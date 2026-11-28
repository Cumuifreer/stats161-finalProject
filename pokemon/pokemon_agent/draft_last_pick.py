"""Audit and benchmark Draft's final pick with known six-card rosters."""

import argparse
from collections import Counter
from contextlib import ExitStack
import json
from math import comb
from pathlib import Path
from random import Random
from statistics import mean
from time import perf_counter

import torch

from .__main__ import HISTORICAL_CARDS, _paired_win_ci, _percentile
from .draft_advantage import AdvantageDraft, load_advantage
from .draft_counterfactual import (HYBRID, PREDICTOR, PROVENANCE, CountedShowdown, LabelGames,
                                   check_hybrid, frozen_splits, sample_prefix)
from .draft_forecast import ForecastDraft, load_predictor
from .draft_league import baseline_hash, disjoint, parallel, pools_for, read, save
from .draft_matchups import ROOT, League
from .draft_value import DraftRollout
from .game import DECISION_LIMIT, DRAFT, DRAFT_SIZE
from .sim import SEATS, fingerprint

PLANS = ("frozen", "tactical", "full", "tactical-compute", "value", "model", "model-full")
OPPONENTS = ("hybrid", "search", "pairwise", "linear", "prior")
REVIEW_PAIRS = 10
REVIEW_BUDGET = 3.5


def older_splits():
    yield from frozen_splits()
    for folder in (ROOT / "benchmark_results/draft-v4", ROOT / "checkpoints/draft-v4"):
        for path in folder.glob("*.json"):
            yield read(path).get("pools", {})


def sample_game(cards, seed):
    rng = Random(seed)
    pool = rng.sample(cards, DRAFT_SIZE)
    first = rng.choice(SEATS)
    seed = tuple(rng.randrange(65536) for _ in range(4))
    return pool, first, seed


def legal_rosters(decision, action, actor, by_id):
    card = decision.validate(action)["card_id"]
    own, foe = (list(decision.observation[key]) for key in ("picked", "opponent_picked"))
    if decision.phase != DRAFT or decision.observation["pick"] != 12 or (len(own), len(foe)) != (5, 6):
        raise ValueError("The final pick requires five own cards and six opposing cards")
    ours = [by_id[entry["card_id"]] for entry in own] + [by_id[card]]
    theirs = [by_id[entry["card_id"]] for entry in foe]
    return {actor: ours, other(actor): theirs}


def other(actor):
    return SEATS[1 - SEATS.index(actor)]


def review_seeds(decision, count):
    rng = Random(fingerprint(["last-pick-review", decision.observation]))
    return [tuple(rng.randrange(65536) for _ in range(4)) for _ in range(count)]


def beats_reference(values):
    if len(values) < 8:
        return False
    wins = sum(value > 0 for value in values)
    losses = sum(value < 0 for value in values)
    decisive = wins + losses
    tail = sum(comb(decisive, n) for n in range(wins, decisive + 1)) / 2**decisive
    return tail <= .125 and mean(values) >= .15


def mean_ci(values):
    if len(values) < 2:
        return None
    rng = Random(0)
    bootstraps = sorted(mean(rng.choices(values, k=len(values))) for _ in range(4000))
    return [bootstraps[100], bootstraps[3900]]


class Experiment:
    def __init__(self, stack, cards):
        self.cards = cards
        self.by_id = {card["card_id"]: card for card in cards}
        self.league = League(stack, cards, "tactical", True)
        self.profiles = self.league.factory().profiles
        self.preview = stack.enter_context(CountedShowdown())
        predictor, _ = load_predictor(PREDICTOR)
        pairwise = self.league.ranks["pairwise"]
        self.hybrid = ForecastDraft(pairwise, predictor, "hybrid", DraftRollout(self.league.simulator, cards, self.profiles))
        self.projection = AdvantageDraft(pairwise, predictor, None, "expanded")
        self.activity = Counter()

    def plan(self, decision, anchor, model=None):
        result = self.projection.options(decision, anchor, all_legal=True)
        if result is None:
            raise ValueError("Could not construct known post-draft rosters")
        actions, _, _, board, slots, _, _ = result
        pool, owners, ply, features = board
        with torch.inference_mode():
            estimates = features[slots, 0]
            if model is not None:
                logits = model(torch.tensor([pool]), torch.tensor([owners]), torch.tensor([ply]), features[None])[0]
                estimates = .65 * logits[slots].sigmoid() + .35 * estimates
        if len(actions) == 1:
            return result, None, 0.
        alternative = max(range(1, len(actions)), key=lambda i: float(estimates[i]))
        return result, alternative, float(estimates[alternative] - estimates[0])

    def review(self, decision, proposal, anchor, method, deadline):
        leaves = LabelGames(self.league.engine, self.preview, self.profiles, self.cards)
        mode = "rollout" if method in {"full", "model-full"} else "tactical"
        draws = review_seeds(decision, 90 if method == "tactical-compute" else REVIEW_PAIRS)
        candidates = (anchor, proposal)
        gains = []
        before = self.preview.games
        for i, seed in enumerate(draws):
            seat = SEATS[i % 2]
            try:
                results = [leaves.run(legal_rosters(decision, action, seat, self.by_id), seat, seed, mode, deadline)
                           for action in candidates]
                if perf_counter() >= deadline:
                    raise TimeoutError("Draft review did not finish its pair")
            except TimeoutError:
                self.activity[f"{method}_budget_fallbacks"] += 1
                break
            gains.append(results[1] - results[0])
        self.activity[f"{method}_leaf_games"] += leaves.activity[mode + "_terminal_games"]
        self.activity[f"{method}_preview_games"] += self.preview.games - before
        self.activity[f"{method}_preview_fallbacks"] += leaves.activity["preview_fallbacks"]
        self.activity[f"{method}_sufficient_reviews"] += len(gains) >= 8
        return proposal if beats_reference(gains) else anchor


def audit_cases(seeds, cards, draws, tactical):
    rows, errors = [], []
    with ExitStack() as stack:
        experiment = Experiment(stack, cards)
        league = experiment.league
        leaves = LabelGames(league.engine, experiment.preview, experiment.profiles, cards)
        for seed in seeds:
            pool, first, _ = sample_game(cards, seed)
            styles = ("hybrid", OPPONENTS[seed % len(OPPONENTS)])
            if first == "p2":
                styles = styles[::-1]
            try:
                _, actor, decision = sample_prefix(pool, first, 12, league, experiment.hybrid, styles)
                if actor != first:
                    raise RuntimeError("Only the first drafter owns pick 12")
                anchor = experiment.hybrid.rank(decision, "value")[0]
                plan, proposal, _ = experiment.plan(decision, anchor)
                actions, _, _, (ids, owners, ply, features), slots, _, _ = plan
                modes = ("rollout", "tactical") if tactical else ("rollout",)
                rng = Random(seed ^ 0xFB8507)
                seeds_for_outcome = [tuple(rng.randrange(65536) for _ in range(4)) for _ in range(draws)]
                outcomes = {mode: [] for mode in modes}
                for action in actions:
                    for mode in modes:
                        values = []
                        for i, draw in enumerate(seeds_for_outcome):
                            seat = SEATS[i % 2]
                            values.append(leaves.run(legal_rosters(decision, action, seat, experiment.by_id), seat, draw, mode))
                        outcomes[mode].append([values])
                rows.append({"seed": seed, "pool": ids, "owners": owners, "ply": ply, "features": features.tolist(),
                             "slots": slots, "proposal": proposal, "opponent": styles[SEATS.index(other(actor))], "results": outcomes})
            except (OSError, RuntimeError, ValueError) as error:
                errors.append({"seed": seed, "error": str(error)})
        counts = leaves.activity
        counts["preview_inner_games"] = experiment.preview.games
        counts["prefix_search_games"] = experiment.hybrid.rollout.activity["rollout_games"] + league.rollout.activity["rollout_games"]
    return rows, errors, counts


def audit_report(rows, draws):
    report = Counter()
    gain, outside, full_vs_fast = [], [], []
    for row in rows:
        if row["proposal"] is None:
            continue
        full = [values[0] for values in row["results"]["rollout"]]
        half = draws // 2

        def discover(candidates, scores):
            return max(candidates, key=lambda i: mean(scores[i][:half]))

        def confirm(a, b):
            return mean(full[a][half:]) - mean(full[b][half:])

        short = (0, row["proposal"])
        inside = discover(short, full)
        best = discover(range(len(full)), full)
        gain.append(confirm(inside, 0))
        if best not in short:
            report["discovery_outside_pair"] += 1
            outside.append(confirm(best, inside))
            report["confirmed_outside_pair"] += outside[-1] > 0
        if "tactical" in row["results"]:
            fast = [values[0] for values in row["results"]["tactical"]]
            fast_choice = discover(short, fast)
            report["preview_disagreements"] += fast_choice != inside
            full_vs_fast.append(confirm(inside, fast_choice))
    return dict(report) | {"auditable_positions": len(gain), "shortlist_mean_gain": mean(gain) if gain else None,
                           "shortlist_gain_ci95": mean_ci(gain), "outside_mean_gain": mean(outside) if outside else None,
                           "outside_gain_ci95": mean_ci(outside), "full_vs_tactical_mean_gain": mean(full_vs_fast) if full_vs_fast else None,
                           "full_vs_tactical_gain_ci95": mean_ci(full_vs_fast)}


def generate(args, cards, seeds, pools):
    started = perf_counter()
    rows, errors, counts = [], [], Counter()
    for records, issues, counts_by_worker in parallel(audit_cases, seeds, args.workers, cards, args.draws, args.tactical):
        rows.extend(records)
        errors.extend(issues)
        counts.update(counts_by_worker)
    rows.sort(key=lambda row: row["seed"])
    report = {"positions": len(rows), "draws": args.draws, "workers": args.workers, "issues": errors,
              "seconds": perf_counter() - started, "audit": audit_report(rows, args.draws), **dict(counts)}
    save(args.output, {"report": report, "pools": pools, "catalog": fingerprint(cards), "frozen": sorted(PROVENANCE), "rows": rows})
    print(json.dumps(report), flush=True)


def evaluate(seeds, cards, plans, opponents, checkpoint, local_draws, review_budget):
    pairs, errors, counts = {}, [], Counter()
    with ExitStack() as stack:
        experiment = Experiment(stack, cards)
        league = experiment.league
        leaves = LabelGames(league.engine, experiment.preview, experiment.profiles, cards)
        model, _ = load_advantage(checkpoint) if checkpoint else (None, None)
        for seed in seeds:
            pool, first, battle_seed = sample_game(cards, seed)
            opponent = opponents[seed % len(opponents)]
            pair = []
            for candidate in SEATS:
                before = perf_counter()
                composition = {candidate: "hybrid", other(candidate): opponent}
                timings = {seat: [] for seat in SEATS}
                try:
                    _, actor, decision = sample_prefix(pool, first, 12, league, experiment.hybrid,
                                                       [composition[seat] for seat in SEATS], timings)
                    if actor != first:
                        raise RuntimeError("Only the first drafter owns pick 12")
                    start = perf_counter()
                    anchor = (experiment.hybrid.rank(decision, "value")[0] if composition[actor] == "hybrid"
                              else league.policy(composition[actor])(decision))
                    anchor_ms = (perf_counter() - start) * 1000
                    record = {"acts": actor == candidate, "candidate": candidate, "last_pick_seat": actor,
                              "opponent": opponent, "draft_ms": timings[candidate],
                              "anchor_ms": anchor_ms, "moves": {}, "real_worlds": 0}
                    outcomes, local = {}, {}
                    for method in plans:
                        start = perf_counter()
                        action = anchor
                        proposed = None
                        if actor == candidate and method != "frozen" and len(decision.actions) > 1:
                            net = model if method in {"model", "model-full"} else None
                            plan, proposal, gain = experiment.plan(decision, anchor, net)
                            selected = plan[0][proposal]
                            proposed = selected["card_id"]
                            if method in {"value", "model"}:
                                if gain >= .035:
                                    action = selected
                            else:
                                clock = DECISION_LIMIT[DRAFT] if decision.seconds_left is None else decision.seconds_left
                                left = clock - .25 - anchor_ms / 1000
                                if left > 0:
                                    deadline = start + min(review_budget, left)
                                    action = experiment.review(decision, selected, anchor, method, deadline)
                        elapsed_ms = (perf_counter() - start) * 1000
                        key = action["card_id"]
                        if key not in outcomes:
                            result = leaves.run(legal_rosters(decision, action, actor, experiment.by_id), candidate, battle_seed, "rollout")
                            outcomes[key] = result
                        if local_draws and action != anchor:
                            if key not in local:
                                rng = Random(seed ^ 0xFE8276)
                                local_gains = []
                                for i in range(local_draws):
                                    draw = tuple(rng.randrange(65536) for _ in range(4))
                                    seat = SEATS[i % 2]
                                    values = []
                                    for option in (anchor, action):
                                        values.append(leaves.run(legal_rosters(decision, option, seat, experiment.by_id), seat, draw, "rollout"))
                                        counts["local_replay_games"] += 1
                                    local_gains.append(values[1] - values[0])
                                local[key] = mean(local_gains)
                        last_ms = elapsed_ms + anchor_ms if actor == candidate else None
                        record["moves"][method] = {"score": outcomes[key], "changed": action != anchor, "last_ms": last_ms,
                                                   "local_replay_gain": local.get(key), "card_id": key,
                                                   "proposed_card_id": proposed}
                    record["real_worlds"] = len(outcomes)
                    pair.append(record)
                except (OSError, RuntimeError, ValueError) as error:
                    errors.append({"seed": seed, "candidate": candidate, "error": str(error)})
                counts["trial_milliseconds"] += int((perf_counter() - before) * 1000)
            if len(pair) == 2:
                if sum(row["acts"] for row in pair) != 1:
                    raise ValueError("Each paired seed must expose exactly one intervention")
                pairs[seed] = pair
        counts.update(experiment.activity)
        counts["terminal_showdown_games"] = leaves.activity["rollout_terminal_games"] - counts["local_replay_games"]
        counts["all_preview_inner_games"] = experiment.preview.games
        counts["prefix_search_games"] = experiment.hybrid.rollout.activity["rollout_games"] + league.rollout.activity["rollout_games"]
    return pairs, errors, counts


def benchmark_report(pairs, method):
    scores = [sum(row["moves"][method]["score"] for row in pair) for pair in pairs.values()]
    baseline = [sum(row["moves"]["frozen"]["score"] for row in pair) for pair in pairs.values()]
    records = [row for pair in pairs.values() for row in pair]
    acting = [row for row in records if row["acts"]]
    chosen = [row for row in acting if row["moves"][method]["changed"]]
    deltas = [row["moves"][method]["score"] - row["moves"]["frozen"]["score"] for row in chosen]
    replicated = [row["moves"][method]["local_replay_gain"] for row in chosen if row["moves"][method].get("local_replay_gain") is not None]
    times = [time for row in records for time in row["draft_ms"]]
    last = [row["moves"][method]["last_ms"] for row in acting]
    times.extend(last)
    d = [(a-b)/2 for a,b in zip(scores, baseline, strict=True)]
    wins = sum(row["moves"][method]["score"] == 1 for row in records)
    draws = sum(row["moves"][method]["score"] == .5 for row in records)
    return {"games": len(records), "wins": wins, "draws": draws, "score": mean(scores)/2 if scores else None,
            "score_ci95": _paired_win_ci(scores),
            "gain_over_frozen": mean(d) if d else None, "gain_ci95": mean_ci(d),
            "last_pick_games": len(acting), "last_pick_changes": len(chosen),
            "last_pick_change_rate": len(chosen) / len(acting) if acting else None,
            "last_pick_win_rate": mean(row["moves"][method]["score"] for row in acting) if acting else None,
            "other_seat_win_rate": mean(row["moves"][method]["score"] for row in records if not row["acts"]) if records else None,
            "change_local_gain": mean(deltas) if deltas else None, "change_local_gain_ci95": mean_ci(deltas),
            "replicated_local_gain": mean(replicated) if replicated else None, "replicated_local_gain_ci95": mean_ci(replicated),
            "draft_p95_ms": _percentile(times, .95), "draft_p99_ms": _percentile(times, .99),
            "last_pick_p95_ms": _percentile(last, .95), "last_pick_p99_ms": _percentile(last, .99),
            "draft_timeouts": sum(time >= 1000 * DECISION_LIMIT[DRAFT] for time in times)}


def benchmark(args, cards, seeds, pools):
    started = perf_counter()
    pairs, errors, counts = {}, [], Counter()
    plans = tuple(dict.fromkeys(("frozen", *args.plans)))
    for results, issues, events in parallel(evaluate, seeds, args.workers, cards, plans, args.opponents,
                                           args.checkpoint, args.local_draws, args.review_budget):
        pairs.update(results)
        errors.extend(issues)
        counts.update(events)
    opponents = {name: {seed: pair for seed, pair in pairs.items() if pair[0]["opponent"] == name} for name in args.opponents}
    output = {"seed": args.seed, "workers": args.workers, "pairs": len(pairs), "local_draws": args.local_draws,
              "extra_review_budget_seconds": args.review_budget,
              "seconds": perf_counter() - started,
              "frozen": HYBRID, "checkpoint": baseline_hash(args.checkpoint) if args.checkpoint else None,
              "results": {plan: benchmark_report(pairs, plan) for plan in plans},
              "by_opponent": {name: {plan: benchmark_report(batch, plan) for plan in plans} for name, batch in opponents.items() if batch},
              "errors": errors, "events": dict(counts)}
    save(args.output, {"report": output, "pools": pools, "paired_games": dict(sorted(pairs.items()))})
    print(json.dumps({key: value for key, value in output.items() if key != "by_opponent"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen, bench = [sub.add_parser(name) for name in ("generate", "bench")]
    for command in (gen, bench):
        command.add_argument("--seed", type=int, required=True)
        command.add_argument("--count", type=int, required=True)
        command.add_argument("--workers", type=int, default=12)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--exclude", type=Path, nargs="*", default=[])
    gen.add_argument("--draws", type=int, default=16)
    gen.add_argument("--tactical", action="store_true")
    bench.add_argument("--plans", choices=PLANS, nargs="+", default=("tactical", "full", "tactical-compute"))
    bench.add_argument("--opponents", choices=OPPONENTS, nargs="+", default=("hybrid", "search", "pairwise"))
    bench.add_argument("--checkpoint", type=Path)
    bench.add_argument("--local-draws", type=int, default=0)
    bench.add_argument("--review-budget", type=float, default=REVIEW_BUDGET)
    args = parser.parse_args()
    if args.output.exists() or min(args.count, args.workers) < 1:
        parser.error("Use a new output path and positive count and worker values")
    check_hybrid()
    cards = read(HISTORICAL_CARDS)
    seeds = range(args.seed, args.seed + args.count)
    pools = pools_for(cards, seeds)
    for split in older_splits():
        disjoint(pools, split)
    for path in args.exclude:
        disjoint(pools, read(path)["pools"])
    if args.command == "generate":
        if args.draws < 8 or args.draws % 4:
            parser.error("Use a multiple of four paired draws, at least eight")
        generate(args, cards, seeds, pools)
    else:
        if args.local_draws < 0 or args.local_draws % 2:
            parser.error("Use an even, nonnegative number of local replays")
        if not 0 < args.review_budget < DECISION_LIMIT[DRAFT]:
            parser.error("Use a positive review budget below the Draft deadline")
        if {"model", "model-full"} & set(args.plans) and not args.checkpoint:
            parser.error("A model plan requires a checkpoint")
        if args.checkpoint:
            _, config = load_advantage(args.checkpoint)
            if config["catalog"] != fingerprint(cards) or set(config["frozen"]) != PROVENANCE or config["trained_picks"] != [12]:
                raise ValueError("Expected a checkpoint trained on pick 12")
            for split in config["splits"].values():
                disjoint(pools, split)
        benchmark(args, cards, seeds, pools)


if __name__ == "__main__":
    main()
