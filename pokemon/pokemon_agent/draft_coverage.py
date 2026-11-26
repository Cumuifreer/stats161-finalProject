"""Compare late Draft coverage against the frozen final-pick champion."""

import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from .__main__ import HISTORICAL_CARDS, _percentile
from .draft_advantage import load_advantage
from .draft_counterfactual import HYBRID, PROVENANCE, LabelGames, check_hybrid, sample_prefix
from .draft_last_pick import (Experiment, beats_reference, benchmark_report, legal_rosters, older_splits,
                              other, review_seeds, sample_game)
from .draft_league import baseline_hash, disjoint, parallel, pools_for, read, save
from .draft_matchups import ROOT
from .game import DECISION_LIMIT, DRAFT
from .sim import SEATS, _draft, fingerprint

CHAMPION = {"hybrid": HYBRID, "pick": 12, "alternatives": 1, "pairs": 10, "review_seconds": 6., "starts": 180}
METHODS = ("four", "all", "learned", "learned-two", "direct", "direct-prior", "search11", "learned11")
RIVALS = ("champion", "hybrid", "search", "pairwise", "linear")


def preview_allocation(actions):
    per_action = (CHAMPION["starts"] - 20) // actions - 4
    seeds = 2 if per_action >= 24 else 1
    choices = min(5, per_action // (4 * seeds))
    if choices < 2:
        raise ValueError("Too many legal final-pick candidates for this budget")
    discovery = (CHAMPION["starts"] - 20 - 4 * actions * choices * seeds) // actions
    return choices, seeds, discovery - discovery % 2


class SearchLimit:
    def __init__(self, deadline, max_starts=CHAMPION["starts"]):
        self.deadline = deadline
        self.max_starts = max_starts
        self.starts = 0
        self.expired = False

    def check(self):
        if perf_counter() >= self.deadline:
            self.expired = True
            raise TimeoutError("Late Draft review reached the decision clock reserve")

    def wrap(self, worker):
        limit = self

        class LimitedWorker:
            def call(self, command, **kwargs):
                limit.check()
                if command == "start":
                    if limit.starts == limit.max_starts:
                        limit.expired = True
                        raise TimeoutError("Late Draft review reached its Showdown budget")
                    limit.starts += 1
                return worker.call(command, **kwargs)

        return LimitedWorker()


class Coverage(Experiment):
    def __init__(self, stack, cards):
        super().__init__(stack, cards)
        self.original_cards = [self.by_id[card] for card in self.projection.prior.index]

    def rank(self, decision, anchor, model):
        plan = self.projection.options(decision, anchor, all_legal=True)
        if plan is None:
            return None
        actions, worlds, _, (pool, owners, ply, features), slots, _, _ = plan
        with torch.inference_mode():
            estimates = features[slots, 0]
            if model is not None:
                prediction = model(torch.tensor([pool]), torch.tensor([owners]), torch.tensor([ply]), features[None])[0]
                estimates = .65 * prediction[slots].sigmoid() + .35 * estimates
        order = sorted(range(1, len(actions)), key=lambda i: float(estimates[i]), reverse=True)
        return actions, worlds, order, estimates.tolist()

    def choose(self, decision, style):
        if style not in {"hybrid", "champion"}:
            return self.league.policy(style)(decision)
        start = perf_counter()
        anchor = self.hybrid.rank(decision, "value")[0]
        if style == "champion" and decision.observation["pick"] == 12 and len(decision.actions) > 1:
            previous = self.activity["frozen_starts"]
            action = self.select(decision, anchor, "frozen", None, start)
            self.activity["other_champion_starts"] += self.activity["frozen_starts"] - previous
            self.activity["frozen_starts"] = previous
            return action
        return anchor

    def select(self, decision, anchor, method, model, started):
        if len(decision.actions) < 2 or (method == "frozen" and decision.observation["pick"] != 12):
            return anchor
        elapsed = perf_counter() - started
        clock = decision.seconds_left if decision.seconds_left is not None else DECISION_LIMIT[DRAFT]
        # Reserve time to deliver the already-validated pick if review stops.
        deadline = perf_counter() + min(CHAMPION["review_seconds"], clock - elapsed - .75)
        if deadline <= perf_counter():
            return anchor
        use_model = model if method in {"learned", "learned-two", "direct", "learned11"} else None
        plan = self.rank(decision, anchor, use_model)
        if plan is None:
            return anchor
        actions, worlds, order, scores = plan
        if not order:
            return anchor
        if method in {"direct", "direct-prior"}:
            return actions[order[0]] if scores[order[0]] - scores[0] >= .035 else anchor
        if method in {"frozen", "learned-two"}:
            previous = self.activity["full_leaf_games"] + self.activity["full_preview_games"]
            result = self.review(decision, actions[order[0]], anchor, "full", deadline)
            used = self.activity["full_leaf_games"] + self.activity["full_preview_games"] - previous
            if used > CHAMPION["starts"]:
                raise RuntimeError("Frozen review exceeded its published start cap")
            self.activity[method + "_starts"] += used
            return result
        if method in {"search11", "learned11"}:
            return self.review11(decision, actions, worlds, order[0], method, deadline)
        selected = [0, *(order if method == "all" else order[:3])]
        if len(selected) == 2:
            previous = self.activity["frozen_starts"]
            action = self.select(decision, anchor, "frozen", None, started)
            self.activity[method + "_starts"] += self.activity["frozen_starts"] - previous
            self.activity["frozen_starts"] = previous
            return action
        return self.review_breadth(decision, actions, selected, scores, method, deadline)

    def budgeted_games(self, limit, options, samples):
        return LabelGames(limit.wrap(self.league.engine), limit.wrap(self.preview), self.profiles, self.cards,
                          preview_options=options, preview_seeds=samples)

    def review_breadth(self, decision, actions, selected, scores, method, deadline):
        options, repeats, discovery = preview_allocation(len(selected))
        limit = SearchLimit(deadline)
        leaves = self.budgeted_games(limit, options, repeats)
        draws = review_seeds(decision, discovery + CHAMPION["pairs"])
        ranking = {i: [] for i in selected}
        gains = []
        candidate = 0
        try:
            for i, seed in enumerate(draws[:discovery]):
                seat = SEATS[i % 2]
                for slot in selected:
                    roster = legal_rosters(decision, actions[slot], seat, self.by_id)
                    ranking[slot].append(leaves.run(roster, seat, seed, "rollout", deadline))
            candidate = max(selected[1:], key=lambda i: (mean(ranking[i]), scores[i]))
            for i, seed in enumerate(draws[discovery:]):
                seat = SEATS[i % 2]
                baseline, alternative = (leaves.run(legal_rosters(decision, actions[slot], seat, self.by_id), seat,
                                                   seed, "rollout", deadline) for slot in (0, candidate))
                limit.check()
                gains.append(alternative - baseline)
        except TimeoutError:
            self.activity[method + "_budget_stops"] += 1
        self.activity[method + "_starts"] += limit.starts
        self.activity[method + "_sufficient_reviews"] += len(gains) >= 8
        return actions[candidate] if beats_reference(gains) else actions[0]

    def review11(self, decision, actions, worlds, candidate, method, deadline):
        replies = []
        for index in (0, candidate):
            possible = [rosters for rosters in worlds[index] if rosters is not None]
            if not possible:
                return actions[0]
            ordered = sorted(Counter(tuple(tuple(team) for team in rosters) for rosters in possible).items(),
                             key=lambda row: row[1], reverse=True)
            replies.append([ordered[0][0], ordered[min(1, len(ordered) - 1)][0]])
        limit = SearchLimit(deadline)
        leaves = self.budgeted_games(limit, 5, 1)
        self.activity[method + "_distinct_reply_positions"] += any(first != second for first, second in replies)
        outcomes = []
        try:
            for i, seed in enumerate(review_seeds(decision, 12)):
                seat = SEATS[(i // 2) % 2]
                results = []
                for ways in replies:
                    teams = ways[i % 2]
                    rosters = {who: [self.original_cards[card] for card in roster]
                               for who, roster in zip((seat, other(seat)), teams, strict=True)}
                    results.append(leaves.run(rosters, seat, seed, "rollout", deadline))
                limit.check()
                outcomes.append(results[1] - results[0])
        except TimeoutError:
            self.activity[method + "_budget_stops"] += 1
        self.activity[method + "_starts"] += limit.starts
        self.activity[method + "_sufficient_reviews"] += len(outcomes) >= 8
        robust = len(outcomes) >= 8 and all(mean(outcomes[i::2]) >= 0 for i in (0, 1))
        return actions[candidate] if robust and beats_reference(outcomes) else actions[0]


def play_terminal(experiment, pool, first, prefix, actor, decision, action, styles, candidate, seed, leaves):
    suffix_ms = None
    if decision.observation["pick"] == 12:
        rosters = legal_rosters(decision, action, actor, experiment.by_id)
    else:
        def pick(seat, public):
            nonlocal suffix_ms
            turn = public.observation["pick"] - 1
            if turn < len(prefix):
                return prefix[turn]
            if turn == len(prefix):
                return action
            start = perf_counter()
            final = public.validate(experiment.choose(public, styles[seat]))
            if seat == candidate:
                suffix_ms = (perf_counter() - start) * 1000
            return final

        rosters = _draft(pool, first, pick)
    return leaves.run(rosters, candidate, seed, "rollout"), suffix_ms


def evaluate(seeds, cards, target, methods, opponents, checkpoint):
    pairs, issues = {}, []
    with ExitStack() as stack:
        experiment = Coverage(stack, cards)
        leaves = LabelGames(experiment.league.engine, experiment.preview, experiment.profiles, cards)
        model = load_advantage(checkpoint)[0] if checkpoint else None
        for seed in seeds:
            pool, first, battle_seed = sample_game(cards, seed)
            opponent = opponents[seed % len(opponents)]
            records = []
            for candidate in SEATS:
                styles = {candidate: "champion", other(candidate): opponent}
                before_target = ["hybrid" if styles[seat] == "champion" else styles[seat] for seat in SEATS]
                times = {seat: [] for seat in SEATS}
                try:
                    prefix, actor, decision = sample_prefix(pool, first, target, experiment.league, experiment.hybrid,
                                                           before_target, times)
                    if actor != (first if target == 12 else other(first)):
                        raise ValueError("The late-pick intervention is in the wrong seat")
                    start = perf_counter()
                    raw = decision.validate(experiment.choose(decision, "hybrid" if actor == candidate else styles[actor]))
                    anchor_ms = (perf_counter() - start) * 1000
                    outcome, moves = {}, {}
                    reference = raw
                    for method in methods:
                        started = perf_counter()
                        choice = (experiment.select(decision, raw, method, model, started - anchor_ms / 1000)
                                  if actor == candidate else raw)
                        elapsed = anchor_ms + (perf_counter() - started) * 1000
                        choice = decision.validate(choice)
                        if method == "frozen":
                            reference = choice
                        key = choice["card_id"]
                        if key not in outcome:
                            outcome[key] = play_terminal(experiment, pool, first, prefix, actor, decision, choice,
                                                         styles, candidate, battle_seed, leaves)
                        score, suffix_ms = outcome[key]
                        moves[method] = {"score": score, "changed": choice != reference, "card_id": key,
                                         "last_ms": elapsed if actor == candidate else None, "suffix_ms": suffix_ms}
                    records.append({"acts": actor == candidate, "draft_ms": times[candidate], "moves": moves, "opponent": opponent})
                except (OSError, RuntimeError, ValueError) as error:
                    issues.append({"seed": seed, "seat": candidate, "error": str(error)})
            if len(records) == 2:
                pairs[seed] = records
        experiment.activity["terminal_games"] = leaves.activity["rollout_terminal_games"]
        experiment.activity["all_preview_inner_games"] = experiment.preview.games
        experiment.activity["hybrid_rollouts"] = experiment.hybrid.rollout.activity["rollout_games"] + experiment.league.rollout.activity["rollout_games"]
    return pairs, issues, experiment.activity


def run(args):
    check_hybrid()
    cards = read(HISTORICAL_CARDS)
    seeds = range(args.seed, args.seed + args.count)
    pools = pools_for(cards, seeds)
    for split in older_splits():
        disjoint(pools, split)
    for group in ("checkpoints", "benchmark_results"):
        for path in (ROOT / group / "draft-v5").glob("*.json"):
            disjoint(pools, read(path).get("pools", {}))
    for path in args.exclude:
        disjoint(pools, read(path)["pools"])
    if args.checkpoint:
        config = read(args.checkpoint / "config.json")
        if config["catalog"] != fingerprint(cards) or set(config["frozen"]) != PROVENANCE or config["trained_picks"] != [args.pick]:
            raise ValueError("Unexpected benchmark checkpoint provenance")
        for split in config["splits"].values():
            disjoint(pools, split)
    methods = ("frozen", *dict.fromkeys(args.methods))
    started = perf_counter()
    pairs, issues, counts = {}, [], Counter()
    for games, errors, events in parallel(evaluate, seeds, args.workers, cards, args.pick, methods, args.opponents, args.checkpoint):
        pairs.update(games)
        issues.extend(errors)
        counts.update(events)
    metrics = {method: benchmark_report(pairs, method) for method in methods}
    for method, data in metrics.items():
        times = [time for pair in pairs.values() for row in pair for time in row["draft_ms"]]
        times.extend(time for pair in pairs.values() for row in pair
                     for key in ("last_ms", "suffix_ms") if (time := row["moves"][method][key]) is not None)
        data.update(draft_p95_ms=_percentile(times, .95), draft_p99_ms=_percentile(times, .99),
                    draft_timeouts=sum(time >= 15000 for time in times))
    report = {"champion": CHAMPION, "seed": args.seed, "workers": args.workers, "pick": args.pick, "pairs": len(pairs),
              "seconds": perf_counter() - started, "results": metrics, "errors": issues, "events": dict(counts),
              "checkpoint": baseline_hash(args.checkpoint) if args.checkpoint else None,
              "by_opponent": {name: {method: benchmark_report({seed: pair for seed, pair in pairs.items() if pair[0]["opponent"] == name}, method)
                                     for method in methods} for name in sorted(set(args.opponents))}}
    save(args.output, {"report": report, "pools": pools, "paired_games": dict(sorted(pairs.items()))})
    print(json.dumps({key: value for key, value in report.items() if key != "by_opponent"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pick", type=int, choices=(11, 12), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--methods", choices=METHODS, nargs="+", required=True)
    parser.add_argument("--opponents", choices=RIVALS, nargs="+", default=("champion", "search", "pairwise"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    if args.output.exists() or min(args.count, args.workers) < 1:
        parser.error("Use a fresh output path and positive count and worker values")
    if any((method.endswith("11")) != (args.pick == 11) for method in args.methods):
        parser.error("Choose methods for the requested Draft pick")
    if {"learned", "learned-two", "direct", "learned11"} & set(args.methods) and not args.checkpoint:
        parser.error("A learned method requires a checkpoint")
    run(args)


if __name__ == "__main__":
    main()
