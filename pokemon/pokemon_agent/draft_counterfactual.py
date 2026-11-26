"""Generate paired Draft labels and audit the effect of the Team Preview policy."""

import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
import json
from pathlib import Path
from random import Random
from statistics import mean
from time import perf_counter

from .__main__ import HISTORICAL_CARDS
from .draft_advantage import AdvantageDraft
from .draft_forecast import ForecastDraft, load_predictor
from .draft_league import baseline_hash, disjoint, parallel, pools_for, read, save
from .draft_matchups import FROZEN, ROOT, STYLES, League, check_frozen, prior_splits
from .draft_value import DraftRollout
from .game import DRAFT_SIZE, PREVIEW
from .joint import JointPolicy
from .lineup import LineupPolicy, index_cards
from .sim import SEATS, Showdown, _draft, fingerprint, play_battle

PREDICTOR = ROOT / "checkpoints/draft-v3/pick-attention"
HYBRID = "9f4ce36ba012548be7795666aa8bd557fb785ea8fe65e3206697a4802332dc4b"
PROVENANCE = {*FROZEN.values(), HYBRID}
PICKS = tuple(range(1, 13))


def frozen_splits():
    yield from prior_splits()
    yield from read(PREDICTOR / "config.json")["splits"].values()
    for version in ("draft-v2", "draft-v3"):
        for path in (ROOT / "benchmark_results" / version).glob("*.json"):
            yield read(path).get("pools", {})


def check_hybrid():
    check_frozen()
    if baseline_hash(PREDICTOR) != HYBRID:
        raise ValueError("Frozen hybrid checkpoint has changed")


class CountedShowdown(Showdown):
    def __init__(self):
        super().__init__()
        self.games = 0

    def call(self, command, **kwargs):
        self.games += command == "start"
        return super().call(command, **kwargs)


class LabelGames:
    def __init__(self, engine, preview, profiles, cards, *, preview_options=5, preview_seeds=2):
        self.engine = engine
        self.preview = preview
        self.profiles = profiles
        self.cards = index_cards(cards)
        self.preview_options = preview_options
        self.preview_seeds = preview_seeds
        self.cache = {}
        self.activity = Counter()

    def run(self, rosters, actor, seed, mode, deadline=None):
        policies = {seat: JointPolicy(self.profiles) for seat in SEATS}

        def choose(seat, decision, _):
            if deadline is not None and perf_counter() >= deadline:
                raise TimeoutError("Draft review exceeded its budget")
            if decision.phase != PREVIEW or mode == "tactical":
                return decision.validate(policies[seat](decision))
            other = SEATS[1 - SEATS.index(seat)]
            key = tuple(tuple(card["card_id"] for card in rosters[side]) for side in (seat, other))
            if key not in self.cache:
                policy = LineupPolicy(self.profiles, self.cards, self.preview,
                                      options_limit=self.preview_options, seed_count=self.preview_seeds)
                if deadline is not None:
                    decision = replace(decision, seconds_left=1 + max(0., deadline - perf_counter()))
                self.cache[key] = decision.validate(policy(decision))
                self.activity["preview_calls"] += 1
                self.activity["preview_fallbacks"] += policy.fallbacks
                self.activity["preview_overrides"] += policy.different_choices
                visible = [decision.observation[side] for side in ("team", "opponent_team")]
                if policy.fallbacks and all(len(team) == 6 and all(mon and mon.get("species") in self.cards for mon in team)
                                            for team in visible):
                    self.activity["preview_supported_fallbacks"] += 1
            policies[seat].observe(decision)
            return decision.validate(self.cache[key])

        self.activity[mode + "_terminal_games"] += 1
        winner = play_battle(self.engine, rosters, policies, 0, seed, choose)
        return .5 if winner is None else float(winner == actor)


class RootFound(Exception):
    def __init__(self, seat, decision):
        self.seat = seat
        self.decision = decision


def sample_prefix(pool, first, pick, league, hybrid, styles, timings=None):
    policies = {seat: league.policy("pairwise" if style == "hybrid" else style)
                for seat, style in zip(SEATS, styles, strict=True)}
    for seat, style in zip(SEATS, styles, strict=True):
        if style == "hybrid":
            policies[seat].draft = hybrid
    prefix = []

    def choose(seat, decision):
        if decision.observation["pick"] == pick:
            raise RootFound(seat, decision)
        start = perf_counter()
        action = decision.validate(policies[seat](decision))
        if timings is not None:
            timings[seat].append((perf_counter() - start) * 1000)
        prefix.append(action)
        return action

    try:
        _draft(pool, first, choose)
    except RootFound as root:
        return prefix, root.seat, root.decision
    raise RuntimeError("The draft did not reach the requested ply")


def finish_draft(pool, first, prefix, forced, actor, league, opponent, hybrid=None):
    foe = SEATS[1 - SEATS.index(actor)]
    rules = {actor: league.policy("pairwise"), foe: league.policy("pairwise" if opponent == "hybrid" else opponent)}
    if opponent == "hybrid":
        if hybrid is None:
            raise ValueError("Hybrid labelling requires its frozen policy")
        rules[foe].draft = hybrid
    history = []

    def choose(seat, decision):
        turn = decision.observation["pick"] - 1
        if turn < len(prefix):
            action = prefix[turn]
        elif turn == len(prefix):
            action = forced
        else:
            action = rules[seat](decision)
        history.append((seat, action["card_id"]))
        return decision.validate(action)

    return _draft(pool, first, choose), history


def label_cases(seeds, cards, repeats, audit, picks, tail_opponents, reference="pairwise", all_legal=False):
    rows, issues = [], []
    with ExitStack() as stack:
        league = League(stack, cards, "tactical", True)
        preview = stack.enter_context(CountedShowdown())
        profiles = league.factory().profiles
        leaves = LabelGames(league.engine, preview, profiles, cards)
        predictor, _ = load_predictor(PREDICTOR)
        hybrid = ForecastDraft(league.ranks["pairwise"], predictor, "hybrid", DraftRollout(league.simulator, cards, profiles))
        candidates = AdvantageDraft(league.ranks["pairwise"], predictor, None, "expanded")
        for seed in seeds:
            rng = Random(seed)
            pool, first = rng.sample(cards, DRAFT_SIZE), rng.choice(SEATS)
            rng = Random(seed ^ 0xA6D3B2)
            pick = picks[seed % len(picks)]
            styles = ("hybrid" if seed % 3 == 0 else "pairwise", ("search", "linear", "prior", "pairwise")[seed % 4])
            alternate = ("search", "prior", "linear", "prior")[seed % 4]
            if not audit and alternate == "search" and seed % 16:
                alternate = "linear"
            opponents = tail_opponents or ("pairwise", alternate)
            try:
                prefix, actor, decision = sample_prefix(pool, first, pick, league, hybrid, styles)
                teacher = hybrid.rank(decision, "value")[0]
                anchor = teacher if reference == "hybrid" else league.ranks["pairwise"].rank(decision, "value")[0]
                plan = candidates.options(decision, anchor, all_legal=audit or all_legal)
                if plan is None:
                    issues.append({"seed": seed, "error": "Invalid reference completions"})
                    continue
                actions, _, _, (indices, owners, ply, features), slots, width, _ = plan
                card_ids = [action["card_id"] for action in actions]
                draws = [[tuple(rng.randrange(65536) for _ in range(4)) for _ in range(repeats)] for _ in opponents]
                modes = ("rollout", "tactical") if audit else ("rollout",)
                branches = {mode: [None] * len(actions) for mode in modes}
                teacher_index = card_ids.index(teacher["card_id"])
                if audit or all_legal:
                    observed = set(range(len(actions)))
                else:
                    outside = list(range(width, len(actions)))
                    strongest = max(outside, key=lambda i: float(features[slots[i], 0])) if outside else 0
                    observed = {0, teacher_index, strongest, rng.randrange(width)}
                    if outside:
                        observed.add(rng.choice(outside))
                    remaining = [i for i in range(len(actions)) if i not in observed]
                    observed.update(remaining[:max(0, 5 - len(observed))])
                steals, reference_replies = [], []
                for i, action in enumerate(actions):
                    if i not in observed:
                        continue
                    completed = []
                    for opponent in opponents:
                        try:
                            completed.append(finish_draft(pool, first, prefix, action, actor, league, opponent, hybrid))
                        except RuntimeError as error:
                            if not str(error).startswith("Item Clause left"):
                                raise
                            break
                    if len(completed) != len(opponents):
                        continue
                    if i == 0:
                        for _, history in completed:
                            replies = []
                            for seat, card_id in history[pick:]:
                                if seat == actor:
                                    break
                                steals.append(card_id)
                                replies.append(card_id)
                            reference_replies.append(replies)
                    for mode in modes:
                        results = []
                        for (rosters, _), scenario in zip(completed, draws, strict=True):
                            opposing_seat = SEATS[1 - SEATS.index(actor)]
                            swapped = {actor: rosters[opposing_seat], opposing_seat: rosters[actor]}
                            results.append([leaves.run(swapped if j % 2 else rosters, opposing_seat if j % 2 else actor,
                                                       battle_seed, mode) for j, battle_seed in enumerate(scenario)])
                        branches[mode][i] = results
                if branches["rollout"][0] is None or sum(item is not None for item in branches["rollout"]) < 2:
                    issues.append({"seed": seed, "error": "No matched alternatives"})
                    continue
                rows.append({"seed": seed, "pool": indices, "owners": owners, "ply": ply, "features": features.tolist(),
                             "slots": slots, "width": width, "teacher": teacher_index,
                             "denied_slots": [slots[i] for i, card_id in enumerate(card_ids) if card_id in steals],
                             "opponents": opponents, "reference_replies": reference_replies,
                             "prefix_styles": styles, "results": branches})
            except (RuntimeError, ValueError) as error:
                issues.append({"seed": seed, "error": str(error)})
        leaves.activity["preview_inner_games"] = preview.games
        leaves.activity["prefix_search_games"] = hybrid.rollout.activity["rollout_games"] + league.rollout.activity["rollout_games"]
    return rows, issues, leaves.activity


def summarize(rows):
    report = Counter()
    outside_deltas, steal_deltas, preview_deltas = [], [], []
    for row in rows:
        full = row["results"]["rollout"]
        if "tactical" not in row["results"]:
            continue
        tactical = row["results"]["tactical"]
        half = len(full[0][0]) // 2

        def score(values, start, stop):
            return mean(value for scenario in values for value in scenario[start:stop])

        valid = [i for i, scenarios in enumerate(full) if scenarios is not None]
        inside = [i for i in valid if i < row["width"]]
        outside = [i for i in valid if i >= row["width"]]
        full_choice = max(valid, key=lambda i: score(full[i], 0, half))
        tactical_choice = max(valid, key=lambda i: score(tactical[i], 0, half))
        report["preview_choice_disagreements"] += full_choice != tactical_choice
        preview_deltas.append(score(full[full_choice], half, None) - score(full[tactical_choice], half, None))
        if outside:
            best_in = max(inside, key=lambda i: score(full[i], 0, half))
            best_out = max(outside, key=lambda i: score(full[i], 0, half))
            delta = score(full[best_out], half, None) - score(full[best_in], half, None)
            outside_deltas.append(delta)
            report["outside_wins_confirmation"] += delta > 0
            report["outside_ties_confirmation"] += delta == 0
            if row["slots"][best_out] in row["denied_slots"]:
                steal_deltas.append(delta)
    return dict(report) | {"audited_positions": len(preview_deltas), "outside_positions": len(outside_deltas),
                           "outside_mean_confirmation_delta": mean(outside_deltas) if outside_deltas else None,
                           "preview_choice_mean_confirmation_delta": mean(preview_deltas) if preview_deltas else None,
                           "outside_denial_positions": len(steal_deltas),
                           "outside_denial_mean_delta": mean(steal_deltas) if steal_deltas else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--positions", type=int, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--all-legal", action="store_true", help="Label every legal card with full Preview")
    parser.add_argument("--picks", type=int, choices=PICKS, nargs="+", default=PICKS)
    parser.add_argument("--tail-opponents", choices=(*STYLES, "hybrid"), nargs=2)
    parser.add_argument("--reference", choices=("pairwise", "hybrid"), default="pairwise")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or min(args.positions, args.workers) < 1 or args.repeats < 2:
        parser.error("Use a new output path and at least two paired draws per opponent")
    check_hybrid()
    cards = read(HISTORICAL_CARDS)
    seeds = range(args.seed, args.seed + args.positions)
    pools = pools_for(cards, seeds)
    for split in frozen_splits():
        disjoint(pools, split)
    rows, issues, counts = [], [], Counter()
    start = perf_counter()
    for results, errors, activity in parallel(label_cases, seeds, args.workers, cards, args.repeats, args.audit, args.picks,
                                              args.tail_opponents, args.reference, args.all_legal):
        rows.extend(results)
        issues.extend(errors)
        counts.update(activity)
    report = {"positions": len(rows), "requested": args.positions, "repeats": args.repeats, "workers": args.workers,
              "seconds": perf_counter() - start, "issues": issues, "target_picks": args.picks,
              "tail_opponents": args.tail_opponents, "reference": args.reference,
              "all_legal": args.audit or args.all_legal, **dict(counts)}
    if args.audit:
        report["audit"] = summarize(rows)
    save(args.output, {"report": report, "pools": pools, "catalog": fingerprint(cards), "frozen": sorted(PROVENANCE),
                       "rows": sorted(rows, key=lambda row: row["seed"])})
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
