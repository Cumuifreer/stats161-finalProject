"""Run paired games, replay decision traces, or compare observed card sets."""

import argparse
from contextlib import nullcontext
from functools import partial
import json
import math
from pathlib import Path
from random import Random
import sys
from time import perf_counter

from .game import BATTLE, CARD_FIELDS, DECISION_LIMIT, DRAFT, DRAFT_SIZE, PREVIEW, Decision, card_set
from .joint import JointPolicy
from .parity import compare_self_play
from .pivot import PivotOpponent
from .policy import RandomPolicy, heuristic, ranked_actions
from .sim import SEATS, Showdown, play_game
from .tactics import load_card_profiles, tactical

HISTORICAL_CARDS = Path(__file__).resolve().parents[1] / "data" / "historical_cards.json"
BENCH_POLICIES = (
    "heuristic", "random", "tactical", "tactical-calibrated", "joint", "llm-heuristic", "llm", "llm-forecast",
)
ACE_POLICIES = ("pivot", "pivot-base")


def _policy(name, seed, profiles=None):
    if name == "random":
        return RandomPolicy(seed)
    if name == "tactical":
        return tactical
    if name == "tactical-calibrated":
        return partial(tactical, card_profiles=profiles)
    if name == "joint":
        return JointPolicy(profiles)
    if name in {"llm", "llm-heuristic", "llm-forecast"}:
        from .llm import LlmPolicy

        if name == "llm-forecast":
            return LlmPolicy(forecast=JointPolicy(profiles))
        return LlmPolicy(ranked_actions) if name == "llm-heuristic" else LlmPolicy()
    return heuristic


def _p95(values):
    values = sorted(values)
    return round(values[math.ceil(len(values) * .95) - 1], 2) if values else None


def _paired_win_ci(wins_per_pair: list[int]) -> list[float] | None:
    pairs = len(wins_per_pair)
    if pairs < 2:
        return None
    rng = Random(0)
    totals = sorted(sum(rng.choices(wins_per_pair, k=pairs)) for _ in range(4000))
    return [totals[int(q * len(totals))] / (2 * pairs) for q in (.025, .975)]


def _testing_pools(paths):
    pools = {}
    for record in _read_traces(paths):
        decision = record["decision"]
        observation = decision["observation"]
        if decision["phase"] == DRAFT and observation["pick"] == 1:
            cards = observation["cards"]
            if len(cards) != DRAFT_SIZE or len({card["card_id"] for card in cards}) != DRAFT_SIZE:
                raise ValueError(f"Expected {DRAFT_SIZE} unique cards in the first pick")
            pools[json.dumps(cards, sort_keys=True)] = cards
    if not pools:
        raise ValueError("No opening draft pools found in the supplied Testing captures")
    return list(pools.values())


def _bench(args):
    if args.pairs < 1:
        raise ValueError("--pairs must be at least 1")
    ace_trainer = args.ace_trainer
    if args.right in ACE_POLICIES and (not ace_trainer or not args.battle_only):
        raise ValueError("Ace Trainer opponents require --ace-trainer CHECKOUT and --battle-only")
    if ace_trainer and args.right not in ACE_POLICIES:
        raise ValueError("--ace-trainer is used with --right pivot or --right pivot-base")
    pools = _testing_pools(args.testing_pools) if args.testing_pools else []
    cards = []
    if not pools:
        cards = json.loads(args.cards.read_text(encoding="utf-8"))
        if len(cards) < DRAFT_SIZE or len({card["card_id"] for card in cards}) != len(cards):
            raise ValueError(f"The draft catalog must contain at least {DRAFT_SIZE} unique card IDs")
    names = [args.left, args.right]
    profiles = None
    with_forecast = bool({"joint", "llm-forecast"}.intersection(names))
    if with_forecast or "tactical-calibrated" in names:
        profiles = load_card_profiles(cards or [card for pool in pools for card in pool], with_sets=with_forecast)
    counters = [dict(wins=0, fallbacks=0, model_choices=0, different_choices=0) for _ in names]
    pair_wins = [[], []]
    paired_scores = {}
    timings = [[], []]
    battle_times = [[], []]
    fixed_policies = {}
    if args.battle_only:
        fixed_policies = {DRAFT: heuristic, PREVIEW: heuristic}
    elif args.preview_only:
        fixed_policies = {DRAFT: heuristic, BATTLE: tactical}
    draws = 0
    completed_pairs = 0
    errors = []
    started = perf_counter()
    opponent = PivotOpponent(ace_trainer, style=args.right == "pivot") if ace_trainer else nullcontext()
    with opponent as pivot, Showdown() as engine:
        for seed in range(args.seed, args.seed + args.pairs):
            outcomes = []
            pool = pools[(seed - args.seed) % len(pools)] if pools else None
            for placement in ((0, 1), (1, 0)):
                policies = {
                    seat: pivot if names[index] in ACE_POLICIES else _policy(names[index], seed + index, profiles)
                    for seat, index in zip(SEATS, placement, strict=True)
                }
                samples = {seat: [] for seat in SEATS}
                try:
                    winner = play_game(engine, cards, policies, seed, samples, pool=pool, fixed_policies=fixed_policies)
                    outcomes.append(placement[SEATS.index(winner)] if winner else None)
                except (ValueError, RuntimeError) as error:
                    errors.append({"seed": seed, "left_seat": SEATS[placement.index(0)], "error": str(error)})
                finally:
                    for seat, index in zip(SEATS, placement, strict=True):
                        timings[index].extend(samples[seat])
                        battle_times[index].append(sum(time for phase, time, _ in samples[seat] if phase == BATTLE) / 1000)
                        for count in ("fallbacks", "model_choices", "different_choices"):
                            counters[index][count] += getattr(policies[seat], count, 0)
            if len(outcomes) == 2:
                completed_pairs += 1
                draws += outcomes.count(None)
                if args.pair_results:
                    paired_scores[seed] = [0.5 if winner is None else int(winner == 0) for winner in outcomes]
                for index, counter in enumerate(counters):
                    wins = outcomes.count(index)
                    pair_wins[index].append(wins)
                    counter["wins"] += wins
    completed = 2 * args.pairs - len(errors)
    elapsed = perf_counter() - started
    for index, counter in enumerate(counters):
        samples = timings[index]
        counter["policy"] = names[index]
        counter["win_rate"] = counter["wins"] / (2 * completed_pairs) if completed_pairs else None
        counter["win_rate_ci95"] = _paired_win_ci(pair_wins[index])
        counter["score_rate"] = (counter["wins"] + .5 * draws) / (2 * completed_pairs) if completed_pairs else None
        counter["decision_p95_ms"] = _p95(time for _, time, _ in samples)
        counter["battle_time_p95_seconds"] = _p95(battle_times[index])
        counter["forfeits"] = sum(event == "forfeit" for _, _, event in samples)
        counter["by_phase"] = {}
        for phase in DECISION_LIMIT:
            decisions = [(time, event) for stage, time, event in samples if stage == phase]
            counter["by_phase"][phase] = {
                "decisions": len(decisions),
                "p95_ms": _p95(time for time, _ in decisions),
                "auto_actions": sum(event == "auto_action" for _, event in decisions),
            }
    print(json.dumps({
        "catalog": None if pools else str(args.cards), "testing_pools": len(pools),
        "first_seed": args.seed, "pairs": args.pairs,
        "battle_only": args.battle_only, "preview_only": args.preview_only,
        "completed_games": completed, "completed_pairs": completed_pairs,
        "draws": draws, "games_per_second": round(completed / elapsed, 2),
        "agents": counters, "errors": errors,
        **({"paired_scores": paired_scores} if args.pair_results else {}),
    }, indent=2))
    return int(bool(errors))


def _read_traces(paths):
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid trace JSON at {path}:{number}") from error


def _replay(args):
    count = 0
    phases: dict[str, int] = {}
    for record in _read_traces(args.traces):
        decision = Decision.from_snapshot(record["decision"])
        decision.validate(record["action"])
        decision.validate(heuristic(decision))
        phases[decision.phase] = phases.get(decision.phase, 0) + 1
        count += 1
    if not count:
        raise ValueError("No decisions were found in the supplied traces")
    print(json.dumps({"decisions": count, "phases": phases}, indent=2))
    return 0


def _catalog(args):
    historical = {
        card["card_id"]: card_set(card)
        for card in json.loads(HISTORICAL_CARDS.read_text(encoding="utf-8"))
    }
    observations: dict[str, list[dict]] = {}
    for record in _read_traces(args.traces):
        decision = record["decision"]
        if decision["phase"] != DRAFT:
            continue
        observation = decision["observation"]
        for field in ("cards", "picked", "opponent_picked"):
            for card in observation.get(field, ()):
                variants = observations.setdefault(card["card_id"], [])
                if card not in variants:
                    variants.append(card)
    if not observations:
        raise ValueError("No draft cards were found in the supplied traces")
    confirmed = []
    changed = {}
    unobserved = {}
    for card_id, variants in sorted(observations.items()):
        if card_id not in historical:
            continue
        baseline = historical[card_id]
        fields = [
            field for field in CARD_FIELDS
            if any(field in card and card[field] != baseline.get(field) for card in variants)
        ]
        missing = [field for field in baseline if all(field not in card for card in variants)]
        if missing:
            unobserved[card_id] = missing
        if fields:
            changed[card_id] = fields
        elif not missing:
            confirmed.append(card_id)
    print(json.dumps({
        "observed_count": len(observations),
        "confirmed": confirmed,
        "changed_fields": changed,
        "unobserved_fields": unobserved,
        "new_card_ids": sorted(observations.keys() - historical.keys()),
        "historical_not_observed": sorted(historical.keys() - observations.keys()),
    }, indent=2))
    return 0


def _parity(args):
    report = compare_self_play((path, _read_traces([path])) for path in args.traces)
    print(json.dumps(report, indent=2))
    return int(bool(report["differences"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("bench", help="Play seeded games with swapped seats")
    bench.add_argument("--pairs", type=int, default=10)
    bench.add_argument("--seed", type=int, default=1)
    bench.add_argument("--pair-results", action="store_true", help="Include left scores by seed, first in seat p1 then p2")
    source = bench.add_mutually_exclusive_group()
    source.add_argument("--cards", type=Path, default=HISTORICAL_CARDS)
    source.add_argument("--testing-pools", type=Path, nargs="+", help="Cycle through pools captured in Testing")
    bench.add_argument("--left", choices=BENCH_POLICIES, default="heuristic")
    bench.add_argument("--right", choices=(*BENCH_POLICIES, *ACE_POLICIES), default="random")
    bench.add_argument("--ace-trainer", type=Path, help="Local Ace Trainer checkout for pivot and pivot-base")
    phase = bench.add_mutually_exclusive_group()
    phase.add_argument("--battle-only", action="store_true", help="Use the same heuristic for both seats' draft and lineup")
    phase.add_argument("--preview-only", action="store_true", help="Use the same draft and tactical battle policies for both seats")
    bench.set_defaults(run=_bench)
    for command, handler in (("replay", _replay), ("catalog", _catalog), ("parity", _parity)):
        mode = sub.add_parser(command)
        mode.add_argument("traces", type=Path, nargs="+")
        mode.set_defaults(run=handler)
    args = parser.parse_args()
    try:
        return args.run(args)
    except (OSError, RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
