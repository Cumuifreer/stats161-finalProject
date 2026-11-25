"""Run paired games, replay decision traces, or compare observed card sets."""

import argparse
from collections import Counter
from contextlib import nullcontext
from functools import partial
import json
import math
from pathlib import Path
from random import Random
import subprocess
import sys
from time import perf_counter

from .game import BATTLE, CARD_FIELDS, DECISION_LIMIT, DRAFT, DRAFT_SIZE, PREVIEW, Decision, card_set
from .forecast import ForecastConfig, ForecastPolicy, fit as fit_forecast
from .joint import JointPolicy
from .parity import compare_self_play
from .pivot import PivotOpponent
from .policy import RandomPolicy, heuristic, ranked_actions
from .sim import SEATS, Showdown, fingerprint, play_game, pool_fingerprint
from .tactics import load_card_profiles, tactical

HISTORICAL_CARDS = Path(__file__).resolve().parents[1] / "data" / "historical_cards.json"
FROZEN_BENCHMARK = HISTORICAL_CARDS.with_name("frozen_benchmark.json")
BENCH_POLICIES = (
    "heuristic", "random", "tactical", "tactical-calibrated", "joint", "llm-heuristic", "llm", "llm-forecast", "bc", "search", "ppo",
    "forecast", "intervention", "lineup-rollout", "roster-net",
)
ACE_POLICIES = ("pivot", "pivot-base")


def _policy(name, seed, profiles=None, search_worker=None, forecast=None):
    if name == "random":
        return RandomPolicy(seed)
    if name == "tactical":
        return tactical
    if name == "tactical-calibrated":
        return partial(tactical, card_profiles=profiles)
    if name == "joint":
        return JointPolicy(profiles)
    if name == "forecast":
        return ForecastPolicy(profiles, forecast)
    if name == "search":
        from .search import SearchPolicy

        return SearchPolicy(profiles, search_worker)
    if name in {"llm", "llm-heuristic", "llm-forecast"}:
        from .llm import LlmPolicy

        if name == "llm-forecast":
            return LlmPolicy(forecast=JointPolicy(profiles))
        return LlmPolicy(ranked_actions) if name == "llm-heuristic" else LlmPolicy()
    return heuristic


def _percentile(values, quantile):
    values = sorted(values)
    return round(values[math.ceil(len(values) * quantile) - 1], 2) if values else None


def _p95(values):
    return _percentile(values, .95)


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


def _frozen_benchmark(args, cards, checkpoints):
    path = getattr(args, "frozen_suite", None) or FROZEN_BENCHMARK
    suite = json.loads(path.read_text(encoding="utf-8"))
    phase_matches = {"lineup": getattr(args, "lineup_only", False), "pipeline": getattr(args, "pipeline", False),
                     "battle": args.battle_only}.get(suite.get("phase", "battle"), False)
    if not phase_matches or args.testing_pools or args.right not in suite["opponents"]:
        raise ValueError("Use the frozen suite's phase, a listed opponent, and the validated catalog")
    if fingerprint(cards) != suite["catalog_sha256"]:
        raise ValueError("Frozen benchmark card catalog differs from the manifest")
    if args.ace_trainer:
        if not args.ace_trainer.is_dir():
            raise ValueError("--ace-trainer must point to the frozen Ace Trainer checkout")
        revision = subprocess.run(["git", "-C", str(args.ace_trainer), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=False)
        if revision.returncode or revision.stdout.strip() != suite["ace_trainer_revision"]:
            raise ValueError("Frozen benchmark requires the pinned Ace Trainer revision")
    seeds = range(suite["seed"], suite["seed"] + suite["pairs"])
    pools = [pool_fingerprint(cards, seed) for seed in seeds]
    if fingerprint(pools) != suite["pool_order_sha256"]:
        raise ValueError("Frozen benchmark draft pool order differs from the manifest")
    frozen_pools = set(pools)
    for checkpoint in checkpoints:
        config = json.loads((checkpoint if checkpoint.is_file() else checkpoint / "config.json").read_text(encoding="utf-8"))
        for group in config["split"].values():
            if any(int(seed) in seeds or pool in frozen_pools for seed, pool in group.items()):
                raise ValueError("Checkpoint data overlaps the frozen benchmark")
    return suite


def _bench(args):
    if args.pairs < 1:
        raise ValueError("--pairs must be at least 1")
    ace_trainer = args.ace_trainer
    lineup_only = getattr(args, "lineup_only", False)
    pipeline = getattr(args, "pipeline", False)
    if args.right in ACE_POLICIES and (not ace_trainer or not (args.battle_only or lineup_only or pipeline)):
        raise ValueError("Ace Trainer opponents require a pinned checkout in a battle, lineup or pipeline benchmark")
    if ace_trainer and args.right not in ACE_POLICIES:
        raise ValueError("--ace-trainer is used with --right pivot or --right pivot-base")
    pools = _testing_pools(args.testing_pools) if args.testing_pools else []
    cards = []
    if not pools:
        cards = json.loads(args.cards.read_text(encoding="utf-8"))
        if len(cards) < DRAFT_SIZE or len({card["card_id"] for card in cards}) != len(cards):
            raise ValueError(f"The draft catalog must contain at least {DRAFT_SIZE} unique card IDs")
    names = [args.left, args.right]
    with_lineup = "lineup-rollout" in names
    if with_lineup and not (lineup_only or pipeline):
        raise ValueError("Lineup rollouts require --lineup-only or --pipeline")
    roster_path = getattr(args, "roster_checkpoint", None)
    roster_battle = getattr(args, "roster_battle_checkpoint", None)
    with_roster = "roster-net" in names
    opposing_draft = getattr(args, "roster_opponent_draft", None)
    if with_roster != bool(roster_path) or (with_roster and not (lineup_only or pipeline)):
        raise ValueError("Roster policy requires --roster-checkpoint with --lineup-only or --pipeline")
    if roster_battle and not with_roster:
        raise ValueError("--roster-battle-checkpoint requires a roster policy")
    if opposing_draft and names != ["roster-net", "roster-net"]:
        raise ValueError("--roster-opponent-draft requires two roster policies")
    if with_roster and pools:
        cards = json.loads(args.cards.read_text(encoding="utf-8"))
        published = {card["card_id"]: card_set(card) for card in cards}
        if any(card_set(card) != published.get(card["card_id"]) for pool in pools for card in pool):
            raise ValueError("Testing pool cards differ from the roster checkpoint catalog")
    roster_preview = getattr(args, "roster_preview", "rollout")
    forecast_path = getattr(args, "forecast_model", None)
    forecast = None
    if "forecast" in names:
        if not forecast_path or not args.battle_only:
            raise ValueError("Forecast requires --forecast-model FILE and --battle-only")
        saved = json.loads(forecast_path.read_text(encoding="utf-8"))
        if saved["catalog_sha256"] != fingerprint(cards):
            raise ValueError("Forecast model and benchmark must use the same card catalog")
        forecast = ForecastConfig(**saved["model"])
    elif forecast_path:
        raise ValueError("--forecast-model is used with --left forecast or --right forecast")
    checkpoint_flags = {"bc": "bc_checkpoint", "ppo": "ppo_checkpoint", "ppo-history": "history_checkpoint",
                        "intervention": "intervention_checkpoint"}
    checkpoints = {name: getattr(args, flag, None) for name, flag in checkpoint_flags.items()}
    frozen = (_frozen_benchmark(args, cards, (path for path in (*checkpoints.values(), forecast_path, roster_path, roster_battle) if path))
              if getattr(args, "frozen", False) or getattr(args, "frozen_suite", None) else None)
    first_seed, pairs = (frozen["seed"], frozen["pairs"]) if frozen else (args.seed, args.pairs)
    learned = {}
    for name, checkpoint in checkpoints.items():
        flag = "--" + checkpoint_flags[name].replace("_", "-")
        if name not in names:
            if checkpoint:
                raise ValueError(f"{flag} is used with --left {name} or --right {name}")
            continue
        if not checkpoint or not (args.battle_only or (pipeline and name in {"bc", "ppo", "ppo-history"})):
            raise ValueError(f"{name.upper()} requires {flag} DIRECTORY and --battle-only")
        import torch
        torch.set_num_threads(1)
        if name == "intervention":
            from .intervention import InterventionPolicy

            learned[name] = InterventionPolicy.load(checkpoint, args.bc_device, getattr(args, "intervention_threshold", None))
            continue
        from .model import ActorCriticNetwork, BattlePolicy

        shortlist = getattr(args, "bc_shortlist", None) if name == "bc" else None
        learned[name] = BattlePolicy.load(checkpoint, args.bc_device, shortlist_only=shortlist)
        if name != "bc" and not isinstance(learned[name].model, ActorCriticNetwork):
            raise ValueError(f"{flag} requires an actor-critic checkpoint")
    audit = getattr(args, "value_audit", False)
    if audit:
        if "ppo" not in names:
            raise ValueError("--value-audit requires a PPO policy")
        from .ppo import ValueAuditPolicy, terminal_reward, value_summary
    profiles = None
    with_forecast = bool({"joint", "llm-forecast", "search", "forecast", "lineup-rollout"}.intersection(names))
    with_search = "search" in names
    if with_search and not args.battle_only:
        raise ValueError("Search requires --battle-only")
    catalog = cards or [card for pool in pools for card in pool]
    if with_forecast or "tactical-calibrated" in names:
        profiles = load_card_profiles(catalog, with_sets=with_forecast)
    counters = [dict(wins=0, fallbacks=0, model_choices=0, different_choices=0) for _ in names]
    pair_wins = [[], []]
    paired_scores = {}
    timings = [[], []]
    battle_times = [[], []]
    search_counts = [Counter() for _ in names]
    search_times = [[] for _ in names]
    value_examples = [[] for _ in names]
    fixed_policies = {}
    if args.battle_only:
        fixed_policies = {DRAFT: heuristic, PREVIEW: heuristic}
    elif args.preview_only:
        fixed_policies = {DRAFT: heuristic, BATTLE: tactical}
    elif lineup_only:
        fixed_policies = {DRAFT: heuristic}
    draws = 0
    completed_pairs = 0
    errors = []
    started = perf_counter()
    opponent = PivotOpponent(ace_trainer, style=args.right == "pivot") if ace_trainer else nullcontext()
    planner = Showdown(public_search=True) if with_search else nullcontext()
    lineup_worker = Showdown() if with_lineup or (with_roster and roster_preview in {"rollout", "value-rollout"}) else nullcontext()
    roster_counts = [Counter() for _ in names]
    with opponent as pivot, Showdown() as engine, planner as search_worker, lineup_worker as lineup_sim:
        if with_lineup:
            from .lineup import LineupPolicy, index_cards

            by_name = index_cards(catalog)
            factory = partial(LineupPolicy, profiles, by_name, lineup_sim)
        if with_roster:
            from .roster import roster_factory

            roster = roster_factory(roster_path, cards, lineup_sim, args.bc_device, args.roster_draft, roster_preview, roster_battle)
            roster_factories = (roster, partial(roster, draft_mode=opposing_draft) if opposing_draft else roster)
        if with_search:
            known_cards = {card["card_id"]: card for card in catalog}
            search_worker.call("configure", cards=list(known_cards.values()))
        for seed in range(first_seed, first_seed + pairs):
            outcomes = []
            pair_values = [[], []]
            pool = pools[(seed - first_seed) % len(pools)] if pools else None
            for placement in ((0, 1), (1, 0)):
                policies = {
                    seat: (roster_factories[index]() if names[index] == "roster-net" else factory() if names[index] == "lineup-rollout" else
                           pivot if names[index] in ACE_POLICIES else learned[names[index]].for_match() if names[index] in learned
                           else _policy(names[index], seed + index, profiles, search_worker, forecast))
                    for seat, index in zip(SEATS, placement, strict=True)
                }
                if audit:
                    for seat, index in zip(SEATS, placement, strict=True):
                        if names[index] == "ppo":
                            policies[seat] = ValueAuditPolicy(policies[seat])
                samples = {seat: [] for seat in SEATS}
                try:
                    winner = play_game(engine, cards, policies, seed, samples, pool=pool, fixed_policies=fixed_policies)
                    outcomes.append(placement[SEATS.index(winner)] if winner else None)
                    if audit:
                        for seat, index in zip(SEATS, placement, strict=True):
                            opening = getattr(policies[seat], "opening_value", None)
                            if opening is not None and not any(event for records in samples.values() for _, _, event in records):
                                pair_values[index].append((opening, terminal_reward(winner, seat)))
                except (ValueError, RuntimeError) as error:
                    errors.append({"seed": seed, "left_seat": SEATS[placement.index(0)], "error": str(error)})
                finally:
                    for seat, index in zip(SEATS, placement, strict=True):
                        timings[index].extend(samples[seat])
                        battle_times[index].append(sum(time for phase, time, _ in samples[seat] if phase == BATTLE) / 1000)
                        for count in ("fallbacks", "model_choices", "different_choices"):
                            counters[index][count] += getattr(policies[seat], count, 0)
                        if names[index] == "roster-net":
                            roster_counts[index].update(policies[seat].roster_activity)
                        if names[index] == "search":
                            search_counts[index].update(policies[seat].search_counts)
                            search_times[index].extend(policies[seat].search_ms)
            if len(outcomes) == 2:
                completed_pairs += 1
                draws += outcomes.count(None)
                if args.pair_results:
                    paired_scores[seed] = [0.5 if winner is None else int(winner == 0) for winner in outcomes]
                for index, counter in enumerate(counters):
                    wins = outcomes.count(index)
                    pair_wins[index].append(wins)
                    counter["wins"] += wins
                    value_examples[index].extend(pair_values[index])
    completed = 2 * pairs - len(errors)
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
        if names[index] == "roster-net":
            counter["roster_activity"] = dict(roster_counts[index])
        if audit and value_examples[index]:
            counter["value_audit"] = value_summary(value_examples[index])
        if names[index] == "search":
            counts = search_counts[index]
            counter["search"] = dict(counts) | {
                "coverage": counts["completed"] / counts["battle_decisions"] if counts["battle_decisions"] else 0.,
                "completed_p95_ms": _p95(search_times[index]),
                "completed_p99_ms": _percentile(search_times[index], .99),
            }
        counter["by_phase"] = {}
        for phase in DECISION_LIMIT:
            decisions = [(time, event) for stage, time, event in samples if stage == phase]
            illegal = sum(event == "illegal_action" for _, event in decisions)
            timeouts = sum(event in {"auto_action", "forfeit"} for _, event in decisions)
            counter["by_phase"][phase] = {
                "decisions": len(decisions),
                "p95_ms": _p95(time for time, _ in decisions),
                "p99_ms": _percentile((time for time, _ in decisions), .99),
                "auto_actions": sum(event == "auto_action" for _, event in decisions),
                "timeout_rate": timeouts / len(decisions) if decisions else None,
                "illegal_actions": illegal,
                "illegal_action_rate": illegal / len(decisions) if decisions else None,
            }
    print(json.dumps({
        "catalog": None if pools else str(args.cards), "testing_pools": len(pools),
        "first_seed": first_seed, "pairs": pairs, "frozen_suite": frozen["name"] if frozen else None,
        "battle_only": args.battle_only, "preview_only": args.preview_only,
        **({"lineup_only": True} if lineup_only else {"pipeline": True} if pipeline else {}),
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


def _bc_train(args):
    from .bc import train

    return train(args)


def _bc_verify(args):
    from .bc import verify

    return verify(args)


def _ppo_train(args):
    from .ppo import train

    return train(args)


def _ppo_verify(args):
    from .ppo import verify

    return verify(args)


def _intervention_train(args):
    from .intervention import train

    return train(args)


def _restore_audit(args):
    from .restoration import audit_restoration

    cards = json.loads(HISTORICAL_CARDS.read_text(encoding="utf-8"))
    report = audit_restoration(cards, args.seed, args.games)
    durations = report.pop("restore_ms")
    print(json.dumps(report | {"restore_p95_ms": _p95(durations), "restore_p99_ms": _percentile(durations, .99)}, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("bench", help="Play seeded games with swapped seats")
    bench.add_argument("--pairs", type=int, default=10)
    bench.add_argument("--seed", type=int, default=1)
    bench.add_argument("--pair-results", action="store_true", help="Include left scores by seed, first in seat p1 then p2")
    suite = bench.add_mutually_exclusive_group()
    suite.add_argument("--frozen", action="store_true", help="Use the fixed seed range and pools from data/frozen_benchmark.json")
    suite.add_argument("--frozen-suite", type=Path, help="Use a specific frozen benchmark manifest")
    source = bench.add_mutually_exclusive_group()
    source.add_argument("--cards", type=Path, default=HISTORICAL_CARDS)
    source.add_argument("--testing-pools", type=Path, nargs="+", help="Cycle through pools captured in Testing")
    bench.add_argument("--left", choices=BENCH_POLICIES, default="heuristic")
    bench.add_argument("--right", choices=(*BENCH_POLICIES, *ACE_POLICIES, "ppo-history"), default="random")
    bench.add_argument("--ace-trainer", type=Path, help="Local Ace Trainer checkout for pivot and pivot-base")
    bench.add_argument("--bc-checkpoint", type=Path, help="BC checkpoint directory used by the bc policy")
    bench.add_argument("--ppo-checkpoint", type=Path, help="Actor-critic checkpoint directory used by the ppo policy")
    bench.add_argument("--history-checkpoint", type=Path, help="Frozen actor-critic checkpoint used by the ppo-history opponent")
    bench.add_argument("--forecast-model", type=Path, help="Fitted JSON forecast configuration")
    bench.add_argument("--roster-checkpoint", type=Path, help="Draft and preview outcome model checkpoint")
    bench.add_argument("--roster-battle-checkpoint", type=Path, help="Replace Joint with a learned battle checkpoint for an ablation")
    bench.add_argument("--roster-draft", choices=("value", "prior", "heuristic"), default="value")
    bench.add_argument("--roster-opponent-draft", choices=("value", "prior", "heuristic"), help="Draft mode for the second roster policy")
    bench.add_argument("--roster-preview", choices=("rollout", "value", "value-rollout", "tactical"), default="rollout")
    bench.add_argument("--intervention-checkpoint", type=Path, help="Early-turn outcome reranker checkpoint")
    bench.add_argument("--intervention-threshold", type=float, help="Required predicted win gain for an override")
    bench.add_argument("--bc-device", choices=("cpu", "mps", "cuda"), default="cpu", help="Inference device for learned policies")
    bc_actions = bench.add_mutually_exclusive_group()
    bc_actions.add_argument("--bc-shortlist", action="store_true", dest="bc_shortlist", default=None,
                            help="Restrict BC inference to Tactical's shortlist")
    bc_actions.add_argument("--bc-full", action="store_false", dest="bc_shortlist", default=None,
                            help="Score all legal actions in a BC ablation")
    bench.add_argument("--value-audit", action="store_true", help="Compare PPO's opening value with the eventual match outcome")
    phase = bench.add_mutually_exclusive_group()
    phase.add_argument("--battle-only", action="store_true", help="Use the same heuristic for both seats' draft and lineup")
    phase.add_argument("--preview-only", action="store_true", help="Use the same draft and tactical battle policies for both seats")
    phase.add_argument("--lineup-only", action="store_true", help="Fix the draft and let each side control preview and battle")
    phase.add_argument("--pipeline", action="store_true", help="Allow draft, preview and battle policy selection")
    bench.set_defaults(run=_bench)
    forecast_fit = sub.add_parser("forecast-fit", help="Fit Joint-based forecast weights on a bounded development suite")
    forecast_fit.add_argument("--cards", type=Path, default=HISTORICAL_CARDS)
    forecast_fit.add_argument("--seed", type=int, default=180001)
    forecast_fit.add_argument("--pairs", type=int, default=32)
    forecast_fit.add_argument("--workers", type=int, default=6)
    forecast_fit.add_argument("--ace-trainer", type=Path, required=True)
    forecast_fit.add_argument("--output", type=Path, required=True)
    forecast_fit.set_defaults(run=fit_forecast)
    intervention_fit = sub.add_parser("intervention-train", help="Learn from matched early-turn actions and terminal outcomes")
    intervention_fit.add_argument("--cards", type=Path, default=HISTORICAL_CARDS)
    intervention_fit.add_argument("--ace-trainer", type=Path, required=True)
    intervention_fit.add_argument("--output", type=Path, required=True)
    intervention_fit.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    intervention_fit.add_argument("--seed", type=int, default=190001)
    intervention_fit.add_argument("--pairs", type=int, default=256)
    intervention_fit.add_argument("--validation-seed", type=int, default=200001)
    intervention_fit.add_argument("--validation-pairs", type=int, default=64)
    intervention_fit.add_argument("--workers", type=int, default=6)

    intervention_fit.set_defaults(run=_intervention_train)
    for command, handler in (("replay", _replay), ("catalog", _catalog), ("parity", _parity)):
        mode = sub.add_parser(command)
        mode.add_argument("traces", type=Path, nargs="+")
        mode.set_defaults(run=handler)
    fit = sub.add_parser("bc-train", help="Train a battle Action-Ranking Network on Joint demonstrations")
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--cards", type=Path, default=HISTORICAL_CARDS)
    fit.add_argument("--ace-trainer", type=Path, help="Include Pivot and Pivot-base as demonstration opponents")
    fit.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    fit.add_argument("--train-pairs", type=int, default=160)
    fit.add_argument("--test-pairs", type=int, default=40)
    fit.add_argument("--seed", type=int, default=81001)
    fit.add_argument("--test-seed", type=int, default=92001)
    fit.add_argument("--epochs", type=int, default=12)
    fit.add_argument("--batch-size", type=int, default=64)
    fit.add_argument("--variant", choices=("v1", "shortlist-hard", "history-hard", "history-soft"), default="v1")
    fit.set_defaults(run=_bc_train)
    verify = sub.add_parser("bc-verify", help="Check checkpoint logits and actions on held-out decisions")
    verify.add_argument("checkpoint", type=Path)
    verify.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    verify.set_defaults(run=_bc_verify)
    ppo = sub.add_parser("ppo-train", help="Train a battle actor-critic on parallel seeded Showdown games")
    ppo.add_argument("--cards", type=Path, default=HISTORICAL_CARDS)
    ppo.add_argument("--bc-init", type=Path, required=True)
    ppo.add_argument("--output", type=Path, required=True)
    ppo.add_argument("--ace-trainer", type=Path, required=True)
    ppo.add_argument("--resume", type=Path)
    ppo.add_argument("--shortlist", action="store_true", help="Use the BC Tactical shortlist for rollouts and inference")
    ppo.add_argument("--history-end", type=float, default=.35, help="Final fraction of pairs against frozen historical policies")
    ppo.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ppo.add_argument("--seed", type=int, default=110001)
    ppo.add_argument("--pairs", type=int, default=48)
    ppo.add_argument("--workers", type=int, default=6)
    ppo.add_argument("--total-updates", type=int, default=6)
    ppo.add_argument("--until-update", type=int)
    ppo.add_argument("--validation-seed", type=int, default=120001)
    ppo.add_argument("--validation-pairs", type=int, default=12)
    ppo.add_argument("--eval-every", type=int, default=2)
    ppo.add_argument("--final-suite", type=Path, default=HISTORICAL_CARDS.with_name("ppo_benchmark.json"))
    ppo.set_defaults(run=_ppo_train)
    ppo_verify = sub.add_parser("ppo-verify", help="Check portable policy logits, actions, and values")
    ppo_verify.add_argument("checkpoint", type=Path)
    ppo_verify.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    ppo_verify.set_defaults(run=_ppo_verify)
    audit = sub.add_parser("restore-audit", help="Check midgame Showdown round-trips on local fixtures")
    audit.add_argument("--seed", type=int, default=97501)
    audit.add_argument("--games", type=int, default=32)
    audit.set_defaults(run=_restore_audit)
    args = parser.parse_args()
    try:
        return args.run(args)
    except (OSError, RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
