"""Measure the frozen champion in complete, seat-swapped local matches."""

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import replace
from hashlib import sha256
from math import comb
from pathlib import Path
from random import Random
from statistics import mean, median
from time import perf_counter
import json

import numpy as np

from .__main__ import _percentile
from .combination_preview import preview_snapshot, terminal
from .draft_counterfactual import check_hybrid
from .draft_coverage import CHAMPION, Coverage
from .draft_last_pick import mean_ci, other, sample_game
from .draft_leaf import CATALOG
from .draft_league import disjoint, parallel, pools_for, read, save
from .draft_matchups import FROZEN, ROOT
from .game import BATTLE, BATTLE_BANK, DECISION_LIMIT, DRAFT, PREVIEW
from .joint import JointPolicy
from .lineup import LineupPolicy, index_cards
from .opening_active import check_fresh, inner_games
from .pivot import PivotOpponent
from .policy import heuristic
from .sim import SEATS, _draft, fingerprint, play_battle
from .tactics import SLOTS, tactical


EXTERNAL = ("ace-code", "pivot")
OPPONENTS = ("random", "heuristic", "tactical", *EXTERNAL)


class ExternalPivot(PivotOpponent):
    def __init__(self, checkout, name, cards):
        super().__init__(checkout, style=name == "pivot")
        self.code_only = name == "ace-code"
        self.cards = {card["card_id"]: card for card in cards}

    def setup(self, root, seat, first):
        self._call(dict(setup=True, seed=root, seat=seat, first=first, style=self.style, code_only=self.code_only))

    def choose_setup(self, decision):
        clock = {"decision_seconds_left": decision.seconds_left, "bank_seconds_left": decision.bank_seconds_left}
        observation = decision.observation.copy()
        if decision.phase == DRAFT:
            for field in ("picked", "opponent_picked"):
                observation[field] = [self.cards[card["card_id"]] for card in observation[field]]
            extra = {"legal_cards": [action["card_id"] for action in decision.actions]}
        else:
            extra = {"roster": list(dict.fromkeys(card for action in decision.actions for card in action["bring"]))}
        answer = self._call(dict(phase=decision.phase, observation=observation, clock=clock, **extra))["action"]
        if decision.phase == PREVIEW:
            # Canonicalize bring order while retaining both lead positions.
            action = next((action for action in decision.actions if set(action["bring"]) == set(answer["bring"])
                           and action["leads"] == answer["leads"]), None)
        else:
            action = answer
        return decision.validate(action)


class Champion:
    def __init__(self, experiment, indexed):
        self.experiment = experiment
        self.preview = LineupPolicy(experiment.profiles, indexed, experiment.preview)
        self.battle = JointPolicy(experiment.profiles)

    def __call__(self, decision):
        if decision.phase == DRAFT:
            return self.experiment.choose(decision, "champion")
        if decision.phase == PREVIEW:
            self.battle.observe(decision)
            return self.preview(decision)
        return self.battle(decision)


def opponent(name, root, profiles, pivot):
    if name == "random":
        # Separate phase streams from each other and from the card pool and battle RNG.
        streams = {phase: Random(f"strength-v1/{root}/{phase}/opponent") for phase in DECISION_LIMIT}
        return lambda decision: streams[decision.phase].choice(decision.actions)
    if name == "heuristic":
        return heuristic
    if name == "tactical":
        return lambda decision: tactical(decision, card_profiles=profiles)
    return pivot


def play(experiment, cards, root, actor, rival, foe, indexed, candidate=None):
    pool, first, battle_seed = sample_game(cards, root)
    champion = candidate if candidate is not None else Champion(experiment, indexed)
    policies = {actor: champion, other(actor): foe}
    banks = dict.fromkeys(SEATS, BATTLE_BANK)
    clocks = {player: defaultdict(list) for player in policies}
    counts = {player: Counter() for player in policies}
    trace, lineups, faint_turns = [], {}, {}
    timeout_rng = Random(f"strength-v1/{root}/clock")

    def choose(seat, decision, view=None):
        phase = decision.phase
        left = min(DECISION_LIMIT[phase], banks[seat]) if phase == BATTLE else DECISION_LIMIT[phase]
        decision = replace(decision, seconds_left=left, bank_seconds_left=banks[seat] if phase == BATTLE else None)
        before = perf_counter()
        policy = policies[seat]
        if isinstance(policy, PivotOpponent):
            action = policy.choose(decision, view) if phase == BATTLE else policy.choose_setup(decision)
        elif phase == BATTLE and hasattr(policy, "choose_with_public_history"):
            action = policy.choose_with_public_history(decision, seat, view.get("search"))
        else:
            action = policy(decision)
        elapsed = perf_counter() - before
        clocks[seat][phase].append(elapsed * 1000)
        if phase == BATTLE:
            banks[seat] -= min(left, elapsed)
            if banks[seat] <= 0:
                counts[seat]["bank_forfeit"] += 1
                return None
        if elapsed >= left:
            counts[seat][f"{phase}_timeout"] += 1
            action = timeout_rng.choice(decision.actions)
        action = decision.validate(action)
        counts[seat][phase] += 1
        if phase == PREVIEW:
            lineups[seat] = action
        elif phase == BATTLE:
            board = decision.observation
            own_alive = sum(bool(mon) for mon in view["raw"]["active_pokemon"] + view["raw"]["team"])
            if own_alive < 4:
                faint_turns.setdefault(seat, board["turn"])
            atoms = [action[slot] for slot in SLOTS]
            counts[seat].update(atom["type"] for atom in atoms)
            counts[seat]["voluntary_switch"] += sum(atom["type"] == "switch" for atom in atoms) if all(board["active"]) else 0
            trace.append(dict(seat=seat, turn=board["turn"], own_alive=own_alive,
                              active=[(mon["species"], round(mon.get("hp", 1), 3)) if mon else None for mon in board["active"]],
                              enemies=[(mon["species"], round(mon.get("hp", 1), 3)) if mon else None for mon in board["opponents"]],
                              conditions=board["side_conditions"], fields=board["fields"], weather=board["weather"], action=action))
        return action

    if isinstance(foe, PivotOpponent):
        foe.setup(root, other(actor), first)
    rosters = _draft(pool, first, choose)
    winner = play_battle(experiment.league.engine, rosters, policies, root, battle_seed, choose)
    if any(counts[seat][DRAFT] != 6 or counts[seat][PREVIEW] != 1 or not counts[seat][BATTLE] for seat in SEATS):
        raise RuntimeError("Benchmark game did not cover all three phases on both sides")
    first_faint = min(faint_turns.items(), key=lambda item: item[1]) if faint_turns else None
    if len(faint_turns) == 2 and len(set(faint_turns.values())) == 1:
        first_faint = ("both", first_faint[1])
    return dict(actor=actor, winner=winner, first=first, battle_seed=battle_seed, opponent=rival,
                roster={seat: [mon["card_id"] for mon in roster] for seat, roster in rosters.items()}, lineups=lineups,
                last_turn=max(row["turn"] for row in trace), first_observed_faint=first_faint,
                counts={seat: dict(totals) for seat, totals in counts.items()}, clocks=clocks,
                champion_preview_fallbacks=champion.preview.fallbacks,
                planning=dict(getattr(champion, "planning_counts", {})),
                opponent_fallbacks=foe.fallbacks if isinstance(foe, PivotOpponent) else 0,
                trace=trace if winner != actor or root % 10 == 0 else [])


def evaluate(roots, cards, rivals, checkout):
    records = {}
    with ExitStack() as stack:
        experiment = Coverage(stack, cards)
        indexed = index_cards(cards)
        external = {name: stack.enter_context(ExternalPivot(checkout, name, cards)) for name in rivals if name in EXTERNAL}
        for root in roots:
            records[root] = {}
            for rival in rivals:
                records[root][rival] = [play(experiment, cards, root, actor, rival,
                                              opponent(rival, root, experiment.profiles, external.get(rival)), indexed)
                                         for actor in SEATS]
        counts = inner_games(experiment)
        counts.update(champion_pick12_starts=experiment.activity["other_champion_starts"],
                      champion_pick12_budget_fallbacks=experiment.activity["full_budget_fallbacks"])
    return records, counts


def phase_controls(roots, cards, records):
    """Swap the external choices one phase at a time on the recorded deployments."""
    measures = {}
    by_id = {card["card_id"]: card for card in cards}
    with ExitStack() as stack:
        experiment = Coverage(stack, cards)
        indexed = index_cards(cards)
        for root in roots:
            result = measures[root] = {}
            for rival, pair in records[str(root)].items():
                scores = result[rival] = []
                for game in pair:
                    actor = game["actor"]
                    foe = other(actor)
                    rosters = {seat: [by_id[card] for card in ids] for seat, ids in game["roster"].items()}
                    seed = game["battle_seed"]
                    own, theirs = (game["lineups"][seat] for seat in (actor, foe))
                    view = preview_snapshot(rosters, seed, experiment.league.engine)[foe]
                    shared_preview = view.validate(LineupPolicy(experiment.profiles, indexed, experiment.preview)(view))
                    actual = .5 if game["winner"] is None else float(game["winner"] == actor)
                    joint = terminal(experiment, rosters, actor, seed, own, theirs)
                    shared = terminal(experiment, rosters, actor, seed, own, shared_preview)
                    scores.append(dict(external=actual, same_lineups_joint=joint, shared_preview_joint=shared))
    return measures


def isolate(args, cards):
    source = read(args.isolate)
    roots = list(map(int, source["records"]))
    began = perf_counter()
    measures = {}
    for result in parallel(phase_controls, roots, args.workers, cards, source["records"]):
        measures.update(result)
    summary = {}
    for rival in source["report"]:
        pairs = [by_rival[rival] for by_rival in measures.values()]
        stages = ("external", "same_lineups_joint", "shared_preview_joint")
        counts = {stage: sum(game[stage] for pair in pairs for game in pair) for stage in stages}
        comparison = {}
        for lo, hi in zip(stages, stages[1:]):
            values = [mean(game[hi] - game[lo] for game in pair) for pair in pairs]
            comparison[hi] = dict(gain=mean(values), ci95=mean_ci(values))
        summary[rival] = dict(counts=counts, comparisons=comparison, games=2 * len(pairs))
    save(args.output, dict(source=str(args.isolate), report=summary, seconds=perf_counter()-began, measures=measures))
    print(json.dumps(summary), flush=True)


def exact_lower(successes, count, alpha):
    if not successes:
        return 0.
    coefficients = [(i, comb(count, i)) for i in range(successes, count + 1)]
    low, high = 0., 1.
    for _ in range(48):
        midpoint = (low + high) / 2
        tail = sum(c * midpoint**i * (1 - midpoint)**(count - i) for i, c in coefficients)
        if tail < alpha:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2


def summarize(pairs):
    rows = [game for pair in pairs for game in pair]
    wins = [sum(game["winner"] == game["actor"] for game in pair) for pair in pairs]
    win_rate = sum(wins) / len(rows)
    losses = [g for g in rows if g["winner"] is not None and g["winner"] != g["actor"]]
    champion_ms = {phase: [ms for g in rows for ms in g["clocks"][g["actor"]][phase]] for phase in DECISION_LIMIT}
    champion_counts = Counter()
    foe_counts = Counter()
    for game in rows:
        champion_counts.update(game["counts"][game["actor"]])
        foe_counts.update(game["counts"][other(game["actor"])])
    # The event of winning both seats bounds the mean game win rate from below;
    # the event of winning either bounds it from above. Each pool is one unit.
    interval = [exact_lower(wins.count(2), len(pairs), .025), 1 - exact_lower(wins.count(0), len(pairs), .025)]
    seats = {phase: [sum(g["winner"] == g["actor"] for g in group), len(group)]
             for phase, group in (("draft_first", [g for g in rows if g["first"] == g["actor"]]),
                                  ("draft_second", [g for g in rows if g["first"] != g["actor"]]))}
    return dict(pairs=len(pairs), games=len(rows), wins=sum(wins), losses=len(losses), draws=len(rows)-sum(wins)-len(losses),
                win_rate=win_rate, paired_bootstrap_95=mean_ci([v / 2 for v in wins]), exact_cluster_95=interval,
                exact_one_sided_lower_95=exact_lower(wins.count(2), len(pairs), .05),
                pairs_won_both=wins.count(2), pairs_won_one=wins.count(1), pairs_won_neither=wins.count(0),
                seats=seats, median_turns=median(g["last_turn"] for g in rows),
                first_faint=dict(Counter("both" if g["first_observed_faint"][0] == "both" else
                                         "champion" if g["first_observed_faint"][0] == g["actor"] else "opponent"
                                         for g in losses if g["first_observed_faint"])),
                chair_clocks={phase: {"p95_ms": _percentile(times, .95), "p99_ms": _percentile(times, .99)}
                              for phase, times in champion_ms.items()},
                champion_actions=dict(champion_counts), opponent_actions=dict(foe_counts),
                opponent_fallbacks=sum(g["opponent_fallbacks"] for g in rows), illegal_actions=0)


def verify_pools(pools, exclusions):
    check_fresh(pools, exclusions)
    for group in ("opening-active-v1", "battle-league-v1", "battle-value-v1", "strength-v1"):
        for path in (ROOT / "benchmark_results" / group).glob("*"):
            if path.suffix == ".json":
                previous = read(path).get("pools", {})
            elif path.suffix == ".npz":
                with np.load(path, allow_pickle=False) as data:
                    previous = json.loads(data["metadata"].item())["pools"]
            else:
                continue
            disjoint(pools, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opponents", choices=OPPONENTS, nargs="+")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--pairs", type=int)
    parser.add_argument("--isolate", type=Path, help="Compare phases on an already completed benchmark")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--ace-trainer", type=Path)
    parser.add_argument("--exclude", nargs="*", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.workers < 1:
        parser.error("Use a fresh output path and positive worker count")
    check_hybrid()
    cards = read(CATALOG)
    if args.isolate:
        isolate(args, cards)
        return
    if args.pairs is None or args.pairs < 1 or args.seed is None or not args.opponents or len(set(args.opponents)) != len(args.opponents):
        parser.error("Use a fresh output path, positive sizes, and unique opponents")
    if any(name in EXTERNAL for name in args.opponents) and not args.ace_trainer:
        parser.error("Provide the pinned external Ace Trainer checkout for Pivot")
    roots = list(range(args.seed, args.seed + args.pairs))
    pools = pools_for(cards, roots)
    verify_pools(pools, args.exclude)
    Random(args.seed ^ 0x762E).shuffle(roots)
    began = perf_counter()
    records, counts = {}, Counter()
    for part, stats in parallel(evaluate, roots, args.workers, cards, args.opponents, args.ace_trainer):
        records.update(part)
        counts.update(stats)
        print(f"Completed {len(records)}/{args.pairs} paired pools", flush=True)
    if len(records) != args.pairs:
        raise RuntimeError("Missing paired benchmark games")
    records = dict(sorted(records.items()))
    report = {name: summarize([row[name] for row in records.values()]) for name in args.opponents}
    spec = {"draft": CHAMPION, "preview": "LineupPolicy", "battle": "JointPolicy", "draft_weights": FROZEN,
            "catalog": fingerprint(cards)}
    external = {}
    if args.ace_trainer:
        external = {name: sha256((args.ace_trainer / name).read_bytes()).hexdigest()
                    for name in ("agent/agent.py", "agent/versions/pivot.py", "agent/pokemon/tuning.py", "sim/translate.py")}
    save(args.output, dict(report=report, spec=spec, external=external, seconds=perf_counter()-began,
                           pools=pools, records=records, counts=dict(counts)))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
