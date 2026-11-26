"""Learn the terminal value of legal joint actions across battle turns."""

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from .draft_counterfactual import check_hybrid
from .draft_coverage import CHAMPION, Coverage
from .draft_early import paired_draws
from .draft_last_pick import mean_ci, other
from .draft_leaf import CATALOG
from .draft_league import parallel, pools_for, read, save
from .draft_matchups import ROOT
from .features import BattleEncoder
from .forecast import ForecastConfig, ForecastPolicy
from .game import BATTLE, PREVIEW, make_decision
from .intervention import _encode
from .joint import BattleHistory, JointPolicy
from .lineup import index_cards
from .opening_active import check_fresh, inner_games
from .opening_audit import TacticalOpponent
from .opening_value import opening_rosters
from .sim import FORMAT, SEATS, Showdown, fingerprint
from .tactics import SLOTS, tactical_actions


RIVALS = ("champion", "search", "pairwise")
STYLES = ("joint", "forecast", "tactical")
FEATURE_VERSION = 1
POSITION = ForecastConfig(position=.55)
SEARCH_REPLY = ForecastConfig(candidates=8, position=.35, replies=True, risk=.20)


def opponent(profiles, style):
    if style == "joint":
        return JointPolicy(profiles)
    if style == "tactical":
        return TacticalOpponent(profiles)
    if style in ("forecast", "forecast-replies"):
        return ForecastPolicy(profiles, POSITION if style == "forecast" else SEARCH_REPLY)
    raise ValueError(f"Unknown Battle style: {style}")


def switch_plan(action):
    return tuple(action[slot].get("species") if action[slot]["type"] == "switch" else None for slot in SLOTS)


def candidates(decision, ranked):
    if len(ranked) < 3 or any(not mon for mon in decision.observation["active"]):
        return ()
    anchor = ranked[0]
    signature = switch_plan(anchor[0])
    changed = next((entry for entry in ranked[1:] if switch_plan(entry[0]) != signature), None)
    if changed is None:
        return ()
    third = next(entry for entry in ranked[1:] if entry != changed)
    return anchor, changed, third


def wider_candidates(decision, ranked, profiles):
    if len(ranked) < 3 or any(not mon for mon in decision.observation["active"]):
        return ()
    base = [action for action, _ in ranked]
    other_actions = [action for action in tactical_actions(decision, card_profiles=profiles) if action not in base]
    plans = {switch_plan(action) for action in base}
    new_switch = next((action for action in other_actions if any(switch_plan(action)) and switch_plan(action) not in plans), None)
    if new_switch is not None:
        other_actions = [new_switch, *(action for action in other_actions if action != new_switch)]
    extras = [(action, {}) for action in other_actions[:2]]
    return tuple(ranked + extras) if extras else ()


def band(turn):
    return 0 if turn == 1 else 1 if turn <= 3 else 2


def encode(encoder, policy, decision, ranked):
    state, actions = _encode(encoder, policy.history, decision, ranked)
    # The incoming card is available in the player's own selected team.
    # Identity helps distinguish similar typings with different exit abilities or move sets.
    board = decision.observation
    own = [mon.get("species") if mon else None for mon in board["active"]]
    context = [float(species in board["reserves"]) for species in encoder.species]
    context.extend(float(species == mon) for mon in own for species in encoder.species)
    extra = [[float(species == target) for target in switch_plan(action) for species in encoder.species]
             for action, _ in ranked]
    return torch.cat((state, torch.tensor(context))), torch.cat((actions, torch.tensor(extra)), dim=1)


def decisions(reply):
    return [(view["seat"], make_decision(view["phase"], view["raw"], view["templates"], view["seat"]))
            for view in reply["views"]]


def advance(engine, reply, policies, *, override=None, lineups=None):
    for seat, decision in decisions(reply):
        policy = policies[seat]
        if decision.phase == PREVIEW:
            policy.observe(decision)
            action = lineups[seat]
        elif override is not None and seat == override[0]:
            policy.observe(decision)
            action = override[1]
            if record_override := getattr(policy, "record_override", None):
                record_override(decision, action)
            policy.record(decision, action)
        else:
            action = policy(decision)
        reply = engine.call("choose", seat=seat, action=decision.validate(action))
    return reply


def terminal(engine, reply, policies, actor, action):
    reply = advance(engine, reply, policies, override=(actor, action))
    for _ in range(499):
        if reply["ended"]:
            return reply["winner"]
        reply = advance(engine, reply, policies)
    raise RuntimeError("Label continuation exceeded 500 decisions")


def collect(seeds, cards, draws, wide):
    torch.set_num_threads(1)
    rows, counts = [], Counter()
    with ExitStack() as stack:
        experiment = Coverage(stack, cards)
        engine = stack.enter_context(Showdown(counterfactual_labels=True))
        encoder, indexed = BattleEncoder(experiment.profiles), index_cards(cards)
        for root in seeds:
            actor, rival = SEATS[root % 2], RIVALS[root // 2 % len(RIVALS)]
            prefix = STYLES[root // 6 % len(STYLES)]
            rosters, actual, lineups, _ = opening_rosters(experiment, indexed, cards, root, actor, rival)
            policies = {actor: JointPolicy(experiment.profiles), other(actor): opponent(experiment.profiles, prefix)}
            reply = engine.call("start", format=FORMAT, seed=actual, rosters=rosters)
            reviewed, reference_winners = set(), []
            for _ in range(500):
                if reply["ended"]:
                    break
                pending = dict(decisions(reply))
                own = pending.get(actor)
                if len(pending) == 2 and own.phase == BATTLE:
                    turn = own.observation["turn"]
                    stage = band(turn)
                    if stage not in reviewed and turn <= 12 and (not wide or turn >= 2):
                        base = policies[actor].rank(own)
                        ranked = wider_candidates(own, base, experiment.profiles) if wide else candidates(own, base)
                        if ranked:
                            # All private simulator state stays in this offline worker. Policy code keeps its player view.
                            saved = engine.call("label_root")
                            if saved != reply:
                                raise RuntimeError("Saving the root changed its player views")
                            histories = {seat: policy.history.snapshot() for seat, policy in policies.items()}
                            features = {}
                            if not wide:
                                state, actions = encode(encoder, policies[actor], own, ranked)
                                features = dict(state=state.tolist(), actions=actions.tolist())
                            menu = [own.validate(action) for action, _ in ranked]

                            def run(style, i, seed=None):
                                restored = engine.call("label_restore", seed=seed)
                                if restored != saved:
                                    raise RuntimeError(f"Restoring changed views at pool {root}, turn {turn}")
                                players = {actor: JointPolicy(experiment.profiles),
                                           other(actor): opponent(experiment.profiles, style)}
                                for seat, policy in players.items():
                                    policy.history = BattleHistory.from_snapshot(histories[seat])
                                return terminal(engine, restored, players, actor, menu[i])

                            reference_winners.append(run(prefix, 0))
                            outcomes = []
                            seeds_by_style = paired_draws(root * 32 + turn, actual, (), len(STYLES), draws)
                            for style, group in zip(STYLES, seeds_by_style, strict=True):
                                outcomes.append([[.5 if (winner := run(style, i, seed)) is None else float(winner == actor)
                                                  for i in range(len(menu))] for seed in group])
                            reply = engine.call("label_restore")
                            if reply != saved:
                                raise RuntimeError("Original root was not restored")
                            rows.append(dict(root=root, turn=turn, actor=actor, rival=rival, prefix=prefix,
                                             fingerprint=fingerprint(own.snapshot()), menu=menu, outcomes=outcomes,
                                             base_count=len(base) if wide else len(menu), **features))
                            reviewed.add(stage)
                            counts["labeled_continuations"] += len(STYLES) * draws * len(menu)
                reply = advance(engine, reply, policies, lineups=lineups)
            if not reply["ended"] or any(result != reply["winner"] for result in reference_winners):
                raise RuntimeError(f"Original seeded continuation diverged at pool {root}")
            counts["natural_prefix_games"] += 1
            counts["original_terminal_checks"] += len(reference_winners)
        counts.update(inner_games(experiment))
    return rows, counts


def audit(rows):
    results = defaultdict(lambda: defaultdict(list))
    chosen = Counter()
    for row in rows:
        rewards = row["outcomes"]
        half = len(rewards[0]) // 2
        changed = [i for i, action in enumerate(row["menu"]) if switch_plan(action) != switch_plan(row["menu"][0])]
        for kind, choices in (("switch", [0, *changed]), ("value", range(3))):
            # Select on Joint and Forecast; leave Tactical and the rest of the seeds out.
            best = max(choices, key=lambda i: mean(y[i] for style in rewards[:2] for y in style[:half]))
            for split, values in (("retained", [y for style in rewards[:2] for y in style[half:]]),
                                  ("heldout_style", rewards[2])):
                gain = mean(y[best] - y[0] for y in values)
                for group in ("all", ("turn_1", "turn_2_3", "turn_4_plus")[band(row["turn"])]):
                    results[(kind, split, group)][row["root"]].append(gain)
            chosen[(kind, "overrides")] += best != 0
    return {"/".join(key): {"positions": sum(map(len, values.values())), "pools": len(values), "delta": mean(map(mean, values.values())),
                            "ci95": mean_ci([mean(gains) for gains in values.values()])}
            for key, values in results.items()} | {"/".join(key): val for key, val in chosen.items()}


def wide_audit(rows, *, equal_budget=False):
    sections = defaultdict(lambda: defaultdict(list))
    outside = Counter()
    for row in rows:
        rewards = row["outcomes"]
        half = len(rewards[0]) // 2
        discovery = [y for style in rewards[:2] for y in style[:half]]
        estimates = [mean(y[i] for y in discovery) for i in range(len(row["menu"]))]
        bins = ("all", "turn_2_3" if row["turn"] <= 3 else "turn_4_plus")
        baseline = row["base_count"]
        tested = len(estimates)
        challengers = [*range(2 * baseline - tested), *range(baseline, tested)] if equal_budget else range(tested)
        select_base = max(range(baseline), key=estimates.__getitem__)
        select_wide = max(challengers, key=estimates.__getitem__)
        for group in bins:
            outside[group] += select_wide >= row["base_count"]
            for split, values in (("retained", [y for style in rewards[:2] for y in style[half:]]), ("heldout_style", rewards[2])):
                sections[(group, split)][row["root"]].append(mean(y[select_wide] - y[select_base] for y in values))
    report = {}
    for (group, split), pools in sections.items():
        gains = [mean(v) for v in pools.values()]
        report[f"{group}/{split}"] = dict(positions=sum(map(len, pools.values())), pools=len(pools), outside=outside[group],
                                           delta=mean(gains), ci95=mean_ci(gains))
    return report


def generate(args, cards):
    roots = range(args.seed, args.seed + args.count)
    pools = pools_for(cards, roots)
    previous = [p for p in (ROOT / "benchmark_results/opening-active-v1").glob("*.json") if "pools" in read(p)]
    check_fresh(pools, [*previous, *args.exclude])
    rows, counts = [], Counter()
    began = perf_counter()
    for positions, stats in parallel(collect, roots, args.workers, cards, args.draws, args.wide_candidates):
        rows.extend(positions)
        counts.update(stats)
    rows.sort(key=lambda row: (row["root"], row["turn"]))
    report = dict(seconds=perf_counter() - began, rows=len(rows), counts=dict(counts),
                  audit=wide_audit(rows) if args.wide_candidates else audit(rows))
    if args.wide_candidates:
        report["equal_budget"] = wide_audit(rows, equal_budget=True)
    save(args.output, dict(report=report, pools=pools, rows=rows, champion=CHAMPION, catalog=fingerprint(cards),
                           styles=STYLES, version=FEATURE_VERSION, draws=args.draws))
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sample = sub.add_parser("collect")
    sample.add_argument("--seed", type=int, required=True)
    sample.add_argument("--count", type=int, required=True)
    sample.add_argument("--workers", type=int, default=12)
    sample.add_argument("--draws", type=int, default=6)
    sample.add_argument("--wide-candidates", action="store_true")
    sample.add_argument("--exclude", type=Path, nargs="*", default=[])
    sample.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or min(args.count, args.workers) < 1 or args.draws < 4:
        parser.error("Use a fresh path, positive sizes and at least four paired draws")
    check_hybrid()
    generate(args, read(CATALOG))


if __name__ == "__main__":
    main()
