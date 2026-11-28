"""Audit first-turn Battle alternatives with champion lineups and paired outcomes."""

import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

from .combination_preview import draft_root, preview_snapshot
from .draft_counterfactual import check_hybrid
from .draft_coverage import CHAMPION, Coverage
from .draft_early import paired_draws
from .draft_last_pick import mean_ci, other
from .draft_leaf import CATALOG, check_pools
from .draft_league import parallel, pools_for, read, save
from .forecast import ForecastConfig, ForecastPolicy
from .game import BATTLE, PREVIEW
from .joint import JointPolicy
from .lineup import index_cards, LineupPolicy
from .sim import SEATS, fingerprint, play_battle
from .tactics import tactical


class MissingOpening(ValueError):
    pass


class TacticalOpponent(JointPolicy):
    def __call__(self, decision):
        return tactical(decision, card_profiles=self.profiles)


def trial(experiment, rosters, lineups, actor, battle_seed, player, branch, *, strict=True, timings=None):
    policies = {seat: JointPolicy(experiment.profiles) for seat in SEATS}
    if player in ("forecast", "forecast-replies"):
        config = (ForecastConfig(position=.55) if player == "forecast" else
                  ForecastConfig(candidates=8, position=.35, replies=True, risk=.20))
        policies[other(actor)] = ForecastPolicy(experiment.profiles, config)
    elif player == "tactical":
        policies[other(actor)] = TacticalOpponent(experiment.profiles)
    elif player != "joint":
        raise ValueError(f"Unknown opening opponent {player}")
    opening = None

    def choose(seat, decision):
        nonlocal opening
        policy = policies[seat]
        if decision.phase == PREVIEW:
            policy.observe(decision)
            return decision.validate(lineups[seat])
        if seat != actor or opening is not None:
            return decision.validate(policy(decision))
        first_turn = decision.phase == BATTLE and decision.observation.get("turn") == 1
        if strict and not first_turn:
            raise MissingOpening("No opening battle decision")
        candidates = policy.rank(decision)[:3]
        ranked = [action for action, _ in candidates]
        if strict and len(ranked) < 2:
            raise MissingOpening("No opening alternative")
        opening = (fingerprint([decision.observation, decision.move_info, ranked]), ranked)
        if callable(branch):
            index = branch(policy, decision, candidates) if first_turn and len(ranked) == 3 else 0
        else:
            index = branch
        move = decision.validate(ranked[index] if 0 <= index < len(ranked) else ranked[0])
        policy.record(decision, move)
        return move

    def measured(seat, decision, _):
        started = perf_counter()
        action = choose(seat, decision)
        if timings is not None and seat == actor and decision.phase == BATTLE:
            timings.append((perf_counter() - started) * 1000)
        return action

    winner = play_battle(experiment.league.engine, rosters, policies, 0, battle_seed, measured)
    if strict and opening is None:
        raise MissingOpening("Battle ended before the opening")
    return opening, .5 if winner is None else float(winner == actor)


def collect(seeds, cards, draws):
    rows, issues, events = [], [], Counter()
    with ExitStack() as stack:
        experiment = Coverage(stack, cards)
        indexed = index_cards(cards)
        for root in seeds:
            actor = SEATS[root % 2]
            opponent = ("champion", "search", "pairwise")[root // 2 % 3]
            try:
                rosters, actual, _ = draft_root(experiment, root, cards, {actor: "champion", other(actor): opponent})
                previews = preview_snapshot(rosters, actual, experiment.league.engine)
                lineups = {seat: previews[seat].validate(LineupPolicy(experiment.profiles, indexed, experiment.preview)(previews[seat]))
                           for seat in SEATS}
                battles = paired_draws(root, actual, (), 2, draws)
                reference = None
                rewards = []
                for style, seeds_by_style in zip(("joint", "forecast"), battles, strict=True):
                    variants = []
                    for seed in seeds_by_style:
                        opening, score = trial(experiment, rosters, lineups, actor, seed, style, 0)
                        events["terminal_games"] += 1
                        if reference is None:
                            reference = opening
                        if reference != opening:
                            raise MissingOpening("Public opening changed across seeds")
                        outcomes = [score]
                        for branch in range(1, len(opening[1])):
                            repeat, value = trial(experiment, rosters, lineups, actor, seed, style, branch)
                            events["terminal_games"] += 1
                            if opening != repeat:
                                raise ValueError("Action branch diverged before intervention")
                            outcomes.append(value)
                        variants.append(outcomes)
                    rewards.append(variants)
                rows.append({"seed": root, "seat": actor, "draft_opponent": opponent,
                             "key": reference[0], "actions": reference[1], "outcomes": rewards})
            except MissingOpening as error:
                events[str(error)] += 1
            except (ValueError, OSError, RuntimeError, KeyError) as error:
                issues.append({"seed": root, "error": str(error)})
        events["preview_and_draft_inner_games"] = experiment.preview.games
        events["champion_review_games"] = experiment.activity["full_leaf_games"]
        events["prefix_rollout_games"] = (experiment.hybrid.rollout.activity["rollout_games"]
                                         + experiment.league.rollout.activity["rollout_games"])
    return rows, issues, events


def summarize(rows):
    retained, alternative, edits = [], [], 0
    for row in rows:
        draws = row["outcomes"]
        cut = len(draws[0]) // 2
        i = max(range(len(row["actions"])), key=lambda k: mean(results[k] for style in draws for results in style[:cut]))
        edits += i != 0
        retained.append(mean(results[i] - results[0] for style in draws for results in style[cut:]))
        j = max(range(len(row["actions"])), key=lambda k: mean(results[k] for results in draws[0][:cut]))
        alternative.append(mean(results[j] - results[0] for results in draws[1][cut:]))
    return {"different_openings": edits, "heldout_delta": mean(retained), "heldout_ci95": mean_ci(retained),
            "unseen_battle_opponent_delta": mean(alternative), "unseen_battle_opponent_ci95": mean_ci(alternative)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--exclude", nargs="*", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or min(args.count, args.workers) < 1 or args.draws < 4:
        parser.error("Use fresh output, positive counts and at least four paired draws")
    check_hybrid()
    cards = read(CATALOG)
    seeds = range(args.seed, args.seed + args.count)
    pools = pools_for(cards, seeds)
    check_pools(pools, args.exclude)
    started = perf_counter()
    rows, errors, events = [], [], Counter()
    for results, issues, totals in parallel(collect, seeds, args.workers, cards, args.draws):
        rows.extend(results)
        errors.extend(issues)
        events.update(totals)
    if not rows:
        raise RuntimeError(f"No fixed opening states: {errors[:4]}")
    rows.sort(key=lambda row: row["seed"])
    report = {"seconds": perf_counter() - started, "positions": len(rows), "workers": args.workers,
              "errors": errors, "counts": dict(events), **summarize(rows)}
    save(args.output, {"report": report, "pools": pools, "champion": CHAMPION, "rows": rows})
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
