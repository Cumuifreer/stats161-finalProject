"""Compare standalone Battle models with fixed Draft and Preview across a policy population."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from multiprocessing import get_context
from pathlib import Path
from statistics import mean
from time import perf_counter

from ..battle_league import BATTLE_LEAGUE, Encoder, LearnedPolicy, configure_torch, load_model
from ..battle_value import opponent
from ..blind85 import used_pools
from ..draft_last_pick import mean_ci
from ..draft_leaf import CATALOG
from ..draft_league import pools_for, read, save
from ..game import BATTLE
from ..joint import JointPolicy
from ..sim import SEATS, Showdown, play_game
from ..tactics import load_card_profiles


class Learner(LearnedPolicy):
    def __call__(self, decision):
        self.observe(decision)
        return super().__call__(decision)


def evaluate(roots, cards, candidates, rivals):
    configure_torch()
    profiles = load_card_profiles(cards, with_sets=True)
    encoder = Encoder(profiles)
    models = [load_model(candidate, cards) for candidate in candidates]
    measurements = {}
    with Showdown() as engine:
        for root in roots:
            measurements[root] = {}
            for name in rivals:
                controls = []
                for model in (None, *models):
                    pair = []
                    for actor, foe in (SEATS, SEATS[::-1]):
                        learner = Learner(profiles, encoder, model, value_only=True) if model else JointPolicy(profiles)
                        rival = opponent(profiles, name)
                        samples = defaultdict(list)
                        winner = play_game(engine, cards, {actor: learner, foe: rival}, root, samples)
                        if any(event for times in samples.values() for _, _, event in times):
                            raise RuntimeError(f"Clock event in qualification root {root}")
                        pair.append(dict(won=winner == actor, drew=winner is None,
                                         moving_ms=[ms for phase, ms, _ in samples[actor] if phase == BATTLE],
                                         chose_diff=getattr(learner, "changes", 0),
                                         n_scored=getattr(learner, "scored", 0)))
                    controls.append(pair)
                measurements[root][name] = controls
    return measurements


def summarize(rows, index, name):
    games = [game for row in rows for game in row[name][index]]
    rates = [mean(match["won"] for match in row[name][index]) for row in rows]
    baseline = [mean(match["won"] for match in row[name][0]) for row in rows]
    clocks = sorted(ms for game in games for ms in game["moving_ms"])
    return dict(games=len(games), wins=sum(game["won"] for game in games), draws=sum(game["drew"] for game in games),
                win_rate=mean(rates), interval=mean_ci(rates), paired_improvement=mean(a-b for a,b in zip(rates, baseline, strict=True)),
                improvement_interval=mean_ci([a-b for a,b in zip(rates, baseline, strict=True)]),
                battle_ms_p95=clocks[int(.95 * len(clocks))], decisions_scored=sum(game["n_scored"] for game in games),
                decisions_changed=sum(game["chose_diff"] for game in games))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pools", type=int, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--rivals", choices=BATTLE_LEAGUE, nargs="+", default=["forecast", "forecast-replies"])
    args = parser.parse_args()
    if args.output.exists() or args.pools < 4 or args.workers < 1:
        parser.error("Provide a fresh directory and positive sample counts")
    cards = read(CATALOG)
    roots = list(range(args.seed, args.seed + args.pools))
    pools = pools_for(cards, roots)
    if not used_pools(cards).isdisjoint(pools.values()):
        raise ValueError("Qualification needs fresh pools")
    save(args.output / "sample.json", {"pools": pools, "rivals": args.rivals,
                                     "candidates": {str(p): sha256((p / "state_dict.pt").read_bytes()).hexdigest()
                                                    for p in args.candidate}})
    started = perf_counter()
    outcomes = {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as workers:
        for result in workers.map(evaluate, [roots[slot::args.workers] for slot in range(args.workers)],
                                  [cards] * args.workers, [args.candidate] * args.workers, [args.rivals] * args.workers):
            outcomes.update(result)
    report = {str(name): {rival: summarize(list(outcomes.values()), index, rival) for rival in args.rivals}
              for index, name in enumerate(("Joint", *args.candidate))}
    save(args.output / "matches.json", dict(pools=pools, outcomes=outcomes, report=report, wall_seconds=perf_counter() - started))
    print(report, flush=True)


if __name__ == "__main__":
    main()
