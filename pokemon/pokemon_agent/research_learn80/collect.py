"""Generate terminal full-game trajectories from a population of legal policies."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from hashlib import sha256
from multiprocessing import get_context
from pathlib import Path
from random import Random
from time import perf_counter

import numpy as np

from ..battle_league import BATTLE_LEAGUE, FIELDS, WIDTH, Encoder, LearnedPolicy, configure_torch, load_model
from ..battle_value import opponent
from ..blind85 import used_pools
from ..draft_leaf import CATALOG
from ..draft_league import pools_for, read, save
from ..game import BATTLE, DRAFT, PREVIEW
from ..joint import JointPolicy
from ..sim import DRAFT_SIZE, SEATS, Showdown, fingerprint, play_game
from ..tactics import load_card_profiles


FLOAT_FIELDS = {"state", "actions", "mons", "outcome", "branches", "propensity"}
BOOL_FIELDS = {"legal", "forced"}
FIELDS = (*FIELDS, "propensity", "style", "self_play")


class Recorder(JointPolicy):
    def __init__(self, profiles, encoder, rng, style, model, exploration):
        super().__init__(profiles)
        self.encoder = encoder
        self.rng = rng
        self.style = style
        self.exploration = exploration
        self.expert = opponent(profiles, BATTLE_LEAGUE[style % len(BATTLE_LEAGUE)])
        self.student = LearnedPolicy(profiles, encoder, model, value_only=True) if model is not None else None
        self.rows = []
        self.changes = 0

    def __call__(self, decision):
        self.observe(decision)
        if decision.phase != BATTLE:
            ranked = self.rank(decision)
            if decision.phase == DRAFT:
                spread = (1, 2, 3, 4)[self.style % 4]
                return ranked[self.rng.randrange(min(len(ranked), spread))][0]
            return ranked[self.rng.randrange(min(len(ranked), 1 + int(self.style % 3 == 2)))][0]
        ranked = self.rank(decision)
        if len(ranked) == 1:
            action = ranked[0][0]
            self.record(decision, action)
            return action
        options = [action for action, _ in ranked]
        self.expert.history = self.history
        teacher = options.index(self.expert.rank(decision, options)[0][0]) if self.style % 4 in (1, 2) else 0
        features = self.encoder(decision, self.history, ranked)
        preferred = teacher if self.student is None else self.student.index(features)
        behavior = self.rng.randrange(len(options)) if self.rng.random() < self.exploration else preferred
        propensity = self.exploration / len(options) + (1. - self.exploration) * (behavior == preferred)
        if self.student is not None:
            self.student.remember(features, behavior)
            self.changes += preferred != teacher
        self.rows.append((*features, teacher, behavior, propensity, decision.observation["turn"],
                          not all(decision.observation["active"])))
        action = options[behavior]
        self.record(decision, action)
        return action


def arrays(rows):
    return {name: np.asarray(values, dtype=(np.float32 if name in FLOAT_FIELDS else
                                           np.bool_ if name in BOOL_FIELDS else np.int32))
            for name, values in zip(FIELDS, zip(*rows, strict=True), strict=True)}


def collect(roots, cards, path, current, previous, exploration):
    configure_torch()
    profiles = load_card_profiles(cards, with_sets=True)
    encoder = Encoder(profiles)
    student = load_model(current, cards) if current else None
    opponent_model = load_model(previous, cards) if previous else student
    counts = Counter()
    rows = []
    cpu_ms = 0.
    began = perf_counter()
    with Showdown() as engine:
        for root in roots:
            pool = Random(root).sample(cards, DRAFT_SIZE)
            for leg in range(2):
                styles = [root % 7, (root // 7) % 7]
                models = [None, None]
                if student is not None:
                    models = [(student if root % 5 != 0 else None), (opponent_model if root % 3 != 0 else None)]
                if leg:
                    models.reverse()
                    styles.reverse()
                policies = {seat: Recorder(profiles, encoder, Random(f"population/v1/{root}/{leg}/{seat}"),
                                           styles[i], models[i], exploration)
                            for i, seat in enumerate(SEATS)}
                samples = defaultdict(list)
                winner = play_game(engine, cards, policies, root * 2 + leg, samples, pool=pool)
                if any(event for times in samples.values() for _, _, event in times):
                    counts["excluded_clock_games"] += 1
                    continue
                if any(sum(phase == DRAFT for phase, _, _ in samples[seat]) != 6 or
                       sum(phase == PREVIEW for phase, _, _ in samples[seat]) != 1 for seat in SEATS):
                    raise RuntimeError("Training replay did not complete Draft and Preview for both seats")
                same_population = int(all(model is not None for model in models))
                counts["games"] += 1
                counts["self_play_games"] += same_population
                counts["student_games"] += any(model is not None for model in models)
                counts["draws"] += winner is None
                cpu_ms += sum(ms for times in samples.values() for _, ms, _ in times)
                for index, (seat, policy) in enumerate(policies.items()):
                    outcome = .5 if winner is None else float(winner == seat)
                    episode = 4 * root + 2 * leg + index
                    counts["learner_decisions"] += len(policy.rows) if models[index] is not None else 0
                    counts["learner_preferences_different"] += policy.changes
                    rows.extend((*features, outcome, [-1.] * WIDTH, episode, turn, forced, propensity,
                                 policy.style, same_population)
                                for *features, propensity, turn, forced in policy.rows)
    np.savez_compressed(path, **arrays(rows))
    return dict(counts, paths=1, positions=len(rows), worker_seconds=perf_counter() - began,
                policy_seconds=cpu_ms / 1000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pools", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shard-pools", type=int, default=64)
    parser.add_argument("--current", type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--prior-workspace", type=Path, action="append", default=[])
    parser.add_argument("--exploration", type=float, default=.27)
    args = parser.parse_args()
    if min(args.pools, args.workers, args.shard_pools) < 1 or not 0 < args.exploration < 1:
        parser.error("Use positive counts and an exploration probability between zero and one")
    if args.previous and not args.current:
        parser.error("A historical snapshot requires a current model")
    # Prevent PyTorch/BLAS oversubscription before Windows workers are spawned.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    cards = read(CATALOG)
    roots = list(range(args.seed, args.seed + args.pools))
    pools = pools_for(cards, roots)
    used = used_pools(cards)
    for workspace in args.prior_workspace:
        used.update(used_pools(cards, workspace))
    if not used.isdisjoint(pools.values()):
        raise ValueError("Pool overlap with a registered experiment")
    args.output.mkdir(parents=True, exist_ok=False)
    checkpoints = {name: {filename: sha256((checkpoint / filename).read_bytes()).hexdigest()
                          for filename in ("config.json", "state_dict.pt")}
                   for name, checkpoint in (("current", args.current), ("previous", args.previous)) if checkpoint}
    manifest = dict(pools=pools, catalog=fingerprint(cards), version=1, exploration=args.exploration,
                    checkpoints=checkpoints, complete=False)
    save(args.output / "pools.json", manifest)
    counts = Counter()
    began = perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as executor:
        jobs = [executor.submit(collect, roots[i:i + args.shard_pools], cards,
                                args.output / f"{roots[i]}.npz", args.current, args.previous, args.exploration)
                for i in range(0, len(roots), args.shard_pools)]
        for job in as_completed(jobs):
            counts.update(job.result())
            print(dict(counts), flush=True)
    if counts["games"] + counts["excluded_clock_games"] != 2 * args.pools or counts["games"] < 1.98 * args.pools:
        raise RuntimeError("Unexpected number of complete games")
    report = dict(complete=True, counts=dict(counts), wall_seconds=perf_counter() - began)
    save(args.output / "completed.json", report)
    print(dict(report, pools=len(pools)), flush=True)


if __name__ == "__main__":
    main()
