"""Compare successive lineup networks in the same complete seat-paired games."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from multiprocessing import get_context
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from ..battle_league import configure_torch
from ..battle_value import opponent
from ..blind85 import used_pools
from ..combination_space import CardSpace
from ..draft_last_pick import mean_ci
from ..draft_leaf import CATALOG
from ..draft_league import pools_for, read, save
from ..game import PREVIEW
from ..joint import JointPolicy
from ..lineup import index_cards
from ..roster import DraftRanker, _load_model
from ..sim import SEATS, Showdown, fingerprint, play_game
from ..tactics import load_card_profiles
from .preview_outcome import PreviewOutcome
from .preview_qualify import LearnedDraft, LearnedPreview
from .qualify import summarize


class LearnedRoster(LearnedDraft):
    def __init__(self, profiles, cards, space, model, projection, indexed):
        super().__init__(profiles, cards, space, model, projection)
        self.preview = LearnedPreview(profiles, indexed, space, model)

    def __call__(self, decision):
        if decision.phase == PREVIEW:
            self.observe(decision)
            return self.preview(decision)
        return super().__call__(decision)


def qualify(roots, cards, checkpoints):
    configure_torch()
    profiles = load_card_profiles(cards, with_sets=True)
    space = CardSpace(cards, profiles)
    indexed = index_cards(cards)
    projection = DraftRanker(cards, *_load_model(Path("checkpoints/roster-v1/expanded"), "cpu"))
    variants = [None]
    for location in checkpoints:
        settings = read(location / "config.json")
        if settings["catalog"] != fingerprint(cards):
            raise ValueError("Lineup comparison requires one frozen card catalog")
        model = PreviewOutcome(**settings["network"]).eval()
        model.load_state_dict(torch.load(location / "state_dict.pt", map_location="cpu", weights_only=True))
        variants.append(model)
    results = {}
    with Showdown() as engine:
        for root in roots:
            rival = "forecast" if root % 2 == 0 else "forecast-replies"
            arms = []
            for model in variants:
                pair = []
                for actor, foe in (SEATS, SEATS[::-1]):
                    own = (LearnedRoster(profiles, cards, space, model, projection, indexed) if model
                           else JointPolicy(profiles))
                    clocks = defaultdict(list)
                    winner = play_game(engine, cards, {actor: own, foe: opponent(profiles, rival)}, root, clocks)
                    if any(problem for series in clocks.values() for _, _, problem in series):
                        raise RuntimeError(f"Clock failure in {root}")
                    pair.append(dict(won=winner == actor, drew=winner is None,
                                     moving_ms=[ms for phase, ms, _ in clocks[actor] if phase == PREVIEW],
                                     chosen_draft=getattr(own, "changed", 0),
                                     chose_diff=own.preview.changed if model else 0, n_scored=int(model is not None)))
                arms.append(pair)
            results[root] = {rival: arms}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pools", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or args.pools < 16 or args.workers < 1:
        parser.error("Use a new result folder and at least 16 pools")
    cards = read(CATALOG)
    roots = list(range(args.seed, args.seed + args.pools))
    pools = pools_for(cards, roots)
    if not used_pools(cards).isdisjoint(pools.values()):
        raise ValueError("Lineup generations must use fresh qualification pools")
    checkpoints = args.parent, args.candidate
    save(args.output / "pools.json", dict(pools=pools, configs={str(p): sha256((p / "state_dict.pt").read_bytes()).hexdigest()
                                                             for p in checkpoints}))
    started = perf_counter()
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        for shard in pool.map(qualify, [roots[i::args.workers] for i in range(args.workers)], [cards] * args.workers,
                              [checkpoints] * args.workers):
            results.update(shard)
    report = {}
    for rival in ("forecast", "forecast-replies"):
        block = [row for row in results.values() if rival in row]
        report[rival] = {name: summarize(block, index, rival) for index, name in enumerate(("joint", "parent", "candidate"))}
        difference = [mean(int(b["won"]) - int(a["won"]) for a,b in zip(row[rival][1], row[rival][2], strict=True))
                      for row in block]
        report[rival]["candidate_vs_parent"] = {"change": mean(difference), "paired95": mean_ci(difference)}
        for name in ("joint", "parent", "candidate"):
            report[rival][name]["preview_ms_p95"] = report[rival][name].pop("battle_ms_p95")
    save(args.output / "matches.json", dict(report=report, records=results, pools=pools, wall_seconds=perf_counter() - started))
    print(report, flush=True)


if __name__ == "__main__":
    main()
