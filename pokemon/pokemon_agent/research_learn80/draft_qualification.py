"""Compare the learned draft-value decisions in complete games."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import torch

from ..battle_league import configure_torch
from ..battle_value import opponent
from ..blind85 import used_pools
from ..combination_space import CardSpace
from ..draft_leaf import CATALOG
from ..draft_league import pools_for, read, save
from ..game import DRAFT
from ..lineup import index_cards
from ..roster import DraftRanker, _load_model
from ..sim import SEATS, Showdown, fingerprint, play_game
from ..tactics import load_card_profiles, tactical_actions
from .draft_outcomes import DraftOutcome, PICKS, encode
from .preview_generations import LearnedRoster
from .preview_outcome import PreviewOutcome
from .qualify import summarize


@torch.inference_mode()
def recommend(outcome, decision, card_ids, anchor):
    options = [anchor, *(action for action in tactical_actions(decision) if action != anchor)][:3]
    if len(options) < 2:
        return anchor
    pool, owners, pick = encode(decision, card_ids)
    positions = [pool.index(card_ids[action["card_id"]]) for action in options]
    values = outcome(torch.tensor([pool]), torch.tensor([owners]), torch.tensor([pick]))[0, positions].sigmoid().tolist()
    best = max(range(len(values)), key=values.__getitem__)
    return options[best] if best and values[best] - values[0] >= .035 else anchor


class DraftPolicy(LearnedRoster):
    def __init__(self, profiles, cards, space, model, projection, indexed, outcome):
        super().__init__(profiles, cards, space, model, projection, indexed)
        self.card_ids = {card["card_id"]: index for index, card in enumerate(cards)}
        self.outcome = outcome
        self.evaluated = 0
        self.revisions = 0

    @torch.inference_mode()
    def __call__(self, decision):
        anchor = super().__call__(decision)
        if self.outcome is None or decision.phase != DRAFT or decision.observation["pick"] not in PICKS:
            return anchor
        if len(decision.actions) < 2:
            return anchor
        selected = recommend(self.outcome, decision, self.card_ids, anchor)
        self.evaluated += 1
        self.revisions += selected != anchor
        return selected


def evaluate(roots, cards, lineup, outcome):
    configure_torch()
    profiles = load_card_profiles(cards, with_sets=True)
    space, indexed = CardSpace(cards, profiles), index_cards(cards)
    projection = DraftRanker(cards, *_load_model(Path("checkpoints/roster-v1/expanded"), "cpu"))
    modules = []
    for kind, directory in ((PreviewOutcome, lineup), (DraftOutcome, outcome)):
        config = read(directory / "config.json")
        if config["catalog"] != fingerprint(cards):
            raise ValueError("Draft qualification uses the training card catalog")
        model = kind(**config["network"]).eval()
        model.load_state_dict(torch.load(directory / "state_dict.pt", map_location="cpu", weights_only=True))
        modules.append(model)
    preview_model, draft_model = modules
    evidence = {}
    with Showdown() as engine:
        for root in roots:
            rival = "forecast" if root % 2 == 0 else "forecast-replies"
            arms = []
            for model in (None, draft_model):
                games = []
                for actor, foe in (SEATS, SEATS[::-1]):
                    agent = DraftPolicy(profiles, cards, space, preview_model, projection, indexed, model)
                    timings = defaultdict(list)
                    winner = play_game(engine, cards, {actor: agent, foe: opponent(profiles, rival)}, root, timings)
                    if any(event for seat in timings.values() for _, _, event in seat):
                        raise RuntimeError(f"Draft qualification clock failed at {root}")
                    games.append(dict(won=winner == actor, drew=winner is None,
                                      moving_ms=[ms for phase, ms, _ in timings[actor] if phase == DRAFT],
                                      chose_diff=agent.revisions, n_scored=agent.evaluated))
                arms.append(games)
            evidence[root] = {rival: arms}
    return evidence


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--lineup", type=Path, required=True)
    cli.add_argument("--draft", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--seed", type=int, required=True)
    cli.add_argument("--pools", type=int, default=384)
    cli.add_argument("--workers", type=int, default=4)
    args = cli.parse_args()
    if args.output.exists() or args.pools < 4 or args.workers < 1:
        cli.error("Use a fresh output directory and positive run sizes")
    cards = read(CATALOG)
    roots = list(range(args.seed, args.seed + args.pools))
    pools = pools_for(cards, roots)
    if not used_pools(cards).isdisjoint(pools.values()):
        raise ValueError("Draft qualification uses new card pools")
    save(args.output / "pools.json", dict(pools=pools, models={str(path): sha256((path / "state_dict.pt").read_bytes()).hexdigest()
                                                             for path in (args.lineup, args.draft)}))
    started = perf_counter()
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as workers:
        for shard in workers.map(evaluate, [roots[i::args.workers] for i in range(args.workers)], [cards] * args.workers,
                                 [args.lineup] * args.workers, [args.draft] * args.workers):
            results.update(shard)
    report = {}
    for opponent_name in ("forecast", "forecast-replies"):
        rows = [row for row in results.values() if opponent_name in row]
        by_model = {name: summarize(rows, index, opponent_name) for index, name in enumerate(("anchor", "learned"))}
        for arm in by_model.values():
            arm["draft_ms_p95"] = arm.pop("battle_ms_p95")
        report[opponent_name] = by_model
    save(args.output / "matches.json", dict(pools=pools, results=results, report=report, wall_seconds=perf_counter() - started))
    print(report, flush=True)


if __name__ == "__main__":
    main()
