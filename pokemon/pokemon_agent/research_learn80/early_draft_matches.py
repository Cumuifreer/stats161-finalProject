"""Evaluate the learned Draft pick value alongside the locked lineup model."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from multiprocessing import get_context
from pathlib import Path
import sys
from unittest.mock import patch

import torch

from .. import blind85, lab85, quicksilver80
from ..battle_league import configure_torch
from ..draft_league import read
from ..sim import fingerprint
from . import draft_matches, draft_outcomes, draft_qualification, preview_matches, preview_outcome
from .draft_outcomes import DraftOutcome, PICKS
from .draft_qualification import recommend


class ValueDraft:
    def __init__(self, anchor, model, card_ids):
        self.anchor = anchor
        self.model = model
        self.card_ids = card_ids
        self.counts = anchor.counts

    def __call__(self, decision):
        anchor = self.anchor(decision)
        if decision.observation["pick"] not in PICKS or len(decision.actions) < 2:
            return anchor
        selected = recommend(self.model, decision, self.card_ids, anchor)
        self.counts["value_draft_scored"] += 1
        self.counts["value_draft_changed"] += selected != anchor
        return selected


def evaluate(roots, cards, rivals, *args, lineup, draft):
    configure_torch()
    config = read(draft / "config.json")
    if config["catalog"] != fingerprint(cards):
        raise ValueError("Draft checkpoint and current card catalog differ")
    model = DraftOutcome(**config["network"]).eval()
    model.load_state_dict(torch.load(draft / "state_dict.pt", map_location="cpu", weights_only=True))
    card_ids = {card["card_id"]: index for index, card in enumerate(cards)}

    def play(*params, **settings):
        candidate = settings.get("candidate") or params[-1]
        candidate.draft = ValueDraft(candidate.draft, model, card_ids)
        return PLAY(*params, **settings)

    with patch.object(lab85, "play", play):
        return draft_matches.evaluate(roots, cards, rivals, *args, checkpoint=lineup, battle_checkpoint=None)


PLAY = lab85.play


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--lineup-net", type=Path, required=True)
    parser.add_argument("--draft-net", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    sys.argv[1:] = remaining
    seed_lock = draft_matches.freeze_for(None, False)

    def freeze(*settings):
        result = seed_lock(*settings)
        paths = [Path(__file__), *(Path(module.__file__) for module in
                                  (draft_outcomes, draft_qualification, preview_matches, preview_outcome))]
        for checkpoint in (args.lineup_net, args.draft_net):
            paths.extend(checkpoint / name for name in ("config.json", "state_dict.pt"))
        for path in paths:
            result["files"][str(path)] = sha256(path.read_bytes()).hexdigest()
        result["research_learning"].update(preview_checkpoint=str(args.lineup_net), draft_checkpoint=str(args.draft_net))
        return result

    def measure(roots, cards, rivals, checkout, variant, workers, *settings):
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            jobs = [pool.submit(evaluate, [root], cards, rivals, checkout, variant, *settings,
                                lineup=args.lineup_net, draft=args.draft_net) for root in roots]
            for job in as_completed(jobs):
                yield job.result()

    with patch.object(blind85, "measure", measure), patch.object(quicksilver80, "freeze", freeze):
        quicksilver80.main()


if __name__ == "__main__":
    main()
