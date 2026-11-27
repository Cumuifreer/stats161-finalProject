"""Test one terminal lineup model in both the Draft projection and Preview search."""

from hashlib import sha256
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import torch

from .. import lab85
from ..battle_league import configure_torch
from ..combination_space import CardSpace
from ..draft_bundles import DraftBundles
from ..draft_league import read
from ..tactics import tactical_actions
from . import matches, preview_matches, preview_qualify
from .preview_outcome import PreviewOutcome
from .preview_qualify import LearnedDraft


def evaluate(roots, cards, rivals, *args, checkpoint, battle_checkpoint):
    configure_torch()
    config = read(checkpoint / "config.json")
    net = PreviewOutcome(**config["network"]).eval()
    net.load_state_dict(torch.load(checkpoint / "state_dict.pt", map_location="cpu", weights_only=True))
    space = CardSpace(cards)

    class BoundDraft(DraftBundles):
        def __init__(self, experiment, **settings):
            super().__init__(experiment, **settings)
            self.value = LearnedDraft(experiment.profiles, cards, space, net, self.prior)

        def __call__(self, decision):
            before = perf_counter()
            anchor = super().__call__(decision)
            if decision.observation["pick"] < 9 or len(decision.actions) < 2:
                return anchor
            candidates = [anchor, *(action for action in tactical_actions(decision) if action != anchor)][:3]
            if len(candidates) < 2:
                return anchor
            reviewed = self.value.reviewed
            choice = self.value.review(decision, anchor, candidates, before)
            self.counts["neural_draft_reviews"] += self.value.reviewed - reviewed
            self.counts["neural_draft_changes"] += choice != anchor
            return choice

    with patch.object(lab85, "DraftBundles", BoundDraft):
        return EVALUATE(roots, cards, rivals, *args, checkpoint=checkpoint, battle_checkpoint=battle_checkpoint)


def freeze_for(checkpoint, asked):
    inherited = FREEZE(checkpoint, asked)

    def freeze(*args):
        result = inherited(*args)
        if not result["candidate"].get("research80", {}).get("draft_bundles"):
            raise ValueError("This experiment needs the frozen DraftBundles anchor")
        for path in (Path(__file__), Path(preview_qualify.__file__)):
            result["files"][str(path)] = sha256(path.read_bytes()).hexdigest()
        result["research_learning"]["learned_draft"] = True
        return result
    return freeze


EVALUATE, FREEZE = preview_matches.evaluate, matches.freeze_for


if __name__ == "__main__":
    with patch.object(preview_matches, "evaluate", evaluate), patch.object(matches, "freeze_for", freeze_for):
        preview_matches.main()
