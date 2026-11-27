"""Offer learned lineup proposals to the intact Showdown Preview search."""

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
from ..combination_space import CardSpace
from ..draft_league import read
from ..preview85 import Preview85
from ..sim import fingerprint
from . import matches, preview_outcome


def evaluate(roots, cards, rivals, *args, checkpoint, battle_checkpoint):
    configure_torch()
    config = read(checkpoint / "config.json")
    if config["catalog"] != fingerprint(cards):
        raise ValueError("Lineup model and match card catalogs differ")
    model = preview_outcome.PreviewOutcome(**config["network"])
    model.load_state_dict(torch.load(checkpoint / "state_dict.pt", map_location="cpu", weights_only=True))
    learner = CardSpace(cards), model.eval()

    class BoundPreview(Preview85):
        def __init__(self, anchor, worker, matchup, indexed, original_learner, reviewer, lead_prior):
            super().__init__(anchor, worker, matchup, indexed, learner, reviewer, lead_prior)

    with patch.object(lab85, "Preview85", BoundPreview):
        if battle_checkpoint:
            return matches.evaluate(roots, cards, rivals, *args, checkpoint=battle_checkpoint, key=None)
        return lab85.evaluate(roots, cards, rivals, *args)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--lineup-net", type=Path, required=True)
    parser.add_argument("--battle-net", type=Path)
    args, remaining = parser.parse_known_args()
    sys.argv[1:] = remaining
    freeze_battle = matches.freeze_for(args.battle_net, False)

    def freeze(*settings):
        report = freeze_battle(*settings)
        if not report["candidate"]["preview"]:
            raise ValueError("A learned Preview ablation needs the original Preview search enabled")
        paths = (Path(__file__), Path(preview_outcome.__file__), args.lineup_net / "state_dict.pt", args.lineup_net / "config.json")
        for path in paths:
            report["files"][str(path)] = sha256(path.read_bytes()).hexdigest()
        report["research_learning"]["preview_checkpoint"] = str(args.lineup_net)
        return report

    def measure(roots, cards, rivals, checkout, variant, workers, *settings):
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
            jobs = [executor.submit(evaluate, [root], cards, rivals, checkout, variant, *settings,
                                    checkpoint=args.lineup_net, battle_checkpoint=args.battle_net) for root in roots]
            for job in as_completed(jobs):
                yield job.result()

    with patch.object(blind85, "measure", measure), patch.object(quicksilver80, "freeze", freeze):
        quicksilver80.main()


if __name__ == "__main__":
    main()
