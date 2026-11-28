"""Nominate legal lineups using match outcomes from spent development pools."""

import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .. import blind85, lab85, quicksilver80
from ..combination_preview import encode_action, team_features
from ..combination_space import CardSpace
from ..draft_leaf import CATALOG
from ..draft_league import read, save
from ..sim import SEATS, fingerprint
from . import distilled


CHECKPOINT = Path("checkpoints/research80/observed-preview-v1.json")
ALLOWED = {"quicksilver-public", "ace-code", "pivot", "pivot-cycle", "pivot-pressure", "champion", "baseline70"}


class ObservedValue(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.register_buffer("weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, teams, roles, features):
        return (features[:, 0] - features[:, 1]) @ self.weights


def examples(paths, cards, space):
    seen = set()
    for path in paths:
        file = path.resolve()
        if not {"quicksilver80-dev", "quicksilver80-matched"}.intersection(file.parts) or any(
            "blind" in part.casefold() or "acceptance" in part.casefold() for part in file.parts
        ):
            raise ValueError("Only prior development matches are eligible for training")
        source = read(file)
        for root, rivals in source["records"].items():
            pool = source["pools"][root]
            for rival, matches in rivals.items():
                if rival not in ALLOWED:
                    continue
                for game in matches:
                    seat = game["actor"]
                    opponent = SEATS[1 - SEATS.index(seat)]
                    key = fingerprint([pool, rival, seat, game["lineups"], game["winner"]])
                    if key in seen:
                        continue
                    seen.add(key)
                    own, foe = [encode_action(space, game["lineups"][player]) for player in (seat, opponent)]
                    edge = np.subtract(team_features(space, own, foe), team_features(space, foe, own))
                    outcome = .5 if game["winner"] is None else float(seat == game["winner"])
                    yield pool, rival, edge, outcome


def train(paths, output):
    start = perf_counter()
    cards = read(CATALOG)
    space = CardSpace(cards)
    rows = list(examples(paths, cards, space))
    training = [row for row in rows if int(row[0][:8], 16) % 5]
    held = [row for row in rows if not int(row[0][:8], 16) % 5]
    if not training or not held:
        raise ValueError("Observed preview training requires distinct held-out pools")
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    totals = Counter(row[1] for row in training)
    wins = Counter()
    for _, rival, _, outcome in training:
        wins[rival] += outcome
    priors = {rival: (wins[rival] + 1.) / (total + 2.) for rival, total in totals.items()}

    def pack(part):
        counts = Counter(row[0] for row in part)
        weights = [1. / counts[row[0]] for row in part]
        x = torch.tensor(np.stack([row[2] for row in part]), dtype=torch.float32, device=device)
        y = torch.tensor([row[3] for row in part], dtype=torch.float32, device=device)
        w = torch.tensor(weights, dtype=torch.float32, device=device)
        intercept = torch.tensor([priors.get(row[1], .5) for row in part], dtype=torch.float32, device=device).logit()
        return x, y, w / w.sum(), intercept

    x, y, w, offset = pack(training)
    check, targets, weights, intercept = pack(held)
    scale = (x.square().mul(w[:, None]).sum(0) + .03).sqrt()
    parameters = nn.Parameter(torch.zeros(x.size(-1), device=device))
    optimizer = torch.optim.AdamW((parameters,), lr=.035, weight_decay=.01)
    began = perf_counter()
    for _ in range(160):
        prediction = (x / scale) @ parameters + offset
        loss = (w * F.binary_cross_entropy_with_logits(prediction, y, reduction="none")).sum()
        loss = loss + .040 * parameters.square().sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        weights_cpu = (parameters / scale).cpu().tolist()
        prediction = (check / scale) @ parameters + intercept
        metric = lambda logits: float((weights * F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none")).sum())
        held_loss = metric(prediction)
        intercept_only = metric(intercept)
    report = {"catalog": fingerprint(cards), "weights": weights_cpu,
              "pools": {pool: pool for pool, *_ in rows}, "train_matches": len(training), "held_matches": len(held),
              "train_pools": len({row[0] for row in training}), "held_pools": len({row[0] for row in held}),
              "intercept_only_log_loss": intercept_only, "held_log_loss": held_loss, "device": str(device),
              "fit_seconds": perf_counter() - began, "seconds": perf_counter() - start,
              "sources": {str(path): sha256(path.read_bytes()).hexdigest() for path in paths}, "api_spend_usd": 0}
    save(output, report)
    print({key: value for key, value in report.items() if key not in {"weights", "pools", "sources"}}, flush=True)


def evaluate(roots, cards, rivals, *args):
    checkpoint = read(CHECKPOINT)
    if checkpoint["catalog"] != fingerprint(cards):
        raise ValueError("Observed preview model requires its original public catalogue")
    space = CardSpace(cards)
    predictor = ObservedValue(checkpoint["weights"])
    base = lab85.Preview85

    class ObservedPreview(base):
        def __init__(self, anchor, worker, matchup, indexed, learner=None, reviewer=None, lead_prior=None):
            super().__init__(anchor, worker, matchup, indexed, (space, predictor), reviewer, lead_prior)

    with patch.object(lab85, "Preview85", ObservedPreview):
        return lab85.evaluate(roots, cards, rivals, *args)


def freeze(*args):
    result = FROZEN(*args)
    for path in (Path(__file__), Path(distilled.__file__), CHECKPOINT):
        result["files"][str(path)] = sha256(path.read_bytes()).hexdigest()
    return result


FROZEN = quicksilver80.freeze


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("train")
        parser.add_argument("--source", nargs="+", type=Path, required=True)
        parser.add_argument("--output", type=Path, default=CHECKPOINT)
        args = parser.parse_args()
        train(args.source, args.output)
    else:
        with patch.object(distilled, "evaluate", evaluate), patch.object(blind85, "measure", distilled.measure), \
             patch.object(quicksilver80, "freeze", freeze):
            quicksilver80.main()
