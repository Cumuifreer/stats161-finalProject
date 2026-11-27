"""Refit a frozen lineup prior on same-roster, same-reply changes."""

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from .combination_preview import freeze, load_model
from .draft_leaf import CATALOG
from .draft_league import read, save
from .preview_fit85 import batch


TEACHER = Path("checkpoints/combination-v1/uniform")


def grouped(rows, device):
    n, a, b = len(rows), len(rows[0]["targets"]), len(rows[0]["targets"][0])
    values = batch(rows, device)
    return tuple(value.reshape(n, a * b, *value.shape[1:]) for value in values), a, b


def forward(net, values, actions, replies):
    prediction = net(*(entry.flatten(0, 1) for entry in values[:-1]))
    return prediction.reshape(len(values[-1]), actions, replies)


def loss(scores, outcomes, prior):
    target = outcomes.mean(-1)
    weights = (target - target[:, :1]).abs()
    preference = (target > target[:, :1]).float()
    relative = scores.mean(-1) - scores.mean(-1)[:, :1]
    ranking = F.binary_cross_entropy_with_logits(relative * 1.6, preference, weight=weights, reduction="sum")
    absolute = F.binary_cross_entropy_with_logits(scores, outcomes)
    preserve = (scores - prior).square().mean()
    return .62 * ranking / weights.sum().clamp_min(1.) + .38 * absolute + .06 * preserve


@torch.inference_mode()
def gain(scores, outcomes):
    probabilities = scores.sigmoid().mean(-1)
    scores = outcomes.mean(-1)
    best = probabilities.argmax(dim=1, keepdim=True)
    lift = scores.gather(1, best).squeeze(-1) - scores[:, 0]
    return float(lift.mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Provide a fresh output")
    torch.manual_seed(8461)
    torch.set_num_threads(1)
    cards = read(CATALOG)
    labels = read(args.labels / "labels.json")
    config = read(args.labels / "config.json")
    held = set(map(str, config["validation_roots"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, actions, replies = grouped([row for root, rows in labels["rows"].items() if root not in held for row in rows], device)
    valid, _, _ = grouped([row for root, rows in labels["rows"].items() if root in held for row in rows], device)
    teacher = load_model(TEACHER, cards).to(device).requires_grad_(False)
    student = deepcopy(teacher)
    student.matchup.requires_grad_(True)
    student.rosters.roles.requires_grad_(True)
    student.rosters.compare.requires_grad_(True)
    parameters = [param for param in student.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=.0006, weight_decay=.02)
    with torch.no_grad():
        train_prior = forward(teacher, train, actions, replies)
        valid_prior = forward(teacher, valid, actions, replies)
    shape = (-1, actions, replies)
    best_loss = float(loss(valid_prior, valid[-1].reshape(shape), valid_prior))
    best = teacher
    started = perf_counter()
    for _ in range(50):
        for index in torch.randperm(len(train[-1]), device=device).split(16):
            items = tuple(entry[index] for entry in train)
            score = forward(student, items, actions, replies)
            objective = loss(score, items[-1].reshape(shape), train_prior[index])
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
        with torch.no_grad():
            score = forward(student, valid, actions, replies)
            objective = float(loss(score, valid[-1].reshape(shape), valid_prior))
            if objective < best_loss:
                best_loss = objective
                best = deepcopy(student)
    with torch.inference_mode():
        score = forward(best, valid, actions, replies)
    metrics = {"held_loss": best_loss, "baseline_held_gain": gain(valid_prior, valid[-1].reshape(shape)),
               "held_gain": gain(score, valid[-1].reshape(shape)), "fit_seconds": perf_counter() - started,
               "trainable_parameters": sum(param.numel() for param in parameters), "device": str(device), "api_spend_usd": 0}
    args.output.mkdir(parents=True)
    torch.save(best.cpu().state_dict(), args.output / "state_dict.pt")
    save(args.output / "config.json", dict(provenance=freeze(cards), width=read(TEACHER / "config.json")["width"],
                                           pools=labels["pools"], teacher=sha256((TEACHER / "state_dict.pt").read_bytes()).hexdigest(),
                                           validation_roots=config["validation_roots"]))
    save(args.output / "metrics.json", metrics)
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
