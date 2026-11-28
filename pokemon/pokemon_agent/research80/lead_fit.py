"""Fit a lead-pair prior across opponent policies with whole pools held out."""

import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from ..combination_space import CardSpace
from ..draft85 import deployment
from ..draft_leaf import CATALOG
from ..draft_league import read, save
from ..lead85 import can_lead
from ..sim import fingerprint
from ..tactics import tactical_actions


def examples(cards, reports):
    index = {card["card_id"]: card for card in cards}
    space = CardSpace(cards)
    seen = set()
    for report in reports:
        for root, entries in report["records"].items():
            pool = report["pools"][root]
            for name, pair in entries.items():
                for game in pair:
                    actor = game["actor"]
                    seat = "p2" if actor == "p1" else "p1"
                    ours, foes = (game["roster"][side] for side in (seat, actor))
                    front = sorted(game["lineups"][seat]["leads"])
                    key = (pool, name, tuple(ours), tuple(foes), tuple(front))
                    if key in seen:
                        continue
                    seen.add(key)
                    view = deployment([index[key] for key in ours], [index[key] for key in foes])
                    pairs = can_lead(view, space, tactical_actions(view))
                    lead = frozenset(space.species[species] for species in front)
                    target = next(i for i, (option, _) in enumerate(pairs) if option == lead)
                    yield pool, name, np.stack([row for _, row in pairs]), target


def batch(rows, device):
    names = Counter(name for _, name, _, _ in rows)
    weights = np.array([names[name] ** -.45 for _, name, _, _ in rows], dtype=np.float32)
    return (
        torch.tensor(np.stack([features for _, _, features, _ in rows]), device=device),
        torch.tensor([target for _, _, _, target in rows], dtype=torch.long, device=device),
        torch.tensor(weights / weights.mean(), device=device),
    )


def summarize(scores, labels, held):
    losses = F.cross_entropy(scores, labels, reduction="none")
    ranks = scores.argsort(dim=1, descending=True)
    out = {}
    for name in ("all", *sorted({row[1] for row in held})):
        indices = [i for i, row in enumerate(held) if name == "all" or name == row[1]]
        out[name] = {"examples": len(indices), "log_loss": float(losses[indices].mean())}
        out[name].update({f"top{k}": float(ranks[indices, :k].eq(labels[indices, None]).any(1).float().mean())
                          for k in (1, 3, 6)})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", nargs="+", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Provide a fresh output")
    cards = read(CATALOG)
    source = read(args.prior)
    prior_pools = set(source["pools"].values())
    reports = [read(path) for path in args.matches]
    rows = list(examples(cards, reports))
    train, held = [], []
    for row in rows:
        cut = int(row[0][:8], 16) % 5
        (held if cut == 0 and row[0] not in prior_pools else train).append(row)
    if {row[0] for row in train} & {row[0] for row in held}:
        raise ValueError("Fit and validation must be grouped by card pool")
    torch.set_num_threads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, y, weights = batch(train, device)
    check, labels, _ = batch(held, device)
    prior = torch.tensor(source["weights"], dtype=torch.float32, device=device)
    norms = x.std(dim=(0, 1)).clamp_min(.12)
    change = torch.nn.Parameter(torch.zeros_like(prior))
    optimizer = torch.optim.AdamW([change], lr=.012)
    started = perf_counter()
    for _ in range(280):
        logits = x @ (prior + change / norms)
        loss = (weights * F.cross_entropy(logits, y, reduction="none")).mean() + .07 * change.square().sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    fitted = prior + change.detach() / norms
    metrics = {"frozen": summarize(check @ prior, labels, held), "new": summarize(check @ fitted, labels, held)}
    pools = {root: pool for report in reports for root, pool in report["pools"].items()}
    # The old checkpoint's source pools also contribute through the initialization and penalty.
    pools.update(source["pools"])
    save(args.output, {"catalog": fingerprint(cards), "weights": fitted.tolist(), "pools": pools,
                       "reports": {str(path): sha256(path.read_bytes()).hexdigest() for path in args.matches},
                       "validation_pools": sorted({row[0] for row in held}),
                       "training_rows": len(train), "validation_rows": len(held), "metrics": metrics,
                       "device": device, "fit_seconds": perf_counter() - started, "api_spend_usd": 0})
    print(metrics, flush=True)


if __name__ == "__main__":
    main()
