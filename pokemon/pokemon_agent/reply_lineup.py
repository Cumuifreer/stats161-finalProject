"""Predict unseen reserves from public drafts and the two revealed leads."""

import argparse
from itertools import combinations
from pathlib import Path
from random import Random
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from .combination_preview import team_features
from .combination_space import CardSpace
from .draft_last_pick import other
from .draft_leaf import CATALOG
from .draft_league import read, save
from .lineup import index_cards
from .sim import fingerprint


def features(space, lineup, leads, rival_draft, rival_leads):
    ids = space.species
    team = [ids[name] for name in lineup]
    front = [ids[name] for name in leads]
    ours = [ids[name] for name in rival_draft]
    other_front = [ids[name] for name in rival_leads]
    reserves = sorted(set(team) - set(front))
    field = team_features(space, (team, front), (ours, other_front))
    anti_front = space.attack[np.ix_(reserves, other_front)]
    front_pressure = space.attack[np.ix_(other_front, reserves)]
    own_cover = space.attack[np.ix_(reserves, ours)].max(axis=0)
    groups = [float(bool(set(reserves) & roles)) for roles in space.roles.values()]
    return np.asarray([*field, anti_front.max(), anti_front.mean(), front_pressure.min(1).mean(), own_cover.mean(),
                       *groups, *(1. if i in reserves else 0. for i in range(len(space.ids)))], dtype=np.float32)


class ReplyLineup:
    def __init__(self, checkpoint, cards, profiles=None):
        raw = read(checkpoint)
        if raw["catalog"] != fingerprint(cards):
            raise ValueError("The inferred-lineup model uses a different card catalog")
        self.space = CardSpace(cards, profiles)
        self.weights = np.asarray(raw["weights"], dtype=np.float32)

    def scores(self, lineups, leads, draft, our_leads):
        return [float(features(self.space, lineup, leads, draft, our_leads) @ self.weights) for lineup in lineups]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Provide a fresh output")
    torch.set_num_threads(1)
    cards = read(CATALOG)
    rows = read(args.source)
    space = CardSpace(cards)
    catalog = index_cards(cards)
    by_id = {card["card_id"]: species for species, card in catalog.items()}
    roots = list(rows["records"])
    Random(26181).shuffle(roots)
    validation_ids = set(roots[::5])
    fit, validation = [], []
    for root in roots:
        dataset = validation if root in validation_ids else fit
        for games in rows["records"][root].values():
            for game in games:
                actor = game["actor"]
                enemy = other(actor)
                deployments = game["lineups"]
                actual = frozenset(deployments[enemy]["bring"])
                leads = deployments[enemy]["leads"]
                ours = [by_id[card] for card in game["roster"][actor]]
                foe = [by_id[card] for card in game["roster"][enemy]]
                roster_options = [tuple(leads) + reserves for reserves in combinations([name for name in foe if name not in leads], 2)]
                labels = [i for i, lineup in enumerate(roster_options) if set(lineup) == actual]
                if len(labels) != 1:
                    raise ValueError("Recorded lead and quartet were inconsistent")
                features_by_option = [features(space, lineup, leads, ours, deployments[actor]["leads"]) for lineup in roster_options]
                dataset.append((np.stack(features_by_option), labels[0]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training = torch.tensor(np.stack([row[0] for row in fit]), device=device)
    target = torch.tensor([row[1] for row in fit], device=device)
    held = torch.tensor(np.stack([row[0] for row in validation]), device=device)
    held_target = torch.tensor([row[1] for row in validation], device=device)
    scaled = (training - training.mean(1, keepdim=True)).square().mean((0, 1)).sqrt().clamp_min(.06)
    weights = torch.nn.Parameter(torch.zeros(training.shape[-1], device=device))
    optimizer = torch.optim.AdamW([weights], lr=.028)
    began = perf_counter()
    for _ in range(350):
        loss = F.cross_entropy((training / scaled) @ weights, target) + .013 * weights.square().sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    elapsed = perf_counter() - began
    with torch.no_grad():
        coef = weights / scaled
        logits = held @ coef
        ranked = logits.argsort(descending=True)
        report = dict(top1=float(ranked[:, 0].eq(held_target).float().mean()),
                      top2=float(ranked[:, :2].eq(held_target[:, None]).any(1).float().mean()),
                      nll=float(F.cross_entropy(logits, held_target)))
    save(args.output, dict(catalog=fingerprint(cards), source=str(args.source), pools=rows["pools"],
                           validation_roots=sorted(validation_ids), device=str(device), weights=coef.tolist(),
                           fit_seconds=elapsed, fit_rows=len(fit), validation_rows=len(validation), report=report,
                           api_spend_usd=0))
    print(report, flush=True)


if __name__ == "__main__":
    main()
