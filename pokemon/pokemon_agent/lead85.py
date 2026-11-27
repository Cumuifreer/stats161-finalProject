"""Separate the probability of bringing a quartet from the probability of its lead pair."""

import argparse
from collections import defaultdict
from itertools import combinations
from math import exp, log1p
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from .combination_preview import signature
from .combination_space import CardSpace
from .draft85 import deployment
from .draft_leaf import CATALOG
from .draft_league import disjoint, read, save
from .sim import fingerprint
from .tactics import tactical_actions


def pair_features(space, own, foe, tactical):
    features = []
    their_speed = space.speeds[foe]
    for front in combinations(own, 2):
        pair = list(front)
        threats = space.attack[np.ix_(foe, pair)].clip(0, 1.5)
        pressure = space.attack[np.ix_(pair, foe)].clip(0, 1.5)
        faster = space.speeds[pair, None] > their_speed[None, :]
        roles = [len(set(pair) & group) for group in space.roles.values()]
        relevance = [float(bool(set(foe) & group)) for group in space.roles.values()]
        position = tactical[frozenset(pair)]
        traits = [log1p(position) / 4., float(position == 0), pressure.max(0).mean(), pressure.min(0).mean(),
                  pressure.mean(), threats.max(1).mean(), threats.min(1).mean(), threats.mean(), faster.mean(),
                  faster.max(0).mean(), faster.min(0).mean(), space.speeds[pair].max() / 250.,
                  space.speeds[pair].min() / 250., *(value * condition for value, condition in zip(roles, relevance)),
                  *roles, *(float(i in pair) for i in range(len(space.ids)))]
        features.append((frozenset(pair), np.asarray(traits, dtype=np.float32)))
    return features


def can_lead(decision, space, ranked):
    own, foe = [[space.species[mon["species"]] for mon in decision.observation[key]] for key in ("team", "opponent_team")]
    tactical = {}
    for i, choice in enumerate(ranked):
        lead = frozenset(space.species[name] for name in choice["leads"])
        tactical.setdefault(lead, i)
    return pair_features(space, own, foe, tactical)


class LeadPrior:
    def __init__(self, checkpoint, cards, profiles=None):
        model = read(checkpoint)
        if model["catalog"] != fingerprint(cards):
            raise ValueError("Lead prior expects its public card catalog")
        self.space = CardSpace(cards, profiles)
        self.weights = np.asarray(model["weights"], dtype=np.float32)

    def scores(self, reverse, ranked):
        return {lead: float(features @ self.weights) for lead, features in can_lead(reverse, self.space, ranked)}

    def scenarios(self, reverse, nominated, ranked):
        spaces = defaultdict(dict)
        for choice in ranked:
            group, lead = signature(choice)
            spaces[group].setdefault(lead, choice)
        order = list(dict.fromkeys(signature(choice)[0] for choice in [*nominated[:2], ranked[0], *nominated[2:], *ranked]))[:4]
        fitted = self.scores(reverse, ranked)
        analytic = {}
        for i, action in enumerate(ranked):
            analytic.setdefault(signature(action), i)
        specialist = {signature(choice) for choice in nominated}
        entries = []
        for team, mass in zip(order, (.40, .27, .18, .15), strict=False):
            ranks = list(spaces[team].values())
            logits = []
            for action in ranks:
                ids = frozenset(self.space.species[name] for name in action["leads"])
                mark = signature(action)
                logits.append(.66 * fitted[ids] - .017 * analytic[mark] + .92 * (mark in specialist))
            maximum = max(logits)
            weights = [exp(value - maximum) for value in logits]
            divisor = sum(weights)
            top = sorted(zip(weights, ranks), key=lambda row: row[0], reverse=True)[:2]
            entries.extend((mass * weight / divisor, choice) for weight, choice in top)
        entries.sort(key=lambda row: row[0], reverse=True)
        return [action for _, action in entries], [value for value, _ in entries]


def data(cards, reports):
    indexed = {card["card_id"]: card for card in cards}
    space = CardSpace(cards)
    features_by_roster, labels = [], []
    for report in reports:
        for opponents in report["records"].values():
            for rival in ("ace-code", "pivot"):
                for game in opponents[rival]:
                    foe = "p2" if game["actor"] == "p1" else "p1"
                    own, other = [[indexed[card] for card in game["roster"][name]] for name in (foe, game["actor"])]
                    decision = deployment(own, other)
                    ranked = tactical_actions(decision)
                    options = can_lead(decision, space, ranked)
                    # Labels are the opening leads subsequently revealed in these development games.
                    actual = frozenset(space.species[name] for name in game["lineups"][foe]["leads"])
                    labels.append(next(i for i, (lead, _) in enumerate(options) if lead == actual))
                    features_by_roster.append(np.stack([values for _, values in options]))
    return np.stack(features_by_roster), np.asarray(labels)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--fit", type=Path, nargs="+", required=True)
    cli.add_argument("--validate", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if args.output.exists():
        cli.error("Provide a fresh output")
    cards = read(CATALOG)
    training = [read(path) for path in args.fit]
    validation = read(args.validate)
    pools = {}
    for report in training:
        disjoint(validation["pools"], report["pools"])
        disjoint(pools, report["pools"])
        pools.update(report["pools"])
    train, target = data(cards, training)
    held, actual = data(cards, [validation])
    torch.set_num_threads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tensor = torch.tensor(train, device=device)
    desired = torch.tensor(target, dtype=torch.long, device=device)
    check = torch.tensor(held, device=device)
    labels = torch.tensor(actual, dtype=torch.long, device=device)
    norms = tensor.std(dim=(0, 1)).clamp_min(.12)
    weights = torch.nn.Parameter(torch.zeros(tensor.shape[-1], device=device))
    optimizer = torch.optim.AdamW([weights], lr=.025)
    began = perf_counter()
    for _ in range(220):
        loss = F.cross_entropy(tensor @ (weights / norms), desired) + .048 * weights.square().sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        learned = check @ (weights / norms)
        prior = -check[..., 0]
        gains = {}
        for mode, score in (("tactical", prior), ("learned", learned)):
            order = score.argsort(dim=1, descending=True)
            gains[mode] = {f"top{i}": float(order[:, :i].eq(labels[:, None]).any(1).float().mean()) for i in (1, 3, 6)}
    save(args.output, dict(catalog=fingerprint(cards), pools=pools | validation["pools"],
                           weights=(weights / norms).detach().tolist(), training_rows=len(train), validation_rows=len(held),
                           metrics=gains, fit_seconds=perf_counter() - began, device=device, api_spend_usd=0))
    print(gains, flush=True)


if __name__ == "__main__":
    main()
