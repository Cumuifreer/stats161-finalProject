"""Compare legal two-card plans when the snake order gives consecutive picks."""

from collections import Counter
from itertools import combinations
from random import Random
from statistics import fmean

import torch

from .draft_forecast import public_draft
from .game import DRAFT
from .sim import fingerprint


class DraftBundles:
    def __init__(self, experiment, *, review_picks=(2, 4, 6, 8, 10), margin=.035):
        self.anchor = experiment
        self.prior = experiment.league.prior
        self.model = experiment.league.ranks["pairwise"].net
        self.review_picks = review_picks
        self.margin = margin
        self.second = None
        self.counts = Counter()

    @torch.inference_mode()
    def __call__(self, decision):
        if decision.phase != DRAFT:
            raise ValueError("A draft bundle requires a Draft decision")
        pick = decision.observation["pick"]
        if self.second:
            turn, previous, card = self.second
            self.second = None
            roster = decision.observation["picked"]
            if pick == turn and roster and roster[-1]["card_id"] == previous:
                selected = next((action for action in decision.actions if action["card_id"] == card), None)
                if selected is not None:
                    self.counts["completed_plans"] += 1
                    return selected
        anchor = self.anchor.choose(decision, "champion")
        if pick not in self.review_picks or len(decision.actions) < 3:
            return anchor
        _, own, foe, available, _, schedule = public_draft(decision, self.prior.index)
        if len(schedule) <= pick or schedule[pick] != 0:
            return anchor
        labels = {self.prior.index[action["card_id"]]: action for action in decision.actions}
        first = self.prior.index[anchor["card_id"]]
        bundles = [(a, b) for a, b in combinations(labels, 2) if self.prior.items[a] != self.prior.items[b]]
        if not bundles or all(first not in pair for pair in bundles):
            return anchor
        # Either ordering is legal with no intervening reply. Preserve the original first pick when it is in the pair.
        plans = [(b, a) if b == first else (a, b) for a, b in bundles]
        rng = Random(fingerprint(["draft-bundles-v1", sorted(own), sorted(foe), sorted(available), pick]))
        noises = [[[0.] * len(self.prior.items) for _ in range(2)]]
        noises.extend([[list(rng.gauss(0, sigma) for _ in self.prior.items) for sigma in (.14, .28 + .05 * (i % 3))]
                       for i in range(11)])
        teams = [[self.prior._finish([card for card in available if card != a], (own + [a], foe), b,
                                     schedule, pick + 1, noise, i + 1)
                  for i, noise in enumerate(noises)] for a, b in plans]
        reference_team = next((state for group in teams for state in group if state is not None), None)
        if reference_team is None:
            return anchor
        tensor = torch.tensor([team if team is not None else reference_team for group in teams for team in group])
        scores = self.model(tensor).sigmoid().view(len(teams), len(noises)).tolist()
        scores = [[value if team is not None else 0. for value, team in zip(row, group, strict=True)]
                  for row, group in zip(scores, teams, strict=True)]
        expected = [fmean(row) for row in scores]
        reference = max((i for i, plan in enumerate(plans) if plan[0] == first), key=expected.__getitem__)
        best = max(range(len(plans)), key=expected.__getitem__)
        differences = [a - b for a, b in zip(scores[best], scores[reference], strict=True)]
        self.counts["reviewed"] += 1
        self.counts["evaluated_bundles"] += len(plans)
        if fmean(differences) < self.margin or sum(value > 0 for value in differences) < 8:
            return anchor
        new_first, new_second = plans[best]
        self.second = (pick + 1, labels[new_first]["card_id"], labels[new_second]["card_id"])
        self.counts["changed"] += 1
        return labels[new_first]
