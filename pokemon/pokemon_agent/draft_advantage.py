"""Score public Draft choices by learning their counterfactual full-game outcomes."""

import json
from random import Random

import torch
from torch import nn

from .draft_forecast import ForecastDraft, encode_owners, public_draft
from .draft_value import DraftRollout
from .sim import fingerprint


class ActionValue(nn.Module):
    def __init__(self, card_count=48, width=48):
        super().__init__()
        self.cards = nn.Embedding(card_count, width)
        self.owners = nn.Embedding(13, width)
        self.plies = nn.Embedding(12, width)
        self.context = nn.TransformerEncoderLayer(width, 4, 2 * width, dropout=.12, batch_first=True)
        self.score = nn.Sequential(nn.Linear(width + 6, width), nn.SiLU(), nn.Linear(width, 1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, pool, owners, ply, features):
        tokens = self.cards(pool) + self.owners(owners) + self.plies(ply)[:, None]
        residual = self.score(torch.cat((self.context(tokens), features), dim=-1)).squeeze(-1)
        projected = features[:, :, 0].clamp(.03, .97).logit()
        return projected + residual


def load_advantage(path, device="cpu"):
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    model = ActionValue(**config["network"])
    model.load_state_dict(torch.load(path / "state_dict.pt", map_location="cpu", weights_only=True))
    return model.to(device).eval(), config


class AdvantageDraft(ForecastDraft):
    def __init__(self, pairwise, predictor, rollout, mode, advantage=None):
        super().__init__(pairwise, predictor, "hybrid", rollout)
        self.mode = mode
        self.advantage = advantage

    @torch.inference_mode()
    def options(self, decision, anchor, *, all_legal=False):
        pool, own, foe, remaining, pick, order = public_draft(decision, self.prior.index)
        prior = self.prior.rank(decision, "prior")
        candidates = [anchor, *(action for action in prior if action != anchor)]
        indices = [self.prior.index[action["card_id"]] for action in candidates]
        zero = [[0.] * len(self.prior.items) for _ in range(2)]
        projected = self.prior._finish(remaining, (own, foe), indices[0], order, pick, zero, 1)
        denied = set()
        if projected is not None:
            next_own = next((i for i, side in enumerate(order[pick:]) if side == 0), len(order) - pick)
            denied.update(projected[1][len(foe):len(foe) + next_own])
        wide = candidates[:7]
        outsider = next((action for action, card in zip(candidates[7:], indices[7:], strict=True) if card in denied), None)
        if outsider:
            wide.append(outsider)
        actions = candidates if all_legal else wide
        cards = [self.prior.index[action["card_id"]] for action in actions]
        rng = Random(fingerprint([pool, own, foe, pick]))
        worlds = self.complete(pool, own, foe, remaining, pick, order, cards, rng)
        if any(rosters is None for rosters in worlds[0][:3]):
            return None
        flat = torch.tensor([rosters if rosters is not None else worlds[0][0] for group in worlds for rosters in group])
        probabilities = self.pairwise.net(flat).sigmoid().view(len(actions), -1)
        valid = torch.tensor([[rosters is not None for rosters in group] for group in worlds])
        values = probabilities * valid
        means = values.mean(-1)
        features = torch.zeros((len(pool), 6))
        position = {card: i for i, card in enumerate(pool)}
        slots = [position[card] for card in cards]
        features[slots] = torch.stack((means, values.amin(-1), values.std(-1, unbiased=False),
                                       torch.tensor([self.prior.strength[card] for card in cards]),
                                       torch.tensor([card in denied for card in cards], dtype=torch.float32),
                                       torch.tensor([float(i == 0) for i in range(len(cards))])), dim=1)
        return actions, worlds, rng, (pool, encode_owners(pool, own, foe), pick - 1, features), slots, len(candidates[:4]), denied

    @torch.inference_mode()
    def rank(self, decision, mode):
        baseline = self.pairwise.rank(decision, mode)
        if not baseline or mode == "prior" or len(baseline) < 2:
            return baseline
        try:
            plan = self.options(decision, baseline[0])
        except (TypeError, KeyError, ValueError):
            self.activity["forecast_fallbacks"] += 1
            return baseline
        if plan is None:
            return baseline
        actions, worlds, rng, (pool, owners, ply, features), slots, width, _ = plan
        estimates = features[slots, 0]
        scores = estimates
        if self.advantage is not None:
            predicted = self.advantage(torch.tensor([pool]), torch.tensor([owners]), torch.tensor([ply]), features[None])[0]
            scores = .65 * predicted[slots].sigmoid() + .35 * estimates
        # Preserve the frozen search's maximum number of checks at each ply.
        budget = 2 if ply >= 7 and self.mode != "reduced" else 1
        self.activity["forecast_positions"] += 1
        alternatives = sorted(range(1, len(actions)), key=lambda i: float(scores[i]), reverse=True)
        if self.mode == "reduced" and scores[alternatives[0]] < scores[0] - .04:
            self.activity["declined_reviews"] += 1
            return baseline
        for i in alternatives[:budget]:
            if estimates[i] < estimates[0] - .075:
                continue
            if self.rollout.compare(worlds[0], worlds[i], rng, decision.seconds_left):
                self.activity["forecast_overrides"] += 1
                self.activity["outside_overrides"] += i >= width
                return [actions[i], *(action for action in baseline if action != actions[i])]
        return baseline


class ReducedReview(DraftRollout):
    def compare(self, baseline, alternative, rng, seconds_left):
        old = [baseline[0], baseline[2]]
        new = [alternative[0], alternative[2]]
        return super().compare(old, new, rng, seconds_left)
