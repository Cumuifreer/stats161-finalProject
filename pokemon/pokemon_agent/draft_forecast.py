"""Predict public draft choices and evaluate their effect on later picks."""

from collections import Counter
import json
from random import Random
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from .draft_value import DraftRollout
from .game import DRAFT, Decision
from .sim import FIRST_DRAFTER_PICKS, fingerprint

STYLES = ("pairwise", "search", "prior", "value", "linear")
UNKNOWN_STYLE = len(STYLES)


def public_draft(decision, index):
    state = decision.observation
    own, foe, available = ([index[card["card_id"]] for card in state[key]]
                           for key in ("picked", "opponent_picked", "cards"))
    pick = state["pick"]
    pool = sorted(own + foe + available)
    if len(pool) != 18 or len(set(pool)) != 18 or pick != 1 + len(own) + len(foe) or not 1 <= pick <= 12:
        raise ValueError("Inconsistent public draft state")
    first = (pick - 1) in FIRST_DRAFTER_PICKS
    order = [int((turn in FIRST_DRAFTER_PICKS) != first) for turn in range(12)]
    if order[:pick - 1].count(0) != len(own):
        raise ValueError("Public rosters disagree with snake order")
    return pool, own, foe, available, pick, order


def encode_owners(pool, own, foe):
    tags = {card: slot for roster, start in ((own, 1), (foe, 7)) for slot, card in enumerate(roster, start)}
    return [tags.get(card, 0) for card in pool]


def legal_cards(pool, owners, items):
    held = {items[card] for card, owner in zip(pool, owners, strict=True) if 1 <= owner <= 6}
    return [owner == 0 and items[card] not in held for card, owner in zip(pool, owners, strict=True)]


class PickModel(nn.Module):
    def __init__(self, card_count=48, width=48):
        super().__init__()
        self.cards = nn.Embedding(card_count, width)
        self.owners = nn.Embedding(13, width)
        self.plies = nn.Embedding(12, width)
        self.styles = nn.Embedding(UNKNOWN_STYLE + 1, width)
        self.context = nn.TransformerEncoderLayer(width, 4, 2 * width, dropout=.05, batch_first=True,
                                                  activation="gelu")
        self.score = nn.Linear(width, 1)
        self.register_buffer("strength", torch.zeros(card_count))

    def encode(self, pool, owners, ply, style):
        state = self.plies(ply) + self.styles(style)
        tokens = self.cards(pool) + self.owners(owners) + state[:, None]
        return self.context(tokens)

    def forward(self, pool, owners, ply, style, mask, reference=None):
        logits = self.score(self.encode(pool, owners, ply, style)).squeeze(-1) + self.strength[pool]
        return logits.masked_fill(~mask, -torch.inf)


class AnchoredPickModel(PickModel):
    """Predict whether the old prior is overridden, then score the alternatives conditional on an override."""

    def __init__(self, card_count=48, width=48):
        super().__init__(card_count, width)
        self.override = nn.Linear(width, 1)
        nn.init.constant_(self.override.bias, -2.)

    def forward(self, pool, owners, ply, style, mask, reference=None):
        context = self.encode(pool, owners, ply, style)
        reference_mask = F.one_hot(reference, num_classes=pool.size(1)).bool()
        reference_context = context[torch.arange(len(pool), device=pool.device), reference]
        alternative = mask & ~reference_mask
        forced = ~alternative.any(1, keepdim=True)
        logit = self.override(reference_context)
        values = self.score(context).squeeze(-1) + self.strength[pool]
        distribution = values.masked_fill(~(alternative | (forced & reference_mask)), -torch.inf).log_softmax(1)
        scores = torch.where(reference_mask, F.logsigmoid(-logit).masked_fill(forced, 0.),
                             distribution + F.logsigmoid(logit))
        return scores.masked_fill(~mask, -torch.inf)


PREDICTORS = {"attention": PickModel, "anchored": AnchoredPickModel}


def load_predictor(path, device="cpu"):
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    model = PREDICTORS[config.get("architecture", "attention")](**config["network"])
    model.load_state_dict(torch.load(path / "state_dict.pt", map_location="cpu", weights_only=True))
    return model.to(device).eval(), config


class ForecastDraft:
    def __init__(self, pairwise, predictor, method, rollout=None):
        self.pairwise = pairwise
        self.prior = pairwise.baseline
        self.model = predictor
        self.method = method
        self.rollout = rollout
        self.activity = Counter()
        self.cards = [{"card_id": card} for card in self.prior.index]

    @torch.inference_mode()
    def rank(self, decision, mode):
        baseline = self.pairwise.rank(decision, mode)
        if not baseline or mode == "prior" or len(baseline) == 1:
            return baseline
        try:
            pool, own, foe, available, pick, order = public_draft(decision, self.prior.index)
        except (KeyError, TypeError, ValueError):
            self.activity["forecast_fallbacks"] += 1
            return baseline
        if self.method == "tail" and pick < 8:
            return baseline
        self.activity["forecast_positions"] += 1
        original = self.prior.rank(decision, "prior")
        options = [baseline[0], *(action for action in original if action != baseline[0])][:4]
        rng = Random(fingerprint([pool, own, foe, pick]))
        actions = [self.prior.index[option["card_id"]] for option in options]
        # All branches share random numbers. In world zero both sides use the old single-card prior.
        worlds = self.complete(pool, own, foe, available, pick, order, actions, rng)
        if any(teams is None for teams in worlds[0][:3]):
            return baseline
        neutral = worlds[0][0]
        teams = torch.tensor([rosters if rosters is not None else neutral for branch in worlds for rosters in branch])
        expected = self.pairwise.net(teams).sigmoid().view(len(options), -1)
        mask = torch.tensor([[rosters is not None for rosters in branch] for branch in worlds])
        scores = (expected * mask).mean(-1).tolist()
        alt = sorted(range(1, len(options)), key=lambda i: scores[i], reverse=True)
        chosen = 0
        if self.rollout is None:
            if scores[alt[0]] >= scores[0] + .025:
                chosen = alt[0]
        else:
            for i in alt[:2 if pick >= 8 else 1]:
                if scores[i] >= scores[0] - .075 and self.rollout.compare(worlds[0], worlds[i], rng, decision.seconds_left):
                    chosen = i
                    break
        if chosen == 0:
            return baseline
        self.activity["forecast_overrides"] += 1
        action = options[chosen]
        return [action, *(item for item in baseline if item != action)]

    def complete(self, pool, own, foe, available, pick, order, actions, rng):
        prior = self.prior
        zero = [[0.] * len(prior.items) for _ in range(2)]
        noise = [[rng.gauss(0, .24) for _ in prior.items] for _ in range(2)]
        worlds = [[prior._finish(available, (own, foe), card, order, pick, bias, 1) for bias in (zero, noise)]
                  for card in actions]
        if self.method == "exact":
            for card, branch in zip(actions, worlds, strict=True):
                branch[0] = self.complete_pairwise(own, foe, available, card, pick, order)
        trials = 6
        states = [[list(own) + [card], list(foe)] for card in actions for _ in range(trials)]
        alive = [True] * len(states)
        pool_ids = torch.tensor(pool).expand(len(states), -1)
        styles = torch.tensor([UNKNOWN_STYLE, 0, 1, 3, 4, UNKNOWN_STYLE] * len(actions))
        for turn, seat in enumerate(order[pick:], pick):
            owners = [encode_owners(pool, teams[seat], teams[1 - seat]) for teams in states]
            masks = [legal_cards(pool, tags, prior.items) for tags in owners]
            for i, mask in enumerate(masks):
                alive[i] &= any(mask)
            mask = torch.tensor(masks)
            reference = None
            if self.method == "control" or isinstance(self.model, AnchoredPickModel):
                covered = [prior._coverage(teams[seat]) for teams in states]
                limits = torch.tensor([[prior.strength[card] + .25 * prior._support(card, motifs) for card in pool]
                                       for motifs in covered])
                reference = limits.masked_fill(~mask, -torch.inf).argmax(1)
            if self.method == "control":
                scores = limits
            else:
                tags = torch.tensor(owners)
                style = styles if seat == 1 else torch.full_like(styles, 0)
                scores = self.model(pool_ids, tags, torch.full_like(style, turn), style, mask, reference)
            draws = torch.tensor([[rng.gauss(0, .48 if trial else 0.) for _ in pool] for trial in range(trials)])
            selected = (scores + draws.repeat(len(actions), 1)).masked_fill(~mask, -torch.inf).argmax(1).tolist()
            for i, slot in enumerate(selected):
                if alive[i]:
                    states[i][seat].append(pool[slot])
        for i, rosters in enumerate(states):
            worlds[i // trials].append(rosters if alive[i] else None)
        return worlds

    def complete_pairwise(self, own, foe, available, card, pick, order):
        rosters = [list(own) + [card], list(foe)]
        remaining = [other for other in available if other != card]
        for turn, side in enumerate(order[pick:], pick + 1):
            used_items = {self.prior.items[mon] for mon in rosters[side]}
            legal = tuple({"type": "draft_pick", **self.cards[mon]} for mon in remaining if self.prior.items[mon] not in used_items)
            if not legal:
                return None
            observation = {"cards": [self.cards[mon] for mon in remaining], "pick": turn,
                           "picked": [self.cards[mon] for mon in rosters[side]],
                           "opponent_picked": [self.cards[mon] for mon in rosters[1 - side]]}
            action = self.pairwise.rank(Decision(DRAFT, observation, legal), "value")[0]
            mon = self.prior.index[action["card_id"]]
            rosters[side].append(mon)
            remaining.remove(mon)
        return rosters


class TailReview(DraftRollout):
    """Spend more distinct battle seeds on short, nearly determined draft tails."""

    def compare(self, baseline, alternative, rng, seconds_left):
        ends = perf_counter() + min(2., (15. if seconds_left is None else seconds_left) - .3)
        self.activity["rollout_checks"] += 1
        gains = []
        try:
            for reference, candidate in zip(baseline[:3], alternative[:3], strict=True):
                if reference is None or candidate is None:
                    return False
                gain = 0.
                if reference != candidate:
                    for _ in range(3):
                        seed = tuple(rng.randrange(65536) for _ in range(4))
                        gain += self._play(candidate, seed, ends) - self._play(reference, seed, ends)
                gains.append(gain)
        except (ValueError, OSError, RuntimeError):
            self.activity["rollout_fallbacks"] += 1
            return False
        return min(gains) >= -1 and sum(gains) >= 3
