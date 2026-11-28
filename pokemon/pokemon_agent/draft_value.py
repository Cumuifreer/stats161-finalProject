"""Evaluate draft completions projected from the acting player's public board."""

from collections import Counter
import json
from random import Random
from time import perf_counter

import torch
from torch import nn

from .game import PREVIEW
from .joint import JointPolicy
from .sim import FIRST_DRAFTER_PICKS, SEATS, fingerprint, play_battle
from .tactics import tactical_candidates


class DraftValue(nn.Module):
    def __init__(self, card_count, width=64):
        super().__init__()
        self.cards = nn.Embedding(card_count, width)
        self.sides = nn.Embedding(2, width)
        self.matchup = nn.TransformerEncoderLayer(width, 4, 2 * width, dropout=0, batch_first=True)
        self.score = nn.Sequential(nn.Linear(width * 3, width), nn.SiLU(), nn.Linear(width, 1))
        self.register_buffer("prior", torch.zeros(card_count))
        self.register_buffer("side_ids", torch.arange(2).view(1, 2, 1), persistent=False)

    def forward(self, rosters):
        both = torch.cat((rosters, rosters.flip(1)))
        entities = self.cards(both) + self.sides(self.side_ids)
        context = self.matchup(entities.flatten(1, 2)).reshape(-1, 2, 6, entities.size(-1)).mean(2)
        own, foe = context.unbind(1)
        first, second = self.score(torch.cat((own, foe, own * foe), dim=-1)).squeeze(-1).chunk(2)
        prior = self.prior[rosters].sum(-1)
        return first - second + prior[:, 0] - prior[:, 1]


class PairwiseDraftValue(nn.Module):
    def __init__(self, card_count, width=32):
        super().__init__()
        self.cards = nn.Embedding(card_count, 3 * width)
        self.synergy = nn.Parameter(torch.zeros(width))
        self.matchup = nn.Parameter(torch.zeros(width))
        self.register_buffer("prior", torch.zeros(card_count))

    def forward(self, rosters):
        combo, lead, answer = self.cards(rosters).chunk(3, dim=-1)
        # The squared sum gives every within-team pair. Subtract self-pairs before comparing sides.
        combinations = ((combo.sum(2).square() - combo.square().sum(2)) / 30 * self.synergy).sum(-1)
        pressure = (lead.mean(2) * answer.mean(2).flip(1) * self.matchup).sum(-1)
        score = self.prior[rosters].sum(-1) + combinations + pressure
        return score[:, 0] - score[:, 1]


NETWORKS = {"attention": DraftValue, "pairwise": PairwiseDraftValue}


class DraftValueRanker:
    def __init__(self, baseline, net, margin, mode, rollout=None):
        self.baseline = baseline
        self.net = net
        self.margin = margin
        self.mode = mode
        self.rollout = rollout
        self.activity = Counter()
        prior = net.prior.detach().cpu()
        self.refit_strength = ((prior - prior.mean()) / prior.std().clamp_min(.05)).tolist()

    @torch.inference_mode()
    def rank(self, decision, mode):
        baseline = self.baseline
        prior = baseline.rank(decision, "prior")
        if not prior or mode == "prior" or len(prior) < 2:
            return prior
        board = decision.observation
        if self.mode == "refit":
            covered = baseline._coverage([baseline.index[card["card_id"]] for card in board["picked"]])
            ranked = [(action, baseline.index[action["card_id"]]) for action in prior]
            ordered = [action for action, _ in sorted(ranked, key=lambda row: self.refit_strength[row[1]]
                                                        + .25 * baseline._support(row[1], covered), reverse=True)]
            self.activity["draft_overrides"] += ordered[0] != prior[0]
            return ordered
        pick = board["pick"]
        if not isinstance(pick, int) or not 1 <= pick <= 8:
            return prior
        own, foe, available = [[baseline.index[card["card_id"]] for card in board[key]]
                               for key in ("picked", "opponent_picked", "cards")]
        schedule = [int((i in FIRST_DRAFTER_PICKS) != (pick - 1 in FIRST_DRAFTER_PICKS)) for i in range(12)]
        public_pool = own + foe + available
        legal_ids = {baseline.index[action["card_id"]] for action in prior}
        if (pick != 1 + len(own) + len(foe) or schedule[:pick - 1].count(0) != len(own)
                or len(public_pool) != 18 or len(set(public_pool)) != 18 or not legal_ids <= set(available)):
            return prior
        self.activity["draft_searches"] += 1
        rng = Random(fingerprint([pick, sorted(own), sorted(foe), sorted(available)]))
        # Scenario zero projects the frozen baseline on both sides. The remainder vary each side's preference.
        noises = [[[0.] * len(baseline.items) for _ in range(2)]]
        noises.extend([[rng.gauss(0, sigma) for _ in baseline.items] for sigma in (.18, .24 if trial % 2 else .34)]
                      for trial in range(11))
        options = prior[:4]
        completions = [[baseline._finish(available, (own, foe), baseline.index[action["card_id"]], schedule, pick, noise, 1)
                        for noise in noises] for action in options]
        device = self.net.prior.device
        live = torch.tensor([[team is not None for team in group] for group in completions], device=device)
        if not live[0].any():
            return prior
        placeholder = next(team for team in completions[0] if team is not None)
        teams = torch.tensor([team if team is not None else placeholder for group in completions for team in group],
                             device=device)
        if self.mode in {"linear", "search-linear"}:
            strength = self.net.prior[teams].sum(-1)
            logits = strength[:, 0] - strength[:, 1]
        else:
            logits = self.net(teams)
        scores = (logits.sigmoid().reshape(len(options), -1) * live).mean(-1).cpu().tolist()
        best = max(range(1, len(options)), key=lambda i: scores[i] - .006 * i)
        if self.rollout is not None:
            # Use the learned value to propose one alternative. Compare it on the same completions and battle seeds.
            if scores[best] < scores[0] - .08:
                return prior
            selected = self.rollout.compare(completions[0], completions[best], rng, decision.seconds_left)
            if not selected:
                return prior
        elif scores[best] - scores[0] < self.margin + .006 * best:
            return prior
        self.activity["draft_overrides"] += 1
        return [prior[best], *prior[:best], *prior[best + 1:]]


class DraftRollout:
    def __init__(self, engine, cards, profiles):
        self.engine = engine
        self.cards = cards
        self.profiles = profiles
        self.activity = Counter()

    def compare(self, baseline, alternative, rng, seconds_left):
        ends = perf_counter() + min(2., (15. if seconds_left is None else seconds_left) - .3)
        seed_draws = [tuple(rng.randrange(65536) for _ in range(4)) for _ in range(2)]
        cache = {}
        gains = []
        self.activity["rollout_checks"] += 1
        try:
            for old, new in zip(baseline[:3], alternative[:3], strict=True):
                if old is None or new is None:
                    return False
                if old == new:
                    gains.append(0.)
                    continue
                result = []
                for rosters in (old, new):
                    key = tuple(tuple(roster) for roster in rosters)
                    if key not in cache:
                        cache[key] = sum(self._play(rosters, seed, ends) for seed in seed_draws)
                    result.append(cache[key])
                gains.append(result[1] - result[0])
        except (ValueError, OSError, RuntimeError):
            self.activity["rollout_fallbacks"] += 1
            return False
        return min(gains) >= 0 and sum(gains) >= 2

    def _play(self, rosters, seed, ends):
        inputs = {seat: [self.cards[card] for card in team] for seat, team in zip(SEATS, rosters, strict=True)}
        policies = {seat: JointPolicy(self.profiles) for seat in SEATS}

        def choose(seat, decision, _):
            if perf_counter() >= ends:
                raise TimeoutError("Draft rollout exceeded its decision budget")
            policy = policies[seat]
            if decision.phase == PREVIEW:
                policy.observe(decision)
                action = tactical_candidates(decision)[0]
            else:
                action = policy(decision)
            return decision.validate(action)

        self.activity["rollout_games"] += 1
        winner = play_battle(self.engine, inputs, policies, 0, seed, choose)
        return .5 if winner is None else float(winner == "p1")


def load_value(path, device):
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    net = NETWORKS[config.get("architecture", "attention")](**config["network"])
    net.load_state_dict(torch.load(path / "state_dict.pt", map_location="cpu", weights_only=True))
    return net.to(device).eval(), config
