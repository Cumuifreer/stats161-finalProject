"""Train a seat-symmetric set model for draft and team preview outcomes."""

import argparse
from collections import Counter
import json
from pathlib import Path
from random import Random
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from .game import BATTLE, DRAFT, PREVIEW, ident, make_decision
from .lineup import LineupPolicy, lineup_options
from .policy import SUPPORT_MOVES, heuristic
from .sim import FIRST_DRAFTER_PICKS, fingerprint
from .tactics import load_card_profiles, tactical_candidates

UNKNOWN_LINEUP = 3


class RosterNet(nn.Module):
    def __init__(self, card_count, width=48):
        super().__init__()
        self.cards = nn.Embedding(card_count, width)
        self.roles = nn.Embedding(4, width)
        self.synergy = nn.MultiheadAttention(width, 4, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.compare = nn.Sequential(nn.Linear(3 * width, 2 * width), nn.SiLU(), nn.Linear(2 * width, 1))
        self.register_buffer("prior", torch.zeros(card_count))

    def forward(self, cards, roles):
        token = self.cards(cards) + self.roles(roles)
        flat = token.flatten(0, 1)
        attention = self.synergy(flat, flat, flat, need_weights=False)[0]
        teams = self.norm(flat + attention).mean(1).unflatten(0, (len(cards), 2))
        left, right = teams.unbind(1)
        interaction = left * right
        left_score = self.compare(torch.cat((left, right, interaction), dim=-1)).squeeze(-1)
        right_score = self.compare(torch.cat((right, left, interaction), dim=-1)).squeeze(-1)
        weights = torch.where(roles == UNKNOWN_LINEUP, 1., torch.where(roles > 0, 1.4, .2))
        prior_left, prior_right = (self.prior[cards] * weights).sum(dim=-1).unbind(1)
        return left_score - right_score + prior_left - prior_right


def _load_data(path, index, device):
    data = json.loads(path.read_text(encoding="utf-8"))
    samples = data["samples"]
    cards = torch.tensor([[[index[name] for name in side] for side in row["cards"]] for row in samples], device=device)
    roles = torch.tensor([row["roles"] for row in samples], device=device)
    scores = torch.tensor([row["p1_score"] for row in samples], dtype=torch.float32, device=device)
    return (cards, roles, scores), data["metrics"]


def _fit_prior(cards, scores, card_count):
    counts = F.one_hot(cards, card_count).float().sum(dim=-2)
    design = counts[:, 0] - counts[:, 1]
    ridge = design.T @ design + 32 * torch.eye(card_count, device=cards.device)
    return 4 * torch.linalg.solve(ridge, design.T @ (scores - .5))


@torch.inference_mode()
def _assess(model, data):
    cards, roles, scores = data
    draft = model(cards, torch.full_like(roles, UNKNOWN_LINEUP))
    preview = model(cards, roles)
    prior = model.prior[cards[:, 0]].sum(dim=-1) - model.prior[cards[:, 1]].sum(dim=-1)
    report = {}
    for name, logits in (("draft", draft), ("preview", preview), ("prior", prior)):
        decisive = scores != .5
        report[name] = {"log_loss": F.binary_cross_entropy_with_logits(logits, scores).item(),
                        "brier": (logits.sigmoid() - scores).square().mean().item(),
                        "accuracy": ((logits[decisive] > 0) == (scores[decisive] > .5)).float().mean().item()}
    return report


def train(args):
    if args.output.exists() or not 16 <= args.width <= 256 or args.width % 4:
        raise ValueError("Use a new checkpoint and a roster width divisible by 4 between 16 and 256")
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    cards = json.loads(args.cards.read_text(encoding="utf-8"))
    index = {card["card_id"]: i for i, card in enumerate(cards)}
    digest = fingerprint(cards)
    device = torch.device(args.device)
    train_data, train_meta = _load_data(args.train, index, device)
    val_data, val_meta = _load_data(args.validation, index, device)
    a, b = (meta["eligible_pools"] for meta in (train_meta, val_meta))
    if a.keys() & b.keys() or set(a.values()) & set(b.values()):
        raise ValueError("Roster train and validation seed or pool overlap")
    if any(meta["catalog_sha256"] != digest or meta["schema_version"] != 1 for meta in (train_meta, val_meta)):
        raise ValueError("Roster dataset and checkpoint catalog or schema mismatch")
    network = {"card_count": len(index), "width": args.width}
    model = RosterNet(**network).to(device)
    model.prior.copy_(_fit_prior(train_data[0], train_data[-1], len(index)))
    optimiser = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.015)
    generator = torch.Generator().manual_seed(args.seed)
    best = float("inf")
    ready = perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, 25):
        model.train()
        for batch in torch.randperm(len(train_data[0]), generator=generator).to(device).split(192):
            rosters, roles, y = (column[batch] for column in train_data)
            logits = model(torch.cat((rosters, rosters)), torch.cat((torch.full_like(roles, UNKNOWN_LINEUP), roles)))
            loss = F.binary_cross_entropy_with_logits(logits, y.repeat(2))
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        model.eval()
        metrics = _assess(model, val_data)
        total = metrics["draft"]["log_loss"] + metrics["preview"]["log_loss"]
        if total < best:
            best, selected = total, epoch
            weights = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    elapsed = perf_counter() - ready
    model.load_state_dict(weights)
    config = {"card_ids": list(index), "network": network, "catalog_sha256": digest, "split": {"train": a, "validation": b},
              "seed": args.seed, "epochs": 24, "selected_epoch": selected, "draft_completions": 12,
              "input": "six public card ids per seat, roles (omitted/back/lead/unknown)",
              "training_sampler": {key: train_meta.get(key) for key in ("first_seed", "seeds", "workers", "behavior_prior_sha256")}}
    cuda = {"gpu": torch.cuda.get_device_name(device), "peak_mib": torch.cuda.max_memory_allocated(device) / 2**20} if device.type == "cuda" else {}
    audit = {"device": str(device), "torch": str(torch.__version__), **cuda,
             "parameters": sum(p.numel() for p in model.parameters()), "fitting_seconds": elapsed,
             "train_games": len(train_data[0]), "validation_games": len(val_data[0]),
             "train": _assess(model, train_data), "validation": _assess(model, val_data)}
    args.output.mkdir(parents=True)
    torch.save(weights, args.output / "state_dict.pt")
    probe_cards, probe_roles, _ = val_data
    with torch.inference_mode():
        probe = {"cards": probe_cards[:8].tolist(), "roles": probe_roles[:8].tolist(),
                 "logits": model(probe_cards[:8], probe_roles[:8]).tolist()}
    for label, obj in (("config", config), ("metrics", audit), ("probes", probe)):
        (args.output / f"{label}.json").write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit), flush=True)
    return 0


def _load_model(directory, device):
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    model = RosterNet(**config["network"])
    model.load_state_dict(torch.load(directory / "state_dict.pt", map_location="cpu", weights_only=True))
    return model.to(device).eval(), config


def verify(args):
    torch.set_num_threads(1)
    model, _ = _load_model(args.checkpoint, args.device)
    probe = json.loads((args.checkpoint / "probes.json").read_text(encoding="utf-8"))
    cards, roles = (torch.tensor(probe[field], device=args.device) for field in ("cards", "roles"))
    with torch.inference_mode():
        predicted = model(cards, roles).cpu()
    target = torch.tensor(probe["logits"])
    torch.testing.assert_close(predicted, target, atol=2e-4, rtol=2e-4)
    print(json.dumps({"device": args.device, "n": len(cards), "max_abs_error": (predicted - target).abs().max().item()}))
    return 0


class DraftRanker:
    def __init__(self, cards, model, config):
        self.model = model
        self.device = model.prior.device
        self.count = config["draft_completions"]
        self.index = {card: i for i, card in enumerate(config["card_ids"])}
        by_id = {card["card_id"]: card for card in cards}
        self.items = []
        self.support = []
        self.guard = []
        motifs = {move: 1 << i for i, move in enumerate(sorted(SUPPORT_MOVES))}
        for key in config["card_ids"]:
            card = by_id[key]
            moves = {ident(move) for move in card["moves"]}
            self.items.append(ident(card["item"]))
            self.support.append(sum(motifs.get(move, 0) for move in moves))
            self.guard.append(.2 if "protect" in moves else 0)
        prior = model.prior.cpu()
        self.strength = ((prior - prior.mean()) / prior.std().clamp_min(.05)).tolist()

    def _coverage(self, team):
        covered = 0
        for card in team:
            covered |= self.support[card]
        return covered

    def _support(self, card, covered):
        return (self.support[card] & ~covered).bit_count() + self.guard[card]

    def _finish(self, available, picks, picked, schedule, ply, noise, scenario):
        teams = [list(picks[0]) + [picked], list(picks[1])]
        available = [card for card in available if card != picked]
        items = [{self.items[c] for c in roster} for roster in teams]
        covered = [self._coverage(team) for team in teams]
        for seat in schedule[ply:]:
            valid = [card for card in available if self.items[card] not in items[seat]]
            if not valid:
                return None
            if seat == 1 and scenario % 3 == 0:
                card = max(valid, key=lambda c: self._support(c, covered[seat]) + .20 * noise[seat][c])
            else:
                card = max(valid, key=lambda c: self.strength[c] + .25 * self._support(c, covered[seat]) + noise[seat][c])
            available.remove(card)
            teams[seat].append(card)
            items[seat].add(self.items[card])
            covered[seat] |= self.support[card]
        return teams

    @torch.inference_mode()
    def rank(self, decision, mode):
        board = decision.observation
        try:
            own, foe = ([self.index[card["card_id"]] for card in board[key]] for key in ("picked", "opponent_picked"))
            actions = [(action, self.index[action["card_id"]]) for action in decision.actions]
            available = [self.index[card["card_id"]] for card in board["cards"]]
        except (KeyError, TypeError):
            return None
        covered = self._coverage(own)
        prior = sorted(actions, key=lambda pair: self.strength[pair[1]] + .25 * self._support(pair[1], covered), reverse=True)
        if mode == "prior" or len(actions) <= 1:
            return [action for action, _ in prior]
        pick = board["pick"]
        if pick != 1 + len(own) + len(foe) or not 1 <= pick <= 12:
            return None
        first = pick - 1 in FIRST_DRAFTER_PICKS
        schedule = [int((i in FIRST_DRAFTER_PICKS) != first) for i in range(12)]
        if schedule[:pick-1].count(0) != len(own) or len(set(available) | set(own) | set(foe)) != 18:
            return None
        rng = Random(fingerprint([pick, sorted(available), sorted(own), sorted(foe)]))
        noises = [[[rng.gauss(0, .45) for _ in self.items] for _ in range(2)] for _ in range(self.count)]
        finished, lengths = [], []
        for _, selected in prior:
            completions = [rosters for trial, noise in enumerate(noises)
                           if (rosters := self._finish(available, (own, foe), selected, schedule, pick, noise, trial)) is not None]
            lengths.append(len(completions))
            finished.extend(completions)
        if not finished:
            return None
        cards = torch.tensor(finished, device=self.device)
        probabilities = self.model(cards, torch.full_like(cards, UNKNOWN_LINEUP)).sigmoid().cpu().tolist()
        scores = []
        offset = 0
        for rank, n in enumerate(lengths):
            score = sum(probabilities[offset:offset + n]) / self.count if n else -1.
            scores.append(score - .007 * rank)
            offset += n
        order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        return [prior[i][0] for i in order]


class RosterPolicy(LineupPolicy):
    def __init__(self, profiles, cards, worker, draft, draft_mode, preview_mode, battle=None):
        super().__init__(profiles, cards, worker)
        self.draft = draft
        self.draft_mode = draft_mode
        self.preview_mode = preview_mode
        self.battle = battle.for_match() if battle is not None else None
        self.roster_activity = Counter()
        self.added = None

    @torch.inference_mode()
    def _preview_values(self, decision, options):
        team, foe = ([mon.get("species") for mon in decision.observation[key] if mon]
                     for key in ("team", "opponent_team"))
        if any(len(line) != 6 or not set(line) <= self.cards.keys() for line in (team, foe)):
            return None
        foe_view = make_decision(PREVIEW, {"your_roster": [self.cards[s] for s in foe],
                                          "opponent_roster": [self.cards[s] for s in team]}, [{"roster": foe}], "p2")
        foe_candidates = tactical_candidates(foe_view)
        foe_choices = [foe_candidates[0], heuristic(foe_view)]
        if foe_choices[0] == foe_choices[1] and len(foe_candidates) > 1:
            foe_choices[1] = foe_candidates[1]

        def roles(names, action):
            return [2 if name in action["leads"] else 1 if name in action["bring"] else 0 for name in names]

        order = [[self.draft.index[self.cards[s]["card_id"]] for s in names] for names in (team, foe)]
        role_rows = [[roles(team, our), roles(foe, their)] for our in options for their in foe_choices]
        device = self.draft.device
        identities = torch.tensor([order], device=device).expand(len(role_rows), -1, -1)
        return self.draft.model(identities, torch.tensor(role_rows, device=device)).sigmoid().view(-1, 2)

    def _preview(self, decision):
        options = lineup_options(decision)
        predicted = self._preview_values(decision, options)
        if predicted is None:
            return None
        gain = predicted - predicted[0]
        edge = gain.mean(dim=1).masked_fill(gain.amin(dim=1) < 0, -torch.inf)
        best = int(edge.argmax())
        return options[best] if edge[best] >= .04 else options[0]

    def options(self, decision):
        options = super().options(decision)
        if self.preview_mode != "value-rollout":
            return options
        arranged = {}
        for action in decision.actions:
            arranged.setdefault((frozenset(action["bring"]), frozenset(action["leads"])), action)
        used = {(frozenset(action["bring"]), frozenset(action["leads"])) for action in options}
        outside = [action for key, action in arranged.items() if key not in used]
        if not outside:
            return options
        predicted = self._preview_values(decision, options + outside)
        if predicted is None:
            return options
        means = predicted.mean(dim=1)
        best = int(means[len(options):].argmax()) + len(options)
        if means[best] > means[:len(options)].max() + .025:
            self.added = outside[best - len(options)]
            self.roster_activity["preview_added"] += 1
            return options + [self.added]
        return options

    def __call__(self, decision):
        if self.battle is not None:
            if decision.phase == BATTLE:
                return self.battle(decision)
            if decision.phase == PREVIEW:
                self.battle.observe(decision)
        if decision.phase == DRAFT:
            if self.draft_mode == "heuristic":
                return heuristic(decision)
            ranked = self.draft.rank(decision, self.draft_mode)
            if ranked:
                changed = ranked[0] != heuristic(decision)
                self.different_choices += changed
                self.roster_activity["draft_changed"] += changed
                return ranked[0]
            self.fallbacks += 1
            return heuristic(decision)
        if decision.phase == PREVIEW and self.preview_mode in {"tactical", "value"}:
            self.observe(decision)
            baseline = tactical_candidates(decision)[0]
            if self.preview_mode == "value":
                choice = self._preview(decision)
                if choice is not None:
                    self.different_choices += choice != baseline
                    self.roster_activity["preview_changed"] += choice != baseline
                    return choice
                self.fallbacks += 1
            return baseline
        if decision.phase == PREVIEW and self.preview_mode == "value-rollout":
            self.added = None
            choice = super().__call__(decision)
            self.roster_activity["preview_added_selected"] += self.added is not None and choice == self.added
            return choice
        return super().__call__(decision)


def roster_factory(path, cards, worker, device, draft_mode, preview_mode, battle_checkpoint=None):
    from functools import partial

    torch.set_num_threads(1)
    model, config = _load_model(path, device)
    if fingerprint(cards) != config["catalog_sha256"]:
        raise ValueError("Roster checkpoint requires its original fixed card catalog")
    draft = DraftRanker(cards, model, config)
    battle = None
    if battle_checkpoint:
        from .model import BattlePolicy

        battle = BattlePolicy.load(battle_checkpoint, device)
    return partial(RosterPolicy, load_card_profiles(cards, with_sets=True),
                   {ident(card["species"]): card for card in cards}, worker, draft,
                   draft_mode=draft_mode, preview_mode=preview_mode, battle=battle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--cards", type=Path, default=Path(__file__).resolve().parents[1] / "data/historical_cards.json")
    fit.add_argument("--train", type=Path, required=True)
    fit.add_argument("--validation", type=Path, required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--seed", type=int, default=921)
    fit.add_argument("--width", type=int, default=48)
    fit.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    fit.set_defaults(run=train)
    probe = commands.add_parser("verify")
    probe.add_argument("--checkpoint", type=Path, required=True)
    probe.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    probe.set_defaults(run=verify)
    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
