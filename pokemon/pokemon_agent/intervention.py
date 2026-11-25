"""Learn candidate overrides from single-action interventions in seeded games."""

from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from .features import HISTORY_FEATURE_VERSION, BattleEncoder
from .game import BATTLE, DRAFT, PREVIEW
from .joint import JointPolicy
from .pivot import PivotOpponent
from .policy import heuristic
from .sim import SEATS, Showdown, fingerprint, play_game, pool_fingerprint
from .tactics import load_card_profiles, tactical


class OutcomeNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        self.board = nn.Linear(state_size, hidden_size)
        self.choices = nn.Linear(action_size, hidden_size)
        self.output = nn.Sequential(nn.LayerNorm(hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size),
                                    nn.SiLU(), nn.Linear(hidden_size, 1))

    def forward(self, states, actions):
        return self.output(self.board(states).unsqueeze(1) + self.choices(actions)).squeeze(-1)


def _encode(encoder, history, decision, ranked):
    features = encoder.encode(decision, history=history, shortlist_only=True)
    indices = {index: row for index, row in zip(features.action_indices, features.actions, strict=True)}
    baseline = ranked[0][1]["score"]
    actions = tuple(indices[decision.actions.index(action)] + (details["score"], details["score"] - baseline, i / 2)
                    for i, (action, details) in enumerate(ranked))
    return torch.tensor(features.state, dtype=torch.float32), torch.tensor(actions, dtype=torch.float32)


class TrialPolicy(JointPolicy):
    def __init__(self, profiles, turn, branch, encoder):
        super().__init__(profiles)
        self.turn = turn
        self.branch = branch
        self.encoder = encoder
        self.root = None

    def __call__(self, decision):
        ranked = self.rank(decision)
        index = 0
        if (self.root is None and decision.phase == BATTLE and decision.observation["turn"] == self.turn
                and len(ranked) >= 3):
            self.root = _encode(self.encoder, self.history, decision, ranked[:3])
            index = self.branch
        action = ranked[index][0]
        self.record(decision, action)
        return action


def _sample(cards, profiles, checkout, seeds):
    torch.set_num_threads(1)
    encoder = BattleEncoder(profiles)
    rows = []
    with Showdown() as engine, PivotOpponent(checkout) as pivot:
        for seed in seeds:
            opponent = seed % 4
            pivot.style = opponent == 2
            for seat, other in (SEATS, SEATS[::-1]):
                root = None
                scores = []
                for branch in range(3):
                    learner = TrialPolicy(profiles, turn=1 + (seed // 4) % 3, branch=branch, encoder=encoder)
                    contender = (JointPolicy(profiles) if opponent == 0 else tactical if opponent == 1 else pivot)
                    samples = {player: [] for player in SEATS}
                    winner = play_game(engine, cards, {seat: learner, other: contender}, seed, samples,
                                       fixed_policies={DRAFT: heuristic, PREVIEW: heuristic})
                    if any(event for records in samples.values() for _, _, event in records):
                        raise RuntimeError(f"An intervention game was invalid or timed out at {seed}/{seat}")
                    if learner.root is None and branch == 0:
                        break
                    if branch == 0:
                        root = learner.root
                    elif learner.root is None or not all(torch.equal(a, b) for a, b in zip(root, learner.root, strict=True)):
                        raise RuntimeError(f"Intervention replay diverged before the chosen action at {seed}/{seat}")
                    scores.append(.5 if winner is None else float(winner == seat))
                if root is not None:
                    rows.append((*root, torch.tensor(scores)))
    # Only player-view features and terminal results leave the simulator worker.
    return tuple(torch.stack(items) for items in zip(*rows, strict=True))


class InterventionPolicy(JointPolicy):
    def __init__(self, profiles, model, encoder, threshold):
        super().__init__(profiles)
        self.model = model
        self.encoder = encoder
        self.threshold = threshold
        self.opportunities = 0
        self.different_choices = 0

    def for_match(self):
        return InterventionPolicy(self.profiles, self.model, self.encoder, self.threshold)

    @classmethod
    def load(cls, checkpoint, device, threshold=None):
        config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        if config["feature_version"] != HISTORY_FEATURE_VERSION:
            raise ValueError("Unsupported intervention feature version")
        model = OutcomeNetwork(**config["network"])
        model.load_state_dict(torch.load(checkpoint / "state_dict.pt", map_location="cpu", weights_only=True))
        profiles = config["profiles"]
        threshold = config["threshold"] if threshold is None else threshold
        if not 0 <= threshold <= 1:
            raise ValueError("Intervention threshold must be a win-probability difference between 0 and 1")
        return cls(profiles, model.to(device).eval(), BattleEncoder(profiles), threshold)

    @torch.inference_mode()
    def __call__(self, decision):
        ranked = self.rank(decision)
        choice = ranked[0][0]
        if (self.different_choices == 0 and decision.phase == BATTLE
                and decision.observation["turn"] in (1, 2, 3) and len(ranked) >= 3):
            self.opportunities += 1
            state, actions = _encode(self.encoder, self.history, decision, ranked[:3])
            device = self.model.board.weight.device
            probs = self.model(state[None].to(device), actions[None].to(device))[0].sigmoid().cpu()
            best = int(probs.argmax())
            if probs[best] - probs[0] > self.threshold:
                choice = ranked[best][0]
                self.different_choices += 1
        self.record(decision, choice)
        return choice


def _loss(scores, targets):
    probs = scores.sigmoid()
    return F.binary_cross_entropy_with_logits(scores, targets) + 2 * F.mse_loss(
        probs[:, 1:] - probs[:, :1], targets[:, 1:] - targets[:, :1])


@torch.inference_mode()
def _diagnose(model, data):
    states, actions, targets = data
    probs = model(states, actions).sigmoid()
    observed = targets[:, 1:] - targets[:, :1]
    predicted = probs[:, 1:] - probs[:, :1]
    informative = observed != 0
    direction = (predicted.sign() == observed.sign())[informative]
    return {"roots": len(states), "changed_outcomes": informative.sum().item(),
            "paired_mse": (predicted - observed).square().mean().item(),
            "zero_delta_mse": observed.square().mean().item(),
            "direction_accuracy": direction.float().mean().item() if direction.numel() else None}


def train(args):
    if min(args.workers, args.pairs, args.validation_pairs) < 1 or args.output.exists():
        raise ValueError("Use positive training sizes and a new intervention output directory")
    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    cards = json.loads(args.cards.read_text(encoding="utf-8"))
    profiles = load_card_profiles(cards, with_sets=True)
    spans = (range(args.seed, args.seed + args.pairs), range(args.validation_seed, args.validation_seed + args.validation_pairs))
    split = {name: {str(seed): pool_fingerprint(cards, seed) for seed in seeds}
             for name, seeds in zip(("train", "validation"), spans, strict=True)}
    train_split, val_split = split.values()
    if train_split.keys() & val_split.keys() or set(train_split.values()) & set(val_split.values()):
        raise ValueError("Intervention training and validation overlap")
    began = perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as executor:
        groups = [[executor.submit(_sample, cards, profiles, args.ace_trainer, seeds[i::args.workers])
                   for i in range(min(len(seeds), args.workers))] for seeds in spans]
        datasets = []
        for group in groups:
            batches = [batch for job in group if (batch := job.result())]
            if not batches:
                raise ValueError("A split had no eligible three-candidate intervention roots")
            datasets.append(tuple(torch.cat(rows).to(device) for rows in zip(*batches, strict=True)))
        train_data, val_data = datasets
    seconds = perf_counter() - began
    model = OutcomeNetwork(train_data[0].shape[-1], train_data[1].shape[-1]).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=.001)
    best_weights, best_epoch, best_mse = None, None, float("inf")
    began = perf_counter()
    for epoch in range(1, 13):
        for index in torch.randperm(len(train_data[0])).split(64):
            states, actions, targets = (tensor[index] for tensor in train_data)
            loss = _loss(model(states, actions), targets)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        metrics = _diagnose(model, val_data)
        if metrics["paired_mse"] < best_mse:
            best_mse = metrics["paired_mse"]
            best_epoch = epoch
            best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_weights)
    config = {"network": {"state_size": train_data[0].shape[-1], "action_size": train_data[1].shape[-1]},
              "feature_version": HISTORY_FEATURE_VERSION, "profiles": profiles, "threshold": .10,
              "catalog_sha256": fingerprint(cards), "split": split, "workers": args.workers,
              "format": "single early-turn intervention, Joint continuation", "device": args.device}
    report = {"train": _diagnose(model, train_data), "validation": _diagnose(model, val_data),
              "sampling_seconds": seconds, "training_seconds": perf_counter() - began, "selected_epoch": best_epoch,
              "parameters": sum(p.numel() for p in model.parameters())}
    args.output.mkdir(parents=True)
    torch.save(best_weights, args.output / "state_dict.pt")
    for name, data in (("config", config), ("metrics", report)):
        (args.output / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0
