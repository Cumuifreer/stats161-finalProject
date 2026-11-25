"""A portable ranker over the action list of a battle Decision."""

import json
from pathlib import Path

import torch
from torch import nn

from .features import FEATURE_VERSION, HISTORY_FEATURE_VERSION, BattleEncoder
from .game import BATTLE, Decision
from .joint import BattleHistory
from .tactics import tactical


class ActionRankingNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int):
        super().__init__()
        self.state = nn.Linear(state_size, hidden_size)
        self.action = nn.Linear(action_size, hidden_size)
        self.score = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, 1))

    def forward(self, states, actions):
        return self._score_actions(self.state(states), actions)

    def _score_actions(self, state, actions):
        context = state.unsqueeze(1) + self.action(actions)
        scores = self.score(context).squeeze(-1)
        # A low-weight tactical prior is shared by training and inference.
        return scores + actions[..., 0] + actions[..., 1]


class ActorCriticNetwork(ActionRankingNetwork):
    def __init__(self, state_size: int, action_size: int, hidden_size: int):
        super().__init__(state_size, action_size, hidden_size)
        self.value_head = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, 1))

    def policy_value(self, states, actions):
        state = self.state(states)
        return self._score_actions(state, actions), self.value_head(state).squeeze(-1).tanh()


class BattlePolicy:
    def __init__(self, model: ActionRankingNetwork, device: torch.device, *,
                 encoder: BattleEncoder | None = None, shortlist_only: bool = False):
        self.model = model.to(device).eval()
        self.device = device
        self.encoder = encoder if encoder is not None else BattleEncoder()
        self.shortlist_only = shortlist_only
        self.history = BattleHistory()

    @classmethod
    def load(cls, directory: Path, device: str = "cpu", *, shortlist_only: bool | None = None) -> "BattlePolicy":
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        version = config["feature_version"]
        if version not in (FEATURE_VERSION, HISTORY_FEATURE_VERSION):
            raise ValueError(f"Unsupported BC feature version {version}")
        profiles = config["card_profiles"] if version == HISTORY_FEATURE_VERSION else None
        architecture = config.get("architecture", "ranking")
        if architecture not in ("ranking", "actor_critic"):
            raise ValueError(f"Unsupported model architecture {architecture}")
        network_type = ActorCriticNetwork if architecture == "actor_critic" else ActionRankingNetwork
        model = network_type(**config["network"])
        state_dict = torch.load(directory / "state_dict.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        return cls(model, torch.device(device), encoder=BattleEncoder(profiles),
                   shortlist_only=config.get("shortlist_only", False) if shortlist_only is None else shortlist_only)

    def for_match(self) -> "BattlePolicy":
        return BattlePolicy(self.model, self.device, encoder=self.encoder, shortlist_only=self.shortlist_only)

    def observe(self, decision: Decision) -> None:
        self.history.observe(decision)

    def encode(self, decision: Decision, *, history: BattleHistory | None = None):
        return self.encoder.encode(decision, history=self.history if history is None else history,
                                   shortlist_only=self.shortlist_only)

    @torch.inference_mode()
    def evaluate(self, decision: Decision, *, history: BattleHistory | None = None) -> tuple[tuple[int, ...], torch.Tensor]:
        features = self.encode(decision, history=history)
        state = torch.tensor([features.state], dtype=torch.float32, device=self.device)
        actions = torch.tensor([features.actions], dtype=torch.float32, device=self.device)
        return features.action_indices, self.model(state, actions)[0].cpu()

    def scores(self, decision: Decision) -> torch.Tensor:
        return self.evaluate(decision)[1]

    def __call__(self, decision: Decision) -> dict:
        self.observe(decision)
        if decision.phase != BATTLE:
            return tactical(decision)
        action = decision.actions[0]
        if len(decision.actions) > 1:
            indices, scores = self.evaluate(decision)
            action = decision.actions[indices[int(scores.argmax())]]
        self.history.record(decision, action)
        return action
