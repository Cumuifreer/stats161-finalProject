"""Use locally learned public action values to nominate legal search candidates."""

import torch

from .draft_league import read
from .features import BattleEncoder
from .model import ActionRankingNetwork
from .reply_learning import _ages, encode
from .sim import fingerprint


class ActionScout85:
    def __init__(self, checkpoint, cards, profiles):
        config = read(checkpoint / "config.json")
        if config["catalog"] != fingerprint(cards):
            raise ValueError("The proposal network expects its public card catalog")
        self.encoder = BattleEncoder(profiles)
        self.model = ActionRankingNetwork(**config["network"]).eval()
        self.model.load_state_dict(torch.load(checkpoint / "state_dict.pt", map_location="cpu", weights_only=True))

    @torch.inference_mode()
    def proposals(self, decision, history, events, limit=5):
        state, actions = encode(self.encoder, decision, history, _ages(events))
        scores = self.model(torch.tensor([state]), torch.tensor([actions]))[0]
        return [decision.actions[i] for i in scores.topk(min(limit, len(scores))).indices.tolist()]
