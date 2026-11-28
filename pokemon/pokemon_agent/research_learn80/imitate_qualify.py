"""Test the distilled search policy in complete matches with its legal action menu."""

from time import perf_counter
from unittest.mock import patch

import torch

from ..battle_league import LearnedPolicy
from ..game import BATTLE
from . import qualify
from .distill_search import SearchExamples


class SearchImitation(LearnedPolicy):
    def __init__(self, profiles, encoder, model, *, value_only):
        super().__init__(profiles, SearchExamples(profiles), model, value_only=False)

    @torch.inference_mode()
    def __call__(self, decision):
        self.observe(decision)
        if decision.phase != BATTLE:
            return super().__call__(decision)
        if len(decision.actions) == 1:
            action = decision.actions[0]
            self.record(decision, action)
            return action
        began = perf_counter()
        features, menu = self.encoder.encode_menu(decision, self)
        state, actions, cards, mons, legal = features
        missing = self.model.history_steps - len(self.trace)
        trace = [(0.,) * (len(state) + len(actions[0]))] * missing + self.trace
        logits, _, _ = self.model(torch.tensor([state]), torch.tensor([actions]), torch.tensor([cards]),
                                  torch.tensor([mons]), torch.tensor([legal]), torch.tensor([trace]))
        index = int(logits[0].argmax())
        action = menu[index]
        self.remember(features, index)
        self.record(decision, action)
        self.inference_ms.append((perf_counter() - began) * 1000)
        self.scored += 1
        self.changes += index != 0
        return action


def evaluate(*args):
    with patch.object(qualify, "Learner", SearchImitation):
        return EVALUATE(*args)


EVALUATE = qualify.evaluate


if __name__ == "__main__":
    with patch.object(qualify, "evaluate", evaluate):
        qualify.main()
