"""Predict replies by running independent local policies on hypothetical player views."""

from dataclasses import replace

from .game import BATTLE, BATTLE_BANK, DECISION_LIMIT
from .search import _key
from .sim import SEATS, fingerprint
from .strength import ExternalPivot


class ProxyReplies:
    def __init__(self, stack, checkout, cards):
        self.cards = cards
        self.models = {name: stack.enter_context(ExternalPivot(checkout, name, list(cards.values())))
                       for name in ("ace-code", "pivot")}

    def probabilities(self, decision, seat, rosters, view, public_log):
        # The forecast actor's request and Preview come from the reconstructed world.
        options = {"pivot": {**view["pivot"], "protocol_log": public_log}, "templates": view["templates"]}
        seed = int(fingerprint(["proxy-reply", seat, decision.observation["turn"], public_log])[:12], 16)
        decision = replace(decision, seconds_left=DECISION_LIMIT[BATTLE], bank_seconds_left=BATTLE_BANK)
        drafted = {player: [self.cards[mon] for mon in rosters[player]] for player in SEATS}
        predictions = {}
        for name, model in self.models.items():
            try:
                model.setup(seed, seat, "p1")
                model.start(seed, seat, drafted)
                action = model.choose(decision, options)
            except (ValueError, RuntimeError):
                continue
            selected = _key(action)
            predictions[name] = [float(_key(legal) == selected) for legal in decision.actions]
        return predictions
