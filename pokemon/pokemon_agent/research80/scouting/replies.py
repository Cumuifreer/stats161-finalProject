"""Use a continuation expert to estimate early opposing actions on public hypotheses."""

from time import perf_counter

from ...search import _key
from ..endgame import CheckedEndgame


class ForecastReplies80(CheckedEndgame):
    def __init__(self, *args, reply_scout, **settings):
        super().__init__(*args, **settings)
        self.reply_scout = reply_scout
        self.reply_model_active = False
        self.reply_cache = None
        self.reply_position = None

    def observe_style(self, agreement):
        if agreement:
            hits, eligible = agreement
            if eligible >= 3 and hits / eligible >= .75:
                self.reply_model_active = True
                self.move_budget += 2.10
                self.expert_prior = {"neutral": .31, "ace-code": .12, "pivot": .12, "continuation": .45}
                self.counts["independent_reply_model_activated"] += 1

    def _supplemental_replies(self, enemy, foe, rosters, view, evidence, deadline):
        turn = enemy.observation["turn"]
        if not self.reply_model_active or turn > 4 or len(enemy.actions) < 4:
            return {}
        events = evidence["events"]
        key = (turn, len(events))
        if key != self.reply_position:
            self.reply_position = key
            self.reply_cache = None
        if self.reply_cache is None:
            if perf_counter() + 1.95 >= deadline:
                return {}
            try:
                forecast, rollouts = self.reply_scout.proposals(
                    enemy, foe, rosters, {"own_proxy": view, "protocol_log": evidence["protocol_log"]},
                )
            except (KeyError, RuntimeError, ValueError, OSError):
                self.counts["independent_reply_model_error"] += 1
                return {}
            self.counts["independent_reply_model_calls"] += 1
            self.counts["independent_reply_model_rollouts"] += rollouts
            self.reply_cache = {_key(action): weight for action, weight in zip(forecast, (.60, .22, .12, .06))}
        legal = {_key(action) for action in enemy.actions}
        support = {choice: weight for choice, weight in self.reply_cache.items() if choice in legal}
        if not support:
            return {}
        self.counts["independent_replies_scored"] += 1
        total = sum(support.values())
        return {"continuation": {choice: weight / total for choice, weight in support.items()}}
