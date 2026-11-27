"""Keep the draft and preview fixed while changing the entire Battle planner."""

from pathlib import Path
from hashlib import sha256
from unittest.mock import patch

from .. import blind85, lab85, quicksilver80
from ..game import BATTLE_BANK, DECISION_LIMIT, BATTLE
from ..research80.scouting.replies import ForecastReplies80
from ..sim import SEATS
from . import distilled


class FullRolloutForecast(ForecastReplies80):
    def __call__(self, decision, seat, evidence):
        if len(decision.actions) < 2 or evidence is None:
            return super().__call__(decision, seat, evidence)
        turn = decision.observation["turn"]
        maximum = 3000 if evidence["force_switch"] else (8000 if turn <= 1 else 6000)
        left = 1000 * min(decision.seconds_left or DECISION_LIMIT[BATTLE], decision.bank_seconds_left or BATTLE_BANK)
        budget = min(maximum, left - 11000)
        if budget < 1400:
            return super().__call__(decision, seat, evidence)
        foe = next(player for player in SEATS if player != seat)
        rosters = {seat: self.our_roster, foe: self.joint.history.opponent_roster}
        prior = self.reply_scout.budget_ms
        try:
            self.reply_scout.budget_ms = int(budget)
            moves, rollouts = self.reply_scout.proposals(decision, seat, rosters, evidence)
        except (KeyError, RuntimeError, ValueError, OSError):
            moves, rollouts = [], 0
        finally:
            self.reply_scout.budget_ms = prior
        self.counts["full_rollout_branches"] += rollouts
        if not moves:
            self.counts["full_rollout_fallbacks"] += 1
            return super().__call__(decision, seat, evidence)
        self.counts["full_rollout_actions"] += 1
        self.counts["decisions"] += 1
        self.joint.observe(decision)
        action = decision.validate(moves[0])
        self.joint.record(decision, action)
        return action


def evaluate(*args):
    with patch.object(lab85, "ForecastReplies80", FullRolloutForecast):
        return lab85.evaluate(*args)


def freeze(*args):
    result = FROZEN(*args)
    for path in (Path(__file__), Path(distilled.__file__)):
        result["files"][str(path)] = sha256(path.read_bytes()).hexdigest()
    return result


FROZEN = quicksilver80.freeze


if __name__ == "__main__":
    with patch.object(distilled, "evaluate", evaluate), patch.object(blind85, "measure", distilled.measure), \
         patch.object(quicksilver80, "freeze", freeze):
        quicksilver80.main()
