"""Evaluate the distilled reply expert on fresh development pools."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from multiprocessing import get_context
from pathlib import Path
from unittest.mock import patch

from .. import blind85, lab85, quicksilver80
from ..game import ident
from ..joint import BattleHistory, PROTECT
from ..reply_learning import LearnedReplies
from ..research80.scouting.replies import ForecastReplies80
from ..search import _key
from ..tactics import load_card_profiles


CHECKPOINT = Path("checkpoints/research80/reply-distilled-v1")


class DistilledForecast(ForecastReplies80):
    def __init__(self, *args, distilled, **settings):
        super().__init__(*args, **settings)
        self.distilled = distilled

    def observe_style(self, agreement):
        super().observe_style(agreement)
        if self.reply_model_active:
            self.expert_prior = {"neutral": .24, "ace-code": .06, "pivot": .06, "continuation": .31, "distilled": .33}

    def _supplemental_replies(self, enemy, foe, rosters, view, evidence, deadline):
        original = super()._supplemental_replies(enemy, foe, rosters, view, evidence, deadline)
        if not self.reply_model_active or len(enemy.actions) < 2:
            return original
        history = BattleHistory()
        history.observe(enemy)
        history.opponent_roster = self.our_roster
        turn = 0
        for event in evidence["events"]:
            fields = event.split("|")
            if fields[1] == "turn":
                turn = int(fields[2])
            elif fields[1] in {"switch", "drag"} and not fields[2].startswith(foe):
                history.revealed.add(ident(fields[3].split(",", 1)[0]))
            elif fields[1] == "move" and fields[2].startswith(foe) and ident(fields[3]) in PROTECT:
                name = ident(fields[2].split(": ", 1)[-1])
                history.protected_at[name] = turn
        probabilities = self.distilled.probabilities(enemy, history, evidence["events"])
        ranked = sorted(zip(enemy.actions, probabilities, strict=True), key=lambda pair: pair[1], reverse=True)[:8]
        normalizer = sum(weight for _, weight in ranked)
        original["distilled"] = {_key(action): weight / normalizer for action, weight in ranked}
        self.counts["distilled_worlds"] += 1
        return original


def evaluate(roots, cards, rivals, *args):
    expert = LearnedReplies(CHECKPOINT, load_card_profiles(cards, with_sets=True))

    class BoundForecast(DistilledForecast):
        def __init__(self, *params, **settings):
            super().__init__(*params, distilled=expert, **settings)

    with patch.object(lab85, "ForecastReplies80", BoundForecast):
        return lab85.evaluate(roots, cards, rivals, *args)


def measure(roots, cards, rivals, checkout, variant, workers, *args):
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        jobs = [pool.submit(evaluate, [root], cards, rivals, checkout, variant, *args) for root in roots]
        for job in as_completed(jobs):
            yield job.result()


def freeze(*args):
    result = FROZEN(*args)
    paths = [Path(__file__), CHECKPOINT / "config.json", CHECKPOINT / "state_dict.pt"]
    for path in paths:
        result["files"][str(path)] = sha256(path.read_bytes()).hexdigest()
    return result


FROZEN = quicksilver80.freeze


if __name__ == "__main__":
    with patch.object(blind85, "measure", measure), patch.object(quicksilver80, "freeze", freeze):
        quicksilver80.main()
