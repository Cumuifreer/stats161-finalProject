"""Test whether a human VGC tactical prior improves full local matches."""

from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

from .. import blind85, lab85, quicksilver80
from ..research80.scouting.replies import ForecastReplies80
from ..search import _key
from . import distilled
from .human import HumanPrior


CHECKPOINT = Path("checkpoints/research80/human-joint-v1")


class HumanForecast(ForecastReplies80):
    def __init__(self, *args, human, **settings):
        super().__init__(*args, **settings)
        self.human = human

    def observe_style(self, agreement):
        super().observe_style(agreement)
        self.expert_prior = {**{name: .80 * mass for name, mass in self.expert_prior.items()}, "human": .20}

    def _supplemental_replies(self, enemy, foe, rosters, view, evidence, deadline):
        learned = super()._supplemental_replies(enemy, foe, rosters, view, evidence, deadline)
        probabilities = self.human.probabilities(enemy, evidence["events"])
        if probabilities:
            learned["human"] = {_key(action): weight for action, weight in probabilities}
            self.counts["human_prior_worlds"] += 1
        return learned


def evaluate(roots, cards, rivals, *args):
    expert = HumanPrior(CHECKPOINT)

    class BoundForecast(HumanForecast):
        def __init__(self, *params, **settings):
            super().__init__(*params, human=expert, **settings)

    with patch.object(lab85, "ForecastReplies80", BoundForecast):
        return lab85.evaluate(roots, cards, rivals, *args)


def freeze(*args):
    result = FROZEN(*args)
    paths = [Path(__file__), Path(__file__).with_name("human.py"), Path(distilled.__file__),
             CHECKPOINT / "config.json", CHECKPOINT / "state_dict.pt"]
    for path in paths:
        result["files"][str(path)] = sha256(path.read_bytes()).hexdigest()
    return result


FROZEN = quicksilver80.freeze


if __name__ == "__main__":
    with patch.object(distilled, "evaluate", evaluate), patch.object(blind85, "measure", distilled.measure), \
         patch.object(quicksilver80, "freeze", freeze):
        quicksilver80.main()
