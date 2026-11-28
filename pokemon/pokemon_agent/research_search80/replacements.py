"""Ablate the existing position-aware rule for forced replacements."""

from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

from .. import blind85, lab85, quicksilver80
from ..research80.scouting.replies import ForecastReplies80
from . import distilled


class ReplacementForecast(ForecastReplies80):
    def __init__(self, *args, **settings):
        super().__init__(*args, **(settings | {"replacements": True}))


def evaluate(*args):
    with patch.object(lab85, "ForecastReplies80", ReplacementForecast):
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
