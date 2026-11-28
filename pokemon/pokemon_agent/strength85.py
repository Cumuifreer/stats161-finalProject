"""Additional local opponents and the preserved 70 percent simulator worker."""

import json
import os
from pathlib import Path
import subprocess
import sys

from .sim import Showdown
from .strength import ExternalPivot


FROZEN_PACKAGE = Path(os.environ.get(
    "POKEMON_FROZEN_BASELINE_PACKAGE",
    Path(__file__).resolve().parent.parent / ".venv" / "frozen70" / "pokemon_agent",
)).resolve()
STYLES = {
    "pivot-cycle": {"switch_p": .99, "double_switch_p": .90, "switch_ratio": .83, "draft_tilt": .85, "disrupt_boost": 2.35},
    "pivot-pressure": {"switch_p": .34, "double_switch_p": .18, "unprotect_p": .86, "draft_tilt": -.25, "disrupt_p": .52},
}


class FrozenSearch(Showdown):
    def __enter__(self):
        bridge = FROZEN_PACKAGE / "showdown.cjs"
        if not bridge.is_file():
            raise ValueError(f"Frozen 70 percent bridge missing at {bridge}. "
                             "Set POKEMON_FROZEN_BASELINE_PACKAGE to its pokemon_agent directory.")
        self.process = subprocess.Popen(["node", str(bridge), "--public-search"],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        return self


class AlternatePivot(ExternalPivot):
    def __init__(self, checkout, name, cards):
        super().__init__(checkout, "pivot", cards)
        self.rules = STYLES[name]

    def __enter__(self):
        environment = {"SystemRoot": os.environ["SystemRoot"], "PIVOT_STYLE": json.dumps(self.rules)}
        worker = Path(__file__).with_name("pivot.py").resolve()
        self.process = subprocess.Popen([sys.executable, "-IB", "-X", "utf8", str(worker)], cwd=self.checkout,
                                        env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE, encoding="utf-8")
        return self
