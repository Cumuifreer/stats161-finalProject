"""Let DeepSeek suggest a tactical alternative for independent Showdown review."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from multiprocessing import get_context
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

from .. import blind85, lab85, quicksilver80
from ..joint import PROTECT
from ..research80.scouting.replies import ForecastReplies80
from ..tactics import SLOTS, SPEED_CONTROL
from . import deepseek, matches, preview_matches, preview_outcome


ROLE_CHECKS = (lambda atom: atom["type"] == "switch" or atom.get("move_id") in {"partingshot", "uturn", "voltswitch"},
               lambda atom: atom.get("move_id") in PROTECT,
               lambda atom: atom.get("move_id") in SPEED_CONTROL,
               lambda atom: atom.get("move_id") in {"spore", "followme", "ragepowder", "encore", "taunt", "fakeout"})


def menu_for(ranked, coarse):
    baseline = ranked[0]
    menu = [baseline]
    for qualifies in ROLE_CHECKS:
        for slot in SLOTS:
            if not qualifies(baseline[slot]):
                option = next((action for action in coarse if action not in menu and qualifies(action[slot])), None)
                if option is not None:
                    menu.append(option)
    return menu


class ProposalForecast(ForecastReplies80):
    def __init__(self, *args, deepseek_key, **settings):
        super().__init__(*args, **settings)
        self.llm = deepseek.DeepSeek(deepseek_key, limit=1)
        self.proposal = None

    def _proposals(self, decision, seat, rosters, evidence):
        existing = super()._proposals(decision, seat, rosters, evidence)
        return [self.proposal, *(action for action in existing if action != self.proposal)] if self.proposal else existing

    def _plan(self, decision, seat, evidence, ranked, coarse):
        started = perf_counter()
        self.proposal = None
        board = decision.observation
        record = None
        left = min((time for time in (decision.seconds_left, decision.bank_seconds_left) if time is not None), default=50.)
        if not self.llm.records and left > self.move_budget + 6 and board["turn"] <= 4 and all(board["active"]) and all(board["opponents"]):
            choices = menu_for(ranked, coarse)
            if len(choices) >= 4:
                context = {"task": "Option 0 is the locally nominated action. Nominate an alternative when protecting, "
                                   "switching or controlling a matchup can make a better two-turn plan. Choose option 0 "
                                   "when the direct line is already strongest. An independent Showdown search will compare it."}
                alternate = self.llm.advise(decision, choices, ranked[0], context=context, timeout=3.0)
                if self.llm.records:
                    record = self.llm.records[-1]
                    record["suggestion_changed"] = record["changed"]
                    record["changed"] = False
                if alternate != ranked[0] and left - (perf_counter() - started) > self.move_budget + 1.5:
                    self.proposal = alternate
                    self.counts["llm_proposals"] += 1
        selected = super()._plan(decision, seat, evidence, ranked, coarse)
        if record:
            record["changed"] = self.proposal is not None and selected == self.proposal
            self.counts["llm_proposals_adopted"] += record["changed"]
        return selected


def evaluate(roots, cards, rivals, *args, checkpoint, key):
    class BoundForecast(ProposalForecast):
        def __init__(self, *args, **settings):
            super().__init__(*args, deepseek_key=key, **settings)

    def play(*args, **settings):
        result = matches.PLAY(*args, **settings)
        candidate = settings.get("candidate") or args[-1]
        result["llm_telemetry"] = candidate.battle.llm.records
        return result

    with patch.object(lab85, "ForecastReplies80", BoundForecast), patch.object(lab85, "play", play):
        return preview_matches.evaluate(roots, cards, rivals, *args, checkpoint=checkpoint, battle_checkpoint=None)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--lineup-net", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if "--seed" not in remaining or "--resume" in remaining:
        parser.error("Use a fresh development sample for remote proposal tests")
    key = deepseek.credential()
    sys.argv[1:] = remaining
    parent = matches.freeze_for(None, True)

    def freeze(*settings):
        result = parent(*settings)
        if not result["candidate"].get("research80", {}).get("reply_rollouts"):
            raise ValueError("Provide the same search anchor used by the neural Preview comparison")
        files = (Path(__file__), Path(preview_matches.__file__), Path(preview_outcome.__file__),
                 args.lineup_net / "config.json", args.lineup_net / "state_dict.pt")
        for path in files:
            result["files"][str(path)] = sha256(path.read_bytes()).hexdigest()
        result["research_learning"].update(proposal_only=True, preview_checkpoint=str(args.lineup_net))
        return result

    def measure(roots, cards, rivals, checkout, variant, workers, *settings):
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
            jobs = [executor.submit(evaluate, [root], cards, rivals, checkout, variant, *settings,
                                    checkpoint=args.lineup_net, key=key) for root in roots]
            for job in as_completed(jobs):
                yield job.result()

    with patch.object(blind85, "measure", measure), patch.object(blind85, "publish_json", matches.publish), \
            patch.object(quicksilver80, "freeze", freeze):
        quicksilver80.main()


if __name__ == "__main__":
    main()
