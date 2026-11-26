"""Research Preview proposals from a public matchup model and short opening search."""

import argparse
from collections import Counter
from contextlib import ExitStack
import os
from pathlib import Path
from random import Random
import subprocess
import sys
from time import perf_counter

from .combination_preview import preview_snapshot
from .draft_leaf import CATALOG
from .draft_league import parallel, read, save
from .game import ident
from .lineup import index_cards
from .pivot import PivotOpponent
from .planning_preview import resolve_species
from .sim import Showdown


class MatchupPreview(PivotOpponent):
    def __init__(self, checkout, indexed, *, budget=2.2):
        super().__init__(checkout)
        self.catalog = indexed
        self.budget = budget
        self.activity = Counter()

    def __enter__(self):
        entry = Path(__file__).with_name("matchup_preview_worker.py").resolve()
        environment = {"SystemRoot": os.environ["SystemRoot"]} if os.name == "nt" else {}
        self.process = subprocess.Popen(
            [sys.executable, "-IB", "-X", "utf8", str(entry)],
            cwd=self.checkout, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE, encoding="utf-8",
        )
        return self

    def proposals(self, decision, top=4):
        own, foe = [[resolve_species(mon["species"], self.catalog) for mon in decision.observation[key]]
                    for key in ("team", "opponent_team")]
        if any(len(team) != 6 or not set(team) <= self.catalog.keys() for team in (own, foe)):
            self.activity["unknown_public_roster"] += 1
            return []
        answer = self._call({"own": {mon: self.catalog[mon] for mon in own}, "foe": {mon: self.catalog[mon] for mon in foe},
                             "budget": self.budget, "top": top})
        legalized = []
        for row in answer["candidates"]:
            bring, leads = set(map(ident, row["bring"])), list(map(ident, row["leads"]))
            action = next((action for action in decision.actions
                           if set(action["bring"]) == bring and action["leads"] == leads), None)
            if action is not None and action not in legalized:
                legalized.append(action)
        self.activity["legal_proposals"] += len(legalized)
        return legalized


def project(roots, rows, checkout, cards, budget):
    output = {}
    by_id = {card["card_id"]: card for card in cards}
    with ExitStack() as stack:
        worker = stack.enter_context(MatchupPreview(checkout, index_cards(cards), budget=budget))
        engine = stack.enter_context(Showdown())
        for root in roots:
            output[root] = {}
            for rival, pair in rows[str(root)].items():
                updated = []
                for game in pair:
                    actor = game["actor"]
                    rosters = {seat: [by_id[card] for card in ids] for seat, ids in game["roster"].items()}
                    views = preview_snapshot(rosters, game["battle_seed"], engine)
                    lineup = worker.proposals(views[actor])
                    recorded = game["lineups"][actor]
                    updated.append(game | {"lineups": {**game["lineups"], actor: lineup[0] if lineup else recorded},
                                           "recorded_own_lineup": recorded})
                output[root][rival] = updated
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--ace-trainer", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=291)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--preview-seconds", type=float, default=2.2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = read(args.source)
    if args.output.exists() or not 1 <= args.pairs <= len(source["records"]) or args.workers < 1:
        parser.error("Use a fresh output and valid positive sizes")
    roots = Random(args.seed).sample(sorted(map(int, source["records"])), args.pairs)
    rows = {}
    began = perf_counter()
    selected = {str(root): source["records"][str(root)] for root in roots}
    for batch in parallel(project, roots, args.workers, selected, args.ace_trainer, read(CATALOG), args.preview_seconds):
        rows.update(batch)
    changed = Counter()
    for block in rows.values():
        for rival, games in block.items():
            changed[rival] += sum(game["recorded_own_lineup"] != game["lineups"][game["actor"]] for game in games)
    save(args.output, dict(records=rows, source=str(args.source), predicted_deployment_changes=changed,
                           seconds=perf_counter() - began, api_spend_usd=0))
    print(dict(changed), flush=True)


if __name__ == "__main__":
    main()
