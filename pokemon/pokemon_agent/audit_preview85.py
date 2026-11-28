"""Audit alternate public lineup forecasts after development matches are complete."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
import json
from multiprocessing import get_context
from pathlib import Path
from random import Random
from time import perf_counter

from .combination_preview import signature
from .draft85 import deployment
from .draft_leaf import CATALOG
from .draft_league import read, save
from .lineup import index_cards
from .matchup_preview import MatchupPreview


def audit(cases, checkout, cards):
    lookup = {card["card_id"]: card for card in cards}
    indexed = index_cards(cards)
    counts = {name: {str(budget): Counter() for budget in (1.2, 19.9)} for name in ("ace-code", "pivot")}
    with ExitStack() as stack:
        models = {str(seconds): stack.enter_context(MatchupPreview(checkout, indexed, budget=seconds)) for seconds in (1.2, 19.9)}
        for name, game in cases:
            other = "p2" if game["actor"] == "p1" else "p1"
            rosters = [[lookup[card] for card in game["roster"][seat]] for seat in (other, game["actor"])]
            view = deployment(*rosters)
            actual = signature(game["lineups"][other])
            for budget, model in models.items():
                options = [signature(action) for action in model.proposals(view, top=4)]
                counts[name][budget].update(games=1, same_first=bool(options) and options[0] == actual,
                                            same_among_four=actual in options,
                                            same_leads_among_four=any(lead == actual[1] for _, lead in options))
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--ace-trainer", type=Path, required=True)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = perf_counter()
    reference = read(args.development)
    if "variant" not in reference:
        parser.error("Use a completed research85 development report")
    if args.count < 2 or args.count % 2 or args.workers < 1:
        parser.error("Provide a positive even sample count and at least one worker")
    cards = read(CATALOG)
    half = args.count // 2
    rng = Random(446183)
    groups = {name: [(name, game) for root in reference["records"].values() for game in root[name]] for name in ("ace-code", "pivot")}
    chosen = [case for name in groups for case in rng.sample(groups[name], half)]
    rng.shuffle(chosen)
    distribution = {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as workers:
        pending = [workers.submit(audit, chosen[i::args.workers], args.ace_trainer, cards) for i in range(args.workers)]
        for work in pending:
            for name, budgets in work.result().items():
                combined = distribution.setdefault(name, {})
                for budget, metrics in budgets.items():
                    combined.setdefault(budget, Counter()).update(metrics)
    report = dict(source=str(args.development), cases=len(chosen), results=distribution,
                  seconds=perf_counter() - started, api_spend_usd=0)
    save(args.output, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
