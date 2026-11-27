"""Compare public deployment forecasts on completed development games."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean
from time import perf_counter

from ..combination_preview import signature
from ..draft85 import deployment
from ..draft_leaf import CATALOG
from ..draft_league import read, save
from ..lead85 import LeadPrior
from ..lineup import index_cards
from ..tactics import tactical_actions
from .forecast import MonteCarloLineups


def evaluate(rows, checkout, cards, budget):
    by_id = {card["card_id"]: card for card in cards}
    lead = LeadPrior(Path("checkpoints/research85/lead-prior.json"), cards)
    counts = Counter()
    timing = []
    with MonteCarloLineups(checkout, index_cards(cards), budget) as forecast:
        for game in rows:
            seat = "p2" if game["actor"] == "p1" else "p1"
            ours, theirs = [[by_id[key] for key in game["roster"][player]] for player in (seat, game["actor"])]
            view = deployment(ours, theirs)
            actual = signature(game["lineups"][seat])
            ranked = tactical_actions(view)
            usual, _ = lead.scenarios(view, [], ranked)
            started = perf_counter()
            proposed = [action for action, _ in forecast.proposals(view, top=4)]
            timing.append(perf_counter() - started)
            counts["games"] += 1
            for name, options in (("tactical", ranked), ("learned", usual), ("monte_carlo", proposed)):
                options = list(dict.fromkeys(map(signature, options)))
                for limit in (1, 2, 4):
                    counts[f"{name}_lineups_{limit}"] += actual in options[:limit]
                    counts[f"{name}_leads_{limit}"] += any(front == actual[1] for _, front in options[:limit])
    return counts, timing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quicksilver", type=Path, required=True)
    parser.add_argument("--read", type=Path, required=True)
    parser.add_argument("--opponent", default="quicksilver-public")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--budget", type=float, default=12.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = read(args.read)
    cards = read(CATALOG)
    rows = [game for group in result["records"].values() for game in group[args.opponent]]
    started = perf_counter()
    counts = Counter()
    latency = []
    with ThreadPoolExecutor(max_workers=args.workers) as workers:
        jobs = [workers.submit(evaluate, rows[i::args.workers], args.quicksilver, cards, args.budget)
                for i in range(min(len(rows), args.workers))]
        for job in jobs:
            part, timing = job.result()
            counts.update(part)
            latency.extend(timing)
    report = dict(counts, forecast_seconds=fmean(latency), seconds=perf_counter() - started,
                  pools=result["pools"], api_spend_usd=0)
    save(args.output, report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
