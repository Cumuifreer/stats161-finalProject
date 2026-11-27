"""Measure the early continuation scout on a registered development pool draw."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from multiprocessing import get_context
from pathlib import Path
from random import Random
from time import time
from unittest.mock import patch

from ... import lab85
from ...draft_leaf import CATALOG
from ...draft_league import pools_for, read
from ...lineup import index_cards
from ...paired_store85 import PairedStore, publish_json
from ...quicksilver80 import FROZEN_85, freeze
from ...sim import fingerprint
from ...strength import summarize
from .battle import RolloutScout, WideScout80


def play_pair(root, cards, ace, quicksilver, rivals):
    with RolloutScout(quicksilver, ace, index_cards(cards)) as scout:
        def planner(*args, move_budget, **settings):
            return WideScout80(*args, rollout_scout=scout, move_budget=move_budget + 1.95, **settings)

        with patch.object(lab85, "WideSearch85", planner):
            return lab85.evaluate(
                [root], cards, rivals, ace, "wide", True, None, None, False, True,
                Path(FROZEN_85["lead_model"]), Path(FROZEN_85["action_model"]), True, quicksilver,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ace-trainer", type=Path, required=True)
    parser.add_argument("--quicksilver", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--opponents", nargs="+", default=[lab85.QUICKSILVER])
    args = parser.parse_args()
    if args.output.exists() != args.resume or args.workers < 1:
        parser.error("Provide a fresh output or request a resume with positive workers")
    output = args.output / "matches.json"
    if output.exists():
        parser.error("The registered trial is already complete")
    prior = read(args.development / "frozen.json")
    if prior["blind"]:
        parser.error("The continuation trial cannot access blind pools")
    sample = read(args.development / "sample.json")
    cards = read(CATALOG)
    pools = pools_for(cards, range(sample["seed"], sample["seed"] + len(sample["fingerprints"])))
    if pools != sample["fingerprints"]:
        raise ValueError("The registered development pool draw changed")

    def lock():
        locked = freeze(args.ace_trainer, args.quicksilver, FROZEN_85, len(pools), False,
                        args.opponents, args.workers, cards)
        files = (Path(__file__), Path(__file__).with_name("battle.py"))
        return locked | {"trial_sources": {str(path): sha256(path.read_bytes()).hexdigest() for path in files}}

    locked = lock()
    journal = PairedStore(args.output, fingerprint([locked, pools]), pools, args.opponents)
    completed, cpu = journal.recover()
    roots = [int(root) for root in pools if int(root) not in completed]
    Random(sample["seed"] ^ 0x762E).shuffle(roots)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as workers:
        jobs = [workers.submit(play_pair, root, cards, args.ace_trainer, args.quicksilver, args.opponents)
                for root in roots]
        for job in as_completed(jobs):
            pairs, spent = job.result()
            journal.append(pairs, spent)
            completed.update(pairs)
            cpu += spent
            print(f"Completed {len(completed)}/{len(pools)} paired pools", flush=True)
    if lock() != locked:
        raise ValueError("The trial inputs changed")
    report = {}
    for name in args.opponents:
        pairs = [match[name] for _, match in sorted(completed.items())]
        result = summarize(pairs)
        result["planning"] = dict(sum((Counter(game["planning"]) for pair in pairs for game in pair), Counter()))
        result["quicksilver"] = dict(sum((Counter(game.get("quicksilver", {})) for pair in pairs for game in pair), Counter()))
        report[name] = result
    publish_json(output, {"pools": pools, "records": completed, "report": report, "sources": locked,
                          "seconds": time() - journal.started, "worker_python_cpu_seconds": cpu, "api_spend_usd": 0})
    for name, result in report.items():
        print(name, result["wins"], result["games"], result["paired_bootstrap_95"], flush=True)


if __name__ == "__main__":
    main()
