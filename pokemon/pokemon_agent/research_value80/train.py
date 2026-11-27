"""Train a long-horizon value model on multiple continuation policies."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from multiprocessing import get_context
from pathlib import Path
import subprocess
from time import perf_counter
from unittest.mock import patch

from .. import value_fit85
from ..blind85 import used_pools
from ..draft_leaf import CATALOG
from ..draft_league import pools_for, read, save
from ..lab85 import MODELS
from ..sim import Showdown


LABELS = Path(__file__).with_name("labels.cjs")


class PolicyLabels(Showdown):
    def __init__(self, engine):
        self.engine = engine
        super().__init__()

    def __enter__(self):
        self.process = subprocess.Popen(
            ["node", str(LABELS), str(self.engine)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )
        return self


def collect(roots, cards, pressure, draws, engine):
    with patch.object(value_fit85, "LabelWorker", lambda: PolicyLabels(engine)):
        return value_fit85.collect(roots, cards, pressure, draws)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quicksilver", type=Path, required=True)
    parser.add_argument("--prior-workspace", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pools", type=int, required=True)
    parser.add_argument("--draws", type=int, default=42)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.output.exists() or args.pools < 5 or min(args.draws, args.workers) <= 0:
        parser.error("Provide a fresh output, at least five pools, and positive counts")
    started = perf_counter()
    cards = read(CATALOG)
    roots = list(range(args.seed, args.seed + args.pools))
    pools = pools_for(cards, roots)
    previous = used_pools(cards)
    for workspace in args.prior_workspace:
        previous.update(used_pools(cards, workspace))
    if not previous.isdisjoint(pools.values()):
        raise ValueError("Training requires fresh pools")
    pressure = read(MODELS / "value-v2.json")["pressure"]
    engine = (args.quicksilver / "pokemon/sim/engine.js").resolve()
    source = {str(path): sha256(path.read_bytes()).hexdigest() for path in (LABELS, Path(__file__), engine)}
    dataset, counts = {}, Counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as workers:
        jobs = [workers.submit(collect, roots[i:i + 8], cards, pressure, args.draws, engine)
                for i in range(0, len(roots), 8)]
        for job in as_completed(jobs):
            rows, stats = job.result()
            dataset.update(rows)
            counts.update(stats)
            print(f"{len(dataset)}/{args.pools} pools, {counts['completed']}/{counts['simulated']} terminal", flush=True)
    if counts["completed"] < .90 * counts["simulated"]:
        raise RuntimeError("Too many candidate games failed to reach a terminal result")
    model = value_fit85.optimize(dataset, pressure, pools, cards, args.seed)
    model.update(counts=dict(counts), time_seconds=perf_counter() - started, api_spend_usd=0, training_source=source)
    save(args.output, model)
    print({key: model[key] for key in ("held_nll", "held_brier", "fit_seconds", "time_seconds", "counts", "device")},
          flush=True)


if __name__ == "__main__":
    main()
