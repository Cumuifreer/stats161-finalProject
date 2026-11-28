"""Test a longer, risk-adjusted continuation on a frozen development candidate."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from multiprocessing import get_context
from pathlib import Path
from statistics import fmean
from time import perf_counter
from unittest.mock import patch

from .. import blind85, lab85, quicksilver80
from ..research80.scouting.replies import ForecastReplies80


class RiskAdjustedHorizon(ForecastReplies80):
    def __init__(self, *args, **settings):
        super().__init__(*args, **settings)
        self.move_budget += 2.1

    def _score(self, world, seat, actions, seeds, deadline, depth):
        if depth != 2 or len(actions) < 2 or perf_counter() + .70 >= deadline:
            return super()._score(world, seat, actions, seeds, deadline, depth)
        world_id, replies, weights = world
        likely = [(action, weight) for action, weight in zip(replies, weights, strict=True) if weight >= .045]
        if len(likely) < 2:
            return super()._score(world, seat, actions, seeds, deadline, depth)
        chosen, mass = zip(*likely, strict=True)
        probabilities = [weight / sum(mass) for weight in mass]
        response = self.worker.call(
            "evaluate", seat=seat, ours=actions, replies=chosen, seeds=seeds,
            budget_ms=1000 * (deadline - perf_counter() - .40), planning=3, world=world_id,
            interaction=True, spread_hp=True,
        )
        if "matrix" not in response:
            self.counts["risk_adjusted_horizon_fallback"] += 1
            return super()._score(world, seat, actions, seeds, deadline, depth)
        averages = [list(map(fmean, row)) for row in response["matrix"]]
        reference = sum(weight * value for weight, value in zip(probabilities, averages[0], strict=True))
        penalty = .24 if reference > 1.1 else .13 if reference > -1.1 else .045
        self.counts["risk_adjusted_horizon_checks"] += 1
        self.counts["branches"] += len(actions) * len(chosen) * len(seeds)
        return [(1 - penalty) * sum(weight * value for weight, value in zip(probabilities, row, strict=True))
                + penalty * min(row) for row in averages]


def evaluate(*args):
    with patch.object(lab85, "ForecastReplies80", RiskAdjustedHorizon):
        return lab85.evaluate(*args)


def measure(roots, cards, rivals, checkout, variant, workers, *args):
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        jobs = [pool.submit(evaluate, [root], cards, rivals, checkout, variant, *args) for root in roots]
        for job in as_completed(jobs):
            yield job.result()


def freeze(*args):
    result = FROZEN(*args)
    script = Path(__file__)
    result["files"][str(script)] = sha256(script.read_bytes()).hexdigest()
    return result


FROZEN = quicksilver80.freeze


if __name__ == "__main__":
    with patch.object(blind85, "measure", measure), patch.object(quicksilver80, "freeze", freeze):
        quicksilver80.main()
