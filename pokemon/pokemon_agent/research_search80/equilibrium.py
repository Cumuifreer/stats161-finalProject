"""Review simultaneous moves against a regularized best-response mixture."""

from hashlib import sha256
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np

from .. import blind85, lab85, quicksilver80
from ..research80.scouting.replies import ForecastReplies80
from . import distilled


def softmax(values):
    raw = np.exp(values - np.max(values))
    return raw / raw.sum()


def equilibrium_scores(payoffs, weights):
    """Keep likely replies in the model while allowing a shared adversarial response."""
    prior = np.asarray(weights, dtype=np.float64).clip(1e-5)
    prior /= prior.sum()
    replies = prior.copy()
    actions = softmax(payoffs @ replies / .40)
    reference = np.log(prior)
    for _ in range(16):
        replies = .66 * replies + .34 * softmax(reference - (actions @ payoffs) / .52)
        actions = .66 * actions + .34 * softmax(payoffs @ replies / .40)
    adverse = .40 * prior + .60 * replies
    return (.91 * (payoffs @ adverse) + .09 * payoffs.min(axis=1)).tolist()


class EquilibriumForecast(ForecastReplies80):
    def _score(self, world, seat, actions, seeds, deadline, depth):
        if depth != 2 or len(actions) < 3 or self.check_endgame or deadline - perf_counter() < .60:
            return super()._score(world, seat, actions, seeds, deadline, depth)
        world_id, replies, weights = world
        result = self.worker.call(
            "evaluate", seat=seat, ours=actions, replies=replies, seeds=seeds,
            budget_ms=1000 * (deadline - perf_counter() - .30), planning=depth, world=world_id,
            interaction=self.interaction, spread_hp=True,
        )
        if "matrix" not in result:
            self.counts["equilibrium_fallbacks"] += 1
            return super()._score(world, seat, actions, seeds, deadline, depth)
        self.counts["equilibrium_reviews"] += 1
        self.counts["branches"] += len(actions) * len(replies) * len(seeds)
        values = np.asarray(result["matrix"], dtype=np.float64).mean(axis=2)
        return equilibrium_scores(values, weights)


def evaluate(*args):
    with patch.object(lab85, "ForecastReplies80", EquilibriumForecast):
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
