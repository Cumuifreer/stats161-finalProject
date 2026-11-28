"""Confirm the current move with a deeper local continuation for the surviving proposals."""

from time import perf_counter

from .planning_refine import robust_scores
from .public_planning import PublicPlanner
from .search85 import Search85


class Horizon85(PublicPlanner):
    def _score(self, world, seat, actions, seeds, deadline, depth):
        if depth != 2 or perf_counter() + .35 >= deadline:
            return super()._score(world, seat, actions, seeds, deadline, depth)
        world_id, replies, weights = world
        # Reserve continuation calls for the replies carrying most of the probability.
        selected = sorted(range(len(replies)), key=weights.__getitem__, reverse=True)[:5]
        likely = [replies[index] for index in selected]
        mass = sum(weights[index] for index in selected)
        probabilities = [weights[index] / mass for index in selected]
        remaining = 1000 * (deadline - perf_counter())
        result = self.worker.call("evaluate", seat=seat, ours=actions, replies=likely, seeds=seeds,
                                  budget_ms=remaining, planning=3, world=world_id, interaction=True, spread_hp=True)
        if reason := result.get("reason"):
            self.counts["horizon_" + reason] += 1
            return None
        self.counts["worlds"] += 1
        self.counts["horizon_worlds"] += 1
        self.counts["branches"] += len(actions) * len(likely) * len(seeds)
        return robust_scores(result["matrix"], probabilities, worst=.13)


class PosteriorHorizon85(Search85, Horizon85):
    """Use the public opponent posterior with the longer confirmation policy."""
