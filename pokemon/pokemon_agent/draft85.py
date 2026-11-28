"""Review late draft picks using legal bring-four games for projected public rosters."""

from collections import Counter
from random import Random
from statistics import fmean
from time import perf_counter

from .game import PREVIEW, ident, make_decision
from .planning_preview import diverse_deployments
from .sim import FORMAT, fingerprint
from .tactics import tactical_actions


def deployment(own, foe):
    return make_decision(PREVIEW, {"your_roster": own, "opponent_roster": foe},
                         [{"roster": [ident(card["species"]) for card in own]}], "p1")


class Draft85:
    def __init__(self, experiment, worker):
        self.experiment = experiment
        self.worker = worker
        self.counts = Counter()

    def __call__(self, decision):
        started = perf_counter()
        anchor = self.experiment.choose(decision, "champion")
        pick = decision.observation["pick"]
        if pick < 10 or len(decision.actions) < 2:
            return anchor
        deadline = started + min(10., (decision.seconds_left or 15.) - 1.5)
        plan = self.experiment.rank(decision, anchor, None)
        if not plan:
            return anchor
        actions, worlds, order, prior = plan
        indices = [0, *(i for i in order[:2] if prior[i] >= prior[0] - .12)]
        if len(indices) < 2:
            return anchor
        rng = Random(fingerprint(["late-draft-v3", decision.observation]))
        seeds = [[rng.randrange(65536) for _ in range(4)] for _ in range(3)]
        horizon = range(min(2, len(worlds[0]))) if pick < 12 else range(1)
        measured = {i: [] for i in indices}
        tested = {}
        for scenario in horizon:
            if not all(worlds[i][scenario] is not None for i in indices):
                continue
            for index in indices:
                if perf_counter() + .25 >= deadline:
                    self.counts["budget_fallback"] += 1
                    return anchor
                pair = worlds[index][scenario]
                key = tuple(tuple(side) for side in pair)
                if key not in tested:
                    try:
                        own, foe = [[self.experiment.original_cards[card] for card in side] for side in pair]
                        own_rank, foe_rank = tactical_actions(deployment(own, foe)), tactical_actions(deployment(foe, own))
                        options = diverse_deployments(own_rank, own_rank[0], 5)
                        replies = diverse_deployments(foe_rank, foe_rank[0], 3)
                        result = self.worker.call("compare", options=options, opponents=replies, weights=[.50, .31, .19],
                                                  format=FORMAT, seeds=seeds, edge=0., budget_ms=1000 * (deadline - perf_counter()),
                                                  stages=[dict(survivors=1, depth=13, start=0, samples=3, opponentCount=3)])
                    except (KeyError, ValueError, RuntimeError, OSError):
                        self.counts["fallback"] += 1
                        return anchor
                    self.counts["rollouts"] += result["calls"]
                    if result["timedOut"] or not result["rows"]:
                        self.counts["budget_fallback"] += 1
                        return anchor
                    tested[key] = result["rows"][0]["score"]
                measured[index].append(tested[key])
        if not measured[0]:
            return anchor
        self.counts["reviewed"] += 1
        gains = {index: [after - before for after, before in zip(measured[index], measured[0], strict=True)] for index in indices}
        values = {index: .86 * fmean(gain) + .14 * min(gain) + .12 * (prior[index] - prior[0]) for index, gain in gains.items()}
        best = max(indices, key=values.__getitem__)
        if best and values[best] >= .12 and min(gains[best]) >= -.02:
            self.counts["changed"] += 1
            return actions[best]
        return anchor
