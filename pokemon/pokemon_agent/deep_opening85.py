"""Combine a public local opponent policy forecast with alternate lead hypotheses."""

from collections import Counter
from math import exp, log

from .combination_preview import signature


def condition_reserves(logits, leads, forecasts):
    forecast_mass = Counter()
    for lineup, mass in forecasts:
        quartet = tuple(sorted(lineup["bring"]))
        if quartet in logits and set(lineup["leads"]) == leads:
            forecast_mass[quartet] += mass
    support = sum(forecast_mass.values())
    if not support or not logits:
        return logits
    peak = max(logits.values())
    prior = {team: exp(value - peak) for team, value in logits.items()}
    scale = sum(prior.values())
    observed = min(.62, .94 * support)
    return {team: log((1 - observed) * value / scale + observed * forecast_mass[team] / support)
            for team, value in prior.items()}


class DeepLeadPrior:
    def __init__(self, prior, specialist):
        self.prior = prior
        self.specialist = specialist
        self.space = prior.space
        self.extra_seconds = specialist.budget + 1.5

    def scenarios(self, reverse, nominated, ranked):
        generalized, weights = self.prior.scenarios(reverse, nominated, ranked)
        try:
            specific = self.specialist.proposals(reverse, top=4)
        except (RuntimeError, ValueError, OSError):
            return generalized, weights
        if not specific:
            return generalized, weights
        lead = signature(specific[0])
        remaining = [(weight, action) for weight, action in zip(weights, generalized, strict=True)
                     if signature(action) != lead][:7]
        divisor = sum(weight for weight, _ in remaining)
        if divisor <= 0:
            return generalized, weights
        scenarios = [specific[0], *[action for _, action in remaining]]
        masses = [.49, *[.51 * weight / divisor for weight, _ in remaining]]
        return scenarios, masses
