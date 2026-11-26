"""Shared weighting and confirmation for public move search."""

from statistics import fmean

from .search import _key


def mix_replies(decision, actions, weights, learned, fraction, limit=8):
    probabilities = {_key(action): probability for action, probability in zip(decision.actions, learned, strict=True)}
    combined = {_key(action): (1 - fraction) * weight for action, weight in zip(actions, weights, strict=True)}
    for key, probability in probabilities.items():
        combined[key] = combined.get(key, 0.) + fraction * probability
    by_key = {_key(action): action for action in decision.actions}
    selected = sorted(combined, key=combined.__getitem__, reverse=True)[:limit]
    total = sum(combined[key] for key in selected)
    return [by_key[key] for key in selected], [combined[key] / total for key in selected]


def robust_scores(matrix, weights, worst=.10):
    scores = []
    for row in matrix:
        by_reply = list(map(fmean, row))
        predicted = sum(w * v for w, v in zip(weights, by_reply, strict=True))
        scores.append((1 - worst) * predicted + worst * min(by_reply))
    return scores


def average_advantages(world_scores):
    offsets = [[score - world[0] for score in world] for world in world_scores]
    return [fmean(values) - .08 * (max(values) - min(values)) for values in zip(*offsets)]


def survivors(scores, limit):
    sorted_ids = sorted(range(1, len(scores)), key=scores.__getitem__, reverse=True)
    return [0, *sorted_ids[:limit - 1]]
