"""Nominate favorable lead pairs within otherwise strong public quartets."""

from collections import defaultdict
from math import log1p

import numpy as np

from .combination_preview import distinct_options, signature


def offense(space, front, enemy, speed_order):
    hits = space.attack[np.ix_(front, enemy)].clip(0., 1.25)
    fast = speed_order * (space.speeds[front, None] - space.speeds[enemy][None, :]) > 0
    # Keep credit for two hits into one target and for threatening both slots.
    uncontested = hits * (.72 + .28 * fast)
    pressure = .68 * uncontested.max(0).mean() + .32 * np.minimum(1.15, uncontested.sum(0)).max()
    swing = max(0., float(hits.max(0).max()) - .65)
    return pressure + .22 * swing


def head_to_head(space, leads, replies):
    room = bool(set(replies) & space.roles["trick_room"])
    ordinary = offense(space, leads, replies, 1) - offense(space, replies, leads, 1)
    if not room:
        return ordinary
    reversed_order = offense(space, leads, replies, -1) - offense(space, replies, leads, -1)
    return .73 * ordinary + .27 * reversed_order


def propose(space, decision, ranked, anchors, opponents, weights, limit=14):
    indices = space.species
    enemy = [list(map(indices.__getitem__, action["leads"])) for action in opponents]
    odds = np.asarray(weights[:len(enemy)])
    odds /= odds.sum()
    cache = {}
    for action in distinct_options(decision).values():
        _, group = signature(action)
        if group not in cache:
            lead = list(map(indices.__getitem__, action["leads"]))
            values = np.asarray([head_to_head(space, lead, other) for other in enemy])
            cache[group] = float(.87 * (values @ odds) + .13 * values.min())
    selected = {signature(action)[0] for action in anchors[:8]}
    by_team = defaultdict(list)
    seen = set()
    for index, action in enumerate(ranked):
        team, leads = signature(action)
        if team not in selected or (team, leads) in seen:
            continue
        seen.add((team, leads))
        by_team[team].append((cache[leads] - .04 * log1p(index), action))
    nominated = []
    for section in by_team.values():
        nominated.extend(sorted(section, key=lambda entry: entry[0], reverse=True)[:2])
    nominated.sort(key=lambda entry: entry[0], reverse=True)
    return [choice for _, choice in nominated[:limit]]
