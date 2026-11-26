"""Update the weights of local reply models from publicly executed actions."""

from collections import Counter
from math import exp, log
import re

from .game import ident
from .search import _key


ACTOR = re.compile(r"^(p[12])([ab]):")


def public_atoms(events, foe):
    """Observe each attempted move and voluntary switch, leaving unobserved slots unknown."""
    atoms = {}
    started = False
    for event in events:
        fields = event.split("|")
        tag = fields[1]
        if tag == "turn":
            break
        if tag in {"move", "cant", "faint"}:
            started = True
        if tag != "move" and (tag != "switch" or started):
            continue
        actor = ACTOR.match(fields[2])
        if actor is None or actor[1] != foe:
            continue
        slot = ord(actor[2]) - ord("a")
        if tag == "switch":
            atoms.setdefault(slot, ("switch", ident(fields[3].split(",", 1)[0]), None))
        elif not any("[from]" in tag for tag in fields[5:]):
            target = ACTOR.match(fields[4]) if len(fields) > 4 else None
            where = (1 if target[1] != foe else -1) * (ord(target[2]) - ord("a") + 1) if target else None
            atoms.setdefault(slot, ("move", ident(fields[3]), where))
    return atoms


def likelihood(distribution, observed):
    def agrees(key, token):
        kind, move, target, species = key
        if kind != token[0]:
            return 0.
        if kind == "switch":
            return float(species == token[1])
        if move != token[1]:
            return 0.
        return 1. if not target or token[2] is None or target == token[2] else .45

    observed_mass = 0.
    for key, probability in distribution.items():
        slots = [agrees(key[slot], token) for slot, token in observed.items()]
        observed_mass += probability * (.72 * min(slots) + .28 * sum(slots) / len(slots))
    return .1 + .9 * observed_mass


class ReplyBelief:
    def __init__(self, experts):
        self.prior = experts.copy() if isinstance(experts, dict) else dict.fromkeys(experts, 1.)
        self.scores = dict.fromkeys(experts, 0.)
        self.last = None
        self.pending = {}

    def observe(self, turn, events, foe):
        if self.last is None or turn <= self.last[0]:
            return False
        _, cursor = self.last
        atoms = public_atoms(events[cursor:], foe)
        if atoms:
            for name in self.scores:
                dist = self.pending.get(name)
                if dist:
                    total = sum(dist.values())
                    chances = {key: value / total for key, value in dist.items()}
                    self.scores[name] = .78 * self.scores[name] + log(likelihood(chances, atoms))
        self.last = None
        self.pending.clear()
        return bool(atoms)

    def remember(self, turn, cursor, name, actions, probabilities):
        self.last = (turn, cursor)
        dist = self.pending.setdefault(name, Counter())
        for action, probability in zip(actions, probabilities, strict=True):
            if probability:
                dist[_key(action)] += probability

    def mix(self, predictions):
        available = [(name, self.prior[name] * exp(self.scores[name])) for name in predictions]
        total = sum(weight for _, weight in available)
        prior_total = sum(self.prior[name] for name in predictions)
        size = len(next(iter(predictions.values())))
        result = [0.] * size
        for name, weight in available:
            ratio = .20 * self.prior[name] / prior_total + .80 * weight / total
            for index, value in enumerate(predictions[name]):
                result[index] += ratio * value
        return result
