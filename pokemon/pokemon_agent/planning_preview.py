"""Rank legal Preview deployments against several public-roster hypotheses."""

from collections import Counter
from dataclasses import replace
from itertools import combinations
from random import Random

from .game import ident
from .sim import FORMAT, fingerprint
from .tactics import tactical_actions


def resolve_species(name, catalog):
    if name in catalog:
        return name
    matches = [species for species in catalog if species.startswith(name)]
    return matches[0] if len(matches) == 1 else name


def publish_sets(observation, cards):
    def enrich(mon):
        species = resolve_species(mon["species"], cards)
        card = cards[species]
        return mon | {"species": species, "moves": [ident(move) for move in card["moves"]], "item": ident(card["item"]),
                      "ability": ident(card["ability"])}

    return {side: [enrich(mon) for mon in team] for side, team in observation.items()}


def opposing_deployments(decision, cards):
    board = publish_sets(decision.observation, cards)
    roster = [mon["species"] for mon in board["opponent_team"]]
    actions = tuple(dict(type="select_lineup", bring=list(bring), leads=list(leads))
                    for bring in combinations(roster, 4) for leads in combinations(bring, 2))
    return tactical_actions(replace(decision, observation={"team": board["opponent_team"], "opponent_team": board["team"]},
                                    actions=actions))


def diverse_deployments(ranked, anchor, limit):
    result = [anchor]
    teams, leads = {frozenset(anchor["bring"])}, {frozenset(anchor["leads"])}
    signatures = {(frozenset(anchor["bring"]), frozenset(anchor["leads"]))}
    for allow_duplicate in (False, True):
        for action in ranked:
            team, lead = frozenset(action["bring"]), frozenset(action["leads"])
            if len(result) >= limit:
                return result
            if (team, lead) in signatures or (not allow_duplicate and team in teams and lead in leads):
                continue
            result.append(action)
            teams.add(team)
            leads.add(lead)
            signatures.add((team, lead))
    return result


class PlanningPreview:
    def __init__(self, anchor, worker, cards, *, width=28, worlds=4, draws=7):
        self.anchor = anchor
        self.worker = worker
        self.cards = cards
        self.width = width
        self.worlds = worlds
        self.draws = draws
        self.counts = Counter()

    @property
    def fallbacks(self):
        return self.anchor.fallbacks + self.counts["fallbacks"]

    def __call__(self, decision):
        anchor = self.anchor(decision)
        try:
            observation = publish_sets(decision.observation, self.cards)
            ranked = tactical_actions(replace(decision, observation=observation))
            foes = opposing_deployments(decision, self.cards)
        except (KeyError, TypeError, ValueError):
            self.counts["fallbacks"] += 1
            return anchor
        options = diverse_deployments(ranked, anchor, self.width)
        opponents = diverse_deployments(foes, foes[0], self.worlds)
        signatures = [[sorted(mon["species"] for mon in team) for team in observation.values()], "preview-planning-v1"]
        rng = Random(fingerprint(signatures))
        seeds = [[rng.randrange(65536) for _ in range(4)] for _ in range(self.draws)]
        try:
            outcome = self.worker.call("plan_preview", format=FORMAT, options=options, opponents=opponents,
                                       seeds=seeds, budget_ms=min(8000., ((decision.seconds_left or 85) - 12.) * 1000))
        except (RuntimeError, OSError):
            self.counts["fallbacks"] += 1
            return anchor
        self.counts["rollouts"] += outcome["calls"]
        self.counts["budget_reached"] += outcome["timedOut"]
        self.counts["changed"] += outcome["index"] != 0
        return options[outcome["index"]]
