"""Enumerate legal fixed-card combinations and audit recorded roster coverage."""

from collections import Counter
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

from .draft_league import read
from .game import ident
from .sim import FIRST_DRAFTER_PICKS
from .tactics import SPREAD, _damage, _speed, load_card_profiles, load_dex

ROLES = {"tailwind": {"tailwind"}, "trick_room": {"trickroom"},
         "redirection": {"ragepowder", "followme"}, "fake_out": {"fakeout"},
         "rain": {"drizzle", "swiftswim", "raindance"}, "sun": {"drought", "protosynthesis", "sunnyday"}}


def legal_groups(items, size):
    return [group for group in combinations(range(len(items)), size)
            if len({items[i] for i in group}) == size]


def disjoint_matchups(items, size):
    counts = {(0, 0): 1}
    for count in Counter(items).values():
        next_counts = Counter()
        for (left, right), total in counts.items():
            for a, b, ways in ((0, 0, 1), (1, 0, count), (0, 1, count), (1, 1, count * (count - 1))):
                if left + a <= size and right + b <= size:
                    next_counts[left + a, right + b] += total * ways
        counts = next_counts
    return counts[size, size] // 2


class CardSpace:
    def __init__(self, cards, profiles=None):
        self.items = [ident(card["item"]) for card in cards]
        self.ids = [card["card_id"] for card in cards]
        self.species = {ident(card["species"]): i for i, card in enumerate(cards)}
        if len(self.species) != len(cards) or len(set(self.ids)) != len(cards):
            raise ValueError("Combination analysis requires a unique public species for each card")
        # Showdown may reveal just the base species at Preview. The fixed pool has one Urshifu form.
        if "urshifurapidstrike" in self.species and "urshifu" not in self.species and "urshifusinglestrike" not in self.species:
            self.species["urshifu"] = self.species["urshifurapidstrike"]
        profiles = profiles if profiles is not None else load_card_profiles(cards, with_sets=True)
        dex = load_dex()
        mons = [profiles[ident(card["species"])] | {"species": ident(card["species"]),
                "ability": ident(card["ability"]), "item": ident(card["item"]), "level": card["level"]} for card in cards]
        moves = [set(ident(move) for move in card["moves"]) for card in cards]
        self.roles = {name: {i for i, mon in enumerate(mons) if (moves[i] | {mon["ability"]}) & actions}
                      for name, actions in ROLES.items()}
        self.speeds = np.array([_speed(mon, [], dex) for mon in mons], dtype=np.float32)
        self.attack = np.array([[max((_damage(move, data, a, b, data.get("target") in SPREAD, dex)
                                     * (data["accuracy"] if data.get("accuracy") is not True else 1.)
                                     for move in attacks if (data := dex["moves"].get(move, {})).get("base_power")),
                                    default=0.) for b in mons] for a, attacks in zip(mons, moves, strict=True)], dtype=np.float32)

    def archetypes(self, group):
        return [name for name, cards in self.roles.items() if cards.intersection(group)] or ["no_listed_control"]

    def enumerate(self, size):
        groups = np.asarray(legal_groups(self.items, size))
        damage = self.attack[groups].clip(0, 1.5)
        champion = damage.max(axis=1)
        strength = damage.mean(axis=2).max(axis=1)
        complement = champion.mean(axis=1) - strength
        pressure = self.attack[:, groups].clip(0, 1.5).min(axis=2).T.mean(axis=1)
        return [[group.tolist(), float(coverage), float(synergy), float(safety)]
                for group, coverage, synergy, safety in zip(groups, champion.mean(axis=1), complement, pressure, strict=True)]


class CoverageCounts:
    def __init__(self, space):
        self.space = space
        self.groups = {size: Counter() for size in range(1, 5)}
        self.styles = Counter()
        self.unique = set()
        self.exposures = 0
        self.leads = Counter()

    def add(self, group, leads=()):
        group = tuple(sorted(group))
        self.unique.add(group)
        self.exposures += 1
        for name in self.space.archetypes(group):
            self.styles[name] += 1
        for size, counts in self.groups.items():
            counts.update(combinations(group, size))
        if leads:
            self.leads[tuple(sorted(leads))] += 1

    def report(self, maximum):
        percentiles = (.1, .5, .9)
        return {"exposures": self.exposures, "unique_teams": len(self.unique),
                "archetype_exposures": dict(self.styles), "lead_pairs": len(self.leads), "groups": {
                    size: {"covered": len(counts), "possible": maximum[size],
                           "exposure_percentiles": np.quantile(list(counts.values()), percentiles).tolist() if counts else [],
                           "at_least_10": sum(n >= 10 for n in counts.values())}
                    for size, counts in self.groups.items()}}


def audit(cards, directory):
    space = CardSpace(cards)
    catalog = {card_id: i for i, card_id in enumerate(space.ids)}
    draft, active = (CoverageCounts(space) for _ in range(2))
    lead_duels, four_duels = set(), set()
    sources = {}
    for name in ("roster-v1/train", "roster-v1/expanded", "draft-v2/train", "draft-v2/league-train", "draft-v3/train"):
        data = read(Path(directory) / (name + ".json"))
        games = 0
        if "samples" in data:
            for row in data["samples"]:
                games += 1
                match_teams, match_leads = [], []
                for roster, roles in zip(row["cards"], row["roles"], strict=True):
                    roster = [catalog[card] for card in roster]
                    draft.add(roster)
                    lineup = tuple(sorted(i for i, role in zip(roster, roles, strict=True) if role))
                    opening = tuple(sorted(i for i, role in zip(roster, roles, strict=True) if role == 2))
                    active.add(lineup, opening)
                    match_teams.append(lineup)
                    match_leads.append(opening)
                lead_duels.add(tuple(sorted(match_leads)))
                four_duels.add(tuple(sorted(match_teams)))
        elif name.startswith("draft-v2/"):
            for row in data["rows"]:
                for teams, _ in row["results"]:
                    games += 1
                    for team in teams:
                        draft.add(team)
        else:
            for row in data["rows"]:
                for game in row["games"]:
                    games += 1
                    own = ([], [])
                    for i, (_, _, slot, _) in enumerate(game["picks"]):
                        own[i not in FIRST_DRAFTER_PICKS].append(game["pool"][slot])
                    for team in own:
                        draft.add(team)
        sources[name] = games
    pairs, triples = (space.enumerate(size) for size in (2, 3))
    maximum = {1: len(cards), 2: len(pairs), 3: len(triples), 4: len(legal_groups(space.items, 4))}
    opposition = {size: disjoint_matchups(space.items, size) for size in (2, 3, 4, 6)}
    return {"card_ids": space.ids, "sources": sources, "draft": draft.report(maximum), "active": active.report(maximum),
            "observed_matchups": {"lead_pairs": len(lead_duels), "four_card_lineups": len(four_duels)},
            "draft_exposures": {card: draft.groups[1][(i,)] for i, card in enumerate(space.ids)},
            "active_exposures": {card: active.groups[1][(i,)] for i, card in enumerate(space.ids)},
            "active_counts": {"leads": [[row, n] for row, n in active.leads.items()],
                              "pairs": [[row, n] for row, n in active.groups[2].items()],
                              "triples": [[row, n] for row, n in active.groups[3].items()]},
            "combinations": {"columns": ["cards", "best_attack_coverage", "coverage_complement", "safest_switch_risk"],
                             "pairs": pairs, "triples": triples},
            "directed_attack": space.attack.tolist(), "disjoint_unordered_matchups": opposition,
            "four_card_ordered_lead_actions": maximum[4] * 12, "four_card_unordered_lead_matchups": opposition[4] * 36,
            "pool_count": comb(len(cards), 18), "preview_actions_per_six": comb(6, 4) * 12}
