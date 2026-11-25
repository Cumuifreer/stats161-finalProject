"""Evaluate legal lineups in hypothetical games made from the public six-card previews."""

from random import Random
from time import perf_counter

from .game import PREVIEW, ident
from .joint import JointPolicy
from .policy import heuristic
from .sim import SEATS, fingerprint, play_battle
from .tactics import tactical_actions, tactical_candidates


def lineup_options(decision, limit=5):
    ranked = tactical_actions(decision)
    selected = [ranked[0]]
    brought = {frozenset(ranked[0]["bring"])}
    for action in ranked:
        team = frozenset(action["bring"])
        if team not in brought:
            selected.append(action)
            brought.add(team)
            if len(selected) == limit - 1:
                break
    diverse_leads = tactical_candidates(decision, ranked=ranked)
    selected.extend(action for action in diverse_leads if action not in selected)
    return selected[:limit]


def _best_lineup(scores):
    best = max(range(len(scores)), key=lambda i: (sum(scores[i]), min(scores[i])))
    baseline = scores[0]
    return best if (sum(scores[best]) - sum(baseline) >= 2
                    and all(new >= old for new, old in zip(scores[best], baseline, strict=True))) else 0


class LineupPolicy(JointPolicy):
    def __init__(self, profiles, cards, worker):
        super().__init__(profiles)
        self.cards = cards
        self.worker = worker
        self.different_choices = 0
        self.fallbacks = 0

    def __call__(self, decision):
        if decision.phase != PREVIEW:
            return super().__call__(decision)
        self.observe(decision)
        options = self.options(decision)
        names = [[mon.get("species") for mon in decision.observation[key] if mon]
                 for key in ("team", "opponent_team")]
        if len(options) == 1:
            return options[0]
        if any(len(team) != 6 or not set(team) <= self.cards.keys() for team in names):
            self.fallbacks += 1
            return options[0]
        rosters = {seat: [self.cards[name] for name in team] for seat, team in zip(SEATS, names, strict=True)}
        rng = Random(fingerprint([sorted(team) for team in names]))
        seeds = [[rng.randrange(65536) for _ in range(4)] for _ in range(2)]
        deadline = perf_counter() + min(5., (decision.seconds_left or 85.) - 1.)
        scores = []
        try:
            for action in options:
                scores.append([sum(self._play(rosters, action, variant, seed, deadline) for seed in seeds)
                               for variant in (0, 1)])
        except (ValueError, OSError, RuntimeError):
            self.fallbacks += 1
            return options[0]
        winner = _best_lineup(scores)
        self.different_choices += winner != 0
        return options[winner]

    def options(self, decision):
        return lineup_options(decision)

    def _play(self, rosters, selected, variant, seed, deadline):
        policies = {seat: JointPolicy(self.profiles) for seat in SEATS}

        def choose(seat, decision, _):
            if perf_counter() >= deadline:
                raise TimeoutError("Lineup evaluation exceeded its budget")
            policy = policies[seat]
            if decision.phase != PREVIEW:
                return decision.validate(policy(decision))
            policy.observe(decision)
            if seat == "p1":
                return decision.validate(selected)
            preferred = tactical_candidates(decision)
            alternative = heuristic(decision)
            if variant:
                return alternative if alternative != preferred[0] else preferred[min(1, len(preferred) - 1)]
            return preferred[0]

        winner = play_battle(self.worker, rosters, policies, 0, seed, choose)
        return .5 if winner is None else float(winner == "p1")


def index_cards(cards):
    return {ident(card["species"]): card for card in cards}
