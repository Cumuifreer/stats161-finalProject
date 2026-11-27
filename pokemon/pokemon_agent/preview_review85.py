"""Double-check Preview finalists with distinct complete-match continuation policies."""

from collections import Counter
from dataclasses import replace
from statistics import fmean
from time import perf_counter

from .game import BATTLE, BATTLE_BANK, DECISION_LIMIT, PREVIEW
from .joint import JointPolicy
from .planning_preview import publish_sets
from .preview85 import independent_draws
from .sim import SEATS, play_battle


class FullPreviewReview:
    def __init__(self, engine, models, cards, profiles):
        self.engine = engine
        self.models = models
        self.cards = cards
        self.profiles = profiles
        self.counts = Counter()

    def __call__(self, decision, options, replies, checked, deadline):
        chosen = checked["index"]
        candidates = list(dict.fromkeys([chosen, *(row["index"] for row in checked["rows"][:2]), 0]))[:3]
        if len(candidates) == 1 or len(replies) < 3 or perf_counter() + 1. >= deadline:
            return chosen
        board = publish_sets(decision.observation, self.cards)
        rosters = {seat: [self.cards[mon["species"]] for mon in board[side]]
                   for seat, side in zip(SEATS, ("team", "opponent_team"), strict=True)}
        draws = independent_draws(decision, "full-preview-review-v1", 8)
        schedule = [("ace-code", 0), ("pivot", 0), ("joint", 1), ("joint", 2),
                    ("ace-code", 1), ("pivot", 1), ("joint", 0), ("joint", 2)]
        results = {index: [] for index in candidates}
        for seed, (style, rival) in zip(draws, schedule, strict=True):
            try:
                paired = {index: self._play(rosters, options[index], replies[rival], seed, style, deadline)
                          for index in candidates}
            except TimeoutError:
                self.counts["full_review_budget"] += 1
                break
            except (RuntimeError, ValueError, OSError):
                self.counts["full_review_error"] += 1
                break
            for index, score in paired.items():
                results[index].append(score)
        if len(results[chosen]) < 4:
            return chosen
        comparisons = {index: [after - before for after, before in zip(outcomes, results[chosen], strict=True)]
                       for index, outcomes in results.items()}
        candidate = max(candidates, key=lambda index: fmean(comparisons[index]))
        margin = comparisons[candidate]
        self.counts["full_reviewed"] += 1
        if candidate != chosen and sum(margin) >= 2 and min(margin[:2]) >= 0.:
            self.counts["full_review_changed"] += 1
            return candidate
        return chosen

    def _play(self, rosters, ours, theirs, seed, style, deadline):
        number = int.from_bytes(b"".join(part.to_bytes(2, "big") for part in seed), "big")
        if style in self.models:
            opponent = self.models[style]
            opponent.setup(number, "p2", "p1")
        else:
            opponent = JointPolicy(self.profiles)
        policies = {"p1": JointPolicy(self.profiles), "p2": opponent}

        def choose(seat, decision, view):
            if perf_counter() >= deadline:
                raise TimeoutError("Preview review exhausted its time")
            policy = policies[seat]
            if decision.phase == PREVIEW:
                if isinstance(policy, JointPolicy):
                    policy.observe(decision)
                return decision.validate(ours if seat == "p1" else theirs)
            if isinstance(policy, JointPolicy):
                return decision.validate(policy(decision))
            decision = replace(decision, seconds_left=DECISION_LIMIT[BATTLE], bank_seconds_left=BATTLE_BANK)
            return decision.validate(policy.choose(decision, view))

        outcome = play_battle(self.engine, rosters, policies, number, seed, choose)
        self.counts["full_rollouts"] += 1
        return .5 if outcome is None else float(outcome == "p1")
