"""Screen many legal joint moves in one public world before rechecking finalists."""

from random import Random
from time import perf_counter

from .planning_refine import survivors
from .public_planning import _diverse, _ordered
from .search import _key
from .search85 import Search85, weighted_advantages
from .sim import SEATS, fingerprint
from .tactics import tactical_actions


class WideSearch85(Search85):
    def __init__(self, *args, scout=None, **settings):
        super().__init__(*args, **settings)
        self.scout = scout

    def _plan(self, decision, seat, evidence, ranked, coarse):
        if len(decision.actions) <= 18:
            return super()._plan(decision, seat, evidence, ranked, coarse)
        base = ranked[0]
        deadline = perf_counter() + min(self.move_budget, (decision.seconds_left or 50.) - 1.)
        if perf_counter() + .10 >= deadline:
            return self._fallback("clock_fallback", base)
        foe = next(player for player in SEATS if player != seat)
        events = evidence["events"]
        turn = decision.observation["turn"]
        self._observe(turn, events, foe)
        self.cursor = (turn, len(events))
        rosters = {seat: self.our_roster, foe: self.joint.history.opponent_roster}
        extra = self._proposals(decision, seat, rosters, evidence)
        learned = self.scout.proposals(decision, self.joint.history, events) if self.scout else []
        informed = tactical_actions(decision, card_profiles=self.profiles)
        order = _ordered([base], extra, learned, ranked[:4], informed[:5], ranked, informed, coarse)
        options = _diverse(order, 49)
        self.counts["scout_proposals"] += len(learned)
        expected = {_key(action) for action in decision.actions}
        rng = Random(fingerprint(["public-planning-v1", seat, events]))
        seeds = [[rng.randrange(65536) for _ in range(4)]]
        worlds = []
        for world_id, (team, weight) in enumerate(self._deployments(foe, events)[:self.worlds]):
            if perf_counter() + .30 >= deadline:
                break
            setup = self._world(decision, seat, foe, evidence, rosters, team, world_id, expected, deadline - .25)
            if setup is None:
                continue
            world, distributions = setup
            self.pending[team] = weight, distributions
            worlds.append((world, weight))
        if not worlds:
            return self._fallback("no_completed_world", base)
        first, _ = worlds[0]
        world_id, replies, weights = first
        mass = sum(weights[:4])
        scores = self._score((world_id, replies[:4], [weight / mass for weight in weights[:4]]), seat, options, seeds,
                             min(deadline, perf_counter() + 1.85), 1)
        if scores is None:
            return self._fallback("wide_screen_failed", base)
        self.counts["searched"] += 1
        self.counts["wide_screen_choices"] += len(options)
        shortlisted = survivors(scores, 8 if learned else 9)
        if learned:
            model_pick = next((index for proposal in learned for index, option in enumerate(options)
                               if proposal == option and index not in shortlisted), None)
            if model_pick is not None and scores[model_pick] >= min(scores[i] for i in shortlisted) - .45:
                shortlisted.append(model_pick)
                self.counts["scout_retained"] += 1
        candidates = [options[i] for i in shortlisted]
        reference = scores[0]
        first_pass = [scores[i] - reference for i in shortlisted]
        refined = []
        for world, weight in worlds:
            values = self._score(world, seat, candidates, seeds, deadline, 1)
            if values is not None:
                refined.append((weight, values))
        predictor = weighted_advantages(refined) if refined else first_pass
        indices = survivors(predictor, 5)
        checking = [candidates[i] for i in indices]
        followup = [[rng.randrange(65536) for _ in range(4)] for _ in range(2)]
        final = []
        for world, weight in worlds:
            values = self._score(world, seat, checking, followup, deadline, 2)
            if values is not None:
                final.append((weight, values))
        if final:
            margin = [.44 * predictor[i] + .56 * score for i, score in zip(indices, weighted_advantages(final), strict=True)]
        else:
            margin = [predictor[i] for i in indices]
        self.counts["confirmed" if final else "confirmation_fallback"] += 1
        best = max(range(len(margin)), key=margin.__getitem__)
        if margin[best] < self.min_edge:
            return base
        chosen = checking[best]
        self.counts["changed"] += 1
        self.counts["model_proposal_chosen"] += chosen in extra
        self.counts["scout_proposal_chosen"] += chosen in learned
        return chosen
