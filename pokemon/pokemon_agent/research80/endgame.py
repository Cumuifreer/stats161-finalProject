"""Check late-game finalists with exact simultaneous replies."""

from time import perf_counter

from ..horizon85 import Horizon85
from ..public_planning import PublicPlanner
from ..wide85 import WideSearch85


class CheckedEndgame(WideSearch85, Horizon85):
    check_endgame = False

    def _plan(self, decision, seat, evidence, ranked, coarse):
        # Late positions often hinge on Protect, forced switches, and a particular response.
        board = decision.observation
        closing = len(decision.actions) <= 26 and (
            not all(board["active"]) or not all(board["opponents"]) or board["turn"] >= 7
        )
        self.check_endgame = closing
        return super()._plan(decision, seat, evidence, ranked, coarse)

    def _score(self, world, seat, actions, seeds, deadline, depth):
        if depth == 2 and self.check_endgame and deadline - perf_counter() > .65:
            # Use the stronger continuation where it can complete. Keep time for the short review.
            score = super()._score(world, seat, actions, seeds, deadline - .38, depth)
            if score is not None:
                self.counts["exact_endgame_reviews"] += 1
                return score
        return PublicPlanner._score(self, world, seat, actions, seeds, deadline, depth)
