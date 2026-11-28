"""Reserve opening confirmation slots for distinct legal tactical plans."""

from hashlib import sha256
from pathlib import Path
from random import Random
from time import perf_counter
from unittest.mock import patch

from .. import blind85, lab85, quicksilver80
from ..joint import PROTECT
from ..planning_refine import survivors
from ..public_planning import _diverse, _ordered
from ..research80.scouting.replies import ForecastReplies80
from ..search import _key
from ..search85 import weighted_advantages
from ..sim import SEATS, fingerprint
from ..tactics import SLOTS, SPEED_CONTROL, tactical_actions
from . import distilled


PIVOT_MOVES = {"uturn", "voltswitch", "flipturn", "partingshot", "chillyreception"}
CONTROL_MOVES = SPEED_CONTROL | {"electroweb", "fakeout", "spore", "yawn", "encore", "taunt"}
SETUP_MOVES = {"followme", "ragepowder", "nastyplot", "calmmind", "swordsdance", "reflect", "lightscreen"}


def roles(action):
    """Identify which slot spends its move on a positional or defensive play."""
    found = set()
    for index, slot in enumerate(SLOTS):
        atom = action[slot]
        move = atom.get("move_id")
        if atom["type"] == "switch" or move in PIVOT_MOVES:
            found.add(("pivot", index))
        elif move in PROTECT:
            found.add(("protect", index))
        elif move in CONTROL_MOVES:
            found.add(("control", index))
        elif move in SETUP_MOVES:
            found.add(("setup", index))
    return found


def shortlist(actions, scores):
    finalists = survivors(scores, 5)
    uses = [roles(action) for action in actions]
    covered = set().union(*(uses[index] for index in finalists))
    floor = max(scores[0] - .28, max(scores) - .55)
    choices = [
        index for index, score in enumerate(scores)
        if index not in finalists and uses[index] and score >= floor
    ]
    for index in sorted(choices, key=scores.__getitem__, reverse=True):
        if uses[index] - covered:
            finalists.append(index)
            covered.update(uses[index])
        if len(finalists) >= 9:
            break
    return finalists


class OpeningPortfolio(ForecastReplies80):
    def _plan(self, decision, seat, evidence, ranked, coarse):
        board = decision.observation
        if (board["turn"] > 4 or len(decision.actions) <= 18 or not all(board["active"])
                or not all(board["opponents"]) or not any(map(roles, decision.actions))):
            return super()._plan(decision, seat, evidence, ranked, coarse)

        self.check_endgame = False
        base = ranked[0]
        deadline = perf_counter() + min(self.move_budget, (decision.seconds_left or 50.) - 1.)
        if deadline - perf_counter() < .25:
            return self._fallback("clock_fallback", base)
        foe = next(player for player in SEATS if player != seat)
        events = evidence["events"]
        self._observe(board["turn"], events, foe)
        self.cursor = (board["turn"], len(events))
        rosters = {seat: self.our_roster, foe: self.joint.history.opponent_roster}
        extra = self._proposals(decision, seat, rosters, evidence)
        learned = self.scout.proposals(decision, self.joint.history, events) if self.scout else []
        informed = tactical_actions(decision, card_profiles=self.profiles)
        options = _diverse(_ordered([base], extra, learned, ranked[:4], informed[:5], ranked, informed, coarse), 42)
        self.counts["scout_proposals"] += len(learned)
        expected = {_key(action) for action in decision.actions}
        worlds = []
        for world_id, (team, weight) in enumerate(self._deployments(foe, events)[:self.worlds]):
            if deadline - perf_counter() < .35:
                break
            setup = self._world(decision, seat, foe, evidence, rosters, team, world_id, expected, deadline - .35)
            if setup is not None:
                world, distributions = setup
                self.pending[team] = weight, distributions
                worlds.append((world, weight))
        if not worlds:
            return self._fallback("no_completed_world", base)

        rng = Random(fingerprint(["public-planning-v1", seat, events]))
        initial = [[rng.randrange(65536) for _ in range(4)]]
        (world_id, replies, weights), _ = worlds[0]
        mass = sum(weights[:4])
        scores = self._score((world_id, replies[:4], [weight / mass for weight in weights[:4]]),
                             seat, options, initial, min(deadline - .55, perf_counter() + 1.85), 1)
        if scores is None:
            return self._fallback("wide_screen_failed", base)
        self.counts["searched"] += 1
        indices = shortlist(options, scores)
        finalists = [options[index] for index in indices]
        self.counts["opening_portfolio_reviews"] += 1
        self.counts["opening_portfolio_reserved"] += len(indices) - min(5, len(options))
        followup = [[rng.randrange(65536) for _ in range(4)] for _ in range(2)]
        confirmed = []
        for world, weight in worlds:
            values = self._score(world, seat, finalists, followup, deadline, 2)
            if values is not None:
                confirmed.append((weight, values))
        if not confirmed:
            return self._fallback("confirmation_fallback", base)
        self.counts["confirmed"] += 1
        margins = [.20 * (scores[index] - scores[0]) + .80 * gain
                   for index, gain in zip(indices, weighted_advantages(confirmed), strict=True)]
        best = max(range(len(margins)), key=margins.__getitem__)
        if margins[best] < self.min_edge:
            return base
        chosen = finalists[best]
        self.counts["changed"] += 1
        self.counts["model_proposal_chosen"] += chosen in extra
        self.counts["scout_proposal_chosen"] += chosen in learned
        self.counts["opening_portfolio_reserved_selected"] += best >= 5
        return chosen


def evaluate(*args):
    with patch.object(lab85, "ForecastReplies80", OpeningPortfolio):
        return lab85.evaluate(*args)


def freeze(*args):
    result = FROZEN(*args)
    for path in (Path(__file__), Path(distilled.__file__)):
        result["files"][str(path)] = sha256(path.read_bytes()).hexdigest()
    return result


FROZEN = quicksilver80.freeze


if __name__ == "__main__":
    with patch.object(distilled, "evaluate", evaluate), patch.object(blind85, "measure", distilled.measure), \
         patch.object(quicksilver80, "freeze", freeze):
        quicksilver80.main()
