"""A bounded one-turn search using a separate, public-evidence Showdown worker."""

from collections import Counter
from random import Random
from statistics import fmean
from time import perf_counter

from .game import BATTLE, PREVIEW, Decision, ident, make_decision
from .joint import JointPolicy, PROTECT
from .sim import FORMAT, SEATS, Showdown, fingerprint
from .tactics import SLOTS, tactical_actions, tactical_candidates

MAX_SEARCH_SECONDS = .25
MIN_EDGE = .12
TIMER_EVENTS = {"upkeep", "-weather", "-fieldstart", "-fieldend", "-sidestart", "-sideend"}


def _candidate_actions(baseline, joint_ranked, ranked, shortlist):
    options = [baseline]
    outside = next((action for action in ranked if action not in shortlist), None)
    limit = 3 if outside else 4
    for action in [ranked[0], *joint_ranked, *ranked]:
        if len(options) == limit:
            break
        if action not in options and action != outside:
            options.append(action)
    if outside:
        options.append(outside)
    return options


def _reply_actions(decision):
    ranked = tactical_actions(decision)
    replies = [ranked[0]]
    for is_variant in (
        lambda slot: slot.get("move_id") in PROTECT,
        lambda slot: slot["type"] == "switch",
    ):
        action = next((row for row in ranked if row not in replies and any(is_variant(row[slot]) for slot in SLOTS)), None)
        if action is not None:
            replies.append(action)
    return (replies + [action for action in ranked if action not in replies])[:3]


def _winner(matrix):
    raw_weights = (.60, .25, .15)[:len(matrix[0])]
    weights = [weight / sum(raw_weights) for weight in raw_weights]

    def score(row):
        by_reply = [fmean(draws) for draws in row]
        return .9 * sum(weight * value for weight, value in zip(weights, by_reply, strict=True)) + .1 * min(by_reply)

    scores = [score(row) for row in matrix]
    best = max(range(len(scores)), key=scores.__getitem__)
    if scores[best] - scores[0] < MIN_EDGE:
        return 0
    supported = 0
    for seed in range(len(matrix[0][0])):
        edge = sum(weight * (alternative[seed] - baseline[seed]) for weight, alternative, baseline
                   in zip(weights, matrix[best], matrix[0], strict=True))
        supported += edge > 0
    return best if supported >= 2 else 0


def _key(action):
    return tuple((atom["type"], atom.get("move_id"), atom.get("target"), ident(atom.get("species", "")))
                 for atom in (action[slot] for slot in SLOTS))


class SearchPolicy:
    def __init__(self, profiles: dict, worker: Showdown):
        self.joint = JointPolicy(profiles)
        self.worker = worker
        self.own_roster = ()
        self.search_counts = Counter()
        self.search_ms = []

    def observe(self, decision: Decision):
        self.joint.observe(decision)
        if decision.phase == PREVIEW:
            self.own_roster = tuple(mon["species"] for mon in decision.observation["team"] if mon and "species" in mon)

    def choose_with_public_history(self, decision: Decision, seat: str, evidence: dict | None) -> dict:
        if decision.phase != BATTLE:
            return self.joint(decision)
        self.search_counts["battle_decisions"] += 1
        ranked = tactical_actions(decision)
        shortlist = tactical_candidates(decision, ranked=ranked)
        joint_ranked = [action for action, _ in self.joint.rank(decision, shortlist)]
        baseline = joint_ranked[0]
        choice = self._search(decision, seat, evidence, baseline, joint_ranked, ranked, shortlist)
        self.joint.record(decision, choice)
        return choice

    def _fallback(self, reason, action):
        self.search_counts[reason] += 1
        return action

    def _search(self, decision, seat, evidence, baseline, joint_ranked, ranked, shortlist):
        if len(decision.actions) == 1:
            return self._fallback("forced_action", baseline)
        if evidence is None or evidence["force_switch"]:
            return self._fallback("no_move_history", baseline)
        if len(self.joint.history.revealed) < 4:
            return self._fallback("unrevealed_roster", baseline)
        board = decision.observation
        with_field = any(board[field] for field in ("weather", "fields", "side_conditions", "opponent_side_conditions"))
        started = perf_counter()
        budget = min([MAX_SEARCH_SECONDS, *(remaining - 1. for remaining in
                      (decision.seconds_left, decision.bank_seconds_left) if remaining is not None)])
        if budget < .01:
            return self._fallback("budget", baseline)
        rosters = [self.own_roster, self.joint.history.opponent_roster]
        if seat == "p2":
            rosters.reverse()
        prepared = self.worker.call("prepare", format=FORMAT, seat=seat, observation=board,
                                    rosters=rosters, events=evidence["events"], own_pp=evidence["own_pp"])
        if reason := prepared.get("reason"):
            return self._fallback(reason, baseline)
        self.search_counts["prepared"] += 1
        try:
            decisions = {view["seat"]: make_decision(BATTLE, view["raw"], view["templates"], view["seat"])
                         for view in prepared["views"]}
        except ValueError:
            return self._fallback("legal_mask_mismatch", baseline)
        if decisions.keys() != set(SEATS) or {_key(action) for action in decisions[seat].actions} != {
            _key(action) for action in decision.actions
        }:
            return self._fallback("legal_mask_mismatch", baseline)
        self.search_counts["legal_mask_matched"] += 1
        ours = _candidate_actions(baseline, joint_ranked, ranked, shortlist)
        replies = _reply_actions(decisions["p2" if seat == "p1" else "p1"])
        previous_events = [event for event in evidence["events"] if event.split("|", 2)[1] not in TIMER_EVENTS]
        rng = Random(fingerprint([seat, previous_events]))
        seeds = [[rng.randrange(65536) for _ in range(4)] for _ in range(3)]
        remaining_ms = 1000 * (budget - (perf_counter() - started))
        if remaining_ms <= 0:
            return self._fallback("budget", baseline)
        outcome = self.worker.call("evaluate", seat=seat, ours=ours, replies=replies, seeds=seeds, budget_ms=remaining_ms)
        elapsed = perf_counter() - started
        if reason := outcome.get("reason"):
            return self._fallback(reason, baseline)
        if elapsed >= budget:
            return self._fallback("budget", baseline)
        self.search_counts["completed"] += 1
        self.search_counts["completed_with_field"] += with_field
        self.search_counts["branches"] += len(ours) * len(replies) * len(seeds)
        self.search_ms.append(1000 * elapsed)
        if any(action not in shortlist for action in ours):
            self.search_counts["outside_shortlist_tested"] += 1
        choice = ours[_winner(outcome["matrix"])]
        self.search_counts["changed"] += choice != baseline
        self.search_counts["outside_shortlist_selected"] += choice not in shortlist
        return choice
