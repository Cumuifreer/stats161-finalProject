"""Compare restored local fixture battles with uninterrupted battles."""

from collections import Counter
from random import Random
from time import perf_counter

from .game import BATTLE, DRAFT_SIZE, PREVIEW, make_decision
from .joint import JointPolicy
from .policy import heuristic
from .sim import FORMAT, SEATS, Showdown, _draft, play_game
from .tactics import SLOTS, load_card_profiles, tactical, tactical_actions


class RestorationAudit:
    def __init__(self, reference: Showdown, restored: Showdown):
        self.reference = reference
        self.restored = restored
        self.checkpoints = Counter()
        self.terminal_checks = 0
        self.restore_ms = []

    def call(self, command, **kwargs):
        expected = self.reference.call(command, **kwargs)
        actual = self.restored.call(command, **kwargs)
        if expected != actual:
            detail = [(board["seat"], board["phase"], board["raw"].get("turn")) for board in expected["views"]]
            raise RuntimeError(f"Restored continuation differs for fixture seed {self.seed}; expected views {detail}")
        if expected["ended"]:
            self.terminal_checks += 1
            return expected

        moving = [board for board in expected["views"] if board["phase"] == BATTLE]
        if not moving or moving[0]["raw"]["turn"] < 2:
            return expected
        self.checkpoints["midgame"] += 1
        self.checkpoints["one_pending_seat" if len(moving) == 1 else "two_pending_seats"] += 1
        if any(all(option["type"] != "move" for slot in board["templates"][0]["slots"]
                   for option in slot["options"]) for board in moving):
            self.checkpoints["replacement"] += 1
        started = perf_counter()
        round_trip = self.restored.call("audit_restore")
        self.restore_ms.append((perf_counter() - started) * 1000)
        if round_trip != expected:
            raise RuntimeError(f"Round-trip changed the decision contract for fixture seed {self.seed}")
        return expected


def audit_restoration(cards, seed: int, games: int) -> dict:
    if games < 1:
        raise ValueError("Restoration audit requires at least one game")
    profiles = load_card_profiles(cards, with_sets=True)
    with Showdown() as reference, Showdown(audit_restoration=True) as restored:
        audit = RestorationAudit(reference, restored)
        for match_seed in range(seed, seed + games):
            audit.seed = match_seed
            play_game(audit, cards, {"p1": JointPolicy(profiles), "p2": tactical}, match_seed,
                      {seat: [] for seat in SEATS})
        demonstration = next((turn for offset in range(16)
                              if (turn := _ambiguous_fixture(reference, restored, cards, seed + offset)) is not None), None)
    if not audit.restore_ms:
        raise RuntimeError("No midgame positions were sampled")
    if demonstration is None:
        raise RuntimeError("No pair of indistinguishable public fixture positions was found")
    return {"fixture_first_seed": seed, "terminal_results_matched": audit.terminal_checks,
            "checkpoints": dict(audit.checkpoints), "restore_ms": audit.restore_ms,
            "same_public_decision_different_legal_foe_switches_at_turn": demonstration}


def _ambiguous_fixture(first, second, cards, seed):
    rng = Random(seed)
    rosters = _draft(rng.sample(cards, DRAFT_SIZE), "p1", lambda _, decision: heuristic(decision))
    settings = {"format": FORMAT, "seed": [rng.randrange(65536) for _ in range(4)], "rosters": rosters}
    engines = (first, second)
    preview = first.call("start", **settings)["views"]
    second.call("start", **settings)
    opening = [None, None]
    for view in preview:
        seat = view["seat"]
        decision = make_decision(PREVIEW, view["raw"], view["templates"], seat)
        primary = decision.actions[0]
        alternative = primary
        if seat == "p2":
            alternative = next(action for action in decision.actions if action["leads"] == primary["leads"]
                               and action["bring"] != primary["bring"])
        for index, (engine, action) in enumerate(zip(engines, (primary, alternative), strict=True)):
            opening[index] = engine.call("choose", seat=seat, action=action)

    after = [None, None]
    for view in opening[0]["views"]:
        seat = view["seat"]
        decision = make_decision(BATTLE, view["raw"], view["templates"], seat)
        action = next(candidate for candidate in tactical_actions(decision)
                      if all(candidate[slot]["type"] != "switch" for slot in SLOTS))
        for index, engine in enumerate(engines):
            after[index] = {board["seat"]: board for board in engine.call("choose", seat=seat, action=action)["views"]}
    if not all(set(views) == set(SEATS) for views in after):
        return None
    turn = after[0]["p1"]["raw"]["turn"]
    switches = [{option["species"] for slot in views["p2"]["templates"][0]["slots"] for option in slot["options"]
                 if option["type"] == "switch"} for views in after]
    if turn >= 2 and after[0]["p1"] == after[1]["p1"] and switches[0] != switches[1]:
        return turn
    return None
