"""Seeded draft and doubles games over a local Showdown process."""

from dataclasses import replace
import json
from hashlib import sha256
from pathlib import Path
import random
import subprocess
from time import perf_counter

from .game import BATTLE, BATTLE_BANK, DECISION_LIMIT, DRAFT, DRAFT_SIZE, PREVIEW, Decision, ident, make_decision
from .pivot import PivotOpponent
from .policy import heuristic

SEATS = ("p1", "p2")
FORMAT = "gen9vgc2025regi"
FIRST_DRAFTER_PICKS = {0, 3, 4, 7, 8, 11}


def fingerprint(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def pool_fingerprint(cards, seed) -> str:
    pool = random.Random(seed).sample(cards, DRAFT_SIZE)
    return fingerprint(sorted(card["card_id"] for card in pool))


class Showdown:
    def __init__(self, *, audit_restoration: bool = False, public_search: bool = False):
        self.audit_restoration = audit_restoration
        self.public_search = public_search

    def __enter__(self):
        bridge = Path(__file__).with_name("showdown.cjs")
        command = ["node", str(bridge)]
        if self.audit_restoration:
            command.append("--audit-restoration")
        if self.public_search:
            command.append("--public-search")
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
        )
        return self

    def __exit__(self, *_):
        self.process.stdin.close()
        self.process.stdout.close()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

    def call(self, command: str, **kwargs) -> dict:
        try:
            self.process.stdin.write(json.dumps({"command": command, **kwargs}) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
        except BrokenPipeError as error:
            raise OSError("Showdown stopped. Verify npm install in pokemon/.") from error
        if not line:
            raise OSError("Showdown stopped. Verify npm install in pokemon/.")
        reply = json.loads(line)
        if error := reply.get("error"):
            raise RuntimeError(f"Showdown: {error}")
        return reply


def _draft(pool: list[dict], first: str, choose) -> dict[str, list[dict]]:
    rosters: dict[str, list[dict]] = {seat: [] for seat in SEATS}
    remaining = list(pool)
    second = next(seat for seat in SEATS if seat != first)
    for pick in range(12):
        seat = first if pick in FIRST_DRAFTER_PICKS else second
        selected_items = {ident(card["item"]) for card in rosters[seat]}
        legal = [card for card in remaining if ident(card["item"]) not in selected_items]
        if not legal:
            raise RuntimeError(f"Item Clause left {seat} without a legal pick at pick {pick + 1}")
        public_rosters = {
            player: [{"card_id": card["card_id"], "species": card["species"]} for card in roster]
            for player, roster in rosters.items()
        }
        decision = make_decision(
            DRAFT,
            {"available_cards": remaining, "rosters": public_rosters, "pick_number": pick + 1},
            legal,
            seat,
            DECISION_LIMIT[DRAFT],
        )
        action = choose(seat, decision)
        card = next(card for card in legal if card["card_id"] == action["card_id"])
        remaining.remove(card)
        rosters[seat].append(card)
    return rosters


def play_game(
    engine: Showdown, cards: list[dict], policies: dict, seed: int, samples: dict, *, pool=None, fixed_policies=None,
) -> str | None:
    rng = random.Random(seed)
    if pool is None:
        pool = rng.sample(cards, DRAFT_SIZE)
    first = rng.choice(SEATS)
    battle_seed = [rng.randrange(65536) for _ in range(4)]
    banks = {seat: BATTLE_BANK for seat in SEATS}
    fixed_policies = fixed_policies or {}

    def choose(seat: str, decision: Decision, view: dict | None = None) -> dict:
        if decision.phase == BATTLE:
            bank = banks[seat]
            decision = replace(decision, seconds_left=min(DECISION_LIMIT[BATTLE], bank), bank_seconds_left=bank)
        elif decision.seconds_left is None:
            decision = replace(decision, seconds_left=DECISION_LIMIT[decision.phase])
        started = perf_counter()
        configured = policies[seat]
        policy = fixed_policies.get(decision.phase, configured)
        if policy is not configured and decision.phase == PREVIEW:
            if observe := getattr(configured, "observe", None):
                observe(decision)
        if isinstance(policy, PivotOpponent):
            action = policy.choose(decision, view) if decision.phase == BATTLE else heuristic(decision)
        elif phase_search := getattr(policy, "choose_with_public_history", None):
            action = phase_search(decision, seat, (view or {}).get("search"))
        else:
            action = policy(decision)
        elapsed = perf_counter() - started
        late = elapsed >= decision.seconds_left
        if decision.phase == BATTLE:
            banks[seat] -= min(elapsed, decision.seconds_left)
        event = None
        if banks[seat] <= 0:
            event = "forfeit"
        elif late:
            event = "auto_action"
            action = rng.choice(decision.actions)
        if event is None:
            try:
                action = decision.validate(action)
            except ValueError:
                samples[seat].append((decision.phase, elapsed * 1000, "illegal_action"))
                raise
        samples[seat].append((decision.phase, elapsed * 1000, event))
        return None if event == "forfeit" else action

    rosters = _draft(pool, first, choose)
    return play_battle(engine, rosters, policies, seed, battle_seed, choose)


def play_battle(engine, rosters, policies, seed, battle_seed, choose):
    """Run from team preview over the same player-view decisions as a full game."""
    start = dict(format=FORMAT, seed=battle_seed, rosters=rosters)
    search_seats = [seat for seat, policy in policies.items() if hasattr(policy, "choose_with_public_history")]
    if search_seats:
        start["search_seats"] = search_seats
    for seat, policy in policies.items():
        if isinstance(policy, PivotOpponent):
            policy.start(seed, seat, rosters)
            start["pivot_seat"] = seat
    reply = engine.call("start", **start)
    for _ in range(500):
        if reply["ended"]:
            return reply["winner"]
        views = reply["views"]
        if not views:
            raise RuntimeError("Showdown has no pending decisions in a running match")
        for view in views:
            seat = view["seat"]
            phase = view["phase"]
            action = choose(seat, make_decision(phase, view["raw"], view["templates"], seat), view)
            if action is None:
                return next(other for other in SEATS if other != seat)
            reply = engine.call("choose", seat=seat, action=action)
    if reply["ended"]:
        return reply["winner"]
    raise RuntimeError("Benchmark match exceeded 500 decision rounds")
