"""Seeded draft and doubles games over a local Showdown process."""

import json
from pathlib import Path
import random
import subprocess
from time import perf_counter

from .game import BATTLE, BATTLE_BANK, DECISION_LIMIT, DRAFT, DRAFT_SIZE, Decision, ident, make_decision

SEATS = ("p1", "p2")
FORMAT = "gen9vgc2025regi"
FIRST_DRAFTER_PICKS = {0, 3, 4, 7, 8, 11}


class Showdown:
    def __enter__(self):
        bridge = Path(__file__).with_name("showdown.cjs")
        self.process = subprocess.Popen(
            ["node", str(bridge)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
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
    engine: Showdown, cards: list[dict], policies: dict, seed: int, samples: dict, *, pool=None,
) -> str | None:
    rng = random.Random(seed)
    if pool is None:
        pool = rng.sample(cards, DRAFT_SIZE)
    first = rng.choice(SEATS)
    battle_seed = [rng.randrange(65536) for _ in range(4)]
    banks = {seat: BATTLE_BANK for seat in SEATS}

    def choose(seat: str, decision: Decision) -> dict:
        started = perf_counter()
        action = policies[seat](decision)
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
        samples[seat].append((decision.phase, elapsed * 1000, event))
        return action if event else decision.validate(action)

    rosters = _draft(pool, first, choose)
    reply = engine.call("start", format=FORMAT, seed=battle_seed, rosters=rosters)
    for _ in range(500):
        if reply["ended"]:
            return reply["winner"]
        views = reply["views"]
        if not views:
            raise RuntimeError("Showdown has no pending decisions in a running match")
        for view in views:
            seat = view["seat"]
            phase = view["phase"]
            bank = banks[seat] if phase == BATTLE else None
            seconds = min(DECISION_LIMIT[phase], bank) if bank is not None else DECISION_LIMIT[phase]
            decision = make_decision(phase, view["raw"], view["templates"], seat, seconds, bank)
            action = choose(seat, decision)
            if banks[seat] <= 0:
                return next(other for other in SEATS if other != seat)
            reply = engine.call("choose", seat=seat, action=action)
    if reply["ended"]:
        return reply["winner"]
    raise RuntimeError("Benchmark match exceeded 500 decision rounds")
