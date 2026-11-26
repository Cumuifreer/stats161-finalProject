"""Run the original Ace Trainer pivot policy as a local benchmark opponent."""

import json
import os
from hashlib import sha256
from pathlib import Path
import subprocess
import sys


class PivotOpponent:
    def __init__(self, checkout: Path, *, style: bool = True):
        self.checkout = checkout.expanduser().resolve()
        self.style = style
        self.fallbacks = 0

    def __enter__(self):
        if not (self.checkout / "agent" / "versions" / "pivot.py").is_file():
            raise ValueError("--ace-trainer must point to an Ace Trainer checkout with agent/versions/pivot.py")
        # Winsock needs SystemRoot even though the benchmark worker receives no API credentials.
        environment = {"SystemRoot": os.environ["SystemRoot"]} if os.name == "nt" else {}
        self.process = subprocess.Popen(
            [sys.executable, "-IB", "-X", "utf8", str(Path(__file__).resolve())],
            cwd=self.checkout, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE, encoding="utf-8",
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

    def _call(self, message: dict) -> dict:
        try:
            self.process.stdin.write(json.dumps(message) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
        except BrokenPipeError as error:
            raise RuntimeError("Ace Trainer benchmark worker stopped") from error
        if not line:
            raise RuntimeError("Ace Trainer benchmark worker stopped. Check the checkout and its Python dependencies.")
        reply = json.loads(line)
        if error := reply.get("error"):
            raise RuntimeError(f"Ace Trainer pivot: {error}")
        return reply

    def start(self, seed: int, seat: str, rosters: dict):
        self.fallbacks = 0
        self._call({"start": True, "seed": seed, "seat": seat, "rosters": rosters, "style": self.style})

    def choose(self, decision, view: dict) -> dict:
        reply = self._call({
            "view": view["pivot"], "slots": view["templates"][0]["slots"],
            "clock": {"decision_seconds_left": decision.seconds_left,
                      "bank_seconds_left": decision.bank_seconds_left},
        })
        self.fallbacks = reply["fallbacks"]
        return decision.validate(reply["action"])


class DecisionLog:
    """Collect upstream errors in memory so a fallback can't hide an adapter bug."""

    def __init__(self):
        self.error = None
        self.fallbacks = 0

    def write(self, kind: str, **fields):
        if kind == "error" and self.error is None:
            self.error = fields["error"]
        if kind == "turn" and fields["code_row"] is None:
            self.fallbacks += 1


def _run_worker():
    protocol = sys.stdout
    sys.stdout = sys.stderr
    sys.path.insert(0, str(Path.cwd()))
    from agent.pokemon.memory import species_key
    from agent.agent import PokemonAgent
    from agent.versions.pivot import PivotAgent
    from altruagent import DecisionContext
    from sim.translate import (BATTLE_FORMAT, GAME_TYPE, LINEUP_INSTRUCTIONS, LINEUP_LABEL, _action, _new_state,
                               battle_state, draft_state)

    full_match = False

    def new_player(seed, seat, style, code_only=False):
        session = f"stats161-{seed}"
        context = DecisionContext(session, None, GAME_TYPE, seat, int(seat == "p2"))
        agent = PokemonAgent(None) if code_only else PivotAgent()
        if not style and not code_only:
            agent.style_turn = lambda _template, _observation, _sheet, answer: (answer, [])
        agent.decision_log = log = DecisionLog()
        agent._print = lambda _: None
        return session, context, agent, log

    for line in sys.stdin:
        try:
            message = json.loads(line)
            if message.get("setup"):
                full_match = True
                seat = message["seat"]
                opponent = "p2" if seat == "p1" else "p1"
                first = message["first"]
                session, context, agent, log = new_player(message["seed"], seat, message["style"], message["code_only"])
                agent.memory.tiebreak_seed = int.from_bytes(sha256(f"{session}|{seat}".encode()).digest()[:8], "big")
                reply = {"ready": True}
            elif message.get("start"):
                seat = message["seat"]
                opponent = "p2" if seat == "p1" else "p1"
                rosters = message["rosters"]
                if not full_match:
                    session, context, agent, log = new_player(message["seed"], seat, message["style"])
                agent.memory.pool_cards.update({
                    card["card_id"]: card for roster in rosters.values() for card in roster
                })
                agent.memory.observe_team_preview({
                    "your_roster": rosters[seat], "opponent_roster": rosters[opponent],
                })
                version = 0
                reply = {"ready": True}
            elif phase := message.get("phase"):
                obs, clock = message["observation"], message["clock"]
                if phase == "draft":
                    picked = {key: iter(value) for key, value in ((seat, obs["picked"]), (opponent, obs["opponent_picked"]))}
                    rosters = {key: obs["picked"] if key == seat else obs["opponent_picked"] for key in ("p1", "p2")}
                    other_first = "p2" if first == "p1" else "p1"
                    order = [first if i in (0, 3, 4, 7, 8, 11) else other_first for i in range(obs["pick"] - 1)]
                    picks = [{"pick_number": i, "seat_id": owner, "card_id": card["card_id"], "species": card["species"]}
                             for i, owner in enumerate(order, 1) for card in [next(picked[owner])]]
                    state = draft_state(session, me=seat, pool=obs["cards"], rosters=rosters, picks=picks,
                                        first_drafter=first, current_seat=seat, version=len(picks))
                    legal = {entry.action_id.split(":", 1)[1] for entry in state.legal_actions}
                    if legal != set(message["legal_cards"]):
                        raise ValueError("Ace Trainer Draft translation changed the legal menu")
                else:
                    names = message["roster"]
                    observation = {"phase": phase, "battle_format": BATTLE_FORMAT, "your_roster": obs["team"],
                                   "opponent_roster": obs["opponent_team"], "clock": clock, "waiting_for_action": True}
                    payload = {"type": "select_lineup", "roster": names, "bring_count": 4, "lead_count": 2,
                               "instructions": LINEUP_INSTRUCTIONS}
                    legal = [_action(session, "select_lineup", LINEUP_LABEL, payload, 0)]
                    state = _new_state(session, phase, observation, legal, 0, actor=seat, position=int(seat == "p2"))
                action = agent.choose_action(state, context).action
                if hasattr(action, "input"):
                    action = action.input["action"]
                if log.error:
                    raise RuntimeError(log.error)
                reply = {"action": action}
            else:
                player_view = message["view"]
                request = player_view["request"]
                if version == 0:
                    agent.memory.my_lineup = [
                        species_key(mon["details"].split(",", 1)[0]) for mon in request["side"]["pokemon"]
                    ]
                snapshot = {
                    "battle_tag": f"battle-{BATTLE_FORMAT}-{session}",
                    "sides": {seat: {
                        "request": request,
                        "preview_request": player_view["preview_request"],
                    }, opponent: {}},
                    "logs": {seat: player_view["protocol_log"]},
                }
                version += 1
                state = battle_state(session, snapshot, seat, version, clock=message["clock"])
                state.raw["observation"]["can_tera"] = [False, False]
                template = state.legal_actions[0].input["action"]
                # Keep Ace Trainer's metadata while using our bridge's actual legal options.
                for slot, legal in zip(template["slots"], message["slots"], strict=True):
                    moves = {option["move_id"]: option for option in slot["options"] if option["type"] == "move"}
                    options = []
                    for option in legal["options"]:
                        move = moves.get(option.get("move_id"), {})
                        enriched = move | option
                        if "target_options" in move:
                            enriched["target_options"] = [
                                target for target in move["target_options"] if target["target"] in option["targets"]
                            ]
                        options.append(enriched)
                    slot["options"] = options
                action = agent.choose_action(state, context).action
                if log.error:
                    raise RuntimeError(log.error)
                reply = {"action": action, "fallbacks": agent.memory.fallbacks + log.fallbacks}
        except Exception as error:
            reply = {"error": f"{type(error).__name__}: {error}"}
        print(json.dumps(reply), file=protocol, flush=True)


if __name__ == "__main__":
    _run_worker()
