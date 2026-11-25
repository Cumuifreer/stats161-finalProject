"""Run the original Ace Trainer pivot policy as a local benchmark opponent."""

import json
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
        self.process = subprocess.Popen(
            [sys.executable, "-IB", str(Path(__file__).resolve())],
            cwd=self.checkout, env={}, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
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
    from agent.versions.pivot import PivotAgent
    from altruagent import DecisionContext
    from sim.translate import BATTLE_FORMAT, GAME_TYPE, battle_state

    for line in sys.stdin:
        try:
            message = json.loads(line)
            if message.get("start"):
                seat = message["seat"]
                opponent = "p2" if seat == "p1" else "p1"
                rosters = message["rosters"]
                session = f"stats161-{message['seed']}"
                context = DecisionContext(session, None, GAME_TYPE, seat, int(seat == "p2"))
                agent = PivotAgent()
                if not message["style"]:
                    agent.style_turn = lambda _template, _observation, _sheet, answer: (answer, [])
                agent.decision_log = log = DecisionLog()
                agent._print = lambda _: None
                agent.memory.pool_cards.update({
                    card["card_id"]: card for roster in rosters.values() for card in roster
                })
                agent.memory.observe_team_preview({
                    "your_roster": rosters[seat], "opponent_roster": rosters[opponent],
                })
                version = 0
                reply = {"ready": True}
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
