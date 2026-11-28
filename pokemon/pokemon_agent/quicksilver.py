"""Run the pinned public Quicksilver policy on the benchmark's player views."""

from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
from queue import Empty, Queue
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Thread
from time import monotonic
import traceback

if __package__:
    from .game import DRAFT, ident
    from .pivot import PivotOpponent
    from .sim import FIRST_DRAFTER_PICKS


    class QuicksilverOpponent(PivotOpponent):
        def __init__(self, checkout, translator):
            super().__init__(checkout)
            self.translator = translator.resolve()
            self.process = None

        def __enter__(self):
            return self

        def __exit__(self, *args):
            if self.process is not None:
                super().__exit__(*args)
                self.process = None

        def _call(self, message):
            self.process.stdin.write(json.dumps(message) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("The Quicksilver benchmark worker stopped")
            reply = json.loads(line)
            if "error" in reply:
                raise RuntimeError(reply["error"])
            self.metrics.update(reply.get("metrics", {}))
            self.fallbacks = sum(count for key, count in self.metrics.items() if "_fallback:" in key)
            return reply

        def setup(self, root, seat, first):
            self.__exit__()
            node = shutil.which("node")
            if node is None:
                raise RuntimeError("Quicksilver requires Node and its lockfile dependencies")
            environment = {key: os.environ[key] for key in ("SystemRoot", "TEMP", "TMP") if key in os.environ}
            environment.update(PATH=str(Path(node).parent), POKEMON_JEV="0")
            self.process = subprocess.Popen(
                [sys.executable, "-IB", "-X", "utf8", str(Path(__file__).resolve()), str(self.translator)],
                cwd=self.checkout, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE, encoding="utf-8",
            )
            self.metrics = Counter()
            self.fallbacks = 0
            self._call({"op": "setup", "seat": seat, "first": first,
                        "first_drafter_picks": list(FIRST_DRAFTER_PICKS)})

        def start(self, seed, seat, rosters):
            self._call({"op": "start", "rosters": rosters})

        @staticmethod
        def clock(decision):
            return {"decision_seconds_left": decision.seconds_left, "bank_seconds_left": decision.bank_seconds_left}

        def choose_setup(self, decision):
            message = {"op": decision.phase, "observation": decision.observation, "clock": self.clock(decision)}
            if decision.phase == DRAFT:
                message["legal"] = decision.actions
                return decision.validate(self._call(message)["action"])
            answer = self._call(message)["action"]
            bring = {ident(mon) for mon in answer["bring"]}
            leads = [ident(mon) for mon in answer["leads"]]
            return decision.validate(next((
                action for action in decision.actions if {ident(mon) for mon in action["bring"]} == bring
                and [ident(mon) for mon in action["leads"]] == leads
            ), None))

        def choose(self, decision, view):
            answer = self._call({"op": "moving", "view": view["pivot"], "clock": self.clock(decision),
                                 "slots": view["templates"][0]["slots"]})["action"]
            return decision.validate(answer)


def windows_engine(base):
    """Replace unsupported Windows pipe polling with a dedicated stdout reader."""
    class Engine(base):
        def _start(self):
            super()._start()
            self.lines = Queue()

            def read(pipe, output):
                try:
                    for line in pipe:
                        output.put(line)
                finally:
                    output.put(None)

            Thread(target=read, args=(self._proc.stdout, self.lines), daemon=True).start()

        def close(self):
            process = self._proc
            super().close()
            if process is not None:
                process.wait()
                process.stdin.close()
                process.stdout.close()

        def _call(self, payload, budget_ms):
            if not self.usable:
                return None
            with self._lock:
                try:
                    if self._proc is None or self._proc.poll() is not None:
                        self.close()
                        self._start()
                    self._next_id += 1
                    self._proc.stdin.write(json.dumps({"id": self._next_id, **payload}) + "\n")
                    self._proc.stdin.flush()
                    deadline = monotonic() + budget_ms / 1000 + 5
                    while True:
                        left = deadline - monotonic()
                        if left <= 0:
                            raise TimeoutError("search timed out")
                        try:
                            line = self.lines.get(timeout=left)
                        except Empty as error:
                            raise TimeoutError("search timed out") from error
                        if line is None:
                            raise RuntimeError("engine closed")
                        reply = json.loads(line)
                        if reply.get("id") == self._next_id:
                            break
                    if reply.get("error"):
                        raise RuntimeError(reply["error"])
                    return reply.get("result")
                except Exception as error:
                    self.last_error = str(error)[:600]
                    self._failures += 1
                    self.close()
                    return None

    return Engine


def get_translator(checkout):
    """Reuse the benchmark's live-compatible observation encoder."""
    sys.path.insert(0, str(checkout))
    try:
        from sim import translate
    finally:
        sys.path.pop(0)
        for name in tuple(sys.modules):
            if name == "agent" or name.startswith("agent."):
                del sys.modules[name]
    return translate


class Trace:
    def __init__(self):
        self.metrics = Counter()
        self.phase = None
        self.error = None

    def write(self, kind, **fields):
        if kind == "error":
            self.error = fields.get("traceback", fields["error"])
        if kind == "sim":
            if fields["ok"]:
                self.metrics[f"{self.phase}_search"] += 1
                self.metrics[f"{self.phase}_rollouts"] += fields.get("rollouts") or 0
                self.metrics[f"{self.phase}_search_ms"] += fields.get("ms") or 0
            else:
                self.metrics[f"{self.phase}_fallback:{fields.get('error')}"] += 1


def run_worker(translator_checkout):
    protocol, sys.stdout = sys.stdout, sys.stderr
    sys.path.insert(0, str(Path.cwd()))
    from altruagent import DecisionContext
    from common import trace as upstream_trace
    from pokemon import simsearch
    from pokemon.agent import PokemonAgent

    translator = get_translator(translator_checkout)
    trace = Trace()
    upstream_trace.write = trace.write
    if os.name == "nt":
        simsearch.Engine = windows_engine(simsearch.Engine)
    if not simsearch.available():
        raise RuntimeError("Quicksilver's pinned simulation dependency is unavailable")

    with TemporaryDirectory(prefix="quicksilver-") as scratch:
        pool = Path(scratch) / "pool.json"
        shutil.copyfile(Path("pokemon") / "pool.json", pool)
        agent = PokemonAgent(pool_path=pool)
        try:
            for line in sys.stdin:
                try:
                    message = json.loads(line)
                    phase = message["op"]
                    trace.phase = phase
                    trace.metrics.clear()
                    if phase == "setup":
                        seat, first = message["seat"], message["first"]
                        foe = "p1" if seat == "p2" else "p2"
                        other_first = "p1" if first == "p2" else "p2"
                        draft_order = [first if pick in message["first_drafter_picks"] else other_first for pick in range(12)]
                        position = int(seat == "p2")
                        session = "local-public-draft"
                        context = DecisionContext(session, None, translator.GAME_TYPE, seat, position)
                        reply = {"ready": True}
                    elif phase == "start":
                        rosters = message["rosters"]
                        public_teams = json.dumps([rosters["p1"], rosters["p2"]], sort_keys=True)
                        session = f"local-public-{sha256(public_teams.encode()).hexdigest()[:16]}"
                        context = DecisionContext(session, None, translator.GAME_TYPE, seat, position)
                        version = 0
                        reply = {"ready": True}
                    else:
                        clock = message["clock"]
                        if phase == "draft":
                            obs = message["observation"]
                            rosters = {seat: obs["picked"], foe: obs["opponent_picked"]}
                            taken = {owner: iter(roster) for owner, roster in rosters.items()}
                            order = draft_order[:obs["pick"] - 1]
                            picks = [dict(next(taken[owner]), pick_number=n, seat_id=owner)
                                     for n, owner in enumerate(order, 1)]
                            version = len(picks)
                            cards = obs["cards"]
                            observation = {
                                "phase": phase, "battle_format": translator.BATTLE_FORMAT, "clock": clock,
                                "available_cards": cards, "available_card_ids": [card["card_id"] for card in cards],
                                "rosters": rosters, "picks": picks, "pick_number": obs["pick"],
                                "picks_remaining": 12 - version, "first_drafter": first, "current_seat": seat,
                            }
                            legal = [translator._action(session, f"draft_pick:{action['card_id']}",
                                                        f"Draft {action['card_id']}", action, version)
                                     for action in message["legal"]]
                        elif phase == "team_preview":
                            observation = {"phase": phase, "your_roster": rosters[seat], "opponent_roster": rosters[foe],
                                           "battle_format": translator.BATTLE_FORMAT, "clock": clock,
                                           "waiting_for_action": True}
                            action = {"type": "select_lineup", "roster": [card["species"] for card in rosters[seat]],
                                      "bring_count": 4, "lead_count": 2, "instructions": translator.LINEUP_INSTRUCTIONS}
                            legal = [translator._action(session, phase, translator.LINEUP_LABEL, action, version)]
                        if phase == "moving":
                            view = message["view"]
                            snapshot = {"battle_tag": session, "sides": {seat: {
                                "request": view["request"], "preview_request": view["preview_request"],
                            }, foe: {}}, "logs": {seat: view["protocol_log"]}}
                            version += 1
                            state = translator.battle_state(session, snapshot, seat, version, clock=clock)
                            observation = state.raw["observation"]
                            observation["can_tera"] = [False, False]
                            template = state.legal_actions[0].input["action"]
                            for slot, actual in zip(template["slots"], message["slots"], strict=True):
                                moves = {option["move_id"]: option for option in slot["options"] if option["type"] == "move"}
                                options = []
                                for option in actual["options"]:
                                    move = moves.get(option.get("move_id"), {})
                                    item = move | option
                                    if "target_options" in move:
                                        item["target_options"] = [
                                            target for target in move["target_options"] if target["target"] in option["targets"]
                                        ]
                                    options.append(item)
                                slot["options"] = options
                        else:
                            state = translator._new_state(session, phase, observation, legal, version,
                                                          actor=seat, position=position)
                        action = agent.choose_action(state, context)
                        if trace.error or agent._sim.last_error:
                            raise RuntimeError(trace.error or agent._sim.last_error)
                        if hasattr(action, "input"):
                            action = action.input["action"]
                        if phase == "moving":
                            for slot in ("slot_0", "slot_1"):
                                if action[slot]["type"] == "move":
                                    action[slot].setdefault("target", 0)
                        reply = {"action": action, "metrics": dict(trace.metrics)}
                except Exception:
                    reply = {"error": traceback.format_exc()}
                print(json.dumps(reply), file=protocol, flush=True)
        finally:
            agent._sim.close()


if __name__ == "__main__":
    run_worker(Path(sys.argv[1]))
