"""Use an independent continuation policy to nominate early legal doubles moves."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

if __package__:
    from ...game import BATTLE_BANK
    from ...search import _key
    from ...wide85 import WideSearch85


class RolloutScout:
    def __init__(self, checkout, translator, indexed, budget_ms=1600):
        self.checkout = checkout.resolve()
        self.translator = translator.resolve()
        self.indexed = indexed
        self.budget_ms = budget_ms

    def __enter__(self):
        node = shutil.which("node")
        if node is None:
            raise RuntimeError("The continuation scout needs Node")
        env = {key: os.environ[key] for key in ("SystemRoot", "TEMP", "TMP") if key in os.environ}
        env.update(PATH=str(Path(node).parent), POKEMON_JEV="0")
        self.process = subprocess.Popen(
            [sys.executable, "-IB", "-X", "utf8", str(Path(__file__).resolve()), str(self.translator)],
            cwd=self.checkout, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, encoding="utf-8",
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

    def proposals(self, decision, seat, public_rosters, evidence):
        observed = evidence["own_proxy"]
        cards = [self.indexed[name] for roster in public_rosters.values() for name in roster]
        request = {"seat": seat, "view": observed["pivot"], "log": evidence["protocol_log"],
                   "template": observed["templates"][0], "cards": cards, "budget_ms": self.budget_ms}
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("The continuation scout stopped")
        reply = json.loads(line)
        if error := reply.get("error"):
            raise RuntimeError(error)
        legal = {_key(action): action for action in decision.actions}
        moves = [legal[key] for answer in reply["plans"] if (key := _key(answer)) in legal]
        return moves[:4], reply["rollouts"]


if __package__:
    class WideScout80(WideSearch85):
        def __init__(self, *args, rollout_scout, **settings):
            super().__init__(*args, **settings)
            self.rollout_scout = rollout_scout

        def _proposals(self, decision, seat, rosters, evidence):
            base = super()._proposals(decision, seat, rosters, evidence)
            turn = decision.observation["turn"]
            if turn > 3 or len(decision.actions) < 18 or (decision.bank_seconds_left or BATTLE_BANK) < 60:
                return base
            try:
                extra, rollouts = self.rollout_scout.proposals(decision, seat, rosters, evidence)
            except (KeyError, RuntimeError, ValueError, OSError):
                self.counts["continuation_scout_errors"] += 1
                return base
            self.counts["continuation_scout_calls"] += 1
            self.counts["continuation_scout_rollouts"] += rollouts
            self.counts["continuation_scout_unique"] += sum(proposal not in base for proposal in extra)
            self.latest_scout = extra
            return base + extra

        def _plan(self, decision, seat, evidence, ranked, coarse):
            self.latest_scout = []
            result = super()._plan(decision, seat, evidence, ranked, coarse)
            self.counts["continuation_scout_chosen"] += result in self.latest_scout
            return result


def serve(translator_checkout):
    protocol, sys.stdout = sys.stdout, sys.stderr
    sys.path[:0] = [str(Path.cwd()), str(Path(__file__).resolve().parents[3])]
    import altruagent
    from pokemon import simsearch
    from pokemon_agent.quicksilver import get_translator, windows_engine

    translator = get_translator(translator_checkout)
    sim = (windows_engine(simsearch.Engine) if os.name == "nt" else simsearch.Engine)()
    try:
        for line in sys.stdin:
            try:
                reply = evaluate(translator, simsearch, sim, json.loads(line))
            except Exception as error:
                reply = {"error": f"{type(error).__name__}: {error}"}
            print(json.dumps(reply), file=protocol, flush=True)
    finally:
        sim.close()


def evaluate(translator, simsearch, sim, request):
    from hashlib import sha256

    seat = request["seat"]
    other = "p2" if seat == "p1" else "p1"
    public_log = request["log"]
    identity = sha256(json.dumps(public_log).encode()).hexdigest()[:16]
    stage = {"battle_tag": f"local-scout-{identity}", "sides": {seat: request["view"], other: {}},
             "logs": {seat: public_log}}
    state = translator.battle_state(stage["battle_tag"], stage, seat, 0)
    synthetic = simsearch.build_state(state.raw["observation"], request["cards"])
    if synthetic is None:
        return {"plans": [], "rollouts": 0}
    output = sim.search(synthetic, request["budget_ms"])
    if output is None and sim.last_error:
        return {"error": sim.last_error}
    if not output:
        return {"plans": [], "rollouts": 0}
    plans = []
    for best in output["ranked"][:6]:
        plan = simsearch.to_action(best, request["template"])
        if plan:
            for slot in ("slot_0", "slot_1"):
                if plan[slot]["type"] == "move":
                    plan[slot].setdefault("target", 0)
            plans.append(plan)
    return {"plans": plans, "rollouts": output["rollouts"]}


if __name__ == "__main__":
    serve(Path(sys.argv[1]))
