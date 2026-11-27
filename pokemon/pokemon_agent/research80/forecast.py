"""Extra deployment scenarios from an independent public roll-out policy."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


class MonteCarloLineups:
    def __init__(self, checkout, indexed, budget):
        self.checkout = checkout.resolve()
        self.indexed = indexed
        self.budget = budget

    def __enter__(self):
        node = shutil.which("node")
        if node is None:
            raise RuntimeError("Node is required for the deployment forecaster")
        env = {key: os.environ[key] for key in ("SystemRoot", "TEMP", "TMP") if key in os.environ}
        env.update(PATH=str(Path(node).parent), POKEMON_JEV="0")
        self.process = subprocess.Popen(
            [sys.executable, "-IB", "-X", "utf8", str(Path(__file__).resolve())],
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

    def proposals(self, decision, top=2, budget=None):
        from ..game import ident

        own, foe = ([self.indexed[mon["species"]] for mon in decision.observation[key]]
                    for key in ("team", "opponent_team"))
        result = self._call({"own": own, "foe": foe, "budget_ms": int((budget or self.budget) * 1000)})
        legal = {(frozenset(map(ident, choice["bring"])), tuple(map(ident, choice["leads"]))): choice
                 for choice in decision.actions}
        return [(legal[(frozenset(map(ident, action["bring"])), tuple(map(ident, action["leads"])))], score)
                for action, score in result["lineups"][:top]]

    def compatibility(self, reverse, pool, model_first):
        own, foe = ([self.indexed[mon["species"]] for mon in reverse.observation[key]]
                    for key in ("team", "opponent_team"))
        result = self._call({"own": own, "foe": foe, "pool": pool, "model_first": model_first})
        return result["hits"], result["eligible"]

    def _call(self, request):
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        row = self.process.stdout.readline()
        if not row:
            raise RuntimeError("The deployment search worker stopped")
        result = json.loads(row)
        if result.get("error"):
            raise RuntimeError(result["error"])
        return result


class EnsembleLeadPrior:
    """Mix predictions from two independent rollout styles and a learned lead prior."""

    def __init__(self, lead_prior, monte_carlo, propose_own=False):
        self.lead_prior = lead_prior
        self.monte_carlo = monte_carlo
        self.space = lead_prior.space
        self.own_budget = 6. if propose_own else 0.
        self.extra_seconds = lead_prior.extra_seconds + monte_carlo.budget + self.own_budget
        self.used = 0
        self.pool = []
        self.model_first = False
        self.agreement = None

    def observe(self, decision):
        from ..game import DRAFT

        if decision.phase != DRAFT or decision.observation["pick"] > 2:
            return
        observation = decision.observation
        self.pool = [*observation["cards"], *observation["opponent_picked"]]
        self.model_first = observation["pick"] == 2
        self.used = 0
        self.agreement = None

    def scenarios(self, reverse, nominated, ranked):
        from ..combination_preview import signature

        lead_prior, weights = self.lead_prior.scenarios(reverse, nominated, ranked)
        self.used = 0
        if not self.pool:
            return lead_prior, weights
        try:
            hits, eligible = self.monte_carlo.compatibility(reverse, self.pool, self.model_first)
            self.agreement = (hits, eligible)
            if eligible < 3 or hits / eligible < .75:
                return lead_prior, weights
            shots = self.monte_carlo.proposals(reverse, top=2)
        except (KeyError, ValueError, RuntimeError, OSError):
            return lead_prior, weights
        if not shots:
            return lead_prior, weights
        self.used = 1
        specialist_mass = .28 + .15 * hits / eligible
        actions = {signature(choice): choice for choice in lead_prior}
        mass = {signature(choice): (1. - specialist_mass) * weight for choice, weight in zip(lead_prior, weights, strict=True)}
        difference = (shots[0][1] - shots[1][1]) if len(shots) > 1 else 1.
        main_mass = max(.65, min(.9, .65 + 2.5 * difference))
        for (choice, _), weight in zip(shots, (main_mass, 1. - main_mass)):
            key = signature(choice)
            actions.setdefault(key, choice)
            mass[key] = mass.get(key, 0.) + specialist_mass * weight
        keys = sorted(mass, key=mass.__getitem__, reverse=True)[:8]
        total = sum(mass[key] for key in keys)
        return [actions[key] for key in keys], [mass[key] / total for key in keys]

    def own_shortlist(self, decision):
        if not self.used or not self.own_budget:
            return []
        try:
            return [choice for choice, _ in self.monte_carlo.proposals(decision, top=3, budget=self.own_budget)]
        except (KeyError, ValueError, RuntimeError, OSError):
            return []


def draft_agreement(draft, request):
    """Score agreement with one published picking style from information already revealed."""
    own, foe = request["own"], request["foe"]
    cards = {card["card_id"]: card for card in [*request["pool"], *own, *foe]}
    if len(cards) != 18:
        raise ValueError("Expected the eighteen published draft cards")
    rosters = {"ours": [], "theirs": []}
    future = {"ours": iter(own), "theirs": iter(foe)}
    model_first = request["model_first"]
    first = "ours" if model_first else "theirs"
    hits = eligible = 0
    for pick in range(1, 13):
        owner = "ours" if draft.pick_owner(pick, we_are_first=model_first) == "us" else "theirs"
        actual = next(future[owner])
        if owner == "ours":
            observation = {"phase": "draft", "pick_number": pick, "first_drafter": first,
                           "available_cards": list(cards.values()), "rosters": rosters}
            ranked = draft.top_cards(observation, agent_id="ours", k=1)
            if ranked:
                key = ranked[0][0]
                taken = {str(card.get("item", "")).casefold() for card in rosters["ours"]}
                if str(cards[key].get("item", "")).casefold() not in taken:
                    eligible += 1
                    hits += key == actual["card_id"]
        del cards[actual["card_id"]]
        rosters[owner].append(actual)
    return {"hits": hits, "eligible": eligible}


def serve():
    protocol, sys.stdout = sys.stdout, sys.stderr
    sys.path[:0] = [str(Path.cwd()), str(Path(__file__).resolve().parents[2])]
    from pokemon import draft, lineup, simsearch
    from pokemon_agent.quicksilver import windows_engine

    engine = (windows_engine(simsearch.Engine) if os.name == "nt" else simsearch.Engine)()
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if "pool" in request:
                    answer = draft_agreement(draft, request)
                    print(json.dumps(answer), file=protocol, flush=True)
                    continue
                own, foe = request["own"], request["foe"]
                players = {"your_roster": own, "opponent_roster": foe}
                cards = [{**card, "_owner": side} for side, roster in (("ours", own), ("theirs", foe)) for card in roster]
                menu = {"roster": [card["species"] for card in own]}
                picked = lineup.engine_lineups(menu, players, pool_cards=cards)
                if picked is None:
                    print('{"lineups":[]}', file=protocol, flush=True)
                    continue
                options, replies = picked
                search = simsearch.lineup_request(
                    [[{"species": name} for name in order] for _, order in options],
                    [([{"species": name} for name in order], weight) for order, weight in replies],
                    cards,
                )
                result = engine.lineups(search, request["budget_ms"]) if search else None
                if result and result["rollouts"]:
                    values = [
                        (option, .65 * value + .35 * worst)
                        for (option, _), value, worst in zip(options, result["values"], result["worst"], strict=True)
                    ]
                    answer = {"lineups": sorted(values, key=lambda item: item[1], reverse=True)}
                else:
                    answer = {"lineups": [], "error": engine.last_error}
            except Exception as error:
                answer = {"error": f"{type(error).__name__}: {error}"}
            print(json.dumps(answer), file=protocol, flush=True)
    finally:
        engine.close()


if __name__ == "__main__":
    serve()
