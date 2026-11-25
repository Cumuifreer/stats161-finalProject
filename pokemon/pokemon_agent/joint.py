"""Rerank the existing shortlist against plausible replies to a doubles turn."""

from itertools import product

from .game import BATTLE, PREVIEW, Decision, ident
from .tactics import SLOTS, SPREAD, _damage, _pressure, _speed, _types, load_dex, tactical_candidates

PROTECT = {"protect", "detect"}
PASS = {"type": "pass"}
POSITIONS = range(4)  # our two slots, then the opponent's two slots


def _move(move_id: str, target: int = 0) -> dict:
    return {"type": "move", "move_id": move_id, "target": target}


class JointPolicy:
    def __init__(self, profiles: dict):
        self.profiles = profiles
        self.opponent_roster = []
        self.revealed = set()
        self.protected_at = {}

    def observe(self, decision: Decision) -> None:
        if decision.phase == PREVIEW:
            self.opponent_roster = [mon["species"] for mon in decision.observation["opponent_team"]
                                    if mon and "species" in mon]
            self.revealed.clear()
            self.protected_at.clear()

    def __call__(self, decision: Decision) -> dict:
        self.observe(decision)
        action = self.rank(decision)[0][0]
        self.record(decision, action)
        return action

    def record(self, decision: Decision, action: dict) -> None:
        if decision.phase == BATTLE:
            for slot, mon in zip(SLOTS, decision.observation["active"], strict=True):
                if mon and mon.get("species") and action[slot].get("move_id") in PROTECT:
                    self.protected_at[mon["species"]] = decision.observation["turn"]

    def rank(self, decision: Decision, candidates: list[dict] | None = None) -> list[tuple[dict, dict]]:
        if candidates is None:
            candidates = tactical_candidates(decision)
        if decision.phase != BATTLE:
            return [(action, {}) for action in candidates]
        self.revealed.update(mon["species"] for mon in decision.observation["opponents"]
                             if mon and "species" in mon)
        if len(candidates) == 1:
            return [(candidates[0], {})]
        unseen = () if len(self.revealed) >= 4 else tuple(
            species for species in self.opponent_roster if species not in self.revealed
        )
        turn = decision.observation.get("turn")
        repeated = {species for species, previous in self.protected_at.items()
                    if isinstance(turn, int) and previous == turn - 1}
        forecast = TurnForecast(decision, self.profiles, unseen, repeated)
        return forecast.rank(candidates)


class TurnForecast:
    def __init__(self, decision: Decision, profiles: dict, unseen: tuple[str, ...], repeated: set[str]):
        self.dex = load_dex()
        self.profiles = profiles
        board = decision.observation
        self.weather = board.get("weather", ())
        self.conditions = (board.get("side_conditions", ()), board.get("opponent_side_conditions", ()))
        self.trick_room = "trickroom" in board.get("fields", ())
        self.opening = board.get("turn") == 1
        self.repeated = repeated
        self.mons = tuple(self._mon(mon) for mon in board["active"] + board["opponents"])
        self.reserve = {name: self._mon({"species": name, **summary})
                        for name, summary in board.get("reserves", {}).items()}
        self.unseen = {name: self._mon({"species": name}, fresh=True) for name in unseen}
        self.our_moves = decision.move_info

    def _mon(self, mon, *, fresh=False):
        if not mon or mon.get("fainted"):
            return {}
        profile = self.profiles.get(mon.get("species"), {})
        merged = profile | mon
        if "ability" not in merged and profile.get("card_ability"):
            merged["ability"] = profile["card_ability"]
        if fresh and profile.get("card_item"):
            merged["item"] = profile["card_item"]
        return merged

    def _info(self, position, move):
        data = self.dex["moves"].get(move, {})
        return data | self.our_moves[position].get(move, {}) if position < 2 else data

    def _targets(self, position, choice, hp):
        move = choice["move_id"]
        kind = self._info(position, move).get("target")
        enemy = 2 if position < 2 else 0
        if kind in SPREAD or kind == "randomnormal":
            slots = [enemy, enemy + 1]
            if kind == "alladjacent":
                slots.append(position ^ 1)
            return [slot for slot in slots if hp[slot] > 0]
        target = choice.get("target", 0)
        if target > 0:
            slot = enemy + target - 1
            return [slot if hp[slot] > 0 else (slot ^ 1)] if hp[enemy] > 0 or hp[enemy + 1] > 0 else []
        if target < 0:
            slot = (position // 2) * 2 - target - 1
            return [slot] if hp[slot] > 0 else []
        return []

    def _switched_mon(self, position, choice):
        species = ident(choice["species"])
        bench = self.reserve if position < 2 else self.unseen
        return bench.get(species) or self._mon({"species": species})

    def _hit(self, move, info, attacker, defender, spread, side):
        accuracy = info.get("accuracy")
        return _damage(move, info, attacker, defender, spread, self.dex, self.weather, self.conditions[side]) * (
            accuracy if isinstance(accuracy, (int, float)) else 1
        )

    def _replies(self, baseline):
        hp = [mon.get("hp", 1) if mon else 0 for mon in self.mons]
        ranked = []
        for position in (2, 3):
            mon = self.mons[position]
            attacks = []
            for move in mon.get("card_moves", mon.get("moves", ())):
                info = self._info(position, move)
                if not info.get("base_power") and move != "ruination":
                    continue
                if move == "fakeout" and not self.opening:
                    continue
                targets = (0,) if info.get("target") in SPREAD | {"randomnormal"} else (1, 2)
                for target in targets:
                    choice = _move(move, target)
                    victims = self._targets(position, choice, hp)
                    if target and hp[target - 1] == 0:
                        continue
                    hits = [self._hit(move, info, mon, victim, len(victims) > 1 and info.get("target") in SPREAD, i // 2)
                            if i in victims else 0 for i, victim in enumerate(self.mons)]
                    value = sum(min(hp[i], hits[i]) + .3 * (hits[i] >= hp[i] > 0) for i in (0, 1))
                    attacks.append((value - hits[position ^ 1], choice, hits))
            attacks.sort(key=lambda attack: attack[0], reverse=True)
            ranked.append(attacks[:3] or [(0, PASS, [0] * 4)])

        def value(pair):
            damage = [sum(attack[2][i] for attack in pair) for i in POSITIONS]
            return sum(min(hp[i], damage[i]) + .3 * (damage[i] >= hp[i] > 0) for i in (0, 1)) - sum(damage[2:])

        attack = [row[1] for row in max(product(*ranked), key=value)]
        replies = [("attack", .72, attack)]
        threatened = sorted((i for i in (2, 3) if hp[i] > 0), key=lambda i: sum(
            self._hit(choice["move_id"], self._info(slot, choice["move_id"]), self.mons[slot], self.mons[i],
                      len(self._targets(slot, choice, hp)) > 1, 1)
            for slot, choice in enumerate(baseline) if choice["type"] == "move" and self.mons[slot]
            and i in self._targets(slot, choice, hp)
        ), reverse=True)
        for position in threatened:
            if move := next((move for move in self.mons[position].get("card_moves", self.mons[position].get("moves", ()))
                             if move in PROTECT), None):
                protect = attack.copy()
                protect[position - 2] = _move(move)
                replies.append(("protect", .18, protect))
                break
        if self.unseen and threatened:
            position = threatened[0]
            attackers = [mon for mon in self.mons[:2] if mon]
            species, mon = min(self.unseen.items(), key=lambda entry: _pressure(entry[1], attackers, self.dex))
            if _pressure(mon, attackers, self.dex) < _pressure(self.mons[position], attackers, self.dex):
                switch = attack.copy()
                switch[position - 2] = {"type": "switch", "species": species}
                replies.append(("switch", .10, switch))
        return replies

    def _evaluate(self, choices):
        mons = list(self.mons)
        initial = [mon.get("hp", 1) if mon else 0 for mon in mons]
        for i, choice in enumerate(choices):
            if choice["type"] == "switch":
                mons[i] = self._switched_mon(i, choice)
                initial[i] = mons[i].get("hp", 1)
        hp = initial.copy()
        tailwind = ["tailwind" in conditions for conditions in self.conditions]
        room = self.trick_room
        shielded, stopped = {}, set()
        order = []
        effects = [0., 0.]
        pending = [i for i in POSITIONS if hp[i] > 0 and choices[i]["type"] == "move"]

        def speed(i):
            return _speed(mons[i], ("tailwind",) if tailwind[i // 2] else (), self.dex)

        def initiative(i):
            move = choices[i]["move_id"]
            return (self._info(i, move).get("priority", 0), speed(i) * (-1 if room else 1), i >= 2)

        while pending:
            i = max(pending, key=initiative)
            pending.remove(i)
            if hp[i] <= 0 or i in stopped:
                continue
            choice = choices[i]
            move = choice["move_id"]
            info = self._info(i, move)
            side = i // 2
            order.append(i)
            if move in PROTECT:
                shielded[i] = 1 / 3 if i < 2 and mons[i].get("species") in self.repeated else 1
                continue
            if move == "tailwind" and not tailwind[side]:
                tailwind[side] = True
                continue
            if move == "trickroom":
                room = not room
                continue
            victims = self._targets(i, choice, hp)
            for j in victims:
                block = 0 if mons[i].get("ability") == "unseenfist" and info.get("contact") else shielded.get(j, 0)
                if block == 1:
                    continue
                if move == "spore" and not mons[j].get("status") and "grass" not in _types(mons[j], self.dex) \
                        and mons[j].get("item") != "safetygoggles":
                    stopped.add(j)
                    effects[side] += .14
                if not info.get("base_power") and move != "ruination":
                    continue
                if move == "fakeout" and not self.opening:
                    continue
                amount = self._hit(move, info, mons[i] | {"hp": hp[i]}, mons[j] | {"hp": hp[j]},
                                   len(victims) > 1 and info.get("target") in SPREAD, j // 2)
                amount *= 1 - block
                if info.get("target") == "randomnormal":
                    amount /= len(victims)
                hp[j] = max(0., hp[j] - amount)
                if move == "fakeout" and amount > 0 and mons[j].get("ability") != "innerfocus" \
                        and mons[j].get("item") != "covertcloak":
                    stopped.add(j)

        tempo = 0.
        if room != self.trick_room or tailwind != ["tailwind" in c for c in self.conditions]:
            comparisons = [(i, j) for i in (0, 1) for j in (2, 3) if hp[i] > 0 and hp[j] > 0]
            if comparisons:
                gained = sum((speed(i) > speed(j)) if not room else (speed(i) < speed(j)) for i, j in comparisons)
                direction = -1 if self.trick_room else 1
                earlier = sum(direction * (_speed(mons[i], self.conditions[0], self.dex)
                                           - _speed(mons[j], self.conditions[1], self.dex)) > 0 for i, j in comparisons)
                tempo = .30 * (gained - earlier) / len(comparisons)
        loss = [initial[i] - hp[i] + .40 * (initial[i] > 0 >= hp[i]) for i in POSITIONS]
        score = sum(loss[2:]) - sum(loss[:2]) + tempo + effects[0] - effects[1]
        our_faints = [i for i in (0, 1) if initial[i] > 0 >= hp[i]]
        return score, {
            "move_order": order,
            "our_ko": our_faints,
            "our_ko_before_move": [i for i in our_faints if choices[i]["type"] == "move" and i not in order],
            "foe_ko": [i - 2 for i in (2, 3) if initial[i] > 0 >= hp[i]],
        }

    def rank(self, candidates):
        replies = self._replies([candidates[0][slot] for slot in SLOTS])
        result = []
        for index, action in enumerate(candidates):
            outcomes = []
            for name, weight, response in replies:
                score, details = self._evaluate([action[slot] for slot in SLOTS] + response)
                outcomes.append((score, weight, name, response, details))
            mean = sum(row[0] * row[1] for row in outcomes) / sum(row[1] for row in outcomes)
            worst, _, reply, response, details = min(outcomes, key=lambda outcome: outcome[0])
            score = .85 * mean + .15 * worst - .035 * index
            result.append((action, {"score": score, "worst_reply": reply, "reply_slots": response, **details}))
        return sorted(result, key=lambda row: row[1]["score"], reverse=True)
