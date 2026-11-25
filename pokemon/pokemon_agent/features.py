"""Encode a battle decision and each of its legal joint actions."""

from dataclasses import dataclass
from functools import cache

from .game import BATTLE, Decision, ident
from .joint import BattleHistory
from .policy import SUPPORT_MOVES
from .tactics import SLOTS, SPREAD, _damage, _pressure, _speed, _types, load_dex, tactical_actions, tactical_candidates

FEATURE_VERSION = 1
HISTORY_FEATURE_VERSION = 2
TYPES = (
    "normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison", "ground", "flying",
    "psychic", "bug", "rock", "ghost", "dragon", "dark", "steel", "fairy",
)
STATS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
STATUSES = ("brn", "frz", "par", "psn", "slp", "tox")
WEATHER = ("sunnyday", "raindance", "sandstorm", "snow", "hail")
FIELDS = ("trickroom", "electricterrain", "grassyterrain", "mistyterrain", "psychicterrain")
SIDE_CONDITIONS = ("tailwind", "reflect", "lightscreen", "auroraveil", "stealthrock", "spikes", "toxicspikes", "stickyweb")
MOVES = tuple(sorted(SUPPORT_MOVES | {
    "protect", "detect", "wideguard", "thunderwave", "helpinghand", "taunt", "partingshot", "uturn", "voltswitch",
    "pollenpuff", "knockoff", "ruination", "eruption", "suckerpunch", "aquajet", "extremespeed", "shadowsneak",
}))


def _flags(options, selected) -> tuple[float, ...]:
    return tuple(float(option in selected) for option in options)


def _hp(mon: dict) -> float:
    return mon.get("hp", 1.) if mon and not mon.get("fainted") else 0.


@dataclass(frozen=True)
class BattleFeatures:
    state: tuple[float, ...]
    actions: tuple[tuple[float, ...], ...]
    action_indices: tuple[int, ...]


class BattleEncoder:
    def __init__(self, profiles: dict | None = None):
        self.dex = load_dex()
        self.profiles = profiles
        self.species = tuple(sorted(profiles or {}))

    def encode(self, decision: Decision, *, history: BattleHistory | None = None, shortlist_only: bool = False) -> BattleFeatures:
        if decision.phase != BATTLE:
            raise ValueError("Battle features require a moving Decision")
        board = decision.observation
        active = tuple(self._profile(mon or {}) for mon in board["active"] + board["opponents"])
        hp = tuple(_hp(mon) for mon in active)
        weather, fields = board.get("weather", ()), board.get("fields", ())
        conditions = board.get("side_conditions", ()), board.get("opponent_side_conditions", ())
        reserves = board.get("reserves", {})
        speed = tuple(_speed(mon, conditions[i // 2], self.dex) if mon else 0. for i, mon in enumerate(active))
        direction = -1 if "trickroom" in fields else 1
        state = (
            min(board.get("turn") or 0, 50) / 50, float(board.get("turn") == 1), len(reserves) / 2,
            sum(_hp(mon) for mon in reserves.values()) / 2,
            *_flags(WEATHER, weather), *_flags(FIELDS, fields),
            *_flags(SIDE_CONDITIONS, conditions[0]), *_flags(SIDE_CONDITIONS, conditions[1]),
        )
        for i, mon in enumerate(active):
            state += (
                float(bool(mon)), hp[i], speed[i] / 250, *_flags(TYPES, _types(mon, self.dex)),
                *_flags(STATUSES, (mon.get("status"),)),
                *(mon.get("boosts", {}).get(stat, 0) / 6 for stat in STATS),
            )
        if self.profiles is not None:
            if history is None:
                raise ValueError("BC v2 requires player-view battle history")
            repeated = history.repeated(board.get("turn"))
            state += (
                *_flags(self.species, history.opponent_roster), *_flags(self.species, history.revealed),
                *(float(mon.get("species") in repeated) for mon in active[:2]),
            )

        @cache
        def slot_features(index: int, kind: str, move: str, target: int, species: str):
            info = decision.move_info[index].get(move, {}) if index < len(decision.move_info) else {}
            data = self.dex["moves"].get(move, {}) | info
            accuracy = data.get("accuracy", 1.)
            accuracy = accuracy if isinstance(accuracy, (int, float)) else 1.
            priority = data.get("priority") or 0
            target_type = data.get("target")
            victims = []
            if target:
                victims = [target + 1 if target > 0 else -target - 1]
            elif target_type in SPREAD or target_type == "randomnormal":
                victims = [2, 3] + ([1 - index] if target_type == "alladjacent" else [])
            victims = [pos for pos in victims if hp[pos] > 0]
            damage = [0.] * 4
            if kind == "move" and (data.get("base_power") or move == "ruination"):
                for pos in victims:
                    amount = _damage(move, data, active[index], active[pos],
                                     target_type in SPREAD and len(victims) > 1, self.dex, weather, conditions[pos // 2])
                    damage[pos] = min(2., amount * accuracy / (len(victims) if target_type == "randomnormal" else 1))

            incoming = self._profile({"species": species, **reserves.get(species, {})}) if kind == "switch" else {}
            incoming_speed = _speed(incoming, conditions[0], self.dex) if incoming else 0.
            values = (
                *_flags(("move", "switch", "pass"), (kind,)), *_flags(range(-2, 3), (target,)),
                *_flags(MOVES, (move,)), *_flags(TYPES, (data.get("type"),)),
                (data.get("base_power") or 0) / 150, accuracy if kind == "move" else 0., priority / 5,
                *_flags(("physical", "special", "status"), (data.get("category"),)),
                *(float(pos in victims and (priority > 0 or (priority == 0 and direction * (speed[index] - speed[pos]) > 0)))
                  for pos in (2, 3)),
                damage[1 - index], damage[2], damage[3],
                _hp(incoming), incoming_speed / 250, *_flags(TYPES, _types(incoming, self.dex)),
                _pressure(incoming, active[2:], self.dex) / 8 if incoming else 0.,
            )
            return values, damage

        ranked = tactical_actions(decision)
        # Tactical returns references to the same action dictionaries held by the Decision.
        ranks = {id(action): i for i, action in enumerate(ranked)}
        shortlist = {id(action) for action in tactical_candidates(decision, ranked=ranked)}
        actions = []
        indices = []
        for index, action in enumerate(decision.actions):
            if shortlist_only and id(action) not in shortlist:
                continue
            indices.append(index)
            choices = [action[slot] for slot in SLOTS]
            encoded = [slot_features(i, choice["type"], choice.get("move_id", ""), choice.get("target", 0),
                                     ident(choice.get("species", ""))) for i, choice in enumerate(choices)]
            (left, left_damage), (right, right_damage) = encoded
            rank = ranks[id(action)]
            feature = (float(id(action) in shortlist), 1. / (1 + rank), rank / max(1, len(ranked) - 1), *left, *right)
            for pos in (2, 3):
                total = left_damage[pos] + right_damage[pos]
                feature += (min(hp[pos], total), float(total >= hp[pos] > 0),
                            float(left_damage[pos] > 0 and right_damage[pos] > 0))
            actions.append(feature)
        return BattleFeatures(state, tuple(actions), tuple(indices))

    def _profile(self, mon: dict) -> dict:
        if self.profiles is None or not mon:
            return mon
        profile = self.profiles.get(mon.get("species"), {})
        merged = profile | mon
        if "ability" not in merged and profile.get("card_ability"):
            merged["ability"] = profile["card_ability"]
        return merged
