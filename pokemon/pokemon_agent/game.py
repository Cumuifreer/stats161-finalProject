"""Shared player-view decisions and legal joint actions."""

from dataclasses import dataclass
from itertools import combinations, permutations, product
import re

GAME_TYPE = "pokemon_vgc_doubles_draft"
DRAFT = "draft"
PREVIEW = "team_preview"
BATTLE = "moving"
CARD_FIELDS = ("card_id", "species", "item", "ability", "nature", "level", "evs", "ivs", "moves")
DRAFT_SIZE = 18
DECISION_LIMIT = {DRAFT: 15, PREVIEW: 85, BATTLE: 50}
BATTLE_BANK = 420


def ident(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def card_set(card: dict) -> dict:
    return {key: card[key] for key in CARD_FIELDS if key in card}


def pokemon_view(pokemon: dict | None) -> dict | None:
    if not pokemon:
        return None
    view = {}
    for key in ("species", "status", "item", "ability"):
        value = pokemon.get(key)
        if isinstance(value, str) and value and not value.startswith("unknown"):
            view[key] = ident(value)
    for key in ("types", "moves"):
        values = pokemon.get(key, ())
        if isinstance(values, (list, dict)):
            view[key] = [ident(value) for value in values if isinstance(value, str)]
    for key in ("boosts", "fainted"):
        if key in pokemon:
            view[key] = pokemon[key]
    if pokemon.get("current_hp_fraction") is not None:
        view["hp"] = pokemon["current_hp_fraction"]
    elif pokemon.get("max_hp"):
        view["hp"] = pokemon["current_hp"] / pokemon["max_hp"]
    return view


def _conditions(value: dict | list | str | None) -> list[str]:
    if isinstance(value, str):
        return [ident(value)] if value else []
    return sorted(ident(key) for key in (value or ()))


def _observation(phase: str, raw: dict, player: str) -> dict:
    if phase == DRAFT:
        rosters = raw.get("rosters") or {}
        return {
            "cards": [card_set(card) for card in raw.get("available_cards", ())],
            "picked": [card_set(card) for card in rosters.get(player, ())],
            "opponent_picked": [
                card_set(card) for seat, cards in rosters.items() if seat != player for card in cards
            ],
            "pick": raw.get("pick_number"),
        }
    if phase == PREVIEW:
        return {
            "team": [pokemon_view(mon) for mon in raw.get("your_roster", ())],
            "opponent_team": [pokemon_view(mon) for mon in raw.get("opponent_roster", ())],
        }
    return {
        "turn": raw.get("turn"),
        "active": [pokemon_view(mon) for mon in raw.get("active_pokemon") or (None, None)],
        "opponents": [
            pokemon_view(mon) for mon in raw.get("opponent_active_pokemon") or (None, None)
        ],
        "weather": _conditions(raw.get("weather")),
        "fields": _conditions(raw.get("fields")),
        "side_conditions": _conditions(raw.get("side_conditions")),
        "opponent_side_conditions": _conditions(raw.get("opponent_side_conditions")),
    }


def _slot_choices(slot: dict) -> list[dict]:
    choices = []
    for option in slot["options"]:
        match option["type"]:
            case "move":
                targets = option.get("targets") or [0]
                choices.extend(
                    {"type": "move", "move_id": option["move_id"], "target": target}
                    for target in targets
                )
            case "switch":
                choices.append({"type": "switch", "species": option["species"]})
            case "pass":
                choices.append({"type": "pass"})
            case kind:
                raise ValueError(f"Unknown Pokémon slot action: {kind!r}")
    return choices


def _joint_choices(slots: list[dict]) -> tuple[dict, ...]:
    by_slot = {slot["slot"]: slot for slot in slots}
    if set(by_slot) != {0, 1} or len(slots) != 2:
        raise ValueError("Expected a template for both doubles slots")
    choices = []
    for left, right in product(_slot_choices(by_slot[0]), _slot_choices(by_slot[1])):
        if left["type"] == right["type"]:
            if left["type"] == "pass":
                continue
            if left["type"] == "switch" and ident(left["species"]) == ident(right["species"]):
                continue
        choices.append({"type": "doubles_turn", "slot_0": left, "slot_1": right})
    return tuple(choices)


def _move_info(slot: dict, raw_moves: list[dict] | None) -> dict:
    by_id = {move["id"]: move for move in (raw_moves or ()) if "id" in move}
    details = {}
    for option in slot["options"]:
        if option["type"] == "move":
            move_id = option["move_id"]
            data = by_id.get(move_id, {}) | option
            details[move_id] = move_metadata({
                key: data[key]
                for key in ("base_power", "accuracy", "priority", "category", "target")
                if key in data
            })
    return details


def _reserves(raw_team: dict | list, slots: list[dict]) -> dict:
    eligible = {ident(option["species"]) for slot in slots for option in slot["options"]
                if option["type"] == "switch"}
    reserves = {}
    for mon in raw_team.values() if isinstance(raw_team, dict) else raw_team:
        view = pokemon_view(mon) or {}
        if (species := view.get("species")) in eligible:
            reserves[species] = {key: view[key] for key in ("hp", "status", "moves") if key in view}
    return dict(sorted(reserves.items()))


def move_metadata(data: dict) -> dict:
    data = dict(data)
    accuracy = data.get("accuracy")
    if isinstance(accuracy, (int, float)):
        data["accuracy"] = accuracy / 100 if accuracy > 1 else float(accuracy)
    for key in ("category", "target"):
        if isinstance(value := data.get(key), str):
            data[key] = ident(value)
    return data


@dataclass(frozen=True)
class Decision:
    phase: str
    observation: dict
    actions: tuple[dict, ...]
    move_info: tuple[dict, ...] = ()
    seconds_left: float | None = None
    bank_seconds_left: float | None = None

    def validate(self, action: dict) -> dict:
        for legal in self.actions:
            if action == legal:
                return legal
        raise ValueError("Policy returned an action outside the current legal choices")

    def snapshot(self) -> dict:
        return {
            "phase": self.phase,
            "observation": self.observation,
            "actions": self.actions,
            "move_info": self.move_info,
            "seconds_left": self.seconds_left,
            "bank_seconds_left": self.bank_seconds_left,
        }

    @classmethod
    def from_snapshot(cls, raw: dict) -> "Decision":
        return cls(
            raw["phase"], raw["observation"], tuple(raw["actions"]),
            tuple({move: move_metadata(data) for move, data in slot.items()}
                  for slot in raw.get("move_info", ())),
            raw.get("seconds_left"), raw.get("bank_seconds_left"),
        )


def make_decision(
    phase: str, raw: dict, templates: list[dict], player: str,
    seconds_left: float | None = None, bank_seconds_left: float | None = None,
) -> Decision:
    observation = _observation(phase, raw, player)
    move_info = ()
    if phase == DRAFT:
        actions = tuple({"type": "draft_pick", "card_id": t["card_id"]} for t in templates)
    elif phase == PREVIEW and len(templates) == 1:
        roster = list(dict.fromkeys(templates[0]["roster"]))
        actions = tuple(
            {"type": "select_lineup", "bring": list(bring), "leads": list(leads)}
            for bring in combinations(roster, 4)
            for leads in permutations(bring, 2)
        )
    elif phase == BATTLE and len(templates) == 1:
        slots = sorted(templates[0]["slots"], key=lambda row: row["slot"])
        actions = _joint_choices(slots)
        if isinstance(raw_team := raw.get("team"), (dict, list)):
            observation["reserves"] = _reserves(raw_team, slots)
        available = raw.get("available_moves") or [[], []]
        move_info = tuple(
            _move_info(slot, raw_moves) for slot, raw_moves in zip(slots, available, strict=True)
        )
    else:
        raise ValueError(f"Unsupported phase or template count: {phase!r}, {len(templates)}")
    if not actions:
        raise ValueError(f"No legal Pokémon actions in phase {phase!r}")
    return Decision(phase, observation, actions, move_info, seconds_left, bank_seconds_left)
