"""Rank legal doubles actions using approximate damage and public board information."""

from functools import cache
import json
from pathlib import Path
import subprocess

from .game import BATTLE, PREVIEW, Decision, ident
from .policy import ALLY_MOVES, ranked_actions

SLOTS = ("slot_0", "slot_1")
SPEED_CONTROL = {"tailwind", "trickroom", "icywind", "thunderwave"}
SPREAD = {"alladjacentfoes", "alladjacent"}


@cache
def load_dex() -> dict:
    bridge = Path(__file__).with_name("showdown.cjs")
    try:
        return json.loads(subprocess.check_output(["node", str(bridge), "--dex"], timeout=5))
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("Could not load the Showdown Dex. Verify Node.js and npm install in pokemon/.") from error


def load_card_profiles(cards: list[dict]) -> dict:
    unique = list({json.dumps(card, sort_keys=True): card for card in cards}.values())
    bridge = Path(__file__).with_name("showdown.cjs")
    try:
        stats = json.loads(subprocess.check_output(
            ["node", str(bridge), "--card-stats"], input=json.dumps(unique), text=True, timeout=5,
        ))
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("Could not calculate fixed-card stats with Showdown") from error
    profiles = {}
    for card, profile in zip(unique, stats, strict=True):
        species = ident(card["species"])
        if profiles.setdefault(species, profile) != profile:
            profiles[species] = None
    return {species: profile for species, profile in profiles.items() if profile is not None}


def _types(mon: dict, dex: dict) -> list[str]:
    return mon.get("types") or dex["species"].get(mon.get("species"), {}).get("types", [])


def _effectiveness(attack: str, target: dict, dex: dict, move: str = "") -> float:
    if attack == "ground" and (target.get("ability") == "levitate" or target.get("item") == "airballoon"):
        return 0
    if attack == "fire" and target.get("ability") == "flashfire":
        return 0
    if attack == "electric" and target.get("ability") in {"voltabsorb", "lightningrod", "motordrive"}:
        return 0
    if attack == "water" and target.get("ability") in {"waterabsorb", "stormdrain", "dryskin"}:
        return 0
    value = 1.
    for defense in _types(target, dex):
        value *= 2 if move == "freezedry" and defense == "water" else dex["types"].get(defense, {}).get(attack, 1)
    return value


def _stat(mon: dict, stat: str, dex: dict) -> float:
    stage = mon.get("boosts", {}).get(stat, 0)
    if "stats" in mon:
        return mon["stats"][stat] * (2 + max(stage, 0)) // (2 + max(-stage, 0))
    base = dex["species"].get(mon.get("species"), {}).get(stat, 80)
    return (base + 20) * (2 + max(stage, 0)) / (2 + max(-stage, 0))


def _speed(mon: dict, conditions: list[str], dex: dict) -> float:
    return (_stat(mon, "spe", dex) * (2 if "tailwind" in conditions else 1)
            * (.5 if mon.get("status") == "par" else 1)
            * (1.5 if mon.get("item") == "choicescarf" else 1))


def _pressure(defender: dict, attackers: list[dict], dex: dict) -> float:
    return sum(max((_effectiveness(kind, defender, dex) for kind in _types(foe, dex)), default=1)
               for foe in attackers if foe)


def _damage(
    move: str, info: dict, attacker: dict, defender: dict, spread: bool, dex: dict,
    weather=(), conditions=(),
) -> float:
    data = dex["moves"].get(move, {})
    kind = data.get("type", "")
    power = info.get("base_power") or 0
    if move == "eruption":
        power *= attacker.get("hp", 1)
    if move == "ruination":
        return defender.get("hp", 1) / 2
    hits = data.get("multihit", 1)
    if isinstance(hits, list):
        hits = sum(hits) / 2 + (1 if attacker.get("item") == "loadeddice" and hits == [2, 5] else 0)
    physical = info.get("category") == "physical"
    offense, defense = ("atk", "def") if physical else ("spa", "spd")
    attack = _stat(defender if move == "foulplay" else attacker, "def" if move == "bodypress" else offense, dex)
    if "stats" in attacker and "stats" in defender:
        guard = _stat(defender, defense, dex)
        attack *= {
            "choiceband": 1.5 if physical else 1, "choicespecs": 1 if physical else 1.5,
        }.get(attacker.get("item"), 1)
        guard *= {"assaultvest": 1 if physical else 1.5, "eviolite": 1.5}.get(defender.get("item"), 1)
        base = ((2 * attacker["level"] // 5 + 2) * power * attack // guard) // 50 + 2
        weather_boost = 1.
        if kind in {"fire", "water"}:
            if "sunnyday" in weather:
                weather_boost = 1.5 if kind == "fire" else .5
            elif "raindance" in weather:
                weather_boost = 1.5 if kind == "water" else .5
        screen = (2 / 3 if ("reflect" if physical else "lightscreen") in conditions
                  or "auroraveil" in conditions else 1)
        burn = .5 if physical and attacker.get("status") == "brn" and attacker.get("ability") != "guts" else 1
        stab = (2 if attacker.get("ability") == "adaptability" else 1.5) if kind in _types(attacker, dex) else 1
        item = 1.3 if attacker.get("item") == "lifeorb" else 1
        modifier = stab * burn * weather_boost * screen * item * _effectiveness(kind, defender, dex, move)
        return base * hits * .925 * modifier * (.75 if spread else 1) / defender["stats"]["hp"]

    if physical and attacker.get("status") == "brn" and attacker.get("ability") != "guts":
        attack *= .5
    hp = dex["species"].get(defender.get("species"), {}).get("hp", 80) + 75
    stab = 1.5 if kind in _types(attacker, dex) else 1
    base = (.44 * power * attack / _stat(defender, defense, dex) + 2) * hits * .925 / hp
    return base * stab * _effectiveness(kind, defender, dex, move) * (.75 if spread else 1)


def _preview_ranks(decision: Decision, dex: dict) -> list[dict]:
    team = {mon["species"]: mon for mon in decision.observation["team"] if mon and "species" in mon}
    foes = [mon for mon in decision.observation["opponent_team"] if mon]
    names = {name: ident(name) for action in decision.actions for name in action["bring"]}
    if not foes or not set(names.values()) <= team.keys():
        return ranked_actions(decision)

    offense = {}
    pressure = {}
    for name, mon in team.items():
        attacks = [(move, dex["moves"].get(move, {})) for move in mon.get("moves", ())]
        offense[name] = [
            max((_damage(move, info, mon, foe, False, dex) * info["accuracy"]
                 for move, info in attacks if info.get("base_power")), default=0.)
            for foe in foes
        ]
        pressure[name] = [_pressure(mon, [foe], dex) for foe in foes]
    speed = {name: _speed(mon, [], dex) for name, mon in team.items()}
    foe_speeds = [_speed(foe, [], dex) for foe in foes]
    matchups = {name: [damage - .12 * risk for damage, risk in zip(offense[name], pressure[name], strict=True)]
                for name in team}

    @cache
    def coverage(lineup):
        matchups_by_foe = zip(*(matchups[name] for name in lineup), strict=True)
        return sum(best + .35 * second for best, second, *_ in
                   (sorted(values, reverse=True) for values in matchups_by_foe)) / len(foes)

    @cache
    def score(bring, leads):
        bench = tuple(name for name in bring if name not in leads)
        pivots = sum(
            max(min(max(0., pressure[lead][i] - pressure[reserve][i]), 2) * min(.4, offense[reserve][i])
                for reserve in bench)
            for lead in leads for i in range(len(foes))
        ) / (2 * len(foes))
        lead_moves = set().union(*(team[name].get("moves", ()) for name in leads))
        tempo = .07 * ("fakeout" in lead_moves) + .06 * any(
            team[name].get("ability") == "intimidate" for name in leads
        )
        control = 0.
        if "tailwind" in lead_moves:
            control = sum(speed[name] < foe <= 2 * speed[name] for name in bring for foe in foe_speeds)
        if "trickroom" in lead_moves:
            flips = sum((speed[name] < foe) - (speed[name] > foe) for name in bring for foe in foe_speeds)
            control = max(control, flips)
        return coverage(bring) + .65 * coverage(leads) + .28 * pivots + tempo + .06 * control / len(foes)

    return sorted(decision.actions, key=lambda action: score(
        tuple(sorted(names[name] for name in action["bring"])),
        tuple(sorted(names[name] for name in action["leads"])),
    ), reverse=True)


def _battle_ranks(decision: Decision, dex: dict, card_profiles: dict | None = None) -> list[dict]:
    board = decision.observation
    active = board["active"]
    opponents = board["opponents"]
    if card_profiles:
        active = [mon | card_profiles.get(mon.get("species"), {}) if mon else None for mon in active]
        opponents = [mon | card_profiles.get(mon.get("species"), {}) if mon else None for mon in opponents]
    live_targets = [index for index, foe in enumerate(opponents) if foe and not foe.get("fainted")]
    foes = [opponents[index] for index in live_targets]
    ours = board.get("side_conditions", [])
    theirs = board.get("opponent_side_conditions", [])
    order = -1 if "trickroom" in board.get("fields", []) else 1
    speeds = [_speed(foe or {}, theirs, dex) for foe in opponents]
    own_speeds = [_speed(mon or {}, ours, dex) for mon in active]
    hp = [(foe or {}).get("hp", 1) for foe in opponents]
    pressures = [_pressure(mon or {}, foes, dex) for mon in active]
    weather = board.get("weather", ())

    # Each individual slot choice repeats across many doubles combinations.
    @cache
    def evaluate(index, kind, move, target, species):
        attacker = active[index] or {}
        other = 1 - index
        if kind == "pass":
            return 0., ()
        if kind == "switch":
            incoming = _pressure({"species": ident(species)}, foes, dex)
            return .08 * (pressures[index] - incoming) - .06, ()
        info = decision.move_info[index].get(move, {})
        if target < 0:
            if move == "pollenpuff":
                ally = active[-target - 1] or {}
                return .4 * min(.5, 1 - ally.get("hp", 1)), ()
            return (.10 if move in ALLY_MOVES else -.7), ()
        spread = info.get("target") in SPREAD
        targets = [target - 1] if target > 0 and target - 1 in live_targets else []
        if not target and (spread or info.get("target") == "randomnormal"):
            targets = live_targets
        if target > 0 and not targets:
            return -.5, ()
        value = 0.
        slow = sum(order * (own_speeds[other] - speeds[foe]) < 0 for foe in live_targets)
        if move == "tailwind" and "tailwind" not in ours and order == 1:
            value = .15 * slow
        elif move == "trickroom":
            value = .14 * (slow - (len(foes) - slow))
        elif move in {"icywind", "thunderwave"} and order == 1 and targets:
            value = .06 * slow
        elif move == "protect":
            value = .04 * min(pressures[index], 4) * (1.2 - attacker.get("hp", 1))
        elif move == "spore" and targets:
            foe = opponents[targets[0]]
            if not foe.get("status") and "grass" not in _types(foe, dex) and foe.get("item") != "safetygoggles":
                value = .28
        elif move == "fakeout" and board.get("turn") == 1 and targets:
            value = .24 * _effectiveness("normal", opponents[targets[0]], dex)
        if not info.get("base_power") and move != "ruination":
            return value, ()
        ally_spread = info.get("target") == "alladjacent" and bool(active[other])
        reduced = spread and len(targets) + ally_spread > 1
        accuracy = info.get("accuracy")
        if not isinstance(accuracy, (int, float)):
            accuracy = 1.
        priority = info.get("priority") or 0
        strikes = []
        if ally_spread:
            damage = _damage(move, info, attacker, active[other], reduced, dex, weather, ours)
            strikes.append((-1, accuracy * damage))
        for foe in targets:
            damage = _damage(move, info, attacker, opponents[foe], reduced, dex, weather, theirs)
            if info.get("target") == "randomnormal":
                damage /= len(targets)
            early = priority > 0 or (priority == 0 and order * (own_speeds[index] - speeds[foe]) > 0)
            value += .16 * accuracy * min(1, damage / max(.01, hp[foe])) if early else 0
            strikes.append((foe, damage * accuracy))
        return value, tuple(strikes)

    def score(action):
        value = 0.
        damage = [0.] * len(opponents)
        for index, slot in enumerate(SLOTS):
            choice = action[slot]
            support, strikes = evaluate(
                index, choice["type"], choice.get("move_id"), choice.get("target", 0), choice.get("species"),
            )
            value += support
            for foe, amount in strikes:
                if foe != -1:
                    damage[foe] += amount
                elif action[SLOTS[1 - index]].get("move_id") != "protect":
                    value -= amount
        for foe in live_targets:
            value += min(hp[foe], damage[foe]) + .3 * min(1, damage[foe] / max(.01, hp[foe])) ** 3
        return value

    return sorted(decision.actions, key=score, reverse=True)


def tactical_actions(decision: Decision, *, card_profiles: dict | None = None) -> list[dict]:
    if decision.phase == PREVIEW:
        return _preview_ranks(decision, load_dex())
    if decision.phase != BATTLE or len(decision.actions) == 1:
        return ranked_actions(decision)
    return _battle_ranks(decision, load_dex(), card_profiles)


def tactical(decision: Decision, *, card_profiles: dict | None = None) -> dict:
    return tactical_actions(decision, card_profiles=card_profiles)[0]


def tactical_candidates(decision: Decision) -> list[dict]:
    ranked = tactical_actions(decision)
    if decision.phase == PREVIEW:
        shortlist = []
        seen = set()
        for action in ranked:
            lineup = (frozenset(action["bring"]), frozenset(action["leads"]))
            if lineup not in seen:
                shortlist.append(action)
                seen.add(lineup)
                if len(shortlist) == 6:
                    break
        return shortlist
    if decision.phase != BATTLE or len(ranked) <= 6:
        return ranked[:6]
    shortlist = ranked[:1]
    for qualifies in (
        lambda slot: slot["type"] == "switch",
        lambda slot: slot.get("move_id") == "protect",
        lambda slot: slot.get("move_id") in SPEED_CONTROL,
    ):
        if not any(qualifies(action[slot]) for action in shortlist for slot in SLOTS):
            best = next((action for action in ranked if any(qualifies(action[slot]) for slot in SLOTS)), None)
            if best is not None:
                shortlist.append(best)
    return (shortlist + [action for action in ranked if action not in shortlist])[:6]
