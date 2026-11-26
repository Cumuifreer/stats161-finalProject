"""Evaluate legal replacements against the visible active Pokémon."""

from .tactics import SLOTS, SPREAD, _damage, _speed, load_dex


def replacement(decision, baseline, profiles, edge=.12):
    board = decision.observation
    dex = load_dex()
    weather = board["weather"]
    room = -1 if "trickroom" in board["fields"] else 1
    theirs = tuple(mon for mon in board["opponents"] if mon)
    if not theirs:
        return baseline

    def profile(entry):
        stats = profiles.get(entry["species"], {})
        result = stats | entry
        if result.get("card_ability"):
            result.setdefault("ability", result["card_ability"])
        if result.get("card_item"):
            result.setdefault("item", result["card_item"])
        return result

    existing = tuple(profile(mon) if mon else None for mon in board["active"])
    foes = tuple(profile(mon) for mon in theirs)
    reserves = {name: profile({"species": name, **mon}) for name, mon in board["reserves"].items()}
    conditions = (board["side_conditions"], board["opponent_side_conditions"])

    def value(action):
        party = list(existing)
        leads = []
        for index, label in enumerate(SLOTS):
            slot = action[label]
            if slot["type"] == "switch" and slot["species"] in reserves:
                incoming = reserves[slot["species"]]
                party[index] = incoming
                leads.append(incoming)
        own = tuple(mon for mon in party if mon)
        if not own or not leads:
            return -100.
        intimidates = sum(mon.get("ability") == "intimidate" for mon in leads)
        opponents = []
        for enemy in foes:
            boosts = enemy.get("boosts", {}).copy()
            if intimidates and enemy.get("ability") not in {"innerfocus", "clearbody", "oblivious", "scrappy"}:
                boosts["atk"] = max(-6, boosts.get("atk", 0) - intimidates)
                if enemy.get("ability") == "defiant":
                    boosts["atk"] = min(6, boosts["atk"] + 2 * intimidates)
                if enemy.get("ability") == "competitive":
                    boosts["spa"] = min(6, boosts.get("spa", 0) + 2 * intimidates)
            opponents.append(enemy | {"boosts": boosts})

        def threats(team, attackers, side):
            rows = []
            for defender in team:
                incoming = [hit(attacker, defender, 1 - side, len(team) > 1) for attacker in attackers]
                hp = defender.get("hp", 1.)
                fastest = max((power for power, early in incoming if early), default=0.)
                physical_loss = sum(power for power, _ in incoming)
                loss = min(hp, physical_loss)
                knockout = min(1., physical_loss / max(.01, hp)) ** 3
                early_ko = min(1., fastest / max(.01, hp)) ** 3
                rows.append((loss, knockout, early_ko))
            return rows

        effective_weather = next((("raindance",) if mon.get("ability") == "drizzle" else ("sunnyday",)
                                  for mon in leads if mon.get("ability") in {"drizzle", "drought"}), weather)

        def hit(attacker, defender, left, spread):
            hp = defender.get("hp", 1.)
            early = room * (_speed(attacker, conditions[left], dex) - _speed(defender, conditions[1 - left], dex)) > 0
            choices = []
            for name in attacker.get("card_moves", ()):
                move = dex["moves"].get(name, {})
                if not move.get("base_power") or name == "fakeout":
                    continue
                damage = _damage(name, move, attacker, defender, spread and move["target"] in SPREAD,
                                 dex, effective_weather, conditions[1 - left])
                fast = move.get("priority", 0) > 0 or (move.get("priority", 0) == 0 and early)
                choices.append((min(hp, damage) + .16 * fast, damage * move.get("accuracy", 1.), fast))
            return max(choices, default=(0., 0., False))[1:]

        danger = threats(own, opponents, 0)
        opportunity = threats(opponents, own, 1)

        def total(rows):
            return sum(.35 * damage + .27 * knockout + .22 * early for damage, knockout, early in rows)

        cycling = .14 * sum("fakeout" in mon.get("card_moves", ()) for mon in leads)
        return total(opportunity) - total(danger) + cycling + .18 * sum(mon.get("hp", 1.) for mon in leads)

    base = value(baseline)
    alternative = max(decision.actions, key=value)
    return alternative if value(alternative) > base + edge else baseline
