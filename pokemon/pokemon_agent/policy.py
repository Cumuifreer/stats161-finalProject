"""Two inexpensive baselines for contract tests and paired evaluation."""

import random

from .game import BATTLE, DRAFT, Decision, ident

SUPPORT_MOVES = {"fakeout", "followme", "ragepowder", "tailwind", "trickroom", "icywind", "spore"}
ALLY_MOVES = {"helpinghand", "pollenpuff", "healpulse", "coaching"}


class RandomPolicy:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def __call__(self, decision: Decision) -> dict:
        return self.rng.choice(decision.actions)


def _draft_score(card: dict, picked: list[dict]) -> float:
    moves = {ident(move) for move in card.get("moves", ())}
    covered = {ident(move) for member in picked for move in member.get("moves", ())}
    return 20 * len((moves & SUPPORT_MOVES) - covered) + 4 * ("protect" in moves)


def _battle_score(decision: Decision, action: dict) -> float:
    score = 0.0
    targets = []
    opponents = decision.observation["opponents"]
    for index in range(2):
        slot = action[f"slot_{index}"]
        if slot["type"] == "switch":
            score += 5
        if slot["type"] != "move":
            continue
        move = slot["move_id"]
        info = decision.move_info[index].get(move, {})
        target = slot["target"]
        power = info.get("base_power") or 0
        accuracy = info.get("accuracy")
        if not isinstance(accuracy, (float, int)):
            accuracy = 1.0
        if accuracy > 1:
            accuracy /= 100
        if target < 0:
            score += 20 if move in ALLY_MOVES else -120
        elif target > 0 and (target > len(opponents) or not opponents[target - 1]
                             or opponents[target - 1].get("fainted")):
            score -= 150
        else:
            score += power * accuracy + 7 * (info.get("priority") or 0)
            target_type = ident(str(info.get("target") or ""))
            if power and target_type in {"alladjacentfoes", "alladjacent"}:
                score += 20 * sum(bool(mon and not mon.get("fainted")) for mon in opponents)
                if target_type == "alladjacent":
                    score -= 35
            if move in SUPPORT_MOVES:
                score += 18
            if target > 0 and power:
                targets.append(target)
                hp = opponents[target - 1].get("hp")
                score += 16 * (1 - hp) if hp is not None else 0
    if len(targets) == 2 and targets[0] == targets[1]:
        score += 12
    return score


def ranked_actions(decision: Decision) -> list[dict]:
    if decision.phase == BATTLE:
        return sorted(decision.actions, key=lambda action: _battle_score(decision, action), reverse=True)
    if decision.phase == DRAFT:
        cards = {card["card_id"]: card for card in decision.observation["cards"]}
        picked = decision.observation["picked"]
        return sorted(
            decision.actions, key=lambda action: _draft_score(cards.get(action["card_id"], {}), picked),
            reverse=True,
        )
    team = {mon["species"]: mon for mon in decision.observation["team"] if mon and "species" in mon}
    return sorted(
        decision.actions,
        key=lambda action: sum(_draft_score(team.get(ident(lead), {}), []) for lead in action["leads"]),
        reverse=True,
    )


def heuristic(decision: Decision) -> dict:
    return ranked_actions(decision)[0]
