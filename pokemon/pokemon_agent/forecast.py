"""Joint-based doubles planning with an optional continuation term."""

from functools import cache
from dataclasses import dataclass
from math import isfinite

from .game import BATTLE
from .joint import JointPolicy, TurnForecast, PROTECT, _move
from .tactics import SLOTS, SPREAD, _speed, tactical_actions, tactical_candidates


@dataclass(frozen=True)
class ForecastConfig:
    candidates: int = 6
    position: float = 0.
    replies: bool = False
    risk: float = .15

    def __post_init__(self):
        if type(self.candidates) is not int or not 6 <= self.candidates <= 20:
            raise ValueError("Forecast requires between 6 and 20 candidates")
        if (type(self.replies) is not bool or not isfinite(self.position) or not 0 <= self.position <= 1
                or not isfinite(self.risk) or not 0 <= self.risk <= .5):
            raise ValueError("Invalid forecast weights")


class ForecastPolicy(JointPolicy):
    def __init__(self, profiles, config):
        super().__init__(profiles)
        self.config = config

    def rank(self, decision, candidates=None):
        if decision.phase != BATTLE:
            return super().rank(decision, candidates)
        self.history.observe(decision)
        if candidates is None:
            candidates = tactical_candidates(decision)
            if self.config.candidates > 6 and len(decision.actions) > len(candidates):
                calibrated = tactical_actions(decision, card_profiles=self.profiles)
                extra = (action for action in calibrated if action not in candidates)
                candidates = [*candidates, *list(extra)[:self.config.candidates - len(candidates)]]
        if len(candidates) == 1:
            return [(candidates[0], {})]
        turn = PositionForecast(decision, self.profiles, self.history.unseen(),
                                self.history.repeated(decision.observation.get("turn")))
        return turn.rank_with_config(candidates, self.config)


class PositionForecast(TurnForecast):
    def _alternative_replies(self, baseline):
        replies = self._replies(baseline)
        attack = replies[0][2]
        # Include a different pressure target and the other Protect branch where feasible.
        healthy = [bool(mon) for mon in self.mons[:2]]
        if all(healthy) and any(slot.get("target", 0) > 0 for slot in attack):
            redirected = [slot | {"target": 3 - slot["target"]} if slot.get("target", 0) > 0 else slot for slot in attack]
            if any(self._hit(move["move_id"], self._info(i, move["move_id"]), self.mons[i], self.mons[move["target"] - 1],
                             False, 0) > .15 for i, move in enumerate(redirected, 2) if move.get("target", 0) > 0):
                replies.append(("alternate_target", .19, redirected))
        for i in (2, 3):
            if any(response[i - 2].get("move_id") in PROTECT for _, _, response in replies):
                continue
            if protect := next((move for move in self.mons[i].get("card_moves", ()) if move in PROTECT), None):
                protected = attack.copy()
                protected[i - 2] = _move(protect)
                replies.append(("alternate_protect", .09, protected))
        return replies

    def rank_with_config(self, candidates, config):
        baseline = [candidates[0][slot] for slot in SLOTS]
        replies = self._alternative_replies(baseline) if config.replies else self._replies(baseline)
        total_weight = sum(weight for _, weight, _ in replies)

        @cache
        def potential(attacker_id, defender_id, foes, defender_side):
            attacker, defender = observed[attacker_id], observed[defender_id]
            hits = []
            for move in attacker.get("card_moves", attacker.get("moves", ())):
                if move == "fakeout":
                    continue
                info = self.dex["moves"].get(move, {})
                if info.get("base_power") or move == "ruination":
                    if info.get("target") == "alladjacent":
                        continue
                    damage = self._hit(move, info, attacker, defender, foes > 1 and info.get("target") in SPREAD, defender_side)
                    if info.get("target") == "randomnormal":
                        damage /= foes
                    hits.append(damage)
            return max(hits, default=0.)

        observed = {}

        def continuation(details):
            mons, hp = details["mons"], details["hp"]
            observed.update((id(mon), mon) for mon in mons)
            speeds = [_speed(mon, ("tailwind",) if details["tailwind"][i // 2] else (), self.dex)
                      for i, mon in enumerate(mons)]
            pressure = [0., 0.]
            for i, mon in enumerate(mons):
                if hp[i] <= 0:
                    continue
                side = i // 2
                foes = [j for j, health in enumerate(hp) if j // 2 != side and health > 0]
                threats = []
                for j in foes:
                    damage = potential(id(mon), id(mons[j]), len(foes), 1 - side)
                    if any(move in mons[j].get("card_moves", ()) for move in PROTECT):
                        damage *= .95
                    early = speeds[i] < speeds[j] if details["room"] else speeds[i] > speeds[j]
                    threats.append(min(hp[j], damage) * (1 if early else .65)
                                   + (.35 if early and damage >= hp[j] else 0.))
                pressure[side] += max(threats, default=0.)
            return pressure[0] - pressure[1]

        rows = []
        for index, action in enumerate(candidates):
            choices = [action[slot] for slot in SLOTS]
            outcomes = []
            for name, weight, response in replies:
                score, details = self._evaluate(choices + response, position=bool(config.position))
                if config.position:
                    score += config.position * continuation(details)
                    for field in ("mons", "hp", "tailwind", "room"):
                        del details[field]
                outcomes.append((score, weight, name, response, details))
            mean = sum(score * weight for score, weight, *_ in outcomes) / total_weight
            worst, _, name, response, details = min(outcomes, key=lambda row: row[0])
            score = (1 - config.risk) * mean + config.risk * worst - .035 * min(index, 5)
            rows.append((action, {"score": score, "worst_reply": name, "reply_slots": response, **details}))
        return sorted(rows, key=lambda row: row[1]["score"], reverse=True)


def _fit_match(args):
    from contextlib import redirect_stdout
    from io import StringIO
    import json

    from .__main__ import _bench

    with redirect_stdout(StringIO()) as output:
        status = _bench(args)
    result = json.loads(output.getvalue())
    if status or result["completed_pairs"] != args.pairs:
        raise ValueError(f"Forecast fitting match failed against {args.right}: {result['errors']}")
    return result


def fit(args):
    """Select a bounded set of forecast weights by development cross-play."""
    from argparse import Namespace
    from concurrent.futures import ProcessPoolExecutor
    from dataclasses import asdict
    import json
    from multiprocessing import get_context
    from statistics import mean
    from tempfile import TemporaryDirectory
    from pathlib import Path

    from .sim import fingerprint, pool_fingerprint

    if args.pairs < 1 or args.workers < 1 or args.output.exists():
        raise ValueError("Use positive fitting sizes and a new forecast output directory")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cards = json.loads(args.cards.read_text(encoding="utf-8"))
    seeds = range(args.seed, args.seed + args.pairs)
    split = {str(seed): pool_fingerprint(cards, seed) for seed in seeds}
    settings = {
        "expanded": ForecastConfig(12),
        "position": ForecastConfig(position=.55),
        "responses": ForecastConfig(replies=True),
        "expanded_position": ForecastConfig(12, .55),
        "expanded_responses": ForecastConfig(12, 0, True),
        "all": ForecastConfig(12, .32, True),
    }
    opponents = ("joint", "pivot", "pivot-base", "tactical")

    def model(config):
        return {"model": asdict(config), "catalog_sha256": fingerprint(cards), "split": {"fit": split}}

    reports = {}
    with TemporaryDirectory(dir=args.output.parent, prefix="forecast-fit-") as tmp, ProcessPoolExecutor(
        max_workers=args.workers, mp_context=get_context("spawn"),
    ) as executor:
        paths = {"joint": None}
        for name, config in settings.items():
            path = Path(tmp) / f"{name}.json"
            path.write_text(json.dumps(model(config)), encoding="utf-8")
            paths[name] = path
        jobs = {(name, opponent): executor.submit(_fit_match, Namespace(
            left="forecast" if path else "joint", right=opponent, cards=args.cards, pairs=args.pairs, seed=args.seed,
            forecast_model=path, battle_only=True, testing_pools=None, preview_only=False, pair_results=True,
            ace_trainer=args.ace_trainer if opponent.startswith("pivot") else None,
        )) for name, path in paths.items() for opponent in opponents}
        for (name, opponent), job in jobs.items():
            report = job.result()
            reports.setdefault(name, {})[opponent] = report
            print(name, opponent, report["agents"][0]["wins"], flush=True)
    scores = {name: mean(result["agents"][0]["score_rate"] for result in records.values())
              for name, records in reports.items()}
    best = max(scores, key=scores.__getitem__)
    args.output.mkdir()
    if best != "joint":
        (args.output / "config.json").write_text(json.dumps(model(settings[best]), indent=2) + "\n", encoding="utf-8")
    report = {"selected": best, "scores": scores, "tested_models": {name: asdict(config) for name, config in settings.items()},
              "benchmark": reports}
    (args.output / "fit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": best, "scores": scores}), flush=True)
    return 0
