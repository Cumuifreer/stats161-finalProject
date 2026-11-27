"""Paired full-match evaluation of terminal learning and optional DeepSeek advice."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from hashlib import sha256
from multiprocessing import get_context
from pathlib import Path
from statistics import fmean
import sys
from time import perf_counter
from unittest.mock import patch

import torch

from .. import blind85, lab85, quicksilver80
from ..battle_league import Encoder, configure_torch, load_model
from ..joint import JointPolicy
from ..research80.scouting.replies import ForecastReplies80
from ..search import _key
from ..tactics import SLOTS
from . import deepseek


class OutcomeScorer:
    def __init__(self, profiles, model):
        self.encoder = Encoder(profiles)
        self.model = model
        self.ranker = JointPolicy(profiles)
        self.trace = []
        self.last = None

    @torch.inference_mode()
    def score(self, decision, history):
        self.ranker.history = history
        ranked = self.ranker.rank(decision)
        if len(ranked) < 2:
            self.last = None
            return {}
        state, actions, cards, mons, legal = self.encoder(decision, history, ranked)
        self.last = state, actions, [action for action, _ in ranked]
        missing = self.model.history_steps - len(self.trace)
        trace = [(0.,) * (len(state) + len(actions[0]))] * missing + self.trace
        logits, q, value = self.model(torch.tensor([state]), torch.tensor([actions]), torch.tensor([cards]),
                                      torch.tensor([mons]), torch.tensor([legal]), torch.tensor([trace]))
        probabilities = q[0].sigmoid().tolist()
        # A small continuation prior. A checked tactical gain can outweigh the network.
        reference = value[0].sigmoid().item()
        return {_key(action): max(-.16, min(.16, .90 * (probabilities[index] - reference)))
                for index, (action, _) in enumerate(ranked)}

    def remember(self, decision, action, history):
        if self.last is None:
            return
        state, actions, options = self.last
        if action in options:
            frame = (*state, *actions[options.index(action)])
        else:
            single = replace(decision, actions=(action,))
            encoded = self.encoder.base.encode(single, history=history)
            switch = [self.encoder.card_ids.get(action[slot].get("species"), 0) / len(self.encoder.card_ids) for slot in SLOTS]
            frame = (*state, *encoded.actions[0], 0., 0., 1., *switch)
        self.trace.append(frame)
        self.trace = self.trace[-self.model.history_steps:]


def advice_menu(chosen, review):
    if not review:
        return [], []
    seen = defaultdict(list)
    actions = {}
    for menu, raw in review:
        if chosen not in menu:
            continue
        base = raw[menu.index(chosen)]
        for action, value in zip(menu, raw, strict=True):
            key = _key(action)
            actions[key] = action
            seen[key].append(value - base)
    keys = [key for key, values in seen.items() if actions[key] != chosen and fmean(values) >= -.20 and min(values) >= -.40]
    keys.sort(key=lambda key: fmean(seen[key]), reverse=True)
    menu = [chosen, *(actions[key] for key in keys[:3])]
    notes = [{"mean_relative_to_default": round(fmean(seen[_key(action)]), 3),
              "worst_relative_to_default": round(min(seen[_key(action)]), 3)} for action in menu] if keys else []
    return menu, notes


class LearningForecast(ForecastReplies80):
    def __init__(self, profiles, worker, *, model, key, **settings):
        super().__init__(profiles, worker, **settings)
        self.neural = OutcomeScorer(profiles, model) if model is not None else None
        self.llm = deepseek.DeepSeek(key, limit=1) if key else None
        self.review = []
        self.prior = {}

    def _score(self, world, seat, actions, seeds, deadline, depth):
        result = super()._score(world, seat, actions, seeds, deadline, depth)
        if result is None:
            return None
        if self.llm is not None and depth == 2:
            self.review.append((actions, result))
        return [value + self.prior.get(_key(action), 0.) for action, value in zip(actions, result, strict=True)] if self.prior else result

    def _plan(self, decision, seat, evidence, ranked, coarse):
        started = perf_counter()
        self.review.clear()
        self.prior = self.neural.score(decision, self.joint.history) if self.neural else {}
        if self.prior:
            self.counts["neural_scored"] += 1
            self.counts["neural_score_ms_sum"] += (perf_counter() - started) * 1000
        chosen = super()._plan(decision, seat, evidence, ranked, coarse)
        if self.llm and decision.observation["turn"] <= 4:
            menu, notes = advice_menu(chosen, self.review)
            if len(menu) >= 2:
                budget = min(7., min((t for t in (decision.seconds_left, decision.bank_seconds_left) if t is not None),
                                     default=50.) - (perf_counter() - started) - 1.5)
                selected = self.llm.advise(decision, menu, chosen, context={"local_rollouts": notes}, timeout=budget, think=True)
                self.counts["llm_actions_changed"] += selected != chosen
                chosen = selected
        if self.prior:
            self.neural.remember(decision, chosen, self.joint.history)
            self.counts["neural_preferred_action_selected"] += _key(chosen) == max(self.prior, key=self.prior.__getitem__)
        return chosen


def evaluate(roots, cards, rivals, *args, checkpoint, key):
    configure_torch()
    model = load_model(checkpoint, cards) if checkpoint else None

    class BoundForecast(LearningForecast):
        def __init__(self, *args, **settings):
            super().__init__(*args, model=model, key=key, **settings)

    def play(*args, **settings):
        result = PLAY(*args, **settings)
        candidate = settings.get("candidate") or args[-1]
        if candidate.battle.llm:
            result["llm_telemetry"] = candidate.battle.llm.records
        return result

    with patch.object(lab85, "ForecastReplies80", BoundForecast), patch.object(lab85, "play", play):
        return lab85.evaluate(roots, cards, rivals, *args)


def measurement(checkpoint, key):
    def measure(roots, cards, rivals, checkout, variant, workers, *args):
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
            jobs = [pool.submit(evaluate, [root], cards, rivals, checkout, variant, *args, checkpoint=checkpoint, key=key)
                    for root in roots]
            for job in as_completed(jobs):
                yield job.result()
    return measure


def freeze_for(checkpoint, asked):
    def freeze(*args):
        result = FROZEN(*args)
        keep = [Path(__file__), Path(deepseek.__file__)]
        if checkpoint:
            keep.extend(checkpoint / name for name in ("config.json", "state_dict.pt"))
        for path in keep:
            result["files"][str(path)] = sha256(path.read_bytes()).hexdigest()
        result["research_learning"] = {"checkpoint": str(checkpoint) if checkpoint else None,
                                       "used_service": "DeepSeek deepseek-flash" if asked else None,
                                       "llm_max_requests_per_game": int(asked)}
        return result
    return freeze


def publish(path, result):
    if "records" in result and "report" in result:
        total = 0.
        for name, metrics in result["report"].items():
            records = [record for row in result["records"].values() for game in row[name]
                       for record in game.get("llm_telemetry", ())]
            if records:
                price = sum(row.get("peak_usd", 0.) for row in records)
                metrics["llm"] = dict(requests=len(records), valid=sum(row["outcome"] == "accepted" for row in records),
                                      changed=sum(row["changed"] for row in records), peak_priced_usage_usd=price,
                                      seconds=fmean(row["seconds"] for row in records))
                total += price
        result["api_spend_usd"] = total
        result["api_spend_basis"] = "reported token use, priced at DeepSeek's peak rate"
    PUBLISH(path, result)


PLAY, FROZEN, PUBLISH = lab85.play, quicksilver80.freeze, blind85.publish_json


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--battle-net", type=Path)
    parser.add_argument("--deepseek-flash", action="store_true")
    args, remaining = parser.parse_known_args()
    if not args.battle_net and not args.deepseek_flash:
        parser.error("Supply --battle-net or --deepseek-flash")
    if args.deepseek_flash and ("--seed" not in remaining or "--resume" in remaining):
        parser.error("The remote model and prompt are still under development. Use a fresh development run.")
    key = deepseek.credential() if args.deepseek_flash else None
    sys.argv[1:] = remaining
    with patch.object(blind85, "measure", measurement(args.battle_net, key)), patch.object(blind85, "publish_json", publish), \
            patch.object(quicksilver80, "freeze", freeze_for(args.battle_net, args.deepseek_flash)):
        quicksilver80.main()


if __name__ == "__main__":
    main()
