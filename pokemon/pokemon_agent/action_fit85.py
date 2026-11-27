"""Distill broad public-world action evaluations into a fast proposal model."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import replace
import json
from multiprocessing import get_context
from pathlib import Path
from random import Random
from statistics import fmean
from time import perf_counter

import torch
from torch.nn import functional as F

from .draft_coverage import Coverage
from .draft_last_pick import other, sample_game
from .draft_leaf import CATALOG
from .draft_league import pools_for, read, save
from .features import BattleEncoder
from .game import BATTLE, BATTLE_BANK, DECISION_LIMIT, PREVIEW
from .joint import BattleHistory, JointPolicy
from .lab85 import BASE, MODELS, SEALED
from .lineup import index_cards
from .model import ActionRankingNetwork
from .planning_bench import configure_worker
from .reply_learning import LearnedReplies, _ages, encode
from .reply_lineup import ReplyLineup
from .search import _key
from .search85 import HypotheticalReplies, Search85
from .sim import SEATS, Showdown, _draft, fingerprint, play_battle
from .strength import EXTERNAL, ExternalPivot, verify_pools
from .tactics import tactical_candidates


class WideTeacher(Search85):
    def __init__(self, profiles, worker, **settings):
        super().__init__(profiles, worker, **settings)
        self.encoder = BattleEncoder(profiles)
        self.rows = []

    def record_labels(self, decision, seat, evidence):
        self.targets = defaultdict(list)
        self.joint.observe(decision)
        history = BattleHistory.from_snapshot(self.joint.history.snapshot())
        chosen = self(decision, seat, evidence)
        if len(self.targets) > 4:
            state, actions = encode(self.encoder, decision, history, _ages(evidence["events"]))
            examples, targets = [], []
            for action, encoded in zip(decision.actions, actions, strict=True):
                if labels := self.targets.get(_key(action)):
                    examples.append(encoded)
                    targets.append(fmean(labels))
            self.rows.append((state, examples, targets))
        return chosen

    def _score(self, world, seat, actions, seeds, deadline, depth):
        if depth == 1:
            random = Random(str(seeds[0]))
            seeds = [seeds[0], *[[random.randrange(65536) for _ in range(4)] for _ in range(2)]]
        scores = super()._score(world, seat, actions, seeds, deadline, depth)
        if depth == 1 and scores is not None:
            for action, score in zip(actions, scores, strict=True):
                self.targets[_key(action)].append(score - scores[0])
        return scores


class PlayerView:
    needs_proxy_evidence = True
    needs_own_proposals = True

    def __init__(self, scout):
        self.choose_with_public_history = scout.record_labels


def collect(roots, cards, checkout):
    torch.set_num_threads(1)
    result = {}
    counts = Counter()
    with ExitStack() as stack:
        experiment = Coverage(stack, cards)
        indexed = index_cards(cards)
        worker = stack.enter_context(Showdown(public_search=True))
        configure_worker(worker, cards, MODELS / "value-v2.json")
        proxy = HypotheticalReplies(stack, checkout, indexed)
        reply = LearnedReplies(MODELS / "reply", experiment.profiles)
        lineup = ReplyLineup(MODELS / "lineup-belief.json", cards, experiment.profiles)
        external = {name: stack.enter_context(ExternalPivot(checkout, name, cards)) for name in EXTERNAL}
        for root in roots:
            pool, first, seed = sample_game(cards, root)
            rows = result[root] = []
            rosters = _draft(pool, first, lambda seat, decision: experiment.choose(decision, "pairwise"))
            for actor in SEATS:
                foe = other(actor)
                teacher = WideTeacher(experiment.profiles, worker, learned=reply, proxy=proxy, lineup_model=lineup,
                                      **(BASE | {"worlds": 1, "candidate_limit": 36, "move_budget": 5.}))
                style = ("ace-code", "pivot", "joint")[(root + int(actor == "p2")) % 3]
                opponent = external.get(style) or JointPolicy(experiment.profiles)
                if style in external:
                    opponent.setup(root, foe, first)
                policies = {actor: PlayerView(teacher), foe: opponent}
                scheduled = {1, 2 + root % 3, 4 + root % 3}
                seen = set()

                def choose(seat, decision, view):
                    if decision.phase == PREVIEW:
                        options = tactical_candidates(decision)
                        choice = options[(root + int(seat == "p2")) % min(3, len(options))]
                        if seat == actor:
                            teacher.observe_preview(decision, choice)
                        elif style not in external:
                            opponent.observe(decision)
                        return choice
                    if seat == foe:
                        if style in external:
                            policy_view = replace(decision, seconds_left=DECISION_LIMIT[BATTLE], bank_seconds_left=BATTLE_BANK)
                            return opponent.choose(policy_view, view)
                        return opponent(decision)
                    evidence = view["search"]
                    turn = decision.observation["turn"]
                    if evidence["force_switch"] or turn not in scheduled or turn in seen:
                        return teacher.joint(decision)
                    seen.add(turn)
                    return teacher.record_labels(decision, seat, evidence)

                play_battle(experiment.league.engine, rosters, policies, root, seed, choose)
                counts["synthetic_games"] += 1
                rows.extend(teacher.rows)
                counts.update(teacher.counts)
    return result, counts


def pack(rows, device):
    width = max(len(actions) for _, actions, _ in rows)
    states, actions, returns, legal = [], [], [], []
    for state, menu, targets in rows:
        pad = width - len(menu)
        states.append(state)
        actions.append(menu + [(0.,) * len(menu[0])] * pad)
        returns.append(targets + [0.] * pad)
        legal.append([True] * len(menu) + [False] * pad)
    return tuple(torch.tensor(column, device=device) for column in (states, actions, returns, legal))


def model_loss(net, states, actions, returns, legal):
    scores = net(states, actions).masked_fill(~legal, -100.)
    targets = (returns / .50).masked_fill(~legal, -100.).softmax(-1)
    return -(targets * scores.log_softmax(-1)).sum(-1).mean()


def fit(dataset, cards, pools, seed, output):
    torch.manual_seed(seed)
    roots = list(sorted(dataset))
    Random(seed).shuffle(roots)
    validation = set(roots[::5])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training = pack([row for root in roots if root not in validation for row in dataset[root]], device)
    held = pack([row for root in roots if root in validation for row in dataset[root]], device)
    sizes = dict(state_size=training[0].shape[-1], action_size=training[1].shape[-1], hidden_size=96)
    net = ActionRankingNetwork(**sizes).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0019, weight_decay=.015)
    best_loss, best = float("inf"), None
    started = perf_counter()
    for _ in range(78):
        for indices in torch.randperm(len(training[0]), device=device).split(128):
            loss = model_loss(net, *(column[indices] for column in training))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            loss = float(model_loss(net, *held))
            if loss < best_loss:
                best_loss = loss
                best = {name: value.detach().cpu().clone() for name, value in net.state_dict().items()}
    net.load_state_dict(best)
    score = net(*held[:2]).detach().masked_fill(~held[-1], -100.)
    truth = held[2].masked_fill(~held[-1], -100.)
    predicted = truth.gather(1, score.topk(5, dim=-1).indices).amax(-1)
    tactical = held[1][..., -2].masked_fill(~held[-1], -100.)
    base = truth.gather(1, tactical.topk(5, dim=-1).indices).amax(-1)
    actual = truth.amax(-1)
    report = dict(held_loss=best_loss, held_top5_regret=float((actual - predicted).mean()), fit_seconds=perf_counter() - started,
                  tactical_top5_regret=float((actual - base).mean()),
                  training_states=len(training[0]), test_states=len(held[0]), device=str(device),
                  parameters=sum(value.numel() for value in best.values()))
    output.mkdir(parents=True)
    torch.save(best, output / "state_dict.pt")
    save(output / "config.json", dict(catalog=fingerprint(cards), network=sizes, pools=pools, validation_roots=sorted(validation)))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pools", type=int, required=True)
    parser.add_argument("--ace-trainer", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.pools < 5 or args.workers < 1:
        parser.error("Provide a fresh output, at least five pools, and a positive number of workers")
    began = perf_counter()
    cards = read(CATALOG)
    roots = list(range(args.seed, args.seed + args.pools))
    pools = pools_for(cards, roots)
    verify_pools(pools, [*SEALED, *Path("benchmark_results/research85").glob("*.json"), *Path("checkpoints/research85").glob("*.json"),
                         *Path("checkpoints/research85").glob("*/config.json")])
    data, counts = {}, Counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as workers:
        jobs = [workers.submit(collect, roots[i:i + 5], cards, args.ace_trainer) for i in range(0, len(roots), 5)]
        for future in as_completed(jobs):
            part, stats = future.result()
            data.update(part)
            counts.update(stats)
            print(f"Labeled action sets from {len(data)}/{len(roots)} pools", flush=True)
    report = fit(data, cards, pools, args.seed, args.output)
    report.update(seconds=perf_counter() - began, api_spend_usd=0, counts=dict(counts))
    save(args.output / "metrics.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
