"""Calibrate the position evaluator on games played by public search."""

import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from random import Random
from time import perf_counter

import torch
from torch.nn import functional as F

from .draft_counterfactual import check_hybrid
from .draft_coverage import Coverage
from .draft_last_pick import sample_game
from .draft_leaf import CATALOG
from .draft_league import disjoint, parallel, pools_for, read, save
from .game import BATTLE, BATTLE_BANK, DECISION_LIMIT, PREVIEW
from .joint import JointPolicy
from .lineup import LineupPolicy, index_cards
from .planning_bench import configure_worker
from .planning_replay import ReplayPolicy
from .planning_value import board, features
from .proxy_replies import ProxyReplies
from .public_planning import PublicPlanner
from .reply_learning import LearnedReplies
from .reply_lineup import ReplyLineup
from .sim import SEATS, Showdown, _draft, fingerprint, play_battle
from .strength import EXTERNAL, ExternalPivot, verify_pools
from .tactics import SLOTS, tactical_candidates


def collect(roots, cards, checkout, initial, reply_model, lineup_model):
    data = {}
    pressure = read(initial)["pressure"]
    with ExitStack() as stack:
        experiment = Coverage(stack, cards)
        profiles = experiment.profiles
        indexed = index_cards(cards)
        search = stack.enter_context(Showdown(public_search=True))
        configure_worker(search, cards, initial)
        proxy = ProxyReplies(stack, checkout, indexed)
        learned = LearnedReplies(reply_model, profiles)
        belief = ReplyLineup(lineup_model, cards, profiles)
        external = {name: stack.enter_context(ExternalPivot(checkout, name, cards)) for name in EXTERNAL}
        for root in roots:
            pool, first, seed = sample_game(cards, root)
            styles = ("ace-code", "pivot", "ace-code", "pivot", "joint")
            rival = styles[root % len(styles)]
            rosters = _draft(pool, first, lambda _, decision: experiment.choose(decision, "pairwise"))
            trace = data[root] = []
            for actor, foe in (SEATS, SEATS[::-1]):
                search_policy = PublicPlanner(profiles, search, worlds=1, samples=1, candidate_limit=9,
                                              proxy=proxy, learned=learned, lineup_model=belief, adaptive=True)
                enemy = external.get(rival) or JointPolicy(profiles)
                if isinstance(enemy, ExternalPivot):
                    enemy.setup(root, foe, first)
                policies = {actor: ReplayPolicy(search_policy), foe: enemy}
                preview = LineupPolicy(profiles, indexed, experiment.preview)
                snapshots, seen = {}, set()
                rows = []

                def choose(seat, decision, view):
                    if decision.phase == PREVIEW:
                        if seat == actor:
                            action = preview(decision)
                            search_policy.observe_preview(decision, action)
                        else:
                            if isinstance(enemy, JointPolicy):
                                enemy.observe(decision)
                            options = tactical_candidates(decision)
                            action = options[(root + (seat == first)) % min(3, len(options))]
                        return action
                    turn = decision.observation["turn"]
                    can_move = any(action[slot]["type"] == "move" for action in decision.actions for slot in SLOTS)
                    if can_move and turn not in seen:
                        slot = snapshots.setdefault(turn, {})
                        slot.setdefault(seat, board(view["raw"]))
                        if set(slot) == set(SEATS):
                            rows.append(features(slot["p1"], slot["p2"], "trickroom" in decision.observation["fields"],
                                                 profiles, pressure))
                            seen.add(turn)
                    if seat == actor:
                        action = search_policy(decision, seat, view.get("search"))
                    elif isinstance(enemy, ExternalPivot):
                        action = enemy.choose(replace(decision, seconds_left=DECISION_LIMIT[BATTLE],
                                                      bank_seconds_left=BATTLE_BANK), view)
                    else:
                        action = enemy(decision)
                    return decision.validate(action)

                winner = play_battle(experiment.league.engine, rosters, policies, root, seed, choose)
                outcome = .5 if winner is None else float(winner == "p1")
                trace.extend((row, outcome, rival) for row in rows)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pools", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ace-trainer", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--reply-model", type=Path, required=True)
    parser.add_argument("--lineup-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.pools < 10 or args.workers < 1:
        parser.error("Use a new output with positive counts and at least ten pools")
    torch.set_num_threads(1)
    check_hybrid()
    roots = list(range(args.seed, args.seed + args.pools))
    cards = read(CATALOG)
    pools = pools_for(cards, roots)
    initial = read(args.initial)
    verify_pools(pools, [args.initial, args.reply_model / "config.json", args.lineup_model])
    for path in (CATALOG.parents[1] / "benchmark_results" / "public-planning-v1").glob("*.json"):
        disjoint(pools, read(path).get("pools", {}))
    began = perf_counter()
    rows = {}
    for batch in parallel(collect, roots, args.workers, cards, args.ace_trainer, args.initial, args.reply_model, args.lineup_model):
        rows.update(batch)
        print(f"Search played both seats on {len(rows)}/{args.pools} paired pools", flush=True)
    data_seconds = perf_counter() - began
    check = set(Random(617223).sample(roots, args.pools // 5))
    samples, heldout = [[row for root in roots if (root in check) == validation for row in rows[root]]
                        for validation in (False, True)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def features_and_targets(group):
        return (torch.tensor([row[0] for row in group], device=device),
                torch.tensor([row[1] for row in group], device=device))

    train, target = features_and_targets(samples)
    test, actual = features_and_targets(heldout)
    scale = train.square().mean(0).sqrt().clamp_min(.25)
    prior = torch.tensor(initial["weights"], device=device)
    weights = torch.nn.Parameter(prior.clone() * scale)
    optimizer = torch.optim.AdamW([weights], lr=.012)
    started = perf_counter()
    for _ in range(220):
        loss = F.binary_cross_entropy_with_logits((train / scale) @ weights, target)
        loss += .006 * (weights - prior * scale).square().sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            weights[:8].clamp_(min=0)
            weights[8:14].clamp_(max=0)
            weights[14:22].clamp_(min=0)
            weights[22].clamp_(max=0)
    with torch.no_grad():
        refined = weights / scale
        probs = {"initial": test @ prior, "refined": test @ refined}
        report = {name: dict(nll=float(F.binary_cross_entropy_with_logits(logits, actual)),
                             accuracy=float(logits.gt(0).eq(actual.gt(.5)).float().mean()))
                  for name, logits in probs.items()}
    save(args.output, dict(version=2, pools=initial["pools"] | pools, teacher_pools=pools, validation_roots=sorted(check),
                           cards=fingerprint(cards), pressure=initial["pressure"], weights=refined.tolist(), report=report,
                           device=str(device), data_seconds=data_seconds, fit_seconds=perf_counter()-started,
                           training_rows=dict(Counter(row[2] for row in samples)),
                           validation_rows=dict(Counter(row[2] for row in heldout)), api_spend_usd=0))
    print(report, flush=True)


if __name__ == "__main__":
    main()
