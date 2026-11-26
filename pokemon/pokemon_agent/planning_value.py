"""Fit a symmetric position evaluator on completed synthetic Showdown games."""

import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
from math import sqrt
from pathlib import Path
from random import Random
from time import perf_counter

import torch
from torch.nn import functional as F

from .draft_last_pick import sample_game
from .draft_leaf import CATALOG
from .draft_league import disjoint, parallel, pools_for, read, save
from .game import BATTLE, BATTLE_BANK, DECISION_LIMIT, PREVIEW, pokemon_view
from .joint import JointPolicy
from .sim import SEATS, Showdown, _draft, fingerprint, play_battle
from .strength import EXTERNAL, ExternalPivot, verify_pools
from .tactics import SLOTS, SPREAD, _damage, _speed, load_card_profiles, load_dex, tactical, tactical_actions, tactical_candidates


STATUSES = ("brn", "par", "slp", "frz", "psn", "tox")
CONDITIONS = ("tailwind", "reflect", "lightscreen", "spikes")
MOVE_GROUPS = ({"tailwind", "trickroom"}, {"fakeout", "spore", "encore"}, {"recover", "protect", "calmmind", "nastyplot"})


def type_pressure(profiles):
    dex = load_dex()
    sets = {
        name: profile | {"species": name, "item": profile["card_item"], "ability": profile["card_ability"],
                         "types": dex["species"][name]["types"], "hp": 1.}
        for name, profile in profiles.items()
    }
    return {name: {
        other: min(1.5, max((
            _damage(move, info, attacker, defender, info["target"] in SPREAD, dex) * info["accuracy"]
            for move in attacker["card_moves"] if (info := dex["moves"].get(move, {})).get("base_power")
            and move != "fakeout"), default=0.))
        for other, defender in sets.items()}
        for name, attacker in sets.items()}


def board(raw):
    return dict(active=[pokemon_view(mon) for mon in raw["active_pokemon"]],
                reserves=[mon for slot in raw["team"] if (mon := pokemon_view(slot))],
                conditions=set(raw["side_conditions"]))


def features(ours, theirs, room, profiles, pressure):
    dex = load_dex()

    def side_features(own, foe):
        active = [mon for mon in own["active"] if mon]
        reserves = own["reserves"]
        living = active + reserves
        enemies = [mon for mon in foe["active"] if mon]
        all_enemies = enemies + foe["reserves"]
        hp = [mon["hp"] for mon in living]
        stats = [(-1 if room and stat == "spe" else 1) *
                 sum(mon.get("boosts", {}).get(stat, 0) * sqrt(mon["hp"]) for mon in active)
                 for stat in ("atk", "spa", "def", "spd", "spe")]
        states = [sum(sqrt(mon["hp"]) for mon in living if mon.get("status") == status) for status in STATUSES]
        conditions = [float(condition in own["conditions"]) for condition in CONDITIONS]
        moves = [sum(sqrt(mon["hp"]) for mon in active if group.intersection(profiles[mon["species"]]["card_moves"]))
                 for group in MOVE_GROUPS]
        speeds = [_speed(profiles[mon["species"]] | mon, own["conditions"], dex) for mon in active]
        foe_speeds = [_speed(profiles[mon["species"]] | mon, foe["conditions"], dex) for mon in enemies]
        faster = sum((left - right) * (-1 if room else 1) > 0 for left in speeds for right in foe_speeds)

        def offense(mon, targets):
            row = pressure[mon["species"]]
            return max((min(row[target["species"]], target["hp"]) + .20 * (row[target["species"]] >= target["hp"])
                        for target in targets), default=0.)

        active_danger = sum(offense(mon, enemies) for mon in active)
        bench_danger = sum(offense(mon, enemies) for mon in reserves)
        coverage = sum(max((pressure[mon["species"]][enemy["species"]] for mon in living), default=0.)
                       for enemy in all_enemies)
        return [len(living), sum(hp), sum(sqrt(value) for value in hp), sum(mon["hp"] for mon in active),
                faster, active_danger, bench_danger, coverage, *states, *stats, *conditions, *moves]

    left, right = side_features(ours, theirs), side_features(theirs, ours)
    return [a - b for a, b in zip(left, right, strict=True)]


def sample(roots, cards, checkout, pressure):
    profiles = load_card_profiles(cards, with_sets=True)
    sampled = {}
    with ExitStack() as stack:
        engine = stack.enter_context(Showdown())
        externals = {name: stack.enter_context(ExternalPivot(checkout, name, cards)) for name in EXTERNAL} if checkout else {}
        styles = ("joint", "tactical", *externals)
        for root in roots:
            rng = Random(root ^ 0x288A)
            pool, first, rng_seed = sample_game(cards, root)
            first_picks = _draft(pool, first, lambda seat, decision: rng.choice(decision.actions))
            opponent = styles[root % len(styles)]
            archive = sampled[root] = []
            for variant in (0, 1):
                actor, foe = SEATS[variant], SEATS[1 - variant]
                policies = {actor: JointPolicy(profiles), foe: externals.get(opponent) or JointPolicy(profiles)}
                if opponent in externals:
                    policies[foe].setup(root, foe, first)
                collected, pending, captured = [], {}, set()

                def choose(seat, decision, view):
                    policy = policies[seat]
                    if decision.phase == PREVIEW:
                        if isinstance(policy, JointPolicy):
                            policy.observe(decision)
                        ranked = tactical_candidates(decision, ranked=tactical_actions(decision))
                        return ranked[(root + (seat == foe)) % min(4, len(ranked))]
                    turn = decision.observation["turn"]
                    can_move = any(action[slot]["type"] == "move" for action in decision.actions for slot in SLOTS)
                    if can_move and turn not in captured:
                        slot = pending.setdefault(turn, {})
                        slot.setdefault(seat, board(view["raw"]))
                        if set(slot) == set(SEATS):
                            room = "trickroom" in decision.observation["fields"]
                            collected.append(features(slot["p1"], slot["p2"], room, profiles, pressure))
                            captured.add(turn)
                    if isinstance(policy, ExternalPivot):
                        decision = replace(decision, seconds_left=DECISION_LIMIT[BATTLE], bank_seconds_left=BATTLE_BANK)
                        return policy.choose(decision, view)
                    if seat == foe and opponent == "tactical":
                        return tactical(decision, card_profiles=profiles)
                    return policy(decision)

                winner = play_battle(engine, first_picks, policies, root, rng_seed, choose)
                result = .5 if winner is None else float(winner == "p1")
                archive.extend((row, result, opponent) for row in collected)
    return sampled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pools", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ace-trainer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if options.output.exists() or options.pools < 10 or options.workers < 1:
        parser.error("Supply a fresh output, at least ten pools, and positive workers")
    torch.set_num_threads(1)
    cards = read(CATALOG)
    roots = list(range(options.seed, options.seed + options.pools))
    pools = pools_for(cards, roots)
    verify_pools(pools, [])
    for path in (CATALOG.parents[1] / "benchmark_results" / "public-planning-v1").glob("*.json"):
        disjoint(pools, read(path).get("pools", {}))
    profiles = load_card_profiles(cards, with_sets=True)
    pressure = type_pressure(profiles)
    began = perf_counter()
    rows = {}
    for batch in parallel(sample, roots, options.workers, cards, options.ace_trainer, pressure):
        rows.update(batch)
        print(f"Generated outcomes from {len(rows)}/{options.pools} pools", flush=True)
    collect_seconds = perf_counter() - began
    split = Random(67812).sample(roots, options.pools // 5)
    verification = set(split)
    train, validation = [[row for root in roots if (root in verification) == validating for row in rows[root]]
                         for validating in (False, True)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tensors(group):
        return torch.tensor([row[0] for row in group], device=device), torch.tensor([row[1] for row in group], device=device)

    examples, targets = tensors(train)
    held, outcomes = tensors(validation)
    start = perf_counter()
    scale = examples.square().mean(0).sqrt().clamp_min(.25)
    weights = torch.nn.Parameter(torch.zeros(examples.shape[-1], device=device))
    optimizer = torch.optim.AdamW([weights], lr=.025)
    scaled = examples / scale
    for _ in range(250):
        loss = F.binary_cross_entropy_with_logits(scaled @ weights, targets) + .0006 * weights.square().sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            weights[:8].clamp_(min=0)
            weights[8:14].clamp_(max=0)
            weights[14:22].clamp_(min=0)
            weights[22].clamp_(max=0)
    fit_seconds = perf_counter() - start
    with torch.no_grad():
        weights = weights / scale
        learned = held @ weights
        linear = .60 * held[:, 0] + 1.10 * held[:, 1] + .12 * held[:, 4]
        report = {name: {"nll": float(F.binary_cross_entropy_with_logits(logits, outcomes)),
                         "accuracy": float((logits.gt(0) == outcomes.gt(.5)).float().mean())}
                  for name, logits in (("learned", learned), ("material_speed", linear))}
    save(options.output, dict(version=2, pools=pools, validation_roots=split, cards=fingerprint(cards), pressure=pressure,
                              weights=weights.detach().cpu().tolist(), report=report, device=str(device),
                              training_rows=dict(Counter(row[2] for row in train)),
                              validation_rows=dict(Counter(row[2] for row in validation)),
                              data_seconds=collect_seconds, fit_seconds=fit_seconds, api_spend_usd=0))
    print(report, flush=True)


if __name__ == "__main__":
    main()
