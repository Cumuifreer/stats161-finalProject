"""Distill the legal choices of strong policies recorded in development matches."""

import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from ..battle_league import Encoder, configure_torch, load_model, make_batch
from ..battle_value import decisions
from ..draft_leaf import CATALOG
from ..draft_league import read, save
from ..game import BATTLE, PREVIEW
from ..joint import JointPolicy
from ..public_planning import _diverse, _ordered
from ..sim import FORMAT, SEATS, Showdown
from ..tactics import SLOTS, load_card_profiles, tactical_actions


WIDTH = 32


class SearchExamples(Encoder):
    def encode_menu(self, decision, policy):
        coarse = tactical_actions(decision)
        joint = policy.rank(decision)
        ranked = [action for action, _ in joint]
        menu = _diverse(_ordered(ranked, coarse), min(WIDTH, len(decision.actions)))
        base = self(decision, policy.history, joint)
        features = self.base.encode(decision, history=policy.history)
        by_index = dict(zip(features.action_indices, features.actions, strict=True))
        scores = [details.get("score", 0.) for _, details in joint]
        rows = []
        for index, action in enumerate(menu):
            local = scores[ranked.index(action)] if action in ranked else scores[-1]
            switching = [self.card_ids.get(action[slot].get("species"), 0) / len(self.card_ids) for slot in SLOTS]
            rows.append((*by_index[decision.actions.index(action)], local, local - scores[0], index / (WIDTH - 1), *switching))
        rows.extend([(0.,) * len(rows[0])] * (WIDTH - len(rows)))
        return (base[0], rows, base[2], base[3], [i < len(menu) for i in range(WIDTH)]), menu


def replay(engine, cards, profiles, encoder, game, count):
    students = {seat: JointPolicy(profiles) for seat in SEATS}
    rows = {seat: [] for seat in SEATS}
    roster = {seat: [cards[card] for card in game["roster"][seat]] for seat in SEATS}
    current = engine.call("start", format=FORMAT, seed=game["battle_seed"], rosters=roster)
    trace = iter(game["trace"])
    verified = False
    try:
        for _ in range(500):
            if current["ended"]:
                verified = current["winner"] == game["winner"] and next(trace, None) is None
                break
            for seat, decision in decisions(current):
                policy = students[seat]
                policy.observe(decision)
                if decision.phase == PREVIEW:
                    action = decision.validate(game["lineups"][seat])
                else:
                    event = next(trace)
                    if decision.phase != BATTLE or seat != event["seat"] or decision.observation["turn"] != event["turn"]:
                        raise ValueError("Trace and observations disagree")
                    action = decision.validate(event["action"])
                    if len(decision.actions) > 1:
                        features, menu = encoder.encode_menu(decision, policy)
                        count["candidate_positions"] += 1
                        if action in menu:
                            rows[seat].append((*features, menu.index(action), decision.observation["turn"]))
                        else:
                            count["outside_candidate_menu"] += 1
                    policy.record(decision, action)
                current = engine.call("choose", seat=seat, action=action)
    except (StopIteration, ValueError):
        pass
    if not verified:
        count["excluded_trace"] += 1
        return {}
    return rows


def collect(sources, cards):
    profiles = load_card_profiles(cards, with_sets=True)
    encoder = SearchExamples(profiles)
    cards = {card["card_id"]: card for card in cards}
    counts = Counter()
    fit, held = [], []
    source_hashes = {}
    known = set()
    with Showdown() as engine:
        for path in sources:
            if read(path.with_name("frozen.json"))["blind"]:
                raise ValueError("Test-suite positions cannot be replayed into training")
            source = read(path)
            source_hashes[str(path)] = sha256(path.read_bytes()).hexdigest()
            for root, pair in source["records"].items():
                pool = source["pools"][root]
                split = held if int(pool[:10], 16) % 6 == 0 else fit
                for games in pair.values():
                    for game in games:
                        if not game["trace"]:
                            continue
                        match = sha256(str((game["battle_seed"], game["roster"], game["lineups"], [row["action"] for row in game["trace"]])).encode()).digest()
                        if match in known:
                            continue
                        known.add(match)
                        positions = replay(engine, cards, profiles, encoder, game, counts)
                        if not positions:
                            continue
                        counts["verified_games"] += 1
                        for seat, episode in positions.items():
                            label = .5 if game["winner"] is None else float(game["winner"] == seat)
                            identity = counts["verified_games"] * 2 + SEATS.index(seat)
                            split.extend((*features, move, move, label, [-1.] * WIDTH, identity, turn)
                                         for *features, move, turn in episode)
            print(path.parent.name, dict(counts), flush=True)
    fields = ("state", "actions", "cards", "mons", "legal", "teacher", "behavior", "outcome", "branches", "root", "turn")
    data = [{name: np.asarray(values, dtype=np.float32 if name in {"state", "actions", "mons", "outcome", "branches"}
                             else np.bool_ if name == "legal" else np.int32)
             for name, values in zip(fields, zip(*examples, strict=True), strict=True)} for examples in (fit, held)]
    return data, dict(counts), source_hashes


def metrics(model, rows):
    model.eval()
    sums = torch.zeros(3, device=rows[0].device)
    with torch.inference_mode():
        for packets in zip(*(field.split(256) for field in rows), strict=True):
            *batch, target, behavior, win, _ = packets
            logits, q, value = model(*batch)
            sums += torch.stack((F.cross_entropy(logits, target, reduction="sum"), (logits.argmax(1) == target).sum(),
                                 F.binary_cross_entropy_with_logits(value, win, reduction="sum")))
    return dict(zip(("policy_nll", "policy_top1", "value_nll"), (sums / len(rows[0])).tolist(), strict=True))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source", nargs="+", type=Path, required=True)
    cli.add_argument("--initialize", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--epochs", type=int, default=20)
    args = cli.parse_args()
    if args.output.exists():
        cli.error("Use a fresh output")
    configure_torch()
    torch.manual_seed(57913)
    torch.set_float32_matmul_precision("high")
    cards = read(CATALOG)
    started = perf_counter()
    data, counts, sources = collect(args.source, cards)
    device = torch.device("cuda")
    model = load_model(args.initialize, cards).to(device)
    train, held = [make_batch(rows, device, model.history_steps) for rows in data]
    save(args.output / "config.json", read(args.initialize / "config.json"))
    references = dict(sources=sources, counts=counts, train_positions=len(train[0]), held_positions=len(held[0]))
    save(args.output / "examples.json", references)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.00017, weight_decay=.025, fused=True)
    best, best_epoch, reports = float("inf"), 0, []
    initial = metrics(model, held)
    began = perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for selection in torch.randperm(len(train[0]), device=device).split(256):
            *batch, target, behavior, win, _ = [field[selection] for field in train]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, q, value = model(*batch)
                objective = (F.cross_entropy(logits, target) + .32 * F.binary_cross_entropy_with_logits(value, win)
                             + .23 * F.binary_cross_entropy_with_logits(q.gather(1, behavior[:, None]).squeeze(1), win))
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
        result = metrics(model, held)
        reports.append(result)
        print(dict(epoch=epoch, **result), flush=True)
        if result["policy_nll"] < best:
            best, best_epoch = result["policy_nll"], epoch
            torch.save(model.state_dict(), args.output / "state_dict.pt")
        if epoch - best_epoch >= 4:
            break
    save(args.output / "training.json", dict(initial=initial, epochs=reports, best_epoch=best_epoch,
                                            training_seconds=perf_counter() - began, seconds=perf_counter() - started,
                                            parameters=sum(p.numel() for p in model.parameters()), cuda=torch.cuda.get_device_name()))


if __name__ == "__main__":
    main()
