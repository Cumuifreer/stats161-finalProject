"""Improve the learned lineup value on rosters and strategies that it actually proposes."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
from multiprocessing import get_context
from pathlib import Path
from random import Random
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from ..battle_league import configure_torch
from ..blind85 import used_pools
from ..combination_preview import encode_action, encode_rows, encode_team, preview_snapshot, signature
from ..combination_space import CardSpace
from ..draft_last_pick import sample_game
from ..draft_leaf import CATALOG
from ..draft_league import pools_for, read, save
from ..planning_preview import diverse_deployments
from ..preview85 import neural_rank
from ..roster import DraftRanker, _load_model
from ..sim import SEATS, Showdown, _draft, fingerprint
from ..tactics import load_card_profiles, tactical_actions
from .preview_outcome import PreviewOutcome, label_battle
from .preview_qualify import LearnedDraft


SHAPE = (4, 3)


def candidates(decision, space, model, opposing, rng, number):
    ranked = tactical_actions(decision)
    first = ranked[0]
    predicted = neural_rank(decision, opposing, space, model)
    response = [first]
    seen = {signature(first)}
    extras = diverse_deployments(ranked, first, 10)[1:]
    rng.shuffle(extras)
    ordered = [predicted[0], predicted[1], *extras, *predicted[2:]] if number == 4 else [predicted[0], *extras]
    for choice in ordered:
        distinct = signature(choice)
        if distinct not in seen:
            seen.add(distinct)
            response.append(choice)
        if len(response) == number:
            break
    return response, predicted[0]


def generate_shard(roots, cards, checkpoint, output):
    configure_torch()
    config = read(checkpoint / "config.json")
    model = PreviewOutcome(**config["network"]).eval()
    model.load_state_dict(torch.load(checkpoint / "state_dict.pt", map_location="cpu", weights_only=True))
    profiles = load_card_profiles(cards, with_sets=True)
    space = CardSpace(cards, profiles)
    projection = DraftRanker(cards, *_load_model(Path("checkpoints/roster-v1/expanded"), "cpu"))
    rows, counts, start = [], Counter(), perf_counter()
    with Showdown() as engine:
        for root in roots:
            pool, first, initial = sample_game(cards, root)
            actor, foe = SEATS if root % 2 == 0 else SEATS[::-1]
            rng = Random(f"learned-lineup-population/v1/{root}")
            draft = {seat: LearnedDraft(profiles, cards, space, model, projection) for seat in SEATS}
            rosters = _draft(pool, first, lambda seat, decision: draft[seat](decision))
            for seat in SEATS:
                counts["draft_network_changes"] += draft[seat].changed
            previews = preview_snapshot(rosters, initial, engine)
            tactical = {seat: tactical_actions(previews[seat]) for seat in SEATS}
            picks, highest = {}, {}
            for own, other, number in ((actor, foe, SHAPE[0]), (foe, actor, SHAPE[1])):
                replies = diverse_deployments(tactical[other], tactical[other][0], 4)
                picks[own], highest[own] = candidates(previews[own], space, model, replies, rng, number)
            if tuple(map(len, (picks[actor], picks[foe]))) != SHAPE:
                raise RuntimeError("Insufficient legal Preview choices")
            board = previews[actor].observation
            teams = [encode_team(space, board, name) for name in ("team", "opponent_team")]
            roles, features = encode_rows(space, teams, *([encode_action(space, action) for action in picks[seat]]
                                                           for seat in (actor, foe)))
            seeds = [[rng.randrange(65536) for _ in range(4)] for _ in range(2)]
            opponent_styles = (("forecast-replies", "forecast"), ("joint", "forecast-replies"))
            outcomes = []
            for own in picks[actor]:
                for reply in picks[foe]:
                    options = {actor: own, foe: reply}
                    results = [label_battle(engine, rosters, actor, options, seed, order, profiles)
                               for seed, order in zip(seeds, opponent_styles, strict=True)]
                    counts["both_learned_previews"] += 2 * (signature(own) == signature(highest[actor]) and
                                                             signature(reply) == signature(highest[foe]))
                    outcomes.append(sum(results) / len(results))
            rows.extend((root, teams, role, feature, result) for role, feature, result in zip(roles, features, outcomes, strict=True))
            counts["terminal_battles"] += len(outcomes) * 2
    fields = ("root", "teams", "roles", "features", "target")
    np.savez_compressed(output, **dict(zip(fields, map(np.asarray, zip(*rows, strict=True)), strict=True)))
    return dict(counts, pools=len(roots), positions=len(rows), worker_seconds=perf_counter() - start)


def group(paths, root, device):
    release = read(root / "pools.json")
    rows = []
    for path in paths:
        with np.load(path, allow_pickle=False) as replay:
            rows.append({key: replay[key] for key in replay.files})
    labels = np.concatenate([row["root"] for row in rows])
    held = np.asarray([int(release["pools"][str(value)][:8], 16) % 8 == 0 for value in labels])
    row_data = [torch.tensor(np.concatenate([row[key] for row in rows]), dtype=dtype, device=device)
                for key, dtype in (("teams", torch.long), ("roles", torch.long), ("features", torch.float32), ("target", torch.float32))]
    return [[field[torch.from_numpy(part).to(device)].unflatten(0, (-1, SHAPE[0] * SHAPE[1])) for field in row_data]
            for part in (~held, held)]


def run(model, rows, batches):
    for index in batches:
        tensor, roles, features, target = [field[index] for field in rows]
        predictions = model(tensor.flatten(0, 1), roles.flatten(0, 1), features.flatten(0, 1)).reshape(target.shape)
        yield predictions.unflatten(1, SHAPE), target.unflatten(1, SHAPE)


@torch.inference_mode()
def assess(model, data):
    predictions, targets = zip(*run(model, data, torch.arange(len(data[0]), device=data[0].device).split(64)), strict=True)
    score, outcome = torch.cat(predictions), torch.cat(targets)
    average = score.sigmoid().mean(2)
    choice = (.85 * average + .15 * score.sigmoid().min(2).values).argmax(1, keepdim=True)
    actual = outcome.mean(2)
    return dict(nll=float(F.binary_cross_entropy_with_logits(score, outcome)),
                lineup_gain=float((actual.gather(1, choice)[:, 0] - actual[:, 0]).mean()))


def fit(args, report, paths):
    configure_torch()
    torch.manual_seed(991178)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    config = read(args.parent / "config.json")
    model = PreviewOutcome(**config["network"]).to(device)
    model.load_state_dict(torch.load(args.parent / "state_dict.pt", map_location=device, weights_only=True))
    training, validation = group(paths, args.data, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.00016, weight_decay=.07, fused=True)
    save(args.output / "config.json", dict(config, parent_sha256=sha256((args.parent / "state_dict.pt").read_bytes()).hexdigest(),
                                         file_hashes={str(path): sha256(path.read_bytes()).hexdigest() for path in paths}))
    initial = assess(model.eval(), validation)
    best, epoch_best, history, updates = float("inf"), 0, [], 0
    started = perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for prediction, actual in run(model, training, torch.randperm(len(training[0]), device=device).split(42)):
            bce = F.binary_cross_entropy_with_logits(prediction, actual)
            goal = (actual.mean(2) / .13).softmax(dim=1)
            relative = -(goal * (prediction.mean(2) / .65).log_softmax(dim=1)).sum(1).mean()
            error = bce + .24 * relative
            optimizer.zero_grad(set_to_none=True)
            error.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
            updates += 1
        result = dict(epoch=epoch, **assess(model.eval(), validation))
        history.append(result)
        quality = result["nll"] - .24 * result["lineup_gain"]
        if quality < best:
            best, epoch_best = quality, epoch
            torch.save(model.state_dict(), args.output / "state_dict.pt")
        print(result, flush=True)
        if epoch - epoch_best >= 6:
            break
    save(args.output / "training.json", dict(initial=initial, epochs=history, best_epoch=epoch_best, source=report,
                                            train_pools=len(training[0]), held_pools=len(validation[0]), updates=updates,
                                            fit_seconds=perf_counter() - started, gpu=torch.cuda.get_device_name()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pools", type=int, required=True)
    parser.add_argument("--prior-workspace", type=Path, action="append", default=[])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=28)
    args = parser.parse_args()
    if args.data.exists() or args.output.exists() or min(args.pools, args.workers, args.epochs) < 1:
        parser.error("Use fresh paths and positive run sizes")
    cards = read(CATALOG)
    roots = list(range(args.seed, args.seed + args.pools))
    pools = pools_for(cards, roots)
    used = used_pools(cards)
    for workspace in args.prior_workspace:
        used.update(used_pools(cards, workspace))
    if not used.isdisjoint(pools.values()) or read(args.parent / "config.json")["catalog"] != fingerprint(cards):
        raise ValueError("Preview populations need unused pools and the original catalog")
    save(args.data / "pools.json", dict(pools=pools, catalog=fingerprint(cards), source=sha256(Path(__file__).read_bytes()).hexdigest()))
    counts, started = Counter(), perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as workers:
        tasks = [workers.submit(generate_shard, roots[i:i + 32], cards, args.parent, args.data / f"{roots[i]}.npz")
                 for i in range(0, len(roots), 32)]
        for task in as_completed(tasks):
            counts.update(task.result())
            print(dict(counts), flush=True)
    report = dict(counts=dict(counts), seconds=perf_counter() - started)
    save(args.data / "completed.json", report)
    fit(args, report, sorted(args.data.glob("*.npz")))


if __name__ == "__main__":
    main()
