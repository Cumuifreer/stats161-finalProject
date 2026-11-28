"""Fit a symmetric nonlinear position value to synthetic complete Showdown games."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from hashlib import sha256
import json
from multiprocessing import get_context
from pathlib import Path
from random import Random
import subprocess
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .draft_coverage import Coverage
from .draft_last_pick import sample_game
from .draft_leaf import CATALOG
from .draft_league import pools_for, read, save
from .lab85 import MODELS, SEALED
from .lineup import index_cards
from .planning_preview import diverse_deployments
from .preview85 import independent_draws
from .combination_preview import preview_snapshot
from .sim import FORMAT, SEATS, Showdown, _draft, fingerprint
from .strength import verify_pools
from .tactics import tactical_actions


class LabelWorker(Showdown):
    def __enter__(self):
        self.process = subprocess.Popen(["node", str(Path(__file__).with_name("value_data85.cjs"))],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        return self


class PositionModel(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.direction = nn.Sequential(nn.Linear(2 * size, 72), nn.SiLU(), nn.Linear(72, 36), nn.SiLU(), nn.Linear(36, 1))

    def forward(self, board, scale):
        difference = (board[:, 0] - board[:, 1]) / scale
        total = (board[:, 0] + board[:, 1]) / scale
        positive = self.direction(torch.cat((difference, total), dim=-1))
        negative = self.direction(torch.cat((-difference, total), dim=-1))
        return (positive - negative).squeeze(-1)


def collect(roots, cards, pressure, draws):
    counts = Counter()
    result = {}
    torch.set_num_threads(1)
    with ExitStack() as stack:
        experiment = Coverage(stack, cards)
        worker = stack.enter_context(LabelWorker())
        worker.call("configure", pressure=pressure)
        index = index_cards(cards)
        for root in roots:
            pool, first, seed = sample_game(cards, root)
            policy = ("pairwise", "champion") if root % 2 else ("champion", "pairwise")
            roster = _draft(pool, first, lambda seat, decision: experiment.choose(decision, policy[SEATS.index(seat)]))
            snapshots = preview_snapshot(roster, seed, experiment.league.engine)
            options = []
            for seat in SEATS:
                ranked = tactical_actions(snapshots[seat])
                options.append(diverse_deployments(ranked, ranked[0], 11))
            rng = Random(f"position-v3/{root}")
            simulated = []
            seeds = independent_draws(snapshots["p1"], "position-labels-v3", draws)
            for entry, hypothetical_seed in enumerate(seeds):
                teams = []
                for choices in options:
                    action = rng.choice(choices if entry % 3 == 0 else choices[:5])
                    order = [*action["leads"], *(mon for mon in action["bring"] if mon not in action["leads"])]
                    teams.append([index[mon] for mon in order])
                simulated.append(dict(teams=teams, seed=hypothetical_seed, style=entry % 3))
            outcome = worker.call("label", games=simulated, format=FORMAT)
            counts.update(simulated=draws, node_cpu_us=sum(outcome["cpu"].values()))
            rows = result[root] = []
            for item in outcome["samples"]:
                winner = item["winner"]
                if winner is None:
                    continue
                counts["completed"] += 1
                target = {"p1": 1., "p2": 0., "draw": .5}[winner]
                rows.extend((board, target) for board in item["states"])
            counts["positions"] += len(rows)
    return result, counts


def materialize(rows, device):
    features, targets = zip(*rows, strict=True)
    return (torch.tensor(np.asarray(features, dtype=np.float32), device=device),
            torch.tensor(targets, dtype=torch.float32, device=device))


def optimize(dataset, pressure, pools, cards, seed):
    torch.manual_seed(seed)
    roots = sorted(dataset)
    Random(seed).shuffle(roots)
    validation = set(roots[::5])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, target = materialize([row for root in roots if root not in validation for row in dataset[root]], device)
    dev, win = materialize([row for root in roots if root in validation for row in dataset[root]], device)
    scale = torch.sqrt(train.square().mean((0, 1))).clamp_min(.15)
    model = PositionModel(train.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0015, weight_decay=.015)
    best, best_nll = None, float("inf")
    started = perf_counter()
    for _ in range(75):
        for idx in torch.randperm(len(train), device=device).split(512):
            loss = F.binary_cross_entropy_with_logits(model(train[idx], scale), target[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            held = model(dev, scale)
            loss = float(F.binary_cross_entropy_with_logits(held, win))
            if loss < best_nll:
                best = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
                best_nll = loss
    model.load_state_dict(best)
    with torch.inference_mode():
        expected = model(dev, scale).sigmoid()
        brier = float((expected - win).square().mean())
        antisymmetry_error = float((expected + model(dev.flip(1), scale).sigmoid() - 1).abs().max())
    elapsed = perf_counter() - started
    layers = [dict(weight=best[f"direction.{i}.weight"].tolist(), bias=best[f"direction.{i}.bias"].tolist()) for i in (0, 2, 4)]
    return dict(version=3, cards=fingerprint(cards), pressure=pressure, scale=scale.tolist(), layers=layers, pools=pools,
                validation_roots=sorted(validation), fit_seconds=elapsed, device=str(device), held_nll=best_nll,
                held_brier=brier, antisymmetry_error=antisymmetry_error, train_positions=len(train), held_positions=len(dev),
                parameters=sum(x.numel() for x in best.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pools", type=int, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--games-per-pool", type=int, default=48)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.pools < 5 or min(args.workers, args.games_per_pool) < 1:
        parser.error("Provide a fresh output, at least five pools, and positive counts")
    started = perf_counter()
    cards = read(CATALOG)
    roots = list(range(args.seed, args.seed + args.pools))
    pools = pools_for(cards, roots)
    verify_pools(pools, [*SEALED, *Path("benchmark_results/research85").glob("*.json"),
                         *Path("checkpoints/research85").glob("*/config.json")])
    pressure = read(MODELS / "value-v2.json")["pressure"]
    data, counts = {}, Counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as workers:
        jobs = [workers.submit(collect, roots[i:i + 5], cards, pressure, args.games_per_pool)
                for i in range(0, len(roots), 5)]
        for job in as_completed(jobs):
            rows, stats = job.result()
            data.update(rows)
            counts.update(stats)
            print(f"Position labels from {len(data)}/{len(roots)} pools", flush=True)
    report = optimize(data, pressure, pools, cards, args.seed)
    report.update(counts=counts, wall_seconds=perf_counter() - started, api_spend_usd=0,
                  label_program=sha256(Path(__file__).read_bytes()).hexdigest())
    save(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key not in {"pressure", "scale", "layers", "pools"}}), flush=True)


if __name__ == "__main__":
    main()
