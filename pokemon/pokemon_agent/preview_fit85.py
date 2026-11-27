"""Learn which lineup proposals survive counterfactual local games."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from hashlib import sha256
import json
from multiprocessing import get_context
from pathlib import Path
from random import Random
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from .combination_preview import ComboValue, encode_action, encode_rows, encode_team, freeze, preview_snapshot
from .combination_space import CardSpace
from .draft_coverage import Coverage
from .draft_last_pick import sample_game
from .draft_leaf import CATALOG
from .draft_league import pools_for, read, save
from .lab85 import MODELS, SEALED
from .lineup import index_cards
from .matchup_preview import MatchupPreview
from .planning_bench import configure_worker
from .planning_preview import diverse_deployments
from .preview85 import PreviewWorker, reverse_view, unique
from .sim import FORMAT, SEATS, _draft
from .strength import verify_pools
from .tactics import tactical_actions


def collect(roots, cards, checkout):
    partitions = {}
    torch.set_num_threads(1)
    with ExitStack() as stack:
        experiment = Coverage(stack, cards)
        indexed = index_cards(cards)
        worker = stack.enter_context(PreviewWorker())
        configure_worker(worker, cards, MODELS / "value-v2.json")
        matchup = stack.enter_context(MatchupPreview(checkout, indexed, budget=.55))
        space = CardSpace(cards, experiment.profiles)
        for root in roots:
            partitions[root] = []
            pool, first, seed = sample_game(cards, root)
            style = ("champion", "pairwise") if root % 3 == 0 else ("pairwise", "pairwise")
            rosters = _draft(pool, first, lambda seat, decision: experiment.choose(decision, style[SEATS.index(seat)]))
            snapshots = preview_snapshot(rosters, seed, experiment.league.engine)
            rng = Random(f"preview-fit85/{root}")
            for seat in SEATS:
                decision = snapshots[seat]
                reverse = reverse_view(decision, indexed)
                ranked, replies = tactical_actions(decision), tactical_actions(reverse)
                proposed = matchup.proposals(decision, top=5)
                opposing = matchup.proposals(reverse, top=4)
                coverage = diverse_deployments(ranked, ranked[0], 15)
                sampled = rng.sample(unique(decision.actions), 8)
                options = unique([ranked[0], *proposed, *coverage[:5], *sampled, *coverage])[0:14]
                opponents = unique([*opposing[:2], replies[0], *opposing[2:], *diverse_deployments(replies, replies[0], 6)])[:4]
                draws = [[rng.randrange(65536) for _ in range(4)] for _ in range(4)]
                outcomes = worker.call("compare", options=options, opponents=opponents, weights=[1.] * len(opponents),
                                       seeds=draws, format=FORMAT, stages=[dict(survivors=len(options), depth=17, start=0,
                                       samples=len(draws), opponentCount=len(opponents))], budget_ms=90000, edge=0.)
                if outcomes["timedOut"] or len(outcomes["rows"]) != len(options):
                    raise TimeoutError("Missing complete groups of paired Preview labels")
                team = [encode_team(space, decision.observation, key) for key in ("team", "opponent_team")]
                roles, features = encode_rows(space, team, [encode_action(space, action) for action in options],
                                              [encode_action(space, action) for action in opponents])
                targets = [[sum(draw) / len(draw) for draw in row["values"]]
                           for row in sorted(outcomes["rows"], key=lambda row: row["index"])]
                partitions[root].append(dict(team=team, roles=roles, features=np.asarray(features, dtype=np.float32).tolist(),
                                             targets=targets,
                                             rollouts=outcomes["calls"], terminal=outcomes["terminals"]))
    return partitions


def batch(rows, device):
    teams, roles, features, targets = [], [], [], []
    for row in rows:
        target = np.asarray(row["targets"], dtype=np.float32)
        teams.append(np.broadcast_to(np.asarray(row["team"]), (target.size, 2, 6)))
        roles.extend(row["roles"])
        features.extend(row["features"])
        targets.append(target.ravel())
    return (torch.tensor(np.concatenate(teams), device=device), torch.tensor(np.asarray(roles), device=device),
            torch.tensor(np.asarray(features, dtype=np.float32), device=device),
            torch.tensor(np.concatenate(targets), device=device))


def fit(rows, cards, seed, output, pools):
    torch.manual_seed(seed)
    roots = sorted(rows)
    Random(seed ^ 0x6BF).shuffle(roots)
    validation_roots = set(roots[::5])
    train = [row for root in roots if root not in validation_roots for row in rows[root]]
    held = [row for root in roots if root in validation_roots for row in rows[root]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training, validation = batch(train, device), batch(held, device)
    width = 96
    network = ComboValue(len(cards), width).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=.0015, weight_decay=.02)
    best = None
    nll = float("inf")
    start = perf_counter()
    for _ in range(90):
        network.train()
        for group in torch.randperm(len(training[-1]), device=device).split(512):
            teams, roles, features, targets = (column[group] for column in training)
            predicted = network(teams, roles, features)
            loss = F.binary_cross_entropy_with_logits(predicted, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        network.eval()
        with torch.no_grad():
            scores = network(*validation[:-1])
            value = float(F.binary_cross_entropy_with_logits(scores, validation[-1]))
            if value < nll:
                nll = value
                best = {name: tensor.detach().cpu().clone() for name, tensor in network.state_dict().items()}
    network.load_state_dict(best)
    gains = []
    with torch.inference_mode():
        for row in held:
            inputs = batch([row], device)
            expected = network(*inputs[:-1]).sigmoid().reshape(len(row["targets"]), -1).mean(1)
            outcomes = inputs[-1].reshape(len(row["targets"]), -1).mean(1)
            gains.append(float(outcomes[int(expected.argmax())] - outcomes[0]))
    elapsed = perf_counter() - start
    torch.save(best, output / "state_dict.pt")
    save(output / "config.json", dict(provenance=freeze(cards), width=width, pools=pools, validation_roots=roots[::5]))
    return dict(training_device=str(device), fit_seconds=elapsed, held_cross_entropy=nll,
                held_label_gain=float(np.mean(gains)), training_rows=len(training[-1]), held_rows=len(validation[-1]),
                parameters=sum(tensor.numel() for tensor in best.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pools", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--ace-trainer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.pools < 5:
        parser.error("Provide a fresh output with at least five training pools")
    torch.set_num_threads(1)
    cards = read(CATALOG)
    roots = list(range(args.seed, args.seed + args.pools))
    pools = pools_for(cards, roots)
    verify_pools(pools, [*SEALED, *Path("benchmark_results/research85").glob("*.json"),
                         *Path("benchmark_results/research85-smoke").glob("*.json")])
    time = perf_counter()
    rows = {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as workers:
        jobs = [workers.submit(collect, roots[i:i + 4], cards, args.ace_trainer) for i in range(0, len(roots), 4)]
        for job in as_completed(jobs):
            rows.update(job.result())
            print(f"Labeled {len(rows)}/{args.pools} training pools", flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "labels.json", dict(pools=pools, rows=rows))
    result = fit(rows, cards, args.seed, args.output, pools)
    result.update(seconds=perf_counter() - time, api_spend_usd=0,
                  games=sum(row["rollouts"] for pair in rows.values() for row in pair),
                  terminal=sum(row["terminal"] for pair in rows.values() for row in pair),
                  label_program=sha256(Path(__file__).read_bytes()).hexdigest())
    save(args.output / "metrics.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
