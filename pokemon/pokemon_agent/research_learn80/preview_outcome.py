"""Learn terminal lineup value from paired full battles on diverse drafted rosters."""

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
from torch import nn
from torch.nn import functional as F

from ..battle_league import configure_torch
from ..battle_value import opponent
from ..blind85 import used_pools
from ..combination_preview import encode_action, encode_rows, encode_team, preview_snapshot
from ..combination_space import CardSpace
from ..draft_last_pick import sample_game
from ..draft_leaf import CATALOG
from ..draft_league import pools_for, read, save
from ..game import PREVIEW
from ..planning_preview import diverse_deployments
from ..sim import SEATS, Showdown, _draft, fingerprint, play_battle
from ..tactics import load_card_profiles, tactical_actions


class PreviewOutcome(nn.Module):
    def __init__(self, cards, hidden=128, depth=3):
        super().__init__()
        self.cards = nn.Embedding(cards, hidden)
        self.roles = nn.Embedding(3, hidden)
        self.sides = nn.Parameter(torch.randn(2, 1, hidden) * .02)
        block = nn.TransformerEncoderLayer(hidden, 4, hidden * 3, dropout=.12, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.interactions = nn.TransformerEncoder(block, depth, enable_nested_tensor=False)
        self.matchup = nn.Linear(12, hidden)
        self.read = nn.Sequential(nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, teams, roles, matchup):
        # Sharing the directed network makes inference antisymmetric under a seat swap.
        tokens = self.cards(teams) + self.roles(roles)
        directional = torch.cat((tokens, tokens.flip(1))) + self.sides
        board = self.interactions(directional.flatten(1, 2)).reshape(directional.shape).mean(2)
        traits = torch.cat((matchup, matchup.flip(1)))
        value, reverse = self.read((board + self.matchup(traits)).flatten(1, 2)).flatten().chunk(2)
        return value - reverse


def label_battle(engine, rosters, seat, choices, seed, styles, profiles):
    policies = {player: opponent(profiles, style) for player, style in zip(SEATS, styles, strict=True)}

    def choose(player, decision, _):
        policy = policies[player]
        if decision.phase == PREVIEW:
            policy.observe(decision)
            return decision.validate(choices[player])
        return decision.validate(policy(decision))

    winner = play_battle(engine, rosters, policies, 0, seed, choose)
    return float(winner == seat) if winner else .5


def collect(roots, cards, output):
    configure_torch()
    profiles = load_card_profiles(cards, with_sets=True)
    space = CardSpace(cards, profiles)
    count = Counter()
    rows, started = [], perf_counter()
    with Showdown() as engine:
        for root in roots:
            pool, first, initial = sample_game(cards, root)
            actor, foe = (SEATS if root % 2 else SEATS[::-1])
            rng = Random(f"lineup-league/v1/{root}")

            def pick(seat, decision):
                spread = (1, 2, 3)[(root + SEATS.index(seat)) % 3]
                choices = tactical_actions(decision)
                return choices[rng.randrange(min(spread, len(choices)))]

            rosters = _draft(pool, first, pick)
            previews = preview_snapshot(rosters, initial, engine)
            ranked = {seat: tactical_actions(previews[seat]) for seat in SEATS}
            viable = {seat: diverse_deployments(menu, menu[0], 10) for seat, menu in ranked.items()}
            actions = [viable[actor][0], *rng.sample(viable[actor][1:], 3)]
            replies = [viable[foe][0], rng.choice(viable[foe][1:])]
            board = previews[actor].observation
            teams = [encode_team(space, board, key) for key in ("team", "opponent_team")]
            roles, features = encode_rows(space, teams, [encode_action(space, action) for action in actions],
                                          [encode_action(space, reply) for reply in replies])
            seeds = [[rng.randrange(65536) for _ in range(4)] for _ in range(2)]
            labels = []
            for action in actions:
                for reply in replies:
                    choices = {actor: action, foe: reply}
                    labels.append([label_battle(engine, rosters, actor, choices, seed, styles, profiles)
                                   for seed, styles in zip(seeds, (("joint", "forecast"), ("forecast-replies", "joint")), strict=True)])
            count["terminal_battles"] += 2 * len(labels)
            rows.extend((root, teams, role, feature, sum(label) / len(label))
                        for role, feature, label in zip(roles, features, labels, strict=True))
    arrays = dict(zip(("root", "teams", "roles", "features", "target"),
                      (np.asarray(x) for x in zip(*rows, strict=True)), strict=True))
    np.savez_compressed(output, **arrays)
    return dict(count, pools=len(roots), positions=len(rows), worker_seconds=perf_counter() - started)


def generate(args, cards):
    roots = list(range(args.seed, args.seed + args.pools))
    pools = pools_for(cards, roots)
    previous = used_pools(cards)
    for workspace in args.prior_workspace:
        previous.update(used_pools(cards, workspace))
    if not previous.isdisjoint(pools.values()):
        raise ValueError("Lineup training requires unused draft pools")
    save(args.output / "pools.json", dict(pools=pools, catalog=fingerprint(cards), source=sha256(Path(__file__).read_bytes()).hexdigest()))
    started = perf_counter()
    counts = Counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as workers:
        jobs = [workers.submit(collect, roots[start:start + 64], cards, args.output / f"{roots[start]}.npz")
                for start in range(0, len(roots), 64)]
        for job in as_completed(jobs):
            counts.update(job.result())
            print(dict(counts), flush=True)
    save(args.output / "completed.json", dict(counts=dict(counts), seconds=perf_counter() - started))


def fit(args, cards):
    configure_torch()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    manifest = read(args.data / "pools.json")
    generated = read(args.data / "completed.json")
    if manifest["catalog"] != fingerprint(cards):
        raise ValueError("Lineup dataset uses a different card catalog")
    files = sorted(args.data.glob("*.npz"))
    rows = []
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            rows.append({key: archive[key] for key in archive.files})
    roots = np.concatenate([row["root"] for row in rows])
    held = np.asarray([int(manifest["pools"][str(root)][:8], 16) % 8 == 0 for root in roots])
    if not held.any() or held.all():
        raise ValueError("Lineup training needs both train and held-out pools")
    device = torch.device("cuda")
    features = [torch.tensor(np.concatenate([row[key] for row in rows]), dtype=dtype, device=device)
                for key, dtype in (("teams", torch.long), ("roles", torch.long), ("features", torch.float32), ("target", torch.float32))]
    training, validation = [[field[torch.from_numpy(split).to(device)] for field in features] for split in (~held, held)]
    network = dict(cards=len(cards), hidden=128, depth=3)
    model = PreviewOutcome(**network).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.04, fused=True)
    save(args.output / "config.json", dict(network=network, catalog=manifest["catalog"], file_hashes={str(path): sha256(path.read_bytes()).hexdigest() for path in files}))
    report = []
    best = float("inf")
    best_epoch = 0
    start = perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for index in torch.randperm(len(training[0]), device=device).split(512):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                objective = F.binary_cross_entropy_with_logits(model(*(field[index] for field in training[:-1])), training[-1][index])
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            estimates = torch.cat([model(*packet) for packet in zip(*(field.split(1024) for field in validation[:-1]), strict=True)])
            nll = float(F.binary_cross_entropy_with_logits(estimates, validation[-1]))
            values = estimates.reshape(-1, 4, 2).mean(2)
            outcome = validation[-1].reshape(-1, 4, 2).mean(2)
            trained_lineup = outcome.gather(1, values.argmax(1, keepdim=True)).squeeze(1)
            delta = float((trained_lineup - outcome[:, 0]).mean())
        report.append(dict(epoch=epoch, held_nll=nll, held_outcome_gain=delta))
        print(report[-1], flush=True)
        if nll < best:
            best, best_epoch = nll, epoch
            torch.save(model.state_dict(), args.output / "state_dict.pt")
        if epoch - best_epoch >= 5:
            break
    save(args.output / "training.json", dict(training=report, best_epoch=best_epoch, training_seconds=perf_counter()-start,
                                            parameters=sum(parameter.numel() for parameter in model.parameters()),
                                            train_positions=int((~held).sum()), held_positions=int(held.sum()), source=generated,
                                            gpu=torch.cuda.get_device_name()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gather = commands.add_parser("generate")
    gather.add_argument("--pools", type=int, required=True)
    gather.add_argument("--workers", type=int, default=6)
    gather.add_argument("--prior-workspace", type=Path, action="append", default=[])
    train = commands.add_parser("fit")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=30)
    for sub in (gather, train):
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    cards = read(CATALOG)
    (generate if args.command == "generate" else fit)(args, cards)


if __name__ == "__main__":
    main()
