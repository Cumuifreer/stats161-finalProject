"""Learn partial Draft value from terminal games after different legal picks."""

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
from ..blind85 import used_pools
from ..combination_preview import preview_snapshot
from ..combination_space import CardSpace
from ..draft_forecast import encode_owners, public_draft
from ..draft_last_pick import sample_game
from ..draft_leaf import CATALOG
from ..draft_league import pools_for, read, save
from ..lineup import index_cards
from ..roster import DraftRanker, _load_model
from ..sim import SEATS, Showdown, _draft, fingerprint
from ..tactics import load_card_profiles, tactical_actions
from .preview_outcome import PreviewOutcome, label_battle
from .preview_qualify import LearnedDraft, LearnedPreview


PICKS = (2, 3, 4, 5, 6, 7, 8, 9, 10)


class DraftOutcome(nn.Module):
    def __init__(self, size, width=104):
        super().__init__()
        self.cards = nn.Embedding(size, width)
        self.owners = nn.Embedding(13, width)
        self.picks = nn.Embedding(13, width)
        block = nn.TransformerEncoderLayer(width, 4, width * 3, dropout=.10, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.rosters = nn.TransformerEncoder(block, 3, enable_nested_tensor=False)
        self.outcome = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, pool, owners, pick):
        tokens = self.cards(pool) + self.owners(owners) + self.picks(pick)[:, None]
        return self.outcome(self.rosters(tokens)).squeeze(-1)


def encode(decision, card_index):
    pool, own, foe, available, pick, order = public_draft(decision, card_index)
    return pool, encode_owners(pool, own, foe), pick


def simulate(roots, cards, lineup, output):
    configure_torch()
    data = read(lineup / "config.json")
    model = PreviewOutcome(**data["network"]).eval()
    model.load_state_dict(torch.load(lineup / "state_dict.pt", map_location="cpu", weights_only=True))
    profiles = load_card_profiles(cards, with_sets=True)
    indexed = index_cards(cards)
    space = CardSpace(cards, profiles)
    card_ids = {card["card_id"]: i for i, card in enumerate(cards)}
    projection = DraftRanker(cards, *_load_model(Path("checkpoints/roster-v1/expanded"), "cpu"))
    rows = []
    counts = Counter()
    since = perf_counter()
    with Showdown() as engine:
        for root in roots:
            pocket, first, seed = sample_game(cards, root)
            pivot_pick = PICKS[root % len(PICKS)]
            chosen_menu = None
            input_state = None
            actor = None
            rng = Random(f"draft-outcome-v1/{root}")
            seeds = [[rng.randrange(65536) for _ in range(4)] for _ in range(2)]
            results = []
            for alternative in range(3):
                draft = {seat: LearnedDraft(profiles, cards, space, model, projection) for seat in SEATS}

                def choose(seat, decision):
                    nonlocal chosen_menu, input_state, actor
                    anchor = draft[seat](decision)
                    if decision.observation["pick"] != pivot_pick:
                        return anchor
                    observed = encode(decision, card_ids)
                    if input_state is None:
                        input_state, actor = observed, seat
                        ranked = tactical_actions(decision)
                        chosen_menu = [anchor, *(action for action in ranked if action != anchor)][:3]
                    if actor != seat or input_state != observed:
                        raise RuntimeError("Counterfactual Drafts disagree before the intervention")
                    return decision.validate(chosen_menu[alternative])

                if chosen_menu is not None and alternative >= len(chosen_menu):
                    break
                rosters = _draft(pocket, first, choose)
                observations = preview_snapshot(rosters, seed, engine)
                choices = {seat: LearnedPreview(profiles, indexed, space, model)(observations[seat]) for seat in SEATS}
                labels = [label_battle(engine, rosters, actor, choices, draw, styles, profiles)
                          for draw, styles in zip(seeds, (("joint", "forecast"), ("forecast-replies", "joint")), strict=True)]
                results.append(sum(labels) / len(labels))
                counts["terminal_games"] += 2
            if len(results) < 2:
                continue
            pool, owners, choice = input_state
            indices = [pool.index(card_ids[action["card_id"]]) for action in chosen_menu]
            target = np.full(18, -1., dtype=np.float32)
            target[indices] = results
            rows.append((root, pool, owners, choice, indices[0], target))
            counts["choice_positions"] += 1
    keys = ("root", "pool", "owners", "pick", "anchor", "target")
    np.savez_compressed(output, **dict(zip(keys, map(np.asarray, zip(*rows, strict=True)), strict=True)))
    return dict(counts, pools=len(roots), worker_seconds=perf_counter() - since)


def inputs(paths, manifest):
    arrays = []
    for path in paths:
        with np.load(path, allow_pickle=False) as file:
            arrays.append({key: file[key] for key in file.files})
    joined = {key: np.concatenate([row[key] for row in arrays]) for key in arrays[0]}
    held = np.asarray([int(manifest[str(i)][:8], 16) % 7 == 0 for i in joined["root"]])
    device = torch.device("cuda")
    data = [torch.tensor(joined[name], dtype=torch.float32 if name == "target" else torch.long, device=device)
            for name in ("pool", "owners", "pick", "anchor", "target")]
    return [[values[torch.from_numpy(flag).to(device)] for values in data] for flag in (~held, held)]


def objective(q, outcome):
    mask = outcome >= 0
    y = outcome.clamp_min(0.)
    raw = F.binary_cross_entropy_with_logits(q, y, reduction="none")
    prior = (outcome / .19).masked_fill(~mask, -1e4).softmax(1)
    rank = -(prior * (q / .8).masked_fill(~mask, -1e4).log_softmax(1)).sum(1).mean()
    return (raw * mask).sum() / mask.sum().clamp_min(1) + .30 * rank


@torch.inference_mode()
def assess(model, held):
    pool, owners, pick, anchor, target = held
    q = model(pool, owners, pick)
    mask = target >= 0
    nll = float(F.binary_cross_entropy_with_logits(q[mask], target[mask]))
    best = q.masked_fill(~mask, -torch.inf).argmax(1)[:, None]
    chosen = target.gather(1, best).flatten()
    before = target.gather(1, anchor[:, None]).flatten()
    return dict(nll=nll, paired_label_gain=float((chosen - before).mean()), changes=int((best[:, 0] != anchor).sum()))


def train(args, cards, pools, files):
    configure_torch()
    torch.manual_seed(902819)
    torch.set_float32_matmul_precision("high")
    learning, validation = inputs(files, pools)
    config = dict(network={"size": len(cards), "width": 104}, catalog=fingerprint(cards))
    model = DraftOutcome(**config["network"]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.00035, weight_decay=.08, fused=True)
    save(args.output / "config.json", dict(config, data_files={str(path): sha256(path.read_bytes()).hexdigest() for path in files}))
    started = perf_counter()
    records = []
    best, best_epoch = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for idx in torch.randperm(len(learning[0]), device="cuda").split(128):
            pool, owners, pick, _, target = [field[idx] for field in learning]
            score = objective(model(pool, owners, pick), target)
            optimizer.zero_grad(set_to_none=True)
            score.backward()
            optimizer.step()
        candidate = dict(epoch=epoch, **assess(model.eval(), validation))
        records.append(candidate)
        measured = candidate["nll"] - .2 * candidate["paired_label_gain"]
        if measured < best:
            best, best_epoch = measured, epoch
            torch.save(model.state_dict(), args.output / "state_dict.pt")
        print(candidate, flush=True)
        if epoch - best_epoch >= 5:
            break
    save(args.output / "training.json", dict(epochs=records, best=best_epoch, train_states=len(learning[0]),
                                            validation_states=len(validation[0]), seconds=perf_counter() - started,
                                            parameters=sum(p.numel() for p in model.parameters()), gpu=torch.cuda.get_device_name()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineup", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pools", type=int, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=24)
    args = parser.parse_args()
    if args.data.exists() or args.output.exists() or min(args.pools, args.workers, args.epochs) < 1:
        parser.error("Use new outputs and positive run sizes")
    cards = read(CATALOG)
    roots = list(range(args.seed, args.seed + args.pools))
    pools = pools_for(cards, roots)
    if not used_pools(cards).isdisjoint(pools.values()):
        raise ValueError("Draft-training pools must be new")
    if read(args.lineup / "config.json")["catalog"] != fingerprint(cards):
        raise ValueError("Lineup and Draft catalog mismatch")
    save(args.data / "pools.json", dict(pools=pools, source=sha256(Path(__file__).read_bytes()).hexdigest(),
                                      advisor=sha256((args.lineup / "state_dict.pt").read_bytes()).hexdigest()))
    started = perf_counter()
    counts = Counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as workers:
        jobs = [workers.submit(simulate, roots[i:i + 32], cards, args.lineup, args.data / f"{roots[i]}.npz")
                for i in range(0, len(roots), 32)]
        for job in as_completed(jobs):
            counts.update(job.result())
            print(dict(counts), flush=True)
    save(args.data / "completed.json", dict(counts=dict(counts), seconds=perf_counter() - started))
    train(args, cards, pools, sorted(args.data.glob("*.npz")))


if __name__ == "__main__":
    main()
