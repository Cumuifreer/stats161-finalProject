"""Train a recurrent Battle policy on terminal outcomes and paired branch labels."""

import argparse
import json
import os
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from ..battle_league import BattleNetwork, CHAMPION, FIELDS, HISTORY_FEATURE_VERSION, VERSION, configure_torch, make_batch
from ..draft_leaf import CATALOG
from ..draft_league import read, save
from ..sim import fingerprint
from ..tactics import load_card_profiles


def dataset(sources, branch_sources, catalog):
    train, held, inputs = [], [], {}
    source_reports = []
    used = set()
    for source in sources:
        manifest = read(source / "pools.json")
        report = read(source / "completed.json")
        if manifest["catalog"] != catalog or not report["complete"]:
            raise ValueError("Expected a finished population dataset using the active public catalog")
        pools = set(manifest["pools"].values())
        if not used.isdisjoint(pools):
            raise ValueError("Replay populations share pools")
        used.update(pools)
        source_reports.append({"source": str(source), "pools": len(pools), **report["counts"]})
        for path in sorted(source.glob("*.npz")):
            with np.load(path, allow_pickle=False) as data:
                rows = {key: data[key] for key in (*FIELDS, "propensity")}
            inputs[str(path)] = sha256(path.read_bytes()).hexdigest()
            validation = np.asarray([int(manifest["pools"][str(root // 4)][:8], 16) % 10 == 0 for root in rows["root"]])
            train.append({key: value[~validation] for key, value in rows.items()})
            held.append({key: value[validation] for key, value in rows.items()})
    for path in branch_sources:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(data["metadata"].item())
            rows = {key: data[key] for key in FIELDS}
        pools = set(metadata["pools"].values())
        if metadata["catalog"] != catalog or not used.isdisjoint(pools):
            raise ValueError("Branch dataset catalog or pool mismatch")
        used.update(pools)
        rows["propensity"] = np.full(len(rows["root"]), .5, dtype=np.float32)
        inputs[str(path)] = sha256(path.read_bytes()).hexdigest()
        train.append(rows)
    keys = (*FIELDS, "propensity")
    arrays = [{key: np.concatenate([rows[key] for rows in split]) for key in keys} for split in (train, held)]
    if min(len(rows["root"]) for rows in arrays) == 0:
        raise ValueError("Replay requires both training and held-out positions")
    return arrays, inputs, source_reports


def prepare(rows, device, history_steps):
    return (*make_batch(rows, device, history_steps), torch.from_numpy(rows["propensity"]).to(device))


def loss(model, batch):
    *features, teacher, behavior, outcome, branch, propensity = batch
    logits, q, value = model(*features)
    # Floor the importance weights to control the variance of rare exploratory actions.
    importance = propensity.clamp_min(.1).rsqrt()
    visited = q.gather(1, behavior[:, None]).squeeze(1)
    action_outcome = F.binary_cross_entropy_with_logits(visited, outcome, reduction="none")
    value_loss = F.binary_cross_entropy_with_logits(value, outcome)
    advantage = outcome - value.detach().sigmoid()
    behavior_loss = F.cross_entropy(logits, behavior, reduction="none")
    imitation = F.cross_entropy(logits, teacher)
    labels = (branch >= 0) & features[4]
    observed = branch.clamp_min(0.)
    regression = F.binary_cross_entropy_with_logits(q, observed, reduction="none")
    paired_error = ((q.sigmoid() - q[:, :1].sigmoid()) - (observed - observed[:, :1])).square()
    branch_loss = ((regression + 2. * paired_error) * labels).sum() / labels.sum().clamp_min(1)
    return ((action_outcome * importance).sum() / importance.sum() + .5 * value_loss + .65 * branch_loss
            + .25 * imitation + .30 * (behavior_loss * (2 * advantage).exp()).mean())


@torch.inference_mode()
def assess(model, batch, batch_size):
    counts = torch.zeros(8, device=batch[0].device)
    paired = torch.zeros(4, device=batch[0].device)
    outputs = []
    for start in range(0, len(batch[0]), batch_size):
        *features, teacher, behavior, outcome, branch, propensity = [field[start:start + batch_size] for field in batch]
        logits, q, value = model(*features)
        visited = q.gather(1, behavior[:, None]).squeeze(1)
        choice = q.masked_fill(~features[4], -torch.inf).argmax(1)
        counts += torch.stack((F.binary_cross_entropy_with_logits(value, outcome, reduction="sum"),
                               F.binary_cross_entropy_with_logits(visited, outcome, reduction="sum"),
                               (value.sigmoid() - outcome).square().sum(), (visited.sigmoid() - outcome).square().sum(),
                               (choice != 0).sum(), (choice == teacher).sum(),
                               (logits.argmax(1) == teacher).sum(),
                               F.binary_cross_entropy_with_logits(value[outcome == 1], outcome[outcome == 1], reduction="sum")))
        outputs.append(torch.stack((value.sigmoid(), outcome), dim=1))
        observed = (branch >= 0) & features[4]
        paired_rows = observed[:, 0] & (observed.sum(1) > 1)
        alternative = q.masked_fill(~observed, -torch.inf).argmax(1)
        label = branch.gather(1, alternative[:, None]).squeeze(1)
        contrast = q.sigmoid() - q[:, :1].sigmoid()
        errors = (contrast - (branch - branch[:, :1])).abs()
        paired += torch.stack((paired_rows.sum(), ((label - branch[:, 0]) * paired_rows).sum(),
                               (errors * (observed & paired_rows[:, None])).sum(), (observed & paired_rows[:, None]).sum()))
    metrics = dict(zip(("value_nll", "q_nll", "value_brier", "q_brier", "q_change_rate", "q_teacher_agreement",
                        "policy_teacher_agreement", "wins_nll_per_position"),
                       (counts / len(batch[0])).tolist(), strict=True))
    predictions = torch.cat(outputs)
    metrics.update(verified_branch_positions=int(paired[0]), paired_label_gain=float(paired[1] / paired[0].clamp_min(1)),
                   paired_delta_mae=float(paired[2] / paired[3].clamp_min(1)))
    start = torch.minimum((predictions[:, 0] * 10).long(), torch.tensor(9, device=counts.device))
    metrics["calibration"] = [{"count": int((start == bin_id).sum()),
                              "predicted": predictions[start == bin_id, 0].mean().nan_to_num().item(),
                              "actual": predictions[start == bin_id, 1].mean().nan_to_num().item()}
                             for bin_id in range(10)]
    return metrics


def batches(train, batch_size, minimum, device):
    indices = torch.randperm(len(train[0]), device=device)
    if minimum <= 0:
        yield from indices.split(batch_size)
        return
    labeled = (train[-2][indices] >= 0).sum(1) > 1
    probe = indices[labeled]
    if not len(probe):
        raise ValueError("Stratified training needs positions with paired outcomes")
    for standard in indices[~labeled].split(batch_size - minimum):
        sampled = probe[torch.randint(len(probe), (minimum,), device=device)]
        yield torch.cat((standard, sampled))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--branches", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=144)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--seed", type=int, default=635501)
    parser.add_argument("--learning-rate", type=float, default=.00041)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--paired-positions-per-batch", type=int, default=0)
    args = parser.parse_args()
    if (args.output.exists() or min(args.epochs, args.batch, args.hidden, args.depth, args.history) < 1
            or args.hidden % 4 or args.patience < 0 or not 0 < args.learning_rate < .01
            or not 0 <= args.paired_positions_per_batch < args.batch):
        parser.error("Provide a fresh output and a positive architecture with hidden size divisible by four")
    os.environ["OMP_NUM_THREADS"] = "1"
    configure_torch()
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested, but PyTorch cannot access the 4090")
    device = torch.device("cuda")
    started = perf_counter()
    cards = read(CATALOG)
    (train_rows, held_rows), data_files, sources = dataset(args.data, args.branches, fingerprint(cards))
    config = dict(network=dict(state_size=train_rows["state"].shape[-1], action_size=train_rows["actions"].shape[-1],
                              card_count=len(load_card_profiles(cards, with_sets=True)), hidden=args.hidden,
                              depth=args.depth, history_steps=args.history), catalog=fingerprint(cards), champion=CHAMPION,
                  version=VERSION, feature_version=HISTORY_FEATURE_VERSION)
    model = BattleNetwork(**config["network"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=.02, fused=True)
    if args.previous:
        if read(args.previous / "config.json") != config:
            raise ValueError("Continued training requires an unchanged architecture and feature encoding")
        model.load_state_dict(torch.load(args.previous / "state_dict.pt", map_location=device, weights_only=True))
        optimizer.load_state_dict(torch.load(args.previous / "optimizer.pt", map_location=device, weights_only=True))
        optimizer.param_groups[0]["lr"] = args.learning_rate
    train, held = prepare(train_rows, device, args.history), prepare(held_rows, device, args.history)
    del train_rows, held_rows
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "config.json", config)
    setup = dict(parameters=sum(parameter.numel() for parameter in model.parameters()), device=torch.cuda.get_device_name(),
                 train_positions=len(train[0]), held_positions=len(held[0]), replay=sources,
                 parent=str(args.previous) if args.previous else None, data_files=data_files,
                 learning_rate=optimizer.param_groups[0]["lr"], paired_positions_per_batch=args.paired_positions_per_batch)
    save(args.output / "setup.json", setup)
    print({key: value for key, value in setup.items() if key != "data_files"}, flush=True)
    model.eval()
    initial = assess(model, held, args.batch)
    best = float("inf")
    best_epoch = 0
    epochs = []
    steps = 0
    moments = perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = torch.tensor(0., device=device)
        visited = 0
        for selection in batches(train, args.batch, args.paired_positions_per_batch, device):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                objective = loss(model, [field[selection] for field in train])
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
            optimizer.step()
            total_loss += objective.detach() * len(selection)
            steps += 1
            visited += len(selection)
        model.eval()
        metrics = assess(model, held, args.batch)
        score = .55 * metrics["q_nll"] + .45 * metrics["value_nll"]
        if args.paired_positions_per_batch:
            score += .30 * metrics["paired_delta_mae"] - .30 * metrics["paired_label_gain"]
        if score < best:
            best = score
            best_epoch = epoch
            torch.save(model.state_dict(), args.output / "state_dict.pt")
            torch.save(optimizer.state_dict(), args.output / "optimizer.pt")
        info = dict(epoch=epoch, optimizer_steps=steps, train_loss=(total_loss / visited).item(),
                    seconds=perf_counter() - moments, **metrics)
        epochs.append(info)
        print({key: value for key, value in info.items() if key != "calibration"}, flush=True)
        if args.patience and epoch - best_epoch >= args.patience:
            break
    report = dict(seconds=perf_counter() - started, training_seconds=perf_counter() - moments,
                  optimizer_steps=steps, best_epoch=best_epoch,
                  initial=initial, epochs=epochs, cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    save(args.output / "training.json", report)


if __name__ == "__main__":
    main()
