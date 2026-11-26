"""Train and test a zero-rollout Battle opening gate on fixed champion lineups."""

import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch
from torch.nn import functional as F

from .__main__ import _percentile
from .combination_preview import draft_root, preview_snapshot
from .draft_counterfactual import check_hybrid
from .draft_coverage import CHAMPION, Coverage
from .draft_early import paired_draws
from .draft_last_pick import mean_ci, other
from .draft_leaf import CATALOG, check_pools
from .draft_league import disjoint, parallel, pools_for, read, save
from .features import HISTORY_FEATURE_VERSION, BattleEncoder
from .intervention import OutcomeNetwork, _encode
from .lineup import LineupPolicy, index_cards
from .opening_audit import trial
from .sim import SEATS, fingerprint

MARGIN = .06


def opening_rosters(experiment, indexed, cards, root, actor, rival):
    rosters, actual, times = draft_root(experiment, root, cards, {actor: "champion", other(actor): rival})
    views = preview_snapshot(rosters, actual, experiment.league.engine)
    lineups = {seat: views[seat].validate(LineupPolicy(experiment.profiles, indexed, experiment.preview)(views[seat]))
               for seat in SEATS}
    return rosters, actual, lineups, times


def restore(rows, cards):
    recovered, errors = [], []
    with ExitStack() as stack:
        experiment = Coverage(stack, cards)
        indexed = index_cards(cards)
        encoder = BattleEncoder(experiment.profiles)
        for row in rows:
            seed, actor = row["seed"], row["seat"]
            try:
                rosters, actual, lineups, _ = opening_rosters(experiment, indexed, cards, seed, actor, row["draft_opponent"])
                first_seed = paired_draws(seed, actual, (), 2, 8)[0][0]
                captured = []

                def encode(policy, decision, ranked):
                    captured.extend(tensor.tolist() for tensor in _encode(encoder, policy.history, decision, ranked))
                    return 0

                game, _ = trial(experiment, rosters, lineups, actor, first_seed, "joint", encode)
                if game[0] != row["key"] or game[1] != row["actions"] or len(captured) != 2:
                    raise ValueError("The restored state does not match the labeled opening")
                recovered.append((captured[0], captured[1], row["outcomes"]))
            except (ValueError, OSError, RuntimeError, KeyError) as error:
                errors.append({"seed": seed, "issue": str(error)})
    return recovered, errors


def paired_loss(scores, paired):
    targets = paired.mean((1, 2))
    values = scores.sigmoid()
    differences = (paired[..., 1:] - paired[..., :1]).flatten(1, 2)
    importance = 1. / (1. + 4. * differences.var(1, unbiased=False))
    return F.binary_cross_entropy_with_logits(scores, targets) + 3. * (
        importance * (values[:, 1:] - values[:, :1] - differences.mean(1)).square()).mean()


@torch.inference_mode()
def assess(model, data):
    states, actions, paired = data
    p = model(states, actions).sigmoid()
    target = paired.mean((1, 2))
    maximum, best = p.max(1)
    edited = maximum - p[:, 0] >= MARGIN
    change = (target.gather(1, best[:, None]).squeeze(1) - target[:, 0]) * edited
    error = (p[:, 1:] - p[:, :1] - (target[:, 1:] - target[:, :1])).square().mean()
    baseline = (target[:, 1:] - target[:, :1]).square().mean()
    return {"positions": len(states), "overrides": int(edited.sum()), "delta": change.mean().item(),
            "paired_mse": error.item(), "anchor_mse": baseline.item()}


def train(args, cards):
    sources = [read(p) for p in (args.train, args.validation)]
    disjoint(sources[0]["pools"], sources[1]["pools"])
    for source in sources:
        if source["champion"] != CHAMPION or source["report"]["errors"]:
            raise ValueError("Incompatible opening labels")
    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    frames, errors = [], []
    began = perf_counter()
    for source in sources:
        all_rows = []
        for rows, issues in parallel(restore, source["rows"], args.workers, cards):
            all_rows.extend(rows)
            errors.extend(issues)
        if not all_rows or len(all_rows) / len(source["rows"]) < .95:
            raise RuntimeError(f"Opening restore did not match labels: {errors[:6]}")
        frames.append(tuple(torch.tensor(field, dtype=torch.float32, device=device) for field in zip(*all_rows, strict=True)))
    sampling = perf_counter() - began
    fit, validation = frames
    architecture = {"state_size": fit[0].size(-1), "action_size": fit[1].size(-1), "hidden_size": 32}
    model = OutcomeNetwork(**architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.05)
    gen = torch.Generator().manual_seed(args.seed)
    best, state = float("inf"), None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    began = perf_counter()
    for epoch in range(24):
        for i in torch.randperm(len(fit[0]), generator=gen).to(device).split(24):
            s, a, y = (t[i] for t in fit)
            loss = paired_loss(model(s, a), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        score = assess(model, validation)
        if score["paired_mse"] < best:
            best, chosen = score["paired_mse"], epoch + 1
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    elapsed = perf_counter() - began
    model.load_state_dict(state)
    args.output.mkdir(parents=True)
    torch.save(state, args.output / "state_dict.pt")
    config = {"network": architecture, "catalog": fingerprint(cards), "champion": CHAMPION, "feature_version": HISTORY_FEATURE_VERSION,
              "epoch": chosen, "margin": MARGIN, "seed": args.seed,
              "splits": {name: source["pools"] for name, source in zip(("train", "validation"), sources, strict=True)}}
    save(args.output / "config.json", config)
    local = load_model(args.output, cards)
    with torch.inference_mode():
        probe = [x[:8] for x in validation[:2]]
        expected, cpu = model(*probe).cpu(), local(*(x.cpu() for x in probe))
        torch.testing.assert_close(cpu, expected, atol=1e-5, rtol=1e-4)
    save(args.output / "probes.json", {"input": [x.tolist() for x in probe], "logits": cpu.tolist()})
    report = {"train": assess(model, fit), "validation": assess(model, validation), "sampling_seconds": sampling,
              "training_seconds": elapsed, "parameters": sum(p.numel() for p in model.parameters()),
              "weights_bytes": (args.output / "state_dict.pt").stat().st_size, "device": str(device), "errors": errors,
              "reload_delta": float((cpu - expected).abs().max())}
    if device.type == "cuda":
        report.update(gpu=torch.cuda.get_device_name(device), peak_mib=torch.cuda.max_memory_allocated(device) / 2**20)
    save(args.output / "metrics.json", report)
    print(json.dumps(report), flush=True)


def load_model(path, cards):
    config = read(path / "config.json")
    if (config["catalog"], config["champion"], config["feature_version"]) != (fingerprint(cards), CHAMPION, HISTORY_FEATURE_VERSION):
        raise ValueError("Opening checkpoint provenance mismatch")
    model = OutcomeNetwork(**config["network"])
    model.load_state_dict(torch.load(path / "state_dict.pt", map_location="cpu", weights_only=True))
    return model.eval()


def evaluate(seeds, cards, checkpoint):
    records, issues = {}, []
    with ExitStack() as stack:
        experiment = Coverage(stack, cards)
        indexed = index_cards(cards)
        model, encoder = load_model(checkpoint, cards), BattleEncoder(experiment.profiles)
        for root in seeds:
            rival = ("champion", "champion", "search", "pairwise")[root % 4]
            style = "forecast" if root % 4 == 3 else "joint"
            paired = []
            for actor in SEATS:
                try:
                    rosters, actual, lineups, draft_times = opening_rosters(experiment, indexed, cards, root, actor, rival)
                    selected, timings = [], []

                    @torch.inference_mode()
                    def choose(policy, decision, ranked):
                        started = perf_counter()
                        state, actions = _encode(encoder, policy.history, decision, ranked)
                        p = model(state[None], actions[None])[0].sigmoid()
                        best = int(p.argmax())
                        selected.append(best if float(p[best] - p[0]) >= MARGIN else 0)
                        timings.append((perf_counter() - started) * 1000)
                        return selected[-1]

                    op, before = trial(experiment, rosters, lineups, actor, actual, style, 0, strict=False)
                    cp, after = trial(experiment, rosters, lineups, actor, actual, style, choose, strict=False)
                    if op != cp:
                        raise ValueError("Opening benchmark diverged before the intervention")
                    paired.append({"old": before, "new": after, "changed": bool(selected and selected[0]),
                                   "decision_ms": timings, "draft_max_ms": max(draft_times), "rival": rival, "style": style})
                except (ValueError, OSError, RuntimeError, KeyError) as error:
                    issues.append({"seed": root, "seat": actor, "issue": str(error)})
            if len(paired) == 2:
                records[root] = paired
    return records, issues


def bench(args, cards):
    config = read(args.checkpoint / "config.json")
    seeds = range(args.seed, args.seed + args.count)
    pools = pools_for(cards, seeds)
    check_pools(pools, args.exclude)
    for split in config["splits"].values():
        disjoint(split, pools)
    games, errors = {}, []
    began = perf_counter()
    for data, issues in parallel(evaluate, seeds, args.workers, cards, args.checkpoint):
        games.update(data)
        errors.extend(issues)
    if not games:
        raise RuntimeError(f"No completed benchmark pairs: {errors[:6]}")
    games = dict(sorted(games.items()))
    ds = [mean(item["new"] - item["old"] for item in pair) for pair in games.values()]
    rows = [item for pair in games.values() for item in pair]
    report = {"pairs": len(games), "games": len(rows), "seconds": perf_counter() - began, "old_wins": sum(r["old"] == 1 for r in rows),
              "new_wins": sum(r["new"] == 1 for r in rows), "delta": mean(ds), "ci95": mean_ci(ds), "overrides": sum(r["changed"] for r in rows),
              "model_p99_ms": _percentile([t for r in rows for t in r["decision_ms"]], .99), "issues": errors,
              "draft_timeouts": sum(r["draft_max_ms"] >= 15000 for r in rows)}
    save(args.output, {"report": report, "pools": pools, "paired": games})
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit, test = (sub.add_parser(name) for name in ("train", "bench"))
    for step in (fit, test):
        step.add_argument("--seed", type=int, required=True)
        step.add_argument("--workers", type=int, default=12)
        step.add_argument("--output", type=Path, required=True)
    fit.add_argument("--train", type=Path, required=True)
    fit.add_argument("--validation", type=Path, required=True)
    fit.add_argument("--device", default="cuda")
    test.add_argument("--checkpoint", type=Path, required=True)
    test.add_argument("--count", type=int, required=True)
    test.add_argument("--exclude", nargs="*", type=Path, default=[])
    args = parser.parse_args()
    if args.output.exists() or args.workers < 1 or (args.command == "bench" and args.count < 1):
        parser.error("Use a fresh output path and positive sizes")
    check_hybrid()
    {"train": train, "bench": bench}[args.command](args, read(CATALOG))


if __name__ == "__main__":
    main()
