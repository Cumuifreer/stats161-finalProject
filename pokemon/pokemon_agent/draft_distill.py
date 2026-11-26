"""Fit counterfactual Draft values and compare them to the frozen hybrid."""

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from .__main__ import HISTORICAL_CARDS, _paired_win_ci, _percentile
from .draft_advantage import ActionValue, AdvantageDraft, ReducedReview, load_advantage
from .draft_counterfactual import HYBRID, PREDICTOR, PROVENANCE, check_hybrid, frozen_splits
from .draft_forecast import ForecastDraft, load_predictor
from .draft_league import baseline_hash, disjoint, parallel, pools_for, read, save
from .draft_matchups import League
from .draft_value import DraftRollout
from .game import BATTLE, DRAFT, PREVIEW
from .sim import SEATS, fingerprint, play_game


def tensors(rows, device):
    inputs = [torch.tensor([row[key] for row in rows], device=device) for key in ("pool", "owners", "ply", "features")]
    shape = inputs[0].shape
    outcomes, weights = torch.zeros(shape), torch.zeros(shape)
    anchors = []
    for r, row in enumerate(rows):
        scenarios = row["results"]["rollout"]
        anchor = torch.tensor(scenarios[0]).flatten()
        anchors.append(row["slots"][0])
        for slot, leaves in zip(row["slots"], scenarios, strict=True):
            if leaves is None:
                continue
            scores = torch.tensor(leaves).flatten()
            if scores.shape != anchor.shape:
                raise ValueError("Matched candidate branches must share the seed samples")
            outcomes[r, slot] = scores.mean()
            weights[r, slot] = 1 / (1 + 8 * (scores - anchor).var(unbiased=False) / len(scores))
    return *inputs, outcomes.to(device), weights.to(device), torch.tensor(anchors, device=device)


def objective(logits, y, weights, anchors):
    p = logits.sigmoid()
    axis = anchors[:, None]
    advantage_error = (p - p.gather(1, axis)) - (y - y.gather(1, axis))
    loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none") + 3 * advantage_error.square()
    return (loss * weights).sum() / weights.sum()


def ranking_objective(logits, y, weights, anchors):
    preferences = (4 * (y[:, :, None] - y[:, None, :])).sigmoid()
    differences = logits[:, :, None] - logits[:, None, :]
    confidence = weights[:, :, None] * weights[:, None, :]
    loss = F.binary_cross_entropy_with_logits(differences, preferences, reduction="none")
    return (loss * confidence).sum() / confidence.sum() + .2 * objective(logits, y, weights, anchors)


@torch.inference_mode()
def assess(net, data):
    pool, owners, ply, features, outcomes, weights, anchors = data
    logits = net(pool, owners, ply, features)
    mask = weights > 0
    true_top = outcomes.masked_fill(~mask, -1).amax(1)
    result = {}
    prior = features[:, :, 0].clamp(.03, .97)
    blend = (.65 * logits.sigmoid() + .35 * prior).logit()
    for name, values in (("model", logits), ("forecast", prior.logit()), ("blend", blend)):
        picked = values.masked_fill(~mask, -torch.inf).argmax(1)
        chosen = outcomes.gather(1, picked[:, None]).squeeze(1)
        result[name] = {"objective": objective(values, outcomes, weights, anchors).item(),
                        "observed_best_choice_rate": (chosen == true_top).float().mean().item(),
                        "chosen_mean_reward": chosen.mean().item()}
    return result


def train(args, cards):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    training = [read(path) for path in args.train]
    validation_sets = [read(path) for path in args.validation]
    catalog = fingerprint(cards)
    if any(data["catalog"] != catalog or set(data["frozen"]) != PROVENANCE for data in (*training, *validation_sets)):
        raise ValueError("Counterfactual dataset provenance mismatch")
    seen = {}
    splits, samples = {}, {}
    for name, datasets in (("train", training), ("validation", validation_sets)):
        pools, rows = {}, []
        for dataset in datasets:
            disjoint(seen, dataset["pools"])
            seen.update(dataset["pools"])
            pools.update(dataset["pools"])
            rows.extend(dataset["rows"])
        splits[name], samples[name] = pools, rows
    device = torch.device(args.device)
    fit, validation = (tensors(samples[name], device) for name in ("train", "validation"))
    selection_picks = getattr(args, "selection_picks", None)
    selection_validation = validation
    if selection_picks:
        selected_turns = torch.isin(validation[2] + 1, torch.tensor(selection_picks, device=device))
        if not selected_turns.any():
            raise ValueError("No validation roots in the requested selection picks")
        selection_validation = tuple(value[selected_turns] for value in validation)
    architecture = {"card_count": len(cards), "width": args.width}
    model = ActionValue(**architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.022)
    rng = torch.Generator().manual_seed(args.seed)
    best = (float("inf"), float("inf"))
    loss_function = ranking_objective if args.loss == "ranking" else objective
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started_at = datetime.now().isoformat(timespec="milliseconds")
    start = perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for indices in torch.randperm(len(fit[0]), generator=rng).to(device).split(64):
            batch = [tensor[indices] for tensor in fit]
            loss = loss_function(model(*batch[:4]), *batch[4:])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        report = assess(model, selection_validation)
        blend = report["blend"]
        selection = (-blend["chosen_mean_reward"], blend["objective"]) if args.loss == "ranking" else (report["model"]["objective"], 0.)
        if selection < best:
            best, selected = selection, epoch
            weights = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    elapsed = perf_counter() - start
    ended_at = datetime.now().isoformat(timespec="milliseconds")
    model.load_state_dict(weights)
    args.output.mkdir(parents=True)
    torch.save(weights, args.output / "state_dict.pt")
    config = {"catalog": catalog, "frozen": sorted(PROVENANCE), "network": architecture, "seed": args.seed, "loss": args.loss,
              "epochs": args.epochs, "selected_epoch": selected,
              "splits": splits, "trained_picks": sorted({row["ply"] + 1 for row in samples["train"]})}
    if selection_picks:
        config["selection_picks"] = selection_picks
    save(args.output / "config.json", config)
    with torch.inference_mode():
        probe = [tensor[:12] for tensor in validation[:4]]
        cpu_model, _ = load_advantage(args.output)
        reference = cpu_model(*(tensor.cpu() for tensor in probe)).sigmoid()
        trained = model(*probe).sigmoid().cpu()
        torch.testing.assert_close(reference, trained, atol=.001, rtol=0)
        save(args.output / "probes.json", {"input": [tensor.tolist() for tensor in probe], "probability": reference.tolist()})
    report = {"training_positions": len(fit[0]), "validation_positions": len(validation[0]),
              "training_terminal_games": sum(dataset["report"]["rollout_terminal_games"] for dataset in training),
              "validation_terminal_games": sum(dataset["report"]["rollout_terminal_games"] for dataset in validation_sets),
              "fitting_seconds": elapsed,
              "fitting_started_local": started_at, "fitting_ended_local": ended_at, "torch": str(torch.__version__),
              "device": str(device), "parameters": sum(p.numel() for p in model.parameters()),
              "weights_bytes": (args.output / "state_dict.pt").stat().st_size,
              "cpu_reload_max_probability_gap": (reference - trained).abs().max().item(),
              "train": assess(model, fit), "validation": assess(model, validation)}
    if selection_picks:
        report["selected_ply_validation"] = assess(model, selection_validation)
    if device.type == "cuda":
        report.update(gpu=torch.cuda.get_device_name(device), peak_mib=torch.cuda.max_memory_allocated(device) / 2**20)
    save(args.output / "metrics.json", report)
    print(json.dumps(report), flush=True)


def evaluate(seeds, cards, checkpoint, method, preview):
    pairs, times, errors, counts = {}, [], [], Counter()
    with ExitStack() as stack:
        league = League(stack, cards, preview, True)
        predictor, _ = load_predictor(PREDICTOR)
        profiles = league.factory().profiles
        pairwise = league.ranks["pairwise"]
        baseline = ForecastDraft(pairwise, predictor, "hybrid", DraftRollout(league.simulator, cards, profiles))
        if method == "frozen":
            challenger = ForecastDraft(pairwise, predictor, "hybrid", DraftRollout(league.simulator, cards, profiles))
        else:
            model = load_advantage(checkpoint)[0] if method in {"learned", "reduced"} else None
            review = ReducedReview if method == "reduced" else DraftRollout
            challenger = AdvantageDraft(pairwise, predictor, review(league.simulator, cards, profiles), method, model)
        for seed in seeds:
            scores = []
            for left in SEATS:
                policies = {seat: league.policy("pairwise") for seat in SEATS}
                for seat in SEATS:
                    policies[seat].draft = challenger if seat == left else baseline
                samples = {seat: [] for seat in SEATS}
                try:
                    winner = play_game(league.engine, cards, policies, seed, samples)
                    scores.append(.5 if winner is None else float(winner == left))
                except (OSError, RuntimeError, ValueError) as error:
                    errors.append({"seed": seed, "seat": left, "error": str(error)})
                times.extend(elapsed for phase, elapsed, _ in samples[left] if phase == DRAFT)
                for seat in SEATS:
                    for phase, _, event in samples[seat]:
                        counts[phase] += 1
                        if event:
                            counts[event] += 1
                    counts["candidate_fallbacks" if seat == left else "baseline_fallbacks"] += policies[seat].fallbacks
            if len(scores) == 2:
                pairs[seed] = scores
        counts.update(challenger.rollout.activity)
        counts.update(challenger.activity)
        counts.update({f"baseline_{key}": value for key, value in baseline.rollout.activity.items()})
    return pairs, times, errors, counts


def benchmark(args, cards):
    seeds = range(args.seed, args.seed + args.pairs)
    pools = pools_for(cards, seeds)
    for split in frozen_splits():
        disjoint(pools, split)
    if args.checkpoint:
        config = read(args.checkpoint / "config.json")
        if config["catalog"] != fingerprint(cards) or set(config["frozen"]) != PROVENANCE:
            raise ValueError("Counterfactual checkpoint provenance mismatch")
        for split in config["splits"].values():
            disjoint(pools, split)
    paired, times, errors, counts = {}, [], [], Counter()
    start = perf_counter()
    for outcomes, latencies, issues, events in parallel(evaluate, seeds, args.workers, cards, args.checkpoint, args.method, args.preview):
        paired.update(outcomes)
        times.extend(latencies)
        errors.extend(issues)
        counts.update(events)
    elapsed = perf_counter() - start
    games = [score for pair in paired.values() for score in pair]
    decisions = counts[DRAFT] + counts[PREVIEW] + counts[BATTLE]
    report = {"method": args.method, "opponent": "frozen hybrid", "preview": args.preview, "seed": args.seed,
              "requested_pairs": args.pairs, "games": len(games), "wins": games.count(1.), "draws": games.count(.5),
              "score": sum(games) / len(games) if games else None, "seconds": elapsed, "workers": args.workers,
              "paired_score_ci95": _paired_win_ci([sum(pair) for pair in paired.values()]),
              "draft_p95_ms": _percentile(times, .95), "draft_p99_ms": _percentile(times, .99),
              "illegal_action_rate": counts["illegal_action"] / decisions if decisions else None,
              "timeout_rate": (counts["auto_action"] + counts["forfeit"]) / decisions if decisions else None,
              "events": dict(counts), "errors": errors, "frozen_hybrid_sha256": HYBRID,
              "checkpoint_sha256": baseline_hash(args.checkpoint) if args.checkpoint else None}
    save(args.output, {"result": report, "pools": pools, "paired_scores": dict(sorted(paired.items()))})
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    fit, bench = (subs.add_parser(name) for name in ("train", "bench"))
    for command in (fit, bench):
        command.add_argument("--seed", type=int, required=True)
        command.add_argument("--output", type=Path, required=True)
    fit.add_argument("--train", type=Path, action="append", required=True)
    fit.add_argument("--validation", type=Path, action="append", required=True)
    fit.add_argument("--device", default="cuda")
    fit.add_argument("--width", type=int, default=48)
    fit.add_argument("--epochs", type=int, default=28)
    fit.add_argument("--loss", choices=("value", "ranking"), default="value")
    fit.add_argument("--selection-picks", nargs="+", type=int, choices=range(1, 13))
    bench.add_argument("--method", choices=("frozen", "expanded", "learned", "reduced"), required=True)
    bench.add_argument("--checkpoint", type=Path)
    bench.add_argument("--preview", choices=("tactical", "rollout"), default="rollout")
    bench.add_argument("--pairs", type=int, required=True)
    bench.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output path")
    check_hybrid()
    cards = read(HISTORICAL_CARDS)
    if args.command == "train":
        if args.width < 4 or args.width % 4 or args.epochs < 1:
            parser.error("Use a width divisible by four and positive epochs")
        train(args, cards)
    else:
        if min(args.pairs, args.workers) < 1:
            parser.error("Use positive pair and worker counts")
        if args.method in {"learned", "reduced"} and not args.checkpoint:
            parser.error("A learned variant requires a checkpoint")
        benchmark(args, cards)


if __name__ == "__main__":
    main()
