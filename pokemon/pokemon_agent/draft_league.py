"""Generate counterfactual drafts, train a value model, and benchmark on separate pools."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from datetime import datetime
from hashlib import sha256
import json
from multiprocessing import get_context
from pathlib import Path
from random import Random
from time import perf_counter

import torch
from torch.nn import functional as F

from .__main__ import HISTORICAL_CARDS, _paired_win_ci, _percentile
from .draft_value import NETWORKS, DraftRollout, DraftValueRanker, load_value
from .game import DECISION_LIMIT, DRAFT, PREVIEW
from .joint import JointPolicy
from .roster import DraftRanker, _fit_prior, _load_model, roster_factory
from .sim import SEATS, Showdown, fingerprint, play_game, pool_fingerprint
from .tactics import load_card_profiles, tactical_candidates


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, separators=(",", ":"))
        output.write("\n")


def baseline_hash(path):
    return sha256((path / "state_dict.pt").read_bytes()).hexdigest()


def pools_for(cards, seeds):
    pools = {str(seed): pool_fingerprint(cards, seed) for seed in seeds}
    if len(set(pools.values())) != len(pools):
        raise ValueError("Repeated card pool in split")
    return pools


def disjoint(left, right):
    if left.keys() & right.keys() or set(left.values()) & set(right.values()):
        raise ValueError("Seed or card pool overlap")


def parallel(function, seeds, workers, *args):
    if workers == 1:
        yield function(seeds, *args)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as executor:
        jobs = [executor.submit(function, seeds[i::workers], *args) for i in range(min(workers, len(seeds)))]
        for job in jobs:
            yield job.result()


def sample(seeds, cards, baseline, behavior):
    torch.set_num_threads(1)
    prior, config = _load_model(baseline, "cpu")
    ranker = DraftRanker(cards, prior, config)
    learned = DraftValueRanker(ranker, load_value(behavior, "cpu")[0], .035, "linear") if behavior else None
    profiles = load_card_profiles(cards, with_sets=True)
    pairs, skips = [], Counter()
    with Showdown() as engine:
        for seed in seeds:
            rng = Random(seed ^ 0x31AF82D)
            branch_pick = 1 + rng.randrange(8)
            branch_rank = rng.randrange(1, 4)
            noise = [[rng.gauss(0, sigma) for _ in cards] for sigma in ((0., .18, .32)[seed % 3],
                                                                    (0., .18, .32)[seed // 3 % 3])]
            learned_seats = (0, 3, 1, 2)[seed % 4] if learned is not None else 0
            pair, sequences = [], []
            for variant in (0, 1):
                rosters = {seat: [] for seat in SEATS}
                sequence = []
                policies = {seat: JointPolicy(profiles) for seat in SEATS}

                def choose(decision, seat):
                    policy = policies[seat]
                    if decision.phase == PREVIEW:
                        policy.observe(decision)
                        return tactical_candidates(decision)[0]
                    if decision.phase != DRAFT:
                        return policy(decision)
                    own = SEATS.index(seat)
                    if learned_seats & (1 << own):
                        ordered = learned.rank(decision, "value")
                    else:
                        covered = ranker._coverage(rosters[seat])

                        def priority(action):
                            card = ranker.index[action["card_id"]]
                            return ranker.strength[card] + .25 * ranker._support(card, covered) + noise[own][card]

                        ordered = sorted(decision.actions, key=priority, reverse=True)
                    rank = min(branch_rank, len(ordered) - 1) if variant and len(sequence) + 1 == branch_pick else 0
                    action = ordered[rank]
                    card = ranker.index[action["card_id"]]
                    sequence.append(card)
                    rosters[seat].append(card)
                    return action

                samples = {seat: [] for seat in SEATS}
                game_policies = {seat: (lambda decision, seat=seat: choose(decision, seat)) for seat in SEATS}
                try:
                    winner = play_game(engine, cards, game_policies, seed, samples)
                except RuntimeError as error:
                    if str(error).startswith("Item Clause left") or str(error) == "Benchmark match exceeded 500 decision rounds":
                        skips[str(error)] += 1
                        break
                    raise
                if any(event for rows in samples.values() for _, _, event in rows):
                    skips["timed out"] += 1
                    break
                pair.append([list(rosters.values()), .5 if winner is None else float(winner == "p1")])
                sequences.append(sequence)
            if len(pair) != 2:
                continue
            if sequences[0][:branch_pick - 1] != sequences[1][:branch_pick - 1]:
                raise RuntimeError("Counterfactual draft prefix changed")
            if sequences[0][branch_pick - 1] == sequences[1][branch_pick - 1]:
                skips["single legal pick"] += 1
                continue
            if all(set(first) == set(second) for first, second in zip(pair[0][0], pair[1][0], strict=True)):
                skips["identical roster sets"] += 1
                continue
            pairs.append({"seed": seed, "pick": branch_pick, "learned_seats": learned_seats, "results": pair})
    return pairs, skips


def generate(args, cards, seeds):
    config = read(args.baseline / "config.json")
    if config["card_ids"] != [card["card_id"] for card in cards] or config["catalog_sha256"] != fingerprint(cards):
        raise ValueError("Dataset requires the baseline's card catalog")
    pools = pools_for(cards, seeds)
    for split in config["split"].values():
        disjoint(pools, split)
    if args.behavior:
        behavior = read(args.behavior / "config.json")
        if (behavior["catalog"], behavior["baseline"]) != (fingerprint(cards), baseline_hash(args.baseline)):
            raise ValueError("Behavior checkpoint provenance mismatch")
        for split in behavior["data"].values():
            disjoint(pools, split)
    rows, skips = [], Counter()
    started = perf_counter()
    for data, errors in parallel(sample, seeds, args.workers, cards, args.baseline, args.behavior):
        rows.extend(data)
        skips.update(errors)
    rows.sort(key=lambda row: row["seed"])
    if not rows:
        raise ValueError("No valid counterfactuals")
    elapsed = perf_counter() - started
    metrics = {"schema_version": 1, "pairs": len(rows), "games": 2 * len(rows), "skips": dict(skips),
               "workers": args.workers, "seconds": elapsed, "games_per_second": 2 * len(rows) / elapsed,
               "different_outcome_pairs": sum(row["results"][0][1] != row["results"][1][1] for row in rows),
               "both_preview": "tactical", "both_battle": "joint", "both_draft": "frozen prior with small preference noise",
               "catalog": fingerprint(cards), "baseline": baseline_hash(args.baseline)}
    if args.behavior:
        metrics.update(behavior_sha256=baseline_hash(args.behavior),
                       pairs_by_learned_seats=dict(Counter(row["learned_seats"].bit_count() for row in rows)))
    save(args.output, {"metrics": metrics, "pools": pools, "rows": rows})
    print(json.dumps(metrics), flush=True)


def tensors(rows, device):
    teams = torch.tensor([[game[0] for game in row["results"]] for row in rows], device=device)
    outcomes = torch.tensor([[game[1] for game in row["results"]] for row in rows], device=device)
    return teams, outcomes


def objective(logits, scores):
    predicted = logits.sigmoid()
    error = (predicted[:, 0] - predicted[:, 1]) - (scores[:, 0] - scores[:, 1])
    return F.binary_cross_entropy_with_logits(logits, scores) + error.square().mean()


@torch.inference_mode()
def assess(net, teams, scores):
    logits = torch.cat([net(batch.flatten(0, 1)).reshape(-1, 2) for batch in teams.split(512)])
    prior = net.prior[teams].sum(-1)
    result = {}
    y = scores[:, 0] - scores[:, 1]
    decisive = y != 0
    for name, x in (("neural", logits), ("linear", prior[:, :, 0] - prior[:, :, 1])):
        delta = x[:, 0].sigmoid() - x[:, 1].sigmoid()
        result[name] = {"objective": objective(x, scores).item(),
                        "brier": (x.sigmoid() - scores).square().mean().item(),
                        "counterfactual_accuracy": ((delta[decisive] * y[decisive]) > 0).float().mean().item()}
    return result


def train(args, cards):
    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    training = [read(path) for path in args.train]
    val_data = read(args.validation)
    fit_rows, fit_pools = [], {}
    for dataset in training:
        disjoint(dataset["pools"], fit_pools)
        disjoint(dataset["pools"], val_data["pools"])
        fit_rows.extend(dataset["rows"])
        fit_pools.update(dataset["pools"])
    provenance = (fingerprint(cards), baseline_hash(args.baseline))
    for dataset in (*training, val_data):
        info = dataset["metrics"]
        if (info["catalog"], info["baseline"]) != provenance or info["schema_version"] != 1:
            raise ValueError("Draft dataset provenance mismatch")
    fit, targets = tensors(fit_rows, device)
    validation, truth = tensors(val_data["rows"], device)
    architecture = {"card_count": len(cards), "width": args.width}
    net = NETWORKS[args.architecture](**architecture).to(device)
    net.prior.copy_(_fit_prior(fit.flatten(0, 1), targets.flatten(), len(cards)))
    optimiser = torch.optim.AdamW(net.parameters(), lr=.0008, weight_decay=.025)
    rng = torch.Generator().manual_seed(args.seed)
    best = float("inf")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started_at = datetime.now().isoformat(timespec="milliseconds")
    started = perf_counter()
    for epoch in range(1, args.epochs + 1):
        net.train()
        for ids in torch.randperm(len(fit), generator=rng).to(device).split(192):
            logits = net(fit[ids].flatten(0, 1)).reshape(-1, 2)
            loss = objective(logits, targets[ids])
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        net.eval()
        metrics = assess(net, validation, truth)
        if metrics["neural"]["objective"] < best:
            best, selected = metrics["neural"]["objective"], epoch
            weights = {name: value.detach().cpu().clone() for name, value in net.state_dict().items()}
    elapsed = perf_counter() - started
    ended_at = datetime.now().isoformat(timespec="milliseconds")
    net.load_state_dict(weights)
    args.output.mkdir(parents=True)
    torch.save(weights, args.output / "state_dict.pt")
    config = dict(architecture=args.architecture, network=architecture, seed=args.seed, epochs=args.epochs, selected_epoch=selected,
                  catalog=provenance[0], baseline=provenance[1], data={"train": fit_pools, "validation": val_data["pools"]})
    with torch.inference_mode():
        probes = validation[:8].flatten(0, 1)
        logit = net(probes)
        torch.testing.assert_close(logit, -net(probes.flip(1)))
        save(args.output / "probes.json", {"rosters": probes.tolist(), "logits": logit.tolist()})
    save(args.output / "config.json", config)
    restored, _ = load_value(args.output, "cpu")
    with torch.inference_mode():
        error = (restored(probes.cpu()) - logit.cpu()).abs().max().item()
        torch.testing.assert_close(restored(probes.cpu()), logit.cpu(), atol=2e-4, rtol=2e-4)
    metrics = dict(parameters=sum(p.numel() for p in net.parameters()), torch=str(torch.__version__), device=str(device),
                   seconds=elapsed, train_games=targets.numel(), validation_games=truth.numel(),
                   train=assess(net, fit, targets), validation=assess(net, validation, truth), cpu_reload_max_abs_error=error,
                   weight_bytes=(args.output / "state_dict.pt").stat().st_size,
                   fitting_started_local=started_at, fitting_ended_local=ended_at)
    if device.type == "cuda":
        metrics.update(gpu=torch.cuda.get_device_name(device), peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20)
    save(args.output / "metrics.json", metrics)
    print(json.dumps(metrics), flush=True)


def evaluate(seeds, cards, baseline, checkpoint, mode, margin, preview, opponent, rival_checkpoint):
    torch.set_num_threads(1)
    model, config = load_value(checkpoint, "cpu")
    provenance = (fingerprint(cards), baseline_hash(baseline))
    if (config["catalog"], config["baseline"]) != provenance:
        raise ValueError("Benchmark checkpoint provenance mismatch")
    rival_model = model
    if rival_checkpoint:
        rival_model, rival_config = load_value(rival_checkpoint, "cpu")
        if (rival_config["catalog"], rival_config["baseline"]) != provenance:
            raise ValueError("Rival checkpoint provenance mismatch")
    paired, draft_times, preview_times, errors, counts = {}, [], [], [], Counter()
    search_worker = Showdown() if preview == "rollout" else nullcontext()
    draft_worker = Showdown() if mode in {"search", "search-linear"} or opponent == "search" else nullcontext()
    with Showdown() as engine, search_worker as worker, draft_worker as simulator:
        factory = roster_factory(baseline, cards, worker, "cpu", "prior", preview)
        base = factory()
        rollout = DraftRollout(simulator, cards, base.profiles) if mode in {"search", "search-linear"} else None
        strong = DraftValueRanker(base.draft, model, margin, mode, rollout)
        rival_rollout = DraftRollout(simulator, cards, base.profiles) if opponent == "search" else None
        rival = (DraftValueRanker(base.draft, rival_model, .035, opponent, rival_rollout)
                 if opponent in {"linear", "neural", "search"} else None)
        for seed in seeds:
            results = []
            for left in SEATS:
                policies = {seat: factory(draft_mode="value" if seat == left or opponent != "prior" else "prior")
                            for seat in SEATS}
                policies[left].draft = strong
                if rival is not None:
                    policies[SEATS[1 - SEATS.index(left)]].draft = rival
                samples = {seat: [] for seat in SEATS}
                try:
                    winner = play_game(engine, cards, policies, seed, samples)
                    results.append(.5 if winner is None else float(winner == left))
                except (ValueError, RuntimeError) as error:
                    errors.append({"seed": seed, "seat": left, "error": str(error)})
                for phase, elapsed, event in samples[left]:
                    if phase == DRAFT:
                        draft_times.append(elapsed)
                    elif phase == PREVIEW:
                        preview_times.append(elapsed)
                for seat in SEATS:
                    side = "candidate" if seat == left else "opponent"
                    counts[f"{side}_fallbacks"] += policies[seat].fallbacks
                    for phase, _, event in samples[seat]:
                        counts[f"{phase}_decisions"] += 1
                        if event:
                            counts[event] += 1
            if len(results) == 2:
                paired[seed] = results
        counts.update(strong.activity)
        if rollout is not None:
            counts.update(rollout.activity)
        if rival_rollout is not None:
            counts.update({f"opponent_{field}": value for field, value in rival_rollout.activity.items()})
    return paired, draft_times, preview_times, errors, counts


def benchmark(args, cards, seeds):
    pool_ids = pools_for(cards, seeds)
    config = read(args.checkpoint / "config.json")
    splits = list(config["data"].values()) + list(read(args.baseline / "config.json")["split"].values())
    if args.opponent_checkpoint:
        splits.extend(read(args.opponent_checkpoint / "config.json")["data"].values())
    for inputs in splits:
        disjoint(pool_ids, inputs)
    paired, draft_ms, preview_ms, errors, counts = {}, [], [], [], Counter()
    started = perf_counter()
    for games, drafts, previews, failures, events in parallel(evaluate, seeds, args.workers, cards, args.baseline,
                                                            args.checkpoint, args.mode, args.margin, args.preview, args.opponent,
                                                            args.opponent_checkpoint):
        paired.update(games)
        draft_ms.extend(drafts)
        preview_ms.extend(previews)
        errors.extend(failures)
        counts.update(events)
    elapsed = perf_counter() - started
    scores = [sum(pair) for pair in paired.values()]
    moves = [score for pair in paired.values() for score in pair]
    decisions = sum(counts[f"{phase}_decisions"] for phase in DECISION_LIMIT)
    result = dict(seed=args.seed, pairs=args.pairs, workers=args.workers, mode=args.mode, margin=args.margin, preview=args.preview,
                  checkpoint_sha256=baseline_hash(args.checkpoint), baseline_sha256=baseline_hash(args.baseline),
                  opponent=args.opponent, games=len(moves), wins=moves.count(1), draws=moves.count(.5),
                  score=sum(moves) / len(moves) if moves else None, paired_score_ci95=_paired_win_ci(scores),
                  draft_p95_ms=_percentile(draft_ms, .95), draft_p99_ms=_percentile(draft_ms, .99),
                  preview_p95_ms=_percentile(preview_ms, .95), preview_p99_ms=_percentile(preview_ms, .99),
                  illegal_action_rate=counts["illegal_action"] / decisions if decisions else None,
                  timeout_rate=(counts["auto_action"] + counts["forfeit"]) / decisions if decisions else None,
                  events=dict(counts), errors=errors, seconds=elapsed, games_per_second=len(moves) / elapsed)
    if args.opponent_checkpoint:
        result["opponent_checkpoint_sha256"] = baseline_hash(args.opponent_checkpoint)
    save(args.output, {"result": result, "pools": pool_ids, "paired_scores": dict(sorted(paired.items()))})
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    data, fit, bench = [commands.add_parser(command) for command in ("generate", "train", "bench")]
    for command in (data, fit, bench):
        command.add_argument("--cards", type=Path, default=HISTORICAL_CARDS)
        command.add_argument("--baseline", type=Path, default=HISTORICAL_CARDS.parents[1] / "checkpoints/roster-v1/expanded")
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--seed", type=int, required=True)
    for command in (data, bench):
        command.add_argument("--pairs", type=int, required=True)
        command.add_argument("--workers", type=int, default=8)
    data.add_argument("--behavior", type=Path, help="Mix in a frozen linear Draft completion policy by seed and seat")
    fit.add_argument("--train", type=Path, action="append", required=True)
    fit.add_argument("--validation", type=Path, required=True)
    fit.add_argument("--width", type=int, default=64)
    fit.add_argument("--architecture", choices=tuple(NETWORKS), default="attention")
    fit.add_argument("--epochs", type=int, default=18)
    fit.add_argument("--device", default="cuda")
    bench.add_argument("--checkpoint", type=Path, required=True)
    bench.add_argument("--mode", choices=("neural", "linear", "refit", "search", "search-linear"), default="neural")
    bench.add_argument("--margin", type=float, default=.035)
    bench.add_argument("--preview", choices=("rollout", "tactical"), default="rollout")
    bench.add_argument("--opponent", choices=("prior", "value", "linear", "neural", "search"), default="prior")
    bench.add_argument("--opponent-checkpoint", type=Path, help="Separate checkpoint for a learned rival")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path")
    cards = read(args.cards)
    if args.command == "train":
        if args.epochs < 1 or args.width < 4 or args.width % 4:
            parser.error("Use positive epochs and a width divisible by four")
        train(args, cards)
        return
    if min(args.pairs, args.workers) < 1:
        parser.error("Use positive pair and worker counts")
    if args.command == "bench" and args.opponent_checkpoint and args.opponent not in {"linear", "neural", "search"}:
        parser.error("A separate opponent checkpoint requires a learned rival")
    seeds = range(args.seed, args.seed + args.pairs)
    (generate if args.command == "generate" else benchmark)(args, cards, seeds)


if __name__ == "__main__":
    main()
