"""Compare active and uniform Battle opening labels at equal training budgets."""

import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
from random import Random
from statistics import mean
from time import perf_counter

import torch

from .__main__ import _percentile
from .draft_counterfactual import check_hybrid
from .draft_coverage import CHAMPION, Coverage
from .draft_early import paired_draws
from .draft_last_pick import mean_ci
from .draft_leaf import CATALOG, check_pools
from .draft_league import baseline_hash, disjoint, parallel, pools_for, read, save
from .draft_matchups import ROOT
from .features import BattleEncoder
from .game import BATTLE, BATTLE_BANK, DECISION_LIMIT, DRAFT
from .intervention import _encode
from .lineup import index_cards
from .opening_audit import MissingOpening, trial
from .opening_value import MARGIN, assess, load_model, opening_rosters, paired_loss
from .sim import SEATS, fingerprint


PRIOR = ROOT / "checkpoints" / "opening-value-v1" / "rank"
STYLES = ("joint", "forecast-replies", "tactical")
RIVALS = ("champion", "search", "pairwise")
METHODS = ("joint", "prior", "uniform", "active")


class EncodedOpening(Exception):
    pass


def check_fresh(pools, extra):
    prior = [p for group, experiment in (("benchmark_results", "opening-audit-v1"), ("checkpoints", "draft-leaf-v1"))
             for p in (ROOT / group / experiment).glob("*.json") if "pools" in read(p)]
    check_pools(pools, [*prior, *extra])


def inner_games(experiment):
    return {"draft_and_preview_games": experiment.preview.games,
            "champion_review_games": experiment.activity["full_leaf_games"],
            "prefix_rollout_games": experiment.hybrid.rollout.activity["rollout_games"] +
            experiment.league.rollout.activity["rollout_games"]}


@torch.inference_mode()
def screen(roots, cards):
    torch.set_num_threads(1)
    prior, indexed = load_model(PRIOR, cards), index_cards(cards)
    rows = []
    with ExitStack() as stack:
        experiment = Coverage(stack, cards)
        encoder = BattleEncoder(experiment.profiles)
        for root in roots:
            actor = SEATS[root % 2]
            rival = RIVALS[root // 2 % len(RIVALS)]
            rosters, actual, lineups, _ = opening_rosters(experiment, indexed, cards, root, actor, rival)
            captured = {}

            def encode(policy, decision, ranked):
                state, actions = _encode(encoder, policy.history, decision, ranked)
                p = prior(state[None], actions[None]).sigmoid()[0]
                best = int(p[1:].argmax()) + 1
                gap = float(p[best] - p[0])
                joint_gap = ranked[0][1]["score"] - ranked[best][1]["score"]
                legal = [decision.validate(action) for action, _ in ranked]
                captured.update(state=state.tolist(), features=actions.tolist(), gap=gap,
                                priority=gap - .08 * joint_gap, actions=legal,
                                opening=fingerprint([decision.observation, decision.move_info, legal]))
                raise EncodedOpening

            try:
                trial(experiment, rosters, lineups, actor, actual, "joint", encode)
            except EncodedOpening:
                rows.append(dict(root=root, actor=actor, rival=rival, rosters=rosters, actual=actual,
                                 lineups=lineups, **captured))
            except MissingOpening:
                continue
        counts = inner_games(experiment)
    return rows, counts


def label(rows, cards, draws):
    labeled = []
    with ExitStack() as stack:
        experiment = Coverage(stack, cards)
        for row in rows:
            reference = (row["opening"], row["actions"])
            values = []
            seeds = paired_draws(row["root"], tuple(row["actual"]), (), len(STYLES), draws)
            for style, group in zip(STYLES, seeds, strict=True):
                outcomes = []
                for seed in group:
                    rewards = []
                    for i in range(len(row["actions"])):
                        opening, result = trial(experiment, row["rosters"], row["lineups"], row["actor"], seed, style, i)
                        if opening != reference:
                            raise ValueError(f"Action or public state changed before intervention at {row['root']}")
                        rewards.append(result)
                    outcomes.append(rewards)
                values.append(outcomes)
            labeled.append({field: row[field] for field in ("root", "actor", "rival", "state", "features", "gap", "priority")} |
                           {"opening": reference[0], "actions": row["actions"], "outcomes": values})
    return labeled


def acquire(args, cards):
    train_roots = range(args.seed, args.seed + args.count)
    val_roots = range(args.validation_seed, args.validation_seed + args.validation_count)
    pools, val_pools = pools_for(cards, train_roots), pools_for(cards, val_roots)
    disjoint(pools, val_pools)
    check_fresh(pools | val_pools, ())
    torch.set_num_threads(1)
    prior_sha = baseline_hash(PRIOR)
    began = perf_counter()
    counts, screened = Counter(), {}
    for rows, events in parallel(screen, [*train_roots, *val_roots], args.workers, cards):
        screened.update((row["root"], row) for row in rows)
        counts.update(events)
    available = [root for root in train_roots if root in screened]
    validation = [root for root in val_roots if root in screened]
    if len(available) < args.sample_count or len(validation) < .9 * len(val_roots):
        raise RuntimeError("Too few openings support the same three legal Joint candidates")
    rng = Random(args.seed ^ 0xBA771E)
    rng.shuffle(available)
    shared_count = args.sample_count // 4
    shared, rest = available[:shared_count], available[shared_count:]
    active = sorted(rest, key=lambda root: screened[root]["priority"], reverse=True)
    groups = {"uniform": available[:args.sample_count], "active": shared + active[:args.sample_count - shared_count],
              "validation": validation}
    selected = sorted(set().union(*groups.values()))
    screening_seconds = perf_counter() - began
    print(json.dumps(dict(screened=len(screened), unique_label_roots=len(selected), screening_seconds=screening_seconds)), flush=True)
    began = perf_counter()
    rows = [row for part in parallel(label, [screened[root] for root in selected], args.workers, cards, args.draws) for row in part]
    counts["terminal_label_games"] = len(rows) * len(STYLES) * args.draws * 3
    report = dict(screened=len(screened), labeled=len(rows), draws=args.draws, workers=args.workers,
                  training_overlap=len(set(groups["uniform"]) & set(groups["active"])),
                  previous_overrides={key: sum(screened[i]["gap"] >= MARGIN for i in ids) for key, ids in groups.items()},
                  screening_seconds=screening_seconds, label_seconds=perf_counter() - began, counts=dict(counts))
    if baseline_hash(PRIOR) != prior_sha:
        raise ValueError("Prior weights changed during sampling")
    save(args.output, dict(report=report, pools=pools | val_pools, splits=groups, styles=STYLES, rows=rows,
                           champion=CHAMPION, prior_sha=prior_sha, seed=args.seed))
    print(json.dumps(report), flush=True)


@torch.inference_mode()
def metrics(model, dataset):
    state, actions, outcomes = dataset
    p = model(state, actions).sigmoid()
    maximum, best = p.max(1)
    choice = torch.where(maximum - p[:, 0] >= MARGIN, best, 0)
    local_gain = outcomes.gather(-1, choice[:, None, None, None].expand(*outcomes.shape[:3], 1)).squeeze(-1) - outcomes[..., 0]
    return assess(model, dataset) | {
        "delta_by_style": dict(zip(STYLES, local_gain.mean((0, 2)).tolist(), strict=True)),
        "interval": mean_ci(local_gain.mean((1, 2)).cpu().tolist()),
    }


def replicated_gain(outcomes):
    """Select on the first half of each style, then replay the same action choices."""
    half = outcomes.shape[2] // 2
    choice = outcomes[:, :, :half].mean((1, 2)).argmax(1)
    retained = outcomes[:, :, half:]
    selected = retained.gather(-1, choice[:, None, None, None].expand(*retained.shape[:3], 1)).squeeze(-1)
    change = selected - retained[..., 0]
    return dict(overrides=int((choice != 0).sum()), delta=change.mean().item(),
                interval=mean_ci(change.mean((1, 2)).cpu().tolist()),
                by_style=dict(zip(STYLES, change.mean((0, 2)).tolist(), strict=True)))


def fit(args, cards):
    source = read(args.data)
    if (source["champion"], tuple(source["styles"]), source["prior_sha"]) != (CHAMPION, STYLES, baseline_hash(PRIOR)):
        raise ValueError("Incompatible counterfactual labels or prior weights")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    base_config = read(PRIOR / "config.json")
    rows = {row["root"]: row for row in source["rows"]}
    datasets = {name: tuple(torch.tensor([rows[root][field] for root in ids], dtype=torch.float32, device=device)
                            for field in ("state", "features", "outcomes")) for name, ids in source["splits"].items()}
    validation = datasets["validation"]
    results = {"prior": metrics(load_model(PRIOR, cards).to(device), validation),
               "label_audit": {name: replicated_gain(data[2]) for name, data in datasets.items()}}
    for name in ("uniform", "active"):
        model = load_model(PRIOR, cards).to(device).train()
        generator = torch.Generator().manual_seed(args.seed)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0004, weight_decay=.05)
        lowest, chosen, best = float("inf"), 0, None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        began = perf_counter()
        data = datasets[name]
        for epoch in range(24):
            for indices in torch.randperm(len(data[0]), generator=generator).to(device).split(24):
                state, actions, targets = (tensor[indices] for tensor in data)
                loss = paired_loss(model(state, actions), targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            score = assess(model, validation)["paired_mse"]
            if score < lowest:
                lowest, chosen = score, epoch + 1
                best = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = perf_counter() - began
        path = args.output / name
        path.mkdir(parents=True)
        model.load_state_dict(best)
        torch.save(best, path / "state_dict.pt")
        splits = {split: {str(root): source["pools"][str(root)] for root in source["splits"][origin]}
                  for split, origin in (("train", name), ("validation", "validation"))}
        save(path / "config.json", base_config | dict(epoch=chosen, method=name, seed=args.seed, prior_sha=source["prior_sha"],
                                                     sampling_seed=source["seed"], splits=splits, pools=source["pools"],
                                                     learning_rate=.0004, max_epochs=24, paired_draws=source["report"]["draws"]))
        probe = [x[:8] for x in validation[:2]]
        with torch.inference_mode():
            expected = model(*probe).cpu()
            actual = load_model(path, cards)(*(x.cpu() for x in probe))
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        save(path / "probes.json", {"input": [x.tolist() for x in probe], "logits": actual.tolist()})
        report = dict(validation=metrics(model, validation), train=metrics(model, data), training_seconds=elapsed,
                      weights_bytes=(path / "state_dict.pt").stat().st_size,
                      parameters=sum(p.numel() for p in model.parameters()), epoch=chosen, device=str(device),
                      reload_error=float((actual - expected).abs().max()))
        if device.type == "cuda":
            report.update(gpu=torch.cuda.get_device_name(device), peak_mib=torch.cuda.max_memory_allocated(device) / 2**20)
        save(path / "metrics.json", report)
        results[name] = report
    save(args.output / "report.json", results)
    print(json.dumps(results), flush=True)


@torch.inference_mode()
def evaluate(seeds, cards, checkpoint):
    torch.set_num_threads(1)
    models = {name: load_model(PRIOR if name == "prior" else checkpoint / name, cards) for name in METHODS[1:]}
    indexed = index_cards(cards)
    records, errors = {}, []
    with ExitStack() as stack:
        experiment = Coverage(stack, cards)
        encoder = BattleEncoder(experiment.profiles)
        for root in seeds:
            matchup = root % 5
            rival = ("champion", "champion", "search", "pairwise", "champion")[matchup]
            style = ("joint", "joint", "joint", "forecast", "forecast-replies")[matchup]
            pair = []
            for actor in SEATS:
                try:
                    rosters, actual, lineups, draft_times = opening_rosters(experiment, indexed, cards, root, actor, rival)
                    moves, latencies = dict.fromkeys(METHODS, 0), {}

                    def predict(policy, decision, ranked):
                        began = perf_counter()
                        state, actions = _encode(encoder, policy.history, decision, ranked)
                        encoding = perf_counter() - began
                        for name, model in models.items():
                            began = perf_counter()
                            p = model(state[None], actions[None]).sigmoid()[0]
                            best = int(p.argmax())
                            moves[name] = best if float(p[best] - p[0]) >= MARGIN else 0
                            latencies[name] = (encoding + perf_counter() - began) * 1000
                        return 0

                    opening, before = trial(experiment, rosters, lineups, actor, actual, style, predict, strict=False)
                    # The baseline callback scored all models together. Replay branch zero for clean latency.
                    times = []
                    repeat, baseline = trial(experiment, rosters, lineups, actor, actual, style, 0, strict=False,
                                             timings=times)
                    if repeat != opening or baseline != before:
                        raise ValueError("Joint control is not reproducible")
                    branches = {0: (before, times)}
                    for i in sorted(set(moves.values()) - {0}):
                        times = []
                        repeat, value = trial(experiment, rosters, lineups, actor, actual, style, i, strict=False,
                                              timings=times)
                        if repeat != opening:
                            raise ValueError("Benchmark changed before intervention")
                        branches[i] = (value, times)
                    battle_ms = {name: branches[i][1].copy() for name, i in moves.items()}
                    for name, elapsed in latencies.items():
                        battle_ms[name][0] += elapsed
                    pair.append(dict(choices=moves, times=latencies, rival=rival, style=style, draft_max_ms=max(draft_times),
                                     wins={name: branches[i][0] for name, i in moves.items()}, battle_ms=battle_ms,
                                     simulated_games=1 + len(branches)))
                except (ValueError, OSError, RuntimeError, KeyError) as error:
                    errors.append(dict(seed=root, seat=actor, issue=str(error)))
            if len(pair) == 2:
                records[root] = pair
        counts = inner_games(experiment)
    return records, errors, counts


def differences(pairs, method, reference):
    return [mean(row["wins"][method] - row["wins"][reference] for row in pair) for pair in pairs]


def comparison(pairs, method, reference):
    delta = differences(pairs, method, reference)
    return dict(delta=mean(delta), ci95=mean_ci(delta))


def summarize(pairs):
    rows = [row for pair in pairs for row in pair]
    report = {}
    for name in METHODS:
        changes = [row for row in rows if row["choices"][name]]
        latencies = [row["times"][name] for row in rows if name in row["times"]]
        opening = [row["battle_ms"][name][0] for row in rows if row["battle_ms"][name]]
        battle = [ms for row in rows for ms in row["battle_ms"][name]]
        report[name] = dict(wins=sum(row["wins"][name] == 1 for row in rows), score=sum(row["wins"][name] for row in rows),
                            eligible=len(latencies) if name != "joint" else len(opening),
                            overrides=len(changes), converted=sum(row["wins"][name] > row["wins"]["joint"] for row in changes),
                            lost=sum(row["wins"][name] < row["wins"]["joint"] for row in changes),
                            gate_p95_ms=_percentile(latencies, .95), gate_p99_ms=_percentile(latencies, .99),
                            opening_p95_ms=_percentile(opening, .95), opening_p99_ms=_percentile(opening, .99),
                            battle_p95_ms=_percentile(battle, .95), battle_p99_ms=_percentile(battle, .99),
                            vs_joint=comparison(pairs, name, "joint"))
    return dict(methods=report, active_vs_uniform=comparison(pairs, "active", "uniform"),
                active_vs_prior=comparison(pairs, "active", "prior"))


def bench(args, cards):
    seeds = range(args.seed, args.seed + args.count)
    pools = pools_for(cards, seeds)
    check_fresh(pools, args.exclude)
    for method in METHODS[2:]:
        config = read(args.checkpoint / method / "config.json")
        if config["prior_sha"] != baseline_hash(PRIOR):
            raise ValueError("Frozen prior differs from training initialization")
        disjoint(pools, config["pools"])
    began = perf_counter()
    records, errors, counts = {}, [], Counter()
    for data, issues, counters in parallel(evaluate, seeds, args.workers, cards, args.checkpoint):
        records.update(data)
        errors.extend(issues)
        counts.update(counters)
    if len(records) != args.count or errors:
        raise RuntimeError(f"Benchmark is incomplete: {len(records)}/{args.count} pairs, {errors[:5]}")
    pairs = [records[root] for root in sorted(records)]
    rows = [row for pair in pairs for row in pair]
    counts["terminal_games"] = sum(row["simulated_games"] for row in rows)
    report = dict(pairs=len(pairs), games=len(rows), seconds=perf_counter() - began, counts=dict(counts),
                  illegal_actions=0, draft_timeouts=sum(row["draft_max_ms"] >= 1000 * DECISION_LIMIT[DRAFT] for row in rows),
                  battle_decisions_over_limit=sum(ms >= 1000 * DECISION_LIMIT[BATTLE] for row in rows
                                                  for game in row["battle_ms"].values() for ms in game),
                  battle_banks_over_limit=sum(sum(game) >= 1000 * BATTLE_BANK for row in rows for game in row["battle_ms"].values()))
    matchups = {(row["rival"], row["style"]) for row in rows}
    report.update(summarize(pairs), by_opponent={f"{rival}/{style}": summarize([pair for pair in pairs
                  if (pair[0]["rival"], pair[0]["style"]) == (rival, style)]) for rival, style in sorted(matchups)})
    save(args.output, dict(report=report, pools=pools, paired=records, champion=CHAMPION))
    print(json.dumps({key: value for key, value in report.items() if key != "by_opponent"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sample, trainer, test = (sub.add_parser(name) for name in ("collect", "train", "bench"))
    for step in (sample, trainer, test):
        step.add_argument("--seed", type=int, required=True)
        step.add_argument("--output", type=Path, required=True)
    for name in ("count", "sample-count", "validation-seed", "validation-count"):
        sample.add_argument(f"--{name}", type=int, required=True)
    sample.add_argument("--draws", type=int, default=6)
    trainer.add_argument("--device", default="cuda")
    trainer.add_argument("--data", type=Path, required=True)
    test.add_argument("--checkpoint", type=Path, required=True)
    test.add_argument("--count", type=int, required=True)
    test.add_argument("--exclude", nargs="*", type=Path, default=[])
    for step in (sample, test):
        step.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    counts = [getattr(args, key, 1) for key in ("count", "sample_count", "validation_count", "workers")]
    if args.output.exists() or min(counts) < 1 or getattr(args, "draws", 4) < 4:
        parser.error("Use a fresh path, positive sizes and at least four paired draws")
    check_hybrid()
    {"collect": acquire, "train": fit, "bench": bench}[args.command](args, read(CATALOG))


if __name__ == "__main__":
    main()
