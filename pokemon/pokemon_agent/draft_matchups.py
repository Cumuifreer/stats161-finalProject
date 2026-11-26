"""Collect strong-player draft traces, fit a pick model, and run paired full games."""

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime
import json
from pathlib import Path
from random import Random
from time import perf_counter

import torch
from torch.nn import functional as F

from .__main__ import HISTORICAL_CARDS, _paired_win_ci, _percentile
from .draft_forecast import (ForecastDraft, PREDICTORS, STYLES, UNKNOWN_STYLE, TailReview,
                             encode_owners, legal_cards, load_predictor, public_draft)
from .draft_league import baseline_hash, disjoint, parallel, pools_for, read, save
from .draft_value import DraftRollout, DraftValueRanker, load_value
from .game import BATTLE, DRAFT, PREVIEW
from .roster import roster_factory
from .sim import SEATS, Showdown, fingerprint, play_game

ROOT = HISTORICAL_CARDS.parents[1]
BASELINE = ROOT / "checkpoints/roster-v1/expanded"
PAIRWISE = ROOT / "checkpoints/draft-v2/league-pairwise"
ATTENTION = ROOT / "checkpoints/draft-v2/neural-pilot"
FROZEN = {
    "prior": "a458a24568361e1515be1ee13e7b8bb094b0e8147a0799e1eecec4ccedc33d5c",
    "pairwise": "c73d26549db2096082cdc93b13191114c99d95156ae9927f4a7f7075e8955927",
    "search": "5d9bfa8863f5891f8573950b8843c504ce5ca5a009c077ce9ed9297b35a79392",
}
PLANNERS = ("forecast", "hybrid", "control", "tail", "exact")


def check_frozen():
    paths = {"prior": BASELINE, "pairwise": PAIRWISE, "search": ATTENTION}
    if any(baseline_hash(path) != FROZEN[style] for style, path in paths.items()):
        raise ValueError("A frozen Draft checkpoint has changed")


class League:
    def __init__(self, stack, cards, preview, needs_search):
        torch.set_num_threads(1)
        self.engine = stack.enter_context(Showdown())
        preview_worker = stack.enter_context(Showdown()) if preview == "rollout" else None
        self.simulator = stack.enter_context(Showdown()) if needs_search else None
        self.factory = roster_factory(BASELINE, cards, preview_worker, "cpu", "prior", preview)
        base = self.factory()
        self.prior = base.draft
        self.rollout = DraftRollout(self.simulator, cards, base.profiles) if self.simulator else None
        pairwise, _ = load_value(PAIRWISE, "cpu")
        attention, _ = load_value(ATTENTION, "cpu")
        self.ranks = {
            "pairwise": DraftValueRanker(self.prior, pairwise, .035, "neural"),
            "search": DraftValueRanker(self.prior, attention, .035, "neural", self.rollout),
            "linear": DraftValueRanker(self.prior, attention, .035, "linear"),
        }

    def policy(self, style):
        agent = self.factory(draft_mode="prior" if style == "prior" else "value")
        if style in self.ranks:
            agent.draft = self.ranks[style]
        return agent


def trace_pick(decision, action, style, prior):
    pool, own, foe, _, pick, _ = public_draft(decision, prior.index)
    owners = encode_owners(pool, own, foe)
    mask = legal_cards(pool, owners, prior.items)
    actual = {prior.index[option["card_id"]] for option in decision.actions}
    if {card for card, enabled in zip(pool, mask, strict=True) if enabled} != actual:
        raise ValueError("Reconstructed draft mask differs from Decision.actions")
    card = prior.index[action["card_id"]]
    if card not in actual:
        raise ValueError("Draft trace contains an illegal card")
    covered = prior._coverage(own)
    scores = [prior.strength[card] + .25 * prior._support(card, covered) for card in pool]
    reference = max((pool.index(prior.index[option["card_id"]]) for option in decision.actions), key=scores.__getitem__)
    return [owners, STYLES.index(style), pool.index(card), reference]


def collect(seeds, cards, search_every):
    rows, styles, errors = [], Counter(), []
    with ExitStack() as stack:
        league = League(stack, cards, "tactical", any(seed % search_every == 0 for seed in seeds))
        for seed in seeds:
            rng = Random(seed ^ 0x91A7E5)
            first = "search" if seed % search_every == 0 else rng.choice(("pairwise", "pairwise", "prior", "value", "linear"))
            second = rng.choice(("pairwise", "pairwise", "prior", "value", "linear"))
            games = []
            for seats in ((first, second), (second, first)):
                agents = {seat: league.policy(style) for seat, style in zip(SEATS, seats, strict=True)}
                picks, pool = [], []

                def choose(seat, decision):
                    action = agents[seat](decision)
                    if decision.phase == DRAFT:
                        if not pool:
                            pool.extend(sorted(league.prior.index[card["card_id"]] for card in decision.observation["cards"]))
                        picks.append(trace_pick(decision, action, seats[SEATS.index(seat)], league.prior))
                    return action

                policies = {seat: (lambda decision, seat=seat: choose(seat, decision)) for seat in SEATS}
                samples = {seat: [] for seat in SEATS}
                try:
                    winner = play_game(league.engine, cards, policies, seed, samples)
                except (RuntimeError, ValueError) as error:
                    errors.append({"seed": seed, "message": str(error)})
                    break
                if len(picks) != 12 or any(event for decisions in samples.values() for _, _, event in decisions):
                    errors.append({"seed": seed, "message": "Invalid game or timing event"})
                    break
                games.append({"pool": pool, "picks": picks, "score": .5 if winner is None else float(winner == "p1")})
            if len(games) == 2:
                styles.update([first, second] * 2)
                rows.append({"seed": seed, "games": games})
    return rows, styles, errors


def prior_splits():
    yield from read(BASELINE / "config.json")["split"].values()
    for path in (PAIRWISE, ATTENTION):
        yield from read(path / "config.json")["data"].values()


def generate(args, cards, pools, seeds):
    rows, styles, errors = [], Counter(), []
    ready = perf_counter()
    for records, labels, issues in parallel(collect, seeds, args.workers, cards, args.search_every):
        rows.extend(records)
        styles.update(labels)
        errors.extend(issues)
    games = len(rows) * 2
    seconds = perf_counter() - ready
    report = {"seed": args.seed, "workers": args.workers, "pairs": len(rows), "games": games,
              "picks": games * 12, "policy_seats": dict(styles), "errors": errors,
              "seconds": seconds, "games_per_second": games / seconds, "search_every": args.search_every}
    save(args.output, {"metrics": report, "pools": pools, "catalog": fingerprint(cards), "frozen": FROZEN,
                       "rows": sorted(rows, key=lambda row: row["seed"])})
    print(json.dumps(report), flush=True)


def data_tensors(dataset, items, device):
    pools, owners, plies, styles, masks, targets, references = [], [], [], [], [], [], []
    for pair in dataset["rows"]:
        for game in pair["games"]:
            pool = game["pool"]
            for ply, (tags, style, label, reference) in enumerate(game["picks"]):
                pools.append(pool)
                owners.append(tags)
                plies.append(ply)
                styles.append(style)
                mask = legal_cards(pool, tags, items)
                if not mask[label] or not mask[reference]:
                    raise ValueError("Invalid cached draft label")
                masks.append(mask)
                targets.append(label)
                references.append(reference)
    return tuple(torch.tensor(column, device=device) for column in (pools, owners, plies, styles, masks, targets, references))


@torch.inference_mode()
def prediction_metrics(model, data, *, known=False):
    pool, owners, ply, styles, mask, labels, prior = data
    conditioning = styles if known else torch.full_like(styles, UNKNOWN_STYLE)
    logits = torch.cat([model(*batch) for batch in zip(*(tensor.split(1024) for tensor in
                                                        (pool, owners, ply, conditioning, mask, prior)), strict=True)])
    top = logits.topk(3).indices
    correct = top[:, 0] == labels
    return {"cross_entropy": F.cross_entropy(logits, labels).item(), "top1": correct.float().mean().item(),
            "top3": (top == labels[:, None]).any(1).float().mean().item(), "prior_top1": (prior == labels).float().mean().item(),
            "early_top1": correct[ply < 8].float().mean().item(), "late_top1": correct[ply >= 8].float().mean().item(),
            "by_style_top1": {name: correct[styles == index].float().mean().item()
                              for index, name in enumerate(STYLES) if (styles == index).any()}}


def train(args, cards):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    datasets = read(args.train), read(args.validation)
    catalog = fingerprint(cards)
    if any(dataset["catalog"] != catalog or dataset["frozen"] != FROZEN or dataset["metrics"]["errors"] for dataset in datasets):
        raise ValueError("Training dataset does not match the frozen league")
    disjoint(datasets[0]["pools"], datasets[1]["pools"])
    prior = roster_factory(BASELINE, cards, None, "cpu", "prior", "tactical")().draft
    device = torch.device(args.device)
    fit, validation = (data_tensors(dataset, prior.items, device) for dataset in datasets)
    model = PREDICTORS[args.architecture](len(cards), args.width).to(device)
    model.strength.copy_(torch.tensor(prior.strength, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.02)
    generator = torch.Generator().manual_seed(args.seed)
    lowest = float("inf")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started_at = datetime.now().isoformat(timespec="milliseconds")
    started = perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for indices in torch.randperm(len(fit[0]), generator=generator).to(device).split(384):
            pool, owners, ply, style, mask, labels, reference = (tensor[indices] for tensor in fit)
            style = style.masked_fill(torch.rand(len(style), device=device) < .5, UNKNOWN_STYLE)
            loss = F.cross_entropy(model(pool, owners, ply, style, mask, reference), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        results = prediction_metrics(model, validation)
        if results["cross_entropy"] < lowest:
            lowest, selected = results["cross_entropy"], epoch
            weights = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    seconds = perf_counter() - started
    ended_at = datetime.now().isoformat(timespec="milliseconds")
    model.load_state_dict(weights)
    config = {"architecture": args.architecture, "network": {"card_count": len(cards), "width": args.width},
              "seed": args.seed, "epochs": args.epochs,
              "selected_epoch": selected, "catalog": catalog, "frozen": FROZEN,
              "splits": dict(zip(("train", "validation"), (dataset["pools"] for dataset in datasets), strict=True))}
    args.output.mkdir(parents=True)
    torch.save(weights, args.output / "state_dict.pt")
    save(args.output / "config.json", config)
    probe = [tensor[:16] for tensor in (*validation[:5], validation[-1])]
    with torch.inference_mode():
        logits = model(*probe)
        reloaded, _ = load_predictor(args.output)
        reference = reloaded(*(tensor.cpu() for tensor in probe))
        probability_gap = (reference.softmax(-1) - logits.cpu().softmax(-1)).abs().max().item()
        torch.testing.assert_close(reference.softmax(-1), logits.cpu().softmax(-1), atol=6e-4, rtol=0)
        torch.testing.assert_close(reference.argmax(-1), logits.cpu().argmax(-1))
        save(args.output / "probes.json", {"input": [tensor.tolist() for tensor in probe],
                                          "logits": reference.masked_fill(~probe[4].cpu(), 0.).tolist()})
    report = {"torch": str(torch.__version__), "device": str(device), "seconds": seconds,
              "fitting_started_local": started_at, "fitting_ended_local": ended_at,
              "training_picks": len(fit[0]), "validation_picks": len(validation[0]),
              "cpu_reload_max_probability_gap": probability_gap,
              "parameters": sum(p.numel() for p in model.parameters()), "weight_bytes": (args.output / "state_dict.pt").stat().st_size,
              "validation": prediction_metrics(model, validation), "known_style_validation": prediction_metrics(model, validation, known=True)}
    if device.type == "cuda":
        report.update(gpu=torch.cuda.get_device_name(device), peak_mib=torch.cuda.max_memory_allocated(device) / 2**20)
    save(args.output / "metrics.json", report)
    print(json.dumps(report), flush=True)


def evaluate(seeds, cards, checkpoint, method, opponent, preview):
    scores, timings, issues, counts = {}, [], [], Counter()
    with ExitStack() as stack:
        model, _ = load_predictor(checkpoint)
        league = League(stack, cards, preview, method != "forecast" or opponent in {"search", "control"})
        review = TailReview if method == "tail" else DraftRollout
        rollout = None if method == "forecast" else review(league.simulator, cards, league.factory().profiles)
        challenger = ForecastDraft(league.ranks["pairwise"], model, method, rollout)
        rival = (ForecastDraft(league.ranks["pairwise"], model, "control",
                              DraftRollout(league.simulator, cards, league.factory().profiles)) if opponent == "control" else None)
        for seed in seeds:
            pair = []
            for left in SEATS:
                policies = {seat: league.policy("pairwise" if seat == left or rival is not None else opponent) for seat in SEATS}
                policies[left].draft = challenger
                if rival is not None:
                    policies[SEATS[1 - SEATS.index(left)]].draft = rival
                samples = {seat: [] for seat in SEATS}
                try:
                    winner = play_game(league.engine, cards, policies, seed, samples)
                    pair.append(.5 if winner is None else float(winner == left))
                except (RuntimeError, ValueError) as error:
                    issues.append({"seed": seed, "seat": left, "message": str(error)})
                timings.extend(time for phase, time, _ in samples[left] if phase == DRAFT)
                for seat in SEATS:
                    counts["candidate_fallbacks" if seat == left else "opponent_fallbacks"] += policies[seat].fallbacks
                    for phase, _, event in samples[seat]:
                        counts[phase] += 1
                        if event:
                            counts[event] += 1
            if len(pair) == 2:
                scores[seed] = pair
        counts.update(challenger.activity)
        if rollout:
            counts.update(rollout.activity)
        if rival is not None:
            counts.update({f"opponent_{name}": count for name, count in rival.rollout.activity.items()})
    return scores, timings, issues, counts


def benchmark(args, cards, pools, seeds):
    config = read(args.checkpoint / "config.json")
    if config["catalog"] != fingerprint(cards) or config["frozen"] != FROZEN:
        raise ValueError("Opponent model checkpoint provenance mismatch")
    for split in config["splits"].values():
        disjoint(pools, split)
    pairs, timings, issues, counts = {}, [], [], Counter()
    started = perf_counter()
    for results, times, errors, events in parallel(evaluate, seeds, args.workers, cards, args.checkpoint,
                                                   args.method, args.opponent, args.preview):
        pairs.update(results)
        timings.extend(times)
        issues.extend(errors)
        counts.update(events)
    games = [score for pair in pairs.values() for score in pair]
    seconds = perf_counter() - started
    decisions = counts[DRAFT] + counts[PREVIEW] + counts[BATTLE]
    report = {"method": args.method, "opponent": args.opponent, "preview": args.preview, "seed": args.seed,
              "workers": args.workers, "pairs": args.pairs, "games": len(games), "wins": games.count(1.), "draws": games.count(.5),
              "score": sum(games) / len(games) if games else None, "paired_score_ci95": _paired_win_ci([sum(pair) for pair in pairs.values()]),
              "draft_p95_ms": _percentile(timings, .95), "draft_p99_ms": _percentile(timings, .99),
              "illegal_action_rate": counts["illegal_action"] / decisions if decisions else None,
              "timeout_rate": (counts["auto_action"] + counts["forfeit"]) / decisions if decisions else None,
              "events": dict(counts), "errors": issues, "seconds": seconds, "games_per_second": len(games) / seconds,
              "checkpoint_sha256": baseline_hash(args.checkpoint), "frozen": FROZEN}
    save(args.output, {"result": report, "pools": pools, "paired_scores": dict(sorted(pairs.items()))})
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen, fit, bench = [sub.add_parser(name) for name in ("generate", "train", "bench")]
    for command in (gen, fit, bench):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--seed", type=int, required=True)
    for command in (gen, bench):
        command.add_argument("--pairs", type=int, required=True)
        command.add_argument("--workers", type=int, default=10)
    gen.add_argument("--search-every", type=int, default=6)
    fit.add_argument("--train", type=Path, required=True)
    fit.add_argument("--validation", type=Path, required=True)
    fit.add_argument("--width", type=int, default=48)
    fit.add_argument("--architecture", choices=tuple(PREDICTORS), default="attention")
    fit.add_argument("--epochs", type=int, default=16)
    fit.add_argument("--device", default="cuda")
    bench.add_argument("--checkpoint", type=Path, required=True)
    bench.add_argument("--method", choices=PLANNERS, default="hybrid")
    bench.add_argument("--opponent", choices=(*STYLES, "control"), default="pairwise")
    bench.add_argument("--preview", choices=("rollout", "tactical"), default="rollout")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output path")
    check_frozen()
    cards = read(HISTORICAL_CARDS)
    if args.command == "train":
        if args.width < 4 or args.width % 4 or args.epochs < 1:
            parser.error("Use a positive width divisible by four and positive epochs")
        train(args, cards)
        return
    if min(args.pairs, args.workers, getattr(args, "search_every", 1)) < 1:
        parser.error("Use positive pair and worker counts")
    seeds = range(args.seed, args.seed + args.pairs)
    pools = pools_for(cards, seeds)
    for split in prior_splits():
        disjoint(pools, split)
    (generate if args.command == "generate" else benchmark)(args, cards, pools, seeds)


if __name__ == "__main__":
    main()
