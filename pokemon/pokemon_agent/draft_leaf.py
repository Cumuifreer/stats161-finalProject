"""Calibrate paired Draft continuation values against complete battle outcomes."""

import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
from random import Random
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from .draft_advantage import load_advantage
from .draft_counterfactual import LabelGames, check_hybrid
from .draft_coverage import CHAMPION
from .draft_last_pick import other, review_seeds, sample_game
from .draft_league import baseline_hash, disjoint, parallel, pools_for, read, save
from .draft_matchups import FROZEN, PAIRWISE, ROOT
from .draft_value import DraftValue, load_value
from .sim import SEATS, fingerprint

CATALOG = ROOT / "data/historical_cards.json"
RANK = ROOT / "checkpoints/draft-early-v1/rank"
MARGIN = .075
DOWNSIDE = -.05


class LeafValue(nn.Module):
    def __init__(self, prior, width=48):
        super().__init__()
        self.prior = prior.requires_grad_(False)
        self.residual = DraftValue(len(prior.prior), width)
        nn.init.zeros_(self.residual.score[-1].weight)
        nn.init.zeros_(self.residual.score[-1].bias)

    def forward(self, rosters):
        return self.prior(rosters) + self.residual(rosters)


def provenance(cards):
    return {"catalog": fingerprint(cards), "champion": CHAMPION, "prior": FROZEN["pairwise"], "proposal": baseline_hash(RANK)}


def load_leaf(path, cards):
    config = read(path / "config.json")
    if config["provenance"] != provenance(cards):
        raise ValueError("Draft leaf checkpoint provenance mismatch")
    model = LeafValue(load_value(PAIRWISE, "cpu")[0], config["width"])
    model.load_state_dict(torch.load(path / "state_dict.pt", map_location="cpu", weights_only=True))
    return model.eval()


@torch.inference_mode()
def advantages(model, pairs):
    teams = torch.as_tensor(pairs)
    value = model(teams.flatten(0, 1)).sigmoid().view(len(pairs), 2)
    return value[:, 1] - value[:, 0]


def accept(gains):
    return float(gains.mean()) >= MARGIN and float(gains.min()) >= DOWNSIDE


def collect(seeds, cards, draws):
    from .draft_early import (LEAGUE, EarlyLeague, acting_seat, paired_draws, prefix_for, projected_options,
                              propose, replay_completion)

    rows, errors, stats = [], [], Counter()
    with ExitStack() as stack:
        experiment = EarlyLeague(stack, cards)
        games = LabelGames(experiment.league.engine, experiment.preview, experiment.profiles, cards)
        model, _ = load_advantage(RANK)
        for root in seeds:
            pool, first, actual = sample_game(cards, root)
            pick = 5 + root % 6
            actor = acting_seat(first, pick)
            rng = Random(root ^ 0x5AD02)
            try:
                styles = {actor: LEAGUE[root // 6 % 3], other(actor): rng.choice(LEAGUE)}
                _, observed, decision = prefix_for(experiment, pool, first, pick, styles)
                if observed != actor:
                    raise ValueError("Reached the wrong seat")
                anchor = decision.validate(experiment.choose(decision, "hybrid"))
                actions, _, encoding, slots = projected_options(experiment, decision, anchor)
                if len(actions) < 2:
                    continue
                learned, _ = propose(encoding, slots, model)
                control, _ = propose(encoding, slots, None)
                indices = list(dict.fromkeys((0, learned, control)))
                seeds_by_style = paired_draws(root, actual, review_seeds(decision, 4), 2, draws)
                cache = {}
                plans = [[replay_completion(experiment, decision, actions[i], style, cache)
                          for style in ("pairwise", "forecast")] for i in indices]
                if any(world is None for group in plans for world in group):
                    stats["projection_fallbacks"] += 1
                    continue
                outcomes = []
                for group in plans:
                    trial = []
                    for teams, battles in zip(group, seeds_by_style, strict=True):
                        trial.append([games.run({owner: [experiment.original_cards[i] for i in team]
                                                for owner, team in zip((SEATS[k % 2], SEATS[1 - k % 2]), teams, strict=True)},
                                                SEATS[k % 2], seed, "rollout") for k, seed in enumerate(battles)])
                    outcomes.append(trial)
                rows.append({"seed": root, "pick": pick, "rosters": plans, "outcomes": outcomes,
                             "learned": indices.index(learned), "control": indices.index(control), "paired_draws": seeds_by_style})
                stats.update(positions=1, learned_disagreement=int(learned != control))
            except (KeyError, OSError, RuntimeError, ValueError) as err:
                errors.append({"seed": root, "issue": str(err)})
        stats.update(games.activity)
        stats["preview_inner_games"] = experiment.preview.games
        stats["prefix_rollouts"] = experiment.hybrid.rollout.activity["rollout_games"] + experiment.league.rollout.activity["rollout_games"]
    return rows, errors, stats


def check_pools(pools, extra):
    from .draft_early import check_pools as check_previous

    previous = [p for group in ("benchmark_results", "checkpoints") for version in ("draft-early-v1", "draft-replay-v1")
                for p in (ROOT / group / version).glob("*.json") if "pools" in read(p)]
    check_previous(pools, [*previous, *extra])


def generate(args, cards):
    seeds = range(args.seed, args.seed + args.count)
    pools = pools_for(cards, seeds)
    check_pools(pools, args.exclude)
    samples, errors, stats = [], [], Counter()
    started = perf_counter()
    for rows, issues, events in parallel(collect, seeds, args.workers, cards, args.draws):
        samples.extend(rows)
        errors.extend(issues)
        stats.update(events)
    if not samples:
        raise RuntimeError(f"No valid counterfactual leaves: {errors[:4]}")
    report = {"seconds": perf_counter() - started, "workers": args.workers, "draws": args.draws,
              "events": dict(stats), "errors": errors}
    save(args.output, {"provenance": provenance(cards), "pools": pools,
                       "rows": sorted(samples, key=lambda x: x["seed"]), "report": report})
    print(json.dumps(report), flush=True)


def tensors(rows, device):
    rosters, scores, masks, learned = [], [], [], []
    for row in rows:
        teams, outcomes = row["rosters"], row["outcomes"]
        n = len(teams)
        rosters.append(teams + [teams[0]] * (3 - n))
        scores.append(outcomes + [outcomes[0]] * (3 - n))
        masks.append([1.] * n + [0.] * (3 - n))
        learned.append(row["learned"])
    return tuple(torch.tensor(x, dtype=t, device=device) for x, t in zip(
        (rosters, scores, masks, learned), (torch.long, torch.float32, torch.float32, torch.long), strict=True))


def objective(logits, paired, mask):
    target = paired.mean(-1)
    outcome = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(-1)
    p = logits.sigmoid()
    raw_deltas = paired[:, 1:] - paired[:, :1]
    weights = 1. / (1. + 4. * raw_deltas.var(-1, unbiased=False))
    pair = ((p[:, 1:] - p[:, :1]) - raw_deltas.mean(-1)).square() * weights
    return (outcome * mask).sum() / mask.sum() + 3. * (pair.mean(-1) * mask[:, 1:]).sum() / mask[:, 1:].sum()


def logits(model, rosters):
    return model(rosters.flatten(0, 2)).view(rosters.shape[:3])


@torch.inference_mode()
def assess(model, data):
    rosters, outcomes, mask, proposed = data
    z = logits(model, rosters)
    point = z.sigmoid()
    row = torch.arange(len(proposed), device=proposed.device)
    expected = point[row, proposed] - point[:, 0]
    observed = (outcomes[row, proposed] - outcomes[:, 0]).mean(-1)
    accepted = (expected.mean(1) >= MARGIN) & (expected.amin(1) >= DOWNSIDE)
    return {"objective": objective(z, outcomes, mask).item(), "gate_overrides": int(accepted.sum()),
            "gate_delta": (observed.mean(1) * accepted).mean().item(),
            "proposal_delta_mse": (expected - observed).square().mean().item()}


def train(args, cards):
    src, held = read(args.train), read(args.validation)
    if src["provenance"] != provenance(cards) or held["provenance"] != provenance(cards):
        raise ValueError("Unexpected leaf training data")
    disjoint(src["pools"], held["pools"])
    if any(block["report"]["errors"] for block in (src, held)):
        raise ValueError("Incomplete leaf data")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    fit = tensors(src["rows"], device)
    val = tensors(held["rows"], device)
    model = LeafValue(load_value(PAIRWISE, "cpu")[0], args.width).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.0007, weight_decay=.025)
    rng = torch.Generator().manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best, chosen, state = float("inf"), None, None
    started = perf_counter()
    for epoch in range(args.epochs):
        for indices in torch.randperm(len(fit[0]), generator=rng).to(device).split(32):
            x, y, mask, _ = (field[indices] for field in fit)
            loss = objective(logits(model, x), y, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        metrics = assess(model, val)
        if metrics["proposal_delta_mse"] < best:
            best, chosen = metrics["proposal_delta_mse"], epoch + 1
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    elapsed = perf_counter() - started
    model.load_state_dict(state)
    args.output.mkdir(parents=True)
    torch.save(state, args.output / "state_dict.pt")
    save(args.output / "config.json", {"provenance": provenance(cards), "width": args.width, "seed": args.seed,
                                      "chosen_epoch": chosen, "margin": MARGIN, "downside": DOWNSIDE,
                                      "splits": {"train": src["pools"], "validation": held["pools"]}})
    reloaded = load_leaf(args.output, cards)
    probes = val[0][0].flatten(0, 1).cpu()
    with torch.inference_mode():
        expected = model(probes.to(device)).cpu()
        actual = reloaded(probes)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    save(args.output / "probes.json", {"rosters": probes.tolist(), "logits": actual.tolist()})
    report = {"train_roots": len(fit[0]), "validation_roots": len(val[0]),
              "train_games": src["report"]["events"]["rollout_terminal_games"], "seconds": elapsed,
              "parameters": sum(p.numel() for p in model.parameters()), "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "weights_bytes": (args.output / "state_dict.pt").stat().st_size, "device": str(device),
              "validation": {"calibrated": assess(model, val), "prior": assess(model.prior, val)},
              "cpu_probe_gap": float((actual - expected).abs().max())}
    if device.type == "cuda":
        report.update(gpu=torch.cuda.get_device_name(device), peak_mib=torch.cuda.max_memory_allocated(device) / 2**20)
    save(args.output / "metrics.json", report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    gen, fit = [command.add_parser(name) for name in ("generate", "train")]
    for step in (gen, fit):
        step.add_argument("--output", type=Path, required=True)
        step.add_argument("--seed", type=int, required=True)
    gen.add_argument("--count", type=int, required=True)
    gen.add_argument("--workers", type=int, default=12)
    gen.add_argument("--draws", type=int, default=6)
    gen.add_argument("--exclude", type=Path, nargs="*", default=[])
    fit.add_argument("--train", type=Path, required=True)
    fit.add_argument("--validation", type=Path, required=True)
    fit.add_argument("--device", default="cuda")
    fit.add_argument("--width", type=int, default=48)
    fit.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    if args.output.exists() or (args.command == "generate" and min(args.count, args.workers, args.draws) < 1):
        parser.error("Use a new output and positive collection sizes")
    if args.command == "train" and (args.epochs < 1 or args.width < 4 or args.width % 4):
        parser.error("Use a positive epoch count and width divisible by four")
    check_hybrid()
    {"generate": generate, "train": train}[args.command](args, read(CATALOG))


if __name__ == "__main__":
    main()
