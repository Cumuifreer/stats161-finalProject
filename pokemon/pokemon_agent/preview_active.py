"""Compare uncertainty-directed and uniform labels for the fifth Preview search slot."""

import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
from random import Random
from statistics import mean
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from .combination_preview import (
    FEATURES, RIVALS, ComboLineup, ComboValue, alternative_reply, battle_report,
    check_pools, distinct_options, draft_root, encode_action, encode_rows, encode_team, freeze,
    load_model, preview_snapshot, signature, terminal,
)
from .combination_space import CardSpace
from .draft_counterfactual import check_hybrid
from .draft_coverage import Coverage
from .draft_last_pick import other
from .draft_league import baseline_hash, disjoint, parallel, pools_for, read, save
from .draft_matchups import ROOT
from .game import PREVIEW, ident, make_decision
from .lineup import LineupPolicy, index_cards, lineup_options
from .sim import SEATS, fingerprint
from .tactics import tactical_actions

BASE = ROOT / "checkpoints/combination-v1/uniform"
BASE_HASH = "bb4052014fb93b2ef4b70587f55b3cf5595b21777cd0f177e01425374463fa2e"
HISTORICAL = ROOT / "data/historical_cards.json"
REFERENCE = "frozen"
TEACHER = "frozen-uniform"
DATA_ARMS = ("uniform", "active")
EXTRA_COUNT = 6


def frozen_config(cards):
    check_hybrid()
    if baseline_hash(BASE) != BASE_HASH:
        raise ValueError("Frozen uniform Preview checkpoint changed")
    return {"baseline": freeze(cards), "teacher": BASE_HASH, "schema": 1}


def assert_pools(pools, extra=()):
    older = [file for folder in ("checkpoints", "benchmark_results")
             for file in (ROOT / folder / "combination-v1").glob("*.json") if "pools" in read(file)]
    check_pools(pools, (*older, *extra))


def extended_actions(decision, original):
    used = {signature(action) for action in original}
    options = {signature(action): action for action in distinct_options(decision).values() if signature(action) not in used}
    return list({signature(action): options[signature(action)] for action in tactical_actions(decision)
                 if signature(action) in options}.values())


def supported(decision, cards):
    return all(len(decision.observation[side]) == 6 and all(mon and mon.get("species") in cards for mon in decision.observation[side])
               for side in ("team", "opponent_team"))


def public_replies(decision, cards):
    ours = [cards[mon["species"]] for mon in decision.observation["team"]]
    theirs = [cards[mon["species"]] for mon in decision.observation["opponent_team"]]
    raw = {"your_roster": theirs, "opponent_roster": ours}
    opponent = make_decision(PREVIEW, raw, [{"roster": [ident(card["species"]) for card in theirs]}], "p2")
    return alternative_reply(opponent)


@torch.inference_mode()
def scores_for(model, space, decision, cards, actions):
    teams = [encode_team(space, decision.observation, side) for side in ("team", "opponent_team")]
    replies = [encode_action(space, action) for action in public_replies(decision, cards)]
    own = [encode_action(space, action) for action in actions]
    roles, features = encode_rows(space, teams, own, replies)
    identities = torch.tensor([teams]).expand(len(roles), -1, -1)
    return model(identities, torch.tensor(roles), torch.tensor(features, dtype=torch.float32)).sigmoid().view(-1, 2)


def acquisition(scores, tactical_order):
    """The first score is the incumbent fifth action. Tactical ranks cover all outside options."""
    avg = scores.mean(1)
    width = len(tactical_order)
    order = list(range(width))
    top = sorted(order, key=lambda i: float(avg[i + 1]), reverse=True)
    gap = scores[1:] - scores[0]
    shortlist = set(top[:12]) | set(tactical_order[:12])
    rank_net = {i: place for place, i in enumerate(top)}
    rank_tactical = {i: place for place, i in enumerate(tactical_order)}
    disagreement = sorted(shortlist, key=lambda i: abs(rank_net[i] - rank_tactical[i]), reverse=True)
    uncertain = sorted(shortlist, key=lambda i: (float(gap[i].max() - gap[i].min()) - abs(float(gap[i].mean())), -i), reverse=True)
    proposal = top[0]
    priority = (float(avg[proposal + 1] - avg[0]) + float(avg[proposal + 1] - avg[tactical_order[0] + 1])
                + float(gap[uncertain[0]].max() - gap[uncertain[0]].min()))
    queue = list(dict.fromkeys(i for ranked in zip(top[:12], tactical_order[:12], disagreement[:12], uncertain[:12], strict=False)
                              for i in ranked))
    return proposal, priority, queue + [i for i in top if i not in queue]


def survey_positions(seeds, cards, rivals):
    rows, errors, counts = [], [], Counter()
    with ExitStack() as stack:
        experiment = Coverage(stack, cards)
        space, indexed = CardSpace(cards, experiment.profiles), index_cards(cards)
        teacher = load_model(BASE, cards)
        for seed in seeds:
            actor = SEATS[seed % 2]
            rival = rivals[seed // 2 % len(rivals)]
            try:
                rosters, battle_seed, times = draft_root(experiment, seed, cards, {actor: "champion", other(actor): rival})
                decision = preview_snapshot(rosters, battle_seed, experiment.league.engine)[actor]
                if not supported(decision, indexed):
                    counts["unsupported"] += 1
                    continue
                original = lineup_options(decision)
                outside = extended_actions(decision, original)
                scores = scores_for(teacher, space, decision, indexed, [original[-1], *outside])
                best, priority, order = acquisition(scores, list(range(len(outside))))
                counts.update(positions=1, model_tactical_disagreements=int(best != 0),
                              teacher_proposals=int(scores[best + 1].mean() >= scores[0].mean() + .015),
                              draft_timeouts=sum(time >= 15000 for time in times))
                rows.append({"seed": seed, "actor": actor, "rival": rival, "rosters": {k: [card["card_id"] for card in v] for k, v in rosters.items()},
                             "battle_seed": battle_seed, "original": original, "outside": outside, "priority": priority,
                             "model_tactical_disagree": best != 0, "acquisition": order})
            except (ValueError, RuntimeError, OSError) as error:
                errors.append({"seed": seed, "error": str(error)})
        counts["draft_shell_games"] = experiment.preview.games + experiment.activity["full_leaf_games"]
        counts["draft_rollout_games"] = (experiment.hybrid.rollout.activity["rollout_games"]
                                        + experiment.league.rollout.activity["rollout_games"])
    return rows, errors, counts


def survey(args, cards):
    provenance = frozen_config(cards)
    seeds = range(args.seed, args.seed + args.count)
    pools = pools_for(cards, seeds)
    assert_pools(pools, args.exclude)
    rows, errors, counts = [], [], Counter()
    started = perf_counter()
    for data, issues, events in parallel(survey_positions, seeds, args.workers, cards, args.rivals):
        rows.extend(data)
        errors.extend(issues)
        counts.update(events)
    report = {"root_attempts": args.count, "seconds": perf_counter() - started, "counts": dict(counts), "errors": errors}
    save(args.output, {"provenance": provenance, "report": report, "pools": pools, "rows": sorted(rows, key=lambda r: r["seed"])})
    print(json.dumps(report), flush=True)


def allocate_extras(rows, balanced):
    ranked = sorted(rows, key=lambda row: (row["priority"], row["seed"]))
    quarter = len(ranked) // 4 if balanced else 0
    counts = {}
    for index, row in enumerate(ranked):
        counts[row["seed"]] = EXTRA_COUNT + 2 * (int(index >= len(ranked) - quarter) - int(index < quarter))
    return counts


def extra_actions(row, count, shared):
    rng = Random(row["seed"] ^ 0xEA06)
    uniform = rng.sample(range(len(row["outside"])), EXTRA_COUNT)
    jointly = uniform[:shared]
    ranked = [i for i in row["acquisition"] if i not in jointly]
    return {"uniform": uniform, "active": jointly + ranked[:count - shared]}


def collect_labels(rows, cards, draws, quotas, shared):
    samples, errors, counts = [], [], Counter()
    with ExitStack() as stack:
        experiment = Coverage(stack, cards)
        space, indexed = CardSpace(cards, experiment.profiles), index_cards(cards)
        teacher = load_model(BASE, cards)
        for row in rows:
            seed, actor = row["seed"], row["actor"]
            foe = other(actor)
            try:
                rosters = {key: [experiment.by_id[c] for c in value] for key, value in row["rosters"].items()}
                views = preview_snapshot(rosters, row["battle_seed"], experiment.league.engine)
                decision, enemy = views[actor], views[foe]
                original = [decision.validate(action) for action in row["original"]]
                incumbent = decision.validate(LineupPolicy(experiment.profiles, indexed, experiment.preview)(decision))
                defender = enemy.validate(LineupPolicy(experiment.profiles, indexed, experiment.preview)(enemy))
                alternative = enemy.validate(ComboLineup(experiment.profiles, indexed, experiment.preview, space, teacher)(enemy))
                if signature(defender) == signature(alternative):
                    alternative = next(action for action in tactical_actions(enemy) if signature(action) != signature(defender))
                else:
                    counts["learned_reply_disagreements"] += 1
                draws_for_reply = Random(seed ^ 0xE9A7)
                replies = [defender, alternative]
                rngs = [[tuple(draws_for_reply.randrange(65536) for _ in range(4)) for _ in range(draws)] for _ in replies]
                methods = extra_actions(row, quotas[seed], shared)
                ids = list(dict.fromkeys(methods["uniform"] + methods["active"]))
                actions = original + [decision.validate(row["outside"][i]) for i in ids]
                indices = {i: j + len(original) for j, i in enumerate(ids)}
                outcomes = [[[terminal(experiment, rosters, actor, battle, action, reply) for battle in seeds]
                             for reply, seeds in zip(replies, rngs, strict=True)] for action in actions]
                counts.update(terminal_games=2 * draws * len(actions), positions=1,
                              model_tactical_disagreements=int(row["model_tactical_disagree"]))
                for arm, choices in methods.items():
                    counts[f"{arm}_label_games"] += 2 * draws * (len(original) + len(choices))
                samples.append({"seed": seed, "rival": row["rival"], "priority": row["priority"],
                                "common": len(original), "champion_action": original.index(incumbent),
                                "teams": [encode_team(space, decision.observation, side) for side in ("team", "opponent_team")],
                                "actions": [encode_action(space, action) for action in actions],
                                "replies": [encode_action(space, reply) for reply in replies],
                                "outcomes": outcomes, "arms": {name: [indices[i] for i in use] for name, use in methods.items()}})
            except (ValueError, RuntimeError, OSError) as error:
                errors.append({"seed": seed, "error": str(error)})
        counts["preview_games"] = experiment.preview.games
    return samples, errors, counts


def label(args, cards):
    provenance = frozen_config(cards)
    roots = read(args.survey)
    if roots["provenance"] != provenance or roots["report"]["errors"] or not roots["rows"]:
        raise ValueError("Require a complete matching Preview survey")
    rows = roots["rows"]
    quotas = allocate_extras(rows, not args.validation)
    started = perf_counter()
    data, issues, counts = [], [], Counter()
    for samples, errors, events in parallel(collect_labels, rows, args.workers, cards, args.draws, quotas, 2):
        data.extend(samples)
        issues.extend(errors)
        counts.update(events)
    if issues or len(data) != len(rows) or counts["uniform_label_games"] != counts["active_label_games"]:
        raise RuntimeError(f"Incompletely matched label generation: {issues[:4]}")
    signals = {arm: Counter() for arm in DATA_ARMS}
    for row in data:
        baseline = row["outcomes"][row["champion_action"]]
        fifth = row["outcomes"][row["common"] - 1]
        for arm, indices in row["arms"].items():
            for i in indices:
                branches = row["outcomes"][i]
                edge = [mean(a) - mean(b) for a, b in zip(branches, baseline, strict=True)]
                signals[arm].update(actions=1, above_champion=int(mean(edge) > 0), robust_above_champion=int(min(edge) >= 0 and mean(edge) > 0),
                                    above_fifth=int(mean(mean(a) - mean(b) for a, b in zip(branches, fifth, strict=True)) > 0))
    report = {"seconds": perf_counter() - started, "workers": args.workers, "draws_per_reply": args.draws, "signals": signals,
              "counts": dict(counts), "extra_quota": dict(Counter(quotas.values())), "errors": issues}
    save(args.output, {"provenance": provenance, "report": report, "pools": roots["pools"],
                       "rows": sorted(data, key=lambda row: row["seed"])})
    print(json.dumps(report), flush=True)


class PreviewRanker(nn.Module):
    def __init__(self, teacher, width=64):
        super().__init__()
        self.teacher = teacher.requires_grad_(False)
        self.cards = nn.Embedding(teacher.rosters.cards.num_embeddings, width)
        self.roles = nn.Embedding(3, width)
        self.teammates = nn.MultiheadAttention(width, 4, dropout=.1, batch_first=True)
        self.opponents = nn.MultiheadAttention(width, 4, dropout=.1, batch_first=True)
        self.team_norm = nn.LayerNorm(width)
        self.foe_norm = nn.LayerNorm(width)
        self.score = nn.Sequential(nn.Linear(2 * width + FEATURES, width), nn.SiLU(), nn.Dropout(.1), nn.Linear(width, 1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, identities, roles, features):
        tokens = (self.cards(identities) + self.roles(roles)).flatten(0, 1)
        teams = self.team_norm(tokens + self.teammates(tokens, tokens, tokens, need_weights=False)[0])
        foes = teams.unflatten(0, (-1, 2)).flip(1).flatten(0, 1)
        mixed = self.foe_norm(teams + self.opponents(teams, foes, foes, need_weights=False)[0])
        weights = (roles + .45).unsqueeze(-1)
        lineups = (mixed.unflatten(0, (-1, 2)) * weights).sum(2) / weights.sum(2)
        scores = self.score(torch.cat((lineups, lineups.flip(1), features), dim=-1)).squeeze(-1)
        return self.teacher(identities, roles, features) + scores[:, 0] - scores[:, 1]


def pack(space, rows, arm, device):
    teams, roles, features, outcomes, masks, anchors = [], [], [], [], [], []
    width = max(row["common"] + len(row["arms"][arm]) if arm else len(row["actions"]) for row in rows)
    for row in rows:
        used = list(range(row["common"])) + row["arms"][arm] if arm else list(range(len(row["actions"])))
        masks.append([1.] * len(used) + [0.] * (width - len(used)))
        used.extend([0] * (width - len(used)))
        actions = [row["actions"][i] for i in used]
        role, feature = encode_rows(space, row["teams"], actions, row["replies"])
        teams.append([row["teams"]] * len(role))
        roles.append(role)
        features.append(feature)
        outcomes.append([row["outcomes"][i] for i in used])
        anchors.append([row["champion_action"], row["common"] - 1])
    return tuple(torch.tensor(data, dtype=dtype, device=device) for data, dtype in zip(
        (teams, roles, features, outcomes, masks, anchors),
        (torch.long, torch.long, torch.float32, torch.float32, torch.float32, torch.long), strict=True))


def objective(logits, outcomes, weights, anchors):
    score = outcomes.mean(-1)
    loss = F.binary_cross_entropy_with_logits(logits, score, reduction="none")
    ref_logits = logits.gather(1, anchors[..., None].expand(-1, -1, 2))
    ref_outcomes = outcomes.gather(1, anchors[..., None, None].expand(-1, -1, 2, outcomes.size(-1)))
    paired = outcomes[:, :, None] - ref_outcomes[:, None]
    wins = (paired > 0).float().mean(-1)
    losses = (paired < 0).float().mean(-1)
    logit_diff = logits[:, :, None] - ref_logits[:, None]
    rank = (wins * F.softplus(-logit_diff) + losses * F.softplus(logit_diff)).mean(2)
    return ((loss + 1.5 * rank) * weights[..., None]).sum() / (2 * weights.sum())


def fit_logits(net, data, indices):
    values = net(*(x[indices].flatten(0, 1) for x in data[:3]))
    return values.reshape(len(indices), -1, 2)


@torch.inference_mode()
def validation_metrics(net, data, rows):
    logits = torch.cat([fit_logits(net, data, batch) for batch in torch.arange(len(rows), device=data[0].device).split(24)])
    score = data[3].mean((-1, -2))
    prediction = logits.sigmoid().mean(-1)
    mask = data[4].bool()
    outsiders = (torch.arange(score.size(1), device=score.device)[None] > data[5][:, 1:2]) & mask
    proposal = prediction.masked_fill(~outsiders, -1.).argmax(1, keepdim=True)
    reference = score.gather(1, data[5])
    reward = score.gather(1, proposal)
    return {"objective": objective(logits, *data[3:]).item(),
            "candidate_minus_fifth": (reward - reference[:, 1:]).mean().item(),
            "candidate_minus_champion": (reward - reference[:, :1]).mean().item()}


def fit(args, cards):
    provenance = frozen_config(cards)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    training, validation = map(read, (args.train, args.validation))
    if any(data["provenance"] != provenance for data in (training, validation)):
        raise ValueError("Training data differs from frozen Preview source")
    disjoint(training["pools"], validation["pools"])
    space, device = CardSpace(cards), torch.device(args.device)
    samples, checks = training["rows"], validation["rows"]
    train, held = pack(space, samples, args.arm, device), pack(space, checks, None, device)
    net = PreviewRanker(load_model(BASE, cards), args.width).to(device)
    optimizer = torch.optim.AdamW((p for p in net.parameters() if p.requires_grad), lr=.0006, weight_decay=.02)
    rng = torch.Generator().manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started, best, state = perf_counter(), float("inf"), None
    for epoch in range(args.epochs):
        net.train()
        for indices in torch.randperm(len(samples), generator=rng).to(device).split(24):
            loss = objective(fit_logits(net, train, indices), *(x[indices] for x in train[3:]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        net.eval()
        report = validation_metrics(net, held, checks)
        if report["objective"] < best:
            best, selected = report["objective"], epoch + 1
            state = {name: value.detach().cpu().clone() for name, value in net.state_dict().items()}
    elapsed = perf_counter() - started
    net.load_state_dict(state)
    args.output.mkdir(parents=True)
    torch.save(state, args.output / "state_dict.pt")
    config = {"provenance": provenance, "width": args.width, "arm": args.arm, "seed": args.seed, "epochs": args.epochs,
              "selected_epoch": selected, "splits": {"train": training["pools"], "validation": validation["pools"]}}
    save(args.output / "config.json", config)
    restored = load_ranker(args.output, cards)
    with torch.inference_mode():
        probe = [value[0, :12].cpu() for value in held[:3]]
        predicted = net(*(x.to(device) for x in probe)).cpu()
        reread = restored(*probe)
        torch.testing.assert_close(predicted, reread, rtol=1e-4, atol=2e-5)
    save(args.output / "probes.json", {"input": [x.tolist() for x in probe], "logits": reread.tolist()})
    report = {"seconds": elapsed, "params": sum(p.numel() for p in net.parameters()),
              "trainable": sum(p.numel() for p in net.parameters() if p.requires_grad),
              "weights_bytes": (args.output / "state_dict.pt").stat().st_size, "train_pools": len(samples),
              "val_pools": len(checks), "train_label_games": training["report"]["counts"][f"{args.arm}_label_games"],
              "validation": validation_metrics(net, held, checks), "device": str(device), "torch": torch.__version__,
              "reload_delta": float((predicted - reread).abs().max())}
    if device.type == "cuda":
        report.update(gpu=torch.cuda.get_device_name(device), peak_mib=torch.cuda.max_memory_allocated(device) / 2**20)
    save(args.output / "metrics.json", report)
    print(json.dumps(report), flush=True)


def load_ranker(path, cards):
    spec = read(path / "config.json")
    if spec["provenance"] != frozen_config(cards):
        raise ValueError("Incompatible Preview ranker")
    net = PreviewRanker(ComboValue(len(cards), read(BASE / "config.json")["width"]), spec["width"])
    net.load_state_dict(torch.load(path / "state_dict.pt", map_location="cpu", weights_only=True))
    return net.eval()


class MeasuredUniform(ComboLineup):
    def options(self, decision):
        before = self.proposed
        candidates = super().options(decision)
        self.inserted = candidates[-1] if self.proposed != before else None
        return candidates


class FifthCandidate(LineupPolicy):
    def __init__(self, profiles, cards, worker, space, model=None):
        super().__init__(profiles, cards, worker)
        self.space, self.model = space, model
        self.inserted = None

    def options(self, decision):
        self.inserted = None
        original = super().options(decision)
        if len(original) != 5 or not supported(decision, self.cards):
            return original
        outside = extended_actions(decision, original)
        if not outside:
            return original
        best = 0
        if self.model is not None:
            scores = scores_for(self.model, self.space, decision, self.cards, [original[-1], *outside]).mean(1)
            best = int(scores[1:].argmax())
            if scores[best + 1] < scores[0] + .015:
                return original
        self.inserted = outside[best]
        return [*original[:-1], self.inserted]


def independent_seeds(decision, seed, actor, actual, count):
    names = [sorted(mon.get("species", "") for mon in decision.observation[side] if mon)
             for side in ("team", "opponent_team")]
    seen = {actual}
    for order in (names, names[::-1]):
        review = Random(fingerprint(order))
        seen.update(tuple(review.randrange(65536) for _ in range(4)) for _ in range(2))
    audit = Random((seed << 3) + SEATS.index(actor) + 0x6C11)
    held = []
    while len(held) < count:
        candidate = tuple(audit.randrange(65536) for _ in range(4))
        if candidate not in seen:
            held.append(candidate)
            seen.add(candidate)
    return held


def evaluate(seeds, cards, rivals, paths, baselines, replays):
    pairs, errors, events = {}, [], Counter()
    with ExitStack() as stack:
        experiment = Coverage(stack, cards)
        profiles, worker = experiment.profiles, experiment.preview
        space, indexed = CardSpace(cards, profiles), index_cards(cards)
        teacher = load_model(BASE, cards)
        models = {name: load_ranker(path, cards) for name, path in paths.items()}
        for seed in seeds:
            rival = rivals[seed % len(rivals)]
            pair = []
            for actor in SEATS:
                foe = other(actor)
                try:
                    rosters, battle_seed, durations = draft_root(experiment, seed, cards, {actor: "champion", foe: rival})
                    views = preview_snapshot(rosters, battle_seed, experiment.league.engine)
                    decision, enemy = views[actor], views[foe]
                    if rival == "search":
                        defender = ComboLineup(profiles, indexed, worker, space, teacher)
                    else:
                        defender = LineupPolicy(profiles, indexed, worker)
                    reply = enemy.validate(defender(enemy))
                    events["opponent_preview_fallbacks"] += defender.fallbacks
                    policies = {REFERENCE: LineupPolicy(profiles, indexed, worker)}
                    if TEACHER in baselines:
                        policies[TEACHER] = MeasuredUniform(profiles, indexed, worker, space, teacher)
                    if "heuristic" in baselines:
                        policies["heuristic"] = FifthCandidate(profiles, indexed, worker, space)
                    policies.update({name: FifthCandidate(profiles, indexed, worker, space, model) for name, model in models.items()})
                    moves, results, cache = {}, {}, {}
                    original_fifth = lineup_options(decision)[-1]
                    for name, policy in policies.items():
                        started, before = perf_counter(), worker.games
                        action = decision.validate(policy(decision))
                        elapsed, starts = (perf_counter() - started) * 1000., worker.games - before
                        proposed = getattr(policy, "inserted", None)
                        if proposed is not None:
                            decision.validate(proposed)
                        moves[name] = action
                        results[name] = {"elapsed_ms": elapsed, "action": encode_action(space, action), "review_starts": starts,
                                         "fallbacks": policy.fallbacks, "proposed": int(proposed is not None),
                                         "inserted": encode_action(space, proposed) if proposed is not None else None,
                                         "used_proposal": proposed is not None and proposed == action,
                                         "displaced_winner": proposed is not None and moves[REFERENCE] == original_fifth and action != moves[REFERENCE]}
                    review_seeds = independent_seeds(decision, seed, actor, battle_seed, replays)

                    def score(action, battle):
                        key = (fingerprint(action), battle)
                        if key not in cache:
                            cache[key] = terminal(experiment, rosters, actor, battle, action, reply)
                            events["benchmark_terminal_games" if battle == battle_seed else "replication_games"] += 1
                        return cache[key]

                    for name, action in moves.items():
                        entry = results[name]
                        entry["score"] = score(action, battle_seed)
                        entry["replay_deltas"] = [score(action, battle) - score(moves[REFERENCE], battle) for battle in review_seeds
                                                  if action != moves[REFERENCE]]
                    pair.append({"rival": rival, "draft_ms": durations, "results": results})
                except (ValueError, RuntimeError, OSError) as error:
                    errors.append({"seed": seed, "seat": actor, "error": str(error)})
            if len(pair) == 2:
                pairs[seed] = pair
        events["preview_and_draft_inner_games"] = worker.games
        events["draft_final_review_games"] = experiment.activity["full_leaf_games"]
        events["draft_rollout_games"] = (experiment.hybrid.rollout.activity["rollout_games"]
                                         + experiment.league.rollout.activity["rollout_games"])
    return pairs, errors, events


def replay_summary(groups):
    groups = [(sum(group), len(group)) for group in groups if group]
    num = sum(n for _, n in groups)
    if not num:
        return {"positions": 0, "delta": None, "ci95": None}
    rng = Random(0)
    draws = []
    for _ in range(4000):
        draw = rng.choices(groups, k=len(groups))
        draws.append(sum(v for v, _ in draw) / sum(n for _, n in draw))
    draws.sort()
    return {"positions": num, "delta": sum(value for value, _ in groups) / num,
            "ci95": [draws[100], draws[3900]] if len(groups) > 1 else None}


def measure(pairs, method):
    report = battle_report(pairs, method, REFERENCE)
    entries = [row["results"][method] for pair in pairs for row in pair]
    report.update(used_proposal=sum(row["used_proposal"] for row in entries), displaced_winner=sum(row["displaced_winner"] for row in entries))
    for name, used in (("all_changed_replays", False), ("selected_proposal_replays", True)):
        report[name] = replay_summary([[mean(row["results"][method]["replay_deltas"]) for row in pair
                                       if row["results"][method]["replay_deltas"] and (not used or row["results"][method]["used_proposal"])]
                                      for pair in pairs])
    return report


def benchmark(args, cards):
    provenance = frozen_config(cards)
    seeds = range(args.seed, args.seed + args.count)
    pools = pools_for(cards, seeds)
    assert_pools(pools, args.exclude)
    models = {path.name: path for path in args.models}
    if len(models) != len(args.models) or set(models) & {REFERENCE, TEACHER, "heuristic"}:
        raise ValueError("Use distinct paths for the new models")
    for path in models.values():
        for split in read(path / "config.json")["splits"].values():
            disjoint(pools, split)
    started = perf_counter()
    pairs, errors, counts = {}, [], Counter()
    for games, failures, totals in parallel(evaluate, seeds, args.workers, cards, args.rivals, models, args.baselines, args.replays):
        pairs.update(games)
        errors.extend(failures)
        counts.update(totals)
    if not pairs:
        raise RuntimeError(f"No valid pairs: {errors[:3]}")
    data = list(pairs.values())
    methods = (REFERENCE, *args.baselines, *models)
    comparison = {f"{name} vs {opponent}": battle_report(data, name, opponent)
                  for name in models for opponent in (*args.baselines, *models) if name != opponent}
    report = {"pairs": len(data), "seconds": perf_counter() - started, "workers": args.workers, "replay_seeds": args.replays,
              "results": {name: measure(data, name) for name in methods}, "comparisons": comparison,
              "opponents": {rival: {name: measure([pair for pair in data if pair[0]["rival"] == rival], name) for name in methods}
                            for rival in {pair[0]["rival"] for pair in data}},
              "draft_timeouts": sum(t >= 15000 for pair in data for row in pair for t in row["draft_ms"]),
              "counts": dict(counts), "errors": errors, "models": {name: baseline_hash(path) for name, path in models.items()}}
    save(args.output, {"provenance": provenance, "report": report, "pools": pools, "paired_games": dict(sorted(pairs.items()))})
    print(json.dumps({k: v for k, v in report.items() if k not in {"opponents", "comparisons"}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    survey_args, bench = [commands.add_parser(command) for command in ("survey", "bench")]
    for cmd in (survey_args, bench):
        cmd.add_argument("--seed", type=int, required=True)
        cmd.add_argument("--count", type=int, required=True)
        cmd.add_argument("--workers", type=int, default=12)
        cmd.add_argument("--rivals", nargs="+", choices=RIVALS, default=RIVALS)
        cmd.add_argument("--exclude", nargs="*", type=Path, default=[])
    bench.add_argument("--models", nargs="*", type=Path, default=[])
    bench.add_argument("--baselines", nargs="*", choices=(TEACHER, "heuristic"), default=(TEACHER, "heuristic"))
    bench.add_argument("--replays", type=int, default=6)
    label_args = commands.add_parser("label")
    label_args.add_argument("--survey", type=Path, required=True)
    label_args.add_argument("--draws", type=int, default=3)
    label_args.add_argument("--workers", type=int, default=12)
    label_args.add_argument("--validation", action="store_true")
    trainer = commands.add_parser("train")
    trainer.add_argument("--train", type=Path, required=True)
    trainer.add_argument("--validation", type=Path, required=True)
    trainer.add_argument("--arm", choices=DATA_ARMS, required=True)
    trainer.add_argument("--device", default="cuda")
    trainer.add_argument("--width", type=int, default=64)
    trainer.add_argument("--seed", type=int, default=881)
    trainer.add_argument("--epochs", type=int, default=32)
    for cmd in (survey_args, label_args, trainer, bench):
        cmd.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output path")
    {"survey": survey, "label": label, "train": fit, "bench": benchmark}[args.command](args, read(HISTORICAL))


if __name__ == "__main__":
    main()
