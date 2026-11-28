"""Test whether coverage-directed lineup labels improve equal-budget Preview search."""

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from itertools import combinations
import json
from pathlib import Path
from random import Random
from statistics import mean
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .__main__ import HISTORICAL_CARDS, _percentile
from .combination_space import CardSpace, CoverageCounts, audit
from .draft_counterfactual import check_hybrid
from .draft_coverage import CHAMPION, Coverage
from .draft_last_pick import mean_ci, older_splits, other, sample_game
from .draft_league import baseline_hash, disjoint, parallel, pools_for, read, save
from .game import PREVIEW, ident, make_decision
from .joint import JointPolicy
from .lineup import LineupPolicy, index_cards, lineup_options
from .policy import heuristic
from .roster import RosterNet
from .sim import FORMAT, SEATS, _draft, fingerprint, play_battle
from .tactics import tactical_actions, tactical_candidates

ROOT = HISTORICAL_CARDS.parents[1]
RIVALS = ("champion", "search", "pairwise")
FEATURES = 12
SCHEMA = 1


def freeze(cards):
    return {"catalog": fingerprint(cards), "draft": CHAMPION, "schema": SCHEMA}


def signature(action):
    return frozenset(action["bring"]), frozenset(action["leads"])


def preview_snapshot(rosters, seed, engine):
    start = engine.call("start", format=FORMAT, seed=seed, rosters=rosters)
    if start["ended"] or len(start["views"]) != 2 or any(view["phase"] != PREVIEW for view in start["views"]):
        raise ValueError("Preview labels require two legal public decisions")
    return {v["seat"]: make_decision(PREVIEW, v["raw"], v["templates"], v["seat"]) for v in start["views"]}


def encode_action(space, action):
    return [[space.species[ident(species)] for species in action[key]] for key in ("bring", "leads")]


def encode_team(space, observation, side):
    return [space.species[entry["species"]] for entry in observation[side]]


def distinct_options(decision):
    actions = {}
    for action in decision.actions:
        actions.setdefault(signature(action), action)
    return actions


def team_features(space, own, foe):
    squad, leads = own
    opposing, opposing_leads = foe
    matrix = space.attack[np.ix_(squad, opposing)].clip(0, 1.5)
    opening = space.attack[np.ix_(leads, opposing_leads)].clip(0, 1.5)
    back = sorted(set(squad) - set(leads))
    incoming = space.attack[np.ix_(opposing_leads, back)].clip(0, 1.5)
    speeds = space.speeds[leads][:, None]
    enemy_speed = space.speeds[opposing_leads][None, :]
    return [matrix.max(0).mean(), matrix.max(0).mean() - matrix.mean(1).max(), matrix.mean(),
            opening.max(0).mean(), opening.min(0).mean(), incoming.min(1).mean(),
            (speeds > enemy_speed).mean(), ((speeds < enemy_speed) & (2 * speeds >= enemy_speed)).mean(),
            *(float(bool(set(leads) & space.roles[key])) for key in ("tailwind", "trick_room", "redirection", "fake_out"))]


def encode_rows(space, teams, actions, replies):
    roles, features = [], []
    for action in actions:
        for reply in replies:
            roles.append([[2 if i in leads else 1 if i in bring else 0 for i in team]
                          for team, (bring, leads) in zip(teams, (action, reply), strict=True)])
            features.append([team_features(space, action, reply), team_features(space, reply, action)])
    return roles, features


class ComboValue(nn.Module):
    def __init__(self, card_count, width):
        super().__init__()
        self.rosters = RosterNet(card_count, width)
        self.matchup = nn.Sequential(nn.Linear(FEATURES, width), nn.SiLU(), nn.Linear(width, 1))

    def forward(self, teams, roles, features):
        edge = self.matchup(features).squeeze(-1)
        return self.rosters(teams, roles) + edge[:, 0] - edge[:, 1]


def choose_stratified(space, options, history, rng, count):
    features = [encode_action(space, action) for action in options]
    ordered = []
    remaining = list(range(len(options)))
    for _ in range(count):
        weights = []
        for i in remaining:
            team, leads = features[i]
            lead = tuple(sorted(leads))
            triple = min(history["triples"][group] for group in combinations(sorted(team), 3))
            # Prioritize untested lead pairs and thinly sampled triples without discarding common cores.
            score = .2 + 1.5 / (1 + history["leads"][lead]) ** .5 + 1.5 / (1 + triple) ** .5
            if any(set(team) == set(features[n][0]) for n in ordered):
                score *= .4
            weights.append(score)
        index = rng.choices(remaining, weights=weights, k=1)[0]
        ordered.append(index)
        remaining.remove(index)
    return [options[i] for i in ordered]


def focused_pool(cards, seed, core):
    pool, first, battle_seed = sample_game(cards, seed)
    required = [cards[i] for i in core]
    selected = [card for card in pool if card not in required]
    pool = required + selected[:len(pool) - len(required)]
    Random(seed ^ 0x7A11).shuffle(pool)
    return pool, first, battle_seed


def draft_root(experiment, seed, cards, styles, focus=None):
    pool, first, battle_seed = focused_pool(cards, seed, focus[1]) if focus else sample_game(cards, seed)
    pending = [cards[i]["card_id"] for i in focus[1]] if focus else []
    times = []

    def choose(seat, decision):
        started = perf_counter()
        forced = None
        if pending and seat == focus[0]:
            available = {action["card_id"]: action for action in decision.actions}
            forced = next((available[card] for card in pending if card in available), None)
        action = decision.validate(forced if forced is not None else experiment.choose(decision, styles[seat]))
        if forced is not None:
            pending.remove(action["card_id"])
        times.append((perf_counter() - started) * 1000)
        return action

    rosters = _draft(pool, first, choose)
    return rosters, battle_seed, times


def terminal(experiment, rosters, actor, seed, own, foe):
    policies = {seat: JointPolicy(experiment.profiles) for seat in SEATS}

    def choose(seat, decision, _):
        policy = policies[seat]
        if decision.phase == PREVIEW:
            policy.observe(decision)
            return decision.validate(own if seat == actor else foe)
        return decision.validate(policy(decision))

    winner = play_battle(experiment.league.engine, rosters, policies, 0, seed, choose)
    return .5 if winner is None else float(winner == actor)


def alternative_reply(decision):
    ranked = tactical_candidates(decision)
    preferred = ranked[0]
    alternate = heuristic(decision)
    if signature(alternate) == signature(preferred):
        alternate = next((action for action in ranked if signature(action) != signature(preferred)), preferred)
    return preferred, alternate


def collect(seeds, cards, history_path, draws, rivals, extras):
    history = {name: Counter({tuple(group): n for group, n in counts})
               for name, counts in read(history_path)["active_counts"].items()}
    rows, errors, counts = [], [], Counter()
    with ExitStack() as stack:
        experiment = Coverage(stack, cards)
        space = CardSpace(cards, experiment.profiles)
        fitted_cards = index_cards(cards)
        for root in seeds:
            actor = SEATS[root % 2]
            rival = rivals[(root // 2) % len(rivals)]
            foe_seat = other(actor)
            try:
                rosters, battle_seed, times = draft_root(experiment, root, cards, {actor: "champion", foe_seat: rival})
                views = preview_snapshot(rosters, battle_seed, experiment.league.engine)
                decision, foe = views[actor], views[foe_seat]
                own_policy = LineupPolicy(experiment.profiles, fitted_cards, experiment.preview)
                opp_policy = LineupPolicy(experiment.profiles, fitted_cards, experiment.preview)
                anchor = decision.validate(own_policy(decision))
                opponent = foe.validate(opp_policy(foe))
                counts["preview_fallbacks"] += own_policy.fallbacks + opp_policy.fallbacks
                counts["draft_timeouts"] += sum(t >= 15000 for t in times)
                options = distinct_options(decision)
                common = {signature(action): action for action in (anchor, *lineup_options(decision))}
                core = list(common.values())
                outside = [action for key, action in options.items() if key not in common]
                rng = Random(root ^ 0x481D)
                arms = {"uniform": rng.sample(outside, extras),
                        "stratified": choose_stratified(space, outside, history, rng, extras)}
                trials = list({signature(action): action for action in (*core, *arms["uniform"], *arms["stratified"])}.values())
                positions = {signature(action): i for i, action in enumerate(trials)}
                # Include the frozen opponent's real lineup in labels and one separate plausible reply.
                alternate = next((action for action in alternative_reply(foe) if signature(action) != signature(opponent)), opponent)
                replies = (opponent, alternate)
                draw_rng = Random(root ^ 0x328F)
                seeds_for_reply = [[tuple(draw_rng.randrange(65536) for _ in range(4)) for _ in range(draws)] for _ in replies]
                outcomes = [[[terminal(experiment, rosters, actor, seed, action, reply) for seed in seeds]
                             for reply, seeds in zip(replies, seeds_for_reply, strict=True)] for action in trials]
                counts["terminal_games"] += len(trials) * len(replies) * draws
                counts["common_actions"] += len(core)
                rows.append({"seed": root, "seat": actor, "rival": rival,
                             "teams": [encode_team(space, decision.observation, side) for side in ("team", "opponent_team")],
                             "actions": [encode_action(space, action) for action in trials],
                             "replies": [encode_action(space, reply) for reply in replies], "outcomes": outcomes,
                             "common": len(core), "arms": {key: [positions[signature(action)] for action in actions]
                                                            for key, actions in arms.items()}})
            except (RuntimeError, ValueError, OSError) as error:
                errors.append({"seed": root, "error": str(error)})
        counts["preview_inner_games"] = experiment.preview.games
        counts["draft_inner_games"] = (experiment.hybrid.rollout.activity["rollout_games"]
                                      + experiment.league.rollout.activity["rollout_games"])
        counts["draft_final_review_battles"] = experiment.activity["full_leaf_games"]
        counts["preview_snapshots"] = len(rows)
    return rows, errors, counts


def prior_files():
    for folder in ("checkpoints", "benchmark_results"):
        for directory in (ROOT / folder).glob("draft-v*"):
            yield from directory.glob("*.json")


def check_pools(pools, excludes):
    for split in older_splits():
        disjoint(pools, split)
    for path in (*prior_files(), *excludes):
        disjoint(pools, read(path).get("pools", {}))


def select_pools(args, cards):
    seeds = range(args.seed, args.seed + args.count)
    pools = pools_for(cards, seeds)
    check_pools(pools, args.exclude)
    return seeds, pools


def generate(args, cards):
    check_hybrid()
    seeds, pools = select_pools(args, cards)
    started = perf_counter()
    rows, errors, counts = [], [], Counter()
    for data, issues, events in parallel(collect, seeds, args.workers, cards, args.coverage, args.draws, args.rivals, args.extras):
        rows.extend(data)
        errors.extend(issues)
        counts.update(events)
    if not rows:
        raise RuntimeError(f"No valid lineup data: {errors[:3]}")
    space = CardSpace(cards)
    counters = {arm: CoverageCounts(space) for arm in ("uniform", "stratified")}
    for row in rows:
        for arm, counts_for_arm in counters.items():
            for i in row["arms"][arm]:
                counts_for_arm.add(*row["actions"][i])
    coverage = read(args.coverage)
    maximum = {i: coverage["active"]["groups"][str(i)]["possible"] for i in range(1, 5)}
    report = {"positions": len(rows), "seconds": perf_counter() - started, "workers": args.workers,
              "draws_per_reply": args.draws, "extra_actions": args.extras, "counts": dict(counts), "errors": errors,
              "arm_coverage": {arm: counter.report(maximum) for arm, counter in counters.items()}}
    save(args.output, {"provenance": freeze(cards), "report": report, "pools": pools, "rows": sorted(rows, key=lambda x: x["seed"])})
    print(json.dumps(report), flush=True)


def core_outcomes(rows):
    observations = defaultdict(list)
    for row in rows:
        positions = defaultdict(list)
        anchor_team, anchor_leads = row["actions"][0]
        anchor_triples = set(combinations(sorted(anchor_team), 3))
        anchor = mean(row["outcomes"][0][0])
        for i in set(row["arms"]["stratified"]) | set(row["arms"]["uniform"]):
            team, leads = row["actions"][i]
            delta = mean(row["outcomes"][i][0]) - anchor
            if set(leads) != set(anchor_leads):
                positions[("lead", tuple(sorted(leads)))].append(delta)
            for core in set(combinations(sorted(team), 3)) - anchor_triples:
                positions[("triple", core)].append(delta)
        for key, deltas in positions.items():
            observations[key].append(mean(deltas))
    return observations


def describe_data(args, cards):
    train, validation = map(read, (args.train, args.validation))
    disjoint(train["pools"], validation["pools"])
    known = {name: Counter({tuple(group): n for group, n in counts})
             for name, counts in read(args.coverage)["active_counts"].items()}
    novelty = {}
    for arm in ("uniform", "stratified"):
        leads, triples = set(), set()
        for row in train["rows"]:
            for i in row["arms"][arm]:
                team, opening = row["actions"][i]
                leads.add(tuple(sorted(opening)))
                triples.update(combinations(sorted(team), 3))
        novelty[arm] = {"previously_unseen_leads": sum(not known["leads"][group] for group in leads),
                        "previously_unseen_triples": sum(not known["triples"][group] for group in triples),
                        "triples_with_under_ten_previous_examples": sum(known["triples"][group] < 10 for group in triples)}
    part, holdout = (core_outcomes(data["rows"]) for data in (train, validation))
    supported = [key for key, scores in part.items() if len(scores) >= 10 and len(holdout[key]) >= 5]
    ranked = sorted(supported, key=lambda key: sum(part[key]) / (len(part[key]) + 8), reverse=True)
    hypotheses = {kind: [{"card_ids": [cards[i]["card_id"] for i in key[1]], "fit_positions": len(part[key]),
                         "fit_delta": mean(part[key]), "validation_positions": len(holdout[key]),
                         "validation_delta": mean(holdout[key]), "validation_ci95": mean_ci(holdout[key])}
                        for key in [key for key in ranked if key[0] == kind][:6]] for kind in ("lead", "triple")}
    report = {"coverage_novelty": novelty, "groups_supported_in_both_splits": len(supported), "candidate_cores": hypotheses}
    save(args.output, report)
    print(json.dumps(report), flush=True)


def focused_labels(seeds, cards, groups, rivals, draws):
    rows, events, errors = [], Counter(), []
    with ExitStack() as stack:
        experiment = Coverage(stack, cards)
        species = index_cards(cards)
        for seed in seeds:
            name, core = groups[seed % len(groups)]
            required = frozenset(ident(cards[i]["species"]) for i in core)
            actor = SEATS[seed // len(groups) % 2]
            foe = other(actor)
            rival = rivals[seed // (2 * len(groups)) % len(rivals)]
            try:
                rosters, battle_seed, _ = draft_root(experiment, seed, cards, {actor: "champion", foe: rival}, (actor, core))
                if not required <= {ident(card["species"]) for card in rosters[actor]}:
                    events[f"{name}_draft_denied"] += 1
                    continue
                views = preview_snapshot(rosters, battle_seed, experiment.league.engine)
                own_policy = LineupPolicy(experiment.profiles, species, experiment.preview)
                foe_policy = LineupPolicy(experiment.profiles, species, experiment.preview)
                anchor = own_policy(views[actor])
                reply = foe_policy(views[foe])
                key = "leads" if len(core) == 2 else "bring"
                candidate = next(action for action in tactical_actions(views[actor])
                                 if required <= set(map(ident, action[key])))
                if required <= set(map(ident, anchor[key])):
                    events[f"{name}_already_used"] += 1
                    continue
                rng = Random(seed ^ 0x2562)
                pairs = []
                for _ in range(draws):
                    battle = tuple(rng.randrange(65536) for _ in range(4))
                    pairs.append([terminal(experiment, rosters, actor, battle, action, reply) for action in (anchor, candidate)])
                rows.append({"seed": seed, "group": name, "rival": rival, "seat": actor, "assigned": candidate,
                             "reference": anchor, "paired_scores": pairs})
            except (ValueError, RuntimeError, OSError) as error:
                errors.append({"seed": seed, "error": str(error)})
        events["preview_inner_games"] = experiment.preview.games
        events["draft_inner_games"] = (experiment.hybrid.rollout.activity["rollout_games"]
                                      + experiment.league.rollout.activity["rollout_games"])
        events["draft_final_review_battles"] = experiment.activity["full_leaf_games"]
    return rows, events, errors


def focus_audit(args, cards):
    check_hybrid()
    coverage = read(args.coverage)
    options = coverage["combinations"]
    history = {kind: {tuple(group) for group, _ in rows} for kind, rows in coverage["active_counts"].items()}
    groups = [("complementary_leads", max(options["pairs"], key=lambda row: row[2])[0]),
              ("unseen_leads", max((row for row in options["pairs"] if tuple(row[0]) not in history["leads"]), key=lambda row: row[1])[0]),
              ("unseen_triple", max((row for row in options["triples"] if tuple(row[0]) not in history["triples"]), key=lambda row: row[1])[0])]
    seeds = range(args.seed, args.seed + args.count)
    pools = {str(seed): fingerprint(sorted(card["card_id"] for card in focused_pool(cards, seed, groups[seed % len(groups)][1])[0]))
             for seed in seeds}
    if len(set(pools.values())) != len(pools):
        raise ValueError("Repeated focused card pool")
    check_pools(pools, args.exclude)
    started = perf_counter()
    rows, events, errors = [], Counter(), []
    for data, counts, issues in parallel(focused_labels, seeds, args.workers, cards, groups, args.rivals, args.draws):
        rows.extend(data)
        events.update(counts)
        errors.extend(issues)
    results = {}
    for name, group in groups:
        sample = [row for row in rows if row["group"] == name]
        deltas = [mean(b - a for a, b in row["paired_scores"]) for row in sample]
        results[name] = {"card_ids": [cards[i]["card_id"] for i in group], "positions": len(deltas),
                         "delta": mean(deltas) if deltas else None, "ci95": mean_ci(deltas),
                         "against": dict(Counter(row["rival"] for row in sample))}
    report = {"seconds": perf_counter() - started, "root_attempts": args.count, "paired_draws": args.draws,
              "terminal_games": 2 * args.draws * len(rows), "results": results, "events": dict(events), "errors": errors}
    save(args.output, {"provenance": freeze(cards), "report": report, "pools": pools, "rows": sorted(rows, key=lambda x: x["seed"])})
    print(json.dumps(report), flush=True)


def pack(space, rows, arm, device):
    teams, roles, features, outcomes, weights = [], [], [], [], []
    for row in rows:
        count = row["common"]
        # The common block occupies the front. Pad it to comparable shape with zero-weight copies.
        if arm:
            fixed_common = 5
            indices = list(range(count)) + [0] * (fixed_common - count) + row["arms"][arm]
            weight = [1] * count + [0] * (fixed_common - count) + [1] * len(row["arms"][arm])
        else:
            indices = list(range(len(row["actions"])))
            weight = [1] * len(indices)
        weights.extend(weight)
        outcomes.extend([row["outcomes"][i] for i in indices])
        actions = [row["actions"][i] for i in indices]
        role, feature = encode_rows(space, row["teams"], actions, row["replies"])
        teams.extend([row["teams"]] * len(role))
        roles.extend(role)
        features.extend(feature)
    return tuple(torch.as_tensor(x, dtype=dtype, device=device) for x, dtype in zip(
        (teams, roles, features, outcomes, weights), (torch.long, torch.long, torch.float32, torch.float32, torch.float32), strict=True))


def network_logits(net, data):
    teams, roles, features = data[:3]
    return torch.cat([net(t, r, f) for t, r, f in zip(teams.split(2048), roles.split(2048), features.split(2048), strict=True)]).reshape(-1, 2)


@torch.inference_mode()
def assess(net, data, rows):
    logits = network_logits(net, data)
    targets = data[3].mean(-1)
    values = logits.sigmoid().mean(-1)
    predictions = []
    start = 0
    for row in rows:
        end = start + len(row["actions"])
        predictions.append((start, start + int(values[start:end].argmax())))
        start = end
    selected = mean(float(targets[i].mean()) for _, i in predictions)
    anchor = mean(float(targets[i].mean()) for i, _ in predictions)
    return {"loss": F.binary_cross_entropy_with_logits(logits, targets).item(), "brier": (logits.sigmoid() - targets).square().mean().item(),
            "selected_reward": selected, "anchor_reward": anchor, "selected_delta": selected - anchor}


def fit(args, cards):
    if args.output.exists():
        raise ValueError("Use a fresh model directory")
    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    source, validation = map(read, (args.train, args.validation))
    if any(block["provenance"] != freeze(cards) for block in (source, validation)):
        raise ValueError("Lineup data provenance differs from frozen champion")
    disjoint(source["pools"], validation["pools"])
    space = CardSpace(cards)
    rows = validation["rows"]
    fixed = pack(space, rows, None, device)
    train = pack(space, source["rows"], args.arm, device)
    net = ComboValue(len(cards), args.width).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.03)
    generator = torch.Generator().manual_seed(args.seed)
    width = train[3].size(0) // len(source["rows"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    best, state = float("inf"), None
    started = perf_counter()
    for epoch in range(args.epochs):
        net.train()
        for batch in torch.randperm(len(source["rows"]), generator=generator).to(device).split(24):
            index = batch[:, None] * width + torch.arange(width, device=device)[None]
            decisions = (index[:, :, None] * 2 + torch.arange(2, device=device)).flatten()
            logits = net(*(tensor[decisions] for tensor in train[:3])).reshape(-1, width, 2)
            y = train[3][index].mean(-1)
            w = train[4][index, None]
            predicted = logits.sigmoid()
            delta = predicted - predicted[:, :1] - (y - y[:, :1])
            loss = ((F.binary_cross_entropy_with_logits(logits, y, reduction="none") + 2 * delta.square()) * w).sum() / (2 * w.sum())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        net.eval()
        with torch.inference_mode():
            logits = network_logits(net, fixed)
            loss = F.binary_cross_entropy_with_logits(logits, fixed[3].mean(-1)).item()
        if loss < best:
            best, selected = loss, epoch + 1
            state = {name: tensor.detach().cpu().clone() for name, tensor in net.state_dict().items()}
    elapsed = perf_counter() - started
    net.load_state_dict(state)
    args.output.mkdir(parents=True)
    weights = args.output / "state_dict.pt"
    torch.save(state, weights)
    config = {"provenance": freeze(cards), "arm": args.arm, "width": args.width, "seed": args.seed, "epochs": args.epochs,
              "selected_epoch": selected, "splits": {"train": source["pools"], "validation": validation["pools"]}}
    save(args.output / "config.json", config)
    cpu = load_model(args.output, cards)
    with torch.inference_mode():
        probes = [tensor[:12] for tensor in fixed[:3]]
        expected = net(*probes).cpu()
        actual = cpu(*(tensor.cpu() for tensor in probes))
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-5)
        save(args.output / "probes.json", {"input": [tensor.tolist() for tensor in probes], "logits": actual.tolist()})
    metrics = {"parameters": sum(p.numel() for p in net.parameters()), "seconds": elapsed, "weights_bytes": weights.stat().st_size,
               "train_positions": len(source["rows"]), "train_games": int(train[4].sum()) * train[3].size(1) * train[3].size(2),
               "validation_positions": len(rows), "device": str(device), "torch": torch.__version__,
               "validation": assess(net, fixed, rows), "cpu_reload_delta": float((expected - actual).abs().max())}
    if device.type == "cuda":
        metrics.update(gpu=torch.cuda.get_device_name(device), peak_mib=torch.cuda.max_memory_allocated(device) / 2**20)
    save(args.output / "metrics.json", metrics)
    print(json.dumps(metrics), flush=True)


def load_model(path, cards):
    config = read(path / "config.json")
    if config["provenance"] != freeze(cards):
        raise ValueError("Lineup model provenance mismatch")
    model = ComboValue(len(cards), config["width"])
    model.load_state_dict(torch.load(path / "state_dict.pt", map_location="cpu", weights_only=True))
    return model.eval()


class ComboLineup(LineupPolicy):
    def __init__(self, profiles, cards, worker, space, model):
        super().__init__(profiles, cards, worker)
        self.space, self.model = space, model
        self.proposed = 0

    @torch.inference_mode()
    def options(self, decision):
        original = super().options(decision)
        if len(original) < 5:
            return original
        try:
            team, foe = [encode_team(self.space, decision.observation, side) for side in ("team", "opponent_team")]
        except (KeyError, TypeError):
            return original
        if len(team) != 6 or len(foe) != 6:
            return original
        if any(mon["species"] not in self.cards for side in ("team", "opponent_team") for mon in decision.observation[side]):
            return original
        theirs = [self.cards[mon["species"]] for mon in decision.observation["opponent_team"]]
        ours = [self.cards[mon["species"]] for mon in decision.observation["team"]]
        reply_names = [ident(card["species"]) for card in theirs]
        public_reply = make_decision(PREVIEW, {"your_roster": theirs, "opponent_roster": ours}, [{"roster": reply_names}], "p2")
        replies = [encode_action(self.space, action) for action in alternative_reply(public_reply)]
        known = {signature(action) for action in original}
        outside = [action for key, action in distinct_options(decision).items() if key not in known]
        if not outside:
            return original
        actions = [encode_action(self.space, action) for action in (original[-1], *outside)]
        roles, features = encode_rows(self.space, (team, foe), actions, replies)
        identities = torch.tensor([[team, foe]]).expand(len(roles), -1, -1)
        scores = self.model(identities, torch.tensor(roles), torch.tensor(features, dtype=torch.float32)).sigmoid().view(-1, 2).mean(1)
        best = int(scores[1:].argmax()) + 1
        if scores[best] >= scores[0] + .015:
            self.proposed += 1
            return original[:-1] + [outside[best - 1]]
        return original


def evaluate(seeds, cards, rivals, weights):
    pairs, errors, counts = {}, [], Counter()
    with ExitStack() as stack:
        experiment = Coverage(stack, cards)
        space = CardSpace(cards, experiment.profiles)
        fitted_cards = index_cards(cards)
        models = {name: load_model(path, cards) for name, path in weights.items()}
        for root in seeds:
            rival = rivals[root % len(rivals)]
            pair = []
            for actor in SEATS:
                foe = other(actor)
                try:
                    rosters, battle_seed, times = draft_root(experiment, root, cards, {actor: "champion", foe: rival})
                    views = preview_snapshot(rosters, battle_seed, experiment.league.engine)
                    opp_policy = LineupPolicy(experiment.profiles, fitted_cards, experiment.preview)
                    reply = views[foe].validate(opp_policy(views[foe]))
                    counts["opponent_preview_fallbacks"] += opp_policy.fallbacks
                    scores, outcomes = {}, {}
                    for name in ("frozen", *models):
                        policy = (LineupPolicy(experiment.profiles, fitted_cards, experiment.preview) if name == "frozen"
                                  else ComboLineup(experiment.profiles, fitted_cards, experiment.preview, space, models[name]))
                        started = perf_counter()
                        previous = experiment.preview.games
                        action = views[actor].validate(policy(views[actor]))
                        elapsed = (perf_counter() - started) * 1000
                        used = experiment.preview.games - previous
                        key = fingerprint(action)
                        if key not in outcomes:
                            outcomes[key] = terminal(experiment, rosters, actor, battle_seed, action, reply)
                            counts["terminal_games"] += 1
                        scores[name] = {"score": outcomes[key], "action": encode_action(space, action), "elapsed_ms": elapsed,
                                        "review_starts": used, "proposed": getattr(policy, "proposed", 0), "fallbacks": policy.fallbacks}
                    pair.append({"rival": rival, "draft_ms": times, "results": scores})
                except (ValueError, RuntimeError, OSError) as error:
                    errors.append({"seed": root, "seat": actor, "error": str(error)})
            if len(pair) == 2:
                pairs[root] = pair
        counts["preview_inner_games"] = experiment.preview.games
        counts["draft_inner_games"] = (experiment.hybrid.rollout.activity["rollout_games"]
                                      + experiment.league.rollout.activity["rollout_games"])
        counts["draft_final_review_battles"] = experiment.activity["full_leaf_games"]
        counts["preview_snapshots"] = 2 * len(pairs)
    return pairs, errors, counts


def battle_report(pairs, method, against):
    rows = [row["results"][method] for pair in pairs for row in pair]
    deltas = [mean(row["results"][method]["score"] - row["results"][against]["score"] for row in pair) for pair in pairs]
    return {"wins": sum(row["score"] == 1 for row in rows), "draws": sum(row["score"] == .5 for row in rows),
            "games": len(rows), "score": mean(row["score"] for row in rows), "delta": mean(deltas), "ci95": mean_ci(deltas),
            "different_actions": sum(row["results"][method]["action"] != row["results"][against]["action"] for pair in pairs for row in pair),
            "preview_p95_ms": _percentile([row["elapsed_ms"] for row in rows], .95),
            "preview_p99_ms": _percentile([row["elapsed_ms"] for row in rows], .99),
            "preview_timeouts": sum(row["elapsed_ms"] >= 85000 for row in rows),
            "review_starts": sum(row["review_starts"] for row in rows), "proposed": sum(row["proposed"] for row in rows),
            "fallbacks": sum(row["fallbacks"] for row in rows)}


def benchmark(args, cards):
    check_hybrid()
    seeds, pools = select_pools(args, cards)
    models = {path.name: path for path in args.models}
    if len(models) != len(args.models) or "frozen" in models:
        raise ValueError("Use distinct model directories")
    for path in models.values():
        for split in read(path / "config.json")["splits"].values():
            disjoint(pools, split)
    started = perf_counter()
    pairs, errors, counts = {}, [], Counter()
    for data, issues, events in parallel(evaluate, seeds, args.workers, cards, args.rivals, models):
        pairs.update(data)
        errors.extend(issues)
        counts.update(events)
    if not pairs:
        raise RuntimeError(f"No completed paired games: {errors[:3]}")
    report = {"pairs": len(pairs), "seconds": perf_counter() - started, "workers": args.workers, "results": {
        name: battle_report(list(pairs.values()), name, "frozen") for name in ("frozen", *models)},
        "vs_first_model": {name: battle_report(list(pairs.values()), name, next(iter(models))) for name in list(models)[1:]},
        "by_rival": {rival: {name: battle_report([p for p in pairs.values() if p[0]["rival"] == rival], name, "frozen")
                    for name in ("frozen", *models)} for rival in {p[0]["rival"] for p in pairs.values()}},
        "draft_timeouts": sum(t >= 15000 for pair in pairs.values() for row in pair for t in row["draft_ms"]),
        "counts": dict(counts), "errors": errors, "models": {name: baseline_hash(path) for name, path in models.items()}}
    save(args.output, {"provenance": freeze(cards), "report": report, "pools": pools, "paired_games": dict(sorted(pairs.items()))})
    print(json.dumps({k: v for k, v in report.items() if k != "by_rival"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    analysis = commands.add_parser("audit")
    analysis.add_argument("--output", type=Path, required=True)
    diagnostic = commands.add_parser("compare-data")
    fit_args = commands.add_parser("train")
    for cmd in (fit_args, diagnostic):
        cmd.add_argument("--train", type=Path, required=True)
        cmd.add_argument("--validation", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True)
    diagnostic.add_argument("--coverage", type=Path, required=True)
    fit_args.add_argument("--arm", choices=("uniform", "stratified"), required=True)
    fit_args.add_argument("--device", default="cuda")
    fit_args.add_argument("--seed", type=int, default=871)
    fit_args.add_argument("--epochs", type=int, default=36)
    fit_args.add_argument("--width", type=int, default=48)
    collect_args, bench_args, focus_args = [commands.add_parser(cmd) for cmd in ("generate", "bench", "audit-cores")]
    for cmd in (collect_args, bench_args, focus_args):
        cmd.add_argument("--seed", type=int, required=True)
        cmd.add_argument("--count", type=int, required=True)
        cmd.add_argument("--workers", type=int, default=12)
        cmd.add_argument("--rivals", choices=RIVALS, nargs="+", default=RIVALS)
        cmd.add_argument("--exclude", type=Path, nargs="*", default=[])
        cmd.add_argument("--output", type=Path, required=True)
    collect_args.add_argument("--coverage", type=Path, required=True)
    collect_args.add_argument("--draws", type=int, default=2)
    collect_args.add_argument("--extras", type=int, default=5)
    focus_args.add_argument("--coverage", type=Path, required=True)
    focus_args.add_argument("--draws", type=int, default=6)
    bench_args.add_argument("--models", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output path")
    cards = read(HISTORICAL_CARDS)
    if args.command == "audit":
        report = audit(cards, ROOT / "checkpoints")
        save(args.output, report)
        print(json.dumps({k: report[k] for k in ("draft", "active", "disjoint_unordered_matchups")}), flush=True)
    else:
        {"generate": generate, "train": fit, "bench": benchmark, "compare-data": describe_data,
         "audit-cores": focus_audit}[args.command](args, cards)


if __name__ == "__main__":
    main()
