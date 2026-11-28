"""Whole-battle policy and value learning with legal paired continuations."""

import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
from random import Random
from statistics import mean
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .__main__ import _percentile
from .battle_value import advance, decisions, opponent, terminal
from .combination_preview import draft_root, preview_snapshot
from .draft_counterfactual import check_hybrid
from .draft_coverage import CHAMPION, Coverage
from .draft_early import paired_draws
from .draft_last_pick import mean_ci, other
from .draft_leaf import CATALOG
from .draft_league import baseline_hash, disjoint, parallel, pools_for, read, save
from .draft_matchups import ROOT
from .features import BattleEncoder, HISTORY_FEATURE_VERSION, STATS, STATUSES
from .forecast import ForecastConfig, ForecastPolicy
from .game import BATTLE, BATTLE_BANK, DECISION_LIMIT, DRAFT, PREVIEW
from .joint import BattleHistory, JointPolicy
from .lineup import LineupPolicy, index_cards
from .opening_active import check_fresh, inner_games
from .opening_value import opening_rosters
from .sim import FORMAT, SEATS, Showdown, fingerprint, play_battle
from .tactics import SLOTS, tactical


VERSION = 1
WIDTH = 6
ENTITY_COUNT = 12
ENTITY_FEATURES = 2 + len(STATUSES) + len(STATS)
HISTORY_STEPS = 2
DRAFT_LEAGUE = ("champion", "pairwise", "search", "linear", "value", "prior")
BATTLE_LEAGUE = ("joint", "forecast", "forecast-replies", "tactical")
FIELDS = ("state", "actions", "cards", "mons", "legal", "teacher", "behavior", "outcome", "branches", "root", "turn", "forced")


def configure_torch():
    torch.set_num_threads(1)
    # Use the same attention implementation in Windows CPU rollouts and CUDA validation.
    torch.backends.mha.set_fastpath_enabled(False)


class Encoder:
    def __init__(self, profiles):
        self.base = BattleEncoder(profiles)
        self.card_ids = {name: i + 1 for i, name in enumerate(self.base.species)}

    def __call__(self, decision, history, ranked):
        feats = self.base.encode(decision, history=history, shortlist_only=True)
        available = dict(zip(feats.action_indices, feats.actions, strict=True))
        lead_score = ranked[0][1].get("score", 0.)
        actions = []
        for i, (action, details) in enumerate(ranked):
            score = details.get("score", 0.)
            switch = tuple(self.card_ids.get(action[slot].get("species"), 0) / len(self.card_ids) for slot in SLOTS)
            actions.append((*available[decision.actions.index(action)], score, score - lead_score, i / (WIDTH - 1), *switch))
        actions.extend([(0.,) * len(actions[0])] * (WIDTH - len(ranked)))
        board = decision.observation
        reserves = list(sorted(board["reserves"].items()))[:2]
        own_bench = [mon | {"species": name} for name, mon in reserves]
        archive = [{"species": name, "seen": name in history.revealed} for name in sorted(history.opponent_roster)[:6]]
        entities = [*board["active"], *board["opponents"], *own_bench, *([None] * (2 - len(own_bench))), *archive]
        entities.extend([None] * (ENTITY_COUNT - len(entities)))
        cards = [self.card_ids.get((mon or {}).get("species"), 0) for mon in entities]
        mons = []
        for mon in entities:
            mon = mon or {}
            mons.append([mon.get("hp", 0.), float(mon.get("seen", False)),
                         *(float(mon.get("status") == status) for status in STATUSES),
                         *(mon.get("boosts", {}).get(stat, 0) / 6 for stat in STATS)])
        return feats.state, actions, cards, mons, [i < len(ranked) for i in range(WIDTH)]


def read_data(path):
    with np.load(path, allow_pickle=False) as data:
        return json.loads(data["metadata"].item()), {name: data[name] for name in FIELDS}


def fresh(pools, paths):
    prior = [read(p)["pools"] for experiment in ("opening-active-v1", "battle-value-v1")
             for p in (ROOT / "benchmark_results" / experiment).glob("*.json") if "pools" in read(p)]
    check_fresh(pools, ())
    for split in prior:
        disjoint(split, pools)
    for path in paths:
        reference = read_data(path)[0] if path.suffix == ".npz" else read(path)
        disjoint(pools, reference["pools"])


def draft_and_preview(experiment, cards, indexed, root, actor):
    # Keep a substantial fraction of the same strong setups used at evaluation.
    ours = DRAFT_LEAGUE[root % len(DRAFT_LEAGUE)]
    theirs = DRAFT_LEAGUE[(root // len(DRAFT_LEAGUE)) % len(DRAFT_LEAGUE)]
    styles = {actor: ours, other(actor): theirs}
    rosters, seed, _ = draft_root(experiment, root, cards, styles)
    previews = preview_snapshot(rosters, seed, experiment.league.engine)
    lineups = {}
    for seat in SEATS:
        decision = previews[seat]
        use_rollout = styles[seat] == "champion" or (root + SEATS.index(seat)) % 3 == 0
        action = (LineupPolicy(experiment.profiles, indexed, experiment.preview)(decision) if use_rollout else
                  tactical(decision, card_profiles=experiment.profiles))
        lineups[seat] = decision.validate(action)
    return rosters, seed, lineups, ours, theirs


def terminal_return(winner, actor):
    return .5 if winner is None else float(winner == actor)


def targets_at_root(engine, reply, actor, ranked, policies, styles, draws, root, turn, actual, profiles, learner):
    saved = engine.call("label_root")
    if saved != reply:
        raise RuntimeError("Saving changed a player's pending decision")
    histories = {seat: policy.history.snapshot() for seat, policy in policies.items()}
    branches = [[] for _ in range(WIDTH)]
    for style, seeds in zip(styles, paired_draws(root * 1024 + turn, actual, (), len(styles), draws), strict=True):
        for seed in seeds:
            for i, (action, _) in enumerate(ranked):
                branch = engine.call("label_restore", seed=seed)
                if branch != saved:
                    raise RuntimeError(f"Simulation changed a legal action or public observation at {root}/{turn}")
                players = {actor: learner() if learner else JointPolicy(profiles),
                           other(actor): learner() if style == "learner" else opponent(profiles, style)}
                for seat in SEATS:
                    players[seat].history = BattleHistory.from_snapshot(histories[seat])
                    if isinstance(players[seat], LearnedPolicy) and isinstance(policies[seat], LearnedPolicy):
                        players[seat].trace = policies[seat].trace.copy()
                branches[i].append(terminal_return(terminal(engine, branch, players, actor, action), actor))
    if engine.call("label_restore") != saved:
        raise RuntimeError("Resume changed the original seeded battle")
    return [mean(row) if row else -1. for row in branches]


def collect(roots, cards, draws, checkpoint, late_labels):
    configure_torch()
    rows, counts = [], Counter()
    with ExitStack() as stack:
        experiment = Coverage(stack, cards)
        engine = stack.enter_context(Showdown(counterfactual_labels=True))
        indexed, encoder = index_cards(cards), Encoder(experiment.profiles)
        frozen = load_model(checkpoint, cards) if checkpoint else None
        for root in roots:
            actor = SEATS[root % 2]
            prefix_style = BATTLE_LEAGUE[(root // 2) % len(BATTLE_LEAGUE)]
            rng = Random(root ^ 0x70A551)
            rosters, actual, lineups, ours, theirs = draft_and_preview(experiment, cards, indexed, root, actor)
            counts[f"draft/{ours}/{theirs}"] += 1
            use_model = frozen is not None and (root // 8) % 4 != 0
            def learner():
                return LearnedPolicy(experiment.profiles, encoder, frozen)

            mimic = opponent(experiment.profiles, "forecast" if root // 7 % 3 == 0 else "joint")
            policies = {actor: learner() if use_model else mimic,
                        other(actor): learner() if use_model and (root // 32) % 2 else opponent(experiment.profiles, prefix_style)}
            counts["model_prefix_games"] += use_model
            counts["self_play_games"] += use_model and isinstance(policies[other(actor)], LearnedPolicy)
            counts[f"battle/{prefix_style}"] += 1
            ranker = JointPolicy(experiment.profiles)
            game = []
            used = set()
            reply = engine.call("start", format=FORMAT, seed=actual, rosters=rosters)
            for tick in range(500):
                if reply["ended"]:
                    break
                current = dict(decisions(reply))
                own = current.get(actor)
                if own is None or own.phase != BATTLE:
                    reply = advance(engine, reply, policies, lineups=lineups)
                    continue
                ranker.history = policies[actor].history
                ranked = ranker.rank(own)
                if len(ranked) == 1:
                    reply = advance(engine, reply, policies, lineups=lineups)
                    continue
                turn = own.observation["turn"]
                forced = any(not mon for mon in own.observation["active"])
                features = encoder(own, ranker.history, ranked)
                if mimic is policies[actor] and isinstance(mimic, ForecastPolicy):
                    demo = mimic.rank(own, [action for action, _ in ranked])[0][0]
                else:
                    demo = ranked[0][0]
                menu = [action for action, _ in ranked]
                teacher = menu.index(demo)
                behavior = policies[actor].index(features) if use_model else teacher
                if root % 5 != 0 and rng.random() < (.08 if use_model else .13):
                    behavior = rng.randrange(min(len(menu), 4))
                    counts["exploratory_actions"] += 1
                # Sample one early and one later intervention, including forced replacements.
                group = "early" if turn <= 2 and not forced else "later"
                if (frozen is not None or late_labels) and turn >= 5:
                    group = "late"
                branches = [-1.] * WIDTH
                if group not in used and (group != "early" or turn == 1 + (root // 3 % 2)):
                    styles = (BATTLE_LEAGUE[root % 4], BATTLE_LEAGUE[(root + 1 + tick % 3) % 4])
                    if use_model and isinstance(policies[other(actor)], LearnedPolicy) and root % 3 == 0:
                        styles = (styles[0], "learner")
                    branches = targets_at_root(engine, reply, actor, ranked, policies, styles, draws, root, tick, actual,
                                               experiment.profiles, learner if use_model else None)
                    counts["branch_games"] += len(menu) * len(styles) * draws
                    counts["branch_positions"] += 1
                    counts[f"branch/{group}"] += 1
                    used.add(group)
                game.append((*features, teacher, behavior, branches, root, turn, forced))
                reply = advance(engine, reply, policies, override=(actor, menu[behavior]), lineups=lineups)
            if not reply["ended"]:
                raise RuntimeError(f"Prefix game did not finish for {root}")
            outcome = terminal_return(reply["winner"], actor)
            rows.extend(row[:7] + (outcome,) + row[7:] for row in game)
            counts["full_games"] += 1
            counts["forced_positions"] += sum(bool(row[-1]) for row in game)
        counts.update(inner_games(experiment))
    return rows, counts


def generate(args, cards):
    seeds = list(range(args.seed, args.seed + args.count))
    pools = pools_for(cards, seeds)
    fresh(pools, args.exclude)
    if args.behavior:
        Random(args.seed ^ 0x756F72).shuffle(seeds)
    began = perf_counter()
    rows, totals = [], Counter()
    for i, (data, counts) in enumerate(parallel(collect, seeds, args.workers, cards, args.draws, args.behavior, args.late_labels), 1):
        rows.extend(data)
        totals.update(counts)
        print(json.dumps(dict(shards=i, games=totals["full_games"], branches=totals["branch_games"])), flush=True)
    rows.sort(key=lambda row: row[FIELDS.index("root")])
    arrays = {name: np.asarray(values, dtype=(np.float32 if name in {"state", "actions", "mons", "outcome", "branches"}
                                             else np.bool_ if name in {"legal", "forced"} else np.int32))
              for name, values in zip(FIELDS, zip(*rows, strict=True), strict=True)}
    report = dict(seconds=perf_counter() - began, positions=len(rows), workers=args.workers, counts=dict(totals), draws=args.draws)
    metadata = dict(report=report, catalog=fingerprint(cards), champion=CHAMPION, pools=pools, version=VERSION,
                    feature_version=HISTORY_FEATURE_VERSION, behavior=baseline_hash(args.behavior) if args.behavior else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as output:
        np.savez_compressed(output, metadata=json.dumps(metadata), **arrays)
    print(json.dumps(report), flush=True)


def masking(logits, legal):
    return logits.masked_fill(~legal, -1e4)


class BattleNetwork(nn.Module):
    def __init__(self, state_size, action_size, card_count, hidden, depth, history_steps=0):
        super().__init__()
        self.board = nn.Linear(state_size, hidden)
        self.cards = nn.Embedding(card_count + 1, hidden, padding_idx=0)
        self.roles = nn.Parameter(torch.randn(ENTITY_COUNT, hidden) * .02)
        self.mons = nn.Linear(ENTITY_FEATURES, hidden)
        self.actions = nn.Linear(action_size, hidden)
        if depth:
            layer = nn.TransformerEncoderLayer(hidden, 4, hidden * 3, dropout=.08, activation="gelu",
                                               batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
            self.read = nn.MultiheadAttention(hidden, 4, batch_first=True)
        else:
            self.encoder = self.read = None
        self.history_steps = history_steps
        if history_steps:
            self.trace_input = nn.Sequential(nn.Linear(state_size + action_size, hidden), nn.SiLU())
            self.trace_cell = nn.GRU(hidden, hidden, batch_first=True)
            self.trace_output = nn.Linear(hidden, hidden, bias=False)
            nn.init.zeros_(self.trace_output.weight)
        self.rank = nn.Sequential(nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 2))
        self.value = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, state, actions, cards, mons, legal, trace=None):
        entity = self.cards(cards) + self.roles + self.mons(mons)
        board = self.board(state)
        menu = self.actions(actions)
        exists = cards != 0
        if self.history_steps:
            observed = (trace != 0).any(dim=-1)
            memory, _ = self.trace_cell(self.trace_input(trace) * observed[..., None])
            board = board + self.trace_output(memory[:, -1]) * observed[:, -1:]
        if self.encoder is not None:
            valid = torch.cat((legal[:, :1], exists), dim=1)
            tokens = self.encoder(torch.cat((board[:, None], entity), dim=1), src_key_padding_mask=~valid)
            board = tokens[:, 0]
            interaction, _ = self.read(menu, tokens, tokens, key_padding_mask=~valid, need_weights=False)
            menu = menu + interaction
        else:
            board = board + (exists[..., None] * entity).sum(1) / exists.sum(1, keepdim=True).clamp_min(1)
        logits, q = self.rank(board[:, None] + menu).unbind(-1)
        return masking(logits, legal), q, self.value(board).squeeze(-1)


def policy_scores(logits, q, legal):
    return (logits + 2. * q.sigmoid()).masked_fill(~legal, -torch.inf)


class LearnedPolicy(JointPolicy):
    def __init__(self, profiles, encoder, model, *, value_only=False):
        super().__init__(profiles)
        self.encoder, self.model = encoder, model
        self.value_only = value_only
        self.trace = []
        self.scored = 0
        self.changes = 0
        self.switch_changes = 0
        self.inference_ms = []

    @torch.inference_mode()
    def index(self, features):
        state, actions, cards, mons, legal = features
        packet = (torch.tensor([state], dtype=torch.float32), torch.tensor([actions], dtype=torch.float32),
                  torch.tensor([cards], dtype=torch.long), torch.tensor([mons], dtype=torch.float32),
                  torch.tensor([legal], dtype=torch.bool))
        if self.model.history_steps:
            missing = self.model.history_steps - len(self.trace)
            trace = [(0.,) * (len(state) + len(actions[0]))] * missing + self.trace
            packet = (*packet, torch.tensor([trace], dtype=torch.float32))
        logits, q, _ = self.model(*packet)
        scores = q.masked_fill(~packet[4], -torch.inf) if self.value_only else policy_scores(logits, q, packet[4])
        return int(scores.argmax())

    def remember(self, features, index):
        if self.model.history_steps:
            self.trace.append((*features[0], *features[1][index]))
            self.trace = self.trace[-self.model.history_steps:]

    def record_override(self, decision, action):
        if self.model.history_steps:
            ranked = self.rank(decision)
            if len(ranked) > 1:
                index = [candidate for candidate, _ in ranked].index(action)
                self.remember(self.encoder(decision, self.history, ranked), index)

    def __call__(self, decision):
        ranked = self.rank(decision)
        index = 0
        if decision.phase == BATTLE and len(ranked) > 1:
            began = perf_counter()
            features = self.encoder(decision, self.history, ranked)
            index = self.index(features)
            self.remember(features, index)
            self.inference_ms.append((perf_counter() - began) * 1000)
            self.scored += 1
            self.changes += index != 0
            self.switch_changes += any(ranked[0][0][slot] != ranked[index][0][slot]
                                       and (ranked[0][0][slot]["type"] == "switch" or ranked[index][0][slot]["type"] == "switch")
                                       for slot in SLOTS)
        action = ranked[index][0]
        self.record(decision, action)
        return action


def load_model(path, cards):
    config = read(path / "config.json")
    if (config["catalog"], config["champion"], config["version"], config["feature_version"]) != (
            fingerprint(cards), CHAMPION, VERSION, HISTORY_FEATURE_VERSION):
        raise ValueError("Checkpoint format or frozen rule set differs")
    model = BattleNetwork(**config["network"])
    model.load_state_dict(torch.load(path / "state_dict.pt", map_location="cpu", weights_only=True))
    return model.eval()


def make_batch(arrays, device, history_steps=0):
    fields = ["state", "actions", "cards", "mons", "legal"]
    if history_steps:
        roots, behavior = arrays["root"], arrays["behavior"]
        frame = np.concatenate((arrays["state"], arrays["actions"][np.arange(len(roots)), behavior]), axis=-1)
        trace = np.zeros((len(roots), history_steps, frame.shape[-1]), dtype=np.float32)
        for i in range(history_steps):
            offset = history_steps - i
            trace[offset:, i] = frame[:-offset] * (roots[offset:] == roots[:-offset])[:, None]
        arrays = dict(arrays, trace=trace)
        fields.append("trace")
    fields.extend(("teacher", "behavior", "outcome", "branches"))
    return [torch.from_numpy(arrays[key]).to(device=device, dtype=torch.long if key in {"cards", "teacher", "behavior"} else None)
            for key in fields]


def objective(model, batch, imitation_weight=.60):
    *features, teacher, behavior, outcome, branches = batch
    legal = features[4]
    logits, q, values = model(*features)
    labeled = (branches >= 0) & legal
    reviewed = labeled.any(1)
    soft = masking(branches / .16, legal).softmax(1)
    ce = -(soft * F.log_softmax(logits, dim=1)).sum(1)
    labels = branches.clamp_min(0.)
    branch_value = F.binary_cross_entropy_with_logits(q, labels, reduction="none")
    difference = q.sigmoid() - q[:, :1].sigmoid()
    paired_error = (difference - (labels - labels[:, :1])).square()
    visited_q = q.gather(1, behavior[:, None]).squeeze(1)
    return (imitation_weight * F.cross_entropy(logits, teacher) + (ce * reviewed).sum() / reviewed.sum().clamp_min(1)
            + ((branch_value + 2. * paired_error) * labeled).sum() / labeled.sum().clamp_min(1)
            + .20 * F.binary_cross_entropy_with_logits(visited_q, outcome)
            + .35 * F.binary_cross_entropy_with_logits(values, outcome))


@torch.inference_mode()
def assess(model, batch):
    *features, teacher, behavior, outcome, branches = batch
    logits, q, value = model(*features)
    action = policy_scores(logits, q, features[4]).argmax(1)
    labeled = (branches >= 0) & features[4]
    rows = labeled.any(1)
    local = (branches.gather(1, action[:, None]).squeeze(1) - branches[:, 0])[rows]
    target = branches[rows]
    diff = q[rows].sigmoid() - q[rows, :1].sigmoid()
    pair_mse = (diff - (target - target[:, :1])).square()[labeled[rows]].mean().item()
    return dict(positions=len(action), reviewed=int(rows.sum()), changes=int((action != 0).sum()),
                imitation=(action == teacher).float().mean().item(),
                local_gain=local.mean().item(), paired_mse=pair_mse,
                value_brier=(value.sigmoid() - outcome).square().mean().item())


def fit(args, cards):
    configure_torch()
    torch.manual_seed(args.seed)
    inputs = [read_data(path) for path in (*args.data, *args.validation)]
    pools = {}
    for config, _ in inputs:
        if (config["catalog"], config["champion"], config["version"], config["feature_version"]) != (
                fingerprint(cards), CHAMPION, VERSION, HISTORY_FEATURE_VERSION):
            raise ValueError("Incompatible training arrays")
        disjoint(config["pools"], pools)
        pools.update(config["pools"])
    device = torch.device(args.device)
    history_steps = HISTORY_STEPS if args.history else 0

    def batch(split):
        return make_batch({name: np.concatenate([arrays[name] for _, arrays in split]) for name in FIELDS}, device, history_steps)

    data = batch(inputs[:len(args.data)])
    validation = batch(inputs[len(args.data):])
    network = dict(state_size=data[0].size(-1), action_size=data[1].size(-1), card_count=len(cards),
                   hidden=32 if args.size == "small" else 128, depth=0 if args.size == "small" else 3, history_steps=history_steps)
    model = BattleNetwork(**network).to(device)
    if args.init:
        loaded = model.load_state_dict(load_model(args.init, cards).state_dict(), strict=not history_steps)
        if loaded.unexpected_keys or any(not key.startswith("trace_") for key in loaded.missing_keys):
            raise ValueError("Warm start does not match Battle Policy architecture")
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0004 if args.init else .001, weight_decay=.025)
    generator = torch.Generator().manual_seed(args.seed)
    best, chosen, state, progress = -float("inf"), 0, None, []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    began = perf_counter()
    for epoch in range(args.epochs):
        model.train()
        for indices in torch.randperm(len(data[0]), generator=generator).to(device).split(256):
            batch = [field[indices] for field in data]
            loss = objective(model, batch, args.imitation_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        model.eval()
        metrics = assess(model, validation)
        grade = metrics["local_gain"] - .1 * metrics["paired_mse"]
        progress.append(metrics)
        if grade > best:
            best, chosen = grade, epoch + 1
            state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - began
    model.load_state_dict(state)
    probe = [t[:16] for t in validation[:-4]]
    cpu_model = BattleNetwork(**network).eval()
    cpu_model.load_state_dict(state)
    with torch.inference_mode():
        expected = tuple(t.cpu() for t in model(*probe))
        actual = cpu_model(*(t.cpu() for t in probe))
        for cpu, gpu in zip(actual, expected, strict=True):
            torch.testing.assert_close(cpu, gpu, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(policy_scores(*actual[:2], probe[4].cpu()).argmax(1),
                                   policy_scores(*expected[:2], probe[4].cpu()).argmax(1))
    reload_delta = max(float((cpu - gpu).abs().max()) for cpu, gpu in zip(actual, expected, strict=True))
    args.output.mkdir(parents=True)
    torch.save(state, args.output / "state_dict.pt")
    save(args.output / "config.json", dict(network=network, seed=args.seed, epochs=args.epochs, chosen=chosen, pools=pools,
                                          imitation_weight=args.imitation_weight,
                                          size=args.size, source=[str(p) for p in args.data], validation=[str(p) for p in args.validation],
                                          init=baseline_hash(args.init) if args.init else None, champion=CHAMPION,
                                          catalog=fingerprint(cards), version=VERSION, feature_version=HISTORY_FEATURE_VERSION))
    with torch.inference_mode():
        reloaded = load_model(args.output, cards)(*(t.cpu() for t in probe))
        for a, b in zip(actual, reloaded, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    save(args.output / "probes.json", dict(inputs=[t.tolist() for t in probe], outputs=[t.tolist() for t in actual]))
    report = dict(seconds=elapsed, samples_per_second=len(data[0]) * args.epochs / elapsed,
                  device=str(device), parameters=sum(p.numel() for p in model.parameters()),
                  weights_bytes=(args.output / "state_dict.pt").stat().st_size, train=assess(model, data),
                  validation=progress[chosen - 1], history=progress, chosen=chosen, reload_max_error=reload_delta)
    if device.type == "cuda":
        report.update(gpu=torch.cuda.get_device_name(device), peak_mib=torch.cuda.max_memory_allocated(device) / 2**20)
    save(args.output / "metrics.json", report)
    print(json.dumps({key: val for key, val in report.items() if key != "history"}), flush=True)


def evaluate(seeds, cards, checkpoints, value_only):
    configure_torch()
    models = {name: load_model(path, cards) for name, path in checkpoints.items()}
    results = {}
    with ExitStack() as stack:
        experiment = Coverage(stack, cards)
        encoder, indexed = Encoder(experiment.profiles), index_cards(cards)
        for root in seeds:
            rival, style = (("champion", "joint"), ("champion", "joint"), ("search", "joint"),
                            ("pairwise", "forecast-replies"), ("champion", "forecast-risk"))[root % 5]
            pair = []
            for actor in SEATS:
                rosters, actual, lineups, draft_times = opening_rosters(experiment, indexed, cards, root, actor, rival)
                outcomes = {}
                for name in ("joint", "position", *models):
                    player = (LearnedPolicy(experiment.profiles, encoder, models[name], value_only=name in value_only) if name in models else
                              opponent(experiment.profiles, "forecast" if name == "position" else "joint"))
                    foe = (ForecastPolicy(experiment.profiles, ForecastConfig(candidates=10, position=.80, replies=True, risk=.28))
                           if style == "forecast-risk" else opponent(experiment.profiles, style))
                    policies = {actor: player, other(actor): foe}
                    times, budget = [], BATTLE_BANK
                    checks = Counter()

                    def choose(seat, decision, _):
                        nonlocal budget
                        policy = policies[seat]
                        if decision.phase == PREVIEW:
                            policy.observe(decision)
                            return decision.validate(lineups[seat])
                        began = perf_counter()
                        action = decision.validate(policy(decision))
                        elapsed = perf_counter() - began
                        if seat == actor:
                            checks["actions"] += 1
                            times.append(elapsed * 1000)
                            if budget <= elapsed:
                                checks["bank_forfeit"] += 1
                                return None
                            if elapsed >= DECISION_LIMIT[BATTLE]:
                                checks["action_timeout"] += 1
                                return None
                            budget -= elapsed
                        return action

                    winner = play_battle(experiment.league.engine, rosters, policies, root, actual, choose)
                    outcomes[name] = dict(score=terminal_return(winner, actor), timing_ms=times, checks=dict(checks),
                                          inference_ms=getattr(player, "inference_ms", []),
                                          scored=getattr(player, "scored", 0), changes=getattr(player, "changes", 0),
                                          switch_changes=getattr(player, "switch_changes", 0))
                pair.append(dict(actor=actor, rival=rival, style=style, draft_max_ms=max(draft_times), method=outcomes))
            results[root] = pair
        counts = inner_games(experiment)
    return results, counts


def comparison(pairs, method, base):
    gains = [mean(row["method"][method]["score"] - row["method"][base]["score"] for row in pair) for pair in pairs]
    return dict(delta=mean(gains), ci95=mean_ci(gains), discordant_pools=sum(v != 0 for v in gains))


def summarize(pairs):
    rows = [row for pair in pairs for row in pair]
    methods = tuple(rows[0]["method"])
    result = {}
    for name in methods:
        versions = [row["method"][name] for row in rows]
        counts = Counter()
        for item in versions:
            counts.update(item["checks"])
        times = [t for item in versions for t in item["timing_ms"]]
        inference = [t for item in versions for t in item["inference_ms"]]
        result[name] = dict(wins=sum(v["score"] == 1 for v in versions), draws=sum(v["score"] == .5 for v in versions),
                            games=len(versions), changes=sum(v["changes"] for v in versions),
                            switch_changes=sum(v["switch_changes"] for v in versions), scored=sum(v["scored"] for v in versions),
                            p95_ms=_percentile(times, .95), p99_ms=_percentile(times, .99),
                            inference_p95_ms=_percentile(inference, .95), inference_p99_ms=_percentile(inference, .99),
                            illegal_actions=0, checks=dict(counts),
                            vs_joint=comparison(pairs, name, "joint"), vs_position=comparison(pairs, name, "position"))
    return result


def bench(args, cards):
    checkpoints = {}
    for value in args.model:
        name, path = value.split("=", 1)
        if name in checkpoints or name in {"joint", "position"}:
            raise ValueError(f"Repeated or reserved benchmark name: {name}")
        checkpoints[name] = Path(path)
    seeds = range(args.seed, args.seed + args.count)
    pools = pools_for(cards, seeds)
    fresh(pools, args.exclude)
    for path in checkpoints.values():
        disjoint(pools, read(path / "config.json")["pools"])
    if set(args.value_only) - checkpoints.keys():
        raise ValueError("Value-only option names must be defined by --model")
    all_pairs, counts = {}, Counter()
    began = perf_counter()
    for i, (pairs, part) in enumerate(parallel(evaluate, seeds, args.workers, cards, checkpoints, args.value_only), 1):
        all_pairs.update(pairs)
        counts.update(part)
        print(json.dumps(dict(shards=i, pairs=len(all_pairs))), flush=True)
    ordered = [all_pairs[root] for root in sorted(all_pairs)]
    if len(ordered) != args.count:
        raise RuntimeError("Incomplete paired benchmark")
    summary = summarize(ordered)
    rivals = {(pair[0]["rival"], pair[0]["style"]) for pair in ordered}
    report = dict(seconds=perf_counter() - began, count=len(ordered), methods=summary, counts=dict(counts),
                  draft_timeouts=sum(row["draft_max_ms"] >= DECISION_LIMIT[DRAFT] * 1000 for pair in ordered for row in pair),
                  by_opponent={f"{rival}/{style}": summarize([pair for pair in ordered if (pair[0]["rival"], pair[0]["style"]) == (rival, style)])
                               for rival, style in sorted(rivals)})
    save(args.output, dict(report=report, pools=pools, paired=all_pairs, champion=CHAMPION, value_only=args.value_only,
                           checkpoints={name: baseline_hash(path) for name, path in checkpoints.items()}))
    print(json.dumps({key: value for key, value in report.items() if key != "by_opponent"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sample, train, test = (sub.add_parser(name) for name in ("collect", "train", "bench"))
    for step in (sample, train, test):
        step.add_argument("--seed", type=int, required=True)
        step.add_argument("--output", type=Path, required=True)
    for step in (sample, test):
        step.add_argument("--count", type=int, required=True)
        step.add_argument("--workers", type=int, default=12)
        step.add_argument("--exclude", type=Path, nargs="*", default=[])
    sample.add_argument("--draws", type=int, default=2)
    sample.add_argument("--behavior", type=Path)
    sample.add_argument("--late-labels", action="store_true", help="Include one extra paired position from turn five or later")
    train.add_argument("--data", type=Path, nargs="+", required=True)
    train.add_argument("--validation", type=Path, nargs="+", required=True)
    train.add_argument("--device", default="cuda")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--size", choices=("small", "large"), required=True)
    train.add_argument("--init", type=Path)
    train.add_argument("--imitation-weight", type=float, default=.60)
    train.add_argument("--history", action="store_true", help="Use the previous two discretionary decisions and played actions")
    test.add_argument("--model", nargs="+", required=True, metavar="NAME=CHECKPOINT")
    test.add_argument("--value-only", nargs="*", default=[], metavar="NAME")
    args = parser.parse_args()
    if (args.output.exists() or min(getattr(args, key, 1) for key in ("count", "workers", "draws", "epochs")) < 1
            or not 0 <= getattr(args, "imitation_weight", .60) <= 1):
        parser.error("Use fresh output paths and positive sizes")
    check_hybrid()
    {"collect": generate, "train": fit, "bench": bench}[args.command](args, read(CATALOG))


if __name__ == "__main__":
    main()
