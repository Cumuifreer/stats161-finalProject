"""Small, seeded PPO runs over complete local Showdown games."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from hashlib import sha256
import json
from multiprocessing import get_context
from pathlib import Path
from random import Random
from statistics import correlation, mean
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from .features import HISTORY_FEATURE_VERSION, BattleEncoder
from .game import BATTLE, DRAFT, PREVIEW, Decision
from .joint import BattleHistory, JointPolicy
from .model import ActionRankingNetwork, ActorCriticNetwork, BattlePolicy
from .pivot import PivotOpponent
from .policy import heuristic
from .sim import SEATS, Showdown, fingerprint, play_game, pool_fingerprint

GAMMA = 1.0
GAE_LAMBDA = .95
CLIP = .2
ENTROPY = .01
LEARNING_RATE = .0003
EPOCHS = 4
MINIBATCH = 128


def _cpu_weights(model):
    return {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}


def _cpu_tree(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu()
    if isinstance(data, dict):
        return {key: _cpu_tree(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_cpu_tree(value) for value in data]
    if isinstance(data, tuple):
        return tuple(_cpu_tree(value) for value in data)
    return data


def terminal_reward(winner, seat):
    return 0. if winner is None else 1. if winner == seat else -1.


def episode_advantages(values: list[float], outcome: float) -> torch.Tensor:
    """Only the final learner decision gets a reward. A complete game ends with zero bootstrap."""
    advantages = torch.empty(len(values), dtype=torch.float32)
    advantage = next_value = 0.
    for index in range(len(values) - 1, -1, -1):
        reward = outcome if index == len(values) - 1 else 0.
        delta = reward + GAMMA * next_value - values[index]
        advantage = delta + GAMMA * GAE_LAMBDA * advantage
        advantages[index] = advantage
        next_value = values[index]
    return advantages


def masked_distribution(logits, valid):
    return torch.distributions.Categorical(logits=logits.masked_fill(~valid, -torch.inf))


class RolloutPolicy(BattlePolicy):
    def __init__(self, model, encoder, seed, *, shortlist_only=False):
        super().__init__(model, torch.device("cpu"), encoder=encoder, shortlist_only=shortlist_only)
        self.rng = torch.Generator(device="cpu").manual_seed(seed)
        self.steps = []
        self.candidates = Counter()

    @torch.inference_mode()
    def __call__(self, decision):
        if decision.phase != BATTLE:
            return super().__call__(decision)
        self.observe(decision)
        features = self.encode(decision)
        state = torch.tensor(features.state, dtype=torch.float32)
        actions = torch.tensor(features.actions, dtype=torch.float32)
        logits, values = self.model.policy_value(state[None], actions[None])
        log_probs = logits[0].log_softmax(dim=0)
        # Forced actions still contribute a value target, with log probability zero.
        # Keep the legacy RNG consumption for v1 checkpoint resumes.
        index = (0 if self.shortlist_only and len(actions) == 1
                 else int(torch.multinomial(log_probs.exp(), 1, generator=self.rng)))
        action = decision.actions[features.action_indices[index]]
        self.steps.append((state, actions, index, log_probs[index].item(), values[0].item()))
        self.candidates[len(actions)] += 1
        self.history.record(decision, action)
        return action


class ValueAuditPolicy(BattlePolicy):
    """Record the first scored decision before the game outcome is known."""

    def __init__(self, policy):
        super().__init__(policy.model, policy.device, encoder=policy.encoder, shortlist_only=policy.shortlist_only)
        self.opening_value = None

    @torch.inference_mode()
    def evaluate(self, decision, *, history=None):
        if self.opening_value is not None:
            return super().evaluate(decision, history=history)
        features = self.encode(decision, history=history)
        states = torch.tensor([features.state], dtype=torch.float32, device=self.device)
        actions = torch.tensor([features.actions], dtype=torch.float32, device=self.device)
        logits, value = self.model.policy_value(states, actions)
        self.opening_value = value.item()
        return features.action_indices, logits[0].cpu()


def value_summary(examples):
    predicted, actual = zip(*examples, strict=True)
    baseline = mean(actual)
    return {"games": len(actual), "mean_value": mean(predicted), "mean_return": baseline,
            "mse": mean((value - outcome) ** 2 for value, outcome in examples),
            "constant_mean_mse": mean((outcome - baseline) ** 2 for outcome in actual),
            "pearson_r": correlation(predicted, actual) if len(set(predicted)) > 1 and len(set(actual)) > 1 else None}


def _initialise_worker():
    torch.set_num_threads(1)


def sample_matches(config, cards, weights, past, ace, schedule):
    """Own simulator and Pivot worker per CPU task. Return on-policy learner steps only."""
    encoder = BattleEncoder(config["card_profiles"])
    shortlist = config["shortlist_only"]
    model = ActorCriticNetwork(**config["network"])
    model.load_state_dict(weights)
    historical = []
    for snapshot in past:
        opponent = ActorCriticNetwork(**config["network"])
        opponent.load_state_dict(snapshot)
        historical.append(BattlePolicy(opponent, torch.device("cpu"), encoder=encoder, shortlist_only=shortlist))
    steps, advantages, outcomes, targets = [], [], Counter(), []
    counts = Counter()
    candidates, by_opponent = Counter(), {}
    started = perf_counter()
    with ExitStack() as stack:
        engine = stack.enter_context(Showdown())
        pivot = stack.enter_context(PivotOpponent(ace))
        for seed, opponent_name, history_index in schedule:
            for seat, other in (SEATS, SEATS[::-1]):
                learner = RolloutPolicy(model, encoder, 2 * seed + int(seat == "p2"), shortlist_only=shortlist)
                if opponent_name == "joint":
                    opposition = JointPolicy(config["card_profiles"])
                elif opponent_name == "historical":
                    opposition = historical[history_index].for_match()
                else:
                    pivot.style = opponent_name == "pivot"
                    opposition = pivot
                samples = {player: [] for player in SEATS}
                winner = play_game(engine, cards, {seat: learner, other: opposition}, seed, samples,
                                   fixed_policies={DRAFT: heuristic, PREVIEW: heuristic})
                if any(event for records in samples.values() for _, _, event in records):
                    raise RuntimeError(f"PPO training game {seed}/{seat} had a timeout or invalid action")
                if not learner.steps:
                    raise RuntimeError(f"PPO training game {seed}/{seat} collected no battle decisions")
                result = terminal_reward(winner, seat)
                values = [row[-1] for row in learner.steps]
                per_episode_advantages = episode_advantages(values, result)
                advantages.extend(per_episode_advantages.tolist())
                targets.extend((per_episode_advantages + torch.tensor(values)).tolist())
                steps.extend(learner.steps)
                counts[opponent_name] += 1
                outcome = ("loss", "draw", "win")[int(result) + 1]
                outcomes[outcome] += 1
                by_opponent.setdefault(opponent_name, Counter())[outcome] += 1
                candidates.update(learner.candidates)
    states, actions, choices, log_probs, _ = zip(*steps, strict=True)
    return {
        "states": torch.stack(states), "actions": torch.cat(actions), "lengths": [len(row) for row in actions],
        "records": torch.tensor(list(zip(choices, log_probs, advantages, targets, strict=True))),
        "opponents": dict(counts), "outcomes": dict(outcomes), "candidates": candidates, "by_opponent": by_opponent,
        "simulator_seconds": perf_counter() - started,
    }


def _schedule(spec, update, history_count):
    n = spec["pairs"]
    start = spec["seed"] + (update - 1) * n
    rng = Random(start)
    increment = spec.get("history_end", .35) - .10 if "history_end" in spec else .25
    fraction = .10 + increment * (update - 1) / max(1, spec["total_updates"] - 1)
    history_games = round(n * fraction)
    choices = [("historical", rng.randrange(history_count)) for _ in range(history_games)]
    choices.extend((name, 0) for name in (("joint", "pivot", "pivot-base")[i % 3]
                                        for i in range(n - history_games)))
    rng.shuffle(choices)
    return [(start + i, name, snapshot) for i, (name, snapshot) in enumerate(choices)]


def collect(executor, config, cards, model, past, ace, update, workers):
    schedule = _schedule(config["ppo"], update, len(past))
    weights = _cpu_weights(model)
    jobs = [executor.submit(sample_matches, config, cards, weights, past, ace, schedule[i::workers])
            for i in range(min(workers, len(schedule)))]
    return [job.result() for job in jobs]


def _batch(chunks, device):
    states = torch.cat([chunk["states"] for chunk in chunks])
    rows = [row for chunk in chunks for row in chunk["actions"].split(chunk["lengths"])]
    padded = pad_sequence(rows, batch_first=True)
    lengths = torch.tensor([len(row) for row in rows])
    valid = torch.arange(padded.size(1))[None, :] < lengths[:, None]
    choices, logp, advantages, returns = torch.cat([chunk["records"] for chunk in chunks]).unbind(dim=1)
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
    return tuple(tensor.to(device) for tensor in (states, padded, valid, choices.long(), logp, advantages, returns))


def optimise(model, optimizer, device, chunks, seed):
    states, actions, valid, choice, old_logp, advantages, returns = _batch(chunks, device)
    # Recreate the CPU rollout distribution on the learner device before any weights change.
    with torch.no_grad():
        errors = []
        for index in torch.arange(len(states), device=device).split(MINIBATCH):
            logits = model(states[index], actions[index])
            errors.append((masked_distribution(logits, valid[index]).log_prob(choice[index]) - old_logp[index]).abs().amax())
        logp_error = torch.stack(errors).amax().item()
        if not logp_error <= 2e-4:
            raise ValueError(f"Rollout and update action log probabilities differ by {logp_error:.6g}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        start_event, stop_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_event.record()
    model.train()
    random = torch.Generator().manual_seed(seed)
    total = torch.zeros(5, device=device)
    started = perf_counter()
    for _ in range(EPOCHS):
        order = torch.randperm(len(states), generator=random).to(device)
        for index in order.split(MINIBATCH):
            logits, values = model.policy_value(states[index], actions[index])
            dist = masked_distribution(logits, valid[index])
            delta = dist.log_prob(choice[index]) - old_logp[index]
            ratio = delta.exp()
            clipped = ratio.clamp(1 - CLIP, 1 + CLIP)
            policy_loss = -torch.minimum(ratio * advantages[index], clipped * advantages[index]).mean()
            value_loss = F.mse_loss(values, returns[index])
            entropy = dist.entropy().mean()
            loss = policy_loss + .5 * value_loss - ENTROPY * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                approx_kl = (ratio - 1 - delta).mean()
                clipped_fraction = ((ratio - 1).abs() > CLIP).float().mean()
                total += torch.stack((policy_loss, value_loss, entropy, approx_kl, clipped_fraction)) * len(index)
    if device.type == "cuda":
        stop_event.record()
        stop_event.synchronize()
    seconds = perf_counter() - started
    passes = len(states) * EPOCHS
    metrics = dict(zip(("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"), (total / passes).tolist(), strict=True))
    if device.type == "cuda":
        metrics["cuda_stream_ms"] = round(start_event.elapsed_time(stop_event), 2)
        metrics["cuda_peak_mib"] = round(torch.cuda.max_memory_allocated(device) / 2**20, 2)
    return metrics | {"decisions": len(states), "old_logp_max_abs_error": logp_error, "update_seconds": round(seconds, 3),
                      "update_samples_per_second": round(passes / seconds, 2)}


def _split(cards, previous, spec, final_suite):
    splits = {f"bc_{name}": group for name, group in previous.items()}
    ranges = {
        "train": range(spec["seed"], spec["seed"] + spec["pairs"] * spec["total_updates"]),
        "validation": range(spec["validation_seed"], spec["validation_seed"] + spec["validation_pairs"]),
        "final": range(final_suite["seed"], final_suite["seed"] + final_suite["pairs"]),
    }
    old_seeds = {int(seed) for group in splits.values() for seed in group}
    old_pools = {pool for group in splits.values() for pool in group.values()}
    for name, seeds in ranges.items():
        group = {str(seed): pool_fingerprint(cards, seed) for seed in seeds}
        if old_seeds.intersection(seeds) or old_pools.intersection(group.values()):
            raise ValueError(f"PPO {name} overlaps another seed or draft pool split")
        old_seeds.update(seeds)
        old_pools.update(group.values())
        if name != "final":
            splits[name] = group
    return splits


def _config(args, cards):
    source = json.loads((args.bc_init / "config.json").read_text(encoding="utf-8"))
    if source.get("architecture", "ranking") != "ranking" or source["feature_version"] != HISTORY_FEATURE_VERSION:
        raise ValueError("PPO warm start requires a history-enabled BC ranking checkpoint")
    if source["catalog_sha256"] != fingerprint(cards):
        raise ValueError("PPO and the BC initializer must use the same catalog")
    shortlist = args.shortlist
    if shortlist and not source.get("shortlist_only"):
        raise ValueError("Shortlisted PPO requires BC weights trained on the Tactical shortlist")
    spec = {key: getattr(args, key) for key in ("pairs", "seed", "workers", "total_updates", "validation_seed",
                                               "validation_pairs", "eval_every")}
    spec.update(gamma=GAMMA, gae_lambda=GAE_LAMBDA, clip=CLIP, entropy=ENTROPY, learning_rate=LEARNING_RATE,
                epochs=EPOCHS, minibatch=MINIBATCH)
    if args.history_end != .35:
        spec["history_end"] = args.history_end
    final = json.loads(args.final_suite.read_text(encoding="utf-8"))
    revision = subprocess.run(["git", "-C", str(args.ace_trainer), "rev-parse", "HEAD"], capture_output=True, text=True)
    if revision.returncode or revision.stdout.strip() != final["ace_trainer_revision"]:
        raise ValueError("PPO requires the Ace Trainer revision pinned by the final benchmark")
    final_pools = [pool_fingerprint(cards, seed) for seed in range(final["seed"], final["seed"] + final["pairs"])]
    if final["catalog_sha256"] != fingerprint(cards) or final["pool_order_sha256"] != fingerprint(final_pools):
        raise ValueError("PPO final benchmark differs from the pinned catalog and pool order")
    splits = _split(cards, source["split"], spec, final)
    return {key: source[key] for key in ("network", "feature_version", "catalog_sha256", "battle_format", "card_profiles")} | {
        "architecture": "actor_critic", "shortlist_only": shortlist, "ppo": spec, "split": splits, "final_suite": final,
        "warm_start_sha256": sha256((args.bc_init / "state_dict.pt").read_bytes()).hexdigest(),
    }


def _golden_samples(model, config, base):
    policy = BattlePolicy(model, torch.device("cpu"), encoder=BattleEncoder(config["card_profiles"]),
                          shortlist_only=config["shortlist_only"])
    samples = []
    with torch.inference_mode():
        for datum in base:
            decision = Decision.from_snapshot(datum["decision"])
            history = BattleHistory.from_snapshot(datum["history"])
            features = policy.encode(decision, history=history)
            states = torch.tensor([features.state], dtype=torch.float32)
            actions = torch.tensor([features.actions], dtype=torch.float32)
            logits, value = model.policy_value(states, actions)
            choice = decision.actions[features.action_indices[int(logits[0].argmax())]]
            samples.append(datum | {"action": choice, "scores": logits[0].tolist(), "value": value.item()})
    return samples


def _save(output, update, config, model, optimizer, history, metrics, samples):
    final = output / f"step-{update:04d}"
    if final.exists():
        raise FileExistsError(final)
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output, prefix=".step-") as scratch:
        pending = Path(scratch) / "checkpoint"
        pending.mkdir()
        policy = _cpu_weights(model)
        torch.save(policy, pending / "state_dict.pt")
        torch.save({"update": update, "optimizer": _cpu_tree(optimizer.state_dict()), "history": history},
                   pending / "training_state.pt")
        cpu_model = ActorCriticNetwork(**config["network"])
        cpu_model.load_state_dict(policy)
        for name, data in (("config", config), ("metrics", metrics), ("test_samples", _golden_samples(cpu_model, config, samples))):
            (pending / f"{name}.json").write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")
        pending.rename(final)
    return final


def evaluate_validation(checkpoint, args):
    # Reuse the normal paired runner and retain the dev results separately from training and final evaluation.
    from argparse import Namespace
    from contextlib import redirect_stdout
    from io import StringIO

    from .__main__ import _bench

    request = Namespace(cards=args.cards, left="ppo", right="joint", pairs=args.validation_pairs, seed=args.validation_seed,
                        battle_only=True, preview_only=False, ppo_checkpoint=checkpoint, bc_device="cpu", testing_pools=None,
                        ace_trainer=None, pair_results=True, value_audit=args.shortlist)
    with redirect_stdout(StringIO()) as output:
        if _bench(request):
            raise RuntimeError(output.getvalue())
    result = json.loads(output.getvalue())
    (checkpoint / "validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result["agents"][0]["win_rate"]


def train(args):
    if args.until_update is None:
        args.until_update = args.total_updates
    if not 1 <= args.until_update <= args.total_updates or min(args.pairs, args.workers, args.validation_pairs, args.eval_every) < 1:
        raise ValueError("Use positive PPO run sizes and an end update within the planned run")
    if not .10 <= args.history_end < 1:
        raise ValueError("Use a final historical opponent fraction between 0.10 and 1")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    cards = json.loads(args.cards.read_text(encoding="utf-8"))
    config = _config(args, cards)
    samples = json.loads((args.bc_init / "test_samples.json").read_text(encoding="utf-8"))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PPO requested CUDA but no CUDA device is available")
    model = ActorCriticNetwork(**config["network"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    if args.resume:
        previous = json.loads((args.resume / "config.json").read_text(encoding="utf-8"))
        if json.loads(json.dumps(config)) != previous:
            raise ValueError("PPO resume requires the original data, schedule and initializer")
        model.load_state_dict(torch.load(args.resume / "state_dict.pt", weights_only=True, map_location="cpu"))
        resume = torch.load(args.resume / "training_state.pt", weights_only=True, map_location="cpu")
        optimizer.load_state_dict(resume["optimizer"])
        history = resume["history"]
        first = resume["update"] + 1
        if first > args.until_update:
            raise ValueError("PPO resume must advance at least one update")
        print(json.dumps({"resumed_update": first - 1, "optimizer_entries": len(optimizer.state)}), flush=True)
    else:
        actor = ActionRankingNetwork(**config["network"])
        actor.load_state_dict(torch.load(args.bc_init / "state_dict.pt", weights_only=True, map_location="cpu"))
        missing, unexpected = model.load_state_dict(actor.state_dict(), strict=False)
        if unexpected or set(missing) != {f"value_head.{key}" for key in model.value_head.state_dict()}:
            raise ValueError("BC actor parameters did not exactly match the PPO policy")
        torch.nn.init.zeros_(model.value_head[-1].weight)
        torch.nn.init.zeros_(model.value_head[-1].bias)
        history = [_cpu_weights(model)]
        first = 1
        _save(args.output, 0, config, model, optimizer, history, {"update": 0}, samples)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn"), initializer=_initialise_worker) as executor:
        for update in range(first, args.until_update + 1):
            started = perf_counter()
            chunks = collect(executor, config, cards, model, history[:-1] or history, args.ace_trainer, update, args.workers)
            elapsed = perf_counter() - started
            counts, outcomes, candidates = Counter(), Counter(), Counter()
            by_opponent = {}
            for chunk in chunks:
                counts.update(chunk["opponents"])
                outcomes.update(chunk["outcomes"])
                candidates.update(chunk["candidates"])
                for opponent, results in chunk["by_opponent"].items():
                    by_opponent.setdefault(opponent, Counter()).update(results)
            metrics = optimise(model, optimizer, device, chunks, args.seed + update)
            history = [*history[-3:], _cpu_weights(model)]
            metrics.update(update=update, device=str(device), workers=args.workers, opponents=dict(counts), outcomes=dict(outcomes),
                           candidate_counts=dict(sorted(candidates.items())), outcomes_by_opponent=by_opponent,
                           games=sum(counts.values()), rollout_seconds=round(elapsed, 3),
                           games_per_second=round(sum(counts.values()) / elapsed, 2),
                           rollout_decisions_per_second=round(metrics["decisions"] / elapsed, 2),
                           parallel_simulator_seconds=round(sum(chunk["simulator_seconds"] for chunk in chunks), 3))
            path = _save(args.output, update, config, model, optimizer, history, metrics, samples)
            if update % args.eval_every == 0 or update == args.total_updates:
                metrics["validation_win_rate"] = evaluate_validation(path, args)
            print(json.dumps(metrics), flush=True)
    return 0


def verify(args):
    from .bc import _verify_samples

    torch.set_num_threads(1)
    policy = BattlePolicy.load(args.checkpoint, args.device)
    if not isinstance(policy.model, ActorCriticNetwork):
        raise ValueError("PPO verification requires an actor-critic checkpoint")
    samples = json.loads((args.checkpoint / "test_samples.json").read_text(encoding="utf-8"))
    report = _verify_samples(policy, samples)
    errors = []
    with torch.inference_mode():
        for datum in samples:
            history = BattleHistory.from_snapshot(datum["history"])
            features = policy.encode(Decision.from_snapshot(datum["decision"]), history=history)
            state = torch.tensor([features.state], dtype=torch.float32, device=policy.device)
            actions = torch.tensor([features.actions], dtype=torch.float32, device=policy.device)
            _, value = policy.model.policy_value(state, actions)
            errors.append(abs(value.item() - datum["value"]))
    if max(errors) > 2e-4:
        raise ValueError("PPO checkpoint value predictions differed from the portable reference")
    print(json.dumps(report | {"max_abs_value_error": max(errors)}, indent=2))
    return 0
