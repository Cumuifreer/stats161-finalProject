"""Collect Joint demonstrations, train the battle ranker, and verify its checkpoint."""

from contextlib import ExitStack
from dataclasses import dataclass
import json
from pathlib import Path
from random import Random
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from .features import FEATURE_VERSION, HISTORY_FEATURE_VERSION, BattleEncoder
from .game import BATTLE, DRAFT, DRAFT_SIZE, PREVIEW, Decision
from .joint import BattleHistory, JointPolicy
from .model import ActionRankingNetwork, BattlePolicy
from .pivot import PivotOpponent
from .policy import heuristic
from .sim import FORMAT, SEATS, Showdown, fingerprint, play_game, pool_fingerprint
from .tactics import load_card_profiles, tactical, tactical_candidates

DISTILL_TEMPERATURE = .25


def _split(cards, train_seeds, test_seeds):
    if set(train_seeds) & set(test_seeds):
        raise ValueError("BC training and holdout seeds overlap")
    split = {name: {seed: pool_fingerprint(cards, seed) for seed in seeds}
             for name, seeds in (("train", train_seeds), ("holdout", test_seeds))}
    if set(split["train"].values()) & set(split["holdout"].values()):
        raise ValueError("BC training and holdout draft pools overlap (independent of card order)")
    return split


@dataclass(frozen=True)
class Example:
    seed: int
    seat: str
    decision: Decision
    target: int
    tactical_target: int
    history: dict
    teacher_scores: dict[int, float]


class Teacher(JointPolicy):
    def __init__(self, profiles, seed, seat):
        super().__init__(profiles)
        self.seed = seed
        self.seat = seat
        self.examples = []

    def __call__(self, decision):
        if decision.phase != BATTLE:
            return super().__call__(decision)
        candidates = tactical_candidates(decision)
        ranked = self.rank(decision, candidates)
        choice = ranked[0][0]
        self.examples.append(Example(
            self.seed, self.seat, decision, decision.actions.index(choice), decision.actions.index(candidates[0]),
            self.history.snapshot(), {decision.actions.index(action): details.get("score", 0.) for action, details in ranked},
        ))
        self.record(decision, choice)
        return choice


def _collect(engine, cards, profiles, opponents, seeds):
    examples = []
    results = {name: dict(wins=0, losses=0, draws=0) for name in opponents}
    names = list(opponents)
    started = perf_counter()
    for index, seed in enumerate(seeds):
        name = names[index % len(names)]
        for seat, other in (SEATS, SEATS[::-1]):
            teacher = Teacher(profiles, seed, seat)
            policies = {seat: teacher, other: opponents[name]()}
            samples = {player: [] for player in SEATS}
            winner = play_game(engine, cards, policies, seed, samples,
                               fixed_policies={DRAFT: heuristic, PREVIEW: heuristic})
            if any(event for decisions in samples.values() for _, _, event in decisions):
                raise RuntimeError(f"BC demonstration game {seed}/{seat} had an automatic or invalid action")
            outcome = "draws" if winner is None else "wins" if winner == seat else "losses"
            results[name][outcome] += 1
            examples.extend(teacher.examples)
    seconds = perf_counter() - started
    return examples, {
        "games": 2 * len(seeds), "games_per_second": round(2 * len(seeds) / seconds, 2),
        "teacher_outcomes": results, "forced_decisions": sum(len(row.decision.actions) == 1 for row in examples),
    }


def _encode(examples, encoder, *, shortlist_only=False, soft=False):
    rows = []
    for row in examples:
        history = BattleHistory.from_snapshot(row.history)
        features = encoder.encode(row.decision, history=history, shortlist_only=shortlist_only)
        distribution = None
        if soft:
            teacher = torch.tensor([row.teacher_scores[index] for index in features.action_indices], dtype=torch.float32)
            distribution = (teacher / DISTILL_TEMPERATURE).softmax(dim=0)
        rows.append((torch.tensor(features.state, dtype=torch.float32),
                     torch.tensor(features.actions, dtype=torch.float32), features.action_indices.index(row.target), distribution))
    return rows


def _batches(rows, order, batch_size, device):
    for start in range(0, len(order), batch_size):
        selected = [rows[i] for i in order[start:start + batch_size]]
        states, actions, targets, distributions = zip(*selected, strict=True)
        padded = pad_sequence(actions, batch_first=True)
        valid = torch.arange(padded.size(1))[None, :] < torch.tensor([len(row) for row in actions])[:, None]
        soft = pad_sequence(distributions, batch_first=True).to(device) if distributions[0] is not None else None
        yield torch.stack(states).to(device), padded.to(device), valid.to(device), torch.tensor(targets, device=device), soft


def _loss(logits, valid, targets, soft):
    scores = logits.masked_fill(~valid, -torch.inf)
    if soft is None:
        return F.cross_entropy(scores, targets)
    log_probs = scores.log_softmax(dim=1).masked_fill(~valid, 0.)
    return -(soft * log_probs).sum(dim=1).mean()


@torch.inference_mode()
def _imitation(model, rows, batch_size, device):
    counts = torch.zeros(7, dtype=torch.long, device=device)
    model.eval()
    for states, actions, valid, targets, _ in _batches(rows, range(len(rows)), batch_size, device):
        scores = model(states, actions).masked_fill(~valid, -torch.inf)
        prediction = scores.argmax(dim=1)
        best = scores.topk(min(3, scores.size(1)), dim=1).indices
        agrees = actions[..., 1].argmax(dim=1) == targets
        correct = prediction == targets
        counts += torch.stack((
            correct.sum(), (best == targets[:, None]).any(dim=1).sum(), agrees.sum(),
            (~valid.gather(1, prediction[:, None])).sum(), (~agrees & correct).sum(),
            (agrees & ~correct).sum(), (~agrees).sum(),
        ))
    top1, top3, tactical_top1, illegal, recovered, spoiled, disagreement = counts.tolist()
    return {"decisions": len(rows), "top1": top1 / len(rows), "top3": top3 / len(rows),
            "tactical_top1": tactical_top1 / len(rows), "illegal_action_rate": illegal / len(rows),
            "teacher_disagreed_with_tactical": disagreement, "bc_recovered_teacher_correction": recovered,
            "bc_spoiled_tactical_teacher_agreement": spoiled}


def _verification_samples(policy, examples):
    samples = []
    seeds = set()
    forced_sampled = non_forced_sampled = 0
    for example in examples:
        decision = example.decision
        forced = len(decision.actions) == 1
        if example.seed in seeds or (forced and forced_sampled) or (not forced and non_forced_sampled >= 5):
            continue
        indices, scores = policy.evaluate(decision, history=BattleHistory.from_snapshot(example.history))
        best = scores.topk(min(2, scores.numel())).values
        if not forced and best[0] - best[1] < 0.002:
            continue
        action = decision.validate(decision.actions[indices[int(scores.argmax())]])
        samples.append({"seed": example.seed, "seat": example.seat, "decision": decision.snapshot(), "history": example.history,
                        "teacher_action": decision.actions[example.target], "action": action, "scores": scores.tolist()})
        seeds.add(example.seed)
        forced_sampled += int(forced)
        non_forced_sampled += int(not forced)
        if len(samples) == 6:
            break
    if not samples:
        raise RuntimeError("Could not select a portable BC verification sample")
    return samples


def train(args):
    if min(args.train_pairs, args.test_pairs, args.epochs, args.batch_size) < 1:
        raise ValueError("Pair counts, epochs and batch size must be positive")
    if args.output.exists():
        raise FileExistsError(f"BC output already exists: {args.output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    cards = json.loads(args.cards.read_text(encoding="utf-8"))
    if len(cards) < DRAFT_SIZE or len({card["card_id"] for card in cards}) != len(cards):
        raise ValueError(f"BC needs at least {DRAFT_SIZE} cards with unique IDs")
    train_seeds = range(args.seed, args.seed + args.train_pairs)
    test_seeds = range(args.test_seed, args.test_seed + args.test_pairs)
    split = _split(cards, train_seeds, test_seeds)

    variant = args.variant
    shortlist_only = variant != "v1"
    use_history = variant.startswith("history-")
    soft = variant == "history-soft"
    profiles = load_card_profiles(cards, with_sets=True)
    encoder = BattleEncoder(profiles if use_history else None)
    opponents = {"tactical": lambda: tactical, "joint": lambda: JointPolicy(profiles)}
    with ExitStack() as stack:
        if args.ace_trainer:
            pivot = stack.enter_context(PivotOpponent(args.ace_trainer))
            pivot_base = stack.enter_context(PivotOpponent(args.ace_trainer, style=False))
            opponents.update(pivot=lambda: pivot, **{"pivot-base": lambda: pivot_base})
        engine = stack.enter_context(Showdown())
        train_examples, train_metrics = _collect(engine, cards, profiles, opponents, train_seeds)
        test_examples, test_metrics = _collect(engine, cards, profiles, opponents, test_seeds)

    started = perf_counter()
    train_rows = _encode((row for row in train_examples if len(row.decision.actions) > 1), encoder,
                         shortlist_only=shortlist_only, soft=soft)
    test_rows = _encode((row for row in test_examples if len(row.decision.actions) > 1), encoder,
                        shortlist_only=shortlist_only)
    encoding_seconds = perf_counter() - started
    if not train_rows or not test_rows:
        raise ValueError("Each BC split needs at least one non-forced battle decision")
    state, actions, *_ = train_rows[0]
    network = dict(state_size=state.numel(), action_size=actions.size(1), hidden_size=64)
    model = ActionRankingNetwork(**network).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    rng = Random(args.seed)
    order = list(range(len(train_rows)))
    losses = []
    started = perf_counter()
    for _ in range(args.epochs):
        rng.shuffle(order)
        total = torch.zeros((), device=device)
        for states, actions, valid, targets, teacher in _batches(train_rows, order, args.batch_size, device):
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(model(states, actions), valid, targets, teacher)
            loss.backward()
            optimizer.step()
            total += loss.detach() * len(targets)
        losses.append(total.item() / len(train_rows))
    seconds = perf_counter() - started
    if not torch.isfinite(torch.tensor(losses)).all():
        raise RuntimeError("BC training produced a non-finite loss")
    report = {
        "variant": variant, "torch_version": str(torch.__version__), "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train": train_metrics | _imitation(model, train_rows, args.batch_size, device),
        "holdout": test_metrics | _imitation(model, test_rows, args.batch_size, device),
        "encoding_seconds": round(encoding_seconds, 3), "training_seconds": round(seconds, 3),
        "training_decisions_per_second": round(args.epochs * len(train_rows) / seconds, 2), "loss_by_epoch": losses,
    }
    config = {
        "feature_version": HISTORY_FEATURE_VERSION if use_history else FEATURE_VERSION,
        "network": network, "battle_format": FORMAT, "shortlist_only": shortlist_only, "variant": variant,
        **({"card_profiles": profiles} if use_history else {}),
        **({"distill_temperature": DISTILL_TEMPERATURE} if soft else {}),
        "catalog_sha256": fingerprint(cards), "teacher": "joint", "opponents": list(opponents),
        "draft_and_preview": "heuristic (existing battle-only benchmark)", "split": split,
        "training": {"seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": 0.002},
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".bc-", dir=args.output.parent) as temporary:
        bundle = Path(temporary) / "bundle"
        bundle.mkdir()
        weights = bundle / "state_dict.pt"
        torch.save({key: tensor.detach().cpu() for key, tensor in model.state_dict().items()}, weights)
        report["state_dict_bytes"] = weights.stat().st_size
        (bundle / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        reloaded = BattlePolicy.load(bundle)
        tests = _verification_samples(reloaded, test_examples)
        (bundle / "test_samples.json").write_text(json.dumps(tests, separators=(",", ":")) + "\n", encoding="utf-8")
        report["verified_samples"] = _verify_samples(reloaded, tests)["samples"]
        (bundle / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        bundle.rename(args.output.resolve())
    print(json.dumps(report, indent=2))
    return 0


def _verify_samples(policy, tests):
    max_error = 0.
    for sample in tests:
        decision = Decision.from_snapshot(sample["decision"])
        history = BattleHistory.from_snapshot(sample["history"]) if "history" in sample else None
        indices, scores = policy.evaluate(decision, history=history)
        expected = torch.tensor(sample["scores"], dtype=torch.float32)
        torch.testing.assert_close(scores, expected, atol=2e-4, rtol=1e-4)
        action = decision.validate(decision.actions[indices[int(scores.argmax())]])
        if action != sample["action"]:
            raise ValueError(f"BC chose a different action on verification seed {sample['seed']}/{sample['seat']}")
        max_error = max(max_error, (scores - expected).abs().max().item())
    return {"device": str(policy.device), "samples": len(tests), "max_abs_logit_error": max_error}


def verify(args):
    torch.set_num_threads(1)
    policy = BattlePolicy.load(args.checkpoint, args.device)
    tests = json.loads((args.checkpoint / "test_samples.json").read_text(encoding="utf-8"))
    try:
        report = _verify_samples(policy, tests)
    except AssertionError as error:
        print(error, file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0
