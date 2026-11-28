"""Learn Battle continuation and advantage-weighted decisions from full replay trajectories."""

import argparse
from copy import deepcopy
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from ..battle_league import configure_torch, load_model
from ..draft_leaf import CATALOG
from ..draft_league import read, save
from ..sim import fingerprint
from . import qualify, train


def successor(episodes):
    index = np.arange(len(episodes), dtype=np.int64)
    continues = np.zeros(len(episodes), dtype=np.bool_)
    continues[:-1] = episodes[1:] == episodes[:-1]
    return index + continues, continues


def update(model, target, replay, index, next_index, continues):
    *features, teacher, behavior, result, branches, propensity = [field[index] for field in replay]
    logits, q, value = model(*features)
    observed = q.gather(1, behavior[:, None]).squeeze(1)
    with torch.no_grad():
        _, _, next_value = target(*(field[next_index[index]] for field in replay[:len(features)]))
        continuation = .5 + .99 * (next_value.float().sigmoid() - .5)
        continuation = torch.where(continues[index], continuation, result)
        target_return = .78 * continuation + .22 * result
        quality = observed.detach().float().sigmoid()
    difference = quality - value.float().sigmoid()
    emphasis = torch.where(difference > 0., .68, .32)
    state_loss = (emphasis * difference.square()).mean()
    temporal = F.binary_cross_entropy_with_logits(observed, target_return)
    advantage = (difference.detach() * 4.).exp().clamp(.12, 5.)
    policy_loss = (advantage * F.cross_entropy(logits, behavior, reduction="none")).mean()
    label = branches.clamp_min(0)
    valid = (branches >= 0) & features[4]
    pair_loss = F.binary_cross_entropy_with_logits(q, label, reduction="none")
    pair_loss = (pair_loss * valid).sum() / valid.sum().clamp_min(1)
    return (temporal + .9 * state_loss + .55 * policy_loss + .23 * F.cross_entropy(logits, teacher) + .45 * pair_loss
            + .18 * F.binary_cross_entropy_with_logits(value, result))


def fit(arguments):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--branches", type=Path, action="append", default=[])
    parser.add_argument("--initialize", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args(arguments)
    if args.output.exists() or args.epochs < 1:
        parser.error("Use a fresh output path and positive training duration")
    started = perf_counter()
    configure_torch()
    torch.manual_seed(71931)
    torch.set_float32_matmul_precision("high")
    cards = read(CATALOG)
    device = torch.device("cuda")
    (training, validation), sources, details = train.dataset(args.data, args.branches, fingerprint(cards))
    model = load_model(args.initialize, cards).to(device)
    target = deepcopy(model).eval()
    initial_model = read(args.initialize / "config.json")
    next_index, continues = [torch.from_numpy(array).to(device) for array in successor(training["root"])]
    replay, held = [train.prepare(rows, device, model.history_steps) for rows in (training, validation)]
    del training, validation
    save(args.output / "config.json", initial_model)
    save(args.output / "replay.json", dict(sources=sources, populations=details, transitions=len(next_index),
                                         episodes=int((~continues).sum()), parent=str(args.initialize)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.00011, weight_decay=.035, fused=True)
    initial = train.assess(model, held, 512)
    began = perf_counter()
    checkpoints = []
    best, best_epoch, steps = float("inf"), 0, 0
    policy_weights, ema_weights = list(model.parameters()), list(target.parameters())
    for epoch in range(1, args.epochs + 1):
        model.train()
        for index in train.batches(replay, 512, 20, device):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                objective = update(model, target, replay, index, next_index, continues)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(policy_weights, 2.)
            optimizer.step()
            with torch.no_grad():
                torch._foreach_lerp_(ema_weights, policy_weights, .015)
            steps += 1
        model.eval()
        result = train.assess(model, held, 512)
        checkpoints.append(result)
        measurement = (result["q_nll"] + .40 * result["value_nll"] + .35 * result["paired_delta_mae"]
                       - .25 * result["paired_label_gain"] - .12 * result["policy_teacher_agreement"])
        if measurement < best:
            best, best_epoch = measurement, epoch
            torch.save(model.state_dict(), args.output / "state_dict.pt")
        print(dict(epoch=epoch, steps=steps, q_nll=result["q_nll"], value_nll=result["value_nll"],
                   paired_label_gain=result["paired_label_gain"], policy_teacher_agreement=result["policy_teacher_agreement"]), flush=True)
    save(args.output / "training.json", dict(initial=initial, epochs=checkpoints, best_epoch=best_epoch,
                                            steps=steps, parameters=sum(p.numel() for p in model.parameters()),
                                            fit_seconds=perf_counter() - began, wall_seconds=perf_counter() - started,
                                            gpu=torch.cuda.get_device_name()))


class TemporalPolicy(qualify.Learner):
    @torch.inference_mode()
    def index(self, features):
        state, actions, cards, mons, legal = features
        missing = self.model.history_steps - len(self.trace)
        trace = [(0.,) * (len(state) + len(actions[0]))] * missing + self.trace
        logits, q, _ = self.model(torch.tensor([state]), torch.tensor([actions]), torch.tensor([cards]),
                                  torch.tensor([mons]), torch.tensor([legal]), torch.tensor([trace]))
        return int((logits + 1.5 * q.sigmoid())[0].argmax())


def evaluate(*args):
    with patch.object(qualify, "Learner", TemporalPolicy):
        return EVALUATE(*args)


EVALUATE = qualify.evaluate


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "qualification":
        del sys.argv[1]
        with patch.object(qualify, "evaluate", evaluate):
            qualify.main()
    else:
        fit(sys.argv[1:])
