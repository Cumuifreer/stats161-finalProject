"""Learn a joint-action distribution from complete, fixed-deployment local replays."""

import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from random import Random
from time import perf_counter

import torch
from torch.nn import functional as F

from .draft_last_pick import other
from .draft_leaf import CATALOG
from .draft_league import parallel, read, save
from .features import BattleEncoder
from .game import BATTLE, DECISION_LIMIT, PREVIEW, ident
from .joint import BattleHistory, JointPolicy
from .model import ActionRankingNetwork
from .sim import Showdown, fingerprint, play_battle
from .strength import EXTERNAL, ExternalPivot
from .tactics import load_card_profiles, tactical_actions


def encode(encoder, decision, history, switch_turns=None):
    batch = encoder.encode(decision, history=history)
    ranks = {id(action): index for index, action in enumerate(tactical_actions(decision, card_profiles=encoder.profiles))}
    switch_turns = switch_turns or {}
    current = decision.observation["turn"]
    ages = tuple(min(5, max(0, current - switch_turns.get(mon["species"], current))) / 5 if mon else 0.
                 for mon in decision.observation["active"] + decision.observation["opponents"])
    moves = tuple((value + (1. / (1 + ranks[id(action)]), ranks[id(action)] / max(1, len(ranks) - 1)))
                  for value, action in zip(batch.actions, decision.actions, strict=True))
    return batch.state + ages, moves


def _ages(raw):
    turns = {}
    turn = 0
    for line in raw:
        fields = line.split("|")
        if len(fields) < 3:
            continue
        if fields[1] == "turn":
            turn = int(fields[2])
        elif fields[1] in {"switch", "drag"}:
            turns[ident(fields[3].split(",", 1)[0])] = turn
    return turns


def sample(roots, source, cards, checkout):
    profiles = load_card_profiles(cards, with_sets=True)
    encoder = BattleEncoder(profiles)
    by_id = {card["card_id"]: card for card in cards}
    rows = {}
    with ExitStack() as stack:
        engine = stack.enter_context(Showdown())
        external = {rival: stack.enter_context(ExternalPivot(checkout, rival, cards)) for rival in EXTERNAL}
        for root in roots:
            data = rows[root] = []
            for rival, foe in external.items():
                for game in source[str(root)][rival]:
                    actor = game["actor"]
                    enemy = other(actor)
                    control = JointPolicy(profiles)
                    history = BattleHistory()
                    roster = {seat: [by_id[card] for card in ids] for seat, ids in game["roster"].items()}
                    foe.setup(root, enemy, game["first"])

                    def choose(seat, decision, view):
                        if decision.phase == PREVIEW:
                            (control if seat == actor else history).observe(decision)
                            return decision.validate(game["lineups"][seat])
                        if seat == actor:
                            return control(decision)
                        history.observe(decision)
                        decision = replace(decision, seconds_left=DECISION_LIMIT[BATTLE], bank_seconds_left=420)
                        action = foe.choose(decision, view)
                        if len(decision.actions) > 1:
                            state, features = encode(encoder, decision, history, _ages(view["pivot"]["protocol_log"]))
                            data.append((rival, state, features, decision.actions.index(action)))
                        history.record(decision, action)
                        return action

                    play_battle(engine, roster, {actor: control, enemy: foe}, root, game["battle_seed"], choose)
    return rows


def tensors(rows, device):
    state = torch.tensor([row[1] for row in rows], dtype=torch.float32)
    mask = torch.arange(max(len(row[2]) for row in rows))[None] < torch.tensor([len(row[2]) for row in rows])[:, None]
    actions = torch.zeros((*mask.shape, len(rows[0][2][0])), dtype=torch.float32)
    actions[mask] = torch.tensor([action for row in rows for action in row[2]], dtype=torch.float32)
    target = torch.tensor([row[3] for row in rows], dtype=torch.long)
    return [tensor.to(device) for tensor in (state, actions, mask, target)]


def metrics(logits, legal, target):
    logits = logits.masked_fill(~legal, -torch.inf)
    order = logits.argsort(dim=1, descending=True)
    return dict(nll=float(F.cross_entropy(logits, target)),
                top1=float(order[:, 0].eq(target).float().mean()),
                top3=float(order[:, :3].eq(target[:, None]).any(dim=1).float().mean()),
                top8=float(order[:, :8].eq(target[:, None]).any(dim=1).float().mean()))


class LearnedReplies:
    def __init__(self, checkpoint, profiles):
        config = read(checkpoint / "config.json")
        self.encoder = BattleEncoder(profiles)
        self.model = ActionRankingNetwork(**config["network"]).eval()
        self.model.load_state_dict(torch.load(checkpoint / "state_dict.pt", map_location="cpu", weights_only=True))
        self.temperature = config["temperature"]

    @torch.inference_mode()
    def probabilities(self, decision, history, events):
        state, actions = encode(self.encoder, decision, history, _ages(events))
        value = self.model(torch.tensor([state]), torch.tensor([actions]))[0]
        return (value / self.temperature).softmax(dim=0).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--ace-trainer", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or min(args.workers, args.epochs) < 1:
        parser.error("Provide a fresh output and positive counts")
    torch.set_num_threads(1)
    torch.manual_seed(29181)
    source = read(args.source)
    roots = sorted(map(int, source["records"]))
    Random(29181).shuffle(roots)
    validation_roots = set(roots[::5])
    cards = read(CATALOG)
    began = perf_counter()
    rows = {}
    for part in parallel(sample, roots, args.workers, source["records"], cards, args.ace_trainer):
        rows.update(part)
        print(f"Encoded {len(rows)}/{len(roots)} paired pools", flush=True)
    data_seconds = perf_counter() - began
    train = [row for root in roots if root not in validation_roots for row in rows[root]]
    validation = [row for root in roots if root in validation_roots for row in rows[root]]
    device = torch.device(args.device)
    states, actions, legal, target = tensors(train, device)
    v_states, v_actions, v_legal, v_target = tensors(validation, device)
    config = dict(state_size=states.shape[-1], action_size=actions.shape[-1], hidden_size=96)
    model = ActionRankingNetwork(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.025)
    best, fitted, curve = float("inf"), None, []
    began = perf_counter()
    for _ in range(args.epochs):
        model.train()
        for batch in torch.randperm(len(train), device=device).split(64):
            prediction = model(states[batch], actions[batch]).masked_fill(~legal[batch], -torch.inf)
            loss = F.cross_entropy(prediction, target[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            model.eval()
            metric = metrics(model(v_states, v_actions), v_legal, v_target)
            curve.append(metric)
            if metric["nll"] < best:
                best = metric["nll"]
                fitted = {key: tensor.detach().to("cpu", copy=True) for key, tensor in model.state_dict().items()}
    model.load_state_dict(fitted)
    with torch.no_grad():
        prediction = model(v_states, v_actions)
        temperature = min((.8, 1., 1.2, 1.5, 2.),
                          key=lambda scale: metrics(prediction / scale, v_legal, v_target)["nll"])
        report = metrics(prediction / temperature, v_legal, v_target)
        report["tactical"] = metrics(-12 * v_actions[:, :, 2], v_legal, v_target)
        report["calibrated_tactical"] = metrics(-12 * v_actions[:, :, -1], v_legal, v_target)
    fit_seconds = perf_counter() - began
    args.output.mkdir(parents=True)
    torch.save(fitted, args.output / "state_dict.pt")
    save(args.output / "config.json", dict(network=config, temperature=temperature, source=str(args.source),
                                           pools={str(root): source["pools"][str(root)] for root in roots},
                                           validation_roots=sorted(validation_roots), card_fingerprint=fingerprint(cards),
                                           device=str(device), training_rows=dict(Counter(row[0] for row in train)),
                                           validation_rows=dict(Counter(row[0] for row in validation)), report=report,
                                           data_seconds=data_seconds, fit_seconds=fit_seconds, epochs=curve,
                                           parameters=sum(parameter.numel() for parameter in model.parameters()), api_spend_usd=0))
    print(report, flush=True)


if __name__ == "__main__":
    main()
