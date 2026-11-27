"""Distill public Quicksilver actions from development replays into a reply prior."""

import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from ..draft_leaf import CATALOG
from ..draft_league import read, save
from ..features import BattleEncoder
from ..game import PREVIEW
from ..joint import BattleHistory
from ..lab85 import MODELS, QUICKSILVER
from ..reply_learning import LearnedReplies, _ages, encode, metrics, tensors
from ..sim import SEATS, Showdown, fingerprint, play_battle
from ..tactics import load_card_profiles


class PublicTrace:
    needs_proxy_evidence = True
    choose_with_public_history = None


def samples(cards, sources):
    lookup = {card["card_id"]: card for card in cards}
    profiles = load_card_profiles(cards, with_sets=True)
    encoder = BattleEncoder(profiles)
    trace = PublicTrace()
    seen = set()
    with Showdown() as engine:
        for source in sources:
            if "blind" in str(source).casefold() or (source.parent / "acceptance.json").exists():
                raise ValueError("Use completed development matches only")
            data = read(source)
            for root, opponents in data["records"].items():
                pool = data["pools"][root]
                for game in opponents.get(QUICKSILVER, ()):
                    if not game["trace"]:
                        continue
                    seat = "p2" if game["actor"] == "p1" else "p1"
                    actions = {player: iter([row["action"] for row in game["trace"] if row["seat"] == player])
                               for player in SEATS}
                    history = BattleHistory()
                    rosters = {player: [lookup[card] for card in ids] for player, ids in game["roster"].items()}
                    labels = []

                    def choose(player, decision, view):
                        if decision.phase == PREVIEW:
                            if player == seat:
                                history.observe(decision)
                            return decision.validate(game["lineups"][player])
                        action = decision.validate(next(actions[player]))
                        if player == seat:
                            history.observe(decision)
                            if len(decision.actions) > 1:
                                state, moves = encode(encoder, decision, history, _ages(view["search"]["protocol_log"]))
                                key = fingerprint([pool, state, moves, action])
                                if key not in seen:
                                    seen.add(key)
                                    labels.append((seat, state, moves, decision.actions.index(action)))
                            history.record(decision, action)
                        return action

                    policies = {player: trace if player == seat else None for player in SEATS}
                    result = play_battle(engine, rosters, policies, int(root), game["battle_seed"], choose)
                    if result != game["winner"]:
                        raise ValueError("A development replay did not reach its recorded outcome")
                    for label in labels:
                        yield pool, label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Give a fresh checkpoint path")
    started = perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(91427)
    cards = read(CATALOG)
    corpus = list(samples(cards, args.development))
    fit, held = [], []
    for pool, row in corpus:
        (held if int(pool[:8], 16) % 4 == 0 else fit).append(row)
    data_seconds = perf_counter() - started
    if min(len(fit), len(held)) < 30:
        raise ValueError("More labeled public actions are needed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    profiles = load_card_profiles(cards, with_sets=True)
    prior = LearnedReplies(MODELS / "reply", profiles)
    model = prior.model.to(device)
    config = read(MODELS / "reply" / "config.json")["network"]
    states, actions, legal, target = tensors(fit, device)
    v_states, v_actions, v_legal, v_target = tensors(held, device)
    original = {name: value.detach().clone() for name, value in model.state_dict().items()}
    with torch.inference_mode():
        baseline = metrics(model(v_states, v_actions) / prior.temperature, v_legal, v_target)
    opt = torch.optim.AdamW(model.parameters(), lr=.0008, weight_decay=.018)
    began = perf_counter()
    best, weights = float("inf"), {}
    for _ in range(56):
        model.train()
        for batch in torch.randperm(len(fit), device=device).split(64):
            logits = model(states[batch], actions[batch]).masked_fill(~legal[batch], -torch.inf)
            pull = sum((parameter - original[name]).square().mean() for name, parameter in model.named_parameters())
            loss = F.cross_entropy(logits, target[batch]) + .07 * pull
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.inference_mode():
            nll = metrics(model(v_states, v_actions), v_legal, v_target)["nll"]
        if nll < best:
            best = nll
            weights = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(weights)
    with torch.inference_mode():
        values = model(v_states, v_actions)
        temperature = min((.8, 1., 1.25, 1.5, 2.), key=lambda x: metrics(values / x, v_legal, v_target)["nll"])
        result = metrics(values / temperature, v_legal, v_target)
    pools = {root: pool for source in args.development for root, pool in read(source)["pools"].items()}
    pools.update(read(MODELS / "reply" / "config.json")["pools"])
    args.output.mkdir(parents=True)
    torch.save(weights, args.output / "state_dict.pt")
    save(args.output / "config.json", {
        "network": config, "temperature": temperature, "pools": pools, "catalog": fingerprint(cards),
        "counts": dict(Counter(pool for pool, _ in corpus)), "training_rows": len(fit), "validation_rows": len(held),
        "files": {str(path): sha256(path.read_bytes()).hexdigest() for path in args.development},
        "device": str(device), "report": {"original": baseline, "distilled": result},
        "data_seconds": data_seconds, "fit_seconds": perf_counter() - began, "api_spend_usd": 0,
    })
    print(len(fit), len(held), baseline, result, flush=True)


if __name__ == "__main__":
    main()
