"""Learn a joint tactical prior from public, pre-Tera VGC replay decisions."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from ..draft_league import read, save
from ..game import ident
from ..reply_learning import _ages


KINDS = ("strike", "spread", "protect", "switch", "setup", "disrupt", "support")
SPREAD = {"alladjacent", "alladjacentfoes", "all"}
PROTECT = {"protect", "detect", "wideguard", "quickguard", "kingsshield", "banefulbunker", "spikyshield", "silktrap"}
SETUP = {"tailwind", "trickroom", "raindance", "sunnyday", "snowscape", "sandstorm", "reflect", "lightscreen",
         "auroraveil", "swordsdance", "nastyplot", "calmmind", "dragondance", "bulkup", "irondefense", "curse"}
DISRUPT = {"fakeout", "icywind", "electroweb", "snarl", "spore", "sleeppowder", "thunderwave", "taunt",
           "encore", "yawn", "willowisp", "haze", "clearsmog", "skillswap"}


def dex():
    query = (
        "const {Dex}=require('pokemon-showdown');"
        "const mons=Object.fromEntries(Dex.species.all().filter(x=>x.exists).map(x=>[x.id,x.baseStats.spe]));"
        "const moves=Object.fromEntries(Dex.moves.all().filter(x=>x.exists).map(x=>[x.id,[x.category,x.target]]));"
        "process.stdout.write(JSON.stringify({mons,moves}));"
    )
    observed = json.loads(subprocess.check_output(["node", "-e", query], text=True, encoding="utf-8"))
    categories = {}
    for move, (category, target) in observed["moves"].items():
        if move in PROTECT:
            tag = "protect"
        elif move in SETUP:
            tag = "setup"
        elif move in DISRUPT:
            tag = "disrupt"
        elif category == "Status":
            tag = "support"
        elif ident(target) in SPREAD:
            tag = "spread"
        else:
            tag = "strike"
        categories[move] = KINDS.index(tag)
    return observed["mons"], categories


def fraction(text):
    amount = text.split(" ", 1)[0]
    if amount == "0":
        return 0.
    if "/" in amount:
        value, total = amount.split("/", 1)
        return float(value) / float(total)
    return float(amount.strip("%")) / 100.


def representation(turn, active, own, foe, wind, room, speeds):
    def mon(slot):
        species, hp, status, switched = active.get(slot, ("", 0., "", turn))
        return [hp, speeds.get(species, 80) / 200., min(4, max(0, turn - switched)) / 4.,
                *[float(status == state) for state in ("brn", "par", "slp", "frz", "psn", "tox")]]

    return [min(turn, 16) / 16., float(own in wind), float(foe in wind), float(room),
            *(value for p in (own, foe) for slot in "ab" for value in mon(p+slot))]


def examples(log, speeds, moves):
    active = {}
    actions = {}
    players = ("p1", "p2")
    frames = {}
    winds = set()
    turn = 0
    room = started = False

    def labels():
        for seat, row in frames.items():
            choices = [actions.get(seat+slot) for slot in "ab"]
            if all(choice is not None for choice in choices):
                yield row, choices[0] * len(KINDS) + choices[1]

    for line in log.splitlines():
        fields = line.split("|")
        if len(fields) < 2:
            continue
        tag = fields[1]
        who = fields[2].split(":", 1)[0] if len(fields) > 2 else ""
        if tag == "-terastallize":
            return
        if tag == "turn":
            yield from labels()
            turn = int(fields[2])
            actions = {}
            started = False
            frames = {
                own: representation(turn, active, own, foe, winds, room, speeds)
                for own, foe in (players, players[::-1]) if all(active.get(own+slot, ("",0.))[1] > 0 for slot in "ab")
            }
        elif tag in {"switch", "drag", "replace"} and len(fields) >= 5 and who in ("p1a", "p1b", "p2a", "p2b"):
            species = ident(fields[3].split(",", 1)[0])
            condition = fields[4].split()
            if tag == "switch" and turn and not started and who[:2] in frames:
                actions.setdefault(who, KINDS.index("switch"))
            active[who] = species, fraction(fields[4]), condition[1] if len(condition) > 1 else "", turn
        elif tag == "move" and len(fields) > 3:
            started = True
            if who in active and not any(part.startswith("[from]") for part in fields[5:]):
                if (kind := moves.get(ident(fields[3]))) is not None:
                    actions.setdefault(who, kind)
        elif tag in {"cant", "faint"}:
            started = True
            if tag == "faint" and who in active:
                species, _, status, since = active[who]
                active[who] = species, 0., status, since
        elif tag in {"-damage", "-heal"} and len(fields) > 3 and who in active:
            species, _, status, since = active[who]
            active[who] = species, fraction(fields[3]), status, since
        elif tag in {"-status", "-curestatus"} and len(fields) > 3 and who in active:
            species, hp, _, since = active[who]
            active[who] = species, hp, fields[3] if tag == "-status" else "", since
        elif tag in {"-sidestart", "-sideend"} and len(fields) > 3 and "tailwind" in fields[3].lower():
            (winds.add if tag == "-sidestart" else winds.discard)(who)
        elif tag in {"-fieldstart", "-fieldend"} and len(fields) > 2 and "trick room" in fields[2].lower():
            room = tag == "-fieldstart"
    yield from labels()


def network(size):
    return nn.Sequential(nn.Linear(size, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, len(KINDS) ** 2))


def scores(logits, target):
    rank = logits.argsort(dim=-1, descending=True)
    return {"nll": float(F.cross_entropy(logits, target)),
            "top1": float(rank[:, 0].eq(target).float().mean()),
            "top3": float(rank[:, :3].eq(target[:, None]).any(1).float().mean())}


class HumanPrior:
    def __init__(self, checkpoint):
        config = read(checkpoint / "config.json")
        self.speeds = config["speeds"]
        self.moves = config["moves"]
        self.temperature = config["temperature"]
        self.net = network(config["input_size"]).eval()
        self.net.load_state_dict(torch.load(checkpoint / "state_dict.pt", weights_only=True, map_location="cpu"))

    @torch.inference_mode()
    def probabilities(self, decision, events):
        board = decision.observation
        turn = board["turn"]
        ages = _ages(events)
        active = {}
        for seat, group in (("p1", board["active"]), ("p2", board["opponents"])):
            for slot, mon in zip("ab", group, strict=True):
                if mon:
                    name = mon["species"]
                    active[seat+slot] = name, mon.get("hp", 1.), mon.get("status", ""), ages.get(name, turn)
        if len(active) != 4:
            return []
        wind = {seat for seat, key in (("p1","side_conditions"), ("p2","opponent_side_conditions"))
                if "tailwind" in list(map(ident, board[key]))}
        room = "trickroom" in list(map(ident, board["fields"]))
        values = self.net(torch.tensor([representation(turn, active, "p1", "p2", wind, room, self.speeds)]))[0]
        grouped = {}
        for action in decision.actions:
            kinds = []
            for slot in ("slot_0", "slot_1"):
                atom = action[slot]
                kind = KINDS.index("switch") if atom["type"] == "switch" else self.moves.get(atom.get("move_id"))
                kinds.append(kind)
            if None not in kinds:
                pair = kinds[0] * len(KINDS) + kinds[1]
                grouped.setdefault(pair, []).append(action)
        if not grouped:
            return []
        groups = list(grouped)
        weights = (values[groups] / self.temperature).softmax(dim=0).tolist()
        return [(action, weight / len(grouped[key])) for key, weight in zip(groups, weights, strict=True)
                for action in grouped[key]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Give a fresh output directory")
    since = perf_counter()
    speeds, moves = dex()
    source = read(args.replays)
    train, held = [], []
    for battle, (_, log) in source.items():
        try:
            rows = list(examples(log, speeds, moves))
        except (TypeError, ValueError, IndexError, ZeroDivisionError):
            continue
        bucket = int(sha256(battle.encode()).hexdigest()[:8], 16) % 5
        (held if bucket == 0 else train).extend(rows)
    prep = perf_counter() - since
    torch.set_num_threads(1)
    torch.manual_seed(28457)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, y = (torch.tensor(list(column), device=device) for column in zip(*train, strict=True))
    v, t = (torch.tensor(list(column), device=device) for column in zip(*held, strict=True))
    model = network(x.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0011, weight_decay=.02)
    best, checkpoint = float("inf"), {}
    began = perf_counter()
    for _ in range(70):
        for batch in torch.randperm(len(train), device=device).split(256):
            loss = F.cross_entropy(model(x[batch]), y[batch])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        with torch.inference_mode():
            loss = float(F.cross_entropy(model(v), t))
        if loss < best:
            best = loss
            checkpoint = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(checkpoint)
    counts = Counter(y.tolist())
    empirical = torch.tensor([counts[i] + 1 for i in range(len(KINDS) ** 2)], device=device)
    with torch.inference_mode():
        predicted = model(v)
        temperature = min((.8,1.,1.15,1.30,1.6), key=lambda value: scores(predicted / value, t)["nll"])
        baseline = scores(empirical.log().expand(len(held), -1), t)
        final = scores(predicted / temperature, t)
    args.output.mkdir(parents=True)
    torch.save(checkpoint, args.output / "state_dict.pt")
    save(args.output / "config.json", {
        "input_size": x.shape[-1], "temperature": temperature, "speeds": speeds, "moves": moves,
        "training_examples": len(train), "validation_examples": len(held), "replays": len(source),
        "sha256": sha256(args.replays.read_bytes()).hexdigest(), "fit_seconds": perf_counter() - began,
        "preparation_seconds": prep, "baseline": baseline, "learned": final, "api_spend_usd": 0, "device": device,
    })
    print(len(train), len(held), baseline, final, flush=True)


if __name__ == "__main__":
    main()
