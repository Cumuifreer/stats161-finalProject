"""Learn lead-pair selection from VGC open team sheets and test transfer to Draft."""

import argparse
from collections import Counter
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..battle_league import configure_torch
from ..draft85 import deployment
from ..draft_leaf import CATALOG
from ..draft_league import read, save
from ..game import ident
from ..lead85 import LeadPrior
from ..research_search80.human import DISRUPT, PROTECT, SETUP, SPREAD
from ..sim import SEATS
from ..tactics import load_dex, tactical_actions


PAIRS = tuple(combinations(range(6), 2))
ROLE_MOVES = (PROTECT, SETUP, DISRUPT, {"tailwind", "icywind", "electroweb"}, {"trickroom"}, {"fakeout"},
              {"followme", "ragepowder", "helpinghand"}, {"spore", "yawn", "sleeppowder"},
              {"uturn", "voltswitch", "flipturn", "partingshot"}, {"recover", "pollenpuff", "healpulse"},
              {"wideguard", "quickguard"}, {"perishsong", "haze", "clearsmog"})
ROLE_ABILITIES = ({"intimidate"}, {"regenerator"}, {"drought", "drizzle", "sandstream", "snowwarning"},
                  {"prankster", "galewings"}, {"competitive", "defiant"}, {"shadowtag", "arenatrap"},
                  {"grassysurge", "psychicsurge", "electricsurge", "mistysurge"}, {"commander"})
ROLE_ITEMS = ({"choicescarf"}, {"choiceband", "choicespecs"}, {"focussash"}, {"assaultvest"}, {"boosterenergy"},
               {"covertcloak", "safetygoggles"}, {"eviolite", "leftovers", "sitrusberry"}, {"protectivepads", "clearamulet"})


class Schema:
    def __init__(self):
        self.dex = load_dex()
        self.types = sorted(self.dex["types"])
        self.cache = {}

    def encode(self, mon, item, ability, moves):
        key = mon, item, ability, tuple(moves)
        if key in self.cache:
            return self.cache[key]
        species = self.dex["species"][mon]
        typing = species["types"]
        moves = set(moves)
        active = [self.dex["moves"][move] for move in moves if move in self.dex["moves"]]
        fields = [float(kind in typing) for kind in self.types]
        fields.extend(species[stat] / 150 for stat in ("hp", "atk", "def", "spa", "spd", "spe"))
        for kind in self.types:
            fields.append(max((min(160, move["base_power"]) / 150 for move in active if move["type"] == kind), default=0.))
        fields.extend(float(bool(moves & role)) for role in ROLE_MOVES)
        fields.extend(float(ability in role) for role in ROLE_ABILITIES)
        fields.extend(float(item in role) for role in ROLE_ITEMS)
        fields.extend((sum(move["target"] in SPREAD for move in active) / 4, sum(move["priority"] > 0 for move in active) / 4))
        self.cache[key] = fields
        return fields

    def showteam(self, packed):
        team = []
        for raw in packed.split("]"):
            fields = raw.split("|")
            if len(fields) < 5:
                raise ValueError("Incomplete team sheet")
            mon, item, ability = ident(fields[1] or fields[0]), ident(fields[2]), ident(fields[3])
            moves = [ident(move) for move in fields[4].split(",")]
            team.append((mon, self.encode(mon, item, ability, moves)))
        return team

    def local(self, team):
        return [self.encode(ident(card["species"]), ident(card["item"]), ident(card["ability"]),
                            [ident(move) for move in card["moves"]]) for card in team]


def logs(path, schema):
    features, labels = [], []
    counts = Counter()
    for _, log in read(path).values():
        teams = {}
        opening = {seat: [] for seat in SEATS}
        counts["logs"] += 1
        try:
            for event in log.splitlines():
                if event.startswith("|turn|"):
                    break
                if event.startswith("|showteam|"):
                    _, _, seat, packed = event.split("|", 3)
                    teams[seat] = schema.showteam(packed)
                if event.startswith("|switch|"):
                    _, _, who, details, _ = event.split("|", 4)
                    opening[who[:2]].append(ident(details.split(",", 1)[0]))
            if len(teams) != 2 or any(len(teams[seat]) != 6 or len(opening[seat]) != 2 for seat in SEATS):
                continue
            match = []
            for seat, foe in (SEATS, SEATS[::-1]):
                own = [mon for mon, _ in teams[seat]]
                lead = tuple(sorted(own.index(mon) for mon in opening[seat]))
                labels_index = PAIRS.index(lead)
                match.append(([features for player in (seat, foe) for _, features in teams[player]], labels_index))
        except (KeyError, ValueError):
            counts["unmatched_forms_or_incomplete"] += 1
            continue
        counts["used"] += 1
        for row, label in match:
            features.append(row)
            labels.append(label)
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64), dict(counts)


class LeadNetwork(nn.Module):
    def __init__(self, features, width=112):
        super().__init__()
        self.encode = nn.Linear(features, width)
        self.side = nn.Parameter(torch.randn(2, 1, width) * .02)
        block = nn.TransformerEncoderLayer(width, 4, width * 3, dropout=.14, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.teams = nn.TransformerEncoder(block, 3, enable_nested_tensor=False)
        self.pair = nn.Sequential(nn.LayerNorm(3 * width), nn.Linear(3 * width, width), nn.SiLU(), nn.Linear(width, 1))
        self.register_buffer("lead_pairs", torch.tensor(PAIRS))

    def forward(self, features):
        embedded = self.encode(features).reshape(len(features), 2, 6, -1) + self.side
        context = self.teams(embedded.flatten(1, 2))
        left, right = context[:, self.lead_pairs].unbind(2)
        counter = context[:, 6:].mean(1, keepdim=True).expand(-1, len(PAIRS), -1)
        return self.pair(torch.cat((left + right, left * right, counter), dim=-1)).squeeze(-1)


@torch.inference_mode()
def metrics(model, features, target):
    logits = torch.cat([model(batch) for batch in features.split(1024)])
    order = logits.argsort(dim=1, descending=True)
    return dict(nll=float(F.cross_entropy(logits, target)), top1=float((order[:, 0] == target).float().mean()),
                top3=float((order[:, :3] == target[:, None]).any(1).float().mean()))


@torch.inference_mode()
def local_check(path, model, schema, cards, prior, device):
    if read(path.with_name("frozen.json"))["blind"]:
        raise ValueError("Transfer diagnostics use only development pools")
    source = read(path)
    lookup = {card["card_id"]: card for card in cards}
    result = Counter()
    for row in source["records"].values():
        for game in row["quicksilver-public"]:
            actor = game["actor"]
            foe = next(seat for seat in SEATS if seat != actor)
            teams = [[lookup[card] for card in game["roster"][seat]] for seat in (foe, actor)]
            own = [ident(card["species"]) for card in teams[0]]
            decision = deployment(*teams)
            ranked = tactical_actions(decision)
            outcomes, _ = prior.scenarios(decision, [], ranked)
            logits = model(torch.tensor([schema.local(teams[0]) + schema.local(teams[1])], device=device))
            top = [frozenset(own[i] for i in PAIRS[index]) for index in logits[0].argsort(descending=True).tolist()]
            actual = frozenset(game["lineups"][foe]["leads"])
            result.update(games=1, human_top1=top[0] == actual, human_top3=actual in top[:3],
                          local_top1=frozenset(outcomes[0]["leads"]) == actual,
                          local_top3=any(frozenset(action["leads"]) == actual for action in outcomes[:3]))
    return dict(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--held", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists() or args.fit.resolve() == args.held.resolve():
        parser.error("Use distinct formats and a fresh output")
    configure_torch()
    torch.manual_seed(80962)
    torch.set_float32_matmul_precision("high")
    schema = Schema()
    start = perf_counter()
    train, label, source_train = logs(args.fit, schema)
    held, target, source_held = logs(args.held, schema)
    device = torch.device("cuda")
    x, y, v, t = (torch.from_numpy(array).to(device) for array in (train, label, held, target))
    config = dict(features=x.shape[-1], width=112)
    model = LeadNetwork(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.00045, fused=True)
    save(args.output / "config.json", dict(network=config, data_sha256={p.name: sha256(p.read_bytes()).hexdigest() for p in (args.fit, args.held)}))
    scores, best, best_epoch = [], float("inf"), 0
    started = perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in torch.randperm(len(x), device=device).split(512):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x[batch]), y[batch])
            loss.backward()
            optimizer.step()
        model.eval()
        scores.append(dict(epoch=epoch, **metrics(model, v, t)))
        print(scores[-1], flush=True)
        if scores[-1]["nll"] < best:
            best, best_epoch = scores[-1]["nll"], epoch
            torch.save(model.state_dict(), args.output / "state_dict.pt")
        if epoch - best_epoch >= 5:
            break
    fit_time = perf_counter() - started
    model.load_state_dict(torch.load(args.output / "state_dict.pt", map_location=device, weights_only=True))
    cards = read(CATALOG)
    prior = LeadPrior(Path("checkpoints/research80/lead-ensemble-v2.json"), cards)
    local = local_check(args.local, model, schema, cards, prior, device)
    record = dict(best_epoch=best_epoch, metrics=scores, local_transfer=local, fit_logs=source_train, held_logs=source_held,
                  train_decisions=len(x), validation_decisions=len(v), train_seconds=fit_time, wall_seconds=perf_counter() - start,
                  device=torch.cuda.get_device_name(), parameters=sum(parameter.numel() for parameter in model.parameters()))
    save(args.output / "training.json", record)
    print(dict(record, metrics=scores[best_epoch - 1]), flush=True)


if __name__ == "__main__":
    main()
