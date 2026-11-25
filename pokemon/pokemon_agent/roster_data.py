"""Generate win-labelled drafts and lineups with the existing Showdown engine."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from multiprocessing import get_context
from pathlib import Path
from random import Random
from time import perf_counter

from .game import BATTLE, DRAFT, PREVIEW, ident
from .joint import JointPolicy
from .lineup import lineup_options
from .policy import _draft_score
from .sim import SEATS, Showdown, fingerprint, play_game, pool_fingerprint
from .tactics import load_card_profiles, tactical


class SamplingPolicy:
    def __init__(self, cards, profiles, seed, battle, prior):
        self.rng = Random(seed)
        self.bias = {card["card_id"]: self.rng.gauss(0, 1) + .8 * prior.get(card["card_id"], 0.) for card in cards}
        self.skill = (.0, .6, 1.4)[seed % 3]
        self.battle = JointPolicy(profiles) if battle == "joint" else tactical
        self.drafted = []
        self.selection = None

    def __call__(self, decision):
        if decision.phase == DRAFT:
            available = {card["card_id"]: card for card in decision.observation["cards"]}
            picked = decision.observation["picked"]
            selected = max(decision.actions, key=lambda action: self.bias[action["card_id"]]
                           + self.skill * _draft_score(available[action["card_id"]], picked) / 20)
            self.drafted.append(available[selected["card_id"]])
            return selected
        if decision.phase == PREVIEW:
            if isinstance(self.battle, JointPolicy):
                self.battle.observe(decision)
            lineups = lineup_options(decision)
            self.selection = lineups[0] if self.rng.random() < .65 else self.rng.choice(lineups)
            return self.selection
        return self.battle(decision)


def _sample(cards, profiles, seeds, prior):
    outcomes, skipped = [], Counter()
    with Showdown() as engine:
        for seed in seeds:
            pair = []
            for swap in (False, True):
                kinds = ("joint", "joint") if seed % 4 else ("joint", "tactical")
                indices = (1, 0) if swap else (0, 1)
                policies = {seat: SamplingPolicy(cards, profiles, seed * 4 + index, kinds[index],
                                                  prior if seed % 4 == 1 or seed % 4 == 2 and index == 0 else {})
                            for seat, index in zip(SEATS, indices, strict=True)}
                samples = {seat: [] for seat in SEATS}
                try:
                    winner = play_game(engine, cards, policies, seed, samples)
                except RuntimeError as error:
                    if str(error).startswith("Item Clause left") or str(error) == "Benchmark match exceeded 500 decision rounds":
                        skipped[str(error)] += 1
                        break
                    raise
                if any(event for rows in samples.values() for _, _, event in rows):
                    skipped["invalid or timed-out decision"] += 1
                    break
                rosters, roles = [], []
                for seat in SEATS:
                    policy = policies[seat]
                    names = [ident(card["species"]) for card in policy.drafted]
                    line = policy.selection
                    rosters.append([card["card_id"] for card in policy.drafted])
                    roles.append([2 if name in line["leads"] else 1 if name in line["bring"] else 0 for name in names])
                    if sorted(roles[-1]) != [0, 0, 1, 1, 2, 2]:
                        raise RuntimeError("Preview roles disagree with the drafted roster")
                pair.append({"seed": seed, "swap": swap, "cards": rosters, "roles": roles,
                             "p1_score": .5 if winner is None else float(winner == "p1")})
            if len(pair) == 2:
                outcomes.extend(pair)
    return outcomes, skipped


def generate(args):
    if min(args.pairs, args.workers) < 1 or args.output.exists():
        raise ValueError("Use positive generation sizes and a new output path")
    cards = json.loads(args.cards.read_text(encoding="utf-8"))
    profiles = load_card_profiles(cards, with_sets=True)
    prior = {}
    prior_sha = None
    if args.behavior_prior:
        from hashlib import sha256
        import torch

        config = json.loads((args.behavior_prior / "config.json").read_text(encoding="utf-8"))
        if config["catalog_sha256"] != fingerprint(cards):
            raise ValueError("Behavior prior must use the same catalog")
        path = args.behavior_prior / "state_dict.pt"
        prior_sha = sha256(path.read_bytes()).hexdigest()
        values = torch.load(path, map_location="cpu", weights_only=True)["prior"]
        standardized = (values - values.mean()) / values.std().clamp_min(.05)
        prior = dict(zip(config["card_ids"], standardized.tolist(), strict=True))
    seeds = range(args.seed, args.seed + args.pairs)
    started = perf_counter()
    samples, skipped = [], Counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as executor:
        jobs = [executor.submit(_sample, cards, profiles, seeds[i::args.workers], prior) for i in range(min(len(seeds), args.workers))]
        for job in jobs:
            rows, errors = job.result()
            samples.extend(rows)
            skipped.update(errors)
    seconds = perf_counter() - started
    metrics = {"seeds": args.pairs, "first_seed": args.seed, "games": len(samples), "skipped_pairs": args.pairs - len(samples) // 2,
               "skip_reasons": dict(skipped), "wall_seconds": seconds, "games_per_second": len(samples) / seconds,
               "workers": args.workers, "eligible_pools": {str(seed): pool_fingerprint(cards, seed) for seed in seeds},
               "catalog_sha256": fingerprint(cards), "schema_version": 1, "behavior_prior_sha256": prior_sha}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"metrics": metrics, "samples": samples}, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in metrics.items() if key != "eligible_pools"}), flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", type=Path, default=Path(__file__).resolve().parents[1] / "data/historical_cards.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pairs", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--behavior-prior", type=Path, help="Use a fitted draft strength prior in a subset of games")
    return generate(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
