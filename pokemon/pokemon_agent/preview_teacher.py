"""Prepare blinded, offline lineup prompts and score legal ID replies without an API."""

import argparse
import json
from pathlib import Path
from random import Random
from statistics import mean
from time import perf_counter

from .game import PREVIEW, Decision
from .lineup import LineupPolicy, index_cards, lineup_options
from .sim import SEATS, Showdown
from .tactics import load_card_profiles


def build(args):
    if args.output.exists() or args.cases < 1:
        raise ValueError("Use a positive case count and a new panel directory")
    cards = json.loads(args.cards.read_text(encoding="utf-8"))
    catalog = index_cards(cards)
    profiles = load_card_profiles(cards, with_sets=True)
    rng = Random(args.seed)
    prompts, evidence, seen = {}, {}, set()
    paths = sorted(args.captures.glob("*.jsonl"))
    rng.shuffle(paths)
    with Showdown() as game:
        for path in paths:
            with path.open(encoding="utf-8") as trace:
                decision = next((Decision.from_snapshot(row["decision"]) for row in map(json.loads, trace)
                                 if row["decision"]["phase"] == PREVIEW), None)
            if decision is None:
                continue
            our, foe = ([mon.get("species") for mon in decision.observation[key] if mon]
                        for key in ("team", "opponent_team"))
            if any(len(team) != 6 or not set(team) <= catalog.keys() for team in (our, foe)):
                continue
            identity = frozenset(frozenset(team) for team in (our, foe))
            if identity in seen:
                continue
            policy = LineupPolicy(profiles, catalog, game)
            options = lineup_options(decision)
            choice = policy(decision)
            if choice == options[0]:
                continue
            rosters = {side: [catalog[name] for name in team] for side, team in zip(SEATS, (our, foe), strict=True)}
            # Use different random seeds for the answer key and keep it out of the model prompt.
            draws = [[rng.randrange(65536) for _ in range(4)] for _ in range(8)]
            mapping = {str(i): action for i, action in enumerate(options)}
            deadline = perf_counter() + 12
            try:
                outcomes = {key: [[policy._play(rosters, action, variant, seed, deadline) for seed in draws]
                                  for variant in (0, 1)] for key, action in mapping.items()}
            except (RuntimeError, ValueError, OSError):
                continue
            key = str(len(prompts))
            prompts[key] = {"task": "Pick four Pokémon and two ordered leads in doubles. The opponent's four are unknown. "
                                    "Return JSON with just the legal candidate id.",
                            "our_public_cards": rosters["p1"], "opponent_public_cards": rosters["p2"],
                            "legal_candidates": mapping}
            evidence[key] = {"baseline": str(options.index(choice)), "tactical": "0", "outcomes": outcomes}
            seen.add(identity)
            if len(prompts) == args.cases:
                break
    if not prompts:
        raise ValueError("No eligible public preview cases with different local recommendations")
    args.output.mkdir(parents=True)
    for filename, content in (("prompts", prompts), ("evidence", evidence)):
        (args.output / f"{filename}.json").write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    (args.output / "tactical-replies.json").write_text(json.dumps({key: {"id": row["tactical"]} for key, row in evidence.items()}),
                                                      encoding="utf-8")
    print(json.dumps({"cases": len(prompts), "requested": args.cases, "seed": args.seed,
                      "validation_games": sum(len(row["outcomes"]) * 16 for row in evidence.values()),
                      "prompt_utf8_bytes": sum(len(json.dumps(row).encode()) for row in prompts.values())}))
    return 0


def score(args):
    evidence = json.loads((args.panel / "evidence.json").read_text(encoding="utf-8"))
    replies = json.loads(args.replies.read_text(encoding="utf-8"))
    if set(replies) != set(evidence):
        raise ValueError("Supply one reply per panel case")
    deltas = []
    different = 0
    for key, case in evidence.items():
        reply = replies[key]
        if not isinstance(reply, dict) or set(reply) != {"id"} or not isinstance(reply["id"], str) or reply["id"] not in case["outcomes"]:
            raise ValueError(f"Invalid reply for panel case {key}")
        baseline = case["baseline"]
        different += baseline != reply["id"]
        diff = [mean(a) - mean(b) for a, b in zip(case["outcomes"][reply["id"]], case["outcomes"][baseline], strict=True)]
        deltas.append(mean(diff))
    print(json.dumps({"cases": len(deltas), "different": different, "paired_delta": mean(deltas), "by_case": deltas}))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(required=True)
    make = modes.add_parser("build")
    make.add_argument("--captures", type=Path, required=True)
    make.add_argument("--cards", type=Path, default=Path(__file__).resolve().parents[1] / "data/historical_cards.json")
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--cases", type=int, default=12)
    make.add_argument("--seed", type=int, default=882)
    make.set_defaults(run=build)
    evaluate = modes.add_parser("score")
    evaluate.add_argument("--panel", type=Path, required=True)
    evaluate.add_argument("--replies", type=Path, required=True)
    evaluate.set_defaults(run=score)
    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
