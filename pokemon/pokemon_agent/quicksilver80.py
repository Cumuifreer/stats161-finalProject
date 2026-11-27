"""Run resumable full-match Quicksilver development or blind evaluation."""

import argparse
from pathlib import Path
from secrets import randbits

from .blind85 import input_files, score_suite, used_pools
from .draft_counterfactual import check_hybrid
from .draft_leaf import CATALOG
from .draft_league import pools_for, read, save
from .lab85 import QUICKSILVER, digests
from .planning_bench import paired_gain
from .sim import fingerprint
from .strength import EXTERNAL
from .strength85 import STYLES


FROZEN_85 = {
    "variant": "wide", "preview": True, "preview_model": "", "position_model": "", "draft_review": False,
    "full_preview": True, "lead_model": "checkpoints/research85/lead-prior.json",
    "action_model": "checkpoints/research85/actions-v1", "deep_preview": True,
}
GENERALIZATION = (*EXTERNAL, *STYLES, "champion", "baseline70")
CLOCK_FAILURES = ("draft_timeout", "team_preview_timeout", "moving_timeout", "bank_forfeit")


def freeze(checkout, quicksilver, candidate, pools, blind, opponents, workers, catalog):
    models = tuple(Path(candidate[key]) if candidate[key] else None
                   for key in ("preview_model", "position_model", "lead_model", "action_model"))
    spec = digests(checkout, *models, quicksilver=quicksilver)
    return {
        "candidate": candidate, "files": input_files(checkout, *models), "quicksilver": spec["external"]["quicksilver"],
        "pairs": pools, "blind": blind, "workers": workers, "catalog": fingerprint(catalog),
        "opponents": list(opponents), "pairing": "common pool, first draft seat and match seed; swap players",
        "api_mode": "no-Jev", "interval": "95% paired bootstrap by card pool",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ace-trainer", type=Path, required=True)
    parser.add_argument("--quicksilver", type=Path, required=True)
    parser.add_argument("--prior-workspace", type=Path, action="append", default=[])
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--pairs", type=int, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--opponents", nargs="+", choices=(QUICKSILVER, *GENERALIZATION), default=[QUICKSILVER])
    parser.add_argument("--generalization", nargs="+", choices=GENERALIZATION, default=[])
    parser.add_argument("--generalization-pairs", type=int, default=64)
    args = parser.parse_args()
    blind = args.seed is None
    if args.pairs < (128 if blind else 1) or args.workers < 1:
        parser.error("Provide positive counts and at least 128 paired pools for a blind evaluation")
    if blind and QUICKSILVER not in args.opponents:
        parser.error("A blind evaluation must include Quicksilver")
    if blind and args.compare:
        parser.error("Control selection by a prior run is restricted to development")
    if args.generalization and (not blind or not 32 <= args.generalization_pairs <= args.pairs
                                or len(set(args.generalization)) != len(args.generalization)):
        parser.error("Generalization needs distinct opponents and at least 32 pools from the blind sample")
    if args.output.exists() != args.resume or len(args.opponents) != len(set(args.opponents)):
        parser.error("Provide a fresh output or resume a locked directory and use distinct opponents")

    check_hybrid()
    cards = read(CATALOG)
    candidate = read(args.candidate) if args.candidate else FROZEN_85
    control = read(args.compare) if args.compare else None
    lock_args = (args.ace_trainer, args.quicksilver, candidate, args.pairs, blind, args.opponents, args.workers, cards)
    locked = freeze(*lock_args)
    locked["generalization"] = {"opponents": args.generalization,
                                "pairs": args.generalization_pairs if args.generalization else 0}
    if args.resume:
        frozen = read(args.output / "frozen.json")
        if any(frozen[key] != value for key, value in locked.items()):
            raise ValueError("A resume cannot change the candidate or fixtures")
        sampled = read(args.output / "sample.json")
        pools = pools_for(cards, range(sampled["seed"], sampled["seed"] + args.pairs))
        if pools != sampled["fingerprints"] or (not blind and sampled["seed"] != args.seed):
            raise ValueError("The locked pool draw changed")
    else:
        previous = used_pools(cards)
        for workspace in args.prior_workspace:
            previous.update(used_pools(cards, workspace))
        frozen = locked | {"excluded_pool_count": len(previous)}
        if blind:
            save(args.output / "frozen.json", frozen)
            while True:
                seed = randbits(63)
                pools = pools_for(cards, range(seed, seed + args.pairs))
                if previous.isdisjoint(pools.values()):
                    break
        else:
            seed = args.seed
            pools = pools_for(cards, range(seed, seed + args.pairs))
            if control is not None:
                if control["pools"] != pools or any(name not in control["report"] for name in args.opponents):
                    raise ValueError("Development control must have the same paired pools and opponents")
            elif not previous.isdisjoint(pools.values()):
                raise ValueError("The development pool range intersects a prior experiment")
            save(args.output / "frozen.json", frozen)
        save(args.output / "sample.json", {"seed": seed, "fingerprints": pools})

    outcome = score_suite(args.output / "matches.json", frozen, pools, cards, args.opponents, args.ace_trainer,
                          args.workers, candidate, args.quicksilver)
    report = outcome["report"]
    if freeze(*lock_args) != {key: value for key, value in locked.items() if key != "generalization"}:
        raise RuntimeError("Inputs changed during the paired run")
    for name, metrics in report.items():
        if any(metrics[side].get(key) for side in ("champion_actions", "opponent_actions") for key in CLOCK_FAILURES):
            raise RuntimeError(f"Clock failure while playing {name}")
        count, total = metrics["wins"], metrics["games"]
        low, high = metrics["paired_bootstrap_95"]
        print(f"{name}: {count}/{total} ({count / total:.2%}), paired 95% {low:.2%}–{high:.2%}", flush=True)
    if blind and not (args.output / "acceptance.json").exists():
        metrics = report[QUICKSILVER]
        save(args.output / "acceptance.json", {
            "target": 0.80, "point_target": metrics["win_rate"] >= 0.80,
            "paired_lower_bound_target": metrics["paired_bootstrap_95"][0] >= 0.80,
        })
    if control is not None and not (args.output / "gains.json").exists():
        save(args.output / "gains.json", {name: paired_gain(outcome["records"], control["records"], name)
                                         for name in args.opponents})
    if args.generalization:
        subset = dict(list(pools.items())[:args.generalization_pairs])
        tested = score_suite(args.output / "generalization.json", frozen, subset, cards, args.generalization,
                             args.ace_trainer, args.workers, candidate, args.quicksilver)
        if freeze(*lock_args) != {key: value for key, value in locked.items() if key != "generalization"}:
            raise RuntimeError("Inputs changed during the generalization checks")
        for name, metrics in tested["report"].items():
            if any(metrics[side].get(key) for side in ("champion_actions", "opponent_actions") for key in CLOCK_FAILURES):
                raise RuntimeError(f"Clock failure while playing {name}")
            low, high = metrics["paired_bootstrap_95"]
            print(f"{name}: {metrics['wins']}/{metrics['games']} ({metrics['win_rate']:.2%}), "
                  f"paired 95% {low:.2%}–{high:.2%}", flush=True)


if __name__ == "__main__":
    main()
