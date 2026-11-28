"""Freeze one research candidate before sampling the acceptance pools."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from secrets import randbits
from time import time

from .draft_counterfactual import check_hybrid
from .draft_leaf import CATALOG
from .draft_league import pools_for, read, save
from .lab85 import digests, measure
from .paired_store85 import PairedStore, publish_json
from .planning_bench import paired_gain
from .sim import DRAFT_SIZE, fingerprint
from .strength import EXTERNAL, summarize
from .strength85 import FROZEN_PACKAGE, STYLES


def input_files(checkout, preview_model, position_model, lead_model, action_model):
    package = Path(__file__).resolve().parent
    paths = [*package.glob("*.py"), *package.glob("*.cjs"), *(package / "research80").rglob("*.py"),
             CATALOG, package.parent / "package-lock.json"]
    paths.extend(FROZEN_PACKAGE.glob("*.cjs"))
    showdown = (package.parent / "node_modules" / "pokemon-showdown").resolve()
    frozen_showdown = (FROZEN_PACKAGE.parent / "node_modules" / "pokemon-showdown").resolve()
    showdown_roots = {showdown}
    if (frozen_showdown / "package.json").is_file():
        showdown_roots.add(frozen_showdown)
    for root in showdown_roots:
        paths.append(root / "package.json")
        for folder, names in (("sim", ("battle", "pokemon", "side", "field", "prng")),
                              ("data", ("moves", "abilities", "conditions", "items"))):
            paths.extend(root / "dist" / folder / f"{name}.js" for name in names)
    for folder in ("agent", "sim"):
        paths.extend((checkout / folder).rglob("*.py"))
    for model_root in ("roster-v1/expanded", "draft-v2", "draft-v3/pick-attention", "draft-leaf-v1/value", "public-planning-v1"):
        paths.extend(path for path in (Path("checkpoints") / model_root).rglob("*")
                     if path.name in {"config.json", "state_dict.pt", "value-v2.json", "lineup-belief.json"})
    if preview_model:
        paths.extend([preview_model / "config.json", preview_model / "state_dict.pt"])
    if position_model:
        paths.append(position_model)
    if lead_model:
        paths.append(lead_model)
    if action_model:
        paths.extend([action_model / "config.json", action_model / "state_dict.pt"])
    return {str(path): sha256(path.read_bytes()).hexdigest() for path in sorted(set(paths))}


def used_pools(cards, workspace=Path(".")):
    used = set()
    for root in (workspace / "benchmark_results", workspace / "checkpoints", workspace / "data"):
        for path in root.rglob("*.json"):
            data = read(path)
            if not isinstance(data, dict):
                continue
            groups = [data.get("pools"), data.get("fingerprints")]
            for key in ("split", "splits"):
                if isinstance(data.get(key), dict):
                    groups.extend(data[key].values())
            for group in groups:
                if isinstance(group, dict):
                    used.update(value for value in group.values() if isinstance(value, str) and len(value) == 64)
            seed, count = data.get("seed"), data.get("pairs")
            if isinstance(seed, int) and isinstance(count, int) and data.get("pool_order_sha256"):
                pools = pools_for(cards, range(seed, seed + count))
                if fingerprint(list(pools.values())) == data["pool_order_sha256"]:
                    used.update(pools.values())
    for path in (workspace / "captures").glob("*.jsonl"):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                decision = json.loads(line).get("decision", {})
                if decision.get("phase") != "draft":
                    continue
                board = decision.get("observation", {})
                ids = {card["card_id"] for key in ("cards", "picked", "opponent_picked") for card in board.get(key, ())}
                if len(ids) == DRAFT_SIZE:
                    used.add(fingerprint(sorted(ids)))
                    break
    return used


def score_suite(path, frozen, pools, cards, opponents, checkout, workers, candidate, quicksilver=None):
    signature = fingerprint([frozen, candidate, list(opponents)])
    if path.exists():
        result = read(path)
        if result["signature"] != signature or result["pools"] != pools:
            raise ValueError("A completed result has a different lock or pool draw")
        return result
    store = PairedStore(path.with_suffix(""), signature, pools, opponents)
    records, cpu = store.recover()
    roots = [int(root) for root in pools if int(root) not in records]
    Random(int(next(iter(pools))) ^ 0x762E).shuffle(roots)
    batches = measure(roots, cards, opponents, checkout, candidate["variant"], workers, candidate["preview"],
                      Path(candidate["preview_model"]) if candidate["preview_model"] else None,
                      Path(candidate["position_model"]) if candidate["position_model"] else None,
                      candidate["draft_review"], candidate["full_preview"],
                      Path(candidate["lead_model"]) if candidate["lead_model"] else None,
                      Path(candidate["action_model"]) if candidate["action_model"] else None, candidate["deep_preview"],
                      quicksilver, candidate.get("research80")) if roots else ()
    for part, python_cpu in batches:
        store.append(part, python_cpu)
        records.update(part)
        cpu += python_cpu
        print(f"Completed {len(records)}/{len(pools)} card pools for {path.stem}", flush=True)
    if len(records) != len(pools):
        raise RuntimeError("Missing paired acceptance games")
    records = dict(sorted(records.items()))
    report = {}
    for name in opponents:
        pairs = [row[name] for row in records.values()]
        metrics = summarize(pairs)
        for key in ("planning", "preview85", "draft85", "quicksilver"):
            totals = Counter()
            for pair in pairs:
                for game in pair:
                    totals.update(game.get(key, {}))
            if totals:
                metrics[key] = dict(totals)
        report[name] = metrics
    result = dict(signature=signature, pools=pools, records=records, report=report, seconds=time() - store.started,
                  worker_python_cpu_seconds=cpu, api_spend_usd=0)
    publish_json(path, result)
    return read(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--ace-trainer", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=256)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.pairs < 128 or args.workers < 1 or args.output.exists() != args.resume:
        parser.error("Provide at least 128 pools and either a fresh directory or --resume for an existing lock")
    development = read(args.development)
    if not all(name in development["report"] for name in EXTERNAL):
        parser.error("Development report must cover both named opponents")
    preview_model = Path(development["preview_model"]) if development["preview_model"] else None
    position_model = Path(development["position_model"]) if development["position_model"] else None
    lead_model = Path(development["lead_model"]) if development["lead_model"] else None
    action_model = Path(development["action_model"]) if development["action_model"] else None
    models = (preview_model, position_model, lead_model, action_model)
    if development["sources"] != digests(args.ace_trainer, *models):
        raise ValueError("Development source, checkpoint, or opponent has changed")
    check_hybrid()
    files = input_files(args.ace_trainer, *models)
    opponents = [*EXTERNAL, "champion", "baseline70"]
    cards = read(CATALOG)
    candidate = {key: development[key] for key in ("variant", "preview", "preview_model", "position_model", "draft_review",
                                                   "full_preview", "lead_model", "action_model", "deep_preview")}
    locked = dict(candidate=candidate, target=.85, pairs=args.pairs, candidate_files=files, opponents=opponents,
                  development=str(args.development), interval="95% paired bootstrap by card pool",
                  style_checks=list(STYLES), style_pairs=min(64, args.pairs))
    if args.resume:
        frozen = read(args.output / "frozen.json")
        if any(frozen[key] != value for key, value in locked.items()):
            raise RuntimeError("The attempted resume changed the locked candidate or protocol")
        sample = read(args.output / "sample.json")
        samples = pools_for(cards, range(sample["seed"], sample["seed"] + args.pairs))
        if samples != sample["fingerprints"]:
            raise RuntimeError("The paired blind draw changed")
    else:
        prior_pools = used_pools(cards)
        frozen = locked | {"previous_pool_count": len(prior_pools)}
        save(args.output / "frozen.json", frozen)
        while True:
            seed = randbits(63)
            samples = pools_for(cards, range(seed, seed + args.pairs))
            if prior_pools.isdisjoint(samples.values()):
                break
        save(args.output / "sample.json", dict(seed=seed, fingerprints=samples))
    outcomes = score_suite(args.output / "matches.json", frozen, samples, cards, opponents, args.ace_trainer,
                           args.workers, candidate)
    if input_files(args.ace_trainer, *models) != files:
        raise RuntimeError("The blind candidate, environment inputs, or opponents changed")
    report = outcomes["report"]
    timeout_keys = ("draft_timeout", "team_preview_timeout", "moving_timeout", "bank_forfeit")
    passed = all(report[name]["win_rate"] >= .85 and report[name]["illegal_actions"] == 0
                 and not any(report[name][side].get(kind, 0) for kind in timeout_keys
                             for side in ("champion_actions", "opponent_actions")) for name in EXTERNAL)
    verdict = dict(point_target=passed, targets={name: report[name]["win_rate"] for name in EXTERNAL},
                   lower_bounds=all(report[name]["paired_bootstrap_95"][0] >= .85 for name in EXTERNAL))
    if not (args.output / "acceptance.json").exists():
        save(args.output / "acceptance.json", verdict)
    print(f"Both 85% full-match targets reached: {passed}", flush=True)
    if passed:
        controls = [*EXTERNAL, "champion"]
        control = dict(variant="baseline70", preview=False, preview_model="", position_model="", draft_review=False,
                       full_preview=False, lead_model="", action_model="", deep_preview=False)
        baseline = score_suite(args.output / "baseline70.json", frozen, samples, cards, controls, args.ace_trainer,
                               args.workers, control)
        if baseline["pools"] != samples or input_files(args.ace_trainer, *models) != files:
            raise RuntimeError("The locked configuration or the paired control has changed")
        gains = {name: paired_gain(outcomes["records"], baseline["records"], name) for name in controls}
        if not (args.output / "gains.json").exists():
            save(args.output / "gains.json", gains)
        subset = dict(list(samples.items())[:frozen["style_pairs"]])
        score_suite(args.output / "style-matches.json", frozen, subset, cards, list(STYLES), args.ace_trainer,
                    args.workers, candidate)
        tested_control = score_suite(args.output / "style-control.json", frozen, subset, cards, list(STYLES), args.ace_trainer,
                                     args.workers, control)
        tested_candidate = read(args.output / "style-matches.json")
        styled_gains = {name: paired_gain(tested_candidate["records"], tested_control["records"], name) for name in STYLES}
        if input_files(args.ace_trainer, *models) != files:
            raise RuntimeError("The lock changed during the style checks")
        if not (args.output / "style-gains.json").exists():
            save(args.output / "style-gains.json", styled_gains)


if __name__ == "__main__":
    main()
