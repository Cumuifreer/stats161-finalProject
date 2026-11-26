"""Lock a candidate, then sample unrelated pools for full-match acceptance."""

import argparse
from hashlib import sha256
from pathlib import Path
from secrets import randbits
import subprocess
import sys

from .draft_league import read, save
from .draft_matchups import ROOT
from .strength import EXTERNAL


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def fingerprint_candidate(development, checkout, reply):
    package = Path(__file__).parent
    source_paths = sorted(package.glob("*.py")) + sorted(package.glob("*.cjs"))
    source_paths += [ROOT / "package-lock.json", ROOT / "data" / "historical_cards.json"]
    source_paths += sorted((checkout / "agent").rglob("*.py"))
    source_paths += sorted((checkout / "sim").rglob("*.py"))
    spec = development["spec"]
    models = spec.get("research_models", {})
    source_paths += [Path(path) for path in models]
    if spec.get("reply_weights"):
        if reply is None:
            raise ValueError("Supply the frozen reply checkpoint")
        source_paths += [reply / file for file in spec["reply_weights"]]
    actual = {str(path): digest(path) for path in source_paths}
    for name, expected in spec["external"].items():
        if actual[str(checkout / name)] != expected:
            raise ValueError(f"The evaluated opponent source changed at {name}")
    for name, expected in spec.get("sources", {}).items():
        if digest(package / name) != expected:
            raise ValueError(f"The development candidate source changed at {name}")
    for name, expected in (spec.get("reply_weights") or {}).items():
        if actual[str(reply / name)] != expected:
            raise ValueError(f"The evaluated reply checkpoint changed at {name}")
    for path, expected in models.items():
        if actual[str(Path(path))] != expected:
            raise ValueError(f"The evaluated research model changed at {path}")
    return actual


def benchmark_flags(spec, reply):
    setting_names = {"depth": "--depth", "worlds": "--worlds", "samples": "--samples",
                     "candidate_limit": "--candidates", "min_edge": "--edge"}
    flags = [flag for name, option in setting_names.items() for flag in (option, str(spec["settings"][name]))]
    for name, option in (("interaction", "--interaction"), ("adaptive", "--adaptive-replies"),
                         ("proposals", "--model-proposals"), ("adaptive_all", "--adaptive-all"),
                         ("replacements", "--replacement-search")):
        if spec["settings"].get(name):
            flags.append(option)
    if spec["settings"].get("move_budget") is not None:
        flags.extend(["--move-budget", str(spec["settings"]["move_budget"])])
    for name, option in (("proxy_replies", "--proxy-replies"), ("preview_search", "--preview-search"),
                         ("draft_bundles", "--draft-bundles")):
        if spec.get(name):
            flags.append(option)
    if spec.get("reply_weights"):
        flags.extend(["--reply-model", str(reply)])
    for path in spec.get("research_models", {}):
        model = read(Path(path))
        flags.extend(["--value-model" if "pressure" in model else "--lineup-model", path])
    return flags


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--ace-trainer", type=Path, required=True)
    parser.add_argument("--reply-model", type=Path)
    parser.add_argument("--pairs", type=int, default=384)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude", nargs="*", type=Path, default=[])
    args = parser.parse_args()
    if args.output.exists() or min(args.workers, args.pairs) < 1:
        parser.error("Use a new output directory and positive counts")
    development = read(args.development)
    if not all(name in development["report"] for name in EXTERNAL):
        parser.error("The locked development result must cover both target opponents")
    files = fingerprint_candidate(development, args.ace_trainer, args.reply_model)
    previous = ROOT / "benchmark_results"
    histories = set(args.exclude + [args.development])
    for folder in previous.glob("public-planning-*"):
        for report in folder.glob("*.json"):
            if read(report).get("pools"):
                histories.add(report)
    for folder in previous.glob("planning-blind-*"):
        report = folder / "matches.json"
        if report.exists():
            histories.add(report)
    opponents = [*EXTERNAL, "champion"]
    save(args.output / "frozen.json", dict(files=files, spec=development["spec"], opponents=opponents,
                                           count=args.pairs, target=.70, interval="95% paired bootstrap",
                                           development=str(args.development), excluded=sorted(map(str, histories))))
    blind_seed = randbits(63)
    save(args.output / "sample.json", {"seed": blind_seed})
    result = args.output / "matches.json"
    command = [sys.executable, "-m", "pokemon_agent.planning_bench", "--opponents", *opponents,
               "--ace-trainer", str(args.ace_trainer), "--seed", str(blind_seed), "--pairs", str(args.pairs),
               "--workers", str(args.workers), "--output", str(result),
               *benchmark_flags(development["spec"], args.reply_model), "--exclude", *sorted(map(str, histories))]
    subprocess.run(command, cwd=ROOT, check=True)
    if fingerprint_candidate(development, args.ace_trainer, args.reply_model) != files:
        raise RuntimeError("The locked candidate changed during blind evaluation")
    report = read(result)["report"]
    reached = all(report[name]["win_rate"] >= .70 for name in EXTERNAL)
    lower_bounds_reached = all(report[name]["paired_bootstrap_95"][0] >= .70 for name in EXTERNAL)
    save(args.output / "acceptance.json", dict(point_target_reached=reached, lower_bounds_reached=lower_bounds_reached,
                                              report=report))
    print(f"70% point target: {reached}. Both 95% lower bounds at least 70%: {lower_bounds_reached}.", flush=True)


if __name__ == "__main__":
    main()
