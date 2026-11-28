"""Research benchmark for public-world planning in complete local matches."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from hashlib import sha256
import json
from multiprocessing import get_context
from pathlib import Path
from random import Random
from statistics import mean
from time import perf_counter

from .draft_bundles import DraftBundles
from .draft_counterfactual import check_hybrid
from .draft_coverage import Coverage
from .draft_last_pick import mean_ci
from .draft_leaf import CATALOG
from .draft_league import disjoint, parallel, pools_for, read, save
from .game import DRAFT, PREVIEW
from .lineup import index_cards
from .opening_active import inner_games
from .planning_preview import PlanningPreview
from .proxy_replies import ProxyReplies
from .public_planning import PublicPlanner
from .reply_learning import LearnedReplies
from .reply_lineup import ReplyLineup
from .sim import SEATS, Showdown, fingerprint
from .strength import Champion, EXTERNAL, ExternalPivot, OPPONENTS, opponent, play, summarize, verify_pools


PEERS = ("champion", "public-one-turn")


class Challenger(Champion):
    def __init__(self, experiment, indexed, worker, settings, learned=None, preview_search=False, proxy=None,
                 lineup_model=None, draft_bundles=False):
        super().__init__(experiment, indexed)
        self.battle = PublicPlanner(experiment.profiles, worker, learned=learned, proxy=proxy,
                                    lineup_model=lineup_model, **settings)
        self.needs_proxy_evidence = proxy is not None
        self.needs_own_proposals = settings.get("proposals", False)
        if preview_search:
            self.preview = PlanningPreview(self.preview, worker, indexed)
        self.planning_counts = self.battle.counts
        self.draft = DraftBundles(experiment) if draft_bundles else None

    def __call__(self, decision):
        if decision.phase == DRAFT and self.draft:
            return self.draft(decision)
        if decision.phase == PREVIEW:
            action = self.preview(decision)
            self.battle.observe_preview(decision, action)
            return action
        return super().__call__(decision)

    def choose_with_public_history(self, decision, seat, evidence):
        return self.battle(decision, seat, evidence)


def configure_worker(worker, cards, model_path):
    parameters = {}
    if model_path:
        model = read(model_path)
        if model["version"] != 2 or model["cards"] != fingerprint(cards):
            raise ValueError("The learned value requires its public card catalog and feature version")
        parameters["position_model"] = {"weights": model["weights"], "pressure": model["pressure"]}
    worker.call("configure", cards=cards, **parameters)


def evaluate(roots, cards, rivals, checkout, settings, reply_model, preview_search, proxy_replies, value_model, lineup_model,
             draft_bundles):
    records, counts = {}, Counter()
    with ExitStack() as stack:
        experiment = Coverage(stack, cards)
        indexed = index_cards(cards)
        worker = stack.enter_context(Showdown(public_search=True))
        configure_worker(worker, cards, value_model)
        learned = LearnedReplies(reply_model, experiment.profiles) if reply_model else None
        proxy = ProxyReplies(stack, checkout, indexed) if proxy_replies else None
        lineup = ReplyLineup(lineup_model, cards, experiment.profiles) if lineup_model else None
        peer = stack.enter_context(Showdown(public_search=True)) if "public-one-turn" in rivals else None
        if peer:
            configure_worker(peer, cards, None)
        external = {name: stack.enter_context(ExternalPivot(checkout, name, cards)) for name in rivals if name in EXTERNAL}
        for root in roots:
            records[root] = {}
            for rival in rivals:
                pair = []
                for actor in SEATS:
                    candidate = Challenger(experiment, indexed, worker, settings, learned, preview_search, proxy, lineup,
                                           draft_bundles)
                    if rival == "champion":
                        foe = Champion(experiment, indexed)
                    elif rival == "public-one-turn":
                        foe = Challenger(experiment, indexed, peer, {})
                    else:
                        foe = opponent(rival, root, experiment.profiles, external.get(rival))
                    game = play(experiment, cards, root, actor, rival, foe, indexed, candidate)
                    pair.append(game)
                    counts.update(candidate.planning_counts)
                    if preview_search:
                        game["preview_planning"] = dict(candidate.preview.counts)
                    if draft_bundles:
                        game["draft_planning"] = dict(candidate.draft.counts)
                records[root][rival] = pair
        counts.update(inner_games(experiment))
    return records, counts


def parts(roots, workers, *inputs):
    if len(roots) <= 6 * workers or workers == 1:
        yield from parallel(evaluate, roots, workers, *inputs)
        return
    batches = [roots[index:index + 6] for index in range(0, len(roots), 6)]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(evaluate, batch, *inputs) for batch in batches]
        for future in as_completed(futures):
            yield future.result()


def paired_gain(records, control, rival):
    def wins(row):
        return sum(game["actor"] == game["winner"] for game in row[rival])

    deltas = [(wins(row) - wins(control[str(root)])) / 2 for root, row in records.items()]
    return {"difference": mean(deltas), "ci95": mean_ci(deltas)}


def source_hashes():
    return {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("public_planning.py", "search.cjs", "planning_rollout.cjs", "showdown.cjs", "strength.py",
                         "planning_refine.py", "planning_preview.py", "planning_preview.cjs", "reply_learning.py",
                         "planning_bench.py", "planning_replay.py", "sim.py", "proxy_replies.py", "planning_interaction.cjs",
                         "planning_value.py", "planning_value.cjs", "reply_belief.py", "reply_lineup.py", "pivot.py",
                         "draft_bundles.py", "forced_position.py")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opponents", nargs="+", choices=(*OPPONENTS, *PEERS), required=True)
    parser.add_argument("--ace-trainer", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pairs", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--worlds", type=int, default=2)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--edge", type=float, default=.08)
    parser.add_argument("--depth", type=int, choices=(0, 1, 2, 3), default=1)
    parser.add_argument("--reply-model", type=Path)
    parser.add_argument("--preview-search", action="store_true")
    parser.add_argument("--proxy-replies", action="store_true")
    parser.add_argument("--interaction", action="store_true")
    parser.add_argument("--adaptive-replies", action="store_true")
    parser.add_argument("--adaptive-all", action="store_true")
    parser.add_argument("--replacement-search", action="store_true")
    parser.add_argument("--draft-bundles", action="store_true")
    parser.add_argument("--value-model", type=Path)
    parser.add_argument("--lineup-model", type=Path)
    parser.add_argument("--model-proposals", action="store_true")
    parser.add_argument("--move-budget", type=float)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--exclude", nargs="*", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or min(args.pairs, args.workers, args.worlds, args.samples, args.candidates) < 1:
        parser.error("Provide a fresh output and positive counts")
    if len(args.opponents) != len(set(args.opponents)) or (set(args.opponents).intersection(EXTERNAL) and not args.ace_trainer):
        parser.error("Provide unique opponents and the external checkout when needed")
    if args.proxy_replies and not args.ace_trainer:
        parser.error("The optional proxy ensemble requires the pinned checkout")
    if (args.adaptive_replies or args.adaptive_all) and not args.proxy_replies:
        parser.error("Adaptive reply weighting requires the proxy ensemble")
    if args.model_proposals and not args.proxy_replies:
        parser.error("Model proposals require the proxy ensemble")
    if args.move_budget is not None and args.move_budget <= 0:
        parser.error("The search budget must be positive")
    check_hybrid()
    cards = read(CATALOG)
    roots = list(range(args.seed, args.seed + args.pairs))
    pools = pools_for(cards, roots)
    verify_pools(pools, args.exclude)
    for model in (args.value_model, args.lineup_model, args.reply_model / "config.json" if args.reply_model else None):
        if model:
            disjoint(pools, read(model)["pools"])
    control = read(args.compare) if args.compare else None
    if control and (control["pools"] != pools or any(rival not in control["report"] for rival in args.opponents)):
        parser.error("The control must cover the same paired pools and opponents")
    for path in args.output.parent.glob("*.json"):
        if args.compare and path.resolve() == args.compare.resolve():
            continue
        previous = read(path)
        if control and previous.get("pools", {}) == pools:
            continue
        disjoint(pools, previous.get("pools", {}))
    Random(args.seed ^ 0x762E).shuffle(roots)
    began = perf_counter()
    settings = dict(worlds=args.worlds, samples=args.samples, candidate_limit=args.candidates,
                    min_edge=args.edge, depth=args.depth, interaction=args.interaction, adaptive=args.adaptive_replies,
                    proposals=args.model_proposals, move_budget=args.move_budget, adaptive_all=args.adaptive_all,
                    replacements=args.replacement_search)
    sources = source_hashes()
    weights = ({name: sha256((args.reply_model / name).read_bytes()).hexdigest()
                for name in ("state_dict.pt", "config.json")} if args.reply_model else None)
    models = {str(path): sha256(path.read_bytes()).hexdigest() for path in (args.value_model, args.lineup_model) if path}
    external = ({name: sha256((args.ace_trainer / name).read_bytes()).hexdigest()
                 for name in ("agent/agent.py", "agent/versions/pivot.py", "agent/pokemon/tuning.py", "sim/translate.py")}
                if args.ace_trainer else None)
    if control and control.get("external", control.get("spec", {}).get("external")) != external:
        raise ValueError("Opponents' source differs from paired control")
    records, counts = {}, Counter()
    for part, stats in parts(roots, args.workers, cards, args.opponents, args.ace_trainer, settings,
                             args.reply_model, args.preview_search, args.proxy_replies, args.value_model, args.lineup_model,
                             args.draft_bundles):
        records.update(part)
        counts.update(stats)
        print(f"Completed {len(records)}/{args.pairs} paired pools", flush=True)
    if len(records) != args.pairs:
        raise RuntimeError("Missing paired benchmark games")
    records = dict(sorted(records.items()))
    report = {}
    for rival in args.opponents:
        pairs = [row[rival] for row in records.values()]
        report[rival] = summarize(pairs)
        planning = Counter()
        preview = Counter()
        draft = Counter()
        for pair in pairs:
            for game in pair:
                planning.update(game["planning"])
                preview.update(game.get("preview_planning", {}))
                draft.update(game.get("draft_planning", {}))
        report[rival]["planning"] = dict(planning)
        if args.draft_bundles:
            report[rival]["draft_planning"] = dict(draft)
        if args.preview_search:
            report[rival]["preview_planning"] = dict(preview)
        if control:
            report[rival]["paired_gain"] = paired_gain(records, control["records"], rival)
    if source_hashes() != sources:
        raise RuntimeError("Planning sources changed during the benchmark")
    spec = {"settings": settings, "catalog": fingerprint(cards), "sources": sources,
            "reply_weights": weights, "research_models": models,
            "preview_search": args.preview_search, "proxy_replies": args.proxy_replies, "draft_bundles": args.draft_bundles}
    if external:
        spec["external"] = external
    save(args.output, dict(report=report, spec=spec, seconds=perf_counter() - began,
                           api_spend_usd=0, pools=pools, records=records, counts=dict(counts)))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
