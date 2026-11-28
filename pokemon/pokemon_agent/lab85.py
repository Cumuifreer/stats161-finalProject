"""Run local research candidates through paired complete matches."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from hashlib import sha256
import json
from multiprocessing import get_context
from pathlib import Path
from random import Random
from time import perf_counter, process_time

import torch

from .action_scout85 import ActionScout85
from .combination_preview import load_model
from .combination_space import CardSpace
from .deep_opening85 import DeepLeadPrior
from .draft85 import Draft85
from .draft_bundles import DraftBundles
from .draft_counterfactual import check_hybrid
from .draft_coverage import Coverage
from .draft_leaf import CATALOG
from .draft_league import disjoint, pools_for, read, save
from .game import PREVIEW
from .horizon85 import PosteriorHorizon85
from .lead85 import LeadPrior
from .lineup import index_cards
from .matchup_preview import MatchupPreview
from .planning_bench import Challenger, configure_worker, paired_gain, source_hashes
from .preview85 import Preview85, PreviewWorker
from .preview_review85 import FullPreviewReview
from .quicksilver import QuicksilverOpponent
from .reply_learning import LearnedReplies
from .reply_lineup import ReplyLineup
from .research80.endgame import CheckedEndgame
from .research80.forecast import EnsembleLeadPrior, MonteCarloLineups
from .research80.scouting.battle import RolloutScout
from .research80.scouting.replies import ForecastReplies80
from .search85 import HypotheticalReplies, Search85
from .sim import SEATS, Showdown, fingerprint
from .strength import Champion, EXTERNAL, ExternalPivot, OPPONENTS, opponent, play, summarize, verify_pools
from .strength85 import AlternatePivot, FROZEN_PACKAGE, FrozenSearch, STYLES
from .wide85 import WideSearch85


MODELS = Path("checkpoints/public-planning-v1")
BASE = dict(worlds=2, samples=1, candidate_limit=15, min_edge=.08, depth=3, interaction=True,
            adaptive=True, proposals=True, move_budget=3.6, adaptive_all=False, replacements=False)
SEALED = [Path("benchmark_results") / name / "matches.json" for name in ("planning-blind-v1", "planning-blind-v2")]
QUICKSILVER = "quicksilver-public"


class OpeningChallenger(Challenger):
    def __call__(self, decision):
        lead_prior = getattr(self.preview, "lead_prior", None)
        if isinstance(lead_prior, EnsembleLeadPrior):
            lead_prior.observe(decision)
        action = super().__call__(decision)
        if decision.phase == PREVIEW:
            self.battle.opening_forecast = self.preview.opposition_forecast
            if isinstance(self.battle, ForecastReplies80):
                self.battle.observe_style(lead_prior.agreement)
        return action


def evaluate(roots, cards, rivals, checkout, variant, preview_type, learned_preview, position_model, draft_review,
             full_preview, lead_model, action_model, deep_preview, quicksilver=None, research80=None):
    torch.set_num_threads(1)
    began = process_time()
    records = {}
    with ExitStack() as stack:
        experiment = Coverage(stack, cards)
        indexed = index_cards(cards)
        worker = stack.enter_context(FrozenSearch() if variant == "baseline70" and not position_model else Showdown(public_search=True))
        if position_model:
            model = read(position_model)
            if model["version"] != 3 or model["cards"] != fingerprint(cards):
                raise ValueError("Expected a value v3 checkpoint for the published card catalog")
            worker.call("configure", cards=cards, position_model={name: model[name] for name in ("version", "scale", "pressure", "layers")})
        else:
            configure_worker(worker, cards, MODELS / "value-v2.json")
        learned = LearnedReplies(MODELS / "reply", experiment.profiles)
        lineup = ReplyLineup(MODELS / "lineup-belief.json", cards, experiment.profiles)
        lead_prior = LeadPrior(lead_model, cards, experiment.profiles) if lead_model else None
        if deep_preview:
            specialist = stack.enter_context(MatchupPreview(checkout, indexed, budget=19.9))
            lead_prior = DeepLeadPrior(lead_prior, specialist)
        research80 = research80 or {}
        if research80.get("rollout_lineups"):
            if not deep_preview or not quicksilver:
                raise ValueError("Second Preview search needs a lead prior and the public engine checkout")
            forecaster = stack.enter_context(MonteCarloLineups(quicksilver, indexed, budget=12.))
            lead_prior = EnsembleLeadPrior(lead_prior, forecaster, propose_own=research80.get("own_preview", False))
        reply_scout = None
        if research80.get("reply_rollouts"):
            if variant != "wide" or not isinstance(lead_prior, EnsembleLeadPrior):
                raise ValueError("The continuation reply expert requires wide search and the public Draft observer")
            reply_scout = stack.enter_context(RolloutScout(quicksilver, checkout, indexed))
        scout = ActionScout85(action_model, cards, experiment.profiles) if action_model else None
        proxy = HypotheticalReplies(stack, checkout, indexed)
        review = (FullPreviewReview(stack.enter_context(Showdown()), proxy.models, indexed, experiment.profiles)
                  if full_preview else None)
        if preview_type or draft_review:
            preview_worker = stack.enter_context(PreviewWorker())
            configure_worker(preview_worker, cards, MODELS / "value-v2.json")
        if preview_type:
            matchup = stack.enter_context(MatchupPreview(checkout, indexed, budget=1.2))
            learner = ((CardSpace(cards, experiment.profiles), load_model(learned_preview, cards))
                       if learned_preview else None)
        if "baseline70" in rivals:
            peer = stack.enter_context(FrozenSearch())
            configure_worker(peer, cards, MODELS / "value-v2.json")
            peer_proxy = HypotheticalReplies(stack, checkout, indexed)
        external = {name: stack.enter_context(ExternalPivot(checkout, name, cards)) for name in rivals if name in EXTERNAL}
        external.update({name: stack.enter_context(AlternatePivot(checkout, name, cards)) for name in rivals if name in STYLES})
        if QUICKSILVER in rivals:
            external[QUICKSILVER] = stack.enter_context(QuicksilverOpponent(quicksilver, checkout))
        for root in roots:
            records[root] = {}
            for rival in rivals:
                pair = []
                for seat in SEATS:
                    challenger = OpeningChallenger if deep_preview else Challenger
                    candidate = challenger(experiment, indexed, worker, BASE, learned, proxy=proxy, lineup_model=lineup)
                    if variant != "baseline70":
                        settings = BASE | {"worlds": 3 if variant == "posterior3" else 2,
                                           "move_budget": {"posterior2": 3.6, "posterior3": 4.3, "horizon": 4.6, "wide": 4.6}[variant]}
                        planner = {"horizon": PosteriorHorizon85, "wide": WideSearch85}.get(variant, Search85)
                        if research80.get("check_endgame"):
                            if variant != "wide":
                                raise ValueError("Endgame review is implemented on the wide candidate")
                            planner = CheckedEndgame
                        if variant == "wide":
                            settings["scout"] = scout
                        if reply_scout:
                            planner = ForecastReplies80
                            settings["reply_scout"] = reply_scout
                        candidate.battle = planner(experiment.profiles, worker, learned=learned, proxy=proxy,
                                                   lineup_model=lineup, **settings)
                        candidate.planning_counts = candidate.battle.counts
                    if preview_type:
                        candidate.preview = Preview85(candidate.preview, preview_worker, matchup, indexed, learner, review, lead_prior)
                    if draft_review:
                        candidate.draft = Draft85(experiment, preview_worker)
                    elif research80.get("draft_bundles"):
                        candidate.draft = DraftBundles(experiment, review_picks=(2, 4, 6, 8))
                    if rival == "baseline70":
                        foe = Challenger(experiment, indexed, peer, BASE, learned, proxy=peer_proxy, lineup_model=lineup)
                    elif rival == "champion":
                        foe = Champion(experiment, indexed)
                    else:
                        foe = opponent(rival, root, experiment.profiles, external.get(rival))
                    game = play(experiment, cards, root, seat, rival, foe, indexed, candidate)
                    if rival == QUICKSILVER:
                        game["quicksilver"] = dict(foe.metrics)
                    if preview_type:
                        game["preview85"] = dict(candidate.preview.counts)
                        if isinstance(candidate.preview.lead_prior, EnsembleLeadPrior):
                            game["preview85"]["independent_search_used"] = candidate.preview.lead_prior.used
                    if draft_review or research80.get("draft_bundles"):
                        game["draft85"] = dict(candidate.draft.counts)
                    pair.append(game)
                records[root][rival] = pair
    return records, process_time() - began


def measure(roots, cards, rivals, checkout, variant, workers, preview_type, learned_preview, position_model, draft_review,
            full_preview=False, lead_model=None, action_model=None, deep_preview=False, quicksilver=None, research80=None):
    batch_size = 1 if len(roots) < 64 else 3
    batches = [roots[i:i + batch_size] for i in range(0, len(roots), batch_size)]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(evaluate, batch, cards, rivals, checkout, variant, preview_type, learned_preview,
                               position_model, draft_review, full_preview, lead_model, action_model, deep_preview, quicksilver,
                               research80)
                   for batch in batches]
        for completed in as_completed(futures):
            yield completed.result()


def digests(checkout, learned_preview=None, position_model=None, lead_model=None, action_model=None, quicksilver=None):
    paths = [Path(__file__), Path(__file__).with_name("search85.py"), MODELS / "value-v2.json",
             MODELS / "lineup-belief.json", MODELS / "reply/config.json", MODELS / "reply/state_dict.pt"]
    paths.extend(Path(__file__).with_name(name) for name in ("preview85.py", "preview85.cjs", "matchup_preview_worker.py",
                                                            "matchup_preview.py", "combination_preview.py", "horizon85.py",
                                                            "draft85.py", "position85.cjs", "endgame85.cjs", "strength85.py",
                                                            "preview_review85.py", "lead85.py", "wide85.py", "action_scout85.py",
                                                            "deep_opening85.py", "opening_scout85.py"))
    paths.extend(sorted(FROZEN_PACKAGE.glob("*.cjs")))
    if learned_preview:
        paths.extend(learned_preview / name for name in ("state_dict.pt", "config.json"))
    if position_model:
        paths.append(position_model)
    if lead_model:
        paths.append(lead_model)
    if action_model:
        paths.extend(action_model / name for name in ("state_dict.pt", "config.json"))
    result = {str(path): sha256(path.read_bytes()).hexdigest() for path in paths}
    result.update(source_hashes())
    result["external"] = {name: sha256((checkout / name).read_bytes()).hexdigest()
                          for name in ("agent/agent.py", "agent/versions/pivot.py", "agent/pokemon/tuning.py", "sim/translate.py")}
    if quicksilver:
        adapter = Path(__file__).with_name("quicksilver.py")
        result[str(adapter)] = sha256(adapter.read_bytes()).hexdigest()
        result["external"]["quicksilver"] = {
            path.relative_to(quicksilver).as_posix(): sha256(path.read_bytes()).hexdigest()
            for directory in ("pokemon", "common", "altruagent")
            for path in sorted((quicksilver / directory).rglob("*"))
            if path.is_file() and path.suffix in (".py", ".json", ".js", ".csv")
            and "__pycache__" not in path.parts
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("baseline70", "posterior2", "posterior3", "horizon", "wide"), required=True)
    parser.add_argument("--opponents", nargs="+", choices=(*OPPONENTS, *STYLES, "champion", "baseline70", QUICKSILVER), required=True)
    parser.add_argument("--ace-trainer", type=Path, required=True)
    parser.add_argument("--quicksilver", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pairs", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-model", type=Path)
    parser.add_argument("--position-model", type=Path)
    parser.add_argument("--draft-review", action="store_true")
    parser.add_argument("--full-preview", action="store_true")
    parser.add_argument("--lead-model", type=Path)
    parser.add_argument("--action-model", type=Path)
    parser.add_argument("--deep-preview", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or min(args.pairs, args.workers) < 1:
        parser.error("Provide a fresh output and positive counts")
    if len(set(args.opponents)) != len(args.opponents):
        parser.error("Provide distinct opponents")
    if QUICKSILVER in args.opponents and not (args.quicksilver and (args.quicksilver / "pokemon/agent.py").is_file()):
        parser.error("Provide the public Quicksilver checkout")
    if args.action_model and args.variant != "wide":
        parser.error("The proposal checkpoint is used by the wide Battle variant")
    if args.deep_preview and (not args.lead_model or args.variant == "baseline70"):
        parser.error("Deep Preview needs the lead checkpoint and a research Battle variant")
    check_hybrid()
    cards = read(CATALOG)
    roots = list(range(args.seed, args.seed + args.pairs))
    pools = pools_for(cards, roots)
    verify_pools(pools, SEALED)
    for file in ("value-v2.json", "lineup-belief.json", "reply/config.json"):
        disjoint(pools, read(MODELS / file)["pools"])
    if args.preview_model:
        disjoint(pools, read(args.preview_model / "config.json").get("pools", {}))
    if args.position_model:
        disjoint(pools, read(args.position_model)["pools"])
    if args.lead_model:
        disjoint(pools, read(args.lead_model)["pools"])
    if args.action_model:
        disjoint(pools, read(args.action_model / "config.json")["pools"])
    preview_type = args.preview or bool(args.preview_model) or args.full_preview or bool(args.lead_model)
    models = (args.preview_model, args.position_model, args.lead_model, args.action_model)
    spec = digests(args.ace_trainer, *models, quicksilver=args.quicksilver)
    control = read(args.compare) if args.compare else None
    if control:
        if (control["pools"] != pools or any(name not in control["report"] for name in args.opponents)
                or control["spec"]["external"] != spec["external"]):
            parser.error("The comparison must cover the same pools and opponents")
    for path in args.output.parent.glob("*.json"):
        registered = read(path).get("pools", {})
        if not control or registered != pools:
            disjoint(pools, registered)
    began = perf_counter()
    Random(args.seed ^ 0x762E).shuffle(roots)
    records, python_seconds = {}, 0.
    for part, python_cpu in measure(roots, cards, args.opponents, args.ace_trainer, args.variant, args.workers,
                                    preview_type, args.preview_model, args.position_model, args.draft_review, args.full_preview,
                                    args.lead_model, args.action_model, args.deep_preview, args.quicksilver):
        records.update(part)
        python_seconds += python_cpu
        print(f"Completed {len(records)}/{args.pairs} paired pools", flush=True)
    if len(records) != args.pairs or digests(args.ace_trainer, *models, quicksilver=args.quicksilver) != spec:
        raise RuntimeError("Missing pairs or research inputs changed during the benchmark")
    records = dict(sorted(records.items()))
    report = {}
    for name in args.opponents:
        pairs = [row[name] for row in records.values()]
        metrics = summarize(pairs)
        planning = Counter()
        preview = Counter()
        draft = Counter()
        for pair in pairs:
            for game in pair:
                planning.update(game["planning"])
                preview.update(game.get("preview85", {}))
                draft.update(game.get("draft85", {}))
        metrics["planning"] = dict(planning)
        if preview_type:
            metrics["preview85"] = dict(preview)
        if args.draft_review:
            metrics["draft85"] = dict(draft)
        if name == QUICKSILVER:
            metrics["quicksilver"] = dict(sum((Counter(game["quicksilver"]) for pair in pairs for game in pair), Counter()))
        if control:
            metrics["paired_gain"] = paired_gain(records, control["records"], name)
        report[name] = metrics
    save(args.output, dict(variant=args.variant, preview=preview_type, preview_model=str(args.preview_model or ""),
                           position_model=str(args.position_model or ""), draft_review=args.draft_review, full_preview=args.full_preview,
                           lead_model=str(args.lead_model or ""), action_model=str(args.action_model or ""),
                           deep_preview=args.deep_preview,
                           sources=spec, spec={"external": spec["external"]},
                           catalog=fingerprint(cards), pools=pools, report=report, records=records,
                           seconds=perf_counter() - began, worker_python_cpu_seconds=python_seconds, api_spend_usd=0))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
