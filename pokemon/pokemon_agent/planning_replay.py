"""Compare Battle policies on saved deployments for development diagnostics."""

import argparse
from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from random import Random
from statistics import mean
from time import perf_counter

from .draft_last_pick import mean_ci, other
from .draft_leaf import CATALOG
from .draft_league import parallel, read, save
from .game import BATTLE, BATTLE_BANK, DECISION_LIMIT, PREVIEW
from .joint import JointPolicy
from .lineup import index_cards
from .planning_bench import configure_worker, source_hashes
from .planning_preview import PlanningPreview
from .proxy_replies import ProxyReplies
from .public_planning import PublicPlanner
from .reply_learning import LearnedReplies
from .reply_lineup import ReplyLineup
from .sim import SEATS, Showdown, play_battle
from .strength import ExternalPivot
from .tactics import load_card_profiles


class ReplayPolicy:
    def __init__(self, planner):
        self.planner = planner
        self.needs_proxy_evidence = planner.proxy is not None
        self.needs_own_proposals = planner.proposals

    def choose_with_public_history(self, decision, seat, evidence):
        return self.planner(decision, seat, evidence)


def trial(engine, root, game, roster, foe, policy, preview_worker=None, indexed=None):
    actor = game["actor"]
    challenger = ReplayPolicy(policy) if isinstance(policy, PublicPlanner) else policy
    players = {actor: challenger, other(actor): foe}
    foe.setup(root, other(actor), game["first"])
    bank = dict.fromkeys(SEATS, BATTLE_BANK)
    clocks = []
    preview = PlanningPreview(lambda decision: decision.validate(game["lineups"][actor]), preview_worker, indexed) if preview_worker else None

    def choose(seat, decision, view):
        if decision.phase == PREVIEW:
            action = decision.validate(game["lineups"][seat])
            if seat == actor:
                if isinstance(policy, PublicPlanner):
                    if preview:
                        action = preview(decision)
                    policy.observe_preview(decision, action)
                else:
                    policy.observe(decision)
            return action
        decision = replace(decision, seconds_left=min(DECISION_LIMIT[BATTLE], bank[seat]), bank_seconds_left=bank[seat])
        began = perf_counter()
        if seat != actor:
            action = foe.choose(decision, view)
        elif isinstance(challenger, ReplayPolicy):
            action = challenger.choose_with_public_history(decision, seat, view.get("search"))
        else:
            action = challenger(decision)
        elapsed = perf_counter() - began
        bank[seat] -= elapsed
        if seat == actor:
            clocks.append(elapsed * 1000)
        if bank[seat] <= 0 or elapsed >= decision.seconds_left:
            raise RuntimeError("Battle diagnostic exceeded a time budget")
        return decision.validate(action)

    winner = play_battle(engine, roster, players, root, game["battle_seed"], choose)
    return {"winner": winner, "actor": actor, "clocks_ms": clocks,
            "planning": dict(policy.counts) if isinstance(policy, PublicPlanner) else {},
            "preview": dict(preview.counts) if preview else {}}


def evaluate(roots, cards, checkout, source, rivals, settings, reply_model, proxy_replies, preview_search, value_model, lineup_model):
    records = {}
    by_id = {card["card_id"]: card for card in cards}
    profiles = load_card_profiles(cards, with_sets=True)
    learned = LearnedReplies(reply_model, profiles) if reply_model else None
    with ExitStack() as stack:
        engine = stack.enter_context(Showdown())
        worker = stack.enter_context(Showdown(public_search=True))
        configure_worker(worker, cards, value_model)
        indexed = index_cards(cards)
        proxy = ProxyReplies(stack, checkout, indexed) if proxy_replies else None
        lineup = ReplyLineup(lineup_model, cards, profiles) if lineup_model else None
        opponents = {name: stack.enter_context(ExternalPivot(checkout, name, cards)) for name in rivals}
        for root in roots:
            records[root] = {}
            for rival in rivals:
                pair = []
                for game in source[str(root)][rival]:
                    roster = {seat: [by_id[card_id] for card_id in ids] for seat, ids in game["roster"].items()}
                    control = trial(engine, root, game, roster, opponents[rival], JointPolicy(profiles))
                    candidate = trial(engine, root, game, roster, opponents[rival],
                                      PublicPlanner(profiles, worker, learned=learned, proxy=proxy, lineup_model=lineup, **settings),
                                      worker if preview_search else None, indexed)
                    pair.append({"control": control, "candidate": candidate})
                records[root][rival] = pair
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--ace-trainer", type=Path, required=True)
    parser.add_argument("--opponents", nargs="+", choices=("ace-code", "pivot"), default=("ace-code", "pivot"))
    parser.add_argument("--pairs", type=int, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=291)
    parser.add_argument("--worlds", type=int, default=2)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--edge", type=float, default=.08)
    parser.add_argument("--depth", type=int, choices=(0, 1, 2, 3), default=1)
    parser.add_argument("--reply-model", type=Path)
    parser.add_argument("--proxy-replies", action="store_true")
    parser.add_argument("--interaction", action="store_true")
    parser.add_argument("--preview-search", action="store_true")
    parser.add_argument("--adaptive-replies", action="store_true")
    parser.add_argument("--adaptive-all", action="store_true")
    parser.add_argument("--replacement-search", action="store_true")
    parser.add_argument("--value-model", type=Path)
    parser.add_argument("--lineup-model", type=Path)
    parser.add_argument("--model-proposals", action="store_true")
    parser.add_argument("--move-budget", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = read(args.source)
    if args.output.exists() or not 1 <= args.pairs <= len(source["records"]) or args.workers < 1:
        parser.error("Provide a fresh output and valid positive counts")
    if (args.adaptive_replies or args.adaptive_all) and not args.proxy_replies:
        parser.error("Adaptive reply weighting requires the proxy ensemble")
    if args.model_proposals and not args.proxy_replies:
        parser.error("Model proposals require the proxy ensemble")
    if args.move_budget is not None and args.move_budget <= 0:
        parser.error("The search budget must be positive")
    roots = Random(args.seed).sample(sorted(map(int, source["records"])), args.pairs)
    settled = {str(root): source["records"][str(root)] for root in roots}
    settings = dict(worlds=args.worlds, samples=args.samples, candidate_limit=args.candidates, min_edge=args.edge,
                    depth=args.depth, interaction=args.interaction, adaptive=args.adaptive_replies,
                    proposals=args.model_proposals, move_budget=args.move_budget, adaptive_all=args.adaptive_all,
                    replacements=args.replacement_search)
    began = perf_counter()
    sources = source_hashes()
    cards = read(CATALOG)
    records = {}
    for part in parallel(evaluate, roots, args.workers, cards, args.ace_trainer, settled, args.opponents, settings,
                         args.reply_model, args.proxy_replies, args.preview_search, args.value_model, args.lineup_model):
        records.update(part)
        print(f"Completed {len(records)}/{args.pairs} paired deployments", flush=True)
    if source_hashes() != sources:
        raise RuntimeError("Planning sources changed during the replay")
    report = {}
    for rival in args.opponents:
        paired = {mode: [] for mode in ("control", "candidate")}
        clocks, counts = [], Counter()
        for pair in records.values():
            rows = pair[rival]
            for mode in paired:
                paired[mode].append(sum(row[mode]["winner"] == row[mode]["actor"] for row in rows))
            for row in rows:
                clocks.extend(row["candidate"]["clocks_ms"])
                counts.update(row["candidate"]["planning"])
                counts.update({f"preview_{key}": value for key, value in row["candidate"]["preview"].items()})
        deltas = [(cand - base) / 2 for cand, base in zip(paired["candidate"], paired["control"], strict=True)]
        clocks.sort()
        report[rival] = dict(games=2 * args.pairs, wins={mode: sum(values) for mode, values in paired.items()},
                             paired_gain=mean(deltas), paired_gain_95=mean_ci(deltas),
                             battle_p95_ms=clocks[int(.95 * (len(clocks) - 1))], stats=dict(counts))
    save(args.output, dict(report=report, source=str(args.source), sources=sources, settings=settings,
                           preview_search=args.preview_search, proxy_replies=args.proxy_replies,
                           reply_model=str(args.reply_model) if args.reply_model else None,
                           models=[str(path) for path in (args.value_model, args.lineup_model) if path],
                           seconds=perf_counter() - began, records=records, api_spend_usd=0))
    print(report, flush=True)


if __name__ == "__main__":
    main()
