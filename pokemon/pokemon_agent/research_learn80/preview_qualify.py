"""Run complete seat-paired matches with only the player's Preview policy changed."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from multiprocessing import get_context
from pathlib import Path
from random import Random
from statistics import fmean
from time import perf_counter

import torch

from ..battle_league import configure_torch
from ..battle_value import opponent
from ..blind85 import used_pools
from ..combination_preview import encode_action, encode_rows, encode_team
from ..combination_space import CardSpace
from ..draft85 import deployment
from ..draft_forecast import public_draft
from ..draft_leaf import CATALOG
from ..draft_league import pools_for, read, save
from ..game import DRAFT, PREVIEW
from ..joint import JointPolicy
from ..lineup import index_cards
from ..planning_preview import diverse_deployments
from ..preview85 import neural_rank, reverse_view
from ..roster import DraftRanker, _load_model
from ..sim import SEATS, Showdown, fingerprint, play_game
from ..tactics import load_card_profiles, tactical_actions
from .preview_outcome import PreviewOutcome
from .qualify import summarize


class LearnedPreview(JointPolicy):
    def __init__(self, profiles, indexed, space, model):
        super().__init__(profiles)
        self.indexed = indexed
        self.space = space
        self.model = model
        self.changed = 0

    def __call__(self, decision):
        baseline = super().__call__(decision)
        if decision.phase != PREVIEW:
            return baseline
        reverse = reverse_view(decision, self.indexed)
        ranked = tactical_actions(reverse)
        alternatives = diverse_deployments(ranked, ranked[0], 4)
        choice = neural_rank(decision, alternatives, self.space, self.model)[0]
        self.changed += choice != baseline
        return choice


class LearnedDraft(JointPolicy):
    def __init__(self, profiles, cards, space, model, projection):
        super().__init__(profiles)
        self.project = projection
        indexed = {card["card_id"]: card for card in cards}
        self.template = [indexed[key] for key in self.project.index]
        self.space = space
        self.model = model
        self.changed = 0
        self.reviewed = 0

    @torch.inference_mode()
    def compare_rosters(self, own, foe):
        deployment_own, deployment_foe = deployment(own, foe), deployment(foe, own)
        ranked = [tactical_actions(decision) for decision in (deployment_own, deployment_foe)]
        options = [diverse_deployments(actions, actions[0], n) for actions, n in zip(ranked, (5, 3), strict=True)]
        teams = [encode_team(self.space, deployment_own.observation, name) for name in ("team", "opponent_team")]
        own_actions, foe_actions = ([encode_action(self.space, action) for action in choices] for choices in options)
        roles, traits = encode_rows(self.space, teams, own_actions, foe_actions)
        expected = self.model(torch.tensor(teams).expand(len(roles), -1, -1), torch.tensor(roles),
                              torch.tensor(traits, dtype=torch.float32)).sigmoid().reshape(len(own_actions), -1)
        return float((.8 * expected.mean(1) + .2 * expected.min(1).values).max())

    def __call__(self, decision):
        started = perf_counter()
        baseline = super().__call__(decision)
        if decision.phase != DRAFT or decision.observation["pick"] < 9 or len(decision.actions) < 2:
            return baseline
        return self.review(decision, baseline, tactical_actions(decision)[:3], started)

    def review(self, decision, baseline, ranked, started):
        pool, own, foe, available, pick, order = public_draft(decision, self.project.index)
        rng = Random(fingerprint(["terminal-draft-v1", pool, own, foe, pick]))
        noise = [[[rng.gauss(0., sigma) for _ in self.template] for sigma in (spread, spread)] for spread in (0., .24, .40)]
        projections = [[self.project._finish(available, (own, foe), self.project.index[action["card_id"]], order, pick,
                                              perturbation, scenario)
                        for scenario, perturbation in enumerate(noise)] for action in ranked]
        comparable = [index for index, column in enumerate(zip(*projections, strict=True)) if all(pair is not None for pair in column)]
        if not comparable:
            return baseline
        values = []
        for hypotheses in projections:
            row = []
            for index in comparable:
                if perf_counter() - started >= min(2.5, (decision.seconds_left or 15.) - 1.):
                    return baseline
                teams = [[self.template[card] for card in roster] for roster in hypotheses[index]]
                row.append(self.compare_rosters(*teams))
            values.append(row)
        best = max(range(1, len(values)), key=lambda index: fmean(values[index]))
        gains = [selected - first for selected, first in zip(values[best], values[0], strict=True)]
        self.reviewed += 1
        if fmean(gains) >= .045 and min(gains) >= -.015:
            self.changed += 1
            return ranked[best]
        return baseline


def evaluate(roots, cards, checkpoint, phase):
    configure_torch()
    config = read(checkpoint / "config.json")
    if config["catalog"] != fingerprint(cards):
        raise ValueError("Card catalog differs from training")
    net = PreviewOutcome(**config["network"]).eval()
    net.load_state_dict(torch.load(checkpoint / "state_dict.pt", map_location="cpu", weights_only=True))
    profiles = load_card_profiles(cards, with_sets=True)
    space, indexed = CardSpace(cards, profiles), index_cards(cards)
    projection = DraftRanker(cards, *_load_model(Path("checkpoints/roster-v1/expanded"), "cpu")) if phase == DRAFT else None
    results = {}
    with Showdown() as engine:
        for root in roots:
            matchup = "forecast" if root % 2 == 0 else "forecast-replies"
            arms = []
            for use_model in (False, True):
                pair = []
                for actor, foe in (SEATS, SEATS[::-1]):
                    if use_model and phase == DRAFT:
                        challenger = LearnedDraft(profiles, cards, space, net, projection)
                    else:
                        challenger = LearnedPreview(profiles, indexed, space, net) if use_model else JointPolicy(profiles)
                    samples = defaultdict(list)
                    winner = play_game(engine, cards, {actor: challenger, foe: opponent(profiles, matchup)}, root, samples)
                    if any(event for times in samples.values() for _, _, event in times):
                        raise RuntimeError(f"Clock event in Preview pool {root}")
                    pair.append(dict(won=winner == actor, drew=winner is None,
                                     moving_ms=[ms for step, ms, _ in samples[actor] if step == phase],
                                     n_scored=getattr(challenger, "reviewed", int(use_model)), chose_diff=getattr(challenger, "changed", 0)))
                arms.append(pair)
            results[root] = {matchup: arms}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pools", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--phase", choices=[DRAFT, PREVIEW], default=PREVIEW)
    args = parser.parse_args()
    if args.output.exists() or args.pools < 4 or args.workers < 1:
        parser.error("Use a fresh output and at least four pools")
    cards = read(CATALOG)
    roots = list(range(args.seed, args.seed + args.pools))
    pools = pools_for(cards, roots)
    if not used_pools(cards).isdisjoint(pools.values()):
        raise ValueError("Qualification must use fresh pools")
    save(args.output / "sample.json", dict(pools=pools, phase=args.phase,
                                         checkpoint=sha256((args.checkpoint / "state_dict.pt").read_bytes()).hexdigest()))
    started = perf_counter()
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
        for part in pool.map(evaluate, [roots[i::args.workers] for i in range(args.workers)],
                             [cards] * args.workers, [args.checkpoint] * args.workers, [args.phase] * args.workers):
            results.update(part)
    report = {}
    for rival in ("forecast", "forecast-replies"):
        games = [row for row in results.values() if rival in row]
        report[rival] = {name: summarize(games, index, rival) for index, name in enumerate(("joint", "learned"))}
        for metrics in report[rival].values():
            metrics["draft_ms_p95" if args.phase == DRAFT else "preview_ms_p95"] = metrics.pop("battle_ms_p95")
    save(args.output / "matches.json", dict(report=report, pools=pools, results=results, wall_seconds=perf_counter() - started))
    print(report, flush=True)


if __name__ == "__main__":
    main()
