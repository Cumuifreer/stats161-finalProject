"""Review public lineup proposals in longer, paired local rollouts."""

from collections import Counter
from pathlib import Path
from random import Random
import subprocess
from time import perf_counter

import torch

from .combination_preview import distinct_options, encode_action, encode_rows, encode_team, signature
from .game import PREVIEW, ident, make_decision
from .opening_scout85 import propose as forecast_openers
from .planning_preview import diverse_deployments, publish_sets
from .sim import FORMAT, Showdown, fingerprint
from .tactics import tactical_actions


class PreviewWorker(Showdown):
    def __enter__(self):
        self.process = subprocess.Popen(["node", str(Path(__file__).with_suffix(".cjs"))],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        return self


def reverse_view(decision, catalog):
    board = publish_sets(decision.observation, catalog)
    own, foe = [[catalog[entry["species"]] for entry in board[key]] for key in ("team", "opponent_team")]
    return make_decision(PREVIEW, {"your_roster": foe, "opponent_roster": own},
                         [{"roster": [ident(card["species"]) for card in foe]}], "p2")


def unique(actions):
    result = {}
    for action in actions:
        result.setdefault(signature(action), action)
    return list(result.values())


@torch.inference_mode()
def neural_rank(decision, replies, space, net):
    own, foe = [encode_team(space, decision.observation, key) for key in ("team", "opponent_team")]
    options = list(distinct_options(decision).values())
    actions = [encode_action(space, action) for action in options]
    opposing = [encode_action(space, action) for action in replies]
    roles, features = encode_rows(space, (own, foe), actions, opposing)
    teams = torch.tensor([[own, foe]]).expand(len(roles), -1, -1)
    estimates = net(teams, torch.tensor(roles), torch.tensor(features, dtype=torch.float32)).sigmoid()
    estimates = estimates.view(len(actions), len(opposing))
    values = .85 * estimates.mean(1) + .15 * estimates.min(1).values
    return [options[i] for i in values.argsort(descending=True).tolist()]


def independent_draws(decision, name, samples):
    seed = fingerprint([name, decision.observation])
    rng = Random(seed)
    return [[rng.randrange(65536) for _ in range(4)] for _ in range(samples)]


class Preview85:
    def __init__(self, anchor, worker, matchup, indexed, learner=None, reviewer=None, lead_prior=None):
        self.anchor = anchor
        self.worker = worker
        self.matchup = matchup
        self.indexed = indexed
        self.learner = learner
        self.reviewer = reviewer
        self.lead_prior = lead_prior
        self.opposition_forecast = ()
        self.counts = Counter()

    @property
    def fallbacks(self):
        return self.anchor.fallbacks + self.counts["fallbacks"]

    def __call__(self, decision):
        started = perf_counter()
        anchor = self.anchor(decision)
        try:
            extra_seconds = getattr(self.lead_prior, "extra_seconds", 0.)
            reverse = reverse_view(decision, self.indexed)
            replies = tactical_actions(reverse)
            predicted = self.matchup.proposals(reverse, top=4)
            if self.lead_prior:
                opposition, weights = self.lead_prior.scenarios(reverse, predicted, replies)
                scenario_counts = (3, 6, 8)
            else:
                opposition = unique([*predicted[:2], replies[0], *predicted[2:], *diverse_deployments(replies, replies[0], 5)])[:4]
                weights, scenario_counts = [.36, .26, .22, .16], (3, 4, 4)
            self.opposition_forecast = tuple(zip(opposition, weights))
            mine = self.matchup.proposals(decision, top=6)
            independent = (self.lead_prior.own_shortlist(decision)
                           if hasattr(self.lead_prior, "own_shortlist") else [])
            ranked = tactical_actions(decision)
            learned = neural_rank(decision, opposition, *self.learner)[:7] if self.learner else []
            diversity = diverse_deployments(ranked, anchor, 12)
            nominating = unique([anchor, *independent, *mine, *diversity]) if extra_seconds else []
            opening = (forecast_openers(self.lead_prior.space, decision, ranked, nominating,
                                        opposition, weights, limit=9) if extra_seconds else [])
            options = unique([anchor, *independent, *mine, *opening, *learned, *diversity])[:22 if extra_seconds else 18]
            clock_limit = (13. if self.lead_prior else 10.) + extra_seconds
            seconds = min(clock_limit, (decision.seconds_left or 85.) - 4.) - (perf_counter() - started)
            if seconds < .5 or len(options) < 2:
                return anchor
            checked = self.worker.call("compare", options=options, opponents=opposition, weights=weights,
                                       seeds=independent_draws(decision, "preview-v2", 7), format=FORMAT,
                                       stages=[dict(survivors=7, depth=3, start=0, samples=1, opponentCount=scenario_counts[0]),
                                               dict(survivors=4, depth=7, start=1, samples=2, opponentCount=scenario_counts[1]),
                                               dict(survivors=1, depth=15, start=3, samples=3 if self.lead_prior else 4,
                                                    opponentCount=scenario_counts[2])],
                                       budget_ms=seconds * 1000, edge=.09)
        except (KeyError, ValueError, RuntimeError, OSError):
            self.counts["fallbacks"] += 1
            return anchor
        self.counts["rollouts"] += checked["calls"]
        self.counts["terminal_rollouts"] += checked["terminals"]
        self.counts["opponent_scenarios"] += len(opposition)
        self.counts["budget_reached"] += checked["timedOut"]
        selected = checked["index"]
        if self.reviewer:
            self.reviewer.counts.clear()
            selected = self.reviewer(decision, options, opposition, checked,
                                     started + min(24. + extra_seconds, (decision.seconds_left or 85.) - 3.))
            self.counts.update(self.reviewer.counts)
        self.counts["changed"] += selected != 0
        choice = options[selected]
        self.counts["learned_lineups_chosen"] += choice != anchor and choice in learned
        if independent:
            self.counts["independent_own_lineups_offered"] += len(independent)
            self.counts["independent_own_lineups_chosen"] += choice != anchor and choice in independent and choice not in mine
        self.counts["forecast_openers_chosen"] += choice != anchor and choice in opening and choice not in mine
        return choice
