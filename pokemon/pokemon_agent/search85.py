"""Belief-weighted search using the published roster and the moving player's history."""

from collections import Counter
from dataclasses import replace
from math import exp, log
from random import Random
from time import perf_counter

from .deep_opening85 import condition_reserves
from .game import BATTLE, BATTLE_BANK, DECISION_LIMIT, ident, make_decision
from .planning_refine import mix_replies, survivors
from .proxy_replies import ProxyReplies
from .public_planning import PublicPlanner, _diverse, _ordered
from .reply_belief import likelihood, public_atoms
from .search import _key
from .sim import FORMAT, SEATS, fingerprint
from .tactics import tactical_actions


PRIOR = {"neutral": .44, "ace-code": .28, "pivot": .28}


def normalize(values):
    total = sum(values.values())
    return {key: weight / total for key, weight in values.items()}


def action_mass(actions, weights):
    return {_key(action): weight for action, weight in zip(actions, weights, strict=True) if weight}


class HypotheticalReplies(ProxyReplies):
    def forecast(self, decision, seat, rosters, view, public_log, deadline):
        # Each sample restarts a published policy. The actual match RNG and identity are absent.
        options = {"pivot": {**view["pivot"], "protocol_log": public_log}, "templates": view["templates"]}
        rng = Random(fingerprint(["local-reply-v2", seat, public_log]))
        decision = replace(decision, seconds_left=DECISION_LIMIT[BATTLE], bank_seconds_left=BATTLE_BANK)
        drafted = {player: [self.cards[mon] for mon in rosters[player]] for player in SEATS}
        result = {}
        for name, model in self.models.items():
            counts = Counter()
            for _ in range(1 if name == "ace-code" else 3):
                if perf_counter() >= deadline:
                    break
                seed = rng.getrandbits(48)
                try:
                    model.setup(seed, seat, "p1")
                    model.start(seed, seat, drafted)
                    counts[_key(model.choose(decision, options))] += 1
                except (ValueError, RuntimeError, OSError):
                    break
            if counts:
                result[name] = normalize(counts)
        return result


class Search85(PublicPlanner):
    def __init__(self, *args, **settings):
        super().__init__(*args, **settings)
        self.deployment_logit = None
        self.opening_forecast = ()
        self.expert_log_likelihood = Counter()
        self.expert_prior = PRIOR
        self.world_log_likelihood = Counter()
        self.pending = {}
        self.cursor = None

    def _deployments(self, foe, events):
        revealed = {}
        for event in events:
            fields = event.split("|")
            if fields[1] in {"switch", "drag"} and fields[2].startswith(foe):
                revealed[ident(fields[3].split(",", 1)[0])] = None
        leads = set(list(revealed)[:2])
        if self.deployment_logit is None:
            ordered = {}
            for index, action in enumerate(self.foe_previews):
                team = tuple(sorted(action["bring"]))
                if not leads.issubset(team):
                    continue
                score = index - 11 * len(leads.intersection(action["leads"]))
                ordered[team] = min(score, ordered.get(team, score))
            ranked = sorted(ordered, key=ordered.__getitem__)
            learned = (self.lineup_model.scores(ranked, sorted(leads), self.our_roster, self.our_leads)
                       if ranked and self.lineup_model and len(leads) == 2 else [0.] * len(ranked))
            self.deployment_logit = {team: score - .28 * i for i, (team, score) in enumerate(zip(ranked, learned))}
            if self.opening_forecast:
                updated = condition_reserves(self.deployment_logit, leads, self.opening_forecast)
                self.counts["preview_reserve_updates"] += updated is not self.deployment_logit
                self.deployment_logit = updated
        choices = {team: exp(logit + self.world_log_likelihood[team]) for team, logit in self.deployment_logit.items()
                   if revealed.keys() <= set(team)}
        if not choices:
            return []
        weights = normalize(choices)
        return sorted(weights.items(), key=lambda entry: entry[1], reverse=True)

    def _observe(self, turn, events, foe):
        if self.cursor is None or turn <= self.cursor[0]:
            return
        atoms = public_atoms(events[self.cursor[1]:], foe)
        if atoms and self.pending:
            prior = self._expert_weights(self.expert_prior)
            aggregated = Counter()
            totals = Counter()
            for team, (mass, distributions) in self.pending.items():
                observed = {name: likelihood(values, atoms) for name, values in distributions.items()}
                experts = normalize({name: prior[name] for name in observed})
                evidence = sum(experts[name] * score for name, score in observed.items())
                reference = observed["neutral"]
                self.world_log_likelihood[team] = .87 * self.world_log_likelihood[team] + .65 * log(evidence / reference)
                for name, score in observed.items():
                    aggregated[name] += mass * score
                    totals[name] += mass
            for name, score in aggregated.items():
                self.expert_log_likelihood[name] = .84 * self.expert_log_likelihood[name] + log(score / totals[name])
            self.counts["posterior_updates"] += 1
        self.pending.clear()

    def _expert_weights(self, available):
        baseline = normalize({name: self.expert_prior[name] for name in available})
        learned = normalize({name: weight * exp(self.expert_log_likelihood[name]) for name, weight in baseline.items()})
        return {name: .25 * baseline[name] + .75 * weight for name, weight in learned.items()}

    def _supplemental_replies(self, enemy, foe, rosters, view, evidence, deadline):
        return {}

    def _world(self, decision, seat, foe, evidence, rosters, team, world_id, expected, deadline):
        lineups = {seat: self.our_lineup, foe: team}
        answer = self.worker.call("prepare", format=FORMAT, seat=seat, observation=decision.observation,
                                  rosters=[rosters[p] for p in SEATS], lineups=[lineups[p] for p in SEATS],
                                  events=evidence["events"], own_pp=evidence["own_pp"], world=world_id, proxy_seat=foe)
        if reason := answer.get("reason"):
            self.counts["prepare_" + reason] += 1
            return None
        views = {view["seat"]: view for view in answer["views"] if view}
        if views.keys() != set(SEATS):
            return None
        try:
            states = {p: make_decision(BATTLE, view["raw"], view["templates"], p) for p, view in views.items()}
        except ValueError:
            self.counts["empty_rebuilt_menu"] += 1
            return None
        if {_key(action) for action in states[seat].actions} != expected:
            self.counts["legal_mask_mismatch"] += 1
            return None
        enemy = states[foe]
        replies, weights = self._replies(enemy, seat, foe, evidence)
        distributions = {"neutral": action_mass(replies, weights)}
        distributions.update(self.proxy.forecast(enemy, foe, rosters, views[foe], evidence["protocol_log"], deadline))
        distributions.update(self._supplemental_replies(enemy, foe, rosters, views[foe], evidence, deadline))
        experts = self._expert_weights(distributions)
        mass = Counter()
        for name, probabilities in distributions.items():
            for action, probability in probabilities.items():
                mass[action] += experts[name] * probability
        mixture = [mass[_key(action)] for action in enemy.actions]
        replies, weights = mix_replies(enemy, replies, weights, mixture, 1., limit=7)
        return (world_id, replies, weights), distributions

    def _plan(self, decision, seat, evidence, ranked, coarse):
        baseline = ranked[0]
        deadline = perf_counter() + min(self.move_budget, (decision.seconds_left or 50.) - 1.)
        if perf_counter() + .10 >= deadline:
            return self._fallback("clock_fallback", baseline)
        foe = next(player for player in SEATS if player != seat)
        events = evidence["events"]
        turn = decision.observation["turn"]
        self._observe(turn, events, foe)
        self.cursor = (turn, len(events))
        rosters = {seat: self.our_roster, foe: self.joint.history.opponent_roster}
        extra = self._proposals(decision, seat, rosters, evidence)
        informed = tactical_actions(decision, card_profiles=self.profiles)
        alternatives = _diverse(_ordered(ranked[:1], extra, ranked[:2], informed[:3], ranked, informed, coarse),
                                self.candidate_limit)
        expected = {_key(action) for action in decision.actions}
        rng = Random(fingerprint(["public-planning-v1", seat, events]))
        seeds = [[rng.randrange(65536) for _ in range(4)]]
        completed = []
        for world_id, (team, weight) in enumerate(self._deployments(foe, events)[:self.worlds]):
            if perf_counter() >= deadline:
                break
            setup = self._world(decision, seat, foe, evidence, rosters, team, world_id, expected, deadline - .25)
            if setup is None:
                continue
            world, distributions = setup
            self.pending[team] = weight, distributions
            scores = self._score(world, seat, alternatives, seeds, deadline, 1)
            if scores is not None:
                completed.append((world, weight, scores))
        if not completed:
            return self._fallback("no_completed_world", baseline)
        self.counts["searched"] += 1
        coarse_scores = weighted_advantages([(weight, scores) for _, weight, scores in completed])
        indices = survivors(coarse_scores, 4)
        options = [alternatives[i] for i in indices]
        seeds = [[rng.randrange(65536) for _ in range(4)] for _ in range(2)]
        confirmation = []
        for world, weight, _ in completed:
            scores = self._score(world, seat, options, seeds, deadline, 2)
            if scores is not None:
                confirmation.append((weight, scores))
        if confirmation:
            self.counts["confirmed"] += 1
            scores = [.4 * coarse_scores[i] + .6 * value
                      for i, value in zip(indices, weighted_advantages(confirmation), strict=True)]
        else:
            self.counts["confirmation_fallback"] += 1
            options, scores = alternatives, coarse_scores
        best = max(range(len(scores)), key=scores.__getitem__)
        if scores[best] < self.min_edge:
            return baseline
        self.counts["changed"] += 1
        self.counts["model_proposal_chosen"] += options[best] in extra
        return options[best]


def weighted_advantages(results):
    total = sum(weight for weight, _ in results)
    offsets = [(weight / total, [score - values[0] for score in values]) for weight, values in results]
    return [sum(weight * scores[i] for weight, scores in offsets)
            - .08 * (max(scores[i] for _, scores in offsets) - min(scores[i] for _, scores in offsets))
            for i in range(len(results[0][1]))]
