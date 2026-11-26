"""Doubles planning over Showdown worlds reconstructed from one player's evidence."""

from collections import Counter
from dataclasses import replace
from itertools import combinations
from random import Random
from statistics import fmean
from time import perf_counter

from .forced_position import replacement
from .game import BATTLE, ident, make_decision
from .joint import JointPolicy, PROTECT
from .planning_preview import resolve_species
from .planning_refine import average_advantages, mix_replies, robust_scores, survivors
from .reply_belief import ReplyBelief
from .search import _key
from .sim import FORMAT, SEATS, fingerprint
from .tactics import SLOTS, SPEED_CONTROL, tactical_actions, tactical_candidates


def _diverse(ranked, limit):
    selected = list(ranked[:min(3, limit)])
    for category in (
        lambda slot: slot.get("move_id") in PROTECT,
        lambda slot: slot["type"] == "switch",
        lambda slot: slot.get("move_id") == "fakeout",
        lambda slot: slot.get("move_id") in SPEED_CONTROL,
        lambda slot: slot.get("move_id") in {"spore", "yawn", "followme", "ragepowder", "encore", "nastyplot", "calmmind"},
    ):
        for slot in SLOTS:
            if len(selected) == limit:
                return selected
            if any(category(action[slot]) for action in selected):
                continue
            choice = next((action for action in ranked if category(action[slot])), None)
            if choice:
                selected.append(choice)
    return (selected + [action for action in ranked if action not in selected])[:limit]


def _ordered(*groups):
    actions = {}
    for group in groups:
        for action in group:
            actions.setdefault(_key(action), action)
    return list(actions.values())


def _reply_distribution(decision, joint, profiles):
    tactical = tactical_actions(decision, card_profiles=profiles)
    strategic = [action for action, _ in joint.rank(decision, tactical_candidates(decision, ranked=tactical))]
    weights = Counter()
    for action, weight in zip(strategic[:3], (.31, .13, .06)):
        weights[_key(action)] += weight
    for action, weight in zip(tactical[:2], (.20, .08)):
        weights[_key(action)] += weight
    alternatives = _diverse(_ordered(strategic[:2], tactical), 8)
    for action in alternatives[3:]:
        weights[_key(action)] += .07
    actions = [action for action in _ordered(strategic[:3], tactical[:2], alternatives) if weights[_key(action)]]
    total = sum(weights.values())
    return actions, [weights[_key(action)] / total for action in actions]


class PublicPlanner:
    def __init__(self, profiles, worker, *, worlds=2, samples=2, candidate_limit=12, min_edge=.08, depth=1,
                 learned=None, proxy=None, interaction=False, lineup_model=None, adaptive=False,
                 proposals=False, move_budget=None, adaptive_all=False, replacements=False):
        self.joint = JointPolicy(profiles)
        self.profiles = profiles
        self.worker = worker
        self.worlds = worlds
        self.samples = samples
        self.candidate_limit = candidate_limit
        self.min_edge = min_edge
        self.depth = depth
        self.learned = learned
        self.proxy = proxy
        self.interaction = interaction
        self.lineup_model = lineup_model
        self.adaptive_all = adaptive_all
        self.belief = ReplyBelief({"neutral": .55, "ace-code": .225, "pivot": .225}) if adaptive_all else (
            ReplyBelief(("ace-code", "pivot")) if adaptive else None)
        self.proposals = proposals
        self.move_budget = move_budget
        self.replacements = replacements
        self.counts = Counter()
        self.our_lineup = None

    def observe_preview(self, decision, chosen):
        board = {
            side: [mon | {"species": resolve_species(mon["species"], self.profiles)} for mon in team]
            for side, team in decision.observation.items()
        }
        self.joint.observe(replace(decision, observation=board))
        self.our_roster = tuple(mon["species"] for mon in board["team"])
        self.our_lineup = tuple(chosen["bring"])
        self.our_leads = tuple(chosen["leads"])
        if self.learned or self.depth == 3:
            def known_set(mon):
                profile = self.profiles.get(mon["species"], {})
                return mon | {"moves": profile.get("card_moves", ()), "item": profile.get("card_item", ""),
                              "ability": profile.get("card_ability", "")}

            board = {side: [known_set(mon) for mon in team] for side, team in board.items()}
        foe_roster = self.joint.history.opponent_roster
        possible = [
            dict(type="select_lineup", bring=list(bring), leads=list(leads))
            for bring in combinations(foe_roster, 4) for leads in combinations(bring, 2)
        ]
        reverse = replace(decision, observation={"team": board["opponent_team"], "opponent_team": board["team"]},
                          actions=tuple(possible))
        self.foe_previews = tactical_actions(reverse)

    def _lineups(self, foe, evidence):
        revealed = {}
        for line in evidence["events"]:
            fields = line.split("|")
            if fields[1] in {"switch", "drag"} and fields[2].startswith(foe):
                revealed.setdefault(ident(fields[3].split(",", 1)[0]), None)
        lead = set(list(revealed)[:2])
        best = {}
        for index, action in enumerate(self.foe_previews):
            lineup = frozenset(action["bring"])
            if not revealed.keys() <= lineup:
                continue
            matching = len(lead.intersection(action["leads"]))
            score = index - 11 * matching
            if lineup not in best or score < best[lineup][0]:
                best[lineup] = score, action["bring"]
        options = [lineup for _, lineup in sorted(best.values())]
        if self.lineup_model and len(lead) == 2 and len(options) > 1:
            belief = self.lineup_model.scores(options, sorted(lead), self.our_roster, self.our_leads)
            ranking = sorted(range(len(options)), key=lambda i: belief[i] - .28 * i, reverse=True)
            options = [options[i] for i in ranking]
        return options[:self.worlds]

    def _fallback(self, reason, baseline):
        self.counts[reason] += 1
        return baseline

    def __call__(self, decision, seat, evidence):
        self.counts["decisions"] += 1
        if self.depth == 0:
            return self.joint(decision)
        coarse = tactical_actions(decision)
        shortlist = tactical_candidates(decision, ranked=coarse)
        ranked = [action for action, _ in self.joint.rank(decision, shortlist)]
        baseline = ranked[0]
        if self.replacements and len(decision.actions) > 1 and evidence and evidence["force_switch"]:
            chosen = replacement(decision, baseline, self.profiles)
            self.counts["replacement_searches"] += 1
            self.counts["replacement_changes"] += chosen != baseline
        elif len(decision.actions) == 1 or evidence is None or evidence["force_switch"] or self.our_lineup is None:
            chosen = self._fallback("one_or_replacement", baseline)
        else:
            chosen = self._plan(decision, seat, evidence, ranked, coarse)
        self.joint.record(decision, chosen)
        return chosen

    def _plan(self, decision, seat, evidence, ranked, coarse):
        baseline = ranked[0]
        foe = next(player for player in SEATS if player != seat)
        turn = decision.observation["turn"]
        if self.belief and self.belief.observe(turn, evidence["events"], foe):
            self.counts["belief_updates"] += 1
        informed = tactical_actions(decision, card_profiles=self.profiles)
        rosters = {seat: self.our_roster, foe: self.joint.history.opponent_roster}
        budget = self.move_budget if self.move_budget is not None else (3.6 if self.depth == 3 else 2.)
        started = perf_counter()
        extra = self._proposals(decision, seat, rosters, evidence) if self.proposals else []
        alternatives = _diverse(_ordered(ranked[:1], extra, ranked[:2], informed[:3], ranked, informed, coarse),
                                self.candidate_limit)
        expected = {_key(action) for action in decision.actions}
        pairs = self._lineups(foe, evidence)
        rng = Random(fingerprint(["public-planning-v1", seat, evidence["events"]]))
        seeds = [[rng.randrange(65536) for _ in range(4)] for _ in range(self.samples)]
        deadline = (started if self.proposals else perf_counter()) + min(budget, (decision.seconds_left or 50) - 1.)
        results = []
        completed = []
        if not pairs:
            self.counts["no_lineup_hypothesis"] += 1
        for world_id, line in enumerate(pairs):
            if perf_counter() >= deadline:
                break
            lineups = {seat: self.our_lineup, foe: line}
            answer = self.worker.call("prepare", format=FORMAT, seat=seat, observation=decision.observation,
                                      rosters=[rosters[p] for p in SEATS], lineups=[lineups[p] for p in SEATS],
                                      events=evidence["events"], own_pp=evidence["own_pp"], world=world_id,
                                      proxy_seat=foe if self.proxy else None)
            if reason := answer.get("reason"):
                self.counts["prepare_" + reason] += 1
                if reason == "prepare_error" and "detail" in answer:
                    self.counts["detail: " + answer["detail"][:90]] += 1
                continue
            try:
                states = {item["seat"]: make_decision(BATTLE, item["raw"], item["templates"], item["seat"])
                          for item in answer["views"] if item}
            except ValueError:
                self.counts["empty_rebuilt_menu"] += 1
                continue
            if states.keys() != set(SEATS) or {_key(action) for action in states[seat].actions} != expected:
                self.counts["legal_mask_mismatch"] += 1
                continue
            replies, weights = self._replies(states[foe], seat, foe, evidence)
            if self.proxy:
                hypothetical = next(item for item in answer["views"] if item["seat"] == foe)
                predictions = self.proxy.probabilities(states[foe], foe, rosters, hypothetical, evidence["protocol_log"])
                if predictions:
                    if self.belief:
                        if self.adaptive_all:
                            neutral = {_key(action): weight for action, weight in zip(replies, weights, strict=True)}
                            predictions["neutral"] = [neutral.get(_key(action), 0.) for action in states[foe].actions]
                        for name, distribution in predictions.items():
                            self.belief.remember(turn, len(evidence["events"]), name, states[foe].actions, distribution)
                        mixture = self.belief.mix(predictions)
                    else:
                        mixture = list(map(fmean, zip(*predictions.values())))
                    replies, weights = mix_replies(states[foe], replies, weights, mixture, 1. if self.adaptive_all else .45)
                    self.counts["proxy_worlds"] += 1
                else:
                    self.counts["proxy_fallbacks"] += 1
            world = (world_id, replies, weights)
            scores = self._score(world, seat, alternatives, seeds, deadline, 1 if self.depth == 3 else self.depth)
            if scores is not None:
                results.append(scores)
                completed.append(world)
        if not results:
            return self._fallback("no_completed_world", baseline)
        self.counts["searched"] += 1
        scores = list(map(fmean, zip(*results)))
        if self.depth == 3:
            alternatives, scores = self._confirm(completed, results, alternatives, seat, rng, deadline)
        best = max(range(len(scores)), key=scores.__getitem__)
        if scores[best] - scores[0] < self.min_edge:
            return baseline
        self.counts["changed"] += 1
        if alternatives[best] in extra:
            self.counts["model_proposal_chosen"] += 1
        return alternatives[best]

    def _proposals(self, decision, seat, rosters, evidence):
        proposals = []
        predictions = self.proxy.probabilities(decision, seat, rosters, evidence["own_proxy"], evidence["protocol_log"])
        for scores in predictions.values():
            proposals.append(decision.actions[max(range(len(scores)), key=scores.__getitem__)])
        self.counts["model_proposals"] += len(proposals)
        return proposals

    def _replies(self, decision, seat, foe, evidence):
        reply_policy = JointPolicy(self.profiles)
        history = reply_policy.history
        history.opponent_roster = self.our_roster
        turn = 0
        for event in evidence["events"]:
            fields = event.split("|")
            if fields[1] == "turn":
                turn = int(fields[2])
            elif fields[1] in {"switch", "drag"} and fields[2].startswith(seat):
                history.revealed.add(ident(fields[3].split(",", 1)[0]))
            elif self.learned and fields[1] == "move" and fields[2].startswith(foe) and ident(fields[3]) in PROTECT:
                name = ident(fields[2].split(": ", 1)[-1])
                history.protected_at[name] = turn
        actions, weights = _reply_distribution(decision, reply_policy, self.profiles)
        if self.learned:
            probabilities = self.learned.probabilities(decision, history, evidence["events"])
            self.counts["learned_worlds"] += 1
            return mix_replies(decision, actions, weights, probabilities, .58)
        return actions, weights

    def _score(self, world, seat, actions, seeds, deadline, depth):
        world_id, replies, weights = world
        remaining = 1000 * (deadline - perf_counter())
        if remaining <= 0:
            self.counts["evaluate_budget"] += 1
            return None
        result = self.worker.call("evaluate", seat=seat, ours=actions, replies=replies, seeds=seeds,
                                  budget_ms=remaining, planning=depth, world=world_id, interaction=self.interaction,
                                  spread_hp=bool(self.learned) or self.depth == 3 or len(seeds) > 3)
        if reason := result.get("reason"):
            self.counts["evaluate_" + reason] += 1
            return None
        self.counts["worlds"] += 1
        self.counts["branches"] += len(actions) * len(replies) * len(seeds)
        return robust_scores(result["matrix"], weights)

    def _confirm(self, worlds, coarse, actions, seat, rng, deadline):
        scores = average_advantages(coarse)
        indices = survivors(scores, 4)
        options = [actions[index] for index in indices]
        seeds = [[rng.randrange(65536) for _ in range(4)] for _ in range(2)]
        results = []
        for world in worlds:
            confirmed = self._score(world, seat, options, seeds, deadline, 2)
            if confirmed is not None:
                results.append(confirmed)
        if not results:
            self.counts["confirmation_fallback"] += 1
            return actions, scores
        self.counts["confirmed"] += 1
        confirmed = average_advantages(results)
        return options, [.40 * scores[index] + .60 * value for index, value in zip(indices, confirmed, strict=True)]
