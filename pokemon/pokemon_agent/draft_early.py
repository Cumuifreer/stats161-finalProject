"""Learn Draft values from legal, paired continuations against a frozen league."""

import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
from random import Random
from statistics import mean
from time import perf_counter

import torch

from .__main__ import HISTORICAL_CARDS, _percentile
from .combination_preview import check_pools as check_prior_pools
from .draft_advantage import load_advantage
from .draft_counterfactual import PROVENANCE, LabelGames, check_hybrid, sample_prefix
from .draft_coverage import CHAMPION, Coverage, SearchLimit
from .draft_forecast import ForecastDraft, legal_cards, public_draft
from .draft_last_pick import mean_ci, other, review_seeds, sample_game
from .draft_league import baseline_hash, disjoint, parallel, pools_for, read, save
from .draft_matchups import ROOT
from .draft_value import DraftValueRanker
from .game import DECISION_LIMIT, DRAFT, Decision
from .sim import FIRST_DRAFTER_PICKS, SEATS, _draft, fingerprint

PICKS = tuple(range(1, 11))
RIVALS = ("champion", "pairwise", "search")
LEAGUE = ("hybrid", "pairwise", "search")
CONTINUATIONS = ("pairwise", "forecast", "attention")
METHODS = ("frozen", "search", "learned", "direct")
REPLAY_METHODS = ("replay-search", "replay-learned")
LEAF_METHODS = ("leaf-control", "leaf-direct", "leaf-guard")
MATCHED_REVIEW_STARTS = 44
REVIEW_SECONDS = 3.0
REVIEW_STARTS = 64


def check_pools(pools, exclusions):
    previous = [p for group in ("benchmark_results", "checkpoints")
                for version in ("combination-v1", "preview-active-v1")
                for p in (ROOT / group / version).glob("*.json") if "pools" in read(p)]
    check_prior_pools(pools, [*previous, *exclusions])


def acting_seat(first, pick):
    return first if pick - 1 in FIRST_DRAFTER_PICKS else other(first)


def prefix_for(experiment, pool, first, pick, styles):
    return sample_prefix(pool, first, pick, experiment.league, experiment.hybrid,
                         ["hybrid" if styles[seat] == "champion" else styles[seat] for seat in SEATS])


def projected_options(experiment, decision, anchor):
    plan = experiment.projection.options(decision, anchor, all_legal=True)
    if plan is None:
        raise ValueError("Cannot project a valid completion for the baseline pick")
    actions, worlds, _, (pool, owners, ply, features), slots, _, _ = plan
    if {pick["card_id"] for pick in actions} != {pick["card_id"] for pick in decision.actions} or actions[0] != anchor:
        raise ValueError("Projected actions differ from Decision.actions")
    legal = legal_cards(pool, owners, experiment.projection.prior.items)
    if {i for i, allowed in enumerate(legal) if allowed} != set(slots):
        raise ValueError("Public feature mask differs from Decision.actions")
    return actions, worlds, (pool, owners, ply, features), slots


def candidate_indices(scores, features, slots, rng, count):
    ranked = sorted(range(1, len(scores)), key=lambda i: scores[i], reverse=True)
    chosen = [0, *ranked[:2]]
    denied = next((i for i in ranked if features[slots[i], 4] and i not in chosen), None)
    if denied is not None and len(chosen) < count - 1:
        chosen.append(denied)
    if len(chosen) < count and (outside := [i for i in ranked if i not in chosen]):
        chosen.append(rng.choice(outside))
    return [*chosen, *(i for i in ranked if i not in chosen)][:count]


def replay_draft(experiment, pool, first, prefix, forced, styles, cache):
    def choose(seat, decision):
        i = decision.observation["pick"] - 1
        if i < len(prefix):
            return decision.validate(prefix[i])
        if i == len(prefix):
            return decision.validate(forced)
        key = (styles[seat], fingerprint(decision.observation))
        if key not in cache:
            cache[key] = decision.validate(experiment.choose(decision, styles[seat]))
        return decision.validate(cache[key])

    return _draft(pool, first, choose)


def paired_draws(seed, actual, excluded, styles, count):
    reserved = {actual, *excluded}
    rng = Random((seed << 9) ^ 0x8DB24)
    results = []
    for _ in range(styles):
        group = []
        while len(group) < count:
            battle = tuple(rng.randrange(65536) for _ in range(4))
            if battle not in reserved:
                reserved.add(battle)
                group.append(battle)
        results.append(group)
    return results


class EarlyLeague(Coverage):
    def __init__(self, stack, cards):
        super().__init__(stack, cards)
        self.forecaster = ForecastDraft(self.hybrid.pairwise, self.hybrid.model, "hybrid")
        attention = self.league.ranks["search"]
        self.attention = DraftValueRanker(attention.baseline, attention.net, attention.margin, "neural")

    def choose(self, decision, style):
        if style == "forecast":
            return self.forecaster.rank(decision, "value")[0]
        if style == "attention":
            return self.attention.rank(decision, "value")[0]
        return super().choose(decision, style)


def continuations(seed, rng, audit, champion_every, actor, first, mix_own):
    order = list(CONTINUATIONS)
    rng.shuffle(order)
    opponents = order if audit else order[:2]
    own = ["forecast" if mix_own and (seed + i) % 2 else "pairwise" for i in range(len(opponents))]
    upgrade = Random(seed ^ 0x41B07).randrange(champion_every)
    if upgrade == 0:
        # Rotate the exact champion across the discovery and held-out styles.
        slot = seed // champion_every % len(opponents)
        if first == actor:
            own[slot] = "champion"
        else:
            opponents[slot] = "champion"
    elif upgrade == 1:
        opponents[seed // champion_every % len(opponents)] = "search"
    return own, opponents


def label_cases(seeds, cards, picks, draws, count, audit, champion_every, mix_own):
    rows, errors, events = [], [], Counter()
    with ExitStack() as stack:
        experiment = EarlyLeague(stack, cards)
        leaves = LabelGames(experiment.league.engine, experiment.preview, experiment.profiles, cards)
        for seed in seeds:
            pool, first, actual = sample_game(cards, seed)
            pick = picks[seed % len(picks)]
            actor = acting_seat(first, pick)
            foe = other(actor)
            rng = Random(seed ^ 0x9A372)
            prefixes = {actor: LEAGUE[seed // len(picks) % 3], foe: rng.choice(LEAGUE)}
            suffix_own, opponents = continuations(seed, rng, audit, champion_every, actor, first, mix_own)
            try:
                prefix, observed, decision = prefix_for(experiment, pool, first, pick, prefixes)
                if observed != actor:
                    raise ValueError("Draft prefix reached the wrong seat")
                anchor = decision.validate(experiment.choose(decision, "hybrid"))
                actions, _, (ids, owners, ply, features), slots = projected_options(experiment, decision, anchor)
                scores = features[slots, 0].tolist()
                label_indices = candidate_indices(scores, features, slots, rng, count)
                seeds_by_style = paired_draws(seed, actual, review_seeds(decision, 12), len(opponents), draws)
                outcomes = [None] * len(actions)
                cache = {}
                for i in label_indices:
                    outcomes[i] = []
                    for own_style, style, battle_seeds in zip(suffix_own, opponents, seeds_by_style, strict=True):
                        rosters = replay_draft(experiment, pool, first, prefix, actions[i], {actor: own_style, foe: style}, cache)
                        values = []
                        for k, battle_seed in enumerate(battle_seeds):
                            swap = k % 2
                            sides = {actor: rosters[foe], foe: rosters[actor]} if swap else rosters
                            values.append(leaves.run(sides, foe if swap else actor, battle_seed, "rollout"))
                        outcomes[i].append(values)
                rows.append({"seed": seed, "pool": ids, "owners": owners, "ply": ply, "features": features.tolist(),
                             "slots": slots, "results": {"rollout": outcomes}, "opponents": opponents,
                             "prefix": [action["card_id"] for action in prefix], "prefix_styles": [prefixes[actor], prefixes[foe]],
                             "actual_battle_seed": actual, "paired_draws": seeds_by_style, "actor": actor,
                             "first": first, "own_suffix": suffix_own})
                events.update(completed=1, **{f"pick_{pick}": 1, f"actor_{actor}": 1})
                for own_style, style in zip(suffix_own, opponents, strict=True):
                    events[f"opponent_{style}"] += 1
                    events["champion_scenarios"] += "champion" in (own_style, style)
            except (KeyError, ValueError, OSError, RuntimeError) as error:
                errors.append({"seed": seed, "pick": pick, "issue": str(error)})
        events.update(leaves.activity)
        events.update(league_draft_games=experiment.hybrid.rollout.activity["rollout_games"] + experiment.league.rollout.activity["rollout_games"],
                      preview_inner_games=experiment.preview.games, champion_terminal_games=experiment.activity["full_leaf_games"],
                      champion_review_fallbacks=experiment.activity["full_budget_fallbacks"])
    return rows, errors, events


def audit_rows(rows):
    deltas, same_style, new_style = [], [], []
    by_prefix, edited = {}, 0
    for row in rows:
        games = row["results"]["rollout"]
        half = len(games[0][0]) // 2
        if len(games[0]) != 3 or half < 2:
            raise ValueError("Audit needs two discovery styles and a third held-out continuation")
        valid = [i for i, game in enumerate(games) if game is not None]
        picked = max(valid, key=lambda i: mean(score for result in games[i][:2] for score in result[:half]))
        edited += picked != 0
        first, second, holdout = [mean(a[half:]) - mean(b[half:]) for a, b in zip(games[picked], games[0], strict=True)]
        deltas.append(mean((first, second, holdout)))
        same_style.append(mean((first, second)))
        new_style.append(holdout)
        by_prefix.setdefault(row["prefix_styles"][0], []).append(deltas[-1])
    return {"positions": len(deltas), "changed": edited, "same_styles": [mean(same_style), mean_ci(same_style)],
            "unseen_style": [mean(new_style), mean_ci(new_style)], "combined": [mean(deltas), mean_ci(deltas)],
            "prefix_styles": {name: [len(values), mean(values), mean_ci(values)] for name, values in by_prefix.items()}}


def generate(args, cards):
    check_hybrid()
    seeds = range(args.seed, args.seed + args.count)
    pools = pools_for(cards, seeds)
    check_pools(pools, args.exclude)
    rows, issues, events = [], [], Counter()
    started = perf_counter()
    for data, errors, stats in parallel(label_cases, seeds, args.workers, cards, args.picks, args.draws, args.candidates,
                                        args.audit, args.champion_every, args.mix_own):
        rows.extend(data)
        issues.extend(errors)
        events.update(stats)
    if not rows:
        raise RuntimeError(f"No valid Draft labels: {issues[:4]}")
    rows.sort(key=lambda row: row["seed"])
    report = {"positions": len(rows), "seconds": perf_counter() - started, "workers": args.workers, "draws": args.draws,
              "candidates": args.candidates, "champion_every": args.champion_every, "events": dict(events), "issues": issues,
              "rollout_terminal_games": events["rollout_terminal_games"], "mix_own": args.mix_own}
    if args.audit:
        report["audit"] = audit_rows(rows)
    save(args.output, {"report": report, "pools": pools, "catalog": fingerprint(cards), "frozen": sorted(PROVENANCE),
                       "champion": CHAMPION, "rows": rows})
    print(json.dumps(report), flush=True)


def review_picks(pool, first):
    rng = Random(fingerprint(["early-review-picks", sorted(card["card_id"] for card in pool)]))
    picked = {}
    for seat in (first, other(first)):
        picked[seat] = {rng.choice([pick for pick in window if acting_seat(first, pick) == seat])
                        for window in (range(5, 8), range(8, 11))}
    return picked


def projection_scores(encoding, slots, model):
    ids, owners, ply, features = encoding
    prior = features[slots, 0]
    if model is None:
        return prior
    with torch.inference_mode():
        logits = model(torch.tensor([ids]), torch.tensor([owners]), torch.tensor([ply]), features[None])[0]
        return .65 * logits[slots].sigmoid() + .35 * prior


def propose(encoding, slots, model):
    scores = projection_scores(encoding, slots, model)
    features = encoding[3]
    prior = features[slots, 0].tolist()
    ordered = sorted(range(1, len(slots)), key=lambda i: prior[i], reverse=True)
    menu = ordered[:6]
    denied = next((i for i in ordered[6:] if features[slots[i], 4]), None)
    if denied is not None:
        menu.append(denied)
    return max(menu, key=lambda i: float(scores[i])), scores


def replay_completion(experiment, decision, forced, opponent, cache):
    prior = experiment.projection.prior
    _, own, foe, available, pick, order = public_draft(decision, prior.index)
    move = prior.index[decision.validate(forced)["card_id"]]
    rosters = [own + [move], list(foe)]
    available.remove(move)
    cards = experiment.original_cards
    for turn, side in enumerate(order[pick:], pick + 1):
        held = {prior.items[card] for card in rosters[side]}
        legal = tuple({"type": "draft_pick", "card_id": cards[i]["card_id"]} for i in available if prior.items[i] not in held)
        if not legal:
            return None
        board = {"pick": turn, "cards": [cards[i] for i in available],
                 "picked": [cards[i] for i in rosters[side]], "opponent_picked": [cards[i] for i in rosters[1 - side]]}
        state = Decision(DRAFT, board, legal)
        strategy = opponent if side else "forecast"
        key = (strategy, turn, tuple(rosters[side]), tuple(rosters[1 - side]))
        if key not in cache:
            cache[key] = state.validate(experiment.choose(state, strategy))
        chosen = prior.index[state.validate(cache[key])["card_id"]]
        available.remove(chosen)
        rosters[side].append(chosen)
    return rosters


def review_worlds(experiment, decision, actions, worlds, alternative, replay):
    if replay:
        cache = {}
        pairs = [(replay_completion(experiment, decision, actions[0], style, cache),
                  replay_completion(experiment, decision, actions[alternative], style, cache)) for style in ("pairwise", "forecast")]
        return [p for p in pairs if all(team is not None for team in p)]
    # Indices 3 and 4 encode Pairwise and Attention + Search in the frozen predictor.
    return [(worlds[0][i], worlds[alternative][i]) for i in (3, 4, 2, 0, 1)
            if worlds[0][i] is not None and worlds[alternative][i] is not None][:2]


def projected_review(experiment, decision, world_pairs, deadline, activity, target_starts=0):
    limit = SearchLimit(deadline, REVIEW_STARTS)
    leaves = experiment.budgeted_games(limit, 3, 1)
    seeds = review_seeds(decision, REVIEW_STARTS // 2 if target_starts else 4)
    changes = [[], []]
    complete = False

    def score_difference(pair, seed, seat):
        scores = []
        for teams in pair:
            rosters = {who: [experiment.original_cards[card] for card in team]
                       for who, team in zip((seat, other(seat)), teams, strict=True)}
            scores.append(leaves.run(rosters, seat, seed, "rollout", deadline))
        return scores[1] - scores[0]

    try:
        for j, pair in enumerate(world_pairs):
            for k, seed in enumerate(seeds[2*j:2*j + 2]):
                changes[j].append(score_difference(pair, seed, SEATS[k]))
        for k, seed in enumerate(seeds[4:]):
            if limit.starts >= target_starts:
                break
            j = k % 2
            changes[j].append(score_difference(world_pairs[j], seed, SEATS[(k // 2) % 2]))
        limit.check()
        complete = limit.starts >= target_starts
    except TimeoutError:
        activity["budget_fallbacks"] += 1
    activity["starts"] += limit.starts
    activity["full_reviews"] += complete
    if not complete:
        return False
    effects = [mean(outcomes) for outcomes in changes]
    return min(effects) >= 0 and mean(effects) >= .375


def intervene(experiment, decision, anchor, method, model, started, activity):
    if len(decision.actions) < 2:
        return anchor
    activity["eligible"] += 1
    try:
        actions, worlds, encoding, slots = projected_options(experiment, decision, anchor)
    except ValueError:
        activity["projection_fallbacks"] += 1
        return anchor
    alternative, scores = propose(encoding, slots, None if method in {"search", "search-matched", "replay-search"} else model)
    if method == "direct":
        return actions[alternative] if scores[alternative] >= scores[0] + .055 else anchor
    pairs = review_worlds(experiment, decision, actions, worlds, alternative, method in (*REPLAY_METHODS, *LEAF_METHODS))
    if len(pairs) < 2:
        activity["projection_fallbacks"] += 1
        return anchor
    activity["duplicate_anchor_replies"] += set(pairs[0][0][1]) == set(pairs[1][0][1])
    if method in LEAF_METHODS:
        from .draft_leaf import accept, advantages

        leaf = experiment.leaf.prior if method == "leaf-control" else experiment.leaf
        gains = advantages(leaf, pairs)
        if method != "leaf-guard":
            return actions[alternative] if accept(gains) else anchor
    now = perf_counter()
    remaining = (decision.seconds_left or DECISION_LIMIT[DRAFT]) - (now - started) - .75
    deadline = now + min(REVIEW_SECONDS, remaining)
    if now >= deadline:
        activity["budget_fallbacks"] += 1
        return anchor
    passed = projected_review(experiment, decision, pairs, deadline, activity,
                              MATCHED_REVIEW_STARTS if method == "search-matched" else 0)
    if method == "leaf-guard" and passed and float(gains.mean()) <= -.045:
        activity["model_vetoes"] += 1
        return anchor
    return actions[alternative] if passed else anchor


def complete_game(experiment, pool, first, actor, opponent, method, model, opportunities, policy_cache, events):
    paths, durations = {}, []
    stats = Counter()

    def choose(seat, decision):
        start = perf_counter()
        ply = decision.observation["pick"]
        style = "champion" if seat == actor else opponent
        key = style, fingerprint(decision.observation)
        if key not in policy_cache:
            action = decision.validate(experiment.choose(decision, style))
            policy_cache[key] = action, (perf_counter() - start) * 1000.
        anchor, reference_ms = policy_cache[key]
        started_intervention = perf_counter()
        pick = anchor
        if seat == actor and method != "frozen" and ply in opportunities:
            pick = intervene(experiment, decision, anchor, method, model, started_intervention - reference_ms / 1000., stats)
            pick = decision.validate(pick)
            stats["overrides"] += pick != anchor
            stats[f"overrides_{ply}"] += pick != anchor
        elapsed = reference_ms + (perf_counter() - started_intervention) * 1000.
        events["timeouts"] += elapsed >= 1000 * DECISION_LIMIT[DRAFT]
        if seat == actor:
            durations.append(elapsed)
        paths[ply] = pick["card_id"]
        return pick

    return _draft(pool, first, choose), paths, durations, stats


def evaluate(seeds, cards, opponents, methods, checkpoint, leaf_checkpoint=None):
    pairs, issues, counts = {}, [], Counter()
    with ExitStack() as stack:
        experiment = (EarlyLeague if set(methods).intersection((*REPLAY_METHODS, *LEAF_METHODS)) else Coverage)(stack, cards)
        leaves = LabelGames(experiment.league.engine, experiment.preview, experiment.profiles, cards)
        model = load_advantage(checkpoint)[0] if checkpoint else None
        if leaf_checkpoint:
            from .draft_leaf import load_leaf

            experiment.leaf = load_leaf(leaf_checkpoint, cards)
        for root in seeds:
            pool, first, seed = sample_game(cards, root)
            opponent = opponents[root % len(opponents)]
            opportunities = review_picks(pool, first)
            pair = []
            for actor in SEATS:
                policy_cache, outcomes, record = {}, {}, {}
                try:
                    for method in methods:
                        rosters, moves, times, stats = complete_game(experiment, pool, first, actor, opponent, method, model,
                                                                     opportunities[actor], policy_cache, counts)
                        keys = tuple(tuple(card["card_id"] for card in rosters[seat]) for seat in SEATS)
                        if keys not in outcomes:
                            outcomes[keys] = leaves.run(rosters, actor, seed, "rollout")
                        record[method] = {"score": outcomes[keys], "cards": moves, "times": times, "review": dict(stats)}
                    pair.append({"opponent": opponent, "first_seat": actor == first, "methods": record})
                except (KeyError, OSError, RuntimeError, ValueError) as error:
                    issues.append({"seed": root, "seat": actor, "issue": str(error)})
            if len(pair) == 2:
                pairs[root] = pair
        counts.update(terminal_games=leaves.activity["rollout_terminal_games"], preview_inner_games=experiment.preview.games,
                      champion_terminal_games=experiment.activity["full_leaf_games"],
                      league_draft_games=experiment.hybrid.rollout.activity["rollout_games"] + experiment.league.rollout.activity["rollout_games"])
    return pairs, issues, counts


def report_results(pairs, method, reference="frozen"):
    records = [row["methods"][method] for pair in pairs for row in pair]
    scores = [row["score"] for row in records]
    difference = [mean(row["methods"][method]["score"] - row["methods"][reference]["score"] for row in pair) for pair in pairs]
    timings = [time for row in records for time in row["times"]]
    stats = Counter()
    for row in records:
        stats.update(row["review"])
    changed = sum(row["methods"][method]["cards"] != row["methods"][reference]["cards"] for pair in pairs for row in pair)
    return {"games": len(scores), "wins": scores.count(1.), "draws": scores.count(.5), "score": mean(scores),
            "delta": mean(difference), "ci95": mean_ci(difference), "different_drafts": changed,
            "draft_p95_ms": _percentile(timings, .95), "draft_p99_ms": _percentile(timings, .99),
            "draft_max_ms": max(timings), "review": dict(stats)}


def benchmark(args, cards):
    check_hybrid()
    seeds = range(args.seed, args.seed + args.count)
    pools = pools_for(cards, seeds)
    check_pools(pools, args.exclude)
    if args.checkpoint:
        config = read(args.checkpoint / "config.json")
        if config["catalog"] != fingerprint(cards) or set(config["frozen"]) != PROVENANCE or config["trained_picks"] != list(PICKS):
            raise ValueError("Unexpected early Draft model provenance")
        for split in config["splits"].values():
            disjoint(pools, split)
    if args.leaf_checkpoint:
        leaf = read(args.leaf_checkpoint / "config.json")
        if leaf["provenance"]["proposal"] != baseline_hash(args.checkpoint):
            raise ValueError("Leaf gate requires its original proposal model")
        for split in leaf["splits"].values():
            disjoint(pools, split)
    methods = tuple(dict.fromkeys(("frozen", *args.methods)))
    started = perf_counter()
    paired, issues, totals = {}, [], Counter()
    for games, errors, counts in parallel(evaluate, seeds, args.workers, cards, args.opponents, methods, args.checkpoint, args.leaf_checkpoint):
        paired.update(games)
        issues.extend(errors)
        totals.update(counts)
    if not paired:
        raise RuntimeError(f"No completed paired games: {issues[:3]}")
    result = [paired[seed] for seed in sorted(paired)]
    groups = {name: [rows for rows in result if rows[0]["opponent"] == name] for name in args.opponents}
    report = {"pairs": len(result), "seconds": perf_counter() - started, "workers": args.workers, "champion": CHAMPION,
              "review_starts": REVIEW_STARTS, "review_seconds": REVIEW_SECONDS, "issues": issues, "counts": dict(totals),
              "results": {method: report_results(result, method) for method in methods},
              "learned_vs_search": report_results(result, "learned", "search") if {"learned", "search"} <= set(methods) else None,
              "by_opponent": {name: {method: report_results(group, method) for method in methods} for name, group in groups.items() if group},
              "model": baseline_hash(args.checkpoint) if args.checkpoint else None,
              "leaf_model": baseline_hash(args.leaf_checkpoint) if args.leaf_checkpoint else None}
    if "search-matched" in methods:
        report["matched_target_starts"] = MATCHED_REVIEW_STARTS
    save(args.output, {"report": report, "pools": pools, "paired_games": dict(sorted(paired.items()))})
    print(json.dumps({key: val for key, val in report.items() if key != "by_opponent"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate_args, bench = [commands.add_parser(command) for command in ("generate", "bench")]
    for cmd in (generate_args, bench):
        cmd.add_argument("--seed", type=int, required=True)
        cmd.add_argument("--count", type=int, required=True)
        cmd.add_argument("--workers", type=int, default=12)
        cmd.add_argument("--output", required=True, type=Path)
        cmd.add_argument("--exclude", nargs="*", default=[], type=Path)
    generate_args.add_argument("--picks", nargs="+", type=int, choices=PICKS, default=PICKS)
    generate_args.add_argument("--draws", type=int, default=6)
    generate_args.add_argument("--candidates", type=int, default=5)
    generate_args.add_argument("--audit", action="store_true")
    generate_args.add_argument("--mix-own", action="store_true")
    generate_args.add_argument("--champion-every", type=int, default=6)
    bench.add_argument("--methods", choices=(*METHODS, "search-matched", *REPLAY_METHODS, *LEAF_METHODS), nargs="+", default=METHODS)
    bench.add_argument("--opponents", choices=RIVALS, nargs="+", default=RIVALS)
    bench.add_argument("--checkpoint", type=Path)
    bench.add_argument("--leaf-checkpoint", type=Path)
    args = parser.parse_args()
    if args.output.exists() or min(args.workers, args.count) < 1:
        parser.error("Use a fresh output, positive count and positive workers")
    if args.command == "generate" and (args.champion_every < 1 or min(args.draws, args.candidates) < 4):
        parser.error("Use a fresh output and at least four candidates and paired draws")
    if args.command == "bench" and {"learned", "direct", "replay-learned", *LEAF_METHODS}.intersection(args.methods) and not args.checkpoint:
        parser.error("Learned Draft methods require a checkpoint")
    if args.command == "bench" and ((set(args.methods).intersection(LEAF_METHODS) and not args.leaf_checkpoint)
                                     or (args.leaf_checkpoint and not args.checkpoint)):
        parser.error("Leaf methods require both the fixed proposal and leaf checkpoints")
    {"generate": generate, "bench": benchmark}[args.command](args, read(HISTORICAL_CARDS))


if __name__ == "__main__":
    main()
