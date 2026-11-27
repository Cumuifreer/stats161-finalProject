"""Gather paired terminal outcomes for different legal moves at the same training position."""

from collections import Counter
from random import Random
from time import perf_counter
from unittest.mock import patch

import numpy as np

from ..battle_league import BATTLE_LEAGUE, WIDTH, Encoder, LearnedPolicy, configure_torch, load_model, targets_at_root
from ..battle_value import advance, decisions, opponent
from ..game import BATTLE
from ..joint import JointPolicy
from ..sim import DRAFT_SIZE, FORMAT, SEATS, Showdown, _draft
from ..tactics import load_card_profiles
from . import collect as population


def collect(roots, cards, path, current, previous, exploration):
    if not current:
        raise ValueError("Counterfactual rollouts require a trained continuation model")
    configure_torch()
    profiles = load_card_profiles(cards, with_sets=True)
    encoder = Encoder(profiles)
    model = load_model(current, cards)
    archived = load_model(previous, cards) if previous else None
    counts = Counter()
    data = []
    started = perf_counter()

    def learned():
        return LearnedPolicy(profiles, encoder, model, value_only=True)

    with Showdown(counterfactual_labels=True) as engine:
        for root in roots:
            pool = Random(root).sample(cards, DRAFT_SIZE)
            for leg in range(2):
                actor = SEATS[leg]
                rival = SEATS[1 - leg]
                rng = Random(f"reanalysis/1/{root}/{leg}")
                real_seed = tuple(rng.randrange(65536) for _ in range(4))
                style = root % len(BATTLE_LEAGUE)
                our = learned()
                foe = (LearnedPolicy(profiles, encoder, archived, value_only=True) if archived and root % 3 == 0
                       else opponent(profiles, BATTLE_LEAGUE[style]))
                policies = {actor: our, rival: foe}
                rosters = _draft(pool, SEATS[root % 2], lambda seat, decision: policies[seat](decision))
                reply = engine.call("start", format=FORMAT, seed=real_seed, rosters=rosters)
                lineups = {seat: JointPolicy(profiles)(decision) for seat, decision in decisions(reply)}
                ranker = JointPolicy(profiles)
                playing = []
                reviewed = False
                for tick in range(500):
                    if reply["ended"]:
                        break
                    pending = dict(decisions(reply)).get(actor)
                    if pending is None or pending.phase != BATTLE:
                        reply = advance(engine, reply, policies, lineups=lineups)
                        continue
                    ranker.history = our.history
                    ranked = ranker.rank(pending)
                    if len(ranked) == 1:
                        reply = advance(engine, reply, policies, lineups=lineups)
                        continue
                    features = encoder(pending, our.history, ranked)
                    turn = pending.observation["turn"]
                    forced = not all(pending.observation["active"])
                    preferred = our.index(features)
                    executed = rng.randrange(len(ranked)) if rng.random() < exploration else preferred
                    propensity = exploration / len(ranked) + (1. - exploration) * (executed == preferred)
                    labels = [-1.] * WIDTH
                    if not reviewed and turn >= (1, 2, 3, 5)[root % 4] and not forced and len(ranked) >= 3:
                        # Test both continuation experts under common alternative simulator draws.
                        styles = (BATTLE_LEAGUE[style], BATTLE_LEAGUE[(style + 1) % len(BATTLE_LEAGUE)])
                        labels = targets_at_root(engine, reply, actor, ranked, policies, styles, 2,
                                                 2 * root + leg, tick, real_seed, profiles, learned)
                        counts["branch_positions"] += 1
                        counts["paired_terminal_games"] += len(ranked) * 4
                        reviewed = True
                    playing.append((*features, 0, executed, labels, turn, forced, propensity))
                    reply = advance(engine, reply, policies, override=(actor, ranked[executed][0]), lineups=lineups)
                if not reply["ended"]:
                    raise RuntimeError("Training game exceeded 500 simulator requests")
                winner = reply["winner"]
                outcome = .5 if winner is None else float(winner == actor)
                episode = 4 * root + 2 * leg + leg
                data.extend((*features, outcome, labels, episode, turn, forced, propensity, style, 0)
                            for *features, labels, turn, forced, propensity in playing)
                counts["games"] += 1
                counts["student_games"] += 1
    np.savez_compressed(path, **population.arrays(data))
    return dict(counts, paths=1, positions=len(data), worker_seconds=perf_counter() - started)


if __name__ == "__main__":
    with patch.object(population, "collect", collect):
        population.main()
