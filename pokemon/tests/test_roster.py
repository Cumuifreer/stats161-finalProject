from time import perf_counter
from itertools import combinations
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from pokemon_agent.combination_preview import ComboValue, FEATURES, focused_pool
    from pokemon_agent.combination_space import disjoint_matchups, legal_groups
    from pokemon_agent.draft_distill import objective as draft_advantage_objective
    from pokemon_agent.draft_coverage import CHAMPION, SearchLimit, preview_allocation
    from pokemon_agent.draft_early import acting_seat, paired_draws, replay_completion, review_picks
    from pokemon_agent.draft_leaf import LeafValue, advantages
    from pokemon_agent.draft_forecast import AnchoredPickModel, PickModel, UNKNOWN_STYLE, encode_owners, legal_cards, public_draft
    from pokemon_agent.draft_last_pick import beats_reference, legal_rosters
    from pokemon_agent.draft_value import DraftValue, DraftValueRanker, PairwiseDraftValue
    from pokemon_agent.game import DRAFT, Decision
    from pokemon_agent.preview_active import EXTRA_COUNT, PreviewRanker, allocate_extras, extra_actions, objective as preview_objective
    from pokemon_agent.roster import DraftRanker, RosterNet


@unittest.skipUnless(torch is not None, "requires PyTorch")
class RosterTests(unittest.TestCase):
    def test_replayed_draft_completion_respects_item_clause_and_perspective(self):
        from types import SimpleNamespace

        cards = [{"card_id": str(i)} for i in range(18)]
        items = [0, 1, 2, 0, *range(4, 18)]
        prior = SimpleNamespace(index={str(i): i for i in range(18)}, items=items)
        experiment = SimpleNamespace(projection=SimpleNamespace(prior=prior), original_cards=cards,
                                     choose=lambda state, style: state.actions[-1] if style == "pairwise" else state.actions[0])
        actions = tuple({"type": "draft_pick", "card_id": str(i)} for i in range(4, 18))
        root = Decision(DRAFT, {"pick": 4, "picked": cards[:1], "opponent_picked": cards[1:3], "cards": cards[3:]}, actions)
        teams = replay_completion(experiment, root, actions[0], "pairwise", {})
        self.assertEqual(teams[0][:2], [0, 4])
        self.assertEqual(teams[1][:2], [1, 2])
        self.assertEqual(len(set(teams[0] + teams[1])), 12)
        self.assertTrue(all(len({items[i] for i in team}) == 6 for team in teams))

    def test_draft_leaf_calibration_starts_from_frozen_prior(self):
        prior = PairwiseDraftValue(48, 16)
        model = LeafValue(prior, 16).eval()
        teams = torch.arange(12).reshape(1, 2, 6)
        with torch.inference_mode():
            torch.testing.assert_close(model(teams), prior(teams))
            torch.testing.assert_close(model(teams.flip(1)), -model(teams))
            torch.testing.assert_close(advantages(model, [[teams[0].tolist(), teams[0].tolist()]]), torch.zeros(1))
        self.assertTrue(all(not p.requires_grad for p in prior.parameters()))

    def test_early_draft_review_is_public_and_covers_each_seats_two_windows(self):
        pool = [{"card_id": str(i)} for i in range(18)]
        for first in ("p1", "p2"):
            positions = review_picks(pool, first)
            self.assertEqual(positions, review_picks(pool[::-1], first))
            for seat, picks in positions.items():
                self.assertEqual(len(picks), 2)
                self.assertTrue(all(acting_seat(first, pick) == seat for pick in picks))
                self.assertTrue(all(5 <= pick <= 10 for pick in picks))
                self.assertEqual({pick <= 7 for pick in picks}, {True, False})

    def test_early_draft_discovery_seeds_do_not_overlap_other_styles_or_game(self):
        seed = 8001
        initial = paired_draws(seed, (0, 0, 0, 0), (), 3, 8)
        actual, review = initial[0][0], initial[2][-1]
        draws = paired_draws(seed, actual, [review], 3, 8)
        unique = {battle for group in draws for battle in group}
        self.assertEqual(len(unique), 24)
        self.assertNotIn(actual, unique)
        self.assertNotIn(review, unique)

    def test_active_preview_keeps_random_examples_and_equal_label_budget(self):
        rows = [{"seed": i, "priority": i, "outside": list(range(16)), "acquisition": list(range(16))} for i in range(12)]
        quotas = allocate_extras(rows, balanced=True)
        self.assertEqual(sum(quotas.values()), EXTRA_COUNT * len(rows))
        self.assertLess(quotas[0], quotas[11])
        for row in rows:
            actions = extra_actions(row, quotas[row["seed"]], 2)
            self.assertEqual(actions["uniform"][:2], actions["active"][:2])
            self.assertEqual(len(set(actions["active"])), quotas[row["seed"]])

    def test_preview_ranker_is_seat_symmetric_and_excludes_padding_from_loss(self):
        net = PreviewRanker(ComboValue(48, 16), 16).eval()
        cards = torch.arange(24).reshape(2, 2, 6)
        roles = torch.arange(24).reshape(2, 2, 6) % 3
        features = torch.zeros((2, 2, FEATURES))
        with torch.inference_mode():
            net.score[-1].weight.fill_(.1)
            result = net(cards, roles, features)
            torch.testing.assert_close(net(cards.flip(1), roles.flip(1), features.flip(1)), -result)
        logits = torch.zeros(1, 3, 2, requires_grad=True)
        targets = torch.tensor([0., 1., 1.]).view(1, 3, 1, 1).expand(-1, -1, 2, 4)
        preview_objective(logits, targets, torch.tensor([[1., 1., 0.]]), torch.tensor([[0, 1]])).backward()
        self.assertTrue(torch.all(logits.grad[0, 2] == 0))
        self.assertTrue(torch.all(logits.grad[0, 1] < 0))

    def test_combination_counts_respect_items_per_side_and_shared_cards(self):
        items = ["a", "a", "b", "c", "d", "d", "e"]
        pairs = legal_groups(items, 2)
        self.assertNotIn((0, 1), pairs)
        for size in (2, 3):
            groups = legal_groups(items, size)
            expected = sum(not set(a).intersection(b) for a, b in combinations(groups, 2))
            self.assertEqual(disjoint_matchups(items, size), expected)

    def test_combination_value_swaps_exactly_with_seats(self):
        model = ComboValue(48, 16).eval()
        cards = torch.arange(12).view(1, 2, 6)
        roles = torch.tensor([[[0, 0, 1, 1, 2, 2], [2, 2, 0, 1, 1, 0]]])
        features = torch.arange(2 * FEATURES).view(1, 2, FEATURES).float()
        with torch.inference_mode():
            expected = model(cards, roles, features)
            torch.testing.assert_close(model(cards.flip(1), roles.flip(1), features.flip(1)), -expected)

    def test_focused_pool_contains_core_and_keeps_eighteen_unique_cards(self):
        cards = [{"card_id": i} for i in range(48)]
        core = [10, 20, 30]
        for seed in range(6):
            pool, _, _ = focused_pool(cards, seed, core)
            ids = {card["card_id"] for card in pool}
            self.assertEqual(len(pool), 18)
            self.assertEqual(len(ids), 18)
            self.assertTrue(set(core) <= ids)

    def test_value_swaps_sign_with_seats_and_ignores_roster_order(self):
        model = RosterNet(18, 16).eval()
        cards = torch.arange(12).view(1, 2, 6)
        roles = torch.tensor([[[0, 0, 1, 1, 2, 2], [2, 2, 0, 1, 1, 0]]])
        with torch.inference_mode():
            baseline = model(cards, roles)
            torch.testing.assert_close(model(cards.flip(1), roles.flip(1)), -baseline)
            torch.testing.assert_close(model(cards.flip(2), roles.flip(2)), baseline)

    def test_draft_completion_obeys_item_clause_and_filters_to_legal_actions(self):
        cards = [{"card_id": str(i), "item": f"item-{i if i != 3 else 0}", "moves": []} for i in range(18)]
        model = RosterNet(18, 16).eval()
        config = {"card_ids": list(map(str, range(18))), "draft_completions": 2}
        planner = DraftRanker(cards, model, config)
        actions = tuple({"type": "draft_pick", "card_id": str(i)} for i in (4, 5, 11))
        decision = Decision(DRAFT, {"pick": 4, "cards": cards[3:], "picked": cards[:1], "opponent_picked": cards[1:3]}, actions)
        self.assertCountEqual(planner.rank(decision, "prior"), actions)
        self.assertCountEqual(planner.rank(decision, "value"), actions)
        neural = DraftValueRanker(planner, DraftValue(18, 16).eval(), .035, "neural")
        self.assertCountEqual(neural.rank(decision, "value"), actions)
        self.assertEqual(neural.rank(decision, "prior"), planner.rank(decision, "prior"))
        schedule = [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]
        teams = planner._finish(list(range(3, 18)), ([0], [1, 2]), 4, schedule, 4, [[0] * 18] * 2, 0)
        self.assertTrue(all(len(roster) == 6 for roster in teams))
        self.assertTrue(all(len({planner.items[card] for card in roster}) == 6 for roster in teams))
        self.assertEqual(len(set(teams[0] + teams[1])), 12)

    def test_cross_team_value_respects_seat_and_card_order(self):
        cards = torch.arange(12).view(1, 2, 6)
        with torch.inference_mode():
            pairwise = PairwiseDraftValue(18, 16).eval()
            pairwise.synergy.fill_(.2)
            pairwise.matchup.fill_(.1)
            for model in (DraftValue(18, 16).eval(), pairwise):
                value = model(cards)
                torch.testing.assert_close(model(cards.flip(1)), -value)
                torch.testing.assert_close(model(cards.flip(2)), value)

    def test_pick_model_matches_player_view_and_masks_identical_items(self):
        cards = [{"card_id": str(i)} for i in range(18)]
        index = {str(i): i for i in range(18)}
        items = [0, 1, 2, 0, *range(4, 18)]
        actions = tuple({"card_id": str(i)} for i in range(4, 18))
        decision = Decision(DRAFT, {"pick": 4, "cards": cards[3:], "picked": cards[:1],
                                    "opponent_picked": cards[1:3]}, actions)
        pool, own, foe, _, pick, schedule = public_draft(decision, index)
        self.assertEqual(schedule[:5], [0, 1, 1, 0, 0])
        owners = encode_owners(pool, own, foe)
        mask = legal_cards(pool, owners, items)
        self.assertEqual([pool[i] for i, enabled in enumerate(mask) if enabled], list(range(4, 18)))
        model = PickModel(len(pool), 16).eval()
        with torch.inference_mode():
            scores = model(torch.tensor([pool]), torch.tensor([owners]), torch.tensor([pick - 1]),
                           torch.tensor([UNKNOWN_STYLE]), torch.tensor([mask]))[0]
        self.assertTrue(torch.isneginf(scores[:4]).all())
        self.assertTrue(torch.isfinite(scores[4:]).all())

    def test_anchored_pick_model_normalizes_legal_alternatives_and_forced_pick(self):
        model = AnchoredPickModel(18, 16).eval()
        pool = torch.arange(18).expand(2, -1)
        mask = torch.zeros_like(pool, dtype=torch.bool)
        mask[0, 4:8] = True
        mask[1, 7] = True
        with torch.inference_mode():
            logits = model(pool, torch.zeros_like(pool), torch.tensor([0, 11]),
                           torch.full((2,), UNKNOWN_STYLE), mask, torch.tensor([5, 7]))
        self.assertTrue(torch.isneginf(logits[~mask]).all())
        torch.testing.assert_close(logits.exp().sum(1), torch.ones(2))
        self.assertEqual(float(logits[1, 7]), 0.)

    def test_counterfactual_loss_uses_only_observed_branches(self):
        logits = torch.zeros((1, 3), requires_grad=True)
        outcome = torch.tensor([[.25, .75, 0.]])
        weights = torch.tensor([[1., .8, 0.]])
        loss = draft_advantage_objective(logits, outcome, weights, torch.tensor([0]))
        loss.backward()
        self.assertGreater(float(logits.grad[0, 0]), 0.)
        self.assertLess(float(logits.grad[0, 1]), 0.)
        self.assertEqual(float(logits.grad[0, 2]), 0.)

    def test_last_pick_only_adds_a_legal_card_to_the_acting_roster(self):
        cards = [{"card_id": str(i)} for i in range(18)]
        by_id = {card["card_id"]: card for card in cards}
        legal = {"card_id": "12"}
        decision = Decision(DRAFT, {"pick": 12, "picked": cards[:5], "opponent_picked": cards[5:11]}, (legal,))
        self.assertEqual(legal_rosters(decision, legal, "p2", by_id), {"p2": cards[:5] + [cards[12]], "p1": cards[5:11]})
        with self.assertRaises(ValueError):
            legal_rosters(decision, {"card_id": "11"}, "p1", by_id)

    def test_last_pick_review_rejects_small_or_ambiguous_samples(self):
        self.assertFalse(beats_reference([1.] * 2))
        self.assertFalse(beats_reference([1., 1.] + [0.] * 8))
        self.assertFalse(beats_reference([1.] * 4 + [-1.] * 4 + [0.] * 2))
        self.assertTrue(beats_reference([1.] * 3 + [0.] * 7))

    def test_breadth_allocation_includes_nested_preview_games(self):
        for count in range(3, 8):
            choices, seeds, discovery = preview_allocation(count)
            self.assertGreaterEqual(discovery, 4)
            self.assertLessEqual(choices, 5)
            self.assertLessEqual(count * (4 * choices * seeds + discovery) + 20, CHAMPION["starts"])

    def test_review_shares_one_cap_between_battle_and_preview_workers(self):
        class Worker:
            def call(self, command, **kwargs):
                return command

        limit = SearchLimit(perf_counter() + 10.)
        limit.starts = CHAMPION["starts"] - 1
        preview, battle = limit.wrap(Worker()), limit.wrap(Worker())
        self.assertEqual(preview.call("start"), "start")
        with self.assertRaises(TimeoutError):
            battle.call("start")
        self.assertEqual(limit.starts, CHAMPION["starts"])
