import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from pokemon_agent.game import DRAFT, Decision
    from pokemon_agent.roster import DraftRanker, RosterNet


@unittest.skipUnless(torch is not None, "requires PyTorch")
class RosterTests(unittest.TestCase):
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
        schedule = [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]
        teams = planner._finish(list(range(3, 18)), ([0], [1, 2]), 4, schedule, 4, [[0] * 18] * 2, 0)
        self.assertTrue(all(len(roster) == 6 for roster in teams))
        self.assertTrue(all(len({planner.items[card] for card in roster}) == 6 for roster in teams))
        self.assertEqual(len(set(teams[0] + teams[1])), 12)
