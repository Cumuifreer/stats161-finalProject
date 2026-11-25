import importlib.util
import json
import unittest
from unittest.mock import patch

from pokemon_agent.features import BattleEncoder
from pokemon_agent.game import BATTLE, PREVIEW, Decision
from pokemon_agent.joint import BattleHistory


class FeatureTests(unittest.TestCase):
    def test_snapshot_preserves_features_in_legal_action_order(self):
        def action(move, target):
            return {"type": "doubles_turn", "slot_0": {"type": "move", "move_id": move, "target": target},
                    "slot_1": {"type": "pass"}}

        decision = Decision(
            BATTLE,
            {"active": [{"species": "arcanine", "hp": 1}, None],
             "opponents": [{"species": "amoonguss", "hp": .5}, None], "turn": 1},
            (action("flamethrower", 1), action("protect", 0)),
            ({"flamethrower": {"base_power": 90, "accuracy": 1., "category": "special"}}, {}),
        )
        dex = {"types": {}, "species": {}, "moves": {"flamethrower": {"type": "fire"}}}
        with patch("pokemon_agent.features.load_dex", return_value=dex), patch(
            "pokemon_agent.tactics.load_dex", return_value=dex,
        ):
            encoder = BattleEncoder()
            expected = encoder.encode(decision)
            serialized = json.loads(json.dumps(decision.snapshot()))
            self.assertEqual(expected, encoder.encode(Decision.from_snapshot(serialized)))
            serialized["actions"].reverse()
            reordered = encoder.encode(Decision.from_snapshot(serialized))
            self.assertEqual(reordered.state, expected.state)
            self.assertEqual(reordered.actions, expected.actions[::-1])
            self.assertGreater(expected.actions[0][1], expected.actions[1][1])

    def test_history_uses_only_preview_reveals_and_past_own_protect(self):
        history = BattleHistory()
        preview = Decision(PREVIEW, {"opponent_team": [{"species": "amoonguss"}, {"species": "tornadus"}]}, ())
        history.observe(preview)
        protect = {"slot_0": {"type": "move", "move_id": "protect"}, "slot_1": {"type": "pass"}}
        turn = Decision(BATTLE, {"turn": 3, "active": [{"species": "arcanine"}, None],
                                 "opponents": [{"species": "amoonguss"}, None]}, (protect,))
        history.observe(turn)
        before = BattleHistory.from_snapshot(json.loads(json.dumps(history.snapshot())))
        history.record(turn, protect)
        self.assertEqual(before.unseen(), ("tornadus",))
        self.assertFalse(before.repeated(4))
        self.assertEqual(history.repeated(4), {"arcanine"})
        history.observe(preview)
        self.assertFalse(history.revealed)
        self.assertFalse(history.repeated(4))


@unittest.skipUnless(importlib.util.find_spec("torch"), "BC requires PyTorch")
class BCTests(unittest.TestCase):
    def test_imitation_excludes_padded_actions_even_when_they_score_higher(self):
        import torch

        from pokemon_agent.bc import _imitation

        class ScoreFirstFeature(torch.nn.Module):
            def forward(self, _, actions):
                return actions[..., 0]

        state = torch.zeros(1)
        rows = [(state, torch.tensor([[-2., .5], [-1., 1.]]), 1, None),
                (state, torch.tensor([[-4., .5], [-3., 1.], [-5., .3]]), 1, None)]
        metrics = _imitation(ScoreFirstFeature(), rows, 2, torch.device("cpu"))
        self.assertEqual(metrics["top1"], 1.)
        self.assertEqual(metrics["illegal_action_rate"], 0.)

    def test_split_rejects_both_seed_and_pool_overlap(self):
        from pokemon_agent.bc import _split

        cards = [{"card_id": str(index)} for index in range(18)]
        with self.assertRaisesRegex(ValueError, "seeds overlap"):
            _split(cards, [1, 2], [2, 3])
        with self.assertRaisesRegex(ValueError, "pools overlap"):
            _split(cards, [1], [2])

    def test_soft_targets_ignore_padding_and_produce_finite_gradients(self):
        import torch

        from pokemon_agent.bc import _loss

        logits = torch.tensor([[1., 2., 100.]], requires_grad=True)
        loss = _loss(logits, torch.tensor([[True, True, False]]), torch.tensor([1]), torch.tensor([[.2, .8, 0.]]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(logits.grad[0, 2].item(), 0.)


if __name__ == "__main__":
    unittest.main()
