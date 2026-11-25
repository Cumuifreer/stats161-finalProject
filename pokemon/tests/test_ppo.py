import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from pokemon_agent.features import BattleEncoder
    from pokemon_agent.game import BATTLE, Decision
    from pokemon_agent.model import ActionRankingNetwork, ActorCriticNetwork, BattlePolicy
    from pokemon_agent.ppo import RolloutPolicy, _schedule, episode_advantages, masked_distribution, terminal_reward
    from pokemon_agent.tactics import tactical_candidates


@unittest.skipUnless(torch is not None, "requires PyTorch")
class PPOTests(unittest.TestCase):
    def test_terminal_reward_and_gae_use_learner_seat_and_zero_terminal_bootstrap(self):
        self.assertEqual(terminal_reward("p2", "p2"), 1.)
        self.assertEqual(terminal_reward("p2", "p1"), -1.)
        self.assertEqual(terminal_reward(None, "p1"), 0.)
        torch.testing.assert_close(episode_advantages([.2, -.4, .3], -1.), torch.tensor([-1.10825, -.535, -1.3]))
        torch.testing.assert_close(episode_advantages([.7], 1.), torch.tensor([.3]))

    def test_mask_removes_padding_from_sampling_entropy_and_gradients(self):
        logits = torch.tensor([[2., -1., 100.], [0., 100., 100.]], requires_grad=True)
        mask = torch.tensor([[True, True, False], [True, False, False]])
        dist = masked_distribution(logits, mask)
        self.assertTrue(torch.equal(dist.probs[~mask], torch.zeros(3)))
        self.assertTrue(mask[torch.arange(2), dist.sample()].all())
        self.assertAlmostEqual(dist.entropy()[1].item(), 0.)
        (-dist.log_prob(torch.tensor([1, 0])).mean() - .01 * dist.entropy().mean()).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue(torch.equal(logits.grad[~mask], torch.zeros(3)))

    def test_bc_initialisation_preserves_logits_and_value_is_state_only(self):
        bc = ActionRankingNetwork(4, 5, 8)
        model = ActorCriticNetwork(4, 5, 8)
        missing, unexpected = model.load_state_dict(bc.state_dict(), strict=False)
        self.assertEqual(set(missing), {f"value_head.{key}" for key in model.value_head.state_dict()})
        self.assertFalse(unexpected)
        states, actions = torch.rand(2, 4), torch.rand(2, 3, 5)
        logits, values = model.policy_value(states, actions)
        torch.testing.assert_close(logits, bc(states, actions))
        torch.testing.assert_close(values, model.policy_value(states, actions[:, :1])[1])

    def test_schedule_is_seeded_and_increases_frozen_opponent_share(self):
        config = dict(pairs=48, seed=110001, total_updates=6)
        first = _schedule(config, 1, 1)
        last = _schedule(config, 6, 3)
        self.assertEqual(first, _schedule(config, 1, 1))
        self.assertFalse({match[0] for match in first} & {match[0] for match in last})
        self.assertEqual(sum(name == "historical" for _, name, _ in first), 5)
        self.assertEqual(sum(name == "historical" for _, name, _ in last), 17)

    def test_shortlist_support_is_shared_by_rollout_and_inference(self):
        moves = {f"move{i}": {"base_power": 20 + 10 * i, "type": "normal", "category": "physical"} for i in range(8)}
        actions = tuple({"type": "doubles_turn", "slot_0": {"type": "move", "move_id": move, "target": 1},
                         "slot_1": {"type": "pass"}} for move in moves)
        board = {"active": [{"species": "arcanine", "hp": 1}, None],
                 "opponents": [{"species": "amoonguss", "hp": 1}, None], "turn": 1}
        dex = {"types": {}, "species": {}, "moves": moves}
        with patch("pokemon_agent.features.load_dex", return_value=dex), patch("pokemon_agent.tactics.load_dex", return_value=dex):
            decision = Decision(BATTLE, board, actions, (moves, {}))
            encoder = BattleEncoder()
            features = encoder.encode(decision)
            model = ActorCriticNetwork(len(features.state), len(features.actions[0]), 8)
            rollout = RolloutPolicy(model, encoder, 1, shortlist_only=True)
            inference = BattlePolicy(model, torch.device("cpu"), encoder=encoder, shortlist_only=True)
            indices, logits = inference.evaluate(decision)
            self.assertEqual(set(indices), {actions.index(action) for action in tactical_candidates(decision)})
            self.assertEqual(len(indices), 6)
            choice = rollout(decision)
            _, encoded, row, old_logp, _ = rollout.steps[-1]
            self.assertEqual(choice, actions[indices[row]])
            self.assertEqual(inference(decision), actions[indices[int(logits.argmax())]])
            self.assertEqual(len(encoded), 6)
            self.assertAlmostEqual(old_logp, logits.log_softmax(0)[row].item(), places=6)

            forced = Decision(BATTLE, board | {"turn": 2}, actions[-1:], (moves, {}))
            self.assertEqual(rollout(forced), actions[-1])
            self.assertEqual(inference(forced), actions[-1])
            self.assertEqual(rollout.steps[-1][2:4], (0, 0.))
