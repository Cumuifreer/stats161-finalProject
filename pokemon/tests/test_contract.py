import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pokemon_agent.__main__ import HISTORICAL_CARDS, _bench, _catalog, _replay
from pokemon_agent.game import BATTLE, DRAFT, GAME_TYPE, PREVIEW, make_decision
from pokemon_agent.live import create_agent
from pokemon_agent.policy import heuristic
from pokemon_agent.sim import SEATS, _draft, play_game

ROSTER = ["amoonguss", "incineroar", "rillaboom", "gholdengo", "arcanine", "gyarados"]
PASS = {"type": "pass"}


def sdk_state(phase, observation, templates, action_ids):
    return SimpleNamespace(
        phase=phase, game_name=GAME_TYPE,
        raw={"observation": observation, "access_token": "secret-fixture-marker"},
        legal_actions=[
            SimpleNamespace(action_id=action_id, input={"action": template})
            for action_id, template in zip(action_ids, templates, strict=True)
        ],
    )


def forced_replacement():
    return {"slots": [
        {"slot": slot, "options": [PASS, {"type": "switch", "species": "gholdengo"}]}
        for slot in (1, 0)
    ]}


class ContractTests(unittest.TestCase):
    def test_two_fainted_one_reserve(self):
        decision = make_decision(BATTLE, {}, [forced_replacement()], "p1")
        self.assertEqual(len(decision.actions), 2)
        for action in decision.actions:
            self.assertEqual(
                {action["slot_0"]["type"], action["slot_1"]["type"]}, {"switch", "pass"},
            )
        with self.assertRaises(ValueError):
            decision.validate({"type": "doubles_turn", "slot_0": PASS, "slot_1": PASS})

    def test_slot_target_and_ranker(self):
        template = {"slots": [
            {"slot": 0, "options": [{"type": "move", "move_id": "flamethrower", "targets": [-2, 1]}]},
            {"slot": 1, "options": [{"type": "move", "move_id": "protect", "targets": [0]}]},
        ]}
        decision = make_decision(BATTLE, {
            "opponent_active_pokemon": [{"species": "Dondozo", "current_hp_fraction": 0.5}, None],
            "available_moves": [[{"id": "flamethrower", "base_power": 90, "accuracy": 100}], []],
        }, [template], "p1")
        action = heuristic(decision)
        self.assertEqual(action["slot_0"]["target"], 1)
        self.assertEqual(action["slot_1"]["target"], 0)
        self.assertEqual(decision.observation["opponents"][0]["species"], "dondozo")

    def test_moves_without_explicit_targets(self):
        template = {"slots": [
            {"slot": 0, "options": [{"type": "move", "move_id": "protect"}]},
            {"slot": 1, "options": [{"type": "move", "move_id": "rockslide", "targets": []}]},
        ]}
        decision = make_decision(BATTLE, {}, [template], "p1")
        self.assertEqual(heuristic(decision), {
            "type": "doubles_turn",
            "slot_0": {"type": "move", "move_id": "protect", "target": 0},
            "slot_1": {"type": "move", "move_id": "rockslide", "target": 0},
        })

    def test_preview_keeps_distinct_bring_and_ordered_leads(self):
        decision = make_decision(PREVIEW, {}, [{"roster": ROSTER}], "p1")
        self.assertEqual(len(decision.actions), 180)
        for action in decision.actions:
            self.assertEqual(len(set(action["bring"])), 4)
            self.assertEqual(len(set(action["leads"])), 2)
            self.assertLessEqual(set(action["leads"]), set(action["bring"]))

    def test_live_capture_can_replay_and_compare_catalog(self):
        cards = json.loads(HISTORICAL_CARDS.read_text())
        changed = dict(cards[1], item="New Item")
        picked, partial = (
            {key: card[key] for key in ("card_id", "species")} for card in (cards[0], cards[3])
        )
        raw = {
            "available_cards": [cards[0], changed],
            "rosters": {"p1": [], "p2": [cards[2], partial]},
            "picks": [{"public_reason": "secret-fixture-marker"}],
        }
        later = {"available_cards": [changed], "rosters": {"p1": [picked], "p2": [cards[2], partial]}}
        states = [
            sdk_state(DRAFT, raw, [{}, {}], [f"draft_pick:{card['card_id']}" for card in raw["available_cards"]]),
            sdk_state(DRAFT, later, [{}], [f"draft_pick:{changed['card_id']}"]),
            sdk_state(PREVIEW, {}, [{"roster": ROSTER}], ["select_lineup"]),
            sdk_state(BATTLE, {}, [forced_replacement()], ["doubles_turn"]),
        ]
        with TemporaryDirectory() as directory, patch.dict(os.environ, {"POKEMON_CAPTURE_DIR": directory}):
            agent = create_agent()
            context = SimpleNamespace(agent_id="p1", game_type=GAME_TYPE)
            self.assertEqual(
                [agent.choose_action(state, context)["type"] for state in states],
                ["draft_pick", "draft_pick", "select_lineup", "doubles_turn"],
            )
            capture, = Path(directory).glob("*.jsonl")
            self.assertNotIn("secret-fixture-marker", capture.read_text())
            args = argparse.Namespace(traces=[capture])
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(_replay(args), 0)
            self.assertEqual(json.loads(output.getvalue())["decisions"], 4)
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(_catalog(args), 0)
            catalog = json.loads(output.getvalue())
            self.assertEqual(catalog["confirmed"], sorted([cards[0]["card_id"], cards[2]["card_id"]]))
            self.assertEqual(catalog["changed_fields"], {changed["card_id"]: ["item"]})
            self.assertEqual(catalog["unobserved_fields"], {
                partial["card_id"]: [field for field in cards[3] if field not in partial],
            })

    def test_draft_snake_order_and_unique_items(self):
        catalog = json.loads(HISTORICAL_CARDS.read_text())
        pool, items = [], set()
        for card in catalog:
            if card["item"] not in items:
                pool.append(card)
                items.add(card["item"])
            if len(pool) == 17:
                break
        pool.append(catalog[8])  # Leftovers is already on Archaludon in this pool.
        order = []

        def choose(seat, decision):
            order.append(seat)
            if decision.observation["pick"] == 6:
                remaining = {card["card_id"] for card in decision.observation["cards"]}
                legal = {action["card_id"] for action in decision.actions}
                self.assertIn(catalog[8]["card_id"], remaining - legal)
                self.assertEqual(set(decision.observation["picked"][0]), {"card_id", "species"})
            return decision.actions[0]

        rosters = _draft(pool, "p2", choose)
        self.assertEqual(order, "p2 p1 p1 p2 p2 p1 p1 p2 p2 p1 p1 p2".split())
        for cards in rosters.values():
            self.assertEqual(len({card["item"] for card in cards}), 6)

    def test_local_clock_auto_drafts_and_forfeits_when_bank_runs_out(self):
        engine = Mock()
        engine.call.side_effect = [
            {"ended": False, "views": [
                {"seat": "p1", "phase": PREVIEW, "raw": {}, "templates": [{"roster": ROSTER}]},
            ]},
            {"ended": False, "views": [
                {"seat": "p1", "phase": BATTLE, "raw": {}, "templates": [forced_replacement()]},
            ]},
        ]
        samples = {seat: [] for seat in SEATS}
        times = [0, 16] + [0, 0] * 11 + [0, 84, 0, 3]
        cards = json.loads(HISTORICAL_CARDS.read_text())

        def policy(decision):
            if decision.phase == BATTLE:
                self.assertEqual(decision.seconds_left, 3)
                self.assertEqual(decision.bank_seconds_left, 3)
            return decision.actions[0]

        with patch("pokemon_agent.sim.perf_counter", side_effect=times), patch("pokemon_agent.sim.BATTLE_BANK", 3):
            winner = play_game(engine, cards, dict.fromkeys(SEATS, policy), 1, samples)
        self.assertEqual(winner, "p2")
        self.assertIn((DRAFT, 16000, "auto_action"), samples["p1"] + samples["p2"])
        self.assertEqual(samples["p1"][-2:], [(PREVIEW, 84000, None), (BATTLE, 3000, "forfeit")])
        self.assertEqual(engine.call.call_count, 2)

    def test_benchmark_uses_only_complete_pairs_for_wins(self):
        args = argparse.Namespace(
            cards=HISTORICAL_CARDS, testing_pools=None, pairs=2, seed=20, left="heuristic", right="random",
        )
        with patch("pokemon_agent.__main__.Showdown"), patch(
            "pokemon_agent.__main__.play_game", side_effect=["p1", "p1", "p1", ValueError("invalid choice")],
        ), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(_bench(args), 1)
        report = json.loads(output.getvalue())
        self.assertEqual(report["completed_games"], 3)
        self.assertEqual(report["completed_pairs"], 1)
        self.assertEqual([agent["win_rate"] for agent in report["agents"]], [0.5, 0.5])
        self.assertEqual(report["errors"][0]["left_seat"], "p2")


if __name__ == "__main__":
    unittest.main()
