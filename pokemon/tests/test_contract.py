import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pokemon_agent.__main__ import HISTORICAL_CARDS, _bench, _catalog, _replay
from pokemon_agent.game import BATTLE, DRAFT, GAME_TYPE, PREVIEW, Decision, make_decision
from pokemon_agent.live import create_agent
from pokemon_agent.llm import LlmPolicy
from pokemon_agent.policy import heuristic
from pokemon_agent.sim import SEATS, _draft, play_game
from pokemon_agent.tactics import tactical, tactical_candidates

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

    def test_tactical_policy_accounts_for_immunity(self):
        options = [
            {"type": "move", "move_id": move, "targets": [1], "base_power": power, "category": "special"}
            for move, power in (("earthpower", 90), ("thunderbolt", 80))
        ]
        template = {"slots": [{"slot": 0, "options": options}, {"slot": 1, "options": [PASS]}]}
        decision = make_decision(BATTLE, {
            "opponent_active_pokemon": [{"species": "Gyarados", "types": ["Water", "Flying"]}, None],
        }, [template], "p1")
        dex = {
            "types": {"flying": {"ground": 0, "electric": 2}, "water": {"electric": 2}},
            "species": {},
            "moves": {"earthpower": {"type": "ground"}, "thunderbolt": {"type": "electric"}},
        }
        with patch("pokemon_agent.tactics.load_dex", return_value=dex):
            self.assertEqual(tactical(decision)["slot_0"]["move_id"], "thunderbolt")

    def test_tactical_shortlist_keeps_legal_alternatives(self):
        options = [
            {"type": "move", "move_id": f"attack{i}", "target": 1} for i in range(6)
        ] + [
            {"type": "switch", "species": "gholdengo"},
            {"type": "move", "move_id": "protect", "target": 0},
            {"type": "move", "move_id": "tailwind", "target": 0},
        ]
        ranked = [{"type": "doubles_turn", "slot_0": action, "slot_1": PASS} for action in options]
        decision = Decision(BATTLE, {}, tuple(ranked))
        with patch("pokemon_agent.tactics.tactical_actions", return_value=ranked):
            shortlist = tactical_candidates(decision)
        self.assertEqual(len(shortlist), 6)
        self.assertEqual(shortlist[0], ranked[0])
        for action in ranked[-3:]:
            self.assertIn(action, shortlist)
        for action in shortlist:
            decision.validate(action)

    def test_calibrated_policy_uses_weather_for_fixed_card_damage(self):
        options = [
            {"type": "move", "move_id": move, "targets": [1], "base_power": power, "category": "special"}
            for move, power in (("flamethrower", 90), ("waterpulse", 60))
        ]
        template = {"slots": [{"slot": 0, "options": options}, {"slot": 1, "options": [PASS]}]}
        raw = {
            "active_pokemon": [{"species": "Hatterene"}, None],
            "opponent_active_pokemon": [{"species": "Porygon2"}, None],
            "weather": {"RAINDANCE": 1},
        }
        decision = make_decision(BATTLE, raw, [template], "p1")
        profile = {"level": 50, "stats": {"hp": 150, "spa": 100, "spd": 100, "spe": 100}}
        profiles = dict.fromkeys(("hatterene", "porygon2"), profile)
        dex = {"types": {}, "species": {}, "moves": {
            "flamethrower": {"type": "fire"}, "waterpulse": {"type": "water"},
        }}
        with patch("pokemon_agent.tactics.load_dex", return_value=dex):
            self.assertEqual(tactical(decision)["slot_0"]["move_id"], "flamethrower")
            action = tactical(decision, card_profiles=profiles)
        self.assertEqual(decision.validate(action)["slot_0"]["move_id"], "waterpulse")

    def test_preview_scores_reserves_and_avoids_reversed_lead_duplicates(self):
        leads = ["incineroar", "rillaboom"]
        bring = leads + ["amoonguss", "arcanine"]
        counter = leads + ["amoonguss", "gholdengo"]
        options = tuple(
            {"type": "select_lineup", "bring": members, "leads": order}
            for members in (bring, counter) for order in (leads, leads[::-1])
        )
        decision = Decision(PREVIEW, {
            "team": [
                {"species": species, "moves": ["thunderbolt" if species == "gholdengo" else "tackle"]}
                for species in set(bring + counter)
            ],
            "opponent_team": [{"species": "gyarados", "types": ["water", "flying"]}],
        }, options)
        dex = {
            "types": {"water": {"electric": 2}, "flying": {"electric": 2}},
            "species": {},
            "moves": {
                "thunderbolt": {"type": "electric", "category": "special", "base_power": 80, "accuracy": 1},
                "tackle": {"type": "normal", "category": "physical", "base_power": 80, "accuracy": 1},
            },
        }
        with patch("pokemon_agent.tactics.load_dex", return_value=dex):
            shortlist = tactical_candidates(decision)
        self.assertEqual(shortlist, [options[2], options[0]])

    def test_battle_reserves_match_own_legal_switches(self):
        raw = {
            "team": {
                "p1: Gholdengo": {"species": "Gholdengo", "current_hp": 68, "max_hp": 100,
                                 "status": "BRN", "moves": {"Shadow Ball": {}}},
                "p1: Amoonguss": {"species": "Amoonguss", "current_hp_fraction": 1},
            },
            "opponent_team": {"p2: Gholdengo": {"species": "Gholdengo", "current_hp_fraction": 1}},
        }
        decision = make_decision(BATTLE, raw, [forced_replacement()], "p1")
        self.assertEqual(decision.observation["reserves"], {
            "gholdengo": {"hp": .68, "status": "brn", "moves": ["shadowball"]},
        })

    def test_deepseek_picks_a_legal_candidate_and_falls_back(self):
        cards = [{"card_id": card} for card in ("a", "b")]
        decision = make_decision(DRAFT, {"available_cards": cards}, cards, "p1")
        options = decision.actions
        client = Mock()
        create = client.chat.completions.create
        create.side_effect = [
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
            for content in ('{"id":"1"}', '{"id":"9"}')
        ]
        factory = Mock(return_value=client)
        with patch.dict(os.environ, {"POKEMON_LLM_PROVIDER": "deepseek", "DEEPSEEK_API_KEY": "test"}, clear=True):
            with patch.dict(sys.modules, {
                "openai": SimpleNamespace(OpenAI=factory),
                "dotenv": SimpleNamespace(load_dotenv=Mock()),
            }):
                policy = LlmPolicy()
        self.assertEqual(policy(decision), options[1])
        self.assertEqual((policy.source, policy.model_choices, policy.different_choices), ("llm", 1, 1))
        policy.in_flight.join()
        self.assertEqual(policy(decision), options[0])
        self.assertEqual((policy.source, policy.fallbacks), ("fallback", 1))
        factory.assert_called_once_with(api_key="test", base_url="https://api.deepseek.com", max_retries=0)
        request = create.call_args.kwargs
        self.assertEqual(request["model"], "deepseek-flash")
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertEqual(request["extra_body"], {"thinking": {"type": "disabled"}})

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
            cards=HISTORICAL_CARDS, testing_pools=None, pairs=3, seed=20, left="heuristic", right="random",
            battle_only=False, preview_only=False, ace_trainer=None, pair_results=True,
        )
        with patch("pokemon_agent.__main__.Showdown"), patch(
            "pokemon_agent.__main__.play_game",
            side_effect=["p1", "p1", "p1", ValueError("invalid choice"), "p2", "p2"],
        ), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(_bench(args), 1)
        report = json.loads(output.getvalue())
        self.assertEqual(report["completed_games"], 5)
        self.assertEqual(report["completed_pairs"], 2)
        self.assertEqual([agent["win_rate"] for agent in report["agents"]], [0.5, 0.5])
        self.assertEqual([agent["win_rate_ci95"] for agent in report["agents"]], [[0.5, 0.5]] * 2)
        self.assertEqual(report["paired_scores"], {"20": [1, 0], "22": [0, 1]})
        self.assertEqual(report["errors"][0]["left_seat"], "p2")


if __name__ == "__main__":
    unittest.main()
