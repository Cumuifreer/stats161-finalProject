import json
from pathlib import Path
from shutil import which
import unittest

from pokemon_agent.game import BATTLE, PREVIEW, make_decision
from pokemon_agent.search import _key, _winner
from pokemon_agent.sim import FORMAT, SEATS, Showdown


ROOT = Path(__file__).resolve().parents[1]


def _slot_pair(kind, left, right):
    if kind == "switch":
        slot = lambda species: {"type": "switch", "species": species}
    else:
        slot = lambda move: {"type": "move", "move_id": move[0], "target": move[1]}
    return {"type": "doubles_turn", "slot_0": slot(left), "slot_1": slot(right)}


class SearchTests(unittest.TestCase):
    def test_benefit_must_repeat_across_common_draws(self):
        base = [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
        outlier = [[2., -.3, -.3], [2., -.3, -.3], [2., -.3, -.3]]
        reliable = [[.3, .2, .3], [.2, .3, .2], [.3, .2, .3]]
        self.assertEqual(_winner([base, outlier]), 0)
        self.assertEqual(_winner([base, reliable]), 1)

    @unittest.skipUnless(which("node") and (ROOT / "node_modules/pokemon-showdown").is_dir(),
                         "requires local Showdown")
    def test_public_worker_rebuilds_both_masks_and_ignores_pending_opponent_action(self):
        cards = json.loads((ROOT / "data/historical_cards.json").read_text(encoding="utf-8"))
        by_name = {card["species"]: card for card in cards}
        names = [["Arcanine", "Archaludon", "Basculegion", "Gholdengo"], ["Cresselia", "Dondozo", "Lucario", "Sylveon"]]
        rosters = {seat: [by_name[name] for name in team] for seat, team in zip(SEATS, names, strict=True)}
        with Showdown() as game, Showdown(public_search=True) as worker:
            worker.call("configure", cards=cards)
            reply = game.call("start", format=FORMAT, seed=[129, 43, 123, 7], rosters=rosters, search_seats=list(SEATS))
            legal_rosters = [list(view["templates"][0]["roster"]) for view in reply["views"]]
            for view in reply["views"]:
                preview = make_decision(PREVIEW, view["raw"], view["templates"], view["seat"])
                reply = game.call("choose", seat=view["seat"], action=preview.actions[0])
            for seat, team in zip(SEATS, legal_rosters, strict=True):
                reply = game.call("choose", seat=seat, action=_slot_pair("switch", *team[2:]))
            game.call("choose", seat="p1", action=_slot_pair("move", ("aquajet", 1), ("shadowball", 2)))
            reply = game.call("choose", seat="p2", action=_slot_pair("move", ("bulletpunch", 1), ("protect", 0)))
            views = {view["seat"]: view for view in reply["views"]}
            our_view = views["p2"]
            evidence = our_view["search"]
            self.assertFalse(evidence["force_switch"])
            actual = {seat: make_decision(BATTLE, view["raw"], view["templates"], seat) for seat, view in views.items()}
            settings = dict(format=FORMAT, seat="p2", observation=actual["p2"].observation, rosters=legal_rosters,
                            events=evidence["events"], own_pp=evidence["own_pp"])
            missing_reveal = {**settings, "events": [line for line in evidence["events"]
                                                    if not line.startswith("|switch|p1a: Basculegion|")]}
            self.assertEqual(worker.call("prepare", **missing_reveal)["reason"], "unrevealed_roster")
            result = worker.call("prepare", **settings)
            self.assertIn("views", result, result)
            for view in result["views"]:
                seat = view["seat"]
                restored = make_decision(BATTLE, view["raw"], view["templates"], seat)
                self.assertEqual({_key(action) for action in actual[seat].actions}, {_key(action) for action in restored.actions})
            attack = _slot_pair("move", ("aquajet", 1), ("shadowball", 2))
            replies = [_slot_pair("move", ("bulletpunch", 1), ("protect", 0)),
                       _slot_pair("move", ("closecombat", 1), ("hypervoice", 0))]
            game_reply = game.call("choose", seat="p1", action=attack)
            self.assertEqual(game_reply["views"], [our_view])
            seeds = [[0, 0, 0, i] for i in (1, 2, 3)]
            matrix = worker.call("evaluate", seat="p2", ours=replies, replies=[attack], seeds=seeds, budget_ms=1000)["matrix"]
            reversed_matrix = worker.call("evaluate", seat="p2", ours=replies[::-1], replies=[attack],
                                          seeds=seeds, budget_ms=1000)["matrix"]
            self.assertEqual(matrix, reversed_matrix[::-1])
            self.assertEqual([len(row[0]) for row in matrix], [3, 3])
            self.assertEqual(worker.call("evaluate", seat="p2", ours=replies, replies=[attack], seeds=seeds,
                                         budget_ms=0)["reason"], "budget")

    @unittest.skipUnless(which("node") and (ROOT / "node_modules/pokemon-showdown").is_dir(),
                         "requires local Showdown")
    def test_midgame_field_durations_survive_branching_and_expire(self):
        cards = json.loads((ROOT / "data/historical_cards.json").read_text(encoding="utf-8"))
        by_name = {card["species"]: card for card in cards}
        names = [["Arcanine", "Archaludon", "Pelipper", "Gholdengo"], ["Cresselia", "Dondozo", "Farigiraf", "Sylveon"]]
        rosters = {seat: [by_name[name] for name in team] for seat, team in zip(SEATS, names, strict=True)}
        with Showdown() as game, Showdown(public_search=True) as worker:
            worker.call("configure", cards=cards)
            reply = game.call("start", format=FORMAT, seed=[129, 43, 123, 7], rosters=rosters, search_seats=list(SEATS))
            legal_rosters = [list(view["templates"][0]["roster"]) for view in reply["views"]]
            for view in reply["views"]:
                preview = make_decision(PREVIEW, view["raw"], view["templates"], view["seat"])
                reply = game.call("choose", seat=view["seat"], action=preview.actions[0])
            for seat, team in zip(SEATS, legal_rosters, strict=True):
                reply = game.call("choose", seat=seat, action=_slot_pair("switch", *team[2:]))

            def rebuild(response):
                view = next(view for view in response["views"] if view["seat"] == "p2")
                decision = make_decision(BATTLE, view["raw"], view["templates"], "p2")
                evidence = view["search"]
                settings = dict(format=FORMAT, seat="p2", observation=decision.observation, rosters=legal_rosters,
                                events=evidence["events"], own_pp=evidence["own_pp"])
                restored = worker.call("prepare", **settings)
                self.assertIn("fields", restored, restored)
                return restored["fields"], settings

            self.assertEqual(rebuild(reply)[0], {"weather": {"raindance": 7}, "fields": {}, "sides": [{}, {}]})
            tailwind = _slot_pair("move", ("tailwind", 0), ("protect", 0))
            room = _slot_pair("move", ("trickroom", 0), ("protect", 0))
            game.call("choose", seat="p1", action=tailwind)
            reply = game.call("choose", seat="p2", action=room)
            timers, settings = rebuild(reply)
            self.assertEqual(timers, {"weather": {"raindance": 6}, "fields": {"trickroom": 4}, "sides": [{"tailwind": 3}, {}]})
            settings["events"] = [event for event in settings["events"] if not (
                event.startswith("|-weather|RainDance|") and "[upkeep]" not in event)]
            self.assertEqual(worker.call("prepare", **settings)["reason"], "field_mismatch")
            rebuild(reply)

            wait_ours = _slot_pair("move", ("helpinghand", -2), ("calmmind", 0))
            wait_theirs = _slot_pair("move", ("protect", 0), ("nastyplot", 0))

            def branch(ours):
                audit = worker.call("evaluate", seat="p2", ours=[ours], replies=[wait_theirs], seeds=[[0, 0, 0, 7]],
                                    budget_ms=1000, audit_fields=True)
                self.assertIn("matrix", audit, audit)
                return audit["matrix"][0][0][0]

            self.assertEqual(branch(wait_ours), {"turn": 4, "fields": {"weather": {"raindance": 5}, "fields": {"trickroom": 3},
                                                                     "sides": [{"tailwind": 2}, {}]}})
            self.assertEqual(branch(room)["fields"]["fields"], {})
            for _ in range(2):
                game.call("choose", seat="p1", action=wait_theirs)
                reply = game.call("choose", seat="p2", action=wait_ours)
            rebuild(reply)
            self.assertEqual(branch(wait_ours), {"turn": 6, "fields": {"weather": {"raindance": 3}, "fields": {"trickroom": 1},
                                                                     "sides": [{}, {}]}})
            game.call("choose", seat="p1", action=wait_theirs)
            reply = game.call("choose", seat="p2", action=wait_ours)
            rebuild(reply)
            self.assertEqual(branch(wait_ours)["fields"]["fields"], {})
            for _ in range(2):
                game.call("choose", seat="p1", action=wait_theirs)
                reply = game.call("choose", seat="p2", action=wait_ours)
            self.assertEqual(rebuild(reply)[0]["weather"], {"raindance": 1})
            self.assertEqual(branch(wait_ours)["fields"]["weather"], {})
