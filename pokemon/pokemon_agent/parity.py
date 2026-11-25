"""Compare local decisions with both seats of official Testing self-play."""

from collections import defaultdict
import json

from .game import BATTLE, DRAFT, DRAFT_SIZE, PREVIEW, Decision, make_decision
from .sim import FORMAT, SEATS, Showdown, _draft


def _compare(report, stage, local, official, **where):
    report["compared"][stage] += 1
    fields = [key for key in sorted(local.observation.keys() | official.observation.keys())
              if local.observation.get(key) != official.observation.get(key)]
    local_actions = {json.dumps(action, sort_keys=True) for action in local.actions}
    official_actions = {json.dumps(action, sort_keys=True) for action in official.actions}
    if fields or local_actions != official_actions or local.move_info != official.move_info:
        report["differences"].append({
            "stage": stage, **where, "observation_fields": fields,
            "local_only_actions": len(local_actions - official_actions),
            "official_only_actions": len(official_actions - local_actions),
            "move_metadata_differs": local.move_info != official.move_info,
        })


def compare_self_play(traces):
    groups = defaultdict(list)
    for path, records in traces:
        records = list(records)
        match = next((row["match"] for row in records if "match" in row), None)
        records = [(Decision.from_snapshot(row["decision"]), row["action"]) for row in records]
        previews = [decision for decision, _ in records if decision.phase == PREVIEW]
        if len(previews) != 1:
            raise ValueError(f"Expected one team preview in {path}")
        roster = previews[0].observation
        teams = [tuple(mon["species"] for mon in roster[field]) for field in ("team", "opponent_team")]
        groups[match or tuple(sorted(teams))].append((path.name, records))

    report = {"matches": len(groups), "compared": {DRAFT: 0, PREVIEW: 0, "opening": 0}, "differences": []}
    with Showdown() as engine:
        for group in groups.values():
            if len(group) != 2:
                raise ValueError(f"Expected both sides of one self-play match for {group[0][0]}")
            traces_by_seat = dict(zip(SEATS, group, strict=True))
            picks = {decision.observation["pick"]: (seat, decision, action)
                     for seat, (_, records) in traces_by_seat.items()
                     for decision, action in records if decision.phase == DRAFT}
            if set(picks) != set(range(1, 13)) or len(picks[1][1].observation["cards"]) != DRAFT_SIZE:
                raise ValueError(f"Expected a complete draft in {group[0][0]}")

            def choose(seat, local):
                pick = local.observation["pick"]
                expected_seat, official, action = picks[pick]
                if expected_seat != seat:
                    raise ValueError(f"Draft order differs at pick {pick} in {group[0][0]}")
                _compare(report, DRAFT, local, official, trace=traces_by_seat[seat][0], pick=pick)
                return local.validate(action)

            first, opening_draft, _ = picks[1]
            rosters = _draft(opening_draft.observation["cards"], first, choose)
            reply = engine.call("start", format=FORMAT, seed=[1, 2, 3, 4], rosters=rosters)
            for view in reply["views"]:
                seat = view["seat"]
                path, records = traces_by_seat[seat]
                official, action = next(record for record in records if record[0].phase == PREVIEW)
                local = make_decision(PREVIEW, view["raw"], view["templates"], seat)
                _compare(report, PREVIEW, local, official, trace=path)
                reply = engine.call("choose", seat=seat, action=local.validate(action))

            for view in reply["views"]:
                seat = view["seat"]
                path, records = traces_by_seat[seat]
                official = next((decision for decision, _ in records if decision.phase == BATTLE), None)
                if official is None or official.observation["turn"] != 1:
                    raise ValueError(f"Expected the opening battle decision in {path}")
                local = make_decision(BATTLE, view["raw"], view["templates"], seat)
                _compare(report, "opening", local, official, trace=path)
    return report
