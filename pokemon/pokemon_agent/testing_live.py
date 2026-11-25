"""Run one Pokémon Testing assignment with the frozen local hybrid."""

import argparse
from datetime import datetime, timezone
from hashlib import blake2s, sha256
import json
import os
from pathlib import Path
from time import monotonic, sleep

from .game import BATTLE, DRAFT, GAME_TYPE, PREVIEW
from .live import LiveAgent
from .sim import Showdown

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "checkpoints/roster-v1/expanded"
WEIGHT_SHA256 = "a458a24568361e1515be1ee13e7b8bb094b0e8147a0799e1eecec4ccedc33d5c"
AGENT_SPEC = "pokemon_agent.testing_live:create_testing_agent"
CONTROL_URL = "https://api.altruagent-game.com"


class TestingPolicy:
    def __init__(self):
        from .roster import roster_factory

        weights = CHECKPOINT / "state_dict.pt"
        if sha256(weights.read_bytes()).hexdigest() != WEIGHT_SHA256:
            raise ValueError("Testing requires the frozen expanded Draft weights")
        cards = json.loads((ROOT / "data/historical_cards.json").read_text(encoding="utf-8"))
        self.roster = roster_factory(CHECKPOINT, cards, None, "cpu", "prior", "rollout")()
        self.source = None

    def __call__(self, decision):
        before = self.roster.fallbacks
        if decision.phase == PREVIEW:
            with Showdown() as worker:
                self.roster.worker = worker
                try:
                    action = self.roster(decision)
                finally:
                    self.roster.worker = None
        else:
            action = self.roster(decision)
        self.source = {DRAFT: "learned_card_prior", PREVIEW: "lineup_rollout", BATTLE: "joint"}[decision.phase]
        if self.roster.fallbacks > before:
            self.source += "_fallback"
        return action


class TestingAgent(LiveAgent):
    def __init__(self):
        super().__init__(TestingPolicy())
        self.receipts = self.capture.with_suffix(".receipts.jsonl") if self.capture else None
        self.started = None
        self.number = 0
        self.phase = None
        self.state_version = None

    def choose_action(self, state, context):
        if context.tournament_id:
            raise ValueError("The experimental entry accepts Testing only")
        if self.number == 0 and self.capture and os.environ.get("POKEMON_TESTING_BATCH") == "1":
            match = blake2s(context.session_id.encode(), digest_size=12).hexdigest()
            folder = self.capture.parent / match
            folder.mkdir(parents=True, exist_ok=True)
            self.capture = folder / self.capture.name
            self.receipts = self.capture.with_suffix(".receipts.jsonl")
        self.started = monotonic()
        self.number += 1
        self.phase = state.phase
        self.state_version = state.state_version
        return super().choose_action(state, context)

    def on_action_result(self, result, context):
        if self.receipts is None:
            return
        row = {"decision_number": self.number, "decision_phase": self.phase, "sent_version": self.state_version,
               "seat": context.seat_position, "worker_pid": os.getpid(), "elapsed_ms": round((monotonic() - self.started) * 1000, 2),
               "source": self.policy.source, "utc": datetime.now(timezone.utc).isoformat()}
        post = result.get("state")
        for key in ("ok", "success", "accepted", "error", "error_code", "status", "phase", "is_terminal", "state_version"):
            if isinstance(value := result.get(key), (bool, int, float, str)):
                row[key] = value[:100] if isinstance(value, str) else value
        if isinstance(post, dict):
            observation = post.get("observation")
            row["post"] = {key: post[key] for key in ("phase", "state_version", "is_terminal") if key in post}
            if isinstance(observation, dict) and isinstance(clock := observation.get("clock"), dict):
                row["post"]["clock"] = {key: clock[key] for key in ("decision_seconds_left", "bank_seconds_left") if key in clock}
        try:
            with self.receipts.open("a", encoding="utf-8") as output:
                output.write(json.dumps(row) + "\n")
        except OSError as error:
            self.receipts = None
            print(f"Testing receipt capture disabled ({type(error).__name__})", flush=True)


def create_testing_agent():
    return TestingAgent()


def _testing(assignment):
    return (assignment.context == "testing" and assignment.game_type == GAME_TYPE
            and not assignment.tournament_id and bool(assignment.match_id))


def main():
    from dotenv import load_dotenv
    from altruagent.errors import AltruAgentError
    from altruagent.official import OfficialAgentClient
    from altruagent.supervisor import TESTING, TournamentState, run_tournament_once

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--match-id", action="append", default=[], help="Play a specific active Testing match (repeat for a batch)")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    if not os.getenv("ALTRUAGENT_OFFICIAL_AGENT_KEY"):
        parser.error("Set ALTRUAGENT_OFFICIAL_AGENT_KEY in pokemon/.env")
    os.environ["ALTRUAGENT_CONTROL_URL"] = CONTROL_URL
    folder = args.capture_dir.resolve()
    os.environ["POKEMON_CAPTURE_DIR"] = str(folder / "decisions")
    requested = set(args.match_id)
    os.environ["POKEMON_TESTING_BATCH"] = "1" if len(requested) > 1 else "0"
    folder.mkdir(parents=True, exist_ok=True)
    TestingPolicy()

    class SelectedTestingMatches(OfficialAgentClient):
        def __init__(self):
            super().__init__(control_url=CONTROL_URL, load_env_file=False)
            self.matches = set(requested)
            self.reported_multiple = False
            self.empty_polls = 0

        def assignments(self):
            seats = [seat for seat in super().assignments() if _testing(seat)]
            if not self.matches:
                matches = {seat.match_id for seat in seats}
                if len(matches) != 1:
                    if len(matches) > 1 and not self.reported_multiple:
                        print("Multiple Pokémon Testing matches are active. Waiting for one.", flush=True)
                        self.reported_multiple = True
                    return []
                self.matches = matches
                print("Locked one Pokémon Testing match.", flush=True)
            selected = [seat for seat in seats if seat.match_id in self.matches]
            self.empty_polls = 0 if selected else self.empty_polls + 1
            if len(self.matches) == 1 and selected and not (folder / "match.json").exists():
                (folder / "match.json").write_text(json.dumps({"match_id": selected[0].match_id, "context": "testing",
                    "game_type": GAME_TYPE, "weights_sha256": WEIGHT_SHA256,
                    "seats": [seat.seat_position for seat in selected]}, indent=2) + "\n", encoding="utf-8")
            return selected

    completed = 0

    def log(message):
        nonlocal completed
        print(message, flush=True)
        completed += message == "Match finished."

    try:
        with SelectedTestingMatches() as client:
            client.authenticate()
            existing = [seat for seat in OfficialAgentClient.assignments(client) if _testing(seat)]
            if not requested <= {seat.match_id for seat in existing}:
                print("A specified Pokémon Testing match is no longer active.", flush=True)
                return 1
            if existing and not requested:
                print("An existing Pokémon Testing match is active. Start this runner with an empty Testing queue.", flush=True)
                return 1
            if len(requested) > 1:
                metadata = [{"match_id": match, "capture_key": blake2s(match.encode(), digest_size=12).hexdigest(),
                             "context": "testing", "game_type": GAME_TYPE, "weights_sha256": WEIGHT_SHA256,
                             "seats": [seat.seat_position for seat in existing if seat.match_id == match],
                             "seat_ids": {seat.seat_id: seat.seat_position for seat in existing if seat.match_id == match}}
                            for match in args.match_id]
                (folder / "matches.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            print(f"Ready for {len(requested) or 1} Pokémon Testing match(es) with the expanded card prior, lineup rollout and Joint. Weights {WEIGHT_SHA256[:12]}.", flush=True)
            state = TournamentState()
            try:
                while completed < (len(client.matches) or 1) and not state.key_refused:
                    run_tournament_once(client, state, agent_spec=AGENT_SPEC, kinds={TESTING}, log=log)
                    if client.matches and client.empty_polls >= 3 and len(state.registry) == 0 and completed < len(client.matches):
                        print(f"Some Testing assignments ended without a completed worker result ({completed}/{len(client.matches)}).", flush=True)
                        break
                    if completed < (len(client.matches) or 1):
                        sleep(1.)
            finally:
                state.registry.terminate_all(4.)
            return 0 if completed == (len(client.matches) or 1) else 1
    except AltruAgentError as error:
        print(f"Testing connection stopped ({type(error).__name__}, HTTP {getattr(error, 'status_code', None)}).", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
