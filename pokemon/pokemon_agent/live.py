"""Official starter hooks. Credentials and state versions stay in the SDK."""

from datetime import datetime, timezone
from hashlib import blake2s
import json
import os
from pathlib import Path
from time import monotonic
from uuid import uuid4

from .game import DRAFT, GAME_TYPE, make_decision
from .policy import heuristic


def _remaining(phase: str, observation: dict) -> float | None:
    if phase == DRAFT and (deadline := observation.get("decision_deadline_at")):
        try:
            end = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
            if end.tzinfo is not None:
                return max(0.0, (end - datetime.now(timezone.utc)).total_seconds())
        except (AttributeError, ValueError):
            pass
    clock = observation.get("clock") or {}
    seconds = clock.get("decision_seconds_left")
    return float(seconds) if isinstance(seconds, (int, float)) else None


class LiveAgent:
    def __init__(self, policy):
        self.policy = policy
        self.capture = None
        if directory := os.environ.get("POKEMON_CAPTURE_DIR"):
            folder = Path(directory)
            folder.mkdir(parents=True, exist_ok=True)
            self.capture = folder / f"{uuid4().hex}.jsonl"

    def choose_action(self, state, context) -> dict:
        if (context.game_type or state.game_name) != GAME_TYPE:
            raise ValueError(f"Expected a {GAME_TYPE} assignment")
        started = monotonic()
        raw = state.raw["observation"]
        templates = []
        for action in state.legal_actions:
            if action.action_id.startswith("draft_pick:"):
                templates.append({"card_id": action.action_id.split(":", 1)[1]})
            else:
                templates.append(action.input["action"])
        clock = raw.get("clock") or {}
        decision = make_decision(
            state.phase, raw, templates, context.agent_id,
            _remaining(state.phase, raw), clock.get("bank_seconds_left"),
        )
        action = decision.validate(self.policy(decision))
        if self.capture:
            record = {
                "decision": decision.snapshot(),
                "action": action,
                "source": getattr(self.policy, "source", "heuristic"),
                "decision_ms": round((monotonic() - started) * 1000, 2),
            }
            if session_id := getattr(context, "session_id", None):
                record["match"] = blake2s(session_id.encode(), digest_size=12).hexdigest()
            try:
                with self.capture.open("a", encoding="utf-8") as output:
                    output.write(json.dumps(record) + "\n")
            except OSError as error:
                # A local logging error should not discard a playable action.
                self.capture = None
                print(f"Pokémon capture disabled ({type(error).__name__})", flush=True)
        return action


def create_agent():
    return LiveAgent(heuristic)


def create_llm_agent():
    from .llm import LlmPolicy

    return LiveAgent(LlmPolicy())
