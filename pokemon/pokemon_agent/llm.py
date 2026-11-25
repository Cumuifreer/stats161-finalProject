"""Bounded model choice over locally validated Pokémon actions."""

import json
import os
from queue import Empty, Queue
from threading import Thread

from .game import Decision, PREVIEW
from .policy import ranked_actions


class LlmPolicy:
    def __init__(self):
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("Set OPENAI_API_KEY to use the LLM policy")
        from openai import OpenAI

        self.client = OpenAI(api_key=key, max_retries=0)
        self.model = os.getenv("POKEMON_LLM_MODEL") or "gpt-5.4-mini"
        self.in_flight: Thread | None = None
        self.source = "local"
        self.fallbacks = 0

    def _ask(self, decision: Decision, candidates: list[dict], budget: float, result: Queue):
        choices = {str(index): action for index, action in enumerate(candidates)}
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": (
                        "Choose one candidate to win this Pokémon VGC doubles match. "
                        "Consider speed control, coordinated targets, Protect, switches and the back line. "
                        "Target 1/2 is the opposing left/right slot, -1/-2 is our left/right slot, "
                        "and 0 means the move takes no explicit target. Return the candidate ID."
                    )},
                    {"role": "user", "content": json.dumps({
                        "phase": decision.phase,
                        "board": decision.observation,
                        "move_info": decision.move_info,
                        "candidates": choices,
                    })},
                ],
                reasoning_effort="none",
                max_completion_tokens=64,
                response_format={"type": "json_schema", "json_schema": {
                    "name": "pokemon_choice",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"id": {"type": "string", "enum": list(choices)}},
                        "required": ["id"],
                        "additionalProperties": False,
                    },
                }},
                timeout=budget,
            )
            content = response.choices[0].message.content
            result.put(choices.get(json.loads(content)["id"]))
        except Exception:
            # Provider failures and malformed replies use the same legal fallback.
            result.put(None)

    def __call__(self, decision: Decision) -> dict:
        ranked = ranked_actions(decision)
        self.source = "forced"
        if len(ranked) == 1:
            return ranked[0]

        budget = 6.0 if decision.phase == PREVIEW else 3.0
        if decision.seconds_left is not None:
            budget = min(budget, decision.seconds_left - 0.75)
        if budget >= 0.25 and not (self.in_flight and self.in_flight.is_alive()):
            result = Queue(maxsize=1)
            self.in_flight = Thread(
                target=self._ask, args=(decision, ranked[:6], budget, result), daemon=True
            )
            self.in_flight.start()
            try:
                action = result.get(timeout=budget)
            except Empty:
                action = None
            if action is not None:
                self.source = "llm"
                return action
        self.source = "fallback"
        self.fallbacks += 1
        return ranked[0]
