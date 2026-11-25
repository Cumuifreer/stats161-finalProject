"""Bounded model choice over locally validated Pokémon actions."""

import json
import os
from pathlib import Path
from queue import Empty, Queue
from threading import Thread

from .game import BATTLE, Decision, PREVIEW
from .joint import JointPolicy
from .tactics import tactical_candidates


class LlmPolicy:
    def __init__(self, candidates=tactical_candidates, *, forecast: JointPolicy | None = None):
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv(Path.cwd() / ".env")
        provider = os.getenv("POKEMON_LLM_PROVIDER") or "openai"
        if provider == "deepseek":
            key_var, base_url, default_model = "DEEPSEEK_API_KEY", "https://api.deepseek.com", "deepseek-flash"
        elif provider == "openai":
            key_var, base_url, default_model = "OPENAI_API_KEY", None, "gpt-5.4-mini"
        else:
            raise ValueError("POKEMON_LLM_PROVIDER must be openai or deepseek")
        if not (key := os.getenv(key_var)):
            raise ValueError(f"Set {key_var} to use the LLM policy")
        self.client = OpenAI(api_key=key, base_url=base_url, max_retries=0)
        self.model = os.getenv("POKEMON_LLM_MODEL") or default_model
        self.provider = provider
        self.candidates = candidates
        self.forecast = forecast
        self.in_flight: Thread | None = None
        self.source = "local"
        self.fallbacks = 0
        self.model_choices = 0
        self.different_choices = 0

    def _ask(self, decision: Decision, candidates: list[dict], budget: float, result: Queue, summaries=None):
        choices = {str(index): action for index, action in enumerate(candidates)}
        if self.provider == "deepseek":
            options = {
                "extra_body": {"thinking": {"type": "disabled"}},
                "max_tokens": 64,
                "response_format": {"type": "json_object"},
            }
        else:
            options = {
                "reasoning_effort": "none",
                "max_completion_tokens": 64,
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "pokemon_choice",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"id": {"type": "string", "enum": list(choices)}},
                        "required": ["id"],
                        "additionalProperties": False,
                    },
                }},
            }
        payload = {
            "phase": decision.phase,
            "board": decision.observation,
            "move_info": decision.move_info,
            "candidates": choices,
        }
        if summaries:
            payload["forecast_notes"] = (
                "Approximate results under each candidate's most dangerous plausible opponent reply. "
                "Move order positions 0 and 1 are ours, 2 and 3 are the opponent's. "
                "KO positions are relative to the specified side. The opponent's two choices are listed in slot order."
            )
            payload["candidate_forecasts"] = summaries
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": (
                        "Choose one candidate to win this Pokémon VGC doubles match. "
                        "Consider speed control, coordinated targets, Protect, switches and the back line. "
                        "Target 1/2 is the opposing left/right slot, -1/-2 is our left/right slot, "
                        'and 0 means the move takes no explicit target. Return JSON like {"id":"0"} '
                        "with exactly one of the supplied candidate IDs."
                    )},
                    {"role": "user", "content": json.dumps(payload)},
                ],
                timeout=budget,
                **options,
            )
            content = response.choices[0].message.content
            result.put(choices.get(json.loads(content)["id"]))
        except Exception:
            # Provider failures and malformed replies use the same legal fallback.
            result.put(None)

    def observe(self, decision: Decision) -> None:
        if self.forecast is not None:
            self.forecast.observe(decision)

    def __call__(self, decision: Decision) -> dict:
        self.observe(decision)
        action = self._choose(decision)
        if self.forecast is not None:
            self.forecast.record(decision, action)
        return action

    def _choose(self, decision: Decision) -> dict:
        ranked = self.candidates(decision)[:6]
        summaries = None
        if self.forecast is not None and decision.phase == BATTLE:
            summaries = {
                str(ranked.index(action)): {key: value for key, value in details.items() if key != "score"}
                for action, details in self.forecast.rank(decision, ranked)
            }
        self.source = "forced"
        if len(ranked) == 1:
            return ranked[0]

        budget = 6.0 if decision.phase == PREVIEW else 3.0
        for seconds in (decision.seconds_left, decision.bank_seconds_left):
            if seconds is not None:
                budget = min(budget, seconds - 0.75)
        if budget >= 0.25 and not (self.in_flight and self.in_flight.is_alive()):
            result = Queue(maxsize=1)
            self.in_flight = Thread(
                target=self._ask, args=(decision, ranked, budget, result, summaries), daemon=True
            )
            self.in_flight.start()
            try:
                action = result.get(timeout=budget)
            except Empty:
                action = None
            if action is not None:
                self.source = "llm"
                self.model_choices += 1
                self.different_choices += action != ranked[0]
                return action
        self.source = "fallback"
        self.fallbacks += 1
        return ranked[0]
