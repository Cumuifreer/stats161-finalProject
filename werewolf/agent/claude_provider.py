"""用 Claude 當 agent 的大腦 (取代官方範例的 OpenAI)。

.env 要有 ANTHROPIC_API_KEY。CLAUDE_MODEL 可以換模型 (預設 claude-opus-5-5)，
自己跟自己測試的時候可以先換 claude-haiku-5-5，比較省。
"""

from __future__ import annotations

import json
import os

import anthropic

from examples.llm.providers import ProviderError

DEFAULT_MODEL = "claude-opus-5-5"


class ClaudeProvider:
    def __init__(self, model=None, effort=None, client=None):
        self.model = model or os.environ.get("CLAUDE_MODEL") or DEFAULT_MODEL
        self.effort = effort or os.environ.get("CLAUDE_EFFORT") or "medium"
        self._client = client or anthropic.Anthropic()  # 從 ANTHROPIC_API_KEY 讀 key

    def complete_structured(self, messages, schema_name, schema, *, timeout=None):
        # 官方範例把 system 放在 messages[0]，Claude 要分開傳
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        chat = [m for m in messages if m["role"] != "system"]
        kw = dict(
            model=self.model,
            max_tokens=4000,
            system=system,
            messages=chat,
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
        )
        if timeout is not None:
            kw["timeout"] = timeout
        if self.model.startswith(("claude-opus-5", "claude-sonnet-5-5", "claude-fable")):
            # 被安全機制擋下時自動換模型重跑，免得卡住
            kw.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        try:
            resp = self._client.beta.messages.create(**kw)
        except anthropic.APIStatusError as e:
            raise ProviderError(f"Claude request failed (HTTP {e.status_code})") from None
        except anthropic.APIConnectionError:
            raise ProviderError("Claude request failed (connection)") from None

        if resp.stop_reason == "refusal":
            raise ProviderError("the model refused to answer")
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            data = json.loads(text)
        except ValueError:
            raise ProviderError("the model's reply was not valid JSON") from None
        if not isinstance(data, dict):
            raise ProviderError("the model's reply was not a JSON object")
        return data


def claude_from_env():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is not set — add it to .env.")
    return ClaudeProvider()
