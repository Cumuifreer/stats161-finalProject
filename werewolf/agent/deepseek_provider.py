"""DeepSeek 版的 provider。

DeepSeek 的 API 跟 OpenAI 差不多，但不支援 json_schema，只有 json_object，
所以把 schema 直接寫進 prompt，回來的答案再交給 LLMAgent 檢查。
.env 要有 DEEPSEEK_API_KEY，模型用 DEEPSEEK_MODEL (預設 deepseek-chat)。
"""

import json
import os

import httpx

from examples.llm.providers import OpenAIProvider, ProviderError


class DeepSeekProvider(OpenAIProvider):
    def complete_structured(self, messages, schema_name, schema, *, timeout=None):
        hint = ("\n\nReply with only a JSON object that matches this schema:\n"
                + json.dumps(schema))
        messages = [{**messages[0], "content": messages[0]["content"] + hint}, *messages[1:]]
        body = {"model": self.model, "messages": messages,
                "response_format": {"type": "json_object"}}
        extra = {"timeout": timeout} if timeout is not None else {}
        try:
            r = self._http.post("/chat/completions", json=body, **extra)
        except httpx.HTTPError as e:
            raise ProviderError(f"DeepSeek request failed ({type(e).__name__})") from None
        if r.status_code != 200:
            raise ProviderError(f"DeepSeek request failed (HTTP {r.status_code})")
        try:
            data = json.loads(r.json()["choices"][0]["message"]["content"] or "")
        except (ValueError, KeyError, IndexError, TypeError):
            raise ProviderError("DeepSeek reply was not valid JSON") from None
        if not isinstance(data, dict):
            raise ProviderError("DeepSeek reply was not a JSON object")
        return data


def deepseek_from_env():
    return DeepSeekProvider(
        os.environ["DEEPSEEK_API_KEY"],
        os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat",
        base_url=os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
    )
