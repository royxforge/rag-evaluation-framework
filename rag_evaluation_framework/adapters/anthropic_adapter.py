"""Anthropic LLM adapter."""

from __future__ import annotations

import os
from typing import Any

from anthropic import AsyncAnthropic

from rag_evaluation_framework.adapters.base import LLMAdapter


class AnthropicAdapter(LLMAdapter):
    """Adapter for Anthropic models (Claude 3, Claude 3.5, etc.)."""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022", **kwargs: Any):
        super().__init__(model, **kwargs)
        api_key = kwargs.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Pass api_key "
                "or set ANTHROPIC_API_KEY environment variable."
            )
        self._client = AsyncAnthropic(api_key=api_key)

    async def complete(self, prompt: str, temperature: float = 0.0, **kwargs: Any) -> str:
        async def _call() -> str:
            # Passed as a mapping so SDK overload resolution is not pinned to a
            # single signature, and extra provider kwargs stay supported.
            params: dict[str, Any] = {
                "model": self.model,
                "max_tokens": kwargs.pop("max_tokens", 4096),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                **kwargs,
            }
            response = await self._client.messages.create(**params)
            return response.content[0].text if response.content else ""

        return await self._retry_with_backoff(_call)
