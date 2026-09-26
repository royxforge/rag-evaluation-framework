"""Abstract base class for LLM adapters."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class LLMAdapter(ABC):
    """Abstract base class for LLM adapters.

    All adapters support async completion, batch processing, and embeddings.
    """

    def __init__(self, model: str, **kwargs: Any):
        self.model: str = model
        self._embedder: Any | None = None
        self._max_retries: int = kwargs.pop("max_retries", 3)
        self._base_delay: float = kwargs.pop("base_delay", 1.0)

    @abstractmethod
    async def complete(self, prompt: str, temperature: float = 0.0, **kwargs: Any) -> str:
        """Send a completion request and return the response text.

        Args:
            prompt: The prompt string.
            temperature: Sampling temperature (0 = deterministic).
            **kwargs: Additional provider-specific parameters.

        Returns:
            The generated text response.
        """
        ...

    async def complete_batch(
        self, prompts: list[str], temperature: float = 0.0, **kwargs: Any
    ) -> list[str]:
        """Process multiple prompts with bounded concurrency.

        Default implementation uses asyncio.gather with a semaphore.
        Override for provider-specific batching.
        """
        max_concurrency = int(kwargs.pop("max_concurrency", 10))
        semaphore = asyncio.Semaphore(max(1, max_concurrency))

        async def _one(prompt: str) -> str:
            async with semaphore:
                return await self.complete(prompt, temperature=temperature, **kwargs)

        tasks = [_one(p) for p in prompts]
        return await asyncio.gather(*tasks)

    async def embed(self, text: str | list[str]) -> np.ndarray:
        """Generate embeddings using sentence-transformers.

        The ``sentence-transformers`` dependency is imported lazily so that
        LLM-only evaluations (OpenAI/etc. judge calls) do not require the
        heavy torch stack at import time.

        Args:
            text: Single string or list of strings.

        Returns:
            numpy array of embeddings.
        """
        embedder = self.get_embedder()
        inputs = [text] if isinstance(text, str) else text
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: embedder.encode(inputs, normalize_embeddings=True)
        )

    def get_embedder(self) -> Any:
        """Return the sentence-transformer embedder (singleton, lazy import)."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "embed() requires the optional 'sentence-transformers' "
                    "package. Install it with: pip install sentence-transformers"
                ) from exc
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self._embedder

    async def _retry_with_backoff(self, coro_factory, retries: int | None = None) -> Any:
        """Execute a coroutine with exponential backoff retry.

        Retries only transient failures (rate limits / 5xx / timeouts /
        connection errors). Authentication and other client errors fail
        immediately so a bad key does not burn through retries.
        """
        # Import defensively: provider SDKs are optional extras, and a
        # missing package must not break retry policy for the providers that
        # *are* installed.
        try:
            import httpx
        except ImportError:  # pragma: no cover - httpx is a declared dependency
            httpx = None
        try:
            import openai
        except ImportError:
            openai = None

        retries = retries or self._max_retries
        last_exception: Exception | None = None
        for attempt in range(retries + 1):
            try:
                return await coro_factory()
            except Exception as e:
                last_exception = e
                if not _is_transient_error(e, httpx=httpx, openai=openai):
                    raise
                if attempt < retries:
                    delay = self._base_delay * (2**attempt)
                    await asyncio.sleep(delay)
        raise last_exception  # type: ignore[misc]


def _is_transient_error(exc: BaseException, *, httpx: Any, openai: Any) -> bool:
    """Return True only for retryable transient failures."""
    # HTTP 429 / 5xx via httpx
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and (status == 429 or 500 <= status < 600):
        return True
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    if isinstance(response_status, int) and (
        response_status == 429 or 500 <= response_status < 600
    ):
        return True
    # OpenAI SDK transient errors
    transient_types: list[type] = []
    for attr in (
        "RateLimitError",
        "InternalServerError",
        "APIConnectionError",
        "APITimeoutError",
    ):
        cls = getattr(openai, attr, None)
        if isinstance(cls, type):
            transient_types.append(cls)
    if transient_types and isinstance(exc, tuple(transient_types)):
        return True
    # Timeouts / connection resets
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    httpx_timeout = getattr(httpx, "TimeoutException", None)
    if isinstance(httpx_timeout, type) and isinstance(exc, httpx_timeout):
        return True
    return False
