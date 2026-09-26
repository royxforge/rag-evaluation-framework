"""Unit tests for LLM adapaters using MockLLMAdapter."""

from __future__ import annotations

import pytest

from rag_evaluation_framework.adapters import get_adapter
from rag_evaluation_framework.adapters.base import LLMAdapter
from rag_evaluation_framework.adapters.openai_adapter import OpenAIAdapter
from tests.conftest import MockLLMAdapter


class TestGetAdapter:
    """Tests for the get_adapter factory function."""

    def test_get_adapter_openai(self):
        """get_adapter returns an LLMAdapter for openai provider."""
        adapter = get_adapter("openai/gpt-4o", config={"api_key": "test-key"})
        assert isinstance(adapter, LLMAdapter)
        assert adapter.model == "gpt-4o"

    def test_get_adapter_invalid_format(self):
        """get_adapter raises ValueError for invalid model string."""
        with pytest.raises(ValueError, match="Model string must be in format"):
            get_adapter("invalid-format")

    def test_get_adapter_unknown_provider(self):
        """get_adapter raises ValueError for unknown provider."""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_adapter("unknown/model")


class TestMockLLMAdapter:
    """Tests for MockLLMAdapter used in testing."""

    @pytest.mark.asyncio
    async def test_complete_default_response(self):
        """MockLLMAdapter returns default response for unknown prompts."""
        adapter = MockLLMAdapter()
        response = await adapter.complete("some random prompt")
        assert response == "Mock response"

    @pytest.mark.asyncio
    async def test_complete_exact_match(self):
        """MockLLMAdapter returns exact match response."""
        adapter = MockLLMAdapter(responses={"hello": "world"})
        response = await adapter.complete("hello")
        assert response == "world"

    @pytest.mark.asyncio
    async def test_complete_partial_match(self):
        """MockLLMAdapter returns response for partial match."""
        adapter = MockLLMAdapter(responses={"decompose": '["claim1"]'})
        response = await adapter.complete("Please decompose this answer")
        assert response == '["claim1"]'

    @pytest.mark.asyncio
    async def test_complete_claim_keyword(self):
        """MockLLMAdapter handles claim-related prompts."""
        adapter = MockLLMAdapter()
        response = await adapter.complete("extract claims from this text")
        assert isinstance(response, str) and len(response) > 0

    @pytest.mark.asyncio
    async def test_complete_tracks_call_history(self):
        """MockLLMAdapter tracks call history."""
        adapter = MockLLMAdapter()
        await adapter.complete("first call")
        await adapter.complete("second call")
        assert len(adapter.call_history) == 2
        assert "first" in adapter.call_history[0]["prompt"]

    @pytest.mark.asyncio
    async def test_embed_single_string(self):
        """MockLLMAdapter.embed returns an array for single string."""
        import numpy as np

        adapter = MockLLMAdapter()
        result = await adapter.embed("test text")
        assert isinstance(result, np.ndarray)
        assert result.shape == (384,)

    @pytest.mark.asyncio
    async def test_embed_multiple_strings(self):
        """MockLLMAdapter.embed returns matrix for multiple strings."""
        import numpy as np

        adapter = MockLLMAdapter()
        result = await adapter.embed(["text1", "text2"])
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 384)

    def test_get_embedder(self):
        """MockLLMAdapter.get_embedder raises a clear error when optional deps are missing."""
        adapter = MockLLMAdapter()
        with pytest.raises(ImportError, match="sentence-transformers"):
            adapter.get_embedder()

    def test_get_embedder_singleton(self):
        """Repeated get_embedder calls raise the same clear ImportError without the optional dep."""
        adapter = MockLLMAdapter()
        with pytest.raises(ImportError, match="sentence-transformers"):
            adapter.get_embedder()
        with pytest.raises(ImportError, match="sentence-transformers"):
            adapter.get_embedder()


class TestLLMAdapterBase:
    """Tests for the LLMAdapter base class."""

    @pytest.mark.asyncio
    async def test_complete_batch(self):
        """complete_batch processes multiple prompts."""
        adapter = MockLLMAdapter(
            responses={
                "hello": "world",
                "goodbye": "farewell",
            }
        )
        results = await adapter.complete_batch(
            ["hello", "goodbye", "unknown"],
            temperature=0.0,
        )
        assert len(results) == 3
        assert results[0] == "world"
        assert results[1] == "farewell"
        assert results[2] == "Mock response"

    @pytest.mark.asyncio
    async def test_embed_via_base(self):
        """embed method works through base class."""
        import numpy as np

        adapter = MockLLMAdapter()
        result = await adapter.embed("test")
        assert isinstance(result, np.ndarray)


class TestRetryPolicy:
    """_retry_with_backoff must retry transient failures only."""

    class _TransientError(Exception):
        status_code = 429

    class _AuthError(Exception):
        status_code = 401

    @pytest.mark.asyncio
    async def test_transient_error_is_retried(self):
        adapter = MockLLMAdapter(base_delay=0.0)
        calls = {"n": 0}

        async def flaky():
            calls["n"] += 1
            if calls["n"] < 2:
                raise self._TransientError("rate limited")
            return "ok"

        assert await adapter._retry_with_backoff(flaky) == "ok"
        assert calls["n"] == 2

    @pytest.mark.asyncio
    async def test_auth_error_fails_fast(self):
        adapter = MockLLMAdapter(base_delay=0.0)
        calls = {"n": 0}

        async def unauthorized():
            calls["n"] += 1
            raise self._AuthError("bad key")

        with pytest.raises(self._AuthError):
            await adapter._retry_with_backoff(unauthorized)
        assert calls["n"] == 1  # no retry burn on a 401


class TestOpenAIAdapter:
    """Tests for OpenAIAdapter initialization."""

    def test_openai_init(self):
        """OpenAIAdapter initializes with model and api_key."""
        adapter = OpenAIAdapter(model="gpt-4o", api_key="test-key")
        assert adapter.model == "gpt-4o"
        assert adapter._max_retries == 3

    def test_openai_init_custom_retries(self):
        """OpenAIAdapter accepts max_retries config."""
        adapter = OpenAIAdapter(
            model="gpt-4o",
            api_key="test-key",
            max_retries=5,
            base_delay=2.0,
        )
        assert adapter._max_retries == 5
        assert adapter._base_delay == 2.0
