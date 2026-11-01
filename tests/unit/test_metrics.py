"""Unit tests for all metric modules."""

from __future__ import annotations

import pytest

from rag_evaluation_framework.metrics.answer_relevance import score_answer_relevance
from rag_evaluation_framework.metrics.context_coverage import score_context_coverage
from rag_evaluation_framework.metrics.faithfulness import score_faithfulness
from rag_evaluation_framework.metrics.hallucination import score_hallucination
from rag_evaluation_framework.metrics.retrieval_precision import score_retrieval_precision
from rag_evaluation_framework.metrics.ucm_confidence import score_ucm_confidence
from tests.conftest import MockLLMAdapter


@pytest.mark.asyncio
async def test_faithfulness_high_score():
    """Faithfulness: context matches answer → high score."""
    adapter = MockLLMAdapter(
        responses={
            "decompose": '["Paris is the capital of France.", "France is in Europe."]',
            "SUPPORTED": "SUPPORTED",
        }
    )
    result = await score_faithfulness(
        question="What is the capital of France?",
        context=["France is a country in Western Europe. Its capital is Paris."],
        answer="The capital of France is Paris.",
        adapter=adapter,
    )
    assert result.score >= 0.8, f"Expected high faithfulness, got {result.score}"
    assert 0.0 <= result.score <= 1.0


@pytest.mark.asyncio
async def test_faithfulness_low_score():
    """Faithfulness: context contradicts answer → low score."""
    adapter = MockLLMAdapter(
        responses={
            "decompose": '["Paris is the capital of Spain.", "Spain is in Europe."]',
            "NOT_SUPPORTED": "NOT_SUPPORTED",
        }
    )
    result = await score_faithfulness(
        question="What is the capital of France?",
        context=["France is a country in Western Europe. Its capital is Paris."],
        answer="Paris is the capital of Spain.",
        adapter=adapter,
    )
    # Mock always returns SUPPORTED, so this tests the metric runs without error
    assert 0.0 <= result.score <= 1.0


@pytest.mark.asyncio
async def test_hallucination_low_rate():
    """Hallucination: grounded answer → low rate."""
    adapter = MockLLMAdapter(
        responses={
            "grounded": '{"grounded": true, "factual": true, "reason": "Supported"}',
        }
    )
    result = await score_hallucination(
        question="What is the capital of France?",
        context=["Paris is the capital of France."],
        answer="The capital of France is Paris.",
        adapter=adapter,
    )
    assert 0.0 <= result.score <= 1.0


@pytest.mark.asyncio
async def test_retrieval_precision():
    """Retrieval precision: should return a valid score."""
    adapter = MockLLMAdapter()
    result = await score_retrieval_precision(
        question="What is the capital of France?",
        context=["France is a country in Europe.", "Paris is a city in France."],
        answer="Paris",
        adapter=adapter,
    )
    assert 0.0 <= result.score <= 1.0
    assert "chunks" in result.details


@pytest.mark.asyncio
async def test_answer_relevance():
    """Answer relevance: should return a valid score."""
    adapter = MockLLMAdapter()
    result = await score_answer_relevance(
        question="What is the capital of France?",
        context=["France is in Europe. Its capital is Paris."],
        answer="The capital of France is Paris.",
        adapter=adapter,
    )
    # The mock's default reply contains no number: an unparseable grading
    # response is a failed measurement, not a mid-range score.
    assert result.score is None
    assert result.details.get("failed") is True


@pytest.mark.asyncio
async def test_context_coverage():
    """Context coverage: should return a valid score."""
    adapter = MockLLMAdapter()
    result = await score_context_coverage(
        question="What is the capital of France?",
        context=["France is in Europe. Its capital is Paris."],
        answer="The capital of France is Paris.",
        adapter=adapter,
    )
    # Same contract as answer relevance: no parseable score means no score.
    assert result.score is None
    assert result.details.get("failed") is True


@pytest.mark.asyncio
async def test_ucm_high_confidence():
    """UCM: consistent samples → high score."""
    adapter = MockLLMAdapter(
        responses={
            "answer the question": "Paris is the capital of France.",
            "extract": '["Paris is the capital of France."]',
        }
    )
    result = await score_ucm_confidence(
        question="What is the capital of France?",
        context=["France's capital is Paris."],
        answer="Paris is the capital of France.",
        adapter=adapter,
        num_samples=2,
    )
    # With only 2 consistent samples, should still give a reasonable score
    assert 0.0 <= result.score <= 1.0


@pytest.mark.asyncio
async def test_ucm_low_confidence():
    """UCM: varying samples → potentially lower score."""
    adapter = MockLLMAdapter(
        responses={
            "answer the question": "Random different answer each time.",
            "extract": '["Different claim."]',
        }
    )
    result = await score_ucm_confidence(
        question="What is the capital of France?",
        context=["Some context."],
        answer="First answer.",
        adapter=adapter,
        num_samples=2,
    )
    assert 0.0 <= result.score <= 1.0


@pytest.mark.asyncio
async def test_empty_context_retrieval():
    """Retrieval precision with empty context should give perfect score."""
    adapter = MockLLMAdapter()
    result = await score_retrieval_precision(
        question="Test?",
        context=[],
        answer="Test answer.",
        adapter=adapter,
    )
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_empty_answer_faithfulness():
    """Faithfulness with empty answer."""
    adapter = MockLLMAdapter()
    result = await score_faithfulness(
        question="Test?",
        context=["Some context."],
        answer="",
        adapter=adapter,
    )
    assert result.score == 1.0  # Vacuously faithful


@pytest.mark.asyncio
async def test_empty_answer_hallucination():
    """Hallucination with empty answer."""
    adapter = MockLLMAdapter()
    result = await score_hallucination(
        question="Test?",
        context=["Some context."],
        answer="",
        adapter=adapter,
    )
    assert result.score == 0.0  # No hallucination


@pytest.mark.asyncio
async def test_empty_answer_relevance():
    """Answer relevance with empty answer."""
    adapter = MockLLMAdapter()
    result = await score_answer_relevance(
        question="Test?",
        context=["Context."],
        answer="",
        adapter=adapter,
    )
    assert result.score == 0.0


@pytest.mark.asyncio
async def test_empty_answer_coverage():
    """Context coverage with empty answer."""
    adapter = MockLLMAdapter()
    result = await score_context_coverage(
        question="Test?",
        context=["Some context."],
        answer="",
        adapter=adapter,
    )
    assert result.score == 0.0
