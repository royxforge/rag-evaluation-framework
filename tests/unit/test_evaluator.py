"""Unit tests for the Evaluator class."""

from __future__ import annotations

import pytest

from rag_evaluation_framework import Evaluator
from rag_evaluation_framework.metrics import METRIC_REGISTRY
from rag_evaluation_framework.models import ComparisonReport, EvalResult


def test_evaluator_init():
    """Evaluator initializes with default params."""
    evaluator = Evaluator(llm="ollama/llama3", cache=True)
    assert evaluator is not None


def test_evaluator_init_local():
    """Evaluator initializes with Ollama."""
    evaluator = Evaluator(llm="ollama/llama3")
    assert evaluator is not None


def test_evaluator_score_sync():
    """Evaluator.score() returns EvalResult (synchronous path)."""
    evaluator = Evaluator(llm="openai/gpt-4o", cache=True, model_config={"api_key": "test"})
    # Override adapter with mock
    from tests.conftest import MockLLMAdapter

    evaluator._adapter = MockLLMAdapter()

    result = evaluator.score(
        question="What is the capital of France?",
        context=["France is in Europe. Its capital is Paris."],
        answer="The capital of France is Paris.",
    )

    assert isinstance(result, EvalResult)
    assert result.question == "What is the capital of France?"
    assert 0.0 <= result.overall_score <= 1.0
    assert result.faithfulness is not None
    assert result.hallucination_rate is not None
    assert result.retrieval_precision is not None
    assert result.answer_relevance is not None
    assert result.context_coverage is not None
    assert result.ucm_confidence is not None


@pytest.mark.asyncio
async def test_evaluator_async_score():
    """Evaluator.async_score() returns EvalResult."""
    evaluator = Evaluator(llm="openai/gpt-4o", model_config={"api_key": "test"})
    from tests.conftest import MockLLMAdapter

    evaluator._adapter = MockLLMAdapter()

    result = await evaluator.async_score(
        question="What is the capital of France?",
        context=["France is in Europe. Its capital is Paris."],
        answer="The capital of France is Paris.",
    )

    assert isinstance(result, EvalResult)
    assert 0.0 <= result.overall_score <= 1.0


def test_evaluator_batch_score():
    """Evaluator.batch_score() returns list of EvalResult."""
    evaluator = Evaluator(llm="openai/gpt-4o", model_config={"api_key": "test"})
    from tests.conftest import MockLLMAdapter

    evaluator._adapter = MockLLMAdapter()

    items = [
        {"question": "Q1?", "context": ["C1"], "answer": "A1"},
        {"question": "Q2?", "context": ["C2"], "answer": "A2"},
    ]

    results = evaluator.batch_score(items)
    assert len(results) == 2
    assert all(isinstance(r, EvalResult) for r in results)


def test_evaluator_compare():
    """Evaluator.compare() returns ComparisonReport."""
    evaluator = Evaluator(llm="ollama/llama3")
    from tests.conftest import MockLLMAdapter

    evaluator._adapter = MockLLMAdapter()

    r1 = evaluator.score(question="Q1?", context=["C1"], answer="A1")
    r2 = evaluator.score(question="Q2?", context=["C2"], answer="A2")

    comparison = evaluator.compare(r1, r2)
    assert isinstance(comparison, ComparisonReport)
    assert comparison.verdict != ""


def test_evaluator_cache_is_metric_aware():
    """Cache keys include the metric set: subset results never satisfy full lookups."""
    evaluator = Evaluator(llm="openai/gpt-4o", cache=True, model_config={"api_key": "test"})
    from tests.conftest import MockLLMAdapter

    evaluator._adapter = MockLLMAdapter()
    kwargs = dict(
        question="Cache test?",
        context=["Cache context."],
        answer="Cache answer.",
    )

    full = evaluator.score(**kwargs)
    assert set(full.metadata["metrics_computed"]) == set(METRIC_REGISTRY.keys())

    subset = evaluator.score(metrics=["faithfulness"], **kwargs)
    assert subset.metadata["metrics_computed"] == ["faithfulness"]
    # A perfect single-metric run averages over 1 metric, not 6.
    assert subset.overall_score == pytest.approx(subset.faithfulness.score)

    again = evaluator.score(**kwargs)
    assert again.id == full.id  # full result still cached under its own key


def test_evaluator_cache_identical_requests_hit():
    """Identical requests with the same metric set hit the cache."""
    evaluator = Evaluator(llm="openai/gpt-4o", cache=True, model_config={"api_key": "test"})
    from tests.conftest import MockLLMAdapter

    evaluator._adapter = MockLLMAdapter()

    result1 = evaluator.score(
        question="Cache test?",
        context=["Cache context."],
        answer="Cache answer.",
    )
    result2 = evaluator.score(
        question="Cache test?",
        context=["Cache context."],
        answer="Cache answer.",
    )

    # Same question/context/answer with same LLM should hit cache
    assert result1.id == result2.id


def test_cache_clear():
    """Clearing cache should work."""
    evaluator = Evaluator(llm="ollama/llama3", cache=True)
    evaluator.clear_cache()
    assert len(evaluator._cache) == 0


@pytest.mark.asyncio
async def test_report_formats():
    """EvalResult.report() should work in all formats."""
    evaluator = Evaluator(llm="openai/gpt-4o", model_config={"api_key": "test"})
    from tests.conftest import MockLLMAdapter

    evaluator._adapter = MockLLMAdapter()

    result = await evaluator.async_score(
        question="Report test?",
        context=["Report context."],
        answer="Report answer.",
    )

    # Dict format
    report_dict = result.report(format="dict")
    assert isinstance(report_dict, dict)
    assert "overall_score" in report_dict

    # JSON format
    report_json = result.report(format="json")
    assert isinstance(report_json, str)
    assert '"overall_score"' in report_json

    # Markdown format
    report_md = result.report(format="markdown")
    assert isinstance(report_md, str)
    assert "RAG Evaluation Framework Report" in report_md

    # HTML format
    report_html = result.report(format="html")
    assert isinstance(report_html, str)
    assert "RAG Evaluation Framework Report" in report_html
