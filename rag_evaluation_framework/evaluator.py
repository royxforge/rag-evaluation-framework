"""Main Evaluator class the public API for RAG Evaluation Framework."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from rag_evaluation_framework.adapters import get_adapter
from rag_evaluation_framework.adapters.base import LLMAdapter
from rag_evaluation_framework.metrics import METRIC_REGISTRY
from rag_evaluation_framework.models import ComparisonReport, EvalResult, MetricScore
from rag_evaluation_framework.utils import clamp_score, generate_cache_key


class Evaluator:
    """RAG Evaluation Framework main evaluation interface.

    Usage:
        evaluator = Evaluator(llm="openai/gpt-4o")
        result = evaluator.score(
            question="What is the capital of France?",
            context=["France is in Europe. Its capital is Paris."],
            answer="Paris is the capital of France."
        )
        print(result.overall_score)
    """

    def __init__(
        self,
        llm: str = "openai/gpt-4o",
        model_config: dict | None = None,
        cache: bool = True,
        cache_backend: str = "memory",
    ):
        """Initialize the evaluator.

        Args:
            llm: Model string in 'provider/model' format.
            model_config: Optional config dict passed to the LLM adapter.
            cache: Whether to cache results.
            cache_backend: Cache backend: 'memory' or 'redis'.
            Redis requires optional 'redis' package.
        """
        self._llm_string: str = llm
        self._model_config: dict = model_config or {}
        self._adapter: LLMAdapter = get_adapter(llm, config=model_config)
        self._cache_enabled: bool = cache
        self._cache: dict[str, EvalResult] = {}
        self._cache_backend: str = cache_backend

    @property
    def adapter(self) -> LLMAdapter:
        """Return the underlying LLM adapter."""
        return self._adapter

    def score(
        self,
        question: str,
        context: list[str],
        answer: str,
        metrics: list[str] | None = None,
        **kwargs: Any,
    ) -> EvalResult:
        """Run a synchronous evaluation.

        Args:
            question: The user's question.
            context: Retrieved context chunks.
            answer: The generated answer.
            metrics: List of metrics to compute (default: all).
            **kwargs: Additional options.

        Returns:
            EvalResult with all metric scores.
        """
        return asyncio.run(self.async_score(question, context, answer, metrics=metrics, **kwargs))

    async def async_score(
        self,
        question: str,
        context: list[str],
        answer: str,
        metrics: list[str] | None = None,
        **kwargs: Any,
    ) -> EvalResult:
        """Run an async evaluation.

        Args:
            question: The user's question.
            context: Retrieved context chunks.
            answer: The generated answer.
            metrics: List of metrics to compute (default: all).
            **kwargs: Additional options.

        Returns:
            EvalResult with all metric scores.
        """
        metrics = metrics or ["all"]

        # Determine which metrics to run first so the cache key can include
        # the resolved set: a single-metric result must never satisfy an
        # "all metrics" lookup (or vice versa).
        if "all" in metrics:
            metric_names = list(METRIC_REGISTRY.keys())
        else:
            metric_names = [m for m in metrics if m in METRIC_REGISTRY]

        if not metric_names:
            raise ValueError(
                f"No valid metrics specified. Available: {list(METRIC_REGISTRY.keys())}"
            )

        # Check cache (key includes metric set + scoring options)
        extra_opts = {k: v for k, v in kwargs.items() if k != "cache_backend"}
        cache_key = generate_cache_key(
            question, context, answer, self._llm_string, metrics=metric_names, extra=extra_opts or None
        )
        if self._cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        start_time = time.time()

        # Run all selected metrics in parallel
        tasks = {}
        for name in metric_names:
            scorer = METRIC_REGISTRY[name]
            tasks[name] = scorer(question, context, answer, self._adapter)

        results = await asyncio.gather(*tasks.values(), return_exceptions=True)

        # Collect results
        metric_scores: dict[str, MetricScore] = {}
        for name, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                metric_scores[name] = MetricScore(
                    score=0.0,
                    explanation=f"Metric {name} failed: {result}",
                    confidence=0.0,
                    details={"error": str(result)},
                )
            else:
                metric_scores[name] = result

        # Build EvalResult
        faithfulness = metric_scores.get("faithfulness", MetricScore(score=0.0))
        hallucination = metric_scores.get("hallucination", MetricScore(score=0.0))
        retrieval_precision = metric_scores.get("retrieval_precision", MetricScore(score=0.0))
        answer_relevance = metric_scores.get("answer_relevance", MetricScore(score=0.0))
        context_coverage = metric_scores.get("context_coverage", MetricScore(score=0.0))
        ucm_confidence = metric_scores.get("ucm_confidence", MetricScore(score=0.0))

        # Overall score: average over the metrics actually computed, so a
        # single-metric evaluation does not divide by 6 (yielding ~0.16 for a
        # perfect score). Hallucination is inverted: lower rate = better.
        computed: list[float] = []
        if "faithfulness" in metric_scores:
            computed.append(faithfulness.score)
        if "hallucination" in metric_scores:
            computed.append(1.0 - hallucination.score)
        if "retrieval_precision" in metric_scores:
            computed.append(retrieval_precision.score)
        if "answer_relevance" in metric_scores:
            computed.append(answer_relevance.score)
        if "context_coverage" in metric_scores:
            computed.append(context_coverage.score)
        if "ucm_confidence" in metric_scores:
            computed.append(ucm_confidence.score)
        overall = clamp_score(sum(computed) / len(computed)) if computed else 0.0

        latency_ms = int((time.time() - start_time) * 1000)

        result = EvalResult(
            question=question,
            context=context,
            answer=answer,
            overall_score=overall,
            faithfulness=faithfulness,
            hallucination_rate=hallucination,
            retrieval_precision=retrieval_precision,
            answer_relevance=answer_relevance,
            context_coverage=context_coverage,
            ucm_confidence=ucm_confidence,
            metadata={"llm": self._llm_string, "metrics_computed": metric_names},
            latency_ms=latency_ms,
        )

        # Cache result
        if self._cache_enabled:
            self._cache[cache_key] = result

        return result

    def batch_score(
        self,
        items: list[dict],
        metrics: list[str] | None = None,
        **kwargs: Any,
    ) -> list[EvalResult]:
        """Run evaluations for multiple items.

        Args:
            items: List of dicts with 'question', 'context', 'answer' keys.
            metrics: List of metrics to compute.
            **kwargs: Additional options.

        Returns:
            List of EvalResult objects.
        """
        return asyncio.run(self._batch_score_async(items, metrics=metrics, **kwargs))

    async def _batch_score_async(
        self,
        items: list[dict],
        metrics: list[str] | None = None,
        **kwargs: Any,
    ) -> list[EvalResult]:
        """Async batch evaluation."""
        tasks = [
            self.async_score(
                item["question"],
                item["context"],
                item["answer"],
                metrics=metrics,
                **kwargs,
            )
            for item in items
        ]
        return await asyncio.gather(*tasks)

    def compare(self, result_a: EvalResult, result_b: EvalResult) -> ComparisonReport:
        """Compare two evaluation results.

        Args:
            result_a: First evaluation result.
            result_b: Second evaluation result.

        Returns:
            ComparisonReport with score deltas and verdict.
        """
        score_deltas: dict[str, float] = {}
        metrics_to_compare = [
            ("overall_score", "overall"),
            ("faithfulness", "faithfulness"),
            ("hallucination_rate", "hallucination rate"),
            ("retrieval_precision", "retrieval precision"),
            ("answer_relevance", "answer relevance"),
            ("context_coverage", "context coverage"),
            ("ucm_confidence", "UCM confidence"),
        ]

        for attr, label in metrics_to_compare:
            val_a = getattr(result_a, attr)
            val_b = getattr(result_b, attr)
            if isinstance(val_a, MetricScore):
                delta = val_b.score - val_a.score
                # Hallucination rate is lower-is-better: invert so that a
                # positive delta always means "B improved".
                if attr == "hallucination_rate":
                    delta = -delta
            else:
                delta = float(val_b) - float(val_a)
            score_deltas[label] = round(delta, 4)

        improved = sum(1 for d in score_deltas.values() if d > 0)
        declined = sum(1 for d in score_deltas.values() if d < 0)
        total = len(score_deltas)

        if improved > declined:
            verdict = f"Result B is better: improved on {improved}/{total} metrics."
        elif declined > improved:
            verdict = f"Result A is better: declined on {declined}/{total} metrics."
        else:
            verdict = f"Results are comparable: {improved}/{total} metrics improved."

        return ComparisonReport(
            result_a=result_a,
            result_b=result_b,
            score_deltas=score_deltas,
            verdict=verdict,
        )

    def clear_cache(self) -> None:
        """Clear the evaluation cache."""
        self._cache.clear()
