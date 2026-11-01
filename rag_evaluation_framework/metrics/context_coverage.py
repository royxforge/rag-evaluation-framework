"""Context coverage metric measures how well the answer covers the information in the context."""

from __future__ import annotations

from rag_evaluation_framework.adapters.base import LLMAdapter
from rag_evaluation_framework.models import MetricScore
from rag_evaluation_framework.prompts import CONTEXT_COVERAGE_PROMPT
from rag_evaluation_framework.utils import clamp_score


async def score_context_coverage(
    question: str,
    context: list[str],
    answer: str,
    adapter: LLMAdapter,
) -> MetricScore:
    """Score how well the answer covers the information present in the context.

    Uses an LLM to evaluate what fraction of relevant context information
    is reflected in the answer.

    Returns:
        MetricScore with score = coverage rating (0.0 - 1.0).
    """
    if not answer.strip():
        return MetricScore(
            score=0.0,
            explanation="Empty answer no context coverage.",
            confidence=1.0,
            details={"reason": "Answer is empty."},
        )

    if not context:
        return MetricScore(
            score=1.0,
            explanation="No context provided vacuously perfect coverage.",
            confidence=0.5,
            details={"reason": "Empty context."},
        )

    context_str = "\n".join(context)
    prompt = CONTEXT_COVERAGE_PROMPT.format(context=context_str, answer=answer)

    try:
        response = await adapter.complete(prompt, temperature=0.0)
        score = _parse_score(response)
    except Exception as exc:
        # Never substitute a plausible number for a failed measurement.
        return MetricScore(
            score=None,
            explanation=f"Context coverage could not be computed: {exc}",
            confidence=0.0,
            details={"error": str(exc), "failed": True},
        )

    if score is None:
        return MetricScore(
            score=None,
            explanation="Context coverage could not be computed: unparseable model response.",
            confidence=0.0,
            details={"error": "unparseable_score", "failed": True},
        )

    score = clamp_score(score)

    if score >= 0.8:
        explanation = (
            f"Answer covers the majority of relevant context information (score={score:.3f})."
        )
    elif score >= 0.5:
        explanation = f"Answer partially covers the context information (score={score:.3f})."
    else:
        explanation = f"Answer misses most of the relevant context information (score={score:.3f})."

    details = {
        "num_context_chunks": len(context),
        "coverage_score": score,
    }

    return MetricScore(score=score, explanation=explanation, confidence=0.8, details=details)


def _parse_score(text: str) -> float | None:
    """Parse a float score from LLM response, or None if it cannot be parsed."""
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        pass

    import re

    match = re.search(r"(\d+\.?\d*)", text)
    if match:
        try:
            val = float(match.group(1))
            return clamp_score(val / 100.0 if val > 1.0 else val)
        except ValueError:
            pass

    return None
