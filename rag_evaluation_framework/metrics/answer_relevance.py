"""Answer relevance metric measures how well the answer addresses the question."""

from __future__ import annotations

from rag_evaluation_framework.adapters.base import LLMAdapter
from rag_evaluation_framework.models import MetricScore
from rag_evaluation_framework.prompts import ANSWER_RELEVANCE_PROMPT
from rag_evaluation_framework.utils import clamp_score


async def score_answer_relevance(
    question: str,
    context: list[str],
    answer: str,
    adapter: LLMAdapter,
) -> MetricScore:
    """Score the relevance of an answer to the question.

    Uses an LLM to evaluate how well the answer addresses the question.

    Returns:
        MetricScore with score = relevance rating (0.0 - 1.0).
    """
    if not answer.strip():
        return MetricScore(
            score=0.0,
            explanation="Empty answer no relevance.",
            confidence=1.0,
            details={"reason": "Answer is empty."},
        )

    prompt = ANSWER_RELEVANCE_PROMPT.format(question=question, answer=answer)

    try:
        response = await adapter.complete(prompt, temperature=0.0)
        score = _parse_score(response)
    except Exception as exc:
        # Never substitute a plausible number for a failed measurement.
        return MetricScore(
            score=None,
            explanation=f"Answer relevance could not be computed: {exc}",
            confidence=0.0,
            details={"error": str(exc), "failed": True},
        )

    if score is None:
        return MetricScore(
            score=None,
            explanation="Answer relevance could not be computed: unparseable model response.",
            confidence=0.0,
            details={"error": "unparseable_score", "failed": True},
        )

    score = clamp_score(score)

    if score >= 0.8:
        explanation = f"Answer is highly relevant to the question (score={score:.3f})."
    elif score >= 0.5:
        explanation = f"Answer is partially relevant to the question (score={score:.3f})."
    else:
        explanation = f"Answer has low relevance to the question (score={score:.3f})."

    details = {
        "question": question,
        "answer_preview": answer[:200],
        "relevance_score": score,
    }

    return MetricScore(score=score, explanation=explanation, confidence=0.8, details=details)


def _parse_score(text: str) -> float | None:
    """Parse a float score from LLM response, or None if it cannot be parsed."""
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        pass

    # Try to find a number in the text
    import re

    match = re.search(r"(\d+\.?\d*)", text)
    if match:
        try:
            val = float(match.group(1))
            return clamp_score(val / 100.0 if val > 1.0 else val)
        except ValueError:
            pass

    return None
