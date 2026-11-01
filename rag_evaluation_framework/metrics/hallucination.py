"""Hallucination rate metric measures unsupported or fabricated claims in the answer."""

from __future__ import annotations

import json
import re
from typing import Any

from rag_evaluation_framework.adapters.base import LLMAdapter
from rag_evaluation_framework.models import MetricScore
from rag_evaluation_framework.prompts import HALLUCINATION_VERIFY_PROMPT
from rag_evaluation_framework.utils import clamp_score, extract_claims


async def score_hallucination(
    question: str,
    context: list[str],
    answer: str,
    adapter: LLMAdapter,
) -> MetricScore:
    """Score the hallucination rate of an answer.

    Distinguishes between:
        - Context hallucination: contradicts the provided context
        - Factual hallucination: makes claims not verifiable from context

    Returns:
        MetricScore where score = hallucination_rate
        (0.0 = no hallucination, 1.0 = full hallucination).
    """
    context_str = "\n".join(context)
    claims = extract_claims(answer)

    if not claims:
        return MetricScore(
            score=0.0,
            explanation="No claims found in answer no hallucination detected.",
            confidence=0.5,
            details={
                "claims": [],
                "total_claims": 0,
                "context_hallucinations": 0,
                "factual_hallucinations": 0,
            },
        )

    supported_claims = []
    context_hallucinations = []
    factual_hallucinations = []
    claim_details = []

    unverified_claims: list[str] = []

    for claim in claims:
        verify_prompt = HALLUCINATION_VERIFY_PROMPT.format(context=context_str, claim=claim)
        try:
            verify_response = await adapter.complete(verify_prompt, temperature=0.0)
            result: dict[str, Any] | None = _parse_verification(verify_response)
        except Exception:
            # An unverifiable claim is unknown, not hallucinated. Treating it as
            # a factual hallucination inflates the rate on provider errors.
            result = None

        if result is None:
            unverified_claims.append(claim)
            claim_details.append(
                {
                    "claim": claim,
                    "grounded": None,
                    "factual": None,
                    "hallucination_type": "unknown",
                    "reason": "Verification failed",
                }
            )
            continue

        is_grounded = result.get("grounded", False)
        is_factual = result.get("factual", False)
        reason = result.get("reason", "")

        if is_grounded:
            supported_claims.append(claim)
        elif is_factual:
            # Factual claim not grounded = factual hallucination
            factual_hallucinations.append(claim)
        else:
            # Claim contradicts context = context hallucination
            context_hallucinations.append(claim)

        claim_details.append(
            {
                "claim": claim,
                "grounded": is_grounded,
                "factual": is_factual,
                "hallucination_type": "none"
                if is_grounded
                else ("factual" if is_factual else "context"),
                "reason": reason,
            }
        )

    total = len(supported_claims) + len(context_hallucinations) + len(factual_hallucinations)
    if total == 0:
        return MetricScore(
            score=None,
            explanation="Hallucination rate could not be computed: no claim could be verified.",
            confidence=0.0,
            details={
                "failed": True,
                "claims": claim_details,
                "unverified_claims": len(unverified_claims),
            },
        )

    total_hallucinations = len(context_hallucinations) + len(factual_hallucinations)
    rate = clamp_score(total_hallucinations / total)

    explanation_parts = []
    if total_hallucinations == 0:
        explanation_parts.append(f"No hallucination detected all {total} claims are grounded.")
    else:
        explanation_parts.append(
            f"Hallucination rate {rate:.3f}: {total_hallucinations}/{total} claims unsupported."
        )
        if context_hallucinations:
            explanation_parts.append(
                f"{len(context_hallucinations)} context hallucination(s) contradicts context."
            )
        if factual_hallucinations:
            explanation_parts.append(
                f"{len(factual_hallucinations)} factual hallucination(s) not in context."
            )

    explanation = " ".join(explanation_parts)

    details = {
        "claims": claim_details,
        "total_claims": total,
        "unverified_claims": len(unverified_claims),
        "unverified_list": unverified_claims[:5],
        "supported_claims": len(supported_claims),
        "context_hallucinations": len(context_hallucinations),
        "context_hallucination_list": context_hallucinations[:5],
        "factual_hallucinations": len(factual_hallucinations),
        "factual_hallucination_list": factual_hallucinations[:5],
    }

    confidence = clamp_score(min(1.0, total / 5.0))

    return MetricScore(
        score=rate,  # Note: higher = worse hallucination
        explanation=explanation,
        confidence=confidence,
        details=details,
    )


def _parse_verification(text: str) -> dict[str, Any]:
    """Parse the LLM verification response."""
    text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Try to extract JSON block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, dict):
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback parsing
    grounded = "true" in text.lower() and "grounded" in text.lower()
    factual = "factual" in text.lower() and "true" in text.lower()
    return {"grounded": grounded, "factual": factual, "reason": text[:100]}
