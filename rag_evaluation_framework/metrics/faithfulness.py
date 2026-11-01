"""Faithfulness metric measures how well the answer is grounded in the provided context."""

from __future__ import annotations

import json
import re
from typing import Any

from rag_evaluation_framework.adapters.base import LLMAdapter
from rag_evaluation_framework.models import MetricScore
from rag_evaluation_framework.prompts import (
    FAITHFULNESS_DECOMPOSE_PROMPT,
    FAITHFULNESS_VERIFY_PROMPT,
)
from rag_evaluation_framework.utils import clamp_score, extract_claims


async def score_faithfulness(
    question: str,
    context: list[str],
    answer: str,
    adapter: LLMAdapter,
) -> MetricScore:
    """Score the faithfulness of an answer given the context.

    Decomposes answer into claims and checks each for support in context.

    Returns:
        MetricScore with score = supported_claims / total_claims.
    """
    context_str = "\n".join(context)

    # Step 1: Decompose answer into atomic claims using LLM
    decompose_prompt = FAITHFULNESS_DECOMPOSE_PROMPT.format(answer=answer)
    try:
        decompose_response = await adapter.complete(decompose_prompt, temperature=0.0)
        claims = _parse_json_list(decompose_response)
    except Exception:
        # Fallback: use simple sentence splitting
        claims = extract_claims(answer)

    if not claims:
        return MetricScore(
            score=1.0,
            explanation="No claims found in answer vacuously faithful.",
            confidence=0.5,
            details={"claims": [], "supported_claims": 0, "total_claims": 0},
        )

    # Step 2: Verify each claim against context
    supported_claims: list[str] = []
    unsupported_claims: list[str] = []
    unverified_claims: list[str] = []
    claim_details: list[dict[str, Any]] = []

    for claim in claims:
        verify_prompt = FAITHFULNESS_VERIFY_PROMPT.format(context=context_str, claim=claim)
        try:
            verify_response = await adapter.complete(verify_prompt, temperature=0.0)
            is_supported: bool | None = _is_supported_verdict(verify_response)
        except Exception as exc:
            # A failed verification is not evidence against the claim. Counting
            # it as "unsupported" deflates the score whenever the provider
            # errors, which is the opposite of what the metric measures.
            is_supported = None
            claim_details.append({"claim": claim, "supported": None, "error": str(exc)})
            unverified_claims.append(claim)
            continue

        if is_supported:
            supported_claims.append(claim)
        else:
            unsupported_claims.append(claim)

        claim_details.append({"claim": claim, "supported": is_supported})

    total = len(supported_claims) + len(unsupported_claims)
    if total == 0:
        return MetricScore(
            score=None,
            explanation="Faithfulness could not be computed: no claim could be verified.",
            confidence=0.0,
            details={
                "failed": True,
                "claims": claim_details,
                "unverified_claims": len(unverified_claims),
            },
        )

    supported = len(supported_claims)
    score = clamp_score(supported / total)

    if supported == total:
        detail_msg = "All claims are grounded."
    else:
        detail_msg = f"{total - supported} claim(s) lack support."
    unverified_msg = (
        f" {len(unverified_claims)} claim(s) could not be verified and are excluded."
        if unverified_claims
        else ""
    )
    explanation = (
        f"Faithfulness score {score:.3f}: {supported}/{total} verified claims supported "
        f"by context. {detail_msg}{unverified_msg}"
    )

    details = {
        "claims": claim_details,
        "supported_claims": supported,
        "total_claims": total,
        "unverified_claims": len(unverified_claims),
        "unverified_list": unverified_claims[:5],
        "supported_list": supported_claims[:5],
        "unsupported_list": unsupported_claims[:5],
    }

    # Confidence: higher when we have more verified claims
    confidence = clamp_score(min(1.0, total / 5.0))

    return MetricScore(
        score=score,
        explanation=explanation,
        confidence=confidence,
        details=details,
    )


def _is_supported_verdict(text: str) -> bool:
    """Return True only for an explicit SUPPORTED verdict.

    A bare ``in`` check matches ``NOT_SUPPORTED`` too (it contains the
    substring ``SUPPORTED``). Require the verdict token at a word boundary
    and reject an explicit NOT_SUPPORTED verdict first.
    """
    upper = text.strip().upper()
    if re.search(r"\bNOT[_\s-]*SUPPORTED\b", upper):
        return False
    return re.search(r"^\s*SUPPORTED\b", upper) is not None


def _parse_json_list(text: str) -> list[str]:
    """Parse a JSON array of strings from LLM response."""
    # Try direct JSON parse
    text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [str(item) for item in result if item]
    except json.JSONDecodeError:
        pass

    # Try to extract JSON array from text
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return [str(item) for item in result if item]
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: split by lines or bullet points
    lines = []
    for line in text.split("\n"):
        line = line.strip().strip('"').strip("-").strip("*").strip()
        if line and len(line) > 5:
            lines.append(line)
    return lines
