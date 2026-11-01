"""UCM (Unsupervised Confidence Metric) flagship module.

Estimates confidence in the LLM's answer without requiring ground truth labels.
Uses multi-sample consistency analysis: semantic, lexical, and factual overlap.
"""

from __future__ import annotations

import json
import re

import numpy as np

from rag_evaluation_framework.adapters.base import LLMAdapter
from rag_evaluation_framework.models import MetricScore
from rag_evaluation_framework.prompts import UCM_CLAIM_EXTRACTION_PROMPT
from rag_evaluation_framework.utils import (
    clamp_score,
    compute_bleu,
    compute_rouge_l,
    cosine_similarity,
    extract_claims,
    jaccard_similarity,
)

# Number of samples to generate
DEFAULT_NUM_SAMPLES = 5
DEFAULT_TEMPERATURE = 0.7

# Weights for UCM score components
SEMANTIC_WEIGHT = 0.4
FACTUAL_WEIGHT = 0.4
LEXICAL_WEIGHT = 0.2


async def score_ucm_confidence(
    question: str,
    context: list[str],
    answer: str,
    adapter: LLMAdapter,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    temperature: float = DEFAULT_TEMPERATURE,
) -> MetricScore:
    """Compute unsupervised confidence by analyzing answer consistency.

    Generates multiple samples for the same (question, context) pair and measures:
        1. Semantic consistency (pairwise cosine similarity of embeddings)
        2. Lexical consistency (pairwise BLEU/ROUGE scores)
        3. Factual claim overlap (Jaccard similarity of extracted claims)

    High UCM = model consistently produces the same answer -> high confidence.
    Low UCM = model produces varying answers -> low confidence.

    Returns:
        MetricScore with score = weighted UCM score (0.0 - 1.0).
    """
    # Step 1: Generate multiple samples
    context_str = "\n".join(context)
    base_prompt = f"""Given this context, answer the following question.

Context:
    {context_str}

Question: {question}

Answer the question concisely using only the information from the context."""

    samples = [answer]  # Include the original answer as the first sample
    try:
        additional_samples = await adapter.complete_batch(
            [base_prompt] * num_samples,
            temperature=temperature,
        )
        samples.extend(additional_samples)
    except Exception as exc:
        # Without additional samples there is no consistency signal to measure;
        # returning a score here would report confidence that was never computed.
        return MetricScore(
            score=None,
            explanation=f"UCM confidence could not be computed: {exc}",
            confidence=0.0,
            details={"error": str(exc), "failed": True},
        )

    # Deduplicate and filter empty samples
    samples = list(dict.fromkeys(s for s in samples if s.strip()))
    if len(samples) < 2:
        return MetricScore(
            score=1.0,
            explanation=(
                "Only one unique sample generated cannot assess consistency. "
                "Defaulting to high confidence."
            ),
            confidence=0.3,
            details={
                "num_samples": len(samples),
                "note": "Insufficient diversity for UCM analysis.",
            },
        )

    # Step 2: Compute semantic consistency
    try:
        embeddings = await adapter.embed(samples)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
    except Exception:
        embeddings = None

    semantic_consistency = 0.0
    if embeddings is not None and len(embeddings) > 1:
        pairwise_sims = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = cosine_similarity(embeddings[i], embeddings[j])
                pairwise_sims.append(sim)
        semantic_consistency = float(np.mean(pairwise_sims)) if pairwise_sims else 0.0

    # Step 3: Compute lexical consistency (BLEU + ROUGE-L)
    lexical_scores = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            bleu = compute_bleu(samples[i], samples[j])
            rouge = compute_rouge_l(samples[i], samples[j])
            lexical_scores.append((bleu + rouge) / 2.0)

    lexical_consistency = float(np.mean(lexical_scores)) if lexical_scores else 0.0

    # Step 4: Compute factual claim overlap
    all_claims = []
    for s in samples:
        try:
            claim_prompt = UCM_CLAIM_EXTRACTION_PROMPT.format(text=s)
            claim_response = await adapter.complete(claim_prompt, temperature=0.0)
            claims = _parse_json_list(claim_response)
        except Exception:
            claims = extract_claims(s)
        all_claims.append(set(c.strip().lower() for c in claims))

    factual_overlaps = []
    for i in range(len(all_claims)):
        for j in range(i + 1, len(all_claims)):
            jaccard = jaccard_similarity(all_claims[i], all_claims[j])
            factual_overlaps.append(jaccard)

    factual_overlap = float(np.mean(factual_overlaps)) if factual_overlaps else 0.0

    # Step 5: Compute weighted UCM score
    ucm_score = (
        SEMANTIC_WEIGHT * clamp_score(semantic_consistency)
        + FACTUAL_WEIGHT * clamp_score(factual_overlap)
        + LEXICAL_WEIGHT * clamp_score(lexical_consistency)
    )
    ucm_score = clamp_score(ucm_score)

    # Build explanation
    explanation_parts = [
        f"UCM score {ucm_score:.3f}: "
        f"semantic_consistency={semantic_consistency:.3f}, "
        f"factual_overlap={factual_overlap:.3f}, "
        f"lexical_consistency={lexical_consistency:.3f}."
    ]

    if ucm_score >= 0.8:
        explanation_parts.append("High consistency across samples the model is confident.")
    elif ucm_score >= 0.5:
        explanation_parts.append("Moderate consistency some variation in outputs.")
    else:
        explanation_parts.append(
            "Low consistency the model produces varied outputs. Low confidence recommended."
        )

    # Uncertainty analysis
    if factual_overlap < 0.5 and len(samples) >= 3:
        explanation_parts.append(
            "The model makes different factual claims across samples high uncertainty."
        )
    elif semantic_consistency > 0.9 and lexical_consistency < 0.5:
        explanation_parts.append(
            "Semantic similarity is high but lexical overlap is low the model conveys "
            "the same meaning using different words, indicating robust understanding."
        )

    explanation = " ".join(explanation_parts)

    details = {
        "num_samples": len(samples),
        "samples": samples[:5],
        "semantic_consistency": float(semantic_consistency),
        "lexical_consistency": float(lexical_consistency),
        "factual_overlap": float(factual_overlap),
        "pairwise_lexical_scores": [float(s) for s in lexical_scores[:10]],
        "weights": {
            "semantic": SEMANTIC_WEIGHT,
            "factual": FACTUAL_WEIGHT,
            "lexical": LEXICAL_WEIGHT,
        },
    }

    # Confidence in the UCM estimate itself higher with more samples
    meta_confidence = clamp_score(min(1.0, len(samples) / 8.0))

    return MetricScore(
        score=ucm_score, explanation=explanation, confidence=meta_confidence, details=details
    )


def _parse_json_list(text: str) -> list[str]:
    """Parse a JSON array of strings from LLM response."""
    text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [str(item) for item in result if item]
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return [str(item) for item in result if item]
        except (json.JSONDecodeError, ValueError):
            pass

    lines = []
    for line in text.split("\n"):
        line = line.strip().strip('"').strip("-").strip("*").strip()
        if line and len(line) > 5:
            lines.append(line)
    return lines
