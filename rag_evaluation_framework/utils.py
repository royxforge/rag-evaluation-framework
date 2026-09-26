"""Utility functions for RAG Evaluation Framework."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def clamp_score(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp a score to [min_val, max_val]."""
    return max(min_val, min(max_val, value))


def extract_claims(text: str) -> list[str]:
    """Extract atomic factual claims from text using sentence splitting.

    Args:
        text: Input text.

    Returns:
        List of atomic claim strings.
    """
    # Split by sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    claims = []
    for s in sentences:
        s = s.strip()
        if s and len(s) > 5:
            claims.append(s)
    return claims


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L F1 score (simplified LCS-based)."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    # LCS length
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[m][n]
    if lcs == 0:
        return 0.0

    precision = lcs / n
    recall = lcs / m
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """Compute a simplified BLEU score."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not hyp_tokens:
        return 0.0

    # Brevity penalty
    bp = min(1.0, float(len(ref_tokens)) / len(hyp_tokens)) if hyp_tokens else 0.0

    # N-gram precision
    scores = []
    for n in range(1, min(max_n, len(hyp_tokens)) + 1):
        hyp_ngrams = {}
        for i in range(len(hyp_tokens) - n + 1):
            gram = " ".join(hyp_tokens[i: i + n])
            hyp_ngrams[gram] = hyp_ngrams.get(gram, 0) + 1

        ref_ngrams = {}
        for i in range(len(ref_tokens) - n + 1):
            gram = " ".join(ref_tokens[i: i + n])
            ref_ngrams[gram] = ref_ngrams.get(gram, 0) + 1

        matches = 0
        total = 0
        for gram, count in hyp_ngrams.items():
            ref_count = ref_ngrams.get(gram, 0)
            matches += min(count, ref_count)
            total += count

        if total > 0:
            scores.append(matches / total)

    if not scores:
        return 0.0

    geo_mean = np.exp(np.mean(np.log(scores))) if all(s > 0 for s in scores) else 0.0
    return float(bp * geo_mean)


def jaccard_similarity(set_a: set[Any], set_b: set[Any]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return intersection / union


def generate_cache_key(
    question: str,
    context: list[str],
    answer: str,
    llm: str,
    metrics: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Generate a deterministic cache key for an evaluation.

    The key includes the resolved metric set (and any extra scoring
    options) so a ``score(metrics=["faithfulness"])`` result can never be
    returned for a later ``score(metrics=["all"])`` call on the same text.
    """
    payload: dict[str, Any] = {
        "question": question,
        "context": sorted(context),
        "answer": answer,
        "llm": llm,
    }
    if metrics is not None:
        payload["metrics"] = sorted(metrics)
    if extra:
        payload["extra"] = extra
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()
