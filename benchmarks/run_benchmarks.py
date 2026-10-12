"""
RAG Evaluation Framework Comprehensive Academic Benchmarking Suite
==================================================================
Designed for PhD-level evaluation. Covers:

    A. Functional Correctness each metric with known inputs
    B. Performance Profiling latency breakdown per metric & end-to-end
    C. Scalability Analysis batch size, context length, answer length
    D. Reliability / Stability run-to-run variance
    E. Edge-Case Robustness empty, adversarial, extreme inputs
    F. Cache Effectiveness hit / miss speedup
    G. Reporting Throughput JSON / Markdown / HTML generation speed
    H. Comparison Engine delta accuracy & verdict correctness
    I. Utility Layer BLEU, ROUGE-L, cosine-sim, Jaccard

All benchmarks use MockLLMAdapter for deterministic, reproducible results.
"""

import gc
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from rag_evaluation_framework import Evaluator
from rag_evaluation_framework.report import ReportBuilder
from rag_evaluation_framework.utils import (
    clamp_score,
    compute_bleu,
    compute_rouge_l,
    cosine_similarity,
    extract_claims,
    generate_cache_key,
    jaccard_similarity,
)
from tests.conftest import MockLLMAdapter


class SmartMockAdapter(MockLLMAdapter):
    """Improved mock that properly handles all RAG Evaluation Framework metric prompts."""

    async def complete(self, prompt: str, temperature: float = 0.0, **kwargs: Any) -> str:
        p = prompt.lower()
        self.call_history.append({"prompt": prompt[:50], "temperature": temperature})

        # Check user-supplied responses first
        if prompt in self.responses:
            return self.responses[prompt]
        for key, response in self.responses.items():
            if key in prompt:
                return response

        # Faithfulness: decompose
        if "decompose" in p or "atomic" in p:
            return '["The capital of France is Paris.", "Paris is in Europe."]'
        # Hallucination must come BEFORE faithfulness verify because the hallucination
        # prompt also contains "fact-checking". Check for "grounded" + "factual" as the unique
        # signature of the HALLUCINATION_VERIFY_PROMPT.
        if "grounded" in p and ("factual" in p or "hallucination" in p):
            return '{"grounded": true, "factual": true, "reason": "Supported by context"}'
        # Faithfulness: verify (SUPPORTED / NOT_SUPPORTED) unique signature: the word ENTAILED
        if "entailed" in p or ("SUPPORTED" in prompt and "claim" in p):
            return "SUPPORTED"
        # Answer relevance prompt says "how relevant" not "relevance"!
        if "how relevant" in p or "relevance" in p or "rate how well this answer" in p:
            return "0.95"
        # Context coverage prompt says "how well an answer covers", not "coverage"!
        if "how well an answer covers" in p or "coverage" in p or "covers the information" in p:
            return "0.90"
        # UCM: claim extraction
        if "extract" in p or "claims" in p and "atomic" not in p:
            return '["Paris is the capital of France."]'
        # Generic answer generation (for UCM samples)
        if "answer the following question" in p or "answer the question" in p:
            return "The capital of France is Paris, located on the Seine River."

        return "Mock response"

    async def embed(self, text: str | list[str]) -> Any:
        """Return deterministic embeddings (not random) for reproducible benchmarks."""
        import hashlib

        import numpy as np

        if isinstance(text, str):
            texts = [text]
        else:
            texts = list(text)

        # Deterministic embedding based on text hash (seed for reproducibility)
        def _det_embed(t: str) -> np.ndarray:
            h = hashlib.sha256(t.encode()).hexdigest()
            seed = int(h[:8], 16)
            rng = np.random.RandomState(seed)
            vec = rng.randn(384).astype(np.float32)
            # Normalize so cosine similarity is meaningful
            vec /= np.linalg.norm(vec) + 1e-10
            return vec

        embeddings = np.array([_det_embed(t) for t in texts])
        if isinstance(text, str):
            return embeddings[0]
        return embeddings


HAS_NUMPY = True  # already imported

# ═══════════════════════════════════════════════════════════════
# SECTION 0 Helpers
# ═══════════════════════════════════════════════════════════════


def _make_eval(adapter=None) -> Evaluator:
    # Use valid provider string, then override with smart mock for deterministic benchmarks
    e = Evaluator(llm="ollama/llama3", cache=True, model_config={"api_key": "test"})
    e._adapter = adapter or SmartMockAdapter()
    return e


def _measure(description: str, fn, *, iterations: int = 5, warmup: int = 1) -> dict:
    """Time a function with warmup rounds, return stats."""
    # Warmup
    for _ in range(warmup):
        fn()
    # Timed runs
    times: list[float] = []
    results = []
    for _ in range(iterations):
        if hasattr(gc, 'disable'):
            gc.disable()
        t0 = time.perf_counter()
        r = fn()
        t1 = time.perf_counter()
        if hasattr(gc, 'enable'):
            gc.enable()
        times.append((t1 - t0) * 1000)  # ms
        results.append(r)
    return {
        "description": description,
        "iterations": iterations,
        "mean_ms": round(statistics.mean(times), 3),
        "median_ms": round(statistics.median(times), 3),
        "min_ms": round(min(times), 3),
        "max_ms": round(max(times), 3),
        "stdev_ms": round(statistics.stdev(times), 3) if len(times) > 1 else 0.0,
        "last_result_preview": str(results[-1])[:120] if results else "",
    }


def _section(title: str):
    sep = "=" * 70
    return f"\n{sep}\n {title}\n{sep}"


# ═══════════════════════════════════════════════════════════════
# A FUNCTIONAL CORRECTNESS
# ═══════════════════════════════════════════════════════════════


def bench_functional_correctness() -> list[dict]:
    """Test each metric with known inputs verify scores are in [0,1] and non-NaN."""
    rows = []
    evaluator = _make_eval(SmartMockAdapter())

    test_cases = [
        # (name, question, context, answer)
        ("Perfect Match",
         "What is the capital of France?",
         ["France is in Europe. Its capital is Paris."],
         "The capital of France is Paris."),
        ("Partial Match",
         "What is the capital of France?",
         ["France is a country. Paris is a city in France."],
         "The capital is Paris, a beautiful city."),
        ("Contradiction",
         "What is the capital of France?",
         ["France is in Europe. Its capital is Paris."],
         "The capital of France is London."),
        ("Empty Answer",
         "What is the capital of France?",
         ["France is in Europe. Its capital is Paris."],
         ""),
        ("No Context",
         "What is 2+2?",
         [],
         "The answer is 4."),
        ("Multi-chunk",
         "What is the capital of France?",
         ["France is a country in Western Europe.",
          "Its capital is Paris, located on the Seine.",
          "Paris is a major cultural and economic center."],
         "The capital of France is Paris, located on the Seine River."),
    ]

    for name, q, ctx, ans in test_cases:
        try:
            r = evaluator.score(question=q, context=ctx, answer=ans)
            issues = []
            for attr in ["faithfulness", "hallucination_rate", "retrieval_precision",
                         "answer_relevance", "context_coverage", "ucm_confidence"]:
                s = getattr(r, attr).score
                if math.isnan(s) or s < -0.001 or s > 1.001:
                    issues.append(f"{attr}={s}")
            rows.append({
                "test": name,
                "question": q[:50],
                "overall": round(r.overall_score, 4),
                "faithfulness": round(r.faithfulness.score, 4),
                "hallucination": round(r.hallucination_rate.score, 4),
                "retrieval": round(r.retrieval_precision.score, 4),
                "relevance": round(r.answer_relevance.score, 4),
                "coverage": round(r.context_coverage.score, 4),
                "ucm": round(r.ucm_confidence.score, 4),
                "latency_ms": r.latency_ms,
                "issues": "; ".join(issues) if issues else "None",
            })
        except Exception as e:
            rows.append({"test": name, "error": str(e), "issues": "CRASHED"})

    return rows


# ═══════════════════════════════════════════════════════════════
# B PERFORMANCE PROFILING
# ═══════════════════════════════════════════════════════════════


def bench_performance() -> dict:
    """Latency breakdown: end-to-end, per metric, overhead."""
    evaluator = _make_eval(SmartMockAdapter())

    q = "What is the capital of France?"
    ctx = ["France is a country in Western Europe. Its capital is Paris.",
           "Paris is located on the Seine River."]
    ans = "The capital of France is Paris, located on the Seine River."

    results = {}

    # B1 Single full evaluation
    def single_eval():
        return evaluator.score(question=q, context=ctx, answer=ans)

    results["end_to_end_full"] = _measure("Single evaluation (all 6 metrics)", single_eval,
                                          iterations=10, warmup=3)

    # B2 Each metric individually
    for metric_name in ["faithfulness", "hallucination", "retrieval_precision",
                        "answer_relevance", "context_coverage"]:
        def make_single(m=metric_name):
            ev = _make_eval(SmartMockAdapter())
            return lambda: ev.score(question=q, context=ctx, answer=ans, metrics=[m])
        results[f"metric_{metric_name}"] = _measure(
            f"Metric: {metric_name}", make_single(), iterations=10, warmup=2)

    return results


# ═══════════════════════════════════════════════════════════════
# C SCALABILITY ANALYSIS
# ═══════════════════════════════════════════════════════════════


def bench_scalability() -> dict:
    """How does performance scale with batch size, context length, answer length?"""
    results = {}

    # C1 Batch sizes
    q = "What is the capital of France?"
    ctx = ["France is a country in Western Europe. Its capital is Paris."]
    ans = "The capital of France is Paris."

    for batch_size in [1, 5, 10, 25, 50, 100]:
        items = [{"question": q, "context": ctx, "answer": ans} for _ in range(batch_size)]

        def make_batch(bs=batch_size, it=items):
            ev = _make_eval(SmartMockAdapter())
            return lambda: ev.batch_score(it)
        r = _measure(f"Batch size={batch_size}", make_batch(),
                     iterations=3, warmup=1)
        items_per_sec = batch_size / (r["mean_ms"] / 1000) if r["mean_ms"] > 0 else 0
        r["throughput_items_per_sec"] = round(items_per_sec, 1)
        results[f"batch_{batch_size}"] = r

    # C2 Context length (number of chunks)
    for n_chunks in [1, 5, 10, 25, 50]:
        ctx_long = [f"Context chunk {i}: Paris is the capital of France." for i in range(n_chunks)]

        def make_ctx(n=n_chunks, c=ctx_long):
            ev = _make_eval(SmartMockAdapter())
            return lambda: ev.score(question=q, context=c, answer=ans)
        results[f"context_{n_chunks}chunks"] = _measure(
            f"Context: {n_chunks} chunks", make_ctx(), iterations=3, warmup=1)

    # C3 Answer length
    for n_words in [5, 25, 100, 500]:
        ans_long = ("The capital of France is Paris. " * (n_words // 7 + 1))[:n_words * 6]

        def make_ans(a=ans_long):
            ev = _make_eval(SmartMockAdapter())
            return lambda: ev.score(question=q, context=ctx[:2], answer=a)
        results[f"answer_{n_words}words"] = _measure(
            f"Answer: ~{n_words} words", make_ans(), iterations=3, warmup=1)

    return results


# ═══════════════════════════════════════════════════════════════
# D RELIABILITY (Run-to-Run Variance)
# ═══════════════════════════════════════════════════════════════


def bench_reliability() -> dict:
    """Run same evaluation N times measure score variance."""
    results = {}

    q = "What is the capital of France?"
    ctx = ["France is a country in Western Europe. Its capital is Paris."]
    ans = "The capital of France is Paris."

    scores: dict[str, list[float]] = {
        "overall": [], "faithfulness": [], "hallucination": [],
        "retrieval": [], "relevance": [], "coverage": [], "ucm": []
    }
    latencies = []

    for _ in range(30):
        e = _make_eval(SmartMockAdapter())
        r = e.score(question=q, context=ctx, answer=ans)
        scores["overall"].append(r.overall_score)
        scores["faithfulness"].append(r.faithfulness.score)
        scores["hallucination"].append(r.hallucination_rate.score)
        scores["retrieval"].append(r.retrieval_precision.score)
        scores["relevance"].append(r.answer_relevance.score)
        scores["coverage"].append(r.context_coverage.score)
        scores["ucm"].append(r.ucm_confidence.score)
        latencies.append(r.latency_ms)

    for metric, vals in scores.items():
        results[metric] = {
            "mean": round(statistics.mean(vals), 4),
            "median": round(statistics.median(vals), 4),
            "stdev": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
            "range": round(max(vals) - min(vals), 4),
            "coeff_variation_pct": round(
                statistics.stdev(vals) / statistics.mean(vals) * 100, 3
            ) if statistics.mean(vals) > 0 and len(vals) > 1 else 0.0,
        }

    results["latency_ms"] = {
        "mean": round(statistics.mean(latencies), 2),
        "stdev": round(statistics.stdev(latencies), 2) if len(latencies) > 1 else 0.0,
        "min": round(min(latencies), 2),
        "max": round(max(latencies), 2),
        "range": round(max(latencies) - min(latencies), 2),
        "coeff_variation_pct": round(
            statistics.stdev(latencies) / statistics.mean(latencies) * 100, 3
        ) if statistics.mean(latencies) > 0 and len(latencies) > 1 else 0.0,
    }
    results["n_runs"] = 30
    return results


# ═══════════════════════════════════════════════════════════════
# E EDGE-CASE ROBUSTNESS
# ═══════════════════════════════════════════════════════════════


def bench_edge_cases() -> list[dict]:
    """Feed adversarial inputs framework should not crash."""
    rows = []

    cases = [
        ("All empty strings", "", [""], ""),
        ("Very long question", "What? " * 1000, ["A" * 100], "Answer."),
        ("Special characters",
         "\u00a7\u00b6\u2190\u00bb\ufffd\n\t\r\x00",
         ["\u03b1\u03b2\u03b3 \u03b4\u03b6\u03b7"],
         "\u221e\u2248\u2260\u2264 \u2265"),
        ("Unicode question", "\u00bfCu\u00e1l es la capital de Francia?",
         ["Francia est\u00e1 en Europa. Su capital es Par\u00eds."],
         "La capital es Par\u00eds."),
        ("JSON injection", '{"question": "x"}', ['context " test'], 'answer " test'),
        ("HTML injection", "<script>alert('xss')</script>",
         ["<b>bold</b> context"], "<i>italic</i> answer"),
        ("50 chunks context", "Question?", ["chunk"] * 50, "Answer."),
        ("Large answer (5000 chars)", "Question?", ["Context."], "X" * 5000),
        ("Numeric inputs", "12345", ["678 910"], "1112 1314"),
        ("Only whitespace", " ", [" ", " "], " "),
    ]

    for name, q, ctx, ans in cases:
        try:
            e = _make_eval(SmartMockAdapter())
            t0 = time.perf_counter()
            r = e.score(question=q, context=ctx, answer=ans)
            elapsed = (time.perf_counter() - t0) * 1000
            rows.append({
                "test": name,
                "crashed": False,
                "overall": round(r.overall_score, 4),
                "latency_ms": round(elapsed, 2),
            })
        except Exception as ex:
            rows.append({
                "test": name,
                "crashed": True,
                "error": str(ex)[:100],
                "latency_ms": -1,
            })

    return rows


# ═══════════════════════════════════════════════════════════════
# F CACHE EFFECTIVENESS
# ═══════════════════════════════════════════════════════════════


def bench_cache() -> dict:
    """Measure speedup from in-memory caching."""
    q = "What is the capital of France?"
    ctx = ["France is in Europe. Its capital is Paris."]
    ans = "The capital of France is Paris."

    # Uncached fresh evaluator each time
    uncached_times = []
    for _ in range(5):
        e = _make_eval(SmartMockAdapter())
        t0 = time.perf_counter()
        e.score(question=q, context=ctx, answer=ans)
        uncached_times.append((time.perf_counter() - t0) * 1000)

    # Cached same evaluator, same inputs
    e = _make_eval(SmartMockAdapter())
    # First call populates cache
    e.score(question=q, context=ctx, answer=ans)
    cached_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        e.score(question=q, context=ctx, answer=ans)
        cached_times.append((time.perf_counter() - t0) * 1000)

    mean_uncached = statistics.mean(uncached_times)
    mean_cached = statistics.mean(cached_times)
    speedup = mean_uncached / mean_cached if mean_cached > 0 else float("inf")

    return {
        "uncached_mean_ms": round(mean_uncached, 3),
        "uncached_stdev_ms": round(statistics.stdev(uncached_times), 3),
        "cached_mean_ms": round(mean_cached, 3),
        "cached_stdev_ms": round(statistics.stdev(cached_times), 3),
        "speedup_x": round(speedup, 2),
        "latency_reduction_pct": round((1 - mean_cached / mean_uncached) * 100, 2),
        "n_uncached": len(uncached_times),
        "n_cached": len(cached_times),
    }


# ═══════════════════════════════════════════════════════════════
# G REPORTING THROUGHPUT
# ═══════════════════════════════════════════════════════════════


def bench_reporting() -> dict:
    """Speed of report generation in all formats + CI badge creation."""
    e = _make_eval(SmartMockAdapter())
    r = e.score(question="What is the capital of France?",
                context=["France is a country. Its capital is Paris."],
                answer="The capital of France is Paris.")
    builder = ReportBuilder(r)

    def gen_json():
        return r.report(format="json")

    def gen_md():
        return r.report(format="markdown")

    def gen_html():
        return r.report(format="html")

    def gen_dict():
        return r.report(format="dict")

    def gen_badge():
        return builder.to_ci_badge("overall_score")

    return {
        "json": _measure("Report: JSON", gen_json, iterations=50, warmup=5),
        "markdown": _measure("Report: Markdown", gen_md, iterations=50, warmup=5),
        "html": _measure("Report: HTML", gen_html, iterations=50, warmup=5),
        "dict": _measure("Report: Dict", gen_dict, iterations=50, warmup=5),
        "ci_badge": _measure("CI Badge URL", gen_badge, iterations=50, warmup=5),
    }


# ═══════════════════════════════════════════════════════════════
# H COMPARISON ENGINE
# ═══════════════════════════════════════════════════════════════


def bench_comparison() -> dict:
    """Compare two results verify delta accuracy & verdict correctness."""
    e = _make_eval(SmartMockAdapter())

    r_good = e.score(question="Q?", context=["Good context."], answer="Good answer.")
    r_bad = e.score(question="Q?", context=["Bad context."], answer="Bad answer.")

    t0 = time.perf_counter()
    comp = e.compare(r_good, r_bad)
    cmp_time = (time.perf_counter() - t0) * 1000

    # Self-comparison (should be zero deltas)
    comp_self = e.compare(r_good, r_good)
    zero_deltas = all(abs(d) < 0.001 for d in comp_self.score_deltas.values())

    return {
        "comparison_latency_ms": round(cmp_time, 3),
        "score_deltas": {k: round(v, 4) for k, v in comp.score_deltas.items()},
        "verdict": comp.verdict,
        "self_comparison_zero_deltas": zero_deltas,
    }


# ═══════════════════════════════════════════════════════════════
# I UTILITY LAYER MICRO-BENCHMARKS
# ═══════════════════════════════════════════════════════════════


def bench_utilities() -> dict:
    """Micro-benchmarks for core utility functions."""
    results = {}

    # I1 Cosine Similarity
    v1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    v2 = np.array([0.4, 0.5, 0.6], dtype=np.float32)
    results["cosine_similarity"] = _measure(
        "cosine_similarity 3-dim",
        lambda: cosine_similarity(v1, v2),
        iterations=1000, warmup=100
    )

    # I2 BLEU
    ref = "The capital of France is Paris"
    hyp = "The capital of France is Paris"
    results["bleu_exact_match"] = _measure(
        "BLEU (exact match)",
        lambda: compute_bleu(ref, hyp),
        iterations=1000, warmup=100
    )
    results["bleu_no_match"] = _measure(
        "BLEU (no match)",
        lambda: compute_bleu("abcdefghij", "klmnopqrst"),
        iterations=1000, warmup=100
    )

    # I3 ROUGE-L
    results["rouge_l"] = _measure(
        "ROUGE-L (partial)",
        lambda: compute_rouge_l("The capital of France is Paris",
                                "Paris is the capital of France"),
        iterations=1000, warmup=100
    )

    # I4 Jaccard
    results["jaccard"] = _measure(
        "Jaccard similarity",
        lambda: jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"}),
        iterations=1000, warmup=100
    )

    # I5 extract_claims
    results["extract_claims"] = _measure(
        "extract_claims 5 sentences",
        lambda: extract_claims("Paris is capital. France is in Europe. "
                               "The Seine flows through Paris. "
                               "France uses Euro. Paris is beautiful."),
        iterations=1000, warmup=100
    )

    # I6 generate_cache_key
    results["cache_key"] = _measure(
        "generate_cache_key",
        lambda: generate_cache_key("What is the capital of France?",
                                   ["France is in Europe. Its capital is Paris."],
                                   "The capital of France is Paris.",
                                   "openai/gpt-4o"),
        iterations=1000, warmup=100
    )

    # I7 clamp_score
    results["clamp_score"] = _measure(
        "clamp_score",
        lambda: clamp_score(-0.5) + clamp_score(1.5) + clamp_score(0.5),
        iterations=1000, warmup=100
    )

    return results


# ═══════════════════════════════════════════════════════════════
# RUN ALL & SAVE
# ═══════════════════════════════════════════════════════════════


def run_all() -> dict:
    # Force UTF-8 for Windows console
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    print("#" + "=" * 68 + "#")
    print("# RAG Evaluation Framework Comprehensive Academic Benchmark Suite #")
    print("#" + "=" * 68 + "#")
    print()

    results: dict[str, Any] = {"_meta": {"suite_version": "1.0.0"}}

    # A
    print(f"{_section('A FUNCTIONAL CORRECTNESS')}")
    results["A_functional"] = bench_functional_correctness()
    n_ok = sum(1 for r in results["A_functional"] if r.get("issues") == "None")
    n_total = len(results["A_functional"])
    print(f" {n_ok}/{n_total} cases passed without issues")
    for r in results["A_functional"]:
        if r.get("issues", "None") != "None":
            print(f" \u26a0 {r['test']}: {r['issues']}")

    # B
    print(f"{_section('B PERFORMANCE PROFILING')}")
    results["B_performance"] = bench_performance()
    for k, v in results["B_performance"].items():
        print(f" {v['description']:<40} {v['mean_ms']:>8.2f} ms \u00b1{v['stdev_ms']:.2f}")

    # C
    print(f"{_section('C SCALABILITY')}")
    results["C_scalability"] = bench_scalability()
    for k, v in results["C_scalability"].items():
        extra = ""
        if "throughput_items_per_sec" in v:
            extra = f" ({v['throughput_items_per_sec']} items/s)"
        print(f" {v['description']:<40} {v['mean_ms']:>8.2f} ms \u00b1{v['stdev_ms']:.2f}{extra}")

    # D
    print(f"{_section('D RELIABILITY (n=30)')}")
    results["D_reliability"] = bench_reliability()
    for metric, data in results["D_reliability"].items():
        if metric == "n_runs":
            continue
        print(f" {metric:<25} mean={data['mean']:.4f} \u03c3={data['stdev']:.4f} "
              f"CV={data.get('coeff_variation_pct', 0):.3f}% "
              f"range=[{data['min']:.4f}, {data['max']:.4f}]")

    # E
    print(f"{_section('E EDGE-CASE ROBUSTNESS')}")
    results["E_edge_cases"] = bench_edge_cases()
    n_crashed = sum(1 for r in results["E_edge_cases"] if r.get("crashed"))
    print(f" {len(results['E_edge_cases']) - n_crashed}/{len(results['E_edge_cases'])} survived")
    for r in results["E_edge_cases"]:
        status = "\U0001f4a5 CRASHED" if r["crashed"] else f"\u2705 {r['overall']:.3f}"
        print(f" {r['test']:<35} {status} ({r['latency_ms']:.1f} ms)")

    # F
    print(f"{_section('F CACHE EFFECTIVENESS')}")
    results["F_cache"] = bench_cache()
    c = results["F_cache"]
    print(f" Uncached: {c['uncached_mean_ms']:.3f} ms")
    print(f" Cached: {c['cached_mean_ms']:.3f} ms")
    print(f" Speedup: {c['speedup_x']}\u00d7")
    print(f" Latency reduction: {c['latency_reduction_pct']:.1f}%")

    # G
    print(f"{_section('G REPORTING THROUGHPUT')}")
    results["G_reporting"] = bench_reporting()
    for k, v in results["G_reporting"].items():
        print(f" {v['description']:<40} {v['mean_ms']:>8.3f} ms \u00b1{v['stdev_ms']:.3f}")

    # H
    print(f"{_section('H COMPARISON ENGINE')}")
    results["H_comparison"] = bench_comparison()
    comp = results["H_comparison"]
    print(f" Comparison latency: {comp['comparison_latency_ms']:.3f} ms")
    print(f" Self-comparison zero deltas: {comp['self_comparison_zero_deltas']}")
    print(f" Verdict: {comp['verdict']}")
    for k, v in comp["score_deltas"].items():
        print(f" {k}: {v:+.4f}")

    # I
    print(f"{_section('I UTILITY MICRO-BENCHMARKS')}")
    results["I_utilities"] = bench_utilities()
    for k, v in results["I_utilities"].items():
        print(f" {v['description']:<40} {v['mean_ms']:>8.3f} ms \u00d7{v['iterations']} iters")

    # Summary timeline
    bar = "\u2550" * 70
    print(f"\n{bar}")
    print(" BENCHMARK COMPLETE")
    print(f"{bar}\n")

    return results


def save_results(results: dict, path: str = "benchmarks/benchmark_results.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types
    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            return super().default(obj)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f" Raw results saved to {path}")


if __name__ == "__main__":
    data = run_all()
    save_results(data)
