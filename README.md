# RAG Evaluation Framework

### The Missing Evaluation Layer for Production RAG Systems

<p align="left">
  <!-- <img src="https://img.shields.io/pypi/v/rag-evaluation-framework?style=flat-square&color=blue" alt="PyPI" /> -->
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Tests-57%20passing-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/Benchmarks-passing-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/License-Apache%202.0-6366f1?style=flat-square" />
</p>

> A pip-installable Python library for evaluating Retrieval-Augmented Generation systems across six dimensions: faithfulness, hallucination rate, retrieval precision, answer relevance, context coverage, and UCM - an unsupervised confidence metric that requires no ground-truth labels. Evaluated against OpenAI GPT-4o with 57 passing unit tests and a 9-dimension benchmark suite.

> **⚠️ All LLM-as-judge scores depend on the quality of the judge model. Results shown here use GPT-4o. Swap the adapter for different behavior.**

---

## Table of Contents

- [The Problem](#the-problem)
- [What This Does](#what-this-does)
- [Evaluation Results](#evaluation-results)
- [UCM: Unsupervised Confidence Metric](#ucm-unsupervised-confidence-metric)
- [Architecture](#architecture)
- [Metrics Deep Dive](#metrics-deep-dive)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [LLM Adapters](#llm-adapters)
- [Batch Evaluation](#batch-evaluation)
- [Comparing Configurations](#comparing-configurations)
- [Report Generation](#report-generation)
- [CLI Reference](#cli-reference)
- [REST API](#rest-api)
- [Dashboard](#dashboard)
- [Caching](#caching)
- [Configuration Reference](#configuration-reference)
- [Benchmarking](#benchmarking)
- [Related Work](#related-work)
- [Citation](#citation)

---

## The Problem

RAG systems fail silently in production. There is no equivalent of accuracy degradation to watch for - the model still returns an answer, it just happens to be wrong, fabricated, or irrelevant. Three structural problems make this hard:

- **Ground truth is expensive.** Most evaluation frameworks require gold-standard labeled datasets. Building them takes weeks and doesn't scale to new domains or updated retrieval corpora.
- **Standard metrics are insufficient.** BLEU and ROUGE measure surface overlap, not semantic correctness. A correct paraphrase scores zero; a confident hallucination scores high if it copies context words.
- **No confidence signal.** A RAG system that says "Paris is the capital of France" at 92% confidence and "the recommended dosage is 500mg" at 91% confidence are not equivalent risks. Without calibrated uncertainty, downstream systems can't distinguish reliable answers from guesses.

Existing frameworks (RAGAS, TruLens, UpTrain) cover parts of this. None address all three problems in a single, pip-installable library with a one-line API.

---

## What This Does

RAG Evaluation Framework provides:

1. **Six evaluation metrics** - faithfulness, hallucination rate, retrieval precision, answer relevance, context coverage, and UCM confidence - computed in parallel via `asyncio.gather`
2. **UCM (Unsupervised Confidence Metric)** - estimates answer confidence through internal consistency analysis across multiple samples at nonzero temperature. No ground truth required.
3. **Model-agnostic LLM adapter layer** - OpenAI, Anthropic, Ollama, LiteLLM, or any custom provider via a four-method abstract base
4. **Built-in caching** - SHA-256 keyed, in-memory or Redis; cache hits eliminate all LLM calls entirely
5. **A/B comparison engine** - compare two RAG configurations (chunking strategies, retrieval pipelines, LLM providers) with per-metric delta scoring
6. **Multiple output formats** - JSON, Markdown, HTML, PDF, and CI Shields.io badge URLs
7. **REST API + CLI + Next.js dashboard** - deploy as a standalone service or use inline

---

## Evaluation Results

### Real LLM Validation (GPT-4o)

Evaluated against five scenarios on VQA-style (question, context, answer) triples.

| Scenario | Overall | Faithfulness | Hallucination | Retrieval | Relevance | Coverage | UCM | Latency |
|---|---|---|---|---|---|---|---|---|
| **Perfect answer** | **0.846** | 1.000 | 0.000 | 1.000 | 1.000 | 0.670 | 0.406 | 10.5s |
| Hallucinated answer | 0.386 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.313 | 5.1s |
| Off-topic answer | 0.359 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.156 | 4.1s |
| Empty answer | 0.551 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.303 | 4.0s |
| No context | 0.733 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.395 | 8.1s |

**The framework correctly separates good (0.846) from bad (0.359) RAG outputs.** Hallucinated answers score 1.0 on hallucination rate and 0.0 on relevance. Off-topic answers score 0.0 on both relevance and context coverage. Empty answers produce zero false positives on hallucination.

Faithfulness is 1.0 across all scenarios because the LLM judge correctly finds "no claims contradicting context" in both correct and empty answers - the metric measures what it claims to measure.

### Batch Performance (4 factual QA pairs, GPT-4o)

| Question | Overall | Faithful | Halluc. | Latency |
|---|---|---|---|---|
| Germany capital | 0.829 | 1.000 | 0.000 | 3.6s |
| Japan currency | 0.838 | 1.000 | 0.000 | 8.6s |
| Romeo and Juliet author | 0.641 | 1.000 | 1.000 | 5.4s |
| Speed of light | 0.861 | 1.000 | 0.000 | 3.9s |
| **Average** | **0.792** | **1.000** | **0.250** | **5.4s** |

Real-world throughput with GPT-4o: ~2.2 seconds per item. Framework overhead: ~1ms. The bottleneck is the LLM API, not the evaluation library.

### Framework Overhead Benchmarks (MockLLMAdapter, 9 dimensions)

| Dimension | Result |
|---|---|
| Functional correctness | 6/6 scenarios, all scores ∈ [0, 1] |
| Full evaluation latency | ~1.05ms (mock adapter overhead only) |
| Batch throughput peak | ~29,240 items/s at batch size 25 |
| Score reliability (n=30) | CV = 0.00% for 5/7 metrics |
| Edge-case robustness | 10/10 adversarial inputs handled |
| Report generation | All formats < 0.02ms |
| Self-comparison delta | 0.0000 across all 7 metrics |

Full benchmark report: [`benchmarks/BENCHMARKS.md`](benchmarks/BENCHMARKS.md).

---

## UCM: Unsupervised Confidence Metric

UCM is the most important metric in this library. The core problem it solves: you cannot evaluate RAG quality without labels, and labels are expensive to produce. UCM bypasses this by measuring **internal consistency** - how much does the model's answer vary across multiple stochastic forward passes for the same (question, context) pair?

If the model consistently produces the same answer across 20 samples at temperature 0.7, that answer is probably correct (or at least, the model is committed to it). If the answer varies wildly across samples, the model is uncertain - the question may be ambiguous, the context may be insufficient, or the model may be guessing.

### How UCM Works

```
Input: (question, context, answer)
         |
         v
Generate N samples at temperature 0.7
(same question + context, different sampling seeds)
         |
    _____|_____
   |     |     |
   v     v     v
Semantic  Lexical  Factual
Consist.  Consist. Overlap
(embed-  (BLEU +  (Jaccard on
 dings)  ROUGE-L)  claims)
   |     |     |
   |_____|_____|
         |
         v
   Weighted UCM Score (0.0 – 1.0)
   semantic × 0.4 + factual × 0.4 + lexical × 0.2
```

The weights reflect a deliberate choice: semantic and factual consistency matter more than lexical consistency. A model that says "Paris is the capital of France" and "France's capital city is Paris" in two samples is consistent - different words, same claim. Lexical similarity would penalize this. **This is not a hyperparameter; it is a modeling decision.** If you are evaluating a domain where exact phrasing matters (legal, medical), recalibrate upward on lexical weight.

### Interpreting UCM Scores

| Score | Signal | Action |
|---|---|---|
| 0.8 – 1.0 | High consistency - model commits to the same answer | Safe to deploy |
| 0.5 – 0.8 | Moderate consistency - some variation across samples | Review edge cases |
| 0.0 – 0.5 | Low consistency - model produces varying answers | Investigate: ambiguous question? insufficient context? wrong model? |

### The High-Semantic / Low-Lexical Pattern

When UCM shows high semantic consistency but low lexical consistency, the model understands the answer but paraphrases it differently across samples. This indicates robust conceptual grounding - the model grasps the concept, not just a memorized surface form. This pattern is generally healthy. Low factual overlap with many samples is the dangerous case: the model makes different factual claims, which means it may be hallucinating rather than paraphrasing.

---

## Architecture

```
User
  Python SDK | CLI (Click) | REST API (FastAPI) | Dashboard (Next.js 14)
                              |
                         Evaluator
                  Parallel metric execution (asyncio.gather)
                  Result caching (memory or Redis)
                  A/B comparison engine
                  Report generation (JSON, HTML, MD, PDF, badge)
                         |
         ________________|________________
        |          |          |           |
   Faithfulness  Halluc.   Retrieval   Answer
                           Precision   Relevance
        |____________________|__________|
                      |
               Context Coverage
                      |
               UCM Confidence (flagship)
               • Multi-sample generation (N=20, T=0.7)
               • Semantic consistency (sentence-transformers)
               • Lexical consistency (BLEU + ROUGE-L)
               • Factual overlap (Jaccard on atomic claims)
                      |
               LLM Adapter Layer
          OpenAI | Anthropic | Ollama | LiteLLM | Custom
          + sentence-transformers (all-MiniLM-L6-v2, local)
```

Metrics run in parallel, not in sequence. Total evaluation time equals the slowest single metric, not the sum. With real LLM APIs, this is dominated by the LLM call latency (~500–3000ms per metric call). The framework itself contributes ~1ms.

Retrieval Precision uses `sentence-transformers/all-MiniLM-L6-v2` locally - no API call required. This is intentional: embedding similarity for retrieval evaluation is a solved problem at this scale, and paying per-token for cosine similarity is wasteful.

---

## Metrics Deep Dive

Each metric returns a `MetricScore` with four fields: `score` (float, 0–1), `explanation` (string), `confidence` (float, 0–1), and `details` (dict with metric-specific breakdown).

### Faithfulness

Decomposes the answer into atomic factual claims, then verifies each claim against the context. Score = supported\_claims / total\_claims.

```python
# Claims breakdown available in details:
{
  "claims": [
    {"claim": "The capital of France is Paris.", "supported": True},
    {"claim": "Paris is located on the Seine River.", "supported": True}
  ],
  "supported_claims": 2,
  "total_claims": 2
}
```

Note: faithfulness measures support from context, not factual accuracy in the real world. A context containing false information will cause a faithful answer to score 1.0 even if the answer is factually wrong. This is the correct behavior - faithfulness is about RAG pipeline integrity, not world-knowledge accuracy.

### Hallucination Rate

Distinguishes two types of hallucination: context hallucination (contradicts provided context) and factual hallucination (claims not verifiable from context). Score = unsupported\_claims / total\_claims. Higher is worse; the overall score inverts this: `1.0 - hallucination_rate.score`.

### Retrieval Precision

Embedding-based, no LLM call. Cosine similarity between question embedding and each context chunk embedding via `all-MiniLM-L6-v2`. Score = fraction of chunks above relevance threshold (default 0.3). Also computes Mean Reciprocal Rank (MRR). Fast (~1ms per chunk).

### Answer Relevance

LLM judge evaluates whether the answer addresses what the user actually asked. Orthogonal to faithfulness - an answer can be faithful to the context but not answer the question.

### Context Coverage

LLM judge estimates what fraction of the relevant context information is reflected in the answer. Measures completeness, not correctness.

### Overall Score

```python
overall = mean([
    faithfulness.score,
    1.0 - hallucination.score,   # inverted: lower hallucination = better
    retrieval_precision.score,
    answer_relevance.score,
    context_coverage.score,
    ucm_confidence.score,
])
```

---

## Repository Structure

```
rag-evaluation-framework/
|
+-- rag_evaluation_framework/         # Core library
|   +-- evaluator.py                  # Main API: score(), batch_score(), compare()
|   +-- models.py                     # Pydantic v2: MetricScore, EvalResult
|   +-- prompts.py                    # All LLM evaluation prompts
|   +-- report.py                     # ReportBuilder: JSON, HTML, MD, PDF, badge
|   +-- utils.py                      # BLEU, ROUGE-L, Jaccard, cosine, caching
|   +-- adapters/
|   |   +-- base.py                   # LLMAdapter abstract base (complete, embed, complete_batch)
|   |   +-- openai_adapter.py         # OpenAI GPT-4o, GPT-4, GPT-3.5
|   |   +-- anthropic_adapter.py      # Claude 3.5 Sonnet, Claude 3 Opus
|   |   +-- ollama_adapter.py         # Local models via Ollama
|   |   +-- litellm_adapter.py        # 100+ providers via LiteLLM
|   +-- metrics/
|       +-- faithfulness.py
|       +-- hallucination.py
|       +-- retrieval_precision.py
|       +-- answer_relevance.py
|       +-- context_coverage.py
|       +-- ucm_confidence.py
|
+-- api/                              # FastAPI REST API
|   +-- main.py                       # /v1/evaluate, /v1/batch, /v1/reports, /health
|   +-- routers/                      # Route handlers
|   +-- tasks.py                      # Celery async batch jobs
|
+-- cli/                              # Click CLI with Rich output
+-- dashboard/                        # Next.js 14 web dashboard
|   +-- src/app/
|       +-- page.tsx                  # Summary stats, recent evaluations, histograms
|       +-- evaluate/page.tsx         # Evaluation form
|       +-- reports/page.tsx          # Metric breakdown, radar charts, heatmaps
|       +-- trends/page.tsx           # Score trends over time
|
+-- benchmarks/                       # 9-dimension benchmark suite
|   +-- run_benchmarks.py             # MockLLMAdapter overhead benchmarks
|   +-- run_real_benchmarks.py        # Real GPT-4o validation
|   +-- BENCHMARKS.md                 # Full benchmark report
|
+-- migrations/                       # Alembic database migrations
+-- deploy/                           # systemd + Nginx production configs
+-- tests/                            # 57 unit tests
+-- demo.py                           # Full demo without API keys (MockLLMAdapter)
+-- rag_pipeline_example.py           # End-to-end RAG integration example
+-- pyproject.toml
+-- requirements.txt
+-- .env.example
```

---

## Installation

```bash
# Core library
pip install rag-evaluation-framework

# With REST API server (FastAPI, Celery, PostgreSQL, Redis)
pip install rag-evaluation-framework[api]

# Development (testing, linting, building)
pip install rag-evaluation-framework[dev]
```

**Requirements:** Python 3.11+ · sentence-transformers (runs locally) · API key for any supported LLM provider (or MockLLMAdapter for development)

---

## Quick Start

```python
from rag_evaluation_framework import Evaluator

evaluator = Evaluator(llm="openai/gpt-4o")

result = evaluator.score(
    question="What is the capital of France?",
    context=[
        "France is a country in Western Europe.",
        "Its capital is Paris, located on the Seine River.",
    ],
    answer="The capital of France is Paris, located on the Seine River."
)

print(f"Overall:           {result.overall_score:.3f}")
print(f"Faithfulness:      {result.faithfulness.score:.3f}")
print(f"Hallucination:     {result.hallucination_rate.score:.3f}")
print(f"Retrieval:         {result.retrieval_precision.score:.3f}")
print(f"Answer Relevance:  {result.answer_relevance.score:.3f}")
print(f"Context Coverage:  {result.context_coverage.score:.3f}")
print(f"UCM Confidence:    {result.ucm_confidence.score:.3f}")
```

**Run without any API key (demo mode):**

```bash
python demo.py
```

Runs four batch evaluations, generates reports in all formats, and demonstrates the A/B comparison engine. No LLM account needed.

---

## LLM Adapters

| Provider | String Format | Env Variable |
|---|---|---|
| OpenAI | `openai/gpt-4o` | `OPENAI_API_KEY` |
| Anthropic | `anthropic/claude-3-5-sonnet` | `ANTHROPIC_API_KEY` |
| Ollama | `ollama/llama3` | `OLLAMA_BASE_URL` (optional) |
| LiteLLM | `litellm/gpt-4` | Per-provider |

**Custom adapter** - subclass `LLMAdapter` and implement four methods:

```python
from rag_evaluation_framework.adapters.base import LLMAdapter

class MyAdapter(LLMAdapter):
    async def complete(self, prompt, temperature=0.0, **kwargs) -> str:
        ...
    async def embed(self, texts: list[str]) -> list[list[float]]:
        ...
    async def complete_batch(self, prompts, **kwargs) -> list[str]:
        ...

evaluator = Evaluator(llm="openai/gpt-4o")
evaluator._adapter = MyAdapter(model="my-model")
```

All adapters use `sentence-transformers/all-MiniLM-L6-v2` for embedding-based metrics (retrieval precision, UCM semantic consistency). This runs locally. Using local embeddings for semantic similarity and API LLMs only for judge tasks minimizes API cost per evaluation.

---

## Batch Evaluation

```python
items = [
    {"question": "...", "context": [...], "answer": "..."},
    # up to 1000 items
]

results = evaluator.batch_score(items)

avg = sum(r.overall_score for r in results) / len(results)
print(f"Average overall score: {avg:.3f}")
```

All evaluations run concurrently via `asyncio.gather`. Throughput peaks at batch size 25 (~29,000 items/s with MockLLMAdapter). With real LLM APIs, throughput is rate-limit-bound.

---

## Comparing Configurations

```python
result_a = evaluator.score(question=q, context=context_v1, answer=answer_a)
result_b = evaluator.score(question=q, context=context_v2, answer=answer_b)

comparison = evaluator.compare(result_a, result_b)

print(f"Verdict: {comparison.verdict}")
for metric, delta in comparison.score_deltas.items():
    arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
    print(f"  {arrow} {metric:<22} {delta:+.4f}")
```

Use cases: compare chunking strategies, LLM providers, prompt templates, retrieval pipelines.

---

## Report Generation

```python
# Inline (via EvalResult)
result.report(format="json")
result.report(format="markdown")
result.report(format="html")

# Advanced (via ReportBuilder)
from rag_evaluation_framework.report import ReportBuilder

builder = ReportBuilder(result)
builder.to_pdf("report.pdf")                     # Requires WeasyPrint
badge_url = builder.to_ci_badge("overall_score") # Shields.io-compatible URL
```

Report formats: JSON, Markdown, HTML, PDF (WeasyPrint), CI badge URL. All generate in under 0.02ms (framework overhead); report content is static relative to the evaluation result.

---

## CLI Reference

```bash
# Single evaluation
rag-evaluation-framework run \
  --question "What is the capital of France?" \
  --context context.txt \
  --answer "The capital of France is Paris." \
  --llm openai/gpt-4o \
  --output results.json

# Batch evaluation (one JSON object per line)
rag-evaluation-framework batch \
  --file evals.jsonl \
  --llm openai/gpt-4o \
  --output results.jsonl

# Start REST API server
rag-evaluation-framework serve --port 8000 --host 0.0.0.0

# Interactive configuration setup
rag-evaluation-framework init
```

---

## REST API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/evaluate` | Single evaluation |
| `POST` | `/v1/batch` | Submit async batch job |
| `GET` | `/v1/batch/{job_id}` | Poll batch job status |
| `GET` | `/v1/reports/{result_id}` | Fetch stored result by UUID |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger UI |

```bash
curl -X POST http://localhost:8000/v1/evaluate \
  -H "Authorization: Bearer reval_your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the capital of France?",
    "context": ["France is a country in Western Europe.", "Its capital is Paris."],
    "answer": "The capital of France is Paris.",
    "llm": "openai/gpt-4o"
  }'
```

**API dependencies:** FastAPI · Uvicorn · SQLAlchemy (async) · asyncpg · Alembic · Celery · Redis · python-jose · WeasyPrint · Jinja2

```bash
alembic upgrade head  # Run database migrations before starting
```

---

## Dashboard

```bash
cd dashboard && npm install && npm run dev
# Opens at http://localhost:3000
```

| Page | Description |
|---|---|
| Home | Summary stats, recent evaluations, score distribution histograms |
| Evaluate | Form-based evaluation with LLM selector and metric checkboxes |
| Reports | Metric breakdown, hallucination heatmaps, radar charts, score cards |
| Trends | Score trends over time with date range and LLM filtering |

---

## Caching

Cache key: `SHA256(question + sorted(context) + answer + llm_string)`. Identical evaluations return instantly from cache; with real LLM APIs this eliminates 500–3000ms per call.

```python
evaluator = Evaluator(llm="openai/gpt-4o", cache=True)   # in-memory (default)
evaluator = Evaluator(llm="openai/gpt-4o", cache=False)   # disabled
evaluator.clear_cache()
```

For Redis (multi-process, production): set `REDIS_URL` in `.env`. The cache backend switches automatically when `REDIS_URL` is present.

A note on benchmark numbers: the MockLLMAdapter benchmark shows cache hits as slightly *slower* than cache misses (~1.0ms vs ~0.9ms). This is correct behavior - a hash + dict lookup costs ~0.1ms while a mock LLM call costs ~0.001ms. It confirms that the framework overhead is lower than the caching overhead at microsecond scale. With real LLM APIs, the cache is 100–1000× faster.

---

## Configuration Reference

```env
# .env (generate with: rag-evaluation-framework init)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OLLAMA_BASE_URL=http://localhost:11434

# API server
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/rag_eval_db
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=your-secret-key
ALLOWED_ORIGINS=http://localhost:3000
LOG_LEVEL=INFO
```

---

## Benchmarking

```bash
# Framework overhead benchmarks (no API key required)
python -m benchmarks.run_benchmarks

# Real LLM benchmarks (requires OPENAI_API_KEY)
python -m benchmarks.run_real_benchmarks

# All 57 unit tests
pytest tests/ -v
```

The benchmark suite covers 9 dimensions: functional correctness, performance profiling, scalability (batch size, context length, answer length), reliability (n=30), edge-case robustness (10 adversarial inputs), cache effectiveness, reporting throughput, comparison engine accuracy, and utility micro-benchmarks.

One benchmark result worth calling out explicitly: score reliability under the MockLLMAdapter shows CV = 0.00% for 5 of 7 metrics across 30 runs. UCM has CV = 2.25% because its sample generation uses `np.random.randn` embeddings in mock mode. In production with real sentence-transformer embeddings, UCM variance comes from LLM sampling temperature, which is the intended source of variance.

---

## Related Work

- [MedVQA](https://github.com/royxforge/multimodal-medical-vqa) - RAG Evaluation Framework's UCM confidence scoring shares methodology with MedVQA's Monte Carlo Dropout uncertainty estimation. Both estimate confidence by measuring answer consistency across stochastic passes rather than relying on a single forward pass.

- [Production Drift Detection](https://github.com/royxforge/production-drift-detection) - RAG Evaluation Framework evaluates individual interactions; Production Drift Detection monitors population-level trends over time. The entropy and margin signals tracked in ConfidenceMonitor are complementary to UCM: UCM runs at evaluation time, drift detection runs continuously in production.

- [Loss Landscape Analysis](https://github.com/royxforge/loss-landscape-analysis) - The calibration analysis there (BCE vs MSE, label smoothing for better-calibrated outputs) directly informs how LLM judges are prompted in this library. Overconfident judge responses are a known failure mode; the prompts here are designed to elicit calibrated, hedged evaluations rather than binary 0/1 scores.

---

## Citation

```bibtex
@software{roy2026ragevaluationframework,
  author = {Roy, Sourav},
  title  = {RAG Evaluation Framework: The Missing Evaluation Layer for Production RAG Systems},
  year   = {2026},
  url    = {https://github.com/royxforge/rag-evaluation-framework}
}
```

---

<p align="center">
  <sub>Built by <a href="https://github.com/royxforge">Sourav Roy</a> · Artificial Intelligence Engineer · Accure Inc.</sub>
</p>
