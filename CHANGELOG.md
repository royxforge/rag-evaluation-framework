# Changelog

All notable changes to the RAG Evaluation Framework are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Security

- **`SECRET_KEY` is required**: importing `api.auth` without it raises `RuntimeError`. Previously a per-process random default silently invalidated every issued JWT on restart and could not be shared across API/worker processes. Generate one with `openssl rand -hex 32` (deployment `setup.sh` already writes it to `.env`).
- **Indexed API-key authentication**: new `api_keys.key_prefix` column (first 12 characters of the raw key) plus Alembic migration `002_add_api_key_prefix` give an indexed lookup and a single bcrypt verification per request. Previously every active key row was bcrypt-verified on every unauthenticated request — an O(N) CPU amplifier. **Breaking: keys created before migration 002 have `key_prefix NULL` and will fail authentication; rotate them (`POST /v1/keys`) after upgrading.**

### Fixed

- **Rate limiting** reuses a process-wide Redis client (previously created and abandoned one connection per request, leaking sockets under load) and re-raises HTTP 429 instead of swallowing it into the in-memory fallback.
- **Cache keys include the resolved metric set** (plus scoring options): a single-metric result can no longer satisfy an "all metrics" lookup (or vice versa).
- **`overall_score` averages only the metrics that were computed** — a perfect single-metric run no longer divides by 6 (yielding ~0.16). `compare()` inverts `hallucination_rate` deltas, so a positive delta always means "B improved".
- **Faithfulness verdict parsing** requires an explicit leading `SUPPORTED` and rejects `NOT_SUPPORTED` (the previous substring check matched both, since `NOT_SUPPORTED` contains `SUPPORTED`).
- **`sentence-transformers` is imported lazily** (LLM-only evaluations no longer pull the torch stack at import time); `embed()` runs in an executor; `complete_batch` bounds concurrency with a semaphore; retries apply only to transient failures (429/5xx/timeout/connection) — authentication errors now fail fast instead of being retried with backoff.

---

## [0.3.0] - 2026-07-20

### Changed

- **Git remote URL**: Updated from `royxlead/rag-evaluation-framework.git` to `royxforge/rag-evaluation-framework.git` to reflect the permanent repository home under the royxforge GitHub organization.
- **Project metadata URLs**: Updated `Homepage`, `Documentation`, and `Repository` URLs in `pyproject.toml` from `royxlead` to `royxforge`.

---

## [Unreleased]

### Changed

- Revised README.md with expanded documentation including evaluation results from real LLM validation against GPT-4o, per-scenario breakdown tables for five VQA-style scenarios, batch performance benchmarks for four factual question-answer pairs, framework overhead benchmarks across nine dimensions, and detailed interpretation guidance for the UCM confidence metric.
- Updated README.md to add links to related projects (MedVQA, Production Drift Detection, Loss Landscape Analysis) and a BibTeX citation entry.
- Refined README.md formatting and content organization to improve readability across the table of contents, installation instructions, API reference, dashboard setup, and configuration documentation.

---

## [0.2.0] - 2026-07-20

### Added

- **Community health files**: Added `CODE_OF_CONDUCT.md` (Contributor Covenant v2.1), `CONTRIBUTING.md` (contribution guidelines), `SECURITY.md` (vulnerability reporting policy), and `CITATION.cff` (citation metadata). These files establish project governance, community participation guidelines, and academic attribution framework.

---

## [0.1.0] - 2026-06-20

### Added

#### Core Evaluation Engine

- Evaluator core with parallel metric execution via `asyncio.gather`. The evaluator runs six metrics concurrently such that total evaluation time equals the slowest single metric rather than the sum.
- Six evaluation metrics computed for every evaluation item:
  - **Faithfulness** decomposes the answer into atomic factual claims and verifies each claim against the provided context. Score equals the ratio of supported claims to total claims. The metric measures RAG pipeline integrity, not real-world factual accuracy. Claim-level breakdowns are available in the metric details.
  - **Hallucination Rate** distinguishes context hallucination (contradicts provided context) from factual hallucination (claims not verifiable from context). Score equals the ratio of unsupported claims to total claims. The overall score inverts this value, so lower hallucination yields a better overall result.
  - **Retrieval Precision** uses embedding-based cosine similarity via `sentence-transformers/all-MiniLM-L6-v2` running locally. It computes the fraction of context chunks above a relevance threshold of 0.3 and also calculates Mean Reciprocal Rank. This metric requires no LLM API call and completes in approximately one millisecond per chunk.
  - **Answer Relevance** uses an LLM judge to determine whether the answer addresses the user's original question, which is orthogonal to faithfulness.
  - **Context Coverage** uses an LLM judge to estimate what fraction of the relevant context information appears in the answer, measuring completeness rather than correctness.
  - **UCM Confidence** estimates answer confidence without ground truth labels through internal consistency analysis across multiple stochastic forward passes at temperature 0.7. It combines semantic consistency (sentence-transformer embeddings, weight 0.4), factual overlap (Jaccard similarity on atomic claims, weight 0.4), and lexical consistency (BLEU and ROUGE-L, weight 0.2). Scores range from 0.0 to 1.0, with higher values indicating greater model certainty.
- UCM adjudication thresholds with three tiers: high consistency (0.8 to 1.0, safe to deploy), moderate consistency (0.5 to 0.8, review edge cases), and low consistency (0.0 to 0.5, investigate root causes including ambiguous questions, insufficient context, or inappropriate model selection).
- Overall score computed as the arithmetic mean of faithfulness, inverted hallucination rate, retrieval precision, answer relevance, context coverage, and UCM confidence.

#### Data Models

- `MetricScore` data model via Pydantic v2 with four fields: `score` (float in [0, 1]), `explanation` (string), `confidence` (float in [0, 1]), and `details` (dictionary with metric-specific breakdown).
- `EvalResult` data model with unique identifier, timestamp, per-metric scores, overall score, and latency tracking.

#### LLM Adapter Layer

- Abstract base class `LLMAdapter` in `rag_evaluation_framework/adapters/base.py` with four abstract methods: `complete`, `embed`, `complete_batch`, and `get_model_name`.
- **OpenAI adapter** supporting GPT-4o, GPT-4 Turbo, GPT-4, and GPT-3.5 Turbo via the `openai/gpt-4o` string format. Configurable via the `OPENAI_API_KEY` environment variable.
- **Anthropic adapter** supporting Claude 3.5 Sonnet and Claude 3 Opus via the `anthropic/claude-3-5-sonnet` string format. Configurable via the `ANTHROPIC_API_KEY` environment variable.
- **Ollama adapter** for local models with optional `OLLAMA_BASE_URL` configuration.
- **LiteLLM adapter** providing access to over 100 providers through a unified interface.
- `MockLLMAdapter` for development and testing without any API key. Returns deterministic mock responses for all metrics.
- Local embedding support via `sentence-transformers/all-MiniLM-L6-v2` for retrieval precision and UCM semantic consistency. All embedding computation runs locally to minimize API cost.

#### Evaluation Prompts

- Full set of LLM judge prompts in `rag_evaluation_framework/prompts.py` designed to elicit calibrated, hedged evaluations rather than binary zero-or-one scores. Prompts address overconfidence as a known failure mode in LLM-as-judge evaluations.

#### Report Generation

- `ReportBuilder` class supporting five output formats: JSON, Markdown, HTML, PDF, and Shields.io badge URL.
- JSON and Markdown reports accessible directly from `EvalResult.report()`.
- PDF generation via WeasyPrint.
- CI badge URLs for embedding overall or per-metric scores in dashboards and repository README files.
- All report formats generate in under 0.02 milliseconds of framework overhead.

#### Caching

- In-memory caching with SHA-256 key derived from the concatenation of question, sorted context, answer, and LLM identifier.
- Automatic Redis backend switch when the `REDIS_URL` environment variable is present, enabling multi-process and production caching.
- Cache control via the `Evaluator` constructor (`cache=True` or `cache=False`) and a `clear_cache()` method.
- Cache hit eliminates all LLM calls entirely, returning results in under one millisecond of framework overhead.

#### A/B Comparison Engine

- `Evaluator.compare()` method computing per-metric deltas between two evaluation results with a human-readable verdict.
- Comparison use cases include chunking strategies, retrieval pipelines, LLM providers, and prompt templates.

#### Batch Evaluation

- `Evaluator.batch_score()` method supporting up to 1000 items with concurrent execution via `asyncio.gather`.
- Peak throughput of approximately 29,240 items per second at batch size 25 with the MockLLMAdapter.

#### REST API

- FastAPI application in `api/main.py` with the following endpoints:
  - `POST /v1/evaluate` for single evaluation requests.
  - `POST /v1/batch` for submitting asynchronous batch jobs.
  - `GET /v1/batch/{job_id}` for polling batch job status.
  - `GET /v1/reports/{result_id}` for retrieving stored results by UUID.
  - `GET /health` for health checks.
  - `GET /docs` for Swagger UI documentation.
- API key authentication with Bearer token format.
- Rate limiting per API key.
- Celery-based background task processing for batch jobs.
- PostgreSQL persistence via SQLAlchemy with Alembic migrations.
- CORS middleware for frontend integration.

#### CLI

- Click-based command line interface with four commands:
  - `run` for single evaluation with question, context file, answer, LLM selector, and output file options.
  - `batch` for batch evaluation from a JSONL file.
  - `serve` for starting the REST API server with configurable host and port.
  - `init` for interactive configuration setup.
- Rich-formatted terminal output.

#### Dashboard

- Next.js 14 web dashboard with TypeScript and Tailwind CSS:
  - Home page with summary statistics, recent evaluations, and score distribution histograms.
  - Evaluate page with form-based evaluation input, LLM selector, and metric checkboxes.
  - Reports page with metric breakdown, hallucination heatmaps, radar charts, and score cards.
  - Trends page with score trends over time, date range filtering, and LLM filtering.
- Client-side API integration module at `dashboard/lib/api.ts`.

#### Benchmarking Suite

- Nine-dimension benchmark suite in `benchmarks/run_benchmarks.py` using MockLLMAdapter, covering functional correctness, performance profiling, scalability (batch size, context length, answer length), reliability across 30 runs, edge-case robustness across 10 adversarial inputs, cache effectiveness, reporting throughput, comparison engine accuracy, and utility micro-benchmarks.
- Real LLM benchmark suite in `benchmarks/run_real_benchmarks.py` for validating against GPT-4o across five VQA-style scenarios.

#### Testing

- 57 unit tests across four test modules:
  - `tests/unit/test_adapters.py` verifying all LLM adapter implementations.
  - `tests/unit/test_evaluator.py` verifying the evaluator core and batch processing.
  - `tests/unit/test_metrics.py` verifying all six metric implementations.
  - `tests/unit/test_report.py` verifying report generation in all formats.
- Integration tests in `tests/integration/test_api.py` covering the REST API.
- Mock fixtures at `tests/fixtures/mock_llm_responses.json` and `tests/fixtures/sample_evals.jsonl`.
- Pytest configuration with async mode support.

#### Deployment

- Nginx reverse proxy configuration at `deploy/nginx.conf`.
- systemd service definitions for the API server and Celery worker at `deploy/rag-evaluation-framework-api.service` and `deploy/rag-evaluation-framework-worker.service`.
- Automated setup script at `deploy/setup.sh`.

#### Development Infrastructure

- Project metadata in `pyproject.toml` with Python 3.11+ requirement, Apache 2.0 license, and optional dependency groups for API and development.
- Ruff linter configuration with target Python 3.11 and 100-character line length.
- MyPy type checking configuration.
- Build system configured with setuptools.
- Demonstrable reference implementation at `demo.py` for running a complete evaluation cycle without any API key.
- End-to-end RAG pipeline integration example at `rag_pipeline_example.py`.

---

[Unreleased]: https://github.com/royxforge/rag-evaluation-framework/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/royxforge/rag-evaluation-framework/releases/tag/v0.2.0
[0.1.0]: https://github.com/royxforge/rag-evaluation-framework/releases/tag/v0.1.0
