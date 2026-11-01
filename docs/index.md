# RAG Evaluation Framework

Six metrics for a retrieval-augmented generation pipeline — **faithfulness, hallucination
rate, retrieval precision, answer relevance, context coverage** and **UCM**, a confidence
score that needs no gold-standard labels — returned by a single evaluator call.

```bash
pip install -e ".[dev]"
python -m benchmarks.run_benchmarks   # CPU only, no API key
```

The full README on GitHub is the primary documentation; these pages cover the parts that
are easiest to read out of order.

- [Metrics](metrics.md) — what each score means and how it is computed
- [CLI](cli.md) — the `rag-eval` commands

## Design notes

- Adapters for OpenAI, Anthropic, Ollama and LiteLLM, behind one `LLMAdapter` interface.
- SHA-256 caching keyed on question, context, answer, model and metric set, so repeat runs
  cost nothing.
- A metric that fails returns `score=None`, never a fabricated `0.0`, and is excluded from
  the aggregate instead of silently pulling it down.
- Every LLM-as-judge score depends on the judge model. Swap the adapter and the numbers move.
