# Metrics

Each metric returns a `MetricScore` with four fields: `score` (0–1), `explanation`,
`confidence` (0–1) and `details` (metric-specific breakdown). A metric that cannot be
computed returns `score=None` and is left out of the aggregate.

## Faithfulness

Decomposes the answer into atomic claims, then verifies each claim against the retrieved
context. Score = supported claims / total claims. Faithfulness measures support *from the
context*, not real-world accuracy: a faithful answer to a false context still scores 1.0,
which is the intended behaviour — it is a pipeline-integrity signal.

## Hallucination rate

Separates context hallucination (contradicts the provided context) from factual
hallucination (not verifiable from it). Score = unsupported claims / total claims. Higher
is worse, so the overall score uses `1.0 - hallucination_rate`.

## Retrieval precision

Embedding-based and LLM-free: cosine similarity between the question and each chunk, using
`all-MiniLM-L6-v2`. Score = fraction of chunks above the relevance threshold (default
`0.3`); MRR is reported in `details`.

## Answer relevance

LLM judge decides whether the answer addresses the question asked. Independent of
faithfulness — an answer can be grounded in the context and still miss the question.

## Context coverage

LLM judge estimates how much of the relevant context is reflected in the answer. Measures
completeness, not correctness.

## UCM (unsupervised confidence metric)

A confidence score computed without reference answers, from agreement across sampled
judgements and the spread of the metric scores. Useful where labels do not exist.
