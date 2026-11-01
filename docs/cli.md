# CLI

The `rag-evaluation-framework` command is installed with the package.

## Single evaluation

```bash
rag-evaluation-framework run \
  --question "What is the capital of France?" \
  --context context.txt \
  --answer "The capital of France is Paris." \
  --llm openai/gpt-4o \
  --output results.json
```

## Batch evaluation

One JSON object per line in, one result per line out.

```bash
rag-evaluation-framework batch \
  --file evals.jsonl \
  --llm openai/gpt-4o \
  --output results.jsonl
```

## Compare two runs

Exits non-zero when the candidate has regressed beyond the tolerance, which makes it usable
as a CI gate.

```bash
rag-evaluation-framework compare \
  --id-a <baseline-uuid> --id-b <candidate-uuid> \
  --fail-on-regression 0.05
```

## Serve the REST API

```bash
rag-evaluation-framework serve --port 8000 --host 0.0.0.0
```

Setting `SECRET_KEY` is required: it signs and verifies the JWTs the API issues.

## Interactive setup

```bash
rag-evaluation-framework init
```
