# Contributing to RAG Evaluation Framework

Thank you for your interest in contributing to RAG Evaluation Framework!
This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Commit Message Conventions](#commit-message-conventions)
- [Issue Reporting](#issue-reporting)
- [Feature Requests](#feature-requests)

## Code of Conduct

This project and everyone participating in it is governed by our
[Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to
uphold this code. Please report unacceptable behavior to royxforge@gmail.com.

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```
   git clone https://github.com/your-username/rag-evaluation-framework.git
   cd rag-evaluation-framework
   ```
3. Add the upstream repository:
   ```
   git remote add upstream https://github.com/royxforge/rag-evaluation-framework.git
   ```

## Development Setup

### Prerequisites

- Python 3.11+
- pip

### Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### API Keys (Required for LLM Evaluation)

Copy `.env.example` to `.env` and add your API keys:

```env
OPENAI_API_KEY=sk-...
```

### Verify Installation

```bash
python -c "from rag_evaluation_framework.evaluator import RAGEvaluator; print('Setup OK')"
```

### Database Setup (for REST API)

```bash
alembic upgrade head
```

## Coding Standards

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guide.
- Use type annotations for all function signatures.
- Maximum line length: 88 characters.
- Document all public classes and functions with Google-style docstrings.

### Imports

Organize imports in the following order:

1. Standard library imports
2. Third-party imports
3. Local application imports

## Testing

```bash
pytest tests/ -v
pytest tests/conftest.py -v
```

### Benchmark Suite

```bash
python -m benchmarks.run_benchmarks
```

Ensure no regressions in evaluation accuracy or latency.

## Pull Request Process

1. Create a new branch from `main`:
   ```
   git checkout -b feature/your-feature-name
   ```

2. Make your changes with clear, descriptive commit messages.

3. Run tests:
   ```bash
   pytest tests/ -v
   ```

4. If adding a new metric, ensure it extends the `BaseMetric` class and
   returns a `MetricScore` with `score`, `explanation`, `confidence`, and `details`.

5. Push your branch and open a Pull Request on GitHub.

6. In your PR description, include:
   - What the change does
   - Any relevant issue numbers
   - How you tested the change
   - Evaluation results if applicable

7. Request review from a maintainer.

## Commit Message Conventions

We follow conventional commit format:

```
<type>(<scope>): <description>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `benchmark`

Examples:
```
feat(metrics): add context coverage metric with cosine similarity
fix(api): correct batch evaluation memory leak
docs(readme): update UCM interpretation table
```

## Issue Reporting

### Bug Reports

When filing a bug report, please include:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior and actual behavior
- Environment details (OS, Python version, LLM provider)
- Relevant logs or error messages

### Feature Requests

We welcome feature suggestions! Please include:

- A clear description of the proposed feature
- The motivation or use case
- Any relevant research or references
- Whether you are willing to implement it

### New LLM Adapter

To add support for a new LLM provider, implement the `BaseLLMAdapter`
abstract base class with the following methods:

- `generate(prompt: str) -> str`
- `generate_batch(prompts: list[str]) -> list[str]`
- `get_confidence(response: str) -> float`

Then submit a PR with the new adapter and corresponding tests.

Thank you for helping make RAG Evaluation Framework better!
