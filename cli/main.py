"""RAG Evaluation Framework CLI evaluate, compare, and report from the terminal.

Usage:
    rag-evaluation-framework run --question "..." \
    --context ctx.txt --answer "..." --llm openai/gpt-4o
    rag-evaluation-framework batch --file evals.jsonl --llm openai/gpt-4o --output results.jsonl
    rag-evaluation-framework report --id <uuid> --format html --output report.html
    rag-evaluation-framework compare --id-a <uuid> --id-b <uuid>
    rag-evaluation-framework serve --port 8000
    rag-evaluation-framework init
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click
from rich import box
from rich.console import Console
from rich.table import Table

from rag_evaluation_framework import Evaluator

console = Console()

# Metrics where a higher score means better quality. `Hallucination Rate` is
# the exception: its score is the rate itself (0.0 = clean, 1.0 = every claim
# unsupported), so an *increase* is the regression.
_HIGHER_IS_BETTER = {
    "Overall",
    "Faithfulness",
    "Retrieval Precision",
    "Answer Relevance",
    "Context Coverage",
    "UCM Confidence",
}
_LOWER_IS_BETTER = {"Hallucination Rate"}


def find_regressions(metrics_to_compare, threshold: float):
    """Return entries that got worse by more than ``threshold`` points.

    ``metrics_to_compare`` is a sequence of ``(name, score_a, score_b)`` where
    A is the baseline and B is the candidate. The threshold is expressed on
    the same 0-1 scale as the metrics, so ``0.05`` means "fail if any metric
    moved more than 5 points in the wrong direction".
    """
    regressed = []
    for name, score_a, score_b in metrics_to_compare:
        delta = score_b - score_a
        if name in _LOWER_IS_BETTER:
            worse = delta > threshold
        elif name in _HIGHER_IS_BETTER:
            worse = delta < -threshold
        else:  # unknown metric: assume higher is better
            worse = delta < -threshold
        if worse:
            regressed.append((name, score_a, score_b, delta))
    return regressed


@click.group()
def cli():
    """RAG Evaluation Framework Production-grade RAG evaluation framework."""
    pass


@cli.command()
@click.option("--question", "-q", required=True, help="The input question")
@click.option(
    "--context",
    "-c",
    "context_path",
    required=True,
    help="Path to context file (one chunk per line)",
)
@click.option("--answer", "-a", required=True, help="The generated answer")
@click.option("--llm", "-l", default="openai/gpt-4o", help="Model string (provider/model)")
@click.option("--output", "-o", default=None, help="Output file path (JSON)")
def run(question: str, context_path: str, answer: str, llm: str, output: str | None):
    """Run a single evaluation."""
    # Read context
    context_path = Path(context_path)
    if not context_path.exists():
        console.print(f"[red]Error:[/red] Context file not found: {context_path}")
        sys.exit(1)

    with open(context_path, "r", encoding="utf-8") as f:
        context = [line.strip() for line in f if line.strip()]

    console.print("[bold]Running evaluation...[/bold]")
    console.print(f" LLM: {llm}")
    console.print(f" Question: {question[:50]}...")
    console.print(f" Context chunks: {len(context)}")
    console.print()

    evaluator = Evaluator(llm=llm, cache=True)
    result = evaluator.score(question=question, context=context, answer=answer)

    # Print results table
    table = Table(title="Evaluation Results", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Explanation")

    metrics = [
        ("Overall", f"{result.overall_score:.3f}", "", ""),
        (
            "Faithfulness",
            f"{result.faithfulness.score:.3f}",
            f"{result.faithfulness.confidence:.2f}",
            result.faithfulness.explanation[:60],
        ),
        (
            "Hallucination Rate",
            f"{result.hallucination_rate.score:.3f}",
            f"{result.hallucination_rate.confidence:.2f}",
            result.hallucination_rate.explanation[:60],
        ),
        (
            "Retrieval Precision",
            f"{result.retrieval_precision.score:.3f}",
            f"{result.retrieval_precision.confidence:.2f}",
            result.retrieval_precision.explanation[:60],
        ),
        (
            "Answer Relevance",
            f"{result.answer_relevance.score:.3f}",
            f"{result.answer_relevance.confidence:.2f}",
            result.answer_relevance.explanation[:60],
        ),
        (
            "Context Coverage",
            f"{result.context_coverage.score:.3f}",
            f"{result.context_coverage.confidence:.2f}",
            result.context_coverage.explanation[:60],
        ),
        (
            "UCM Confidence",
            f"{result.ucm_confidence.score:.3f}",
            f"{result.ucm_confidence.confidence:.2f}",
            result.ucm_confidence.explanation[:60],
        ),
    ]

    for name, score, confidence, explanation in metrics:
        table.add_row(name, score, confidence, explanation)

    console.print(table)
    console.print(f"\n[dim]Latency: {result.latency_ms}ms | ID: {result.id[:8]}...[/dim]")

    # Save output
    if output:
        output_path = Path(output)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result.report(format="json"))
        console.print(f"\n[green]✓[/green] Results saved to {output_path}")


@cli.command()
@click.option("--file", "-f", "input_file", required=True, help="Path to JSONL input file")
@click.option("--llm", "-l", default="openai/gpt-4o", help="Model string")
@click.option("--output", "-o", default="results.jsonl", help="Output JSONL file")
def batch(input_file: str, llm: str, output: str):
    """Run batch evaluation from a JSONL file.

    Input format: one JSON per line: {"question": "...", "context": [...], "answer": "..."}
    """
    input_path = Path(input_file)
    if not input_path.exists():
        console.print(f"[red]Error:[/red] Input file not found: {input_file}")
        sys.exit(1)

    # Read items
    items = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    console.print(f"[bold]Batch evaluating {len(items)} items...[/bold]")
    console.print(f" LLM: {llm}")
    console.print()

    evaluator = Evaluator(llm=llm, cache=True)

    with click.progressbar(items, label="Evaluating") as bar:
        results = evaluator.batch_score(bar)

    # Write results
    output_path = Path(output)
    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(result.model_dump_json() + "\n")

    console.print(f"\n[green]✓[/green] {len(results)} results saved to {output_path}")

    # Summary stats
    avg_overall = sum(r.overall_score for r in results) / len(results)
    console.print(f" Average overall score: {avg_overall:.3f}")


@cli.command()
@click.option("--id", "result_id", required=True, help="Result UUID")
@click.option(
    "--format", "-f", "format_type", default="html", help="Output format: json, html, markdown"
)
@click.option("--output", "-o", required=True, help="Output file path")
def report(result_id: str, format_type: str, output: str):
    """Generate a report for a stored evaluation result."""
    import httpx

    api_url = os.environ.get("RAG_EVALUATION_FRAMEWORK_API_URL", "http://localhost:8000")
    api_key = os.environ.get("RAG_EVALUATION_FRAMEWORK_API_KEY", "")

    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        response = httpx.get(
            f"{api_url}/v1/reports/{result_id}?format={format_type}",
            headers=headers,
            timeout=30.0,
        )
        response.raise_for_status()

        output_path = Path(output)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        console.print(f"[green]✓[/green] Report saved to {output_path}")
    except Exception as e:
        console.print(f"[red]Error fetching report:[/red] {e}")
        console.print("Make sure the API is running and the result ID is valid.")
        sys.exit(1)


@cli.command()
@click.option("--id-a", required=True, help="First result UUID")
@click.option("--id-b", required=True, help="Second result UUID")
@click.option(
    "--fail-on-regression",
    type=float,
    default=None,
    help="Exit 1 if any metric worsens by more than this many points (0-1 scale).",
)
def compare(id_a: str, id_b: str, fail_on_regression: float | None):
    """Compare two evaluation results side-by-side (A is the baseline)."""
    import httpx

    api_url = os.environ.get("RAG_EVALUATION_FRAMEWORK_API_URL", "http://localhost:8000")
    api_key = os.environ.get("RAG_EVALUATION_FRAMEWORK_API_KEY", "")

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    try:
        r1 = httpx.get(f"{api_url}/v1/reports/{id_a}?format=json", headers=headers, timeout=30.0)
        r2 = httpx.get(f"{api_url}/v1/reports/{id_b}?format=json", headers=headers, timeout=30.0)
        r1.raise_for_status()
        r2.raise_for_status()

        data_a = r1.json()
        data_b = r2.json()

        # Build comparison
        table = Table(title="Comparison", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Result A", justify="right")
        table.add_column("Result B", justify="right")
        table.add_column("Delta", justify="right")

        metrics_to_compare = [
            ("Overall", data_a.get("overall_score", 0), data_b.get("overall_score", 0)),
            (
                "Faithfulness",
                data_a.get("faithfulness", {}).get("score", 0),
                data_b.get("faithfulness", {}).get("score", 0),
            ),
            (
                "Hallucination Rate",
                data_a.get("hallucination_rate", {}).get("score", 0),
                data_b.get("hallucination_rate", {}).get("score", 0),
            ),
            (
                "Retrieval Precision",
                data_a.get("retrieval_precision", {}).get("score", 0),
                data_b.get("retrieval_precision", {}).get("score", 0),
            ),
            (
                "Answer Relevance",
                data_a.get("answer_relevance", {}).get("score", 0),
                data_b.get("answer_relevance", {}).get("score", 0),
            ),
            (
                "Context Coverage",
                data_a.get("context_coverage", {}).get("score", 0),
                data_b.get("context_coverage", {}).get("score", 0),
            ),
            (
                "UCM Confidence",
                data_a.get("ucm_confidence", {}).get("score", 0),
                data_b.get("ucm_confidence", {}).get("score", 0),
            ),
        ]

        for name, score_a, score_b in metrics_to_compare:
            delta = score_b - score_a
            delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
            table.add_row(name, f"{score_a:.3f}", f"{score_b:.3f}", delta_str)

        console.print(table)

        if fail_on_regression is not None:
            regressed = find_regressions(metrics_to_compare, fail_on_regression)
            if regressed:
                for name, score_a, score_b, delta in regressed:
                    console.print(
                        f"[red]REGRESSION[/red] {name}: "
                        f"{score_a:.3f} -> {score_b:.3f} ({delta:+.3f})"
                    )
                console.print(
                    f"[red]{len(regressed)} metric(s) regressed by more than "
                    f"{fail_on_regression:.3f}[/red]"
                )
                sys.exit(1)
            console.print(
                f"[green]OK[/green] no metric regressed by more than "
                f"{fail_on_regression:.3f}"
            )
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@cli.command()
@click.option("--port", "-p", default=8000, help="Port to serve on")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
def serve(port: int, host: str):
    """Start the RAG Evaluation Framework API server directly."""
    # Lazy import to avoid requiring API dependencies for core CLI commands
    try:
        import uvicorn  # noqa: F401
    except ImportError:
        console.print("[red]Error:[/red] API dependencies not installed.")
        console.print("Install them with: [bold]pip install rag-evaluation-framework[api][/bold]")
        sys.exit(1)

    console.print("[bold]Starting RAG Evaluation Framework API...[/bold]")
    console.print(f" http://{host}:{port}")
    console.print(f" Docs: http://{host}:{port}/docs")
    console.print()

    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )


@cli.command()
def init():
    """Create .rag-evaluation-framework.env configuration file."""
    env_path = Path(".rag-evaluation-framework.env")
    if env_path.exists():
        console.print(
            "[yellow]Warning:[/yellow] .rag-evaluation-framework.env"
            " already exists. Overwrite? [y/N]"
        )
        response = input().strip().lower()
        if response != "y":
            console.print("Aborted.")
            return

    console.print("[bold]Setting up RAG Evaluation Framework configuration...[/bold]")
    console.print()

    openai_key = click.prompt("OpenAI API Key", default="", show_default=False)
    anthropic_key = click.prompt("Anthropic API Key", default="", show_default=False)
    ollama_url = click.prompt("Ollama Base URL", default="http://localhost:11434")
    db_url = click.prompt(
        "Database URL",
        default="postgresql+asyncpg://rag_evaluation_framework_user:rag_evaluation_framework_pass@localhost/rag_evaluation_framework_db",
    )
    redis_url = click.prompt("Redis URL", default="redis://localhost:6379/0")
    secret_key = click.prompt("Secret Key", default=os.urandom(32).hex())

    env_content = f"""# RAG Evaluation Framework Configuration
# Generated by `rag-evaluation-framework init`

DATABASE_URL={db_url}
REDIS_URL={redis_url}
SECRET_KEY={secret_key}
OPENAI_API_KEY={openai_key}
ANTHROPIC_API_KEY={anthropic_key}
OLLAMA_BASE_URL={ollama_url}
ALLOWED_ORIGINS=http://localhost:3000
DEBUG=false
LOG_LEVEL=INFO
RAG_EVALUATION_FRAMEWORK_API_URL=http://localhost:8000
"""

    with open(env_path, "w", encoding="utf-8") as f:
        f.write(env_content)

    console.print(f"\n[green]✓[/green] Configuration saved to {env_path}")
    console.print(
        "Run [bold]source .rag-evaluation-framework.env[/bold] to load it,"
        " or copy it to /opt/rag-evaluation-framework/.env"
    )


if __name__ == "__main__":
    cli()
