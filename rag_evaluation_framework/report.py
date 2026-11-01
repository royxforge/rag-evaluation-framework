"""Report builder generates reports in multiple formats."""

from __future__ import annotations

from typing import Any

from rag_evaluation_framework.models import EvalResult


class ReportBuilder:
    """Builds formatted reports from evaluation results."""

    def __init__(self, result: EvalResult):
        self.result = result

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a dict."""
        return self.result.model_dump()

    def to_json(self, indent: int = 2) -> str:
        """Return the result as a JSON string."""
        return self.result.model_dump_json(indent=indent)

    def to_markdown(self) -> str:
        """Return the result as a Markdown string."""
        return self.result.report(format="markdown")

    def to_html(self) -> str:
        """Return the result as an HTML string."""
        return self.result.report(format="html")

    def to_pdf(self, output_path: str) -> str:
        """Generate a PDF report using WeasyPrint.

        Args:
            output_path: Path to save the PDF file.

        Returns:
            The output path.
        """
        html_content = self.to_html()
        try:
            from weasyprint import HTML

            HTML(string=html_content).write_pdf(output_path)
        except (ImportError, OSError):
            # Fallback: save as HTML when WeasyPrint or its system deps are missing
            html_path = output_path.replace(".pdf", ".html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            output_path = html_path
        return output_path

    def to_ci_badge(self, metric: str = "overall_score") -> str:
        """Generate a Shields.io-style badge URL for CI pipelines.

        Args:
            metric: Which metric to display ('overall_score', 'faithfulness', etc.)

        Returns:
            Badge URL string.
        """
        score: float | None
        if metric == "overall_score":
            score = self.result.overall_score
            label = "overall"
        else:
            score_obj = getattr(self.result, metric, None)
            if score_obj is None:
                return ""
            score = score_obj.score if hasattr(score_obj, "score") else None
            if score is None:
                return ""
            label = metric.replace("_", " ")

        color = "green" if score >= 0.8 else "orange" if score >= 0.6 else "red"
        pct = int(score * 100)
        return f"https://img.shields.io/badge/{label}-{pct}%25-{color}"

    def summary_stats(self) -> dict[str, Any]:
        """Return summary statistics for this result."""
        metrics = {
            "faithfulness": self.result.faithfulness.score,
            "hallucination_rate": self.result.hallucination_rate.score,
            "retrieval_precision": self.result.retrieval_precision.score,
            "answer_relevance": self.result.answer_relevance.score,
            "context_coverage": self.result.context_coverage.score,
            "ucm_confidence": self.result.ucm_confidence.score,
        }
        return {
            "id": self.result.id,
            "timestamp": self.result.timestamp.isoformat(),
            "overall_score": self.result.overall_score,
            "metrics": metrics,
            "latency_ms": self.result.latency_ms,
            "llm": self.result.metadata.get("llm", "unknown"),
        }


def build_report(result: EvalResult, format: str = "dict", **kwargs: Any) -> dict | str:
    """Convenience function to build a report.

    Args:
        result: The evaluation result.
        format: Output format: 'dict', 'json', 'html', 'markdown', 'pdf'.
        **kwargs: Additional arguments passed to the ReportBuilder.

    Returns:
        Formatted report.
    """
    builder = ReportBuilder(result)

    if format == "dict":
        return builder.to_dict()
    elif format == "json":
        return builder.to_json(**kwargs)
    elif format == "html":
        return builder.to_html()
    elif format == "markdown":
        return builder.to_markdown()
    elif format == "pdf":
        output_path = kwargs.get("output_path", f"report_{result.id[:8]}.pdf")
        return builder.to_pdf(output_path)
    else:
        raise ValueError(f"Unknown format: {format}")
