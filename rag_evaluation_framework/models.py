"""Pydantic v2 models for RAG Evaluation Framework."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal, Optional, overload

from pydantic import BaseModel, Field


class MetricScore(BaseModel):
    """Score for a single evaluation metric.

    ``score`` is ``None`` when the metric could not be computed (for example a
    provider outage). A failed metric must never be reported as a plausible
    numeric value, because such a value is averaged into ``overall_score`` and
    silently corrupts the aggregate.
    """

    score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Score between 0.0 and 1.0, or None if it failed"
    )
    explanation: str = Field(default="", description="Human-readable explanation of the score")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence in this metric score"
    )
    details: dict[str, Any] = Field(default_factory=dict, description="Metric-specific breakdown")


class EvalResult(BaseModel):
    """Complete evaluation result for a single RAG query."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    question: str
    context: list[str]
    answer: str
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    faithfulness: MetricScore = Field(default_factory=MetricScore)
    hallucination_rate: MetricScore = Field(default_factory=MetricScore)
    retrieval_precision: MetricScore = Field(default_factory=MetricScore)
    answer_relevance: MetricScore = Field(default_factory=MetricScore)
    context_coverage: MetricScore = Field(default_factory=MetricScore)
    ucm_confidence: MetricScore = Field(default_factory=MetricScore)
    metadata: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int = Field(default=0)

    @overload
    def report(self, format: Literal["dict"] = "dict") -> dict: ...

    @overload
    def report(self, format: Literal["json", "html", "markdown"]) -> str: ...

    @overload
    def report(self, format: str) -> dict | str: ...

    def report(self, format: str = "dict") -> dict | str:
        """Generate a formatted report.

        Args:
            format: One of "dict", "json", "html", "markdown".

        Returns:
            Formatted report as dict or string.
        """
        if format == "dict":
            return self.model_dump()
        elif format == "json":
            return self.model_dump_json(indent=2)
        elif format == "markdown":
            lines = [
                f"# RAG Evaluation Framework Report: `{self.id[:8]}...`",
                "",
                f"- **Question:** {self.question}",
                f"- **Answer:** {self.answer[:100]}{'...' if len(self.answer) > 100 else ''}",
                f"- **Overall Score:** {self.overall_score:.3f}",
                f"- **Timestamp:** {self.timestamp.isoformat()}",
                f"- **Latency:** {self.latency_ms}ms",
                "",
                "## Metrics",
                "",
                "| Metric | Score | Confidence |",
                "|--------|-------|------------|",
            ]
            # Build metric rows (keep lines under 100 chars)
            for label, metric in [
                ("Faithfulness", self.faithfulness),
                ("Hallucination Rate (inv.)", self.hallucination_rate),
                ("Retrieval Precision", self.retrieval_precision),
                ("Answer Relevance", self.answer_relevance),
                ("Context Coverage", self.context_coverage),
                ("UCM Confidence", self.ucm_confidence),
            ]:
                if metric.score is None:
                    formatted = "n/a"
                elif "Hallucination" in label:
                    formatted = f"{1 - metric.score:.3f}"
                else:
                    formatted = f"{metric.score:.3f}"
                lines.append(f"| {label} | {formatted} | {metric.confidence:.3f} |")

            lines.extend([
                "",
                "## Explanations",
                "",
                "### Faithfulness",
                f"{self.faithfulness.explanation}",
                "",
                "### Hallucination Rate",
                f"{self.hallucination_rate.explanation}",
                "",
                "### Retrieval Precision",
                f"{self.retrieval_precision.explanation}",
                "",
                "### Answer Relevance",
                f"{self.answer_relevance.explanation}",
                "",
                "### Context Coverage",
                f"{self.context_coverage.explanation}",
                "",
                "### UCM Confidence",
                f"{self.ucm_confidence.explanation}",
            ])
            return "\n".join(lines)
        elif format == "html":
            score_color = (
                "green"
                if self.overall_score >= 0.8
                else "orange"
                if self.overall_score >= 0.6
                else "red"
            )
            lines = [
                "<!DOCTYPE html>",
                '<html><head><meta charset="utf-8">'
                "<title>RAG Evaluation Framework Report</title>",
                "<style>",
                ("body{font-family:-apple-system,BlinkMacSystemFont,"
                 "'Segoe UI',sans-serif;max-width:800px;"
                 "margin:40px auto;padding:0 20px;"
                 "color:#333;line-height:1.6}"),
                ("h1{color:#1a1a2e}"
                 "h2{color:#16213e;border-bottom:2px solid #eaeaea;"
                 "padding-bottom:8px}"),
                (".score{font-size:2em;font-weight:700;padding:16px;"
                 "border-radius:8px;text-align:center;margin:20px 0}"),
                "table{width:100%;border-collapse:collapse;margin:16px 0}",
                "th,td{padding:12px;text-align:left;"
                "border-bottom:1px solid #ddd}",
                "th{background:#f5f5f5;font-weight:600}",
                ".metric-detail{background:#f9f9f9;padding:16px;"
                "border-radius:8px;margin:12px 0}",
                (".badge{display:inline-block;padding:2px 8px;"
                 "border-radius:4px;font-size:0.8em;font-weight:600}"),
                ".badge-green{background:#d4edda;color:#155724}",
                ".badge-red{background:#f8d7da;color:#721c24}",
                ".badge-amber{background:#fff3cd;color:#856404}",
                "</style></head><body>",
                "<h1>RAG Evaluation Framework Report</h1>",
                f"<div class='score' "
                f"style='background:{score_color}20;"
                f"color:{score_color}'>",
                f"Overall Score: {self.overall_score:.3f}</div>",
                f"<p><strong>Question:</strong> {self.question}</p>",
                f"<p><strong>Answer:</strong> {self.answer}</p>",
                f"<p><strong>Latency:</strong> {self.latency_ms}ms</p>",
                "<h2>Metrics</h2>",
                "<table><tr><th>Metric</th><th>Score</th>"
                "<th>Confidence</th><th>Explanation</th></tr>",
            ]
            for name, metric in [
                ("Faithfulness", self.faithfulness),
                ("Hallucination Rate", self.hallucination_rate),
                ("Retrieval Precision", self.retrieval_precision),
                ("Answer Relevance", self.answer_relevance),
                ("Context Coverage", self.context_coverage),
                ("UCM Confidence", self.ucm_confidence),
            ]:
                if metric.score is None:
                    badge_class = "badge-amber"
                    score_text = "n/a"
                else:
                    badge_class = (
                        "badge-green"
                        if metric.score >= 0.8
                        else "badge-amber"
                        if metric.score >= 0.6
                        else "badge-red"
                    )
                    score_text = f"{metric.score:.3f}"
                expl = metric.explanation[:80]
                if len(metric.explanation) > 80:
                    expl += "..."
                lines.append(
                    f"<tr><td>{name}</td>"
                    f"<td><span class='badge {badge_class}'>"
                    f"{score_text}</span></td>"
                    f"<td>{metric.confidence:.2f}</td>"
                    f"<td>{expl}</td></tr>"
                )
            lines.append("</table>")
            lines.append("</body></html>")
            return "\n".join(lines)
        else:
            return self.model_dump()


class EvalRequest(BaseModel):
    """Request payload for a single evaluation."""

    question: str
    context: list[str]
    answer: str
    metrics: list[str] = Field(default_factory=lambda: ["all"])
    llm: str = "openai/gpt-4o"
    options: dict[str, Any] = Field(default_factory=dict)


class BatchEvalRequest(BaseModel):
    """Request payload for batch evaluation."""

    items: list[dict] = Field(..., min_length=1, max_length=1000)
    llm: str = "openai/gpt-4o"
    webhook_url: Optional[str] = None


class BatchJobStatus(BaseModel):
    """Status of a batch evaluation job."""

    job_id: str
    status: str  # queued, running, completed, failed
    total_items: int
    completed_items: int = 0
    result_ids: list[str] = Field(default_factory=list)
    error: Optional[str] = None


class ComparisonReport(BaseModel):
    """Comparison between two evaluation results."""

    result_a: EvalResult
    result_b: EvalResult
    score_deltas: dict[str, float] = Field(default_factory=dict)
    verdict: str = ""


class MetricScoreResponse(BaseModel):
    """API response wrapper for a metric score."""

    score: Optional[float] = None
    explanation: str
    confidence: float
    details: dict[str, Any]
