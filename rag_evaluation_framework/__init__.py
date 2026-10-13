"""RAG Evaluation Framework — Production-grade RAG evaluation framework."""

from rag_evaluation_framework.evaluator import Evaluator
from rag_evaluation_framework.models import EvalResult, MetricScore
from rag_evaluation_framework.report import ReportBuilder, build_report

__all__ = [
    "Evaluator",
    "EvalResult",
    "MetricScore",
    "ReportBuilder",
    "build_report",
]

__version__ = "0.5.0"
