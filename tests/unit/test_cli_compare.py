"""Tests for the ``compare`` CLI command and its regression gate."""

from __future__ import annotations

import httpx
from click.testing import CliRunner

from cli.main import cli, find_regressions

BASELINE = {
    "overall_score": 0.80,
    "faithfulness": {"score": 0.85},
    "hallucination_rate": {"score": 0.10},
    "retrieval_precision": {"score": 0.70},
    "answer_relevance": {"score": 0.75},
    "context_coverage": {"score": 0.72},
    "ucm_confidence": {"score": 0.66},
}

# Candidate: overall + faithfulness dropped, hallucination rate rose.
REGRESSED = {
    "overall_score": 0.60,
    "faithfulness": {"score": 0.55},
    "hallucination_rate": {"score": 0.40},
    "retrieval_precision": {"score": 0.69},
    "answer_relevance": {"score": 0.74},
    "context_coverage": {"score": 0.71},
    "ucm_confidence": {"score": 0.65},
}

# Candidate: everything stable, hallucination rate unchanged.
STABLE = dict(BASELINE, overall_score=0.80)

METRICS = [
    ("Overall", 0.80, 0.60),
    ("Hallucination Rate", 0.10, 0.40),
    ("Faithfulness", 0.85, 0.84),
]


def test_find_regressions_detects_drops_and_rate_increases():
    regressed = find_regressions(METRICS, threshold=0.05)
    names = [name for name, _a, _b, _d in regressed]
    assert "Overall" in names  # higher-is-better dropped
    assert "Hallucination Rate" in names  # lower-is-better increased
    # A 0.01 move is inside the 0.05 threshold.
    assert "Faithfulness" not in names


def test_find_regressions_respects_threshold():
    assert find_regressions(METRICS, threshold=0.5) == []


def test_find_regressions_detects_hallucination_improvement_as_not_regressed():
    metrics = [("Hallucination Rate", 0.40, 0.10)]  # went down = better
    assert find_regressions(metrics, threshold=0.01) == []


def _patch_reports(monkeypatch, candidate: dict) -> None:
    class _Resp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def fake_get(url, *args, **kwargs):
        return _Resp(BASELINE if "baseline-id" in url else candidate)

    monkeypatch.setattr(httpx, "get", fake_get)


def test_compare_gate_fails_on_regression(monkeypatch):
    _patch_reports(monkeypatch, REGRESSED)
    result = CliRunner().invoke(
        cli,
        ["compare", "--id-a", "baseline-id", "--id-b", "candidate-id",
         "--fail-on-regression", "0.05"],
    )
    assert result.exit_code == 1
    assert "REGRESSION" in result.output


def test_compare_without_gate_exits_zero(monkeypatch):
    _patch_reports(monkeypatch, REGRESSED)
    result = CliRunner().invoke(
        cli, ["compare", "--id-a", "baseline-id", "--id-b", "candidate-id"]
    )
    assert result.exit_code == 0
    assert "REGRESSION" not in result.output


def test_compare_gate_passes_when_stable(monkeypatch):
    _patch_reports(monkeypatch, STABLE)
    result = CliRunner().invoke(
        cli,
        ["compare", "--id-a", "baseline-id", "--id-b", "candidate-id",
         "--fail-on-regression", "0.05"],
    )
    assert result.exit_code == 0
    assert "OK" in result.output


def test_compare_reports_error_and_exits_nonzero_when_api_unreachable(monkeypatch):
    def _boom(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "get", _boom)
    result = CliRunner().invoke(
        cli, ["compare", "--id-a", "a", "--id-b", "b"]
    )
    assert result.exit_code == 1
    assert "Error" in result.output
