"use client";

import { useState } from "react";
import { EvalResult } from "@/lib/api";
import EvalForm from "@/components/EvalForm";
import MetricRadar from "@/components/MetricRadar";
import ScoreCard from "@/components/ScoreCard";

const METRICS_CONFIG = [
  { title: "Faithfulness", key: "faithfulness" as const, invert: false },
  { title: "Hallucination", key: "hallucination_rate" as const, invert: true },
  { title: "Retrieval Precision", key: "retrieval_precision" as const, invert: false },
  { title: "Answer Relevance", key: "answer_relevance" as const, invert: false },
  { title: "Context Coverage", key: "context_coverage" as const, invert: false },
  { title: "UCM Confidence", key: "ucm_confidence" as const, invert: false },
] as const;

export default function EvaluatePage() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<EvalResult | null>(null);

  const handleResult = (data: EvalResult) => {
    setResult(data);
  };

  const handleError = (msg: string) => {
    setError(msg);
  };

  const handleLoadingChange = (isLoading: boolean) => {
    setLoading(isLoading);
  };

  // Failed metrics have a null score; drop them from the radar rather than
  // plotting NaN and visually implying a zero score.
  const radarData = result
    ? [
        { metric: "Faithfulness", score: result.faithfulness.score },
        {
          metric: "Hallucination",
          score:
            result.hallucination_rate.score === null
              ? null
              : 1 - result.hallucination_rate.score,
        },
        { metric: "Retrieval", score: result.retrieval_precision.score },
        { metric: "Relevance", score: result.answer_relevance.score },
        { metric: "Coverage", score: result.context_coverage.score },
        { metric: "UCM", score: result.ucm_confidence.score },
      ]
        .filter((d): d is { metric: string; score: number } => d.score !== null)
        .map((d) => ({ metric: d.metric, score: d.score * 100 }))
    : [];

  return (
    <div className="space-y-8">
      <h1 className="text-3xl font-bold">Run Evaluation</h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Form */}
        <div className="card p-6 space-y-5">
          <EvalForm
            onResult={handleResult}
            onError={handleError}
            onLoadingChange={handleLoadingChange}
          />

          {error && (
            <div className="bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400 px-4 py-3 rounded-lg text-sm">
              {error}
            </div>
          )}
        </div>

        {/* Results */}
        <div className="space-y-4">
          {loading && (
            <div className="card p-8 flex items-center justify-center">
              <div className="text-center">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600 mx-auto mb-3" />
                <p className="text-sm text-slate-500">Running evaluation...</p>
              </div>
            </div>
          )}

          {result && !loading && (
            <>
              {/* Overall Score */}
              <div className="card p-6 text-center">
                <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Overall Score</p>
                <p className={`text-4xl font-bold mt-1 ${
                  result.overall_score >= 0.8 ? "text-emerald-600 dark:text-emerald-400" :
                  result.overall_score >= 0.6 ? "text-amber-600 dark:text-amber-400" :
                  "text-red-600 dark:text-red-400"
                }`}>
                  {(result.overall_score * 100).toFixed(0)}%
                </p>
                <p className="text-xs text-slate-400 mt-1">{result.latency_ms}ms</p>
              </div>

              {/* Radar Chart */}
              <div className="card p-6">
                <h3 className="text-sm font-medium mb-3">Metric Overview</h3>
                <MetricRadar data={radarData} height={256} />
              </div>

              {/* Metrics Grid */}
              <div className="grid grid-cols-2 gap-3">
                {METRICS_CONFIG.map(({ title, key, invert }) => (
                  <ScoreCard
                    key={key}
                    title={title}
                    score={result[key].score}
                    explanation={result[key].explanation}
                    invert={invert}
                    size="sm"
                  />
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
