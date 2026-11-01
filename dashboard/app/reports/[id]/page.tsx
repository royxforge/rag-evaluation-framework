"use client";

import { useState, useEffect } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { getReport, EvalResult } from "@/lib/api";
import HallucinationHeatmap from "@/components/HallucinationHeatmap";

export default function ReportPage() {
  const params = useParams();
  const id = params.id as string;
  const [result, setResult] = useState<EvalResult | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const data = await getReport(id, "json");
        setResult(data as EvalResult);
      } catch {
        // Will show error state
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [id]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600" />
      </div>
    );
  }

  if (!result) {
    return (
      <div className="text-center py-16">
        <p className="text-4xl mb-4">🔍</p>
        <h2 className="text-xl font-semibold mb-2">Report Not Found</h2>
        <p className="text-slate-500 dark:text-slate-400 mb-4">
          Could not load report {id}
        </p>
        <Link href="/" className="btn-primary">Back to Home</Link>
      </div>
    );
  }

  const metrics = [
    { name: "Faithfulness", data: result.faithfulness },
    { name: "Hallucination Rate", data: result.hallucination_rate, invert: true },
    { name: "Retrieval Precision", data: result.retrieval_precision },
    { name: "Answer Relevance", data: result.answer_relevance },
    { name: "Context Coverage", data: result.context_coverage },
    { name: "UCM Confidence", data: result.ucm_confidence },
  ];

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-3xl font-bold">Evaluation Report</h1>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
            ID: {result.id} &middot; {new Date(result.timestamp).toLocaleString()}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => {
              const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
              const url = URL.createObjectURL(blob);
              const a = document.createElement("a");
              a.href = url;
              a.download = `report_${result.id.slice(0, 8)}.json`;
              a.click();
              URL.revokeObjectURL(url);
            }}
            className="btn-secondary text-sm"
          >
            Export JSON
          </button>
        </div>
      </div>

      {/* Overall Score */}
      <div className={`card p-8 text-center ${
        result.overall_score >= 0.8 ? "border-emerald-500/30" :
        result.overall_score >= 0.6 ? "border-amber-500/30" :
        "border-red-500/30"
      }`}>
        <p className="text-sm font-medium text-slate-500 dark:text-slate-400">Overall Score</p>
        <p className={`text-5xl font-bold mt-2 ${
          result.overall_score >= 0.8 ? "text-emerald-600 dark:text-emerald-400" :
          result.overall_score >= 0.6 ? "text-amber-600 dark:text-amber-400" :
          "text-red-600 dark:text-red-400"
        }`}>
          {(result.overall_score * 100).toFixed(0)}%
        </p>
        <p className="text-sm text-slate-400 mt-2">Latency: {result.latency_ms}ms</p>
      </div>

      {/* Question & Answer */}
      <div className="card p-6 space-y-4">
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-1">Question</h3>
          <p className="text-sm">{result.question}</p>
        </div>
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-1">Answer</h3>
          <p className="text-sm">{result.answer}</p>
        </div>
        <div>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-1">Context ({result.context.length} chunks)</h3>
          <div className="space-y-1">
            {result.context.map((chunk, i) => (
              <p key={i} className="text-xs text-slate-500 dark:text-slate-400 bg-slate-50 dark:bg-slate-800 p-2 rounded">
                [{i + 1}] {chunk}
              </p>
            ))}
          </div>
        </div>
      </div>

      {/* Metrics */}
      <div className="space-y-4">
        <h2 className="text-lg font-semibold">Metrics</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {metrics.map(({ name, data, invert }) => {
            // A null score means the metric failed to compute; show n/a and
            // leave the bar empty instead of painting it red as a 0% score.
            const effectiveScore =
              data.score === null ? null : invert ? 1 - data.score : data.score;
            const colorClass =
              effectiveScore === null ? "" :
              effectiveScore >= 0.8 ? "score-green" :
              effectiveScore >= 0.6 ? "score-amber" : "score-red";

            return (
              <div key={name} className="card p-5">
                <div className="flex items-start justify-between mb-3">
                  <h3 className="font-medium text-sm">{name}</h3>
                  <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${colorClass}`}>
                    {effectiveScore === null ? "n/a" : `${(effectiveScore * 100).toFixed(0)}%`}
                  </span>
                </div>

                {/* Score bar */}
                <div className="h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden mb-3">
                  <div
                    className={`h-full rounded-full transition-all ${
                      effectiveScore === null ? "bg-slate-400 dark:bg-slate-600" :
                      effectiveScore >= 0.8 ? "bg-emerald-500" :
                      effectiveScore >= 0.6 ? "bg-amber-500" : "bg-red-500"
                    }`}
                    style={{ width: `${effectiveScore === null ? 0 : (effectiveScore * 100).toFixed(0)}%` }}
                  />
                </div>

                <p className="text-xs text-slate-500 dark:text-slate-400">
                  {data.explanation || "No explanation"}
                </p>

                {name === "Hallucination Rate" && Array.isArray(data.details?.claims) && (
                  <div className="mt-3">
                    <HallucinationHeatmap claims={data.details.claims as Array<{ claim: string; grounded: boolean; factual: boolean; hallucination_type: string; reason?: string }>} />
                  </div>
                )}

                {name === "UCM Confidence" && Array.isArray(data.details?.samples) && (
                  <div className="mt-3 space-y-1">
                    <p className="text-xs font-medium text-slate-500">Samples:</p>
                    {(data.details.samples as unknown as string[]).slice(0, 3).map((s, i) => (
                      <p key={i} className="text-xs text-slate-500 dark:text-slate-400 bg-slate-50 dark:bg-slate-800 p-2 rounded">
                        [{i + 1}] {s.slice(0, 100)}{s.length > 100 ? "..." : ""}
                      </p>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
