"use client";

interface HallucinationDetail {
  claim: string;
  /** null when the claim could not be verified (provider failure). */
  grounded: boolean | null;
  factual: boolean | null;
  hallucination_type: string;
  reason?: string;
}

interface HallucinationHeatmapProps {
  claims: HallucinationDetail[];
}

export default function HallucinationHeatmap({ claims }: HallucinationHeatmapProps) {
  if (!claims || claims.length === 0) {
    return (
      <div className="text-center py-8 text-slate-500 dark:text-slate-400 text-sm">
        No claims to display.
      </div>
    );
  }

  const supported = claims.filter((c) => c.grounded === true).length;
  const unsupported = claims.filter((c) => c.grounded === false).length;
  const unknown = claims.filter((c) => c.grounded === null || c.grounded === undefined).length;

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3 text-sm">
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm bg-emerald-500" />
          Supported: {supported}
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm bg-red-500" />
          Unsupported: {unsupported}
        </span>
        {unknown > 0 && (
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded-sm bg-slate-400" />
            Unverified: {unknown}
          </span>
        )}
      </div>

      <div className="space-y-1">
        {claims.map((claim, i) => {
          const state: "supported" | "unsupported" | "unknown" =
            claim.grounded === true ? "supported" : claim.grounded === false ? "unsupported" : "unknown";
          const rowClass =
            state === "supported"
              ? "bg-emerald-50 dark:bg-emerald-900/20"
              : state === "unsupported"
              ? "bg-red-50 dark:bg-red-900/20"
              : "bg-slate-50 dark:bg-slate-800/40";
          const dotClass =
            state === "supported"
              ? "bg-emerald-500"
              : state === "unsupported"
              ? "bg-red-500"
              : "bg-slate-400";
          const textClass =
            state === "supported"
              ? "text-emerald-600 dark:text-emerald-400"
              : state === "unsupported"
              ? "text-red-600 dark:text-red-400"
              : "text-slate-500 dark:text-slate-400";

          return (
            <div key={i} className={`flex items-start gap-2 p-2 rounded text-xs ${rowClass}`}>
              <span className={`mt-0.5 w-2 h-2 rounded-full flex-shrink-0 ${dotClass}`} />
              <div className="flex-1 min-w-0">
                <p className="text-slate-700 dark:text-slate-300 truncate">{claim.claim}</p>
                {state === "unsupported" && claim.reason && (
                  <p className="text-red-600 dark:text-red-400 mt-0.5">{claim.reason}</p>
                )}
              </div>
              <span className={`flex-shrink-0 text-xs font-medium ${textClass}`}>
                {claim.hallucination_type ||
                  (state === "supported" ? "supported" : state === "unsupported" ? "unsupported" : "unknown")}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
