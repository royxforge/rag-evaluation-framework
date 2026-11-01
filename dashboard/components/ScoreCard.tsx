"use client";

import clsx from "clsx";

interface ScoreCardProps {
  title: string;
  /** null when the metric failed to compute. */
  score: number | null;
  subtitle?: string;
  icon?: string;
  size?: "sm" | "md" | "lg";
  explanation?: string;
  invert?: boolean;
}

export default function ScoreCard({ title, score, subtitle, icon, size = "md", explanation, invert }: ScoreCardProps) {
  const effectiveScore = score === null ? null : invert ? 1 - score : score;

  const colorClass =
    effectiveScore === null
      ? "text-slate-400 dark:text-slate-500"
      : effectiveScore >= 0.8
      ? "text-emerald-600 dark:text-emerald-400"
      : effectiveScore >= 0.6
      ? "text-amber-600 dark:text-amber-400"
      : "text-red-600 dark:text-red-400";

  const bgClass =
    effectiveScore === null
      ? "bg-slate-50 dark:bg-slate-800/40"
      : effectiveScore >= 0.8
      ? "bg-emerald-50 dark:bg-emerald-900/20"
      : effectiveScore >= 0.6
      ? "bg-amber-50 dark:bg-amber-900/20"
      : "bg-red-50 dark:bg-red-900/20";

  return (
    <div className={clsx("card p-4", bgClass)}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-medium text-slate-500 dark:text-slate-400">{title}</p>
          <p className={clsx("font-bold", {
            "text-2xl": size === "lg",
            "text-xl": size === "md",
            "text-lg": size === "sm",
          }, colorClass)}>
            {effectiveScore === null ? "n/a" : `${(effectiveScore * 100).toFixed(0)}%`}
          </p>
          {subtitle && (
            <p className="text-xs text-slate-400 dark:text-slate-500 mt-0.5">{subtitle}</p>
          )}
        </div>
        {icon && <span className="text-xl">{icon}</span>}
      </div>
      <div className="mt-3 h-1.5 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
        <div
          className={clsx("h-full rounded-full transition-all duration-500", {
            "bg-slate-400 dark:bg-slate-600": effectiveScore === null,
            "bg-emerald-500": effectiveScore !== null && effectiveScore >= 0.8,
            "bg-amber-500": effectiveScore !== null && effectiveScore >= 0.6 && effectiveScore < 0.8,
            "bg-red-500": effectiveScore !== null && effectiveScore < 0.6,
          })}
          style={{ width: `${effectiveScore === null ? 0 : (effectiveScore * 100).toFixed(0)}%` }}
        />
      </div>
      {explanation && (
        <p className="text-xs text-slate-400 dark:text-slate-500 mt-2 line-clamp-2">
          {explanation}
        </p>
      )}
    </div>
  );
}
