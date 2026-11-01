/** API client for the RAG Evaluation Framework backend. */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface EvaluateRequest {
  question: string;
  context: string[];
  answer: string;
  metrics?: string[];
  llm?: string;
  options?: Record<string, unknown>;
}

interface MetricScore {
  /** null when the metric failed to compute; render as "n/a", not as 0. */
  score: number | null;
  explanation: string;
  confidence: number;
  details: Record<string, unknown>;
}

export interface EvalResult {
  id: string;
  timestamp: string;
  question: string;
  context: string[];
  answer: string;
  overall_score: number;
  faithfulness: MetricScore;
  hallucination_rate: MetricScore;
  retrieval_precision: MetricScore;
  answer_relevance: MetricScore;
  context_coverage: MetricScore;
  ucm_confidence: MetricScore;
  metadata: Record<string, unknown>;
  latency_ms: number;
}

export interface BatchJob {
  job_id: string;
  status: string;
  total_items: number;
  completed_items: number;
  result_ids: string[];
  error: string | null;
}

/**
 * Get API key from localStorage or prompt.
 */
function getApiKey(): string {
  if (typeof window === "undefined") return "";
  const key = localStorage.getItem("rag_evaluation_framework_api_key");
  if (key) return key;
  return "";
}

/**
 * Set the API key (user-facing).
 */
export function setApiKey(key: string): void {
  localStorage.setItem("rag_evaluation_framework_api_key", key);
}

/**
 * Headers for authenticated requests.
 */
function headers(): Record<string, string> {
  const h: Record<string, string> = { "Content-Type": "application/json" };
  const key = getApiKey();
  if (key) h["Authorization"] = `Bearer ${key}`;
  return h;
}

/**
 * Run a single evaluation.
 */
export async function evaluate(data: EvaluateRequest): Promise<EvalResult> {
  const res = await fetch(`${API_BASE}/v1/evaluate`, {
    method: "POST",
    headers: headers(),
    body: JSON.stringify(data),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Evaluation failed: ${err}`);
  }
  return res.json();
}

/**
 * Submit a batch evaluation job.
 */
export async function batchEvaluate(
  items: Array<{ question: string; context: string[]; answer: string }>,
  llm = "openai/gpt-4o",
  webhookUrl?: string
): Promise<{ job_id: string; status: string; estimated_seconds: number }> {
  const res = await fetch(`${API_BASE}/v1/batch`, {
    method: "POST",
    headers: headers(),
    body: JSON.stringify({ items, llm, webhook_url: webhookUrl }),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`Batch submission failed: ${err}`);
  }
  return res.json();
}

/**
 * Get job status.
 */
export async function getJobStatus(jobId: string): Promise<BatchJob> {
  const res = await fetch(`${API_BASE}/v1/jobs/${jobId}`, {
    headers: headers(),
  });
  if (!res.ok) throw new Error("Failed to get job status");
  return res.json();
}

/**
 * Get a report by result ID.
 */
export async function getReport(
  resultId: string,
  format: "json" | "html" = "json"
): Promise<EvalResult | string> {
  const res = await fetch(`${API_BASE}/v1/reports/${resultId}?format=${format}`, {
    headers: headers(),
  });
  if (!res.ok) throw new Error("Failed to get report");
  if (format === "html") return res.text();
  return res.json();
}

/**
 * List recent evaluation results (summary data).
 */
export interface ReportSummary {
  id: string;
  question: string;
  overall_score: number;
  latency_ms: number;
  created_at: string;
  faithfulness_score: number | null;
  hallucination_rate: number | null;
  retrieval_precision: number | null;
  answer_relevance: number | null;
  context_coverage: number | null;
  ucm_score: number | null;
}

export async function getReports(
  limit = 100,
  offset = 0
): Promise<ReportSummary[]> {
  const res = await fetch(
    `${API_BASE}/v1/reports?limit=${limit}&offset=${offset}`,
    { headers: headers() }
  );
  if (!res.ok) throw new Error("Failed to fetch reports");
  return res.json();
}

/**
 * Get health status.
 */
export async function getHealth(): Promise<{
  status: string;
  version: string;
  database: string;
  redis: string;
}> {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}
