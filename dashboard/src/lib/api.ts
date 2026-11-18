// Client for greenquery_server. The dashboard still works without it: the
// archived snapshot is bundled, and live features show as "backend offline".

const CONFIGURED_API = (process.env.NEXT_PUBLIC_GREENQUERY_API ?? "http://localhost:8000").replace(/\/$/, "");
const isLocalHost = (host: string) => ["localhost", "127.0.0.1", "[::1]"].includes(host);
// A phone that opened the dashboard over the network can't reach a localhost API, so it goes through the
// dashboard's own /gq-api proxy (next.config.ts).
export const API_BASE = typeof window !== "undefined" && !isLocalHost(window.location.hostname)
  && isLocalHost(new URL(CONFIGURED_API).hostname) ? "/gq-api" : CONFIGURED_API;

export type SystemInfo = {
  key: string;
  label: string;
  description: string;
  status: "available" | "experimental" | "planned";
  workloads: string[];
  notes: string;
  providers: Array<"openai" | "anthropic" | "openai_compatible">;
  multi_model: boolean;
};
export type WorkloadInfo = {
  key: string;
  dataset: string;
  queries: string[];
  description: string;
  single_query_runs: boolean;
  preparation: string | null;
};
export type ModelInfo = {
  key: string;
  provider: "openai" | "anthropic" | "openai_compatible";
  api_model: string;
  display_name: string;
  deployment: string;
  price_in: number;
  price_out: number;
  energy: "ecologits" | "ecologits_proxy" | "not_available";
  ecologits_model: string | null;
  credential_env: string | null;
  credential_available: boolean;
  custom?: boolean;
};
export type Catalog = {
  systems: SystemInfo[];
  workloads: WorkloadInfo[];
  models: ModelInfo[];
  limits: { max_run_cost_usd: number; max_repetitions: number };
  executor: "local" | "slurm";
};
export type JobStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";
export type Job = {
  id: string;
  created_at: number;
  updated_at: number;
  status: JobStatus;
  step: string | null;
  spec: { system: string; workload: string; model: string; query: string | null; repetitions: number; cache_policy: string; operator_models?: string | null };
  cost_estimate_usd: number | null;
  validation_status: string | null;
  record_count: number;
  error: string | null;
};
export type CostEstimate = { total_usd: number | null; per_query_usd: Record<string, number | null>; basis: string };
export type RunRequest = {
  system: string;
  workload: string;
  model: string;
  query: string | null;
  repetitions: number;
  cache_policy: "disabled" | "default";
  // Multi-model mode: alternative operator models, cheapest first; null = single-model.
  operator_models: string[] | null;
  accept_unknown_cost?: boolean;
  dry_run?: boolean;
};
export type JobEvent = { type: "status"; job: Job | null } | { type: "log"; line: string } | { type: "ping" };

export class ApiError extends Error {
  constructor(message: string, readonly status: number, readonly detail: unknown) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = (body as { detail?: unknown }).detail;
    const message = typeof detail === "string" ? detail : (detail as { message?: string } | undefined)?.message ?? response.statusText;
    throw new ApiError(message, response.status, detail);
  }
  return body as T;
}

export const fetchCatalog = () => request<Catalog>("/api/catalog");
export const fetchJobs = () => request<{ jobs: Job[] }>("/api/runs").then((body) => body.jobs);
export const fetchLiveRecords = () =>
  request<{ records: Array<Record<string, unknown>> }>("/api/records?source=live").then((body) => body.records);
export const estimateRun = (run: RunRequest) =>
  request<{ estimate: CostEstimate }>("/api/runs", { method: "POST", body: JSON.stringify({ ...run, dry_run: true }) }).then((body) => body.estimate);
export const startRun = (run: RunRequest) =>
  request<{ job: Job; estimate: CostEstimate }>("/api/runs", { method: "POST", body: JSON.stringify(run) });
/** Where a model added from the dashboard comes from ("bring your own model"). */
export type ModelSource = "ollama" | "huggingface" | "openai" | "openrouter";
export const MODEL_SOURCES: { key: ModelSource; label: string; placeholder: string; local: boolean }[] = [
  { key: "ollama", label: "Ollama library", placeholder: "e.g. phi3:mini or mistral:7b", local: true },
  { key: "huggingface", label: "Hugging Face (GGUF)", placeholder: "e.g. bartowski/Phi-3.5-mini-instruct-GGUF:Q4_K_M", local: true },
  { key: "openai", label: "OpenAI API", placeholder: "e.g. gpt-4.1-mini", local: false },
  { key: "openrouter", label: "OpenRouter API", placeholder: "e.g. anthropic/claude-haiku-5.5", local: false },
];
/**
 * Add a model. Local models (Ollama library, Hugging Face GGUF) are downloaded by the next run on the DGX
 * and get measured energy; API models (OpenAI, OpenRouter) run through the logging proxy with a spending
 * cap and have no measured energy. OpenAI models need their prices (USD per million tokens).
 */
export const addCustomModel = (source: ModelSource, model: string, prices?: { price_in?: number; price_out?: number }) =>
  request<{ model: ModelInfo; added: boolean }>("/api/models/custom",
    { method: "POST", body: JSON.stringify({ source, model, ...prices }) });
export const cancelRun = (jobId: string) => request<{ job: Job }>(`/api/runs/${jobId}/cancel`, { method: "POST" });

/** Stream a job's log and status; returns a function that closes the stream. */
export function subscribeToJob(jobId: string, onEvent: (event: JobEvent) => void): () => void {
  const source = new EventSource(`${API_BASE}/api/runs/${jobId}/events`);
  source.onmessage = (message) => onEvent(JSON.parse(message.data) as JobEvent);
  source.onerror = () => source.close();
  return () => source.close();
}

// ------------------------------------------------------------------ challenge
export type LiveProgress = {
  phase: string;
  energy_wh: number;
  llm_calls: number;
  llm_errors: number;
  prompt_tokens: number;
  completion_tokens: number;
  elapsed_s: number;
  done: boolean;
  error?: string;
  record?: LiveRecord | null;
  // Heartbeat: GPU power in 1 s buckets as [seconds since the query started, W], LLM call start times.
  power?: Array<[number, number]>;
  call_starts?: number[];
  idle_w?: number | null;
  power_w?: number | null;
};
export type LiveRecord = {
  energy_kwh: number;
  runtime_seconds: number | null;
  error: string | null;
  quality: { status: string; score: number | null; f1: number | null; relative_error: number | null };
  measurement: {
    gpu_wh: number; gpu_dynamic_wh: number; cpu_job_wh: number; gpu_idle_w: number | null;
    llm_calls: number | null; prompt_tokens: number | null; completion_tokens: number | null; timed_out: boolean;
  };
};
export type BoardEntry = {
  id?: number;
  name: string;
  query: string;
  system: string;
  model: string;
  energy_wh: number;
  quality: number;
  mode: "live" | "recorded";
  job_id?: string | null;
  created_at?: number;
};

export const fetchLive = (jobId: string) => request<{ job: Job; progress: LiveProgress }>(`/api/runs/${jobId}/live`);
export const fetchBoard = (query: string) =>
  request<{ entries: BoardEntry[] }>(`/api/challenge/board?query=${encodeURIComponent(query)}`).then((body) => body.entries);
export const postBoardEntry = (entry: BoardEntry) =>
  request<{ entry: BoardEntry }>("/api/challenge/board", { method: "POST", body: JSON.stringify(entry) });
export const resetBoard = () => request<{ deleted: number }>("/api/challenge/reset", { method: "POST" });
