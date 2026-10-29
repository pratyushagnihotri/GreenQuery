import animals from "./animals_query_metrics.json";
import artwork from "./artwork_query_metrics.json";
import medical from "./medical_query_metrics.json";
import medicalOnline from "./medical_online_query_metrics.json";
import movie from "./movie_query_metrics.json";
import movieLocal from "./movie_local_query_metrics.json";
import rotowire from "./rotowire_query_metrics.json";
import rotowireOnline from "./rotowire_online_query_metrics.json";

export type PlanEdge = { from: string; to: string; kind: string };
export type PlanStep = {
  id: string;
  type?: string;
  operator?: string;
  explanation?: string;
  expression?: string;
  logical_step_id?: string;
  inputs?: string[];
  output?: string;
  operator_config?: Record<string, unknown>;
};
export type StructuredExecutionPlan = {
  schema_version?: number;
  logical?: { steps: PlanStep[]; edges: PlanEdge[] };
  physical?: { steps: PlanStep[]; edges: PlanEdge[] };
  edges?: PlanEdge[];
  optimization?: { candidate_steps?: unknown[] };
};

type RawRecord = {
  query: string;
  model: string;
  run: string | number;
  runtime_seconds: number;
  energy_kwh: number;
  co2_g: number | null;
  physical_operators?: string[];
  execution_plan?: StructuredExecutionPlan;
  quality?: Record<string, unknown>;
  [key: string]: unknown;
};

export type BenchmarkRecord = RawRecord & {
  dataset: string;
  run: string;
  source: "archive" | "live";
  system_label: string;
  validation_status: string | null;
  energy_wh: number | null;
  planner_energy_wh: number | null;
  execution_energy_wh: number | null;
  planner_co2_g: number | null;
  execution_co2_g: number | null;
  planner_runtime_seconds: number | null;
  execution_runtime_seconds: number | null;
  quality_status: string;
  precision: number | null;
  recall: number | null;
  f1: number | null;
  absolute_error: number | null;
  ground_truth: string;
  correct_rows: number | null;
  returned_rows: number | null;
  expected_rows: number | null;
  expected_count: number | null;
  observed_count: number | null;
  expected_average_age: number | null;
  observed_average_age: number | null;
  expected_ratio: number | null;
  observed_ratio: number | null;
  false_positive_rows: number | null;
  false_negative_rows: number | null;
  group_columns: string[];
  correct: boolean | null;
  quality_metric: string;
  quality_selection_f1: number | null;
  quality_century_accuracy: number | null;
  quality_century_checked_rows: number | null;
  accuracy: number | null;
  checked_fields: number | null;
  correct_fields: number | null;
  deployment: "Local DGX" | "Online API" | "Local server";
  energy_provenance: string;
};

export type QualityMetricDisplay = {
  label: string;
  value: number | string | boolean;
  direction: "higher is better" | "lower is better" | "informational";
  provenance: "recorded" | "derived";
};

export type PrimaryQuality = {
  label: string;
  value: number;
  direction: "higher is better" | "lower is better";
  secondary?: { label: string; value: number };
};

const QUERY_TEXT: Record<string, string> = {
  Q1_allergy_patients: "Which patients have symptoms that indicate allergies?",
  Q3_cancer_family_lung_problems_patients: "Find sick patients with cancer in their family history and lung problems visible in their X-rays.",
  Q4_average_age_acne_patients: "What is the average age of patients with acne?",
  Q9_sick_patients_skin_xray: "Find patients considered sick by both skin-mole and lung X-ray evidence.",
  R1_player_points_assists: "How many points and assists did each player record according to the game report?",
  R2_player_points_home_team_won: "How many points did each player score in reports where the home team won?",
  Q1_positive_reviews: "Which are five clearly positive reviews?",
  Q2_positive_reviews_for_taken_3: "Which are five clearly positive reviews for the movie Taken 3?",
  Q3_reviews_same_sentiment: "Find pairs of reviews that have the same sentiment.",
  Q4_positive_reviews_for_taken_3_ratio: "What is the positivity ratio of reviews for Taken 3?",
  A1_madonna_and_child: "Find artworks depicting Madonna and Child.",
  A2_more_than_two_people: "Does the artwork show more than two people?",
  Q1_zebra_image_count: "How many zebra images are available in the collection?",
  Q2_lion_audio_count: "How many lion audio recordings are available in the collection?",
  Q3_top_zebra_city: "Which city has the highest number of zebra observations?",
  Q10_top_zebra_city_station: "Which city and station have the highest number of zebra observations?",
};

export const DATASET_INFO: Record<string, { description: string; workload: string }> = {
  Animals: {
    description: "Multimodal wildlife records with image, audio, and grouped observation questions.",
    workload: "Visual/audio filtering, aggregation, and top-group queries",
  },
  Artwork: {
    description: "Artwork image-understanding workloads paired with metadata and century checks.",
    workload: "Visual selection, extraction, and metadata validation",
  },
  Medical: {
    description: "Patient records combining structured attributes with text and clinical-image reasoning.",
    workload: "Patient filtering, joins, aggregation, and multimodal evidence",
  },
  Movie: {
    description: "Movie-review workloads covering sentiment, joins, result limits, and ratios.",
    workload: "Text extraction, filtering, joins, and aggregate ratios",
  },
  RotoWire: {
    description: "Sports-report workloads that extract player statistics and combine report facts.",
    workload: "Text extraction, filtering, projection, and joins",
  },
};

const sources: Array<[string, RawRecord[]]> = [
  ["Animals", animals as RawRecord[]],
  ["Artwork", artwork as RawRecord[]],
  ["Medical", medical as RawRecord[]],
  ["Medical", medicalOnline as RawRecord[]],
  ["Movie", movie as RawRecord[]],
  ["Movie", movieLocal as RawRecord[]],
  ["RotoWire", rotowire as RawRecord[]],
  ["RotoWire", rotowireOnline as RawRecord[]],
];

const number = (value: unknown) => typeof value === "number" ? value : null;
const boolean = (value: unknown) => typeof value === "boolean" ? value : null;
const strings = (value: unknown) => Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];

function normalize(dataset: string, record: RawRecord, origin: "archive" | "live"): BenchmarkRecord {
    const quality = record.quality ?? {};
    const planning = (record.planning ?? {}) as Record<string, unknown>;
    const execution = (record.execution ?? {}) as Record<string, unknown>;
    const online = record.energy_source === "estimated_by_ecologits";
    // Missing telemetry stays missing: never show an unmeasured run as 0 Wh.
    const measured = record.energy_source !== "not_available";
    const deployment: BenchmarkRecord["deployment"] =
      record.deployment_label === "Local server" ? "Local server" : online ? "Online API" : "Local DGX";
    const plannerEnergy = measured ? number(record.planner_energy_kwh) ?? number(planning.energy_kwh) : null;
    const executionEnergy = measured ? number(record.execution_energy_kwh) ?? number(execution.energy_kwh) : null;
    const proxy = typeof record.ecologits_model === "string" ? ` (proxy: ${record.ecologits_model})` : "";
    // Live runs from other systems/optimizers or multi-model mode get their own model
    // label so existing views (grouped by model) keep them apart from plain Stretto runs.
    const systemLabel = typeof record.system_label === "string" ? record.system_label : "Stretto";
    const operatorModels = Array.isArray(record.operator_models) ? (record.operator_models as string[]) : [];
    const multi = record.operator_mode === "multi_model" && operatorModels.length > 0;
    const short = (model: string) => model.split("/").at(-1) ?? model;
    const variant = [
      systemLabel !== "Stretto" ? systemLabel : null,
      multi ? `(multi: ${operatorModels.map(short).join("+")})` : null,
    ].filter(Boolean).join(" ");
    return {
      ...record,
      model: variant ? `${record.model} · ${variant}` : record.model,
      dataset,
      run: String(record.run),
      source: origin,
      system_label: typeof record.system_label === "string" ? record.system_label : "Stretto",
      validation_status: typeof record.validation_status === "string" ? record.validation_status : null,
      co2_g: measured ? record.co2_g : null,
      energy_wh: measured ? record.energy_kwh * 1000 : null,
      planner_energy_wh: plannerEnergy === null ? null : plannerEnergy * 1000,
      execution_energy_wh: executionEnergy === null ? null : executionEnergy * 1000,
      planner_co2_g: number(record.planner_co2_g) ?? number(planning.co2_g),
      execution_co2_g: number(record.execution_co2_g) ?? number(execution.co2_g),
      planner_runtime_seconds: number(record.planner_runtime_seconds) ?? number(planning.total_runtime_seconds),
      execution_runtime_seconds: number(record.execution_runtime_seconds) ?? number(execution.total_runtime_seconds),
      planner_input_tokens: number(record.planner_input_tokens) ?? number(planning.total_prompt_tokens),
      planner_output_tokens: number(record.planner_output_tokens) ?? number(planning.total_completion_tokens),
      execution_input_tokens: number(record.execution_input_tokens) ?? number(execution.total_prompt_tokens),
      execution_output_tokens: number(record.execution_output_tokens) ?? number(execution.total_completion_tokens),
      planner_requests: number(record.planner_requests) ?? number(planning.total_calls),
      local_requests: number(record.local_requests) ?? number(execution.total_calls),
      quality_status: String(quality.status ?? "unavailable"),
      precision: number(quality.precision),
      recall: number(quality.recall),
      f1: number(quality.f1),
      absolute_error: number(quality.absolute_error),
      ground_truth: String(quality.ground_truth ?? ""),
      correct_rows: number(quality.correct_rows),
      returned_rows: number(quality.returned_rows),
      expected_rows: number(quality.expected_rows),
      expected_count: number(quality.expected_count),
      observed_count: number(quality.observed_count),
      expected_average_age: number(quality.expected_average_age),
      observed_average_age: number(quality.observed_average_age),
      expected_ratio: number(quality.expected_ratio),
      observed_ratio: number(quality.observed_ratio),
      false_positive_rows: number(quality.false_positive_rows),
      false_negative_rows: number(quality.false_negative_rows),
      group_columns: strings(quality.group_columns),
      correct: boolean(quality.correct),
      quality_metric: String(quality.metric ?? "unavailable"),
      quality_selection_f1: number(record.quality_selection_f1),
      quality_century_accuracy: number(record.quality_century_accuracy),
      quality_century_checked_rows: number(record.quality_century_checked_rows),
      accuracy: number(quality.accuracy),
      checked_fields: number(quality.checked_fields),
      correct_fields: number(quality.correct_fields),
      deployment,
      energy_provenance: !measured ? "Not measured" : online ? `EcoLogits estimate${proxy}` : "NVML measurement",
    };
}

export const records: BenchmarkRecord[] = sources.flatMap(([dataset, source]) =>
  source.map((record) => normalize(dataset, record, "archive")),
);

const liveIds = new Set<unknown>();

/** Merge records produced by greenquery_server runs; returns how many were new. */
export function addLiveRecords(raw: Array<Record<string, unknown>>): number {
  let added = 0;
  for (const record of raw) {
    if (liveIds.has(record.record_id)) continue;
    liveIds.add(record.record_id);
    records.push(normalize(String(record.dataset), record as RawRecord, "live"));
    added += 1;
  }
  return added;
}

export const queryText = (query: string) => QUERY_TEXT[query] ?? query.replaceAll("_", " ");
export const datasetDescription = (dataset: string) => DATASET_INFO[dataset]?.description ?? "Recorded benchmark workload.";
export const datasetWorkload = (dataset: string) => DATASET_INFO[dataset]?.workload ?? "Recorded query workload";
export const queryWorkload = (query: string) => {
  if (/zebra|lion/i.test(query)) return "Multimodal observation analysis";
  if (/madonna|saints|death|people/i.test(query)) return "Artwork image understanding";
  if (/patient|acne|cancer|allergy/i.test(query)) return "Clinical record analysis";
  if (/review|sentiment|taken/i.test(query)) return "Movie review analysis";
  if (/player|team/i.test(query)) return "Sports report extraction";
  return "Recorded query workload";
};
export const queryLabel = (query: string) => query.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
export const modelLabel = (model: string) => model.split("/").at(-1)?.replace("-Instruct", "") ?? model;
export const average = (values: Array<number | null | undefined>) => {
  const valid = values.filter((value): value is number => typeof value === "number");
  return valid.length ? valid.reduce((sum, value) => sum + value, 0) / valid.length : null;
};
export const median = (values: Array<number | null | undefined>) => {
  const valid = values.filter((value): value is number => typeof value === "number").sort((a, b) => a - b);
  if (!valid.length) return null;
  const middle = Math.floor(valid.length / 2);
  return valid.length % 2 ? valid[middle] : (valid[middle - 1] + valid[middle]) / 2;
};

const recorded = (label: string, value: number | string | boolean | null, direction: QualityMetricDisplay["direction"] = "informational") => value === null ? [] : [{ label, value, direction, provenance: "recorded" as const }];

export function qualityMetrics(record: BenchmarkRecord): QualityMetricDisplay[] {
  switch (record.quality_metric) {
    case "exact_aggregate_count": {
      const error = record.expected_count === null || record.observed_count === null ? [] : [{ label: "Absolute count error", value: Math.abs(record.observed_count - record.expected_count), direction: "lower is better" as const, provenance: "derived" as const }];
      return [...recorded("Expected count", record.expected_count), ...recorded("Observed count", record.observed_count), ...error, ...recorded("Correct", record.correct)];
    }
    case "top_group_set_precision_recall":
    case "patient_id_set_precision_recall":
    case "pair_precision_recall":
    case "artwork_title_set_precision_recall":
      return [
        ...recorded("Precision", record.precision, "higher is better"), ...recorded("Recall", record.recall, "higher is better"), ...recorded("F1 score", record.f1, "higher is better"),
        ...recorded("Expected rows", record.expected_rows), ...recorded("Returned rows", record.returned_rows), ...recorded("Correct rows", record.correct_rows),
        ...recorded("False positives", record.false_positive_rows), ...recorded("False negatives", record.false_negative_rows), ...recorded("Group columns", record.group_columns.join(", ") || null),
      ];
    case "exact_aggregate_average_age":
      return [...recorded("Expected age", record.expected_average_age), ...recorded("Observed age", record.observed_average_age), ...recorded("Absolute error", record.absolute_error, "lower is better"), ...recorded("Correct", record.correct)];
    case "precision_at_result_limit":
      return [...recorded("Precision", record.precision, "higher is better"), ...recorded("Returned rows", record.returned_rows), ...recorded("Correct rows", record.correct_rows)];
    case "ratio_absolute_error":
      return [...recorded("Expected ratio", record.expected_ratio), ...recorded("Observed ratio", record.observed_ratio), ...recorded("Absolute error", record.absolute_error, "lower is better")];
    case "artwork_selection_and_century":
      return [...recorded("Selection F1", record.quality_selection_f1, "higher is better"), ...recorded("Century accuracy", record.quality_century_accuracy, "higher is better"), ...recorded("Checked century rows", record.quality_century_checked_rows)];
    case "exact_numeric_field_accuracy":
      return [...recorded("Correct fields", record.correct_fields), ...recorded("Checked fields", record.checked_fields), ...recorded("Legacy scorer output", record.accuracy, "higher is better")];
    default:
      return [];
  }
}

export function primaryQuality(record: BenchmarkRecord): PrimaryQuality | null {
  if (["top_group_set_precision_recall", "patient_id_set_precision_recall", "pair_precision_recall", "artwork_title_set_precision_recall"].includes(record.quality_metric) && record.f1 !== null) return { label: "F1 score", value: record.f1, direction: "higher is better" };
  if (record.quality_metric === "precision_at_result_limit" && record.precision !== null) return { label: "Precision", value: record.precision, direction: "higher is better" };
  if (["exact_aggregate_count", "exact_aggregate_average_age", "ratio_absolute_error"].includes(record.quality_metric)) {
    const value = record.quality_metric === "exact_aggregate_count" && record.expected_count !== null && record.observed_count !== null ? Math.abs(record.observed_count - record.expected_count) : record.absolute_error;
    return value === null ? null : { label: "Absolute error", value, direction: "lower is better" };
  }
  if (record.quality_metric === "artwork_selection_and_century" && record.quality_selection_f1 !== null) return { label: "Selection F1", value: record.quality_selection_f1, direction: "higher is better", secondary: record.quality_century_accuracy === null ? undefined : { label: "Century accuracy", value: record.quality_century_accuracy } };
  return null;
}

export const qualityFamily = (record: BenchmarkRecord) => ({
  exact_aggregate_count: "count error",
  exact_aggregate_average_age: "average error",
  ratio_absolute_error: "ratio error",
  precision_at_result_limit: "precision",
  artwork_selection_and_century: "selection + century",
  exact_numeric_field_accuracy: "legacy audit",
}[record.quality_metric] ?? "precision / recall / F1");

export const qualitySummary = (record: BenchmarkRecord) => {
  if (record.quality_metric === "exact_numeric_field_accuracy") return "Audit-only legacy scorer";
  const primary = primaryQuality(record);
  return primary ? `${primary.label}: ${primary.value.toLocaleString(undefined, { maximumFractionDigits: 3 })}` : "No comparable quality value";
};
