import { request } from "./client";
import { getActiveProjectId } from "./projectSelection";
import type { DatasetSummary } from "./types";

export interface Parameter {
  type: "number" | "select" | "column" | "json";
  default: unknown;
  min?: number | null;
  max?: number | null;
  options?: string[];
}
export interface ModelSpec {
  id: string; name: string; family: string; parameters: Record<string, Parameter>;
  available: boolean; availability_reason: string | null; probabilities: boolean;
  state_count: boolean; decoding: boolean; criteria: string[];
}
export interface Options {
  models: ModelSpec[]; datasets: DatasetSummary[]; plans: DatasetSummary[];
  features: string[]; columns: string[]; preprocessing: Record<string, unknown>[];
  optimization: Record<string, boolean>;
}
export interface ExperimentConfig {
  name: string;
  inputs: { dataset_id: string; split_plan_id: string; features: string[]; fold: number;
    market_dataset_id?: string | null; price_column?: string | null; return_column?: string | null;
    return_type?: "simple" | "log" | null;
    adx_column?: string | null; rsi_column?: string | null; volatility_column?: string | null };
  model: string; parameters: Record<string, unknown>; seed: number;
  states: { mode: string; count: number; candidates: number[]; template: string; minimum_occupancy: number; criterion: string };
  target: { source: string; column?: string | null; rule_feature?: string | null; objective: string; horizon: number; availability_lag: number; lower: number; upper: number } | null;
  search: { method: string; space: Record<string, unknown[]>; trials: number };
}
export interface Accepted { job_id: string; status: string; poll: string }
export interface Artifact {
  id: string; kind: string; name: string; created_at: string; model?: string;
  config?: ExperimentConfig; diagnostics?: Record<string, unknown>; state_count?: number; fitted_parameters?: Record<string, unknown>;
  contract?: { asset: string; timeframe: string; features: string[]; train_start: string; train_end: string; preprocessing: unknown[] };
  labels?: Record<string, string>; experiment_id?: string; approval_id?: string; selected_experiment_id?: string;
  trials?: Record<string, unknown>[]; experiment_ids?: string[]; validation_id?: string;
  rows?: Record<string, unknown>[]; folds?: Record<string, unknown>[]; test?: Partition;
  snapshot?: Snapshot; monitoring?: Monitoring; transitions?: Transition;
  events?: Record<string, unknown>[]; causal?: boolean; observation_count?: number;
  decoding_note?: string; integrity_error?: boolean; rationale?: string; version?: string; request?: Record<string, unknown>;
}
export interface Profile extends Record<string, unknown> {
  state_id: number; count: number; occupancy: number; mean_return: number | null;
  return_volatility: number | null; mean_duration: number | null; feature_means: Record<string, number>;
}
export interface Transition {
  probabilities: number[][]; fitted_probabilities?: number[][] | null;
  switching_frequency: number; expected_duration: (number | null)[];
  empirical_duration: (number | null)[]; events: Record<string, unknown>[];
  durations: { state_id: number; bars: number; censored: boolean }[];
  unavailable?: string;
}
export interface Partition {
  occupancy_note?: string;
  observations: number; profiles: Profile[]; transitions: Transition;
  predictive: Record<string, unknown>; separation?: Record<string, unknown>;
  log_likelihood_per_observation?: number;
}
export interface EvaluationResult {
  experiment_id: string; model: string; diagnostics: Record<string, unknown>; fitted_parameters?: Record<string, unknown>;
  partitions: Record<string, Partition>; labels: Record<string, string>; label_proposal: Record<string, string>;
  state_distribution_drift: Record<string, number | null>; economic_note: string; structural_note: string | null;
  charts: { timestamps: string[]; state_ids: number[]; price: (number | null)[] | null;
    returns: (number | null)[] | null; return_type?: "simple" | "log" | null; volatility: (number | null)[] | null; probabilities: number[][] | null; sampling: string };
}
export interface Snapshot {
  state_id: number; raw_state_id: number; regime: string; trend: string | null; volatility: string | null;
  probabilities: number[] | null; distances: number[] | null; confidence: number | null;
  duration_bars: number; previous_state: number | null; last_transition: Record<string, unknown> | null;
  model: string; version: string; asset: string; timeframe: string; timestamp: string; inferred_at: string;
  available_at: string; objective: string; horizon: number; retrospective: boolean; live_current_regime: boolean;
}
export interface Monitoring {
  warnings: { kind: string; message: string }[]; feature_drift: { feature: string; drift: number; method: string }[];
  state_occupancy_drift: number | null; cusum_events: Record<string, unknown>[];
  performance: { available: boolean; reason?: string; metrics?: Record<string, unknown> };
}
export interface History { rows: Record<string, unknown>[]; total: number; offset: number }
const post = <T,>(path: string, body: unknown) => request<T>(`/regime${path}`, { method: "POST", body: JSON.stringify(body) });
export const regimes = {
  options: (id = "") => request<Options>(`/regime/options${id ? `?dataset_id=${encodeURIComponent(id)}` : ""}`),
  plan: (id: string) => request<{ fold: number; role: string; start_time: string; end_time: string; row_count: number }[]>(`/regime/plans/${id}`),
  fit: (config: ExperimentConfig) => post<Artifact | Accepted>("/experiments", config),
  list: (kind = "experiment") => request<Artifact[]>(`/regime/artifacts?kind=${kind}`),
  get: (id: string) => request<Artifact>(`/regime/artifacts/${id}`),
  history: (id: string, offset = 0) => request<History>(`/regime/artifacts/${id}/history?offset=${offset}`),
  evaluate: (id: string) => request<EvaluationResult>(`/regime/experiments/${id}/evaluation`),
  label: (id: string, body: unknown) => post<Artifact>(`/experiments/${id}/labels`, body),
  validate: (id: string, body: unknown) => post<Accepted>(`/experiments/${id}/validation`, body),
  robustness: (id: string, body: unknown) => post<Accepted>(`/experiments/${id}/robustness`, body),
  approve: (id: string, body: unknown) => post<Artifact>(`/experiments/${id}/approval`, body),
  compare: (ids: string[]) => post<{ rows: Record<string, unknown>[]; comparable: boolean; note: string }>("/comparison", { experiment_ids: ids }),
  infer: (body: unknown) => post<Accepted>("/inference", body),
  exportUrl: (id: string, format: string) => `/api/regime/artifacts/${id}/export?${new URLSearchParams({ format, project_id: getActiveProjectId() })}`,
  bundleUrl: (id: string) => `/api/regime/artifacts/${id}/bundle?${new URLSearchParams({ project_id: getActiveProjectId() })}`,
};
