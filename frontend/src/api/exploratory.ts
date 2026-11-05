import type { DatasetMeta, DatasetSummary, ScalingAccepted, SplitPlanSummary } from "./types";
import { failureMessage, waitForJob } from "./jobs";

export type AnalysisScope = "full_dataset" | "custom_date_range" | "train_span" | "rolling";
export type MissingPolicy = "pairwise_complete" | "drop_any_missing" | "median_impute_scope" | "mean_impute_scope" | "leave_nan";
export type PreTransform = "none" | "first_difference" | "percent_change" | "log_difference" | "absolute_return" | "squared_return" | "rank_transform" | "winsorize_1_99" | "robust_scale" | "standard_scale" | "minmax_scale";
export type ConstantPolicy = "exclude_constants" | "warn_only" | "refuse_if_constants";
export type CorrelationMethod = "pearson" | "spearman" | "kendall" | "partial_pearson" | "partial_spearman";
export interface ExploratoryOptions {
  datasets: DatasetSummary[];
  dataset_id: string | null;
  features: string[];
  columns: string[];
  time_column: string | null;
  plans: SplitPlanSummary[];
  methods: CorrelationMethod[];
  warnings: string[];
  feature_groups: Record<string, string[]>;
  ignored_columns: string[];
  computational_limits: Record<string, number>;
  target_candidates: string[];
  group_by_candidates: string[];
  has_close_time: boolean;
  supported_dataset_categories: string[];
  dataset_naming: "canonical_auto";
  default_parameters: Record<string, unknown>;
}
export interface ScopeRequest {
  dataset_id: string;
  features: string[];
  scope: AnalysisScope;
  start_time?: string;
  end_time?: string;
  split_plan_id?: string;
  fold?: number;
  pre_transform?: PreTransform;
  missing_policy?: MissingPolicy;
  constant_policy?: ConstantPolicy;
  min_obs?: number;
}
export interface CorrelationRequest extends ScopeRequest {
  method: CorrelationMethod;
  target: string | null;
  controls: string[];
  group_column: string | null;
  redundancy_threshold: number;
  max_lag: number;
  rolling_window: number | null;
  group_min_obs?: number;
  group_bins?: number[];
  alpha?: number;
  confidence_level?: number;
  multiple_testing?: "none" | "bonferroni" | "benjamini_hochberg";
  compute_p_values?: boolean;
  compute_confidence_intervals?: boolean;
  ci_method?: "bootstrap" | "none";
  calculate_vif?: boolean;
  vif_threshold?: number;
  include_intercept?: boolean;
  enable_ccf?: boolean;
  lag_min?: number;
  lag_max?: number;
  ccf_min_obs?: number;
  enable_rolling?: boolean;
  rolling_step?: number;
  rolling_min_periods?: number;
  rolling_pair_selection?: "all_pairs" | "top_n" | "selected_pairs";
  rolling_top_pairs?: number;
  rolling_pairs?: [string, string][];
  enable_redundancy_groups?: boolean;
  strict_causal_mode?: boolean;
}
export interface PCARequest extends ScopeRequest {
  standardizer: "none" | "standard" | "robust" | "minmax";
  component_selection: "manual" | "variance_threshold" | "eigenvalue_gt_1" | "scree" | "parallel";
  n_components: number;
  variance_threshold: number;
  write_dataset: boolean;
  strict_causal: boolean;
  keep_original_features: boolean;
  carry_columns: string[];
  name: string | null;
  destination?: "outputs" | "processed_data";
  whiten?: boolean;
  component_prefix?: string;
  parallel_permutations?: number;
  include_target_feature?: boolean;
  include_carried_columns?: boolean;
  target_feature?: string | null;
}
export interface AnalysisResult {
  dataset_id: string;
  dataset_name: string;
  features: string[];
  fit_scope: AnalysisScope;
  causal_for_downstream_training: boolean;
  n_scope_rows: number;
  time_column: string | null;
  split_plan_id: string | null;
  fold: number | null;
  scope_start: string | null;
  scope_end: string | null;
  warnings: string[];
  notes: string[];
  excluded_constants: string[];
  missing_summary: Record<string, unknown>;
  preprocessing_parameters: Record<string, unknown>;
  metadata: Record<string, unknown>;
  fit_rows: number;
  transform_rows: number;
}
export interface CorrelationPair {
  feature_x: string;
  feature_y: string;
  correlation: number | null;
  p_value: number | null;
  ci_low: number | null;
  ci_high: number | null;
  n_used: number;
  feature_a: string;
  feature_b: string;
  abs_correlation: number | null;
  adjusted_p_value: number | null;
  ci_method: string;
  significant: boolean;
  redundancy_flag: boolean;
}
export interface CorrelationResult extends AnalysisResult {
  method: CorrelationMethod;
  matrix: (number | null)[][];
  pairwise: CorrelationPair[];
  vif: { feature: string; vif: number | null; n_used: number; status: string; tolerance: number | null; condition_flag: string; suggested_action: string }[];
  redundancy_groups: { group_id: number; members: string[]; representative: string; min_pairwise_abs_correlation: number; max_pairwise_abs_correlation: number; average_pairwise_abs_correlation: number; suggested_action: string }[];
  redundancy_threshold: number;
  ccf: (CorrelationPair & { lag: number })[];
  rolling: { feature: string; time: string; correlation: number | null; n_used: number }[];
  conditional: { group: string | null; rows: number; pairwise: CorrelationPair[]; matrix: (number | null)[][]; warnings: string[] }[];
  ccf_summary: { feature: string; best_positive_lag: number | null; best_negative_lag: number | null; max_absolute_predictive_correlation: number | null }[];
  rolling_summary: { feature_a: string; feature_b: string; mean_corr: number | null; std_corr: number | null; min_corr: number | null; max_corr: number | null; latest_corr: number | null; stability_score: number | null; n_points: number }[];
  multiple_testing_method: string;
}
export interface PCAResult extends AnalysisResult {
  n_components: number;
  n_fit_rows: number;
  standardizer: string;
  component_selection: string;
  eigenvalues: number[];
  explained_variance: number[];
  explained_variance_ratio: number[];
  cumulative_explained_variance_ratio: number[];
  loadings: number[][];
  component_summary: { component: string; eigenvalue: number; explained_variance_ratio: number;
    cumulative_ratio: number; selected: boolean }[];
  biplot: { scores: number[][]; loadings: number[][]; row_indices: number[]; times?: string[] };
  dataset: DatasetMeta | null;
  params_dataset: DatasetMeta | null;
  component_columns: string[];
  reconstruction_error: number | null;
  n_nan_scores: number;
  variance_threshold: number;
}

/** A read-only preview can be immediate; saves always return a job. */
export async function resolveAnalysis<T extends AnalysisResult>(value: T | ScalingAccepted, signal?: AbortSignal): Promise<T> {
  if (!("job_id" in value)) return value;
  const job = await waitForJob<T>(value.job_id, signal ? { signal } : {});
  if (job.status === "failed" || !job.result) throw new Error(failureMessage(job));
  return job.result;
}

export function featuresInGroups(groups: Record<string, string[]>, selected: string[]): string[] {
  return [...new Set(selected.flatMap((group) => groups[group] ?? []))];
}

export function scopeIsReady(scope: AnalysisScope, timeColumn: string | null, start: string, end: string, hasPlan: boolean): boolean {
  if (scope === "full_dataset") return true;
  if (!timeColumn) return false;
  if (scope === "rolling") return true;
  if (scope === "train_span") return hasPlan;
  return start !== "" && end !== "" && new Date(start).getTime() <= new Date(end).getTime();
}

export function heatmapValues(matrix: (number | null)[][], triangle: string, threshold: number): (number | null)[][] {
  return matrix.map((row, i) => row.map((value, j) => (triangle === "upper" && i > j)
    || (triangle === "lower" && i < j) || (value !== null && Math.abs(value) < threshold) ? null : value));
}
