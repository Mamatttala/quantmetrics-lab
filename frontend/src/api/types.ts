/**
 * Typed mirrors of the backend's Pydantic DTOs.
 *
 * These are hand-written rather than generated. There is no schema-generation step
 * in this project, so the contract is kept honest by the backend tests asserting
 * exact field names and by `exactOptionalPropertyTypes` catching drift at the
 * call site. Nullable fields are `| null`, never optional — the backend commits to
 * emitting the key.
 */

export type DatasetCategory =
  | "raw_data"
  | "uploaded_data"
  | "processed_data"
  | "outputs";

export type Provider = "binance" | "yahoo" | "upload" | "derived";

export type AssetClass =
  | "crypto_spot"
  | "crypto_futures"
  | "equity_spot"
  | "equity_futures";

export type Timeframe = "1m" | "5m" | "15m" | "30m" | "1h" | "4h" | "1d" | "1w";

export const TIMEFRAMES: readonly Timeframe[] = [
  "1m",
  "5m",
  "15m",
  "30m",
  "1h",
  "4h",
  "1d",
  "1w",
];

export const CATEGORY_LABELS: Record<DatasetCategory, string> = {
  raw_data: "Raw Data",
  uploaded_data: "Uploaded Data",
  processed_data: "Processed Data",
  outputs: "Outputs",
};

export const CATEGORY_ORDER: readonly DatasetCategory[] = [
  "raw_data",
  "uploaded_data",
  "processed_data",
  "outputs",
];

export interface ColumnMeta {
  name: string;
  dtype: string;
  nullable: boolean;
}

export interface DatasetSummary {
  dataset_id: string;
  name: string;
  category: DatasetCategory;
  source: Provider;
  symbol: string | null;
  timeframe: Timeframe | null;
  row_count: number;
  start_timestamp: string | null;
  end_timestamp: string | null;
  created_at: string;
  parent_dataset_id: string | null;
  /** The operation that produced this dataset, lifted out of its provenance. Null for a
   *  dataset that was fetched or uploaded rather than computed here.
   *
   *  It is on the summary rather than only in the detail because it is what lets C6 and C7
   *  hide their own artifacts from their dataset pickers: a split plan is a table of row
   *  indices, and offering it as something to split would produce a plan over a plan. */
  operation: string | null;
}

export interface DatasetMeta extends DatasetSummary {
  asset_class: AssetClass;
  market_type: string | null;
  timezone: string | null;
  columns: ColumnMeta[];
  checksum: string;
  storage_path: string;
  unpopulated_fields: string[];
  provenance: Record<string, string>;
}

export interface PreviewResponse {
  dataset_id: string;
  row_count: number;
  columns: ColumnMeta[];
  head: Record<string, unknown>[];
  tail: Record<string, unknown>[];
}

export interface PageResponse {
  dataset_id: string;
  offset: number;
  limit: number;
  total: number;
  rows: Record<string, unknown>[];
}

export interface CellEdit {
  row: number;
  column: string;
  value: unknown;
}

export interface ColumnOp {
  op: "rename" | "drop" | "add" | "reorder";
  column: string;
  new_name?: string | null;
  /** 0-based place in the *final* schema. Only meaningful for `reorder`, which the
   *  backend applies after every rename and add so it can name the column as it will
   *  finally be called. */
  position?: number | null;
}

export interface DeriveRequest {
  name?: string | null;
  operation?: string;
  edits: CellEdit[];
  column_ops: ColumnOp[];
}

export interface FetchResult {
  dataset: DatasetMeta;
  deduplicated: boolean;
  warnings: string[];
}

export interface FetchRun {
  run_id: string;
  provider: string;
  endpoint: string;
  params: string;
  status: string;
  dataset_id: string | null;
  message: string | null;
  started_at: string;
  finished_at: string | null;
}

/** A single OHLCV bar. `time` is a UNIX timestamp in **seconds**, which is what
 *  lightweight-charts requires — it rejects milliseconds, and a milliseconds value
 *  renders as a bar in the year 57000, i.e. an empty chart. The backend converts at
 *  the source so there is no factor-of-1000 decision left at the chart layer. */
export interface OhlcvBar {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface OhlcvResponse {
  dataset_id: string;
  bars: OhlcvBar[];
  total: number;
  truncated: boolean;
}

export type QualityKind =
  | "duplicate_timestamp"
  | "timestamp_out_of_order"
  | "missing_value"
  | "missing_bar"
  | "invalid_dtype"
  | "invalid_numeric"
  | "non_positive_price"
  | "high_below_low"
  | "extreme_return";

/**
 * Every key the backend's `counts` map can carry, with the sentence the quality
 * panel shows for it. One list rather than one per page: Raw Data View and Cleaning
 * read the same report, and a check that appeared on one page and not the other
 * would read as "not checked" on the page that omitted it.
 *
 * `invalid_dtype` and `invalid_numeric` count *columns* (and unparseable values),
 * not rows, which is why their labels say so — a column count rendered under a
 * row-shaped heading invites exactly the wrong reading of a report.
 */
export const QUALITY_LABELS: Record<QualityKind, string> = {
  duplicate_timestamp: "Duplicate timestamp",
  timestamp_out_of_order: "Out of order",
  missing_value: "Missing value",
  missing_bar: "Missing bar",
  invalid_dtype: "Unsupported column type",
  invalid_numeric: "Non-numeric value in a text column",
  non_positive_price: "Non-positive price",
  high_below_low: "high < low",
  extreme_return: "Extreme return",
};

/** The order the report is read in: structure of the timeline first, then the
 *  columns, then the values. Declared once so the panel cannot shuffle itself
 *  between renders. */
export const QUALITY_ORDER: readonly QualityKind[] = [
  "duplicate_timestamp",
  "timestamp_out_of_order",
  "missing_value",
  "missing_bar",
  "invalid_dtype",
  "invalid_numeric",
  "non_positive_price",
  "high_below_low",
  "extreme_return",
];

export interface QualityMarker {
  /** UNIX seconds, or `null` when the row's timestamp is unparseable. Such a marker
   *  cannot be placed on the chart but is still reported in the list — a defective
   *  row with a broken timestamp is precisely what the checks are for. */
  time: number | null;
  kind: QualityKind;
  detail: string;
}

export interface QualityColumn {
  name: string;
  dtype: string;
  missing: number;
  /** False when the column's type is outside the numeric/boolean/timestamp/text
   *  vocabulary the rest of the app computes on. */
  supported: boolean;
}

export interface QualityResponse {
  dataset_id: string;
  time_column: string;
  markers: QualityMarker[];
  counts: Record<string, number>;
  /** True when a check could not run because the dataset lacks the needed columns,
   *  so the UI can say "not checked" rather than implying "clean". */
  skipped: string[];
  /** Detail behind a count that has no per-row marker — a check that ran and has
   *  something to say. Kept separate from `skipped` because the two are different
   *  claims: rendering a finding under "not checked" reports the report as not having
   *  run at the moment it produced a number. */
  notes: string[];
  columns: QualityColumn[];
}

export const MARKET_OPTIONS: readonly {
  value: AssetClass;
  label: string;
  provider: "binance" | "yahoo";
}[] = [
  { value: "crypto_spot", label: "Crypto · Spot", provider: "binance" },
  { value: "crypto_futures", label: "Crypto · Futures", provider: "binance" },
  { value: "equity_spot", label: "Equity · Cash", provider: "yahoo" },
  { value: "equity_futures", label: "Equity · Futures", provider: "yahoo" },
];

export const CRYPTO_DATA_TYPES = ["klines", "funding", "metrics"] as const;
export type CryptoDataType = (typeof CRYPTO_DATA_TYPES)[number];

/** Timeframes each market/type combination actually serves. Mirrors the backend's
 *  frozensets so a doomed request is not offered in the first place. */
export function timeframesFor(
  assetClass: AssetClass,
  dataType: CryptoDataType,
): readonly Timeframe[] {
  if (assetClass === "equity_spot" || assetClass === "equity_futures") {
    return ["1h", "4h", "1d"];
  }
  if (dataType === "metrics") return ["5m", "15m", "30m", "1h", "4h", "1d"];
  if (dataType === "funding") return [];
  return TIMEFRAMES;
}

export interface ApiErrorPayload {
  error: {
    code: string;
    message: string;
    details: Record<string, unknown>;
  };
}

export const ERROR_CODES = {
  validationFailure: "validation_failure",
  spanTooWide: "span_too_wide",
  unsupportedRequest: "unsupported_request",
  datasetNotFound: "dataset_not_found",
  immutableDataset: "immutable_dataset",
  upstreamError: "upstream_error",
} as const;

/* ================================================================= Part B ===
 * Cleaning (B3) and Resample & Merge (B4).
 *
 * The policy unions below mirror `app/models/enums.py` exactly. Where this project
 * offers a *different* set of options than the specification listed, the deviation is
 * recorded on the option itself (`caution`) and in DECISIONS.md, because a menu is
 * where a methodological choice becomes invisible: the user sees "Forward Fill" and
 * not "this one is safe, that one is a look-ahead".
 */

export type DuplicatePolicy =
  | "keep"
  | "drop_exact"
  | "keep_first_by_time"
  | "keep_last_by_time";

export type MissingValuePolicy =
  | "keep"
  | "ffill"
  | "bfill"
  | "interpolate_time"
  | "zero"
  | "drop_rows";

export type OutlierPolicy = "keep" | "winsorize" | "remove" | "iqr";

export type AlignMethod = "backward" | "exact";

export type MergeHow = "left" | "inner";

/** One selectable option, with the sentence that has to appear next to it. */
export interface PolicyOption<T extends string> {
  value: T;
  label: string;
  detail: string;
  /** Present when the choice can introduce look-ahead or discard observations. */
  caution?: string;
}

export const DUPLICATE_OPTIONS: readonly PolicyOption<DuplicatePolicy>[] = [
  {
    value: "keep",
    label: "Do nothing",
    detail: "Leave repeated rows exactly as they are.",
  },
  {
    value: "drop_exact",
    label: "Remove exact duplicate rows",
    detail:
      "Drop a row only when every column matches an earlier row. Rows sharing a timestamp but differing in a value are kept.",
  },
  {
    value: "keep_first_by_time",
    label: "One row per timestamp — keep the first",
    detail:
      "Also removes rows that share a timestamp but disagree, keeping the first of them.",
    caution:
      "Not in the specification's list, which offers only exact-duplicate removal. A repeated timestamp carrying different values is exactly what the quality report flags as a duplicate timestamp, and exact-duplicate removal cannot repair it.",
  },
  {
    value: "keep_last_by_time",
    label: "One row per timestamp — keep the last",
    detail:
      "As above, but keeps the last row for each timestamp. Useful when a later row is a correction of an earlier one.",
    caution:
      "Also not in the specification's list. Choose deliberately: which row is kept depends on the file's row order unless you also sort by time.",
  },
];

export const MISSING_OPTIONS: readonly PolicyOption<MissingValuePolicy>[] = [
  {
    value: "keep",
    label: "Do nothing",
    detail: "Leave empty cells empty. Every check downstream will still report them.",
  },
  {
    value: "ffill",
    label: "Forward fill",
    detail:
      "Carry the last known value forward into the gap. A leading empty cell has nothing before it and stays empty.",
  },
  {
    value: "bfill",
    label: "Backward fill",
    detail:
      "Carry the next known value backwards into the gap. The only fill that can repair a hole at the very start.",
    caution:
      "A filled cell holds a value that had not been observed at that time. If this dataset feeds a backtest, that is look-ahead.",
  },
  {
    value: "interpolate_time",
    label: "Time-based linear interpolation",
    detail:
      "Draw a straight line between the two known values either side of the hole, weighted by the actual time gap. Leading and trailing holes are left empty, since filling them is extrapolation rather than interpolation.",
    caution:
      "Interpolated cells are fabricated values that were never observed. Needs a parseable timestamp on every row.",
  },
  {
    value: "zero",
    label: "Replace with zero",
    detail: "Substitute 0 for every empty numeric cell.",
    caution:
      "On a price column a zero is a −100% return, not a neutral value. Only safe where zero is a real observation — open interest or volume, for instance.",
  },
  {
    value: "drop_rows",
    label: "Remove rows with missing values",
    detail:
      "Drop any row containing at least one empty cell, including in columns unrelated to the one you were looking at.",
  },
];

export const OUTLIER_OPTIONS: readonly PolicyOption<OutlierPolicy>[] = [
  {
    value: "keep",
    label: "Do nothing",
    detail: "Leave every value as it is.",
  },
  {
    value: "winsorize",
    label: "Winsorization",
    detail:
      "Clamp values outside the chosen lower and upper quantiles to the fence. The values are pulled in, not deleted, so the row count is unchanged.",
  },
  {
    value: "iqr",
    label: "Interquartile range (IQR)",
    detail:
      "Clamp values outside Q1 − k·IQR and Q3 + k·IQR. The fences come from the data's own spread rather than from a fixed quantile.",
  },
  {
    value: "remove",
    label: "Removing extremes",
    detail:
      "Delete the rows that fall outside the fence.",
    caution:
      "This leaves holes in the timeline, which the missing-bar check will then report. The row count drops.",
  },
];

export const ALIGN_OPTIONS: readonly PolicyOption<AlignMethod>[] = [
  {
    value: "backward",
    label: "Most recent value at or before the bar",
    detail:
      "Each bar takes the latest value published at or before its timestamp. A funding rate quoted at 00:00 reaches the 03:00 bar; one quoted at 04:00 does not.",
  },
  {
    value: "exact",
    label: "Exact timestamp match",
    detail:
      "Keep only bars whose timestamp appears in both datasets. Everything else is dropped.",
  },
];

export const HOW_OPTIONS: readonly PolicyOption<MergeHow>[] = [
  {
    value: "left",
    label: "Keep every base bar",
    detail: "Bars that found no match keep an empty cell in the joined columns.",
  },
  {
    value: "inner",
    label: "Keep only matched bars",
    detail: "Bars with no counterpart are dropped, so the result has fewer rows.",
  },
];

/** One join in a merge request. */
export interface JoinRequest {
  dataset_id: string;
  how: MergeHow;
  method: AlignMethod;
  prefix: string;
  /** Maximum staleness accepted, in seconds. `null` accepts a value of any age. */
  tolerance_seconds: number | null;
}

export interface CleanRequest {
  name: string | null;
  duplicates: DuplicatePolicy;
  missing: MissingValuePolicy;
  outliers: OutlierPolicy;
  outlier_columns: string[];
  lower_quantile: number;
  upper_quantile: number;
  iqr_multiplier: number;
  sort_by_time: boolean;
  /** `false` runs the plan and returns the report and preview without storing a
   *  dataset. That is what makes the dry run use the same code path as the save. */
  write: boolean;
}

export interface MergeRequest {
  base_dataset_id: string;
  name: string | null;
  timeframe: Timeframe | null;
  joins: JoinRequest[];
  column_ops: ColumnOp[];
  write: boolean;
}

/** The JupyterLab-style window, without the dataset id the stored `/preview` adds. */
export interface HeadTailPreview {
  row_count: number;
  columns: ColumnMeta[];
  head: Record<string, unknown>[];
  tail: Record<string, unknown>[];
}

export interface OutlierFence {
  lower: number;
  upper: number;
  rule: string;
  /** How many values fell outside the fence, counted before the repair. */
  outside: number;
}

export interface CleaningStep {
  step: string;
  policy: string;
  detail?: string;
  warnings?: string[];
  rows_removed?: number;
  filled?: number;
  cells_changed?: number;
  columns?: string[];
  fences?: Record<string, OutlierFence>;
}

export interface CleaningReport {
  time_column: string;
  rows_before: number;
  rows_after: number;
  rows_removed: number;
  columns_before: string[];
  columns_after: string[];
  steps: CleaningStep[];
  warnings: string[];
}

export interface ResampleReport {
  timeframe: string;
  rows_before: number;
  rows_after: number;
  detail?: string;
  /** The aggregation applied per column, so the OHLC rules are inspectable. */
  aggregations: Record<string, string>;
  warnings: string[];
}

export interface JoinReport {
  dataset_id: string;
  label: string;
  method: AlignMethod;
  how: MergeHow;
  prefix: string;
  tolerance_seconds: number | null;
  columns_added: string[];
  rows: number;
  unmatched: number;
  detail?: string;
  warnings: string[];
}

export interface MergeReport {
  base: { dataset_id: string; name: string };
  resample: ResampleReport | null;
  joins: JoinReport[];
  rows_before: number;
  rows_after: number;
  column_ops?: string[];
  warnings: string[];
}

/** `dataset` is null for a dry run — the same shape either way, so the preview a user
 *  approves is produced by the code that would do the work. */
export interface CleanResult {
  dataset: DatasetMeta | null;
  report: CleaningReport;
  preview: HeadTailPreview;
}

export interface MergeResult {
  dataset: DatasetMeta | null;
  report: MergeReport;
  preview: HeadTailPreview;
}

// ============================================================ C5 · Feature engineering

/**
 * Which price series every price-derived feature is computed from.
 *
 * `weighted` is one series with a resolved definition rather than a fixed formula: it is
 * the exact bar VWAP when the dataset carries both `volume` and `quote_volume`, and the
 * typical price otherwise. The report says which, because the two are not the same number.
 */
export type PriceSource = "close" | "weighted";

export const PRICE_SOURCE_OPTIONS: readonly PolicyOption<PriceSource>[] = [
  {
    value: "close",
    label: "Closing price",
    detail:
      "Use the bar's closing price for every feature that needs a price. This is the value a daily bar is conventionally quoted at.",
  },
  {
    value: "weighted",
    label: "Volume-weighted price",
    detail:
      "Use a volume-weighted average for the bar. Where the dataset carries both volume and quote volume this is the exact traded average; otherwise it is the typical price (high + low + close) / 3, which is an estimator. The report states which one was used.",
    caution:
      "Only the exact form weighs by real volume. On a dataset without quote volume the typical price is a midpoint estimate, not a traded average.",
  },
];

export type FeatureGroup =
  | "price"
  | "volume"
  | "returns"
  | "volatility"
  | "indicators"
  | "time"
  | "exogenous";

export const FEATURE_GROUP_LABELS: Record<FeatureGroup, string> = {
  price: "Price",
  volume: "Volume",
  returns: "Returns",
  volatility: "Volatility",
  indicators: "Indicators",
  time: "Time",
  exogenous: "Exogenous",
};

/** Whether selecting an entry computes a column or keeps one that is already there. */
export type FeatureKind = "computed" | "carried";

/** How a feature's single varying parameter is expressed. */
export type WindowKind = "bars" | "duration";

export interface FeatureParam {
  kind: WindowKind;
  /** The tick options. A duration token (`"4h"`) or a bar count (`"20"`), as text. */
  ticks: string[];
  default: string;
  /** Smallest window the feature is defined for, in bars, after any conversion. */
  minimum: number;
  /** Whether the user may type a value the ticks do not list. */
  custom: boolean;
}

export interface FeatureWindow {
  /** What the user picked, and what the request sends back. */
  token: string;
  /** Resolved window length in bars of *this* dataset, or null when unresolvable. */
  bars: number | null;
  /** Warm-up rows for this window. Null exactly when `bars` is null. */
  warmup_bars: number | null;
  available: boolean;
  reason: string | null;
}

export interface FeatureSpec {
  id: string;
  label: string;
  group: FeatureGroup;
  kind: FeatureKind;
  /** Shown verbatim in the info panel — the backend's own statement of the formula. */
  formula: string;
  detail: string;
  /** Column name templates. `{arg}` stands in for the window. */
  columns: string[];
  /** Source columns required from the dataset. */
  needs: string[];
  needs_price: boolean;
  caution: string | null;
  /** Warm-up rows for a feature with no parameter; null when it has one. */
  warmup_bars: number | null;
  available: boolean;
  reason: string | null;
  param: FeatureParam | null;
  windows: FeatureWindow[];
}

export interface FeatureGroupPayload {
  group: FeatureGroup;
  features: FeatureSpec[];
}

export interface FeatureCatalog {
  dataset_id: string;
  asset_class: AssetClass;
  timeframe: Timeframe | null;
  timezone: string | null;
  columns: string[];
  unpopulated_fields: string[];
  has_time: boolean;
  groups: FeatureGroupPayload[];
  available_count: number;
  total_count: number;
}

/** One request line: a feature, and the window it was given. */
export interface FeatureSelection {
  feature_id: string;
  /** Null for a feature with no parameter. */
  window: string | null;
}

export interface FeatureRequest {
  name: string | null;
  selections: FeatureSelection[];
  price_source: PriceSource;
  /** Store only the generated columns (plus the time column) instead of adding to them. */
  keep_only_selected: boolean;
  write: boolean;
}

export interface FeatureAdded {
  feature_id: string;
  label: string;
  group: FeatureGroup;
  /** The token as requested. */
  window: string | null;
  /** The same window as a number of bars of this dataset. */
  window_bars: number | null;
  columns: string[];
  warmup_bars: number;
}

export interface CollinearityPair {
  a: string;
  b: string;
  r: number;
}

export interface FeatureReport {
  rows_before: number;
  rows_after: number;
  price_source: { source: PriceSource; definition: string; basis: string };
  added: FeatureAdded[];
  /** Columns kept as-is because they were already in the dataset. */
  carried: string[];
  generated_columns: string[];
  dropped_columns: string[];
  /** The longest warm-up among the selections: rows without a value for every feature. */
  warmup_bars: number;
  /** Timestamp of the first fully-warmed row, or null when there is none. */
  usable_from: string | null;
  time_column: string;
  timezone: string;
  notes: string[];
  collinearity: CollinearityPair[];
  /** The absolute correlation at or above which a pair is listed. */
  collinearity_threshold: number;
  selection_hash: string;
}

export interface FeatureResult {
  dataset: DatasetMeta | null;
  report: FeatureReport;
  preview: HeadTailPreview;
}

/* ============================================================ D · Regime detection

   Module D's contract. Two shapes here are worth reading before the rest:

   `RegimeOptions` is the *verdict* the config panel renders — which models and which
   observations this dataset can support, each with a reason when it cannot. It is computed
   server-side in one function and never re-derived here, so a control greyed out in the
   browser and a refusal from the run endpoint cannot disagree about why.

   `RegimeReport` is the diagnostics table. Every field is `| null` where the engine can
   legitimately have nothing to report — a run with no labelled rows has no BIC, and a JSON
   `null` is the honest encoding of that. `undefined` would be indistinguishable from a
   field this frontend had not been updated to read. */

export type RegimeModel = "hmm" | "gmm" | "hybrid" | "rules";

export type RegimeFeatureId =
  | "log_return"
  | "realized_volatility"
  | "volume_change"
  | "adx"
  | "rsi";

export type RegimeStateName =
  | "trending_up"
  | "trending_down"
  | "ranging"
  | "crisis"
  | "transition";

export type JobStatus = "running" | "succeeded" | "failed";

export interface RegimeFeatureOption {
  id: RegimeFeatureId;
  label: string;
  /** `computed` from the prices here, or `carried` in from C5. */
  kind: "computed" | "carried";
  /** The dataset columns this observation is built from. */
  needs: string[];
  available: boolean;
  /** Why not, when `available` is false. Never null when it is false. */
  reason: string | null;
}

export interface RegimeModelOption {
  id: RegimeModel;
  available: boolean;
  requires: string[];
  reason: string | null;
}

export interface RegimeWarning {
  /** `error` blocks the run; `warning` is a caution the user may overrule. */
  level: "error" | "warning";
  code: string;
  message: string;
}

export interface RegimeAnnualisation {
  available: boolean;
  /** Bars per year for this dataset's timeframe and asset class, or null when unknown. */
  periods_per_year: number | null;
  timeframe: Timeframe | null;
  reason: string;
}

export interface RegimeDefaults {
  n_states: number;
  min_state_duration: number;
  switching_cost_penalty: number;
  crisis_threshold: number;
  min_train_bars: number;
  refit_every: number;
  volatility_window: number;
}

export interface RegimeOptions {
  dataset_id: string;
  name: string;
  rows: number;
  timeframe: Timeframe | null;
  asset_class: AssetClass;
  columns: string[];
  features: RegimeFeatureOption[];
  available_features: RegimeFeatureId[];
  annualisation: RegimeAnnualisation;
  models: RegimeModelOption[];
  defaults: RegimeDefaults;
  warnings: RegimeWarning[];
}

export interface RegimeDetectRequest {
  dataset_id: string;
  model_type: RegimeModel;
  n_states: number;
  features: RegimeFeatureId[];
  min_state_duration: number;
  switching_cost_penalty: number;
  crisis_threshold: number;
  min_train_bars: number;
  refit_every: number;
  volatility_window: number;
  name: string | null;
  write: boolean;
}

export interface RegimeRunAccepted {
  job_id: string;
  status: JobStatus;
  /** Where to poll. Sent by the server so the path is not spelled out in two places. */
  poll: string;
}

/** A job record from the generic job registry.
 *
 *  `result` is parameterised because the registry is shared: Module D's jobs carry a
 *  `RegimeRunResult` and C7's carry a `ScalingCreated`, and the envelope around them is
 *  identical. A single non-generic type would have to say `RegimeRunResult | ScalingCreated`
 *  and every reader would then have to narrow a union that the job's own `kind` already
 *  decided — so the caller names what it is polling for instead. The default keeps D1's
 *  existing call sites unchanged. */
export interface JobRecord<TResult = RegimeRunResult> {
  job_id: string;
  kind: string;
  status: JobStatus;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  dataset_id: string | null;
  output_dataset_id: string | null;
  message: string | null;
  /** The refusal envelope, when `status` is `failed`. */
  error: { code: string; message: string; details: Record<string, unknown> } | null;
  result: TResult | null;
}

export interface RegimeStateDuration {
  /** How many times this state was committed to. */
  runs: number;
  /** Mean length of those runs, in bars. */
  mean_bars: number;
  /** The longest single run, in bars. */
  longest_bars: number;
}

export interface RegimeNetValueAdd {
  definition: string;
  /** Always true today. Present so a future out-of-sample figure can be told apart. */
  in_sample: boolean;
  bars_scored: number;
  /**
   * Bars the strategy actually held a position on. Zero means the rule never went long, so
   * `return_improvement` is exactly `-benchmark_log_return` and the figure says nothing
   * about the model — which the panel has to say out loud rather than present as a score.
   */
  bars_long: number;
  strategy_log_return: number | null;
  benchmark_log_return: number | null;
  return_improvement: number | null;
  /**
   * Position entries and exits — *not* the report's `switch_count`, which counts committed
   * state changes. Moving from ranging to crisis is a state change that costs nothing.
   */
  position_changes: number;
  switching_cost_penalty: number;
  switching_cost: number | null;
  net_value_add: number | null;
}

export interface RegimeStationarity {
  feature: string;
  /** Null when the series was too short or constant for the test to say anything. */
  stationary: boolean | null;
  /** ADF p-value, absent when the test did not run. */
  p_value?: number;
  adf_statistic?: number;
  detail: string;
}

export interface RegimeRefitDiagnostics {
  /** Index of the first bar in the training window that produced this fit. */
  train_from_bar?: number;
  /** Index of the bar the training window stopped at — the fit never saw it or later. */
  trained_through_bar?: number;
  labels_from_bar?: number;
  labels_to_bar?: number;
  training_bars: number;
  bic: number | null;
  aic: number | null;
  n_parameters: number;
  n_iter: number | null;
  converged: boolean | null;
  [key: string]: unknown;
}

export interface RegimeReport {
  method: string;
  model_type: RegimeModel;
  n_states_requested: number;
  features: RegimeFeatureId[];
  periods_per_year: number;
  volatility_window: number;
  crisis_threshold: number;
  min_state_duration: number;
  transition_confidence: number;
  commit_confidence: number;
  min_train_bars: number;
  refit_every: number;
  /** Bars consumed by the observation window before a feature vector exists at all. */
  warmup_bars: number;
  /**
   * The whole null prefix: `warmup_bars` plus the bars the first model trained on and so
   * never labelled. Equal to `state_counts.warmup`, and different from `warmup_bars`.
   */
  unlabelled_bars: number;
  rows: number;
  labelled_rows: number;
  /** One diagnostic block per refit, in order. */
  refit_points: RegimeRefitDiagnostics[];
  refit_count: number;
  switch_count: number;
  crisis_override_bars: number;
  crisis_bars_in_warmup: number;
  stationarity: RegimeStationarity[];
  net_value_add: RegimeNetValueAdd;
  /** Every row is accounted for; the unlabelled prefix is keyed `warmup`. */
  state_counts: Record<string, number>;
  state_durations: Record<string, RegimeStateDuration>;
  /** Row/column order, so the heatmap's axes cannot be mislabelled. */
  transition_matrix: { labels: RegimeStateName[]; values: number[][] };
  bic_latest_refit: number | null;
  aic_latest_refit: number | null;
}

export interface RegimeRunResult {
  dataset: DatasetMeta | null;
  report: RegimeReport;
  preview: HeadTailPreview;
}

/* ============================================================ C6 · splits

   A split plan is a *description* of a dataset, not a change to it: it says which row
   ranges are train, validation and test, and which fold each belongs to. Nothing here
   produces a new price frame, so there is no `preview` in the create response — the plan
   table is the result.

   `SplitPlanRow` carries both an index pair and a timestamp pair, and they are not the
   same convention: `end_index` is **exclusive** (it slices), `end_time` is **inclusive**
   (it reads). Both are shipped because a plan expressed only in timestamps has to be
   reconstructed by searching the time column, which silently picks the wrong row on a
   frame with a repeated or missing bar. */

export type SplitType =
  | "date_3way"
  | "date_2way"
  | "percentage_70_15_15"
  | "percentage_60_20_20"
  | "tscv_expanding"
  | "tscv_rolling";

export type SplitRole = "train" | "val" | "test";

export const SPLIT_TYPE_OPTIONS: readonly PolicyOption<SplitType>[] = [
  {
    value: "percentage_70_15_15",
    label: "70 / 15 / 15 by row count",
    detail:
      "The first 70% of bars train, the next 15% validate, the last 15% test. The cuts are floor(0.70·N) and floor(0.85·N), so no bar is dropped or counted twice — the validation span absorbs the rounding.",
  },
  {
    value: "percentage_60_20_20",
    label: "60 / 20 / 20 by row count",
    detail:
      "As above with a larger out-of-sample share. Fewer bars to train on, more to score against.",
  },
  {
    value: "date_3way",
    label: "Three spans by date",
    detail:
      "You name all six boundaries. Useful when the split has to line up with something outside the data — a regime change, a listing date, a known structural break.",
  },
  {
    value: "date_2way",
    label: "Two spans by date",
    detail:
      "You name the training and test boundaries and there is no validation span.",
  },
  {
    value: "tscv_expanding",
    label: "Walk-forward — expanding window",
    detail:
      "The series is cut into (folds + 1) equal blocks. Fold k tests block k and trains on everything before it, so early folds train on little and late folds on much. The first block is only ever training data.",
  },
  {
    value: "tscv_rolling",
    label: "Walk-forward — rolling window",
    detail:
      "As above, but each fold trains on the single block immediately before its test block rather than on all history. Suits a non-stationary series where old bars describe a regime that no longer holds — and leaves every fold with the same amount of training data.",
    caution:
      "Rolling discards history deliberately. On a short series that can leave a fold with very few training bars, which the report states rather than hides.",
  },
];

export interface SplitParameters {
  train_start: string | null;
  train_end: string | null;
  val_start: string | null;
  val_end: string | null;
  test_start: string | null;
  test_end: string | null;
  n_folds: number;
  gap_size: number;
  purge_flag: boolean;
  label_horizon_bars: number | null;
}

export interface SplitCreateRequest {
  dataset_id: string;
  split_type: SplitType;
  parameters: Partial<SplitParameters>;
  name: string | null;
}

export interface SplitPlanRow {
  split_id: string;
  fold: number;
  role: SplitRole;
  /** Exclusive: slices the frame. */
  start_index: number;
  end_index: number;
  /** Inclusive: the timestamp of the span's first bar. */
  start_time: string;
  /** Inclusive: the timestamp of the span's *last* bar. */
  end_time: string;
  row_count: number;
  dataset_checksum: string;
}

/** One boundary where the application chose the cut. `gap_rows_removed` and
 *  `purge_rows_removed` are the rows each rule *actually* removed, which is not the same
 *  as the size of each knob: they overlap, so with `gap_size >= label_horizon_bars` the
 *  purge removes nothing and the report says so. */
export interface SplitBoundary {
  fold: number;
  role: SplitRole;
  cut_index: number;
  gap_rows_removed: number;
  purge_rows_removed: number;
  effective_end_index: number;
}

export interface SplitReport {
  split_type: SplitType;
  time_column: string;
  rows: number;
  dataset_checksum: string;
  parameters: Record<string, unknown>;
  folds: number;
  rows_by_role: Record<string, number>;
  rows_tested: number;
  rows_never_tested: number;
  boundaries: SplitBoundary[];
  warnings: string[];
}

export interface SplitPlanCreated {
  dataset: DatasetMeta;
  plan: SplitPlanRow[];
  report: SplitReport;
}

export interface SplitPlanSummary {
  split_plan_id: string;
  name: string;
  split_type: SplitType;
  folds: number;
  rows: number;
  created_at: string;
  parameters: Record<string, unknown>;
}

/* ============================================================ C7 · scaling

   The rule the whole module exists to hold: a scaler is fitted on the training span and
   applied to everything. `split_plan_id` is therefore *required* for the three fitted
   scalers — the backend refuses a request without one at the schema, so a page that let
   the user press Run would render a 422 rather than a number. Which is why the option
   list below is rendered from `ScalingOptions.scalers[].available` and never re-derived
   here. */

export type ScalerType = "minmax" | "standard" | "robust" | "rolling_zscore";

export const SCALER_OPTIONS: readonly PolicyOption<ScalerType>[] = [
  {
    value: "standard",
    label: "Standard (z-score)",
    detail:
      "(x − mean) / standard deviation, with both estimated on the training span and applied to every row. The classic choice when a model assumes roughly centred inputs.",
    caution:
      "The mean and deviation come from the training rows only. On the test span the result is not centred at zero — that is the point.",
  },
  {
    value: "minmax",
    label: "Min-Max to [0, 1]",
    detail:
      "(x − min) / (max − min), with the bounds taken from the training span.",
    caution:
      "Values outside the training range come out below 0 or above 1. That is correct behaviour, not an error — but it does mean the test span is not confined to the unit interval.",
  },
  {
    value: "robust",
    label: "Robust (median / IQR)",
    detail:
      "(x − median) / (Q3 − Q1) over the training span. Ignores the tails, so a handful of outliers moves the scale far less than it moves a mean and a standard deviation.",
  },
  {
    value: "rolling_zscore",
    label: "Rolling z-score",
    detail:
      "At every row, subtract the mean of the preceding N bars and divide by their standard deviation. Nothing is fitted, so no split plan is needed and the first N−1 rows are empty.",
  },
];

export interface ScalerOption {
  id: ScalerType;
  needs_split_plan: boolean;
  available: boolean;
  /** Never null when `available` is false. */
  reason: string | null;
}

export interface ScalingDefaults {
  features: string[];
  window: number;
  scaler_type: ScalerType;
}

export interface ScalingOptions {
  dataset_id: string;
  name: string;
  rows: number;
  features: string[];
  defaults: ScalingDefaults;
  scalers: ScalerOption[];
  plans: SplitPlanSummary[];
  warnings: RegimeWarning[];
}

export interface ScalingRequest {
  dataset_id: string;
  scaler_type: ScalerType;
  split_plan_id: string | null;
  fold: number | null;
  features: string[];
  window: number | null;
  write: boolean;
  name: string | null;
}

export interface ScalerParameter {
  feature: string;
  scaler_type: ScalerType;
  /** The inversion pair: `x = scaled · scale_ + center_` undoes any of the fitted
   *  scalers, which is why all three are reduced to it. */
  center_: number;
  scale_: number;
  /** True when the training span held no dispersion for this column, so it was left
   *  alone rather than divided by zero. */
  constant: boolean;
  fitted_rows: number;
  /** The raw sklearn attributes, under the library's own names. */
  attributes: Record<string, number | null>;
  formula: string;
}

export interface ScalingReport {
  dataset_id: string;
  dataset_name: string;
  scaler_type: ScalerType;
  split_plan_id: string | null;
  split_plan_name: string | null;
  fold: number | null;
  /** Which rows the parameters were estimated from — the sentence that says whether the
   *  run was legitimate. */
  fit_role: string | null;
  fit_rows: number;
  fit_start_time: string | null;
  fit_end_time: string | null;
  rows: number;
  features: string[];
  window: number | null;
  rows_used_for_fit: number;
  per_feature_rows: Record<string, number>;
  warnings: string[];
}

export interface ScalingAccepted {
  job_id: string;
  status: JobStatus;
  poll: string;
}

export interface ScalingCreated {
  dataset: DatasetMeta | null;
  params_dataset: DatasetMeta | null;
  report: ScalingReport;
  parameters: ScalerParameter[];
  preview: HeadTailPreview;
}

/* ================================================ E · volatility forecasting

   Module E is the first module that *predicts* rather than describes, and the types
   below are shaped by the two consequences of that.

   **There are two forecasts, and they are different objects.** The *scored* forecast is
   one-step-ahead at each walk-forward position: it is what MQLIKE, the Mincer–Zarnowitz
   regression and Diebold–Mariano are defined over, and it is what lands in the output
   dataset. The *forward path* is `forecast_horizon` steps past the last observation — it is
   what the fan chart draws, it travels only in the report, and nothing scores it. A single
   type carrying both would invite a reader to compare a projection against a realised
   value, which is a category error the backend already refuses to make.

   **Every metric is nullable, and that is information.** A model whose optimiser failed on
   enough steps has no MQLIKE — not a zero one. `unscored_reason` says which it is, so the
   table can print a sentence instead of an em-dash the reader has to interpret.

   Like Module D's, every optional number is `| null` rather than `?`: the server sends an
   explicit `null` for "there is no such number", and `undefined` would be
   indistinguishable from a field this frontend had not been updated to read. */

export type VolModel =
  | "garch"
  | "egarch"
  | "gjr_garch"
  | "arma_egarch_xgboost"
  | "tdnn_gjr_garch";

export type VolDistribution = "normal" | "students_t" | "skew_student" | "ged";

export type VolMeanModel = "constant" | "arma";

export type VolMetric = "mqlike" | "rmse" | "mae" | "mz_r2" | "dm";

/** Which of `p`, `o`, `q`, `lags` a model actually consumes. Read from the enum on the
 *  server so a control that changes nothing is visibly inert rather than silently so. */
export interface VolModelOption {
  id: VolModel;
  label: string;
  /** `simple` is the arch family; `hybrid` is a linear leg times a learned correction. */
  family: "simple" | "hybrid";
  needs: string[];
  note: string;
}

export interface VolMetricOption {
  id: VolMetric;
  label: string;
  note: string;
  /** False only for MZ R², which is a calibration diagnostic rather than a ranking. */
  lower_is_better: boolean;
  /**
   * Whether this metric orders the models at all. Not the same question as
   * `lower_is_better`: MZ R² has a direction (higher is better) and is still not a
   * ranking, so a table that crowned the maximum would contradict the note beside it.
   */
  is_ranking: boolean;
}

export interface VolWarning {
  level: "error" | "warning";
  code: string;
  message: string;
}

/** One column a request may take returns from.
 *
 *  The two named sources are always offered, with `available` saying whether this dataset
 *  can support them; `close` derives `diff(log(close))` and `log_return` uses a column of
 *  that name if one exists. A price level is never a model input. */
export interface VolSourceOption {
  id: string;
  label: string;
  available: boolean;
  reason: string | null;
}

/** A stored regime output offered as a conditioning dataset for the hybrid legs.
 *
 *  `usable` is false when the dataset carries no observation-time column — a Module D
 *  output produced before `close_time` was renamed away. It is still listed, with the
 *  reason, because a dataset silently missing from the dropdown is a puzzle rather than a
 *  verdict. */
export interface VolConditioningOption {
  dataset_id: string;
  name: string;
  rows: number;
  columns: string[];
  usable: boolean;
  reason: string | null;
}

export interface VolDefaults {
  price_source: string;
  models: VolModel[];
  distribution: VolDistribution;
  mean_model: VolMeanModel;
  p: number;
  o: number;
  q: number;
  lags: number;
  train_window: number;
  test_window: number;
  step_size: number;
  forecast_horizon: number;
}

/** The bounds the request schema enforces, sent so the form's live arithmetic uses the
 *  same numbers the refusal would. Restating them here would be a second copy of a
 *  guardrail, and the copy is the one that drifts. */
export interface VolLimits {
  min_train_window: number;
  min_test_window: number;
  min_scored_steps: number;
  min_step_size: number;
  max_forecast_horizon: number;
  max_order: number;
  max_lags: number;
}

export interface VolOptions {
  dataset_id: string;
  name: string;
  rows: number;
  /** Bars the return series actually has: `rows` less the first difference's missing
   *  predecessor. The walk-forward is defined over *this* count, and a form that shows
   *  only `rows` will look one bar longer than the schedule it offers. */
  usable_returns: number;
  timeframe: Timeframe | null;
  asset_class: AssetClass;
  columns: string[];
  sources: VolSourceOption[];
  models: VolModelOption[];
  metrics: VolMetricOption[];
  conditioning: VolConditioningOption[];
  defaults: VolDefaults;
  limits: VolLimits;
  feasible: boolean;
  warnings: VolWarning[];
}

export interface WalkForwardSpec {
  train_window: number;
  test_window: number;
  step_size: number;
}

export interface VolForecastRequest {
  dataset_id: string;
  price_source: string;
  models: VolModel[];
  distribution: VolDistribution;
  mean_model: VolMeanModel;
  p: number;
  o: number;
  q: number;
  lags: number;
  walk_forward: WalkForwardSpec;
  forecast_horizon: number;
  conditioning_dataset_id: string | null;
  name: string | null;
  write: boolean;
}

export interface VolForecastAccepted {
  job_id: string;
  status: JobStatus;
  poll: string;
}

/** One model's metrics over the scored walk-forward span.
 *
 *  `n_nan` and `n_inf` are counted, never filled. A model that produced forty non-finite
 *  forecasts and had them forward-filled is indistinguishable in a chart from one that
 *  produced none, and the two say opposite things about whether the optimiser converged. */
export interface VolModelScore {
  model: VolModel;
  family: "simple" | "hybrid";
  n_steps: number;
  n_finite: number;
  n_nan: number;
  n_inf: number;
  mqlike: number | null;
  rmse: number | null;
  mae: number | null;
  mz_intercept: number | null;
  mz_slope: number | null;
  mz_r2: number | null;
  /** Why the metrics are null, when they are. Never null in that case. */
  unscored_reason: string | null;
  seconds: number;
}

export interface VolComparison {
  model_a: VolModel;
  model_b: VolModel;
  /** Steps both models produced a finite forecast on — DM is run on the intersection, so
   *  a "difference" is not partly a difference between two samples. */
  n_effective: number;
  statistic: number;
  p_value: number;
  /** The model with the lower mean loss, or null on an exact tie. */
  better: VolModel | null;
  significant: boolean;
}

export interface VolFanBand {
  label: string;
  low: number[];
  high: number[];
}

/** The projection beyond the last observation — the fan chart.
 *
 *  Not scored, and not written to the dataset: MQLIKE, MZ and DM are all defined against a
 *  realised value, and a projected bar has none. */
export interface VolForwardPath {
  model: VolModel;
  times: string[];
  variance: number[];
  volatility: number[];
  bands: VolFanBand[];
}

export interface VolForecastReport {
  dataset_id: string;
  dataset_name: string;
  price_source: string;
  return_column: string;
  derived_returns: boolean;
  distribution: VolDistribution;
  mean_model: VolMeanModel;
  p: number;
  o: number;
  q: number;
  lags: number;
  walk_forward: WalkForwardSpec;
  forecast_horizon: number;
  rows: number;
  return_rows: number;
  /** The scored region, in bar indices and timestamps. Repeated rather than left to be
   *  re-derived, because this is the sentence that says what the metrics are over. */
  first_scored_index: number;
  last_scored_index: number;
  first_scored_time: string | null;
  last_scored_time: string | null;
  /** arch is fitted on percentage returns; variances are divided by this squared. Stated
   *  because it is a number in the artifact, not a modelling choice. */
  scale_factor: number;
  adf_statistic: number;
  adf_pvalue: number;
  stationary: boolean;
  /** The realised-variance proxy, e.g. "squared log return". */
  actual_proxy: string;
  scores: VolModelScore[];
  comparisons: VolComparison[];
  /** Empirical quantiles of the walk-forward error ratio `actual/forecast`, per model.
   *  These are the fan bands — empirical because deriving them from the fitted
   *  distribution would assume the specification the walk-forward exists to question. */
  error_quantiles: Record<string, Record<string, number>>;
  forward_paths: VolForwardPath[];
  forward_step_seconds: number | null;
  forward_step_source: string | null;
  conditioning_dataset_id: string | null;
  conditioning_dataset_name: string | null;
  conditioning_columns: string[];
  warnings: string[];
}

export interface VolForecastRunResult {
  dataset: DatasetMeta | null;
  report: VolForecastReport;
  preview: HeadTailPreview;
}
