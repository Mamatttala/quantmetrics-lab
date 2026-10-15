/**
 * Backend client.
 *
 * Every failure is normalised into `ApiError` carrying the backend's machine
 * readable `code`. Callers branch on the code (e.g. offer a narrower date range
 * for SPAN_TOO_WIDE) rather than matching on prose, which would silently break the
 * moment a message is reworded.
 */

import type {
  ApiErrorPayload,
  AssetClass,
  CleanRequest,
  CleanResult,
  DatasetCategory,
  DatasetMeta,
  DatasetSummary,
  DeriveRequest,
  FeatureCatalog,
  FeatureRequest,
  FeatureResult,
  FetchResult,
  FetchRun,
  MergeRequest,
  MergeResult,
  OhlcvResponse,
  PageResponse,
  PreviewResponse,
  QualityResponse,
  JobRecord,
  RegimeDetectRequest,
  RegimeOptions,
  RegimeRunAccepted,
  RegimeRunResult,
  ScalingAccepted,
  ScalingOptions,
  ScalingRequest,
  SplitCreateRequest,
  SplitPlanCreated,
  SplitPlanSummary,
  Timeframe,
  VolForecastAccepted,
  VolForecastRequest,
  VolOptions,
} from "./types";

const BASE = "/api";

export class ApiError extends Error {
  readonly code: string;
  readonly status: number;
  readonly details: Record<string, unknown>;

  constructor(
    message: string,
    code: string,
    status: number,
    details: Record<string, unknown> = {},
  ) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
    this.details = details;
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const options: RequestInit = { ...init };
  if (!(options.body instanceof FormData)) {
    // Only set for JSON bodies: a FormData body must be left alone so the browser
    // can generate the multipart boundary itself. Nothing in this client passes
    // custom headers, so there is no merge to preserve.
    options.headers = { "Content-Type": "application/json" };
  }

  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, options);
  } catch (cause) {
    // A transport failure means the backend is not running, which is a different
    // situation from a rejected request and must say so.
    throw new ApiError(
      "Cannot reach the backend. Is the API server running on port 8000?",
      "transport_error",
      0,
      { cause: String(cause) },
    );
  }

  if (!response.ok) {
    let payload: ApiErrorPayload | null = null;
    try {
      payload = (await response.json()) as ApiErrorPayload;
    } catch {
      payload = null;
    }
    // FastAPI's own validation errors have a different shape than ours; fall back
    // to a generic message rather than rendering "[object Object]".
    const message =
      payload?.error?.message ??
      (response.status === 404
        ? `No such endpoint: ${path}`
        : `Request failed (${response.status})`);
    // The backend puts the actionable half of a rejection in `details.hint` — "Storing
    // it would round the value without saying so", "Use true or false". Every display
    // site in this app renders `error.message` and nothing else, so a hint kept only in
    // `details` was written, sent, and stored without ever being shown once. Folding it
    // into the message is what gets it onto the screen; `details` still carries it
    // structured for anything that wants to branch on it.
    const hint = payload?.error?.details?.hint;
    const shown = typeof hint === "string" && hint !== "" ? `${message} ${hint}` : message;
    throw new ApiError(
      shown,
      payload?.error?.code ?? "http_error",
      response.status,
      payload?.error?.details ?? {},
    );
  }

  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

export const api = {
  health: () =>
    request<{ status: string; version: string; data_dir: string }>("/health"),

  listDatasets: (category?: DatasetCategory) =>
    request<DatasetSummary[]>(
      category ? `/datasets?category=${category}` : "/datasets",
    ),

  getDataset: (id: string) => request<DatasetMeta>(`/datasets/${id}`),

  preview: (id: string, head = 5, tail = 5) =>
    request<PreviewResponse>(`/datasets/${id}/preview?head=${head}&tail=${tail}`),

  page: (id: string, offset: number, limit: number, sort?: string) => {
    const params = new URLSearchParams({
      offset: String(offset),
      limit: String(limit),
    });
    if (sort) params.set("sort", sort);
    return request<PageResponse>(`/datasets/${id}/page?${params.toString()}`);
  },

  /** Rename catalog metadata without touching the immutable Parquet payload. */
  renameDataset: (datasetId: string, newName: string) =>
    request<void>(`/datasets/${encodeURIComponent(datasetId)}/rename`, {
      method: "POST",
      body: JSON.stringify({ name: newName }),
    }),

  /** Delete one dataset. The backend refuses a source that still has descendants. */
  deleteDataset: (datasetId: string) =>
    request<void>(`/datasets/${encodeURIComponent(datasetId)}`, {
      method: "DELETE",
    }),

  deleteAllDatasets: () =>
    request<{ deleted: number }>("/datasets", { method: "DELETE" }),

  deriveDataset: (id: string, body: DeriveRequest) =>
    request<DatasetMeta>(`/datasets/${id}/derive`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  uploadCsv: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<DatasetMeta>("/datasets/upload", { method: "POST", body: form });
  },

  // ------------------------------------------------------------------- fetch

  fetchKlines: (body: {
    asset_class: AssetClass;
    symbol: string;
    timeframe: Timeframe;
    start: string;
    end: string;
  }) =>
    request<FetchResult>("/fetch/binance/klines", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  fetchFunding: (body: { symbol: string; start: string; end: string }) =>
    request<FetchResult>("/fetch/binance/funding", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  fetchMetrics: (body: {
    symbol: string;
    timeframe: Timeframe;
    start: string;
    end: string;
  }) =>
    request<FetchResult>("/fetch/binance/metrics", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  fetchYahoo: (body: {
    asset_class: AssetClass;
    symbol: string;
    timeframe: Timeframe;
    start: string;
    end: string;
  }) =>
    request<FetchResult>("/fetch/yahoo", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  fetchRuns: (limit = 50) =>
    request<FetchRun[]>(`/fetch/runs?limit=${limit}`),

  // ------------------------------------------------------------------ charts

  ohlcv: (id: string, limit = 20000) =>
    request<OhlcvResponse>(`/datasets/${id}/ohlcv?limit=${limit}`),

  quality: (id: string) => request<QualityResponse>(`/datasets/${id}/quality`),

  // -------------------------------------------------------------- preparation

  /** Run a cleaning plan. With `write: false` this is the dry run: the same
   *  computation, the same report, the same preview, nothing stored. */
  cleanDataset: (id: string, body: CleanRequest) =>
    request<CleanResult>(`/datasets/${id}/clean`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  mergeDatasets: (body: MergeRequest) =>
    request<MergeResult>("/datasets/merge", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  // -------------------------------------------------------- C5 · preprocessing

  /** The catalogue for one dataset. Availability is resolved server-side against that
   *  dataset's columns, timeframe and asset class, so the UI never re-derives it.
   *
   *  The query name is `dataset_id`, not `datasetId`: every parameter this API accepts is
   *  snake_case on the wire (route params `{dataset_id}`, body fields `base_dataset_id`),
   *  and `datasetId` here was a local variable's camelCase leaking into a URL. It is the
   *  only multi-word query parameter in the application, which is why nothing else caught
   *  it — and it 422'd every catalogue request, so the whole page rendered an error. */
  featureCatalog: (datasetId: string) =>
    request<FeatureCatalog>(
      `/features/catalog?dataset_id=${encodeURIComponent(datasetId)}`,
    ),

  /** Compute features. With `write: false` this is the dry run: the same computation,
   *  the same report, the same preview, nothing stored. */
  computeFeatures: (id: string, body: FeatureRequest) =>
    request<FeatureResult>(`/datasets/${id}/features`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  // ------------------------------------------------------- D · regime detection

  /** What this dataset can support, and why not for the rest.
   *
   *  `dataset_id`, not `datasetId` — the wire format is snake_case throughout. */
  regimeOptions: (datasetId: string) =>
    request<RegimeOptions>(
      `/regime/options?dataset_id=${encodeURIComponent(datasetId)}`,
    ),

  /** Accept a detection run. Resolves with a job handle, not a result: the fit takes
   *  longer than a request should, so the caller polls `job()` until it is terminal. */
  detectRegime: (body: RegimeDetectRequest) =>
    request<RegimeRunAccepted>("/regime/detect", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  /** Poll a job. The result type is named by the caller because the registry is shared
   *  between D1's detections and C7's scaling runs; see `JobRecord`. */
  job: <TResult = RegimeRunResult>(jobId: string) =>
    request<JobRecord<TResult>>(`/jobs/${encodeURIComponent(jobId)}`),

  jobs: (limit = 25) => request<JobRecord[]>(`/jobs?limit=${limit}`),

  // ------------------------------------------------------- C6 · chronological splits

  /** Compute and store a split plan.
   *
   *  **201, not 202.** The plan is finished when this resolves — the response carries the
   *  stored dataset, every span, and the report. A page that treated it as a job handle
   *  would show an empty table under a request that had already succeeded. */
  createSplitPlan: (body: SplitCreateRequest) =>
    request<SplitPlanCreated>("/splits/create", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  /** Every plan already made from this dataset, newest first. Read from the lineage table,
   *  so it is exactly this dataset's plans and not every plan in the store. */
  splitPlans: (datasetId: string) =>
    request<SplitPlanSummary[]>(
      `/splits/plans?dataset_id=${encodeURIComponent(datasetId)}`,
    ),

  // --------------------------------------------------------------- C7 · scaling

  /** Numeric features, available scalers, and this dataset's plans, in one call.
   *
   *  Availability is a *verdict* computed server-side against this dataset — which scalers
   *  can run, and why not for the rest. The page renders that and never re-derives it, so a
   *  control shown as available and a refusal from the run endpoint cannot disagree. */
  scalingOptions: (datasetId: string) =>
    request<ScalingOptions>(
      `/scaling/options?dataset_id=${encodeURIComponent(datasetId)}`,
    ),

  /** Accept a scaling run. Resolves with a job handle, not a result: the fit reads a whole
   *  Parquet file and writes a whole one back, so the caller polls `job()` until terminal. */
  runScaling: (body: ScalingRequest) =>
    request<ScalingAccepted>("/scaling/fit_transform", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  // ------------------------------------------------ E · volatility forecasting

  /** Sources, the model catalogue, the metrics and their direction, any conditioning
   *  datasets, and this dataset's default walk-forward schedule — one call for the page.
   *
   *  The default schedule is derived from the row count server-side and is one the dataset
   *  can actually run. Deriving it here would be a second implementation of the same
   *  arithmetic, and the two would disagree on exactly the short datasets where the
   *  difference matters. */
  volatilityOptions: (datasetId: string) =>
    request<VolOptions>(
      `/volatility/options?dataset_id=${encodeURIComponent(datasetId)}`,
    ),

  /** Accept a forecasting run. Resolves with a job handle, not a result: a walk-forward
   *  refits at every scored step — twice per step for a hybrid — so the run takes minutes
   *  and the caller polls `job<VolForecastRunResult>()` until it is terminal. */
  runVolatilityForecast: (body: VolForecastRequest) =>
    request<VolForecastAccepted>("/volatility/forecast", {
      method: "POST",
      body: JSON.stringify(body),
    }),
};
