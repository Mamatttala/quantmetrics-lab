import { cloneElement, useEffect, useId, useRef, useState } from "react";
import type { ReactElement, ReactNode } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import type { Data, Layout } from "plotly.js";

import { api } from "@/api/client";
import { notifyDatasetsChanged, subscribeToDatasets } from "@/api/datasetsChanged";
import { isAbort } from "@/api/jobs";
import type { DatasetSummary } from "@/api/types";
import type {
  AnalysisResult, AnalysisScope, CorrelationMethod, CorrelationPair, CorrelationResult,
  ExploratoryOptions, PCARequest, PCAResult, ScopeRequest, CorrelationRequest,
} from "@/api/exploratory";
import { featuresInGroups, heatmapValues, resolveAnalysis, scopeIsReady } from "@/api/exploratory";
import { ResponsivePlot as Plot } from "@/charts/ResponsivePlot";
import { DatasetOptions } from "@/components/DatasetOptions";
import { PageHeader } from "@/components/PageHeader";
import { ResultGrid } from "@/components/ResultGrid";
import { chartTheme, useTheme } from "@/theme";
import "./exploratory.css";

const scopeLabels: Record<AnalysisScope, string> = {
  full_dataset: "Full dataset", custom_date_range: "Custom date range", train_span: "Training span",
  rolling: "Rolling / time-varying",
};
const methodLabels: Record<CorrelationMethod, string> = {
  pearson: "Pearson", spearman: "Spearman", kendall: "Kendall tau-b",
  partial_pearson: "Partial Pearson", partial_spearman: "Partial Spearman",
};
const message = (cause: unknown) => cause instanceof Error ? cause.message : String(cause);
const display = (value: unknown): string => value === null || value === undefined ? "—"
  : typeof value === "number" ? Number.isInteger(value) ? String(value) : value.toPrecision(5)
    : typeof value === "boolean" ? value ? "Yes" : "No" : String(value);

export function ExploratoryAnalysisPage() {
  const { datasetId } = useParams<{ datasetId?: string }>();
  const navigate = useNavigate();
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [options, setOptions] = useState<ExploratoryOptions | null>(null);
  const [loadError, setLoadError] = useState("");
  const [features, setFeatures] = useState<string[]>([]);
  const [scope, setScope] = useState<AnalysisScope>("full_dataset");
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [planId, setPlanId] = useState("");
  const [fold, setFold] = useState(1);
  const [target, setTarget] = useState("");
  const [controls, setControls] = useState<string[]>([]);
  const [group, setGroup] = useState("");
  const [method, setMethod] = useState<CorrelationMethod>("pearson");
  const [threshold, setThreshold] = useState(0.9);
  const [ccf, setCcf] = useState(false);
  const [windowSize, setWindowSize] = useState(60);
  const [standardizer, setStandardizer] = useState<PCARequest["standardizer"]>("standard");
  const [selection, setSelection] = useState<PCARequest["component_selection"]>("manual");
  const [components, setComponents] = useState(2);
  const [variance, setVariance] = useState(0.95);
  const [strict, setStrict] = useState(false);
  const [keepOriginal, setKeepOriginal] = useState(false);
  const [carry, setCarry] = useState<string[]>([]);
  const [name, setName] = useState("");
  const [datasetSearch, setDatasetSearch] = useState("");
  const [configOpen, setConfigOpen] = useState(true);
  const [corrOptions, setCorrOptions] = useState<Partial<CorrelationRequest>>({
    min_obs: 30, group_min_obs: 30, alpha: .05, confidence_level: .95,
    multiple_testing: "none", compute_p_values: true, compute_confidence_intervals: true,
    calculate_vif: true, vif_threshold: 5, include_intercept: true, enable_redundancy_groups: true,
    rolling_step: 1, rolling_min_periods: 30, rolling_pair_selection: "all_pairs", rolling_top_pairs: 10,
    lag_min: -10, lag_max: 10, ccf_min_obs: 30,
  });
  const [corrPolicy, setCorrPolicy] = useState<Policy>({ pre_transform: "none", missing_policy: "pairwise_complete", constant_policy: "exclude_constants" });
  const [pcaPolicy, setPcaPolicy] = useState<Policy>({ pre_transform: "none", missing_policy: "pairwise_complete", constant_policy: "exclude_constants" });
  const [pcaOptions, setPcaOptions] = useState<Partial<PCARequest>>({ whiten: false, component_prefix: "pc_", parallel_permutations: 100, include_target_feature: false, include_carried_columns: false });
  const [correlation, setCorrelation] = useState<CorrelationResult | null>(null);
  const [pca, setPca] = useState<PCAResult | null>(null);
  const [corrBusy, setCorrBusy] = useState(false);
  const [pcaBusy, setPcaBusy] = useState(false);
  const [corrError, setCorrError] = useState("");
  const [pcaError, setPcaError] = useState("");
  const corrAbort = useRef<AbortController | null>(null);
  const pcaAbort = useRef<AbortController | null>(null);

  useEffect(() => {
    let active = true;
    const refresh = () => api.exploratoryOptions().then((value) => {
      if (active) setDatasets(value.datasets);
    }).catch((cause: unknown) => { if (active) setLoadError(message(cause)); });
    void refresh();
    const unsubscribe = subscribeToDatasets(() => { void refresh(); });
    return () => { active = false; unsubscribe(); };
  }, []);

  useEffect(() => {
    let active = true;
    setOptions(null);
    setLoadError("");
    setFeatures([]);
    setControls([]);
    setCarry([]);
    setTarget("");
    setGroup("");
    setScope("full_dataset");
    if (datasetId) void api.exploratoryOptions(datasetId).then((value) => {
      if (!active) return;
      setOptions(value);
      setDatasets(value.datasets);
      setFeatures(value.features.slice(0, 12));
      setComponents(Math.max(1, Math.min(2, value.features.length)));
      setPlanId(value.plans[0]?.split_plan_id ?? "");
      setFold(1);
      const dataset = value.datasets.find((item) => item.dataset_id === datasetId);
      setStart(dataset?.start_timestamp?.slice(0, 19) ?? "");
      setEnd(dataset?.end_timestamp?.slice(0, 19) ?? "");
    }).catch((cause: unknown) => { if (active) setLoadError(message(cause)); });
    return () => { active = false; };
  }, [datasetId]);
  useEffect(() => () => { corrAbort.current?.abort(); pcaAbort.current?.abort(); }, []);
  useEffect(() => {
    if (!datasetId && datasets.length) {
      const suitable = datasets.filter((item) => item.row_count >= 2 && item.artifact_kind === "observations")
        .sort((a, b) => b.created_at.localeCompare(a.created_at))[0];
      if (suitable) navigate(`/exploratory/${suitable.dataset_id}`, { replace: true });
    }
  }, [datasetId, datasets, navigate]);

  const plan = options?.plans.find((item) => item.split_plan_id === planId);
  const walkForward = plan?.split_type.startsWith("tscv") ?? false;
  const partial = method.startsWith("partial");
  const ready = options?.dataset_id === datasetId && features.length >= 2 && features.length <= (options?.computational_limits.max_features ?? 250);
  const scopeReady = scopeIsReady(scope, options?.time_column ?? null, start, end, plan !== undefined);
  const selectedDataset = datasets.find((item) => item.dataset_id === datasetId);
  function baseRequest(): ScopeRequest {
    return { dataset_id: datasetId ?? "", features, scope,
      ...(scope === "custom_date_range" ? { start_time: `${start}Z`, end_time: `${end}Z` } : {}),
      ...(scope === "train_span" ? { split_plan_id: planId, ...(walkForward ? { fold } : {}) } : {}),
    };
  }
  async function runCorrelation() {
    if (corrAbort.current && !corrAbort.current.signal.aborted && corrBusy) return;
    const abort = new AbortController();
    corrAbort.current = abort;
    setCorrBusy(true);
    setCorrError("");
    try {
      const accepted = await api.runCorrelation({ ...baseRequest(), ...corrOptions, ...corrPolicy, method, target: target || null,
        controls: partial ? controls : [], group_column: scope === "rolling" ? null : group || null,
        redundancy_threshold: threshold, max_lag: 0, enable_ccf: scope === "rolling" ? false : ccf,
        rolling_window: scope === "rolling" ? windowSize : null, enable_rolling: scope === "rolling",
        calculate_vif: scope === "rolling" ? false : corrOptions.calculate_vif ?? true,
        enable_redundancy_groups: scope === "rolling" ? false : corrOptions.enable_redundancy_groups ?? true,
        strict_causal_mode: strict });
      const result = await resolveAnalysis<CorrelationResult>(accepted, abort.signal);
      if (abort.signal.aborted) return;
      setCorrelation(result);
    } catch (cause) { if (!isAbort(cause) && !abort.signal.aborted) setCorrError(message(cause)); }
    finally { if (!abort.signal.aborted) setCorrBusy(false); }
  }
  async function runPCA(write: boolean) {
    if (pcaAbort.current && !pcaAbort.current.signal.aborted && pcaBusy) return;
    const abort = new AbortController();
    pcaAbort.current = abort;
    setPcaBusy(true);
    setPcaError("");
    try {
      const accepted = await api.runPCA({ ...baseRequest(), ...pcaPolicy, ...pcaOptions, standardizer,
        component_selection: selection, n_components: components, variance_threshold: variance,
        write_dataset: write, strict_causal: strict, keep_original_features: keepOriginal,
        carry_columns: carry, target_feature: target || null, name: name.trim() || null,
        destination: pcaOptions.destination ?? (scope === "train_span" ? "processed_data" : "outputs") });
      const result = await resolveAnalysis<PCAResult>(accepted, abort.signal);
      if (abort.signal.aborted) return;
      setPca(result);
      if (result.dataset) notifyDatasetsChanged();
    } catch (cause) { if (!isAbort(cause) && !abort.signal.aborted) setPcaError(message(cause)); }
    finally { if (!abort.signal.aborted) setPcaBusy(false); }
  }

  return <div className="page exploratory">
    <PageHeader title="Exploratory Analysis" lead="Explore relationships and static principal components in any dataset. Full-dataset analysis is available by default." />
    <button className="btn" aria-expanded={configOpen} onClick={() => setConfigOpen(!configOpen)}>{configOpen ? "Collapse configuration" : "Show configuration"}</button>
    <div className="exploratory__workspace">
      <section className="exploratory__config" aria-label="Analysis configuration" hidden={!configOpen}>
        <Section title="Dataset and scope">
          <Field label="Search datasets"><input value={datasetSearch} onChange={(e) => setDatasetSearch(e.target.value)} /></Field>
          <Field label="Source dataset"><select id="exploratory-dataset" value={datasetId ?? ""}
            onChange={(event) => navigate(event.target.value ? `/exploratory/${event.target.value}` : "/exploratory")}>
            <option value="">Select a dataset…</option><DatasetOptions datasets={datasets.filter((item) => item.dataset_id === datasetId || item.name.toLowerCase().includes(datasetSearch.toLowerCase()))} />
          </select></Field>
          {loadError && <ErrorNotice text={loadError} />}
          {datasetId && !options && !loadError && <p role="status">Loading dataset options…</p>}
          {options && <>
            {selectedDataset && <p className="muted">{selectedDataset.row_count.toLocaleString()} rows · {options.columns.length} columns · {selectedDataset.timeframe ?? "No timeframe"}<br />Lineage: {selectedDataset.parent_dataset_id ?? "Source dataset"} · {selectedDataset.operation ?? "Imported"}</p>}
            <p className="muted">Observation column: {options.time_column ?? "None detected; full-sample diagnostics are available."}</p>
            {options.has_close_time && <p className="muted">close_time is present but is never used as C8's observation timeline.</p>}
            <details><summary>Dataset compatibility</summary><p>{options.features.length} numeric columns; {options.plans.length} split plans.</p><p>Ignored non-numeric/time columns: {options.ignored_columns.join(", ") || "None"}</p><p>Regime/volatility: {featuresInGroups(options.feature_groups, ["regime", "volatility", "volatility_forecast"]).join(", ") || "None"}</p>{options.warnings.map((warning) => <p key={warning}>{warning}</p>)}</details>
            <MultiSelect label="Numeric features" options={options.features} selected={features}
              onChange={(value) => { setFeatures(value); setControls((old) => old.filter((c) => !value.includes(c))); }} />
            <div className="exploratory__actions">{[["Returns / volatility", "returns", "volatility"], ["Regime features", "regime"], ["Exogenous features", "exogenous"]].map(([label, ...groups]) => <button className="btn" key={label} onClick={() => setFeatures(featuresInGroups(options.feature_groups, groups))}>{label}</button>)}</div>
            <details><summary>Feature groups</summary>{Object.entries(options.feature_groups).map(([label, members]) => <button className="btn" key={label} onClick={() => setFeatures(members)}>{label} ({members.length})</button>)}</details>
            {features.length > 100 && <p className="exploratory__notice">More than 100 features: inference and rolling diagnostics can be expensive.</p>}
            {features.length > options.computational_limits.max_features! && <ErrorNotice text={`Select at most ${options.computational_limits.max_features} features.`} />}
            {options.features.length === 0 && <p>This dataset has no numeric features.</p>}
            <Field label="Analysis scope"><select value={scope} onChange={(e) => setScope(e.target.value as AnalysisScope)}>
              <option value="full_dataset">Full dataset (default)</option>
              <option value="custom_date_range" disabled={!options.time_column}>Custom date range</option>
              <option value="train_span" disabled={!options.time_column || !options.plans.length}>Training span</option>
              <option value="rolling" disabled={!options.time_column}>Rolling / time-varying correlation</option>
            </select></Field>
            {scope === "custom_date_range" && <>
              <Field label="Start (UTC, inclusive)"><input type="datetime-local" step="1" value={start} onChange={(e) => setStart(e.target.value)} /></Field>
              <Field label="End (UTC, inclusive)"><input type="datetime-local" step="1" value={end} onChange={(e) => setEnd(e.target.value)} /></Field>
            </>}
            {scope === "train_span" && <>
              <Field label="Split plan"><select value={planId} onChange={(e) => { setPlanId(e.target.value); setFold(1); }}>
                {options.plans.map((item) => <option key={item.split_plan_id} value={item.split_plan_id}>{item.name}</option>)}
              </select></Field>
              {walkForward && <Field label="Training fold"><select value={fold} onChange={(e) => setFold(Number(e.target.value))}>
                {Array.from({ length: plan?.folds ?? 0 }, (_, index) => <option key={index} value={index + 1}>{index + 1}</option>)}
              </select></Field>}
            </>}
            <p className="exploratory__notice">{scope === "rolling" ? "Rolling correlation uses causal preceding windows. Rolling PCA is deferred." : scope === "train_span" ? "Fit only on the selected training span."
              : "Full-sample and custom-range fits are exploratory and labeled non-causal for downstream training."}
              {" "}The causal label covers C8 fitting only; it does not certify the whole pipeline.</p>
          </>}
        </Section>
        <Section title="Correlation Analysis">
          <Preprocessing value={corrPolicy} onChange={setCorrPolicy} scope={scope} />
          <Field label="Correlation method"><select value={method} onChange={(e) => setMethod(e.target.value as CorrelationMethod)}>
            {Object.entries(methodLabels).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
          </select></Field>
           <CorrelationControls value={corrOptions} onChange={setCorrOptions} features={[...new Set([...features, ...(target ? [target] : [])])]} rollingScope={scope === "rolling"} />
          <Field label="Target feature (optional)"><select value={target} onChange={(e) => {
            setTarget(e.target.value); setControls((old) => old.filter((c) => c !== e.target.value));
           }}><option value="">None</option>{options?.target_candidates.map((f) => <option key={f}>{f}</option>)}</select></Field>
          {partial && <MultiSelect label="Control features" options={(options?.features ?? []).filter((f) => !features.includes(f) && f !== target)} selected={controls} onChange={setControls} />}
           {scope !== "rolling" && <><Field label="Grouping column (optional)"><select value={group} onChange={(e) => setGroup(e.target.value)}>
             <option value="">None</option>{options?.group_by_candidates.map((column) => <option key={column}>{column}</option>)}
           </select></Field>
           <NumberField label="Redundancy threshold |r|" value={threshold} onChange={setThreshold} min={0.01} max={1} step={0.01} />
           <Check label="Lead–lag diagnostic against target" checked={ccf} onChange={setCcf} />
           {ccf && <><NumberField label="Minimum lag (bars)" value={corrOptions.lag_min ?? -10} onChange={(value) => setCorrOptions({ ...corrOptions, lag_min: value })} min={-(options?.computational_limits.max_ccf_lag_abs ?? 120)} max={options?.computational_limits.max_ccf_lag_abs ?? 120} />
             <NumberField label="Maximum lag (bars)" value={corrOptions.lag_max ?? 10} onChange={(value) => setCorrOptions({ ...corrOptions, lag_max: value })} min={-(options?.computational_limits.max_ccf_lag_abs ?? 120)} max={options?.computational_limits.max_ccf_lag_abs ?? 120} />
            <NumberField label="CCF minimum observations" value={corrOptions.ccf_min_obs ?? 30} onChange={(value) => setCorrOptions({ ...corrOptions, ccf_min_obs: value })} min={3} max={1000000} />
             <p className="exploratory__notice">Lag L = corr(x_t, y_&#123;t+L&#125;). Positive lags use future target values: x leads y. Diagnostic only; never saved as row features.</p></>}</>}
           {scope === "rolling" && <><NumberField label="Past rows in rolling window" value={windowSize} onChange={setWindowSize} min={3} max={options?.computational_limits.max_rolling_window ?? 5000} />
             <p className="muted">Uses preceding rows only, excluding the current row. All-pairs mode returns summaries; select explicit pairs for plotted series.</p></>}
          <button className="btn btn--primary" disabled={!ready || !scopeReady || corrBusy || (partial && !controls.length)
             || (scope !== "rolling" && ccf && !target) || ((ccf || scope === "rolling") && !options?.time_column)
             || new Set([...features, ...(target ? [target] : [])]).size < 2} onClick={() => { void runCorrelation(); }}>{scope === "rolling" ? "Run Rolling Correlation" : "Run Correlation"}</button>
          {corrBusy && <p role="status">Correlation job running…</p>}
          {corrError && <ErrorNotice text={corrError} />}
        </Section>
        <Section title="PCA Analysis">
          <Preprocessing value={pcaPolicy} onChange={setPcaPolicy} scope={scope} />
          <Field label="Standardizer"><select value={standardizer} onChange={(e) => setStandardizer(e.target.value as PCARequest["standardizer"])}>
            {["none", "standard", "robust", "minmax"].map((value) => <option key={value}>{value}</option>)}
          </select></Field>
          <p className="muted">Static PCA always mean-centers after standardization.</p>
          <Field label="Component selection"><select value={selection} onChange={(e) => setSelection(e.target.value as PCARequest["component_selection"])}>
            <option value="manual">Manual component count</option><option value="variance_threshold">Explained variance threshold</option>
            <option value="eigenvalue_gt_1">Eigenvalue &gt; 1</option>
            <option value="scree">Scree heuristic</option><option value="parallel">Parallel analysis</option>
          </select></Field>
          {selection === "manual" && <NumberField label="Number of components" value={components} onChange={setComponents} min={1} max={Math.max(1, features.length)} />}
          {selection === "variance_threshold" && <NumberField label="Explained variance threshold" value={variance} onChange={setVariance} min={0.01} max={1} step={0.01} />}
           {selection === "parallel" && <NumberField label="Parallel permutations" value={pcaOptions.parallel_permutations ?? 100} onChange={(value) => setPcaOptions({ ...pcaOptions, parallel_permutations: value })} min={1} max={options?.computational_limits.max_parallel_permutations ?? 200} />}
          <Check label="Whiten scores" checked={pcaOptions.whiten ?? false} onChange={(value) => setPcaOptions({ ...pcaOptions, whiten: value })} />
          <Field label="Component prefix"><input value={pcaOptions.component_prefix ?? "pc_"} onChange={(e) => setPcaOptions({ ...pcaOptions, component_prefix: e.target.value })} /></Field>
          <Field label="Save destination"><select value={pcaOptions.destination ?? (scope === "train_span" ? "processed_data" : "outputs")} onChange={(e) => setPcaOptions({ ...pcaOptions, destination: e.target.value as NonNullable<PCARequest["destination"]> })}><option value="outputs">Outputs</option><option value="processed_data">Processed data</option></select></Field>
          {scope !== "train_span" && pcaOptions.destination === "processed_data" && <p className="exploratory__notice">Saving non-causal PCA to processed data preserves its non-causal label.</p>}
           <Field label="Saved dataset name (optional)"><input type="text" placeholder="Leave blank for an auto-generated canonical name" value={name} onChange={(e) => setName(e.target.value)} maxLength={200} /></Field>
           <p className="muted">Blank names are generated uniquely from source, scope, operation, and dataset ID. Duplicate custom names are refused.</p>
          <Check label="Keep selected original features" checked={keepOriginal} onChange={setKeepOriginal} />
          <Check label="Include target feature" checked={pcaOptions.include_target_feature ?? false} onChange={(value) => setPcaOptions({ ...pcaOptions, include_target_feature: value })} />
          <Check label="Include all non-selected columns" checked={pcaOptions.include_carried_columns ?? false} onChange={(value) => setPcaOptions({ ...pcaOptions, include_carried_columns: value })} />
          <MultiSelect label="Carry target / other columns" options={(options?.columns ?? []).filter((c) => c !== options?.time_column)} selected={carry} onChange={setCarry} />
          <Check label="Strict causal mode for saving (default off)" checked={strict} onChange={setStrict} />
          <p className="muted">Saving creates a transformed dataset and a parameter artifact. All source rows are transformed with the fitted parameters; incomplete rows keep null scores.</p>
          {!options?.time_column && options && <p>Saving requires an observation/time column.</p>}
          {strict && scope !== "train_span" && <p className="exploratory__notice">Strict mode requires training-span fitting to save. Diagnostics remain available.</p>}
          <div className="exploratory__actions">
            <button className="btn" disabled={!ready || !scopeReady || pcaBusy || scope === "rolling"} onClick={() => { void runPCA(false); }}>Run PCA Diagnostics</button>
            <button className="btn btn--primary" disabled={!ready || !scopeReady || pcaBusy || scope === "rolling" || !options?.time_column || (strict && scope !== "train_span") || pcaPolicy.pre_transform === "rank_transform"}
              onClick={() => { void runPCA(true); }}>Run PCA and Save Dataset</button>
          </div>
          <p className="muted">Fixed deterministic signs. Rolling PCA is deferred. Rank transform is diagnostic-only and cannot be saved.</p>
          {pcaBusy && <p role="status">PCA job running…</p>}
          {pcaError && <ErrorNotice text={pcaError} />}
        </Section>
      </section>
      <div className="exploratory__results" role="region" aria-label="Analysis results">
        <div className="exploratory__actions"><button className="btn" disabled={corrBusy || pcaBusy || (!correlation && !pca && !corrError && !pcaError)} onClick={() => {
          setCorrelation(null); setPca(null); setCorrError(""); setPcaError("");
        }}>Clear Results</button></div>
        <Section title="Correlation results">
          {correlation ? <CorrelationResults result={correlation} /> : <p className="muted">Run Correlation to see relationships, multicollinearity, and redundancy here.</p>}
        </Section>
        <Section title="PCA results">
          {pca ? <PCAResults result={pca} /> : <p className="muted">Run PCA Diagnostics for immediate results, or save reusable PCA datasets.</p>}
        </Section>
      </div>
    </div>
  </div>;
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return <section className="panel exploratory__section"><h2 className="section__title">{title}</h2>{children}</section>;
}
function Field({ label, children }: { label: string; children: ReactElement<{ id?: string }> }) {
  const generatedId = useId();
  const id = children.props.id ?? generatedId;
  return <div className="exploratory__field"><label htmlFor={id}>{label}</label>{cloneElement(children, { id })}</div>;
}
function Check({ label, checked, onChange }: { label: string; checked: boolean; onChange: (value: boolean) => void }) {
  return <label className="exploratory__check"><input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />{label}</label>;
}
function NumberField({ label, value, onChange, min, max, step = 1 }: {
  label: string; value: number; onChange: (value: number) => void; min: number; max: number; step?: number;
}) {
  return <Field label={label}><input type="number" value={value} min={min} max={max} step={step} onChange={(e) => onChange(Number(e.target.value))} /></Field>;
}
function MultiSelect({ label, options, selected, onChange }: {
  label: string; options: string[]; selected: string[]; onChange: (value: string[]) => void;
}) {
  const [search, setSearch] = useState("");
  return <fieldset className="exploratory__picker"><legend>{label} ({selected.length})</legend>
    <input aria-label={`Search ${label}`} value={search} onChange={(e) => setSearch(e.target.value)} placeholder="Search features" />
    <div className="exploratory__actions"><button type="button" className="btn" onClick={() => onChange(options)}>Select all</button>
      <button type="button" className="btn" onClick={() => onChange([])}>None</button></div>
    <div className="exploratory__choices">{options.filter((item) => item.toLowerCase().includes(search.toLowerCase())).map((item) => <Check key={item} label={item} checked={selected.includes(item)} onChange={(checked) => onChange(checked ? [...selected, item] : selected.filter((f) => f !== item))} />)}</div>
    {!options.length && <span className="muted">No available columns.</span>}
  </fieldset>;
}
type Policy = Required<Pick<ScopeRequest, "pre_transform" | "missing_policy" | "constant_policy">>;
function Preprocessing({ value, onChange, scope }: { value: Policy; onChange: (value: Policy) => void; scope: AnalysisScope }) {
  return <details><summary>Preprocessing and missing values</summary>
    <Field label="Pre-analysis transform"><select value={value.pre_transform} onChange={(e) => onChange({ ...value, pre_transform: e.target.value as Policy["pre_transform"] })}>
      {["none", "first_difference", "percent_change", "log_difference", "absolute_return", "squared_return", "rank_transform", "winsorize_1_99", "robust_scale", "standard_scale", "minmax_scale"].map((name) => <option key={name}>{name}</option>)}</select></Field>
    <Field label="Missing policy"><select value={value.missing_policy} onChange={(e) => onChange({ ...value, missing_policy: e.target.value as Policy["missing_policy"] })}>
      {["pairwise_complete", "drop_any_missing", "median_impute_scope", "mean_impute_scope", "leave_nan"].map((name) => <option key={name}>{name}</option>)}</select></Field>
    <Field label="Constant policy"><select value={value.constant_policy} onChange={(e) => onChange({ ...value, constant_policy: e.target.value as Policy["constant_policy"] })}>
      {["exclude_constants", "warn_only", "refuse_if_constants"].map((name) => <option key={name}>{name}</option>)}</select></Field>
    <p className="muted">Parameters fit within {scopeLabels[scope]}. {scope === "full_dataset" || scope === "custom_date_range" ? "Fitted preprocessing is non-causal and diagnostic." : "Only selected training rows or preceding rolling windows fit parameters."}</p>
  </details>;
}
function CorrelationControls({ value, onChange, features, rollingScope }: { value: Partial<CorrelationRequest>; onChange: (value: Partial<CorrelationRequest>) => void; features: string[]; rollingScope: boolean }) {
  const set = <K extends keyof CorrelationRequest>(key: K, next: CorrelationRequest[K]) => onChange({ ...value, [key]: next });
  const [bins, setBins] = useState("");
  const pairNames = features.flatMap((a, index) => features.slice(index + 1).map((b) => JSON.stringify([a, b])));
  return <>
    {!rollingScope && <details><summary>Inference and conditional groups</summary>
      <NumberField label="Minimum pair observations" value={value.min_obs ?? 30} onChange={(n) => set("min_obs", n)} min={3} max={1000000} />
      <NumberField label="Minimum group observations" value={value.group_min_obs ?? 30} onChange={(n) => set("group_min_obs", n)} min={3} max={1000000} />
      <Field label="Numeric group bin boundaries (comma separated)"><input value={bins} onChange={(e) => { setBins(e.target.value); set("group_bins", e.target.value.trim() ? e.target.value.split(",").map(Number) : undefined); }} /></Field>
      <Field label="Significance alpha"><select value={value.alpha} onChange={(e) => set("alpha", Number(e.target.value))}>{[.01, .05, .1].map((n) => <option key={n}>{n}</option>)}</select></Field>
      <Field label="Confidence level"><select value={value.confidence_level} onChange={(e) => set("confidence_level", Number(e.target.value))}>{[.9, .95, .99].map((n) => <option value={n} key={n}>{n * 100}%</option>)}</select></Field>
      <Field label="Multiple-testing correction"><select value={value.multiple_testing} onChange={(e) => set("multiple_testing", e.target.value as CorrelationRequest["multiple_testing"])}>{["none", "bonferroni", "benjamini_hochberg"].map((name) => <option key={name}>{name}</option>)}</select></Field>
      <Check label="Compute p-values" checked={value.compute_p_values ?? true} onChange={(checked) => set("compute_p_values", checked)} />
      <Check label="Compute confidence intervals" checked={value.compute_confidence_intervals ?? true} onChange={(checked) => set("compute_confidence_intervals", checked)} />
      <Check label="Paired bootstrap confidence intervals (slower)" checked={value.ci_method === "bootstrap"} onChange={(checked) => set("ci_method", checked ? "bootstrap" : undefined)} />
      <p className="muted">Pearson uses Fisher z. Rank-method CIs need bootstrap. Inference assumes independent observations.</p>
    </details>}
    {!rollingScope && <details><summary>VIF and redundancy</summary>
      <Check label="Calculate VIF" checked={value.calculate_vif ?? true} onChange={(checked) => set("calculate_vif", checked)} />
      <NumberField label="VIF threshold" value={value.vif_threshold ?? 5} onChange={(n) => set("vif_threshold", n)} min={1.01} max={1000} step={.5} />
      <Check label="Include intercept in VIF" checked={value.include_intercept ?? true} onChange={(checked) => set("include_intercept", checked)} />
      <Check label="Enable redundancy groups" checked={value.enable_redundancy_groups ?? true} onChange={(checked) => set("enable_redundancy_groups", checked)} />
    </details>}
    {rollingScope && <details open><summary>Rolling pair selection and sampling</summary>
      <NumberField label="Rolling step (bars)" value={value.rolling_step ?? 1} onChange={(n) => set("rolling_step", n)} min={1} max={100000} />
      <NumberField label="Rolling minimum periods" value={value.rolling_min_periods ?? 30} onChange={(n) => set("rolling_min_periods", n)} min={3} max={5000} />
      <Field label="Rolling pair selection"><select value={value.rolling_pair_selection} onChange={(e) => set("rolling_pair_selection", e.target.value as CorrelationRequest["rolling_pair_selection"])}><option value="all_pairs">All pairs summary only</option><option value="selected_pairs">User-selected pairs with series</option></select></Field>
      {value.rolling_pair_selection === "selected_pairs" && <MultiSelect label="Rolling pairs" options={pairNames} selected={(value.rolling_pairs ?? []).map((pair) => JSON.stringify(pair))} onChange={(selected) => set("rolling_pairs", selected.map((pair) => JSON.parse(pair) as [string, string]))} />}
    </details>}
  </>;
}
function ErrorNotice({ text }: { text: string }) { return <p className="result__bar result__bar--error" role="alert">{text}</p>; }
function Summary({ result }: { result: AnalysisResult }) {
  return <div className="exploratory__summary">
    <p><strong>{result.dataset_name}</strong> · {scopeLabels[result.fit_scope]} · {result.n_scope_rows.toLocaleString()} scope rows</p>
    <p>{result.causal_for_downstream_training ? "Causal C8 fit (training span or rolling series)" : "Non-causal for downstream training"}</p>
    <p>{result.features.length} features · {result.fit_rows} fit rows · {result.transform_rows} transform rows. Excluded constants: {result.excluded_constants.join(", ") || "None"}</p>
    <p className="muted">The causal_for_downstream_training flag describes C8 fitting only. It does not certify upstream features or the downstream pipeline.</p>
    <details><summary>Missing data and fitted parameters</summary><pre>{JSON.stringify({ missing: result.missing_summary, preprocessing: result.preprocessing_parameters }, null, 2)}</pre></details>
    {result.scope_start && <p className="muted">{result.time_column}: {result.scope_start} → {result.scope_end}</p>}
    {result.split_plan_id && <p>Split plan: <Link to={`/data/${result.split_plan_id}`}>{result.split_plan_id}</Link>{result.fold === null ? "" : ` · fold ${result.fold}`}</p>}
    <details><summary>Analysis notes and warnings ({result.warnings.length})</summary><ul>{result.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul></details>
  </div>;
}
function Table<T extends object>({ caption, rows, columns }: { caption: string; rows: T[]; columns: [keyof T, string][] }) {
  return <ResultGrid caption={caption} rows={rows} columns={columns} />;
}
const pairColumns: [keyof CorrelationPair, string][] = [["feature_x", "Feature X"], ["feature_y", "Feature Y"],
  ["correlation", "Correlation"], ["abs_correlation", "|r|"], ["p_value", "p-value"], ["adjusted_p_value", "Adjusted p"], ["ci_low", "CI low"], ["ci_high", "CI high"], ["ci_method", "CI method"], ["n_used", "n used"], ["significant", "Significant"], ["redundancy_flag", "Redundant"]];

function Chart({ title, data, extra }: { title: string; data: Data[]; extra?: Partial<Layout> }) {
  useTheme();
  const colors = chartTheme();
  const axis = { gridcolor: colors.grid, zerolinecolor: colors.zero, color: colors.axis };
  return <div className="exploratory__chart"><h3>{title}</h3><Plot data={data} layout={{
    height: 330, paper_bgcolor: colors.paper, plot_bgcolor: colors.plot, font: { color: colors.font },
    colorway: colors.series, margin: { l: 70, r: 25, t: 25, b: 75 },
    ...extra,
    xaxis: { ...axis, ...extra?.xaxis }, yaxis: { ...axis, ...extra?.yaxis },
    legend: { font: { color: colors.legend } },
  }} config={{ responsive: true, displaylogo: false }} style={{ width: "100%" }} /></div>;
}
export function Heatmap({ title, x, y, z, pairs }: { title: string; x: string[]; y: string[]; z: (number | null)[][]; pairs?: CorrelationPair[] }) {
  // Matplotlib RdBu_r: negative blue, neutral white, positive red.
  // Keep the entire heatmap independent of chartTheme, CSS variables and Plotly templates.
  const colorscale: [number, string][] = [
    [0, "#053061"], [0.1, "#2166ac"], [0.2, "#4393c3"], [0.3, "#92c5de"],
    [0.4, "#d1e5f0"], [0.5, "#f7f7f7"], [0.6, "#fddbc7"], [0.7, "#f4a582"],
    [0.8, "#d6604d"], [0.9, "#b2182b"], [1, "#67001f"],
  ];
  const textColor = "#1f2937";
  const [triangle, setTriangle] = useState("full");
  const [annotate, setAnnotate] = useState(false);
  const [threshold, setThreshold] = useState(0);
  const [highlight, setHighlight] = useState(true);
  const [ordered, setOrdered] = useState(false);
  const indices = x.map((_, i) => i);
  if (ordered && pairs) indices.sort((a, b) => z[b]!.reduce<number>((sum, n) => sum + Math.abs(n ?? 0), 0) - z[a]!.reduce<number>((sum, n) => sum + Math.abs(n ?? 0), 0) || x[a]!.localeCompare(x[b]!));
  const names = indices.map((i) => x[i]!);
  const yNames = pairs ? names : y;
  const matrix = pairs ? indices.map((i) => indices.map((j) => z[i]?.[j] ?? null)) : z;
  const values = heatmapValues(matrix, pairs ? triangle : "full", threshold);
  const hover = yNames.map((a, i) => names.map((b, j) => {
    const pair = pairs?.find((row) => row.feature_a === a && row.feature_b === b || row.feature_a === b && row.feature_b === a);
    return `${a} / ${b}<br>r: ${display(values[i]?.[j])}${pair ? `<br>p: ${display(pair.p_value)}; adjusted p: ${display(pair.adjusted_p_value)}<br>n: ${pair.n_used}<br>CI: ${display(pair.ci_low)} to ${display(pair.ci_high)} (${pair.ci_method})` : ""}`;
  }));
  const marked = (pairs ?? []).filter((row) => row.significant && (row.abs_correlation ?? 0) >= threshold);
  return <>
    <div className="exploratory__actions">
      {pairs && <Field label="Heatmap triangle"><select value={triangle} onChange={(e) => setTriangle(e.target.value)}>{["full", "upper", "lower"].map((name) => <option key={name}>{name}</option>)}</select></Field>}
      <NumberField label="Heatmap minimum absolute value" value={threshold} onChange={setThreshold} min={0} max={1} step={.05} />
      <Check label="Annotate cells" checked={annotate} onChange={setAnnotate} />
      {pairs && <><Check label="Highlight significant pairs" checked={highlight} onChange={setHighlight} /><Check label="Order by total absolute correlation" checked={ordered} onChange={setOrdered} /></>}
    </div>
    <div className="exploratory__chart"><h3>{title}</h3><Plot data={[{
      type: "heatmap", x: names, y: yNames, z: values,
      zmin: -1, zmax: 1, zmid: 0, zauto: false,
      colorscale, autocolorscale: false, reversescale: false, showscale: true,
      colorbar: { tickvals: [-1, -0.5, 0, 0.5, 1], tickfont: { color: textColor } },
      text: hover, hoverinfo: "text", hoverongaps: false,
      hoverlabel: { bgcolor: "#ffffff", font: { color: textColor } },
    }, ...(pairs && highlight ? [{
      type: "scatter" as const, mode: "markers" as const, name: "Significant after correction",
      x: marked.map((p) => p.feature_b), y: marked.map((p) => p.feature_a),
      marker: { symbol: "square-open" as const, size: 18, color: textColor },
      hoverinfo: "skip" as const,
    }] : [])]} layout={{
      template: {}, paper_bgcolor: "#ffffff", plot_bgcolor: "#ffffff", font: { color: textColor },
      height: Math.max(330, Math.min(750, y.length * 25)),
      margin: { l: 110, b: 110, r: 80, t: 25 },
      xaxis: { color: textColor, showgrid: false, zeroline: false },
      yaxis: { color: textColor, showgrid: false, zeroline: false },
      legend: { font: { color: textColor } },
      annotations: annotate ? values.flatMap((row, i) => row.flatMap((value, j) => value === null ? [] : [{
        x: names[j]!, y: yNames[i]!, text: value.toFixed(2), showarrow: false,
        font: { color: textColor, size: 10 }, bgcolor: "rgba(255,255,255,0.85)", borderpad: 2,
      }])) : [],
    }} config={{ responsive: true, displaylogo: false }} style={{ width: "100%" }} /></div>
  </>;
}
function CorrelationResults({ result }: { result: CorrelationResult }) {
  const [tab, setTab] = useState(result.matrix.length ? "Correlation Heatmap" : "Overview");
  const [groupIndex, setGroupIndex] = useState(0);
  const [minAbs, setMinAbs] = useState(0);
  const [maxP, setMaxP] = useState(1);
  const [significant, setSignificant] = useState(false);
  const [redundant, setRedundant] = useState(false);
  const colors = chartTheme();
  useTheme();
  const hasStatic = result.matrix.length > 0;
  const hasRolling = result.rolling.length > 0 || result.rolling_summary.length > 0;
  const tabs = ["Overview", ...(hasStatic ? ["Correlation Heatmap", "Pairwise Table"] : []), ...(result.conditional.length ? ["Conditional Correlation"] : []), ...(hasStatic ? ["VIF / Multicollinearity"] : []), ...(result.ccf.length ? ["Lead–lag / CCF"] : []), ...(hasRolling ? ["Rolling Correlation"] : []), ...(hasStatic ? ["Redundancy Groups"] : [])];
  const pairs = result.pairwise.filter((pair) => (pair.abs_correlation ?? -1) >= minAbs && (maxP === 1 || (pair.adjusted_p_value ?? 2) <= maxP) && (!significant || pair.significant) && (!redundant || pair.redundancy_flag));
  const selectedGroup = result.conditional[Math.min(groupIndex, result.conditional.length - 1)];
  const vif = [...result.vif].sort((a, b) => (b.vif ?? 0) - (a.vif ?? 0));
  const vifThreshold = Number(result.metadata.vif_threshold ?? 5);
  return <div data-testid="correlation-results">
    <Summary result={result} />
    <div className="exploratory__tabs" role="tablist">{tabs.map((name) => <button className="btn" role="tab" aria-selected={tab === name} key={name} onClick={() => setTab(name)}>{name}</button>)}</div>
    {tab === "Overview" && <><p>Correction: {result.multiple_testing_method}. {result.notes.map((note) => <span key={note}>{note} </span>)}</p><DownstreamGuidance /></>}
    {tab === "Correlation Heatmap" && <><Heatmap title={`${methodLabels[result.method]} correlation matrix`} x={result.features} y={result.features} z={result.matrix} pairs={result.pairwise} /><Table caption="Pairwise correlations" rows={result.pairwise} columns={pairColumns} /></>}
    {tab === "Pairwise Table" && <><div className="exploratory__actions"><NumberField label="Minimum |correlation|" value={minAbs} onChange={setMinAbs} min={0} max={1} step={.05} /><NumberField label="Maximum adjusted p-value" value={maxP} onChange={setMaxP} min={0} max={1} step={.01} /><Check label="Significant only" checked={significant} onChange={setSignificant} /><Check label="Redundant only" checked={redundant} onChange={setRedundant} /></div><Table caption="Filtered pairwise correlations" rows={pairs} columns={pairColumns} /></>}
    {tab === "Conditional Correlation" && selectedGroup && <><Field label="Conditional group"><select value={groupIndex} onChange={(e) => setGroupIndex(Number(e.target.value))}>{result.conditional.map((item, index) => <option key={index} value={index}>{item.group ?? "Missing"} · {item.rows} rows</option>)}</select></Field>{selectedGroup.warnings.map((warning) => <p className="exploratory__notice" key={warning}>{warning}</p>)}<Heatmap title="Conditional correlation" x={result.features} y={result.features} z={selectedGroup.matrix} pairs={selectedGroup.pairwise} /><Table caption="Group pairwise correlations" rows={selectedGroup.pairwise} columns={pairColumns} /></>}
    {tab === "VIF / Multicollinearity" && <><p>Intercept: {display(result.metadata.include_intercept)}. Perfect collinearity is shown as 1e30 (effectively infinite).</p><Chart title="Variance inflation factors" data={[{ type: "bar", x: vif.map((row) => row.feature), y: vif.map((row) => row.vif), marker: { color: vif.map((row) => (row.vif ?? 0) >= vifThreshold ? colors.series[3]! : colors.series[0]!) } }]} extra={{ yaxis: { type: "log" }, shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: vifThreshold, y1: vifThreshold, line: { color: colors.axis, dash: "dash" } }] }} /><Table caption="Variance inflation factors (VIF)" rows={vif} columns={[["feature", "Feature"], ["vif", "VIF"], ["tolerance", "Tolerance"], ["n_used", "n used"], ["condition_flag", "Condition"], ["suggested_action", "Suggested action"]]} /></>}
    {tab === "Redundancy Groups" && <Table caption={`Redundancy groups · |r| ≥ ${result.redundancy_threshold}`} rows={result.redundancy_groups} columns={[["group_id", "Group"], ["members", "Members"], ["representative", "Representative"], ["min_pairwise_abs_correlation", "Minimum |r|"], ["max_pairwise_abs_correlation", "Maximum |r|"], ["average_pairwise_abs_correlation", "Mean |r|"], ["suggested_action", "Suggested action"]]} />}
    {tab === "Lead–lag / CCF" && <>
      <p className="exploratory__notice">Positive-lag CCF uses future target values and must not be used as causal trading features. Create causal lagged features in C5. corr_L = corr(x_t, y_&#123;t+L&#125;); positive L means x leads y.</p>
      <Chart title="Lead–lag cross-correlation" data={[...new Set(result.ccf.map((p) => p.feature_x))].map((feature) => {
        const rows = result.ccf.filter((p) => p.feature_x === feature);
        return { type: "scatter", mode: "lines+markers", name: feature, x: rows.map((r) => r.lag), y: rows.map((r) => r.correlation) };
      })} extra={{ xaxis: { title: { text: "Lag L (observed rows); positive = x leads target" } }, yaxis: { range: [-1, 1] } }} />
      <Table caption="Lead–lag pairs" rows={result.ccf} columns={[["lag", "Lag L"], ...pairColumns]} />
      <Table caption="Best lead–lag relationships" rows={result.ccf_summary} columns={[["feature", "Feature"], ["best_positive_lag", "Best positive lag"], ["best_negative_lag", "Best negative lag"], ["max_absolute_predictive_correlation", "Max absolute predictive correlation"]]} />
    </>}
    {tab === "Rolling Correlation" && <>
      <p className="muted">Causal {methodLabels[result.method]} windows use preceding rows only. Stability = 1 / (1 + sample standard deviation). Rolling PCA is deferred.</p>
      {result.rolling.length ? <Chart title={`Causal rolling ${methodLabels[result.method]} correlation`} data={[...new Set(result.rolling.map((p) => p.feature))].map((feature) => {
        const rows = result.rolling.filter((p) => p.feature === feature);
        return { type: "scatter", mode: "lines", name: feature, x: rows.map((r) => r.time), y: rows.map((r) => r.correlation), connectgaps: false };
      })} extra={{ yaxis: { range: [-1, 1] } }} /> : <p className="muted">All-pairs mode returns the complete summary without materializing every row-level series. Choose user-selected pairs to plot lines.</p>}
      {result.rolling.length > 0 && <Table caption="Rolling correlations" rows={result.rolling} columns={[["time", "Observation time"], ["feature", "Feature"], ["correlation", `${methodLabels[result.method]} correlation`], ["n_used", "n used"]]} />}
      <Table caption="Rolling summary" rows={result.rolling_summary} columns={[["feature_a", "Feature A"], ["feature_b", "Feature B"], ["mean_corr", "Mean"], ["std_corr", "Std"], ["min_corr", "Minimum"], ["max_corr", "Maximum"], ["latest_corr", "Latest"], ["stability_score", "Stability"], ["n_points", "Points"]]} />
    </>}
  </div>;
}
function DownstreamGuidance() {
  return <div id="exploratory-downstream"><p><strong>D — Regime detection:</strong> Prefer train-span PCA components or low-VIF features for causal training. Full-sample PCA remains available for exploration; rolling and conditional correlations can reveal changing relationships.</p><p><strong>E — Volatility forecasting:</strong> Use VIF to prune redundant estimators and train-span PCA for causal feature sets. Full-sample PCA is labeled non-causal.</p><p><strong>F — Return prediction:</strong> Use CCF to discover candidate lead–lag relationships, then reconstruct them causally in C5 with explicit lags. Partial correlation controls for other factors; train-span PCA provides orthogonal predictors.</p></div>;
}
function PCAResults({ result }: { result: PCAResult }) {
  const names = result.component_columns;
  const components = result.component_summary.map((row) => row.component);
  const scores = result.biplot.scores;
  const [xPC, setXPC] = useState(0);
  const [yPC, setYPC] = useState(Math.min(1, result.n_components - 1));
  const [topN, setTopN] = useState(10);
  const [loadingThreshold, setLoadingThreshold] = useState(0);
  const colors = chartTheme();
  useTheme();
  const xIndex = Math.min(xPC, result.n_components - 1), yIndex = Math.min(yPC, result.n_components - 1);
  const topFeatures = new Set<number>();
  for (let i = 0; i < result.n_components; i++) {
    result.features.map((_, index) => index).filter((index) => Math.abs(result.loadings[index]?.[i] ?? 0) >= loadingThreshold)
      .sort((a, b) => Math.abs(result.loadings[b]?.[i] ?? 0) - Math.abs(result.loadings[a]?.[i] ?? 0)).slice(0, topN).forEach((index) => topFeatures.add(index));
  }
  const displayed = [...topFeatures].sort((a, b) => a - b);
  return <div data-testid="pca-results">
    <Summary result={result} />
    <p>{result.n_components} components · {result.n_fit_rows.toLocaleString()} complete fit rows · {result.standardizer} scaling · {result.component_selection}</p>
    <PCASavedBanner result={result} />
    <p>NaN score rows: {result.n_nan_scores}. Full-component reconstruction error: {display(result.reconstruction_error)}.</p>
    <div className="exploratory__chart-grid">
      <Chart title="Scree plot" data={[{ type: "bar", x: components, y: result.eigenvalues, name: "Eigenvalue" }]} extra={{ yaxis: { title: { text: "Eigenvalue / explained variance" } } }} />
      <Chart title="Cumulative explained variance" data={[{ type: "scatter", mode: "lines+markers", x: components, y: result.cumulative_explained_variance_ratio, name: "Cumulative ratio" }]} extra={{ yaxis: { range: [0, 1.05], tickformat: ".0%" }, shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: result.variance_threshold, y1: result.variance_threshold, line: { color: colors.axis, dash: "dash" } }] }} />
    </div>
    <div className="exploratory__actions"><NumberField label="Top features per component" value={topN} onChange={setTopN} min={1} max={result.features.length} /><NumberField label="Minimum absolute loading" value={loadingThreshold} onChange={setLoadingThreshold} min={0} max={1} step={.05} /></div>
    <Heatmap title="Component loadings (unit eigenvectors)" x={names} y={displayed.map((i) => result.features[i]!)} z={displayed.map((i) => result.loadings[i]!)} />
    <Table caption="Component summary" rows={result.component_summary} columns={[["component", "Component"], ["eigenvalue", "Eigenvalue"], ["explained_variance_ratio", "Variance ratio"], ["cumulative_ratio", "Cumulative ratio"], ["selected", "Retained"]]} />
    <div className="exploratory__actions"><Field label="Biplot X component"><select value={xIndex} onChange={(e) => setXPC(Number(e.target.value))}>{names.map((name, i) => <option key={name} value={i}>{name}</option>)}</select></Field><Field label="Biplot Y component"><select value={yIndex} onChange={(e) => setYPC(Number(e.target.value))}>{names.map((name, i) => <option key={name} value={i}>{name}</option>)}</select></Field></div>
    <Chart title="PCA biplot" data={[
      { type: "scatter", mode: "markers", name: "Fit observations", x: scores.map((row) => row[xIndex] ?? 0), y: scores.map((row) => result.n_components > 1 ? row[yIndex] ?? 0 : 0),
        text: result.biplot.times ?? result.biplot.row_indices.map(String), marker: { size: 5, opacity: 0.65 } },
      ...result.features.map((feature, index): Data => ({ type: "scatter", mode: "lines+markers+text", name: feature,
        x: [0, result.biplot.loadings[index]?.[xIndex] ?? 0], y: [0, result.n_components > 1 ? result.biplot.loadings[index]?.[yIndex] ?? 0 : 0],
        text: ["", feature], textposition: "top center", marker: { size: 6 }, })),
    ]} extra={{ xaxis: { title: { text: `${names[xIndex]} score / loading direction` } }, yaxis: { title: { text: result.n_components > 1 ? `${names[yIndex]} score / loading direction` : "One retained component (y = 0)" } } }} />
    <p className="muted">Up to 500 evenly sampled fit observations. Lines show unit loading directions; their lengths are not score magnitudes. Largest absolute loading is positive, with ties resolved in feature order.</p>
    <DownstreamGuidance />
  </div>;
}
export function PCASavedBanner({ result }: { result: PCAResult }) {
  if (!result.dataset) return null;
  return <div className={result.causal_for_downstream_training ? "result__bar result__bar--ok" : "exploratory__notice"} data-testid="pca-saved" role="status">
    Saved dataset: <Link to={`/data/${result.dataset.dataset_id}`}>{result.dataset.name}</Link>
    {result.params_dataset && <> · <Link to={`/data/${result.params_dataset.dataset_id}`}>PCA parameters and loadings</Link></>}
    <p>{result.dataset.category} · {scopeLabels[result.fit_scope]} · {result.causal_for_downstream_training ? "Train-span C8 fit" : "Non-causal fit"}</p>
    <p>The causal flag describes C8 fitting only; review upstream features before downstream training.</p>
  </div>;
}
