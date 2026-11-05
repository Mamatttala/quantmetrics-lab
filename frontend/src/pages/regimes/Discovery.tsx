import { useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { regimes } from "@/api/regimes";
import type { Artifact, ExperimentConfig, ModelSpec, Options } from "@/api/regimes";
import { PageHeader } from "@/components/PageHeader";
import { Details, Empty, Field, Metrics, Navigation, Parameters, Section, Status, Table, useTask } from "./shared";
import { Chart } from "./charts";

const initial: ExperimentConfig = {
  name: "BTCUSDT regime experiment", inputs: { dataset_id: "", split_plan_id: "", features: [], fold: 0 },
  model: "gaussian_hmm", parameters: {}, seed: 42,
  states: { mode: "fixed", count: 3, candidates: [2, 3, 4], template: "statistical", minimum_occupancy: 0.01, criterion: "bic" },
  target: null, search: { method: "manual", space: {}, trials: 12 },
};
const targetDefault = { source: "rules", rule_feature: "", column: "", objective: "current", horizon: 1, availability_lag: 0, lower: -0.001, upper: 0.001 };

export function Discovery() {
  const [options, setOptions] = useState<Options | null>(null);
  const [config, setConfig] = useState<ExperimentConfig>(initial);
  const [folds, setFolds] = useState<number[]>([]);
  const [experiments, setExperiments] = useState<Artifact[]>([]);
  const [result, setResult] = useState<Artifact | null>(null);
  const [queue, setQueue] = useState<ExperimentConfig[]>([]);
  const [selected, setSelected] = useState<Artifact | null>(null);
  const [spaceText, setSpaceText] = useState("{}");
  const [candidateText, setCandidateText] = useState("2, 3, 4");
  const task = useTask();
  const location = useLocation();
  const model = options?.models.find(item => item.id === config.model);
  const features = options?.features ?? [];
  const selectedFamily = model?.family;

  useEffect(() => {
    let active = true;
    void regimes.options(config.inputs.dataset_id).then(value => { if (active) setOptions(value); }).catch(error => { if (active) task.setError(String(error)); });
    return () => { active = false; };
  }, [config.inputs.dataset_id]);
  useEffect(() => {
    let active = true;
    void regimes.list().then(rows => { if (active) setExperiments(rows); }).catch(error => { if (active) task.setError(String(error)); });
    return () => { active = false; };
  }, []);
  useEffect(() => {
    if (!config.inputs.split_plan_id) { setFolds([]); return; }
    let active = true;
    void regimes.plan(config.inputs.split_plan_id).then(rows => {
      if (!active) return;
      const values = [...new Set(rows.map(row => row.fold))]; setFolds(values);
      setConfig(current => ({ ...current, inputs: { ...current.inputs, fold: values[0] ?? 0 } }));
    }).catch(error => task.setError(String(error)));
    return () => { active = false; };
  }, [config.inputs.split_plan_id]);
  useEffect(() => {
    if (location.hash) document.getElementById(location.hash.slice(1))?.scrollIntoView();
  }, [location.hash, options]);

  function choose(spec: ModelSpec) {
    setConfig(current => ({ ...current, model: spec.id, parameters: {},
      target: spec.family === "hybrid" ? { ...targetDefault, source: "teacher" } : spec.family === "supervised" ? { ...targetDefault } : null,
      states: { ...current.states, mode: "fixed", criterion: spec.criteria[0] ?? "bic" }, search: { ...current.search, method: "manual", space: {} } }));
    setSpaceText("{}");
  }
  const input = (key: string, value: unknown) => setConfig(current => ({ ...current, inputs: { ...current.inputs, [key]: value } }));
  function validatedConfig() {
    if (document.querySelector('.regime-workspace textarea[aria-invalid="true"]')) throw new Error("Correct invalid model-parameter JSON before submitting.");
    const parsed: unknown = JSON.parse(spaceText);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed) || Object.values(parsed).some(value => !Array.isArray(value))) throw new Error("Search space must map supported parameter names to arrays of candidate values.");
    const candidates = candidateText.split(",").map(value => Number(value.trim()));
    if (candidates.some(value => !Number.isInteger(value) || value < 2 || value > 12)) throw new Error("Candidate counts must be integers between 2 and 12.");
    if (!config.inputs.dataset_id || !config.inputs.split_plan_id || !config.inputs.features.length) throw new Error("Select prepared data, its C6 split plan, and input features.");
    if (config.inputs.return_column && !config.inputs.return_type) throw new Error("Select simple or log returns for the return column.");
    return { ...config, search: { ...config.search, space: parsed as Record<string, unknown[]> }, states: { ...config.states, candidates } };
  }
  async function run() {
    let submissions: ExperimentConfig[];
    try { submissions = queue.length ? queue : [validatedConfig()]; } catch (error) { task.setError(String(error)); return; }
    // Each request captures an independent configuration and saves its own models.
    await task.run<Artifact>(async (signal, progress) => {
      let last: Artifact | null = null;
      const { waitForJob, failureMessage } = await import("@/api/jobs");
      for (const item of submissions) {
        if (signal.aborted) throw new Error("Page closed; remaining staged submissions were not started.");
        const accepted = await regimes.fit(item);
        if ("job_id" in accepted) {
          const job = await waitForJob<Artifact>(accepted.job_id, { signal, onPoll: row => progress(row.message ?? "Training…") });
          if (job.status === "failed" || !job.result) throw new Error(failureMessage(job));
          last = job.result;
        } else last = accepted;
      }
      if (!last) throw new Error("No experiment was submitted.");
      return last;
    }, async value => { setResult(value); setQueue([]); setExperiments(await regimes.list()); if (value.selected_experiment_id) setSelected(await regimes.get(value.selected_experiment_id)); });
  }
  function family(id: string, title: string, families: string[]) {
    const entries = options?.models.filter(item => families.includes(item.family)) ?? [];
    return <Section id={id} title={title}><div className="regime-models">{entries.map(item => <button key={item.id} type="button" className={`regime-model ${config.model === item.id ? "selected" : ""}`} aria-pressed={config.model === item.id} disabled={!item.available || task.busy} onClick={() => choose(item)}>
      <strong>{item.name}</strong><small>{item.available ? item.probabilities ? "State / class probabilities" : item.family === "structural" ? "Structural change events" : "Assignments without probabilities" : item.availability_reason}</small>
    </button>)}</div>
      {model && families.includes(model.family) && <><h3>{model.name} configuration</h3><Parameters fields={model.parameters} values={config.parameters} columns={config.inputs.features} onChange={(key, value) => setConfig(current => ({ ...current, parameters: { ...current.parameters, [key]: value } }))} /></>}
      {id === "baseline" && <p className="regime-note">Periods belong to the C5 indicator configuration. Select those prepared MA, ADX, directional-index and volatility columns here; thresholds are expressed in the selected columns’ units.</p>}
      {id === "unsupervised" && <p className="regime-note">PELT performs offline segmentation. CUSUM reports sequential changes. Their segment IDs do not represent recurring hidden states.</p>}
    </Section>;
  }

  return <div className="page regime-workspace"><PageHeader title="D9 · Model Selection & State Discovery" lead="Configure independent experiments against prepared observations. Fit on training data; compare candidates without using the test span." /><Navigation step="discovery" /><Status task={task} />
    <Section id="inputs" title="Prepared research inputs"><div className="regime-form">
      <Field label="Prepared dataset"><select value={config.inputs.dataset_id} onChange={e => setConfig(current => ({ ...current, parameters: {}, inputs: { ...current.inputs, dataset_id: e.target.value, split_plan_id: "", features: [] } }))}><option value="">Select C5–C8 observations</option>{options?.datasets.map(row => <option key={row.dataset_id} value={row.dataset_id}>{row.name} · {row.symbol} · {row.timeframe}</option>)}</select></Field>
      <Field label="C6 chronological split plan"><select value={config.inputs.split_plan_id} onChange={e => input("split_plan_id", e.target.value)}><option value="">Select saved plan</option>{options?.plans.map(row => <option key={row.dataset_id} value={row.dataset_id}>{row.name}</option>)}</select></Field>
      <Field label="Training fold"><select value={config.inputs.fold} onChange={e => input("fold", Number(e.target.value))}>{folds.map(fold => <option key={fold} value={fold}>{fold}</option>)}</select></Field>
    </div>{!features.length ? <Empty>Select a prepared dataset to inspect its existing features and fitted artifacts.</Empty> : <><h3>Input features · {config.inputs.features.length} selected</h3><div className="regime-chips">{features.map(feature => <label key={feature}><input type="checkbox" checked={config.inputs.features.includes(feature)} onChange={e => input("features", e.target.checked ? [...config.inputs.features, feature] : config.inputs.features.filter(value => value !== feature))} />{feature}</label>)}</div></>}
      <Details title="Associated C7 / C8 fitted preprocessing" value={options?.preprocessing ?? []} />
      <details><summary>Economic diagnostics inputs</summary><div className="regime-form"><Field label="Original market observations"><select value={config.inputs.market_dataset_id ?? ""} onChange={e => input("market_dataset_id", e.target.value || null)}><option value="">Resolve original-price ancestor</option>{options?.datasets.map(row => <option key={row.dataset_id} value={row.dataset_id}>{row.name}</option>)}</select></Field>{["price_column", "return_column", "adx_column", "rsi_column", "volatility_column"].map(key => <Field key={key} label={key.replaceAll("_", " ")}><input placeholder={key === "price_column" ? "close" : "Existing column name"} onChange={e => input(key, e.target.value || null)} /></Field>)}<Field label="Return convention" hint="Required for a return column. Price-derived returns are simple."><select value={config.inputs.return_type ?? ""} onChange={e => input("return_type", e.target.value || null)}><option value="">Select convention</option><option value="simple">Simple returns</option><option value="log">Log returns</option></select></Field></div></details>
    </Section>
    {family("baseline", "D9.1 · Baseline Models", ["baseline"])}
    {family("unsupervised", "D9.2 · Unsupervised Models & Structural Changes", ["hmm", "clustering", "structural"])}
    {family("supervised", "D9.3 · Supervised Models", ["supervised"])}
    {family("hybrid", "D9.4 · Hybrid Models", ["hybrid"])}
    {config.target && <Section id="targets" title="Training target & prediction origin"><div className="regime-form">
      <Field label="Target source"><select value={config.target.source} disabled={selectedFamily === "hybrid"} onChange={e => setConfig(current => ({ ...current, target: { ...current.target!, source: e.target.value } }))}>{selectedFamily === "hybrid" ? <option value="teacher">Causal HMM teacher</option> : <><option value="rules">Threshold labels from an upstream feature</option><option value="column">Existing label column</option></>}</select></Field>
      <Field label="Objective"><select value={config.target.objective} onChange={e => setConfig(current => ({ ...current, target: { ...current.target!, objective: e.target.value } }))}><option value="current">Current-state classification / imitation</option><option value="future">Future-state forecasting</option></select></Field>
      {config.target.source !== "teacher" && <Field label="Target feature / label column"><select value={config.target.source === "column" ? config.target.column ?? "" : config.target.rule_feature ?? ""} onChange={e => setConfig(current => ({ ...current, target: { ...current.target!, [current.target!.source === "column" ? "column" : "rule_feature"]: e.target.value } }))}><option value="">Select upstream column</option>{options?.columns.map(column => <option key={column}>{column}</option>)}</select></Field>}
      {(config.target.objective === "future" ? ["horizon", "availability_lag"] : ["availability_lag"]).map(key => <Field key={key} label={key.replaceAll("_", " ")}><input type="number" min={key === "horizon" ? 1 : 0} value={config.target![key as "horizon" | "availability_lag"]} onChange={e => setConfig(current => ({ ...current, target: { ...current.target!, [key]: Number(e.target.value) } }))} /></Field>)}
      {config.target.source === "rules" && ["lower", "upper"].map(key => <Field key={key} label={`${key} label threshold`}><input type="number" step="any" value={config.target![key as "lower" | "upper"]} onChange={e => setConfig(current => ({ ...current, target: { ...current.target!, [key]: Number(e.target.value) } }))} /></Field>)}
    </div><p className="regime-note">Labels and availability lags are purged at chronological boundaries. Teacher agreement measures imitation of the teacher’s definition; it does not establish independent economic validity.</p></Section>}
    <Section id="states" title="D9.5 · State Configuration & Discovery">{model?.state_count ? <div className="regime-form">
      <Field label="State configuration"><select value={config.states.template} onChange={e => { const template = e.target.value; const counts: Record<string, number> = { two: 2, bull_bear_sideways: 3, four: 4, trend_volatility: 6 }; setConfig(current => ({ ...current, states: { ...current.states, template, count: counts[template] ?? current.states.count } })); }}>{["statistical", "two", "bull_bear_sideways", "four", "trend_volatility", "custom"].map(value => <option key={value} value={value}>{value.replaceAll("_", " ")}</option>)}</select></Field>
      <Field label="Count selection"><select value={config.states.mode} onChange={e => setConfig(current => ({ ...current, states: { ...current.states, mode: e.target.value } }))}><option value="fixed">Fixed count</option><option value="automatic">Automatic candidate evaluation</option></select></Field>
      {config.states.mode === "fixed" ? <Field label="States / clusters"><input type="number" min={2} max={12} value={config.states.count} onChange={e => setConfig(current => ({ ...current, states: { ...current.states, count: Number(e.target.value) } }))} /></Field> : <Field label="Candidate counts"><input value={candidateText} onChange={e => setCandidateText(e.target.value)} /></Field>}
      <Field label="Minimum training occupancy"><input type="number" min={0} max={0.49} step={0.01} value={config.states.minimum_occupancy} onChange={e => setConfig(current => ({ ...current, states: { ...current.states, minimum_occupancy: Number(e.target.value) } }))} /></Field>
    </div> : <Empty>This method determines classes from its rules, supervised targets, or structural segments.</Empty>}<p className="regime-note">Templates set a starting count. Statistical state IDs remain unchanged; semantic labels are assigned separately in D10.</p></Section>
    <Section id="optimization" title="D9.6 · Hyperparameter Optimization"><div className="regime-form"><Field label="Optimization method"><select value={config.search.method} onChange={e => setConfig(current => ({ ...current, search: { ...current.search, method: e.target.value } }))}>{Object.entries(options?.optimization ?? { manual: true }).map(([key, enabled]) => <option key={key} value={key} disabled={!enabled || (key !== "manual" && !model?.criteria.length)}>{key}</option>)}</select></Field>
      {(config.search.method !== "manual" || config.states.mode === "automatic") && <Field label="Selection criterion"><select value={config.states.criterion} onChange={e => setConfig(current => ({ ...current, states: { ...current.states, criterion: e.target.value } }))}>{model?.criteria.map(key => <option key={key}>{key}</option>)}</select></Field>}
      {config.search.method !== "manual" && <Field label="Trial budget"><input type="number" min={1} max={100} value={config.search.trials} onChange={e => setConfig(current => ({ ...current, search: { ...current.search, trials: Number(e.target.value) } }))} /></Field>}
    </div>{config.search.method !== "manual" && <Field label="Model-specific search space (JSON)" hint={`Supported: ${Object.keys(model?.parameters ?? {}).join(", ")}. Bayesian and Optuna use seeded TPE sampling.`}><textarea rows={4} value={spaceText} onChange={e => setSpaceText(e.target.value)} placeholder={'{"max_iter": [50, 100, 200]}'} /></Field>}</Section>
    <Section id="training" title="D9.7 · Model Fitting & Training"><div className="regime-form"><Field label="Experiment name"><input value={config.name} onChange={e => setConfig(current => ({ ...current, name: e.target.value }))} /></Field><Field label="Random seed"><input type="number" min={0} value={config.seed} onChange={e => setConfig(current => ({ ...current, seed: Number(e.target.value) }))} /></Field></div>
      <div className="regime-actions"><button disabled={task.busy} onClick={() => { try { const staged = structuredClone(validatedConfig()); setQueue(items => [...items, staged]); } catch (error) { task.setError(String(error)); } }}>Stage independent experiment</button><button className="primary" disabled={task.busy || !model?.available} onClick={() => void run()}>{task.busy ? "Training…" : queue.length ? `Run ${queue.length} staged experiments` : "Fit model & save artifacts"}</button>{queue.length > 0 && <button onClick={() => setQueue([])}>Clear staged experiments</button>}</div>
      {queue.length > 0 && <Table caption="Staged experiments" rows={queue.map(item => ({ name: item.name, model: item.model, features: item.inputs.features.join(", "), count: item.states.count, seed: item.seed }))} />}<Status task={task} />
    </Section>
    <Section id="results" title="D9.8 · Training Results">{result?.trials && <Table caption="Candidate trials" rows={result.trials} keys={["number", "count", "status", "score", "seconds", "artifact_id", "error"]} />}
      {experiments.length ? <><div className="regime-form"><Field label="Saved experiment"><select value={selected?.id ?? ""} onChange={e => setSelected(experiments.find(item => item.id === e.target.value) ?? null)}><option value="">Inspect a trained candidate</option>{experiments.map(item => <option key={item.id} value={item.id}>{item.name} · {item.model} · {item.id.slice(-6)}</option>)}</select></Field></div><Table caption="Saved trained candidates" rows={experiments.map(item => ({ id: item.id, name: item.name, model: item.model, states: item.state_count, created: item.created_at, seconds: item.diagnostics?.seconds }))} /></> : <Empty>Run an experiment to create genuine state assignments, diagnostics, training logs, and model artifacts.</Empty>}
      {selected && <><Metrics values={selected.diagnostics ?? {}} /><Details title="Fitted diagnostics and convergence" value={selected.diagnostics} /><Details title="Learned model parameters" value={selected.fitted_parameters} /><Details title="Saved experiment configuration" value={selected.config} />{Array.isArray(selected.diagnostics?.log_likelihood_history) && <Chart title="EM convergence · training log likelihood" data={[{ type: "scatter", mode: "lines", y: selected.diagnostics.log_likelihood_history as number[] }]} />}<div className="regime-actions"><Link className="primary" to={`/regime/evaluation?experiment=${selected.id}`}>Evaluate in D10 →</Link><a href={regimes.exportUrl(selected.id, "json")} target="_blank" rel="noreferrer">Export experiment JSON</a><a href={regimes.exportUrl(selected.id, "csv")}>Assignments CSV</a><a href={regimes.bundleUrl(selected.id)}>Model & preprocessing bundle</a></div></>}
    </Section>
  </div>;
}
