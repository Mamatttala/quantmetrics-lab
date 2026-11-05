import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { regimes } from "@/api/regimes";
import type { Artifact, EvaluationResult, Options } from "@/api/regimes";
import { PageHeader } from "@/components/PageHeader";
import { ArtifactSelect, Details, Empty, Field, Metrics, Navigation, Section, Status, Table, useTask } from "./shared";
import { Chart, Matrix, StateChart } from "./charts";
import { chartTheme, useTheme } from "@/theme";

export function Evaluation() {
  useTheme();
  const theme = chartTheme();
  const [query] = useSearchParams();
  const [experiments, setExperiments] = useState<Artifact[]>([]);
  const [id, setId] = useState(query.get("experiment") ?? "");
  const [result, setResult] = useState<EvaluationResult | null>(null);
  const [partition, setPartition] = useState("test");
  const [labels, setLabels] = useState<Record<string, string>>({});
  const [labelMode, setLabelMode] = useState("automatic");
  const [template, setTemplate] = useState("trend");
  const [validationMode, setValidationMode] = useState("chronological");
  const [validationPlan, setValidationPlan] = useState("");
  const [foldInputs, setFoldInputs] = useState("{}");
  const [validation, setValidation] = useState<Artifact | null>(null);
  const [robust, setRobust] = useState<Artifact | null>(null);
  const [robustText, setRobustText] = useState('{"seeds": [7, 42, 123], "counts": [], "feature_sets": [], "configurations": [], "training_fractions": [1], "initializations": []}');
  const [options, setOptions] = useState<Options | null>(null);
  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [comparison, setComparison] = useState<{ rows: Record<string, unknown>[]; comparable: boolean; note: string } | null>(null);
  const [rationale, setRationale] = useState("");
  const [approval, setApproval] = useState<Artifact | null>(null);
  const task = useTask();
  const data = result?.partitions[partition];
  const selected = experiments.find(row => row.id === id);

  useEffect(() => {
    let active = true;
    void Promise.all([regimes.list(), regimes.options()]).then(([rows, opts]) => { if (active) { setExperiments(rows); setOptions(opts); } }).catch(error => { if (active) task.setError(String(error)); });
    return () => { active = false; };
  }, []);
  useEffect(() => {
    setResult(null); setValidation(null); setRobust(null); setApproval(null); setComparison(null);
    if (!id) return;
    let active = true;
    void Promise.all([regimes.evaluate(id), regimes.list("validation"), regimes.list("robustness")]).then(([report, validations, robustness]) => {
      if (active) { setResult(report); setLabels(report.labels); setValidation(validations.find(row => row.experiment_id === id) ?? null); setRobust(robustness.find(row => row.experiment_id === id) ?? null); }
    }).catch(error => { if (active) task.setError(String(error)); });
    return () => { active = false; };
  }, [id]);

  function runValidation() {
    void task.run<Artifact>(async () => regimes.validate(id, { mode: validationMode, split_plan_id: validationPlan || null, prepared_datasets: JSON.parse(foldInputs) }), setValidation);
  }
  const mappingNames = data?.profiles.map(row => `${row.state_id} · ${result?.labels[String(row.state_id)] ?? "State"}`) ?? [];
  const featureNames = selected?.config?.inputs.features ?? [];
  const profiles = data?.profiles ?? [];

  return <div className="page regime-workspace"><PageHeader title="D10 · Regime Evaluation & Interpretation" lead="Inspect statistical fit, state behavior, persistence, and chronological generalization. Approve a model together with its exact preprocessing and label mapping." /><Navigation step="evaluation" /><Status task={task} />
    <div className="regime-form"><ArtifactSelect label="Trained candidate" value={id} items={experiments} onChange={setId} /><Field label="Analysis partition"><select value={partition} onChange={e => setPartition(e.target.value)}><option value="train">Train · in sample</option><option value="val">Validation · out of sample</option><option value="test">Test · out of sample</option></select></Field></div>
    {!id && <Empty>No model is selected. Fit an experiment in D9, then choose its saved candidate here.</Empty>}
    {id && !result && !task.error && <div role="status">Loading computed evaluation…</div>}
    <Section id="diagnostics" title="D10.1 · Model Diagnostics">{result ? <><Metrics values={result.diagnostics} />{data && <><Metrics values={{ observations: data.observations, log_likelihood_per_observation: data.log_likelihood_per_observation, ...data.separation, ...data.predictive }} /><Details value={data.predictive} title="Training target evaluation" /></>}<Details value={result.diagnostics} title="Model-specific fit diagnostics" />{Array.isArray(result.diagnostics.log_likelihood_history) && <Chart title="Training convergence" data={[{ type: "scatter", mode: "lines", y: result.diagnostics.log_likelihood_history as number[], line: { color: theme.gold } }]} />}</> : <Empty>Model diagnostics appear after selecting a trained candidate.</Empty>}</Section>
    <Section id="analysis" title="D10.2 · State / Cluster Analysis">{data && result ? <><Table caption={`${partition} state profiles`} rows={profiles} keys={["state_id", "count", "occupancy", "mean_return", "median_return", "return_volatility", "skewness", "kurtosis", "downside_deviation", "maximum_drawdown", "adx_mean", "rsi_mean", "mean_duration"]} />
      <p className="regime-note">{data.occupancy_note}</p><div className="regime-charts"><Chart title="State occupancy · assigned observations" data={[{ type: "bar", x: mappingNames, y: profiles.map(row => row.occupancy), marker: { color: theme.series } }]} /><Chart title="State return & volatility profiles" data={[{ type: "bar", name: "Mean return", x: mappingNames, y: profiles.map(row => row.mean_return) }, { type: "bar", name: "Return volatility", x: mappingNames, y: profiles.map(row => row.return_volatility) }]} /></div>
      <Chart title="Feature means by state · selected model units" data={[{ type: "heatmap", x: featureNames, y: mappingNames, z: profiles.map(row => featureNames.map(feature => row.feature_means[feature] ?? null)), colorscale: [[0, theme.heatmapLow], [1, theme.heatmapHigh]] }]} />
      {result.charts.price && <StateChart title="Historical price colored by raw state · all partitions" times={result.charts.timestamps} values={result.charts.price} states={result.charts.state_ids} labels={result.labels} />}
      {result.charts.returns && <Chart title="Return distributions by state · all partitions" data={profiles.map((row, index) => ({ type: "histogram", name: `${row.state_id} · ${result.labels[String(row.state_id)] ?? "State"}`, x: result.charts.returns!.filter((_, i) => result.charts.state_ids[i] === row.state_id), opacity: 0.6, marker: { color: theme.series[index % theme.series.length] ?? theme.gold } }))} />}
      <p className="regime-note">{result.economic_note} {result.charts.sampling}</p>{result.structural_note && <p className="regime-note">{result.structural_note}</p>}
      {result.charts.volatility ? <Chart title="Upstream volatility distributions by state" data={profiles.map((row, index) => ({ type: "histogram", name: `State ${row.state_id}`, x: result.charts.volatility!.filter((_, i) => result.charts.state_ids[i] === row.state_id), opacity: 0.6, marker: { color: theme.series[index % theme.series.length] ?? theme.gold } }))} /> : <p className="regime-note">Select an existing volatility column in D9’s economic inputs to view volatility distributions.</p>}
    </> : <Empty>{id && result ? "This split plan has no observations for the selected partition." : "State analysis uses actual assignments and original-unit market observations."}</Empty>}</Section>
    <Section id="labeling" title="D10.3 · Regime Interpretation & Labeling"><div className="regime-form"><Field label="Labeling method"><select value={labelMode} onChange={e => setLabelMode(e.target.value)}><option value="automatic">Automatic statistical labels</option><option value="template">Template-based labels</option><option value="manual">Manual / custom labels</option></select></Field><Field label="Statistical template"><select value={template} onChange={e => setTemplate(e.target.value)}>{["trend", "volatility", "trend_volatility", "stress"].map(value => <option key={value}>{value}</option>)}</select></Field></div>
      {result && <div className="regime-form">{Object.entries(labels).map(([key, value]) => <Field key={key} label={`Original state ID ${key}`}><input value={value} disabled={labelMode !== "manual"} onChange={e => setLabels(current => ({ ...current, [key]: e.target.value }))} /></Field>)}</div>}
      <p className="regime-note">Automatic labels use training-state return statistics. Original state IDs are preserved. Approval saves an immutable copy of this mapping.</p>
      <div className="regime-actions"><button disabled={!result || task.busy} onClick={() => void task.run<Artifact>(() => regimes.label(id, { mode: labelMode, template, labels }), async artifact => { setLabels(artifact.labels ?? {}); setResult(await regimes.evaluate(id)); })}>Save label mapping</button></div>
    </Section>
    <Section id="transitions" title="D10.4 · Transition & Persistence Analysis">{data ? <><Metrics values={{ switching_frequency: data.transitions.switching_frequency }} /><div className="regime-charts"><Matrix title="Empirical transition probabilities" values={data.transitions.probabilities} labels={mappingNames} />{data.transitions.fitted_probabilities && <Matrix title="Fitted HMM transition probabilities" values={data.transitions.fitted_probabilities} labels={mappingNames} />}<Chart title="Expected and observed duration · bars" data={[{ type: "bar", name: "Expected", x: mappingNames, y: data.transitions.expected_duration }, { type: "bar", name: "Empirical", x: mappingNames, y: data.transitions.empirical_duration }]} /><Chart title="Observed duration distribution · censored edge runs included" data={[{ type: "histogram", x: data.transitions.durations.map(row => row.bars), marker: { color: theme.gold } }]} /></div><Table caption="State entry / exit timeline" rows={data.transitions.events} /></> : <Empty>Transition statistics require a model and a nonempty partition.</Empty>}</Section>
    <Section id="stability" title="D10.5 · Stability & Robustness"><Field label="Independent perturbations (JSON)" hint="Change seeds, counts, feature_sets, configurations, training_fractions, or initializations. Shorter windows require compatible C7/C8 fit spans."><textarea rows={5} value={robustText} onChange={e => setRobustText(e.target.value)} /></Field><div className="regime-actions"><button disabled={!id || task.busy} onClick={() => void task.run<Artifact>(async () => regimes.robustness(id, JSON.parse(robustText)), setRobust)}>Run robustness refits</button></div>{robust?.rows && <Table caption="Robustness results" rows={robust.rows} />}<p className="regime-note">State numbering is aligned using maximum-overlap assignment. Adjusted Rand Index is invariant to label permutations. Unsupported variants show their refusal reason.</p></Section>
    <Section id="validation" title="D10.6 · Out-of-Sample & Walk-Forward Validation"><div className="regime-form"><Field label="Validation mode"><select value={validationMode} onChange={e => setValidationMode(e.target.value)}><option value="chronological">Saved train / validation / test split</option><option value="expanding">Expanding-window refitting</option><option value="rolling">Rolling-window refitting</option></select></Field>{validationMode !== "chronological" && <Field label="Existing C6 walk-forward plan"><select value={validationPlan} onChange={e => setValidationPlan(e.target.value)}><option value="">Select saved plan</option>{options?.plans.map(row => <option key={row.dataset_id} value={row.dataset_id}>{row.name}</option>)}</select></Field>}</div>
      {validationMode !== "chronological" && <Field label="Prepared dataset per fold (JSON)" hint={'Example: {"1": "C7_DATASET_ID", "2": "C7_DATASET_ID"}. Each scaler/PCA fit must be inside its fold’s training window.'}><textarea rows={3} value={foldInputs} onChange={e => setFoldInputs(e.target.value)} /></Field>}
      <div className="regime-actions"><button disabled={!id || task.busy} className="primary" onClick={runValidation}>Run chronological validation</button></div><Status task={task} />
      {validation && <><Metrics values={{ validation_id: validation.id, test_observations: validation.test?.observations, ...validation.test?.predictive }} /><Details title="Saved validation evidence" value={validation} />{validation.folds?.length ? <Table caption="Walk-forward folds" rows={validation.folds} keys={["fold", "train_end", "test_start", "test_end", "observations", "state_distribution_drift"]} /> : null}</>}
      {result && <Metrics values={Object.fromEntries(Object.entries(result.state_distribution_drift).map(([key, value]) => [`${key}_occupancy_drift`, value]))} />}
    </Section>
    <Section id="comparison" title="D10.7 · Model Comparison & Final Selection"><div className="regime-chips">{experiments.map(item => <label key={item.id}><input type="checkbox" checked={compareIds.includes(item.id)} onChange={e => setCompareIds(current => e.target.checked ? [...current, item.id] : current.filter(value => value !== item.id))} />{item.name} · {item.model} · {item.id.slice(-6)}</label>)}</div><div className="regime-actions"><button disabled={!compareIds.length || task.busy} onClick={() => void task.run(() => regimes.compare(compareIds), setComparison)}>Compare selected candidates</button></div>
      {comparison && <><Table caption="Candidate comparison" rows={comparison.rows} /><p className="regime-note">{comparison.note} {!comparison.comparable && "Input or target contracts differ; these candidates cannot be ranked as one contest."}</p></>}
      <Field label="Approval rationale"><textarea rows={3} value={rationale} onChange={e => setRationale(e.target.value)} placeholder="Explain why the evaluated state behavior and validation evidence support deployment." /></Field>
      <div className="regime-actions"><button className="primary" disabled={!validation || rationale.trim().length < 5 || task.busy} onClick={() => void task.run<Artifact>(() => regimes.approve(id, { validation_id: validation!.id, rationale }), setApproval)}>Approve validated model for D11</button></div>
      {approval && <div className="regime-status">Approved model version {approval.version}. <Link to={`/regime/inference?approval=${approval.id}`}>Deploy in D11 →</Link></div>}
      {!validation && <p className="regime-note">Complete chronological validation of this candidate before approving it.</p>}
    </Section>
  </div>;
}
