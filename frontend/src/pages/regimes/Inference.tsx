import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { regimes } from "@/api/regimes";
import type { Artifact, History, Options } from "@/api/regimes";
import { PageHeader } from "@/components/PageHeader";
import { chartTheme, useTheme } from "@/theme";
import { ArtifactSelect, Details, Empty, Field, Metrics, Navigation, Section, Status, Table, useTask } from "./shared";
import { Chart, Matrix, ProbabilityChart, StateChart } from "./charts";

export function Inference() {
  useTheme();
  const theme = chartTheme();
  const [query] = useSearchParams();
  const [approvals, setApprovals] = useState<Artifact[]>([]);
  const [id, setId] = useState(query.get("approval") ?? "");
  const [options, setOptions] = useState<Options | null>(null);
  const [dataset, setDataset] = useState("");
  const [mode, setMode] = useState("prepared");
  const [asOf, setAsOf] = useState("");
  const [decoding, setDecoding] = useState("filter");
  const [lag, setLag] = useState(5);
  const [minimum, setMinimum] = useState(3);
  const [threshold, setThreshold] = useState(0.55);
  const [hysteresis, setHysteresis] = useState(0.1);
  const [confidence, setConfidence] = useState(0.5);
  const [drift, setDrift] = useState(0.2);
  const [cusum, setCusum] = useState("");
  const [result, setResult] = useState<Artifact | null>(null);
  const [history, setHistory] = useState<History | null>(null);
  const [runs, setRuns] = useState<Artifact[]>([]);
  const task = useTask();
  const approval = approvals.find(item => item.id === id);
  const model = options?.models.find(item => item.id === approval?.model);
  const snapshot = result?.snapshot;
  const hasProbabilities = model?.probabilities;

  useEffect(() => {
    let active = true;
    void Promise.all([regimes.list("approval"), regimes.options(), regimes.list("inference")]).then(([models, opts, inferences]) => { if (active) { setApprovals(models); setOptions(opts); setRuns(inferences); } }).catch(error => { if (active) task.setError(String(error)); });
    return () => { active = false; };
  }, []);
  async function opened(value: Artifact) {
    const [savedHistory, savedRuns] = await Promise.all([regimes.history(value.id), regimes.list("inference")]);
    const saved = value.request;
    setId(String(saved?.approval_id ?? value.approval_id ?? ""));
    setDataset(String(saved?.dataset_id ?? ""));
    setMode(String(saved?.input_mode ?? "prepared"));
    setAsOf(String(saved?.as_of ?? ""));
    setDecoding(String(saved?.decoding ?? "filter"));
    setLag(Number(saved?.lag ?? 5));
    setMinimum(Number(saved?.minimum_persistence ?? 3));
    setThreshold(Number(saved?.probability_threshold ?? 0.55));
    setHysteresis(Number(saved?.hysteresis ?? 0.1));
    setConfidence(Number(saved?.low_confidence ?? 0.5));
    setDrift(Number(saved?.drift_threshold ?? 0.2));
    setCusum(saved?.cusum_threshold == null ? "" : String(saved.cusum_threshold));
    setResult(value); setHistory(savedHistory); setRuns(savedRuns);
  }
  const rows = history?.rows ?? [];
  const times = rows.map(row => String(row.timestamp));
  const states = rows.map(row => Number(row.confirmed_state_id));
  const probabilityKeys = Object.keys(rows[0] ?? {}).filter(key => key.startsWith("probability_")).sort((a, b) => Number(a.split("_")[1]) - Number(b.split("_")[1]));
  const labels = result?.labels ?? {};
  const transition = result?.transitions;
  const payload = { approval_id: id, dataset_id: dataset, input_mode: mode, as_of: asOf || null, decoding, lag,
    minimum_persistence: minimum, probability_threshold: threshold, hysteresis, low_confidence: confidence,
    drift_threshold: drift, cusum_threshold: cusum ? Number(cusum) : null };

  return <div className="page regime-workspace"><PageHeader title="D11 · Regime Inference & Current Market Regime" lead="Apply an approved model and its fitted preprocessing to completed market observations. Filtering is causal; smoothing and Viterbi are retrospective research tools." /><Navigation step="inference" /><Status task={task} />
    <Section id="deployment" title="D11.1 · Model Deployment & Inference Configuration"><div className="regime-form"><ArtifactSelect label="Approved model version" value={id} items={approvals} onChange={value => { setId(value); setResult(null); setHistory(null); setDecoding("filter"); }} /><Field label="Market observations"><select value={dataset} onChange={e => setDataset(e.target.value)}><option value="">Select prepared or upstream feature data</option>{options?.datasets.filter(item => !approval?.contract || item.symbol === approval.contract.asset && item.timeframe === approval.contract.timeframe).map(row => <option key={row.dataset_id} value={row.dataset_id}>{row.name} · {row.symbol} · {row.timeframe}</option>)}</select></Field><Field label="Preprocessing mode"><select value={mode} onChange={e => setMode(e.target.value)}><option value="prepared">Verify already-prepared artifact identity</option><option value="replay">Replay approved C7 / C8 parameters</option></select></Field><Field label="Historical prediction origin (optional)" hint="ISO timestamp with timezone; leave empty for now. Example: 2026-01-01T12:00:00Z."><input value={asOf} onChange={e => setAsOf(e.target.value)} placeholder="Latest completed observation" /></Field></div>
      {approval ? <><Metrics values={{ asset: approval.contract?.asset, timeframe: approval.contract?.timeframe, model: approval.model, version: approval.version, training_end: approval.contract?.train_end }} /><Details title="Pinned features, preprocessing and approval rationale" value={{ contract: approval.contract, rationale: approval.rationale, labels: approval.labels }} /></> : <Empty>Approve a validated candidate in D10 to make it available for deployment.</Empty>}
    </Section>
    <Section id="inference" title="D11.2 · Regime Inference & State Probabilities"><div className="regime-actions"><button className="primary" disabled={!id || !dataset || task.busy} onClick={() => void task.run<Artifact>(() => regimes.infer(payload), opened)}>{task.busy ? "Inferring…" : "Run approved model"}</button></div><Status task={task} />
      {result && <><p className="regime-note">Predictions available for {result.observation_count} completed observations.</p><p className="regime-note">{result.monitoring?.performance.available ? "Realized-target scoring is available." : `Realized-target scoring is unavailable. ${result.monitoring?.performance.reason ?? "Realized targets are required."}`}</p></>}
      {snapshot ? <><Metrics values={{ raw_state_id: snapshot.raw_state_id, confirmed_state_id: snapshot.state_id, confidence: snapshot.confidence, observation_time: snapshot.timestamp, available_at: snapshot.available_at }} />{snapshot.probabilities && <Chart title="Latest state / class probabilities" data={[{ type: "bar", x: snapshot.probabilities.map((_, k) => `${k} · ${labels[String(k)] ?? "State"}`), y: snapshot.probabilities, marker: { color: theme.series } }]} />}{snapshot.distances && <Chart title="Latest distances to cluster centers" data={[{ type: "bar", x: snapshot.distances.map((_, k) => `Cluster ${k}`), y: snapshot.distances, marker: { color: theme.gold } }]} />}</> : <Empty>Inference results appear here after a successful run. Deterministic baselines, K-Means, and change detectors do not receive invented probabilities.</Empty>}
    </Section>
    <Section id="filtering" title="D11.3 · Filtering & Smoothing"><div className="regime-form"><Field label="Decoding method"><select value={decoding} onChange={e => setDecoding(e.target.value)}><option value="filter">Causal filtering / native prediction</option>{model?.decoding && <><option value="smooth">Forward-backward smoothing · retrospective</option><option value="fixed_lag">Fixed-lag smoothing · delayed retrospective estimate</option><option value="viterbi">Viterbi path · retrospective</option></>}</select></Field>{decoding === "fixed_lag" && <Field label="Future lag (bars)"><input type="number" min={1} max={100} value={lag} onChange={e => setLag(Number(e.target.value))} /></Field>}</div>{decoding !== "filter" && <div className="regime-status">Retrospective mode uses observations after the timestamp being labeled. This output is not a historical live signal.</div>}<p className="regime-note">{result?.decoding_note ?? "Real-time inference defaults to causal filtering. Supervised forecasting is displayed as a forecast at its origin."}</p></Section>
    <Section id="detection" title="D11.4 · Regime Transition Detection"><div className="regime-form"><Field label="Minimum persistence (bars)"><input type="number" min={1} value={minimum} onChange={e => setMinimum(Number(e.target.value))} /></Field>{hasProbabilities && <><Field label="Confirmation probability threshold"><input type="number" min={0} max={1} step={0.05} value={threshold} onChange={e => setThreshold(Number(e.target.value))} /></Field><Field label="Probability hysteresis"><input type="number" min={0} max={1} step={0.05} value={hysteresis} onChange={e => setHysteresis(Number(e.target.value))} /></Field></>}</div>{result?.events && <Table caption="Confirmed transition events" rows={result.events} />}<p className="regime-note">A transition is recorded on its confirmation bar. Detection delay measures confirmation lag; it is not a claim about the time of an unobserved economic regime change.</p></Section>
    <Section id="dashboard" title="D11.5 · Current Market Regime Dashboard">{snapshot ? <>{snapshot.retrospective && <div className="regime-status">Retrospective research result · {String(result?.request?.decoding ?? "saved decoder unavailable")}</div>}{snapshot.objective === "future" && <div className="regime-status">Future-state forecast · {snapshot.horizon} bars after the observation origin</div>}<div className="regime-current">{snapshot.regime}</div><Metrics values={{ state_id: snapshot.state_id, trend: snapshot.trend, volatility: snapshot.volatility, duration_bars: snapshot.duration_bars, previous_state: snapshot.previous_state, confidence: snapshot.confidence, model: snapshot.model, asset: snapshot.asset, timeframe: snapshot.timeframe, latest_observation: snapshot.timestamp, inference_timestamp: snapshot.inferred_at }} /><Details title="Last transition and model version" value={{ transition: snapshot.last_transition, version: snapshot.version }} /></> : <Empty>The current dashboard requires an inference result from an approved model.</Empty>}</Section>
    <Section id="history" title="D11.6 · Historical Regime Visualization">{history && rows.length ? <>{rows[0]?.price !== undefined && <StateChart title="Price colored by confirmed regime" times={times} values={rows.map(row => row.price === null ? null : Number(row.price))} states={states} labels={labels} />}<StateChart title="Historical confirmed regime timeline" times={times} values={states} states={states} labels={labels} />{probabilityKeys.length > 0 && <ProbabilityChart times={times} values={rows.map(row => probabilityKeys.map(key => Number(row[key])))} labels={labels} />}
      {transition && !transition.unavailable && <div className="regime-charts"><Matrix title="Historical transition matrix" values={transition.probabilities} labels={transition.probabilities.map((_, k) => `${k} · ${labels[String(k)] ?? "State"}`)} /><Chart title="Regime duration distribution" data={[{ type: "histogram", x: transition.durations.map(row => row.bars), marker: { color: theme.gold } }]} /><Chart title="Regime counts · assigned visible history" data={[{ type: "bar", x: [...new Set(states.filter(state => state >= 0))].map(String), y: [...new Set(states.filter(state => state >= 0))].map(state => states.filter(value => value === state).length), marker: { color: theme.series } }]} /></div>}
      <p className="regime-note">Showing rows {history.offset + 1}–{history.offset + rows.length} of {history.total}. Exports include all observations.</p><div className="regime-actions"><button disabled={history.offset === 0 || task.busy} onClick={() => void task.run(() => regimes.history(result!.id, Math.max(0, history.offset - 2500)), setHistory)}>Previous history</button><button disabled={history.offset + rows.length >= history.total || task.busy} onClick={() => void task.run(() => regimes.history(result!.id, history.offset + rows.length), setHistory)}>Next history</button></div></> : <Empty>Historical charts use saved inference assignments and probabilities.</Empty>}</Section>
    <Section id="monitoring" title="D11.7 · Monitoring & Alerts"><div className="regime-form">{hasProbabilities && <Field label="Low-confidence threshold"><input type="number" min={0} max={1} step={0.05} value={confidence} onChange={e => setConfidence(Number(e.target.value))} /></Field>}<Field label="Distribution drift threshold"><input type="number" min={0.01} max={1} step={0.05} value={drift} onChange={e => setDrift(Number(e.target.value))} /></Field><Field label="CUSUM alert threshold (optional)"><input type="number" min={0.01} value={cusum} onChange={e => setCusum(e.target.value)} /></Field></div>
      {result?.monitoring ? <>{result.monitoring.warnings.length ? result.monitoring.warnings.map((warning, index) => <div className="regime-status" key={index}><b>{warning.kind.replaceAll("_", " ")}</b> · {warning.message}</div>) : <p className="regime-note">No configured monitoring threshold was exceeded for this run.</p>}<Table caption="Feature distribution drift" rows={result.monitoring.feature_drift} /><Metrics values={{ state_occupancy_drift: result.monitoring.state_occupancy_drift }} /><Table caption="CUSUM alerts" rows={result.monitoring.cusum_events} /><Details title="Performance monitoring availability and evidence" value={result.monitoring.performance} /></> : <Empty>Monitoring is computed against the model’s training reference on each inference run.</Empty>}</Section>
    <Section id="output" title="D11.8 · Output & Integration"><ArtifactSelect label="Saved inference history" value={result?.id ?? ""} items={runs} onChange={value => { const record = runs.find(item => item.id === value); if (record) void task.run(() => regimes.get(record.id), opened); }} />
      {result ? <><div className="regime-actions"><a href={regimes.exportUrl(result.id, "json")} target="_blank" rel="noreferrer">Structured JSON + metadata</a><a href={regimes.exportUrl(result.id, "csv")}>Historical assignments CSV</a><a href={regimes.bundleUrl(result.id)}>Complete deployment bundle</a></div><Details title="Reusable inference API request" value={{ method: "POST", endpoint: "/api/regime/inference", body: result.request ?? { unavailable: "Saved request metadata is unavailable." } }} /><Details title="Model output and explicit state-label mapping" value={{ snapshot, labels: result.labels, approval_id: result.approval_id, artifact_id: result.id }} /><Table caption="Historical state assignments" rows={rows.slice(0, 500)} /></> : <Empty>Every successful run saves its request, model version, label mapping, monitoring results, and complete assignments.</Empty>}
    </Section>
  </div>;
}
