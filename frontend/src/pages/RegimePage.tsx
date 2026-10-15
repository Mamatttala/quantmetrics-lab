/**
 * Module D — Regime & Trend Detection.
 *
 * The page is a config panel over a results dashboard. Three things about it are worth
 * stating up front, because each is a place the obvious implementation is the wrong one.
 *
 * **The run is a job, not a request.** A fit over a long series takes seconds to minutes,
 * so `POST /api/regime/detect` returns a handle and this page polls. The button's
 * "Running…" state is therefore genuinely unbounded rather than a spinner with a timeout
 * behind it, and unmounting the page cancels the *waiting*, not the run — the job keeps
 * going server-side and lands in the registry either way. Abandoning a page is not the
 * same as asking for the work to stop, and pretending otherwise would lose a five-minute
 * fit because someone clicked a sidebar link.
 *
 * **The gauge reports the committed state's own probability.** This is the number the
 * specification's "below 55%, treat it as a gray zone" line is about, and it is *not* the
 * largest probability in the row. Anti-thrashing means the label can lag the argmax — the
 * model may be most confident of Ranging while the committed state is still Trending Up —
 * and reporting the maximum there would show a confident gauge for a state the label does
 * not name. The backend computes it; this page only renders it.
 *
 * **Nothing here is a hard label.** The frame carries four soft probabilities that sum to
 * one, and the chart draws them stacked. A dashboard that showed only the discrete state
 * would hide the difference between a bar where crisis held 0.92 and one where it scraped
 * past the threshold, and that difference is most of what the module is for.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import type { Data, Layout } from "plotly.js";

import { ApiError, api } from "@/api/client";
import { notifyDatasetsChanged, subscribeToDatasets } from "@/api/datasetsChanged";
import { failureMessage, isAbort, waitForJob } from "@/api/jobs";
import type {
  DatasetSummary,
  HeadTailPreview,
  JobRecord,
  RegimeFeatureId,
  RegimeModel,
  RegimeOptions,
  RegimeReport,
  RegimeStateName,
} from "@/api/types";
import { HeadTailTable } from "@/components/HeadTailTable";
import { InfoTip } from "@/components/InfoTip";
import { PageHeader } from "@/components/PageHeader";
import { chartTheme, useTheme } from "@/theme";

const Plot = createPlotlyComponent(Plotly);

/* ------------------------------------------------------------------ palette

   The five state colours, in one place. Gold for an up-trend and violet for a down-trend
   rather than the conventional green/red: green/red is already spoken for by the candle
   chart's up/down bodies, and a state band that reused those would read as "price rose"
   instead of "the model called a trend". Crisis is the exception — it takes the danger red,
   because the one band a user must not misread is that one.

   `transition` is deliberately the dimmest of the five. It is the absence of a call, and a
   band that competed for attention with the states it sits between would misrepresent it. */

const STATE_COLOUR: Record<RegimeStateName, string> = {
  trending_up: "#d4a843",
  trending_down: "#8b5cf6",
  ranging: "#888888",
  crisis: "#ef4444",
  transition: "#2a2a2a",
};

const STATE_LABEL: Record<RegimeStateName, string> = {
  trending_up: "Trending Up",
  trending_down: "Trending Down",
  ranging: "Ranging",
  crisis: "Crisis",
  transition: "Transition",
};

/** The stack order, bottom band first. Crisis rides the top of the stack so a spike is
 *  visible against the axis rather than buried under three other bands. */
const STACK: readonly { key: string; state: RegimeStateName }[] = [
  { key: "prob_ranging", state: "ranging" },
  { key: "prob_trending_up", state: "trending_up" },
  { key: "prob_trending_down", state: "trending_down" },
  { key: "prob_crisis", state: "crisis" },
];

/** The confidence below which the current reading is a gray zone rather than a call.
 *  Mirrors `transition_confidence` on the backend, which owns the threshold. */
const GRAY_ZONE = 0.55;

const MODEL_OPTIONS: readonly { value: RegimeModel; label: string; detail: string }[] = [
  {
    value: "hmm",
    label: "Gaussian HMM",
    detail:
      "A hidden Markov model over log return and realized volatility. States are latent and persistent by construction, which is what a regime is supposed to be.",
  },
  {
    value: "gmm",
    label: "Gaussian mixture",
    detail:
      "A mixture over the same observations plus volume change. No transition structure, so it can relabel consecutive bars freely — read its switch count as a lower bound.",
  },
  {
    value: "hybrid",
    label: "Hybrid HMM + gradient boosting",
    detail:
      "A gradient-boosted classifier trained to reproduce the HMM's soft labels from faster indicators (ADX, RSI). Needs those columns carried in from C5.",
  },
  {
    value: "rules",
    label: "Rules + crisis override",
    detail:
      "A disclosed heuristic on returns and volatility, with the same volatility circuit-breaker. No fitted parameters — the one path that works on a close column alone.",
  },
];

interface Notice {
  message: string;
  kind: "ok" | "error";
}

export function RegimePage() {
  const { datasetId } = useParams<{ datasetId?: string }>();
  const navigate = useNavigate();

  const [candidates, setCandidates] = useState<DatasetSummary[]>([]);
  const [options, setOptions] = useState<RegimeOptions | null>(null);
  const [optionsStatus, setOptionsStatus] = useState<"idle" | "loading" | "ready" | "error">(
    "idle",
  );
  const [optionsError, setOptionsError] = useState<string | null>(null);

  // ---- configuration
  const [model, setModel] = useState<RegimeModel>("hmm");
  const [nStates, setNStates] = useState(4);
  const [features, setFeatures] = useState<ReadonlySet<RegimeFeatureId>>(new Set());
  const [minDuration, setMinDuration] = useState(5);
  const [switchingCost, setSwitchingCost] = useState("0.005");
  const [crisisThreshold, setCrisisThreshold] = useState("0.35");
  const [outName, setOutName] = useState("");
  const [seededFor, setSeededFor] = useState<string | null>(null);

  // ---- run state
  const [running, setRunning] = useState(false);
  const [polls, setPolls] = useState(0);
  const [result, setResult] = useState<JobRecord | null>(null);
  const [notice, setNotice] = useState<Notice | null>(null);
  const abort = useRef<AbortController | null>(null);

  // ------------------------------------------------------------ dataset list

  const loadCandidates = useCallback(async () => {
    try {
      const rows = await api.listDatasets();
      setCandidates(rows.filter((row) => row.row_count > 0));
    } catch {
      setCandidates([]);
    }
  }, []);

  useEffect(() => {
    void loadCandidates();
  }, [loadCandidates]);

  useEffect(() => subscribeToDatasets(() => void loadCandidates()), [loadCandidates]);

  // ---------------------------------------------------------------- options

  useEffect(() => {
    if (datasetId === undefined) {
      setOptions(null);
      setOptionsStatus("idle");
      return;
    }
    let cancelled = false;
    setOptionsStatus("loading");
    setOptionsError(null);

    api
      .regimeOptions(datasetId)
      .then((payload) => {
        if (cancelled) return;
        setOptions(payload);
        setOptionsStatus("ready");
      })
      .catch((cause: unknown) => {
        if (cancelled) return;
        setOptionsError(cause instanceof ApiError ? cause.message : String(cause));
        setOptionsStatus("error");
      });

    return () => {
      cancelled = true;
    };
  }, [datasetId]);

  // The panel seeds from the server's defaults once per dataset, and then belongs to the
  // user. Re-seeding on every options change would silently discard a slider the user had
  // already moved while a later request was in flight.
  useEffect(() => {
    if (options === null || seededFor === options.dataset_id) return;
    setSeededFor(options.dataset_id);
    setNStates(options.defaults.n_states);
    setMinDuration(options.defaults.min_state_duration);
    setSwitchingCost(String(options.defaults.switching_cost_penalty));
    setCrisisThreshold(String(options.defaults.crisis_threshold));
    setModel(options.models.find((entry) => entry.available)?.id ?? "rules");
    setFeatures(new Set(options.available_features.filter((id) => id !== "adx" && id !== "rsi")));
    setResult(null);
    setNotice(null);
  }, [options, seededFor]);

  // Cancel the *wait*, never the job. See the module docstring.
  useEffect(() => () => abort.current?.abort(), []);

  const toggleFeature = useCallback((id: RegimeFeatureId) => {
    setFeatures((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const modelOption = options?.models.find((entry) => entry.id === model) ?? null;

  const problems = useMemo(() => {
    const found: string[] = [];
    if (options === null) return found;
    const blocking = options.warnings.filter((warning) => warning.level === "error");
    for (const warning of blocking) found.push(warning.message);
    if (modelOption !== null && !modelOption.available) {
      found.push(modelOption.reason ?? "That model cannot run on this dataset.");
    }
    if (features.size === 0 && model !== "rules") {
      found.push("Select at least one observation — a model needs something to fit on.");
    }
    const cost = Number(switchingCost);
    if (!Number.isFinite(cost) || cost < 0) {
      found.push("Switching cost penalty must be a number at or above zero.");
    }
    const threshold = Number(crisisThreshold);
    if (!Number.isFinite(threshold) || threshold <= 0 || threshold >= 5) {
      found.push(
        "Crisis threshold must be a positive annualised volatility, e.g. 0.35 for 35%.",
      );
    }
    return found;
  }, [options, modelOption, model, features, switchingCost, crisisThreshold]);

  const canRun = options !== null && problems.length === 0 && !running;

  const run = useCallback(async () => {
    if (options === null) return;
    const controller = new AbortController();
    abort.current = controller;

    setRunning(true);
    setPolls(0);
    setNotice(null);
    setResult(null);

    try {
      const accepted = await api.detectRegime({
        dataset_id: options.dataset_id,
        model_type: model,
        n_states: nStates,
        features: [...features],
        min_state_duration: minDuration,
        switching_cost_penalty: Number(switchingCost),
        crisis_threshold: Number(crisisThreshold),
        min_train_bars: options.defaults.min_train_bars,
        refit_every: options.defaults.refit_every,
        volatility_window: options.defaults.volatility_window,
        name: outName.trim() === "" ? null : outName.trim(),
        write: true,
      });

      const job = await waitForJob(accepted.job_id, {
        signal: controller.signal,
        onPoll: (_record, attempt) => setPolls(attempt + 1),
      });
      setResult(job);
      if (job.status === "succeeded") {
        setNotice({ message: job.message ?? "Detection finished.", kind: "ok" });
        notifyDatasetsChanged();
      } else {
        setNotice({ message: failureMessage(job), kind: "error" });
      }
    } catch (cause: unknown) {
      if (isAbort(cause)) return;
      setNotice({
        message: cause instanceof ApiError ? cause.message : String(cause),
        kind: "error",
      });
    } finally {
      setRunning(false);
      abort.current = null;
    }
  }, [options, model, nStates, features, minDuration, switchingCost, crisisThreshold, outName]);

  const empty = datasetId === undefined;

  return (
    <div className="page">
      <PageHeader
        title="Regime & Trend Detection"
        lead={
          <>
            Label every bar with a market state — trending, ranging, or crisis — as a{" "}
            <strong>soft probability</strong> rather than a hard call, so a downstream
            strategy can size its exposure to how sure the model is. The output is a new{" "}
            <span className="mono">outputs</span> dataset; the source is never modified.
          </>
        }
      />

      <div className="row">
        <div className="field field--tight">
          <label htmlFor="regime-dataset">Dataset</label>
          <select
            id="regime-dataset"
            value={datasetId ?? ""}
            onChange={(event) => {
              const next = event.target.value;
              navigate(next === "" ? "/regime" : `/regime/${next}`);
            }}
          >
            <option value="">— choose a dataset —</option>
            {candidates.map((row) => (
              <option key={row.dataset_id} value={row.dataset_id}>
                {row.name} · {row.category} · {row.row_count.toLocaleString()} rows
                {row.timeframe === null ? "" : ` · ${row.timeframe}`}
              </option>
            ))}
          </select>
        </div>
      </div>

      {empty && (
        <div className="panel muted">
          No dataset selected. Regimes are detected over stored bars — fetch one in A1 or
          upload a file in Data Management. The result is saved as a new{" "}
          <span className="mono">outputs</span> child, leaving the input untouched.
        </div>
      )}

      {optionsStatus === "loading" && <div className="panel muted">Reading the dataset…</div>}

      {optionsStatus === "error" && (
        <div className="result__bar result__bar--error">
          <span>{optionsError}</span>
        </div>
      )}

      {notice !== null && (
        <div
          className={
            notice.kind === "ok"
              ? "result__bar result__bar--ok"
              : "result__bar result__bar--error"
          }
        >
          <span>{notice.message}</span>
        </div>
      )}

      {optionsStatus === "ready" && options !== null && (
        <>
          {/* ======================================================== config */}
          <div className="section">
            <div className="section__head">
              <h2 className="section__title">Source</h2>
              <span className="mono muted" style={{ fontSize: 11 }}>
                {options.dataset_id} · {options.rows.toLocaleString()} rows ·{" "}
                {options.timeframe === null ? "no declared timeframe" : options.timeframe} ·{" "}
                {options.asset_class}
              </span>
            </div>

            <div className="panel">
              <div className="regime__annualisation">
                <span className="report__key">Annualisation</span>
                <span className={options.annualisation.available ? "" : "regime__bad"}>
                  {options.annualisation.available
                    ? `×${options.annualisation.periods_per_year?.toLocaleString()} — ${options.annualisation.reason}`
                    : options.annualisation.reason}
                </span>
                <InfoTip
                  label="What annualisation means for the crisis threshold"
                  formula="annualised volatility = per-bar volatility × √(bars per year)"
                  detail="The crisis threshold is stated per year, so a per-bar volatility has to be scaled to compare against it. The factor comes from the dataset's own timeframe and asset class — 8760 bars a year for hourly crypto, 252 days for equities — and is refused when the dataset declares no timeframe, because a wrong factor is a silent multiplicative error in the number that triggers a crisis label."
                />
              </div>

              {options.warnings.map((warning) => (
                <div
                  key={warning.code}
                  className={
                    warning.level === "error"
                      ? "regime__warning regime__warning--error"
                      : "regime__warning"
                  }
                >
                  <span className="badge">{warning.level}</span> {warning.message}
                </div>
              ))}
            </div>
          </div>

          <div className="section">
            <div className="section__head">
              <h2 className="section__title">Model</h2>
            </div>

            <div className="panel">
              <fieldset className="policy" disabled={running}>
                <legend className="policy__legend">Detection method</legend>
                <p className="policy__hint muted">
                  Each of these answers the same question differently. The mixture and the
                  rules are here for comparison, not because they are better — a regime that
                  only one method finds is worth knowing about before it is traded.
                </p>

                {MODEL_OPTIONS.map((option) => {
                  const availability = options.models.find(
                    (entry) => entry.id === option.value,
                  );
                  const unavailable = availability !== undefined && !availability.available;
                  const selected = option.value === model;
                  return (
                    <label
                      className={`policy__option${selected ? " policy__option--on" : ""}${
                        unavailable ? " policy__option--off" : ""
                      }`}
                      key={option.value}
                    >
                      <input
                        type="radio"
                        name="regime-model"
                        value={option.value}
                        checked={selected}
                        disabled={unavailable}
                        onChange={() => setModel(option.value)}
                      />
                      <span className="policy__body">
                        <span className="policy__label">{option.label}</span>
                        <span className="policy__detail muted">{option.detail}</span>
                        {unavailable && (
                          <span className="policy__caution">{availability?.reason}</span>
                        )}
                      </span>
                    </label>
                  );
                })}
              </fieldset>
            </div>
          </div>

          <div className="section">
            <div className="section__head">
              <h2 className="section__title">Configuration</h2>
            </div>

            <div className="panel">
              <div className="regime__grid">
                <div className="field">
                  <label htmlFor="regime-states">
                    States <span className="mono">{nStates}</span>
                  </label>
                  <input
                    id="regime-states"
                    type="range"
                    min={2}
                    max={5}
                    step={1}
                    value={nStates}
                    disabled={running}
                    onChange={(event) => setNStates(Number(event.target.value))}
                  />
                  <p className="field__note">
                    How many regimes the model may use. The four named states are fixed, so a
                    count below four forces two of them to share one statistical regime —
                    which is legitimate for a market that only ranges, and misleading if it
                    does not. Choosing <span className="mono">k</span> is a modelling
                    decision; the diagnostics below report the BIC so the two can be
                    compared rather than guessed.
                  </p>
                </div>

                <div className="field">
                  <label htmlFor="regime-duration">
                    Minimum state duration <span className="mono">{minDuration}</span> bars
                  </label>
                  <input
                    id="regime-duration"
                    type="range"
                    min={1}
                    max={10}
                    step={1}
                    value={minDuration}
                    disabled={running}
                    onChange={(event) => setMinDuration(Number(event.target.value))}
                  />
                  <p className="field__note">
                    The anti-thrashing rule: an incumbent state cannot be replaced until it
                    has been held this long, however confident a challenger looks. It is a
                    pure cost — it delays every genuine turn — and it buys a label series a
                    strategy can act on without paying the switching cost on every flicker.
                    The crisis override deliberately ignores it.
                  </p>
                </div>

                <div className="field">
                  <label htmlFor="regime-cost">Switching cost penalty</label>
                  <input
                    id="regime-cost"
                    type="number"
                    min={0}
                    step={0.001}
                    value={switchingCost}
                    disabled={running}
                    onChange={(event) => setSwitchingCost(event.target.value)}
                  />
                  <p className="field__note">
                    Charged against Net Value-Add for every state change, as a fraction of
                    the position. It does not change the labels — it changes the score those
                    labels are judged by, so a detector that thrashes is penalised rather
                    than flattered. Set it to what a switch actually costs you.
                  </p>
                </div>

                <div className="field">
                  <label htmlFor="regime-crisis">Crisis threshold (annualised)</label>
                  <input
                    id="regime-crisis"
                    type="number"
                    min={0.01}
                    step={0.01}
                    value={crisisThreshold}
                    disabled={running}
                    onChange={(event) => setCrisisThreshold(event.target.value)}
                  />
                  <p className="field__note">
                    A realized volatility above this forces the bar to Crisis with at least
                    90% probability, overriding the minimum-duration rule and the model's own
                    ranking. 0.35 means 35% annualised. The override is a fact about the
                    market rather than a fit, so it raises{" "}
                    <span className="mono">prob_crisis</span> itself — the label and the
                    column never disagree.
                  </p>
                </div>
              </div>

              <fieldset className="policy" disabled={running}>
                <legend className="policy__legend">Observations</legend>
                <p className="policy__hint muted">
                  What the model is allowed to see. Every one of these is causal — computed
                  from the bar and the bars before it, never from the series as a whole.
                </p>
                {options.features.map((feature) => (
                  <label
                    className={`feature__pick${feature.available ? "" : " feature__row--off"}`}
                    key={feature.id}
                  >
                    <input
                      type="checkbox"
                      checked={features.has(feature.id)}
                      disabled={!feature.available || running}
                      onChange={() => toggleFeature(feature.id)}
                    />
                    <span className="feature__label">{feature.label}</span>
                    <span className="badge">{feature.kind}</span>
                    {!feature.available && (
                      <span className="feature__reason">{feature.reason}</span>
                    )}
                  </label>
                ))}
              </fieldset>

              <div className="field">
                <label htmlFor="regime-name">Output dataset name</label>
                <input
                  id="regime-name"
                  type="text"
                  placeholder="(defaults to the source name plus the model)"
                  value={outName}
                  disabled={running}
                  onChange={(event) => setOutName(event.target.value)}
                />
              </div>
            </div>
          </div>

          <div className="section">
            <div className="actions">
              <button
                type="button"
                className="btn"
                onClick={() => void run()}
                disabled={!canRun}
              >
                {running
                  ? `Running… (${polls} poll${polls === 1 ? "" : "s"})`
                  : "Run Detection"}
              </button>
              <button
                type="button"
                className="btn btn--ghost btn--sm"
                onClick={() => abort.current?.abort()}
                disabled={!running}
              >
                Stop waiting
              </button>
            </div>

            {problems.length > 0 && (
              <div className="result__bar result__bar--error">
                <span>the run cannot start yet:</span>
                {problems.map((problem) => (
                  <div key={problem}>· {problem}</div>
                ))}
              </div>
            )}

            <div className="panel muted" style={{ fontSize: 12 }}>
              The fit runs in the background: pressing Run returns a job immediately and the
              page polls it. Stopping the wait does not stop the job — it still lands in the
              registry, and its output is still saved. The result is written as a new{" "}
              <strong>outputs</strong> dataset carrying the source as its parent and the
              whole request in its lineage, so the same labels can be reproduced from the
              record alone.
            </div>
          </div>

          {/* ======================================================= results */}
          {result !== null && <Results job={result} onOpenDataset={navigate} />}
        </>
      )}
    </div>
  );
}

/* ============================================================== results */

function Results({ job, onOpenDataset }: { job: JobRecord; onOpenDataset: (to: string) => void }) {
  const result = job.result;
  if (result === null) {
    return (
      <div className="section">
        <div className="section__head">
          <h2 className="section__title">Result</h2>
        </div>
        <div className="result__bar result__bar--error">
          <span>{failureMessage(job)}</span>
        </div>
      </div>
    );
  }

  const { report } = result;
  const latest = latestLabelled(report, result.preview);

  return (
    <>
      <div className="section">
        <div className="section__head">
          <h2 className="section__title">Current state</h2>
          <span className="mono muted" style={{ fontSize: 11 }}>
            {report.labelled_rows.toLocaleString()} of {report.rows.toLocaleString()} bars
            labelled · {report.refit_count} refit
            {report.refit_count === 1 ? "" : "s"}
          </span>
        </div>

        <div className="regime__gauges">
          <CurrentStateCard latest={latest} />
          <div className="panel regime__counts">
            <div className="report">
              {Object.entries(report.state_counts)
                .sort((a, b) => b[1] - a[1])
                .map(([name, count]) => (
                  <div className="report__row" key={name}>
                    <span className="report__key">
                      {name === "warmup" ? (
                        // Not "warm-up": the report's `warmup_bars` is only the observation
                        // window, and this bucket is that plus the bars the first model
                        // trained on. Naming both "warm-up" is what made the two disagree.
                        "Unlabelled (warm-up + training)"
                      ) : (
                        <>
                          <span
                            className="regime__swatch"
                            style={{
                              background:
                                STATE_COLOUR[name as RegimeStateName] ?? "transparent",
                            }}
                          />
                          {STATE_LABEL[name as RegimeStateName] ?? name}
                        </>
                      )}
                    </span>
                    <span className="report__value mono">
                      {count.toLocaleString()}{" "}
                      <span className="muted">
                        ({((count / report.rows) * 100).toFixed(1)}%)
                      </span>
                    </span>
                  </div>
                ))}
            </div>
          </div>
        </div>
      </div>

      <div className="section">
        <div className="section__head">
          <h2 className="section__title">Regime timeline</h2>
          <span className="muted" style={{ fontSize: 12 }}>
            {report.method}
          </span>
        </div>
        <div className="panel">
          <Timeline report={report} preview={result.preview} />
          <p className="field__note">
            The bands are the model's <strong>soft probabilities</strong>, stacked to fill
            the panel — a wide band means the model was sure, a smeared boundary means it was
            not. The ribbon underneath is the committed label after anti-thrashing, so the
            dark stretches are bars where no state held a mandate. Hover any bar to read all
            four figures at once.
          </p>
        </div>
      </div>

      <div className="section">
        <div className="section__head">
          <h2 className="section__title">Transition matrix</h2>
        </div>
        <div className="panel">
          <TransitionMatrix report={report} />
          <p className="field__note">
            Row-normalised: each row is the distribution of where the <em>next</em> labelled
            bar goes, given the current one. Built from committed labels rather than raw
            probabilities, so it describes the series the output actually carries.
          </p>
        </div>
      </div>

      <div className="section">
        <div className="section__head">
          <h2 className="section__title">Diagnostics</h2>
        </div>
        <div className="panel">
          <Diagnostics report={report} />
        </div>
      </div>

      <div className="section">
        <div className="section__head">
          <h2 className="section__title">Output preview</h2>
          {result.dataset !== null && (
            <button
              type="button"
              className="btn btn--ghost btn--sm"
              onClick={() => onOpenDataset(`/view/${result.dataset?.dataset_id ?? ""}`)}
            >
              Open {result.dataset.name}
            </button>
          )}
        </div>
        <div className="panel">
          <HeadTailTable
            columns={result.preview.columns}
            head={result.preview.head}
            tail={result.preview.tail}
            rowCount={report.rows}
          />
        </div>
      </div>
    </>
  );
}

interface Latest {
  state: RegimeStateName;
  confidence: number;
  time: string;
}

/**
 * The last labelled bar, read from the preview's tail.
 *
 * From the preview rather than a dedicated endpoint because the tail is already on the
 * wire: a second route returning "the current regime" would be a second place for the
 * definition of "current" to live, and the two would eventually disagree about whether
 * warm-up rows count.
 */
function latestLabelled(report: RegimeReport, preview: HeadTailPreview): Latest | null {
  const names = preview.columns.map((column) => column.name);
  if (!names.includes("dominant_state") || !names.includes("confidence")) return null;
  const timeAt = names.indexOf("open_time");

  for (let index = preview.tail.length - 1; index >= 0; index -= 1) {
    const row = preview.tail[index];
    if (row === undefined) continue;
    const state = row["dominant_state"];
    if (typeof state !== "string" || state === "") continue;
    if (!(state in STATE_COLOUR)) continue;
    const raw = row["confidence"];
    const confidence = typeof raw === "number" ? raw : Number(raw);
    if (!Number.isFinite(confidence)) continue;
    return {
      state: state as RegimeStateName,
      confidence,
      time: timeAt >= 0 ? String(row["open_time"] ?? "") : `${report.rows} bars in`,
    };
  }
  return null;
}

function CurrentStateCard({ latest }: { latest: Latest | null }) {
  if (latest === null) {
    return (
      <div className="panel">
        <div className="report__key">Current state</div>
        <div className="muted">
          No labelled rows in this range. The model needs the observation warm-up plus the
          training window before it can call anything — lower the minimum training window,
          or detect over a longer series.
        </div>
      </div>
    );
  }

  const colour = STATE_COLOUR[latest.state];
  const grey = latest.confidence < GRAY_ZONE;
  const percent = latest.confidence * 100;

  return (
    <div className={`panel regime__gauge${grey ? " regime__gauge--grey" : ""}`}>
      <div className="report__key">Current state</div>
      <div className="regime__gauge-state" style={{ color: colour }}>
        {STATE_LABEL[latest.state]}
      </div>
      <div className="regime__gauge-bar" role="presentation">
        <span style={{ width: `${percent.toFixed(1)}%`, background: colour }} />
      </div>
      <div className="mono regime__gauge-value">{percent.toFixed(1)}%</div>
      <div className="field__note">
        Probability of the committed state, at {latest.time}.
      </div>
      {grey && (
        <div className="regime__warning regime__warning--grey">
          <span className="badge">gray zone</span> Below {(GRAY_ZONE * 100).toFixed(0)}% the
          model is not committed to this state. The label is a booking, not a call — treat
          it as transitional and let a position sizing rule decide, rather than reading it
          as a regime change.
        </div>
      )}
    </div>
  );
}

/* --------------------------------------------------------------- charts */

function Timeline({ report, preview }: { report: RegimeReport; preview: HeadTailPreview }) {
  const { theme } = useTheme();
  const rows = useMemo(() => [...preview.head, ...preview.tail], [preview]);

  const { data, layout } = useMemo(() => buildTimeline(report, rows), [report, rows, theme]);

  if (rows.length === 0) {
    return <div className="chart__empty">No rows to draw.</div>;
  }

  return (
    <Plot
      data={data}
      layout={layout}
      config={{ responsive: true, displaylogo: false }}
      style={{ width: "100%", height: "420px" }}
      useResizeHandler
    />
  );
}

/**
 * The stacked area, plus the committed-label ribbon.
 *
 * The ribbon is a one-row heatmap on a second y-axis rather than a fifth area: the
 * committed label is categorical and includes Transition, which has no probability column
 * of its own, so there is nothing for it to be a band *of*. Drawing it as a strip keeps it
 * on the same time axis as the stack it was derived from, which is the whole point of
 * showing it.
 */
function buildTimeline(
  report: RegimeReport,
  rows: Record<string, unknown>[],
): { data: Data[]; layout: Partial<Layout> } {
  const times = rows.map((row) => String(row["open_time"] ?? ""));
  const timesMs = times.map((value) => new Date(value).getTime());

  const data: Data[] = STACK.map(({ key, state }) => ({
    type: "scatter",
    mode: "lines",
    name: STATE_LABEL[state],
    x: timesMs,
    y: rows.map((row) => numeric(row[key])),
    stackgroup: "probabilities",
    line: { width: 0, color: STATE_COLOUR[state] },
    fillcolor: STATE_COLOUR[state],
    hovertemplate: `${STATE_LABEL[state]} %{y:.1%}<extra></extra>`,
  }));

  const order = report.transition_matrix.labels;
  const ribbon = rows.map((row) => {
    const state = row["dominant_state"];
    if (typeof state !== "string" || state === "") return null;
    const index = order.indexOf(state as RegimeStateName);
    return index < 0 ? null : index;
  });

  data.push({
    type: "heatmap",
    name: "Committed label",
    x: timesMs,
    y: ["label"],
    z: [ribbon],
    yaxis: "y2",
    showscale: false,
    zmin: 0,
    zmax: Math.max(order.length - 1, 1),
    colorscale: order.map((state, index) => [
      index / Math.max(order.length - 1, 1),
      STATE_COLOUR[state],
    ]),
    customdata: [ribbon.map((index) => (index === null ? "warm-up" : STATE_LABEL[order[index] ?? "transition"]))],
    hovertemplate: "%{customdata}<extra>Committed label</extra>",
    showlegend: false,
  } as Data);

  const t = chartTheme();
  const layout: Partial<Layout> = {
    paper_bgcolor: t.paper,
    plot_bgcolor: t.plot,
    font: { family: "Inter, system-ui, sans-serif", color: t.font, size: 11 },
    margin: { l: 52, r: 12, t: 8, b: 34 },
    hovermode: "x unified",
    legend: { orientation: "h", y: 1.12, font: { size: 12, color: t.legend } },
    xaxis: {
      type: "date",
      gridcolor: t.grid,
      linecolor: t.line,
      zerolinecolor: t.zero,
      tickfont: { color: t.axis },
      zeroline: false,
    },
    yaxis: {
      domain: [0.2, 1],
      range: [0, 1],
      tickformat: ".0%",
      gridcolor: t.grid,
      linecolor: t.line,
      zerolinecolor: t.zero,
      tickfont: { color: t.axis },
      zeroline: false,
      title: { text: "probability", font: { size: 10 } },
    },
    yaxis2: { domain: [0, 0.13], showticklabels: false, showgrid: false, zeroline: false },
  };

  return { data, layout };
}

function TransitionMatrix({ report }: { report: RegimeReport }) {
  const { theme } = useTheme();
  const { data, layout } = useMemo(() => buildMatrix(report), [report, theme]);
  return (
    <Plot
      data={data}
      layout={layout}
      config={{ responsive: true, displaylogo: false }}
      style={{ width: "100%", height: `${Math.max(260, report.transition_matrix.labels.length * 62)}px` }}
      useResizeHandler
    />
  );
}

function buildMatrix(report: RegimeReport): { data: Data[]; layout: Partial<Layout> } {
  const { labels, values } = report.transition_matrix;
  const text = values.map((row) => row.map((value) => (value > 0 ? value.toFixed(2) : "")));
  const t = chartTheme();

  const data: Data[] = [
    {
      type: "heatmap",
      x: labels.map((state) => STATE_LABEL[state]),
      y: labels.map((state) => STATE_LABEL[state]),
      z: values,
      text,
      texttemplate: "%{text}",
      textfont: { size: 11, family: "ui-monospace, SFMono-Regular, Menlo, monospace" },
      colorscale: [
        [0, t.plot],
        [1, t.gold],
      ],
      zmin: 0,
      zmax: 1,
      showscale: false,
      hovertemplate: "%{y} → %{x}: %{z:.1%}<extra></extra>",
      xgap: 2,
      ygap: 2,
    },
  ];

  const layout: Partial<Layout> = {
    paper_bgcolor: t.paper,
    plot_bgcolor: t.plot,
    font: { family: "Inter, system-ui, sans-serif", color: t.font, size: 11 },
    margin: { l: 110, r: 12, t: 8, b: 46 },
    xaxis: {
      title: { text: "to", font: { size: 10 } },
      linecolor: t.line,
      tickfont: { color: t.axis },
      side: "bottom",
    },
    yaxis: {
      title: { text: "from", font: { size: 10 } },
      autorange: "reversed",
      linecolor: t.line,
      tickfont: { color: t.axis },
    },
  };

  return { data, layout };
}

/* ------------------------------------------------------------ diagnostics */

function Diagnostics({ report }: { report: RegimeReport }) {
  const durations = Object.entries(report.state_durations);
  const totalRuns = durations.reduce((sum, [, entry]) => sum + entry.runs, 0);
  const averageDuration = totalRuns === 0 ? null : report.labelled_rows / totalRuns;
  const nva = report.net_value_add;

  return (
    <>
      <div className="regime__table mono">
        <Row label="BIC (latest refit)" value={format(report.bic_latest_refit, 2)} />
        <Row label="AIC (latest refit)" value={format(report.aic_latest_refit, 2)} />
        <Row label="Switch count" value={report.switch_count.toLocaleString()} />
        <Row
          label="Average state duration"
          value={averageDuration === null ? "—" : `${averageDuration.toFixed(1)} bars`}
        />
        <Row
          label="Net Value-Add"
          value={nva.net_value_add === null ? "—" : nva.net_value_add.toFixed(4)}
          // Spread rather than `tone={undefined}`: `exactOptionalPropertyTypes` treats the
          // two as different things, and it is right to — "no tone" and "a tone that
          // happens to be undefined" should not be the same call.
          {...(nva.net_value_add === null
            ? {}
            : { tone: nva.net_value_add >= 0 ? ("ok" as const) : ("bad" as const) })}
        />
      </div>

      <div className="report">
        <div className="report__row">
          <span className="report__key">Net Value-Add</span>
          <span className="report__value">
            {nva.net_value_add === null ? (
              "Not computable — no bars were scored."
            ) : (
              <>
                {nva.return_improvement === null ? "—" : signed(nva.return_improvement)} return
                improvement − {nva.switching_cost === null ? "—" : nva.switching_cost.toFixed(4)}{" "}
                switching cost over {nva.position_changes} position change
                {nva.position_changes === 1 ? "" : "s"} ={" "}
                <strong>{signed(nva.net_value_add)}</strong>
              </>
            )}
            {/* A run where the rule never held a position has an improvement that is just
                the negative of the benchmark, so the figure measures the market rather
                than the model. Say so, rather than let it read as a score. */}
            {nva.bars_long === 0 && nva.bars_scored > 0 ? (
              <div className="regime__warning regime__warning--grey">
                The rule never went long: no bar was committed to <em>Trending Up</em>, so the
                improvement is exactly the negative of the benchmark and this figure says
                nothing about the model.
              </div>
            ) : null}
            <div className="muted" style={{ fontSize: 11 }}>
              {nva.definition}. Scored over {nva.bars_scored.toLocaleString()} bars, long on{" "}
              {nva.bars_long.toLocaleString()}.
              {nva.in_sample
                ? " In-sample: this is the same data the model was fitted on, so it is a diagnostic, not evidence."
                : ""}
            </div>
          </span>
        </div>

        <div className="report__row">
          <span className="report__key">Refits</span>
          <span className="report__value mono" style={{ fontSize: 11 }}>
            {report.refit_points
              .map((point) =>
                point.trained_through_bar === undefined
                  ? "?"
                  : `≤${point.trained_through_bar}`,
              )
              .join(", ") || "none"}
            <span className="muted">
              {" "}
              — each label is produced by a fit that saw only earlier bars
            </span>
          </span>
        </div>

        <div className="report__row">
          <span className="report__key">Crisis override</span>
          <span className="report__value">
            fired on {report.crisis_override_bars.toLocaleString()} bar
            {report.crisis_override_bars === 1 ? "" : "s"}
            {report.crisis_bars_in_warmup > 0 && (
              <span className="regime__bad">
                {" "}
                · {report.crisis_bars_in_warmup} further breach
                {report.crisis_bars_in_warmup === 1 ? "" : "es"} fell in the warm-up, where
                there was no posterior to override. The detector was blind for those bars.
              </span>
            )}
          </span>
        </div>

        {report.stationarity.length > 0 && (
          <div className="report__row">
            <span className="report__key">Stationarity (ADF)</span>
            <span className="report__value">
              {report.stationarity.map((entry) => (
                <div key={entry.feature}>
                  <span className="mono">{entry.feature}</span> —{" "}
                  {entry.stationary === null ? (
                    <span className="muted">{entry.detail}</span>
                  ) : entry.stationary ? (
                    <span className="regime__ok">stationary at 5%</span>
                  ) : (
                    <span className="regime__bad">{entry.detail}</span>
                  )}
                </div>
              ))}
            </span>
          </div>
        )}

        {durations.length > 0 && (
          <div className="report__row">
            <span className="report__key">State durations</span>
            <span className="report__value mono" style={{ fontSize: 11 }}>
              {durations
                .map(
                  ([name, entry]) =>
                    `${name}: ${entry.runs}× mean ${entry.mean_bars.toFixed(1)} / longest ${entry.longest_bars}`,
                )
                .join(" · ")}
            </span>
          </div>
        )}
      </div>
    </>
  );
}

function Row({ label, value, tone }: { label: string; value: string; tone?: "ok" | "bad" }) {
  return (
    <div className="regime__table-row">
      <span className="regime__table-key">{label}</span>
      <span
        className={
          tone === undefined ? "regime__table-value" : `regime__table-value regime__${tone}`
        }
      >
        {value}
      </span>
    </div>
  );
}

/* ----------------------------------------------------------------- helpers */

function numeric(value: unknown): number {
  if (typeof value === "number") return Number.isFinite(value) ? value : 0;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function format(value: number | null, digits: number): string {
  return value === null ? "—" : value.toFixed(digits);
}

function signed(value: number): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(4)}`;
}
