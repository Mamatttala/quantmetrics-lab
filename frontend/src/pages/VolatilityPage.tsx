/**
 * Module E — Volatility Forecasting.
 *
 * This is the first page in the application whose subject is something that has not
 * happened yet, and three things about it follow from that.
 *
 * **Two charts' worth of information, and they are not the same object.** The *scored*
 * forecast is one-step-ahead at each walk-forward position: it is what MQLIKE, the
 * Mincer–Zarnowitz regression and Diebold–Mariano are defined over, and it is what the run
 * saves. The *forward path* is `forecast_horizon` steps past the last observation: it is
 * what the fan chart draws, and nothing scores it, because there is no realised value to
 * score it against. They are drawn on one time axis — the scored span, then the projection
 * continuing off its end — which is the only arrangement in which the fan can be read
 * against how the model has actually been behaving. But they are labelled separately and
 * the table below scores only the first.
 *
 * **An empty metric is information.** A model whose optimiser failed on enough steps has
 * no MQLIKE; it does not have a MQLIKE of zero. The table prints the model's own
 * `unscored_reason` rather than an em-dash, because "—" beside a real number reads as a
 * very good score.
 *
 * **No verdict is computed here.** Which metric is better in which direction, which models
 * form a family, which parameters a model actually consumes, and whether a walk-forward
 * schedule fits the series all arrive from the server. Marking the highest MQLIKE as best
 * is a one-character mistake that inverts the conclusion of the whole report, and it should
 * not be possible to make it in two places independently.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { useNavigate, useParams } from "react-router-dom";

import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-dist-min";
import type { Data, Layout } from "plotly.js";

import { ApiError, api } from "@/api/client";
import { notifyDatasetsChanged, subscribeToDatasets } from "@/api/datasetsChanged";
import { failureMessage, isAbort, waitForJob } from "@/api/jobs";
import type {
  DatasetSummary,
  JobRecord,
  VolDistribution,
  VolForecastReport,
  VolForecastRunResult,
  VolMeanModel,
  VolMetricOption,
  VolModel,
  VolOptions,
} from "@/api/types";
import { HeadTailTable } from "@/components/HeadTailTable";
import { PageHeader } from "@/components/PageHeader";
import { chartTheme, useTheme } from "@/theme";

const Plot = createPlotlyComponent(Plotly);

/* ------------------------------------------------------------------ palette

   One colour per model, and a muted grey for the realised series. Literal strings rather
   than CSS variables for the same reason D1's state colours are: Plotly draws the traces
   and needs the values, so a token lookup here would be a second copy of the palette that
   could drift from the swatches beside the table. */

const MODEL_COLOUR: Record<VolModel, string> = {
  garch: "#d4a843",
  egarch: "#8b5cf6",
  gjr_garch: "#4ade80",
  arma_egarch_xgboost: "#60a5fa",
  tdnn_gjr_garch: "#f59e0b",
};

/** The realised series is deliberately the dimmest trace on the chart. It is the thing
 *  being forecast, not a forecast; a bright line for it would compete with the fan the
 *  page exists to show. */
const REALISED_COLOUR = "#888888";

const DISTRIBUTIONS: readonly { value: VolDistribution; label: string; detail: string }[] = [
  {
    value: "normal",
    label: "Normal",
    detail: "Symmetric, thin-tailed. Understates how often a large move happens.",
  },
  {
    value: "students_t",
    label: "Student's t",
    detail:
      "Symmetric with fitted tail weight. The usual default for daily and intraday returns.",
  },
  {
    value: "skew_student",
    label: "Skewed Student's t",
    detail: "Adds a shape parameter, so a large fall need not be as likely as a large rise.",
  },
  {
    value: "ged",
    label: "Generalised error",
    detail: "Nests the normal and the Laplace; the tail index is estimated, not assumed.",
  },
];

const MEAN_MODELS: readonly { value: VolMeanModel; label: string; detail: string }[] = [
  {
    value: "constant",
    label: "Constant",
    detail: "One intercept, estimated jointly with the variance equation.",
  },
  {
    value: "arma",
    label: "AR(p)",
    detail:
      "An autoregressive mean estimated jointly with the variance, using the same p. Not a full ARMA — arch has no ARMA mean beside a variance equation, and an MA part on top of an EGARCH(q) is not identifiable. A mis-specified order inflates the apparent persistence of the variance, so this is not the safe default.",
  },
];

interface Notice {
  message: string;
  kind: "ok" | "error";
}

/** One row of the output dataset, reduced to what the chart overlays. */
interface ScoredPoint {
  time: number;
  actual: number;
}

export function VolatilityPage() {
  const { datasetId } = useParams<{ datasetId?: string }>();
  const navigate = useNavigate();

  const [candidates, setCandidates] = useState<DatasetSummary[]>([]);
  const [options, setOptions] = useState<VolOptions | null>(null);
  const [optionsStatus, setOptionsStatus] = useState<"idle" | "loading" | "ready" | "error">(
    "idle",
  );
  const [optionsError, setOptionsError] = useState<string | null>(null);

  // ---- configuration
  const [priceSource, setPriceSource] = useState("close");
  const [models, setModels] = useState<ReadonlySet<VolModel>>(new Set(["garch"]));
  const [distribution, setDistribution] = useState<VolDistribution>("students_t");
  const [meanModel, setMeanModel] = useState<VolMeanModel>("constant");
  const [order, setOrder] = useState({ p: 1, o: 1, q: 1 });
  const [lags, setLags] = useState(5);
  const [trainWindow, setTrainWindow] = useState(500);
  const [testWindow, setTestWindow] = useState(250);
  const [stepSize, setStepSize] = useState(1);
  const [horizon, setHorizon] = useState(24);
  const [conditioning, setConditioning] = useState("");
  const [outName, setOutName] = useState("");
  const [pathModel, setPathModel] = useState<VolModel | null>(null);
  const [seededFor, setSeededFor] = useState<string | null>(null);

  // ---- run state
  const [running, setRunning] = useState(false);
  const [polls, setPolls] = useState(0);
  const [result, setResult] = useState<JobRecord<VolForecastRunResult> | null>(null);
  const [scored, setScored] = useState<Record<string, ScoredPoint[]>>({});
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
      .volatilityOptions(datasetId)
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

  // Seeded once per dataset, then the panel belongs to the user. Re-seeding on every
  // options change would discard a window the user had already typed while a later request
  // was in flight — and the defaults are derived from the row count, so they *do* change.
  useEffect(() => {
    if (options === null || seededFor === options.dataset_id) return;
    setSeededFor(options.dataset_id);
    const defaults = options.defaults;
    setPriceSource(defaults.price_source);
    setModels(new Set(defaults.models));
    setDistribution(defaults.distribution);
    setMeanModel(defaults.mean_model);
    setOrder({ p: defaults.p, o: defaults.o, q: defaults.q });
    setLags(defaults.lags);
    setTrainWindow(defaults.train_window);
    setTestWindow(defaults.test_window);
    setStepSize(defaults.step_size);
    setHorizon(defaults.forecast_horizon);
    setConditioning("");
    setOutName("");
    setPathModel(defaults.models[0] ?? null);
    setResult(null);
    setScored({});
    setNotice(null);
  }, [options, seededFor]);

  // Cancel the *wait*, never the job. A walk-forward that has been running for four minutes
  // keeps running and keeps its output when someone clicks a sidebar link.
  useEffect(() => () => abort.current?.abort(), []);

  const toggleModel = useCallback((id: VolModel) => {
    setModels((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  // ------------------------------------------------------- live arithmetic

  const selected = useMemo(
    () => (options?.models ?? []).filter((entry) => models.has(entry.id)),
    [options, models],
  );

  const needs = useMemo(() => {
    const union = new Set<string>();
    for (const entry of selected) for (const name of entry.needs) union.add(name);
    return union;
  }, [selected]);

  const needsLags = needs.has("lags");
  const needsO = needs.has("o");

  /** Ceiling division, matching `WalkForwardSpec.n_scored` on the server. */
  const nScored = stepSize >= 1 ? Math.ceil(testWindow / stepSize) : 0;

  const problems = useMemo(() => {
    const found: string[] = [];
    if (options === null) return found;

    for (const warning of options.warnings) {
      if (warning.level === "error") found.push(warning.message);
    }
    if (selected.length === 0) {
      found.push("Select at least one model — a run with no model produces nothing.");
    }

    const limits = options.limits;
    if (trainWindow < limits.min_train_window) {
      found.push(
        `Training window must be at least ${limits.min_train_window} bars. Below that a ` +
          "GARCH is fitting three parameters to noise, and the forecast it produces is " +
          "a constant that looks like a working model in a chart.",
      );
    }
    if (testWindow < limits.min_test_window) {
      found.push(`Test window must be at least ${limits.min_test_window} bars.`);
    }
    if (stepSize < limits.min_step_size) {
      found.push(`Step size must be at least ${limits.min_step_size}.`);
    }
    if (stepSize > testWindow) {
      found.push(
        `Step size (${stepSize}) exceeds the test window (${testWindow}), so the run would ` +
          "skip scored bars entirely while still paying for a refit at each position it " +
          "does visit.",
      );
    }
    if (nScored < limits.min_scored_steps) {
      found.push(
        `A ${testWindow}-bar test window at a step of ${stepSize} scores ${nScored} bars, ` +
          `and a metric needs at least ${limits.min_scored_steps} to be an estimate of ` +
          "anything. Lengthen the test window or shorten the step.",
      );
    }
    // The schedule is over the *return* series, which is one bar shorter than the frame
    // when returns are derived from close. Sizing against `rows` is how a form comes to
    // open on its own refusal.
    const needed = trainWindow + testWindow;
    if (needed > options.usable_returns) {
      found.push(
        `A ${trainWindow}-bar training window and a ${testWindow}-bar test window need at ` +
          `least ${needed} bars, and this series has ${options.usable_returns} usable ` +
          `returns (${options.rows} rows, less the first difference that has no ` +
          "predecessor).",
      );
    }
    if (needsO && order.o < 1) {
      found.push(
        "A selected model estimates an asymmetry, so the asymmetry order must be at " +
          "least 1. Use GARCH for a symmetric model.",
      );
    }
    if (horizon > limits.max_forecast_horizon) {
      found.push(`Forecast horizon must be at most ${limits.max_forecast_horizon} steps.`);
    }
    if (lags > limits.max_lags) {
      found.push(`Lags must be at most ${limits.max_lags}.`);
    }
    if (conditioning !== "") {
      const entry = options.conditioning.find((row) => row.dataset_id === conditioning);
      if (entry !== undefined && !entry.usable) {
        found.push(entry.reason ?? "That conditioning dataset cannot be joined.");
      }
    }
    return found;
  }, [
    options,
    selected,
    trainWindow,
    testWindow,
    stepSize,
    nScored,
    needsO,
    order.o,
    horizon,
    lags,
    conditioning,
  ]);

  const canRun = options !== null && problems.length === 0 && !running;

  // ------------------------------------------------------------ scored span

  /** The realised half of the chart, read back from the saved output.
   *
   *  From the stored dataset rather than from the job payload because the payload carries
   *  the metrics and the projection, not 250 rows of walk-forward output — and re-deriving
   *  the realised series here would be a second implementation of the return derivation.
   *  The frame is the module's own artifact, so what the chart overlays is exactly what was
   *  saved. */
  const loadScored = useCallback(async (outputId: string) => {
    try {
      const payload = await api.page(outputId, 0, 2000);
      const byModel: Record<string, ScoredPoint[]> = {};
      for (const row of payload.rows) {
        const name = row["model"];
        if (typeof name !== "string") continue;
        const time = new Date(String(row["open_time"] ?? "")).getTime();
        const actual = Number(row["actual_volatility"]);
        if (!Number.isFinite(time) || !Number.isFinite(actual)) continue;
        (byModel[name] ??= []).push({ time, actual });
      }
      setScored(byModel);
    } catch {
      // A chart overlay that cannot be loaded is not a failed run — the metrics and the
      // projection are already on screen. Missing the realised line is a smaller loss
      // than replacing the result with an error.
      setScored({});
    }
  }, []);

  const run = useCallback(async () => {
    if (options === null) return;
    const controller = new AbortController();
    abort.current = controller;

    setRunning(true);
    setPolls(0);
    setNotice(null);
    setResult(null);
    setScored({});

    try {
      const accepted = await api.runVolatilityForecast({
        dataset_id: options.dataset_id,
        price_source: priceSource,
        models: [...models],
        distribution,
        mean_model: meanModel,
        p: order.p,
        o: order.o,
        q: order.q,
        lags,
        walk_forward: {
          train_window: trainWindow,
          test_window: testWindow,
          step_size: stepSize,
        },
        forecast_horizon: horizon,
        conditioning_dataset_id: conditioning === "" ? null : conditioning,
        name: outName.trim() === "" ? null : outName.trim(),
        write: true,
      });

      const job = await waitForJob<VolForecastRunResult>(accepted.job_id, {
        signal: controller.signal,
        onPoll: (_record, attempt) => setPolls(attempt + 1),
      });
      setResult(job);
      if (job.status === "succeeded") {
        setNotice({ message: job.message ?? "Forecast finished.", kind: "ok" });
        notifyDatasetsChanged();
        const outputId = job.result?.dataset?.dataset_id;
        if (outputId !== undefined) void loadScored(outputId);
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
  }, [
    options,
    priceSource,
    models,
    distribution,
    meanModel,
    order,
    lags,
    trainWindow,
    testWindow,
    stepSize,
    horizon,
    conditioning,
    outName,
    loadScored,
  ]);

  const empty = datasetId === undefined;
  const chartModel =
    pathModel !== null && models.has(pathModel) ? pathModel : (selected[0]?.id ?? null);

  return (
    <div className="page">
      <PageHeader
        title="Volatility Forecasting"
        lead={
          <>
            Fit a variance model on a return series, score it <strong>out of sample</strong>{" "}
            with a walk-forward, and project the next bars as a fan chart. The scored run is
            saved as a new <span className="mono">outputs</span> dataset; the source is never
            modified.
          </>
        }
      />

      <div className="row">
        <div className="field field--tight">
          <label htmlFor="vol-dataset">Dataset</label>
          <select
            id="vol-dataset"
            value={datasetId ?? ""}
            onChange={(event) => {
              const next = event.target.value;
              navigate(next === "" ? "/volatility" : `/volatility/${next}`);
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
          No dataset selected. A forecast needs a return series — fetch bars in A1, or derive
          one in cleaning. The model is never fitted to a price level: GARCH describes the
          variance of returns, and a price has near-unit-root behaviour that fits beautifully
          and means nothing.
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
          {/* ======================================================== source */}
          <div className="section">
            <div className="section__head">
              <h2 className="section__title">Return series</h2>
              <span className="mono muted" style={{ fontSize: 11 }}>
                {options.dataset_id} · {options.rows.toLocaleString()} rows ·{" "}
                {options.usable_returns.toLocaleString()} usable returns ·{" "}
                {options.timeframe === null ? "no declared timeframe" : options.timeframe} ·{" "}
                {options.asset_class}
              </span>
            </div>

            <div className="panel">
              <fieldset className="policy" disabled={running}>
                <legend className="policy__legend">Where the returns come from</legend>
                <p className="policy__hint muted">
                  Every model here is a model of <em>returns</em>. Choosing close derives{" "}
                  <span className="mono">diff(log(close))</span>; a column is used exactly as
                  it is stored, so pointing this at a price, a running total or a cumulative
                  volume is the one mistake that would matter — and the stationarity check
                  is what catches it.
                </p>

                {options.sources.map((source) => (
                  <label
                    className={`policy__option${
                      source.id === priceSource ? " policy__option--on" : ""
                    }${source.available ? "" : " policy__option--off"}`}
                    key={source.id}
                  >
                    <input
                      type="radio"
                      name="vol-source"
                      value={source.id}
                      checked={source.id === priceSource}
                      disabled={!source.available}
                      onChange={() => setPriceSource(source.id)}
                    />
                    <span className="policy__body">
                      <span className="policy__label">{source.label}</span>
                      {!source.available && source.reason !== null && (
                        <span className="policy__caution">{source.reason}</span>
                      )}
                    </span>
                  </label>
                ))}

                <div className="field" style={{ marginTop: 10 }}>
                  <label htmlFor="vol-column">Or name a column</label>
                  <input
                    id="vol-column"
                    list="vol-columns"
                    value={priceSource}
                    disabled={running}
                    onChange={(event) => setPriceSource(event.target.value)}
                  />
                  <datalist id="vol-columns">
                    {options.columns.map((name) => (
                      <option key={name} value={name} />
                    ))}
                  </datalist>
                  <div className="field__note">
                    Any numeric column. It is used as given, so it must already be a return —
                    the stationarity test refuses one that is not.
                  </div>
                </div>
              </fieldset>

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

          {/* ======================================================== models */}
          <div className="section">
            <div className="section__head">
              <h2 className="section__title">Models</h2>
            </div>

            <div className="panel">
              <fieldset className="policy" disabled={running}>
                <legend className="policy__legend">Estimators</legend>
                <p className="policy__hint muted">
                  Every model refits at every scored step — that is what makes the causality
                  guarantee possible — so a run's cost is the number of scored bars times the
                  cost of one fit. The hybrids train a second model inside that loop.
                </p>

                {(["simple", "hybrid"] as const).map((family) => (
                  <div key={family}>
                    <div className="vol__family">
                      {family === "simple"
                        ? "Family A · arch estimators"
                        : "Family B · linear leg × learned correction"}
                    </div>
                    {options.models
                      .filter((entry) => entry.family === family)
                      .map((entry) => (
                        <label
                          className={`policy__option${
                            models.has(entry.id) ? " policy__option--on" : ""
                          }`}
                          key={entry.id}
                        >
                          <input
                            type="checkbox"
                            value={entry.id}
                            checked={models.has(entry.id)}
                            disabled={running}
                            onChange={() => toggleModel(entry.id)}
                          />
                          <span className="policy__body">
                            <span className="policy__label">
                              <span
                                className="regime__swatch"
                                style={{ background: MODEL_COLOUR[entry.id] }}
                              />{" "}
                              {entry.label}
                            </span>
                            <span className="policy__detail muted">{entry.note}</span>
                            {family === "hybrid" && (
                              <span className="policy__caution">
                                Refits a second model at every scored step — the slowest
                                option here by a wide margin.
                              </span>
                            )}
                          </span>
                        </label>
                      ))}
                  </div>
                ))}
              </fieldset>

              <div className="vol__grid">
                <div className="field">
                  <label>Distribution of the standardised shock</label>
                  <div className="field__note" style={{ marginTop: 0 }}>
                    How the model describes the shape of what it cannot predict. It changes
                    the likelihood, so it changes the fitted parameters — not just the
                    confidence bands.
                  </div>
                  {DISTRIBUTIONS.map((entry) => (
                    <label className="vol__radio" key={entry.value}>
                      <input
                        type="radio"
                        name="vol-distribution"
                        value={entry.value}
                        checked={entry.value === distribution}
                        disabled={running}
                        onChange={() => setDistribution(entry.value)}
                      />
                      <span>
                        <strong>{entry.label}</strong>{" "}
                        <span className="muted">{entry.detail}</span>
                      </span>
                    </label>
                  ))}
                </div>

                <div className="field">
                  <label>Mean equation</label>
                  {MEAN_MODELS.map((entry) => (
                    <label className="vol__radio" key={entry.value}>
                      <input
                        type="radio"
                        name="vol-mean"
                        value={entry.value}
                        checked={entry.value === meanModel}
                        disabled={running}
                        onChange={() => setMeanModel(entry.value)}
                      />
                      <span>
                        <strong>{entry.label}</strong>{" "}
                        <span className="muted">{entry.detail}</span>
                      </span>
                    </label>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* ==================================================== walk-forward */}
          <div className="section">
            <div className="section__head">
              <h2 className="section__title">Walk-forward</h2>
              <span className="mono muted" style={{ fontSize: 11 }}>
                {nScored} scored step{nScored === 1 ? "" : "s"}
              </span>
            </div>

            <div className="panel">
              <div className="vol__grid">
                <div className="field">
                  <label htmlFor="vol-train">Training window (bars)</label>
                  <input
                    id="vol-train"
                    type="number"
                    min={options.limits.min_train_window}
                    value={trainWindow}
                    disabled={running}
                    onChange={(event) => setTrainWindow(Number(event.target.value))}
                  />
                  <div className="field__note">
                    Bars one fit may see, immediately before the bar it forecasts.
                  </div>
                </div>

                <div className="field">
                  <label htmlFor="vol-test">Test window (bars)</label>
                  <input
                    id="vol-test"
                    type="number"
                    min={options.limits.min_test_window}
                    value={testWindow}
                    disabled={running}
                    onChange={(event) => setTestWindow(Number(event.target.value))}
                  />
                  <div className="field__note">
                    The <strong>last</strong> this many bars are scored, so the sample size
                    the Diebold–Mariano test has to work with is a number you choose rather
                    than one that follows from the series length.
                  </div>
                </div>

                <div className="field">
                  <label htmlFor="vol-step">Step size (bars)</label>
                  <input
                    id="vol-step"
                    type="number"
                    min={options.limits.min_step_size}
                    value={stepSize}
                    disabled={running}
                    onChange={(event) => setStepSize(Number(event.target.value))}
                  />
                  <div className="field__note">
                    How far the window rolls between refits. This is a cost knob, not a
                    leakage knob — every fit still sees only its own past — but a larger
                    step scores fewer bars.
                  </div>
                </div>

                <div className="field">
                  <label>Schedule</label>
                  <div className="mono" style={{ fontSize: 12 }}>
                    {trainWindow} train + {testWindow} test ={" "}
                    {trainWindow + testWindow} bars needed,{" "}
                    {options.usable_returns} available
                  </div>
                  <div className="field__note">
                    {nScored} one-step forecasts will be scored, each from a model refitted
                    on the {trainWindow} bars behind it.
                    {needsLags && (
                      <>
                        {" "}
                        A hybrid additionally trains its correction at each of those steps.
                      </>
                    )}
                  </div>
                </div>
              </div>

              <fieldset className="policy" disabled={running}>
                <legend className="policy__legend">Parameters</legend>
                <p className="policy__hint muted">
                  Only the parameters a selected model consumes are live. A control that
                  changes nothing is disabled with the reason, rather than left to look
                  effective.{" "}
                  {selected.length === 0
                    ? "No model is selected."
                    : `Selected: ${selected.map((entry) => entry.label).join(", ")}.`}
                </p>

                <div className="vol__grid">
                  <div className="field">
                    <label htmlFor="vol-p">ARCH order (p)</label>
                    <input
                      id="vol-p"
                      type="number"
                      min={1}
                      max={options.limits.max_order}
                      value={order.p}
                      disabled={running}
                      onChange={(event) =>
                        setOrder((current) => ({ ...current, p: Number(event.target.value) }))
                      }
                    />
                  </div>
                  <div className="field">
                    <label htmlFor="vol-o">Asymmetry order (o)</label>
                    <input
                      id="vol-o"
                      type="number"
                      min={needsO ? 1 : 0}
                      max={options.limits.max_order}
                      value={order.o}
                      disabled={running || !needsO}
                      onChange={(event) =>
                        setOrder((current) => ({ ...current, o: Number(event.target.value) }))
                      }
                    />
                    <div className="field__note">
                      {needsO
                        ? "EGARCH needs at least 1; GJR-GARCH is defined by exactly 1."
                        : "No selected model has an asymmetry term — a plain GARCH does not read this."}
                    </div>
                  </div>
                  <div className="field">
                    <label htmlFor="vol-q">GARCH order (q)</label>
                    <input
                      id="vol-q"
                      type="number"
                      min={1}
                      max={options.limits.max_order}
                      value={order.q}
                      disabled={running}
                      onChange={(event) =>
                        setOrder((current) => ({ ...current, q: Number(event.target.value) }))
                      }
                    />
                  </div>
                  <div className="field">
                    <label htmlFor="vol-lags">Feature lags</label>
                    <input
                      id="vol-lags"
                      type="number"
                      min={1}
                      max={options.limits.max_lags}
                      value={lags}
                      disabled={running || !needsLags}
                      onChange={(event) => setLags(Number(event.target.value))}
                    />
                    <div className="field__note">
                      {needsLags
                        ? "How far the hybrid features reach back. Each lag contributes the return and its absolute value, plus the same lags of any conditioning columns — all strictly in the past."
                        : "Only the hybrid models read this."}
                    </div>
                  </div>
                </div>
              </fieldset>

              <div className="vol__grid">
                <div className="field">
                  <label htmlFor="vol-horizon">Forward path (steps)</label>
                  <input
                    id="vol-horizon"
                    type="number"
                    min={1}
                    max={options.limits.max_forecast_horizon}
                    value={horizon}
                    disabled={running}
                    onChange={(event) => setHorizon(Number(event.target.value))}
                  />
                  <div className="field__note">
                    Bars projected past the last observation — what the fan chart draws. This
                    is <strong>not</strong> the scored forecast: MQLIKE and the rest are
                    defined against a realised value, and a projection has none.
                  </div>
                </div>

                <div className="field">
                  <label htmlFor="vol-conditioning">Conditioning dataset</label>
                  <select
                    id="vol-conditioning"
                    value={conditioning}
                    disabled={running}
                    onChange={(event) => setConditioning(event.target.value)}
                  >
                    <option value="">— none —</option>
                    {options.conditioning.map((entry) => (
                      <option key={entry.dataset_id} value={entry.dataset_id}>
                        {entry.name} · {entry.rows.toLocaleString()} rows
                        {entry.usable ? "" : " · unusable"}
                      </option>
                    ))}
                  </select>
                  <div className="field__note">
                    A Module D regime output, joined on{" "}
                    <span className="mono">open_time</span> and used as{" "}
                    <strong>lagged</strong> features by the hybrid models. The plain arch
                    family ignores it. Unusable entries are listed with their reason rather
                    than hidden.
                  </div>
                  {conditioning !== "" && (
                    <div
                      className={
                        options.conditioning.find((row) => row.dataset_id === conditioning)
                          ?.usable === false
                          ? "regime__warning regime__warning--error"
                          : "regime__warning"
                      }
                    >
                      {options.conditioning.find((row) => row.dataset_id === conditioning)
                        ?.usable === false
                        ? (options.conditioning.find(
                            (row) => row.dataset_id === conditioning,
                          )?.reason ?? "")
                        : `Joins ${(
                            options.conditioning.find(
                              (row) => row.dataset_id === conditioning,
                            )?.columns ?? []
                          ).join(", ")} as lagged features.`}
                    </div>
                  )}
                </div>

                <div className="field">
                  <label htmlFor="vol-name">Output name</label>
                  <input
                    id="vol-name"
                    value={outName}
                    placeholder="leave blank for a generated name"
                    disabled={running}
                    onChange={(event) => setOutName(event.target.value)}
                  />
                  <div className="field__note">
                    The run is saved as a new <span className="mono">outputs</span> dataset
                    carrying the whole request in its lineage, so it can be reproduced from
                    the record alone.
                  </div>
                </div>
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
                  : "Run Forecast"}
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
              The run happens in the background: pressing Run returns a job immediately and
              the page polls it. Stopping the wait does not stop the job — it still lands in
              the registry, and its output is still saved.
            </div>
          </div>

          {/* ======================================================= results */}
          {result !== null && (
            <Results
              job={result}
              scored={scored}
              metrics={options.metrics}
              chartModel={chartModel}
              onPickModel={setPathModel}
              onOpenDataset={navigate}
            />
          )}
        </>
      )}
    </div>
  );
}

/* ============================================================== results */

function Results({
  job,
  scored,
  metrics,
  chartModel,
  onPickModel,
  onOpenDataset,
}: {
  job: JobRecord<VolForecastRunResult>;
  scored: Record<string, ScoredPoint[]>;
  metrics: VolMetricOption[];
  chartModel: VolModel | null;
  onPickModel: (model: VolModel) => void;
  onOpenDataset: (to: string) => void;
}) {
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

  return (
    <>
      <div className="section">
        <div className="section__head">
          <h2 className="section__title">Fan chart</h2>
          <span className="mono muted" style={{ fontSize: 11 }}>
            {report.return_rows.toLocaleString()} returns · scored bars{" "}
            {report.first_scored_index.toLocaleString()}–
            {report.last_scored_index.toLocaleString()} · {report.forecast_horizon}-step
            projection
          </span>
        </div>

        <div className="panel">
          <div className="vol__path-picker">
            <span className="report__key">Bands shown for</span>
            <select
              value={chartModel ?? ""}
              onChange={(event) => onPickModel(event.target.value as VolModel)}
            >
              {report.forward_paths.map((path) => (
                <option key={path.model} value={path.model}>
                  {labelFor(report, path.model)}
                </option>
              ))}
            </select>
            <span className="muted" style={{ fontSize: 12 }}>
              The band is the empirical quantiles of the model's own walk-forward error ratio
              — how wrong it has actually been, not how wrong a correctly specified model
              would be. Every model's projection is drawn; only one set of bands is, because
              two overlapping fans are unreadable.
            </span>
          </div>

          <FanChart report={report} scored={scored} chartModel={chartModel} />
        </div>
      </div>

      <div className="section">
        <div className="section__head">
          <h2 className="section__title">Out-of-sample scores</h2>
          <span className="mono muted" style={{ fontSize: 11 }}>
            one-step-ahead · {report.walk_forward.test_window}-bar test window at a step of{" "}
            {report.walk_forward.step_size}
          </span>
        </div>

        <div className="panel">
          <MetricsTable report={report} metrics={metrics} />
          <div className="regime__warning">
            Scored against the{" "}
            <strong>{report.actual_proxy}</strong> as the realised variance — the standard
            proxy, and a very noisy one. A Mincer–Zarnowitz R² near zero is therefore the
            expected result for a correctly specified model, not evidence of a broken one:
            most of the proxy's variation is unpredictable by construction. Rank on MQLIKE,
            which is robust to the proxy, and read the R² as a calibration diagnostic only.
          </div>
        </div>
      </div>

      {report.comparisons.length > 0 && (
        <div className="section">
          <div className="section__head">
            <h2 className="section__title">Diebold–Mariano</h2>
            <span className="mono muted" style={{ fontSize: 11 }}>
              pairwise · Newey–West HAC variance, Harvey–Laybourne–Newbold corrected
            </span>
          </div>

          <div className="panel">
            <ComparisonTable report={report} />
            <div className="regime__warning">
              A positive statistic means the second model has the lower mean loss, and a
              p-value above 0.05 means the difference is not distinguishable from noise at
              that sample size — which is the usual outcome between two volatility models on
              a few hundred bars. The test is run on the bars both models scored a finite
              forecast on, so a "difference" is never partly a difference between two
              samples.
            </div>
          </div>
        </div>
      )}

      <div className="section">
        <div className="section__head">
          <h2 className="section__title">Run report</h2>
        </div>

        <div className="panel">
          <div className="report">
            <Row k="Return series">
              <span className="mono">{report.return_column}</span>{" "}
              <span className="muted">
                {report.derived_returns
                  ? "derived as diff(log(close))"
                  : `used as stored from price_source='${report.price_source}'`}
              </span>
            </Row>
            <Row k="Rows">
              <span className="mono">
                {report.return_rows.toLocaleString()} returns from{" "}
                {report.rows.toLocaleString()} rows
              </span>
            </Row>
            <Row k="Scored region">
              <span className="mono">
                bars {report.first_scored_index.toLocaleString()}–
                {report.last_scored_index.toLocaleString()}
              </span>{" "}
              <span className="muted">
                {report.first_scored_time === null
                  ? ""
                  : `${stamp(report.first_scored_time)} → ${
                      report.last_scored_time === null
                        ? ""
                        : stamp(report.last_scored_time)
                    }`}
              </span>
            </Row>
            <Row k="Stationarity (ADF)">
              <span className={report.stationary ? "regime__ok" : "regime__bad"}>
                {report.stationary ? "rejected unit root" : "not rejected"}
              </span>{" "}
              <span className="mono muted">
                statistic {report.adf_statistic.toFixed(4)} · p {report.adf_pvalue.toFixed(4)}
              </span>
            </Row>
            <Row k="Orders">
              <span className="mono">
                p={report.p} o={report.o} q={report.q}
                {report.lags > 0 ? ` lags=${report.lags}` : ""} · {report.mean_model}{" "}
                mean · {report.distribution}
              </span>
            </Row>
            <Row k="Fitting scale">
              <span className="mono">×{report.scale_factor.toFixed(0)}</span>{" "}
              <span className="muted">
                arch is fitted on percentage returns because its optimiser is poorly
                conditioned at 1e-4. The variance is divided by{" "}
                {Math.round(report.scale_factor ** 2).toLocaleString()} on the way out, which
                is exact — a conditioning transform, not a modelling choice.
              </span>
            </Row>
            <Row k="Forward path">
              <span className="mono">{report.forecast_horizon} steps</span>{" "}
              <span className="muted">
                {report.forward_step_seconds === null
                  ? "bar length unknown"
                  : `stepped by ${formatSeconds(report.forward_step_seconds)} — ${
                      report.forward_step_source ?? "source unrecorded"
                    }`}
              </span>
            </Row>
            {report.conditioning_dataset_id !== null && (
              <Row k="Conditioning">
                <span className="mono">{report.conditioning_dataset_name}</span>{" "}
                <span className="muted">
                  joined on open_time, used lagged: {report.conditioning_columns.join(", ")}
                </span>
              </Row>
            )}
          </div>

          {report.warnings.map((warning) => (
            <div className="regime__warning" key={warning}>
              <span className="badge">note</span> {warning}
            </div>
          ))}
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
            rowCount={result.dataset?.row_count ?? 0}
          />
          <div className="field__note">
            The stored frame is the <strong>scored walk-forward span only</strong> — real
            bars, each with the forecast that was made for it before it happened. The forward
            path is not written to the store: a dataset row in this application is an
            observation, and a projection of the next {report.forecast_horizon} bars is not
            one.
          </div>
        </div>
      </div>
    </>
  );
}

function Row({ k, children }: { k: string; children: ReactNode }) {
  return (
    <div className="report__row">
      <span className="report__key">{k}</span>
      <span className="report__value">{children}</span>
    </div>
  );
}

function labelFor(report: VolForecastReport, model: VolModel): string {
  const score = report.scores.find((entry) => entry.model === model);
  if (score === undefined) return model;
  return score.mqlike === null ? model : `${model} · MQLIKE ${score.mqlike.toFixed(4)}`;
}

/* -------------------------------------------------------------- metrics */

/** The scalar metrics, paired with the `VolModelScore` field each is read from.
 *
 *  `VolMetric` also has a `dm` member, which is a pairwise comparison with no per-model
 *  scalar — so this table can only be built by naming the four fields it reads. Which of
 *  them is a *ranking*, and which direction is better, is not decided here: both come from
 *  the server's metric options, because crowning the wrong row is how a report's
 *  conclusion gets inverted. */
const SCORE_COLUMNS = [
  { id: "mqlike", field: "mqlike" },
  { id: "rmse", field: "rmse" },
  { id: "mae", field: "mae" },
  { id: "mz_r2", field: "mz_r2" },
] as const;

function MetricsTable({
  report,
  metrics,
}: {
  report: VolForecastReport;
  metrics: VolMetricOption[];
}) {
  const optionFor = (id: string): VolMetricOption | undefined =>
    metrics.find((entry) => entry.id === id);

  const best: Record<string, VolModel | null> = {};
  for (const column of SCORE_COLUMNS) {
    const option = optionFor(column.id);
    const values = report.scores
      .map((score) => ({ model: score.model, value: score[column.field] }))
      .filter((entry): entry is { model: VolModel; value: number } => entry.value !== null);
    // Nothing is crowned in a column the server does not call a ranking (MZ R²), nor in one
    // where fewer than two models produced a finite number — with one candidate there is no
    // comparison for a mark to represent.
    if (option === undefined || !option.is_ranking || values.length < 2) {
      best[column.id] = null;
      continue;
    }
    const numbers = values.map((entry) => entry.value);
    const target = option.lower_is_better ? Math.min(...numbers) : Math.max(...numbers);
    const winners = values.filter((entry) => entry.value === target);
    best[column.id] = winners.length === 1 ? (winners[0]?.model ?? null) : null;
  }

  return (
    <table className="vol__table mono">
      <thead>
        <tr>
          <th>Model</th>
          <th>Family</th>
          {SCORE_COLUMNS.map((column) => {
            const option = optionFor(column.id);
            return (
              <th key={column.id}>
                {option?.label ?? column.id}
                {option?.is_ranking === true && (option.lower_is_better ? " ↓" : " ↑")}
              </th>
            );
          })}
          <th>Scored</th>
          <th>NaN / Inf</th>
          <th>Seconds</th>
        </tr>
      </thead>
      <tbody>
        {report.scores.map((score) => (
          <tr key={score.model}>
            <td>
              <span
                className="regime__swatch"
                style={{ background: MODEL_COLOUR[score.model] }}
              />{" "}
              {score.model}
            </td>
            <td className="muted">{score.family}</td>
            {SCORE_COLUMNS.map((column) => (
              <td
                key={column.id}
                className={best[column.id] === score.model ? "vol__best" : ""}
                title={optionFor(column.id)?.note ?? ""}
              >
                {fixed(score[column.field], column.id === "mz_r2" ? 3 : 4)}
              </td>
            ))}
            <td>
              {score.n_finite}/{score.n_steps}
            </td>
            <td className={score.n_nan + score.n_inf > 0 ? "regime__bad" : ""}>
              {score.n_nan} / {score.n_inf}
            </td>
            <td>{score.seconds.toFixed(2)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function fixed(value: number | null, digits = 4): string {
  return value === null ? "—" : value.toFixed(digits);
}

function ComparisonTable({ report }: { report: VolForecastReport }) {
  return (
    <table className="vol__table mono">
      <thead>
        <tr>
          <th>A</th>
          <th>B</th>
          <th>Statistic</th>
          <th>p-value</th>
          <th>Bars</th>
          <th>Reading</th>
        </tr>
      </thead>
      <tbody>
        {report.comparisons.map((entry) => (
          <tr key={`${entry.model_a}|${entry.model_b}`}>
            <td>{entry.model_a}</td>
            <td>{entry.model_b}</td>
            <td>{entry.statistic.toFixed(4)}</td>
            <td className={entry.significant ? "vol__best" : ""}>{entry.p_value.toFixed(4)}</td>
            <td>{entry.n_effective}</td>
            <td>
              {entry.significant && entry.better !== null ? (
                <>
                  <span className="regime__ok">{entry.better}</span> has the lower mean loss
                </>
              ) : (
                <span className="muted">
                  no significant difference at 5%
                  {entry.better === null ? "" : ` (${entry.better} lower, on noise)`}
                </span>
              )}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/* --------------------------------------------------------------- charts */

function FanChart({
  report,
  scored,
  chartModel,
}: {
  report: VolForecastReport;
  scored: Record<string, ScoredPoint[]>;
  chartModel: VolModel | null;
}) {
  const { theme } = useTheme();
  const { data, layout } = useMemo(
    () => buildFan(report, scored, chartModel),
    [report, scored, chartModel, theme],
  );

  if (report.forward_paths.length === 0) {
    return <div className="chart__empty">No forward path was produced.</div>;
  }

  return (
    <Plot
      data={data}
      layout={layout}
      config={{ responsive: true, displaylogo: false }}
      style={{ width: "100%", height: "460px" }}
      useResizeHandler
    />
  );
}

/**
 * The realised walk-forward volatility, then the projected fan off its end.
 *
 * One time axis, so the projection sits where it belongs — immediately after the last bar
 * anyone knows the answer for. The realised series comes from the saved output dataset, so
 * what is drawn is the same numbers the artifact holds rather than a re-derivation.
 *
 * The y-axis is *volatility*, not variance, because that is the unit a reader compares
 * against a return: a 3% bar. Variance is what the models are fitted on and what the table
 * scores, and mixing the two on one axis is the commonest way a volatility chart misleads.
 */
function buildFan(
  report: VolForecastReport,
  scored: Record<string, ScoredPoint[]>,
  chartModel: VolModel | null,
): { data: Data[]; layout: Partial<Layout> } {
  const data: Data[] = [];

  const realised = chartModel === null ? [] : (scored[chartModel] ?? []);
  if (realised.length > 0) {
    data.push({
      type: "scatter",
      mode: "lines",
      name: "realised (scored span)",
      x: realised.map((point) => point.time),
      y: realised.map((point) => point.actual),
      line: { color: REALISED_COLOUR, width: 1 },
      hovertemplate: "realised %{y:.4f}<extra></extra>",
    });
  }

  for (const path of report.forward_paths) {
    const times = path.times.map((value) => new Date(value).getTime());
    const isBandModel = path.model === chartModel;
    const colour = MODEL_COLOUR[path.model];

    if (isBandModel) {
      for (const band of path.bands) {
        data.push({
          type: "scatter",
          mode: "lines",
          name: `${path.model} ${band.label}`,
          x: times,
          y: band.high,
          line: { color: colour, width: 0 },
          hoverinfo: "skip",
          showlegend: false,
        } as Data);
        data.push({
          type: "scatter",
          mode: "lines",
          name: `${path.model} ${band.label}`,
          x: times,
          y: band.low,
          line: { color: colour, width: 0 },
          fill: "tonexty",
          fillcolor: fillFor(band.label, colour),
          hoverinfo: "skip",
          showlegend: false,
        } as Data);
      }
    }

    data.push({
      type: "scatter",
      mode: "lines+markers",
      name: isBandModel ? `${path.model} (bands)` : path.model,
      x: times,
      y: path.volatility,
      line: { color: colour, width: 2, dash: isBandModel ? "solid" : "dot" },
      marker: { size: 3, color: colour },
      hovertemplate: `${path.model} %{y:.4f}<extra></extra>`,
    });
  }

  const t = chartTheme();
  const layout: Partial<Layout> = {
    paper_bgcolor: t.paper,
    plot_bgcolor: t.plot,
    font: { family: "Inter, system-ui, sans-serif", color: t.font, size: 11 },
    margin: { l: 60, r: 12, t: 8, b: 34 },
    hovermode: "x unified",
    legend: { orientation: "h", y: 1.14, font: { size: 12, color: t.legend } },
    xaxis: {
      type: "date",
      gridcolor: t.grid,
      linecolor: t.line,
      zerolinecolor: t.zero,
      tickfont: { color: t.axis },
      zeroline: false,
    },
    yaxis: {
      gridcolor: t.grid,
      linecolor: t.line,
      zerolinecolor: t.zero,
      tickfont: { color: t.axis },
      zeroline: false,
      rangemode: "tozero",
      title: { text: "volatility (per bar)", font: { size: 10 } },
    },
  };

  return { data, layout };
}

/** The two bands, at different opacities, so the 25–75% is legible inside the 5–95%. */
function fillFor(label: string, colour: string): string {
  const alpha = label.startsWith("5") ? 0.16 : 0.24;
  return hexAlpha(colour, alpha);
}

function hexAlpha(hex: string, alpha: number): string {
  const value = hex.replace("#", "");
  const r = Number.parseInt(value.slice(0, 2), 16);
  const g = Number.parseInt(value.slice(2, 4), 16);
  const b = Number.parseInt(value.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/* -------------------------------------------------------------- helpers */

function stamp(value: string): string {
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toISOString().replace("T", " ").slice(0, 16);
}

function formatSeconds(seconds: number): string {
  if (seconds >= 86400) return `${(seconds / 86400).toFixed(2)}d`;
  if (seconds >= 3600) return `${(seconds / 3600).toFixed(2)}h`;
  if (seconds >= 60) return `${(seconds / 60).toFixed(1)}m`;
  return `${seconds.toFixed(0)}s`;
}
