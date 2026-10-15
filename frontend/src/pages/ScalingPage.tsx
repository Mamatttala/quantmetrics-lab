import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { ApiError, api } from "@/api/client";
import { notifyDatasetsChanged, subscribeToDatasets } from "@/api/datasetsChanged";
import { failureMessage, isAbort, waitForJob } from "@/api/jobs";
import type {
  DatasetSummary,
  JobRecord,
  ScalerType,
  ScalingCreated,
  ScalingOptions,
  SplitPlanSummary,
} from "@/api/types";
import { SCALER_OPTIONS } from "@/api/types";
import { HeadTailTable } from "@/components/HeadTailTable";
import { PageHeader } from "@/components/PageHeader";

interface Notice {
  message: string;
  kind: "ok" | "error";
}

/**
 * Scaling (C7).
 *
 * One rule governs this page, and every control on it is shaped by it:
 *
 * > **Fit on the training span. Transform everything. Never fit on anything the model will
 * > be scored on.**
 *
 * A scaler fitted on the whole series carries the test span's mean and range into every
 * training row. The model has seen the future before it is trained, and the transformed
 * frame looks entirely normal — the only symptom is a backtest that is too good. So the
 * three fitted scalers (min-max, standard, robust) cannot be *run* without a split plan:
 * the backend refuses the request at the schema, before any work happens. The page does not
 * re-derive that rule — it renders `options.scalers[].available` and the reason attached to
 * it, so a control greyed out here and a refusal from the endpoint cannot disagree.
 *
 * **The rolling z-score is the one member that is different in kind.** It estimates a mean
 * and a standard deviation from the trailing window at each row, so it is causal by
 * construction: nothing is fitted, nothing can leak, and it takes no plan. It is offered
 * even when the dataset has no plan at all.
 *
 * **A walk-forward plan needs a fold.** The plan holds one train/test pair per fold, and
 * pooling every fold's training rows into one fit would compute fold 1's statistics from
 * rows lying *after* fold 1's test span — look-ahead relative to that fold, and invisible.
 * So the fold control appears exactly when the selected plan is a walk-forward one.
 */

/** Plan/parameter artifacts are not frames to scale: a split plan has no timeline, and a
 *  scaler-parameter table's own columns are the numbers — offering it would invite scaling
 *  `center_` and `scale_`. */
const ARTIFACT_OPERATIONS = new Set(["split", "scaler_params"]);

function isWalkForwardPlan(plan: SplitPlanSummary | undefined): boolean {
  return plan !== undefined && plan.split_type.startsWith("tscv");
}

function formatInstant(value: string | null): string {
  if (value === null) return "—";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toISOString().replace("T", " ").replace("Z", " UTC");
}

export function ScalingPage() {
  const { datasetId } = useParams<{ datasetId?: string }>();
  const navigate = useNavigate();

  const [candidates, setCandidates] = useState<DatasetSummary[]>([]);
  const [options, setOptions] = useState<ScalingOptions | null>(null);
  const [optionsStatus, setOptionsStatus] = useState<
    "idle" | "loading" | "ready" | "error"
  >("idle");
  const [loadError, setLoadError] = useState<string | null>(null);

  const [scaler, setScaler] = useState<ScalerType>("standard");
  const [planId, setPlanId] = useState("");
  const [fold, setFold] = useState(1);
  const [picked, setPicked] = useState<ReadonlySet<string>>(new Set());
  // Not named `window`: it would shadow the global, and this page opens the parameters
  // export in a new tab from inside the same component.
  const [trailing, setTrailing] = useState("60");
  const [outName, setOutName] = useState("");

  const [result, setResult] = useState<JobRecord<ScalingCreated> | null>(null);
  const [running, setRunning] = useState(false);
  const [polls, setPolls] = useState(0);
  const [notice, setNotice] = useState<Notice | null>(null);

  const abort = useRef<AbortController | null>(null);

  // ------------------------------------------------------------ dataset list

  const loadCandidates = useCallback(async () => {
    try {
      const rows = await api.listDatasets();
      setCandidates(
        rows.filter(
          (row) => row.row_count > 0 && !ARTIFACT_OPERATIONS.has(row.operation ?? ""),
        ),
      );
    } catch {
      setCandidates([]);
    }
  }, []);

  useEffect(() => {
    void loadCandidates();
  }, [loadCandidates]);

  useEffect(() => subscribeToDatasets(() => void loadCandidates()), [loadCandidates]);

  useEffect(() => {
    if (datasetId === undefined) {
      setOptions(null);
      setOptionsStatus("idle");
      return;
    }
    let cancelled = false;
    setOptionsStatus("loading");
    setLoadError(null);

    api
      .scalingOptions(datasetId)
      .then((payload) => {
        if (cancelled) return;
        setOptions(payload);
        setOptionsStatus("ready");
        // The server's own defaults seed the form, so the first Run is the run the server
        // would have chosen rather than one this page invented.
        setPicked(new Set(payload.defaults.features));
        setScaler(payload.defaults.scaler_type);
        setTrailing(String(payload.defaults.window));
        setPlanId(payload.plans[0]?.split_plan_id ?? "");
        setFold(1);
      })
      .catch((cause: unknown) => {
        if (cancelled) return;
        setLoadError(cause instanceof ApiError ? cause.message : String(cause));
        setOptionsStatus("error");
      });

    return () => {
      cancelled = true;
    };
  }, [datasetId]);

  useEffect(() => {
    setResult(null);
    setNotice(null);
    setOutName("");
  }, [datasetId]);

  useEffect(() => () => abort.current?.abort(), []);

  // ------------------------------------------------------------------ inputs

  const scalerOption = useMemo(
    () => options?.scalers.find((entry) => entry.id === scaler),
    [options, scaler],
  );

  const needsPlan = scalerOption?.needs_split_plan ?? true;

  const selectedPlan = useMemo(
    () => options?.plans.find((plan) => plan.split_plan_id === planId),
    [options, planId],
  );

  const walkForward = needsPlan && isWalkForwardPlan(selectedPlan);

  const problems = useMemo<string[]>(() => {
    const found: string[] = [];
    if (picked.size === 0) {
      found.push("Tick at least one feature to scale.");
    }
    if (needsPlan && planId === "") {
      found.push(
        `${scaler} is fitted on a stretch of history, so it needs a split plan to say which ` +
          "rows it may be fitted on. Make one in C6 Chronological Splits first.",
      );
    }
    if (walkForward && (fold < 1 || fold > (selectedPlan?.folds ?? 0))) {
      found.push(
        `Pick a fold between 1 and ${selectedPlan?.folds ?? 0}: this is a walk-forward ` +
          "plan, and each fold has its own training span.",
      );
    }
    if (scaler === "rolling_zscore") {
      const bars = Number.parseInt(trailing, 10);
      if (!Number.isFinite(bars) || bars < 2) {
        found.push(
          "The trailing window needs at least two bars. A one-bar window has no dispersion " +
            "to divide by.",
        );
      }
    }
    return found;
  }, [picked.size, needsPlan, planId, scaler, walkForward, fold, selectedPlan, trailing]);

  const canRun = options !== null && problems.length === 0 && !running;

  // -------------------------------------------------------------------- run

  const run = async () => {
    if (options === null || !canRun) return;
    const controller = new AbortController();
    abort.current = controller;

    setRunning(true);
    setPolls(0);
    setNotice(null);
    setResult(null);

    try {
      const accepted = await api.runScaling({
        dataset_id: options.dataset_id,
        scaler_type: scaler,
        split_plan_id: needsPlan ? planId : null,
        fold: walkForward ? fold : null,
        // Sent in the options' own order so two identical runs produce identical bytes —
        // which is what makes the stored provenance comparable between them.
        features: options.features.filter((name) => picked.has(name)),
        window: scaler === "rolling_zscore" ? Number.parseInt(trailing, 10) : null,
        write: true,
        name: outName.trim() === "" ? null : outName.trim(),
      });

      const job = await waitForJob<ScalingCreated>(accepted.job_id, {
        signal: controller.signal,
        onPoll: (_record, attempt) => setPolls(attempt + 1),
      });
      setResult(job);
      if (job.status === "succeeded") {
        setNotice({ message: job.message ?? "Scaling finished.", kind: "ok" });
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
  };

  const empty = datasetId === undefined;
  const payload = result?.status === "succeeded" ? result.result : null;

  return (
    <div className="page">
      <PageHeader
        title="Scaling"
        lead={
          <>
            Rescale feature columns to a comparable range. Every scaler is fitted on the{" "}
            <strong>training span alone</strong> and applied to every row, so the test span
            never contributes a mean, a range or a median to the numbers a model is trained
            on. The frame keeps its rows and its other columns: only the ticked features are
            replaced.
          </>
        }
      />

      <div className="row" style={{ marginBottom: 14 }}>
        <div className="field">
          <label htmlFor="scaling-dataset">Dataset</label>
          <select
            id="scaling-dataset"
            value={datasetId ?? ""}
            onChange={(event) => {
              const next = event.target.value;
              navigate(next === "" ? "/scaling" : `/scaling/${next}`);
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
          <p className="field__note">
            A split plan must already exist for this dataset before min-max, standard or
            robust can be run — plans and parameter tables are not offered here.
          </p>
        </div>
      </div>

      {empty && (
        <div className="panel muted">
          No dataset selected. Scaling writes a new{" "}
          <span className="mono">processed_data</span> dataset and a separate{" "}
          <span className="mono">outputs</span> table of the fitted parameters, leaving the
          input untouched.
        </div>
      )}

      {optionsStatus === "loading" && <div className="panel muted">Loading options…</div>}

      {optionsStatus === "error" && (
        <div className="result__bar result__bar--error">
          <span>{loadError}</span>
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
          <div className="section">
            <div className="section__head">
              <h2 className="section__title">Source</h2>
              <span className="mono muted" style={{ fontSize: 11 }}>
                {options.dataset_id} · {options.rows.toLocaleString()} rows ·{" "}
                {options.features.length} numeric column(s)
              </span>
            </div>

            <div className="panel">
              {options.warnings.map((warning) => (
                <div
                  key={warning.code}
                  className={
                    warning.level === "error"
                      ? "scaling__warning scaling__warning--error"
                      : "scaling__warning"
                  }
                >
                  <span className="badge">{warning.level}</span> {warning.message}
                </div>
              ))}
              {options.warnings.length === 0 && (
                <span className="muted" style={{ fontSize: 12 }}>
                  Nothing to flag about this dataset.
                </span>
              )}
            </div>
          </div>

          <div className="section">
            <div className="section__head">
              <h2 className="section__title">Scaler</h2>
            </div>

            <div className="panel">
              <fieldset className="policy" disabled={running}>
                <legend className="policy__legend">How to rescale the columns</legend>
                <p className="policy__hint muted">
                  The three fitted scalers are marked unavailable until a split plan exists.
                  That is not a limitation to work around: without a plan the only stretch of
                  history available to fit on is the whole series, which includes the rows
                  the model will be scored on.
                </p>

                {SCALER_OPTIONS.map((option) => {
                  const availability = options.scalers.find(
                    (entry) => entry.id === option.value,
                  );
                  const unavailable =
                    availability !== undefined && !availability.available;
                  const selected = option.value === scaler;
                  return (
                    <label
                      className={`policy__option${selected ? " policy__option--on" : ""}${
                        unavailable ? " policy__option--off" : ""
                      }`}
                      key={option.value}
                    >
                      <input
                        type="radio"
                        name="scaler-type"
                        value={option.value}
                        checked={selected}
                        disabled={unavailable}
                        onChange={() => setScaler(option.value)}
                      />
                      <span className="policy__body">
                        <span className="policy__label">{option.label}</span>
                        <span className="policy__detail muted">{option.detail}</span>
                        {option.caution !== undefined && (
                          <span className="policy__caution">{option.caution}</span>
                        )}
                        {unavailable && (
                          <span className="policy__caution">{availability?.reason}</span>
                        )}
                      </span>
                    </label>
                  );
                })}

                {needsPlan && (
                  <div className="scaling__plan">
                    <div className="field field--tight">
                      <label htmlFor="scaling-plan">Split plan</label>
                      {options.plans.length === 0 ? (
                        <>
                          <select id="scaling-plan" value="" disabled>
                            <option value="">— no plan exists for this dataset —</option>
                          </select>
                          <p className="field__note">
                            Make one in{" "}
                            <Link to={`/splits/${options.dataset_id}`}>
                              C6 Chronological Splits
                            </Link>
                            . Until then only the rolling z-score can run here, because it
                            is the only scaler that fits nothing.
                          </p>
                        </>
                      ) : (
                        <>
                          <select
                            id="scaling-plan"
                            value={planId}
                            onChange={(event) => {
                              setPlanId(event.target.value);
                              setFold(1);
                            }}
                            disabled={running}
                          >
                            {options.plans.map((plan) => (
                              <option key={plan.split_plan_id} value={plan.split_plan_id}>
                                {plan.name} · {plan.split_type} · {plan.folds} fold(s) ·{" "}
                                {plan.rows.toLocaleString()} span(s)
                              </option>
                            ))}
                          </select>
                          <p className="field__note">
                            The parameters are estimated from this plan&apos;s training span
                            and applied to every row. The plan records the checksum of the
                            frame it was computed on, so a stale plan is refused rather than
                            silently applied.
                          </p>
                        </>
                      )}
                    </div>

                    {walkForward && (
                      <div className="field field--tight">
                        <label htmlFor="scaling-fold">Fold</label>
                        <select
                          id="scaling-fold"
                          value={String(fold)}
                          onChange={(event) => setFold(Number(event.target.value))}
                          disabled={running}
                        >
                          {Array.from(
                            { length: selectedPlan?.folds ?? 0 },
                            (_unused, index) => index + 1,
                          ).map((number) => (
                            <option key={number} value={String(number)}>
                              fold {number}
                            </option>
                          ))}
                        </select>
                        <p className="field__note">
                          A walk-forward plan is a set of train/test pairs, one per fold.
                          Fitting on every fold&apos;s training rows at once would compute
                          the first fold&apos;s statistics from rows lying{" "}
                          <strong>after</strong> its test span — look-ahead relative to that
                          fold, and invisible in the resulting frame.
                        </p>
                      </div>
                    )}
                  </div>
                )}

                {scaler === "rolling_zscore" && (
                  <div className="field field--tight">
                    <label htmlFor="scaling-window">Trailing window (bars)</label>
                    <input
                      id="scaling-window"
                      type="number"
                      min={2}
                      value={trailing}
                      onChange={(event) => setTrailing(event.target.value)}
                      disabled={running}
                    />
                    <p className="field__note">
                      Each row is standardised against the mean and standard deviation of
                      the <em>preceding</em> {trailing || "0"} bars, so its value never
                      depends on a row after it. The first {Math.max(
                        Number.parseInt(trailing, 10) - 1 || 0,
                        0,
                      )}{" "}
                      rows are left empty — there is no window behind them yet, and null is
                      the honest value.
                    </p>
                  </div>
                )}
              </fieldset>

              <div className="field" style={{ marginTop: 12 }}>
                <label>Features ({picked.size} of {options.features.length})</label>
                <div className="columnpick">
                  {options.features.map((name) => (
                    <label className="columnpick__row" key={name}>
                      <input
                        type="checkbox"
                        checked={picked.has(name)}
                        onChange={(event) => {
                          setPicked((current) => {
                            const next = new Set(current);
                            if (event.target.checked) next.add(name);
                            else next.delete(name);
                            return next;
                          });
                        }}
                        disabled={running}
                      />
                      <span className="columnpick__name mono">{name}</span>
                    </label>
                  ))}
                </div>
                <p className="field__note">
                  Only these columns are replaced. Every other column — including the
                  timestamp — is carried through unchanged, and no row is added or removed.
                </p>
              </div>

              <div className="field">
                <label htmlFor="scaling-name">Output dataset name</label>
                <input
                  id="scaling-name"
                  type="text"
                  placeholder="(defaults to the source name plus the scaler)"
                  value={outName}
                  onChange={(event) => setOutName(event.target.value)}
                  disabled={running}
                />
              </div>

              <div className="actions">
                <button
                  type="button"
                  className="btn"
                  onClick={() => void run()}
                  disabled={!canRun}
                >
                  {running
                    ? `Running… (${polls} poll${polls === 1 ? "" : "s"})`
                    : "Run Scaling"}
                </button>
                {running && (
                  <button
                    type="button"
                    className="btn btn--ghost btn--sm"
                    onClick={() => abort.current?.abort()}
                  >
                    stop waiting
                  </button>
                )}
                <span className="actions__spacer" />
                <span className="mono muted" style={{ fontSize: 11 }}>
                  {scaler}
                  {needsPlan && planId !== "" ? ` · ${planId}` : ""}
                  {walkForward ? ` · fold ${fold}` : ""}
                </span>
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
                Stopping the wait does not stop the job — it is running on the server and
                still lands in the job registry. Both artifacts are stored: the scaled frame
                in <strong>processed_data</strong> and the fitted parameters in{" "}
                <strong>outputs</strong>, so a prediction can be inverted later without the
                original being to hand.
              </div>
            </div>
          </div>

          {/* ============================================================ result */}
          {payload !== null && (
            <div className="section">
              <div className="section__head">
                <h2 className="section__title">
                  Result · {payload.report.scaler_type}
                </h2>
                <span className="mono muted" style={{ fontSize: 11 }}>
                  {payload.report.rows.toLocaleString()} rows · {payload.report.features.length}{" "}
                  feature(s) rescaled · fitted on{" "}
                  {payload.report.fit_rows.toLocaleString()} row(s)
                </span>
              </div>

              <div className="panel">
                <div className="report">
                  <div className="report__row">
                    <span className="report__key">Fitted on</span>
                    <span className="report__value">
                      {payload.report.fit_role ?? "—"}
                      {payload.report.split_plan_name !== null
                        ? ` — ${payload.report.split_plan_name}`
                        : ""}
                      {payload.report.fold !== null ? ` · fold ${payload.report.fold}` : ""}
                    </span>
                  </div>
                  <div className="report__row">
                    <span className="report__key">Fit span</span>
                    <span className="report__value mono" style={{ fontSize: 11 }}>
                      {formatInstant(payload.report.fit_start_time)} →{" "}
                      {formatInstant(payload.report.fit_end_time)}
                    </span>
                  </div>
                  <div className="report__row">
                    <span className="report__key">Rows fitted on</span>
                    <span className="report__value mono">
                      {payload.report.rows_used_for_fit.toLocaleString()} of{" "}
                      {payload.report.rows.toLocaleString()}
                      {payload.report.fit_rows > payload.report.rows_used_for_fit
                        ? ` — ${(
                            payload.report.fit_rows - payload.report.rows_used_for_fit
                          ).toLocaleString()} row(s) in the span carry an empty value for at least one feature`
                        : ""}
                    </span>
                  </div>
                  {payload.report.window !== null && (
                    <div className="report__row">
                      <span className="report__key">Window</span>
                      <span className="report__value mono">
                        {payload.report.window} bars
                      </span>
                    </div>
                  )}
                  <div className="report__row">
                    <span className="report__key">Plan</span>
                    <span className="report__value mono" style={{ fontSize: 11 }}>
                      {payload.report.split_plan_id ?? "none — nothing was fitted"}
                    </span>
                  </div>
                </div>

                <div className="section__head" style={{ marginTop: 14 }}>
                  <h3 className="section__title" style={{ fontSize: 12 }}>
                    Parameters ({payload.parameters.length}) — the inversion pair
                  </h3>
                </div>
                <table className="dtable">
                  <thead>
                    <tr>
                      <th>Feature</th>
                      <th>center_</th>
                      <th>scale_</th>
                      <th>Fitted rows</th>
                      <th>Inversion</th>
                    </tr>
                  </thead>
                  <tbody>
                    {payload.parameters.map((parameter) => (
                      <tr key={parameter.feature}>
                        <td className="mono">
                          {parameter.feature}
                          {parameter.constant && (
                            <span className="badge" style={{ marginLeft: 6 }}>
                              constant
                            </span>
                          )}
                        </td>
                        <td className="mono">{parameter.center_.toPrecision(6)}</td>
                        <td className="mono">{parameter.scale_.toPrecision(6)}</td>
                        <td className="mono">{parameter.fitted_rows.toLocaleString()}</td>
                        <td className="mono" style={{ fontSize: 11 }}>
                          {parameter.formula}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="field__note">
                  Every fitted scaler reduces to this one affine pair, so{" "}
                  <span className="mono">x = scaled · scale_ + center_</span> inverts
                  min-max, standard and robust alike — a prediction can be returned to price
                  units without knowing which one produced it. The raw library attributes
                  are stored beside it in the parameters dataset.
                </p>

                {payload.report.warnings.length > 0 && (
                  <div className="result__bar" style={{ marginTop: 12, marginBottom: 0 }}>
                    <span>what this run did:</span>
                    {payload.report.warnings.map((warning, index) => (
                      <div key={`${index}-${warning.slice(0, 20)}`}>· {warning}</div>
                    ))}
                  </div>
                )}
              </div>

              <div className="section__head" style={{ marginTop: 18 }}>
                <h3 className="section__title" style={{ fontSize: 13 }}>
                  {payload.preview.columns.length} column(s), first 5 and last 5 of{" "}
                  {payload.preview.row_count.toLocaleString()} rows
                </h3>
              </div>
              <HeadTailTable
                columns={payload.preview.columns}
                head={payload.preview.head}
                tail={payload.preview.tail}
                rowCount={payload.preview.row_count}
              />

              <div className="actions">
                <span className="mono muted" style={{ fontSize: 11 }}>
                  {payload.dataset?.dataset_id ?? "not stored"}
                  {payload.params_dataset !== null
                    ? ` · params ${payload.params_dataset.dataset_id}`
                    : ""}
                </span>
                <span className="actions__spacer" />
                {payload.dataset !== null && (
                  <>
                    <Link
                      className="btn btn--ghost btn--sm"
                      to={`/data/${payload.dataset.dataset_id}`}
                    >
                      open in Data Management
                    </Link>
                    <Link
                      className="btn btn--ghost btn--sm"
                      to={`/features/${payload.dataset.dataset_id}`}
                    >
                      add features
                    </Link>
                  </>
                )}
                {payload.params_dataset !== null && (
                  <Link
                    className="btn btn--ghost btn--sm"
                    to={`/data/${payload.params_dataset.dataset_id}`}
                  >
                    inspect the parameters
                  </Link>
                )}
                {payload.params_dataset !== null && (
                  <button
                    type="button"
                    className="btn btn--ghost btn--sm"
                    onClick={() =>
                      window.open(
                        `/api/datasets/${payload.params_dataset?.dataset_id}/export?format=json`,
                      )
                    }
                  >
                    export parameters as JSON
                  </button>
                )}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
