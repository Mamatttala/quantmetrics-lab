import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { ApiError, api } from "@/api/client";
import { notifyDatasetsChanged, subscribeToDatasets } from "@/api/datasetsChanged";
import type {
  CleaningStep,
  CleanResult,
  DatasetSummary,
  DuplicatePolicy,
  MissingValuePolicy,
  OutlierPolicy,
  QualityResponse,
} from "@/api/types";
import {
  DUPLICATE_OPTIONS,
  MISSING_OPTIONS,
  OUTLIER_OPTIONS,
  QUALITY_LABELS,
  QUALITY_ORDER,
} from "@/api/types";
import { HeadTailTable, isNumericDtype } from "@/components/HeadTailTable";
import { PageHeader } from "@/components/PageHeader";
import { PolicyGroup } from "@/components/PolicyGroup";

interface Notice {
  message: string;
  kind: "ok" | "error";
}

/** The plan as the page holds it. Same fields as `CleanRequest` minus `write`, which
 *  is what the two buttons differ on. */
interface Plan {
  duplicates: DuplicatePolicy;
  missing: MissingValuePolicy;
  outliers: OutlierPolicy;
  outlierColumns: string[];
  lowerQuantile: number;
  upperQuantile: number;
  iqrMultiplier: number;
  sortByTime: boolean;
}

const DEFAULT_PLAN: Plan = {
  duplicates: "keep",
  missing: "keep",
  outliers: "keep",
  outlierColumns: [],
  lowerQuantile: 0.01,
  upperQuantile: 0.99,
  iqrMultiplier: 1.5,
  sortByTime: false,
};

/**
 * Cleaning (B3).
 *
 * The two halves of the specification live on one screen on purpose. Section A is the
 * **report** — what is actually wrong with this file — and section B is the **plan** —
 * which of those defects to repair. Splitting them across two pages would let a user
 * choose a repair without having seen the count that motivates it, and the whole risk
 * with a cleaning menu is repairing something that was never broken.
 *
 * Nothing here is repaired until the user names it. The default plan is all "do
 * nothing", and the backend refuses a request with no columns selected for outlier
 * handling rather than silently applying it to every numeric column — clamping price
 * levels is the single most destructive thing a one-click outlier button can do.
 *
 * **Preview and Save are the same computation.** Preview posts the identical plan with
 * `write: false`; the backend runs it, returns the report and the 5+5 preview, and
 * stores nothing. That is what makes the preview trustworthy: there is no second code
 * path that could disagree with the save.
 *
 * Saving does not navigate away, unlike Data Management. The report is the artifact of
 * this step — which rows were removed, which cells were substituted, where the fences
 * fell — and it is not recoverable from the saved dataset's preview. Navigating would
 * destroy the evidence at the moment it became worth reading. The child is offered as
 * a link instead.
 */
export function Cleaning() {
  const { datasetId } = useParams<{ datasetId?: string }>();
  const navigate = useNavigate();

  const [candidates, setCandidates] = useState<DatasetSummary[]>([]);
  const [quality, setQuality] = useState<QualityResponse | null>(null);
  const [qualityStatus, setQualityStatus] = useState<
    "idle" | "loading" | "ready" | "error"
  >("idle");
  const [loadError, setLoadError] = useState<string | null>(null);

  const [plan, setPlan] = useState<Plan>(DEFAULT_PLAN);
  const [name, setName] = useState("");
  const [result, setResult] = useState<CleanResult | null>(null);
  const [resultName, setResultName] = useState<string | null>(null);
  const [running, setRunning] = useState<"preview" | "save" | null>(null);
  const [notice, setNotice] = useState<Notice | null>(null);

  // ------------------------------------------------------------ dataset list

  const loadCandidates = useCallback(async () => {
    try {
      const rows = await api.listDatasets();
      setCandidates(rows.filter((dataset) => dataset.row_count > 0));
    } catch {
      setCandidates([]);
    }
  }, []);

  useEffect(() => {
    void loadCandidates();
  }, [loadCandidates]);

  // A dataset cleaned on this page, or fetched on another, must appear in the picker
  // without a reload.
  useEffect(() => subscribeToDatasets(() => void loadCandidates()), [loadCandidates]);

  useEffect(() => {
    if (datasetId === undefined) {
      setQuality(null);
      setQualityStatus("idle");
      return;
    }
    let cancelled = false;
    setQualityStatus("loading");
    setLoadError(null);

    api
      .quality(datasetId)
      .then((report) => {
        if (cancelled) return;
        setQuality(report);
        setQualityStatus("ready");
      })
      .catch((cause: unknown) => {
        if (cancelled) return;
        setLoadError(cause instanceof ApiError ? cause.message : String(cause));
        setQualityStatus("error");
      });

    return () => {
      cancelled = true;
    };
  }, [datasetId]);

  // A plan and a result belong to the dataset they were made for. Carrying either
  // across a change of dataset would apply a fence computed on one series to another
  // — and would show a report about a file that is no longer selected.
  useEffect(() => {
    setPlan(DEFAULT_PLAN);
    setName("");
    setResult(null);
    setResultName(null);
    setNotice(null);
  }, [datasetId]);

  // ------------------------------------------------------- candidate columns

  /**
   * Columns the outlier policy may target: numeric, and not the timestamp.
   *
   * The timestamp is excluded because a fence on it has no meaning — clamping a
   * timestamp moves an observation in time — and because the backend would refuse it
   * as non-numeric anyway once it is stored as an integer.
   */
  const outlierCandidates = useMemo(() => {
    if (quality === null) return [];
    return quality.columns.filter(
      (column) =>
        column.supported &&
        column.name !== quality.time_column &&
        isNumericDtype(column.dtype),
    );
  }, [quality]);

  const selectedOutliers = useMemo(
    () => new Set(plan.outlierColumns),
    [plan.outlierColumns],
  );

  const toggleOutlierColumn = useCallback((column: string) => {
    setPlan((current) => {
      const chosen = current.outlierColumns.includes(column);
      return {
        ...current,
        outlierColumns: chosen
          ? current.outlierColumns.filter((name) => name !== column)
          : [...current.outlierColumns, column],
      };
    });
  }, []);

  /**
   * Reasons this plan cannot be run, checked here so the user is told before a
   * round trip rather than by a rejection.
   *
   * These mirror the backend's own guards exactly, and the backend remains the
   * authority: everything it can check that this cannot — an unparseable timestamp, a
   * column that is not numeric — is left to it and surfaced as an error.
   */
  const planProblem = useMemo<string | null>(() => {
    if (plan.outliers !== "keep" && plan.outlierColumns.length === 0) {
      return (
        "Outlier handling is set but no column is selected. Choose the column(s) to " +
        "treat — applying a fence to every numeric column would clamp price levels."
      );
    }
    if (plan.outliers === "winsorize") {
      if (plan.lowerQuantile >= plan.upperQuantile) {
        return (
          `The lower quantile (${plan.lowerQuantile}) must be below the upper one ` +
          `(${plan.upperQuantile}); the fences would be inverted and every value ` +
          "would count as an outlier."
        );
      }
    }
    if (plan.missing === "interpolate_time" && quality !== null && !quality.time_column) {
      return (
        "Time-based interpolation needs a recognisable timestamp column, and this " +
        "dataset has none. Use forward fill, backward fill, or row removal instead."
      );
    }
    if (
      plan.sortByTime &&
      quality !== null &&
      qualityStatus === "ready" &&
      !quality.time_column
    ) {
      return "Sorting by time needs a recognisable timestamp column.";
    }
    return null;
  }, [plan, quality, qualityStatus]);

  // -------------------------------------------------------------- quality view

  const qualityTotals = useMemo(() => {
    if (quality === null) return { markers: 0, counted: 0 };
    return {
      markers: quality.markers.length,
      counted: Object.values(quality.counts).reduce((total, count) => total + count, 0),
    };
  }, [quality]);

  // -------------------------------------------------------------------- run

  const run = async (write: boolean) => {
    if (datasetId === undefined || planProblem !== null) return;
    setRunning(write ? "save" : "preview");
    setNotice(null);
    try {
      const response = await api.cleanDataset(datasetId, {
        name: name.trim() === "" ? null : name.trim(),
        duplicates: plan.duplicates,
        missing: plan.missing,
        outliers: plan.outliers,
        outlier_columns: plan.outlierColumns,
        lower_quantile: plan.lowerQuantile,
        upper_quantile: plan.upperQuantile,
        iqr_multiplier: plan.iqrMultiplier,
        sort_by_time: plan.sortByTime,
        write,
      });
      setResult(response);
      setResultName(name.trim() === "" ? null : name.trim());
      if (write && response.dataset !== null) {
        const child = response.dataset;
        setNotice({
          message:
            `Saved as ${child.dataset_id} — ${child.row_count.toLocaleString()} rows, ` +
            `${child.columns.length} column(s), in processed_data. The source dataset ` +
            "is unchanged.",
          kind: "ok",
        });
        // A new catalog row, so the explorer's list and this page's picker are stale.
        notifyDatasetsChanged();
      } else if (!write) {
        setNotice({
          message:
            "Preview only — nothing was stored. The report and the table below are what " +
            "saving would produce.",
          kind: "ok",
        });
      }
    } catch (cause) {
      setNotice({
        message: cause instanceof ApiError ? cause.message : String(cause),
        kind: "error",
      });
    } finally {
      setRunning(null);
    }
  };

  const empty = datasetId === undefined;

  return (
    <div className="page">
      <PageHeader
        title="Cleaning"
        lead={
          <>
            Check a dataset for defects, then choose which ones to repair. Every repair
            writes a <strong>new</strong> dataset and records the plan in its lineage —
            the source is never modified. Nothing is changed unless it is named here.
          </>
        }
      />

      <div className="row" style={{ marginBottom: 14 }}>
        <div className="field">
          <label htmlFor="dataset">Dataset</label>
          <select
            id="dataset"
            value={datasetId ?? ""}
            onChange={(e) => {
              const next = e.target.value;
              navigate(next === "" ? "/cleaning" : `/cleaning/${next}`);
            }}
          >
            <option value="">— choose a dataset —</option>
            {candidates.map((dataset) => (
              <option key={dataset.dataset_id} value={dataset.dataset_id}>
                {dataset.name} · {dataset.category} · {dataset.row_count.toLocaleString()}{" "}
                rows
              </option>
            ))}
          </select>
        </div>
      </div>

      {empty && (
        <div className="panel muted">
          No dataset selected. Pick one above, or click a dataset in the Data Explorer
          and open it here. A dataset cleaned here lands in <strong>processed_data</strong>{" "}
          and keeps <strong>parent_dataset_id</strong> pointing at its source.
        </div>
      )}

      {qualityStatus === "loading" && <div className="panel muted">Running checks…</div>}

      {qualityStatus === "error" && (
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

      {qualityStatus === "ready" && quality !== null && (
        <>
          {/* ============================================== A · quality report */}
          <div className="section">
            <div className="section__head">
              <h2 className="section__title">Data quality checks</h2>
              <span className="mono muted" style={{ fontSize: 11 }}>
                {quality.time_column
                  ? `time column: ${quality.time_column}`
                  : "no recognisable timestamp column"}
              </span>
            </div>

            <div className="panel">
              {!quality.time_column && (
                <div className="result__bar result__bar--error" style={{ marginTop: 0 }}>
                  <span>
                    No timestamp column could be identified, so ordering, duplicate
                    timestamps, bar spacing, and time-based interpolation cannot be
                    checked on this dataset.
                  </span>
                </div>
              )}

              <div className="meta">
                {QUALITY_ORDER.map((kind) => {
                  const count = quality.counts[kind];
                  return (
                    <div className="meta__item" key={kind}>
                      <div className="meta__key">{QUALITY_LABELS[kind]}</div>
                      <div
                        className="meta__value"
                        style={{
                          color:
                            count === undefined
                              ? "var(--muted)"
                              : count > 0
                                ? "var(--warn)"
                                : "var(--ok)",
                        }}
                      >
                        {count === undefined ? "not checked" : count.toLocaleString()}
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* A check that could not run is not a check that passed. Saying which
                  ones did not run is the difference between the two. */}
              {quality.skipped.length > 0 && (
                <div className="result__bar" style={{ marginBottom: 0 }}>
                  <span>not checked:</span>
                  {quality.skipped.map((reason) => (
                    <div key={reason}>· {reason}</div>
                  ))}
                </div>
              )}

              {/* Findings that have no per-row marker of their own — kept out of the
                  list above, which claims a check did not run. */}
              {quality.notes.length > 0 && (
                <div className="result__bar" style={{ marginBottom: 0 }}>
                  <span>also found:</span>
                  {quality.notes.map((note) => (
                    <div key={note}>· {note}</div>
                  ))}
                </div>
              )}

              {quality.markers.length > 0 && (
                <p className="muted" style={{ marginBottom: 0, fontSize: 12 }}>
                  {quality.markers.length.toLocaleString()} marker(s) recorded
                  {qualityTotals.markers < qualityTotals.counted
                    ? ` of ${qualityTotals.counted.toLocaleString()} counted — the API caps each kind`
                    : ""}
                  . Rows are listed on <strong>Raw Data View</strong> (A2), where each one
                  is plotted against its bar.
                </p>
              )}
            </div>

            {/* Column types are the evidence behind "invalid data types" and
                "invalid numeric values", which are per-column counts rather than
                per-row markers. Without this table those two numbers are unreadable. */}
            <div className="section__head" style={{ marginTop: 18 }}>
              <h2 className="section__title">Columns</h2>
              <span className="mono muted" style={{ fontSize: 11 }}>
                {quality.columns.length} column(s)
              </span>
            </div>
            <div className="panel">
              <table className="dtable">
                <thead>
                  <tr>
                    <th>Column</th>
                    <th>Type</th>
                    <th>Empty</th>
                    <th>Usable</th>
                  </tr>
                </thead>
                <tbody>
                  {quality.columns.map((column) => (
                    <tr key={column.name}>
                      <td className="mono">{column.name}</td>
                      <td className="mono muted">{column.dtype}</td>
                      <td
                        className="mono"
                        style={{ color: column.missing > 0 ? "var(--warn)" : undefined }}
                      >
                        {column.missing.toLocaleString()}
                      </td>
                      <td>
                        {column.supported ? (
                          <span className="badge badge--ok">yes</span>
                        ) : (
                          <span className="badge">no — unsupported type</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* ================================================ B · repair plan */}
          <div className="section">
            <div className="section__head">
              <h2 className="section__title">Handling</h2>
              <span className="mono muted" style={{ fontSize: 11 }}>
                every option defaults to “do nothing”
              </span>
            </div>

            <div className="stack">
              <PolicyGroup
                name="duplicates"
                legend="Duplicates"
                hint={
                  "Exact duplicates are two rows identical in every column. Two rows " +
                  "sharing a timestamp but differing in a value are a different defect " +
                  "and need one of the two time-based options."
                }
                options={DUPLICATE_OPTIONS}
                value={plan.duplicates}
                onChange={(duplicates) =>
                  setPlan((current) => ({ ...current, duplicates }))
                }
              />

              <PolicyGroup
                name="missing"
                legend="Missing values"
                hint={
                  "Applies to numeric columns only. The timestamp column is never " +
                  "filled — inventing a bar timestamp invents an observation, and it " +
                  "would change the bar spacing the missing-bar check measures."
                }
                options={MISSING_OPTIONS}
                value={plan.missing}
                onChange={(missing) => setPlan((current) => ({ ...current, missing }))}
              />

              <PolicyGroup
                name="outliers"
                legend="Extreme outliers"
                hint={
                  "The fence is estimated from the column's own distribution. Choose " +
                  "the columns deliberately: on a price level it clamps the trend, " +
                  "which is why the column list is required rather than defaulted."
                }
                options={OUTLIER_OPTIONS}
                value={plan.outliers}
                onChange={(outliers) => setPlan((current) => ({ ...current, outliers }))}
              >
                {plan.outliers !== "keep" && (
                  <div className="policy__aside">
                    <div>
                      <label className="policy__label" htmlFor="outlier-columns">
                        Columns to treat{" "}
                        <span className="mono muted">
                          ({plan.outlierColumns.length} selected)
                        </span>
                      </label>
                      <p className="field__note">
                        Numeric columns only. A column with empty cells shows its count —
                        a fence computed over few observations is a weak fence.
                      </p>
                      {outlierCandidates.length === 0 ? (
                        <div className="result__bar result__bar--error" style={{ margin: 0 }}>
                          <span>
                            This dataset has no numeric column other than the timestamp,
                            so there is nothing to apply a fence to.
                          </span>
                        </div>
                      ) : (
                        <div className="columnpick" id="outlier-columns">
                          {outlierCandidates.map((column) => (
                            <label className="columnpick__row" key={column.name}>
                              <input
                                type="checkbox"
                                checked={selectedOutliers.has(column.name)}
                                onChange={() => toggleOutlierColumn(column.name)}
                              />
                              <span className="columnpick__name" title={column.name}>
                                {column.name}
                              </span>
                              <span className="columnpick__dtype">{column.dtype}</span>
                              <span className="columnpick__missing">
                                {column.missing > 0
                                  ? `${column.missing.toLocaleString()} empty`
                                  : "complete"}
                              </span>
                            </label>
                          ))}
                        </div>
                      )}
                    </div>

                    {plan.outliers === "iqr" ? (
                      <div className="field field--tight">
                        <label htmlFor="iqr">IQR multiplier (k)</label>
                        <input
                          id="iqr"
                          type="number"
                          min={0.1}
                          max={20}
                          step={0.1}
                          value={plan.iqrMultiplier}
                          onChange={(e) =>
                            setPlan((current) => ({
                              ...current,
                              iqrMultiplier: Number(e.target.value),
                            }))
                          }
                        />
                        <p className="field__note">
                          Fences at Q1 − k·IQR and Q3 + k·IQR. 1.5 is the conventional
                          choice; 3.0 marks only far-out values.
                        </p>
                      </div>
                    ) : (
                      <div className="row">
                        <div className="field field--tight">
                          <label htmlFor="lower-q">Lower quantile</label>
                          <input
                            id="lower-q"
                            type="number"
                            min={0}
                            max={0.49}
                            step={0.005}
                            value={plan.lowerQuantile}
                            onChange={(e) =>
                              setPlan((current) => ({
                                ...current,
                                lowerQuantile: Number(e.target.value),
                              }))
                            }
                          />
                        </div>
                        <div className="field field--tight">
                          <label htmlFor="upper-q">Upper quantile</label>
                          <input
                            id="upper-q"
                            type="number"
                            min={0.51}
                            max={1}
                            step={0.005}
                            value={plan.upperQuantile}
                            onChange={(e) =>
                              setPlan((current) => ({
                                ...current,
                                upperQuantile: Number(e.target.value),
                              }))
                            }
                          />
                        </div>
                        <p className="field__note" style={{ flex: "1 1 260px" }}>
                          The conventional 0.01/0.99 pair clamps roughly 2% of every
                          column it is applied to. Widen the interval to treat only
                          genuinely extreme values.
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </PolicyGroup>

              <fieldset className="policy">
                <legend className="policy__legend">Order</legend>
                <label className="policy__option">
                  <input
                    type="checkbox"
                    checked={plan.sortByTime}
                    onChange={(e) =>
                      setPlan((current) => ({ ...current, sortByTime: e.target.checked }))
                    }
                  />
                  <span className="policy__body">
                    <span className="policy__label">Sort rows by timestamp first</span>
                    <span className="policy__detail muted">
                      A stable ascending sort by{" "}
                      {quality.time_column || "the timestamp column"}. Needed before
                      “keep the first” or “keep the last” per timestamp means anything:
                      on an unsorted file those words describe the row order in the file,
                      not in time. Ties keep their original order.
                    </span>
                  </span>
                </label>
              </fieldset>
            </div>
          </div>

          {/* ===================================================== run controls */}
          <div className="section">
            <div className="section__head">
              <h2 className="section__title">Run</h2>
            </div>

            <div className="actions">
              <div className="field" style={{ maxWidth: 320 }}>
                <label htmlFor="out-name">Output dataset name</label>
                <input
                  id="out-name"
                  type="text"
                  placeholder="(defaults to the source name plus “cleaned”)"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                />
              </div>
              <span className="actions__spacer" />
              <button
                type="button"
                className="btn btn--ghost"
                onClick={() => void run(false)}
                disabled={running !== null || planProblem !== null}
              >
                {running === "preview" ? "Running…" : "Preview (store nothing)"}
              </button>
              <button
                type="button"
                className="btn"
                onClick={() => void run(true)}
                disabled={running !== null || planProblem !== null}
              >
                {running === "save" ? "Saving…" : "Apply and save as new dataset"}
              </button>
              <button
                type="button"
                className="btn btn--ghost"
                onClick={() => {
                  setPlan(DEFAULT_PLAN);
                  setResult(null);
                  setNotice({ message: "Plan reset to “do nothing”.", kind: "ok" });
                }}
                disabled={running !== null}
              >
                Reset plan
              </button>
            </div>

            {planProblem !== null && (
              <div className="result__bar result__bar--error">
                <span>{planProblem}</span>
              </div>
            )}

            <div className="panel muted" style={{ fontSize: 12 }}>
              Preview runs the whole plan against the real file and throws the result
              away — the same code path as saving, so the two cannot disagree. Saving
              writes a new <strong>processed_data</strong> dataset; the fences and the
              plan are recorded in its provenance so a later reader can audit what was
              done to the data.
            </div>
          </div>

          {/* ========================================================== result */}
          {result !== null && (
            <div className="section">
              <div className="section__head">
                <h2 className="section__title">
                  {result.dataset === null ? "Preview" : "Saved"} · result
                </h2>
                <span className="mono muted" style={{ fontSize: 11 }}>
                  {result.report.rows_before.toLocaleString()} →{" "}
                  {result.report.rows_after.toLocaleString()} rows ·{" "}
                  {result.report.columns_after.length} column(s)
                </span>
              </div>

              <div className="panel">
                <div className="report">
                  <div className="report__row">
                    <span className="report__key">Rows before</span>
                    <span className="report__value">
                      {result.report.rows_before.toLocaleString()}
                    </span>
                  </div>
                  <div className="report__row">
                    <span className="report__key">Rows removed</span>
                    <span
                      className={
                        result.report.rows_removed > 0
                          ? "report__value report__value--removed"
                          : "report__value"
                      }
                    >
                      {result.report.rows_removed.toLocaleString()}
                    </span>
                  </div>
                  <div className="report__row">
                    <span className="report__key">Rows after</span>
                    <span className="report__value">
                      {result.report.rows_after.toLocaleString()}
                    </span>
                  </div>
                  <div className="report__row">
                    <span className="report__key">Time column</span>
                    <span className="report__value mono">
                      {result.report.time_column || "none recognised"}
                    </span>
                  </div>
                </div>

                {/* Steps that were applied, in the order they ran. An empty list means
                    the plan changed nothing — and is shown as such rather than as an
                    empty panel, because "nothing was wrong" and "nothing was checked"
                    look identical otherwise. */}
                <div className="section__head" style={{ marginTop: 14 }}>
                  <h3 className="section__title" style={{ fontSize: 12 }}>
                    Steps applied ({result.report.steps.length})
                  </h3>
                </div>
                {result.report.steps.length === 0 ? (
                  <p className="muted" style={{ margin: 0, fontSize: 12 }}>
                    No step was applied — every policy was left at “do nothing”, so the
                    output would be identical to the source.
                  </p>
                ) : (
                  <table className="dtable">
                    <thead>
                      <tr>
                        <th>Step</th>
                        <th>Policy</th>
                        <th>Effect</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.report.steps.map((step, index) => (
                        <tr key={`${step.step}-${index}`}>
                          <td className="mono">{step.step}</td>
                          <td className="mono muted">{step.policy}</td>
                          <td>
                            <StepEffect step={step} />
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}

                {result.report.warnings.length > 0 && (
                  <div className="result__bar" style={{ marginBottom: 0 }}>
                    <span>what this did to the data:</span>
                    {result.report.warnings.map((warning, index) => (
                      <div key={`${index}-${warning.slice(0, 24)}`}>· {warning}</div>
                    ))}
                  </div>
                )}
              </div>

              {/* Fences are the one part of the report that cannot be recovered from
                  the resulting frame, so they are shown in full rather than summarised. */}
              {result.report.steps.some(
                (step) => Object.keys(step.fences ?? {}).length > 0,
              ) && (
                <div className="panel">
                  <div className="report__key" style={{ marginBottom: 8 }}>
                    Outlier fences
                  </div>
                  <table className="dtable">
                    <thead>
                      <tr>
                        <th>Column</th>
                        <th>Lower</th>
                        <th>Upper</th>
                        <th>Rule</th>
                        <th>Outside</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.report.steps.flatMap((step) =>
                        Object.entries(step.fences ?? {}).map(([column, fence]) => (
                          <tr key={`${step.step}-${column}`}>
                            <td className="mono">{column}</td>
                            <td className="mono">{fence.lower}</td>
                            <td className="mono">{fence.upper}</td>
                            <td className="mono muted">{fence.rule}</td>
                            <td className="mono">{fence.outside.toLocaleString()}</td>
                          </tr>
                        )),
                      )}
                    </tbody>
                  </table>
                </div>
              )}

              <div className="section__head" style={{ marginTop: 18 }}>
                <h3 className="section__title" style={{ fontSize: 13 }}>
                  {result.preview.row_count.toLocaleString()} rows, first 5 and last 5
                  {result.dataset === null ? " · not stored" : ""}
                </h3>
              </div>
              <HeadTailTable
                columns={result.preview.columns}
                head={result.preview.head}
                tail={result.preview.tail}
                rowCount={result.preview.row_count}
              />

              {result.dataset !== null && (
                <div className="actions">
                  <span className="mono muted" style={{ fontSize: 11 }}>
                    {result.dataset.dataset_id}
                    {resultName !== null ? ` · ${resultName}` : ""}
                  </span>
                  <span className="actions__spacer" />
                  <Link className="btn btn--ghost btn--sm" to={`/data/${result.dataset.dataset_id}`}>
                    open in Data Management
                  </Link>
                  <Link className="btn btn--ghost btn--sm" to={`/view/${result.dataset.dataset_id}`}>
                    view chart
                  </Link>
                  {(["csv", "parquet"] as const).map((format) => (
                    <button
                      key={format}
                      type="button"
                      className="btn btn--ghost btn--sm"
                      onClick={() =>
                        window.open(
                          `/api/datasets/${result.dataset?.dataset_id}/export?format=${format}`,
                        )
                      }
                    >
                      export {format}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}

/** One line describing what a step did, built from whichever counters that step
 *  carries — the report is a plain dict, so the fields present differ per policy. */
function StepEffect({ step }: { step: CleaningStep }) {
  const parts: string[] = [];
  if (step.rows_removed !== undefined && step.rows_removed > 0) {
    parts.push(`${step.rows_removed.toLocaleString()} row(s) removed`);
  }
  if (step.filled !== undefined) parts.push(`${step.filled.toLocaleString()} cell(s) filled`);
  if (step.cells_changed !== undefined) {
    parts.push(`${step.cells_changed.toLocaleString()} cell(s) clamped`);
  }
  if (step.columns !== undefined && step.columns.length > 0) {
    parts.push(`columns: ${step.columns.join(", ")}`);
  }
  return (
    <>
      {step.detail !== undefined && <div>{step.detail}</div>}
      {parts.length > 0 && <div className="mono muted">{parts.join(" · ")}</div>}
    </>
  );
}
