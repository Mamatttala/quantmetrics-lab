import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { ApiError, api } from "@/api/client";
import { notifyDatasetsChanged, subscribeToDatasets } from "@/api/datasetsChanged";
import type {
  DatasetMeta,
  DatasetSummary,
  SplitCreateRequest,
  SplitParameters,
  SplitPlanCreated,
  SplitPlanSummary,
  SplitType,
} from "@/api/types";
import { SPLIT_TYPE_OPTIONS } from "@/api/types";
import { PageHeader } from "@/components/PageHeader";
import { PolicyGroup } from "@/components/PolicyGroup";
import { Timeline } from "@/components/Timeline";

interface Notice {
  message: string;
  kind: "ok" | "error";
}

/**
 * Chronological Splits (C6).
 *
 * The page plans a split; it does not perform one. Nothing here produces a new price
 * frame, and that is a deliberate architectural statement rather than an omission: a plan
 * is a table of row ranges, so the dataset a model trains on is the one the user already
 * has, sliced by the same indices every other module reads. Making a second, cut-up copy
 * of the data would mean two frames that could drift apart — and the one that drifted
 * would be the one being trained on.
 *
 * **What is on screen and why.** Three things get more space than a plain form would give
 * them, because each is a decision whose consequence is invisible in the resulting table:
 *
 * * The split-type radios carry the consequence. "Walk-forward — rolling window" discards
 *   history on purpose, and that is only legible beside the option rather than in a
 *   document.
 * * The gap and purge controls report the rows each rule *actually* removed. They overlap
 *   rather than add up — the embargo drops `[c-gap, c)` and the purge drops `[c-h, c)` — so
 *   a plan with `gap_size >= label_horizon_bars` removed nothing extra, and the panel says
 *   so instead of presenting two knobs as twice the safety.
 * * The timeline is positioned by row index rather than by wall-clock time, so an embargo
 *   or a purge is visible as the gap it is. On a time axis a five-bar gap in an hourly
 *   series is invisible.
 */

/** The date fields, in the order the spans read left to right. */
const DATE_FIELDS = [
  { key: "train_start", label: "Train from", role: "train" },
  { key: "train_end", label: "Train to", role: "train" },
  { key: "val_start", label: "Validation from", role: "val" },
  { key: "val_end", label: "Validation to", role: "val" },
  { key: "test_start", label: "Test from", role: "test" },
  { key: "test_end", label: "Test to", role: "test" },
] as const;

type DateFieldKey = (typeof DATE_FIELDS)[number]["key"];

/** Which date fields each split type takes. Mirrors the backend's `_REQUIRED_DATES`. */
const DATE_FIELDS_FOR: Record<string, readonly DateFieldKey[]> = {
  date_3way: DATE_FIELDS.map((field) => field.key),
  date_2way: ["train_start", "train_end", "test_start", "test_end"],
};

/** The proportions each percentage member stands for, for display. The *row counts* are
 *  computed by the backend and read back from the report; this is a label, not arithmetic,
 *  so the two cannot disagree about where the cut falls. */
const PROPORTIONS: Record<string, string> = {
  percentage_70_15_15: "70% train · 15% validation · 15% test",
  percentage_60_20_20: "60% train · 20% validation · 20% test",
};

const ARTIFACT_OPERATIONS = new Set(["split", "scaler_params"]);

function isDateSplit(splitType: SplitType): boolean {
  return splitType === "date_3way" || splitType === "date_2way";
}

function isWalkForward(splitType: SplitType): boolean {
  return splitType === "tscv_expanding" || splitType === "tscv_rolling";
}

/** A `datetime-local` value in the browser's own zone, sent as the instant it names.
 *
 *  The input gives `2024-01-01T00:00` with no offset, and the backend refuses a zone-less
 *  boundary outright — because it would be read in the *server's* zone, which may be
 *  nowhere near the user's, and would silently move bars between spans. Converting here
 *  means the instant the user typed is the instant that is sent, with the offset made
 *  explicit. The note under the inputs says so. */
function toUtcInstant(local: string): string | null {
  if (local.trim() === "") return null;
  const parsed = new Date(local);
  if (Number.isNaN(parsed.getTime())) return null;
  return parsed.toISOString();
}

function formatTimestamp(value: string): string {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toISOString().replace("T", " ").replace("Z", " UTC");
}

export function SplitsPage() {
  const { datasetId } = useParams<{ datasetId?: string }>();
  const navigate = useNavigate();

  const [candidates, setCandidates] = useState<DatasetSummary[]>([]);
  const [dataset, setDataset] = useState<DatasetMeta | null>(null);
  const [datasetStatus, setDatasetStatus] = useState<"idle" | "loading" | "ready" | "error">(
    "idle",
  );
  const [loadError, setLoadError] = useState<string | null>(null);

  const [plans, setPlans] = useState<SplitPlanSummary[]>([]);

  const [splitType, setSplitType] = useState<SplitType>("percentage_70_15_15");
  const [dates, setDates] = useState<Record<string, string>>({});
  const [nFolds, setNFolds] = useState(5);
  const [gapSize, setGapSize] = useState(0);
  const [purge, setPurge] = useState(false);
  const [horizon, setHorizon] = useState("5");
  const [outName, setOutName] = useState("");

  const [result, setResult] = useState<SplitPlanCreated | null>(null);
  const [running, setRunning] = useState(false);
  const [notice, setNotice] = useState<Notice | null>(null);

  // ------------------------------------------------------------ dataset list

  const loadCandidates = useCallback(async () => {
    try {
      const rows = await api.listDatasets();
      // Plans and parameter tables are excluded from the picker. A split plan is a table
      // of row indices, so offering it as something to split would invite a plan over a
      // plan; a scaler-parameter table has no timeline at all and would be refused by the
      // backend anyway. Filtering here is a convenience, not the guard — the backend
      // refuses either one on its own terms.
      setCandidates(
        rows.filter(
          (row) =>
            row.row_count > 0 &&
            !ARTIFACT_OPERATIONS.has(row.operation ?? ""),
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
      setDataset(null);
      setDatasetStatus("idle");
      return;
    }
    let cancelled = false;
    setDatasetStatus("loading");
    setLoadError(null);

    api
      .getDataset(datasetId)
      .then((meta) => {
        if (cancelled) return;
        setDataset(meta);
        setDatasetStatus("ready");
      })
      .catch((cause: unknown) => {
        if (cancelled) return;
        setLoadError(cause instanceof ApiError ? cause.message : String(cause));
        setDatasetStatus("error");
      });

    return () => {
      cancelled = true;
    };
  }, [datasetId]);

  const loadPlans = useCallback(async (id: string) => {
    try {
      setPlans(await api.splitPlans(id));
    } catch {
      setPlans([]);
    }
  }, []);

  useEffect(() => {
    if (datasetId === undefined) {
      setPlans([]);
      return;
    }
    void loadPlans(datasetId);
  }, [datasetId, loadPlans]);

  // A plan is a statement about one exact frame, so the controls reset with the dataset.
  // Carrying dates across would leave boundaries that may not even fall inside the new
  // frame's range, and a result computed against different data than the form describes.
  useEffect(() => {
    setDates({});
    setNFolds(5);
    setGapSize(0);
    setPurge(false);
    setHorizon("5");
    setOutName("");
    setResult(null);
    setNotice(null);
  }, [datasetId]);

  // ------------------------------------------------------------------ inputs

  const requiredDates = DATE_FIELDS_FOR[splitType] ?? [];
  const showsDerivedControls = !isDateSplit(splitType);

  const problems = useMemo<string[]>(() => {
    const found: string[] = [];
    if (isDateSplit(splitType)) {
      for (const key of requiredDates) {
        const raw = dates[key] ?? "";
        if (raw.trim() === "") {
          const field = DATE_FIELDS.find((entry) => entry.key === key);
          found.push(`${field?.label ?? key} needs a date and time.`);
        } else if (toUtcInstant(raw) === null) {
          found.push(`${key.replace("_", " ")} is not a timestamp this browser can read.`);
        }
      }
      const instants = requiredDates.map((key) => toUtcInstant(dates[key] ?? ""));
      if (instants.every((value) => value !== null)) {
        for (let index = 1; index < instants.length; index += 1) {
          if (instants[index]! <= instants[index - 1]!) {
            const later = DATE_FIELDS.find((f) => f.key === requiredDates[index]);
            const earlier = DATE_FIELDS.find((f) => f.key === requiredDates[index - 1]);
            found.push(
              `${later?.label ?? ""} must be after ${earlier?.label ?? ""} — spans may not ` +
                "share a bar.",
            );
          }
        }
      }
    }
    if (purge && showsDerivedControls) {
      const bars = Number.parseInt(horizon, 10);
      if (!Number.isFinite(bars) || bars < 1) {
        found.push(
          "The label horizon must be at least one bar: it is how far forward a label looks, " +
            "and it is the number of training rows the purge removes.",
        );
      }
    }
    return found;
  }, [splitType, requiredDates, dates, purge, showsDerivedControls, horizon]);

  const canRun = problems.length === 0 && datasetId !== undefined;

  const parameters = useMemo<Partial<SplitParameters>>(() => {
    if (isDateSplit(splitType)) {
      const built: Partial<SplitParameters> = {};
      for (const key of requiredDates) {
        built[key] = toUtcInstant(dates[key] ?? "");
      }
      return built;
    }
    const built: Partial<SplitParameters> = {
      gap_size: gapSize,
      purge_flag: purge,
      label_horizon_bars: purge ? Number.parseInt(horizon, 10) : null,
    };
    if (isWalkForward(splitType)) built.n_folds = nFolds;
    return built;
  }, [splitType, requiredDates, dates, gapSize, purge, horizon, nFolds]);

  // -------------------------------------------------------------------- run

  const run = async () => {
    if (datasetId === undefined || !canRun) return;
    setRunning(true);
    setNotice(null);
    try {
      const body: SplitCreateRequest = {
        dataset_id: datasetId,
        split_type: splitType,
        parameters,
        name: outName.trim() === "" ? null : outName.trim(),
      };
      const created = await api.createSplitPlan(body);
      setResult(created);
      const neverTested = created.report.rows_never_tested;
      setNotice({
        message:
          `Plan stored as ${created.dataset.dataset_id} in outputs — ` +
          `${created.plan.length} span(s) over ${created.report.rows.toLocaleString()} rows. ` +
          (neverTested > 0
            ? `${neverTested.toLocaleString()} row(s) are in no test span.`
            : "Every row falls in some test span."),
        kind: "ok",
      });
      notifyDatasetsChanged();
      await loadPlans(datasetId);
    } catch (cause) {
      setNotice({
        message: cause instanceof ApiError ? cause.message : String(cause),
        kind: "error",
      });
    } finally {
      setRunning(false);
    }
  };

  const empty = datasetId === undefined;

  return (
    <div className="page">
      <PageHeader
        title="Chronological Splits"
        lead={
          <>
            Describe a dataset as train, validation and test spans that never overlap in
            time. A plan is a <strong>table of row ranges</strong> — the dataset itself is
            not modified, re-written or copied, and the plan records the checksum of the
            frame it was computed on so it can never be applied to a different one.
          </>
        }
      />

      <div className="row" style={{ marginBottom: 14 }}>
        <div className="field">
          <label htmlFor="split-dataset">Dataset</label>
          <select
            id="split-dataset"
            value={datasetId ?? ""}
            onChange={(event) => {
              const next = event.target.value;
              navigate(next === "" ? "/splits" : `/splits/${next}`);
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
            Split plans themselves are not offered here — a plan is a set of row indices,
            not a price series. The dataset must be sorted by its observation time.
          </p>
        </div>
      </div>

      {empty && (
        <div className="panel muted">
          No dataset selected. A plan is computed from a stored dataset and saved to the{" "}
          <span className="mono">outputs</span> library. Nothing in the source dataset is
          touched.
        </div>
      )}

      {datasetStatus === "loading" && <div className="panel muted">Loading dataset…</div>}

      {datasetStatus === "error" && (
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

      {datasetStatus === "ready" && dataset !== null && (
        <>
          <div className="section">
            <div className="section__head">
              <h2 className="section__title">Source</h2>
              <span className="mono muted" style={{ fontSize: 11 }}>
                {dataset.dataset_id} · {dataset.row_count.toLocaleString()} rows ·{" "}
                {dataset.columns.length} column(s) ·{" "}
                {dataset.timeframe === null ? "no declared timeframe" : dataset.timeframe}
              </span>
            </div>
            <div className="panel muted" style={{ fontSize: 12 }}>
              The spans below are laid out along the dataset&apos;s{" "}
              <strong>observation time</strong> — the instant a bar opens, never the instant
              it closes. Dating a span by bar-close would move the last bar of training into
              the test span, a one-bar leak with nothing in the resulting table to show it.
            </div>
          </div>

          <div className="section">
            <div className="section__head">
              <h2 className="section__title">Split</h2>
            </div>

            <div className="panel">
              <PolicyGroup
                name="split-type"
                legend="How to cut the series"
                hint="Every plan is contiguous and ordered: no span starts before the one before it ends, and no bar is in two spans."
                options={SPLIT_TYPE_OPTIONS}
                value={splitType}
                onChange={setSplitType}
                disabled={running}
              >
                {isDateSplit(splitType) && (
                  <div className="split__dates">
                    {DATE_FIELDS.filter((field) =>
                      requiredDates.includes(field.key),
                    ).map((field) => (
                      <div className="field field--tight" key={field.key}>
                        <label htmlFor={`split-${field.key}`}>{field.label}</label>
                        <input
                          id={`split-${field.key}`}
                          type="datetime-local"
                          value={dates[field.key] ?? ""}
                          onChange={(event) =>
                            setDates((current) => ({
                              ...current,
                              [field.key]: event.target.value,
                            }))
                          }
                          disabled={running}
                        />
                      </div>
                    ))}
                    <p className="field__note">
                      Read in this browser&apos;s timezone and sent as that exact instant in
                      UTC. Both ends are inclusive: a span runs from its first bar up to and
                      including its last. Gap and purge are not offered on a date split —
                      move the date instead, so nothing shrinks a span you named.
                    </p>
                  </div>
                )}

                {showsDerivedControls && (
                  <div className="split__derived">
                    {isWalkForward(splitType) ? (
                      <div className="field field--tight">
                        <label htmlFor="split-folds">
                          Folds <span className="mono">{nFolds}</span>
                        </label>
                        <input
                          id="split-folds"
                          type="range"
                          min={2}
                          max={10}
                          step={1}
                          value={nFolds}
                          onChange={(event) => setNFolds(Number(event.target.value))}
                          disabled={running}
                        />
                        <p className="field__note">
                          The series is cut into {nFolds + 1} equal blocks; fold <em>k</em>{" "}
                          tests block <em>k</em>. The first block is never tested — it is
                          the training data every fold starts from.
                        </p>
                      </div>
                    ) : (
                      <div className="field field--tight">
                        <label>Proportions</label>
                        <p className="mono muted" style={{ fontSize: 12 }}>
                          {PROPORTIONS[splitType] ?? ""}
                        </p>
                        <p className="field__note">
                          The cut rows are computed from this dataset&apos;s row count and
                          reported below — the counts are the engine&apos;s, not this
                          page&apos;s estimate of them.
                        </p>
                      </div>
                    )}

                    <div className="field field--tight">
                      <label htmlFor="split-gap">
                        Embargo <span className="mono">{gapSize}</span> bar(s)
                      </label>
                      <input
                        id="split-gap"
                        type="range"
                        min={0}
                        max={20}
                        step={1}
                        value={gapSize}
                        onChange={(event) => setGapSize(Number(event.target.value))}
                        disabled={running}
                      />
                      <p className="field__note">
                        Bars dropped between one span&apos;s end and the next span&apos;s
                        start. A feature computed over a trailing window at the end of
                        training still reaches into the first bars of the test span without
                        one.
                      </p>
                    </div>

                    <div className="field field--tight">
                      <label className="feature__keep" htmlFor="split-purge">
                        <input
                          id="split-purge"
                          type="checkbox"
                          checked={purge}
                          onChange={(event) => setPurge(event.target.checked)}
                          disabled={running}
                        />
                        <span>Purge training rows whose label reaches past the cut</span>
                      </label>
                      {purge && (
                        <div className="field field--tight">
                          <label htmlFor="split-horizon">Label horizon (bars)</label>
                          <input
                            id="split-horizon"
                            type="number"
                            min={1}
                            value={horizon}
                            onChange={(event) => setHorizon(event.target.value)}
                            disabled={running}
                          />
                        </div>
                      )}
                      <p className="field__note">
                        A label with a forward horizon reads bars in the next span, so the
                        last rows of training are not really training data. The embargo and
                        the purge <strong>overlap rather than add up</strong>: the training
                        span ends at the cut minus whichever is larger, and the report below
                        states how many rows each rule actually removed.
                      </p>
                    </div>
                  </div>
                )}
              </PolicyGroup>

              <div className="field" style={{ marginTop: 12 }}>
                <label htmlFor="split-name">Plan name</label>
                <input
                  id="split-name"
                  type="text"
                  placeholder="(defaults to the source name plus the split type)"
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
                  disabled={running || !canRun}
                >
                  {running ? "Generating…" : "Generate Split Plan"}
                </button>
                <span className="actions__spacer" />
                <span className="mono muted" style={{ fontSize: 11 }}>
                  {plans.length} existing plan(s) for this dataset
                </span>
              </div>

              {problems.length > 0 && (
                <div className="result__bar result__bar--error">
                  <span>the plan cannot be generated yet:</span>
                  {problems.map((problem) => (
                    <div key={problem}>· {problem}</div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* ============================================================ result */}
          {result !== null && (
            <div className="section">
              <div className="section__head">
                <h2 className="section__title">Plan · {result.report.split_type}</h2>
                <span className="mono muted" style={{ fontSize: 11 }}>
                  {result.report.rows.toLocaleString()} rows · {result.report.folds} fold(s)
                  · {result.report.rows_tested.toLocaleString()} tested ·{" "}
                  {result.report.rows_never_tested.toLocaleString()} never tested
                </span>
              </div>

              <Timeline rows={result.plan} totalRows={result.report.rows} />

              <div className="section__head" style={{ marginTop: 16 }}>
                <h3 className="section__title" style={{ fontSize: 13 }}>
                  Spans
                </h3>
              </div>
              <table className="dtable split__table">
                <thead>
                  <tr>
                    <th>Fold</th>
                    <th>Role</th>
                    <th>Start</th>
                    <th>End</th>
                    <th>Rows</th>
                  </tr>
                </thead>
                <tbody>
                  {result.plan.map((row) => (
                    <tr key={`${row.fold}-${row.role}-${row.start_index}`}>
                      <td className="mono">{row.fold}</td>
                      <td>
                        <span className={`split__role split__role--${row.role}`}>
                          {row.role}
                        </span>
                      </td>
                      <td className="mono">{formatTimestamp(row.start_time)}</td>
                      <td className="mono">{formatTimestamp(row.end_time)}</td>
                      <td className="mono">
                        {row.row_count.toLocaleString()}
                        <span className="muted" style={{ fontSize: 10 }}>
                          {" "}
                          [{row.start_index}–{row.end_index})
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <div className="panel" style={{ marginTop: 12 }}>
                <div className="report">
                  <div className="report__row">
                    <span className="report__key">Rows per role</span>
                    <span className="report__value mono">
                      {Object.entries(result.report.rows_by_role)
                        .map(([role, count]) => `${role} ${count.toLocaleString()}`)
                        .join(" · ")}
                    </span>
                  </div>
                  <div className="report__row">
                    <span className="report__key">Time column</span>
                    <span className="report__value mono">{result.report.time_column}</span>
                  </div>
                  <div className="report__row">
                    <span className="report__key">Source checksum</span>
                    <span className="report__value mono" style={{ fontSize: 11 }}>
                      {result.report.dataset_checksum.slice(0, 24)}…
                    </span>
                  </div>
                </div>

                {result.report.boundaries.length > 0 && (
                  <>
                    <div className="section__head" style={{ marginTop: 14 }}>
                      <h3 className="section__title" style={{ fontSize: 12 }}>
                        Boundaries — what each rule actually removed
                      </h3>
                    </div>
                    <table className="dtable">
                      <thead>
                        <tr>
                          <th>Fold</th>
                          <th>Cut at row</th>
                          <th>Embargo removed</th>
                          <th>Purge removed</th>
                          <th>Training ends at</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.report.boundaries.map((boundary) => (
                          <tr key={`${boundary.fold}-${boundary.role}`}>
                            <td className="mono">{boundary.fold}</td>
                            <td className="mono">{boundary.cut_index}</td>
                            <td className="mono">{boundary.gap_rows_removed}</td>
                            <td className="mono">{boundary.purge_rows_removed}</td>
                            <td className="mono">{boundary.effective_end_index}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </>
                )}

                {result.report.warnings.length > 0 && (
                  <div className="result__bar" style={{ marginTop: 12, marginBottom: 0 }}>
                    <span>what this plan means:</span>
                    {result.report.warnings.map((warning, index) => (
                      <div key={`${index}-${warning.slice(0, 20)}`}>· {warning}</div>
                    ))}
                  </div>
                )}
              </div>

              <div className="actions">
                <span className="mono muted" style={{ fontSize: 11 }}>
                  {result.dataset.dataset_id}
                </span>
                <span className="actions__spacer" />
                <Link
                  className="btn btn--ghost btn--sm"
                  to={`/data/${result.dataset.dataset_id}`}
                >
                  open the plan table
                </Link>
                <Link className="btn btn--ghost btn--sm" to={`/scaling/${datasetId ?? ""}`}>
                  scale against this plan
                </Link>
              </div>
            </div>
          )}

          {/* ==================================================== existing plans */}
          <div className="section">
            <div className="section__head">
              <h2 className="section__title">Existing plans ({plans.length})</h2>
            </div>

            {plans.length === 0 ? (
              <div className="panel muted">
                No plan has been made from this dataset yet. C7 Scaling refuses to fit
                min-max, standard or robust until one exists — fitting on the whole series
                would carry the test span&apos;s statistics into every training row.
              </div>
            ) : (
              <table className="dtable">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Type</th>
                    <th>Folds</th>
                    {/* The plan's own row count — one row per span — not the dataset's. The
                        Spans table above uses "Rows" for the bars inside a span, and one
                        word for two quantities on one page is how a reader ends up quoting
                        the wrong one. */}
                    <th>Spans</th>
                    <th>Created</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {plans.map((plan) => (
                    <tr key={plan.split_plan_id}>
                      <td>{plan.name}</td>
                      <td className="mono">{plan.split_type}</td>
                      <td className="mono">{plan.folds}</td>
                      <td className="mono">{plan.rows.toLocaleString()}</td>
                      <td className="mono">{formatTimestamp(plan.created_at)}</td>
                      <td>
                        <Link
                          className="btn btn--ghost btn--sm"
                          to={`/data/${plan.split_plan_id}`}
                        >
                          inspect
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </>
      )}
    </div>
  );
}
