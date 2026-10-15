import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { ApiError, api } from "@/api/client";
import { notifyDatasetsChanged, subscribeToDatasets } from "@/api/datasetsChanged";
import type {
  AlignMethod,
  ColumnMeta,
  ColumnOp,
  DatasetMeta,
  DatasetSummary,
  HeadTailPreview,
  MergeHow,
  MergeResult,
  Timeframe,
} from "@/api/types";
import { ALIGN_OPTIONS, HOW_OPTIONS, TIMEFRAMES } from "@/api/types";
import { HeadTailTable } from "@/components/HeadTailTable";
import { PageHeader } from "@/components/PageHeader";
import { PolicyGroup } from "@/components/PolicyGroup";

interface Notice {
  message: string;
  kind: "ok" | "error";
}

/** One dataset to align onto the base. `key` is a client-side handle only — it is
 *  never sent, and exists so removing the middle row of four does not re-key the
 *  others and reset their inputs. */
interface JoinRow {
  key: number;
  datasetId: string;
  method: AlignMethod;
  how: MergeHow;
  prefix: string;
  /** Seconds, as typed. Held as a string because an empty box means "no limit" and
   *  `Number("")` is 0, which would silently mean "no tolerance at all". */
  tolerance: string;
}

/** One column of the merged output, as the user is choosing to keep it. `origin` is
 *  the name the merge produced and never changes, so a rename is always expressed as
 *  a diff against the merge rather than against the previous rename. */
interface ColumnEdit {
  origin: string;
  name: string;
  dropped: boolean;
}

function newJoinRow(key: number): JoinRow {
  return {
    key,
    datasetId: "",
    // Most recent value at or before the bar is the only alignment that cannot
    // introduce look-ahead, so it is the default.
    method: "backward",
    how: "left",
    prefix: "",
    tolerance: "",
  };
}

/**
 * Resample & Merge (B4).
 *
 * Three things happen in one pass, in this order: the base is aggregated up to the
 * target timeframe, every other dataset is aligned onto the base's timestamps, and
 * the whole thing is joined into one frame. The order matters and is not
 * interchangeable — aligning first and resampling second would attach a value to a
 * bar that the aggregation then moves.
 *
 * **Alignment is as-of, never as-of-reversed.** Each bar takes the latest value
 * published *at or before* its own timestamp. The alternative — nearest, or forward —
 * reaches into the future, and a funding rate attached to the bar before it was
 * published is a look-ahead that leaves no trace in the resulting frame. A staleness
 * cap is offered per join for the same reason: an un-capped as-of match will happily
 * attach a six-month-old funding rate to today's bar and call it aligned.
 *
 * **The column edits are a diff, applied on save.** Removing or renaming a column is
 * a presentation change to already-computed rows — no statistic is involved — so the
 * table updates immediately without a round trip. On save the diff is sent with the
 * request, `_apply_column_ops` applies it to the merged frame on the server, and the
 * response's own preview of the saved dataset is then shown below so the two can be
 * compared rather than assumed to agree.
 */
export function ResampleMerge() {
  const { datasetId } = useParams<{ datasetId?: string }>();
  const navigate = useNavigate();

  const [candidates, setCandidates] = useState<DatasetSummary[]>([]);
  const [base, setBase] = useState<DatasetMeta | null>(null);
  const [baseStatus, setBaseStatus] = useState<"idle" | "loading" | "ready" | "error">(
    "idle",
  );
  const [loadError, setLoadError] = useState<string | null>(null);

  const [timeframe, setTimeframe] = useState<Timeframe | "">("");
  const [outName, setOutName] = useState("");
  const [joins, setJoins] = useState<JoinRow[]>([]);
  const [nextKey, setNextKey] = useState(1);

  const [result, setResult] = useState<MergeResult | null>(null);
  const [editor, setEditor] = useState<ColumnEdit[]>([]);
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

  useEffect(() => subscribeToDatasets(() => void loadCandidates()), [loadCandidates]);

  useEffect(() => {
    if (datasetId === undefined) {
      setBase(null);
      setBaseStatus("idle");
      return;
    }
    let cancelled = false;
    setBaseStatus("loading");
    setLoadError(null);

    api
      .getDataset(datasetId)
      .then((meta) => {
        if (cancelled) return;
        setBase(meta);
        setBaseStatus("ready");
      })
      .catch((cause: unknown) => {
        if (cancelled) return;
        setLoadError(cause instanceof ApiError ? cause.message : String(cause));
        setBaseStatus("error");
      });

    return () => {
      cancelled = true;
    };
  }, [datasetId]);

  // A merged result belongs to the base it was computed from. Keeping it across a
  // change of base would leave a table on screen that is not the table these controls
  // describe.
  useEffect(() => {
    setResult(null);
    setEditor([]);
    setNotice(null);
    setTimeframe("");
    setOutName("");
    setJoins([]);
  }, [datasetId]);

  // -------------------------------------------------------------------- joins

  const addJoin = useCallback(() => {
    setJoins((current) => [...current, newJoinRow(nextKey)]);
    setNextKey((key) => key + 1);
  }, [nextKey]);

  const updateJoin = useCallback((key: number, patch: Partial<JoinRow>) => {
    setJoins((current) =>
      current.map((row) => (row.key === key ? { ...row, ...patch } : row)),
    );
  }, []);

  const removeJoin = useCallback((key: number) => {
    setJoins((current) => current.filter((row) => row.key !== key));
  }, []);

  /** Datasets offered in a join row. The base is excluded because the backend refuses
   *  a self-merge, and offering a choice that is always rejected is worse than not
   *  offering it. */
  const joinCandidates = useMemo(
    () => candidates.filter((dataset) => dataset.dataset_id !== datasetId),
    [candidates, datasetId],
  );

  const problems = useMemo<string[]>(() => {
    const found: string[] = [];
    const chosen = joins.filter((row) => row.datasetId !== "").map((row) => row.datasetId);
    const duplicates = chosen.filter((id, index) => chosen.indexOf(id) !== index);
    if (duplicates.length > 0) {
      found.push(
        `The same dataset appears in more than one join (${[...new Set(duplicates)].join(", ")}). ` +
          "Its columns would arrive twice; give each its own copy or remove one.",
      );
    }
    if (joins.some((row) => row.datasetId === "")) {
      found.push("Every join row needs a dataset selected, or it must be removed.");
    }
    for (const row of joins) {
      if (row.datasetId === "") continue;
      if (row.tolerance !== "") {
        const seconds = Number(row.tolerance);
        if (!Number.isFinite(seconds) || seconds < 1) {
          found.push(
            `A staleness cap must be a whole number of seconds of at least 1 (got “${row.tolerance}”).`,
          );
        }
      }
      if (row.prefix !== "" && !/^[A-Za-z0-9_]+$/.test(row.prefix)) {
        found.push(
          `The prefix “${row.prefix}” must be letters, digits, or underscores — it is prepended to a column name.`,
        );
      }
    }
    // Saying *why* the buttons are disabled. Without this the first-run screen is a
    // timeframe selector and two dead buttons, with nothing to explain the connection.
    if (timeframe === "" && joins.length === 0) {
      found.push("Choose a timeframe to resample to, or add a dataset to merge in.");
    }
    return found;
  }, [joins, timeframe]);

  // A join is not required. Resampling to a different timeframe is the first half of
  // what this step is for, so "pick a timeframe and press Preview" has to work on its
  // own — the backend accepts it, and a UI that disabled the button until a dataset was
  // added would make a supported operation unreachable. The "nothing to do" case is
  // reported through `problems`, so this needs no separate clause.
  const canRun = problems.length === 0;

  // ----------------------------------------------------------- column editor

  /**
   * Rebuild the editor only when the merged schema actually changed.
   *
   * Rebuilding on every preview would silently discard the renames the user has
   * already typed, because a second preview of the same plan returns the same
   * columns. Comparing the origin list is what distinguishes "the merge produced
   * something else" from "the same merge, run again".
   *
   * Held in a ref rather than derived from `result` in an effect: the signature is
   * written by the same handler that sets `result`, so the two cannot be observed out
   * of step by the render in between.
   */
  const schemaRef = useRef<string>("");

  const keptColumns = useMemo(() => editor.filter((edit) => !edit.dropped), [editor]);

  const editorProblem = useMemo<string | null>(() => {
    if (editor.length === 0) return null;
    if (keptColumns.length === 0) {
      return "Every column is marked for removal. At least one must remain.";
    }
    const names = keptColumns.map((edit) => edit.name.trim());
    if (names.some((name) => name === "")) {
      return "A renamed column cannot be empty.";
    }
    const duplicate = names.find((name, index) => names.indexOf(name) !== index);
    if (duplicate !== undefined) {
      return `Two columns would be called “${duplicate}”. Names must be unique — a duplicate makes column selection ambiguous.`;
    }
    return null;
  }, [editor, keptColumns]);

  /** The edits, expressed as a diff against the merge. Drops run before renames on the
   *  server, and each phase is applied as a set, so the order within the array is not
   *  meaningful. */
  const columnOps = useMemo<ColumnOp[]>(() => {
    const ops: ColumnOp[] = [];
    for (const edit of editor) {
      if (edit.dropped) {
        ops.push({ op: "drop", column: edit.origin });
      } else if (edit.name.trim() !== edit.origin) {
        ops.push({ op: "rename", column: edit.origin, new_name: edit.name.trim() });
      }
    }
    return ops;
  }, [editor]);

  /** The merged preview as the current column edits would leave it: a header renamed
   *  and a column dropped, over rows the server already returned. No value is
   *  computed here, which is why this can be done in the browser at all. */
  const projected = useMemo<HeadTailPreview | null>(() => {
    if (result === null) return null;
    const byName = new Map(result.preview.columns.map((column) => [column.name, column]));
    const columns: ColumnMeta[] = keptColumns.map((edit) => {
      const source = byName.get(edit.origin);
      return {
        name: edit.name.trim(),
        dtype: source?.dtype ?? "object",
        nullable: source?.nullable ?? true,
      };
    });
    const project = (rows: Record<string, unknown>[]): Record<string, unknown>[] =>
      rows.map((row) =>
        Object.fromEntries(keptColumns.map((edit) => [edit.name.trim(), row[edit.origin]])),
      );
    return {
      row_count: result.preview.row_count,
      columns,
      head: project(result.preview.head),
      tail: project(result.preview.tail),
    };
  }, [result, keptColumns]);

  // -------------------------------------------------------------------- run

  const run = async (write: boolean) => {
    if (datasetId === undefined || base === null || !canRun) return;
    if (write && editorProblem !== null) {
      setNotice({ message: editorProblem, kind: "error" });
      return;
    }
    setRunning(write ? "save" : "preview");
    setNotice(null);
    try {
      const response = await api.mergeDatasets({
        base_dataset_id: datasetId,
        name: outName.trim() === "" ? null : outName.trim(),
        timeframe: timeframe === "" ? null : timeframe,
        joins: joins.map((row) => ({
          dataset_id: row.datasetId,
          how: row.how,
          method: row.method,
          prefix: row.prefix,
          tolerance_seconds: row.tolerance === "" ? null : Number(row.tolerance),
        })),
        // A preview always describes the *merge*, with no column edits applied, so the
        // editor's baseline is the merge's own output however many times it is re-run.
        column_ops: write ? columnOps : [],
        write,
      });
      setResult(response);
      // The editor's baseline is the merge's own schema. Rebuilt only when that schema
      // changes, so a re-preview does not throw away renames already typed.
      const signature = response.preview.columns.map((column) => column.name).join("\u0000");
      if (signature !== schemaRef.current) {
        schemaRef.current = signature;
        setEditor(
          response.preview.columns.map((column) => ({
            origin: column.name,
            name: column.name,
            dropped: false,
          })),
        );
      }
      if (write && response.dataset !== null) {
        const child = response.dataset;
        setNotice({
          message:
            `Saved as ${child.dataset_id} — ${child.row_count.toLocaleString()} rows, ` +
            `${child.columns.length} column(s), in processed_data. Every input is ` +
            "recorded as a parent, so the merged file's lineage names each source.",
          kind: "ok",
        });
        notifyDatasetsChanged();
      } else if (!write) {
        setNotice({
          message:
            "Preview only — nothing was stored. Removing or renaming a column below " +
            "changes the table but is only applied when you save.",
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

  const reset = () => {
    setTimeframe("");
    setOutName("");
    setJoins([]);
    setResult(null);
    setEditor([]);
    setNotice({ message: "Plan cleared.", kind: "ok" });
  };

  const empty = datasetId === undefined;

  return (
    <div className="page">
      <PageHeader
        title="Resample & Merge"
        lead={
          <>
            Aggregate the base up to a coarser timeframe, align every other dataset onto
            its timestamps, and join the result into one file. Alignment is{" "}
            <strong>as-of</strong> — a bar only ever sees values published at or before
            it.
          </>
        }
      />

      <div className="row" style={{ marginBottom: 14 }}>
        <div className="field">
          <label htmlFor="base">Base dataset</label>
          <select
            id="base"
            value={datasetId ?? ""}
            onChange={(e) => {
              const next = e.target.value;
              navigate(next === "" ? "/resample" : `/resample/${next}`);
            }}
          >
            <option value="">— choose a dataset —</option>
            {candidates.map((dataset) => (
              <option key={dataset.dataset_id} value={dataset.dataset_id}>
                {dataset.name} · {dataset.category} · {dataset.row_count.toLocaleString()}{" "}
                rows{dataset.timeframe === null ? "" : ` · ${dataset.timeframe}`}
              </option>
            ))}
          </select>
        </div>
      </div>

      {empty && (
        <div className="panel muted">
          No base dataset selected. The base supplies the timeline every other dataset is
          aligned onto, so it must have a recognisable timestamp column. Pick one above,
          or click a dataset in the Data Explorer.
        </div>
      )}

      {baseStatus === "loading" && <div className="panel muted">Loading base dataset…</div>}

      {baseStatus === "error" && (
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

      {baseStatus === "ready" && base !== null && (
        <>
          {/* ====================================================== base + target */}
          <div className="section">
            <div className="section__head">
              <h2 className="section__title">Base</h2>
              <span className="mono muted" style={{ fontSize: 11 }}>
                {base.dataset_id} · {base.row_count.toLocaleString()} rows ·{" "}
                {base.columns.length} column(s)
              </span>
            </div>

            <div className="panel">
              <div className="row">
                <div className="field">
                  <label htmlFor="timeframe">Target timeframe</label>
                  <select
                    id="timeframe"
                    value={timeframe}
                    onChange={(e) => setTimeframe(e.target.value as Timeframe | "")}
                  >
                    <option value="">— no resampling —</option>
                    {TIMEFRAMES.map((value) => (
                      <option key={value} value={value}>
                        {value}
                      </option>
                    ))}
                  </select>
                  <p className="field__note">
                    {base.timeframe === null
                      ? "This dataset declares no timeframe, so the API cannot check the target against the source interval — it will warn that a finer target would fabricate bars."
                      : `This dataset declares ${base.timeframe}. A coarser target aggregates its bars; a finer one is refused, because building four 15-minute bars from one hourly bar invents price paths no market produced.`}
                  </p>
                </div>

                <div className="field">
                  <label htmlFor="merge-name">Output dataset name</label>
                  <input
                    id="merge-name"
                    type="text"
                    placeholder="(defaults to the base name plus “merged”)"
                    value={outName}
                    onChange={(e) => setOutName(e.target.value)}
                  />
                  <p className="field__note">
                    Aggregation is fixed by column name: <span className="mono">open</span>{" "}
                    first, <span className="mono">high</span> max,{" "}
                    <span className="mono">low</span> min, <span className="mono">close</span>{" "}
                    last, <span className="mono">volume</span> summed, and any other
                    column takes its <em>last</em> value in the interval — a level such as
                    open interest or a funding rate is observed at an instant, and
                    averaging it reports a value that was never observed.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* ============================================================== joins */}
          <div className="section">
            <div className="section__head">
              <h2 className="section__title">
                Datasets to align and merge ({joins.length})
              </h2>
            </div>

            {joins.length === 0 && (
              <div className="panel muted">
                No dataset selected. Add one below — funding rates, open interest, or
                metrics for the same symbol — and it will be aligned onto the base&apos;s
                timeline. Each one becomes a parent of the merged dataset in the lineage.
              </div>
            )}

            <div className="stack">
              {joins.map((row, index) => (
                <div className="join" key={row.key}>
                  <div className="join__head">
                    <span className="join__label">join {index + 1}</span>
                    <button
                      type="button"
                      className="btn btn--ghost btn--sm"
                      onClick={() => removeJoin(row.key)}
                    >
                      remove
                    </button>
                  </div>

                  <div className="row">
                    <div className="field">
                      <label htmlFor={`join-dataset-${row.key}`}>Dataset</label>
                      <select
                        id={`join-dataset-${row.key}`}
                        value={row.datasetId}
                        onChange={(e) =>
                          updateJoin(row.key, { datasetId: e.target.value })
                        }
                      >
                        <option value="">— choose a dataset —</option>
                        {joinCandidates.map((dataset) => (
                          <option key={dataset.dataset_id} value={dataset.dataset_id}>
                            {dataset.name} · {dataset.category} ·{" "}
                            {dataset.row_count.toLocaleString()} rows
                          </option>
                        ))}
                      </select>
                    </div>

                    <div className="field field--tight">
                      <label htmlFor={`join-prefix-${row.key}`}>Prefix</label>
                      <input
                        id={`join-prefix-${row.key}`}
                        type="text"
                        placeholder="none"
                        value={row.prefix}
                        onChange={(e) => updateJoin(row.key, { prefix: e.target.value })}
                      />
                    </div>

                    <div className="field field--tight">
                      <label htmlFor={`join-tolerance-${row.key}`}>
                        Staleness cap (s)
                      </label>
                      <input
                        id={`join-tolerance-${row.key}`}
                        type="number"
                        min={1}
                        step={1}
                        placeholder="none"
                        value={row.tolerance}
                        onChange={(e) =>
                          updateJoin(row.key, { tolerance: e.target.value })
                        }
                      />
                      <p className="field__note">
                        Leave empty to accept a value of any age. A funding rate published
                        six months ago would then still be attached to today&apos;s bar.
                      </p>
                    </div>
                  </div>

                  <div className="join__options">
                    <PolicyGroup
                      name={`method-${row.key}`}
                      legend="Alignment"
                      options={ALIGN_OPTIONS}
                      value={row.method}
                      onChange={(method) => updateJoin(row.key, { method })}
                    />
                    <PolicyGroup
                      name={`how-${row.key}`}
                      legend="Match"
                      options={HOW_OPTIONS}
                      value={row.how}
                      onChange={(how) => updateJoin(row.key, { how })}
                    />
                  </div>
                </div>
              ))}
            </div>

            <div className="actions">
              <button
                type="button"
                className="btn btn--ghost btn--sm"
                onClick={addJoin}
                disabled={joinCandidates.length === 0}
              >
                Add dataset to merge
              </button>
              {joinCandidates.length === 0 && (
                <span className="muted" style={{ fontSize: 12 }}>
                  No other dataset is available to merge. Fetch or upload one first.
                </span>
              )}
            </div>
          </div>

          {/* ======================================================= run controls */}
          <div className="section">
            <div className="section__head">
              <h2 className="section__title">Run</h2>
            </div>

            <div className="actions">
              <button
                type="button"
                className="btn btn--ghost"
                onClick={() => void run(false)}
                disabled={running !== null || !canRun}
              >
                {running === "preview" ? "Running…" : "Preview (store nothing)"}
              </button>
              <button
                type="button"
                className="btn"
                onClick={() => void run(true)}
                disabled={running !== null || !canRun || editorProblem !== null}
              >
                {running === "save" ? "Saving…" : "Apply and save as new dataset"}
              </button>
              <button
                type="button"
                className="btn btn--ghost"
                onClick={reset}
                disabled={running !== null}
              >
                Reset plan
              </button>
            </div>

            {problems.length > 0 && (
              <div className="result__bar result__bar--error">
                <span>the plan cannot run yet:</span>
                {problems.map((problem) => (
                  <div key={problem}>· {problem}</div>
                ))}
              </div>
            )}

            <div className="panel muted" style={{ fontSize: 12 }}>
              Preview runs the whole pipeline — resample, align, and join — against the
              real frames and stores nothing. Removing or renaming a column below changes
              the table immediately but is only sent with the save, so saving writes one
              new <strong>processed_data</strong> dataset whose lineage lists every input
              as a parent, with the alignment method recorded per input.
            </div>
          </div>

          {/* ============================================================ result */}
          {result !== null && (
            <div className="section">
              <div className="section__head">
                <h2 className="section__title">
                  {result.dataset === null ? "Preview" : "Saved"} · merged output
                </h2>
                <span className="mono muted" style={{ fontSize: 11 }}>
                  {result.report.rows_before.toLocaleString()} →{" "}
                  {result.report.rows_after.toLocaleString()} rows ·{" "}
                  {columnOps.length} pending column operation(s)
                </span>
              </div>

              <div className="panel">
                <div className="report">
                  <div className="report__row">
                    <span className="report__key">Base</span>
                    <span className="report__value">
                      {result.report.base.name}{" "}
                      <span className="mono muted">{result.report.base.dataset_id}</span>
                    </span>
                  </div>
                  <div className="report__row">
                    <span className="report__key">Rows in</span>
                    <span className="report__value">
                      {result.report.rows_before.toLocaleString()}
                    </span>
                  </div>
                  <div className="report__row">
                    <span className="report__key">Rows out</span>
                    <span className="report__value">
                      {result.report.rows_after.toLocaleString()}
                    </span>
                  </div>
                </div>

                {result.report.resample !== null && (
                  <>
                    <div className="section__head" style={{ marginTop: 14 }}>
                      <h3 className="section__title" style={{ fontSize: 12 }}>
                        Resampling
                      </h3>
                    </div>
                    <div className="report">
                      <div className="report__row">
                        <span className="report__key">Target</span>
                        <span className="report__value mono">
                          {result.report.resample.timeframe}
                        </span>
                      </div>
                      <div className="report__row">
                        <span className="report__key">Rows</span>
                        <span className="report__value">
                          {result.report.resample.rows_before.toLocaleString()} →{" "}
                          {result.report.resample.rows_after.toLocaleString()}
                        </span>
                      </div>
                      {result.report.resample.detail !== undefined && (
                        <div className="report__row">
                          <span className="report__key">Note</span>
                          <span className="report__value">
                            {result.report.resample.detail}
                          </span>
                        </div>
                      )}
                    </div>
                    {/* The per-column aggregation is shown rather than described: it is
                        the whole difference between a resampled bar and an invented
                        one. */}
                    <div className="report__row" style={{ marginTop: 8 }}>
                      <span className="report__key">Aggregation</span>
                      <span className="report__value mono" style={{ fontSize: 11 }}>
                        {Object.entries(result.report.resample.aggregations)
                          .map(([column, rule]) => `${column}:${rule}`)
                          .join("  ") || "—"}
                      </span>
                    </div>
                  </>
                )}

                <div className="section__head" style={{ marginTop: 14 }}>
                  <h3 className="section__title" style={{ fontSize: 12 }}>
                    Joins ({result.report.joins.length})
                  </h3>
                </div>
                <table className="dtable">
                  <thead>
                    <tr>
                      <th>Dataset</th>
                      <th>Alignment</th>
                      <th>Match</th>
                      <th>Prefix</th>
                      <th>Cap</th>
                      <th>Unmatched</th>
                      <th>Columns added</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.report.joins.map((join, index) => (
                      <tr key={`${join.dataset_id}-${index}`}>
                        <td className="mono" title={join.dataset_id}>
                          {join.label}
                        </td>
                        <td className="mono muted">{join.method}</td>
                        <td className="mono muted">{join.how}</td>
                        <td className="mono muted">{join.prefix || "—"}</td>
                        <td className="mono muted">
                          {join.tolerance_seconds === null
                            ? "none"
                            : `${join.tolerance_seconds}s`}
                        </td>
                        <td
                          className="mono"
                          style={{ color: join.unmatched > 0 ? "var(--warn)" : undefined }}
                        >
                          {join.unmatched.toLocaleString()}
                        </td>
                        <td className="mono muted" style={{ fontSize: 11 }}>
                          {join.columns_added.join(", ") || "none"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>

                {result.report.joins.some((join) => join.detail !== undefined) && (
                  <div className="report" style={{ marginTop: 8 }}>
                    {result.report.joins.map((join, index) =>
                      join.detail === undefined ? null : (
                        <div className="report__row" key={`detail-${index}`}>
                          <span className="report__key">{join.label}</span>
                          <span className="report__value">{join.detail}</span>
                        </div>
                      ),
                    )}
                  </div>
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

              {/* ------------------------------------------------- column edits */}
              <div className="section__head" style={{ marginTop: 18 }}>
                <h3 className="section__title" style={{ fontSize: 13 }}>
                  Columns ({keptColumns.length} of {editor.length} kept)
                </h3>
                <span className="mono muted" style={{ fontSize: 11 }}>
                  applied when you save
                </span>
              </div>

              <div className="panel">
                <table className="dtable">
                  <thead>
                    <tr>
                      <th>Keep</th>
                      <th>Merged name</th>
                      <th>Output name</th>
                      <th>Type</th>
                    </tr>
                  </thead>
                  <tbody>
                    {editor.map((edit) => {
                      const source = result.preview.columns.find(
                        (column) => column.name === edit.origin,
                      );
                      return (
                        <tr key={edit.origin}>
                          <td>
                            <input
                              type="checkbox"
                              checked={!edit.dropped}
                              aria-label={`keep ${edit.origin}`}
                              onChange={(e) =>
                                setEditor((current) =>
                                  current.map((item) =>
                                    item.origin === edit.origin
                                      ? { ...item, dropped: !e.target.checked }
                                      : item,
                                  ),
                                )
                              }
                            />
                          </td>
                          <td className="mono muted">{edit.origin}</td>
                          <td>
                            <input
                              type="text"
                              value={edit.name}
                              disabled={edit.dropped}
                              aria-label={`name for ${edit.origin}`}
                              onChange={(e) =>
                                setEditor((current) =>
                                  current.map((item) =>
                                    item.origin === edit.origin
                                      ? { ...item, name: e.target.value }
                                      : item,
                                  ),
                                )
                              }
                            />
                          </td>
                          <td className="mono muted">{source?.dtype ?? "—"}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>

                {editorProblem !== null && (
                  <div className="result__bar result__bar--error" style={{ marginBottom: 0 }}>
                    <span>{editorProblem}</span>
                  </div>
                )}
              </div>

              {/* --------------------------------------------------- output table */}
              <div className="section__head" style={{ marginTop: 18 }}>
                <h3 className="section__title" style={{ fontSize: 13 }}>
                  {keptColumns.length} column(s), first 5 and last 5 of{" "}
                  {result.preview.row_count.toLocaleString()} rows
                  {result.dataset === null ? " · not stored" : ""}
                </h3>
              </div>

              {result.dataset === null && projected !== null ? (
                <HeadTailTable
                  columns={projected.columns}
                  head={projected.head}
                  tail={projected.tail}
                  rowCount={projected.row_count}
                />
              ) : (
                // After a save the table is the *saved dataset's* own preview, produced
                // by the server with the column operations applied. Showing it rather
                // than the local projection is what lets the two be compared: if the
                // rename did not land, this is where it shows.
                <HeadTailTable
                  columns={result.preview.columns}
                  head={result.preview.head}
                  tail={result.preview.tail}
                  rowCount={result.preview.row_count}
                />
              )}

              {result.dataset !== null && (
                <div className="actions">
                  <span className="mono muted" style={{ fontSize: 11 }}>
                    {result.dataset.dataset_id}
                    {result.dataset.timeframe === null
                      ? ""
                      : ` · ${result.dataset.timeframe}`}
                  </span>
                  <span className="actions__spacer" />
                  <Link
                    className="btn btn--ghost btn--sm"
                    to={`/data/${result.dataset.dataset_id}`}
                  >
                    open in Data Management
                  </Link>
                  <Link
                    className="btn btn--ghost btn--sm"
                    to={`/view/${result.dataset.dataset_id}`}
                  >
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
