import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import { ApiError, api } from "@/api/client";
import { notifyDatasetsChanged } from "@/api/datasetsChanged";
import type { ColumnOp, DatasetMeta, PreviewResponse } from "@/api/types";
import { DataGrid, type GridColumn, type PendingEdit } from "@/components/DataGrid";
import { HeadTailTable } from "@/components/HeadTailTable";
import { PageHeader } from "@/components/PageHeader";

function formatStamp(value: string | null): string {
  if (value === null) return "—";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toISOString().replace("T", " ");
}

interface Notice {
  message: string;
  kind: "ok" | "error";
}

/**
 * Data Management (DM).
 *
 * Two views of the same dataset: a JupyterLab-style head/tail preview, which is the
 * quick "does this look right?" check, and the full editable grid below it.
 *
 * Edits never touch the dataset being viewed. Saving posts them to `/derive`, which
 * writes a **new** `processed_data` dataset whose `parent_dataset_id` points back at
 * this one, and the page then navigates to that child. That is the spec's
 * immutability rule made visible: you can see the fork happen rather than being told
 * about it.
 */
export function DataManagement() {
  const { datasetId } = useParams<{ datasetId?: string }>();
  const navigate = useNavigate();

  const [meta, setMeta] = useState<DatasetMeta | null>(null);
  const [preview, setPreview] = useState<PreviewResponse | null>(null);
  const [columns, setColumns] = useState<GridColumn[]>([]);
  const [pendingEdits, setPendingEdits] = useState<Record<string, unknown>>({});
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<Notice | null>(null);
  const [saving, setSaving] = useState(false);

  // The id `save` navigated to. Loading a *different* dataset starts a clean slate, so
  // the notice is cleared — but the load that immediately follows a save is of the
  // child that save just created, and clearing there erased the confirmation before it
  // could be read. The ref distinguishes the two without suppressing the reset itself.
  const savedTo = useRef<string | null>(null);

  const originalNames = useMemo(
    () => (meta === null ? [] : meta.columns.map((c) => c.name)),
    [meta],
  );

  useEffect(() => {
    if (datasetId === undefined) {
      setMeta(null);
      setPreview(null);
      setColumns([]);
      setStatus("idle");
      return;
    }
    let cancelled = false;
    setStatus("loading");
    setError(null);

    Promise.all([api.getDataset(datasetId), api.preview(datasetId, 5, 5)])
      .then(([metaResponse, previewResponse]) => {
        if (cancelled) return;
        setMeta(metaResponse);
        setPreview(previewResponse);
        // Fresh column model per dataset. Carrying one across datasets would apply
        // a rename meant for one file to an unrelated one.
        setColumns(
          metaResponse.columns.map((c) => ({
            name: c.name,
            origin: c.name,
            dtype: c.dtype,
            hidden: false,
            isNew: false,
          })),
        );
        setPendingEdits({});
        if (savedTo.current !== datasetId) setNotice(null);
        setStatus("ready");
      })
      .catch((cause: unknown) => {
        if (cancelled) return;
        setError(cause instanceof ApiError ? cause.message : String(cause));
        setStatus("error");
      });

    return () => {
      cancelled = true;
    };
  }, [datasetId]);

  const noticeHandler = useCallback((message: string, kind: "ok" | "error") => {
    setNotice({ message, kind });
  }, []);

  // ------------------------------------------------------------------- edits

  const applyEdit = useCallback((edit: PendingEdit) => {
    setPendingEdits((current) => ({ ...current, [`${edit.row}|${edit.column}`]: edit.value }));
  }, []);

  const applyBulkEdit = useCallback((edits: PendingEdit[]) => {
    setPendingEdits((current) => {
      const next = { ...current };
      for (const edit of edits) next[`${edit.row}|${edit.column}`] = edit.value;
      return next;
    });
  }, []);

  // -------------------------------------------------------------- column ops

  const renameColumn = useCallback(
    (from: string, to: string) => {
      if (columns.some((c) => c.name === to)) {
        setNotice({ message: `A column named "${to}" already exists.`, kind: "error" });
        return;
      }
      // Pending edits are keyed by column name, and the backend applies column
      // operations before cell edits. Re-keying here keeps an edit attached to the
      // column it was made on rather than to the name that column used to have.
      //
      // The decision to rename is made against the state in scope rather than inside
      // the updater: an updater must be pure, and StrictMode invokes it twice.
      setPendingEdits((edits) => {
        const next: Record<string, unknown> = {};
        for (const [key, value] of Object.entries(edits)) {
          const split = key.indexOf("|");
          const row = key.slice(0, split);
          const column = key.slice(split + 1);
          next[`${row}|${column === from ? to : column}`] = value;
        }
        return next;
      });
      setColumns((current) => current.map((c) => (c.name === from ? { ...c, name: to } : c)));
      setNotice({ message: `Renamed "${from}" to "${to}".`, kind: "ok" });
    },
    [columns],
  );

  const dropColumn = useCallback(
    (column: string) => {
      if (columns.length <= 1) {
        setNotice({ message: "Cannot drop the last remaining column.", kind: "error" });
        return;
      }
      setPendingEdits((edits) => {
        const next: Record<string, unknown> = {};
        for (const [key, value] of Object.entries(edits)) {
          if (!key.endsWith(`|${column}`)) next[key] = value;
        }
        return next;
      });
      setColumns((current) => current.filter((c) => c.name !== column));
      setNotice({ message: `Dropped "${column}".`, kind: "ok" });
    },
    [columns],
  );

  const hideColumn = useCallback((column: string) => {
    setColumns((current) =>
      current.map((c) => (c.name === column ? { ...c, hidden: !c.hidden } : c)),
    );
  }, []);

  const addColumn = useCallback(() => {
    // The prompt is asked outside the updater. React invokes a state updater twice
    // under StrictMode and may re-run it later, so a side effect in there would open
    // the dialog more than once and let the discarded run's answer win.
    const proposed = window.prompt("Name for the new column:", "new_column");
    if (proposed === null) return;
    const name = proposed.trim();
    if (name === "") return;
    if (columns.some((c) => c.name === name)) {
      setNotice({ message: `A column named "${name}" already exists.`, kind: "error" });
      return;
    }
    // `origin: null` is what marks this as a column that exists only in the derived
    // dataset — it has no counterpart in the parent to rename or drop.
    setColumns((current) => [
      ...current,
      { name, origin: null, dtype: "object", hidden: false, isNew: true },
    ]);
    setNotice({ message: `Added empty column "${name}". Fill it in the grid.`, kind: "ok" });
  }, [columns]);

  const reorderColumns = useCallback((order: string[]) => {
    setColumns((current) => {
      const byName = new Map(current.map((c) => [c.name, c]));
      const ordered = order
        .map((name) => byName.get(name))
        .filter((c): c is GridColumn => c !== undefined);
      // Anything the grid did not mention is appended rather than lost.
      const seen = new Set(ordered.map((c) => c.name));
      return [...ordered, ...current.filter((c) => !seen.has(c.name))];
    });
  }, []);

  /**
   * Column operations as a diff against the parent's schema, not as a log of clicks.
   *
   * A log would replay every intermediate state — rename `a`→`b` then `b`→`c` sends
   * both, and a column renamed and renamed back sends a no-op pair. A diff sends the
   * minimum set that turns the parent into what is on screen, which is also what
   * makes the recorded lineage meaningful.
   */
  const columnOps = useMemo<ColumnOp[]>(() => {
    if (meta === null) return [];
    const renames: ColumnOp[] = [];
    for (const column of columns) {
      if (column.origin !== null && column.name !== column.origin) {
        renames.push({ op: "rename", column: column.origin, new_name: column.name });
      }
    }
    const dropped: ColumnOp[] = originalNames
      .filter((name) => !columns.some((c) => c.origin === name))
      .map((name) => ({ op: "drop" as const, column: name }));
    const added: ColumnOp[] = columns
      .filter((c) => c.origin === null)
      .map((c) => ({ op: "add" as const, column: c.name, new_name: c.name }));

    // Reordering is a diff too, and the baseline is what the first three phases would
    // produce on their own: survivors in the parent's order, under their new names,
    // with the added columns appended. Sending a `reorder` for every column *only*
    // when the screen disagrees with that baseline is what keeps a mere click on a
    // header from marking the dataset dirty — and it is why the positions can be a
    // total order rather than a partial one the backend would have to guess at.
    const finalOrder = columns.map((c) => c.name);
    const baseline: string[] = [];
    for (const origin of originalNames) {
      const current = columns.find((c) => c.origin === origin);
      if (current !== undefined) baseline.push(current.name);
    }
    for (const c of columns) if (c.origin === null) baseline.push(c.name);

    const inParentOrder = finalOrder.every((name, index) => name === baseline[index]);
    const reorder: ColumnOp[] = inParentOrder
      ? []
      : finalOrder.map((column, position) => ({ op: "reorder" as const, column, position }));

    // Order matters: drops free names, renames reoccupy them, adds append, and
    // reorders place columns in the schema those three produced.
    return [...renames, ...dropped, ...added, ...reorder];
  }, [meta, columns, originalNames]);

  const dirtyCount = Object.keys(pendingEdits).length;
  const hasChanges = dirtyCount > 0 || columnOps.length > 0;

  // -------------------------------------------------------------------- save

  const save = async () => {
    if (meta === null) return;
    if (!hasChanges) {
      setNotice({ message: "Nothing to save.", kind: "error" });
      return;
    }
    const childName = window.prompt(
      "Name for the derived dataset:",
      `${meta.name} (edited)`,
    );
    if (childName === null || childName.trim() === "") return;

    setSaving(true);
    try {
      const edits = Object.entries(pendingEdits).map(([key, value]) => {
        const split = key.indexOf("|");
        return {
          row: Number(key.slice(0, split)),
          column: key.slice(split + 1),
          value,
        };
      });
      const child = await api.deriveDataset(meta.dataset_id, {
        name: childName.trim(),
        operation: "edit",
        edits,
        column_ops: columnOps,
      });
      setNotice({
        message: `Saved as ${child.dataset_id} — ${child.row_count.toLocaleString()} rows in processed_data.`,
        kind: "ok",
      });
      // The child is a new catalog row, so the explorer's list is now stale.
      notifyDatasetsChanged();
      // Navigate to the child. Staying on the parent would leave the grid showing
      // edits that have already been committed elsewhere, and a second Save would
      // fork the parent again rather than update the copy.
      savedTo.current = child.dataset_id;
      navigate(`/data/${child.dataset_id}`);
    } catch (cause) {
      setNotice({
        message: cause instanceof ApiError ? cause.message : String(cause),
        kind: "error",
      });
    } finally {
      setSaving(false);
    }
  };

  const discard = () => {
    if (meta === null) return;
    setColumns(
      meta.columns.map((c) => ({
        name: c.name,
        origin: c.name,
        dtype: c.dtype,
        hidden: false,
        isNew: false,
      })),
    );
    setPendingEdits({});
    setNotice({ message: "Discarded unsaved changes.", kind: "ok" });
  };

  // ------------------------------------------------------------------ render

  if (datasetId === undefined) {
    return (
      <div className="page">
        <PageHeader
          title="Data Management"
          lead={
            <>
              Select a dataset in the <strong>Data Explorer</strong> to preview and edit
              it. Raw data is immutable — saving an edit writes a new{" "}
              <strong>processed_data</strong> dataset.
            </>
          }
        />
        <div className="panel muted">
          No dataset selected. Choose one on the left, or fetch one from Raw Data Fetch
          (A1).
        </div>
      </div>
    );
  }

  if (status === "error") {
    return (
      <div className="page">
        <PageHeader title="Data Management" lead="Could not load this dataset." />
        <div className="result__bar result__bar--error">
          <span>{error}</span>
        </div>
      </div>
    );
  }

  if (status !== "ready" || meta === null || preview === null) {
    return (
      <div className="page">
        <PageHeader title="Data Management" lead="Loading dataset…" />
        <div className="panel muted">Loading…</div>
      </div>
    );
  }

  const unpopulated = new Set(meta.unpopulated_fields);

  return (
    <div className="page">
      <PageHeader
        title={meta.name}
        lead={
          <>
            <strong>{meta.dataset_id}</strong> · {meta.category} ·{" "}
            {meta.row_count.toLocaleString()} rows ·{" "}
            {meta.parent_dataset_id === null ? (
              <>immutable source</>
            ) : (
              <>
                derived from <strong>{meta.parent_dataset_id}</strong>
              </>
            )}
          </>
        }
      />

      {notice !== null && (
        <div
          className={
            notice.kind === "ok" ? "result__bar result__bar--ok" : "result__bar result__bar--error"
          }
        >
          <span>{notice.message}</span>
        </div>
      )}

      {/* ------------------------------------------------------------ metadata */}
      <div className="section">
        <div className="section__head">
          <h2 className="section__title">Dataset metadata</h2>
        </div>
        <div className="panel">
          <div className="meta">
            {(
              [
                ["Source", meta.source],
                ["Asset class", meta.asset_class],
                ["Symbol", meta.symbol ?? "—"],
                ["Market type", meta.market_type ?? "—"],
                ["Timeframe", meta.timeframe ?? "—"],
                ["Timezone", meta.timezone ?? "not stated"],
                ["Start", formatStamp(meta.start_timestamp)],
                ["End", formatStamp(meta.end_timestamp)],
                ["Columns", String(meta.columns.length)],
                ["Created", formatStamp(meta.created_at)],
                ["Parent", meta.parent_dataset_id ?? "—"],
                ["Checksum", `${meta.checksum.slice(0, 24)}…`],
              ] as const
            ).map(([key, value]) => (
              <div className="meta__item" key={key}>
                <div className="meta__key">{key}</div>
                <div className="meta__value" title={String(value)}>
                  {value}
                </div>
              </div>
            ))}
          </div>

          {meta.unpopulated_fields.length > 0 && (
            <div className="meta__item" style={{ marginTop: 12 }}>
              <div className="meta__key">Declared but unpopulated</div>
              <div className="meta__value meta__value--unpopulated">
                {meta.unpopulated_fields.join(", ")}
                <span className="badge">null</span>
              </div>
            </div>
          )}

          <div className="meta__item" style={{ marginTop: 12 }}>
            <div className="meta__key">Storage</div>
            <div className="meta__value" title={meta.storage_path}>
              {meta.storage_path}
            </div>
          </div>
        </div>
      </div>

      {/* ------------------------------------------------------------- preview */}
      <div className="section">
        <div className="section__head">
          <h2 className="section__title">Data Preview</h2>
        </div>
        <HeadTailTable
          columns={preview.columns}
          head={preview.head}
          tail={preview.tail}
          rowCount={preview.row_count}
        />
      </div>

      {/* ---------------------------------------------------------------- grid */}
      <div className="section">
        <div className="section__head">
          <h2 className="section__title">Edit grid</h2>
          <span className="mono muted" style={{ fontSize: 11 }}>
            {hasChanges
              ? `${dirtyCount} cell edit(s), ${columnOps.length} column operation(s) pending`
              : "no pending changes"}
          </span>
        </div>

        <div className="actions">
          <button type="button" className="btn" onClick={() => void save()} disabled={saving}>
            {saving ? "Saving…" : "Save as new dataset"}
          </button>
          <button
            type="button"
            className="btn btn--ghost"
            onClick={discard}
            disabled={!hasChanges || saving}
          >
            Discard changes
          </button>
          <button type="button" className="btn btn--ghost btn--sm" onClick={addColumn}>
            Add column
          </button>

          {columns.some((c) => c.hidden) && (
            <button
              type="button"
              className="btn btn--ghost btn--sm"
              onClick={() =>
                setColumns((current) => current.map((c) => ({ ...c, hidden: false })))
              }
            >
              Show all columns ({columns.filter((c) => c.hidden).length} hidden)
            </button>
          )}

          <span className="actions__spacer" />

          <span className="mono muted" style={{ fontSize: 11 }}>
            export
          </span>
          {(["csv", "json", "parquet"] as const).map((format) => (
            <button
              key={format}
              type="button"
              className="btn btn--ghost btn--sm"
              onClick={() =>
                window.open(`/api/datasets/${meta.dataset_id}/export?format=${format}`)
              }
            >
              {format === "csv" ? "Export .csv" : format}
            </button>
          ))}
        </div>

        <div className="panel muted" style={{ marginBottom: 10, fontSize: 12 }}>
          Right-click a column header for rename, remove, select all, cut, copy, and
          paste. Drag a header to reorder — the order you leave the columns in is the
          order that gets saved. Click a cell then shift-click to extend a selection, then use{" "}
          <span className="mono">Ctrl+C</span> / <span className="mono">Ctrl+X</span> /{" "}
          <span className="mono">Ctrl+V</span>. Sorting is applied by the server across
          the whole dataset, not just the page on screen. Columns marked{" "}
          <span className="mono">italic violet</span> were added here and do not exist in
          the parent.
          {unpopulated.size > 0 && (
            <>
              {" "}
              This dataset declares {unpopulated.size} column(s) the provider leaves
              null: {[...unpopulated].join(", ")}.
            </>
          )}
        </div>

        <DataGrid
          datasetId={meta.dataset_id}
          columns={columns}
          pendingEdits={pendingEdits}
          onEdit={applyEdit}
          onBulkEdit={applyBulkEdit}
          onRenameColumn={renameColumn}
          onDropColumn={dropColumn}
          onHideColumn={hideColumn}
          onAddColumn={addColumn}
          onReorderColumns={reorderColumns}
          onNotice={noticeHandler}
        />
      </div>
    </div>
  );
}
