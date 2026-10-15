import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { MouseEvent as ReactMouseEvent } from "react";
import { useLocation, useNavigate } from "react-router-dom";

import { ApiError, api } from "@/api/client";
import { notifyDatasetsChanged, subscribeToDatasets } from "@/api/datasetsChanged";
import {
  CATEGORY_LABELS,
  CATEGORY_ORDER,
  type DatasetCategory,
  type DatasetSummary,
} from "@/api/types";
import { ThemeToggle } from "@/components/ThemeToggle";

interface DataExplorerProps {
  onHide: () => void;
}

/** Width limits for the drag-to-resize handle (§1: min 200, max 500). */
const MIN_WIDTH = 200;
const MAX_WIDTH = 500;
const DEFAULT_WIDTH = 280;
const WIDTH_KEY = "qca.ui.explorerWidth";

function clampWidth(value: number): number {
  return Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, value));
}

function readStoredWidth(): number {
  try {
    const raw = window.localStorage.getItem(WIDTH_KEY);
    const parsed = raw === null ? Number.NaN : Number(raw);
    return Number.isFinite(parsed) ? clampWidth(parsed) : DEFAULT_WIDTH;
  } catch {
    // localStorage throws in private windows; a layout preference is not worth a
    // blank pane, so the default is used instead.
    return DEFAULT_WIDTH;
  }
}

/**
 * Data Explorer — the tree pane between the sidebar and the main content.
 *
 * A dark-navy tree of the catalog, grouped by category. Clicking a dataset opens it
 * in Data Management; the row also keeps the small, high-frequency actions close to
 * the dataset they affect without making the row's navigation ambiguous.
 */
export function DataExplorer({ onHide }: DataExplorerProps) {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [error, setError] = useState<string | null>(null);
  // Whether the failure was transport or application. Both land in `error`, but they
  // are not the same claim: a rejected upload — a non-CSV, a header-only file — means
  // the backend answered, and announcing "Backend unreachable" over that message sends
  // the user to check a server that is running fine.
  const [unreachable, setUnreachable] = useState(false);
  const [width, setWidth] = useState<number>(readStoredWidth);
  const navigate = useNavigate();

  // `useParams` is deliberately not used here. DataExplorer renders inside the
  // pathless layout route, whose match carries no params — react-router's `useParams`
  // reads only the *last* match in the current route context, so it returns `{}` for
  // everything rendered under a layout. Only the leaf route's own element
  // (DataManagement, RawDataView) sees `:datasetId`. Reading it from the location is
  // what actually identifies the dataset open in the main pane.
  const { pathname } = useLocation();
  const openDatasetId = useMemo(() => {
    const match = /^\/(?:data|view)\/([^/]+)\/?$/.exec(pathname);
    return match?.[1] ?? null;
  }, [pathname]);
  const fileInput = useRef<HTMLInputElement>(null);
  const resize = useRef<{ startX: number; startWidth: number } | null>(null);
  // Mirrors `width` so the resize handler reads the live value without being rebuilt
  // (and re-bound) on every pointer move.
  const widthRef = useRef(width);

  const refresh = useCallback(async () => {
    setStatus("loading");
    try {
      const rows = await api.listDatasets();
      setDatasets(rows);
      setError(null);
      setUnreachable(false);
      setStatus("ready");
    } catch (cause) {
      // Distinguish "backend unreachable" from "no datasets" — an empty list
      // shown for a downed backend would read as data loss.
      setError(cause instanceof ApiError ? cause.message : String(cause));
      setUnreachable(cause instanceof ApiError && cause.code === "transport_error");
      setStatus("error");
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // The shell mounts this pane once, so a fetch or derive performed in the main pane
  // would leave the list stale until Refresh was pressed. This pane's own mutations
  // (upload, delete all) refresh directly and do not need the signal.
  useEffect(() => subscribeToDatasets(() => void refresh()), [refresh]);

  useEffect(() => {
    widthRef.current = width;
    try {
      window.localStorage.setItem(WIDTH_KEY, String(width));
    } catch {
      /* preference not persisted; not worth surfacing */
    }
  }, [width]);

  const handleRename = async (dataset: DatasetSummary) => {
    const proposed = window.prompt("New name", dataset.name);
    if (proposed === null) return;
    const newName = proposed.trim();
    if (newName === "" || newName === dataset.name) return;

    try {
      await api.renameDataset(dataset.dataset_id, newName);
      await refresh();
    } catch (cause) {
      window.alert(cause instanceof ApiError ? cause.message : String(cause));
    }
  };

  const handleDelete = async (dataset: DatasetSummary) => {
    const confirmed = window.confirm(
      `Delete "${dataset.name}"?\n\nThis removes the catalog entry and its Parquet ` +
        `file. A dataset that others were derived from cannot be deleted — delete ` +
        `those first.`,
    );
    if (!confirmed) return;

    try {
      await api.deleteDataset(dataset.dataset_id);
      if (openDatasetId === dataset.dataset_id) navigate("/data");
      await refresh();
    } catch (cause) {
      window.alert(cause instanceof ApiError ? cause.message : String(cause));
    }
  };

  const openQuickAction = (event: ReactMouseEvent<HTMLButtonElement>, path: string) => {
    event.stopPropagation();
    navigate(path);
  };

  const startResize = useCallback((event: ReactMouseEvent<HTMLDivElement>) => {
    event.preventDefault();
    resize.current = { startX: event.clientX, startWidth: widthRef.current };
    const onMove = (move: globalThis.MouseEvent) => {
      const state = resize.current;
      if (state === null) return;
      setWidth(clampWidth(state.startWidth + move.clientX - state.startX));
    };
    const onUp = () => {
      resize.current = null;
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }, []);

  const handleDeleteAll = async () => {
    if (datasets.length === 0) return;
    const confirmed = window.confirm(
      `Delete all ${datasets.length} dataset(s)?\n\nThis removes every catalog entry ` +
        `and its Parquet file from the Data Explorer. This cannot be undone.`,
    );
    if (!confirmed) return;
    setStatus("loading");
    try {
      await api.deleteAllDatasets();
      navigate("/data");
      notifyDatasetsChanged();
      await refresh();
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : String(cause));
      setUnreachable(cause instanceof ApiError && cause.code === "transport_error");
      setStatus("error");
    }
  };

  const handleUpload = async (file: File) => {
    setStatus("loading");
    try {
      const uploaded = await api.uploadCsv(file);
      await refresh();
      navigate(`/data/${uploaded.dataset_id}`);
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : String(cause));
      setUnreachable(cause instanceof ApiError && cause.code === "transport_error");
      setStatus("error");
    }
  };

  // Category groups are collapsible. The set holds categories the user has
  // folded away; default state per the redesign spec is expanded.
  const [folded, setFolded] = useState<ReadonlySet<string>>(new Set());

  const grouped = CATEGORY_ORDER.map((category: DatasetCategory) => ({
    category,
    rows: datasets.filter((d) => d.category === category),
  }));

  const toggleFold = (category: string) => {
    setFolded((prev) => {
      const next = new Set(prev);
      if (next.has(category)) {
        next.delete(category);
      } else {
        next.add(category);
      }
      return next;
    });
  };

  return (
    <aside className="explorer" style={{ width }}>
      <div className="explorer__header">
        <span className="explorer__title">Data Explorer</span>
        <div className="explorer__toolbar">
          <button
            type="button"
            className="explorer__btn"
            onClick={() => fileInput.current?.click()}
            title="Import a CSV as an immutable uploaded_data dataset"
          >
            Upload
          </button>
          <button type="button" className="explorer__btn" onClick={onHide} title="Hide the Data Explorer">
            Hide
          </button>
          <button
            type="button"
            className="explorer__btn explorer__btn--danger"
            onClick={() => void handleDeleteAll()}
            disabled={datasets.length === 0}
            title="Remove every dataset from the Data Explorer"
          >
            Delete
          </button>
          <button
            type="button"
            className="explorer__btn"
            onClick={() => void refresh()}
            title="Reload the dataset list"
          >
            Refresh
          </button>
          <ThemeToggle />
          <input
            ref={fileInput}
            type="file"
            accept=".csv"
            hidden
            onChange={(e) => {
              const file = e.target.files?.[0];
              // Clear the input so choosing the same file twice in a row still
              // fires a change event.
              e.target.value = "";
              if (file !== undefined) void handleUpload(file);
            }}
          />
        </div>
      </div>

      <div className="explorer__body">
        {status === "error" && (
          <div className="explorer__empty">
            {unreachable ? "Backend unreachable." : "That did not work."}
            <br />
            {error}
          </div>
        )}

        {status === "ready" && datasets.length === 0 && (
          <div className="explorer__empty">
            No datasets yet. Fetch one from Raw Data Fetch (A1).
          </div>
        )}

        {grouped.map(({ category, rows }) =>
          rows.length === 0 ? null : (
            <div className="explorer__group" key={category}>
              <button
                type="button"
                className="explorer__grouphead"
                aria-expanded={!folded.has(category)}
                onClick={() => toggleFold(category)}
                title={CATEGORY_LABELS[category]}
              >
                <span
                  className={
                    folded.has(category)
                      ? "explorer__chevron"
                      : "explorer__chevron explorer__chevron--open"
                  }
                  aria-hidden="true"
                >
                  ▶
                </span>
                <span className="explorer__grouplabel">{CATEGORY_LABELS[category]}</span>
                <span className="explorer__count">{rows.length}</span>
              </button>
              <div
                className={
                  folded.has(category)
                    ? "explorer__children"
                    : "explorer__children explorer__children--open"
                }
              >
                <div className="explorer__childreninner">
                  {rows.map((dataset) => (
                    <div
                      key={dataset.dataset_id}
                      className={
                        dataset.dataset_id === openDatasetId
                          ? "explorer__row explorer__row--selected"
                          : "explorer__row"
                      }
                      data-dataset-id={dataset.dataset_id}
                      onClick={() => navigate(`/data/${dataset.dataset_id}`)}
                    >
                      <button
                        type="button"
                        className={
                          dataset.dataset_id === openDatasetId
                            ? "explorer__item explorer__item--selected"
                            : "explorer__item"
                        }
                        title={`${dataset.name}\n${dataset.symbol ?? "—"} · ${
                          dataset.timeframe ?? "—"
                        }\n${dataset.row_count.toLocaleString()} rows`}
                        onClick={(event) => {
                          event.stopPropagation();
                          navigate(`/data/${dataset.dataset_id}`);
                        }}
                      >
                        {dataset.name}
                      </button>

                      <div className="explorer__actions" onClick={(event) => event.stopPropagation()}>
                        <button
                          type="button"
                          className="explorer__quick"
                          title="Open in Raw Data View (A2)"
                          onClick={(event) =>
                            openQuickAction(event, `/view/${dataset.dataset_id}`)
                          }
                        >
                          view
                        </button>
                        <button
                          type="button"
                          className="explorer__quick"
                          title="Open in Cleaning (B3)"
                          onClick={(event) =>
                            openQuickAction(event, `/cleaning/${dataset.dataset_id}`)
                          }
                        >
                          clean
                        </button>
                        <button
                          type="button"
                          className="explorer__quick"
                          title="Open as the base in Resample & Merge (B4)"
                          onClick={(event) =>
                            openQuickAction(event, `/resample/${dataset.dataset_id}`)
                          }
                        >
                          merge
                        </button>
                        <span className="explorer__action-spacer" aria-hidden="true" />
                        <button
                          type="button"
                          className="icon-btn"
                          aria-label={`Rename dataset ${dataset.name}`}
                          title="Rename dataset"
                          onClick={(event) => {
                            event.stopPropagation();
                            void handleRename(dataset);
                          }}
                        >
                          ✏️
                        </button>
                        <button
                          type="button"
                          className="icon-btn danger"
                          aria-label={`Delete dataset ${dataset.name}`}
                          title="Delete dataset"
                          onClick={(event) => {
                            event.stopPropagation();
                            void handleDelete(dataset);
                          }}
                        >
                          🗑️
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ),
        )}
      </div>

      <div
        className="explorer__resize"
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize Data Explorer"
        onMouseDown={startResize}
      />
    </aside>
  );
}
