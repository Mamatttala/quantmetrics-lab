import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { useLocation, useNavigate } from "react-router-dom";

import { ApiError, api } from "@/api/client";
import { notifyDatasetsChanged, subscribeToDatasets } from "@/api/datasetsChanged";
import { getActiveProjectId } from "@/api/projectSelection";
import { useResizablePane } from "./useResizablePane";
import { Description } from "@/components/Description";
import {
  CATEGORY_LABELS,
  CATEGORY_ORDER,
  OPERATION_LABELS,
  type DatasetCategory,
  type DatasetSummary,
  type Project,
} from "@/api/types";

interface DataExplorerProps {
  project: Project | null;
  projectControls: ReactNode;
  onHide: () => void;
}

/**
 * Data Explorer — the tree pane between the sidebar and the main content.
 *
 * A dark-navy tree of the catalog, grouped by category. Clicking a dataset opens it
 * in Data Management; rename and delete remain available beside each dataset.
 */
export function DataExplorer({ project, projectControls, onHide }: DataExplorerProps) {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [error, setError] = useState<string | null>(null);
  // Whether the failure was transport or application. Both land in `error`, but they
  // are not the same claim: a rejected upload — a non-CSV, a header-only file — means
  // the backend answered, and announcing "Backend unreachable" over that message sends
  // the user to check a server that is running fine.
  const [unreachable, setUnreachable] = useState(false);
  const { width, resizeHandleProps } = useResizablePane("qca.ui.explorerWidth", 280);
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

  const refresh = useCallback(async () => {
    if (!project) return;
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
  }, [project]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // The shell mounts this pane once, so a fetch or derive performed in the main pane
  // would leave the list stale until Refresh was pressed. This pane's own mutations
  // (upload, per-row mutations, delete all) refresh directly and do not need the signal.
  useEffect(() => subscribeToDatasets(() => void refresh()), [refresh]);

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

  const handleDeleteAll = async () => {
    if (datasets.length === 0) return;
    const confirmed = window.confirm(
      `Delete all ${datasets.length} dataset(s) in "${project?.name}"?\n\n` +
        `This removes this project's datasets and their Parquet files. This cannot be undone.`,
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
      if (getActiveProjectId() !== project?.project_id) return;
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

  const grouped = CATEGORY_ORDER.map((category: DatasetCategory) => {
    const rows = datasets.filter((d) => d.category === category);
    const operations = new Map<string, DatasetSummary[]>();
    for (const row of rows) {
      const operation =
        row.operation ??
        (category === "raw_data" ? "fetch" : category === "uploaded_data" ? "upload" : "table");
      operations.set(operation, [...(operations.get(operation) ?? []), row]);
    }
    return { category, rows, operations: [...operations.entries()] };
  });

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
      {projectControls}
      <div className="explorer__header">
        <span className="explorer__title">Data Explorer</span>
        <div className="explorer__toolbar">
          <button
            type="button"
            className="explorer__btn"
            onClick={() => fileInput.current?.click()}
            disabled={!project}
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
            disabled={!project || datasets.length === 0 || status === "loading"}
            title="Remove every dataset in the selected project"
          >
            Delete
          </button>
          <button
            type="button"
            className="explorer__btn"
            onClick={() => void refresh()}
            disabled={!project}
            title="Reload the dataset list"
          >
            Refresh
          </button>
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
        {project && <details className="explorer__project" open>
          <summary className="explorer__project-name" title={project.name}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
              strokeWidth="1.6" aria-hidden="true">
              <path d="M3 7V5h6l2 2h10v12H3Z" />
            </svg>
            <span>{project.name}</span>
            <span className="explorer__count">{datasets.length}</span>
          </summary>
        {status === "loading" && <div className="explorer__empty" role="status">Loading datasets…</div>}
        {status === "error" && (
          <div className="explorer__empty">
            {unreachable ? "Backend unreachable." : "That did not work."}
            <br />
            {error}
          </div>
        )}

        {status === "ready" && datasets.length === 0 && (
          <div className="explorer__empty">
            No datasets in this project yet.
            <Description anchor=".explorer__empty">
              Upload a CSV or fetch data from Raw Data Fetch (A1).
            </Description>
          </div>
        )}

        {grouped.map(({ category, rows, operations }) =>
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
                  {operations.map(([operation, operationRows]) => (
                    <div className="explorer__operation" key={operation}>
                      <div className="explorer__operation-label">
                        <span>{OPERATION_LABELS[operation] ?? operation.replaceAll("_", " ")}</span>
                        <span className="explorer__count">{operationRows.length}</span>
                      </div>
                  {operationRows.map((dataset) => (
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
                  ))}
                </div>
              </div>
            </div>
          ),
        )}
        </details>}
      </div>

      <div
        {...resizeHandleProps}
        className="pane-resize explorer__resize"
        aria-label="Resize Data Explorer"
      />
    </aside>
  );
}
