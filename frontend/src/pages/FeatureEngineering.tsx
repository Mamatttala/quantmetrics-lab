import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { ApiError, api } from "@/api/client";
import { notifyDatasetsChanged, subscribeToDatasets } from "@/api/datasetsChanged";
import type {
  DatasetMeta,
  DatasetSummary,
  FeatureCatalog,
  FeatureGroupPayload,
  FeatureResult,
  FeatureSelection,
  FeatureSpec,
  PriceSource,
} from "@/api/types";
import { PRICE_SOURCE_OPTIONS } from "@/api/types";
import { FeaturePicker, firstWindow } from "@/components/FeaturePicker";
import { HeadTailTable } from "@/components/HeadTailTable";
import { PageHeader } from "@/components/PageHeader";
import { PolicyGroup } from "@/components/PolicyGroup";

interface Notice {
  message: string;
  kind: "ok" | "error";
}

/**
 * Features Engineering (C5).
 *
 * The page is a catalogue browser: the backend serves one entry per feature — its
 * formula, its parameters, the columns it produces, and whether it can be built from
 * *this* dataset — and the page renders that and nothing else. It publishes no formula
 * of its own. A frontend that restated the mathematics would eventually describe
 * something the engine does not do, and the user reading the info icon would have no
 * way to tell.
 *
 * **Availability is decided server-side and shown inline.** Entries that cannot be
 * computed here are disabled with the reason in place, never hidden: `hour_of_day` on
 * daily bars is a column of one repeated value, and `realized_vol` needs a bar length
 * the dataset may not declare. Both are legitimate features that this dataset cannot
 * support, which is a different thing from a feature that does not exist.
 *
 * **Nothing here drops rows.** Features are appended as new columns, so the warm-up
 * gap — the leading rows where a 20-bar average has no value yet — stays visible as
 * empty cells. Filling it would fabricate observations; deleting it here would be a
 * second, hidden way to lose the timeline, when B3 already owns that decision.
 */
export function FeatureEngineering() {
  const { datasetId } = useParams<{ datasetId?: string }>();
  const navigate = useNavigate();

  const [candidates, setCandidates] = useState<DatasetSummary[]>([]);
  const [dataset, setDataset] = useState<DatasetMeta | null>(null);
  const [datasetStatus, setDatasetStatus] = useState<
    "idle" | "loading" | "ready" | "error"
  >("idle");
  const [loadError, setLoadError] = useState<string | null>(null);

  const [catalog, setCatalog] = useState<FeatureCatalog | null>(null);
  const [catalogStatus, setCatalogStatus] = useState<"idle" | "loading" | "ready" | "error">(
    "idle",
  );
  const [catalogError, setCatalogError] = useState<string | null>(null);

  const [picked, setPicked] = useState<ReadonlySet<string>>(new Set());
  const [windows, setWindows] = useState<Record<string, string[]>>({});
  const [priceSource, setPriceSource] = useState<PriceSource>("close");
  const [keepOnlySelected, setKeepOnlySelected] = useState(false);
  const [outName, setOutName] = useState("");

  const [result, setResult] = useState<FeatureResult | null>(null);
  const [running, setRunning] = useState<"preview" | "save" | null>(null);
  const [notice, setNotice] = useState<Notice | null>(null);

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

  useEffect(() => {
    if (datasetId === undefined) {
      setCatalog(null);
      setCatalogStatus("idle");
      return;
    }
    let cancelled = false;
    setCatalogStatus("loading");
    setCatalogError(null);

    api
      .featureCatalog(datasetId)
      .then((payload) => {
        if (cancelled) return;
        setCatalog(payload);
        setCatalogStatus("ready");
      })
      .catch((cause: unknown) => {
        if (cancelled) return;
        setCatalogError(cause instanceof ApiError ? cause.message : String(cause));
        setCatalogStatus("error");
      });

    return () => {
      cancelled = true;
    };
  }, [datasetId]);

  // A selection is a set of ticked boxes against one catalogue. Carrying it across a
  // change of dataset would leave ticked features that the new dataset may not support,
  // and a result computed from a different frame than the controls describe.
  useEffect(() => {
    setPicked(new Set());
    setWindows({});
    setKeepOnlySelected(false);
    setResult(null);
    setNotice(null);
    setOutName("");
  }, [datasetId]);

  // ------------------------------------------------------------------ selection

  const specs = useMemo(() => {
    const found: FeatureSpec[] = [];
    for (const group of catalog?.groups ?? []) {
      found.push(...group.features);
    }
    return found;
  }, [catalog]);

  const specById = useMemo(
    () => new Map(specs.map((spec) => [spec.id, spec])),
    [specs],
  );

  /** The request's selections, in catalogue order so two identical runs send identical
   *  bytes — which is what makes the selection hash comparable between them. */
  const selections = useMemo<FeatureSelection[]>(() => {
    const lines: FeatureSelection[] = [];
    for (const spec of specs) {
      if (!picked.has(spec.id)) continue;
      if (spec.param === null) {
        lines.push({ feature_id: spec.id, window: null });
        continue;
      }
      for (const token of windows[spec.id] ?? []) {
        lines.push({ feature_id: spec.id, window: token });
      }
    }
    return lines;
  }, [specs, picked, windows]);

  const togglePick = useCallback(
    (spec: FeatureSpec) => {
      setPicked((current) => {
        const next = new Set(current);
        if (next.has(spec.id)) {
          next.delete(spec.id);
        } else {
          next.add(spec.id);
        }
        return next;
      });
      // Ticking a parameterised feature with no window chosen yet would produce a
      // selection the backend refuses. `firstWindow` reads the catalogue's own default
      // rather than one invented here, and falls back to the first window this dataset
      // can actually resolve, so the first tick always runs.
      if (spec.param !== null) {
        setWindows((current) => {
          if ((current[spec.id] ?? []).length > 0) return current;
          return { ...current, [spec.id]: firstWindow(spec) };
        });
      }
    },
    [],
  );

  const setFeatureWindows = useCallback((featureId: string, tokens: string[]) => {
    setWindows((current) => ({ ...current, [featureId]: tokens }));
  }, []);

  const toggleGroup = useCallback((group: FeatureGroupPayload, on: boolean) => {
    const ids = group.features.filter((spec) => spec.available);
    setPicked((current) => {
      const next = new Set(current);
      for (const spec of ids) {
        if (on) next.add(spec.id);
        else next.delete(spec.id);
      }
      return next;
    });
    setWindows((current) => {
      const next = { ...current };
      for (const spec of ids) {
        if (!on || spec.param === null) continue;
        if ((next[spec.id] ?? []).length === 0) next[spec.id] = firstWindow(spec);
      }
      return next;
    });
  }, []);

  const problems = useMemo<string[]>(() => {
    const found: string[] = [];
    if (selections.length === 0) {
      found.push("Tick at least one feature to compute.");
    }
    for (const spec of specs) {
      if (!picked.has(spec.id) || spec.param === null) continue;
      if ((windows[spec.id] ?? []).length === 0) {
        found.push(`${spec.label} takes a window — tick one, or type one in its box.`);
      }
    }
    return found;
  }, [selections.length, specs, picked, windows]);

  const canRun = problems.length === 0;

  // The columns the request will produce, so the count is on screen before the run.
  // The names are templates with the window substituted exactly as the backend does —
  // no arithmetic, only the same string substitution, which is why it is safe here.
  const plannedColumns = useMemo<string[]>(() => {
    const names: string[] = [];
    for (const line of selections) {
      const spec = specById.get(line.feature_id);
      if (spec === undefined) continue;
      for (const template of spec.columns) {
        names.push(template.replace("{arg}", line.window ?? ""));
      }
    }
    return names;
  }, [selections, specById]);

  // -------------------------------------------------------------------- run

  const run = async (write: boolean) => {
    if (datasetId === undefined || !canRun) return;
    setRunning(write ? "save" : "preview");
    setNotice(null);
    try {
      const response = await api.computeFeatures(datasetId, {
        name: outName.trim() === "" ? null : outName.trim(),
        selections,
        price_source: priceSource,
        keep_only_selected: keepOnlySelected,
        write,
      });
      setResult(response);
      if (write && response.dataset !== null) {
        const child = response.dataset;
        setNotice({
          message:
            `Saved as ${child.dataset_id} — ${child.row_count.toLocaleString()} rows, ` +
            `${child.columns.length} column(s), in processed_data. The first ` +
            `${response.report.warmup_bars} row(s) are empty for at least one feature; ` +
            "they are the warm-up, and no value exists for them.",
          kind: "ok",
        });
        notifyDatasetsChanged();
      } else if (!write) {
        setNotice({
          message:
            "Preview only — nothing was stored. The column names, the warm-up and the " +
            "collinearity scan are the ones a save would produce.",
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
    setPicked(new Set());
    setWindows({});
    setKeepOnlySelected(false);
    setOutName("");
    setResult(null);
    setNotice({ message: "Selection cleared.", kind: "ok" });
  };

  const empty = datasetId === undefined;

  return (
    <div className="page">
      <PageHeader
        title="Preprocessing & Feature Selection"
        lead={
          <>
            Configure preprocessing steps and select features for the dataset. Every
            column is <strong>point-in-time</strong>: a value at bar <em>t</em> is
            computed from bars at or before <em>t</em> and from nothing else, so
            computing the same series over a shorter history leaves every row it shares
            unchanged.
          </>
        }
      />

      <div className="row" style={{ marginBottom: 14 }}>
        <div className="field">
          <label htmlFor="feature-dataset">Dataset</label>
          <select
            id="feature-dataset"
            value={datasetId ?? ""}
            onChange={(event) => {
              const next = event.target.value;
              navigate(next === "" ? "/features" : `/features/${next}`);
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
            The catalogue is resolved against the chosen dataset: a feature that needs a
            column, a bar length, or a crypto market this dataset does not have is
            offered disabled, with the reason beside it.
          </p>
        </div>
      </div>

      {empty && (
        <div className="panel muted">
          No dataset selected. Features are computed from a stored dataset — fetch one in
          A1 or upload a file in Data Management — and the result is saved as a new{" "}
          <span className="mono">processed_data</span> child, leaving the input untouched.
        </div>
      )}

      {datasetStatus === "loading" && <div className="panel muted">Loading dataset…</div>}

      {datasetStatus === "error" && (
        <div className="result__bar result__bar--error">
          <span>{loadError}</span>
        </div>
      )}

      {catalogStatus === "error" && (
        <div className="result__bar result__bar--error">
          <span>{catalogError}</span>
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

      {datasetStatus === "ready" && dataset !== null && catalogStatus === "ready" && (
        <>
          {/* ==================================================== source + prices */}
          <div className="section">
            <div className="section__head">
              <h2 className="section__title">Source</h2>
              <span className="mono muted" style={{ fontSize: 11 }}>
                {dataset.dataset_id} · {dataset.row_count.toLocaleString()} rows ·{" "}
                {dataset.columns.length} column(s) · {dataset.asset_class}
                {dataset.timeframe === null ? " · no declared timeframe" : ` · ${dataset.timeframe}`}
              </span>
            </div>

            <div className="panel">
              <div className="row">
                <PolicyGroup
                  name="price-source"
                  legend="Price series"
                  hint="Used by every feature that needs a price. The choice is recorded in the report and in the saved dataset's lineage."
                  options={PRICE_SOURCE_OPTIONS}
                  value={priceSource}
                  onChange={setPriceSource}
                />

                <div className="field">
                  <label htmlFor="features-name">Output dataset name</label>
                  <input
                    id="features-name"
                    type="text"
                    placeholder="(defaults to the source name plus “features”)"
                    value={outName}
                    onChange={(event) => setOutName(event.target.value)}
                  />

                  <label className="feature__keep" htmlFor="keep-only">
                    <input
                      id="keep-only"
                      type="checkbox"
                      checked={keepOnlySelected}
                      onChange={(event) => setKeepOnlySelected(event.target.checked)}
                    />
                    <span>Keep only the generated columns</span>
                  </label>
                  <p className="field__note">
                    Off, the dataset&apos;s own columns are kept alongside the new ones.
                    On, the output holds the generated columns and the timestamp only —
                    which is what a model matrix usually wants, but it also means the
                    close price a return was computed from is no longer in the frame.
                    {catalog !== null && !catalog.has_time
                      ? " This dataset has no recognisable timestamp column, so nothing would be kept alongside them."
                      : ""}
                  </p>
                </div>
              </div>

              {catalog !== null && catalog.unpopulated_fields.length > 0 && (
                <div className="report__row">
                  <span className="report__key">Empty columns</span>
                  <span className="report__value">
                    {catalog.unpopulated_fields.join(", ")} — empty in every row, so any
                    feature built on them is offered disabled.
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* ========================================================= catalogue */}
          <div className="section">
            <div className="section__head">
              <h2 className="section__title">
                Features ({picked.size} selected · {plannedColumns.length} column
                {plannedColumns.length === 1 ? "" : "s"})
              </h2>
              <span className="muted" style={{ fontSize: 12 }}>
                {catalog === null
                  ? ""
                  : `${catalog.available_count} of ${catalog.total_count} available on this dataset`}
              </span>
            </div>

            {catalog?.groups.map((group) => (
              <FeaturePicker
                key={group.group}
                group={group}
                picked={picked}
                windows={windows}
                onTogglePick={togglePick}
                onSetWindows={setFeatureWindows}
                onToggleGroup={toggleGroup}
                busy={running !== null}
              />
            ))}
          </div>

          {/* ====================================================== run controls */}
          <div className="section">
            <div className="section__head">
              <h2 className="section__title">Run</h2>
            </div>

            <div className="actions feature__run">
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
                className="btn feature__runprimary"
                onClick={() => void run(true)}
                disabled={running !== null || !canRun}
              >
                {running === "save" ? "Saving…" : "Apply and save as new dataset"}
              </button>
              <button
                type="button"
                className="btn btn--ghost btn--sm"
                onClick={reset}
                disabled={running !== null}
              >
                Clear selection
              </button>
            </div>

            <div className="feature__status">
              {running !== null
                ? running === "preview"
                  ? "Running preview…"
                  : "Saving…"
                : result !== null
                  ? result.dataset === null
                    ? "Preview ready."
                    : "Saved."
                  : problems.length > 0
                    ? "Waiting for a valid selection."
                    : "Idle."}
            </div>

            {problems.length > 0 && (
              <div className="result__bar result__bar--error">
                <span>the selection cannot run yet:</span>
                {problems.map((problem) => (
                  <div key={problem}>· {problem}</div>
                ))}
              </div>
            )}

            <div className="panel muted" style={{ fontSize: 12 }}>
              Preview computes every column over the real frame and stores nothing. Saving
              writes one new <strong>processed_data</strong> dataset whose lineage records
              the source dataset, the price series, and a hash of the exact selection —
              so the same request can be reproduced from the report alone. No row is
              added or removed by either.
            </div>
          </div>

          {/* ============================================================= result */}
          {result !== null && (
            <div className="section">
              <div className="section__head">
                <h2 className="section__title">
                  {result.dataset === null ? "Preview" : "Saved"} · computed columns
                </h2>
                <span className="mono muted" style={{ fontSize: 11 }}>
                  {result.report.rows_before.toLocaleString()} →{" "}
                  {result.report.rows_after.toLocaleString()} rows ·{" "}
                  {result.report.generated_columns.length} generated ·{" "}
                  {result.report.warmup_bars} warm-up
                </span>
              </div>

              <div className="panel">
                <div className="report">
                  <div className="report__row">
                    <span className="report__key">Rows</span>
                    <span className="report__value">
                      {result.report.rows_before.toLocaleString()} in →{" "}
                      {result.report.rows_after.toLocaleString()} out
                      {result.report.rows_before === result.report.rows_after
                        ? " — no row was dropped"
                        : ""}
                    </span>
                  </div>
                  <div className="report__row">
                    <span className="report__key">Price series</span>
                    <span className="report__value">
                      {result.report.price_source.source} —{" "}
                      {result.report.price_source.definition}{" "}
                      <span className="muted">({result.report.price_source.basis})</span>
                    </span>
                  </div>
                  <div className="report__row">
                    <span className="report__key">Warm-up</span>
                    <span className="report__value">
                      {result.report.warmup_bars === 0
                        ? "no feature needs a warm-up"
                        : `the first ${result.report.warmup_bars} row(s) carry no value for at least one feature`}
                    </span>
                  </div>
                  <div className="report__row">
                    <span className="report__key">Usable from</span>
                    <span className="report__value mono">
                      {result.report.usable_from ?? "—"}
                      {result.report.usable_from !== null && (
                        <span className="muted">
                          {" "}
                          in {result.report.timezone}
                        </span>
                      )}
                    </span>
                  </div>
                  <div className="report__row">
                    <span className="report__key">Selection</span>
                    <span className="report__value mono" style={{ fontSize: 11 }}>
                      {result.report.selection_hash}
                    </span>
                  </div>
                </div>

                <div className="section__head" style={{ marginTop: 14 }}>
                  <h3 className="section__title" style={{ fontSize: 12 }}>
                    Added ({result.report.added.length})
                  </h3>
                </div>
                <table className="dtable">
                  <thead>
                    <tr>
                      <th>Feature</th>
                      <th>Window</th>
                      <th>Warm-up</th>
                      <th>Columns</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.report.added.map((entry) => (
                      <tr key={`${entry.feature_id}-${entry.window ?? ""}`}>
                        <td>{entry.label}</td>
                        <td className="mono muted">
                          {entry.window ?? "—"}
                          {entry.window_bars === null ? "" : ` (${entry.window_bars} bars)`}
                        </td>
                        <td className="mono muted">{entry.warmup_bars}</td>
                        <td className="mono" style={{ fontSize: 11 }}>
                          {entry.columns.join(", ")}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>

                {result.report.carried.length > 0 && (
                  <div className="report__row" style={{ marginTop: 8 }}>
                    <span className="report__key">Carried</span>
                    <span className="report__value mono" style={{ fontSize: 11 }}>
                      {result.report.carried.join(", ")} — kept as they arrived; nothing
                      was computed for them.
                    </span>
                  </div>
                )}

                {result.report.dropped_columns.length > 0 && (
                  <div className="report__row">
                    <span className="report__key">Dropped</span>
                    <span className="report__value mono" style={{ fontSize: 11 }}>
                      {result.report.dropped_columns.join(", ")} —{" "}
                      <span className="muted">
                        left out because only the generated columns were kept
                      </span>
                    </span>
                  </div>
                )}

                {result.report.collinearity.length > 0 && (
                  <>
                    <div className="section__head" style={{ marginTop: 14 }}>
                      <h3 className="section__title" style={{ fontSize: 12 }}>
                        Near-identical columns ({result.report.collinearity.length})
                      </h3>
                    </div>
                    <div className="result__bar" style={{ marginBottom: 8 }}>
                      <span>
                        these pairs move together with |r| ≥{" "}
                        {result.report.collinearity_threshold} across the sample shown
                        here. They are not errors — a 5-bar and a 10-bar average of the
                        same series are genuinely similar — but a model given both is
                        being given the same information twice, and any coefficient
                        attributed to one of them is attributed to the pair.
                      </span>
                    </div>
                    <table className="dtable">
                      <thead>
                        <tr>
                          <th>Column</th>
                          <th>Column</th>
                          <th>r</th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.report.collinearity.map((pair) => (
                          <tr key={`${pair.a}-${pair.b}`}>
                            <td className="mono">{pair.a}</td>
                            <td className="mono">{pair.b}</td>
                            <td className="mono">{pair.r.toFixed(4)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </>
                )}

                {result.report.notes.length > 0 && (
                  <div className="result__bar" style={{ marginBottom: 0, marginTop: 12 }}>
                    <span>what this did to the data:</span>
                    {result.report.notes.map((note, index) => (
                      <div key={`${index}-${note.slice(0, 24)}`}>· {note}</div>
                    ))}
                  </div>
                )}
              </div>

              {/* --------------------------------------------------- output table */}
              <div className="section__head" style={{ marginTop: 18 }}>
                <h3 className="section__title" style={{ fontSize: 13 }}>
                  {result.preview.columns.length} column(s), first 5 and last 5 of{" "}
                  {result.preview.row_count.toLocaleString()} rows
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
                    {result.dataset.timeframe === null ? "" : ` · ${result.dataset.timeframe}`}
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
                  <Link
                    className="btn btn--ghost btn--sm"
                    to={`/features/${result.dataset.dataset_id}`}
                  >
                    add more features
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
