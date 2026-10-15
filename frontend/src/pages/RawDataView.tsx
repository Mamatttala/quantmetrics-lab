import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

import { ApiError, api } from "@/api/client";
import { subscribeToDatasets } from "@/api/datasetsChanged";
import type { DatasetSummary, OhlcvResponse, QualityResponse } from "@/api/types";
import { QUALITY_LABELS, QUALITY_ORDER } from "@/api/types";
import { Candles } from "@/charts/Candles";
import { PageHeader } from "@/components/PageHeader";

function formatTime(seconds: number | null): string {
  if (seconds === null) return "— no timestamp —";
  return new Date(seconds * 1000).toISOString().replace("T", " ").slice(0, 19) + "Z";
}

/** Everything that is not an extreme return. Kept as one list so the "structural
 *  anomalies" toggle and its count cannot drift apart. */
const ANOMALY_KINDS = [
  "duplicate_timestamp",
  "missing_bar",
  "non_positive_price",
  "high_below_low",
] as const;

/** How many markers the list renders. The panel can hold tens of thousands, so the
 *  remainder is counted rather than drawn. */
const MARKER_LIST_LIMIT = 300;

/**
 * Raw Data View (A2).
 *
 * A TradingView-style read of one dataset: candlesticks, volume, crosshair readout,
 * zoom/pan, and a fullscreen mode — plus optional overlays for structural anomalies
 * and extreme returns.
 *
 * Both overlays are computed by the **backend**. The spec is explicit that the
 * frontend must not implement statistical mathematics, and a robust z-score is
 * exactly the sort of thing that has to come out identical on every machine.
 * Overlays are off by default: a chart with markers on it invites the reading that
 * the marked bars are the interesting ones, which is a conclusion the user should
 * reach rather than be handed.
 */
export function RawDataView() {
  const { datasetId } = useParams<{ datasetId?: string }>();
  const navigate = useNavigate();

  const [candidates, setCandidates] = useState<DatasetSummary[]>([]);
  const [ohlcv, setOhlcv] = useState<OhlcvResponse | null>(null);
  const [quality, setQuality] = useState<QualityResponse | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const [showAnomalies, setShowAnomalies] = useState(false);
  const [showExtremes, setShowExtremes] = useState(false);

  // Every dataset is offered, including ones this page cannot draw. Filtering here
  // would need a per-dataset metadata fetch, and the backend already refuses a
  // non-OHLC frame with a precise reason ("missing open, high, low, close") that is
  // more useful than a silently shortened list.
  const loadCandidates = useCallback(async () => {
    try {
      const rows = await api.listDatasets();
      setCandidates(rows.filter((d) => d.row_count > 0));
    } catch {
      // The main load below reports the failure; a second identical banner would
      // just be noise.
      setCandidates([]);
    }
  }, []);

  useEffect(() => {
    void loadCandidates();
  }, [loadCandidates]);

  // Built once on mount, so a dataset fetched or uploaded elsewhere afterwards would
  // never appear in the picker — including one the user was just sent here to look at.
  useEffect(() => subscribeToDatasets(() => void loadCandidates()), [loadCandidates]);

  useEffect(() => {
    if (datasetId === undefined) {
      setOhlcv(null);
      setQuality(null);
      setStatus("idle");
      return;
    }
    let cancelled = false;
    setStatus("loading");
    setError(null);

    Promise.all([api.ohlcv(datasetId), api.quality(datasetId)])
      .then(([bars, report]) => {
        if (cancelled) return;
        setOhlcv(bars);
        setQuality(report);
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

  const selectedMarkers = useMemo(() => {
    if (quality === null) return [];
    return quality.markers.filter((marker) => {
      if (marker.kind === "extreme_return") return showExtremes;
      return showAnomalies;
    });
  }, [quality, showAnomalies, showExtremes]);

  // The markers are computed over the whole dataset while the chart holds at most the
  // most recent 20,000 bars, so a marker can name a bar that is not loaded. It has
  // nowhere to sit on the axis, and counting it in "N marker(s) shown" would report
  // something the user cannot see. Both are cut to the loaded window together, and the
  // difference is stated rather than dropped silently.
  const chartWindow = useMemo(() => {
    const first = ohlcv?.bars[0];
    const last = ohlcv === null ? undefined : ohlcv.bars[ohlcv.bars.length - 1];
    if (first === undefined || last === undefined) return null;
    return { from: first.time, to: last.time };
  }, [ohlcv]);

  const visibleMarkers = useMemo(() => {
    if (chartWindow === null) return [];
    return selectedMarkers.filter(
      (marker) =>
        marker.time !== null &&
        marker.time >= chartWindow.from &&
        marker.time <= chartWindow.to,
    );
  }, [selectedMarkers, chartWindow]);

  const offWindowMarkers = selectedMarkers.length - visibleMarkers.length;

  const anomalyCount = useMemo(
    () =>
      quality === null
        ? 0
        : ANOMALY_KINDS.reduce((total, kind) => total + (quality.counts[kind] ?? 0), 0),
    [quality],
  );

  const extremeCount = quality?.counts["extreme_return"] ?? 0;

  // The backend caps each kind at 200 markers, so the list can hold fewer entries than
  // the counts add up to. Comparing the two says exactly whether anything was dropped;
  // a fixed threshold would report truncation on a dataset that had none, and miss it
  // on one whose markers were spread thinly across several kinds.
  const listedMarkers = quality?.markers.length ?? 0;
  const countedMarkers = useMemo(
    () => Object.values(quality?.counts ?? {}).reduce((total, count) => total + count, 0),
    [quality],
  );
  const markersTruncated = listedMarkers < countedMarkers;

  return (
    <div className="page">
      <PageHeader
        title="Raw Data View"
        lead={
          <>
            Candlesticks and volume with a crosshair, zoom and pan. Structural anomalies
            and <strong>extreme returns</strong> are computed on the backend and overlaid
            on request.
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
              navigate(next === "" ? "/view" : `/view/${next}`);
            }}
          >
            <option value="">— choose a dataset —</option>
            {candidates.map((dataset) => (
              <option key={dataset.dataset_id} value={dataset.dataset_id}>
                {dataset.name} · {dataset.category} · {dataset.row_count.toLocaleString()} rows
              </option>
            ))}
          </select>
        </div>
      </div>

      {status === "error" && (
        <div className="result__bar result__bar--error">
          <span>{error}</span>
        </div>
      )}

      {status === "idle" && (
        <div className="panel muted">
          No dataset selected. Pick one above, or click a dataset in the Data Explorer
          and open it here.
        </div>
      )}

      {status === "loading" && <div className="panel muted">Loading bars…</div>}

      {status === "ready" && ohlcv !== null && quality !== null && (
        <>
          {ohlcv.truncated && (
            <div className="result__bar">
              <span>
                showing the most recent {ohlcv.bars.length.toLocaleString()} of{" "}
                {ohlcv.total.toLocaleString()} bars
              </span>
            </div>
          )}

          <div className="toggle-row">
            <label className="chip">
              <input
                type="checkbox"
                checked={showAnomalies}
                onChange={(e) => setShowAnomalies(e.target.checked)}
              />
              Structural anomalies
              <span className="mono muted">{anomalyCount}</span>
            </label>
            <label className="chip">
              <input
                type="checkbox"
                checked={showExtremes}
                onChange={(e) => setShowExtremes(e.target.checked)}
              />
              Extreme returns
              <span className="mono muted">{extremeCount}</span>
            </label>
            <span className="actions__spacer" />
            <span className="mono muted" style={{ fontSize: 11 }}>
              drag to pan · scroll to zoom · {visibleMarkers.length} marker(s) on the
              chart
              {offWindowMarkers > 0 && (
                <> · {offWindowMarkers} outside the loaded window</>
              )}
            </span>
            {datasetId !== undefined && (
              <button
                type="button"
                className="btn btn--ghost btn--sm"
                onClick={() =>
                  window.open(`/api/datasets/${datasetId}/export?format=csv`)
                }
              >
                export CSV
              </button>
            )}
          </div>

          <Candles bars={ohlcv.bars} markers={visibleMarkers} />

          {/* ------------------------------------------------------- quality */}
          <div className="section">
            <div className="section__head">
              <h2 className="section__title">Data quality</h2>
              <span className="mono muted" style={{ fontSize: 11 }}>
                time column: {quality.time_column || "none recognised"}
              </span>
            </div>

            <div className="panel">
              <div className="meta">
                {QUALITY_ORDER.map((kind) => {
                  const count = quality.counts[kind];
                  return (
                    <div className="meta__item" key={kind}>
                      <div className="meta__key">{QUALITY_LABELS[kind]}</div>
                      <div
                        className="meta__value"
                        style={{ color: count === undefined ? "var(--muted)" : count > 0 ? "var(--warn)" : "var(--ok)" }}
                      >
                        {count === undefined ? "not checked" : count.toLocaleString()}
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* "Not checked" and "clean" are different claims. Saying which checks
                  could not run is the difference between the two. */}
              {quality.skipped.length > 0 && (
                <div className="result__bar" style={{ marginBottom: 0 }}>
                  <span>not checked:</span>
                  {quality.skipped.map((reason) => (
                    <div key={reason}>· {reason}</div>
                  ))}
                </div>
              )}

              {quality.notes.length > 0 && (
                <div className="result__bar" style={{ marginBottom: 0 }}>
                  <span>also found:</span>
                  {quality.notes.map((note) => (
                    <div key={note}>· {note}</div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {quality.markers.length > 0 && (
            <div className="section">
              <div className="section__head">
                <h2 className="section__title">
                  Markers ({quality.markers.length.toLocaleString()}
                  {markersTruncated ? " of " + countedMarkers.toLocaleString() : ""})
                </h2>
              </div>
              <div className="panel markers">
                {quality.markers.slice(0, MARKER_LIST_LIMIT).map((marker, index) => (
                  <div className="markers__row" key={`${marker.time}-${marker.kind}-${index}`}>
                    <span>{formatTime(marker.time)}</span>
                    <span className={`markers__kind markers__kind--${marker.kind}`}>
                      {QUALITY_LABELS[marker.kind] ?? marker.kind}
                    </span>                    <span className="muted">{marker.detail}</span>
                  </div>
                ))}
                {/* Two different truncations, counted separately. The rows actually
                    rendered are bounded by the list cap; the number of markers that
                    exist at all is bounded by the backend's per-kind cap. Reporting
                    one as the other claimed entries that were never sent. */}
                {listedMarkers > MARKER_LIST_LIMIT && (
                  <div className="muted" style={{ padding: 6 }}>
                    … {(listedMarkers - MARKER_LIST_LIMIT).toLocaleString()} more in
                    this list not shown.
                  </div>
                )}
                {markersTruncated && (
                  <div className="muted" style={{ padding: 6 }}>
                    {(countedMarkers - listedMarkers).toLocaleString()} further marker(s)
                    were counted but not sent — the backend caps each kind. Export the
                    dataset if you need the full set.
                  </div>
                )}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
