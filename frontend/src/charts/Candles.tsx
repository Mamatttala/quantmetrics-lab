import {
  CandlestickSeries,
  ColorType,
  CrosshairMode,
  HistogramSeries,
  createChart,
  createSeriesMarkers,
  type CandlestickData,
  type IChartApi,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  type MouseEventParams,
  type SeriesMarker,
  type Time,
  type UTCTimestamp,
} from "lightweight-charts";
import { useEffect, useRef, useState } from "react";

import type { OhlcvBar, QualityMarker, QualityKind } from "@/api/types";
import { candleTheme, useTheme } from "@/theme";

interface MarkerStyle {
  color: string;
  shape: "circle" | "square" | "arrowUp" | "arrowDown";
  position: "aboveBar" | "belowBar";
  label: string;
}

/** Marker styling per quality kind. Colour carries the category; the shape carries
 *  the direction, so the overlay stays readable without relying on hue alone.
 *
 *  Partial, not exhaustive. Two of the checks in `QUALITY_LABELS` count columns
 *  rather than rows and can never produce a marker, so demanding a style for them
 *  would be demanding a style for something that does not exist — and the entry added
 *  to satisfy the compiler would be dead code pretending to be a mapping. The kinds
 *  that *are* listed are the ones the API emits markers for. */
const MARKER_STYLE: Partial<Record<QualityKind, MarkerStyle>> = {
  extreme_return: {
    color: "#8b5cf6",
    shape: "circle",
    position: "aboveBar",
    label: "extreme return",
  },
  non_positive_price: {
    color: "#ef4444",
    shape: "square",
    position: "belowBar",
    label: "non-positive price",
  },
  high_below_low: {
    color: "#fbbf24",
    shape: "square",
    position: "aboveBar",
    label: "high < low",
  },
  duplicate_timestamp: {
    color: "#fbbf24",
    shape: "circle",
    position: "belowBar",
    label: "duplicate timestamp",
  },
  timestamp_out_of_order: {
    color: "#fbbf24",
    shape: "arrowDown",
    position: "aboveBar",
    label: "out of order",
  },
  missing_value: {
    color: "#888888",
    shape: "square",
    position: "belowBar",
    label: "missing value",
  },
  missing_bar: {
    color: "#888888",
    shape: "arrowUp",
    position: "belowBar",
    label: "missing bar",
  },
};

/** Style for a kind this build has no entry for. The API can add a check without the
 *  bundle being rebuilt, and a marker drawn generically is strictly better than the
 *  crash a missing entry used to cause: `style.position` on `undefined` threw inside
 *  the effect, taking the whole chart with it, on exactly the datasets whose
 *  anomalies were worth looking at. */
const DEFAULT_MARKER_STYLE: MarkerStyle = {
  color: "#888888",
  shape: "circle",
  position: "aboveBar",
  label: "quality marker",
};

interface CandlesProps {
  bars: OhlcvBar[];
  markers: QualityMarker[];
  height?: number;
}

interface Readout {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

function formatPrice(value: number): string {
  // Six decimals for sub-dollar instruments, two otherwise. A fixed 2 would show a
  // $0.000012 alt-coin as "0.00" — indistinguishable from zero.
  return Math.abs(value) < 1 ? value.toPrecision(6) : value.toFixed(2);
}

function formatTime(seconds: number): string {
  return new Date(seconds * 1000).toISOString().replace("T", " ").slice(0, 19) + "Z";
}

/**
 * Candlestick + volume viewer.
 *
 * lightweight-charts wants **seconds**; the API returns bars already converted, and
 * markers come from `/quality` in the same unit. That consistency is the whole reason
 * the conversion lives in one place on the backend.
 *
 * Zoom, pan, and the crosshair are the library's own behaviours — reimplementing them
 * would be strictly worse. Fit and fullscreen are the two additions it does not ship.
 */
export function Candles({ bars, markers, height = 480 }: CandlesProps) {
  const { theme } = useTheme();
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const volumeRef = useRef<ISeriesApi<"Histogram"> | null>(null);
  const markersRef = useRef<ISeriesMarkersPluginApi<Time> | null>(null);
  const [readout, setReadout] = useState<Readout | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Bumped every time a chart instance is created. The data and marker effects key off
  // this as well as their own props, so a chart rebuilt for a reason unrelated to the
  // data — a changed `height`, a remount — is repopulated. Without it they would fire
  // only when `bars` or `markers` changed, and a rebuild would leave every series empty
  // with no further render coming to fill them: a blank chart over data that is loaded.
  const [chartEpoch, setChartEpoch] = useState(0);

  // Whether the container below is actually rendered. It is omitted entirely in the
  // empty state, so the chart-creating effect has to re-run when this flips — with
  // only `height` in its dependency list the effect would run once against a null
  // container and never run again, leaving a permanently blank chart after the user
  // moved from an empty dataset to a non-empty one.
  const hasBars = bars.length > 0;

  // ---------------------------------------------------------------- lifecycle

  useEffect(() => {
    const container = containerRef.current;
    if (container === null) return;

    const colours = candleTheme();
    const chart = createChart(container, {
      width: container.clientWidth,
      height,
      layout: {
        background: { type: ColorType.Solid, color: colours.bg },
        textColor: colours.muted,
        fontFamily: '"Inter", system-ui, sans-serif',
        fontSize: 11,
      },
      grid: {
        vertLines: { color: colours.grid },
        horzLines: { color: colours.grid },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: colours.accent, width: 1, style: 2, labelBackgroundColor: colours.label },
        horzLine: { color: colours.accent, width: 1, style: 2, labelBackgroundColor: colours.label },
      },
      rightPriceScale: { borderColor: colours.border },
      timeScale: {
        borderColor: colours.border,
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 4,
      },
      localization: {
        timeFormatter: (time: Time) =>
          typeof time === "number" ? formatTime(time) : String(time),
      },
    });

    const candles = chart.addSeries(CandlestickSeries, {
      upColor: colours.up,
      downColor: colours.down,
      borderUpColor: colours.up,
      borderDownColor: colours.down,
      wickUpColor: colours.up,
      wickDownColor: colours.down,
      priceLineColor: colours.accent,
      priceLineStyle: 2,
    });

    // The volume histogram shares the pane, pinned to the bottom fifth via scale
    // margins. A second pane would need a synced price scale for no extra insight.
    const volume = chart.addSeries(HistogramSeries, {
      priceFormat: { type: "volume" },
      priceScaleId: "volume",
      color: "rgba(136,136,136,0.5)",
    });
    chart.priceScale("volume").applyOptions({
      scaleMargins: { top: 0.82, bottom: 0 },
    });

    markersRef.current = createSeriesMarkers(candles, []);
    chartRef.current = chart;
    candleRef.current = candles;
    volumeRef.current = volume;
    setChartEpoch((epoch) => epoch + 1);

    chart.subscribeCrosshairMove((param: MouseEventParams<Time>) => {
      if (param.time === undefined || param.point === undefined) {
        setReadout(null);
        return;
      }
      const candle = param.seriesData.get(candles) as CandlestickData<Time> | undefined;
      const bar = param.seriesData.get(volume) as { value?: number } | undefined;
      if (candle === undefined) {
        setReadout(null);
        return;
      }
      setReadout({
        time: typeof param.time === "number" ? param.time : 0,
        open: candle.open,
        high: candle.high,
        low: candle.low,
        close: candle.close,
        volume: bar?.value ?? 0,
      });
    });

    const observer = new ResizeObserver(() => {
      // clientWidth is 0 while an element is hidden, and applying a 0 width makes
      // lightweight-charts throw. Skip those frames; the next visible one corrects it.
      if (container.clientWidth > 0) {
        chart.applyOptions({ width: container.clientWidth });
      }
    });
    observer.observe(container);

    const onFullscreenChange = () => {
      const active = document.fullscreenElement === container;
      setIsFullscreen(active);
      // Give the browser a frame to lay the element out at its new size before
      // measuring, or the chart resizes to the old dimensions.
      requestAnimationFrame(() => {
        if (container.clientWidth > 0) {
          chart.applyOptions({
            width: container.clientWidth,
            height: active ? window.innerHeight : height,
          });
        }
      });
    };
    document.addEventListener("fullscreenchange", onFullscreenChange);

    return () => {
      document.removeEventListener("fullscreenchange", onFullscreenChange);
      observer.disconnect();
      chart.remove();
      chartRef.current = null;
      candleRef.current = null;
      volumeRef.current = null;
      markersRef.current = null;
    };
  }, [height, hasBars, theme]);

  // -------------------------------------------------------------------- data

  useEffect(() => {
    const candles = candleRef.current;
    const volume = volumeRef.current;
    if (candles === null || volume === null) return;

    candles.setData(
      bars.map((bar) => ({
        time: bar.time as UTCTimestamp,
        open: bar.open,
        high: bar.high,
        low: bar.low,
        close: bar.close,
      })),
    );
    volume.setData(
      bars.map((bar) => ({
        time: bar.time as UTCTimestamp,
        value: bar.volume,
        // Volume bars take the candle's direction so a glance at the lower band
        // tells you whether the move was bought or sold.
        color:
          bar.close >= bar.open ? "rgba(74,222,128,0.42)" : "rgba(239,68,68,0.42)",
      })),
    );
    chartRef.current?.timeScale().fitContent();
  }, [bars, chartEpoch]);

  useEffect(() => {
    const plugin = markersRef.current;
    if (plugin === null) return;
    // Markers whose timestamp could not be parsed have nowhere to sit on the time
    // axis. They remain in the page's marker list, which is where they are useful.
    const mapped: SeriesMarker<Time>[] = markers
      .filter((marker) => marker.time !== null)
      .map((marker) => {
        const style = MARKER_STYLE[marker.kind] ?? DEFAULT_MARKER_STYLE;
        return {
          time: marker.time as UTCTimestamp,
          position: style.position,
          shape: style.shape,
          color: style.color,
          text: style.label,
          size: 1,
        };
      });
    plugin.setMarkers(mapped);
  }, [markers, chartEpoch]);

  // ----------------------------------------------------------------- actions

  const toggleFullscreen = () => {
    const container = containerRef.current;
    if (container === null) return;
    if (document.fullscreenElement === container) {
      void document.exitFullscreen();
    } else {
      void container.requestFullscreen().catch(() => {
        // Some browsers refuse fullscreen outside a user gesture, and iframes need
        // an allow attribute. Nothing to recover — just don't claim it worked.
      });
    }
  };

  if (bars.length === 0) {
    return (
      <div className="chart">
        <div className="chart__empty">No bars to draw.</div>
      </div>
    );
  }

  return (
    <div className="chart" ref={containerRef}>
      <div className="chart__overlay">
        {readout === null ? (
          <span className="muted">Hover the chart for OHLCV</span>
        ) : (
          <>
            <div>
              <b>{formatTime(readout.time)}</b>
            </div>
            <div>
              O {formatPrice(readout.open)} · H {formatPrice(readout.high)}
            </div>
            <div>
              L {formatPrice(readout.low)} · C {formatPrice(readout.close)}
            </div>
            <div className="muted">Vol {readout.volume.toLocaleString()}</div>
          </>
        )}
      </div>

      {!isFullscreen && (
        <div style={{ position: "absolute", top: 10, right: 12, zIndex: 3, display: "flex", gap: 6 }}>
          <button
            type="button"
            className="btn btn--sm btn--util"
            onClick={() => chartRef.current?.timeScale().fitContent()}
            title="Fit all loaded bars into view"
          >
            Fit
          </button>
          <button
            type="button"
            className="btn btn--sm btn--util"
            onClick={toggleFullscreen}
            title="Fullscreen"
          >
            ⛶
          </button>
        </div>
      )}
      {isFullscreen && (
        <button
          type="button"
          className="btn btn--sm btn--util"
          style={{ position: "absolute", top: 10, right: 12, zIndex: 3 }}
          onClick={toggleFullscreen}
        >
          Exit fullscreen
        </button>
      )}
    </div>
  );
}
