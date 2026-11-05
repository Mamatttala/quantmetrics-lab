import { afterEach, describe, expect, it, vi } from "vitest";
import type { VolForecastReport } from "@/api/types";
import { candleTheme, chartTheme } from "@/theme";
import { matrixTrace, stateTraces } from "@/pages/regimes/charts";
import { buildFan } from "@/pages/VolatilityPage";

vi.mock("react-plotly.js/factory", () => ({ default: () => () => null }));
vi.mock("plotly.js-dist-min", () => ({ default: {} }));

afterEach(() => document.documentElement.removeAttribute("style"));

function tokens(values: Record<string, string>) {
  for (const [name, value] of Object.entries(values)) {
    document.documentElement.style.setProperty(name, value);
  }
}

describe("chart color consumers", () => {
  it("reads candle direction, volume, and chrome from the current CSS variables", () => {
    tokens({
      "--bg-chart": "#102030", "--color-positive": "#123456", "--color-negative": "#654321",
      "--chart-volume-up": "rgba(18,52,86,0.42)", "--chart-volume-down": "rgba(101,67,33,0.42)",
    });
    expect(candleTheme()).toMatchObject({
      bg: "#102030", up: "#123456", down: "#654321",
      volumeUp: "rgba(18,52,86,0.42)", volumeDown: "rgba(101,67,33,0.42)",
    });
    tokens({ "--color-positive": "#abcdef" });
    expect(candleTheme().up).toBe("#abcdef");
  });

  it("uses active state colors in regime traces and the active heatmap scale", () => {
    tokens({
      "--chart-state-up": "#123456", "--chart-neutral": "#334455",
      "--chart-state-down": "#654321", "--chart-state-crisis": "#778899",
      "--chart-heatmap-low": "#102030", "--chart-heatmap-high": "#abcdef",
    });
    tokens({ "--chart-series-1": "#123456" });
    const timeline = stateTraces(["2026-01-01T00:00:00Z"], [1], [0], { "0": "Bullish" });
    expect(timeline[0]).toMatchObject({ marker: { color: "#123456" } });
    expect(matrixTrace([[0.8, 0.2], [0.3, 0.7]], ["0", "1"])).toMatchObject({
      colorscale: [[0, "#102030"], [1, "#abcdef"]],
    });
    tokens({ "--chart-series-1": "#fedcba" });
    expect(stateTraces(["2026-01-01"], [1], [0], {})[0])
      .toMatchObject({ marker: { color: "#fedcba" } });
  });

  it("uses active categorical colors and distinct forecast-band opacity tokens", () => {
    tokens({
      "--chart-series-1": "#123456", "--chart-series-2": "#abcdef",
      "--chart-band-outer-opacity": "0.12", "--chart-band-inner-opacity": "0.26",
    });
    const report = { forward_paths: [{
      model: "garch", times: ["2026-01-01T00:00:00Z"], volatility: [0.2],
      bands: [
        { label: "5–95%", high: [0.4], low: [0.1] },
        { label: "25–75%", high: [0.3], low: [0.15] },
      ],
    }] } as unknown as VolForecastReport;
    const fan = buildFan(report, {}, "garch");
    expect(fan.data[1]).toMatchObject({ fillcolor: "rgba(18, 52, 86, 0.12)" });
    expect(fan.data[3]).toMatchObject({ fillcolor: "rgba(18, 52, 86, 0.26)" });
    expect(fan.data[4]).toMatchObject({ line: { color: "#123456" }, name: "garch (bands)" });
    expect(chartTheme().series.slice(0, 2)).toEqual(["#123456", "#abcdef"]);
  });
});
