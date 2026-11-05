import type { Data, Layout } from "plotly.js";
import { ResponsivePlot } from "@/charts/ResponsivePlot";
import { chartTheme, useTheme } from "@/theme";

export function chartLayout(title: string): Partial<Layout> {
  const theme = chartTheme();
  return { title: { text: title, font: { size: 13 } }, height: 320, paper_bgcolor: theme.paper, plot_bgcolor: theme.plot,
    font: { color: theme.font, size: 11 }, margin: { t: 42, b: 45, l: 55, r: 25 },
    xaxis: { gridcolor: theme.grid, zerolinecolor: theme.zero }, yaxis: { gridcolor: theme.grid, zerolinecolor: theme.zero },
    legend: { orientation: "h", y: -0.2 }, hovermode: "closest" };
}
export function matrixTrace(matrix: number[][], labels: string[]): Data {
  const theme = chartTheme();
  return { type: "heatmap", z: matrix, x: labels, y: labels, colorscale: [[0, theme.heatmapLow], [1, theme.heatmapHigh]] };
}
export function stateTraces(times: string[], values: (number | null)[], states: number[], labels: Record<string, string>): Data[] {
  const theme = chartTheme();
  return [...new Set(states)].filter(k => k >= 0).sort((a, b) => a - b).map((state, i) => ({
    type: "scatter", mode: "markers", name: labels[String(state)] ? `${state} · ${labels[String(state)]}` : `State ${state}`,
    x: times.filter((_, j) => states[j] === state), y: values.filter((_, j) => states[j] === state),
    marker: { size: 4, color: theme.series[i % theme.series.length] ?? theme.gold },
  }));
}
export function Chart({ title, data }: { title: string; data: Data[] }) {
  useTheme();
  return <ResponsivePlot data={data} layout={chartLayout(title)} config={{ responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"] }} style={{ width: "100%" }} />;
}
export function Matrix({ title, values, labels }: { title: string; values: number[][]; labels: string[] }) {
  useTheme();
  return <Chart title={title} data={[matrixTrace(values, labels)]} />;
}
export function StateChart({ title, times, values, states, labels }: { title: string; times: string[]; values: (number | null)[]; states: number[]; labels: Record<string, string> }) {
  useTheme();
  return <Chart title={title} data={stateTraces(times, values, states, labels)} />;
}
export function ProbabilityChart({ times, values, labels }: { times: string[]; values: number[][]; labels: Record<string, string> }) {
  useTheme();
  const theme = chartTheme();
  return <Chart title="State probabilities" data={(values[0] ?? []).map((_, k) => ({ type: "scatter", mode: "lines", name: `${k} · ${labels[String(k)] ?? "State"}`,
    x: times, y: values.map(row => row[k] ?? null), line: { color: theme.series[k % theme.series.length] ?? theme.gold, width: 1.5 } }))} />;
}
