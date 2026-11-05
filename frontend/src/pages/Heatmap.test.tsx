import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import type { PlotParams } from "react-plotly.js";
import { afterEach, expect, it, vi } from "vitest";

import { THEMES, applyTheme } from "@/theme";
import { Heatmap } from "./ExploratoryAnalysisPage";

const plot = vi.hoisted(() => vi.fn());
vi.mock("@/charts/ResponsivePlot", () => ({
  ResponsivePlot: (props: PlotParams) => { plot(props); return <div data-testid="heatmap" />; },
}));

afterEach(() => {
  cleanup(); plot.mockClear();
  document.documentElement.removeAttribute("style");
  document.documentElement.removeAttribute("data-theme");
});

it("keeps RdBu_r, colorbar limits and dark annotations fixed across every app theme", () => {
  const heatmap = <Heatmap title="Correlation matrix" x={["x", "y", "z"]}
    y={["x", "y", "z"]} z={[[1, 0, -.5], [0, 1, .5], [-.5, .5, 1]]} pairs={[]} />;
  const view = render(heatmap);
  fireEvent.click(screen.getByRole("checkbox", { name: "Annotate cells" }));
  const initial = plot.mock.calls.at(-1)![0] as PlotParams;
  expect(initial.data[0]).toMatchObject({
    zmin: -1, zmax: 1, zmid: 0, zauto: false,
    autocolorscale: false, reversescale: false, showscale: true,
    colorscale: [
      [0, "#053061"], [0.1, "#2166ac"], [0.2, "#4393c3"], [0.3, "#92c5de"],
      [0.4, "#d1e5f0"], [0.5, "#f7f7f7"], [0.6, "#fddbc7"], [0.7, "#f4a582"],
      [0.8, "#d6604d"], [0.9, "#b2182b"], [1, "#67001f"],
    ],
    colorbar: { tickvals: [-1, -.5, 0, .5, 1], tickfont: { color: "#1f2937" } },
  });
  expect(initial.layout.template).toEqual({});
  expect(initial.layout.annotations).toHaveLength(9);
  expect(initial.layout.annotations).toEqual(expect.arrayContaining([
    expect.objectContaining({ text: "1.00", font: { color: "#1f2937", size: 10 } }),
    expect.objectContaining({ text: "0.00", font: { color: "#1f2937", size: 10 } }),
    expect.objectContaining({ text: "-0.50", font: { color: "#1f2937", size: 10 } }),
  ]));
  for (const { id } of THEMES) {
    applyTheme(id);
    // Deliberately unrelated theme colors must not enter either Plotly trace or layout.
    for (const token of ["--chart-series-1", "--chart-series-4", "--chart-heatmap-low", "--chart-heatmap-high", "--bg-chart", "--text-primary"]) {
      document.documentElement.style.setProperty(token, "#00ff00");
    }
    view.rerender(<Heatmap title="Correlation matrix" x={["x", "y", "z"]}
      y={["x", "y", "z"]} z={[[1, 0, -.5], [0, 1, .5], [-.5, .5, 1]]} pairs={[]} />);
    const current = plot.mock.calls.at(-1)![0] as PlotParams;
    expect(current.data).toEqual(initial.data);
    expect(current.layout).toEqual(initial.layout);
  }
});
