/// <reference types="vite/client" />
import source from "./ExploratoryAnalysisPage.tsx?raw";
import css from "./exploratory.css?raw";
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { api } from "@/api/client";
import { featuresInGroups, heatmapValues, resolveAnalysis, scopeIsReady } from "@/api/exploratory";
import type { AnalysisResult, CorrelationResult, PCAResult } from "@/api/exploratory";
import { ExploratoryAnalysisPage, PCASavedBanner } from "./ExploratoryAnalysisPage";

vi.mock("@/charts/ResponsivePlot", () => ({ ResponsivePlot: () => <div data-testid="plot" /> }));
vi.mock("@/components/ResultGrid", () => ({ ResultGrid: ({ caption }: { caption: string }) => <div>{caption}</div> }));
vi.mock("@/theme", () => ({ useTheme: () => ({ theme: "dark" }), chartTheme: () => ({ series: [], paper: "", plot: "" }) }));

afterEach(() => { cleanup(); vi.restoreAllMocks(); });

describe("C8 input and response helpers", () => {
  it("selects feature groups without duplicate features", () => {
    expect(featuresInGroups({ returns: ["r"], volatility: ["r", "v"] }, ["returns", "volatility"])).toEqual(["r", "v"]);
  });
  it("validates scope without requiring a split for full-sample exploration", () => {
    expect(scopeIsReady("full_dataset", null, "", "", false)).toBe(true);
    expect(scopeIsReady("train_span", "open_time", "", "", false)).toBe(false);
    expect(scopeIsReady("custom_date_range", "open_time", "2024-02-01", "2024-01-01", false)).toBe(false);
    expect(scopeIsReady("rolling", "open_time", "", "", false)).toBe(true);
  });
  it("filters heatmap triangles and thresholds without changing the report", () => {
    const values = [[1, .2], [.2, 1]];
    expect(heatmapValues(values, "upper", 0)).toEqual([[1, .2], [null, 1]]);
    expect(heatmapValues(values, "full", .3)).toEqual([[1, null], [null, 1]]);
    expect(values[1]?.[0]).toBe(.2);
  });
  it("handles immediate reports and pollable jobs uniformly", async () => {
    const report = { dataset_id: "source" } as AnalysisResult;
    const job = vi.spyOn(api, "job").mockResolvedValue({ status: "succeeded", result: report } as never);
    expect(await resolveAnalysis(report)).toBe(report);
    expect(job).not.toHaveBeenCalled();
    expect(await resolveAnalysis({ job_id: "job", status: "running", poll: "/api/jobs/job" })).toBe(report);
    expect(job).toHaveBeenCalledWith("job");
  });
});

const common = { dataset_id: "source", dataset_name: "Synthetic", fit_scope: "full_dataset", n_scope_rows: 100,
  features: ["x", "y"], fit_rows: 100, transform_rows: 100, warnings: [], notes: [],
  metadata: {}, excluded_constants: [], preprocessing_parameters: {}, missing_summary: {},
  causal_for_downstream_training: false };
const pca = { ...common, n_components: 2, n_fit_rows: 100, component_columns: ["pc_1", "pc_2"],
  component_summary: [], biplot: { scores: [], loadings: [], row_indices: [] }, loadings: [[1, 0], [0, 1]],
  dataset: { dataset_id: "saved", name: "PCA result", category: "outputs" }, params_dataset: null,
} as unknown as PCAResult;

it("shows a non-causal warning and causal success on saved PCA datasets", () => {
  const { rerender } = render(<MemoryRouter><PCASavedBanner result={pca} /></MemoryRouter>);
  expect(screen.getByTestId("pca-saved").textContent).toContain("Non-causal fit");
  expect(screen.getByTestId("pca-saved").className).toBe("exploratory__notice");
  rerender(<MemoryRouter><PCASavedBanner result={{ ...pca, fit_scope: "train_span", causal_for_downstream_training: true }} /></MemoryRouter>);
  expect(screen.getByTestId("pca-saved").textContent).toContain("Train-span C8 fit");
});

it("preserves correlation and PCA results across independent runs", async () => {
  vi.spyOn(api, "exploratoryOptions").mockResolvedValue({ dataset_id: "source", datasets: [], features: ["x", "y"], columns: ["x", "y"], plans: [], time_column: "open_time", methods: ["pearson"], warnings: [], ignored_columns: [], feature_groups: {}, computational_limits: { max_features: 250 }, target_candidates: ["x", "y"], group_by_candidates: [], has_close_time: false, supported_dataset_categories: ["raw_data", "uploaded_data", "processed_data", "outputs"], dataset_naming: "canonical_auto", default_parameters: {} });
  const correlation = { ...common, method: "pearson", pairwise: [], matrix: [[1, .5], [.5, 1]], vif: [], redundancy_groups: [], conditional: [], ccf: [], rolling: [], rolling_summary: [], ccf_summary: [] } as unknown as CorrelationResult;
  vi.spyOn(api, "runCorrelation").mockResolvedValue(correlation);
  vi.spyOn(api, "runPCA").mockResolvedValue(pca);
  render(<MemoryRouter initialEntries={["/exploratory/source"]}><Routes><Route path="/exploratory/:datasetId" element={<ExploratoryAnalysisPage />} /></Routes></MemoryRouter>);
  await waitFor(() => expect((screen.getByRole("button", { name: "Run Correlation" }) as HTMLButtonElement).disabled).toBe(false));
  fireEvent.click(screen.getByRole("button", { name: "Run Correlation" }));
  await screen.findByTestId("correlation-results");
  fireEvent.click(screen.getByRole("button", { name: "Run PCA Diagnostics" }));
  await screen.findByTestId("pca-results");
  expect(screen.getByTestId("correlation-results")).toBeTruthy();
  fireEvent.click(screen.getByRole("button", { name: "Run Correlation" }));
  expect(screen.getByTestId("pca-results")).toBeTruthy();
});

it("uses theme tokens and preserves the positive-lag warning", () => {
  // The heatmap deliberately has its own fixed palette; other charts use theme tokens.
  const themedSource = source.slice(0, source.indexOf("export function Heatmap("))
    + source.slice(source.indexOf("function CorrelationResults("));
  expect(themedSource + css).not.toMatch(/(?<!&)#[0-9a-fA-F]{3,8}\b/);
  expect(source).toContain("chartTheme()");
  expect(source).toContain("Positive-lag CCF uses future target values");
  expect(source).not.toContain("Causal rolling Pearson diagnostic");
  expect(source).toContain("All pairs summary only");
  expect(source).toContain("auto-generated canonical name");
});
