import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "@/api/client";
import { notifyDatasetsChanged } from "@/api/datasetsChanged";
import type { DatasetSummary, JobRecord, ScalingOptions, SplitPlanSummary } from "@/api/types";
import { ScalingPage } from "./ScalingPage";

const source = {
  dataset_id: "source", name: "Prepared data", category: "processed_data",
  operation: "features", artifact_kind: "observations", row_count: 200, timeframe: "1h",
} as DatasetSummary;
const savedPlan = {
  split_plan_id: "saved-plan", name: "Saved chronological split",
  split_type: "percentage_70_15_15", folds: 1, rows: 3,
} as SplitPlanSummary;
const planDataset = {
  ...source, dataset_id: savedPlan.split_plan_id, name: savedPlan.name,
  category: "outputs", operation: "split", artifact_kind: "split_plan", row_count: 3,
} as DatasetSummary;
const base: ScalingOptions = {
  dataset_id: "source", name: source.name, rows: 200, features: ["close"],
  defaults: { features: ["close"], window: 60, scaler_type: "standard", split_plan_id: null },
  scalers: ["standard", "minmax", "robust", "rolling_zscore"].map((id) => ({
    id: id as ScalingOptions["scalers"][number]["id"], available: true, reason: null,
    needs_split_plan: false, supports_split_plan: id !== "rolling_zscore",
  })),
  plans: [], warnings: [],
};

function renderPage(path = "/scaling/source") {
  return render(<MemoryRouter initialEntries={[path]}>
    <Routes>
      <Route path="/scaling/:datasetId?" element={<ScalingPage />} />
    </Routes>
  </MemoryRouter>);
}

beforeEach(() => {
  vi.spyOn(api, "listDatasets").mockResolvedValue([source, planDataset]);
  vi.spyOn(api, "scalingOptions").mockResolvedValue(base);
  vi.spyOn(api, "runScaling").mockResolvedValue({
    job_id: "job", status: "running", poll: "/api/jobs/job",
  });
  vi.spyOn(api, "job").mockResolvedValue({
    job_id: "job", status: "succeeded", result: null, message: "Scaling finished.",
  } as unknown as JobRecord);
});
afterEach(() => { cleanup(); vi.restoreAllMocks(); });

describe("C7 scaling scopes", () => {
  it.each(["standard", "minmax", "robust"])("runs %s on the entire dataset without splitting", async (scaler) => {
    renderPage();
    const radio = await screen.findByDisplayValue(scaler);
    expect((radio as HTMLInputElement).disabled).toBe(false);
    fireEvent.click(radio);
    const scope = screen.getByLabelText("Fit scope / split plan") as HTMLSelectElement;
    expect(scope.value).toBe("");
    expect(scope.disabled).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: "Run Scaling" }));
    await waitFor(() => expect(api.runScaling).toHaveBeenCalledWith(expect.objectContaining({
      dataset_id: "source", scaler_type: scaler, split_plan_id: null, fold: null,
    })));
  });

  it("loads a previously saved plan when selecting a dataset from the C7 menu", async () => {
    vi.mocked(api.scalingOptions).mockResolvedValue({
      ...base, plans: [savedPlan], defaults: { ...base.defaults, split_plan_id: "saved-plan" },
    });
    renderPage("/scaling");
    await screen.findByRole("option", { name: /Prepared data/ });
    fireEvent.change(screen.getByLabelText("Dataset"), { target: { value: "source" } });
    const scope = await screen.findByLabelText("Fit scope / split plan") as HTMLSelectElement;
    expect(scope.value).toBe("saved-plan");
    fireEvent.click(screen.getByRole("button", { name: "Run Scaling" }));
    await waitFor(() => expect(api.runScaling).toHaveBeenCalledWith(expect.objectContaining({
      split_plan_id: "saved-plan",
    })));
  });

  it("offers saved split datasets directly and permits switching their source to full scaling", async () => {
    vi.mocked(api.scalingOptions).mockResolvedValue({
      ...base, plans: [savedPlan], defaults: { ...base.defaults, split_plan_id: "saved-plan" },
    });
    renderPage("/scaling/saved-plan");
    await screen.findByRole("option", { name: /Saved chronological split.*spans/ });
    const scope = await screen.findByLabelText("Fit scope / split plan");
    fireEvent.change(scope, { target: { value: "" } });
    fireEvent.click(screen.getByRole("button", { name: "Run Scaling" }));
    await waitFor(() => expect(api.runScaling).toHaveBeenCalledWith(expect.objectContaining({
      dataset_id: "source", split_plan_id: null,
    })));
  });

  it("refreshes saved plans after catalog changes while preserving the chosen scaler", async () => {
    renderPage();
    fireEvent.click(await screen.findByDisplayValue("robust"));
    vi.mocked(api.scalingOptions).mockResolvedValue({ ...base, plans: [savedPlan] });
    notifyDatasetsChanged();
    await screen.findByRole("option", { name: /Saved chronological split.*fold/ });
    expect((screen.getByDisplayValue("robust") as HTMLInputElement).checked).toBe(true);
    expect((screen.getByLabelText("Fit scope / split plan") as HTMLSelectElement).value).toBe("");
  });
});
