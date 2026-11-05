import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, useLocation } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "@/api/client";
import type { DatasetSummary } from "@/api/types";
import { ThemeProvider } from "@/theme";

import { DataExplorer } from "./DataExplorer";

const DATASET: DatasetSummary = {
  dataset_id: "dataset-1",
  name: "BTC hourly",
  category: "raw_data",
  source: "binance",
  symbol: "BTCUSDT",
  timeframe: "1h",
  row_count: 100,
  start_timestamp: "2026-01-01T00:00:00Z",
  end_timestamp: "2026-01-05T03:00:00Z",
  created_at: "2026-01-05T04:00:00Z",
  parent_dataset_id: null,
  operation: null,
  artifact_kind: "observations",
  origin_page: "A1",
};

function renderExplorer(): void {
  render(
    <MemoryRouter initialEntries={["/data"]}>
      <ThemeProvider>
        <DataExplorer
          project={{ project_id: "default", name: "Default project", created_at: "2026-01-01T00:00:00Z" }}
          projectControls={null}
          onHide={vi.fn()}
        />
        <LocationProbe />
      </ThemeProvider>
    </MemoryRouter>,
  );
}

function LocationProbe() {
  return <output data-testid="location">{useLocation().pathname}</output>;
}

describe("DataExplorer dataset actions", () => {
  beforeEach(() => {
    vi.spyOn(api, "listDatasets").mockResolvedValue([DATASET]);
    vi.spyOn(api, "renameDataset").mockResolvedValue(undefined);
    vi.spyOn(api, "deleteDataset").mockResolvedValue(undefined);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders rename and delete buttons for every dataset row", async () => {
    renderExplorer();

    expect(await screen.findByText("BTC hourly")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Rename dataset BTC hourly" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Delete dataset BTC hourly" })).toBeTruthy();
  });

  it("deletes the selected dataset after confirmation", async () => {
    renderExplorer();
    await screen.findByText("BTC hourly");
    vi.spyOn(window, "confirm").mockReturnValue(true);

    fireEvent.click(screen.getByRole("button", { name: "Delete dataset BTC hourly" }));

    await waitFor(() => {
      expect(api.deleteDataset).toHaveBeenCalledWith("dataset-1");
    });
    expect(api.listDatasets).toHaveBeenCalledTimes(2);
  });

  it("prompts for and submits a new dataset name", async () => {
    renderExplorer();
    await screen.findByText("BTC hourly");
    vi.spyOn(window, "prompt").mockReturnValue("Renamed BTC");

    fireEvent.click(screen.getByRole("button", { name: "Rename dataset BTC hourly" }));

    await waitFor(() => {
      expect(api.renameDataset).toHaveBeenCalledWith("dataset-1", "Renamed BTC");
    });
    expect(window.prompt).toHaveBeenCalledWith("New name", "BTC hourly");
    expect(api.listDatasets).toHaveBeenCalledTimes(2);
  });

  it("opens Data Management when a dataset is clicked", async () => {
    renderExplorer();
    fireEvent.click(await screen.findByRole("button", { name: "BTC hourly" }));

    await waitFor(() =>
      expect(screen.getByTestId("location").textContent).toBe("/data/dataset-1"),
    );
  });

  it("groups observation splits separately from their boundary plans", async () => {
    vi.mocked(api.listDatasets).mockResolvedValue([
      {
        ...DATASET,
        dataset_id: "partition-1",
        name: "BTC split train",
        category: "processed_data",
        operation: "split_partition",
        origin_page: "C6",
      },
      {
        ...DATASET,
        dataset_id: "plan-1",
        name: "BTC split plan",
        category: "outputs",
        operation: "split",
        artifact_kind: "split_plan",
        origin_page: "C6",
      },
    ]);
    renderExplorer();

    expect(await screen.findByText("Processed Data")).toBeTruthy();
    expect(screen.getByText("Chronological Splits")).toBeTruthy();
    expect(screen.getByText("Outputs")).toBeTruthy();
    expect(screen.getByText("Split Plans")).toBeTruthy();
  });
});
