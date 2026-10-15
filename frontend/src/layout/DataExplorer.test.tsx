import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
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
};

function renderExplorer(): void {
  render(
    <MemoryRouter initialEntries={["/data"]}>
      <ThemeProvider>
        <DataExplorer onHide={vi.fn()} />
      </ThemeProvider>
    </MemoryRouter>,
  );
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
});
