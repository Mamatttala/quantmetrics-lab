import { useState } from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "@/api/client";
import { getActiveProjectId, setActiveProjectId } from "@/api/projectSelection";
import type { DatasetSummary, Project } from "@/api/types";
import { ThemeProvider } from "@/theme";
import { Shell } from "./Shell";

const PROJECTS: Project[] = [
  { project_id: "default", name: "Default project", created_at: "2026-01-01T00:00:00Z" },
  { project_id: "research", name: "Bitcoin research", created_at: "2026-01-02T00:00:00Z" },
];
const dataset = (name: string): DatasetSummary => ({
  dataset_id: name, name, category: "raw_data", source: "binance", symbol: "BTCUSDT",
  timeframe: "1h", row_count: 10, start_timestamp: null, end_timestamp: null,
  created_at: "2026-01-01T00:00:00Z", parent_dataset_id: null, operation: null,
  artifact_kind: "observations", origin_page: "A1",
});

function Page() {
  const [value, setValue] = useState("");
  return <>
    <output data-testid="location">{useLocation().pathname}</output>
    <input aria-label="Page draft" value={value} onChange={(event) => setValue(event.target.value)} />
  </>;
}

function renderShell() {
  return render(<MemoryRouter initialEntries={["/data"]}><ThemeProvider>
    <Routes><Route element={<Shell />}><Route path="/data/:datasetId?" element={<Page />} /></Route></Routes>
  </ThemeProvider></MemoryRouter>);
}

describe("project selection", () => {
  beforeEach(() => {
    window.localStorage.clear();
    window.sessionStorage.clear();
    setActiveProjectId("default");
    vi.spyOn(api, "health").mockResolvedValue({ status: "ok", version: "test", data_dir: "test" });
    vi.spyOn(api, "listProjects").mockResolvedValue(PROJECTS);
    vi.spyOn(api, "listDatasets").mockImplementation(async () => {
      if (getActiveProjectId() === "default") return [dataset("Legacy data")];
      if (getActiveProjectId() === "research") return [dataset("Research data")];
      return [];
    });
  });

  afterEach(() => { vi.restoreAllMocks(); setActiveProjectId("default"); });

  it("switches datasets and clears the previous dataset route and page state", async () => {
    renderShell();
    fireEvent.click(await screen.findByRole("button", { name: "Legacy data" }));
    fireEvent.change(screen.getByLabelText("Page draft"), { target: { value: "old dataset edit" } });
    expect(screen.getByTestId("location").textContent).toBe("/data/Legacy data");

    fireEvent.change(screen.getByLabelText("Project"), { target: { value: "research" } });
    expect(await screen.findByRole("button", { name: "Research data" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Legacy data" })).toBeNull();
    expect(screen.getByTestId("location").textContent).toBe("/data");
    expect((screen.getByLabelText("Page draft") as HTMLInputElement).value).toBe("");
    expect(window.sessionStorage.getItem("qca.project")).toBe("research");
    expect(screen.getAllByRole("button", { name: "Choose theme" })).toHaveLength(1);

    fireEvent.change(screen.getByLabelText("Project"), { target: { value: "default" } });
    expect(await screen.findByRole("button", { name: "Legacy data" })).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Research data" })).toBeNull();
  });

  it("creates and selects an empty project, and restores that selection on remount", async () => {
    const created = { project_id: "new-project", name: "New study", created_at: "2026-01-03T00:00:00Z" };
    vi.spyOn(api, "createProject").mockResolvedValue(created);
    const view = renderShell();
    await screen.findByRole("button", { name: "Legacy data" });
    fireEvent.click(screen.getByRole("button", { name: "+ New project" }));
    fireEvent.change(screen.getByLabelText("Project name"), { target: { value: " New study " } });
    fireEvent.click(screen.getByRole("button", { name: "Create project" }));
    await screen.findByText(/No datasets in this project yet/);
    expect(api.createProject).toHaveBeenCalledWith("New study");
    expect((screen.getByLabelText("Project") as HTMLSelectElement).value).toBe("new-project");
    expect(screen.queryByRole("button", { name: "Legacy data" })).toBeNull();

    view.unmount();
    vi.mocked(api.listProjects).mockResolvedValue([...PROJECTS, created]);
    renderShell();
    await screen.findByText(/No datasets in this project yet/);
    expect((screen.getByLabelText("Project") as HTMLSelectElement).value).toBe("new-project");
  });

  it("keeps the active project and shows the error when creation is rejected", async () => {
    vi.spyOn(api, "createProject").mockRejectedValue(new Error("A project with this name already exists."));
    renderShell();
    await screen.findByRole("button", { name: "Legacy data" });
    fireEvent.click(screen.getByRole("button", { name: "+ New project" }));
    fireEvent.change(screen.getByLabelText("Project name"), { target: { value: "Default project" } });
    fireEvent.click(screen.getByRole("button", { name: "Create project" }));
    expect(await screen.findByRole("alert")).toBeTruthy();
    expect(getActiveProjectId()).toBe("default");
    expect(screen.getByRole("button", { name: "Legacy data" })).toBeTruthy();
  });

  it("does not expose datasets until a saved project has been validated", async () => {
    setActiveProjectId("missing-project");
    vi.mocked(api.listProjects).mockRejectedValueOnce(new Error("Connection unavailable"));
    renderShell();
    await screen.findByRole("alert");
    expect(api.listDatasets).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    await waitFor(() => expect(getActiveProjectId()).toBe("default"));
    expect(await screen.findByRole("button", { name: "Legacy data" })).toBeTruthy();
  });
});
