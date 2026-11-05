import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, api } from "./client";
import { setActiveProjectId } from "./projectSelection";

afterEach(() => {
  vi.unstubAllGlobals();
  setActiveProjectId("default");
});

describe("project-aware requests", () => {
  it("sends the selected project on both JSON and multipart requests", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response("{}"));
    vi.stubGlobal("fetch", fetchMock);
    setActiveProjectId("research");
    await api.renameDataset("dataset", "new name");
    expect(new Headers(fetchMock.mock.calls[0]?.[1].headers).get("X-Project-Id")).toBe("research");
    fetchMock.mockResolvedValue(new Response("{}"));
    await api.uploadCsv(new File(["value\n1"], "sample.csv"));
    const uploadHeaders = new Headers(fetchMock.mock.calls[1]?.[1].headers);
    expect(uploadHeaders.get("X-Project-Id")).toBe("research");
    expect(uploadHeaders.has("Content-Type")).toBe(false);
    expect(api.datasetExportUrl("dataset", "csv")).toContain("project_id=research");
  });

  it("rejects a late response from a previously selected project", async () => {
    let complete!: (response: Response) => void;
    vi.stubGlobal("fetch", vi.fn(() => new Promise<Response>((resolve) => { complete = resolve; })));
    setActiveProjectId("first");
    const pending = api.listDatasets();
    setActiveProjectId("second");
    complete(new Response("[]"));
    await expect(pending).rejects.toMatchObject({ code: "project_changed" });
  });
});

describe("dataset mutation client methods", () => {
  it("posts a rename action and resolves successful responses", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ dataset_id: "dataset-1" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await api.renameDataset("dataset/1", "new name");

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/datasets/dataset%2F1/rename",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ name: "new name" }),
      }),
    );
  });

  it("deletes one dataset and accepts the backend's 200 response", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ deleted: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await api.deleteDataset("dataset-1");

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/datasets/dataset-1",
      expect.objectContaining({ method: "DELETE" }),
    );
  });

  it("throws ApiError for a rejected mutation", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ error: { code: "conflict", message: "Cannot delete" } }), {
          status: 409,
          headers: { "Content-Type": "application/json" },
        }),
      ),
    );

    const result = api.deleteDataset("dataset-1");

    await expect(result).rejects.toBeInstanceOf(ApiError);
    await expect(result).rejects.toMatchObject({ status: 409, code: "conflict" });
  });
});
