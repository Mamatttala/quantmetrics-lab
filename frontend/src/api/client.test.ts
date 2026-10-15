import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, api } from "./client";

afterEach(() => {
  vi.unstubAllGlobals();
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

  it("deletes one dataset and accepts the backend's empty 204 response", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 204 }));
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
