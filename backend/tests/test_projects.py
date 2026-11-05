"""Projects isolate real storage, including exports, deletes, and background jobs."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi import Depends
from fastapi.testclient import TestClient

from app.config import get_project_settings, get_settings
from app.main import create_app
from app.routers.projects import project_scope
from app.services.jobs import JobService, spawn
from tests.conftest import hourly_frame, register


@pytest.fixture
def project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    root = tmp_path / "project-data"
    monkeypatch.setenv("QML_DATA_DIR", str(root))
    get_settings.cache_clear()
    yield root
    get_settings.cache_clear()


@pytest.fixture
def client(project_root: Path) -> Iterator[TestClient]:
    with TestClient(create_app()) as instance:
        yield instance


def create(client: TestClient, name: str) -> str:
    response = client.post("/api/projects", json={"name": name})
    assert response.status_code == 201, response.text
    return response.json()["project_id"]


def upload(client: TestClient, project_id: str = "default") -> dict:
    response = client.post(
        "/api/datasets/upload",
        headers={"X-Project-Id": project_id},
        files={"file": ("sample.csv", b"value\n1\n2\n3\n", "text/csv")},
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_existing_catalog_is_preserved_as_default(settings, catalog, monkeypatch) -> None:
    meta = register(settings, catalog, hourly_frame(12))
    original = Path(meta.storage_path).read_bytes()
    monkeypatch.setenv("QML_DATA_DIR", str(settings.data_dir))
    get_settings.cache_clear()
    try:
        with TestClient(create_app()) as client:
            assert client.get("/api/projects").json()[0]["name"] == "Default project"
            assert client.get("/api/datasets").json()[0]["dataset_id"] == meta.dataset_id
            project_id = create(client, "New study")
            assert client.get("/api/datasets", headers={"X-Project-Id": project_id}).json() == []
            restored = client.get(f"/api/datasets/{meta.dataset_id}").json()
            assert restored["storage_path"] == meta.storage_path
            assert restored["checksum"] == hashlib.sha256(original).hexdigest()
            assert Path(meta.storage_path).read_bytes() == original
    finally:
        get_settings.cache_clear()


def test_projects_reopen_after_restart_and_validate_names(client: TestClient) -> None:
    project_id = create(client, "  Bitcoin research  ")
    dataset = upload(client, project_id)
    assert client.post("/api/projects", json={"name": "BITCOIN RESEARCH"}).status_code == 422
    for name in ("   ", "a" * 81, "line\nbreak"):
        assert client.post("/api/projects", json={"name": name}).status_code == 422
    with TestClient(create_app()) as reopened:
        projects = reopened.get("/api/projects").json()
        assert [(p["project_id"], p["name"]) for p in projects][1] == (
            project_id, "Bitcoin research"
        )
        rows = reopened.get("/api/datasets", headers={"X-Project-Id": project_id}).json()
        assert [row["dataset_id"] for row in rows] == [dataset["dataset_id"]]


def test_upload_derive_export_and_bulk_delete_are_scoped(
    client: TestClient, project_root: Path
) -> None:
    legacy = upload(client)
    first = create(client, "First")
    second = create(client, "Second")
    dataset = upload(client, first)
    untouched = upload(client, second)
    headers = {"X-Project-Id": first}
    dataset_id = dataset["dataset_id"]
    assert Path(dataset["storage_path"]).is_relative_to(project_root / "projects" / first)

    derived = client.post(
        f"/api/datasets/{dataset_id}/derive", headers=headers,
        json={"edits": [{"row": 0, "column": "value", "value": 4}], "name": "Edited"},
    )
    assert derived.status_code == 201, derived.text
    child = derived.json()
    assert child["parent_dataset_id"] == dataset_id
    assert Path(child["storage_path"]).is_relative_to(project_root / "projects" / first)
    listed = client.get("/api/datasets", headers=headers).json()
    assert {row["dataset_id"] for row in listed} == {dataset_id, child["dataset_id"]}

    for other in ("default", second):
        foreign = {"X-Project-Id": other}
        assert client.get(f"/api/datasets/{dataset_id}", headers=foreign).status_code == 404
        assert client.get(f"/api/datasets/{dataset_id}/preview", headers=foreign).status_code == 404
        assert client.delete(f"/api/datasets/{dataset_id}", headers=foreign).status_code == 404
        assert client.post(f"/api/datasets/{dataset_id}/rename", headers=foreign,
                           json={"name": "Wrong project"}).status_code == 404

    export = client.get(f"/api/datasets/{dataset_id}/export",
                        params={"format": "csv", "project_id": first})
    assert export.status_code == 200, export.text
    assert "value" in export.text
    assert client.get(f"/api/datasets/{dataset_id}/export",
                      params={"format": "csv", "project_id": second}).status_code == 404

    deleted = client.delete("/api/datasets", headers=headers)
    assert deleted.json() == {"deleted": 2}
    assert client.get("/api/datasets", headers=headers).json() == []
    assert not Path(dataset["storage_path"]).exists()
    assert not Path(child["storage_path"]).exists()
    assert Path(legacy["storage_path"]).exists()
    assert Path(untouched["storage_path"]).exists()
    assert len(client.get("/api/datasets").json()) == 1
    assert len(client.get("/api/datasets", headers={"X-Project-Id": second}).json()) == 1


def test_unknown_projects_never_fall_back_or_create_directories(
    client: TestClient, project_root: Path
) -> None:
    upload(client)
    for project_id in ("", "missing", "../outside", "..\\outside"):
        response = client.delete("/api/datasets", headers={"X-Project-Id": project_id})
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "project_not_found"
    assert len(client.get("/api/datasets").json()) == 1
    assert not (project_root.parent / "outside").exists()


def test_fetch_deduplication_stays_within_each_project(client: TestClient, monkeypatch) -> None:
    from app.services import binance_fetch

    async def fake_klines(*args):
        return hourly_frame(12)

    monkeypatch.setattr(binance_fetch, "fetch_klines", fake_klines)
    first, second = create(client, "Fetch A"), create(client, "Fetch B")
    body = {"asset_class": "crypto_spot", "symbol": "BTCUSDT", "timeframe": "1h",
            "start": "2024-01-01T00:00:00Z", "end": "2024-01-02T00:00:00Z"}
    outputs = []
    for project_id in (first, first, second):
        response = client.post("/api/fetch/binance/klines", json=body,
                               headers={"X-Project-Id": project_id})
        assert response.status_code == 201, response.text
        outputs.append(response.json())
    assert [row["deduplicated"] for row in outputs] == [False, True, False]
    assert outputs[0]["dataset"]["dataset_id"] != outputs[2]["dataset"]["dataset_id"]
    assert client.get("/api/datasets").json() == []


async def test_background_task_keeps_its_project_after_request_ends(project_root: Path) -> None:
    app = create_app()
    gate = asyncio.Event()
    tasks = []

    @app.post("/test-project-job", dependencies=[Depends(project_scope)])
    async def launch():
        job_id = await JobService().create("test")

        async def work():
            await gate.wait()
            await JobService().succeed(
                job_id, result={"folder": str(get_project_settings().data_dir)}
            )

        tasks.append(spawn(work()))
        return {"job_id": job_id}

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client,
    ):
        first = (await client.post("/api/projects", json={"name": "Job A"})).json()["project_id"]
        second = (await client.post("/api/projects", json={"name": "Job B"})).json()["project_id"]
        launched = await client.post("/test-project-job", headers={"X-Project-Id": first})
        job = launched.json()["job_id"]
        assert (await client.get("/api/jobs", headers={"X-Project-Id": second})).json() == []
        gate.set()
        await asyncio.gather(*tasks)
        result = await client.get(f"/api/jobs/{job}", headers={"X-Project-Id": first})
        assert result.status_code == 200, result.text
        assert result.json()["status"] == "succeeded"
        assert Path(result.json()["result"]["folder"]) == project_root / "projects" / first
        foreign = await client.get(f"/api/jobs/{job}", headers={"X-Project-Id": second})
        assert foreign.status_code == 404
        assert (await client.get("/api/jobs")).json() == []
