"""The real C8 HTTP surface, including failures before and after job acceptance."""

import time

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import create_app
from app.models.enums import DatasetCategory
from app.storage.catalog import Catalog
from tests.conftest import register
from tests.test_exploratory import sample


@pytest.fixture
def api_env(tmp_path, monkeypatch):
    data = tmp_path / "c8-api"
    monkeypatch.setenv("QML_DATA_DIR", str(data))
    get_settings.cache_clear()
    settings = Settings(data_dir=data)
    settings.ensure_dirs()
    catalog = Catalog(settings.catalog_path)
    catalog.initialise()
    with TestClient(create_app()) as client:
        yield client, settings, catalog
    catalog.close()
    get_settings.cache_clear()


def poll(client, job_id):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        body = response.json()
        if body["status"] != "running":
            return body
        time.sleep(0.02)
    pytest.fail("C8 job never finished")


def test_options_lists_every_category(api_env):
    client, settings, catalog = api_env
    for category in DatasetCategory:
        source = register(settings, catalog, sample(), category=category)
    options = client.get("/api/exploratory/options")
    assert options.status_code == 200
    assert {r["category"] for r in options.json()["datasets"]} == {c.value for c in DatasetCategory}
    options = client.get("/api/exploratory/options", params={"dataset_id": source.dataset_id})
    assert options.json()["features"] == ["x", "y", "z"]


@pytest.mark.parametrize("endpoint", ["correlation", "pca"])
def test_heavy_job_pattern_returns_immediate_diagnostics(api_env, endpoint):
    client, settings, catalog = api_env
    source = register(settings, catalog, sample())
    accepted = client.post(
        f"/api/exploratory/{endpoint}",
        json={"dataset_id": source.dataset_id, "features": ["x", "y"]},
    )
    assert accepted.status_code == 202
    assert set(accepted.json()) == {"job_id", "status", "poll"}
    body = poll(client, accepted.json()["job_id"])
    assert body["status"] == "succeeded", body
    assert body["result"]["causal_for_downstream_training"] is False
    assert "matrix" in body["result"] if endpoint == "correlation" else "loadings" in body["result"]
    assert len(catalog.list_datasets()) == 1


def test_strict_and_bad_feature_refusals_are_422(api_env):
    client, settings, catalog = api_env
    source = register(settings, catalog, sample())
    base = {"dataset_id": source.dataset_id, "features": ["x", "y"]}
    for params in [
        {"strict_causal": True, "write_dataset": True},
        {"features": ["close_time"]},
        {"scope": "train_span"},
        {"n_components": 0},
    ]:
        response = client.post("/api/exploratory/pca", json={**base, **params})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_failure"


def test_save_catalog_and_job_validation_failure(api_env):
    client, settings, catalog = api_env
    source = register(settings, catalog, sample())
    base = {"dataset_id": source.dataset_id, "features": ["x", "y"], "write_dataset": True}
    response = client.post("/api/exploratory/pca", json=base)
    result = poll(client, response.json()["job_id"])
    assert result["status"] == "succeeded", result
    ids = {r["dataset_id"] for r in client.get("/api/datasets").json()}
    assert result["result"]["dataset"]["dataset_id"] in ids
    assert result["result"]["params_dataset"]["dataset_id"] in ids
    response = client.post(
        "/api/exploratory/pca", json={**base, "n_components": 3, "name": "invalid-count"}
    )
    failure = poll(client, response.json()["job_id"])
    assert failure["status"] == "failed"
    assert failure["error"]["code"] == "validation_failure"


def test_rolling_scope_is_causal_and_static_diagnostics_are_absent(api_env):
    client, settings, catalog = api_env
    source = register(settings, catalog, sample())
    accepted = client.post(
        "/api/exploratory/correlation",
        json={
            "dataset_id": source.dataset_id,
            "features": ["x", "y"],
            "scope": "rolling",
            "rolling_window": 40,
        },
    )
    assert accepted.status_code == 202
    result = poll(client, accepted.json()["job_id"])["result"]
    assert result["causal_for_downstream_training"] is True
    assert result["matrix"] == result["pairwise"] == result["vif"] == []
    assert result["ccf"] == result["conditional"] == result["redundancy_groups"] == []
    assert result["rolling"] == []
    assert result["rolling_summary"]

    selected = client.post(
        "/api/exploratory/correlation",
        json={
            "dataset_id": source.dataset_id,
            "features": ["x", "y"],
            "scope": "rolling",
            "rolling_window": 40,
            "rolling_pair_selection": "selected_pairs",
            "rolling_pairs": [["x", "y"]],
        },
    )
    result = poll(client, selected.json()["job_id"])["result"]
    assert result["rolling"] and result["rolling_summary"]


@pytest.mark.parametrize(
    "extra",
    [
        {"enable_rolling": True, "rolling_window": 40},
        {"scope": "rolling", "rolling_pair_selection": "top_n"},
        {"scope": "rolling", "enable_ccf": True, "target": "y"},
        {"scope": "rolling", "group_column": "group"},
    ],
)
def test_mixed_or_future_selected_rolling_requests_are_422(api_env, extra):
    client, settings, catalog = api_env
    source = register(settings, catalog, sample())
    response = client.post(
        "/api/exploratory/correlation",
        json={"dataset_id": source.dataset_id, "features": ["x", "y"], **extra},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_failure"
