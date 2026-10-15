"""C7's HTTP surface: the job handshake, the options call, and the guardrail.

Three things are under test, and the third is the one that matters.

**The handshake.** `POST /api/scaling/fit_transform` returns 202 and a job id. The fit reads
a whole Parquet file and writes a whole one; a 200 here would mean the request waited for it.

**The options call.** One request answers everything the page needs to render itself, and
the availability it reports is the same rule the service refuses on — so a control the page
greys out cannot disagree with a run the backend rejects.

**The guardrail.** A fitted scaler without a split plan is refused. It is refused twice, and
both are asserted: at the schema (422, before the handler runs) and, for a caller that
builds a request some other way, in the service. The schema check is the one the page hits.

These go through the real app because the frontend's types are hand-written — a renamed JSON
field would surface in the browser as `undefined`, not as a compile error.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.test_regime_api import register


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    resolved = tmp_path / "scaling-api-data"
    monkeypatch.setenv("QML_DATA_DIR", str(resolved))
    from app.config import get_settings

    get_settings.cache_clear()
    return resolved


@pytest.fixture
def client(data_dir: Path) -> Iterator[TestClient]:
    from app.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client
    from app.config import get_settings

    get_settings.cache_clear()


@pytest.fixture
def dataset_id(data_dir: Path) -> str:
    from tests.conftest import hourly_frame

    return register(data_dir, hourly_frame(200), name="btc 1h")


@pytest.fixture
def plan_id(client: TestClient, dataset_id: str) -> str:
    response = client.post(
        "/api/splits/create",
        json={"dataset_id": dataset_id, "split_type": "percentage_70_15_15"},
    )
    assert response.status_code == 201, response.text
    return response.json()["dataset"]["dataset_id"]


def run_job(client: TestClient, body: dict[str, Any]) -> dict[str, Any]:
    """POST, then poll the jobs route until the run reaches a terminal state."""
    accepted = client.post("/api/scaling/fit_transform", json=body)
    assert accepted.status_code == 202, accepted.text
    handle = accepted.json()
    assert handle["status"] == "running"
    assert handle["poll"] == f"/api/jobs/{handle['job_id']}"

    for _ in range(600):
        record = client.get(handle["poll"]).json()
        if record["status"] != "running":
            return record
        time.sleep(0.05)
    raise AssertionError("the scaling job never finished")


# ------------------------------------------------------------- the handshake


def test_fit_transform_returns_202_and_a_job(
    client: TestClient, dataset_id: str, plan_id: str
) -> None:
    accepted = client.post(
        "/api/scaling/fit_transform",
        json={
            "dataset_id": dataset_id,
            "split_plan_id": plan_id,
            "scaler_type": "standard",
            "features": ["close"],
        },
    )
    assert accepted.status_code == 202
    assert set(accepted.json()) == {"job_id", "status", "poll"}


def test_the_finished_job_carries_the_frame_and_the_parameters(
    client: TestClient, dataset_id: str, plan_id: str
) -> None:
    record = run_job(
        client,
        {
            "dataset_id": dataset_id,
            "split_plan_id": plan_id,
            "scaler_type": "standard",
            "features": ["close"],
        },
    )
    assert record["status"] == "succeeded"
    result = record["result"]
    assert result["dataset"]["operation"] == "scaling"
    assert result["params_dataset"]["category"] == "outputs"
    assert len(result["parameters"]) == 1
    assert result["preview"]["row_count"] == 200


def test_the_report_names_the_plan_and_the_rows_it_fitted_on(
    client: TestClient, dataset_id: str, plan_id: str
) -> None:
    record = run_job(
        client,
        {
            "dataset_id": dataset_id,
            "split_plan_id": plan_id,
            "scaler_type": "standard",
            "features": ["close"],
        },
    )
    report = record["result"]["report"]
    assert report["split_plan_id"] == plan_id
    assert report["fit_role"] == "train"
    assert report["fit_rows"] == 140
    assert report["scaler_type"] == "standard"


def test_the_parameter_rows_carry_the_affine_pair(
    client: TestClient, dataset_id: str, plan_id: str
) -> None:
    """`center_` and `scale_` are what Modules E and F invert a prediction with."""
    record = run_job(
        client,
        {
            "dataset_id": dataset_id,
            "split_plan_id": plan_id,
            "scaler_type": "standard",
            "features": ["close", "volume"],
        },
    )
    for parameter in record["result"]["parameters"]:
        assert {"feature", "center_", "scale_", "constant", "fitted_rows", "formula"} <= set(
            parameter
        )
        assert parameter["attributes"]["mean_"] is not None


def test_a_rolling_run_needs_no_plan(client: TestClient, dataset_id: str) -> None:
    record = run_job(
        client,
        {"dataset_id": dataset_id, "scaler_type": "rolling_zscore", "features": ["close"]},
    )
    assert record["status"] == "succeeded"
    assert record["result"]["report"]["split_plan_id"] is None
    assert record["result"]["report"]["window"] == 60


def test_the_rolling_window_is_echoed_back(client: TestClient, dataset_id: str) -> None:
    record = run_job(
        client,
        {
            "dataset_id": dataset_id,
            "scaler_type": "rolling_zscore",
            "features": ["close"],
            "window": 30,
        },
    )
    assert record["result"]["report"]["window"] == 30
    assert any("30 bars before it" in note for note in record["result"]["report"]["warnings"])


# ------------------------------------------------------------- the guardrail


def test_minmax_without_a_split_plan_is_a_422(client: TestClient, dataset_id: str) -> None:
    """The specification's first guardrail, asserted at the schema.

    A 422 rather than the specification's 400: `app/core/errors.py` maps nothing to 400, and
    DECISIONS §57 corrected five driver assertions from 400 to 422 for the same reason. What
    matters is that the request is refused before any work happens, not which of the two
    client-error codes carries the refusal.
    """
    response = client.post(
        "/api/scaling/fit_transform",
        json={"dataset_id": dataset_id, "scaler_type": "minmax", "features": ["close"]},
    )
    assert response.status_code == 422
    assert "split_plan_id" in response.text


def test_every_fitted_scaler_is_refused_without_a_plan(client: TestClient, dataset_id: str) -> None:
    for scaler in ("minmax", "standard", "robust"):
        response = client.post(
            "/api/scaling/fit_transform",
            json={"dataset_id": dataset_id, "scaler_type": scaler, "features": ["close"]},
        )
        assert response.status_code == 422, scaler


def test_a_rolling_run_refuses_a_plan(client: TestClient, dataset_id: str, plan_id: str) -> None:
    response = client.post(
        "/api/scaling/fit_transform",
        json={
            "dataset_id": dataset_id,
            "split_plan_id": plan_id,
            "scaler_type": "rolling_zscore",
            "features": ["close"],
        },
    )
    assert response.status_code == 422


def test_a_window_on_a_fitted_scaler_is_a_422(
    client: TestClient, dataset_id: str, plan_id: str
) -> None:
    response = client.post(
        "/api/scaling/fit_transform",
        json={
            "dataset_id": dataset_id,
            "split_plan_id": plan_id,
            "scaler_type": "standard",
            "features": ["close"],
            "window": 30,
        },
    )
    assert response.status_code == 422


def test_an_unknown_dataset_is_a_404(client: TestClient) -> None:
    response = client.post(
        "/api/scaling/fit_transform",
        json={"dataset_id": "nope", "scaler_type": "rolling_zscore", "features": ["close"]},
    )
    assert response.status_code == 404


def test_a_bad_plan_fails_the_job_rather_than_the_request(
    client: TestClient, dataset_id: str
) -> None:
    """The plan is read inside the job, because reading it means opening a file.

    So a plan id that turns out not to be a plan surfaces through the job's `error` field —
    by then there is no request left to attach a status code to.
    """
    record = run_job(
        client,
        {
            "dataset_id": dataset_id,
            "split_plan_id": dataset_id,
            "scaler_type": "standard",
            "features": ["close"],
        },
    )
    assert record["status"] == "failed"
    message = record["error"]["message"]
    assert record["error"]["code"] == "not_a_plan" or "not a split plan" in message


def test_a_feature_that_does_not_exist_fails_the_job(
    client: TestClient, dataset_id: str, plan_id: str
) -> None:
    record = run_job(
        client,
        {
            "dataset_id": dataset_id,
            "split_plan_id": plan_id,
            "scaler_type": "standard",
            "features": ["nope"],
        },
    )
    assert record["status"] == "failed"
    assert "not a column" in record["error"]["message"]


# ------------------------------------------------------------- the options call


def test_options_answers_the_whole_page_in_one_call(client: TestClient, dataset_id: str) -> None:
    body = client.get("/api/scaling/options", params={"dataset_id": dataset_id}).json()
    fields = ("dataset_id", "name", "rows", "features", "defaults", "scalers", "plans", "warnings")
    for field in fields:
        assert field in body, field
    assert body["features"] == ["open", "high", "low", "close", "volume"]
    assert body["defaults"]["features"] == body["features"]
    assert body["defaults"]["window"] == 60


def test_options_offers_the_rolling_scaler_but_not_the_fitted_ones(
    client: TestClient, dataset_id: str
) -> None:
    body = client.get("/api/scaling/options", params={"dataset_id": dataset_id}).json()
    by_id = {entry["id"]: entry for entry in body["scalers"]}
    assert by_id["rolling_zscore"]["available"] is True
    assert by_id["minmax"]["available"] is False
    assert by_id["standard"]["available"] is False
    assert by_id["robust"]["available"] is False
    assert "C6" in by_id["standard"]["reason"]


def test_options_offers_the_fitted_scalers_once_a_plan_exists(
    client: TestClient, dataset_id: str, plan_id: str
) -> None:
    body = client.get("/api/scaling/options", params={"dataset_id": dataset_id}).json()
    assert all(entry["available"] for entry in body["scalers"])
    assert [entry["split_plan_id"] for entry in body["plans"]] == [plan_id]


def test_options_warns_what_scaling_against_a_plan_means(
    client: TestClient, dataset_id: str, plan_id: str
) -> None:
    body = client.get("/api/scaling/options", params={"dataset_id": dataset_id}).json()
    assert any(entry["code"] == "fit_on_train_only" for entry in body["warnings"])


def test_options_for_an_unknown_dataset_is_a_404(client: TestClient) -> None:
    assert client.get("/api/scaling/options", params={"dataset_id": "nope"}).status_code == 404
