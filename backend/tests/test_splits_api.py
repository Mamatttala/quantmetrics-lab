"""C6's HTTP surface: the 201 envelope, the read endpoint, and the refusals.

The route is exercised through the real app rather than by calling the service, for the
reason `test_features_api.py` and `test_regime_api.py` do the same: the frontend's types are
hand-written, so a renamed JSON field surfaces in the browser as `undefined` rather than as a
compile error. Every field `SplitsPage.tsx` reads is therefore asserted by name here.

The status code itself is under test too. `POST /api/splits/create` returns **201**, not
202 — the plan is finished when the response is sent, and a caller that rendered a 202 as a
result would show an empty table. Asserting the exact code is the only way to hold that.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.test_regime_api import register


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    resolved = tmp_path / "splits-api-data"
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


def create(client: TestClient, dataset: str, **body: Any) -> dict[str, Any]:
    payload = {"dataset_id": dataset, "split_type": "percentage_70_15_15", **body}
    response = client.post("/api/splits/create", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def test_create_returns_201_not_202(client: TestClient, dataset_id: str) -> None:
    """201, because the work is done. A 202 would mean the plan is not in the body."""
    response = client.post(
        "/api/splits/create",
        json={"dataset_id": dataset_id, "split_type": "percentage_70_15_15"},
    )
    assert response.status_code == 201


def test_the_envelope_carries_the_dataset_the_plan_and_the_report(
    client: TestClient, dataset_id: str
) -> None:
    body = create(client, dataset_id)
    assert set(body) == {"dataset", "plan", "report"}
    assert body["dataset"]["operation"] == "split"
    assert body["dataset"]["category"] == "outputs"
    assert isinstance(body["plan"], list)
    assert body["report"]["rows"] == 200


def test_every_plan_row_field_the_page_reads_is_present(
    client: TestClient, dataset_id: str
) -> None:
    body = create(client, dataset_id)
    row = body["plan"][0]
    # The table columns in SplitsPage: Fold | Role | Start | End | Rows.
    for field in (
        "split_id",
        "fold",
        "role",
        "start_index",
        "end_index",
        "start_time",
        "end_time",
        "row_count",
        "dataset_checksum",
    ):
        assert field in row, field


def test_the_report_carries_everything_the_page_renders(
    client: TestClient, dataset_id: str
) -> None:
    report = create(client, dataset_id)["report"]
    for field in (
        "split_type",
        "time_column",
        "rows",
        "dataset_checksum",
        "parameters",
        "folds",
        "rows_by_role",
        "rows_tested",
        "rows_never_tested",
        "boundaries",
        "warnings",
    ):
        assert field in report, field


def test_tscv_parameters_round_trip_through_the_api(client: TestClient, dataset_id: str) -> None:
    body = create(
        client,
        dataset_id,
        split_type="tscv_expanding",
        parameters={"n_folds": 3, "gap_size": 2, "purge_flag": True, "label_horizon_bars": 5},
    )
    assert body["report"]["folds"] == 3
    assert body["report"]["parameters"]["n_folds"] == 3
    assert body["report"]["boundaries"][0]["gap_rows_removed"] == 2
    assert len(body["plan"]) == 6


def test_a_date_split_accepts_its_dates(client: TestClient, dataset_id: str) -> None:
    body = create(
        client,
        dataset_id,
        split_type="date_2way",
        parameters={
            "train_start": "2024-01-01T00:00:00Z",
            "train_end": "2024-01-04T00:00:00Z",
            "test_start": "2024-01-04T01:00:00Z",
            "test_end": "2024-01-09T00:00:00Z",
        },
    )
    assert [row["role"] for row in body["plan"]] == ["train", "test"]
    assert body["plan"][0]["row_count"] == 73


def test_a_zoneless_date_is_a_422(client: TestClient, dataset_id: str) -> None:
    response = client.post(
        "/api/splits/create",
        json={
            "dataset_id": dataset_id,
            "split_type": "date_2way",
            "parameters": {
                "train_start": "2024-01-01T00:00:00",
                "train_end": "2024-01-04T00:00:00",
                "test_start": "2024-01-04T01:00:00",
                "test_end": "2024-01-09T00:00:00",
            },
        },
    )
    assert response.status_code == 422


def test_gap_on_a_date_split_is_a_422(client: TestClient, dataset_id: str) -> None:
    response = client.post(
        "/api/splits/create",
        json={
            "dataset_id": dataset_id,
            "split_type": "date_2way",
            "parameters": {
                "train_start": "2024-01-01T00:00:00Z",
                "train_end": "2024-01-04T00:00:00Z",
                "test_start": "2024-01-04T01:00:00Z",
                "test_end": "2024-01-09T00:00:00Z",
                "gap_size": 3,
            },
        },
    )
    assert response.status_code == 422


def test_an_unknown_split_type_is_a_422(client: TestClient, dataset_id: str) -> None:
    response = client.post(
        "/api/splits/create",
        json={"dataset_id": dataset_id, "split_type": "fifty_fifty"},
    )
    assert response.status_code == 422


def test_an_unknown_dataset_is_a_404(client: TestClient) -> None:
    response = client.post(
        "/api/splits/create",
        json={"dataset_id": "nope", "split_type": "percentage_70_15_15"},
    )
    assert response.status_code == 404


def test_an_unsorted_dataset_is_a_422(client: TestClient, data_dir: Path) -> None:
    from tests.conftest import hourly_frame

    frame = hourly_frame(200).iloc[::-1].reset_index(drop=True)
    dataset = register(data_dir, frame, name="reversed")
    response = client.post(
        "/api/splits/create",
        json={"dataset_id": dataset, "split_type": "percentage_70_15_15"},
    )
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"]
    assert "not sorted" in body["error"]["message"]


def test_plans_lists_what_was_made(client: TestClient, dataset_id: str) -> None:
    assert client.get("/api/splits/plans", params={"dataset_id": dataset_id}).json() == []
    created = create(client, dataset_id)
    listed = client.get("/api/splits/plans", params={"dataset_id": dataset_id}).json()
    assert len(listed) == 1
    assert listed[0]["split_plan_id"] == created["dataset"]["dataset_id"]
    assert listed[0]["split_type"] == "percentage_70_15_15"


def test_plans_for_an_unknown_dataset_is_a_404(client: TestClient) -> None:
    assert client.get("/api/splits/plans", params={"dataset_id": "nope"}).status_code == 404


def test_the_source_dataset_is_not_listed_as_a_plan(client: TestClient, dataset_id: str) -> None:
    create(client, dataset_id)
    listed = client.get("/api/splits/plans", params={"dataset_id": dataset_id}).json()
    assert dataset_id not in [entry["split_plan_id"] for entry in listed]


def test_the_plan_records_the_utc_timestamps_the_page_formats(
    client: TestClient, dataset_id: str
) -> None:
    row = create(client, dataset_id)["plan"][0]
    start = datetime.fromisoformat(row["start_time"])
    assert start.tzinfo is not None
    assert start.astimezone(UTC) == datetime(2024, 1, 1, tzinfo=UTC)
