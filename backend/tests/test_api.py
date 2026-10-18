"""HTTP surface smoke tests.

These go through the real FastAPI app rather than the service layer, so they cover the
pieces the service tests cannot: request validation, error-to-status mapping, and the
exact JSON field names the frontend's TypeScript types declare. That last one matters
because those types are hand-written — nothing generates them, so a renamed field
would be a silent `undefined` in the browser.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from tests.conftest import hourly_frame


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A fresh app bound to a throwaway data directory.

    Settings are cached and read from the environment at first construction, so the
    cache is cleared on both sides: leaving a tmp-path Settings cached would leak into
    any later test in the same process.
    """
    monkeypatch.setenv("QML_DATA_DIR", str(tmp_path / "api-data"))

    from app.config import get_settings

    get_settings.cache_clear()
    from app.main import create_app

    application = create_app()
    with TestClient(application) as test_client:
        yield test_client
    get_settings.cache_clear()


def upload(client: TestClient, frame: pd.DataFrame, filename: str = "sample.csv"):
    buffer = io.BytesIO(frame.to_csv(index=False).encode("utf-8"))
    return client.post(
        "/api/datasets/upload",
        files={"file": (filename, buffer, "text/csv")},
    )


def test_health_responds(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] in {"ok", "healthy"}


def test_upload_returns_the_full_metadata_contract(client: TestClient) -> None:
    response = upload(client, hourly_frame(30))
    assert response.status_code == 201
    payload = response.json()

    # Every field DatasetMeta declares in the frontend's types.ts.
    for field in (
        "dataset_id",
        "name",
        "category",
        "source",
        "asset_class",
        "symbol",
        "market_type",
        "timeframe",
        "timezone",
        "start_timestamp",
        "end_timestamp",
        "row_count",
        "columns",
        "parent_dataset_id",
        "checksum",
        "storage_path",
        "created_at",
        "unpopulated_fields",
        "provenance",
    ):
        assert field in payload, f"{field} missing from the dataset response"

    assert payload["category"] == "uploaded_data"
    assert payload["row_count"] == 30
    assert isinstance(payload["columns"][0], dict)

    # A CSV carries no timezone, so claiming one would be a fabrication.
    assert payload["timezone"] is None


def test_uploaded_dataset_appears_in_the_listing(client: TestClient) -> None:
    created = upload(client, hourly_frame(20)).json()
    listing = client.get("/api/datasets").json()
    assert created["dataset_id"] in {row["dataset_id"] for row in listing}


def test_listing_can_be_filtered_by_category(client: TestClient) -> None:
    upload(client, hourly_frame(20))
    assert client.get("/api/datasets?category=uploaded_data").json() != []
    assert client.get("/api/datasets?category=raw_data").json() == []


def test_unknown_dataset_is_404_with_our_error_shape(client: TestClient) -> None:
    response = client.get("/api/datasets/DOESNOTEXIST")
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "dataset_not_found"
    assert body["error"]["message"]


def test_page_endpoint_returns_the_declared_envelope(client: TestClient) -> None:
    created = upload(client, hourly_frame(120)).json()
    response = client.get(f"/api/datasets/{created['dataset_id']}/page?offset=0&limit=50")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"dataset_id", "offset", "limit", "total", "rows"}
    assert body["total"] == 120
    assert len(body["rows"]) == 50


def test_preview_endpoint_returns_head_and_tail(client: TestClient) -> None:
    created = upload(client, hourly_frame(120)).json()
    body = client.get(f"/api/datasets/{created['dataset_id']}/preview?head=5&tail=5").json()
    assert len(body["head"]) == 5
    assert len(body["tail"]) == 5
    assert body["row_count"] == 120
    assert len(body["columns"]) > 0


def test_bad_sort_is_a_422_not_a_500(client: TestClient) -> None:
    created = upload(client, hourly_frame(30)).json()
    response = client.get(
        f"/api/datasets/{created['dataset_id']}/page?offset=0&limit=10&sort=nope:desc"
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_failure"


def test_derive_creates_a_child_and_leaves_the_parent_alone(client: TestClient) -> None:
    parent = upload(client, hourly_frame(40)).json()
    response = client.post(
        f"/api/datasets/{parent['dataset_id']}/derive",
        json={
            "name": "child",
            "operation": "edit",
            "edits": [{"row": 0, "column": "close", "value": 4242.0}],
            "column_ops": [],
        },
    )
    assert response.status_code == 201
    child = response.json()
    assert child["parent_dataset_id"] == parent["dataset_id"]
    assert child["category"] == "processed_data"

    unchanged = client.get(f"/api/datasets/{parent['dataset_id']}").json()
    assert unchanged["checksum"] == parent["checksum"]


def test_derive_with_nothing_to_do_is_rejected(client: TestClient) -> None:
    created = upload(client, hourly_frame(20)).json()
    response = client.post(
        f"/api/datasets/{created['dataset_id']}/derive",
        json={"edits": [], "column_ops": []},
    )
    assert response.status_code == 422


def test_rename_changes_the_name_and_keeps_the_checksum(client: TestClient) -> None:
    created = upload(client, hourly_frame(20)).json()
    response = client.patch(
        f"/api/datasets/{created['dataset_id']}", json={"name": "renamed"}
    )
    assert response.status_code == 200
    assert response.json()["name"] == "renamed"
    assert response.json()["checksum"] == created["checksum"]


def test_explorer_rename_action_uses_the_explicit_route(client: TestClient) -> None:
    created = upload(client, hourly_frame(20)).json()
    response = client.post(
        f"/api/datasets/{created['dataset_id']}/rename", json={"name": "renamed_action"}
    )
    assert response.status_code == 200
    assert response.json()["name"] == "renamed_action"
    assert response.json()["checksum"] == created["checksum"]


def test_rename_to_an_empty_name_is_rejected(client: TestClient) -> None:
    created = upload(client, hourly_frame(20)).json()
    response = client.patch(f"/api/datasets/{created['dataset_id']}", json={"name": ""})
    assert response.status_code == 422


def test_delete_removes_the_dataset(client: TestClient) -> None:
    created = upload(client, hourly_frame(20)).json()
    response = client.delete(f"/api/datasets/{created['dataset_id']}")
    assert response.status_code == 200
    assert response.json() == {"deleted": True}
    assert client.get(f"/api/datasets/{created['dataset_id']}").status_code == 404


def test_deleting_an_unknown_dataset_is_404_not_500(client: TestClient) -> None:
    """A stale entry in the explorer is the likeliest way to ask for a dataset that is
    already gone, and `Catalog.delete_dataset` raises a bare `KeyError` for a missing
    row. Unmapped, that surfaced as a 500 with no useful message."""
    response = client.delete("/api/datasets/DOESNOTEXIST")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "dataset_not_found"


def test_deleting_a_parent_with_derivatives_is_refused(client: TestClient) -> None:
    """Deleting a parent would leave its child claiming a provenance that can no longer
    be verified."""
    parent = upload(client, hourly_frame(20)).json()
    client.post(
        f"/api/datasets/{parent['dataset_id']}/derive",
        json={"edits": [{"row": 0, "column": "close", "value": 1.0}], "column_ops": []},
    )
    response = client.delete(f"/api/datasets/{parent['dataset_id']}")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "immutable_dataset"


def test_export_csv_has_a_download_filename(client: TestClient) -> None:
    created = upload(client, hourly_frame(20)).json()
    response = client.get(f"/api/datasets/{created['dataset_id']}/export?format=csv")
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    assert response.headers["content-disposition"].endswith('.csv"')


def test_export_of_an_unknown_format_is_rejected(client: TestClient) -> None:
    created = upload(client, hourly_frame(20)).json()
    response = client.get(f"/api/datasets/{created['dataset_id']}/export?format=xlsx")
    assert response.status_code == 422


def test_upload_of_a_non_csv_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("data.parquet", io.BytesIO(b"not a csv"), "application/octet-stream")},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_failure"


def test_upload_of_an_empty_file_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")},
    )
    assert response.status_code == 422


def test_upload_of_a_header_only_csv_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/datasets/upload",
        files={"file": ("headers.csv", io.BytesIO(b"a,b,c\n"), "text/csv")},
    )
    assert response.status_code == 422


def test_ohlcv_converts_timestamps_to_seconds(client: TestClient) -> None:
    """lightweight-charts takes seconds. Milliseconds would put every bar in the year
    57000 and render an empty chart."""
    created = upload(client, hourly_frame(50)).json()
    body = client.get(f"/api/datasets/{created['dataset_id']}/ohlcv").json()

    assert body["dataset_id"] == created["dataset_id"]
    assert body["total"] == 50
    assert len(body["bars"]) == 50

    first = body["bars"][0]
    assert set(first) == {"time", "open", "high", "low", "close", "volume"}
    # 2024-01-01T00:00:00Z in seconds.
    assert first["time"] == 1704067200


def test_ohlcv_times_are_strictly_ascending(client: TestClient) -> None:
    created = upload(client, hourly_frame(50)).json()
    payload = client.get(f"/api/datasets/{created['dataset_id']}/ohlcv").json()
    times = [bar["time"] for bar in payload["bars"]]
    assert times == sorted(times)
    assert len(set(times)) == len(times), "lightweight-charts rejects duplicate times"


def test_a_zone_less_timestamp_is_resolved_as_utc_by_both_chart_and_quality() -> None:
    """A CSV whose timestamps carry no offset is read back as text, and
    `pd.Timestamp(value).timestamp()` reads a zone-less value as the *machine's* local
    time — so the same upload charted differently in two timezones. The quality checks
    already resolve such a value as UTC (`pd.to_datetime(..., utc=True)`), so the bars
    and the anomaly markers disagreed by the machine's offset everywhere except UTC,
    putting a marker on the wrong candle with nothing in the UI to say so.

    This asserts the two layers agree; the literal values then pin which way.
    """
    from app.services.datasets import _epoch_seconds
    from app.services.quality import _time_seconds

    naive = ["2024-01-01 00:00:00", "2024-01-01 01:00:00"]
    from_chart = [_epoch_seconds(value) for value in naive]
    from_quality = [int(value) for value in _time_seconds(pd.Series(naive))]

    assert from_quality == [1704067200, 1704070800]
    assert from_chart == from_quality


def test_an_unusable_timestamp_is_skipped_rather_than_taking_the_chart_down() -> None:
    """One broken cell in a large file must not turn the whole viewer into a 500. The
    quality overlay is where a defective timestamp is supposed to surface."""
    from app.services.datasets import _epoch_seconds

    assert _epoch_seconds("not a date") is None
    assert _epoch_seconds(float("nan")) is None
    assert _epoch_seconds(None) is None


def test_ohlcv_on_a_dataset_without_prices_is_a_422(client: TestClient) -> None:
    frame = pd.DataFrame(
        {
            "create_time": pd.date_range("2024-01-01", periods=10, freq="1D", tz="UTC"),
            "count_long_short_ratio": [1.0] * 10,
        }
    )
    created = upload(client, frame).json()
    response = client.get(f"/api/datasets/{created['dataset_id']}/ohlcv")
    assert response.status_code == 422
    assert "open" in response.json()["error"]["message"]


def test_ohlcv_reads_capitalised_equity_columns(client: TestClient) -> None:
    """A Yahoo equity frame is stored as Date/Open/High/Low/Close/Adj Close/Volume.
    The viewer matched lower-case names exactly, so it told the user such a file had
    no candle data; the columns are now resolved case-insensitively."""
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=12, freq="1D", tz="UTC"),
            "Open": [100.0 + i for i in range(12)],
            "High": [101.0 + i for i in range(12)],
            "Low": [99.0 + i for i in range(12)],
            "Close": [100.5 + i for i in range(12)],
            "Adj Close": [100.4 + i for i in range(12)],
            "Volume": [1_000 + i for i in range(12)],
        }
    )
    created = upload(client, frame, "AAPL.csv").json()
    response = client.get(f"/api/datasets/{created['dataset_id']}/ohlcv")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total"] == 12
    assert body["bars"][0]["open"] == 100.0
    assert body["bars"][0]["volume"] == 1000.0


def test_quality_reads_capitalised_equity_prices(client: TestClient) -> None:
    """The price checks matched lower-case names too, so an equity file reported
    "no OHLC columns present" and never ran price sanity or extreme returns."""
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=60, freq="1D", tz="UTC"),
            "Open": [100.0] * 60,
            # Row 0 has high below low and a zero price: both must be counted.
            "High": [101.0] * 60,
            "Low": [99.0] * 60,
            "Close": [100.0] * 60,
            "Volume": [1_000.0] * 60,
        }
    )
    frame.loc[0, "High"] = 90.0
    frame.loc[1, "Low"] = 0.0
    created = upload(client, frame, "MSFT.csv").json()
    body = client.get(f"/api/datasets/{created['dataset_id']}/quality").json()
    assert body["counts"]["high_below_low"] == 1
    assert body["counts"]["non_positive_price"] == 1
    assert "Price sanity: no OHLC columns present." not in body["skipped"]


def test_delete_all_removes_every_dataset(client: TestClient) -> None:
    upload(client, hourly_frame(10), "a.csv")
    upload(client, hourly_frame(12), "b.csv")
    assert len(client.get("/api/datasets").json()) == 2

    response = client.delete("/api/datasets")
    assert response.status_code == 200
    assert response.json() == {"deleted": 2}
    assert client.get("/api/datasets").json() == []


def test_quality_endpoint_returns_the_declared_envelope(client: TestClient) -> None:
    created = upload(client, hourly_frame(120)).json()
    body = client.get(f"/api/datasets/{created['dataset_id']}/quality").json()
    assert set(body) == {
        "dataset_id",
        "time_column",
        "markers",
        "counts",
        "skipped",
        "notes",
        "columns",
    }
    assert body["time_column"] == "open_time"


def test_quality_reports_an_unpopulated_timeframe_as_skipped(client: TestClient) -> None:
    """An uploaded CSV has no declared timeframe, so gap detection cannot run. The
    response must say so rather than reporting zero gaps."""
    created = upload(client, hourly_frame(120)).json()
    body = client.get(f"/api/datasets/{created['dataset_id']}/quality").json()
    assert "missing_bar" not in body["counts"]
    assert any("timeframe" in reason for reason in body["skipped"])


def test_dataset_columns_carry_name_dtype_and_nullable(client: TestClient) -> None:
    created = upload(client, hourly_frame(10)).json()
    column = created["columns"][0]
    assert set(column) == {"name", "dtype", "nullable"}


def test_row_count_is_zero_safe_for_an_empty_listing(client: TestClient) -> None:
    assert client.get("/api/datasets").json() == []


def test_cors_allows_the_vite_origin(client: TestClient) -> None:
    """The dev server and the API are different origins, so a misconfigured CORS
    policy would break every request in the browser while curl kept working."""
    response = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_health_reports_a_data_dir_inside_the_test_sandbox(client: TestClient) -> None:
    """A test run must never write into the project's real data directory."""
    reported = Path(client.get("/api/health").json()["data_dir"])
    assert "api-data" in reported.parts
    assert "backend" not in reported.parts, "leaked out of the tmp sandbox"
