"""The features HTTP surface.

These go through the real app for the same reason `test_api.py` does: the frontend's
TypeScript types are hand-written, so a renamed JSON field is a silent `undefined` in the
browser rather than a compile error. Anything the feature page reads by name is asserted
here by name.

An uploaded CSV is a useful subject for these tests because it is the worst case — no
declared timeframe, a placeholder asset class. A feature page that only worked on
perfectly-labelled data would be a feature page that did not work.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from tests.conftest import hourly_frame


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("QML_DATA_DIR", str(tmp_path / "features-api-data"))

    from app.config import get_settings

    get_settings.cache_clear()
    from app.main import create_app

    application = create_app()
    with TestClient(application) as test_client:
        yield test_client
    get_settings.cache_clear()


def upload(client: TestClient, frame: pd.DataFrame, filename: str = "sample.csv") -> str:
    """Upload a frame and return the new dataset's id."""
    buffer = io.BytesIO(frame.to_csv(index=False).encode("utf-8"))
    response = client.post(
        "/api/datasets/upload",
        files={"file": (filename, buffer, "text/csv")},
    )
    assert response.status_code == 201, response.text
    return str(response.json()["dataset_id"])


def daily_frame(rows: int = 40) -> pd.DataFrame:
    close = 100.0 + np.cumsum(np.random.default_rng(7).normal(0, 1, rows))
    return pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-01", periods=rows, freq="1D", tz="UTC"),
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(rows, 5.0),
        }
    )


def compute(client: TestClient, dataset_id: str, body: dict) -> object:
    return client.post(f"/api/datasets/{dataset_id}/features", json=body)


def selection(feature_id: str, window: str | None = None) -> dict:
    return {"feature_id": feature_id, "window": window}


# ------------------------------------------------------------------------- catalog


def test_catalog_lists_every_group_in_order(client: TestClient) -> None:
    dataset_id = upload(client, hourly_frame(60))
    response = client.get("/api/features/catalog", params={"dataset_id": dataset_id})
    assert response.status_code == 200
    payload = response.json()

    assert [group["group"] for group in payload["groups"]] == [
        "price",
        "volume",
        "returns",
        "volatility",
        "indicators",
        "time",
        "exogenous",
    ]
    assert payload["dataset_id"] == dataset_id
    assert payload["total_count"] == payload["available_count"] + sum(
        1
        for group in payload["groups"]
        for entry in group["features"]
        if not entry["available"]
    )


def test_catalog_entries_carry_what_the_info_icon_renders(client: TestClient) -> None:
    dataset_id = upload(client, hourly_frame(60))
    payload = client.get("/api/features/catalog", params={"dataset_id": dataset_id}).json()
    entries = {entry["id"]: entry for group in payload["groups"] for entry in group["features"]}

    bollinger = entries["bollinger"]
    assert bollinger["formula"]
    assert bollinger["detail"]
    assert bollinger["columns"] == ["bb_upper_20_2", "bb_lower_20_2", "bb_width_20_2"]
    assert bollinger["param"] is None
    assert bollinger["warmup_bars"] == 19

    sma = entries["sma"]
    assert sma["param"]["kind"] == "bars"
    assert sma["param"]["default"] == "5"
    assert sma["param"]["custom"] is True
    assert sma["warmup_bars"] is None  # per window, not per feature
    windows = {window["token"]: window for window in sma["windows"]}
    assert windows["20"]["warmup_bars"] == 19


def test_catalog_marks_an_interval_feature_unavailable_without_a_timeframe(
    client: TestClient,
) -> None:
    dataset_id = upload(client, hourly_frame(60))
    payload = client.get("/api/features/catalog", params={"dataset_id": dataset_id}).json()
    entries = {entry["id"]: entry for group in payload["groups"] for entry in group["features"]}
    assert entries["realized_vol"]["available"] is False
    assert "no timeframe" in entries["realized_vol"]["reason"]
    # A bar count needs no declared bar length, so it survives.
    assert entries["sma"]["available"] is True


def test_catalog_reads_the_bars_to_decide_whether_a_clock_time_exists(client: TestClient) -> None:
    """The uploaded hourly file declares no timeframe; its timestamps still answer."""
    hourly = client.get(
        "/api/features/catalog", params={"dataset_id": upload(client, hourly_frame(60))}
    ).json()
    daily = client.get(
        "/api/features/catalog",
        params={"dataset_id": upload(client, daily_frame(), filename="daily.csv")},
    ).json()

    def hour_of_day(payload: dict) -> dict:
        return next(
            entry
            for group in payload["groups"]
            for entry in group["features"]
            if entry["id"] == "hour_of_day"
        )

    assert hour_of_day(hourly)["available"] is True
    assert hour_of_day(daily)["available"] is False
    assert "same time of day" in hour_of_day(daily)["reason"]


def test_catalog_for_an_unknown_dataset_is_404(client: TestClient) -> None:
    response = client.get("/api/features/catalog", params={"dataset_id": "NOPE"})
    assert response.status_code == 404


def test_catalog_requires_a_dataset_id(client: TestClient) -> None:
    assert client.get("/api/features/catalog").status_code == 422


# ------------------------------------------------------------------------ compute


def test_a_preview_does_not_write_a_dataset(client: TestClient) -> None:
    dataset_id = upload(client, hourly_frame(60))
    before = len(client.get("/api/datasets").json())
    response = compute(
        client,
        dataset_id,
        {"selections": [selection("sma", "20")], "write": False},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["dataset"] is None
    assert len(client.get("/api/datasets").json()) == before
    assert "sma_20" in [column["name"] for column in payload["preview"]["columns"]]


def test_a_write_creates_a_processed_child_carrying_the_new_column(client: TestClient) -> None:
    dataset_id = upload(client, hourly_frame(60))
    response = compute(client, dataset_id, {"selections": [selection("sma", "20")]})
    assert response.status_code == 201
    payload = response.json()

    child = payload["dataset"]
    assert child["category"] == "processed_data"
    assert child["parent_dataset_id"] == dataset_id
    assert child["row_count"] == 60
    assert "sma_20" in [column["name"] for column in child["columns"]]

    # The report the page renders.
    report = payload["report"]
    assert report["rows_before"] == report["rows_after"] == 60
    assert report["generated_columns"] == ["sma_20"]
    assert report["warmup_bars"] == 19
    assert report["price_source"]["source"] == "close"
    # The page prints this threshold above the collinearity table, so it is asserted here by
    # name: a field the page reads and no test asserts is a field that can vanish silently
    # into a blank space on screen.
    assert report["collinearity_threshold"] == 0.99
    assert report["selection_hash"]
    assert report["added"][0]["feature_id"] == "sma"
    assert report["added"][0]["columns"] == ["sma_20"]

    # And the column is really in the stored frame, not only in the preview.
    stored = client.get(f"/api/datasets/{child['dataset_id']}/preview", params={"head": 25})
    assert stored.status_code == 200
    assert "sma_20" in [column["name"] for column in stored.json()["columns"]]


def test_the_parent_is_left_untouched(client: TestClient) -> None:
    dataset_id = upload(client, hourly_frame(60))
    before = client.get(f"/api/datasets/{dataset_id}").json()
    compute(client, dataset_id, {"selections": [selection("sma", "20")]})
    after = client.get(f"/api/datasets/{dataset_id}").json()
    assert after["checksum"] == before["checksum"]
    assert "sma_20" not in [column["name"] for column in after["columns"]]


def test_the_child_records_where_it_came_from_and_what_was_asked_for(client: TestClient) -> None:
    dataset_id = upload(client, hourly_frame(60))
    child = compute(client, dataset_id, {"selections": [selection("rsi", "14")]}).json()["dataset"]
    provenance = child["provenance"]
    assert provenance["operation"] == "features"
    assert provenance["derived_from"] == dataset_id
    assert provenance["feature_price_source"] == "close"
    assert provenance["feature_selection_hash"]
    # The upload's own note is inherited rather than dropped.
    assert provenance["uploaded_filename"] == "sample.csv"


def test_keep_only_selected_projects_the_stored_frame(client: TestClient) -> None:
    dataset_id = upload(client, hourly_frame(60))
    response = compute(
        client,
        dataset_id,
        {"selections": [selection("sma", "20")], "keep_only_selected": True},
    )
    assert response.status_code == 201
    columns = [column["name"] for column in response.json()["dataset"]["columns"]]
    assert columns == ["open_time", "sma_20"]
    # `close_time` is here because the uploaded file carried it: `hourly_frame` writes one,
    # an upload stores the CSV's columns as they came, and a projection that kept it would
    # keep a column that says nothing a bar's own timestamp does not. Naming it keeps the
    # assertion complete, which is the point — a report that listed four of the six dropped
    # columns would pass a sloppier version of this test.
    assert set(response.json()["report"]["dropped_columns"]) == {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
    }


def test_many_features_together(client: TestClient) -> None:
    dataset_id = upload(client, hourly_frame(120))
    response = compute(
        client,
        dataset_id,
        {
            "selections": [
                selection("sma", "20"),
                selection("ema", "10"),
                selection("rsi", "14"),
                selection("bollinger"),
                selection("macd"),
                selection("parkinson", "5"),
                selection("atr", "14"),
                selection("obv"),
                selection("day_of_week"),
                selection("price_lag", "1"),
            ]
        },
    )
    assert response.status_code == 201, response.text
    report = response.json()["report"]
    assert len(report["added"]) == 10
    assert report["warmup_bars"] == 33  # MACD's signal line
    assert report["generated_columns"] == [
        "sma_20",
        "ema_10",
        "rsi_14",
        "bb_upper_20_2",
        "bb_lower_20_2",
        "bb_width_20_2",
        "macd_line_12_26",
        "macd_signal_12_26_9",
        "macd_hist_12_26_9",
        "parkinson_5",
        "atr_14",
        "obv",
        "day_of_week",
        "price_lag_1",
    ]


def test_a_custom_window_from_the_blank_box(client: TestClient) -> None:
    dataset_id = upload(client, hourly_frame(120))
    response = compute(client, dataset_id, {"selections": [selection("sma", "37")]})
    assert response.status_code == 201, response.text
    assert response.json()["report"]["generated_columns"] == ["sma_37"]


def test_the_weighted_price_source_is_reported(client: TestClient) -> None:
    dataset_id = upload(client, hourly_frame(60))
    response = compute(
        client,
        dataset_id,
        {"selections": [selection("price_lag", "1")], "price_source": "weighted"},
    )
    assert response.status_code == 201
    report = response.json()["report"]
    assert report["price_source"]["source"] == "weighted"
    assert "typical price" in report["price_source"]["definition"]
    assert any("typical price" in note for note in report["notes"])


def test_an_uploaded_dataset_is_flagged_as_having_a_placeholder_asset_class(
    client: TestClient,
) -> None:
    dataset_id = upload(client, hourly_frame(60))
    report = compute(client, dataset_id, {"selections": [selection("sma", "20")]}).json()["report"]
    assert any("asset class is a placeholder" in note for note in report["notes"])


def test_recursive_features_are_flagged(client: TestClient) -> None:
    dataset_id = upload(client, hourly_frame(60))
    report = compute(client, dataset_id, {"selections": [selection("rsi", "14")]}).json()["report"]
    assert any("recursive" in note for note in report["notes"])


def test_the_warm_up_rows_are_visible_as_empty_in_the_stored_frame(client: TestClient) -> None:
    """The gap is left in place, not filled: it is the one visible sign that the first
    rows of a moving average are not data."""
    dataset_id = upload(client, hourly_frame(60))
    child = compute(client, dataset_id, {"selections": [selection("sma", "20")]}).json()["dataset"]
    page = client.get(
        f"/api/datasets/{child['dataset_id']}/page", params={"offset": 0, "limit": 30}
    ).json()
    rows = page["rows"]
    assert rows[0]["sma_20"] is None  # inside the 19-row warm-up
    assert rows[18]["sma_20"] is None
    assert rows[19]["sma_20"] is not None  # the first row with a full window


# ----------------------------------------------------------------------- refusals


def assert_refused(response: object, fragment: str = "") -> dict:
    assert response.status_code == 422, response.text
    body = response.json()
    assert body["error"]["code"] == "validation_failure"
    assert body["error"]["message"]
    if fragment:
        assert fragment in body["error"]["message"]
    return body


def test_an_empty_selection_is_refused(client: TestClient) -> None:
    dataset_id = upload(client, hourly_frame(60))
    body = assert_refused(compute(client, dataset_id, {"selections": []}), "No features")
    # The hint is where the actionable half of a refusal lives, and the page renders it
    # beside the message — so it is asserted through the shape the page reads.
    assert body["error"]["details"]["hint"]


def test_an_unknown_feature_is_refused(client: TestClient) -> None:
    dataset_id = upload(client, hourly_frame(60))
    assert_refused(
        compute(client, dataset_id, {"selections": [selection("not_a_feature")]}),
        "no feature called",
    )


def test_a_duplicate_selection_is_refused(client: TestClient) -> None:
    dataset_id = upload(client, hourly_frame(60))
    assert_refused(
        compute(
            client,
            dataset_id,
            {"selections": [selection("sma", "20"), selection("sma", "20")]},
        ),
        "twice",
    )


def test_an_interval_window_is_refused_on_a_dataset_with_no_declared_timeframe(
    client: TestClient,
) -> None:
    dataset_id = upload(client, hourly_frame(60))
    assert_refused(
        compute(client, dataset_id, {"selections": [selection("realized_vol", "4h")]}),
        "no timeframe",
    )


def test_annualized_volatility_is_refused_without_a_declared_timeframe(client: TestClient) -> None:
    dataset_id = upload(client, hourly_frame(60))
    assert_refused(
        compute(client, dataset_id, {"selections": [selection("annualized_vol", "7d")]}),
        "bar length",
    )


def test_a_collision_with_an_existing_column_is_refused(client: TestClient) -> None:
    """Silently renaming would make the output schema depend on the input schema."""
    dataset_id = upload(client, hourly_frame(60))
    child = compute(client, dataset_id, {"selections": [selection("sma", "20")]}).json()["dataset"]
    assert_refused(
        compute(client, child["dataset_id"], {"selections": [selection("sma", "20")]}),
        "already exist",
    )


def test_a_window_below_the_minimum_is_refused(client: TestClient) -> None:
    dataset_id = upload(client, hourly_frame(60))
    assert_refused(
        compute(client, dataset_id, {"selections": [selection("sma", "0")]}),
        "at least",
    )


def test_a_window_that_is_not_a_number_is_refused(client: TestClient) -> None:
    dataset_id = upload(client, hourly_frame(60))
    assert_refused(
        compute(client, dataset_id, {"selections": [selection("sma", "twenty")]}),
        "count of candles",
    )


def test_an_unknown_price_source_is_rejected_by_the_request_model(client: TestClient) -> None:
    dataset_id = upload(client, hourly_frame(60))
    response = compute(
        client,
        dataset_id,
        {"selections": [selection("sma", "20")], "price_source": "vwap"},
    )
    assert response.status_code == 422


def test_features_on_an_unknown_dataset_is_404(client: TestClient) -> None:
    response = compute(client, "NOPE", {"selections": [selection("sma", "20")]})
    assert response.status_code == 404
