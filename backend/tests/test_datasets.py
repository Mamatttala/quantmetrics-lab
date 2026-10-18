"""Dataset upload behavior that crosses the CSV/Parquet boundary."""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """A fresh HTTP app and catalog for each upload test."""
    monkeypatch.setenv("QML_DATA_DIR", str(tmp_path / "api-data"))

    from app.config import get_settings

    get_settings.cache_clear()
    from app.main import create_app

    application = create_app()
    with TestClient(application) as test_client:
        yield test_client
    get_settings.cache_clear()


def upload_csv(client: TestClient, csv: str, filename: str = "dates.csv") -> dict:
    response = client.post(
        "/api/datasets/upload",
        files={"file": (filename, io.BytesIO(csv.encode("utf-8")), "text/csv")},
    )
    assert response.status_code == 201, response.text
    return response.json()


def read_uploaded_frame(meta: dict) -> pd.DataFrame:
    return pd.read_parquet(Path(meta["storage_path"]))


def test_non_iso_upload_timestamps_are_stored_as_utc_and_sort_chronologically(
    client: TestClient,
) -> None:
    meta = upload_csv(
        client,
        "Date,open,high,low,close,volume\n"
        "1/10/2024,10,11,9,10,100\n"
        "1/2/2024,20,21,19,20,200\n",
    )

    frame = read_uploaded_frame(meta)
    assert isinstance(frame["Date"].dtype, pd.DatetimeTZDtype)
    assert str(frame["Date"].dtype) == "datetime64[ns, UTC]"

    page = client.get(
        f"/api/datasets/{meta['dataset_id']}/page",
        params={"offset": 0, "limit": 10, "sort": "Date:asc"},
    )
    assert page.status_code == 200
    assert [row["close"] for row in page.json()["rows"]] == [20.0, 10.0]


def test_iso_upload_timestamps_are_still_parsed(client: TestClient) -> None:
    meta = upload_csv(
        client,
        "date,close\n2024-01-02,20\n2024-01-10,10\n",
        filename="iso.csv",
    )

    frame = read_uploaded_frame(meta)
    assert str(frame["date"].dtype) == "datetime64[ns, UTC]"


def test_non_date_values_in_a_timestamp_named_column_remain_text(
    client: TestClient,
) -> None:
    meta = upload_csv(
        client,
        "Date,close\nfirst,20\nsecond,10\n",
        filename="labels.csv",
    )

    frame = read_uploaded_frame(meta)
    assert pd.api.types.is_object_dtype(frame["Date"])
    assert frame["Date"].tolist() == ["first", "second"]
