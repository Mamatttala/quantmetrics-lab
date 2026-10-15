"""Storage guarantees: immutability, content addressing, and path derivation.

These are the invariants the rest of the application is built on. If write-once
stops holding, "raw datasets are permanently read-only" stops being true and every
result computed from a dataset becomes unverifiable after the fact.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from app.config import Settings
from app.core.errors import ValidationFailure
from app.models.enums import DatasetCategory
from app.storage import parquet_store
from tests.conftest import hourly_frame


def test_writing_twice_to_the_same_id_is_refused(settings: Settings) -> None:
    frame = hourly_frame(10)
    dataset_id = parquet_store.new_dataset_id()
    parquet_store.write_dataset(frame, settings.data_dir, DatasetCategory.RAW, dataset_id)

    with pytest.raises(ValidationFailure, match="already exists"):
        parquet_store.write_dataset(
            frame, settings.data_dir, DatasetCategory.RAW, dataset_id
        )


def test_refused_overwrite_leaves_the_original_bytes_untouched(settings: Settings) -> None:
    """The check must happen *before* anything is written.

    A guard that raised after truncating would pass a naive "does it raise?" test
    while destroying the data it claims to protect.
    """
    original = hourly_frame(10)
    dataset_id = parquet_store.new_dataset_id()
    written = parquet_store.write_dataset(
        original, settings.data_dir, DatasetCategory.RAW, dataset_id
    )
    before = written.storage_path.read_bytes()

    with pytest.raises(ValidationFailure):
        parquet_store.write_dataset(
            hourly_frame(999), settings.data_dir, DatasetCategory.RAW, dataset_id
        )

    assert written.storage_path.read_bytes() == before
    assert parquet_store.row_count(written.storage_path) == 10


def test_overwrite_flag_permits_replacement(settings: Settings) -> None:
    dataset_id = parquet_store.new_dataset_id()
    first = parquet_store.write_dataset(
        hourly_frame(10), settings.data_dir, DatasetCategory.PROCESSED, dataset_id
    )
    second = parquet_store.write_dataset(
        hourly_frame(25),
        settings.data_dir,
        DatasetCategory.PROCESSED,
        dataset_id,
        overwrite=True,
    )
    assert first.checksum != second.checksum
    assert second.row_count == 25


def test_checksum_matches_the_bytes_on_disk(settings: Settings) -> None:
    written = parquet_store.write_dataset(
        hourly_frame(50),
        settings.data_dir,
        DatasetCategory.RAW,
        parquet_store.new_dataset_id(),
    )
    actual = hashlib.sha256(written.storage_path.read_bytes()).hexdigest()
    assert written.checksum == actual
    assert parquet_store.verify_checksum(written.storage_path) == actual


def test_path_is_derived_from_id_and_category_not_name(settings: Settings) -> None:
    root = settings.data_dir
    a = parquet_store.dataset_path(root, DatasetCategory.RAW, "01ABC")
    b = parquet_store.dataset_path(root, DatasetCategory.PROCESSED, "01ABC")

    assert a.name == "01ABC.parquet"
    assert a.parent.name == "raw_data"
    assert b.parent.name == "processed_data"
    # Same id, different category, different file — so a rename can never collide.
    assert a != b


def test_dataset_ids_are_unique_and_sort_by_creation() -> None:
    ids = [parquet_store.new_dataset_id() for _ in range(500)]
    assert len(set(ids)) == 500
    assert all(len(i) == 26 for i in ids), "ULIDs are 26 Crockford base32 characters"
    # Time-sortable: the first 10 characters encode the millisecond timestamp, so the
    # sequence must never go backwards. Generation order and lexicographic order agree,
    # which is what makes an id usable as a tiebreaker and a range key.
    prefixes = [i[:10] for i in ids]
    assert prefixes == sorted(prefixes)


def test_naive_timestamps_are_localised_to_utc(settings: Settings) -> None:
    """A tz-naive column is a bug waiting to happen, so it is coerced at the boundary."""
    naive = pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-01", periods=3, freq="1h"),
            "close": [1.0, 2.0, 3.0],
        }
    )
    written = parquet_store.write_dataset(
        naive, settings.data_dir, DatasetCategory.RAW, parquet_store.new_dataset_id()
    )
    dtypes = {c.name: c.dtype for c in written.columns}
    assert "UTC" in dtypes["open_time"]


def test_timezone_aware_column_is_converted_not_relabelled(settings: Settings) -> None:
    """Conversion must move the instant, not just rename the zone.

    12:00 in +05:00 is 07:00 UTC. `tz_localize` here would produce 12:00 UTC — a
    five-hour error that no downstream check would catch.
    """
    frame = pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-01 12:00", periods=1, freq="1h", tz="Asia/Karachi"),
            "close": [1.0],
        }
    )
    written = parquet_store.write_dataset(
        frame, settings.data_dir, DatasetCategory.RAW, parquet_store.new_dataset_id()
    )
    stored = parquet_store.read_dataset(written.storage_path)
    assert stored["open_time"].iloc[0].hour == 7
    assert str(stored["open_time"].dt.tz) == "UTC"


def test_empty_frame_is_refused(settings: Settings) -> None:
    with pytest.raises(ValidationFailure, match="empty"):
        parquet_store.write_dataset(
            pd.DataFrame({"close": []}),
            settings.data_dir,
            DatasetCategory.RAW,
            parquet_store.new_dataset_id(),
        )


def test_head_tail_does_not_repeat_rows_on_a_short_frame() -> None:
    frame = hourly_frame(8)
    head, tail = parquet_store.frame_head_tail(frame, 5, 5)
    assert len(head) == 5
    # 8 rows cannot yield a 5+5 split without showing three of them twice.
    assert tail == []


def test_head_tail_splits_a_long_frame() -> None:
    frame = hourly_frame(100)
    head, tail = parquet_store.frame_head_tail(frame, 5, 5)
    assert len(head) == 5
    assert len(tail) == 5
    assert head[0]["close"] != tail[0]["close"]


def test_nan_is_exported_as_json_null() -> None:
    """`json.dumps` writes a bare `NaN`, which `JSON.parse` in the browser rejects.

    A single null close would otherwise break the whole grid rather than one cell.
    """
    import json

    frame = pd.DataFrame({"close": [1.0, float("nan")]})
    payload = json.loads(parquet_store.to_json_bytes(frame))
    assert payload[1]["close"] is None


def test_read_missing_file_raises_rather_than_returning_empty(tmp_path: Path) -> None:
    with pytest.raises(ValidationFailure, match="missing"):
        parquet_store.read_dataset(tmp_path / "nope.parquet")
