"""Canonical Parquet storage.

Three guarantees this module exists to provide:

1. **Immutability.** `write_dataset` refuses to overwrite an existing path. Raw and
   uploaded datasets are write-once; edits fork a new dataset.
2. **Content addressing.** The stored `checksum` is the sha256 of the exact bytes on
   disk, so "did this dataset change?" is answerable without trusting the catalog.
3. **Type preservation.** Timestamps are written as `timestamp[ns, tz=UTC]`. CSV
   cannot carry that, which is why CSV is export-only and never canonical.

All functions here are synchronous; callers must go through
`app.core.blocking.run_blocking`.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from app.core.errors import ValidationFailure
from app.models.dataset import ColumnMeta
from app.models.enums import DatasetCategory

# ------------------------------------------------------------------------ ULID

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_dataset_id() -> str:
    """A ULID: 48-bit millisecond timestamp then 80 bits of randomness.

    Lexicographic order matches creation order, which makes directory listings and
    any future range scan on id behave sensibly. Not a UUID — those sort randomly.
    """
    ts = int(time.time() * 1000) & ((1 << 48) - 1)
    rand = secrets.randbits(80)

    chars: list[str] = []
    for shift in range(45, -1, -5):
        chars.append(_CROCKFORD[(ts >> shift) & 0x1F])
    for shift in range(75, -1, -5):
        chars.append(_CROCKFORD[(rand >> shift) & 0x1F])
    return "".join(chars)


# ------------------------------------------------------------------- locations


def dataset_path(root: Path, category: DatasetCategory, dataset_id: str) -> Path:
    """Storage path for a dataset.

    Derived from the *id*, never the name: renaming must not move a file, and two
    datasets sharing a name must not collide.
    """
    return root / category.value / f"{dataset_id}.parquet"


# ----------------------------------------------------------------------- write


@dataclass(frozen=True)
class WrittenDataset:
    storage_path: Path
    checksum: str
    row_count: int
    columns: list[ColumnMeta]
    start_timestamp: datetime | None
    end_timestamp: datetime | None


def _normalise_timezones(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce every datetime column to UTC.

    A tz-naive timestamp column in this application is a bug waiting to happen: it
    will be silently interpreted as local time somewhere downstream. Better to fail
    or coerce here, at the boundary.
    """
    out = frame
    for name in out.columns:
        series = out[name]
        # `isinstance(dtype, DatetimeTZDtype)` rather than the `is_datetime64tz_dtype`
        # helper: that helper is deprecated in pandas 2.1 and the project turns
        # FutureWarnings into errors, so using it would fail the suite outright.
        if isinstance(series.dtype, pd.DatetimeTZDtype):
            out[name] = series.dt.tz_convert(UTC)
        elif pd.api.types.is_datetime64_any_dtype(series.dtype):
            out[name] = series.dt.tz_localize(UTC)
    return out


def _columns_from_schema(schema: pa.Schema) -> list[ColumnMeta]:
    return [
        ColumnMeta(name=field.name, dtype=str(field.type), nullable=field.nullable)
        for field in schema
    ]


def columns_of(frame: pd.DataFrame) -> list[ColumnMeta]:
    """Column metadata for a frame that has not been written yet.

    The Cleaning and Resample steps show the result of a plan *before* it is saved, so
    they need the schema of a frame that has no Parquet file to read it back from.
    """
    return _columns_from_schema(pa.Table.from_pandas(frame, preserve_index=False).schema)


def _detect_time_range(
    frame: pd.DataFrame, columns: list[ColumnMeta]
) -> tuple[datetime | None, datetime | None]:
    """Earliest and latest timestamp across the dataset's datetime columns.

    Uses the union of all timestamp columns rather than assuming a particular one is
    the index — the schema differs by provider, and guessing wrong would silently
    report a narrower range than the data covers.
    """
    starts: list[datetime] = []
    ends: list[datetime] = []
    for column in columns:
        if "timestamp" not in column.dtype:
            continue
        series = frame[column.name].dropna()
        if series.empty:
            continue
        starts.append(series.min().to_pydatetime())
        ends.append(series.max().to_pydatetime())
    if not starts:
        return None, None
    return min(starts), max(ends)


def write_dataset(
    frame: pd.DataFrame,
    root: Path,
    category: DatasetCategory,
    dataset_id: str,
    *,
    overwrite: bool = False,
) -> WrittenDataset:
    """Write `frame` to canonical Parquet and return its content description.

    Refuses to clobber unless `overwrite=True`, which only `processed_data` may pass.
    """
    if frame.empty:
        raise ValidationFailure(
            "Refusing to write an empty dataset.",
            hint="An empty frame usually means the fetch returned nothing; check the "
            "symbol, date range, and timeframe.",
        )

    frame = _normalise_timezones(frame)

    path = dataset_path(root, category, dataset_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        raise ValidationFailure(
            f"Dataset {dataset_id} already exists at {path.name}.",
            hint="Raw and uploaded datasets are immutable. Derive a new dataset instead.",
        )

    table = pa.Table.from_pandas(frame, preserve_index=False)

    # Write to a sibling temp file, hash it, then move into place. A crash mid-write
    # leaves a stray temp file, never a half-written dataset that looks complete.
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd", version="2.6")

    digest = hashlib.sha256()
    with tmp.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    checksum = digest.hexdigest()

    # `Path.replace` is `os.replace`: atomic on the same filesystem, which the temp
    # file is by construction (it is written as a sibling of the target).
    tmp.replace(path)

    columns = _columns_from_schema(table.schema)
    start, end = _detect_time_range(frame, columns)

    return WrittenDataset(
        storage_path=path,
        checksum=checksum,
        row_count=len(frame),
        columns=columns,
        start_timestamp=start,
        end_timestamp=end,
    )


# ------------------------------------------------------------------------ read


def read_dataset(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise ValidationFailure(f"Dataset file is missing: {path}")
    return pd.read_parquet(path, columns=columns)


def read_schema(path: Path) -> list[ColumnMeta]:
    if not path.exists():
        raise ValidationFailure(f"Dataset file is missing: {path}")
    return _columns_from_schema(pq.read_schema(path))


def row_count(path: Path) -> int:
    """Row count from Parquet metadata — no data pages read."""
    if not path.exists():
        raise ValidationFailure(f"Dataset file is missing: {path}")
    return int(pq.ParquetFile(path).metadata.num_rows)


def verify_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------- export


def to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """CSV export.

    Explicitly not the canonical format: this loses timezone and dtype information,
    which is why nothing in the application reads CSV back as a dataset.
    """
    return frame.to_csv(index=False).encode("utf-8")


def to_json_bytes(frame: pd.DataFrame) -> bytes:
    """Record-oriented JSON, ISO-8601 timestamps preserved as UTC strings."""
    return frame.to_json(orient="records", date_format="iso", indent=2).encode("utf-8")


def frame_head_tail(
    frame: pd.DataFrame, head: int, tail: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """JupyterLab-style preview: first `head` rows and last `tail` rows.

    When the frame is short enough that the two windows would overlap, only the head
    is returned — showing the same row twice under two headings misrepresents how
    much data there is.
    """
    head_rows = _records(frame.head(head))
    if len(frame) <= head + tail:
        return head_rows, []
    return head_rows, _records(frame.tail(tail))


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """JSON-safe records.

    NaN and NaT are converted to null. `json.dumps` emits bare `NaN`, which is not
    valid JSON and which JavaScript's `JSON.parse` rejects outright — so leaving it
    would break the grid rather than degrade it.

    The frame is cast to `object` *before* the substitution, because `where` on a
    float column would otherwise put NaN back where the None went.
    """
    safe = frame.astype(object).where(pd.notna(frame), None)
    # `to_dict(orient="records")` is typed as `dict[Hashable, Any]`, because pandas only
    # guarantees a *hashable* column label. Parquet guarantees a string one — the format
    # stores names as strings and this module is the only writer — so the keys are
    # converted here rather than casting the claim onto the result.
    return [
        {str(key): value for key, value in record.items()}
        for record in safe.to_dict(orient="records")
    ]
