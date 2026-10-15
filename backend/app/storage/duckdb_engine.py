"""DuckDB query layer over canonical Parquet.

The data grid must never pull a whole dataset into the browser, so paging, sorting,
and preview windows are all resolved here in SQL and only the requested slice
crosses the wire.

Every identifier that reaches the SQL text is validated against the file's actual
schema first. Column names in this application originate from provider payloads and
user uploads, which is exactly the input you do not interpolate into SQL.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb

from app.core.errors import ValidationFailure


#: Identifiers are only ever drawn from a real schema, but they are still quoted, so
#: a column legitimately named `select` or `weird "name"` works rather than breaking.
def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    # Timestamps must come back in UTC regardless of the host's local timezone,
    # otherwise a dataset read on a machine set to another zone silently shifts.
    con.execute("SET timezone = 'UTC'")
    return con


def _jsonable(value: Any) -> Any:
    """Convert DuckDB values into JSON-safe Python.

    NaN and infinity must become null: Python's `json.dumps` writes a bare `NaN`
    token, which is not valid JSON and which `JSON.parse` in the browser rejects
    outright. A single NaN in one cell would otherwise break the entire grid.
    """
    if value is None:
        return None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Decimal):
        as_float = float(value)
        return as_float if math.isfinite(as_float) else None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray, memoryview)):
        # Binary columns are not renderable in a grid; surface that explicitly
        # rather than shipping a repr that looks like data.
        return f"<{len(bytes(value))} bytes>"
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def describe(path: Path) -> list[tuple[str, str]]:
    """(column_name, duckdb_type) pairs, in file order."""
    con = _connect()
    try:
        rows = con.execute(
            "SELECT column_name, column_type FROM (DESCRIBE SELECT * FROM read_parquet(?))",
            [str(path)],
        ).fetchall()
    finally:
        con.close()
    return [(str(r[0]), str(r[1])) for r in rows]


def count_rows(path: Path) -> int:
    con = _connect()
    try:
        row = con.execute("SELECT count(*) FROM read_parquet(?)", [str(path)]).fetchone()
    finally:
        con.close()
    return int(row[0]) if row else 0


def _order_clause(sort: str | None, valid: set[str]) -> str:
    """Build an ORDER BY clause from a `column` or `column:desc` spec.

    The column is checked against the real schema and then quoted. An unknown name is
    an error rather than a silent fallback to unordered, because a caller that asked
    for a sort and got arbitrary order has no way to notice.
    """
    if not sort:
        return ""
    column, _, direction = sort.partition(":")
    column = column.strip()
    if column not in valid:
        raise ValidationFailure(
            f"Cannot sort by unknown column {column!r}.",
            hint=f"Available columns: {', '.join(sorted(valid))}",
        )
    direction = direction.strip().lower()
    if direction in ("", "asc"):
        keyword = "ASC"
    elif direction == "desc":
        keyword = "DESC"
    else:
        raise ValidationFailure(
            f"Sort direction must be 'asc' or 'desc', got {direction!r}."
        )
    # NULLS LAST in both directions: a missing value is not a small value, and
    # floating nulls to the top on a DESC sort buries the actual extremes.
    return f"ORDER BY {_quote(column)} {keyword} NULLS LAST"


def page(
    path: Path,
    offset: int,
    limit: int,
    sort: str | None = None,
    columns: list[str] | None = None,
) -> tuple[list[dict[str, Any]], int, list[str]]:
    """Return (rows, total_row_count, column_names) for one window of the dataset."""
    if offset < 0:
        raise ValidationFailure("offset must be >= 0")
    if limit <= 0:
        raise ValidationFailure("limit must be > 0")

    con = _connect()
    try:
        available = [name for name, _ in describe(path)]
        valid = set(available)

        projection = "*"
        if columns is not None:
            unknown = [c for c in columns if c not in valid]
            if unknown:
                raise ValidationFailure(
                    f"Unknown column(s): {', '.join(unknown)}",
                    hint=f"Available columns: {', '.join(available)}",
                )
            projection = ", ".join(_quote(c) for c in columns)

        total = count_rows(path)

        order = _order_clause(sort, valid)
        sql = f"SELECT {projection} FROM read_parquet(?) {order} LIMIT ? OFFSET ?"
        cursor = con.execute(sql, [str(path), limit, offset])
        names = [d[0] for d in cursor.description]
        raw = cursor.fetchall()
    finally:
        con.close()

    rows = [
        {name: _jsonable(value) for name, value in zip(names, record, strict=True)}
        for record in raw
    ]
    return rows, total, names


def preview(
    path: Path, head: int, tail: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, list[str]]:
    """JupyterLab-style first-N / last-N window, read in SQL.

    Deliberately does not load the frame: `head`/`tail` on a 40M-row Parquet file must
    cost the same as on a 40-row one.
    """
    con = _connect()
    try:
        total = count_rows(path)
        names = [name for name, _ in describe(path)]

        head_cursor = con.execute(
            "SELECT * FROM read_parquet(?) LIMIT ?", [str(path), head]
        )
        head_names = [d[0] for d in head_cursor.description]
        head_rows = head_cursor.fetchall()

        # Overlapping windows would show the same row twice under two headings and
        # misrepresent how much data exists, so a short file gets head only.
        if total <= head + tail:
            tail_rows: list[tuple[Any, ...]] = []
            tail_names = head_names
        else:
            tail_cursor = con.execute(
                "SELECT * FROM read_parquet(?) LIMIT ? OFFSET ?",
                [str(path), tail, max(0, total - tail)],
            )
            tail_names = [d[0] for d in tail_cursor.description]
            tail_rows = tail_cursor.fetchall()
    finally:
        con.close()

    def convert(names_: list[str], rows_: list[tuple[Any, ...]]) -> list[dict[str, Any]]:
        return [
            {name: _jsonable(value) for name, value in zip(names_, record, strict=True)}
            for record in rows_
        ]

    return convert(head_names, head_rows), convert(tail_names, tail_rows), total, names


def distinct_values(path: Path, column: str, limit: int = 200) -> list[Any]:
    """Distinct values for a low-cardinality column, for filter UI."""
    con = _connect()
    try:
        valid = {name for name, _ in describe(path)}
        if column not in valid:
            raise ValidationFailure(f"Unknown column {column!r}.")
        rows = con.execute(
            f"SELECT DISTINCT {_quote(column)} FROM read_parquet(?) "
            f"ORDER BY 1 NULLS LAST LIMIT ?",
            [str(path), limit],
        ).fetchall()
    finally:
        con.close()
    return [_jsonable(r[0]) for r in rows]
