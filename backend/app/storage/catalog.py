"""SQLite metadata catalog: datasets, column schemas, lineage, and fetch runs.

Parquet holds the *data*; this holds the *meaning*. Keeping them separate is what
lets the Parquet files stay plain, portable, and readable by DuckDB without our
schema getting in the way.

All methods here are synchronous. Callers in the service layer must invoke them
through ``app.core.blocking.run_blocking``.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.models.dataset import ColumnMeta, DatasetMeta, DatasetSummary
from app.models.enums import (
    AssetClass,
    DatasetCategory,
    JobStatus,
    Provider,
    Timeframe,
)

#: Bumped only for a change that requires the user to act — a dropped column, a changed
#: meaning, a table that must be rebuilt. `initialise` refuses to start on a mismatch, so
#: bumping it for an *additive* table would tell every existing installation to migrate or
#: delete its catalog in order to gain a table that `CREATE TABLE IF NOT EXISTS` had
#: already created a moment earlier. The `jobs` table was added that way and the version
#: deliberately did not move. See DECISIONS §76.
SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS datasets (
    dataset_id          TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    category            TEXT NOT NULL,
    source              TEXT NOT NULL,
    asset_class         TEXT NOT NULL,
    symbol              TEXT,
    market_type         TEXT,
    timeframe           TEXT,
    timezone            TEXT,
    start_timestamp     TEXT,
    end_timestamp       TEXT,
    row_count           INTEGER NOT NULL,
    parent_dataset_id   TEXT REFERENCES datasets(dataset_id),
    checksum            TEXT NOT NULL,
    storage_path        TEXT NOT NULL UNIQUE,
    created_at          TEXT NOT NULL,
    unpopulated_fields  TEXT NOT NULL DEFAULT '[]',
    provenance          TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS dataset_columns (
    dataset_id  TEXT NOT NULL REFERENCES datasets(dataset_id) ON DELETE CASCADE,
    ordinal     INTEGER NOT NULL,
    name        TEXT NOT NULL,
    dtype       TEXT NOT NULL,
    nullable    INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (dataset_id, ordinal)
);

-- Append-only. A dataset's lineage must be reconstructible even after the child
-- is deleted, so there is deliberately no cascade from datasets here.
CREATE TABLE IF NOT EXISTS lineage (
    child_id    TEXT NOT NULL,
    parent_id   TEXT NOT NULL,
    operation   TEXT NOT NULL,
    detail      TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fetch_runs (
    run_id       TEXT PRIMARY KEY,
    provider     TEXT NOT NULL,
    endpoint     TEXT NOT NULL,
    params       TEXT NOT NULL,
    status       TEXT NOT NULL,
    dataset_id   TEXT,
    message      TEXT,
    started_at   TEXT NOT NULL,
    finished_at  TEXT
);

-- Background jobs. Module D's regime fit runs hundreds of EM refits and cannot hold a
-- request open while it does, so the work is handed to a task and the caller polls here.
--
-- The row is the durable record, not an in-memory one. A completed job's diagnostics
-- stay readable after a restart, which an in-memory registry cannot promise — and for a
-- research tool the diagnostics *are* the artifact: they are what makes a saved regime
-- dataset interpretable a month later.
CREATE TABLE IF NOT EXISTS jobs (
    job_id            TEXT PRIMARY KEY,
    kind              TEXT NOT NULL,
    status            TEXT NOT NULL,
    dataset_id        TEXT,
    params            TEXT NOT NULL DEFAULT '{}',
    message           TEXT,
    error             TEXT,
    result            TEXT,
    output_dataset_id TEXT,
    created_at        TEXT NOT NULL,
    started_at        TEXT,
    finished_at       TEXT
);

CREATE INDEX IF NOT EXISTS idx_datasets_category ON datasets(category);
CREATE INDEX IF NOT EXISTS idx_datasets_parent   ON datasets(parent_dataset_id);
CREATE INDEX IF NOT EXISTS idx_columns_dataset   ON dataset_columns(dataset_id);
CREATE INDEX IF NOT EXISTS idx_lineage_child     ON lineage(child_id);
CREATE INDEX IF NOT EXISTS idx_lineage_parent    ON lineage(parent_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status       ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created      ON jobs(created_at);
"""


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


class Catalog:
    """Thin, explicit wrapper over the catalog database.

    A single connection guarded by a lock is intentional. This is a single-user
    local application; the alternative (a pool plus per-thread connections) buys
    concurrency we do not have work for and makes the write path harder to reason
    about.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None

    # ---------------------------------------------------------------- lifecycle

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # check_same_thread=False because calls arrive from the asyncio default
            # executor, where the thread is not stable. The RLock is what actually
            # makes that safe.
            conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            # WAL keeps a long read (browsing the explorer) from blocking a write
            # (finishing a fetch).
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            self._conn = conn
        return self._conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def initialise(self) -> None:
        with self._lock:
            self.conn.executescript(_SCHEMA)
            cur = self.conn.execute("SELECT version FROM schema_version LIMIT 1")
            row = cur.fetchone()
            if row is None:
                self.conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,)
                )
            elif row["version"] != SCHEMA_VERSION:
                raise RuntimeError(
                    f"Catalog schema is v{row['version']}, this build expects "
                    f"v{SCHEMA_VERSION}. Migrate or remove {self.path}."
                )
            self.conn.commit()

    # ------------------------------------------------------------------ datasets

    def insert_dataset(self, meta: DatasetMeta) -> None:
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO datasets (
                    dataset_id, name, category, source, asset_class, symbol,
                    market_type, timeframe, timezone, start_timestamp, end_timestamp,
                    row_count, parent_dataset_id, checksum, storage_path, created_at,
                    unpopulated_fields, provenance
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    meta.dataset_id,
                    meta.name,
                    meta.category.value,
                    meta.source.value,
                    meta.asset_class.value,
                    meta.symbol,
                    meta.market_type,
                    meta.timeframe.value if meta.timeframe else None,
                    meta.timezone,
                    meta.start_timestamp.isoformat() if meta.start_timestamp else None,
                    meta.end_timestamp.isoformat() if meta.end_timestamp else None,
                    meta.row_count,
                    meta.parent_dataset_id,
                    meta.checksum,
                    meta.storage_path,
                    meta.created_at.isoformat(),
                    json.dumps(meta.unpopulated_fields),
                    json.dumps(meta.provenance),
                ),
            )
            self.conn.executemany(
                """
                INSERT INTO dataset_columns (dataset_id, ordinal, name, dtype, nullable)
                VALUES (?,?,?,?,?)
                """,
                [
                    (meta.dataset_id, i, c.name, c.dtype, int(c.nullable))
                    for i, c in enumerate(meta.columns)
                ],
            )
            self.conn.commit()

    def get_dataset(self, dataset_id: str) -> DatasetMeta | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)
            ).fetchone()
            if row is None:
                return None
            columns = self.conn.execute(
                "SELECT name, dtype, nullable FROM dataset_columns "
                "WHERE dataset_id = ? ORDER BY ordinal",
                (dataset_id,),
            ).fetchall()
        return _row_to_meta(row, columns)

    def list_datasets(self, category: DatasetCategory | None = None) -> list[DatasetSummary]:
        sql = "SELECT * FROM datasets"
        params: tuple[Any, ...] = ()
        if category is not None:
            sql += " WHERE category = ?"
            params = (category.value,)
        sql += " ORDER BY created_at DESC"
        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
        return [_row_to_summary(r) for r in rows]

    def rename_dataset(self, dataset_id: str, new_name: str) -> None:
        """Rename is metadata-only. Storage paths are derived from the id, so the
        file never moves and the checksum is untouched."""
        with self._lock:
            cur = self.conn.execute(
                "UPDATE datasets SET name = ? WHERE dataset_id = ?", (new_name, dataset_id)
            )
            self.conn.commit()
            if cur.rowcount == 0:
                raise KeyError(dataset_id)

    def delete_dataset(self, dataset_id: str) -> str | None:
        """Remove the catalog row and return its storage path for the caller to unlink.

        Lineage rows are intentionally left in place (see schema comment), so a
        deleted dataset still appears as an ancestor of its derivatives.
        """
        with self._lock:
            row = self.conn.execute(
                "SELECT storage_path FROM datasets WHERE dataset_id = ?", (dataset_id,)
            ).fetchone()
            if row is None:
                raise KeyError(dataset_id)
            self.conn.execute("DELETE FROM datasets WHERE dataset_id = ?", (dataset_id,))
            self.conn.commit()
        return row["storage_path"]

    def children_of(self, dataset_id: str) -> list[DatasetSummary]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM datasets WHERE parent_dataset_id = ? ORDER BY created_at",
                (dataset_id,),
            ).fetchall()
        return [_row_to_summary(r) for r in rows]

    def derived_datasets(self, parent_id: str, operation: str) -> list[DatasetMeta]:
        """Every dataset produced from `parent_id` by one named operation, newest first.

        Queried through `lineage` rather than through `datasets.parent_dataset_id` because
        the question is "which *split plans* were made from this dataset", and the parent
        column cannot answer it: a regime run and a split plan can both be outputs of the
        same dataset. The operation is on the lineage row, which is where the verb lives.

        Full `DatasetMeta` rather than `DatasetSummary` because the caller needs the
        provenance, and the provenance is where a plan records its split type and
        parameters. The result is a handful of rows — one per plan a user has made — so
        reading each one's columns costs nothing worth avoiding.
        """
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT d.* FROM lineage l
                JOIN datasets d ON d.dataset_id = l.child_id
                WHERE l.parent_id = ? AND l.operation = ?
                ORDER BY d.created_at DESC
                """,
                (parent_id, operation),
            ).fetchall()
            # Fetched per row rather than joined, for the same reason `get_dataset` does:
            # one mapper from `dataset_columns` to `ColumnMeta`, so the two paths cannot
            # disagree about ordinal order.
            metas = [
                _row_to_meta(
                    row,
                    self.conn.execute(
                        "SELECT name, dtype, nullable FROM dataset_columns "
                        "WHERE dataset_id = ? ORDER BY ordinal",
                        (row["dataset_id"],),
                    ).fetchall(),
                )
                for row in rows
            ]
        return metas

    def find_by_checksum(
        self, checksum: str, category: DatasetCategory | None = None
    ) -> DatasetMeta | None:
        """Locate an existing dataset with identical content.

        The checksum is sha256 over the stored Parquet bytes, so a match means the
        frames are byte-identical, not merely similar. Used to keep a repeated fetch
        from producing a duplicate.
        """
        sql = "SELECT dataset_id FROM datasets WHERE checksum = ?"
        params: list[Any] = [checksum]
        if category is not None:
            sql += " AND category = ?"
            params.append(category.value)
        sql += " ORDER BY created_at LIMIT 1"
        with self._lock:
            row = self.conn.execute(sql, params).fetchone()
        if row is None:
            return None
        return self.get_dataset(row["dataset_id"])

    # ------------------------------------------------------------------- lineage

    def add_lineage(
        self, child_id: str, parent_id: str, operation: str, detail: dict[str, Any] | None = None
    ) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO lineage (child_id, parent_id, operation, detail, created_at) "
                "VALUES (?,?,?,?,?)",
                (child_id, parent_id, operation, json.dumps(detail or {}), _utcnow()),
            )
            self.conn.commit()

    def ancestors(self, dataset_id: str) -> list[str]:
        """Walk to the root. Bounded by depth so a corrupted cycle cannot hang the
        request thread.

        Follows `datasets.parent_dataset_id` rather than the `lineage` table. A merge
        records one lineage row per input, so `lineage` has several rows for one child
        and "the" parent would be whichever the database happened to return first —
        which is not a property of the data. The column on the dataset row is the single
        authoritative parent, and the extra inputs stay recorded in `lineage` and in the
        child's provenance where they belong.
        """
        chain: list[str] = []
        seen = {dataset_id}
        current = dataset_id
        for _ in range(64):
            with self._lock:
                row = self.conn.execute(
                    "SELECT parent_dataset_id FROM datasets WHERE dataset_id = ?",
                    (current,),
                ).fetchone()
            parent = row["parent_dataset_id"] if row is not None else None
            if parent is None or parent in seen:
                break
            current = parent
            seen.add(current)
            chain.append(current)
        return chain

    # ---------------------------------------------------------------- fetch runs

    def start_run(
        self, run_id: str, provider: Provider, endpoint: str, params: dict[str, Any]
    ) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO fetch_runs (run_id, provider, endpoint, params, status, started_at) "
                "VALUES (?,?,?,?,?,?)",
                (run_id, provider.value, endpoint, json.dumps(params), "running", _utcnow()),
            )
            self.conn.commit()

    def finish_run(
        self,
        run_id: str,
        status: str,
        dataset_id: str | None = None,
        message: str | None = None,
    ) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE fetch_runs SET status = ?, dataset_id = ?, message = ?, finished_at = ? "
                "WHERE run_id = ?",
                (status, dataset_id, message, _utcnow(), run_id),
            )
            self.conn.commit()

    def recent_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM fetch_runs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ---------------------------------------------------------------------- jobs

    def create_job(
        self,
        job_id: str,
        kind: str,
        *,
        dataset_id: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Record a job as running, before any work starts.

        Written first, deliberately: a job that crashed between being accepted and being
        recorded would leave a caller polling an id that does not exist, with no way to
        tell that apart from a typo.
        """
        with self._lock:
            self.conn.execute(
                "INSERT INTO jobs (job_id, kind, status, dataset_id, params, created_at) "
                "VALUES (?,?,?,?,?,?)",
                (
                    job_id,
                    kind,
                    JobStatus.RUNNING.value,
                    dataset_id,
                    json.dumps(params or {}, default=str),
                    _utcnow(),
                ),
            )
            self.conn.commit()

    def start_job(self, job_id: str) -> None:
        """Mark the worker as having picked the job up.

        Distinct from creation because a job can sit queued behind another; without the
        distinction "accepted 40 seconds ago" and "running for 40 seconds" look the same
        to a caller deciding whether to keep waiting.
        """
        with self._lock:
            self.conn.execute(
                "UPDATE jobs SET started_at = ? WHERE job_id = ?", (_utcnow(), job_id)
            )
            self.conn.commit()

    def finish_job(
        self,
        job_id: str,
        *,
        status: JobStatus,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        message: str | None = None,
        output_dataset_id: str | None = None,
    ) -> None:
        """Write a terminal state. The only place a job stops being RUNNING."""
        with self._lock:
            self.conn.execute(
                "UPDATE jobs SET status = ?, result = ?, error = ?, message = ?, "
                "output_dataset_id = ?, finished_at = ? WHERE job_id = ?",
                (
                    status.value,
                    json.dumps(result, default=str) if result is not None else None,
                    json.dumps(error, default=str) if error is not None else None,
                    message,
                    output_dataset_id,
                    _utcnow(),
                    job_id,
                ),
            )
            self.conn.commit()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return None if row is None else _job_row(row)

    def list_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_job_row(row) for row in rows]


# ------------------------------------------------------------------ row mapping


def _row_to_meta(row: sqlite3.Row, columns: Sequence[sqlite3.Row]) -> DatasetMeta:
    return DatasetMeta(
        dataset_id=row["dataset_id"],
        name=row["name"],
        category=DatasetCategory(row["category"]),
        source=Provider(row["source"]),
        asset_class=AssetClass(row["asset_class"]),
        symbol=row["symbol"],
        market_type=row["market_type"],
        timeframe=Timeframe(row["timeframe"]) if row["timeframe"] else None,
        timezone=row["timezone"],
        start_timestamp=_parse_dt(row["start_timestamp"]),
        end_timestamp=_parse_dt(row["end_timestamp"]),
        row_count=row["row_count"],
        columns=[
            ColumnMeta(name=c["name"], dtype=c["dtype"], nullable=bool(c["nullable"]))
            for c in columns
        ],
        parent_dataset_id=row["parent_dataset_id"],
        checksum=row["checksum"],
        storage_path=row["storage_path"],
        created_at=_parse_dt(row["created_at"]),
        unpopulated_fields=json.loads(row["unpopulated_fields"]),
        provenance=json.loads(row["provenance"]),
    )


def _row_to_summary(row: sqlite3.Row) -> DatasetSummary:
    return DatasetSummary(
        dataset_id=row["dataset_id"],
        name=row["name"],
        category=DatasetCategory(row["category"]),
        source=Provider(row["source"]),
        symbol=row["symbol"],
        timeframe=Timeframe(row["timeframe"]) if row["timeframe"] else None,
        row_count=row["row_count"],
        start_timestamp=_parse_dt(row["start_timestamp"]),
        end_timestamp=_parse_dt(row["end_timestamp"]),
        created_at=_parse_dt(row["created_at"]),
        parent_dataset_id=row["parent_dataset_id"],
        operation=json.loads(row["provenance"]).get("operation") or None,
    )


def _parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    # Timestamps are stored with an offset; anything naive would silently be
    # interpreted as local time by the frontend, which is the bug class this
    # whole module is trying to avoid.
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _job_row(row: sqlite3.Row) -> dict[str, Any]:
    """A job row with its JSON columns parsed.

    Parsed here rather than by the caller so the ``NULL``-versus-``{}`` distinction stays
    where it was written: a job with no result has ``result = None``, while a job whose
    result was genuinely an empty object has ``{}``. Collapsing the two would make a
    completed run with nothing to report indistinguishable from one still in flight.
    """
    record = dict(row)
    for key in ("params", "result", "error"):
        raw = record.get(key)
        record[key] = json.loads(raw) if raw else ({} if key == "params" else None)
    return record


def summarise(rows: Iterable[DatasetSummary]) -> dict[str, int]:
    """Count datasets per category — used by the explorer's bucket headers."""
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.category.value] = counts.get(row.category.value, 0) + 1
    return counts
