"""Dataset operations: the layer that ties the catalog to the Parquet files.

Every public function is async and routes its blocking work through
`run_blocking`. That is not decoration — the DuckDB and pandas calls here are
synchronous, and calling them directly from a coroutine stalls the whole server.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from app.config import Settings, get_settings
from app.core.blocking import run_blocking
from app.core.errors import (
    DatasetNotFound,
    ImmutableDataset,
    ValidationFailure,
)
from app.models.dataset import DatasetMeta, DatasetSummary
from app.models.enums import DatasetCategory, Timeframe
from app.services import cleaning
from app.services import feature_catalog as feature_catalogue
from app.services import merge as merge_frames
from app.services import quality as quality_checks
from app.services.cleaning import CleaningPlan
from app.services.feature_catalog import DatasetContext
from app.services.features import FeatureRequest, compute_features
from app.services.merge import JoinSpec
from app.storage import duckdb_engine, parquet_store
from app.storage.catalog import Catalog


@dataclass(frozen=True)
class CellEdit:
    row: int
    column: str
    value: Any


@dataclass(frozen=True)
class ColumnOp:
    op: Literal["rename", "drop", "add", "reorder"]
    column: str
    new_name: str | None = None
    position: int | None = None


class DatasetService:
    """Reads and writes datasets. Holds no per-request state."""

    def __init__(self, settings: Settings | None = None, catalog: Catalog | None = None) -> None:
        self.settings = settings or get_settings()
        self.catalog = catalog or Catalog(self.settings.catalog_path)

    # ------------------------------------------------------------------ lookups

    def _require(self, dataset_id: str) -> DatasetMeta:
        meta = self.catalog.get_dataset(dataset_id)
        if meta is None:
            raise DatasetNotFound(f"No dataset with id {dataset_id}.")
        return meta

    def _path(self, meta: DatasetMeta) -> Path:
        """Resolve from the catalog's recorded path, not by recomputing it.

        Recomputing would silently point at a different file if the storage layout
        ever changed, and would report "missing" for a dataset that is present.
        """
        return resolved_path(meta, self.settings)

    async def list_datasets(
        self, category: DatasetCategory | None = None
    ) -> list[DatasetSummary]:
        return await run_blocking(self.catalog.list_datasets, category)

    async def get(self, dataset_id: str) -> DatasetMeta:
        return await run_blocking(self._require, dataset_id)

    # ------------------------------------------------------------------ reads

    async def preview(
        self, dataset_id: str, head: int = 5, tail: int = 5
    ) -> dict[str, Any]:
        meta = await run_blocking(self._require, dataset_id)
        path = self._path(meta)

        def work() -> dict[str, Any]:
            head_rows, tail_rows, total, _ = duckdb_engine.preview(path, head, tail)
            return {
                "dataset_id": dataset_id,
                "row_count": total,
                "columns": [c.model_dump() for c in meta.columns],
                "head": head_rows,
                "tail": tail_rows,
            }

        return await run_blocking(work)

    async def page(
        self,
        dataset_id: str,
        offset: int,
        limit: int,
        sort: str | None = None,
    ) -> dict[str, Any]:
        meta = await run_blocking(self._require, dataset_id)
        path = self._path(meta)

        # Silently honouring a huge limit is how a browser tab dies. The caller learns
        # what actually happened from the echoed `limit` in the response, which is what
        # the frontend computes its page count from.
        capped = min(limit, self.settings.max_page_size)

        def work() -> dict[str, Any]:
            rows, total, _ = duckdb_engine.page(path, offset, capped, sort)
            return {
                "dataset_id": dataset_id,
                "offset": offset,
                "limit": capped,
                "total": total,
                "rows": rows,
            }

        return await run_blocking(work)

    # --------------------------------------------------------------- mutations

    async def rename(self, dataset_id: str, new_name: str) -> DatasetMeta:
        name = new_name.strip()
        if not name:
            raise ValidationFailure("A dataset name cannot be empty.")

        def work() -> None:
            self._require(dataset_id)
            self.catalog.rename_dataset(dataset_id, name)

        await run_blocking(work)
        return await run_blocking(self._require, dataset_id)

    async def delete(self, dataset_id: str) -> None:
        """Remove a dataset. Refuses to orphan derivatives.

        Deleting a parent would leave its children pointing at a dataset that no
        longer exists — the lineage chain would silently claim provenance it can no
        longer verify.
        """

        def work() -> None:
            # Resolved through `_require` so an unknown id is a 404. `delete_dataset`
            # raises a bare `KeyError` for a missing row, which would otherwise surface
            # as a 500 — and a stale entry in the explorer is the likeliest way to ask
            # for a dataset that is already gone.
            self._require(dataset_id)
            children = self.catalog.children_of(dataset_id)
            if children:
                names = ", ".join(c.name for c in children[:5])
                raise ImmutableDataset(
                    f"'{dataset_id}' has {len(children)} derived dataset(s) and cannot be deleted.",
                    hint=f"Delete the derivatives first: {names}",
                )
            stored = self.catalog.delete_dataset(dataset_id)
            if stored:
                path = Path(stored)
                if not path.is_absolute():
                    path = self.settings.data_dir / path
                path.unlink(missing_ok=True)

        await run_blocking(work)

    async def delete_all(self) -> int:
        """Remove every dataset, returning how many were deleted.

        Children are deleted before their parents even though nothing is being kept:
        the schema's self-referencing foreign key is enforced, so a parent that still
        has a child would otherwise raise instead of being removed. Filtering to
        leaves and repeating terminates for any DAG, and the catalog stores no cycles.
        """

        def work() -> int:
            rows = self.catalog.list_datasets()
            remaining = {row.dataset_id for row in rows}
            deleted = 0
            while remaining:
                parents = {
                    row.parent_dataset_id
                    for row in rows
                    if row.dataset_id in remaining and row.parent_dataset_id in remaining
                }
                leaves = [dataset_id for dataset_id in remaining if dataset_id not in parents]
                # Every remaining dataset is a parent of another only in a cycle, which
                # the append-only lineage model cannot produce. Falling back to the full
                # set removes the block rather than looping forever on it.
                if not leaves:
                    leaves = list(remaining)
                for dataset_id in leaves:
                    remaining.discard(dataset_id)
                    stored = self.catalog.delete_dataset(dataset_id)
                    if stored:
                        path = Path(stored)
                        if not path.is_absolute():
                            path = self.settings.data_dir / path
                        path.unlink(missing_ok=True)
                    deleted += 1
            return deleted

        return await run_blocking(work)

    async def derive(
        self,
        dataset_id: str,
        *,
        name: str | None = None,
        edits: list[CellEdit] | None = None,
        column_ops: list[ColumnOp] | None = None,
        operation: str = "edit",
    ) -> DatasetMeta:
        """Fork a dataset with edits applied.

        This is the only path by which data changes. The parent is never touched,
        which is what makes "raw datasets are permanently read-only" true rather
        than aspirational.
        """
        parent = await run_blocking(self._require, dataset_id)
        path = self._path(parent)
        edits = edits or []
        column_ops = column_ops or []

        if not edits and not column_ops:
            raise ValidationFailure("Nothing to save: no cell edits or column operations.")

        def work() -> DatasetMeta:
            frame = parquet_store.read_dataset(path)
            frame = _apply_column_ops(frame, column_ops, parent)
            frame = _apply_cell_edits(frame, edits)

            return self._persist(
                parent,
                frame,
                name=name or f"{parent.name} (edited)",
                operation=operation,
                detail={
                    "cell_edits": len(edits),
                    "column_ops": [op.op for op in column_ops],
                },
            )

        return await run_blocking(work)

    # ---------------------------------------------------------------- cleaning

    async def clean(
        self,
        dataset_id: str,
        plan: CleaningPlan,
        *,
        name: str | None = None,
        write: bool = True,
    ) -> dict[str, Any]:
        """Apply a cleaning plan, returning the report and a preview of the result.

        With `write=False` nothing is stored: the plan runs, the frame is discarded, and
        the caller sees exactly what saving it would do. That is the difference between
        a repair the user chose and one they discovered afterwards.
        """
        parent = await run_blocking(self._require, dataset_id)
        path = self._path(parent)

        def work() -> dict[str, Any]:
            frame = parquet_store.read_dataset(path)
            time_column = quality_checks.find_time_column(frame)
            cleaned, report = cleaning.clean_frame(frame, plan, time_column)
            report["time_column"] = time_column or ""

            meta: DatasetMeta | None = None
            if write:
                meta = self._persist(
                    parent,
                    cleaned,
                    name=name or f"{parent.name} (cleaned)",
                    operation="clean",
                    detail={
                        "plan": {
                            "duplicates": plan.duplicates.value,
                            "missing": plan.missing.value,
                            "outliers": plan.outliers.value,
                            "outlier_columns": list(plan.outlier_columns),
                            "sort_by_time": plan.sort_by_time,
                        },
                        "rows_removed": report["rows_removed"],
                        "steps": [step["step"] for step in report["steps"]],
                    },
                    # The fences are the one part of a cleaning run that cannot be
                    # reconstructed from the result, so they are stored on the dataset
                    # rather than only reported once.
                    extra_provenance=_fence_provenance(report),
                )
            return {
                "dataset": meta.model_dump(mode="json") if meta is not None else None,
                "report": report,
                "preview": _preview_of(cleaned),
            }

        return await run_blocking(work)

    # ------------------------------------------------------------------ merging

    async def merge(
        self,
        base_dataset_id: str,
        joins: list[JoinSpec],
        *,
        timeframe: Timeframe | None = None,
        name: str | None = None,
        column_ops: list[ColumnOp] | None = None,
        write: bool = True,
    ) -> dict[str, Any]:
        """Resample the base, align every other dataset onto it, and merge.

        Called with `write=False` this is the preview: the whole pipeline runs against
        the real frames and the result is thrown away.

        A join is not required. Resampling to a different timeframe is a first-class
        operation — turning 1-minute bars into 1-hour bars changes the observation
        frequency, which is a data-preparation decision in its own right and the first
        half of what this step is named for. Requiring a join would force a user who
        only wants to resample to attach an unrelated dataset to the request.
        """
        if not joins and timeframe is None:
            raise ValidationFailure(
                "Nothing to do: no target timeframe and no dataset to merge.",
                hint="Choose a timeframe to resample to, or add a dataset to align "
                "onto the base timeline.",
            )

        base = await run_blocking(self._require, base_dataset_id)
        inputs: list[tuple[JoinSpec, DatasetMeta, Path]] = []
        for spec in joins:
            if spec.dataset_id == base_dataset_id:
                raise ValidationFailure(
                    f"{base.name} cannot be merged with itself."
                )
            meta = await run_blocking(self._require, spec.dataset_id)
            inputs.append((spec, meta, self._path(meta)))

        base_path = self._path(base)
        column_ops = column_ops or []

        def work() -> dict[str, Any]:
            frame = parquet_store.read_dataset(base_path)
            time_column = quality_checks.find_time_column(frame)
            if time_column is None:
                raise ValidationFailure(
                    f"{base.name} has no recognisable timestamp column, so there is no "
                    "timeline to align the other datasets onto.",
                    hint="Fetch or upload a dataset with an open_time, create_time, or "
                    "Date column to use as the base.",
                )

            reports: dict[str, Any] = {
                "base": {"dataset_id": base.dataset_id, "name": base.name},
                "resample": None,
                "joins": [],
                "rows_before": int(len(frame)),
                "warnings": [],
            }

            if timeframe is not None:
                frame, resample_report = merge_frames.resample_frame(
                    frame, time_column, timeframe, base.timeframe
                )
                reports["resample"] = resample_report
                reports["warnings"].extend(resample_report.get("warnings", []))

            loaded: list[tuple[JoinSpec, pd.DataFrame, str, str]] = []
            for spec, meta, path in inputs:
                other = parquet_store.read_dataset(path)
                right_time = quality_checks.find_time_column(other)
                loaded.append((spec, other, meta.name, right_time or ""))

            frame, join_reports = merge_frames.align_and_merge(
                frame, time_column, loaded
            )
            reports["joins"] = join_reports
            for join_report in join_reports:
                reports["warnings"].extend(join_report.get("warnings", []))

            if column_ops:
                # Reuses the grid's own column vocabulary, so "remove this column" means
                # the same thing here as it does in Data Management.
                frame = _apply_column_ops(frame, column_ops, base)
                reports["column_ops"] = [op.op for op in column_ops]

            reports["rows_after"] = int(len(frame))

            result_meta: DatasetMeta | None = None
            if write:
                result_meta = self._persist(
                    base,
                    frame,
                    # Named for what actually happened. A resample-only child called
                    # "... (merged)" would describe an input that does not exist.
                    name=name
                    or (f"{base.name} (merged)" if joins else f"{base.name} (resampled)"),
                    operation="merge" if joins else "resample",
                    detail={
                        "timeframe": timeframe.value if timeframe else None,
                        "inputs": [
                            {
                                "dataset_id": spec.dataset_id,
                                "method": spec.method.value,
                                "how": spec.how.value,
                                "prefix": spec.prefix,
                                "tolerance_seconds": spec.tolerance_seconds,
                            }
                            for spec in joins
                        ],
                        "column_ops": [op.op for op in column_ops],
                        "rows_after": reports["rows_after"],
                    },
                    timeframe=timeframe,
                    extra_parents=[
                        (
                            spec.dataset_id,
                            {"role": "merged input", "method": spec.method.value},
                        )
                        for spec in joins
                    ],
                    # Provenance must answer "what is this file", and for a merge the
                    # answer is not recoverable from the frame alone. A resample-only
                    # run has no inputs, so the keys are omitted rather than written as
                    # empty strings — "merge_inputs: ''" reads as a merge whose inputs
                    # were lost.
                    extra_provenance=(
                        {
                            "merge_inputs": ", ".join(spec.dataset_id for spec in joins),
                            "merge_methods": ", ".join(
                                spec.method.value for spec in joins
                            ),
                        }
                        if joins
                        else None
                    ),
                )
            return {
                "dataset": (
                    result_meta.model_dump(mode="json")
                    if result_meta is not None
                    else None
                ),
                "report": reports,
                "preview": _preview_of(frame),
            }

        return await run_blocking(work)

    # ----------------------------------------------------------------- features

    async def feature_catalog(self, dataset_id: str) -> dict[str, Any]:
        """Every catalogue entry, marked available or not *for this dataset*.

        The availability verdict is computed here and never in the frontend. One function
        decides, so a checkbox the UI renders as disabled and a refusal from the compute
        endpoint can never disagree about the reason.

        Nothing but the time column is read from disk, and that only because
        `DatasetMeta` records a dataset's start and end timestamps but not the name of the
        column holding them. Everything else is a decision about the metadata: asset
        class, timeframe, the column list, and which of those columns are known-empty.
        """
        meta = await run_blocking(self._require, dataset_id)
        path = self._path(meta)

        def work() -> dict[str, Any]:
            frame = parquet_store.read_dataset(path)
            time_column = quality_checks.find_time_column(frame)
            context = _feature_context(
                meta,
                has_time=time_column is not None,
                intraday=quality_checks.has_intraday_pattern(frame, time_column),
            )
            groups: list[dict[str, Any]] = []
            for group in feature_catalogue.GROUP_ORDER:
                entries = [
                    feature_catalogue.spec_payload(
                        spec, feature_catalogue.availability(spec, context)
                    )
                    for spec in feature_catalogue.catalogue()
                    if spec.group is group
                ]
                groups.append({"group": group.value, "features": entries})
            return {
                "dataset_id": meta.dataset_id,
                "asset_class": meta.asset_class.value,
                "timeframe": meta.timeframe.value if meta.timeframe else None,
                "timezone": meta.timezone,
                "columns": [column.name for column in meta.columns],
                "unpopulated_fields": list(meta.unpopulated_fields),
                "has_time": context.has_time,
                "groups": groups,
                "available_count": sum(
                    1 for group in groups for entry in group["features"] if entry["available"]
                ),
                "total_count": sum(len(group["features"]) for group in groups),
            }

        return await run_blocking(work)

    async def features(
        self,
        dataset_id: str,
        request: FeatureRequest,
        *,
        name: str | None = None,
        write: bool = True,
    ) -> dict[str, Any]:
        """Apply a feature selection, returning the report and a preview of the result.

        With `write=False` the whole computation runs and the frame is thrown away, so the
        report describes what saving would actually produce rather than what it is expected
        to produce.
        """
        parent = await run_blocking(self._require, dataset_id)
        path = self._path(parent)

        def work() -> dict[str, Any]:
            frame = parquet_store.read_dataset(path)
            time_column = quality_checks.find_time_column(frame)
            context = _feature_context(
                parent,
                has_time=time_column is not None,
                intraday=quality_checks.has_intraday_pattern(frame, time_column),
            )
            result, report = compute_features(
                frame,
                time_column,
                request,
                context,
                timezone=parent.timezone,
            )

            meta: DatasetMeta | None = None
            if write:
                meta = self._persist(
                    parent,
                    result,
                    name=name or f"{parent.name} (features)",
                    operation="features",
                    detail={
                        "selections": [
                            {"feature_id": entry["feature_id"], "window": entry["window"]}
                            for entry in report["added"]
                        ],
                        "generated_columns": report["generated_columns"],
                        "carried_columns": report["carried"],
                        "dropped_columns": report["dropped_columns"],
                        "price_source": report["price_source"]["definition"],
                        "keep_only_selected": request.keep_only_selected,
                        "warmup_bars": report["warmup_bars"],
                        "selection_hash": report["selection_hash"],
                    },
                    extra_provenance={
                        "feature_price_source": report["price_source"]["definition"],
                        "feature_selection_hash": report["selection_hash"],
                    },
                )
            return {
                "dataset": meta.model_dump(mode="json") if meta is not None else None,
                "report": report,
                "preview": _preview_of(result),
            }

        return await run_blocking(work)

    def _persist(
        self,
        parent: DatasetMeta,
        frame: pd.DataFrame,
        *,
        name: str,
        operation: str,
        detail: dict[str, Any],
        extra_provenance: dict[str, str] | None = None,
        timeframe: Timeframe | None = None,
        extra_parents: list[tuple[str, dict[str, Any]]] | None = None,
        category: DatasetCategory = DatasetCategory.PROCESSED,
    ) -> DatasetMeta:
        """Write `frame` as a new dataset and record where it came from.

        Thin delegation to ``persist_dataset``, which is where the work lives. The split
        exists so that services outside this class — Module D writing its regime output,
        and later modules writing their own — can reach the persistence path without
        calling a method whose leading underscore says it is private. One implementation,
        two entry points, and no second copy of the immutability and lineage rules.
        """
        return persist_dataset(
            parent=parent,
            frame=frame,
            data_dir=self.settings.data_dir,
            catalog=self.catalog,
            category=category,
            name=name,
            operation=operation,
            detail=detail,
            extra_provenance=extra_provenance,
            timeframe=timeframe,
            extra_parents=extra_parents,
        )

    # ------------------------------------------------------------------ charts

    async def ohlcv(self, dataset_id: str, limit: int = 20000) -> dict[str, Any]:
        """Bars for the candlestick viewer, oldest first.

        `time` is emitted in **seconds**. lightweight-charts requires seconds while
        the rest of this application speaks milliseconds; converting once, here, keeps
        that factor-of-1000 mismatch from appearing at three different call sites.
        """
        meta = await run_blocking(self._require, dataset_id)
        path = self._path(meta)

        def work() -> dict[str, Any]:
            # Column names are resolved case-insensitively: a Yahoo equity dataset
            # stores `Open`/`High`/`Low`/`Close`, and an exact lower-case membership
            # test rejected it as having no candlestick data.
            names = [c.name for c in meta.columns]
            ohlcv = quality_checks.resolve_ohlcv(names)
            needed = ("open", "high", "low", "close")
            missing = [field for field in needed if field not in ohlcv]
            if missing:
                raise ValidationFailure(
                    f"Dataset has no candlestick data: missing {', '.join(missing)}.",
                    hint="Raw Data View needs OHLC columns. Funding and metrics "
                    "datasets have none.",
                )
            time_column = quality_checks.find_time_column_in(names)
            if time_column is None:
                raise ValidationFailure(
                    "Dataset has no recognisable timestamp column.",
                    hint="Expected one of: open_time, create_time, calc_time, Date.",
                )

            total = duckdb_engine.count_rows(path)
            # Take the most recent `limit` bars when the file is larger than the
            # window. A chart of the first 20k bars of a 5M-row file is not what
            # anyone wants, and `truncated` says so plainly rather than implying the
            # dataset is short.
            offset = max(0, total - limit)
            rows, _, _ = duckdb_engine.page(
                path,
                offset=offset,
                limit=limit,
                sort=f"{time_column}:asc",
            )

            bars: list[dict[str, Any]] = []
            for row in rows:
                seconds = _epoch_seconds(row.get(time_column))
                if seconds is None:
                    continue
                bars.append(
                    {
                        "time": seconds,
                        "open": _as_float(row.get(ohlcv["open"])),
                        "high": _as_float(row.get(ohlcv["high"])),
                        "low": _as_float(row.get(ohlcv["low"])),
                        "close": _as_float(row.get(ohlcv["close"])),
                        "volume": (
                            (_as_float(row.get(ohlcv["volume"])) or 0.0)
                            if "volume" in ohlcv
                            else 0.0
                        ),
                    }
                )

            # lightweight-charts requires strictly ascending, unique times and throws
            # on a violation. De-duplicating here means a dataset with a repeated
            # timestamp still renders, with the defect surfaced by the quality
            # overlay rather than by a blank chart.
            deduped: dict[int, dict[str, Any]] = {}
            for bar in bars:
                deduped[bar["time"]] = bar
            ordered = [deduped[k] for k in sorted(deduped)]

            return {
                "dataset_id": dataset_id,
                "bars": ordered,
                "total": total,
                "truncated": total > len(ordered),
            }

        return await run_blocking(work)

    async def quality(self, dataset_id: str) -> dict[str, Any]:
        meta = await run_blocking(self._require, dataset_id)
        path = self._path(meta)

        def work() -> dict[str, Any]:
            frame = parquet_store.read_dataset(path)
            report = quality_checks.analyse(
                frame, meta.timeframe, asset_class=meta.asset_class
            )
            return {"dataset_id": dataset_id, **report}

        return await run_blocking(work)

    # ------------------------------------------------------------------ export

    async def export(self, dataset_id: str, fmt: str) -> tuple[bytes, str, str]:
        """Return (payload, media_type, filename)."""
        meta = await run_blocking(self._require, dataset_id)
        path = self._path(meta)

        def work() -> tuple[bytes, str, str]:
            frame = parquet_store.read_dataset(path)
            stem = _safe_filename(meta.name)
            if fmt == "csv":
                return parquet_store.to_csv_bytes(frame), "text/csv", f"{stem}.csv"
            if fmt == "json":
                return (
                    parquet_store.to_json_bytes(frame),
                    "application/json",
                    f"{stem}.json",
                )
            if fmt == "parquet":
                return path.read_bytes(), "application/vnd.apache.parquet", f"{stem}.parquet"
            raise ValidationFailure(
                f"Unsupported export format {fmt!r}.", hint="Use csv, json, or parquet."
            )

        return await run_blocking(work)


# ------------------------------------------------------------------- helpers


def resolved_path(meta: DatasetMeta, settings: Settings) -> Path:
    """Where a dataset's Parquet file actually is.

    Read from ``storage_path`` as recorded at write time, never recomputed from the id and
    the category. The two agree today; the point is that if the layout ever changes, every
    dataset written before the change still resolves, and "missing file" keeps meaning the
    file is missing rather than that our arithmetic disagrees with our history.
    """
    path = Path(meta.storage_path)
    if not path.is_absolute():
        path = settings.data_dir / path
    return path


def persist_dataset(
    *,
    parent: DatasetMeta,
    frame: pd.DataFrame,
    data_dir: Path,
    catalog: Catalog,
    category: DatasetCategory,
    name: str,
    operation: str,
    detail: dict[str, Any],
    extra_provenance: dict[str, str] | None = None,
    timeframe: Timeframe | None = None,
    extra_parents: list[tuple[str, dict[str, Any]]] | None = None,
    dataset_id: str | None = None,
) -> DatasetMeta:
    """Write `frame` as a new dataset and record where it came from.

    Synchronous, and called from inside a `run_blocking` work function. Every mutation
    path in this application ends here, so the immutability and lineage rules are applied
    in exactly one place.

    ``dataset_id`` lets a caller that must know the id *before* the frame is written
    supply it. C6 is the case: a split plan carries its own id as a column inside the
    table it writes, so that a plan exported out of this application is still traceable to
    the row it came from — and a plan whose plan-id and dataset-id were two different
    ULIDs would be two identifiers for one object.

    **On the category.** Prepared data — edited, cleaned, resampled, joined, featurised —
    belongs in ``processed_data``: it is still a frame of bars, and it still claims the
    provenance its parent claimed. A regime label series is a different kind of object. It
    is the *product* of an analysis rather than an input to one, its rows are states rather
    than prices, and it will be consumed by a backtest rather than by another cleaning
    step. That is what ``outputs`` is for, and the caller says so explicitly rather than
    the choice being made on its behalf by a hardcoded constant.
    """
    new_id = dataset_id or parquet_store.new_dataset_id()
    written = parquet_store.write_dataset(frame, data_dir, category, new_id)
    meta = DatasetMeta(
        dataset_id=new_id,
        name=name,
        category=category,
        source=parent.source,
        asset_class=parent.asset_class,
        symbol=parent.symbol,
        market_type=parent.market_type,
        timeframe=timeframe if timeframe is not None else parent.timeframe,
        timezone=parent.timezone,
        start_timestamp=written.start_timestamp,
        end_timestamp=written.end_timestamp,
        row_count=written.row_count,
        columns=written.columns,
        parent_dataset_id=parent.dataset_id,
        checksum=written.checksum,
        storage_path=str(written.storage_path),
        created_at=datetime.now(UTC),
        # Recomputed rather than inherited. A join that matched nothing leaves an
        # all-null column that the parent never had, and a column that was entirely
        # null until the user filled it must stop being labelled as such.
        unpopulated_fields=[
            str(column) for column in frame.columns if bool(frame[column].isna().all())
        ],
        provenance={
            **parent.provenance,
            **(extra_provenance or {}),
            "derived_from": parent.dataset_id,
            "operation": operation,
        },
    )
    catalog.insert_dataset(meta)
    catalog.add_lineage(new_id, parent.dataset_id, operation, detail)
    for parent_id, parent_detail in extra_parents or []:
        catalog.add_lineage(new_id, parent_id, operation, parent_detail)
    return meta


def _feature_context(
    meta: DatasetMeta, *, has_time: bool = True, intraday: bool | None = None
) -> DatasetContext:
    """What a feature's availability is decided from — the metadata, plus two observations.

    ``has_time`` and ``intraday`` are the inputs that cannot come from the metadata: a
    dataset records its start and end timestamps but not the name of the column holding
    them, nor whether the bars within a day repeat a clock time. The caller reads the frame
    and passes both answers in.
    """
    return DatasetContext(
        asset_class=meta.asset_class,
        timeframe=meta.timeframe,
        columns=frozenset(column.name for column in meta.columns),
        unpopulated=frozenset(meta.unpopulated_fields),
        category=meta.category.value,
        has_time=has_time,
        intraday=intraday,
    )


def _preview_of(frame: pd.DataFrame) -> dict[str, Any]:
    """JupyterLab-style head/tail window over a frame that has not been stored."""
    head_rows, tail_rows = parquet_store.frame_head_tail(frame, 5, 5)
    return {
        "row_count": int(len(frame)),
        "columns": [c.model_dump() for c in parquet_store.columns_of(frame)],
        "head": head_rows,
        "tail": tail_rows,
    }


def _fence_provenance(report: dict[str, Any]) -> dict[str, str]:
    """Flatten an outlier step's fences into provenance strings.

    Provenance is `dict[str, str]` because it is a note attached to a dataset, not a
    place to store a schema. One entry per column, so a reader asking "what was the
    lower fence on close?" finds an answer without parsing a nested document.
    """
    notes: dict[str, str] = {}
    for step in report.get("steps", []):
        for column, fence in (step.get("fences") or {}).items():
            notes[f"outlier_fence[{column}]"] = (
                f"{fence['rule']} lower={fence['lower']!r} upper={fence['upper']!r}"
            )
    return notes


def _safe_filename(name: str) -> str:
    """Strip characters that are illegal or dangerous in a download filename.

    The browser sanitises much of this anyway, but `..` and path separators are
    worth removing ourselves rather than trusting the client.
    """
    cleaned = "".join(ch if ch.isalnum() or ch in " -_." else "_" for ch in name)
    cleaned = cleaned.strip(" .")
    return cleaned.replace("..", "_") or "dataset"


def _as_float(value: Any) -> float | None:
    """Coerce a cell to float for chart consumption.

    Returns None rather than 0.0 for unparseable input: a zero price would draw a
    plausible-looking spike to the axis, whereas a gap is visibly a gap.
    """
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _epoch_seconds(stamp: Any) -> int | None:
    """UNIX seconds for the chart's time axis, or None if there is no usable instant.

    `pd.Timestamp(value).timestamp()` reads a zone-less value as the *machine's* local
    time, so the same dataset would chart differently in two timezones. An uploaded
    CSV is exactly the case that carries no zone: its timestamp column is whatever text
    the file held, and it stays text. The quality checks already resolve such a value
    as UTC (`quality._time_seconds` uses `pd.to_datetime(..., utc=True)`), so the two
    had to be made to agree — a marker computed one way and a bar placed the other
    would put the anomaly on the wrong candle, by the machine's UTC offset, on some
    machines and not others.

    NaT and unparseable values return None rather than raising: one broken cell in a
    five-million-row file must not take the whole chart down, and skipping it is what
    the quality overlay is there to report.
    """
    if stamp is None:
        return None
    try:
        moment = pd.Timestamp(stamp)
    except (TypeError, ValueError):
        return None
    if pd.isna(moment):
        return None
    if moment.tzinfo is None:
        moment = moment.tz_localize(UTC)
    return int(moment.timestamp())


def _apply_column_ops(
    frame: pd.DataFrame, ops: list[ColumnOp], parent: DatasetMeta
) -> pd.DataFrame:
    # Drops run before renames, adds next, and reorders last. Drops and renames
    # interact: replacing a redundant column with a renamed one means dropping "b" and
    # renaming "a" to "b" in the same save. Nothing is lost by dropping first, because a
    # drop always names a column the parent already has and so cannot depend on a rename
    # having happened. Adds come after for the mirror reason: they occupy names the other
    # two may free. Reorders come last because their position denotes a place in the
    # *final* schema, and name the column as it is called after renaming.
    #
    # Each phase is applied as a *set*, never as a sequence of one-at-a-time moves. The
    # operations are a diff the UI computes — what the screen shows versus the parent —
    # so the same end state can be serialised in more than one order, and a phase whose
    # result depends on the arrival order is a contract that holds for exactly one caller.
    drops = [op for op in ops if op.op == "drop"]
    renames = [op for op in ops if op.op == "rename"]
    adds = [op for op in ops if op.op == "add"]
    reorders = [op for op in ops if op.op == "reorder"]
    unsupported = [op for op in ops if op.op not in ("drop", "rename", "add", "reorder")]
    if unsupported:
        raise ValidationFailure(f"Unsupported column operation {unsupported[0].op!r}.")

    known = set(frame.columns)

    # ------------------------------------------------------------------- drops
    for op in drops:
        if op.column not in known:
            raise ValidationFailure(f"Cannot drop missing column {op.column!r}.")
        if len(known) == 1:
            raise ValidationFailure("Cannot drop the last remaining column.")
        known.discard(op.column)
    if drops:
        frame = frame.drop(columns=[op.column for op in drops])

    # ----------------------------------------------------------------- renames
    mapping: dict[str, str] = {}
    for op in renames:
        if op.column not in known:
            raise ValidationFailure(f"Cannot rename missing column {op.column!r}.")
        if not op.new_name:
            raise ValidationFailure("Rename requires new_name.")
        if op.column in mapping:
            raise ValidationFailure(f"Column {op.column!r} is renamed twice in one save.")
        mapping[op.column] = op.new_name

    if mapping:
        # Applied as one simultaneous mapping rather than one `rename` per op. Renaming
        # `a`->`b` together with `b`->`c` shifts a name along, and it is an ordinary thing
        # to do: the screen shows no conflict, so the frontend allows it. Applied one at a
        # time it depends on which runs first — `a`->`b` before `b`->`c` is refused
        # because `b` is still occupied — so the user would get an error they cannot
        # explain by looking at a grid whose columns are exactly the ones they meant.
        # pandas renames a mapping simultaneously, which removes the ordering rather than
        # picking a winner between two orders that are both defensible.
        final = [mapping.get(name, name) for name in known]
        if len(set(final)) != len(final):
            duplicate = sorted({name for name in final if final.count(name) > 1})[0]
            raise ValidationFailure(
                f"A column named {duplicate!r} already exists.",
                hint="Names must be unique — a duplicate would make column "
                "selection ambiguous.",
            )
        frame = frame.rename(columns=mapping)
        known = set(final)

    # -------------------------------------------------------------------- adds
    for op in adds:
        target = op.new_name or "new_column"
        if target in known:
            raise ValidationFailure(f"A column named {target!r} already exists.")
        # Added empty, as specified. Typed object rather than float so that
        # filling it with strings later does not require a dtype migration.
        frame = frame.assign(**{target: pd.Series([None] * len(frame), dtype="object")})
        known.add(target)

    # ----------------------------------------------------------------- reorders
    if reorders:
        # Placed as a set rather than moved one at a time. A sequence of moves gives a
        # result that depends on the order the operations arrived in, and these are a
        # *diff* the UI computes — a contract that only holds for one particular
        # arrival order is a trap for the next caller. Sorting by the requested index
        # is order-independent. A column the caller did not mention sorts after every
        # column that was placed, keeping its existing position among its peers, which
        # makes a partial list behave as "move these to the front" rather than as an
        # error.
        positions: dict[str, int] = {}
        for op in reorders:
            if op.column not in known:
                raise ValidationFailure(f"Cannot reorder missing column {op.column!r}.")
            if op.position is None or op.position < 0:
                raise ValidationFailure(
                    f"Reorder of {op.column!r} needs a non-negative position."
                )
            positions[op.column] = op.position

        current = list(frame.columns)
        # Each column is sorted as a (requested position, current position, name) triple
        # rather than having its current position looked up by name inside the key
        # function. The triple carries both ranks with it — one source of truth for where
        # a column sits now — and the generator gives the sort key's parameter a type
        # mypy can infer, which it cannot from `sorted`'s overloaded call context.
        ordered = sorted(
            (positions.get(name, len(current)), index, name)
            for index, name in enumerate(current)
        )
        frame = frame[[name for _, _, name in ordered]]
    return frame


def _fit_to_dtype(value: Any, dtype: Any, column: str) -> Any:
    """Fit one edited cell to its column's stored type, or refuse it.

    The grid's cell editor is a text box and its paste path falls back to text, so a
    value arriving here is frequently a *string* even when the column is numeric.
    Handing that straight to pandas is unsafe in two different ways:

    * A value numpy cannot convert — `"abc"` into a float column — raises out of the
      storage layer as an unhandled `ValueError`, which the API reports as a 500 the
      user can do nothing with.
    * A value numpy *can* convert by widening the block — `"1.5"` into a float column,
      or a null into an integer one — silently changes the column's type for every row.
      A `num_trades` column that has quietly become `object` breaks every numeric
      operation built on it, and nothing in the grid would show that it had happened.
      That is a data-integrity failure, not a type quirk.

    So the conversion is explicit, and anything the column cannot hold is refused with
    a message naming the column. Valid input is passed through unchanged: this runs on
    every edit, and it must not be the thing that alters a value.
    """
    if value is None:
        if pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
            raise ValidationFailure(
                f"Column {column!r} cannot store a blank: it is stored as {dtype}, "
                "which has no missing value.",
                hint="Enter a value instead, or drop the column and add a new one.",
            )
        return None

    # Booleans first. `pd.Series(["false"], dtype=bool)` is `True` — numpy's rule is
    # "non-empty string", not "the word false" — so the generic path below would
    # invert this one case rather than reject it, which is the worst outcome of all.
    if pd.api.types.is_bool_dtype(dtype) and isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1"):
            return True
        if lowered in ("false", "0"):
            return False
        raise ValidationFailure(
            f"{value!r} is not a boolean, and column {column!r} holds booleans.",
            hint="Use true or false.",
        )

    # Integers next, because `pd.Series([1.5], dtype="int64")` truncates without
    # complaint. A price typed into a trade-count column must not become 1.
    if pd.api.types.is_integer_dtype(dtype):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValidationFailure(
                f"{value!r} is not a number, and column {column!r} holds whole numbers."
            ) from exc
        if not math.isfinite(number) or not number.is_integer():
            raise ValidationFailure(
                f"{value!r} is not a whole number, and column {column!r} holds whole "
                f"numbers ({dtype}).",
                hint="Storing it would round the value without saying so.",
            )
        return int(number)

    # Timestamps explicitly, rather than through the generic path below. That path
    # would work only by way of pandas' parsing of a dtype *string* such as
    # "datetime64[ns, UTC]", and the conversion it performs is the subtle part: a value
    # carrying an offset must be moved to the column's zone, not relabelled with it.
    # 12:00+05:00 in a UTC column is 07:00, and a relabelling would store it as 12:00
    # while looking entirely reasonable.
    if pd.api.types.is_datetime64_any_dtype(dtype):
        try:
            moment = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise ValidationFailure(
                f"{value!r} is not a timestamp, and column {column!r} holds timestamps."
            ) from exc
        if isinstance(dtype, pd.DatetimeTZDtype):
            # A zone-less value is read as UTC — the same resolution the chart and the
            # quality checks make (DECISIONS §23), so all three agree.
            if moment.tzinfo is None:
                moment = moment.tz_localize(UTC)
            return moment.tz_convert(dtype.tz)
        # A naive column. Nothing in this application stores one — every write goes
        # through `_normalise_timezones` — but an aware value arriving here would
        # otherwise raise out of pandas as an unhandled TypeError.
        return moment.tz_convert(UTC).tz_localize(None) if moment.tzinfo else moment

    # Everything else goes through pandas, so the spellings accepted here are pandas'
    # own rather than a hand-rolled subset. A failure that escapes as a ValueError is
    # reported as a validation failure rather than as a 500.
    try:
        return pd.Series([value], dtype=dtype).iloc[0]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationFailure(
            f"{value!r} cannot be stored in column {column!r}, which holds {dtype}.",
            hint="The column's type comes from the dataset, not from this edit.",
        ) from exc


def _apply_cell_edits(frame: pd.DataFrame, edits: list[CellEdit]) -> pd.DataFrame:
    if not edits:
        return frame
    for edit in edits:
        if edit.column not in frame.columns:
            raise ValidationFailure(f"Edit targets unknown column {edit.column!r}.")
        if not 0 <= edit.row < len(frame):
            raise ValidationFailure(
                f"Edit targets row {edit.row}, outside the dataset's {len(frame)} rows."
            )
    frame = frame.copy()
    for edit in edits:
        series = frame[edit.column]
        fitted = _fit_to_dtype(edit.value, series.dtype, edit.column)
        # `get_loc` is typed as `int | slice | ndarray` because a frame *may* carry
        # duplicate labels, in which case it returns a slice or a boolean mask. A Parquet
        # dataset cannot: duplicates would already have made `frame[edit.column]` above
        # return a frame rather than a Series, and `series.dtype` would have raised.
        position = frame.columns.get_loc(edit.column)
        if not isinstance(position, int):
            raise ValidationFailure(
                f"Column {edit.column!r} is not uniquely named, so a cell in it is ambiguous."
            )
        frame.iat[edit.row, position] = fitted
    return frame
