"""Turning a provider response into a catalogued, immutable raw dataset.

This is the only place a fetched frame becomes a dataset, so it is where the
immutability and deduplication rules are enforced once rather than at each call
site.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from app.config import Settings, get_settings
from app.core.blocking import run_blocking
from app.models.dataset import DatasetMeta
from app.models.enums import (
    AssetClass,
    DatasetCategory,
    Provider,
    Timeframe,
)
from app.storage import parquet_store
from app.storage.catalog import Catalog


def default_name(
    symbol: str,
    timeframe: Timeframe | None,
    asset_class: AssetClass,
    start: datetime | None,
    end: datetime | None,
) -> str:
    """A name that identifies the dataset without the user having to invent one."""
    parts = [symbol]
    if timeframe is not None:
        parts.append(timeframe.value)
    parts.append(asset_class.value.replace("_", " "))
    span = ""
    if start is not None and end is not None:
        span = f"  {start.date()} → {end.date()}"
    return " ".join(parts) + span


class IngestService:
    def __init__(self, settings: Settings | None = None, catalog: Catalog | None = None) -> None:
        self.settings = settings or get_settings()
        self.catalog = catalog or Catalog(self.settings.catalog_path)

    async def register(
        self,
        frame: pd.DataFrame,
        *,
        name: str,
        category: DatasetCategory,
        source: Provider,
        asset_class: AssetClass,
        symbol: str | None,
        market_type: str | None,
        timeframe: Timeframe | None,
        timezone: str | None,
        unpopulated_fields: list[str] | None = None,
        provenance: dict[str, str] | None = None,
        dedupe: bool = True,
    ) -> tuple[DatasetMeta, bool]:
        """Write `frame` as a raw dataset.

        Returns (metadata, deduplicated). With `dedupe=True`, an existing dataset
        whose content hashes identically is returned instead of a duplicate being
        written — re-running the same fetch is a normal thing to do, and silently
        accumulating identical copies would make the explorer useless.
        """
        if frame.empty:
            raise ValueError("Refusing to register an empty frame.")

        def work() -> tuple[DatasetMeta, bool]:
            dataset_id = parquet_store.new_dataset_id()
            written = parquet_store.write_dataset(
                frame, self.settings.data_dir, category, dataset_id
            )

            if dedupe:
                existing = self.catalog.find_by_checksum(written.checksum, category)
                if existing is not None:
                    # Identical content already catalogued. Remove the copy we just
                    # wrote — leaving it would give two datasets with one checksum,
                    # which makes the checksum useless as an identity.
                    written.storage_path.unlink(missing_ok=True)
                    return existing, True

            meta = DatasetMeta(
                dataset_id=dataset_id,
                name=name,
                category=category,
                source=source,
                asset_class=asset_class,
                symbol=symbol,
                market_type=market_type,
                timeframe=timeframe,
                timezone=timezone,
                start_timestamp=written.start_timestamp,
                end_timestamp=written.end_timestamp,
                row_count=written.row_count,
                columns=written.columns,
                parent_dataset_id=None,
                checksum=written.checksum,
                storage_path=str(written.storage_path),
                created_at=datetime.now(UTC),
                unpopulated_fields=unpopulated_fields or [],
                provenance=provenance or {},
            )
            self.catalog.insert_dataset(meta)
            return meta, False

        return await run_blocking(work)

    async def record_run(
        self,
        run_id: str,
        provider: Provider,
        endpoint: str,
        params: dict[str, Any],
        *,
        dataset_id: str | None,
        status: str,
        message: str | None = None,
    ) -> None:
        def work() -> None:
            self.catalog.start_run(run_id, provider, endpoint, params)
            self.catalog.finish_run(run_id, status, dataset_id, message)

        await run_blocking(work)

    async def catalog_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        return await run_blocking(self.catalog.recent_runs, limit)
