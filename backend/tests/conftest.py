"""Shared fixtures.

Every test runs against a throwaway data directory. Nothing here touches
`backend/data/`, so running the suite can never damage real datasets — which matters
because the whole storage design is built on the promise that a raw dataset is
write-once.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.config import Settings
from app.models.dataset import DatasetMeta
from app.models.enums import AssetClass, DatasetCategory, Provider, Timeframe
from app.storage import parquet_store
from app.storage.catalog import Catalog


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    resolved = Settings(data_dir=tmp_path / "data")
    resolved.ensure_dirs()
    return resolved


@pytest.fixture
def catalog(settings: Settings) -> Catalog:
    instance = Catalog(settings.catalog_path)
    instance.initialise()
    return instance


def hourly_frame(rows: int = 200, *, start: str = "2024-01-01") -> pd.DataFrame:
    """A deterministic OHLCV frame on an exact hourly grid.

    Deterministic is the point: a seeded generator means a checksum comparison across
    two runs is a real assertion rather than a coincidence.
    """
    rng = np.random.default_rng(20240101)
    index = pd.date_range(start=start, periods=rows, freq="1h", tz="UTC")
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, rows))
    return pd.DataFrame(
        {
            "open_time": index,
            "open": close,
            "high": close + np.abs(rng.normal(0, 0.3, rows)),
            "low": close - np.abs(rng.normal(0, 0.3, rows)),
            "close": close,
            "volume": np.abs(rng.normal(1000, 100, rows)),
            "close_time": index + timedelta(hours=1) - timedelta(milliseconds=1),
        }
    )


def register(
    settings: Settings,
    catalog: Catalog,
    frame: pd.DataFrame,
    *,
    name: str = "test dataset",
    category: DatasetCategory = DatasetCategory.RAW,
    source: Provider = Provider.BINANCE,
    asset_class: AssetClass = AssetClass.CRYPTO_SPOT,
    timeframe: Timeframe | None = Timeframe.H1,
    symbol: str | None = "BTCUSDT",
) -> DatasetMeta:
    """Write `frame` as a canonical dataset and record it in the catalog."""
    dataset_id = parquet_store.new_dataset_id()
    written = parquet_store.write_dataset(frame, settings.data_dir, category, dataset_id)
    meta = DatasetMeta(
        dataset_id=dataset_id,
        name=name,
        category=category,
        source=source,
        asset_class=asset_class,
        symbol=symbol,
        market_type=None,
        timeframe=timeframe,
        timezone="UTC" if timeframe is not None else None,
        start_timestamp=written.start_timestamp,
        end_timestamp=written.end_timestamp,
        row_count=written.row_count,
        columns=written.columns,
        parent_dataset_id=None,
        checksum=written.checksum,
        storage_path=str(written.storage_path),
        created_at=datetime.now(UTC),
    )
    catalog.insert_dataset(meta)
    return meta
