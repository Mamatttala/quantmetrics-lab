"""Shared API dataset fixture, independent of any application module."""
from datetime import UTC, datetime
from pathlib import Path
import pandas as pd
from app.models.dataset import DatasetMeta
from app.models.enums import AssetClass, DatasetCategory, Provider, Timeframe
from app.storage import parquet_store
from app.storage.catalog import Catalog

def register(
    data_dir: Path,
    frame: pd.DataFrame,
    *,
    name: str = "btc 1h",
    timeframe: Timeframe | None = Timeframe.H1,
    asset_class: AssetClass = AssetClass.CRYPTO_SPOT,
) -> str:
    """Write a canonical dataset the app can see, and return its id.

    Written straight through the store rather than uploaded, because an uploaded CSV
    declares no timeframe — and Module D *refuses* a dataset without one, since the crisis
    threshold is stated in annualised terms and there is no factor to annualise by. That
    refusal is a guardrail worth its own test (below), but it makes uploads useless as a
    subject for the other thirty assertions here. A Binance-fetched dataset is the realistic
    input anyway.

    The catalog is opened per call rather than shared: SQLite serialises writers on the
    file, and a long-lived connection held by the test thread while the app's own requests
    run on the portal thread is a threading question this test has no business raising.
    """
    from app.config import Settings

    catalog = Catalog(Settings(data_dir=data_dir).catalog_path)
    catalog.initialise()
    dataset_id = parquet_store.new_dataset_id()
    written = parquet_store.write_dataset(frame, data_dir, DatasetCategory.RAW, dataset_id)
    catalog.insert_dataset(
        DatasetMeta(
            dataset_id=dataset_id,
            name=name,
            category=DatasetCategory.RAW,
            source=Provider.BINANCE,
            asset_class=asset_class,
            symbol="BTCUSDT",
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
    )
    return dataset_id


