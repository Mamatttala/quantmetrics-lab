"""Fetch endpoints for A1 · Raw Data Fetch.

Each handler validates parameters, calls a provider, and registers the result as an
immutable raw dataset. The response reports whether the content was deduplicated, so
a repeated request is visibly a no-op rather than a silently growing pile of copies.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from app.core.errors import ValidationFailure
from app.models.dataset import DatasetMeta
from app.models.enums import (
    AssetClass,
    DatasetCategory,
    Provider,
    Timeframe,
)
from app.services import binance_fetch, yahoo_fetch
from app.services.ingest import IngestService, default_name
from app.storage import parquet_store

router = APIRouter(prefix="/fetch", tags=["fetch"])


class FetchResult(BaseModel):
    dataset: DatasetMeta
    deduplicated: bool
    warnings: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------- crypto


class CryptoKlinesRequest(BaseModel):
    asset_class: AssetClass
    symbol: str = Field(min_length=1)
    timeframe: Timeframe
    start: datetime
    end: datetime


@router.post("/binance/klines", response_model=FetchResult, status_code=201)
async def fetch_binance_klines(body: CryptoKlinesRequest) -> FetchResult:
    if body.asset_class not in (AssetClass.CRYPTO_SPOT, AssetClass.CRYPTO_FUTURES):
        raise ValidationFailure(
            f"{body.asset_class.value} is not a Binance market.",
            hint="Use crypto_spot or crypto_futures.",
        )

    frame = await binance_fetch.fetch_klines(
        body.asset_class, body.symbol, body.timeframe, body.start, body.end
    )
    symbol = binance_fetch.normalise_symbol(body.symbol)
    market_type = (
        "spot" if body.asset_class == AssetClass.CRYPTO_SPOT else "futures"
    )

    service = IngestService()
    run_id = parquet_store.new_dataset_id()

    meta, deduped = await service.register(
        frame,
        name=default_name(symbol, body.timeframe, body.asset_class, body.start, body.end),
        category=DatasetCategory.RAW,
        source=Provider.BINANCE,
        asset_class=body.asset_class,
        symbol=symbol,
        market_type=market_type,
        timeframe=body.timeframe,
        timezone="UTC",
        provenance={
            "endpoint": "binance REST klines",
            "market": market_type,
            "note": "open_time is the bar's opening instant; close_time is its close.",
        },
    )

    # Recorded after the dataset is committed, and carrying its id. The other order —
    # a run marked "succeeded" before `register` had run — would leave a success in the
    # run log for a fetch that then failed to store anything.
    await service.record_run(
        run_id, Provider.BINANCE, "/klines", body.model_dump(mode="json"),
        dataset_id=meta.dataset_id, status="succeeded",
    )

    warnings: list[str] = []
    if deduped:
        warnings.append("Identical content was already catalogued; existing dataset returned.")

    return FetchResult(dataset=meta, deduplicated=deduped, warnings=warnings)


class CryptoFundingRequest(BaseModel):
    symbol: str = Field(min_length=1)
    start: datetime
    end: datetime


@router.post("/binance/funding", response_model=FetchResult, status_code=201)
async def fetch_binance_funding(body: CryptoFundingRequest) -> FetchResult:
    frame = await binance_fetch.fetch_funding(body.symbol, body.start, body.end)
    symbol = binance_fetch.normalise_symbol(body.symbol)

    service = IngestService()
    run_id = parquet_store.new_dataset_id()
    meta, deduped = await service.register(
        frame,
        name=f"{symbol} funding rate  {body.start.date()} → {body.end.date()}",
        category=DatasetCategory.RAW,
        source=Provider.BINANCE,
        asset_class=AssetClass.CRYPTO_FUTURES,
        symbol=symbol,
        market_type="futures",
        timeframe=None,
        timezone="UTC",
        provenance={
            "endpoint": "data.binance.vision fundingRate dump",
            "note": "funding_interval_hours is supplied by the source, not derived.",
        },
    )

    # Recorded here for the same reason as the klines handler above — and it was the
    # handler that *had* it. With only klines calling `record_run`, the run log said
    # that funding, metrics, and Yahoo fetches had never happened: `/fetch/runs` would
    # report a history with three quarters of its entries missing, and a gap in an
    # audit log reads as "this never ran", not as "this was not recorded".
    await service.record_run(
        run_id, Provider.BINANCE, "/funding", body.model_dump(mode="json"),
        dataset_id=meta.dataset_id, status="succeeded",
    )
    return FetchResult(dataset=meta, deduplicated=deduped)


class CryptoMetricsRequest(BaseModel):
    symbol: str = Field(min_length=1)
    timeframe: Timeframe = Timeframe.H1
    start: datetime
    end: datetime


@router.post("/binance/metrics", response_model=FetchResult, status_code=201)
async def fetch_binance_metrics(body: CryptoMetricsRequest) -> FetchResult:
    frame = await binance_fetch.fetch_metrics(
        body.symbol, body.start, body.end, body.timeframe
    )
    symbol = binance_fetch.normalise_symbol(body.symbol)

    service = IngestService()
    run_id = parquet_store.new_dataset_id()
    meta, deduped = await service.register(
        frame,
        name=default_name(symbol, body.timeframe, AssetClass.CRYPTO_FUTURES, body.start, body.end)
        + " metrics",
        category=DatasetCategory.RAW,
        source=Provider.BINANCE,
        asset_class=AssetClass.CRYPTO_FUTURES,
        symbol=symbol,
        market_type="futures",
        timeframe=body.timeframe,
        timezone="UTC",
        provenance={
            "endpoint": "data.binance.vision metrics dump",
            "note": "Source granularity is 5m; coarser timeframes take the last "
            "observation in each bar, not a mean — these are point-in-time levels.",
            "additional_columns_in_source": ", ".join(
                binance_fetch.METRICS_EXTRA_AVAILABLE
            ),
        },
    )

    await service.record_run(
        run_id, Provider.BINANCE, "/metrics", body.model_dump(mode="json"),
        dataset_id=meta.dataset_id, status="succeeded",
    )
    return FetchResult(dataset=meta, deduplicated=deduped)


# ------------------------------------------------------------------ equities


class YahooRequest(BaseModel):
    asset_class: AssetClass
    symbol: str = Field(min_length=1)
    timeframe: Timeframe
    start: datetime
    end: datetime


@router.post("/yahoo", response_model=FetchResult, status_code=201)
async def fetch_yahoo(body: YahooRequest) -> FetchResult:
    if body.asset_class not in (AssetClass.EQUITY_SPOT, AssetClass.EQUITY_FUTURES):
        raise ValidationFailure(
            f"{body.asset_class.value} is not a Yahoo market.",
            hint="Use equity_spot or equity_futures.",
        )

    frame, meta_extra = await yahoo_fetch.fetch_equities(
        body.asset_class, body.symbol, body.timeframe, body.start, body.end
    )
    symbol = body.symbol.strip().upper()

    service = IngestService()
    run_id = parquet_store.new_dataset_id()
    meta, deduped = await service.register(
        frame,
        name=default_name(symbol, body.timeframe, body.asset_class, body.start, body.end),
        category=DatasetCategory.RAW,
        source=Provider.YAHOO,
        asset_class=body.asset_class,
        symbol=symbol,
        market_type=(
            "cash_equity" if body.asset_class == AssetClass.EQUITY_SPOT else "equity_futures"
        ),
        timeframe=body.timeframe,
        # Yahoo reports no exchange timezone through this path, and guessing one from
        # the ticker suffix would be unreliable. Null means "unstated".
        timezone=None,
        unpopulated_fields=meta_extra.get("unpopulated_fields", []),
        provenance={
            "endpoint": "yfinance",
            "note": "Raw Close and Adj Close are both retained; adjusted prices are "
            "revised retroactively and are not point-in-time.",
            **(
                {"unpopulated_reason": meta_extra["unpopulated_reason"]}
                if "unpopulated_reason" in meta_extra
                else {}
            ),
        },
    )

    warnings: list[str] = []
    if meta.unpopulated_fields:
        warnings.append(
            "Columns "
            + ", ".join(meta.unpopulated_fields)
            + " are declared but unpopulated: Yahoo publishes no source for them."
        )
    if deduped:
        warnings.append("Identical content was already catalogued; existing dataset returned.")

    await service.record_run(
        run_id, Provider.YAHOO, "/yahoo", body.model_dump(mode="json"),
        dataset_id=meta.dataset_id, status="succeeded",
    )
    return FetchResult(dataset=meta, deduplicated=deduped, warnings=warnings)


# ---------------------------------------------------------------------- runs


@router.get("/runs")
async def recent_runs(limit: int = Query(default=50, ge=1, le=200)) -> list[dict[str, Any]]:
    service = IngestService()
    return await service.catalog_runs(limit)
