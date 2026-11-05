"""Range filtering and resampling for the two dump-sourced families.

Funding and metrics come from `data.binance.vision` monthly/daily ZIPs rather than the
REST API, so unlike klines they filter the requested range **client-side**, comparing
`calc_time` / `create_time` against the request bounds. Both columns are parsed
tz-aware UTC (`pd.to_datetime(..., utc=True)`) while a bound arriving as a plain
`datetime` from the request body is naive, and pandas does not coerce between the two
— it raises `TypeError: Invalid comparison between dtype=datetime64[ns, UTC] and
datetime`.

Nothing covered either function, so both endpoints returned HTTP 500 for every
request that reached them: the missing-timestamp columns were only exercised through
the parser, never through the filter.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta, timezone

import pandas as pd
import pytest

from app.core.errors import UpstreamError, ValidationFailure
from app.models.enums import Timeframe
from app.services import binance_fetch


@pytest.fixture
def daily_metrics(monkeypatch):
    """Serve two five-minute observations for every requested UTC day."""
    state = {"urls": [], "active": 0, "peak": 0}

    async def download(client, url, *, columns=None):
        state["urls"].append(url)
        state["active"] += 1
        state["peak"] = max(state["peak"], state["active"])
        await asyncio.sleep(0)
        state["active"] -= 1
        day = url.removesuffix(".zip")[-10:]
        return pd.DataFrame({
            "create_time": [f"{day} 00:00:00", f"{day} 00:05:00"],
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "sum_open_interest": [1000, 1001],
            "count_long_short_ratio": [1.5, 1.6],
            "sum_taker_long_short_vol_ratio": [1.0, 1.1],
        })

    monkeypatch.setattr(binance_fetch, "_download_zip_csv", download)
    return state


@pytest.mark.parametrize("timeframe,rows", [(Timeframe.M5, 2010), (Timeframe.H1, 1005)])
async def test_metrics_downloads_1005_days_at_requested_resolution(daily_metrics, timeframe, rows):
    start = datetime(2022, 1, 1, tzinfo=UTC)
    end = start + timedelta(days=1005) - timedelta(seconds=1)
    frame = await binance_fetch.fetch_metrics("BTC", start, end, timeframe)
    assert len(frame) == rows
    assert len(daily_metrics["urls"]) == 1005
    assert len(set(daily_metrics["urls"])) == 1005
    assert all("/daily/metrics/" in url for url in daily_metrics["urls"])
    assert daily_metrics["peak"] <= binance_fetch._DUMP_CONCURRENCY
    assert frame["create_time"].min() == pd.Timestamp(start)
    assert frame["create_time"].max().date() == end.date()
    assert frame["create_time"].is_monotonic_increasing
    assert not frame["create_time"].duplicated().any()
    if timeframe == Timeframe.H1:
        assert (frame["sum_open_interest"] == 1001).all()


def test_metrics_endpoint_stores_a_multi_year_dataset(tmp_path, monkeypatch, daily_metrics):
    from fastapi.testclient import TestClient

    from app.config import get_settings
    from app.main import create_app

    monkeypatch.setenv("QML_DATA_DIR", str(tmp_path / "fetch-api"))
    get_settings.cache_clear()
    start = datetime(2022, 1, 1, tzinfo=UTC)
    end = start + timedelta(days=1005) - timedelta(seconds=1)
    try:
        with TestClient(create_app()) as client:
            response = client.post("/api/fetch/binance/metrics", json={
                "symbol": "BTC", "timeframe": "1h",
                "start": start.isoformat(), "end": end.isoformat(),
            })
            assert response.status_code == 201, response.text
            dataset = response.json()["dataset"]
            assert dataset["row_count"] == 1005
            assert dataset["timeframe"] == "1h"
            preview = client.get(f"/api/datasets/{dataset['dataset_id']}/preview").json()
            assert preview["row_count"] == 1005
            assert preview["head"][0]["sum_open_interest"] == 1001
    finally:
        get_settings.cache_clear()


async def test_a_failed_daily_file_still_refuses_to_store_partial_metrics(monkeypatch):
    async def download(client, url, *, columns=None):
        if "2024-01-02" in url:
            raise UpstreamError("CDN timeout")
        return metrics_dump(60)

    monkeypatch.setattr(binance_fetch, "_download_zip_csv", download)
    with pytest.raises(UpstreamError, match="1 of 3 metrics dump"):
        await binance_fetch.fetch_metrics(
            "BTC", datetime(2024, 1, 1), datetime(2024, 1, 3), Timeframe.H1,
        )


async def test_funding_fills_a_missing_month_even_when_earlier_months_exist(monkeypatch):
    urls = []

    async def download(client, url, *, columns=None):
        urls.append(url)
        if "/monthly/" in url:
            return funding_dump(1) if "2024-01.zip" in url else None
        day = url.removesuffix(".zip")[-10:]
        return funding_dump(1, start=f"{day}T00:00:00Z")

    monkeypatch.setattr(binance_fetch, "_download_zip_csv", download)
    frame = await binance_fetch.fetch_funding(
        "BTC", datetime(2024, 1, 1), datetime(2024, 2, 2, 23, 59, 59),
    )
    assert list(frame["calc_time"].dt.strftime("%Y-%m-%d")) == [
        "2024-01-01", "2024-02-01", "2024-02-02",
    ]
    assert len(urls) == 4
    assert all("2024-02" in url for url in urls if "/daily/" in url)


def funding_dump(hours: int = 12, *, start: str = "2024-01-01T00:00:00Z") -> pd.DataFrame:
    """A funding dump as it leaves `_download_zip_csv`: `calc_time` in milliseconds,
    the two numeric columns as text (the file is CSV)."""
    origin = datetime.fromisoformat(start)
    rows = []
    for step in range(hours):
        moment = origin + timedelta(hours=8 * step)
        rows.append(
            {
                "calc_time": str(int(moment.timestamp() * 1000)),
                "funding_interval_hours": "8",
                "last_funding_rate": f"0.0001{step}",
            }
        )
    return pd.DataFrame(rows)


def metrics_dump(minutes: int = 180) -> pd.DataFrame:
    """A metrics dump: `create_time` is written space-separated and parsed with
    `errors="coerce"`, so it arrives as text."""
    origin = datetime(2024, 1, 1, tzinfo=UTC)
    rows = []
    for step in range(minutes // 5):
        moment = origin + timedelta(minutes=5 * step)
        rows.append(
            {
                "create_time": moment.strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": "BTCUSDT",
                "sum_open_interest": str(1000 + step),
                "count_long_short_ratio": str(1.5 + step / 1000),
                "sum_taker_long_short_vol_ratio": str(1.0 + step / 1000),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def dumping(monkeypatch: pytest.MonkeyPatch) -> dict[str, pd.DataFrame]:
    """Serve the crafted dump instead of downloading it. Recorded so a test can assert
    which files the range would have selected."""
    served: dict[str, pd.DataFrame] = {"frames": pd.DataFrame()}

    async def fake_collect(
        client: object,
        kind: str,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        monthly: bool,
    ) -> list[pd.DataFrame]:
        served["kind"] = kind
        served["start"] = start
        served["end"] = end
        served["monthly"] = monthly
        # Mirrors the real collector, which appends a frame only when it is non-empty
        # (a file that downloads but holds no rows is skipped, not kept). Returning
        # `[empty]` here instead would hand `assemble` a zero-column frame and turn the
        # "empty dump" case into a KeyError rather than the typed error it must raise.
        return [] if served["frames"].empty else [served["frames"]]

    monkeypatch.setattr(binance_fetch, "_collect_dump", fake_collect)
    return served


async def test_funding_accepts_naive_bounds_against_a_utc_column(dumping: dict) -> None:
    """The regression. Naive bounds + tz-aware column raised TypeError, which the
    router had no handler for, so the endpoint answered 500."""
    dumping["frames"] = funding_dump(6)
    frame = await binance_fetch.fetch_funding(
        "BTCUSDT", datetime(2024, 1, 1), datetime(2024, 1, 3)
    )

    assert len(frame) == 6
    assert list(frame.columns) == binance_fetch.FUNDING_COLUMNS
    assert frame["calc_time"].dt.tz is not None


async def test_funding_excludes_rows_outside_the_requested_range(dumping: dict) -> None:
    dumping["frames"] = funding_dump(12)  # 8-hourly over four days
    frame = await binance_fetch.fetch_funding(
        "BTCUSDT",
        datetime(2024, 1, 1),
        datetime(2024, 1, 1, 23, 59, 59),
    )

    assert len(frame) == 3, "only the 00:00, 08:00 and 16:00 fundings are in range"
    assert frame["calc_time"].max() <= pd.Timestamp("2024-01-01T16:00:00Z")


async def test_an_offset_aware_bound_is_converted_not_discarded(dumping: dict) -> None:
    """`02:00+02:00` is midnight UTC, two hours earlier than the wall clock reads.

    Dropping the offset would read it as 02:00 UTC and silently narrow the range by two
    hours at each end — here that would drop the 00:00 funding and return two rows
    instead of three, with nothing in the frame to say a row was missing.
    """
    dumping["frames"] = funding_dump(12)
    plus_two = timezone(timedelta(hours=2))
    frame = await binance_fetch.fetch_funding(
        "BTCUSDT",
        datetime(2024, 1, 1, 2, tzinfo=plus_two),  # 2024-01-01T00:00:00Z
        datetime(2024, 1, 1, 18, tzinfo=plus_two),  # 2024-01-01T16:00:00Z
    )

    assert dumping["start"] == datetime(2024, 1, 1, tzinfo=UTC)
    assert dumping["end"] == datetime(2024, 1, 1, 16, tzinfo=UTC)
    assert list(frame["calc_time"]) == [
        pd.Timestamp("2024-01-01T00:00:00Z"),
        pd.Timestamp("2024-01-01T08:00:00Z"),
        pd.Timestamp("2024-01-01T16:00:00Z"),
    ]


async def test_the_utc_date_is_what_selects_the_dump_file(dumping: dict) -> None:
    """A bound at 23:00+02:00 is 21:00 UTC on the same day; a bound at 01:00+02:00 is
    23:00 UTC on the *previous* day. File selection uses the UTC date, so the
    conversion has to happen before the files are listed."""
    dumping["frames"] = funding_dump(3)
    await binance_fetch.fetch_funding(
        "BTCUSDT",
        datetime(2024, 3, 1, 1, tzinfo=timezone(timedelta(hours=2))),
        datetime(2024, 3, 2, 1, tzinfo=timezone(timedelta(hours=2))),
    )

    assert dumping["start"].tzinfo is not None
    assert dumping["start"].date().isoformat() == "2024-02-29"


async def test_metrics_accept_naive_bounds_and_resample_to_the_target(dumping: dict) -> None:
    """180 five-minute rows is 15 hours; at 1h that is 15 bars, and the retained value
    must be the *last* in each interval — open interest is a level, not a flow."""
    dumping["frames"] = metrics_dump(15 * 60)
    frame = await binance_fetch.fetch_metrics(
        "BTCUSDT", datetime(2024, 1, 1), datetime(2024, 1, 2), Timeframe.H1
    )

    assert len(frame) == 15
    assert list(frame.columns) == binance_fetch.METRICS_COLUMNS
    # The 09:00 bar carries the 09:55 observation (step 119), not the 09:00 one.
    at_nine = frame.loc[frame["create_time"] == pd.Timestamp("2024-01-01T09:00:00Z")]
    assert int(at_nine["sum_open_interest"].iloc[0]) == 1119


async def test_metrics_at_five_minutes_are_not_resampled(dumping: dict) -> None:
    dumping["frames"] = metrics_dump(60)
    frame = await binance_fetch.fetch_metrics(
        "BTCUSDT", datetime(2024, 1, 1), datetime(2024, 1, 2), Timeframe.M5
    )

    assert len(frame) == 12


async def test_metrics_refuses_a_timeframe_the_dump_does_not_serve(dumping: dict) -> None:
    dumping["frames"] = metrics_dump(60)
    with pytest.raises(ValidationFailure) as raised:
        await binance_fetch.fetch_metrics(
            "BTCUSDT", datetime(2024, 1, 1), datetime(2024, 1, 2), Timeframe.W1
        )

    assert "not published" in raised.value.message


async def test_an_empty_dump_is_a_typed_error_not_an_empty_dataset(dumping: dict) -> None:
    """An empty dump must reach `UpstreamError`, not `assemble`.

    The distinction is the difference between "this range has no funding data" and a
    `KeyError: 'calc_time'` from a zero-column frame, which the router would answer with
    a 500.
    """
    dumping["frames"] = pd.DataFrame()
    with pytest.raises(UpstreamError):
        await binance_fetch.fetch_funding(
            "BTCUSDT", datetime(2024, 1, 1), datetime(2024, 1, 3)
        )
