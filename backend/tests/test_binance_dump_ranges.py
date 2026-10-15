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

from datetime import UTC, datetime, timedelta, timezone

import pandas as pd
import pytest

from app.core.errors import UpstreamError, ValidationFailure
from app.models.enums import Timeframe
from app.services import binance_fetch


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
