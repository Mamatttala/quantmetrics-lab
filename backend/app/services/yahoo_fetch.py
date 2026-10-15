"""Yahoo Finance ingestion for equities.

Two things here are deliberate and easy to get wrong:

* **`auto_adjust=False`.** Yahoo's adjusted prices are revised retroactively whenever
  a split or dividend occurs, so a series of them is not reproducible point-in-time:
  the value shown for a past date today is not the value that existed then. Both raw
  `Close` and `Adj Close` are stored; neither replaces the other (DECISIONS.md §15).

* **Daily bars keep their exchange-local session date.** A US equity session opening
  09:30 America/New_York is not a UTC midnight boundary, so coercing daily bars to
  UTC midnight misaligns them against crypto daily bars by up to a day (§14).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd

from app.core.blocking import run_blocking
from app.core.errors import SpanTooWide, UpstreamError, ValidationFailure
from app.models.enums import AssetClass, Timeframe

#: yfinance's intraday intervals. `4h` is absent on purpose — Yahoo does not serve
#: it, so a 4h series is aggregated from 1h bars below rather than requested.
_NATIVE_INTERVAL = {
    Timeframe.H1: "1h",
    Timeframe.D1: "1d",
}

#: Yahoo restricts intraday history without documenting a firm number; 730 days is
#: the widely observed ceiling for hourly bars. Requests beyond it come back short,
#: which the post-check below turns into a hard error rather than silent data loss.
_INTRADAY_MAX_DAYS = 729

CASH_EQUITY_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
EQUITY_FUTURES_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "settlement",
    "open_interest",
]

#: Columns the equity-futures schema declares but no available source populates.
#: Recorded in dataset metadata so nulls are read as "not available" rather than
#: as missing values (DECISIONS.md §2).
EQUITY_FUTURES_UNPOPULATED = ["settlement", "open_interest"]


def _flatten(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse yfinance's (Price, Ticker) MultiIndex columns to plain names.

    yfinance returns a MultiIndex for a single ticker in recent versions, and flat
    columns in older ones. Handling both here keeps that inconsistency out of the
    rest of the module.
    """
    if isinstance(frame.columns, pd.MultiIndex):
        # Level 0 carries the field name; the ticker level is redundant for us
        # because one dataset is one symbol.
        frame = frame.copy()
        frame.columns = [str(c[0]) for c in frame.columns]
    return frame


def _to_utc_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Move the date index to a UTC column named `Date`.

    Daily bars from Yahoo arrive as tz-naive session dates. They are localised to UTC
    as a *labelling* convention only — the value stored is the session date, and the
    dataset's `timezone` field records the exchange zone so the original meaning stays
    recoverable. Converting a session date to a true UTC instant is a decision for
    whoever builds features on it, not something ingest should silently impose.
    """
    frame = frame.copy()
    index = pd.DatetimeIndex(frame.index)
    # Labelling, not conversion: a naive session date is *declared* UTC in the sense of
    # "this is the date, pinned to UTC midnight". An aware index is a real instant and is
    # converted, so 12:00 in +05:00 becomes 07:00Z rather than being relabelled 12:00Z.
    index = index.tz_localize(UTC) if index.tz is None else index.tz_convert(UTC)
    frame["Date"] = index
    return frame.reset_index(drop=True)


def _aggregate_to_4h(frame: pd.DataFrame) -> pd.DataFrame:
    """Build 4h bars from 1h bars.

    `label="left", closed="left"` matches how Binance stamps its own bars: the
    timestamp is the *opening* time of the interval. Using pandas' defaults would
    label each bar by its close, which would shift every 4h bar three hours against
    the crypto series and quietly break any cross-asset alignment.
    """
    if frame.empty:
        return frame
    frame = frame.set_index("Date")
    aggregated = frame.resample("4h", label="left", closed="left").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Adj Close": "last",
            "Volume": "sum",
        }
    )
    # A bar with no trades inside it has no open; dropping beats emitting a row of
    # NaN that looks like a zero-volume session.
    aggregated = aggregated.dropna(subset=["Open", "Close"])
    return aggregated.reset_index()


async def fetch_equities(
    asset_class: AssetClass,
    symbol_input: str,
    timeframe: Timeframe,
    start: datetime,
    end: datetime,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Download equity bars.

    Returns the frame plus a metadata dict carrying the exchange timezone (when
    Yahoo reports one) and any columns declared but not populated.
    """
    if asset_class not in (AssetClass.EQUITY_SPOT, AssetClass.EQUITY_FUTURES):
        raise ValidationFailure(f"Yahoo ingestion does not serve {asset_class.value}.")
    if timeframe not in (Timeframe.H1, Timeframe.H4, Timeframe.D1):
        raise ValidationFailure(
            f"Yahoo does not serve {timeframe.value} bars for equities.",
            hint="Available: 1h, 4h, 1d.",
        )
    symbol = symbol_input.strip().upper()
    if not symbol:
        raise ValidationFailure("Symbol is required.")
    if start >= end:
        raise ValidationFailure("Start date must be before end date.")

    if timeframe in (Timeframe.H1, Timeframe.H4):
        span_days = (end - start).days
        if span_days > _INTRADAY_MAX_DAYS:
            raise SpanTooWide(
                f"Requested {span_days} days of {timeframe.value} equity data.",
                hint=f"Yahoo serves at most about {_INTRADAY_MAX_DAYS} days of intraday "
                "history. Narrow the range, or request 1d bars.",
                requested_days=span_days,
                max_days=_INTRADAY_MAX_DAYS,
            )

    # 4h is built from 1h; both need the intraday endpoint.
    request_interval = "1h" if timeframe in (Timeframe.H1, Timeframe.H4) else "1d"

    def download() -> pd.DataFrame:
        import yfinance as yf

        try:
            frame = yf.download(
                symbol,
                start=start.date().isoformat(),
                # yfinance's `end` is exclusive. Passing the user's end date directly
                # would drop the final bar of every request.
                end=(end.date() + pd.Timedelta(days=1)).isoformat(),
                interval=request_interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception as exc:
            raise UpstreamError(f"Yahoo request failed: {exc}", symbol=symbol) from exc

        return _flatten(frame) if frame is not None else pd.DataFrame()

    frame = await run_blocking(download)

    if frame.empty:
        raise UpstreamError(
            f"Yahoo returned no data for {symbol}.",
            hint="Check the ticker. Equity futures usually need an '=' suffix, "
            "for example ES=F or NQ=F.",
            symbol=symbol,
        )
    if "Close" not in frame.columns:
        raise UpstreamError(
            f"Yahoo response for {symbol} has no Close column.",
            columns=[str(c) for c in frame.columns],
        )

    def shape() -> pd.DataFrame:
        out = _to_utc_index(frame)
        if timeframe == Timeframe.H4:
            out = _aggregate_to_4h(out)
        return out.reset_index(drop=True)

    out = await run_blocking(shape)

    # Compare what came back against what was asked for. Yahoo truncates intraday
    # history silently, and a short frame that reports success is the exact failure
    # mode DECISIONS.md §7 exists to prevent.
    is_futures = asset_class == AssetClass.EQUITY_FUTURES

    def verify_and_label() -> tuple[pd.DataFrame, dict[str, Any]]:
        """Span check, schema shaping, and the metadata describing both.

        Kept inside a blocking helper rather than the coroutine body: the span check is
        a pandas reduction over a column that can hold millions of rows, and the schema
        shaping writes two columns across the whole frame. Both belong on a worker
        thread with the download and the shaping above.
        """
        if out.empty:
            raise UpstreamError(f"No bars survived processing for {symbol}.")

        actual_start = pd.Timestamp(out["Date"].min())
        requested_start = pd.Timestamp(start if start.tzinfo else start.replace(tzinfo=UTC))
        tolerance = pd.Timedelta(days=5 if request_interval == "1d" else 1)
        if actual_start > requested_start + tolerance:
            raise SpanTooWide(
                f"Yahoo returned data starting {actual_start.date()}, but "
                f"{requested_start.date()} was requested.",
                hint="Yahoo restricts how far back intraday and some daily history "
                "goes. Narrow the requested range to match what is available.",
                returned_start=str(actual_start),
                requested_start=str(requested_start),
            )

        metadata: dict[str, Any] = {"yahoo_symbol": symbol}
        if is_futures:
            # Declared in the schema, unavailable from the source. Created as all-null
            # so the schema is stable and a future real source can backfill into it.
            out["settlement"] = pd.Series([None] * len(out), dtype="float64")
            out["open_interest"] = pd.Series([None] * len(out), dtype="float64")
            metadata["unpopulated_fields"] = EQUITY_FUTURES_UNPOPULATED
            metadata["unpopulated_reason"] = (
                "Yahoo does not publish settlement price or open interest for futures. "
                "Columns are declared for schema stability and left null."
            )
            declared = EQUITY_FUTURES_COLUMNS
        else:
            metadata["unpopulated_fields"] = []
            declared = CASH_EQUITY_COLUMNS

        ordered = [c for c in declared if c in out.columns]
        return out[ordered].reset_index(drop=True), metadata

    return await run_blocking(verify_and_label)
