"""Binance ingestion.

Two transports, chosen per dataset and documented in DECISIONS.md §4:

* **Klines** come from the REST API. It is current to the minute and accepts an
  arbitrary date range, which matters because price data is needed up to now.
* **Funding rate and metrics** come from the `data.binance.vision` public dumps.
  These are the only source carrying `funding_interval_hours`, and the metrics REST
  endpoint is capped at roughly 30 days per request. Neither series is needed
  intraday, so the dumps' one-day publication lag is acceptable.

Both transports are async at the network layer; every pandas operation is pushed to
a worker thread.
"""

from __future__ import annotations

import asyncio
import io
import zipfile
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal

import httpx
import pandas as pd

from app.config import get_settings
from app.core.blocking import run_blocking
from app.core.errors import SpanTooWide, UpstreamError, ValidationFailure
from app.models.enums import AssetClass, Timeframe

DUMP_BASE = "https://data.binance.vision/data"

#: REST page sizes. Spot caps at 1000, futures at 1500; asking for more returns an
#: error rather than a truncated page, but using the documented maximum minimises
#: round trips.
_SPOT_PAGE = 1000
_FUTURES_PAGE = 1500

#: Canonical column order, per DECISIONS.md §6. One scheme for all three families
#: so a spot/futures merge produces one column, not two half-empty ones.
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "num_trades",
    "taker_buy_volume",
    "taker_buy_quote_volume",
]

#: Binance's own dump headers, in file order. Used to select a subset positionally
#: when a file turns out to be headerless (§5).
DUMP_KLINE_FIELDS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]

FUNDING_COLUMNS = ["calc_time", "funding_interval_hours", "last_funding_rate"]

METRICS_COLUMNS = [
    "create_time",
    "symbol",
    "sum_open_interest",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
]

#: The metrics dump's *full* positional layout, in file order. `METRICS_COLUMNS` above
#: is the five-column subset the specification asks us to keep; this is what a
#: headerless file needs in order to be named at all. Naming a headerless file with the
#: five-column list raises a length mismatch on an eight-column file — a confusing
#: failure at the very moment the sniffing was supposed to handle the difference.
METRICS_DUMP_FIELDS = [
    "create_time",
    "symbol",
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
]

#: The metrics dump carries three further columns we deliberately do not store,
#: because the specification names exactly five. Recorded here so the omission is a
#: decision rather than an oversight.
METRICS_EXTRA_AVAILABLE = [
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
]


def normalise_symbol(raw: str) -> str:
    """Accept `BTC`, `btcusdt`, or `BTCUSDT` and return `BTCUSDT`.

    The specification says the user types a symbol for a USDT pair, so a bare base
    asset is the common case. Anything already ending in a quote suffix is left alone.
    """
    symbol = raw.strip().upper().replace("/", "").replace("-", "")
    if not symbol:
        raise ValidationFailure("Symbol is required.")
    if not symbol.isalnum():
        raise ValidationFailure(
            f"Symbol {raw!r} contains unexpected characters.",
            hint="Use a form like BTC, BTCUSDT, or BTC/USDT.",
        )
    for quote in ("USDT", "USDC", "BUSD", "FDUSD", "TUSD"):
        if symbol.endswith(quote):
            return symbol
    return f"{symbol}USDT"


def _timeframe_delta(timeframe: Timeframe) -> timedelta:
    from app.models.enums import TIMEFRAME_SECONDS

    return timedelta(seconds=TIMEFRAME_SECONDS[timeframe])


def _ms(moment: datetime) -> int:
    """Milliseconds since epoch. Binance speaks in integer milliseconds."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return int(moment.timestamp() * 1000)


# --------------------------------------------------------------------- HTTP


def _client() -> httpx.AsyncClient:
    settings = get_settings()
    return httpx.AsyncClient(
        timeout=settings.http_timeout_seconds,
        headers={"User-Agent": "quantmetrics-lab/0.1 (local research)"},
        follow_redirects=True,
    )


async def _get_json(
    client: httpx.AsyncClient, url: str, params: dict[str, Any]
) -> Any:
    settings = get_settings()
    last: Exception | None = None
    for attempt in range(settings.http_max_retries):
        try:
            response = await client.get(url, params=params)
        except httpx.HTTPError as exc:
            last = exc
            await asyncio.sleep(0.6 * (attempt + 1))
            continue

        if response.status_code == 200:
            return response.json()

        # 429/418 are Binance's rate-limit and ban codes; backing off is the only
        # correct response. Retrying immediately escalates a soft limit to a ban.
        if response.status_code in (429, 418):
            retry_after = float(response.headers.get("Retry-After", 1.0))
            last = UpstreamError(
                f"Binance rate limited the request (HTTP {response.status_code}).",
                retry_after=retry_after,
            )
            await asyncio.sleep(min(retry_after, 5.0) * (attempt + 1))
            continue

        if 400 <= response.status_code < 500:
            raise UpstreamError(
                f"Binance rejected the request: HTTP {response.status_code}",
                body=response.text[:400],
            )
        last = UpstreamError(f"Binance returned HTTP {response.status_code}")

    raise last if isinstance(last, Exception) else UpstreamError("Binance request failed.")


# ------------------------------------------------------------------- klines


def _klines_to_frame(raw: list[list[Any]]) -> pd.DataFrame:
    """Convert a REST kline payload into the canonical frame.

    Binance returns positional arrays, so the mapping is by index. Timestamps are
    integer milliseconds and are converted explicitly to UTC — `unit="ms"` with
    `utc=True` rather than a local-time default.
    """
    if not raw:
        return pd.DataFrame(columns=KLINE_COLUMNS)

    frame = pd.DataFrame(raw)
    frame = frame.iloc[:, :12]
    frame.columns = [*DUMP_KLINE_FIELDS[:12]]  # same positional layout as the dump

    out = pd.DataFrame(
        {
            "open_time": pd.to_datetime(frame["open_time"].astype("int64"), unit="ms", utc=True),
            "open": frame["open"].astype(float),
            "high": frame["high"].astype(float),
            "low": frame["low"].astype(float),
            "close": frame["close"].astype(float),
            "volume": frame["volume"].astype(float),
            "close_time": pd.to_datetime(
                frame["close_time"].astype("int64"), unit="ms", utc=True
            ),
            "quote_volume": frame["quote_volume"].astype(float),
            "num_trades": frame["count"].astype("int64"),
            "taker_buy_volume": frame["taker_buy_volume"].astype(float),
            "taker_buy_quote_volume": frame["taker_buy_quote_volume"].astype(float),
        }
    )
    return out[KLINE_COLUMNS]


async def fetch_klines(
    asset_class: AssetClass,
    symbol_input: str,
    timeframe: Timeframe,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Download OHLCV klines, paginating until the requested window is covered."""
    if asset_class not in (AssetClass.CRYPTO_SPOT, AssetClass.CRYPTO_FUTURES):
        raise ValidationFailure(f"Binance klines do not serve {asset_class.value}.")
    if start >= end:
        raise ValidationFailure("Start date must be before end date.")

    symbol = normalise_symbol(symbol_input)
    settings = get_settings()
    is_futures = asset_class == AssetClass.CRYPTO_FUTURES
    base = settings.binance_futures_base if is_futures else settings.binance_spot_base
    path = "/fapi/v1/klines" if is_futures else "/api/v3/klines"
    page_size = _FUTURES_PAGE if is_futures else _SPOT_PAGE

    collected: list[list[Any]] = []
    cursor = _ms(start)
    end_ms = _ms(end)
    # The width of one bar in milliseconds — a duration, not a timestamp. Computing
    # this as `_ms(start + delta)` yields an epoch value, which makes the pagination
    # cursor leap past `end_ms` and silently returns a single page.
    step_ms = int(_timeframe_delta(timeframe).total_seconds() * 1000)

    async with _client() as client:
        while cursor < end_ms:
            batch = await _get_json(
                client,
                f"{base}{path}",
                {
                    "symbol": symbol,
                    "interval": timeframe.value,
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": page_size,
                },
            )
            if not batch:
                # An empty page means the listing did not exist for this window, or
                # we have walked past the present. Either way there is nothing more.
                break

            collected.extend(batch)
            last_open = int(batch[-1][0])
            next_cursor = last_open + step_ms
            if next_cursor <= cursor:
                break
            cursor = next_cursor

            if len(batch) < page_size:
                break
            # Public endpoints tolerate a steady pace; this keeps us well inside the
            # weight budget without needing to track it.
            await asyncio.sleep(0.12)

    if not collected:
        raise UpstreamError(
            f"Binance returned no klines for {symbol} {timeframe.value}.",
            hint="Check that the symbol is listed on the selected market and that the "
            "date range is not entirely in the future.",
        )

    frame = await run_blocking(_klines_to_frame, collected)
    frame = frame.drop_duplicates(subset=["open_time"]).sort_values("open_time")
    return frame.reset_index(drop=True)


# --------------------------------------------------------------- dump reader


async def _download_zip_csv(
    client: httpx.AsyncClient,
    url: str,
    *,
    columns: list[str] | None = None,
) -> pd.DataFrame | None:
    """Fetch one dump ZIP and parse its single CSV member.

    Returns None on 404: a missing file is a normal occurrence (the symbol was not
    listed yet, the month is not published, or the day is still in the future), not an
    error. Any other failure is raised, because silently skipping it would leave a hole
    that looks like a gap in the market data.

    `columns` names the file positionally when it turns out to be headerless. It has to
    be the file's *full* layout, not the subset being kept: a length mismatch here
    would surface as a pandas error rather than as the read it was meant to be.
    """
    settings = get_settings()
    response: httpx.Response | None = None
    last: Exception | None = None

    # Retried, unlike a one-shot download. A dump may span hundreds of daily files, and
    # a caller that treats any failed file as fatal (see `_collect_dump`) would then
    # abort a fetch over one transient CDN hiccup.
    for attempt in range(settings.http_max_retries):
        try:
            response = await client.get(url)
        except httpx.HTTPError as exc:
            last = exc
            await asyncio.sleep(0.5 * (attempt + 1))
            continue
        if response.status_code == 200 or response.status_code == 404:
            break
        last = UpstreamError(
            f"Dump download failed: HTTP {response.status_code}", url=url
        )
        await asyncio.sleep(0.5 * (attempt + 1))

    if response is None or response.status_code not in (200, 404):
        raise last if isinstance(last, Exception) else UpstreamError(
            "Dump download failed.", url=url
        )
    if response.status_code == 404:
        return None

    blob = response.content

    def parse() -> pd.DataFrame:
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            if not names:
                # Every Binance dump archive holds exactly one CSV. An archive without
                # one is a corrupted or substituted download, not a period with no
                # data — returning None here would file it under "no data for this
                # date" and leave a silent hole, which is the outcome the 404 handling
                # above exists to keep separate.
                raise UpstreamError(
                    "Dump archive contained no CSV member.",
                    url=url,
                    members=archive.namelist()[:10],
                )
            with archive.open(names[0]) as handle:
                head = handle.readline().decode("utf-8", errors="replace").strip()
                handle.seek(0)
                # §5: the spot kline dumps have no header row. Detect it by asking
                # whether the first field is numeric rather than keeping a table of
                # which families carry headers, which would rot.
                first_field = head.split(",")[0].strip()
                has_header = not _looks_numeric(first_field)
                frame = pd.read_csv(handle, header=0 if has_header else None)
                if not has_header and columns is not None:
                    if len(frame.columns) != len(columns):
                        raise UpstreamError(
                            f"Headerless dump {url} has {len(frame.columns)} columns, "
                            f"expected {len(columns)}; its layout has changed.",
                            url=url,
                        )
                    frame.columns = columns
                return frame

    return await run_blocking(parse)


def _looks_numeric(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _month_starts(start: date, end: date) -> list[date]:
    months: list[date] = []
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        months.append(cursor)
        cursor = date(cursor.year + (cursor.month // 12), (cursor.month % 12) + 1, 1)
    return months


async def _collect_dump(
    client: httpx.AsyncClient,
    kind: Literal["fundingRate", "metrics"],
    symbol: str,
    start: datetime,
    end: datetime,
    *,
    monthly: bool,
) -> list[pd.DataFrame]:
    """Walk monthly then daily dump files across the requested range."""
    frames: list[pd.DataFrame] = []
    start_d, end_d = start.date(), end.date()
    # The file's own full layout, used only when a file turns out to be headerless.
    layout = FUNDING_COLUMNS if kind == "fundingRate" else METRICS_DUMP_FIELDS

    if monthly:
        requests = [
            f"{DUMP_BASE}/futures/um/monthly/{kind}/{symbol}/"
            f"{symbol}-{kind}-{m.strftime('%Y-%m')}.zip"
            for m in _month_starts(start_d, end_d)
        ]
    else:
        # Metrics are published daily only; funding has both. Daily walking covers
        # the partial months at each end of a range.
        span = (end_d - start_d).days + 1
        if span > 800:
            raise SpanTooWide(
                f"Daily dump walk of {span} days is too wide for {kind}.",
                hint="The monthly files cover whole months; narrow the range or use "
                "monthly granularity.",
                days=span,
            )
        requests = [
            f"{DUMP_BASE}/futures/um/daily/{kind}/{symbol}/"
            f"{symbol}-{kind}-{(start_d + timedelta(days=i)).isoformat()}.zip"
            for i in range(span)
        ]

    # Modest concurrency: these are static files on a CDN, but hammering it is both
    # rude and liable to get us throttled.
    semaphore = asyncio.Semaphore(8)

    async def one(url: str) -> pd.DataFrame | None:
        async with semaphore:
            return await _download_zip_csv(client, url, columns=layout)

    results = await asyncio.gather(*(one(u) for u in requests), return_exceptions=True)

    hard_failures: list[BaseException] = []
    for item in results:
        if isinstance(item, BaseException):
            hard_failures.append(item)
        elif item is not None and not item.empty:
            frames.append(item)

    if hard_failures:
        # Deliberately fatal even when some files arrived. Storing the subset would
        # produce a dataset with holes that are indistinguishable from periods when
        # the market genuinely had no data — the exact confusion this function's
        # 404 handling exists to prevent, and one that no downstream check could
        # detect afterwards. A missing file is a 404 and returns None above; reaching
        # here means the download actually failed, and retrying is the honest answer.
        first = hard_failures[0]
        raise UpstreamError(
            f"{len(hard_failures)} of {len(requests)} {kind} dump file(s) failed to "
            f"download; refusing to store a dataset with holes. First error: {first}",
            failures=len(hard_failures),
            succeeded=len(frames),
            first_error=str(first),
        )
    return frames


# ------------------------------------------------------------------ funding


def _as_utc(moment: datetime) -> datetime:
    """Read a request bound as UTC.

    The dump files carry UTC timestamps, and pandas does **not** coerce a tz-aware
    column against a naive bound — it raises
    `TypeError: Invalid comparison between dtype=datetime64[ns, UTC] and datetime`,
    which surfaced as an HTTP 500 on every funding and metrics request. A bound that
    carries an offset is converted rather than discarded, so an explicit `+02:00` is
    honoured; a naive one is read as UTC, which is the only reading consistent with the
    rest of the ingestion path — Binance klines are UTC and every stored timestamp is
    UTC. Reading a naive bound as local time would silently shift the requested range
    by the machine's offset, and the resulting dataset would look plausible.

    Applied before the dump files are listed as well as before the rows are filtered:
    a bound near midnight can fall on the previous UTC day, and selecting the wrong
    monthly file returns an empty frame with no error.
    """
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


async def fetch_funding(
    symbol_input: str, start: datetime, end: datetime
) -> pd.DataFrame:
    """Funding rate history. Futures only — spot has no funding mechanism."""
    start, end = _as_utc(start), _as_utc(end)
    symbol = normalise_symbol(symbol_input)
    async with _client() as client:
        frames = await _collect_dump(client, "fundingRate", symbol, start, end, monthly=True)

    if not frames:
        # Fall back to the daily files. Monthly funding files are published after the
        # month closes, so a range ending in the current month has no monthly file yet.
        async with _client() as client:
            frames = await _collect_dump(
                client, "fundingRate", symbol, start, end, monthly=False
            )

    if not frames:
        raise UpstreamError(
            f"No funding-rate data found for {symbol}.",
            hint="Funding exists only for futures symbols. Check the symbol and range.",
        )

    def assemble() -> pd.DataFrame:
        frame = pd.concat(frames, ignore_index=True)
        # Column names are settled at parse time: a header row supplies them, and a
        # headerless file is named positionally from `_download_zip_csv`'s `columns`
        # argument. Nothing is inferred here.
        #
        # Cast to a number before asking for a unit. The CSV supplies `calc_time` as
        # text, and `to_datetime` with a `unit` on *strings* is deprecated — pandas 3
        # will parse them as datetime strings instead, at which point "1704067200000"
        # becomes NaT and every funding row is dropped without an error. Unparseable
        # values become NaT and are dropped, exactly as in the metrics path below; the
        # range filter then removes anything that coerced to a plausible-looking but
        # wrong epoch (0 would otherwise be 1970-01-01).
        frame["calc_time"] = pd.to_datetime(
            pd.to_numeric(frame["calc_time"], errors="coerce"), unit="ms", utc=True
        )
        frame = frame.dropna(subset=["calc_time"])
        for column in ("funding_interval_hours", "last_funding_rate"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.drop_duplicates(subset=["calc_time"]).sort_values("calc_time")
        frame = frame[(frame["calc_time"] >= start) & (frame["calc_time"] <= end)]
        return frame[FUNDING_COLUMNS].reset_index(drop=True)

    return await run_blocking(assemble)


# ------------------------------------------------------------------ metrics


async def fetch_metrics(
    symbol_input: str, start: datetime, end: datetime, timeframe: Timeframe
) -> pd.DataFrame:
    """Open interest and long/short ratios. Daily dumps only, 5m granularity at source."""
    from app.models.enums import CRYPTO_METRICS_TIMEFRAMES

    start, end = _as_utc(start), _as_utc(end)

    if timeframe not in CRYPTO_METRICS_TIMEFRAMES:
        raise ValidationFailure(
            f"Metrics are not published at {timeframe.value}.",
            hint="Available: "
            + ", ".join(sorted(t.value for t in CRYPTO_METRICS_TIMEFRAMES)),
        )

    symbol = normalise_symbol(symbol_input)
    async with _client() as client:
        frames = await _collect_dump(client, "metrics", symbol, start, end, monthly=False)

    if not frames:
        raise UpstreamError(
            f"No metrics data found for {symbol}.",
            hint="Metrics exist only for futures symbols, and only from around 2021.",
        )

    def assemble() -> pd.DataFrame:
        frame = pd.concat(frames, ignore_index=True)
        # Named at parse time (see `_download_zip_csv`): a rename here would have to
        # guess the file's width, and naming an eight-column headerless file with the
        # five-column keep-list raises rather than reads.
        # The dump writes a space-separated timestamp; anything unparseable becomes
        # NaT and is then dropped rather than being stored as a plausible-looking
        # epoch value.
        frame["create_time"] = pd.to_datetime(frame["create_time"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["create_time"])
        numeric_columns = (
            "sum_open_interest",
            "count_long_short_ratio",
            "sum_taker_long_short_vol_ratio",
        )
        for column in numeric_columns:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.drop_duplicates(subset=["create_time"]).sort_values("create_time")
        frame = frame[(frame["create_time"] >= start) & (frame["create_time"] <= end)]

        # Resample to the requested timeframe. `last` is correct for open interest
        # and for ratios: these are point-in-time levels, not flows, so a mean would
        # invent a value that was never observed. Volume-like fields would need `sum`.
        if timeframe != Timeframe.M5 and not frame.empty:
            frame = (
                frame.set_index("create_time")
                .resample(timeframe.value)
                .last()
                .dropna(how="all")
                .reset_index()
            )
        keep = [c for c in METRICS_COLUMNS if c in frame.columns]
        return frame[keep].reset_index(drop=True)

    return await run_blocking(assemble)
