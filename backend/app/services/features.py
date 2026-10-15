"""Feature computation — the engine behind C5.

One invariant governs everything here, and every function below is written so it holds by
construction rather than by care:

    every column is causal. For any row *t*, its value depends only on rows at or before
    *t*. Recomputed on a truncated series, the last row is unchanged.

That is why nothing in this module calls a whole-sample statistic. No ``.mean()`` over a
column, no ``.std()`` without a window, no ``.max()`` without one either — each of those
would make row *t* depend on rows after *t*, and the resulting dataset would backtest too
well for a reason invisible in the frame. ``tests/test_features_compute.py`` asserts the
property directly, over every entry in the catalogue.

The other decision worth stating: **no row is ever dropped.** Warm-up rows are left with
empty feature values and the report says how many there are. Deleting them here would
make the output row count silently disagree with the input's, and the timeline is the one
thing a research pipeline cannot afford to lose track of.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.core.errors import ValidationFailure
from app.models.enums import (
    TIMEFRAME_SECONDS,
    AssetClass,
    FeatureKind,
    PriceSource,
    Timeframe,
)
from app.services import feature_catalog as catalogue
from app.services.feature_catalog import DatasetContext, FeatureSpec, WindowOption

#: Bounds the peak size of the sliding-window arrays in `_rolling_slope`. The window view
#: itself is free, but the `windows * weights` product is not, so it is taken in blocks.
_CHUNK = 65_536

#: Absolute correlation at which two generated columns are reported as carrying one signal.
#: Deliberately strict: the scan's value is that a report means "these two are the same
#: column", so it stays silent about pairs that are merely correlated. It travels in the
#: report as `collinearity_threshold` so the sentence the page prints above the table is the
#: number the scan actually used rather than a copy of it.
COLLINEARITY_R = 0.99

#: How many of the qualifying pairs are reported. A selection large enough to produce more
#: than this has a collinearity problem the list is already making; the tail adds nothing.
COLLINEARITY_MAX_PAIRS = 20


# --------------------------------------------------------------------------------------
# Request types
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Selection:
    """One checked feature, with the window the user chose for it.

    ``window`` is the token as the request spells it — ``"20"`` for a bar count, ``"4h"``
    for an interval — and is ``None`` for a feature that takes no parameter.
    """

    feature_id: str
    window: str | None = None


@dataclass(frozen=True)
class FeatureRequest:
    selections: tuple[Selection, ...]
    price_source: PriceSource = PriceSource.CLOSE
    #: The specification's "should only be kept in the new processed_data, nothing more".
    #: When set, the output is the time column plus the selected columns and no others;
    #: the report lists everything dropped, so the projection is never silent.
    keep_only_selected: bool = False


@dataclass(frozen=True)
class Resolved:
    """A selection after validation: which window, how many bars, which columns."""

    spec: FeatureSpec
    token: str | None
    bars: int
    columns: tuple[str, ...]

    @property
    def label(self) -> str:
        return self.spec.label if self.token is None else f"{self.spec.label} ({self.token})"


# --------------------------------------------------------------------------------------
# Causal primitives
# --------------------------------------------------------------------------------------


def _shift(values: np.ndarray, bars: int) -> np.ndarray:
    """Backward shift by ``bars``: row t receives the value that was at t − bars."""
    out = np.full(values.shape, np.nan, dtype=float)
    if bars <= 0:
        return values.astype(float)
    if bars < values.size:
        out[bars:] = values[:-bars]
    return out


def _first_full_window(finite: np.ndarray, window: int) -> int:
    """Index of the first row whose trailing ``window`` values are all finite, or −1."""
    run = 0
    for index, ok in enumerate(finite):
        run = run + 1 if ok else 0
        if run >= window:
            return index
    return -1


def _recursive(values: np.ndarray, window: int, alpha: float) -> np.ndarray:
    """A recursive average, seeded with the mean of the first full window.

    One routine serves both Wilder's smoothing (``alpha = 1/n``, used by ATR, RSI,
    DI and ADX) and an exponential moving average (``alpha = 2/(n+1)``), because the two
    differ only in the constant.

    An empty cell inside the series produces an empty output at that row and leaves the
    running state untouched, so the recursion resumes from the last real observation
    rather than from a filled value. That is a convention, not a fact about the data, so
    it is reported as a note whenever a recursive feature is selected.
    """
    out = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    start = _first_full_window(finite, window)
    if start < 0:
        return out
    previous = float(values[start - window + 1 : start + 1].mean())
    out[start] = previous
    for index in range(start + 1, values.size):
        if not finite[index]:
            continue
        previous = alpha * float(values[index]) + (1.0 - alpha) * previous
        out[index] = previous
    return out


def _wilder(values: np.ndarray, window: int) -> np.ndarray:
    return _recursive(values, window, 1.0 / window)


def _rolling_slope(values: np.ndarray, window: int) -> np.ndarray:
    """OLS slope of ``values`` against the bar index over the trailing ``window`` rows.

    Because the regressor is always ``0, 1, … n−1``, the usual covariance numerator
    collapses: ``Σ(xᵢ−x̄)(yᵢ−ȳ) = Σ(xᵢ−x̄)yᵢ``. That turns the whole thing into one
    weighted sum per window, which is why this can be done without a Python-level loop
    over rows.
    """
    out = np.full(values.shape, np.nan, dtype=float)
    if window < 2 or values.size < window:
        return out
    weights = np.arange(window, dtype=float)
    weights -= weights.mean()
    denominator = float((weights**2).sum())
    span = values.size - window + 1
    for start in range(0, span, _CHUNK):
        stop = min(start + _CHUNK, span)
        block = np.lib.stride_tricks.sliding_window_view(values[start : stop + window - 1], window)
        usable = ~np.isnan(block).any(axis=1)
        if usable.any():
            out[start + window - 1 : stop + window - 1][usable] = (
                block[usable] * weights
            ).sum(axis=1) / denominator
    return out


def _rolling_argmax(values: np.ndarray, window: int) -> np.ndarray:
    """Position of the maximum within each trailing window, from the window's left edge."""
    out = np.full(values.shape, np.nan, dtype=float)
    if values.size < window:
        return out
    span = values.size - window + 1
    for start in range(0, span, _CHUNK):
        stop = min(start + _CHUNK, span)
        block = np.lib.stride_tricks.sliding_window_view(values[start : stop + window - 1], window)
        usable = ~np.isnan(block).any(axis=1)
        if usable.any():
            out[start + window - 1 : stop + window - 1][usable] = np.argmax(
                block[usable], axis=1
            )
    return out


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Wilder's true range. ``fmax`` propagates a value when only one side is empty."""
    previous_close = _shift(close, 1)
    return np.fmax(np.fmax(high - low, np.abs(high - previous_close)), np.abs(low - previous_close))


def _direction(up: np.ndarray, down: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Wilder's +DM and −DM: the larger of the two moves, and only if it is upward/downward."""
    plus = np.where((up > down) & (up > 0), up, 0.0)
    minus = np.where((down > up) & (down > 0), down, 0.0)
    seed = ~np.isfinite(up) | ~np.isfinite(down)
    plus[seed] = np.nan
    minus[seed] = np.nan
    return plus, minus


def _rolling(values: np.ndarray, window: int, how: str, ddof: int = 1) -> np.ndarray:
    """Trailing rolling statistic. ``min_periods=window``, so a gap yields an empty cell."""
    series = pd.Series(values)
    if how == "mean":
        return series.rolling(window, min_periods=window).mean().to_numpy()
    if how == "sum":
        return series.rolling(window, min_periods=window).sum().to_numpy()
    if how == "std":
        return series.rolling(window, min_periods=window).std(ddof=ddof).to_numpy()
    if how == "max":
        return series.rolling(window, min_periods=window).max().to_numpy()
    if how == "min":
        return series.rolling(window, min_periods=window).min().to_numpy()
    raise AssertionError(f"unknown rolling statistic {how!r}")  # noqa: S101


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide, leaving an empty cell wherever the denominator is zero or missing."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = numerator / denominator
    return np.where(np.isfinite(out), out, np.nan)


# --------------------------------------------------------------------------------------
# Time helpers
# --------------------------------------------------------------------------------------


def _local_index(times: pd.Series, timezone: str | None) -> pd.DatetimeIndex:
    """The bar timestamps, converted into the dataset's own timezone and made naive.

    Calendar features are a property of the *market's* day, not of the machine running the
    server, so the dataset's declared timezone decides where a day starts. A dataset that
    declares none is treated as UTC and the report says so.
    """
    index = pd.DatetimeIndex(pd.to_datetime(times))
    if index.tz is not None:
        index = index.tz_convert(timezone or "UTC").tz_localize(None)
    return index


# --------------------------------------------------------------------------------------
# Compute arguments
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Args:
    frame: pd.DataFrame
    #: The resolved price series — `close`, or the disclosed weighted series.
    price: np.ndarray
    #: The output column names, in the order the catalogue declares them.
    names: tuple[str, ...]
    bars: int
    token: str | None
    time_column: str
    timezone: str | None
    asset_class: AssetClass
    timeframe: Timeframe | None
    #: Set by the engine so a computer can record a one-off fact for the report.
    notes: list[str]

    def column(self, name: str) -> np.ndarray:
        return self.frame[name].to_numpy(dtype="float64")

    def local(self) -> pd.DatetimeIndex:
        return _local_index(self.frame[self.time_column], self.timezone)


Computer = Callable[[_Args], dict[str, np.ndarray]]


# --------------------------------------------------------------------------------------
# Computers
# --------------------------------------------------------------------------------------


def _price_lag(args: _Args) -> dict[str, np.ndarray]:
    return {args.names[0]: _shift(args.price, args.bars)}


def _high_low_range(args: _Args) -> dict[str, np.ndarray]:
    return {args.names[0]: args.column("high") - args.column("low")}


def _normalized_high_low_range(args: _Args) -> dict[str, np.ndarray]:
    return {
        args.names[0]: _safe_ratio(
            args.column("high") - args.column("low"), args.column("close")
        )
    }


def _candle_body_ratio(args: _Args) -> dict[str, np.ndarray]:
    return {
        args.names[0]: _safe_ratio(
            args.column("close") - args.column("open"),
            args.column("high") - args.column("low"),
        )
    }


def _typical_price(args: _Args) -> np.ndarray:
    return (args.column("high") + args.column("low") + args.column("close")) / 3.0


def _session_vwap(args: _Args, period: str) -> dict[str, np.ndarray]:
    typical = _typical_price(args)
    volume = args.column("volume")
    key = args.local().to_period(period)
    weighted = pd.Series(typical * volume).groupby(key).cumsum().to_numpy()
    total = pd.Series(volume).groupby(key).cumsum().to_numpy()
    return {args.names[0]: _safe_ratio(weighted, total)}


def _vwap_daily(args: _Args) -> dict[str, np.ndarray]:
    return _session_vwap(args, "D")


def _vwap_weekly(args: _Args) -> dict[str, np.ndarray]:
    return _session_vwap(args, "W")


def _vwap_monthly(args: _Args) -> dict[str, np.ndarray]:
    return _session_vwap(args, "M")


def _rolling_vwap(args: _Args, window: int) -> np.ndarray:
    typical = _typical_price(args)
    volume = args.column("volume")
    return _safe_ratio(
        _rolling(typical * volume, window, "sum"), _rolling(volume, window, "sum")
    )


def _vwap_rolling(args: _Args) -> dict[str, np.ndarray]:
    return {args.names[0]: _rolling_vwap(args, args.bars)}


def _distance_to_vwap(args: _Args) -> dict[str, np.ndarray]:
    vwap = _rolling_vwap(args, args.bars)
    return {args.names[0]: _safe_ratio(args.price - vwap, vwap)}


def _volume_change(args: _Args) -> dict[str, np.ndarray]:
    volume = args.column("volume")
    return {args.names[0]: _safe_ratio(volume, _shift(volume, args.bars)) - 1.0}


def _log_volume_change(args: _Args) -> dict[str, np.ndarray]:
    volume = args.column("volume")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(volume) - np.log(_shift(volume, args.bars))
    return {args.names[0]: np.where(np.isfinite(out), out, np.nan)}


def _turnover(args: _Args) -> dict[str, np.ndarray]:
    quote = args.frame.get("quote_volume")
    if quote is not None and bool(quote.notna().any()):
        args.notes.append(
            "Turnover was taken from the provider's quote_volume column, which is the "
            "exact notional traded and needs no reconstruction."
        )
        return {args.names[0]: quote.to_numpy(dtype="float64")}
    args.notes.append(
        "This dataset has no populated quote_volume column, so turnover was "
        "reconstructed as close × volume. That is the bar's notional only if every "
        "trade printed at the close, so it is a lower bound on a bar that trended."
    )
    return {args.names[0]: args.column("close") * args.column("volume")}


def _obv_values(args: _Args) -> np.ndarray:
    close = args.column("close")
    volume = args.column("volume")
    direction = np.sign(close - _shift(close, 1))
    contribution = direction * volume
    contribution[0] = np.nan
    out = np.full(close.shape, np.nan, dtype=float)
    running = 0.0
    for index in range(1, close.size):
        if not np.isfinite(contribution[index]):
            continue
        running += float(contribution[index])
        out[index] = running
    return out


def _obv(args: _Args) -> dict[str, np.ndarray]:
    return {args.names[0]: _obv_values(args)}


def _obv_slope_10(args: _Args) -> dict[str, np.ndarray]:
    return {args.names[0]: _rolling_slope(_obv_values(args), 10)}


def _return_1(args: _Args) -> dict[str, np.ndarray]:
    return {args.names[0]: _safe_ratio(args.price, _shift(args.price, 1)) - 1.0}


def _log_return_1(args: _Args) -> dict[str, np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(_safe_ratio(args.price, _shift(args.price, 1)))
    return {args.names[0]: out}


def _cumulative_return(args: _Args) -> dict[str, np.ndarray]:
    return {args.names[0]: _safe_ratio(args.price, args.price[:1]) - 1.0}


def _squared_return_1(args: _Args) -> dict[str, np.ndarray]:
    return {args.names[0]: _log_returns(args) ** 2}


def _absolute_return_1(args: _Args) -> dict[str, np.ndarray]:
    return {args.names[0]: np.abs(_log_returns(args))}


def _log_return_lag(args: _Args) -> dict[str, np.ndarray]:
    # The one-bar log return is itself already a backward view, so lagging it by n bars
    # puts a return that was fully realised n+1 bars earlier on row t.
    return {args.names[0]: _shift(_log_returns(args), args.bars)}


def _roc(args: _Args) -> dict[str, np.ndarray]:
    return {args.names[0]: _safe_ratio(args.price, _shift(args.price, args.bars)) - 1.0}


def _momentum(args: _Args) -> dict[str, np.ndarray]:
    return {args.names[0]: args.price - _shift(args.price, args.bars)}


def _log_returns(args: _Args) -> np.ndarray:
    return _log_return_1(args)[args.names[0]]


def _realized_vol(args: _Args) -> dict[str, np.ndarray]:
    return {args.names[0]: _rolling(_log_returns(args), args.bars, "std", ddof=1)}


def _annualisation_factor(
    timeframe: Timeframe | None, asset_class: AssetClass
) -> tuple[float, str] | None:
    """Bars per year, and a phrase explaining where the number came from.

    The factor is a convention, not a measurement, and getting it wrong scales every
    volatility by a constant — √(365/252) ≈ 20%, enough to move a position size. So it is
    derived from the dataset's own declarations and reported, never assumed.
    """
    if timeframe is None:
        return None
    seconds = TIMEFRAME_SECONDS[timeframe]
    crypto = asset_class in (AssetClass.CRYPTO_SPOT, AssetClass.CRYPTO_FUTURES)
    if crypto:
        return 365.0 * 86400.0 / seconds, "crypto trades every day, so 365 days a year"
    if seconds >= 86400:
        return 252.0 * 86400.0 / seconds, "equities trade about 252 days a year"
    # 6.5 hours of regular session, 09:30–16:00.
    return 252.0 * (23400.0 / seconds), (
        "equities trade about 252 days a year, each with a 6.5-hour regular session"
    )


def _annualized_vol(args: _Args) -> dict[str, np.ndarray]:
    factor = _annualisation_factor(args.timeframe, args.asset_class)
    assert factor is not None  # noqa: S101 — availability refuses a null timeframe
    per_year, why = factor
    args.notes.append(
        f"annualized_vol used {per_year:,.0f} bars per year ({why}), so a volatility of "
        f"1% per bar becomes {0.01 * per_year**0.5:.1%} per year."
    )
    return {args.names[0]: _realized_vol(args)[args.names[0]] * float(per_year) ** 0.5}


def _parkinson(args: _Args) -> dict[str, np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        squared = np.log(_safe_ratio(args.column("high"), args.column("low"))) ** 2
    return {args.names[0]: np.sqrt(_rolling(squared / (4.0 * np.log(2.0)), args.bars, "mean"))}


def _garman_klass(args: _Args) -> dict[str, np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        range_term = 0.5 * np.log(_safe_ratio(args.column("high"), args.column("low"))) ** 2
        body_term = (2.0 * np.log(2.0) - 1.0) * np.log(
            _safe_ratio(args.column("close"), args.column("open"))
        ) ** 2
    mean = _rolling(range_term - body_term, args.bars, "mean")
    args.notes.append(
        "Garman–Klass applies max(mean, 0) before the square root: a bar that gapped and "
        "then barely moved contributes a negative term, and a negative window mean has no "
        "square root."
    )
    return {args.names[0]: np.sqrt(np.maximum(mean, 0.0))}


def _rogers_satchell(args: _Args) -> dict[str, np.ndarray]:
    high, low = args.column("high"), args.column("low")
    close, open_ = args.column("close"), args.column("open")
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.log(_safe_ratio(high, close)) * np.log(_safe_ratio(high, open_)) + np.log(
            _safe_ratio(low, close)
        ) * np.log(_safe_ratio(low, open_))
    mean = _rolling(term, args.bars, "mean")
    return {args.names[0]: np.sqrt(np.maximum(mean, 0.0))}


def _max_drawdown(args: _Args) -> dict[str, np.ndarray]:
    price = args.price
    window = args.bars
    out = np.full(price.shape, np.nan, dtype=float)
    if price.size < window:
        return {args.names[0]: out}
    span = price.size - window + 1
    for start in range(0, span, _CHUNK):
        stop = min(start + _CHUNK, span)
        block = np.lib.stride_tricks.sliding_window_view(price[start : stop + window - 1], window)
        usable = ~np.isnan(block).any(axis=1)
        if not usable.any():
            continue
        selected = block[usable]
        running_peak = np.maximum.accumulate(selected, axis=1)
        drawdown = (selected / running_peak - 1.0).min(axis=1)
        out[start + window - 1 : stop + window - 1][usable] = drawdown
    return {args.names[0]: out}


def _true_range_series(args: _Args) -> np.ndarray:
    return _true_range(args.column("high"), args.column("low"), args.column("close"))


def _atr(args: _Args) -> dict[str, np.ndarray]:
    return {args.names[0]: _wilder(_true_range_series(args), args.bars)}


def _normalized_atr(args: _Args) -> dict[str, np.ndarray]:
    return {args.names[0]: _safe_ratio(_atr(args)[args.names[0]], args.column("close"))}


def _sma(args: _Args) -> dict[str, np.ndarray]:
    return {args.names[0]: _rolling(args.price, args.bars, "mean")}


def _ema(args: _Args) -> dict[str, np.ndarray]:
    return {args.names[0]: _recursive(args.price, args.bars, 2.0 / (args.bars + 1.0))}


def _sma_20_slope_5(args: _Args) -> dict[str, np.ndarray]:
    return {args.names[0]: _rolling_slope(_rolling(args.price, 20, "mean"), 5)}


def _ema_20_slope_5(args: _Args) -> dict[str, np.ndarray]:
    return {args.names[0]: _rolling_slope(_recursive(args.price, 20, 2.0 / 21.0), 5)}


def _distance_to_sma(args: _Args) -> dict[str, np.ndarray]:
    sma = _rolling(args.price, args.bars, "mean")
    return {args.names[0]: _safe_ratio(args.price - sma, sma)}


def _vwma_values(args: _Args, window: int) -> np.ndarray:
    volume = args.column("volume")
    return _safe_ratio(
        _rolling(args.price * volume, window, "sum"), _rolling(volume, window, "sum")
    )


def _vwma(args: _Args) -> dict[str, np.ndarray]:
    return {args.names[0]: _vwma_values(args, args.bars)}


def _distance_to_vwma(args: _Args) -> dict[str, np.ndarray]:
    vwma = _vwma_values(args, args.bars)
    return {args.names[0]: _safe_ratio(args.price - vwma, vwma)}


def _vwma_20_slope(args: _Args) -> dict[str, np.ndarray]:
    return {args.names[0]: _rolling_slope(_vwma_values(args, 20), args.bars)}


def _evwma(args: _Args) -> dict[str, np.ndarray]:
    """Elastic volume-weighted moving average.

    The memory is set by traded volume: the estimator is pulled toward price in
    proportion to how much volume has just printed. The trailing n-bar volume stands in
    for the share float the original indicator was defined against.
    """
    price = args.price
    volume = args.column("volume")
    window = args.bars
    total = _rolling(volume, window, "sum")
    out = np.full(price.shape, np.nan, dtype=float)
    finite = np.isfinite(price) & np.isfinite(volume)
    start = _first_full_window(finite, window)
    if start < 0:
        return {args.names[0]: out}
    previous = float(price[start - window + 1 : start + 1].mean())
    out[start] = previous
    for index in range(start + 1, price.size):
        working = total[index - 1]
        if not (np.isfinite(working) and np.isfinite(price[index]) and np.isfinite(volume[index])):
            continue
        if working <= 0:
            out[index] = previous
            continue
        previous = ((working - volume[index]) * previous + volume[index] * price[index]) / working
        out[index] = previous
    return {args.names[0]: out}


def _stoch_k(args: _Args, window: int = 14, smoothing: int = 3) -> np.ndarray:
    highest = _rolling(args.column("high"), window, "max")
    lowest = _rolling(args.column("low"), window, "min")
    raw = 100.0 * _safe_ratio(args.column("close") - lowest, highest - lowest)
    return _rolling(raw, smoothing, "mean")


def _stoch_k_14_3(args: _Args) -> dict[str, np.ndarray]:
    return {args.names[0]: _stoch_k(args)}


def _stoch_d_14_3_3(args: _Args) -> dict[str, np.ndarray]:
    return {args.names[0]: _rolling(_stoch_k(args), 3, "mean")}


def _directional(args: _Args, window: int = 14) -> tuple[np.ndarray, np.ndarray]:
    high, low = args.column("high"), args.column("low")
    plus_dm, minus_dm = _direction(high - _shift(high, 1), _shift(low, 1) - low)
    atr = _wilder(_true_range(high, low, args.column("close")), window)
    return (
        100.0 * _safe_ratio(_wilder(plus_dm, window), atr),
        100.0 * _safe_ratio(_wilder(minus_dm, window), atr),
    )


def _directional_index(args: _Args) -> dict[str, np.ndarray]:
    plus, minus = _directional(args)
    return {args.names[0]: plus, args.names[1]: minus}


def _dx(args: _Args, window: int) -> np.ndarray:
    """DX at the same period as the directional indicators it is built from.

    The window has no default on purpose. ADX(n) is Wilder's average of DX(n) — the
    textbook definition, and the one the catalogue's `2n − 1` warm-up describes — so a
    silent 14 here would make `adx_1` an average of DX(14) while the column name, the
    formula and the declared warm-up all say 1.
    """
    plus, minus = _directional(args, window)
    return 100.0 * _safe_ratio(np.abs(plus - minus), plus + minus)


def _adx(args: _Args) -> dict[str, np.ndarray]:
    return {args.names[0]: _wilder(_dx(args, args.bars), args.bars)}


def _aroon(args: _Args) -> dict[str, np.ndarray]:
    """Aroon over a window of ``n + 1`` bars, which is the indicator's own convention."""
    window = 15
    up = 100.0 * _rolling_argmax(args.column("high"), window) / (window - 1)
    down = 100.0 * _rolling_argmax(args.column("low"), window) / (window - 1)
    return {args.names[0]: up, args.names[1]: down, args.names[2]: up - down}


def _williams_r(args: _Args) -> dict[str, np.ndarray]:
    highest = _rolling(args.column("high"), 14, "max")
    lowest = _rolling(args.column("low"), 14, "min")
    return {
        args.names[0]: -100.0
        * _safe_ratio(highest - args.column("close"), highest - lowest)
    }


def _rsi_values(args: _Args, window: int) -> np.ndarray:
    delta = args.price - _shift(args.price, 1)
    finite = np.isfinite(delta)
    gain = np.where(finite & (delta > 0), delta, np.where(finite, 0.0, np.nan))
    loss = np.where(finite & (delta < 0), -delta, np.where(finite, 0.0, np.nan))
    average_gain = _wilder(gain, window)
    average_loss = _wilder(loss, window)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_strength = average_gain / average_loss
    index = 100.0 - 100.0 / (1.0 + relative_strength)
    # A window with no losses at all has infinite relative strength: RSI is 100. A window
    # with neither gains nor losses is flat, and a flat series has no RSI — 50 would be a
    # fabricated neutral reading, so it is left empty.
    flat = (average_gain == 0) & (average_loss == 0)
    return np.where(flat, np.nan, index)


def _rsi(args: _Args) -> dict[str, np.ndarray]:
    return {args.names[0]: _rsi_values(args, args.bars)}


def _rsi_slope_5(args: _Args) -> dict[str, np.ndarray]:
    return {args.names[0]: _rolling_slope(_rsi_values(args, 14), 5)}


def _distance_to_rsi_50(args: _Args) -> dict[str, np.ndarray]:
    return {args.names[0]: _rsi_values(args, 14) - 50.0}


def _bollinger(args: _Args) -> dict[str, np.ndarray]:
    middle = _rolling(args.price, 20, "mean")
    deviation = _rolling(args.price, 20, "std", ddof=0)
    upper = middle + 2.0 * deviation
    lower = middle - 2.0 * deviation
    return {
        args.names[0]: upper,
        args.names[1]: lower,
        args.names[2]: _safe_ratio(upper - lower, middle),
    }


def _macd(args: _Args) -> dict[str, np.ndarray]:
    line = _recursive(args.price, 12, 2.0 / 13.0) - _recursive(args.price, 26, 2.0 / 27.0)
    signal = _recursive(line, 9, 2.0 / 10.0)
    return {args.names[0]: line, args.names[1]: signal, args.names[2]: line - signal}


def _vortex(args: _Args) -> dict[str, np.ndarray]:
    high, low = args.column("high"), args.column("low")
    movement_plus = np.abs(high - _shift(low, 1))
    movement_minus = np.abs(low - _shift(high, 1))
    range_sum = _rolling(_true_range(high, low, args.column("close")), args.bars, "sum")
    return {
        args.names[0]: _safe_ratio(_rolling(movement_plus, args.bars, "sum"), range_sum),
        args.names[1]: _safe_ratio(_rolling(movement_minus, args.bars, "sum"), range_sum),
    }


def _time_component(args: _Args, kind: str) -> dict[str, np.ndarray]:
    local = args.local()
    if kind == "hour_of_day":
        values: Any = local.hour
    elif kind == "day_of_week":
        values = local.dayofweek
    elif kind == "day_of_month":
        values = local.day
    elif kind == "month":
        values = local.month
    elif kind == "quarter":
        values = local.quarter
    elif kind == "month_end":
        values = local.is_month_end.astype(int)
    elif kind == "days_to_month_end":
        values = ((local + pd.offsets.MonthEnd(0)).normalize() - local.normalize()).days
    elif kind == "days_to_quarter_end":
        values = ((local + pd.offsets.QuarterEnd(0)).normalize() - local.normalize()).days
    elif kind == "time_idx":
        values = np.arange(local.size)
    else:
        raise AssertionError(f"unknown time feature {kind!r}")  # noqa: S101
    return {args.names[0]: np.asarray(values)}


def _hour_of_day(args: _Args) -> dict[str, np.ndarray]:
    return _time_component(args, "hour_of_day")


def _day_of_week(args: _Args) -> dict[str, np.ndarray]:
    return _time_component(args, "day_of_week")


def _day_of_month(args: _Args) -> dict[str, np.ndarray]:
    return _time_component(args, "day_of_month")


def _month(args: _Args) -> dict[str, np.ndarray]:
    return _time_component(args, "month")


def _quarter(args: _Args) -> dict[str, np.ndarray]:
    return _time_component(args, "quarter")


def _month_end(args: _Args) -> dict[str, np.ndarray]:
    return _time_component(args, "month_end")


def _days_to_month_end(args: _Args) -> dict[str, np.ndarray]:
    return _time_component(args, "days_to_month_end")


def _days_to_quarter_end(args: _Args) -> dict[str, np.ndarray]:
    return _time_component(args, "days_to_quarter_end")


def _time_idx(args: _Args) -> dict[str, np.ndarray]:
    return _time_component(args, "time_idx")


def _carry(args: _Args) -> dict[str, np.ndarray]:
    # A carried column is already in the frame; it is recorded, not rebuilt. Rebuilding it
    # would risk the copy differing from the original, which is the one thing carrying
    # must never do.
    return {}


#: Every entry in the catalogue must appear here. `tests/test_features_catalog.py`
#: asserts it, so a feature cannot be declared available and then fail at run time.
COMPUTERS: dict[str, Computer] = {
    "price_lag": _price_lag,
    "high_low_range": _high_low_range,
    "normalized_high_low_range": _normalized_high_low_range,
    "candle_body_ratio": _candle_body_ratio,
    "vwap_daily": _vwap_daily,
    "vwap_weekly": _vwap_weekly,
    "vwap_monthly": _vwap_monthly,
    "vwap_rolling": _vwap_rolling,
    "distance_to_vwap": _distance_to_vwap,
    "volume": _carry,
    "quote_volume": _carry,
    "taker_buy_volume": _carry,
    "taker_buy_quote_volume": _carry,
    "volume_change": _volume_change,
    "log_volume_change": _log_volume_change,
    "turnover": _turnover,
    "obv": _obv,
    "obv_slope_10": _obv_slope_10,
    "return_1": _return_1,
    "log_return_1": _log_return_1,
    "cumulative_return": _cumulative_return,
    "squared_return_1": _squared_return_1,
    "absolute_return_1": _absolute_return_1,
    "log_return_lag": _log_return_lag,
    "roc": _roc,
    "momentum": _momentum,
    "realized_vol": _realized_vol,
    "annualized_vol": _annualized_vol,
    "parkinson": _parkinson,
    "garman_klass": _garman_klass,
    "rogers_satchell": _rogers_satchell,
    "max_drawdown": _max_drawdown,
    "atr": _atr,
    "normalized_atr": _normalized_atr,
    "sma": _sma,
    "ema": _ema,
    "sma_20_slope_5": _sma_20_slope_5,
    "ema_20_slope_5": _ema_20_slope_5,
    "distance_to_sma": _distance_to_sma,
    "vwma": _vwma,
    "distance_to_vwma": _distance_to_vwma,
    "vwma_20_slope": _vwma_20_slope,
    "evwma": _evwma,
    "stoch_k_14_3": _stoch_k_14_3,
    "stoch_d_14_3_3": _stoch_d_14_3_3,
    "directional_index": _directional_index,
    "aroon": _aroon,
    "williams_r_14": _williams_r,
    "rsi": _rsi,
    "rsi_slope_5": _rsi_slope_5,
    "distance_to_rsi_50": _distance_to_rsi_50,
    "bollinger": _bollinger,
    "macd": _macd,
    "adx": _adx,
    "vortex": _vortex,
    "hour_of_day": _hour_of_day,
    "day_of_week": _day_of_week,
    "day_of_month": _day_of_month,
    "month": _month,
    "quarter": _quarter,
    "month_end": _month_end,
    "days_to_month_end": _days_to_month_end,
    "days_to_quarter_end": _days_to_quarter_end,
    "time_idx": _time_idx,
    "last_funding_rate": _carry,
    "sum_open_interest": _carry,
    "count_long_short_ratio": _carry,
    "sum_taker_long_short_vol_ratio": _carry,
    "num_trades": _carry,
    "settlement": _carry,
    "yield_curve_slope": _carry,
}

#: Features whose value at *t* is built from earlier feature values rather than from a
#: trailing slice. Selecting one changes how a gap should be read, so it is reported.
_RECURSIVE = frozenset(
    {
        "ema",
        "ema_20_slope_5",
        "obv",
        "obv_slope_10",
        "evwma",
        "atr",
        "normalized_atr",
        "rsi",
        "rsi_slope_5",
        "distance_to_rsi_50",
        "macd",
        "adx",
        "directional_index",
    }
)


# --------------------------------------------------------------------------------------
# Price source
# --------------------------------------------------------------------------------------


def _resolve_price(frame: pd.DataFrame, source: PriceSource) -> tuple[np.ndarray, str, str]:
    """The series every price-derived feature is measured on, and how it was defined.

    The specification names ``weighted_price`` without saying what it is, so the choice is
    made explicit and reported rather than left to the reader. Both branches are stated in
    the report and written to provenance, because a downstream reader who assumes the
    other one is looking at a different dataset.
    """
    if source is PriceSource.CLOSE:
        if "close" not in frame.columns:
            raise ValidationFailure(
                "This dataset has no close column, so the close price cannot be used.",
                hint="Choose the weighted price, or merge in a dataset that has a close.",
            )
        return (
            frame["close"].to_numpy(dtype="float64"),
            "close",
            "close",
        )

    quote = frame.get("quote_volume")
    volume = frame.get("volume")
    if (
        quote is not None
        and volume is not None
        and bool(quote.notna().any())
        and bool(volume.notna().any())
    ):
        values = _safe_ratio(quote.to_numpy(dtype="float64"), volume.to_numpy(dtype="float64"))
        return (
            values,
            "quote_volume / volume — the exact volume-weighted average price of the bar",
            "quote_volume / volume",
        )

    if {"high", "low", "close"} <= set(frame.columns):
        values = (
            frame["high"].to_numpy(dtype="float64")
            + frame["low"].to_numpy(dtype="float64")
            + frame["close"].to_numpy(dtype="float64")
        ) / 3.0
        return (
            values,
            "(high + low + close) / 3 — the typical price, an estimate of the bar's VWAP",
            "(high + low + close) / 3",
        )

    raise ValidationFailure(
        "This dataset cannot produce a weighted price: it has neither "
        "quote_volume alongside volume, nor high/low/close.",
        hint="Use the close price, or merge in a dataset carrying those columns.",
    )


# --------------------------------------------------------------------------------------
# Selection resolution
# --------------------------------------------------------------------------------------


def _check_duplicates(request: FeatureRequest) -> None:
    seen: set[tuple[str, str | None]] = set()
    for selection in request.selections:
        key = (selection.feature_id, selection.window)
        if key in seen:
            label = selection.feature_id
            if selection.window is not None:
                label = f"{label} ({selection.window})"
            raise ValidationFailure(
                f"{label} is selected twice.",
                hint="Remove one of the two checkboxes with that window.",
            )
        seen.add(key)


def _resolve_one(selection: Selection, context: DatasetContext) -> Resolved:
    spec = catalogue.get(selection.feature_id)
    if spec is None:
        raise ValidationFailure(
            f"There is no feature called '{selection.feature_id}'.",
            hint="Reload the page: the catalogue may have changed.",
        )

    if spec.param is None:
        if selection.window is not None:
            raise ValidationFailure(f"{spec.label} takes no window.")
        return Resolved(spec=spec, token=None, bars=0, columns=spec.column_names())

    if selection.window is None:
        raise ValidationFailure(
            f"{spec.label} needs a window.",
            hint=f"Choose one of {', '.join(spec.param.ticks)}, or type your own.",
        )

    option: WindowOption
    if selection.window in spec.param.ticks:
        option = catalogue.window_option(spec, context, selection.window)
    else:
        option = catalogue.resolve_custom(spec, context, selection.window)

    if not option.available:
        raise ValidationFailure(
            f"{spec.label} cannot use the window '{selection.window}': {option.reason}",
        )
    return Resolved(
        spec=spec,
        token=option.token,
        bars=option.bars or 0,
        columns=spec.column_names(option.token),
    )


# --------------------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------------------


def compute_features(
    frame: pd.DataFrame,
    time_column: str | None,
    request: FeatureRequest,
    context: DatasetContext,
    *,
    timezone: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add the selected features to ``frame`` and describe exactly what was added.

    Returns the new frame and the report. The report is not decoration: it is the record
    of which price series was used, which columns were carried, how many leading rows each
    feature leaves empty, and which pairs of the result are near-duplicates of each other.
    """
    if not request.selections:
        raise ValidationFailure(
            "No features are selected.",
            hint="Tick at least one feature, or use a group's Select all.",
        )

    _check_duplicates(request)

    resolved: list[Resolved] = []
    taken: set[str] = set()
    for selection in request.selections:
        spec = catalogue.get(selection.feature_id)
        if spec is None:
            raise ValidationFailure(f"There is no feature called '{selection.feature_id}'.")
        available = catalogue.availability(spec, context)
        if not available.available:
            raise ValidationFailure(
                f"{spec.label} is not available on this dataset: {available.reason}"
            )
        if spec.id in catalogue.NEEDS_TIME and time_column is None:
            raise ValidationFailure(
                f"{spec.label} needs a time column, and this dataset has none."
            )
        item = _resolve_one(selection, context)
        for column in item.columns:
            if column in taken:
                raise ValidationFailure(
                    f"Two selections both produce a column called '{column}'.",
                    hint="Untick one of them.",
                )
            taken.add(column)
        resolved.append(item)

    computed = [item for item in resolved if item.spec.kind is FeatureKind.COMPUTED]
    carried = [item for item in resolved if item.spec.kind is FeatureKind.CARRIED]

    # Only *generated* columns can collide. A carried column is by definition one that is
    # already in the frame, so checking it here would refuse `volume`, `num_trades` and the
    # rest of the raw columns — the very entries the specification says to keep.
    generated_names = {column for item in computed for column in item.columns}
    collisions = sorted(column for column in generated_names if column in set(frame.columns))
    if collisions:
        raise ValidationFailure(
            "These columns already exist in the dataset: " + ", ".join(collisions) + ".",
            hint=(
                "Rename them in Resample & merge, or unselect the features that produce "
                "them. Renaming is done automatically nowhere, because a generated column "
                "that silently becomes `sma_20_2` makes the output schema depend on the "
                "input schema."
            ),
        )

    price, definition, basis = _resolve_price(frame, request.price_source)

    notes: list[str] = []
    result = frame.copy()
    added: list[dict[str, Any]] = []

    for item in computed:
        computer = COMPUTERS[item.spec.id]
        args = _Args(
            frame=frame,
            price=price,
            names=item.columns,
            bars=item.bars,
            token=item.token,
            time_column=time_column or "",
            timezone=timezone,
            notes=notes,
            asset_class=context.asset_class,
            timeframe=context.timeframe,
        )
        produced = computer(args)
        missing = [name for name in item.columns if name not in produced]
        if missing:
            raise AssertionError(  # noqa: S101 — a programming error, not user input
                f"{item.spec.id} did not produce {missing}"
            )
        for name in item.columns:
            result[name] = pd.Series(produced[name], index=frame.index, dtype="float64")
        added.append(
            {
                "feature_id": item.spec.id,
                "label": item.spec.label,
                "group": item.spec.group.value,
                "window": item.token,
                "window_bars": item.bars or None,
                "columns": list(item.columns),
                "warmup_bars": item.spec.warmup(item.bars),
            }
        )

    dropped: list[str] = []
    if request.keep_only_selected:
        keep = {*(column for item in resolved for column in item.columns)}
        if time_column:
            keep.add(time_column)
        dropped = [column for column in result.columns if column not in keep]
        result = result[[column for column in result.columns if column in keep]]

    # The longest of the selected warm-ups, not the first: the row the report dates as
    # usable is the row at which every selected column has a value. The same rule the
    # catalogue applies within an entry that emits several columns.
    warmup = max((item.spec.warmup(item.bars) for item in resolved), default=0)
    usable_from: str | None = None
    if time_column and warmup < len(frame):
        usable_from = str(pd.Timestamp(frame[time_column].iloc[warmup]))

    if any(item.spec.id in _RECURSIVE for item in computed):
        notes.append(
            "At least one selected feature is recursive, carrying state from bar to bar. "
            "Where such a series has an empty cell the recursion resumes from the last "
            "real observation instead of a filled value, and the affected rows are left "
            "empty rather than guessed."
        )
    if request.price_source is PriceSource.WEIGHTED:
        notes.append(f"The weighted price was resolved as: {definition}.")
    if context.category == "uploaded_data":
        notes.append(
            "This dataset was uploaded, so its asset class is a placeholder rather than "
            "a declaration. Anything that depends on the asset class — the volatility "
            "annualisation factor above all — should be read with that in mind."
        )
    if carried:
        notes.append(
            "Carried columns are only as point-in-time as the merge that produced them. "
            "A funding rate aligned forward rather than backward is already a look-ahead "
            "before it reaches this step."
        )

    generated = [column for item in computed for column in item.columns]
    report: dict[str, Any] = {
        "rows_before": int(len(frame)),
        "rows_after": int(len(result)),
        "price_source": {
            "source": request.price_source.value,
            "definition": definition,
            "basis": basis,
        },
        "added": added,
        "carried": [column for item in carried for column in item.columns],
        "generated_columns": generated,
        "dropped_columns": dropped,
        "warmup_bars": warmup,
        "usable_from": usable_from,
        "time_column": time_column or "",
        "timezone": timezone or "UTC",
        "notes": notes,
        "collinearity": _collinearity(result, generated),
        # Sent rather than restated in the frontend: the sentence above the table names the
        # threshold, and a number written twice is a number that can disagree with itself.
        "collinearity_threshold": COLLINEARITY_R,
        "selection_hash": _selection_hash(resolved, request),
    }
    return result, report


def _selection_hash(resolved: list[Resolved], request: FeatureRequest) -> str:
    """A stable digest of exactly what was asked for, for provenance and reproducibility."""
    entries = sorted(
        (
            {
                "feature_id": item.spec.id,
                "window": item.token,
                "bars": item.bars,
                "columns": list(item.columns),
            }
            for item in resolved
        ),
        key=lambda entry: (entry["feature_id"], entry["window"] or ""),
    )
    payload = {
        "price_source": request.price_source.value,
        "keep_only_selected": request.keep_only_selected,
        "selections": entries,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _collinearity(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    """Pairs of generated columns whose correlation is at least `COLLINEARITY_R`.

    Selecting SMA 5/10/20/50/100 produces five columns that are almost the same column.
    A model will not tell you that; this will. The scan is a description of the sample,
    not an input to any computation, so it does not violate causality.
    """
    usable = [column for column in columns if column in frame.columns]
    if len(usable) < 2 or len(frame) < 3:
        return []
    correlations = frame[usable].corr(numeric_only=True)
    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(usable):
        for right in usable[left_index + 1 :]:
            cell: Any = correlations.at[left, right]
            value = float(cell)
            if not np.isfinite(value) or abs(value) < COLLINEARITY_R:
                continue
            pairs.append({"a": left, "b": right, "r": round(value, 6)})
    pairs.sort(key=lambda item: (-abs(item["r"]), item["a"], item["b"]))
    return pairs[:COLLINEARITY_MAX_PAIRS]
