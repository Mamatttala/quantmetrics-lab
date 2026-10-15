"""The feature catalogue: one entry per feature the specification asks for.

This module is deliberately pure metadata and imports no numeric library. It is read by
three consumers that must never disagree — the engine (which computes), the API (which
serves it to the UI so the purple info icon shows the same formula the code implements),
and the tests (which assert that no entry exists without a stated formula). Keeping pandas
out of it also keeps it inside the async rule: everything under ``app/`` that touches
pandas must go through ``app/core/blocking.py``, and a catalogue that never touches it
cannot break that.

Two things the specification leaves implicit are made explicit here, because both would
otherwise be invented once per reader:

* the **formula** for every feature, shown verbatim in the UI;
* the **column name** each feature produces, with any baked-in constant (Bollinger's
  20/2, MACD's 12/26/9, Williams %R's 14) spelled into the name rather than hidden.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.models.enums import (
    TIMEFRAME_SECONDS,
    AssetClass,
    FeatureGroup,
    FeatureKind,
    Timeframe,
    WindowKind,
)

# --------------------------------------------------------------------------------------
# Column vocabulary
# --------------------------------------------------------------------------------------

#: Every source column the application can put in a frame, across both providers. A
#: feature may only declare a requirement drawn from this set; the integrity test enforces
#: it, so a typo in ``needs`` fails a test rather than silently disabling a feature.
SOURCE_COLUMNS: frozenset[str] = frozenset(
    {
        # Binance klines
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
        # Binance funding
        "calc_time",
        "funding_interval_hours",
        "last_funding_rate",
        # Binance open-interest metrics
        "create_time",
        "symbol",
        "sum_open_interest",
        "count_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
        # Yahoo equity futures carry a settlement column that is empty for every row the
        # provider actually returns; it exists in the schema so the shape is stable.
        "settlement",
        # Not fetched by this application. Present only so an uploaded CSV that carries a
        # macro series can have it kept as an exogenous feature.
        "yield_curve_slope",
    }
)

#: Asset-class gates. A feature is offered only to the classes named here.
CRYPTO: frozenset[AssetClass] = frozenset({AssetClass.CRYPTO_SPOT, AssetClass.CRYPTO_FUTURES})
CRYPTO_FUTURES: frozenset[AssetClass] = frozenset({AssetClass.CRYPTO_FUTURES})
EQUITY: frozenset[AssetClass] = frozenset({AssetClass.EQUITY_SPOT, AssetClass.EQUITY_FUTURES})
EQUITY_FUTURES: frozenset[AssetClass] = frozenset({AssetClass.EQUITY_FUTURES})

# --------------------------------------------------------------------------------------
# Duration windows
# --------------------------------------------------------------------------------------

#: A window written as a duration ("4h", "30d") is an interval, not a count of candles, so
#: it has to be converted against the dataset's timeframe before it means anything. Only
#: the durations the specification actually names are listed; anything else can still be
#: typed into the custom box and is parsed by `parse_duration`.
WINDOW_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
    "2d": 172800,
    "3d": 259200,
    "7d": 604800,
    "10d": 864000,
    "14d": 1209600,
    "20d": 1728000,
    "21d": 1814400,
    "30d": 2592000,
    "60d": 5184000,
    "90d": 7776000,
}

_DURATION_RE = re.compile(r"^(\d+)([mhdw])$")
_UNIT_SECONDS = {"m": 60, "h": 3600, "d": 86400, "w": 604800}

#: A custom window wider than this is refused. It is not a statistical rule, it is a
#: guard against a typo like `999999999d` turning into an allocation the machine cannot
#: serve; the frame's own length is the real limit and is checked separately.
MAX_CUSTOM_BARS = 100_000


def parse_duration(text: str) -> int | None:
    """Seconds in a duration written as ``<int><m|h|d|w>``, or ``None`` if unparseable."""
    match = _DURATION_RE.match(text.strip().lower())
    if match is None:
        return None
    return int(match.group(1)) * _UNIT_SECONDS[match.group(2)]


def duration_bars(seconds: int, timeframe: Timeframe | None) -> int | None:
    """Whole bars of ``timeframe`` in ``seconds``, or ``None`` if it is not a whole number.

    The exactness is the point. A "4h" window on a 1h dataset is four bars; on a 3h
    dataset it is not a whole number of bars, and rounding to one bar would silently turn
    a four-hour return into a three-hour one. Refusing is the only honest answer.
    """
    if timeframe is None:
        return None
    bar_seconds = TIMEFRAME_SECONDS[timeframe]
    if seconds % bar_seconds != 0:
        return None
    return seconds // bar_seconds


def is_intraday(timeframe: Timeframe | None) -> bool:
    """True when the bar is shorter than a day, which is what gates the clock features."""
    return timeframe is not None and TIMEFRAME_SECONDS[timeframe] < 86400


# --------------------------------------------------------------------------------------
# Dataset context and availability
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetContext:
    """Everything availability depends on. Derived from `DatasetMeta`, never guessed."""

    asset_class: AssetClass
    timeframe: Timeframe | None
    columns: frozenset[str]
    #: Columns present in the schema but empty in every row — Yahoo's equity `settlement`,
    #: for instance. Kept available-but-useless would be worse than disabled-with-a-reason.
    unpopulated: frozenset[str] = frozenset()
    category: str = ""
    #: Whether a time column could be identified. Checked at read time, because a dataset's
    #: metadata records its timestamps but not which column holds them.
    has_time: bool = True
    #: Whether the timestamps themselves show more than one time of day. ``None`` means the
    #: question could not be answered from the data, in which case the declared timeframe
    #: decides. This is asked of the timestamps rather than of the metadata because an
    #: uploaded file declares no timeframe at all, and refusing `hour_of_day` on such a file
    #: would hide a column that is perfectly well defined in it.
    intraday: bool | None = None


@dataclass(frozen=True)
class WindowOption:
    """One tick a parameterised feature offers, and whether this dataset can support it."""

    token: str
    bars: int | None
    available: bool
    reason: str | None


@dataclass(frozen=True)
class Availability:
    available: bool
    reason: str | None
    windows: tuple[WindowOption, ...] = ()


Guard = Callable[[DatasetContext], "str | None"]


# --------------------------------------------------------------------------------------
# Specs
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureParam:
    """The one varying parameter a feature may take.

    ``ticks`` are spelled exactly as the request spells them: ``"20"`` for a bar count
    and ``"4h"`` for a duration. The engine resolves a duration to bars once, server-side,
    and echoes both in the report, so the request stays self-describing.
    """

    kind: WindowKind
    ticks: tuple[str, ...]
    default: str
    minimum: int = 1
    #: The specification's "blank option in which the user can write the number of candles
    #: he wants, plus the option of adding more". A custom value is accepted unless the
    #: feature says otherwise — every feature here can take one.
    custom: bool = True

    @property
    def maximum(self) -> int:
        """A ceiling for a custom bar count. Durations have their own parser."""
        return MAX_CUSTOM_BARS


@dataclass(frozen=True)
class FeatureSpec:
    id: str
    label: str
    group: FeatureGroup
    #: Shown verbatim in the info icon. Never restated in the frontend.
    formula: str
    detail: str
    kind: FeatureKind
    #: Templates; ``{arg}`` is replaced by the resolved token. Fixed features have none.
    columns: tuple[str, ...]
    #: Number of leading rows the value is undefined for, given the resolved window — the
    #: index of the first row that has a value. One definition, so a test can assert the
    #: exact NaN count rather than a range.
    #:
    #: An entry with several columns declares the row at which *every* one of them has a
    #: value, which is the last of their individual first-defined rows rather than the
    #: first. MACD is the case that makes it concrete: the line exists from row 25, its
    #: signal from row 33, and the entry declares 33 — because this number is what
    #: `usable_from` reports, and a row where the signal is still empty is not a usable
    #: row for the feature. Declaring the earliest instead would name a row the user
    #: cannot use; declaring a later one would discard rows that are populated.
    warmup: Callable[[int], int]
    param: FeatureParam | None = None
    needs: tuple[str, ...] = ()
    #: True when the feature reads the request's selected price source.
    needs_price: bool = True
    caution: str | None = None
    asset_classes: frozenset[AssetClass] | None = None

    @property
    def parameterised(self) -> bool:
        return self.param is not None

    def column_names(self, token: str | None = None) -> tuple[str, ...]:
        if token is None:
            return self.columns
        return tuple(template.format(arg=token) for template in self.columns)


def _fixed(
    id: str,
    label: str,
    group: FeatureGroup,
    *,
    formula: str,
    detail: str,
    columns: tuple[str, ...],
    warmup: int,
    needs: tuple[str, ...] = (),
    needs_price: bool = True,
    caution: str | None = None,
    asset_classes: frozenset[AssetClass] | None = None,
    kind: FeatureKind = FeatureKind.COMPUTED,
) -> FeatureSpec:
    return FeatureSpec(
        id=id,
        label=label,
        group=group,
        formula=formula,
        detail=detail,
        kind=kind,
        columns=columns,
        warmup=lambda _bars: warmup,
        needs=needs,
        needs_price=needs_price,
        caution=caution,
        asset_classes=asset_classes,
    )


def _bars(
    id: str,
    label: str,
    group: FeatureGroup,
    *,
    formula: str,
    detail: str,
    ticks: tuple[int, ...],
    columns: tuple[str, ...],
    warmup: Callable[[int], int],
    minimum: int = 1,
    default: int | None = None,
    needs: tuple[str, ...] = (),
    needs_price: bool = True,
    caution: str | None = None,
    asset_classes: frozenset[AssetClass] | None = None,
) -> FeatureSpec:
    """A feature whose parameter is a count of candles. Available on every dataset."""
    first = default if default is not None else ticks[0]
    return FeatureSpec(
        id=id,
        label=label,
        group=group,
        formula=formula,
        detail=detail,
        kind=FeatureKind.COMPUTED,
        columns=columns,
        warmup=warmup,
        param=FeatureParam(
            kind=WindowKind.BARS,
            ticks=tuple(str(t) for t in ticks),
            default=str(first),
            minimum=minimum,
        ),
        needs=needs,
        needs_price=needs_price,
        caution=caution,
        asset_classes=asset_classes,
    )


def _duration(
    id: str,
    label: str,
    group: FeatureGroup,
    *,
    formula: str,
    detail: str,
    ticks: tuple[str, ...],
    columns: tuple[str, ...],
    warmup: Callable[[int], int],
    minimum: int = 1,
    default: str | None = None,
    needs: tuple[str, ...] = (),
    needs_price: bool = True,
    caution: str | None = None,
) -> FeatureSpec:
    """A feature whose parameter is an interval. Needs a declared timeframe to resolve."""
    return FeatureSpec(
        id=id,
        label=label,
        group=group,
        formula=formula,
        detail=detail,
        kind=FeatureKind.COMPUTED,
        columns=columns,
        warmup=warmup,
        param=FeatureParam(
            kind=WindowKind.DURATION,
            ticks=ticks,
            default=default if default is not None else ticks[0],
            minimum=minimum,
        ),
        needs=needs,
        needs_price=needs_price,
        caution=caution,
    )


def _carried(
    column: str,
    label: str,
    group: FeatureGroup,
    *,
    detail: str,
    size_note: str = "",
    asset_classes: frozenset[AssetClass] | None = None,
    caution: str | None = None,
) -> FeatureSpec:
    """A column that is kept, never computed — the specification's exogenous features.

    Nothing is calculated, so there is no formula to state and no methodology to invent.
    The one rule is that the column must actually be there and actually be populated.
    """
    return FeatureSpec(
        id=column,
        label=label,
        group=group,
        formula=f"kept as-is: {column}[t]",
        detail=(
            f"Not computed. This is the dataset's own `{column}` column, carried into the "
            f"processed dataset unchanged so it is available alongside the generated "
            f"features. {size_note}".strip()
        ),
        kind=FeatureKind.CARRIED,
        columns=(column,),
        warmup=lambda _bars: 0,
        needs=(column,),
        needs_price=False,
        asset_classes=asset_classes,
        caution=caution,
    )


# --------------------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------------------

#: Timestamp columns that describe when something *finished*, not when it was observed:
#: ``close_time`` is the instant a bar closed and ``create_time``/``calc_time`` are
#: publication instants for the metrics and funding series. A feature whose value is
#: stamped with the close of bar t already contains bar t, so building one of these into
#: an input is the exact look-ahead Part A's leakage rule exists to prevent. No entry may
#: require one, and the integrity test reads this set rather than restating it.
LEAKY_COLUMNS: frozenset[str] = frozenset(
    {"close_time", "calc_time", "create_time", "open_time"}
)

#: Features that read the time column: the clock and calendar features, and the session
#: VWAPs, whose whole definition is "since the start of the current session".
NEEDS_TIME: frozenset[str] = frozenset(
    {
        "vwap_daily",
        "vwap_weekly",
        "vwap_monthly",
        "hour_of_day",
        "day_of_week",
        "day_of_month",
        "month",
        "quarter",
        "month_end",
        "days_to_month_end",
        "days_to_quarter_end",
        "time_idx",
    }
)

#: Features that cannot be computed without a declared bar length. Annualized volatility
#: is the only one, and it is here rather than merely warned about because its scaling
#: factor is not a detail: √(365/252) is about 20%, which is the difference between two
#: position sizes. A dataset with no timeframe makes the correct factor unknowable, and
#: guessing one would be a silent multiplicative error rather than a missing value.
NEEDS_TIMEFRAME: frozenset[str] = frozenset({"annualized_vol"})


def _needs_intraday(context: DatasetContext) -> str | None:
    """Why a clock-time feature cannot say anything here, or ``None`` if it can.

    The timestamps are asked first: a dataset whose bars all fall at the same time of day
    would get a constant column, whatever its metadata claims. Only when the timestamps
    cannot answer — no time column, or an unreadable one — does the declared timeframe
    decide, and if that is missing too the feature is refused rather than guessed at.
    """
    if context.intraday is False:
        return (
            "Every bar in this dataset falls at the same time of day, so a column holding "
            "that time would be constant and carry no information."
        )
    if context.intraday is True:
        return None
    if context.timeframe is None:
        return (
            "This dataset declares no timeframe and its timestamps do not show more than "
            "one time of day, so there is no time of day to record."
        )
    if not is_intraday(context.timeframe):
        return (
            "This dataset's bars are a day or longer, so every bar in it has the same "
            "time of day and the column would carry no information."
        )
    return None


# --------------------------------------------------------------------------------------
# The catalogue
# --------------------------------------------------------------------------------------

_CATALOGUE: tuple[FeatureSpec, ...] = (
    # ---------------------------------------------------------------------------- PRICE
    FeatureSpec(
        id="price_lag",
        label="Price lag",
        group=FeatureGroup.PRICE,
        formula="price_lag(n)[t] = P[t − n]",
        detail=(
            "The selected price series shifted back by n bars. A plain backward shift: "
            "row t carries the value that was already known n bars earlier, with no "
            "interpolation and no fill."
        ),
        kind=FeatureKind.COMPUTED,
        columns=("price_lag_{arg}",),
        warmup=lambda n: n,
        param=FeatureParam(WindowKind.BARS, ("1", "2", "3", "5", "10", "20", "30"), "1"),
        needs_price=True,
    ),
    _fixed(
        "high_low_range",
        "High–low range",
        FeatureGroup.PRICE,
        formula="high_low_range[t] = high[t] − low[t]",
        detail="The bar's full trading range in price units.",
        columns=("high_low_range",),
        warmup=0,
        needs=("high", "low"),
        needs_price=False,
    ),
    _fixed(
        "normalized_high_low_range",
        "Normalized high–low range",
        FeatureGroup.PRICE,
        formula="normalized_high_low_range[t] = (high[t] − low[t]) / close[t]",
        detail=(
            "The bar's range as a fraction of its close, so it is comparable across "
            "assets and across time. This is the scale-free form the volatility "
            "estimators below generalise."
        ),
        columns=("normalized_high_low_range",),
        warmup=0,
        needs=("high", "low", "close"),
        needs_price=False,
    ),
    _fixed(
        "candle_body_ratio",
        "Candle body ratio",
        FeatureGroup.PRICE,
        formula="candle_body_ratio[t] = (close[t] − open[t]) / (high[t] − low[t])",
        detail=(
            "Where the close sits relative to the bar's body, in [−1, 1] while the bar "
            "has any range at all. Positive means the bar closed above its open."
        ),
        columns=("candle_body_ratio",),
        warmup=0,
        needs=("open", "high", "low", "close"),
        needs_price=False,
        caution=(
            "A bar with high == low has no range; the ratio is left empty rather than "
            "set to zero, because zero would read as 'a doji' rather than 'undefined'."
        ),
    ),
    _fixed(
        "vwap_daily",
        "VWAP (session, daily)",
        FeatureGroup.PRICE,
        formula="vwap_daily[t] = Σ(session, ≤t) typical·volume / Σ(session, ≤t) volume",
        detail=(
            "Volume-weighted average price accumulated from the start of the current "
            "calendar day up to and including bar t, using typical price "
            "(high + low + close)/3. The accumulation resets at each session boundary."
        ),
        columns=("vwap_daily",),
        warmup=0,
        needs=("high", "low", "close", "volume"),
        needs_price=False,
        caution=(
            "Anchored on the dataset's declared timezone. The session boundary is "
            "therefore a property of the data, not of this machine's clock."
        ),
    ),
    _fixed(
        "vwap_weekly",
        "VWAP (session, weekly)",
        FeatureGroup.PRICE,
        formula="vwap_weekly[t] = Σ(week, ≤t) typical·volume / Σ(week, ≤t) volume",
        detail="As daily VWAP, with the accumulation resetting at each calendar week.",
        columns=("vwap_weekly",),
        warmup=0,
        needs=("high", "low", "close", "volume"),
        needs_price=False,
    ),
    _fixed(
        "vwap_monthly",
        "VWAP (session, monthly)",
        FeatureGroup.PRICE,
        formula="vwap_monthly[t] = Σ(month, ≤t) typical·volume / Σ(month, ≤t) volume",
        detail="As daily VWAP, with the accumulation resetting at each calendar month.",
        columns=("vwap_monthly",),
        warmup=0,
        needs=("high", "low", "close", "volume"),
        needs_price=False,
    ),
    _bars(
        "vwap_rolling",
        "VWAP (rolling)",
        FeatureGroup.PRICE,
        formula="vwap(n)[t] = Σ_{i<t−n+1..t} typical·volume / Σ volume",
        detail=(
            "Volume-weighted average price over the trailing n bars, using typical price. "
            "Unlike the session VWAP this window does not reset, so it is continuous "
            "across session boundaries."
        ),
        ticks=(10, 20, 50),
        columns=("vwap_{arg}",),
        warmup=lambda n: n - 1,
        needs=("high", "low", "close", "volume"),
        needs_price=False,
    ),
    _bars(
        "distance_to_vwap",
        "Distance to VWAP (rolling)",
        FeatureGroup.PRICE,
        formula="distance_to_vwap(n)[t] = (P[t] − vwap(n)[t]) / vwap(n)[t]",
        detail=(
            "How far the selected price sits above or below the trailing VWAP, as a "
            "fraction of that VWAP."
        ),
        ticks=(10, 20, 50),
        columns=("distance_to_vwap_{arg}",),
        warmup=lambda n: n - 1,
        needs=("high", "low", "close", "volume"),
    ),
    # --------------------------------------------------------------------------- VOLUME
    _carried("volume", "Volume (carried)", FeatureGroup.VOLUME, detail=""),
    _carried("quote_volume", "Quote volume (carried)", FeatureGroup.VOLUME, detail=""),
    _carried(
        "taker_buy_volume",
        "Taker buy volume (carried)",
        FeatureGroup.VOLUME,
        detail="",
    ),
    _carried(
        "taker_buy_quote_volume",
        "Taker buy quote volume (carried)",
        FeatureGroup.VOLUME,
        detail="",
    ),
    _duration(
        "volume_change",
        "Volume change",
        FeatureGroup.VOLUME,
        formula="volume_change(w)[t] = volume[t] / volume[t − w] − 1",
        detail="Proportional change in volume over the window.",
        ticks=("15m", "1h", "4h", "1d", "7d", "21d"),
        columns=("volume_change_{arg}",),
        warmup=lambda n: n,
        needs=("volume",),
        needs_price=False,
    ),
    _duration(
        "log_volume_change",
        "Log volume change",
        FeatureGroup.VOLUME,
        formula="log_volume_change(w)[t] = ln(volume[t]) − ln(volume[t − w])",
        detail=(
            "The log difference. Preferred to the proportional change when volume spans "
            "orders of magnitude, because it is symmetric: doubling and halving are +ln2 "
            "and −ln2 rather than +1 and −0.5."
        ),
        ticks=("15m", "1h", "4h", "1d", "7d", "21d"),
        columns=("log_volume_change_{arg}",),
        warmup=lambda n: n,
        needs=("volume",),
        needs_price=False,
    ),
    _fixed(
        "turnover",
        "Turnover (notional)",
        FeatureGroup.VOLUME,
        formula="turnover[t] = quote_volume[t]  (else close[t] · volume[t])",
        detail=(
            "The notional value traded in the bar. Taken directly from the provider's "
            "quote volume where the dataset has it, and otherwise reconstructed as "
            "close × volume. Which of the two was used is recorded in the report."
        ),
        columns=("turnover",),
        warmup=0,
        needs=("volume", "close"),
        needs_price=False,
    ),
    _fixed(
        "obv",
        "On-balance volume",
        FeatureGroup.VOLUME,
        formula="obv[t] = obv[t−1] + sign(close[t] − close[t−1]) · volume[t]",
        detail=(
            "A running total of volume, signed by the direction of the close-to-close "
            "move. It is a level, not a rate, so its starting value is arbitrary — only "
            "its changes carry information."
        ),
        columns=("obv",),
        warmup=1,
        needs=("volume", "close"),
        needs_price=False,
        caution=(
            "OBV is a cumulative sum, so it drifts with the sample and is not comparable "
            "between two datasets that start at different times. Prefer obv_slope_10 or "
            "a difference of it."
        ),
    ),
    _fixed(
        "obv_slope_10",
        "OBV slope (10)",
        FeatureGroup.VOLUME,
        formula="obv_slope_10[t] = OLS slope of obv against bar index over the last 10 bars",
        detail=(
            "The ordinary-least-squares slope of on-balance volume regressed on the bar "
            "index over the trailing ten bars — a local rate of accumulation, scaled per "
            "bar. Being a slope it drops the arbitrary starting level that makes raw OBV "
            "hard to compare."
        ),
        columns=("obv_slope_10",),
        warmup=10,
        needs=("volume", "close"),
        needs_price=False,
    ),
    # -------------------------------------------------------------------------- RETURNS
    _fixed(
        "return_1",
        "Simple return (1 bar)",
        FeatureGroup.RETURNS,
        formula="return_1[t] = P[t] / P[t−1] − 1",
        detail="One-bar proportional return of the selected price series.",
        columns=("return_1",),
        warmup=1,
    ),
    _fixed(
        "log_return_1",
        "Log return (1 bar)",
        FeatureGroup.RETURNS,
        formula="log_return_1[t] = ln(P[t] / P[t−1])",
        detail=(
            "One-bar log return. It is the return the rest of this application measures "
            "volatility with, because log returns are additive through time."
        ),
        columns=("log_return_1",),
        warmup=1,
    ),
    _fixed(
        "cumulative_return",
        "Cumulative return",
        FeatureGroup.RETURNS,
        formula="cumulative_return[t] = P[t] / P[0] − 1",
        detail=(
            "Total return since the first row of the dataset. The base row is the "
            "dataset's own first row, so this column changes if the dataset is truncated "
            "— it is a description of the sample, not a point-in-time signal."
        ),
        columns=("cumulative_return",),
        warmup=0,
        caution=(
            "Measured from the first row of the data, so every row carries the sample's "
            "start date with it. Safe as a chart, unsafe as a model input under a "
            "chronological split, where training and test have different bases."
        ),
    ),
    _fixed(
        "squared_return_1",
        "Squared return (1 bar)",
        FeatureGroup.RETURNS,
        formula="squared_return_1[t] = log_return_1[t]²",
        detail=(
            "The one-bar squared log return. Its trailing average is realized variance; "
            "it is the standard proxy for that bar's realised volatility."
        ),
        columns=("squared_return_1",),
        warmup=1,
    ),
    _fixed(
        "absolute_return_1",
        "Absolute return (1 bar)",
        FeatureGroup.RETURNS,
        formula="absolute_return_1[t] = |log_return_1[t]|",
        detail=(
            "The one-bar absolute log return — a more robust volatility proxy than the "
            "squared return when the sample contains jumps."
        ),
        columns=("absolute_return_1",),
        warmup=1,
    ),
    _bars(
        "log_return_lag",
        "Log return lag",
        FeatureGroup.RETURNS,
        formula="log_return_lag(n)[t] = ln(P[t−n] / P[t−n−1])",
        detail=(
            "The one-bar log return, shifted back a further n bars. Row t carries a "
            "return that was fully realised n+1 bars before t, so it is usable as a "
            "predictor without any shift of its own."
        ),
        ticks=(1, 2, 3, 5, 10),
        columns=("log_return_lag_{arg}",),
        warmup=lambda n: n + 1,
    ),
    _duration(
        "roc",
        "Rate of change",
        FeatureGroup.RETURNS,
        formula="roc(w)[t] = P[t] / P[t − w] − 1",
        detail="Proportional change in price over the window.",
        ticks=("15m", "1h", "4h", "1d", "7d", "10d", "20d", "30d"),
        columns=("roc_{arg}",),
        warmup=lambda n: n,
    ),
    _duration(
        "momentum",
        "Momentum",
        FeatureGroup.RETURNS,
        formula="momentum(w)[t] = P[t] − P[t − w]",
        detail=(
            "Change in price over the window, in price units. Unlike rate of change it "
            "is not scale-free, so it is not comparable across assets with different "
            "price levels."
        ),
        ticks=("15m", "1h", "4h", "1d", "7d", "10d", "20d", "30d"),
        columns=("momentum_{arg}",),
        warmup=lambda n: n,
        caution=(
            "In raw price units, so a $60,000 asset and a $3 asset produce quantities "
            "that cannot be compared or pooled."
        ),
    ),
    # ----------------------------------------------------------------------- VOLATILITY
    _duration(
        "realized_vol",
        "Realized volatility",
        FeatureGroup.VOLATILITY,
        formula="realized_vol(w)[t] = sqrt( (1/(n−1)) · Σ_{i=0}^{n−1} (r[t−i] − r̄)² )",
        detail=(
            "The sample standard deviation of the one-bar log returns over the trailing "
            "window, where n is the window expressed in bars. Not annualised: it is "
            "already in the units of one bar's return."
        ),
        ticks=("15m", "1h", "4h", "12h", "1d", "7d", "21d", "30d"),
        columns=("realized_vol_{arg}",),
        warmup=lambda n: n,
        minimum=2,
        caution=(
            "Offered only when the window is at least two bars. A one-bar realized "
            "volatility is a single absolute return wearing a volatility's name, and the "
            "sample standard deviation of one observation is undefined."
        ),
    ),
    _duration(
        "annualized_vol",
        "Annualized volatility",
        FeatureGroup.VOLATILITY,
        formula="annualized_vol(w)[t] = realized_vol(w)[t] · sqrt(bars per year)",
        detail=(
            "The trailing realized volatility, measured over the window w, scaled to a "
            "yearly figure by sqrt(bars per year). The factor is derived from the "
            "dataset's own declared timeframe and asset class and written into the "
            "report so it is always inspectable: crypto trades 365 days a year, equities "
            "about 252. The square root is what makes the result a per-year volatility "
            "rather than a per-window one — volatility scales with the square root of "
            "time, not linearly in it."
        ),
        ticks=("15m", "1h", "4h", "1d", "7d", "21d", "30d"),
        columns=("annualized_vol_{arg}",),
        warmup=lambda n: n,
        default="1d",
        minimum=2,
        caution=(
            "Sensitive to the annualisation factor, which is a convention rather than a "
            "property of the data. 365/252 differs by about 20% — large enough to change "
            "a position size. The factor actually used is reported. The window is a "
            "duration, so it must divide the bar length: a 15m window on a daily dataset "
            "is refused rather than rounded to one bar."
        ),
    ),
    _bars(
        "parkinson",
        "Parkinson volatility",
        FeatureGroup.VOLATILITY,
        formula="sqrt( (1/(4n·ln2)) · Σ ln(high/low)² )  over the trailing n bars",
        detail=(
            "A range-based volatility estimator using only the bar's high and low. It is "
            "substantially more efficient than close-to-close when prices move "
            "continuously, but it is biased downward when the true path is a jump."
        ),
        ticks=(1, 5, 10, 21, 63),
        columns=("parkinson_{arg}",),
        warmup=lambda n: n - 1,
        needs=("high", "low"),
        needs_price=False,
    ),
    _bars(
        "garman_klass",
        "Garman–Klass volatility",
        FeatureGroup.VOLATILITY,
        formula=(
            "sqrt( (1/n) · Σ [ ½·ln(high/low)² − (2ln2−1)·ln(close/open)² ] ) "
            "over the trailing n bars"
        ),
        detail=(
            "Uses the whole bar — open, high, low, close — and is the most efficient of "
            "the three range estimators under continuous prices."
        ),
        ticks=(1, 5, 10, 21, 63),
        columns=("garman_klass_{arg}",),
        warmup=lambda n: n - 1,
        needs=("open", "high", "low", "close"),
        needs_price=False,
        caution=(
            "The summand can be negative for an individual bar (a bar that gapped and "
            "then barely moved), so a window's mean can in principle go negative and the "
            "square root of it is taken over max(mean, 0)."
        ),
    ),
    _bars(
        "rogers_satchell",
        "Rogers–Satchell volatility",
        FeatureGroup.VOLATILITY,
        formula=(
            "sqrt( (1/n) · Σ [ ln(high/close)·ln(high/open) + ln(low/close)·ln(low/open) ] )"
        ),
        detail=(
            "A drift-independent range estimator: it stays unbiased when the price trends "
            "within the window, where Parkinson and Garman–Klass do not."
        ),
        ticks=(1, 5, 10, 21, 63),
        columns=("rogers_satchell_{arg}",),
        warmup=lambda n: n - 1,
        needs=("open", "high", "low", "close"),
        needs_price=False,
    ),
    _duration(
        "max_drawdown",
        "Max drawdown",
        FeatureGroup.VOLATILITY,
        formula="max_drawdown(w)[t] = min_{i in window} ( P[i] / max_{j ≤ i} P[j] − 1 )",
        detail=(
            "The deepest peak-to-trough decline in the selected price over the trailing "
            "window, as a negative fraction. The running peak is taken from within the "
            "window only, so the value is a property of that window and not of the whole "
            "sample."
        ),
        ticks=("4h", "1d", "7d", "10d", "20d", "30d", "60d"),
        columns=("max_drawdown_{arg}",),
        warmup=lambda n: n - 1,
        minimum=2,
    ),
    _bars(
        "atr",
        "Average true range",
        FeatureGroup.VOLATILITY,
        formula=(
            "TR[t] = max(high−low, |high−close[t−1]|, |low−close[t−1]|); "
            "atr(n)[t] = Wilder average of TR over n bars"
        ),
        detail=(
            "The trailing mean of true range, where true range allows for the gap from "
            "the previous close. Smoothed by Wilder's recursive average, chosen because "
            "it is what ATR conventionally means."
        ),
        ticks=(1, 7, 14, 21),
        columns=("atr_{arg}",),
        warmup=lambda n: n - 1,
        needs=("high", "low", "close"),
        needs_price=False,
        caution=(
            "In price units, so it scales with the instrument's price level. Use "
            "normalized_atr to compare across assets or across time."
        ),
    ),
    _bars(
        "normalized_atr",
        "Normalized ATR",
        FeatureGroup.VOLATILITY,
        formula="normalized_atr(n)[t] = atr(n)[t] / close[t]",
        detail=(
            "ATR divided by the bar's close, so it is a fraction of price rather than a "
            "number of dollars, and stays comparable as the price level drifts."
        ),
        ticks=(1, 7, 14, 21),
        columns=("normalized_atr_{arg}",),
        warmup=lambda n: n - 1,
        needs=("high", "low", "close"),
        needs_price=False,
    ),
    # ----------------------------------------------------------------------- INDICATORS
    _bars(
        "sma",
        "Simple moving average",
        FeatureGroup.INDICATORS,
        formula="SMA(n)[t] = (1/n) · Σ_{i=0}^{n−1} P[t−i]",
        detail=(
            "The trailing arithmetic mean of the selected price over n bars. Every value "
            "in the average is at or before t."
        ),
        ticks=(5, 10, 20, 50, 100),
        columns=("sma_{arg}",),
        warmup=lambda n: n - 1,
    ),
    _bars(
        "ema",
        "Exponential moving average",
        FeatureGroup.INDICATORS,
        formula="EMA(n)[t] = α·P[t] + (1−α)·EMA(n)[t−1],  α = 2/(n+1)",
        detail=(
            "A recursive average giving more weight to recent bars. Seeded with the SMA "
            "of the first n bars and then carried forward one bar at a time, so each "
            "value depends only on the past."
        ),
        ticks=(5, 10, 20, 50, 100),
        columns=("ema_{arg}",),
        warmup=lambda n: n - 1,
        caution=(
            "Recursive: every later value depends on the seed, and the seed is the "
            "average of the first n bars. The first n−1 rows are left empty rather than "
            "started from an arbitrary first price, but even the first defined values "
            "still carry the seed's influence — discarding a further window before "
            "modelling is a reasonable precaution."
        ),
    ),
    _fixed(
        "sma_20_slope_5",
        "SMA(20) slope over 5 bars",
        FeatureGroup.INDICATORS,
        formula="OLS slope of SMA(20) against bar index over the trailing 5 bars",
        detail=(
            "The least-squares slope of the 20-bar simple moving average regressed on the "
            "bar index over the last five bars — the moving average's local direction, per "
            "bar. The windows (20 and 5) are fixed by the specification, which gives no "
            "parameter, and are named here so they are not invisible."
        ),
        columns=("sma_20_slope_5",),
        warmup=23,
    ),
    _fixed(
        "ema_20_slope_5",
        "EMA(20) slope over 5 bars",
        FeatureGroup.INDICATORS,
        formula="OLS slope of EMA(20) against bar index over the trailing 5 bars",
        detail=(
            "As SMA(20) slope, computed on the 20-bar exponential moving average. Both "
            "windows are fixed by the specification and named here."
        ),
        columns=("ema_20_slope_5",),
        warmup=23,
    ),
    _bars(
        "distance_to_sma",
        "Distance to SMA",
        FeatureGroup.INDICATORS,
        formula="distance_to_sma(n)[t] = (P[t] − SMA(n)[t]) / SMA(n)[t]",
        detail=(
            "How far the selected price sits from its own moving average, as a fraction "
            "of that average. Scale-free, so it is comparable across assets."
        ),
        ticks=(20, 50),
        columns=("distance_to_sma_{arg}",),
        warmup=lambda n: n - 1,
    ),
    _bars(
        "vwma",
        "Volume-weighted moving average",
        FeatureGroup.INDICATORS,
        formula="VWMA(n)[t] = Σ P·volume / Σ volume  over the trailing n bars",
        detail="The trailing average of price weighted by each bar's volume.",
        ticks=(10, 20, 50),
        columns=("vwma_{arg}",),
        warmup=lambda n: n - 1,
        needs=("volume",),
    ),
    _bars(
        "distance_to_vwma",
        "Distance to VWMA",
        FeatureGroup.INDICATORS,
        formula="distance_to_vwma(n)[t] = (P[t] − VWMA(n)[t]) / VWMA(n)[t]",
        detail="How far the selected price sits from its volume-weighted moving average.",
        ticks=(10, 20, 50),
        columns=("distance_to_vwma_{arg}",),
        warmup=lambda n: n - 1,
        needs=("volume",),
    ),
    _bars(
        "vwma_20_slope",
        "VWMA(20) slope",
        FeatureGroup.INDICATORS,
        formula="OLS slope of VWMA(20) against bar index over the trailing n bars",
        detail=(
            "The least-squares slope of the 20-bar volume-weighted moving average over the "
            "trailing window. The 20 is fixed by the specification and named in the column."
        ),
        ticks=(5, 10),
        columns=("vwma_20_slope_{arg}",),
        warmup=lambda n: n + 18,
        needs=("volume",),
    ),
    _bars(
        "evwma",
        "Elastic volume-weighted moving average",
        FeatureGroup.INDICATORS,
        formula=(
            "EVWMA(n)[t] = ((V[t−1] − V̄)·EVWMA[t−1] + V[t]·P[t]) / V̄, "
            "V̄ = Σ volume over the trailing n bars"
        ),
        detail=(
            "A recursive average whose memory is set by how much volume has traded: a "
            "quiet bar moves it little, a busy bar moves it far. The window n sets how "
            "much volume the estimator treats as its working float."
        ),
        ticks=(5, 10, 20, 50),
        columns=("evwma_{arg}",),
        warmup=lambda n: n - 1,
        needs=("volume",),
        caution=(
            "An adaptation. The indicator was defined for equities, where the float is a "
            "known number of shares outstanding; a crypto pair has no such quantity, so "
            "the trailing n-bar volume stands in for it. The number is a choice, and a "
            "different choice gives a different line."
        ),
    ),
    _fixed(
        "stoch_k_14_3",
        "Stochastic %K (14, 3)",
        FeatureGroup.INDICATORS,
        formula="%K[t] = 100 · (close[t] − LL14[t]) / (HH14[t] − LL14[t])",
        detail=(
            "Where the close sits within the highest–lowest range of the last 14 bars. "
            "The 14 and the 3-bar smoothing are the conventional defaults; the "
            "specification gives no parameter, so they are named in the column."
        ),
        columns=("stoch_k_14_3",),
        warmup=15,
        needs=("high", "low", "close"),
        needs_price=False,
        caution=(
            "Undefined when the 14-bar high equals the 14-bar low (a perfectly flat "
            "window); left empty rather than set to 50."
        ),
    ),
    _fixed(
        "stoch_d_14_3_3",
        "Stochastic %D (14, 3, 3)",
        FeatureGroup.INDICATORS,
        formula="%D[t] = SMA(3) of %K(14, 3)[t]",
        detail=(
            "The three-bar moving average of %K, the conventional signal line. Both "
            "smoothings are fixed by convention and named in the column."
        ),
        columns=("stoch_d_14_3_3",),
        warmup=17,
        needs=("high", "low", "close"),
        needs_price=False,
    ),
    _fixed(
        "directional_index",
        "Directional index (DI+, DI−)",
        FeatureGroup.INDICATORS,
        formula=(
            "+DM = max(high[t]−high[t−1], 0) if > low[t−1]−low[t] else 0; "
            "DI± = 100 · Wilder(±DM, 14) / ATR(14)"
        ),
        detail=(
            "The two Wilder directional indicators. They are produced together because "
            "neither is meaningful alone: DI+ above DI− is an uptrend, and the pair is "
            "also the input to ADX. Producing one column per checkbox would put the same "
            "information in the dataset twice under two names."
        ),
        columns=("di_plus_14", "di_minus_14"),
        warmup=14,
        needs=("high", "low", "close"),
        needs_price=False,
    ),
    _fixed(
        "aroon",
        "Aroon (up, down, oscillator)",
        FeatureGroup.INDICATORS,
        formula=(
            "AroonUp[t] = 100·(14 − bars since 14-bar high)/14; "
            "AroonDown[t] = 100·(14 − bars since 14-bar low)/14; Osc = Up − Down"
        ),
        detail=(
            "How recently the window's high and low were set. The specification lists "
            "this indicator twice — once as 'Aroon Up/Down' and once as 'AroonD_14' / "
            "'AroonU_14' — so it is emitted once, under one name, rather than as two "
            "identical pairs of columns."
        ),
        columns=("aroon_up_14", "aroon_down_14", "aroon_osc_14"),
        warmup=14,
        needs=("high", "low"),
        needs_price=False,
    ),
    _fixed(
        "williams_r_14",
        "Williams %R (14)",
        FeatureGroup.INDICATORS,
        formula="%R[t] = −100 · (HH14[t] − close[t]) / (HH14[t] − LL14[t])",
        detail=(
            "Where the close sits in the 14-bar range, on a scale from −100 (at the low) "
            "to 0 (at the high). The 14 is the conventional default; the specification "
            "gives no parameter, so it is named in the column."
        ),
        columns=("williams_r_14",),
        warmup=13,
        needs=("high", "low", "close"),
        needs_price=False,
        caution=(
            "Undefined when the 14-bar high equals the 14-bar low; left empty rather than "
            "set to −50."
        ),
    ),
    _bars(
        "rsi",
        "Relative strength index",
        FeatureGroup.INDICATORS,
        formula=(
            "RSI(n)[t] = 100 − 100 / (1 + Wilder(gain, n)/Wilder(loss, n))"
        ),
        detail=(
            "The ratio of Wilder-smoothed gains to Wilder-smoothed losses, mapped to "
            "0–100. A window of 1 makes the smoothing the bare last observation, which is "
            "why RSI(1) is only ever 0 or 100 — allowed, but named for what it is."
        ),
        ticks=(7, 14, 21),
        columns=("rsi_{arg}",),
        warmup=lambda n: n,
        default=14,
        caution=(
            "Recursive through Wilder's smoothing, so the first rows depend on where the "
            "series starts. Rows before the warm-up are left empty and should be dropped "
            "before modelling rather than filled."
        ),
    ),
    _fixed(
        "rsi_slope_5",
        "RSI(14) slope over 5 bars",
        FeatureGroup.INDICATORS,
        formula="OLS slope of RSI(14) against bar index over the trailing 5 bars",
        detail=(
            "Whether RSI is rising or falling, and how fast, per bar. The windows (14 and "
            "5) are fixed by the specification and named here."
        ),
        columns=("rsi_slope_5",),
        warmup=18,
    ),
    _fixed(
        "distance_to_rsi_50",
        "Distance to RSI 50",
        FeatureGroup.INDICATORS,
        formula="distance_to_rsi_50[t] = RSI(14)[t] − 50",
        detail=(
            "RSI measured from its neutral midpoint. This one is a difference rather "
            "than the ratio the other distance_to_* features use, because RSI is already "
            "a bounded index and dividing by 50 would make the column mean something "
            "different from the others under the same prefix."
        ),
        columns=("distance_to_rsi_50",),
        warmup=14,
    ),
    _fixed(
        "bollinger",
        "Bollinger bands",
        FeatureGroup.INDICATORS,
        formula=(
            "mid = SMA(20); upper = mid + 2·σ(20); lower = mid − 2·σ(20); "
            "width = (upper − lower)/mid"
        ),
        detail=(
            "The 20-bar moving average with bands two population standard deviations "
            "wide, plus the band width as a fraction of the middle band. The window and "
            "the multiplier are the conventional 20/2; the specification gives no "
            "parameter, so both are named in the columns."
        ),
        columns=("bb_upper_20_2", "bb_lower_20_2", "bb_width_20_2"),
        warmup=19,
    ),
    _fixed(
        "macd",
        "MACD",
        FeatureGroup.INDICATORS,
        formula=(
            "line = EMA(12) − EMA(26); signal = EMA(9) of line; histogram = line − signal"
        ),
        detail=(
            "The moving-average convergence–divergence indicator: the 12/26 difference, "
            "its 9-bar exponential average, and the gap between them. All three periods "
            "are the conventional defaults and are named in the columns."
        ),
        columns=("macd_line_12_26", "macd_signal_12_26_9", "macd_hist_12_26_9"),
        warmup=33,
    ),
    _bars(
        "adx",
        "Average directional index",
        FeatureGroup.INDICATORS,
        formula="DX = 100·|DI+ − DI−| / (DI+ + DI−);  ADX(n) = Wilder average of DX over n",
        detail=(
            "The strength of the trend regardless of its direction, built from the "
            "directional indicators above."
        ),
        ticks=(1, 7, 14, 21),
        columns=("adx_{arg}",),
        # Two smoothings, not one: DI± need n bars of Wilder smoothing before DX exists,
        # and ADX is a further n bars of smoothing on top of DX. So the first defined row
        # is 2n−1, and the declared warm-up has to be that. It is emphatically *not*
        # n + 13, which is 2n−1 only at n = 14 — the value the formula was first written
        # for. Declaring it that way would leave the catalogue claiming a 34-row warm-up
        # for ADX(21) when the engine produces a value at row 41, and would make
        # `usable_from` point at a row that is still empty.
        warmup=lambda n: 2 * n - 1,
        default=14,
        needs=("high", "low", "close"),
        needs_price=False,
        caution=(
            "ADX(1) is degenerate. Wilder's smoothing with a period of one is no "
            "smoothing at all, so ADX(1) equals the raw DX and swings between 0 and 100 "
            "bar to bar. It is offered because it is defined, and named adx_1 so the "
            "column does not claim to be a trend measure it is not."
        ),
    ),
    _bars(
        "vortex",
        "Vortex indicator",
        FeatureGroup.INDICATORS,
        formula=(
            "VI+ = Σ|high[t]−low[t−1]| / Σ TR;  VI− = Σ|low[t]−high[t−1]| / Σ TR, "
            "over the trailing n bars"
        ),
        detail=(
            "Two lines measuring whether upward or downward movement dominates over the "
            "window, both normalised by true range. The specification draws one checkbox "
            "and two lines; two lines is what the indicator is, so one checkbox emits "
            "both columns."
        ),
        ticks=(7, 14),
        columns=("vortex_plus_{arg}", "vortex_minus_{arg}"),
        warmup=lambda n: n,
        default=14,
        needs=("high", "low", "close"),
        needs_price=False,
    ),
    # ----------------------------------------------------------------------------- TIME
    _fixed(
        "hour_of_day",
        "Hour of day",
        FeatureGroup.TIME,
        formula="hour_of_day[t] = the bar's hour in the dataset's timezone, 0–23",
        detail=(
            "Which hour the bar opens in, taken from the dataset's declared timezone "
            "rather than this machine's. Useful for intraday seasonality."
        ),
        columns=("hour_of_day",),
        warmup=0,
        needs_price=False,
    ),
    _fixed(
        "day_of_week",
        "Day of week",
        FeatureGroup.TIME,
        formula="day_of_week[t] = the bar's weekday in the dataset's timezone, 0=Monday",
        detail="Monday is 0 and Sunday is 6. A calendar fact about the bar, known at t.",
        columns=("day_of_week",),
        warmup=0,
        needs_price=False,
    ),
    _fixed(
        "day_of_month",
        "Day of month",
        FeatureGroup.TIME,
        formula="day_of_month[t] = the bar's day of the month, 1–31",
        detail="A calendar fact about the bar, known at t.",
        columns=("day_of_month",),
        warmup=0,
        needs_price=False,
    ),
    _fixed(
        "month",
        "Month",
        FeatureGroup.TIME,
        formula="month[t] = the bar's month, 1–12",
        detail="A calendar fact about the bar, known at t.",
        columns=("month",),
        warmup=0,
        needs_price=False,
    ),
    _fixed(
        "quarter",
        "Quarter",
        FeatureGroup.TIME,
        formula="quarter[t] = the bar's calendar quarter, 1–4",
        detail="A calendar fact about the bar, known at t.",
        columns=("quarter",),
        warmup=0,
        needs_price=False,
    ),
    _fixed(
        "month_end",
        "Month end",
        FeatureGroup.TIME,
        formula="month_end[t] = 1 if the bar's calendar date is the last day of its month",
        detail=(
            "A flag marking bars that fall on the last calendar day of a month. This is "
            "the causal definition: whether a date is the last of its month follows from "
            "the calendar and is known at t. The tempting alternative — 'the last bar "
            "observed in the month' — cannot be known until the next bar arrives, and "
            "stamping the flag on the earlier bar would be a look-ahead."
        ),
        columns=("month_end",),
        warmup=0,
        needs_price=False,
        caution=(
            "On a dataset with gaps, the last *observed* bar of a month may not be the "
            "last *calendar* day. This flag follows the calendar, so a month whose final "
            "days are missing gets no flag at all. days_to_month_end is the more robust "
            "form."
        ),
    ),
    _fixed(
        "days_to_month_end",
        "Days to month end",
        FeatureGroup.TIME,
        formula="days_to_month_end[t] = (last calendar day of the month) − (bar's date)",
        detail=(
            "How many days remain in the month, counting the last day as zero. Unlike "
            "month_end it stays informative on a dataset with gaps, and it decreases "
            "monotonically within each month."
        ),
        columns=("days_to_month_end",),
        warmup=0,
        needs_price=False,
    ),
    _fixed(
        "days_to_quarter_end",
        "Days to quarter end",
        FeatureGroup.TIME,
        formula="days_to_quarter_end[t] = (last calendar day of the quarter) − (bar's date)",
        detail="As days_to_month_end, on the quarter boundary.",
        columns=("days_to_quarter_end",),
        warmup=0,
        needs_price=False,
    ),
    _fixed(
        "time_idx",
        "Time index",
        FeatureGroup.TIME,
        formula="time_idx[t] = rows since the first row of the dataset",
        detail=(
            "A monotone counter over the dataset's rows. It encodes absolute position in "
            "the sample rather than anything about the market."
        ),
        columns=("time_idx",),
        warmup=0,
        needs_price=False,
        caution=(
            "Strongly discouraged. Under a chronological split every test value lies "
            "outside the range seen in training, so a tree cannot split on it and a "
            "linear model extrapolates a trend that does not exist. It is offered "
            "because the specification asks for it, not because it is advisable."
        ),
    ),
    # ------------------------------------------------------------------------ EXOGENOUS
    _carried(
        "last_funding_rate",
        "Funding rate (carried)",
        FeatureGroup.EXOGENOUS,
        detail="",
        size_note=(
            "Funding is published every eight hours, so this column is flat within a "
            "funding interval and steps at the boundary — unless the merge that produced "
            "it was backward-aligned, which is the only alignment that cannot read ahead."
        ),
        asset_classes=CRYPTO_FUTURES,
    ),
    _carried(
        "sum_open_interest",
        "Open interest (carried)",
        FeatureGroup.EXOGENOUS,
        detail="",
        asset_classes=CRYPTO_FUTURES,
    ),
    _carried(
        "count_long_short_ratio",
        "Long/short account ratio (carried)",
        FeatureGroup.EXOGENOUS,
        detail="",
        asset_classes=CRYPTO_FUTURES,
    ),
    _carried(
        "sum_taker_long_short_vol_ratio",
        "Taker long/short volume ratio (carried)",
        FeatureGroup.EXOGENOUS,
        detail="",
        asset_classes=CRYPTO_FUTURES,
    ),
    _carried(
        "num_trades",
        "Number of trades (carried)",
        FeatureGroup.EXOGENOUS,
        detail="",
        size_note=(
            "The kline's trade count. The specification also lists `trade_count`; in this "
            "application's schema they are one field, so it appears once."
        ),
    ),
    _carried(
        "settlement",
        "Settlement price (carried)",
        FeatureGroup.EXOGENOUS,
        detail="",
        asset_classes=EQUITY_FUTURES,
        caution=(
            "Yahoo returns no settlement price, so on a fetched equity-futures dataset "
            "this column exists but is empty in every row. It is offered only when it "
            "actually carries values."
        ),
    ),
    _carried(
        "yield_curve_slope",
        "Yield curve slope (carried)",
        FeatureGroup.EXOGENOUS,
        detail="",
        size_note=(
            "This application fetches no macro data, so the column is offered only when "
            "an uploaded or merged dataset already carries one under this name."
        ),
    ),
)

#: By id, for O(1) lookup from the engine and the request validator.
CATALOGUE: dict[str, FeatureSpec] = {spec.id: spec for spec in _CATALOGUE}

#: The order the UI renders. Group order follows the specification; within a group the
#: order is the one the entries were declared in above.
GROUP_ORDER: tuple[FeatureGroup, ...] = (
    FeatureGroup.PRICE,
    FeatureGroup.VOLUME,
    FeatureGroup.RETURNS,
    FeatureGroup.VOLATILITY,
    FeatureGroup.INDICATORS,
    FeatureGroup.TIME,
    FeatureGroup.EXOGENOUS,
)


def catalogue() -> tuple[FeatureSpec, ...]:
    return _CATALOGUE


def get(feature_id: str) -> FeatureSpec | None:
    return CATALOGUE.get(feature_id)


# --------------------------------------------------------------------------------------
# Availability
# --------------------------------------------------------------------------------------


def _missing_columns(spec: FeatureSpec, context: DatasetContext) -> list[str]:
    return [name for name in spec.needs if name not in context.columns]


def availability(spec: FeatureSpec, context: DatasetContext) -> Availability:
    """Whether this feature can be computed on this dataset, and if not, why.

    The UI never re-derives this. One function decides, so a disabled checkbox and a
    refusal from the server can never disagree about the reason.
    """
    if spec.asset_classes is not None and context.asset_class not in spec.asset_classes:
        return Availability(
            False,
            f"{spec.label} applies to {_classes(spec.asset_classes)}; this dataset is "
            f"{context.asset_class.value}.",
        )

    missing = _missing_columns(spec, context)
    if missing:
        return Availability(
            False,
            f"This dataset has no {', '.join(missing)} column. Merge or upload a dataset "
            f"that carries one first.",
        )

    empty = [name for name in spec.needs if name in context.unpopulated]
    if empty:
        return Availability(
            False,
            f"The {', '.join(empty)} column exists in this dataset but is empty in every "
            f"row, so there is nothing to carry.",
        )

    if spec.id in NEEDS_TIME and not context.has_time:
        return Availability(
            False,
            "No time column could be identified in this dataset, so there are no "
            "timestamps to take a day of week from.",
        )

    if spec.id in NEEDS_TIMEFRAME and context.timeframe is None:
        return Availability(
            False,
            f"{spec.label} needs the dataset's bar length. Without a declared timeframe "
            f"there is no way to know how many bars a year holds, and assuming one would "
            f"scale every value by a constant nobody could see.",
        )

    if spec.id == "hour_of_day":
        reason = _needs_intraday(context)
        if reason:
            return Availability(False, reason)

    if spec.param is None:
        return Availability(True, None)

    windows = tuple(window_option(spec, context, token) for token in spec.param.ticks)
    if not any(window.available for window in windows):
        first = windows[0]
        return Availability(False, first.reason, windows)
    return Availability(True, None, windows)


def window_option(spec: FeatureSpec, context: DatasetContext, token: str) -> WindowOption:
    assert spec.param is not None  # noqa: S101 — guarded by every caller
    param = spec.param
    if param.kind is WindowKind.BARS:
        bars = int(token)
    else:
        seconds = parse_duration(token)
        if seconds is None:
            return WindowOption(token, None, False, f"'{token}' is not a duration.")
        if context.timeframe is None:
            return WindowOption(
                token,
                None,
                False,
                "This dataset declares no timeframe, so an interval cannot be turned "
                "into a number of bars. Use a bar count instead.",
            )
        converted = duration_bars(seconds, context.timeframe)
        if converted is None:
            bar_seconds = TIMEFRAME_SECONDS[context.timeframe]
            return WindowOption(
                token,
                None,
                False,
                f"{token} is not a whole number of {context.timeframe.value} bars "
                f"({bar_seconds}s). Use a bar count instead.",
            )
        bars = converted
    if bars < param.minimum:
        return WindowOption(
            token,
            bars,
            False,
            f"{spec.label} needs at least {param.minimum} of this dataset's bars; "
            f"{token} is {bars}.",
        )
    return WindowOption(token, bars, True, None)


def _classes(classes: frozenset[AssetClass]) -> str:
    return " and ".join(sorted(c.value for c in classes))


def resolve_custom(spec: FeatureSpec, context: DatasetContext, raw: str) -> WindowOption:
    """Resolve a value typed into the specification's blank box.

    The blank box is one control for both kinds: on a bar-count feature the user types a
    number of candles, on a duration feature they type an interval. Which one is expected
    follows from the feature, and a value of the wrong shape is refused with the shape
    that was wanted.
    """
    assert spec.param is not None  # noqa: S101 — guarded by every caller
    text = raw.strip().lower()
    if not text:
        return WindowOption(raw, None, False, "Empty window.")
    if spec.param.kind is WindowKind.BARS:
        if not text.isdigit():
            return WindowOption(
                raw,
                None,
                False,
                f"{spec.label} takes a count of candles, so '{raw}' should be a whole "
                f"number.",
            )
        value = int(text)
        if value > MAX_CUSTOM_BARS:
            return WindowOption(
                raw, None, False, f"A window wider than {MAX_CUSTOM_BARS:,} bars is refused."
            )
        if value < spec.param.minimum:
            return WindowOption(
                raw,
                value,
                False,
                f"{spec.label} needs at least {spec.param.minimum} bars; {value} is fewer.",
            )
        return WindowOption(str(value), value, True, None)
    return window_option(spec, context, text)


def spec_payload(spec: FeatureSpec, avail: Availability) -> dict[str, Any]:
    """The wire form of one catalogue entry. The UI renders this and nothing else.

    The warm-up is reported *per window* rather than per feature, because for a
    parameterised feature it is a function of the window and a single number would be
    wrong for every tick but one. A parameterless feature gets a plain number.
    """
    return {
        "id": spec.id,
        "label": spec.label,
        "group": spec.group.value,
        "kind": spec.kind.value,
        "formula": spec.formula,
        "detail": spec.detail,
        "columns": list(spec.columns),
        "needs": list(spec.needs),
        "needs_price": spec.needs_price,
        "caution": spec.caution,
        "warmup_bars": spec.warmup(0) if spec.param is None else None,
        "available": avail.available,
        "reason": avail.reason,
        "param": None
        if spec.param is None
        else {
            "kind": spec.param.kind.value,
            "ticks": list(spec.param.ticks),
            "default": spec.param.default,
            "minimum": spec.param.minimum,
            "custom": spec.param.custom,
        },
        "windows": [
            {
                "token": window.token,
                "bars": window.bars,
                "warmup_bars": None if window.bars is None else spec.warmup(window.bars),
                "available": window.available,
                "reason": window.reason,
            }
            for window in avail.windows
        ],
    }
