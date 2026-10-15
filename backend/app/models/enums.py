"""Closed vocabularies shared across the app.

These are enums rather than free strings because every one of them is used as a
branch condition somewhere; a typo should fail at import, not return an empty frame.
"""

from __future__ import annotations

from enum import StrEnum


class AssetClass(StrEnum):
    CRYPTO_SPOT = "crypto_spot"
    CRYPTO_FUTURES = "crypto_futures"
    EQUITY_SPOT = "equity_spot"
    EQUITY_FUTURES = "equity_futures"


class Provider(StrEnum):
    BINANCE = "binance"
    YAHOO = "yahoo"
    UPLOAD = "upload"
    DERIVED = "derived"


class DatasetCategory(StrEnum):
    """The four buckets the Data Explorer groups datasets into.

    `UPLOADED` is deliberately distinct from `RAW`: both are immutable, but a raw
    dataset has verifiable provider provenance and an uploaded one does not.
    """

    RAW = "raw_data"
    UPLOADED = "uploaded_data"
    PROCESSED = "processed_data"
    OUTPUTS = "outputs"


class Timeframe(StrEnum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


class DuplicatePolicy(StrEnum):
    """What to do about repeated rows.

    `DROP_EXACT` and the two timestamp policies repair different defects. Two rows can
    share a timestamp and differ in every other column — a provider correction appended
    rather than substituted — and dropping exact duplicates leaves that untouched, so
    the two are offered separately rather than folded into one "dedupe" switch.
    """

    KEEP = "keep"
    DROP_EXACT = "drop_exact"
    KEEP_FIRST_BY_TIME = "keep_first_by_time"
    KEEP_LAST_BY_TIME = "keep_last_by_time"


class MissingValuePolicy(StrEnum):
    """How to treat empty cells.

    The spec lists forward and backward fill on one line. They are separate members here
    because they differ in exactly the way that matters: a forward fill carries the last
    *past* observation into a gap, while a backward fill carries the next *future* one
    backwards. On a price series that is the difference between a defensible assumption
    and a look-ahead, so it must not be a single checkbox.
    """

    KEEP = "keep"
    FFILL = "ffill"
    BFILL = "bfill"
    INTERPOLATE_TIME = "interpolate_time"
    ZERO = "zero"
    DROP_ROWS = "drop_rows"


class OutlierPolicy(StrEnum):
    """How to treat values outside a fence.

    `WINSORIZE` and `REMOVE` share the quantile fence; `IQR` uses the Tukey fence
    (Q1 − k·IQR, Q3 + k·IQR) and clamps, because "Removing Extremes" is already offered
    as its own option. The fence actually applied is reported for every column.
    """

    KEEP = "keep"
    WINSORIZE = "winsorize"
    REMOVE = "remove"
    IQR = "iqr"


class AlignMethod(StrEnum):
    """How a joined dataset is placed on the base dataset's timeline.

    Only two are offered, and the omission is the point. `backward` takes the most
    recent value *at or before* each bar, which is the only direction that cannot read
    the future. A `forward` fill would stamp a bar with a funding rate that had not been
    published yet, and `nearest` can do the same whenever the next observation is
    closer than the last — a look-ahead that is invisible in the merged frame and would
    show up only as an unexplained edge in a backtest.
    """

    BACKWARD = "backward"
    EXACT = "exact"


class MergeHow(StrEnum):
    """Which rows survive a join."""

    LEFT = "left"
    INNER = "inner"


class FeatureGroup(StrEnum):
    """The sections of the feature catalogue, in the specification's order.

    ``EXOGENOUS`` is structurally different from the rest and is kept last for that
    reason: those entries are not computed at all. They are columns another dataset
    already carries — a funding rate, an open interest — and selecting one keeps it,
    which is exactly what the specification says to do with them.
    """

    PRICE = "price"
    VOLUME = "volume"
    RETURNS = "returns"
    VOLATILITY = "volatility"
    INDICATORS = "indicators"
    TIME = "time"
    EXOGENOUS = "exogenous"


class PriceSource(StrEnum):
    """Which price series every price-derived feature is computed from.

    The specification offers exactly two, and the choice propagates into every other
    feature, so it is a field of the request rather than a per-feature option.
    """

    CLOSE = "close"
    WEIGHTED = "weighted"


class WindowKind(StrEnum):
    """How a feature's single varying parameter is expressed.

    The specification makes this distinction itself: some features take *intervals*
    ("15m, 1h, 4h, 1d, 7d, 21d") and some take *counts of candles* ("1, 2, 3, 5, 10"),
    and it notes that the candle-count kind is available for every dataset. A
    duration window is only meaningful when the dataset declares a timeframe to
    divide it by, which is why the two cannot be one control.
    """

    BARS = "bars"
    DURATION = "duration"


class FeatureKind(StrEnum):
    """What selecting an entry actually does."""

    #: Computed from the dataset's own columns.
    COMPUTED = "computed"
    #: Kept as-is. Nothing is calculated, so no methodology can be invented for it.
    CARRIED = "carried"


class SplitType(StrEnum):
    """How a dataset is cut into spans a model may and may not see.

    Six members rather than three with options, because each one *is* a different
    procedure. The two percentage members carry their ratio in the name: a plan made as
    ``percentage_70_15_15`` and one made as ``percentage_60_20_20`` produce different row
    counts, and a stored plan that said only "percentage" would leave the reader to
    reconstruct the ratio from its rows.

    ``TSCV_EXPANDING`` and ``TSCV_ROLLING`` differ in what each fold trains on — everything
    before the cut, or only the block immediately before it. For a non-stationary series
    that is the difference between a training set that reaches back years and one that
    describes the regime the test block actually sits in, so they cannot be one member
    with a flag.
    """

    DATE_3WAY = "date_3way"
    DATE_2WAY = "date_2way"
    PERCENTAGE_70_15_15 = "percentage_70_15_15"
    PERCENTAGE_60_20_20 = "percentage_60_20_20"
    TSCV_EXPANDING = "tscv_expanding"
    TSCV_ROLLING = "tscv_rolling"

    @property
    def is_tscv(self) -> bool:
        """Whether this split produces one train/test pair per fold.

        Read by the scaling module, which must fit per fold: statistics pooled across
        folds would be computed from rows that lie *after* an early fold's test span,
        which is look-ahead relative to that fold.
        """
        return self in (SplitType.TSCV_EXPANDING, SplitType.TSCV_ROLLING)


class SplitRole(StrEnum):
    """What a span in a split plan is for.

    ``VAL`` is absent from every TSCV plan by construction rather than by omission: a
    walk-forward scheme has no separate validation span, because each fold's test block
    already plays that part for the fold before it.
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class ScalerType(StrEnum):
    """How feature columns are rescaled.

    ``ROLLING_ZSCORE`` is not a member of the same family as the other three and is not
    labelled as one. Min-Max, Standard and Robust each estimate a *fixed* pair of numbers
    over a span of history and apply it to every row, so they need a split plan to say
    which history may be looked at. The rolling z-score estimates a fresh mean and
    standard deviation from the trailing window at each row, so it needs no fit, no plan,
    and cannot be given one — there is nothing to leak.
    """

    MINMAX = "minmax"
    STANDARD = "standard"
    ROBUST = "robust"
    ROLLING_ZSCORE = "rolling_zscore"

    @property
    def needs_split_plan(self) -> bool:
        """Whether this scaler must be fitted on a training span.

        The single place the rule is written. The service refuses on it, the API echoes
        it to the page, and the page greys the split-plan control with it — one source,
        so a control the UI presents as optional and a refusal from the backend cannot
        disagree.
        """
        return self is not ScalerType.ROLLING_ZSCORE


#: Bar length in seconds. Used for gap detection and span arithmetic, never for
#: string manipulation of timestamps.
TIMEFRAME_SECONDS: dict[Timeframe, int] = {
    Timeframe.M1: 60,
    Timeframe.M5: 300,
    Timeframe.M15: 900,
    Timeframe.M30: 1800,
    Timeframe.H1: 3600,
    Timeframe.H4: 14400,
    Timeframe.D1: 86400,
    Timeframe.W1: 604800,
}

#: Timeframes each provider actually serves. The spec's UI offers a superset, so
#: the fetch form filters against this rather than letting a doomed request through.
BINANCE_KLINES_TIMEFRAMES: frozenset[Timeframe] = frozenset(Timeframe)
CRYPTO_METRICS_TIMEFRAMES: frozenset[Timeframe] = frozenset(
    {Timeframe.M5, Timeframe.M15, Timeframe.M30, Timeframe.H1, Timeframe.H4, Timeframe.D1}
)
EQUITY_TIMEFRAMES: frozenset[Timeframe] = frozenset(
    {Timeframe.H1, Timeframe.H4, Timeframe.D1}
)


class RegimeModel(StrEnum):
    """How Module D infers market state.

    ``HYBRID`` and ``RULES`` are separate members rather than one "advanced" option
    because they answer different questions. ``HYBRID`` keeps the statistical model and
    adds a classifier trained to reproduce it from faster indicators; ``RULES`` discards
    the statistical model entirely and fires on a volatility breach alone. The second is
    a risk control, not an estimator, and conflating them would hide that.
    """

    HMM = "hmm"
    GMM = "gmm"
    HYBRID = "hybrid"
    RULES = "rules"


class RegimeFeature(StrEnum):
    """The observation vector a regime model is fitted on.

    ``ADX`` and ``RSI`` are deliberately *not* computed by this module. They are
    methodology-laden indicators the feature catalogue already defines, names, and
    documents; a second implementation here would be a second formula for one concept,
    which is the drift the catalogue exists to prevent. They are carried from a C5
    feature dataset and the request is refused if they are absent.
    """

    LOG_RETURN = "log_return"
    REALIZED_VOLATILITY = "realized_volatility"
    VOLUME_CHANGE = "volume_change"
    ADX = "adx"
    RSI = "rsi"


class RegimeState(StrEnum):
    """The named market states, plus the explicit gray zone.

    ``TRANSITION`` is a state in its own right rather than an absence of one. Forcing a
    binary decision when no state holds a clear majority is what produces the thrashing
    that the minimum-duration rule exists to suppress; naming the ambiguity is the
    honest alternative to guessing.
    """

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    CRISIS = "crisis"
    TRANSITION = "transition"


#: The four states that carry a probability column, in the order the specification lists
#: them. ``TRANSITION`` is absent on purpose: it is derived from the *shape* of these four
#: (no clear winner), not a fifth competing hypothesis, so giving it its own column would
#: double-count the same probability mass.
REGIME_PROBABILITY_COLUMNS: tuple[RegimeState, ...] = (
    RegimeState.TRENDING_UP,
    RegimeState.TRENDING_DOWN,
    RegimeState.RANGING,
    RegimeState.CRISIS,
)


class JobStatus(StrEnum):
    """Lifecycle of a background job.

    ``FAILED`` carries the refusal rather than raising it, because by the time the work
    runs there is no request left to attach a 422 to — the caller is polling, and the
    message has to reach them down the same channel as the result.
    """

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class VolModel(StrEnum):
    """The estimators Module E can fit.

    Two families, and the split is not cosmetic. ``VOL_MODEL_FAMILY`` below records it,
    and the UI groups by it, because the difference a user has to be told about is
    *refit cost*: every member refits at every walk-forward step (that is what makes the
    causality test passable), and the two hybrids train a second model inside that loop.

    Named ``VolModel`` rather than ``VolatilityModel`` so it does not collide with the
    runtime protocol of the same concept in ``app/services/volatility_models.py`` — the
    enum is what a request *asks for*, the protocol is what runs.
    """

    GARCH = "garch"
    EGARCH = "egarch"
    GJR_GARCH = "gjr_garch"
    ARMA_EGARCH_XGBOOST = "arma_egarch_xgboost"
    TDNN_GJR_GARCH = "tdnn_gjr_garch"

    @property
    def family(self) -> str:
        """``"simple"`` or ``"hybrid"``. Defined once, in ``VOL_MODEL_FAMILY``."""
        return VOL_MODEL_FAMILY[self]

    @property
    def needs(self) -> tuple[str, ...]:
        """Which optional parameters this model actually consumes.

        The single place the rule is written, so the router, the options endpoint and the
        page cannot disagree about whether a control is meaningful or noise.
        """
        return VOL_MODEL_NEEDS[self]


#: Which family each model belongs to. ``"simple"`` is a single arch estimator;
#: ``"hybrid"`` combines an arch variance with a machine-learned correction and therefore
#: costs an order of magnitude more per walk-forward step.
VOL_MODEL_FAMILY: dict[VolModel, str] = {
    VolModel.GARCH: "simple",
    VolModel.EGARCH: "simple",
    VolModel.GJR_GARCH: "simple",
    VolModel.ARMA_EGARCH_XGBOOST: "hybrid",
    VolModel.TDNN_GJR_GARCH: "hybrid",
}

#: The parameters each model reads. ``p``/``o``/``q`` are the arch lag orders; the hybrids
#: additionally read ``lags`` (the tapped delay line of past observations). A parameter a
#: model does not list here is ignored, and the options endpoint reports it as such rather
#: than letting a control sit on the page that changes nothing.
VOL_MODEL_NEEDS: dict[VolModel, tuple[str, ...]] = {
    VolModel.GARCH: ("p", "q"),
    VolModel.EGARCH: ("p", "o", "q"),
    VolModel.GJR_GARCH: ("p", "o", "q"),
    VolModel.ARMA_EGARCH_XGBOOST: ("p", "o", "q", "lags"),
    VolModel.TDNN_GJR_GARCH: ("p", "o", "q", "lags"),
}


class VolDistribution(StrEnum):
    """The conditional distribution of the standardised innovation ``z``.

    These are the specification's names. They are deliberately *not* arch's own strings
    (``t``, ``skewt``, ``ged``): the mapping lives in one table in
    ``app/services/volatility_models.py``, so a rename on either side is a one-line change
    in a place that is obviously the translation rather than a scattering of literals.
    """

    NORMAL = "normal"
    STUDENTS_T = "students_t"
    SKEW_STUDENT = "skew_student"
    GED = "ged"


class VolMeanModel(StrEnum):
    """How the conditional mean of the return is modelled.

    ``CONSTANT`` estimates a single intercept. ``ARMA`` lets arch estimate an
    autoregressive mean **jointly** with the variance equation, so the two are not fitted
    in two stages. It is an AR(p) mean rather than a full ARMA one: arch has no ARMA mean
    for a univariate series alongside a variance equation, and adding a separate MA part on
    top of an EGARCH(q) is not identifiable — the q lagged variance terms already carry the
    residual autocorrelation. The report labels it ``ar{p}`` for that reason.

    The more elaborate option is not "better": a mis-specified AR order inflates the
    apparent persistence of the variance, which is why the choice is explicit rather than
    defaulted to it.
    """

    CONSTANT = "constant"
    ARMA = "arma"


class VolMetric(StrEnum):
    """The scoring functions the report carries.

    ``MZ_R2`` is the Mincer–Zarnowitz regression's coefficient of determination, and it is
    a *diagnostic*, not a ranking: against a squared-return proxy its value is near zero
    even for a correctly specified model, because the proxy is enormously noisy. It is
    reported, and the report says this in words, so the number is not misread as failure.
    ``DM`` is Diebold–Mariano, which is a pairwise comparison and therefore has no
    per-model scalar — it appears in its own matrix rather than in the metrics table.
    """

    MQLIKE = "mqlike"
    RMSE = "rmse"
    MAE = "mae"
    MZ_R2 = "mz_r2"
    DM = "dm"


#: The metrics that produce one number per model, in report order. ``DM`` is excluded: it
#: compares two models and has no meaning alone, so a table with a DM column per row would
#: be inviting a reader to compare incomparable things.
VOL_SCALAR_METRICS: tuple[VolMetric, ...] = (
    VolMetric.MQLIKE,
    VolMetric.RMSE,
    VolMetric.MAE,
    VolMetric.MZ_R2,
)

#: Which direction is better, for the metrics table's "best" marking. Lower is better for
#: every loss; higher is better for the R² diagnostic. Written down rather than assumed,
#: because marking the highest MQLIKE as best is a plausible-looking way to invert the
#: entire conclusion of the report.
VOL_METRIC_LOWER_IS_BETTER: dict[VolMetric, bool] = {
    VolMetric.MQLIKE: True,
    VolMetric.RMSE: True,
    VolMetric.MAE: True,
    VolMetric.MZ_R2: False,
}

#: Whether a metric is a *ranking* at all — as opposed to a number to read.
#:
#: The direction table above is not enough on its own. ``VOL_METRIC_LOWER_IS_BETTER`` says
#: ``False`` for MZ_R2, and a table that read that as "higher is better" would crown the
#: largest R² in the column — which is the opposite of what the report says about it. The
#: R² is a calibration diagnostic against a proxy so noisy that near-zero is the expected
#: value for a correctly specified model, so there is no best row to mark. Saying which
#: metrics are ordered, in the same place the ordering itself is stated, is what stops the
#: page from having to decide it a second time.
VOL_METRIC_RANKING: dict[VolMetric, bool] = {
    VolMetric.MQLIKE: True,
    VolMetric.RMSE: True,
    VolMetric.MAE: True,
    VolMetric.MZ_R2: False,
}
