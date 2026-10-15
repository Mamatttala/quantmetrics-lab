"""Module E — volatility forecasting over HTTP.

Three things in this module are shaped by the fact that a forecast, unlike everything the
modules before it produced, **cannot be checked by re-reading its input**.

The first is the walk-forward contract. ``WalkForwardSpec`` is a request for a protocol,
not for a model: it says how many bars a fit may see, which bars are scored, and how far
the window rolls between refits. The arithmetic it implies — ``train_window + test_window
<= rows`` — cannot be checked until the file has been read, because it needs the series
length; it is checked the moment the series exists and before the first fit, so an
impossible schedule is refused naming both numbers rather than discovered forty seconds
into a walk-forward. On the HTTP path that refusal is the job's own error
(``validation_failure``) — a 202 has already been returned by then — while the inline
``run`` path used by the drivers raises it as a 422.

The second is that there are **two different forecasts** in one run, and the schema keeps
them apart. The *scored* forecast is one-step-ahead at each walk-forward position: it is
what MQLIKE, the Mincer–Zarnowitz regression and Diebold–Mariano are defined over, and
every one of them would be meaningless against a multi-step forecast. The *forward path* is
``forecast_horizon`` steps beyond the last observation, which is what the fan chart draws
and what a user actually wants to see. They are not two kinds of row in one frame — the
scored run goes to the output dataset and the forward path travels only in the report — so
there is no column labelling one against the other and no way to filter the wrong one in.

The third is that the time column is ``open_time``: the instant the bar was *observed*, and
therefore the earliest instant at which a forecast for it could have been made. The
specification for this module asked for the frame to be keyed on ``close_time``, which is
one bar later — the same inversion that §§14 and 86 exist to prevent, and the reason
``app/services/regime_engine.py`` was repaired before this module was written. A forecast
stamped with the bar's close is a forecast that knows the answer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, computed_field, model_validator

from app.models.dataset import DatasetMeta
from app.models.enums import (
    JobStatus,
    VolDistribution,
    VolMeanModel,
    VolMetric,
    VolModel,
)

#: The smallest in-sample window an arch estimator is given. Below roughly this many
#: observations a GARCH(1,1) is fitting three parameters to noise: the optimiser converges
#: on a boundary and the forecast is a constant, which looks like a working model in a
#: chart. Refused rather than warned about, because the failure is silent.
MIN_TRAIN_WINDOW = 50

#: The fewest finite scored steps a metric may be computed from. Below this the number is
#: not an estimate of anything, and reporting it beside a real one invites a comparison
#: between them.
MIN_SCORED_STEPS = 30


class WalkForwardSpec(BaseModel):
    """The evaluation protocol: how much a fit may see, and which bars are scored.

    Defined so that every scored step is a genuine out-of-sample **one-step-ahead**
    forecast: the model is fitted on ``train_window`` bars ending at ``t``, and asked for
    ``t``. Nothing else is a walk-forward, and the causality test in
    ``tests/test_volatility_models.py`` is what holds this to it.

    The evaluation region is the **last** ``test_window`` bars of the series rather than
    the whole series after the first fit. That is a deliberate choice with a visible
    consequence: it makes the scored sample size exactly ``ceil(test_window / step_size)``,
    which is the number a Diebold–Mariano test's power depends on, so it is a decision the
    user makes once rather than a quantity that silently follows from the series length.
    """

    #: Bars a single fit is allowed to see, immediately preceding the bar it forecasts.
    train_window: int = Field(default=500, ge=MIN_TRAIN_WINDOW)

    #: The length of the out-of-sample region, in bars, at the end of the series.
    test_window: int = Field(default=250, ge=MIN_SCORED_STEPS)

    #: Bars the window rolls between refits. Every step refits — this controls how *often*,
    #: which is a cost knob, not a leakage knob. It cannot change the causality property
    #: (each fit still sees only its own past) and the report states the cost it implies.
    step_size: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _check(self) -> WalkForwardSpec:
        if self.step_size > self.test_window:
            raise ValueError(
                f"step_size ({self.step_size}) exceeds test_window ({self.test_window}), "
                "so the run would skip scored bars entirely while still paying for a refit "
                "at each position it does visit. Use a step no larger than the region it "
                "walks."
            )
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def n_scored(self) -> int:
        """How many one-step forecasts this protocol produces. One reading of the
        arithmetic, so the report, the page's live hint and the engine cannot disagree.

        A computed field rather than a plain property because it has to survive
        serialisation: the report embeds this spec, the page shows the count beside the
        schedule, and the lineage row records it. A property exists only in Python, so the
        figure the page needs would arrive as ``undefined`` — the failure mode this project
        keeps meeting, and the reason the API tests assert every field by name.
        """
        return -(-self.test_window // self.step_size)

    def schedule(self, n_rows: int) -> list[int]:
        """The bar offsets this protocol scores, given a series of ``n_rows`` bars.

        Raises ``ValueError`` with the computed numbers when the schedule does not fit. The
        engine calls this twice — once through ``_validate_schedule``, before the first fit,
        and once to enumerate the scored offsets — so "the window is impossible" is one
        sentence in one place rather than two that drift. It is deliberately not reachable
        from ``VolForecastRequest``'s validator: the check needs the series length, which no
        request carries, so ``train_window`` and ``test_window`` on their own cannot be
        known to be too large for the data.
        """
        first = n_rows - self.test_window
        if first < self.train_window:
            raise ValueError(
                f"A {self.train_window}-bar training window and a {self.test_window}-bar "
                f"test window need at least {self.train_window + self.test_window} bars; "
                f"this series has {n_rows}. Shorten either window, or use a longer dataset."
            )
        return list(range(first, n_rows, self.step_size))


class VolForecastRequest(BaseModel):
    """Parameters for one forecasting run.

    Every field is a modelling decision that lands in the output's lineage. A volatility
    forecast is uninterpretable next to a different distribution, a different lag order or
    a different train window, and storing the frame without them would make the artifact
    unreproducible — the one thing a research artifact must never be.
    """

    dataset_id: str

    #: Where the return series comes from. ``"close"`` means "derive log returns from the
    #: close column"; ``"log_return"`` means "use the column of that name if it is one";
    #: anything else names a column to use directly. Raw ``close`` is never handed to a
    #: model — GARCH is a model of returns, and fitting it to a price level produces a
    #: near-unit-root variance equation that fits beautifully and means nothing.
    price_source: str = "close"

    models: list[VolModel] = Field(default_factory=lambda: [VolModel.GARCH])
    distribution: VolDistribution = VolDistribution.STUDENTS_T
    mean_model: VolMeanModel = VolMeanModel.CONSTANT

    #: arch lag orders. ``p`` is the GARCH/AR lag, ``q`` the ARCH/MA lag, ``o`` the
    #: asymmetry order (EGARCH and GJR only — EGARCH needs at least 1, GJR is defined by
    #: exactly 1).
    p: int = Field(default=1, ge=1, le=5)
    o: int = Field(default=1, ge=0, le=5)
    q: int = Field(default=1, ge=1, le=5)

    #: Tapped delay line for the hybrid legs: how many past observations the XGBoost and
    #: TDNN features reach back over. Ignored by the simple family, and the options
    #: endpoint says so rather than leaving a control that changes nothing.
    lags: int = Field(default=5, ge=1, le=100)

    walk_forward: WalkForwardSpec = Field(default_factory=WalkForwardSpec)

    #: Steps of the *forward path* beyond the last observation — the fan chart. This is not
    #: the scored forecast and is reported separately; see the module docstring.
    forecast_horizon: int = Field(default=24, ge=1, le=500)

    #: Optional second lineage parent. A Module D regime dataset supplies the posterior
    #: columns the hybrid legs may use as *lagged* features. Joined on ``open_time``.
    conditioning_dataset_id: str | None = None

    write: bool = True
    name: str | None = None

    @model_validator(mode="after")
    def _check(self) -> VolForecastRequest:
        if not self.models:
            raise ValueError("Select at least one model to fit.")
        if len(set(self.models)) != len(self.models):
            raise ValueError("A model is selected more than once.")

        # The same rule the options endpoint reports from: a parameter a model does not
        # consume is not silently ignored. An asymmetry order of 0 asked of an EGARCH is a
        # user believing they configured something; arch would accept it and fit a
        # symmetric model under an asymmetric name.
        for model in self.models:
            needs = model.needs
            if "o" in needs and self.o < 1:
                raise ValueError(
                    f"{model.value} models an asymmetry, so it needs o >= 1; got o={self.o}. "
                    "Use garch for a symmetric model."
                )
        if self.price_source.strip() == "":
            raise ValueError("price_source must name a column or 'close'.")
        return self

    @property
    def needs_lags(self) -> bool:
        """Whether any selected model consumes ``lags``. Reported so a control that does
        nothing is visibly doing nothing."""
        return any("lags" in model.needs for model in self.models)


class VolForecastAccepted(BaseModel):
    """What ``POST /api/volatility/forecast`` returns immediately.

    Same contract as the other two async modules: the work has not happened, and ``poll``
    is the only way to learn whether it succeeded.
    """

    job_id: str
    status: JobStatus
    poll: str


class VolRow(BaseModel):
    """One scored forecast, as a row of the output dataset.

    The stored frame is a DataFrame rather than a list of these — this type exists so the
    schema is written down once, in the same place a reader looks for the model of the
    output. The column order is ``VOL_FORECAST_COLUMNS``.
    """

    open_time: Any
    model: str
    forecast_variance: float
    forecast_volatility: float
    actual_variance: float | None
    actual_volatility: float | None
    error_variance: float | None
    ratio: float | None
    finite: bool


#: The output dataset's columns, in order. Asserted as an ordered tuple by the API and
#: driver tests: a consumer reading these by position is a thing that will eventually
#: happen, and the specification asks for an exact schema.
#:
#: **The dataset holds the scored walk-forward span only** — real bars, each with the
#: forecast that was made for it before it happened. The forward path is *not* written
#: here. A dataset row in this application is an observation, and a projection of the next
#: ``h`` bars is not one; storing future timestamps in the store would put rows into it
#: that no cleaning step, no split plan and no backtest is entitled to treat as data. The
#: forward path travels in the report instead, where the fan chart reads it and where the
#: ``is_forward_path`` distinction cannot be lost by a consumer that filters nothing.
#:
#: There is likewise no ``close_time`` column: carrying one "for readability" would
#: reintroduce the name that makes a mistaken join look correct, and every consumer that
#: needs to join a forecast to anything joins it on the instant the bar was observed.
VOL_FORECAST_COLUMNS: tuple[str, ...] = (
    "open_time",
    "model",
    "forecast_variance",
    "forecast_volatility",
    "actual_variance",
    "actual_volatility",
    "error_variance",
    "ratio",
    "finite",
)


class VolModelScore(BaseModel):
    """One model's metrics over the scored walk-forward span.

    ``n_nan`` and ``n_inf`` are counted rather than filled. A model that produced 400
    non-finite forecasts and had them forward-filled is indistinguishable in a chart from
    one that produced none, and the two mean entirely different things about whether the
    optimiser converged.
    """

    model: VolModel
    family: str

    #: Steps the protocol produced, and how many of them were finite and therefore scored.
    n_steps: int
    n_finite: int
    n_nan: int
    n_inf: int

    #: ``None`` when fewer than ``MIN_SCORED_STEPS`` steps were finite. The reason is in
    #: ``unscored_reason`` rather than left to a null the reader has to interpret.
    mqlike: float | None = None
    rmse: float | None = None
    mae: float | None = None

    #: The Mincer–Zarnowitz regression ``actual = a + b·forecast``.
    mz_intercept: float | None = None
    mz_slope: float | None = None
    mz_r2: float | None = None

    unscored_reason: str | None = None

    #: Seconds the walk-forward took, for the run report's cost comparison between the
    #: two families. Wall-clock, so it is a hint rather than a benchmark.
    seconds: float = 0.0


class VolComparison(BaseModel):
    """One pairwise Diebold–Mariano test.

    ``significant`` is the 5% verdict written as a boolean beside the p-value, and ``better``
    names the model with the lower mean loss — so the page can say "no significant
    difference" in words rather than leaving a bare statistic for the reader to over-read.
    """

    model_a: VolModel
    model_b: VolModel
    n_effective: int
    statistic: float
    p_value: float
    better: VolModel | None
    significant: bool


class VolFanBand(BaseModel):
    """One fan-chart band, as an array aligned with its path's ``times``.

    Per-step rather than a single interval around the whole path. A single interval would
    be a statement about the path's average width and not about any bar in it, and it would
    hide the shape a volatility forecast is actually about — the term structure flattening
    as the horizon grows.

    Derived from the empirical quantiles of the model's own walk-forward error ratio, so the
    interval says how wrong this model has actually been rather than how wrong a correctly
    specified model would be.
    """

    label: str
    low: list[float] = Field(default_factory=list)
    high: list[float] = Field(default_factory=list)


class VolForwardPath(BaseModel):
    """The projection beyond the last observation — the fan chart, and not the scored run.

    Separate from the dataset for the reason given on ``VOL_FORECAST_COLUMNS``, and
    separate from the scores for a second reason: MQLIKE, Mincer–Zarnowitz and
    Diebold–Mariano are all defined against a realised value, and a projected bar has none.
    A single object carrying both would invite scoring one against the other.
    """

    model: VolModel
    times: list[Any] = Field(default_factory=list)
    variance: list[float] = Field(default_factory=list)
    volatility: list[float] = Field(default_factory=list)
    bands: list[VolFanBand] = Field(default_factory=list)


class VolForecastReport(BaseModel):
    """What the run did, and against exactly what."""

    dataset_id: str
    dataset_name: str

    #: The return series the models were actually fitted on, and how it was obtained. If
    #: ``price_source`` was ``"close"`` this is the derived column, and saying so is what
    #: makes the run reproducible.
    price_source: str
    return_column: str
    derived_returns: bool

    distribution: VolDistribution
    mean_model: VolMeanModel
    p: int
    o: int
    q: int
    lags: int

    walk_forward: WalkForwardSpec
    forecast_horizon: int

    rows: int
    return_rows: int

    #: The scored region, in bars and as timestamps. Repeated from the schedule rather than
    #: left to be re-derived: this is the sentence that says what the metrics are over.
    first_scored_index: int
    last_scored_index: int

    #: Typed ``datetime`` rather than left ``Any``. The value handed in is a
    #: ``pd.Timestamp``, and this is one of the two fields the report publishes *and* the
    #: lineage row records — and the lineage row is JSON text in sqlite, written with a
    #: plain ``json.dumps``. Naming the type makes "this has to serialise" a property of
    #: the schema, checked at construction, rather than a hope about what arrives.
    first_scored_time: datetime | None = None
    last_scored_time: datetime | None = None

    #: arch is fitted on percentage returns. The factor is stated, and the variance
    #: rescaling back to decimal units is exact, so this is a conditioning transform rather
    #: than a modelling choice — but it is a number in the artifact either way.
    scale_factor: float

    #: Stationarity, checked before fitting and reported whether or not it refused.
    adf_statistic: float
    adf_pvalue: float
    stationary: bool

    #: The realised-variance proxy the metrics are computed against, and the standing
    #: caution about it. Against squared returns the Mincer–Zarnowitz R² is near zero even
    #: for a correctly specified model — the proxy is that noisy — and a reader who does not
    #: know that will read a working model as a broken one.
    actual_proxy: str

    scores: list[VolModelScore] = Field(default_factory=list)
    comparisons: list[VolComparison] = Field(default_factory=list)

    #: Empirical quantiles of the walk-forward error ratio ``actual/forecast``, per model.
    #: These are the fan chart's bands. They are stated as empirical because the alternative
    #: — deriving bands from the fitted distribution — assumes the model is correctly
    #: specified, which is the thing the walk-forward exists to question.
    error_quantiles: dict[str, dict[str, float]] = Field(default_factory=dict)

    #: The projection beyond the last observation, one entry per model. Not scored, not
    #: written to the dataset — see ``VolForwardPath``.
    forward_paths: list[VolForwardPath] = Field(default_factory=list)

    #: The bar length the forward path's timestamps were projected with, and where it came
    #: from. A projection off a mis-inferred step is wrong by a multiple, so the rule that
    #: produced it is stated rather than implied.
    forward_step_seconds: float | None = None
    forward_step_source: str | None = None

    conditioning_dataset_id: str | None = None
    conditioning_dataset_name: str | None = None
    conditioning_columns: list[str] = Field(default_factory=list)

    warnings: list[str] = Field(default_factory=list)


class VolForecastCreated(BaseModel):
    """The terminal payload of a forecast job — what ``GET /api/jobs/{id}`` carries.

    The scores and the comparisons live on the report rather than being repeated here. Two
    copies of the same numbers in one payload is a second thing to keep in step, and the
    page reads whichever it reached first.
    """

    dataset: DatasetMeta | None
    report: VolForecastReport
    preview: dict[str, Any]


#: The report field each metric is read from, so the metrics table's columns and the
#: report's own field names cannot drift apart. ``DM`` is absent because it is pairwise;
#: the list of scalar metrics it must cover is ``enums.VOL_SCALAR_METRICS``.
VOL_METRIC_FIELDS: dict[VolMetric, str] = {
    VolMetric.MQLIKE: "mqlike",
    VolMetric.RMSE: "rmse",
    VolMetric.MAE: "mae",
    VolMetric.MZ_R2: "mz_r2",
}
