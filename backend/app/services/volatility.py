"""Module E over HTTP: the walk-forward loop, the scoring, and the artifact.

Everything statistical lives in ``volatility_models`` (the estimators) and
``volatility_eval`` (the metrics). What this module decides is the three things neither of
those can, and one of them is the whole reason the module exists:

1. **What a model is allowed to see.** Every scored forecast is produced by a model fitted
   on a slice of the return series that ends *at* the bar being forecast. The loop here is
   the only place slices are chosen, and it is written so there is no expression that could
   produce a slice reaching past its own origin. ``tests/test_volatility_models.py`` checks
   the property behaviourally — a run over 80% of the series must reproduce the first 80%
   of a run over all of it, to within 1e-6.

2. **What is refused, and when.** A dataset that does not exist is a 404 on the ``POST``.
   Everything else surfaces through the job's ``error`` field with the computed numbers in
   the message, because none of it is knowable until the file has been read. The schedule is
   the one that *looks* knowable from the request and is not: the series a walk-forward runs
   over is the return series, which is one bar shorter than the frame when returns are
   derived from close and shorter still when the usable span starts after a leading hole.
   Validating it against ``row_count`` would accept schedules that cannot run and refuse
   schedules that can, so it is checked against ``series.returns.size`` — the array the
   models are actually handed.

3. **What is recorded.** A volatility forecast is uninterpretable next to a different
   distribution, lag order or train window. The whole request goes into lineage as
   ``parameters_used``, beside the stationarity test and the scored region.

The dataset written is the **scored walk-forward span**: real bars, each with the forecast
that was made for it before it happened. The forward path — the projection the fan chart
draws — is not written to the store. A dataset row in this application is an observation,
and a projection is not one; putting future timestamps into the store would create rows
that no cleaning step, no split plan and no backtest is entitled to treat as data.
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from app.config import Settings, get_settings
from app.core.blocking import run_blocking
from app.core.errors import ValidationFailure
from app.models.dataset import DatasetMeta
from app.models.enums import (
    VOL_METRIC_LOWER_IS_BETTER,
    VOL_METRIC_RANKING,
    VOL_SCALAR_METRICS,
    DatasetCategory,
    JobStatus,
    Timeframe,
    VolDistribution,
    VolMetric,
    VolModel,
)
from app.models.jobs import JobRecord
from app.models.volatility import (
    MIN_SCORED_STEPS,
    MIN_TRAIN_WINDOW,
    VOL_FORECAST_COLUMNS,
    VOL_METRIC_FIELDS,
    VolComparison,
    VolFanBand,
    VolForecastAccepted,
    VolForecastReport,
    VolForecastRequest,
    VolForwardPath,
    VolModelScore,
    WalkForwardSpec,
)
from app.services import quality as quality_checks
from app.services import volatility_eval as evaluation
from app.services import volatility_models as models
from app.services.datasets import DatasetService, _preview_of, persist_dataset, resolved_path
from app.services.jobs import JobService, failure_payload, spawn
from app.storage import parquet_store
from app.storage.catalog import Catalog

#: Written into lineage so a saved forecast can be traced back to the module that produced
#: it without reading the operation string.
MODULE_SOURCE = "volatility"

JOB_KIND = "volatility_forecast"

#: ADF p-value above which the return series is refused. Not a tuning knob dressed as one:
#: the test is being used as a guard against a column that is not a return series at all,
#: and for that purpose the conventional 5% is right. The flag is reported either way.
STATIONARITY_ALPHA = 0.05

#: The conditioning columns a Module D regime dataset contributes. Probabilities and the
#: confidence they were committed with — the four ``REGIME_PROBABILITY_COLUMNS`` plus
#: ``confidence``. Listed here rather than imported from the regime module so the coupling
#: is one visible tuple rather than an implicit dependency, and so a rename over there
#: shows up as a missing column with a clear message rather than as a silently empty
#: feature block.
CONDITIONING_COLUMNS: tuple[str, ...] = (
    "prob_trending_up",
    "prob_trending_down",
    "prob_ranging",
    "prob_crisis",
    "confidence",
)

#: The realised-variance proxy. Stated once and repeated in the report, because a reader
#: who does not know it will read an MZ R² of 0.03 as a broken model rather than as the
#: expected value against a squared-return proxy.
ACTUAL_PROXY = "squared log return"

#: How many quantiles of the walk-forward error ratio the report carries.
ERROR_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)

#: How far above the largest squared return *in its own training window* a one-step variance
#: forecast may sit before it is recorded as a diverged step rather than as a forecast.
#:
#: **Why the bound is relative.** A fixed absolute ceiling cannot be right: a variance of
#: ``1e-6`` is enormous for a series of daily decimal returns and negligible for one quoted
#: in annualised basis points, and the same dataset can arrive either way. The window's own
#: largest squared return is the only yardstick that travels with the data.
#:
#: **Why a hundred.** A one-step variance forecast is anchored to the sample it was fitted
#: on. GARCH's recursion is a weighted average of past squared returns, so its forecast lies
#: *inside* the range the window has already seen; GJR's adds an asymmetric term to the same
#: recursion. EGARCH's is exponential and can step further — the previous conditional
#: variance multiplied by ``exp(α(|z| − E|z|) + γz)`` — but for parameters near the optimum
#: and a standardised shock within the window's own range that factor is single-digit. So a
#: forecast one *order of magnitude* above the window's largest squared return is already at
#: the edge of what these models can mean, and a hundredfold one cannot come from the model
#: at all: it comes from a fitted level that has left the data's scale.
#:
#: Measured on this project's own fixture (GARCH(1,1) returns, 30 rolling windows), the
#: ratio of forecast to window-maximum squared return is: GARCH ≤ 0.36, GJR ≤ 0.38, EGARCH
#: median 0.2 with the largest legitimate value at 15, and then a gap to 3 495 for a window
#: whose EGARCH fit pinned ``β = 0.999`` — a unit root in log-variance, which is a random
#: walk the likelihood cannot pin down on a short window. The bound sits above the former by
#: an order of magnitude and below the latter by more than an order of magnitude, which is
#: what a threshold separating "large" from "impossible" should look like.
#:
#: arch reports that runaway fit with ``convergence_flag = 0``: its flag is scipy's optimiser
#: status, which says the solver stopped, not that the answer is usable.
#:
#: The cost of a false refusal is one step counted as unscorable, printed with its reason;
#: the cost of *not* refusing is measured in what one diverged step does to everything
#: downstream. A single such step turned one run's headline MQLIKE into ``4.7e293``, made its
#: RMSE ``inf`` — which cannot even be serialised, since the API renders JSON with
#: ``allow_nan=False`` and a browser's ``JSON.parse`` throws on ``Infinity`` — wrote a
#: variance of ``8.3e213`` into the saved dataset as if it were an observation, and silently
#: emptied that model's comparison row.
MAX_FORECAST_VARIANCE_MULTIPLE = 100.0

#: How far *below* the variance of its own training window a one-step forecast may fall
#: before it is refused. The same breakdown as the ceiling above, approached from the other
#: end: an EGARCH whose fitted level has been driven to ``log σ² ≈ −300`` returns ``1e-130``,
#: which passes every "is it positive and finite" test and is no more a forecast of a calm
#: market than ``8e213`` is a forecast of a violent one.
#:
#: **Why the reference is the window's mean rather than its maximum.** The two bounds answer
#: different questions. "Beyond anything the window ever saw" is a comparison against the
#: largest observation, and it is tight because a variance recursion is anchored to its
#: sample. "Orders of magnitude below the window's own level" is a comparison against the
#: window's variance, because that is what the fit's level should be commensurate with — and
#: measuring it against the *smallest* squared return instead would make the bound vacuous
#: the moment one bar happened to close where it opened.
#:
#: **Why 1e-30.** The same measurement that fixed the ceiling gives the legitimate range.
#: Across GARCH, GJR and (on a window long enough to identify it) EGARCH, a one-step forecast
#: lands between **0.35 and 7.9** times its own window's mean squared return — a spread of
#: well under two orders of magnitude, centred on one, which is what a recursion anchored to
#: its sample should produce. Thirty orders is therefore not a threshold separating large from
#: small; nothing legitimate comes within twenty-five orders of it. It is the line past which
#: the value is a floating-point boundary rather than an estimate, and the same measurement
#: put the breakdown far beyond it: unidentified EGARCH fits returned ``1e-190`` and
#: ``3e-39`` on 100- and 140-bar windows of a series whose identified range is two orders
#: wide, and the XGBoost hybrid on the e2e driver's fixture returned ``1e-127`` while its two
#: companions returned MQLIKEs of 0.88 and 0.91. A tighter floor would risk refusing a
#: genuinely quiet forecast after a regime shift; this one only ever fires on a recursion that
#: has collapsed.
MIN_FORECAST_VARIANCE_RATIO = 1e-30

#: The prefix every refusal from :func:`_forecast_refusal` carries, so the scored-step
#: report can tell a diverged fit from an exception raised by arch and say something useful
#: about each. A tag rather than a second list on ``_WalkForward``: the failures list is
#: already the one place a step's reason is recorded, and it travels into the report and the
#: lineage row as it stands.
FORECAST_REFUSAL_TAG = "diverged forecast: "

_MAX_TRAIN_WINDOW = "train_window"


@dataclass
class _Series:
    """The return series every model in one run is fitted on, with its alignment.

    ``returns`` is what the models see: a finite array. ``offset`` is the index in the
    *original frame* of ``returns[0]``, so every scored step can be mapped back to the bar
    it belongs to. ``times`` is the observation instant of each element, already UTC.
    """

    returns: np.ndarray
    offset: int
    column: str
    derived: bool
    times: pd.DatetimeIndex
    exog: np.ndarray | None = None
    exog_columns: list[str] = field(default_factory=list)
    exog_rows_dropped: int = 0


@dataclass
class _WalkForward:
    """One model's walk-forward output, before it is scored."""

    model: VolModel
    indices: np.ndarray
    forecast: np.ndarray
    actual: np.ndarray
    seconds: float
    clamped: int
    untrained: bool
    failures: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- request


def _validate_schedule(spec: WalkForwardSpec, n_rows: int) -> list[int]:
    """The scored bar offsets, or a 422 naming the numbers.

    Called before the job exists on the ``run`` path so the driver and the tests get an
    exception rather than a failed job; on the ``forecast`` path the same arithmetic is
    checked here too, which is why it lives in a function rather than inline.
    """
    try:
        return spec.schedule(n_rows)
    except ValueError as exc:
        raise ValidationFailure(
            str(exc),
            hint="The scored region is the last `test_window` bars of the series, and the "
            "first of them needs `train_window` bars of history behind it.",
        ) from exc


# ---------------------------------------------------------------------- the series


def _resolve_returns(
    frame: pd.DataFrame, price_source: str
) -> tuple[np.ndarray, str, bool]:
    """The return series a request asks for, and how it was obtained.

    ``price_source`` is ``"close"``, ``"log_return"``, or the name of a column. ``"close"``
    derives ``diff(log(close))``; ``"log_return"`` prefers a column of that name and falls
    back to deriving it; anything else names a column and is used as given.

    **A price level is never handed to a model.** GARCH is a model of returns; fitting it
    to a price produces a near-unit-root variance equation whose parameters are dominated
    by where in the price range the sample happens to sit, and whose forecasts look
    plausible in a chart. The derivation happens here, once, rather than being a rule every
    caller has to remember.
    """
    requested = price_source.strip()
    if requested == "log_return":
        if "log_return" in frame.columns:
            return frame["log_return"].to_numpy(dtype=float), "log_return", False
        requested = "close"
    if requested == "close":
        if "close" not in frame.columns:
            raise ValidationFailure(
                "This dataset has no 'close' column, so log returns cannot be derived from it.",
                hint="Name a return column explicitly, or use a dataset that carries OHLCV "
                "bars. A price level is not accepted as a model input.",
            )
        close = frame["close"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            returns = np.diff(np.log(np.where(close > 0.0, close, np.nan)), prepend=np.nan)
        return returns, "log_return", True

    if requested not in frame.columns:
        raise ValidationFailure(
            f"This dataset has no '{requested}' column.",
            hint="Pick a column that exists, or use 'close' to derive log returns from it.",
        )
    return frame[requested].to_numpy(dtype=float), requested, False


def _conditioning_exog(
    frame: pd.DataFrame,
    time_column: str,
    parent_times: pd.Series,
    conditioning: pd.DataFrame | None,
    conditioning_name: str | None,
) -> tuple[np.ndarray | None, list[str], int]:
    """Align a regime dataset's posteriors onto the parent's bar timeline.

    **Joined on ``open_time``** — the observation instant — which is the whole reason the
    regime engine's mislabelled time column was repaired before this module was written. A
    join on the bar-close instant would align every posterior with the following bar: off
    by one, with no error, and a hybrid whose conditioning feature set was silently one bar
    stale — which, in a module about volatility clustering, is not a rounding error.

    A left join, so a parent bar with no regime row keeps its place with NaN features. The
    hybrid's feature builder drops rows with NaN and falls back to the linear forecast when
    the row it needs is one of them, so a partially-covered conditioning dataset degrades
    the model rather than corrupting the alignment.
    """
    if conditioning is None:
        return None, [], 0

    if time_column not in conditioning.columns:
        raise ValidationFailure(
            f"The conditioning dataset '{conditioning_name}' has no '{time_column}' column, "
            "so its rows cannot be aligned with the bars being forecast.",
            hint=f"A Module D regime output carries '{time_column}'. One produced before "
            "that column was named correctly carries the bar-close instant instead, and "
            "joining on it would date every posterior one bar late.",
        )

    available = [name for name in CONDITIONING_COLUMNS if name in conditioning.columns]
    if not available:
        raise ValidationFailure(
            f"The conditioning dataset '{conditioning_name}' has none of the columns a "
            f"regime output carries ({', '.join(CONDITIONING_COLUMNS)}).",
            hint="Conditioning expects a Module D regime dataset.",
        )

    right = conditioning[[time_column, *available]].copy()
    right[time_column] = pd.to_datetime(right[time_column], utc=True)
    # A duplicated key would fan the left join out and change the row count, which is the
    # one thing a merge must never do silently.
    right = right.drop_duplicates(subset=[time_column], keep="last")

    left = pd.DataFrame({time_column: pd.to_datetime(parent_times, utc=True)})
    merged = left.merge(right, on=time_column, how="left")
    exog = merged[available].to_numpy(dtype=float)
    dropped = int((~np.isfinite(exog).all(axis=1)).sum())
    return exog, available, dropped


def _build_series(
    frame: pd.DataFrame,
    *,
    time_column: str,
    price_source: str,
    conditioning: pd.DataFrame | None,
    conditioning_name: str | None,
) -> _Series:
    if time_column not in frame.columns:
        raise ValidationFailure(f"This dataset has no '{time_column}' column.")
    raw, column, derived = _resolve_returns(frame, price_source)

    times_all = pd.DatetimeIndex(pd.to_datetime(frame[time_column], utc=True))
    if times_all.hasnans:
        raise ValidationFailure(
            "The time column has empty entries, so the bars cannot be ordered.",
            hint="Cleaning (B1) drops or fills empty timestamps.",
        )
    if not times_all.is_monotonic_increasing:
        raise ValidationFailure(
            "The time column is not sorted, so a walk-forward over it would not be a "
            "walk-forward.",
            hint="The scored region is chosen by position; on an unsorted series those "
            "positions are not chronological.",
        )
    if times_all.has_duplicates:
        raise ValidationFailure(
            "The time column has duplicate entries, so bar positions are ambiguous.",
        )

    exog, exog_columns, dropped = _conditioning_exog(
        frame, time_column, frame[time_column], conditioning, conditioning_name
    )

    usable = np.isfinite(raw)
    # Conditioning rows with no regime yet are still usable by a plain GARCH — the
    # probabilities are only read by the hybrids — so the mask here is the returns alone.
    first = int(np.argmax(usable)) if usable.any() else 0
    if not usable.any():
        raise ValidationFailure(
            "The selected source produces no finite return values.",
            hint="Pick a different column, or clean the dataset first.",
        )
    last = int(len(usable) - np.argmax(usable[::-1]))
    # A hole *inside* the usable span would silently change what "the last `test_window`
    # bars" means, so it is refused rather than bridged.
    if not usable[first:last].all():
        holes = int((~usable[first:last]).sum())
        raise ValidationFailure(
            f"The return series has {holes} empty values inside its usable span.",
            hint="Refit on a contiguous run of bars; a gap would make the walk-forward's "
            "positions non-contiguous without anything in the output saying so.",
        )

    returns = raw[first:last].astype(float)
    exog_slice = exog[first:last] if exog is not None else None
    return _Series(
        returns=returns,
        offset=first,
        column=column,
        derived=derived,
        times=times_all[first:last],
        exog=exog_slice,
        exog_columns=exog_columns,
        exog_rows_dropped=dropped,
    )


# ------------------------------------------------------------------ walk-forward


def _forecast_refusal(
    value: float, training_max_square: float, training_mean_square: float
) -> str | None:
    """Why this one-step variance forecast cannot be scored, or ``None`` if it can.

    The ways a fit that *returned* something is still not a forecast, with the numbers in
    the sentence, because "the model diverged" is not actionable and "EGARCH returned
    8.28e+213 from a window whose largest squared return is 2.1e-05" is.

    Two references from the same window, because they answer different questions.
    ``training_max_square`` is the largest realised squared return in it: a forecast beyond
    a hundred times that is beyond anything the window ever saw, which is the *tight* test
    and the one measured against real fits. ``training_mean_square`` is the window's own
    variance: a forecast orders of magnitude below it is the same breakdown from the other
    end — a log-variance recursion whose level has been driven to −300 does not predict a
    quiet market, it predicts ``1e-130``. Both are relative because a fixed bound cannot be
    right here: a variance of ``1e-6`` is enormous for a series quoted in daily decimal
    returns and negligible for one quoted in annualised basis points, and the same dataset
    can arrive either way.

    A non-finite value is reported rather than allowed to propagate for the reason the
    metric module gives: one ``inf`` erases a mean. A value at or below zero is a recursion
    at its boundary — arch's variance recursion is a sum of squares and cannot legitimately
    reach zero for a series with any variation, so reaching it means an optimiser has walked
    into a corner of the parameter space, not that the market is about to go quiet.

    Every message carries ``FORECAST_REFUSAL_TAG``, so ``_score`` can tell this kind of
    failed step from an exception and answer it with the right advice.
    """
    if not math.isfinite(value):
        return FORECAST_REFUSAL_TAG + (
            f"a walk-forward step produced a non-finite variance forecast ({value!r})"
        )
    if value <= 0.0:
        return FORECAST_REFUSAL_TAG + (
            f"a walk-forward step produced a variance forecast of {value:.3g}; a variance at "
            "or below zero is a recursion at its boundary, not a prediction of a calm market"
        )
    if training_max_square <= 0.0:
        return FORECAST_REFUSAL_TAG + (
            f"a walk-forward step produced a variance forecast of {value:.3g} from a "
            "training window whose returns are all exactly zero, which carries no volatility "
            "to forecast"
        )
    if value > MAX_FORECAST_VARIANCE_MULTIPLE * training_max_square:
        return FORECAST_REFUSAL_TAG + (
            f"a walk-forward step produced a variance forecast of {value:.3g}, more than "
            f"{MAX_FORECAST_VARIANCE_MULTIPLE:.0f} times the largest squared return in its "
            f"own training window ({training_max_square:.3g}); the fit has left the "
            "parameter space rather than forecast anything"
        )
    if training_mean_square > 0.0 and value < MIN_FORECAST_VARIANCE_RATIO * training_mean_square:
        return FORECAST_REFUSAL_TAG + (
            f"a walk-forward step produced a variance forecast of {value:.3g}, more than "
            f"{abs(math.log10(MIN_FORECAST_VARIANCE_RATIO)):.0f} orders of magnitude below "
            f"the variance of its own training window ({training_mean_square:.3g}); a "
            "recursion at that level is a boundary, not a prediction of a calm market"
        )
    return None


def _walk_forward(
    model: models.ArchModel | models.HybridModel,
    series: _Series,
    spec: WalkForwardSpec,
) -> _WalkForward:
    """Fit and forecast one bar at a time, refitting at every step.

    Refitting at every scored bar is the expensive choice and the obviously correct one.
    Reusing parameters for a few bars would still be causal — parameters estimated from
    data up to ``t`` say nothing about ``t+1`` — but it makes the answer depend on where
    the refit points fall, and a reader comparing two runs would have to reconstruct the
    refit schedule to know whether the difference was the model or the schedule. Paying the
    cost removes the question. ``step_size`` is the knob for trading it back.
    """
    train = spec.train_window
    indices = np.asarray(spec.schedule(series.returns.size), dtype=int)
    forecast = np.full(indices.size, np.nan)
    actual = np.full(indices.size, np.nan)
    failures: list[str] = []
    clamped = 0
    untrained = False
    started = time.perf_counter()

    for position, t in enumerate(indices):
        window = slice(t - train, t)
        exog_window = series.exog[window] if series.exog is not None else None
        try:
            model.fit(series.returns[window], exog_window)
            value = float(model.forecast_variance(1)[0])
            refusal = _forecast_refusal(
                value,
                float(np.max(series.returns[window] ** 2)),
                float(np.mean(series.returns[window] ** 2)),
            )
            if refusal is not None:
                # Recorded, not raised: the step becomes a NaN and is counted, exactly like
                # a fit that threw. What it must never be is *scored* — see the constant.
                if len(failures) < 5:
                    failures.append(refusal)
                value = np.nan
            forecast[position] = value
        except Exception as exc:  # noqa: BLE001 - one bad step must not lose the other 249
            # Recorded as a non-finite step and counted. Letting the exception out would
            # abandon a run that is 90% finished over one optimiser failure; swallowing it
            # without a trace would make the model look like it converged on every step.
            if len(failures) < 5:
                failures.append(f"{type(exc).__name__}: {exc}")
            forecast[position] = np.nan
        actual[position] = float(series.returns[t] ** 2)
        if isinstance(model, models.HybridModel):
            clamped += model.correction_clamped
            untrained = untrained or model.untrained

    return _WalkForward(
        model=model.name,
        indices=indices + series.offset,
        forecast=forecast,
        actual=actual,
        seconds=time.perf_counter() - started,
        clamped=clamped,
        untrained=untrained,
        failures=failures,
    )


def _score(run: _WalkForward, family: str) -> VolModelScore:
    """The metrics for one model, over the steps that produced a finite forecast."""
    finite = np.isfinite(run.forecast) & np.isfinite(run.actual)
    n = int(run.forecast.size)
    n_finite = int(finite.sum())
    nan_count = int(np.isnan(run.forecast).sum())
    inf_count = int(np.isinf(run.forecast).sum())

    score = VolModelScore(
        model=run.model,
        family=family,
        n_steps=n,
        n_finite=n_finite,
        n_nan=nan_count,
        n_inf=inf_count,
        seconds=round(run.seconds, 4),
    )
    if n_finite < MIN_SCORED_STEPS:
        # Two different failures, and conflating them puts the blame in the wrong place. A
        # protocol that scored ten bars and got ten finite forecasts has a model that is
        # working and a sample that is too small; the sentence below is what tells the user
        # to widen the window rather than to distrust the optimiser.
        if n_finite == n and run.failures == []:
            score.unscored_reason = (
                f"All {n} steps produced a finite forecast, but a metric needs at least "
                f"{MIN_SCORED_STEPS} to be an estimate of anything. Widen the test window, "
                "or shorten the step size so more of it is scored."
            )
        else:
            score.unscored_reason = (
                f"Only {n_finite} of {n} steps produced a finite forecast, and a metric "
                f"needs at least {MIN_SCORED_STEPS} to be an estimate of anything. "
                + (f"First failure: {run.failures[0]}" if run.failures else "")
                # A diverged fit and an exception call for different responses, and the two
                # arrive here in the same list. A divergence is a model whose parameter count
                # the window cannot identify — more data is the fix. An exception from arch
                # is usually a step the optimiser could not solve at all, which a wider
                # window may or may not help. Saying nothing leaves the user to guess which
                # they have, and the first guess anyone makes is "the model is broken".
                + (
                    " A fit that leaves the parameter space usually has more parameters than "
                    "the window can identify, so a longer train_window is the likeliest fix."
                    if any(
                        failure.startswith(FORECAST_REFUSAL_TAG) for failure in run.failures
                    )
                    else ""
                )
            ).strip()
        return score

    a = run.actual[finite]
    f = run.forecast[finite]
    score.mqlike = evaluation.mqlike(a, f)
    score.rmse = evaluation.rmse(a, f)
    score.mae = evaluation.mae(a, f)
    mz = evaluation.mincer_zarnowitz(a, f)
    score.mz_intercept = mz.intercept
    score.mz_slope = mz.slope
    score.mz_r2 = mz.r2
    return score


@dataclass(frozen=True)
class _Comparisons:
    """The pairwise tests, and the pairs that could not be tested.

    Two lists rather than one because a comparison that did not happen is a *finding*, not an
    absence. ``_compare`` used to skip a pair whose jointly-scorable mask fell below the
    floor, so a two-model run could return ``comparisons: []`` with nothing anywhere saying
    why — the reader sees two models and no verdict, and has to guess whether the test
    failed, was skipped, or was never asked for.
    """

    rows: list[VolComparison]
    gaps: list[str]


def _compare(runs: list[_WalkForward]) -> _Comparisons:
    """Pairwise Diebold–Mariano over the steps **both** models and the actual are finite for.

    Intersecting the masks rather than letting each pair find its own is what makes the
    comparison a comparison: on two different subsets of the walk-forward span, a
    "difference" between two models is partly a difference between two samples.

    A pair with too little overlap for the test is reported as a gap naming both models and
    the count, never dropped in silence. This is the case a two-model run on a short window
    lands in whenever one of the two diverges on most of its steps, and it is exactly the
    case where the reader most needs to be told the verdict is missing.
    """
    rows: list[VolComparison] = []
    gaps: list[str] = []
    for i, first in enumerate(runs):
        for second in runs[i + 1 :]:
            mask = (
                np.isfinite(first.actual)
                & np.isfinite(first.forecast)
                & np.isfinite(second.forecast)
                & (first.forecast > 0.0)
                & (second.forecast > 0.0)
            )
            n_shared = int(mask.sum())
            if n_shared < MIN_SCORED_STEPS:
                gaps.append(
                    f"{first.model.value} and {second.model.value} share {n_shared} steps "
                    "where both produced a usable forecast, fewer than the "
                    f"{MIN_SCORED_STEPS} a Diebold–Mariano test needs to estimate a "
                    "variance; no comparison is reported for that pair."
                )
                continue
            actual = first.actual[mask]
            loss_a = evaluation.qlike_loss_series(actual, first.forecast[mask])
            loss_b = evaluation.qlike_loss_series(actual, second.forecast[mask])
            result = evaluation.diebold_mariano(loss_a, loss_b, horizon=1)
            better: VolModel | None = None
            if result.mean_differential < 0.0:
                better = first.model
            elif result.mean_differential > 0.0:
                better = second.model
            rows.append(
                VolComparison(
                    model_a=first.model,
                    model_b=second.model,
                    n_effective=result.n_effective,
                    statistic=result.statistic,
                    p_value=result.p_value,
                    better=better,
                    significant=result.p_value < 0.05,
                )
            )
    return _Comparisons(rows=rows, gaps=gaps)


def _forward_step(times: pd.DatetimeIndex, timeframe: Timeframe | None) -> tuple[float, str]:
    """The bar length the forward path's timestamps are projected with.

    The declared timeframe when the catalog has one, otherwise the median spacing actually
    observed. The second is a fallback and not a preference: a dataset whose metadata says
    "1h" but whose rows are 4 hours apart is a real thing, and projecting its future bars
    three hours out would put the fan chart on a timeline nothing else in the application
    shares. Which rule was used is reported beside the number.
    """
    if timeframe is not None:
        from app.models.enums import TIMEFRAME_SECONDS

        return float(TIMEFRAME_SECONDS[timeframe]), "declared timeframe"
    if times.size >= 2:
        deltas = np.diff(times.view("int64")) / 1e9
        median = float(np.median(deltas))
        if median > 0.0 and math.isfinite(median):
            return median, "median observed spacing"
    return 3600.0, "assumed hourly (no timeframe and no spacing to measure)"


def _forward_path(
    model: models.ArchModel | models.HybridModel,
    series: _Series,
    spec: WalkForwardSpec,
    horizon: int,
    step_seconds: float,
    quantiles: dict[str, float],
) -> VolForwardPath:
    """The projection beyond the last observation, with its empirical bands.

    Fitted on the last ``train_window`` bars **up to and including the final bar** — the
    span the last scored step used, plus that one bar. The extra bar is allowed here and
    was not allowed there: bar ``t`` can only be *scored* by a fit that never saw it, while
    a projection made *from* ``t`` may use it. Nothing past the end of the series is
    touched, so there is no leak either way.

    The window is ``train_window`` rather than the whole series deliberately. It is the span
    the user chose and the span the scored run's own final refit used, so the fan chart and
    the last scored forecast beside it are two readings of the same estimates. An
    all-history refit here would silently re-estimate the model on a different sample at the
    one point where the reader is most likely to compare the two. Raising ``train_window``
    widens both fits together.

    The bands are the fitted variance scaled by the empirical quantiles of the model's own
    walk-forward error ratio, so each band follows the forecast's own term structure rather
    than being one interval drawn around the whole path.

    The steps are put through the same refusal as the scored ones, and the reason is sharper
    here: this is the series the fan chart draws as its headline. A fitted recursion that has
    left the parameter space would set the chart's axis at ``8e213`` and flatten every bar,
    every band and the realised line onto zero — a picture that looks like a rendering bug and
    is really a diverged fit. Refusing leaves a gap, and the report already warns that a gap
    in a forward path is a gap rather than an interpolated value.
    """
    window = slice(series.returns.size - spec.train_window, series.returns.size)
    exog_window = series.exog[window] if series.exog is not None else None
    model.fit(series.returns[window], exog_window)
    raw = np.asarray(model.forecast_variance(horizon), dtype=float).ravel()[:horizon]
    training_max_square = float(np.max(series.returns[window] ** 2))
    training_mean_square = float(np.mean(series.returns[window] ** 2))
    variance = np.array(
        [
            float(value)
            if _forecast_refusal(float(value), training_max_square, training_mean_square) is None
            else np.nan
            for value in raw
        ]
    )

    last = series.times[-1]
    offset = pd.Timedelta(seconds=step_seconds)
    times = [last + offset * (k + 1) for k in range(variance.size)]

    bands = [
        VolFanBand(
            label=label,
            low=[_scaled(value, low_q) for value in variance],
            high=[_scaled(value, high_q) for value in variance],
        )
        for label, low_q, high_q in evaluation.fan_bands(quantiles)
    ]

    return VolForwardPath(
        model=model.name,
        times=list(times),
        variance=[float(value) for value in variance],
        volatility=[_sqrt_or_nan(value) for value in variance],
        bands=bands,
    )


def _scaled(value: float, factor: float) -> float:
    scaled = float(value) * float(factor)
    return scaled if math.isfinite(scaled) else math.nan


def _sqrt_or_nan(value: float) -> float:
    return math.sqrt(value) if math.isfinite(value) and value >= 0.0 else math.nan


# ---------------------------------------------------------------------- the service


class VolatilityService:
    """Runs Module E. Public methods are coroutines."""

    def __init__(self, settings: Settings | None = None, catalog: Catalog | None = None) -> None:
        self.settings = settings or get_settings()
        self.catalog = catalog or Catalog(self.settings.catalog_path)
        self.datasets = DatasetService(self.settings, self.catalog)
        self.jobs = JobService(self.settings, self.catalog)

    # ------------------------------------------------------------------- entry

    async def forecast(self, request: VolForecastRequest) -> VolForecastAccepted:
        """Accept the work and return a handle to it.

        The dataset is resolved here, before the job is created, so an unknown id is an
        immediate 404 with the id in the message rather than a 202 for a run that cannot
        start. Everything else about the data needs the file read, so it surfaces through
        the job.
        """
        parent = await self.datasets.get(request.dataset_id)
        job_id = await self.jobs.create(
            JOB_KIND,
            dataset_id=parent.dataset_id,
            params=request.model_dump(mode="json"),
        )
        spawn(self._execute(job_id, request, parent))
        return VolForecastAccepted(
            job_id=job_id, status=JobStatus.RUNNING, poll=f"/api/jobs/{job_id}"
        )

    async def run(self, request: VolForecastRequest) -> dict[str, Any]:
        """Compute inline and return the payload, without scheduling a job."""
        parent = await self.datasets.get(request.dataset_id)
        return await self._compute(request, parent)

    async def job(self, job_id: str) -> JobRecord:
        return await self.jobs.get(job_id)

    async def wait(
        self, job_id: str, *, timeout: float = 3600.0, interval: float = 0.1
    ) -> JobRecord:
        """Block until a job reaches a terminal state. For tests and scripted drivers.

        A longer default than Module D's: a hybrid walk-forward refits a second model at
        every scored step, and a 900-second ceiling would be a ceiling the slowest
        legitimate run hits.
        """
        waited = 0.0
        while waited < timeout:
            record = await self.jobs.get(job_id)
            if record.status is not JobStatus.RUNNING:
                return record
            await asyncio.sleep(interval)
            waited += interval
        raise TimeoutError(f"Job {job_id} did not finish within {timeout:.0f}s.")

    # ----------------------------------------------------------------- options

    async def options(self, dataset_id: str) -> dict[str, Any]:
        """What this dataset can support, decided here and not in the browser.

        The same rule the feature catalogue and Module D follow: one function answers "is
        this available, and if not why", so a disabled control and a refusal from the run
        endpoint cannot disagree about the reason.
        """
        meta = await self.datasets.get(dataset_id)
        conditioning = await self._conditioning_options()

        def work() -> dict[str, Any]:
            present = {column.name for column in meta.columns}
            derived = "close" in present
            sources = [
                {
                    "id": "close",
                    "label": "Log return from close",
                    "available": derived,
                    "reason": None
                    if derived
                    else "This dataset has no 'close' column, so log returns cannot be "
                    "derived from it.",
                },
                {
                    "id": "log_return",
                    "label": "Existing log_return column",
                    "available": "log_return" in present,
                    "reason": None
                    if "log_return" in present
                    else "No 'log_return' column. Produce one in C5, or derive it from close.",
                },
            ]

            # A default schedule the dataset can actually run. Derived from the row count
            # rather than fixed, because the fixed default would be refused on any dataset
            # shorter than 750 rows — and a form that opens on its own error message is a
            # form that teaches the user the module is broken.
            #
            # Derived from the *return* count, not the row count. Log returns derived from a
            # close column are one bar shorter than the frame: the first difference has no
            # predecessor. Sizing the windows to the frame makes the default schedule exceed
            # the series by exactly that bar, so every dataset short enough to reach the
            # `rows // 2` branch would open on a refusal — a failure of the form, reported
            # far from its cause. Reserving the bar here is the whole fix.
            rows = meta.row_count
            usable = max(0, rows - 1)
            train = min(500, max(MIN_TRAIN_WINDOW, usable // 2))
            test = min(250, max(MIN_SCORED_STEPS, usable - train))
            feasible = usable >= train + test

            return {
                "dataset_id": meta.dataset_id,
                "name": meta.name,
                "rows": rows,
                "usable_returns": usable,
                "timeframe": meta.timeframe.value if meta.timeframe else None,
                "asset_class": meta.asset_class.value,
                "columns": [column.name for column in meta.columns],
                "sources": sources,
                "models": _model_options(),
                "metrics": _metric_options(),
                "conditioning": conditioning,
                # The bounds the request schema enforces, so the page's live arithmetic
                # uses the same numbers the refusal would. Restating 50 and 30 in
                # TypeScript would be a second copy of a guardrail, and the copy is what
                # would drift: the form would open on a schedule the endpoint refuses.
                "limits": {
                    "min_train_window": MIN_TRAIN_WINDOW,
                    "min_test_window": MIN_SCORED_STEPS,
                    "min_scored_steps": MIN_SCORED_STEPS,
                    "min_step_size": 1,
                    "max_forecast_horizon": 500,
                    "max_order": 5,
                    "max_lags": 100,
                },
                "defaults": {
                    "price_source": "close" if derived else "log_return",
                    "models": [VolModel.GARCH.value],
                    "distribution": VolDistribution.STUDENTS_T.value,
                    "mean_model": "constant",
                    "p": 1,
                    "o": 1,
                    "q": 1,
                    "lags": 5,
                    "train_window": train,
                    "test_window": test,
                    "step_size": 1,
                    "forecast_horizon": 24,
                },
                "feasible": feasible,
                "warnings": _option_warnings(meta, train, test, usable, feasible),
            }

        return await run_blocking(work)

    async def _conditioning_options(self) -> list[dict[str, Any]]:
        """The datasets that can condition a hybrid, as the page's dropdown needs them.

        Filtered by the columns a regime output carries rather than by the category, so a
        regime dataset that was re-saved into another category is still offered and a
        non-regime dataset is not.

        The schema is read per dataset rather than taken from the explorer listing, which
        carries no columns at all — the filter *is* a column test, so the cheap projection
        cannot answer it. Both reads happen inside the one blocking closure, so this costs
        a single thread hand-off for the whole loop rather than one per dataset.

        A dataset without ``open_time`` is offered but marked unusable. It is listed rather
        than hidden so the reason is visible, and there is no fallback to the bar-close
        instant: a posterior joined on it would be dated one bar late, which is the
        observation-time rule this module is most exposed to.
        """

        def work() -> list[dict[str, Any]]:
            catalog = self.datasets.catalog
            found: list[dict[str, Any]] = []
            for summary in catalog.list_datasets(DatasetCategory.OUTPUTS):
                meta = catalog.get_dataset(summary.dataset_id)
                if meta is None:
                    continue
                names = {column.name for column in meta.columns}
                if "prob_crisis" not in names:
                    continue
                joinable = "open_time" in names
                found.append(
                    {
                        "dataset_id": meta.dataset_id,
                        "name": meta.name,
                        "rows": meta.row_count,
                        "columns": sorted(names & set(CONDITIONING_COLUMNS)),
                        "usable": joinable,
                        "reason": None
                        if joinable
                        else "This regime output carries no observation-time column, so its "
                        "posteriors cannot be aligned with the bars being forecast.",
                    }
                )
            return found

        return await run_blocking(work)

    # ------------------------------------------------------------------ worker

    async def _execute(
        self, job_id: str, request: VolForecastRequest, parent: DatasetMeta
    ) -> None:
        """The background task: run, then record the terminal state.

        Failures are written to the job rather than raised — there is no request left to
        carry a 422, and an exception escaping into the event loop would leave the job at
        RUNNING forever while the traceback went somewhere the user is not reading.
        """
        await self.jobs.start(job_id)
        try:
            payload = await self._compute(request, parent)
        except Exception as exc:  # noqa: BLE001 - the job record is the error channel
            await self.jobs.fail(job_id, failure_payload(exc), message=str(exc))
            return
        dataset = payload.get("dataset") or {}
        await self.jobs.succeed(
            job_id,
            result=payload,
            message=_summary_message(payload),
            output_dataset_id=dataset.get("dataset_id"),
        )

    async def _compute(
        self, request: VolForecastRequest, parent: DatasetMeta
    ) -> dict[str, Any]:
        path = resolved_path(parent, self.settings)
        conditioning_meta: DatasetMeta | None = None
        if request.conditioning_dataset_id is not None:
            conditioning_meta = await self.datasets.get(request.conditioning_dataset_id)

        def work() -> dict[str, Any]:
            frame = parquet_store.read_dataset(path)
            time_column = quality_checks.find_time_column(frame)
            if time_column is None:
                raise ValidationFailure(
                    "This dataset has no time column, so the forecasts cannot be stamped "
                    "with the bars they belong to.",
                    hint="A walk-forward is defined over a chronological series.",
                )

            conditioning_frame: pd.DataFrame | None = None
            if conditioning_meta is not None:
                conditioning_frame = parquet_store.read_dataset(
                    resolved_path(conditioning_meta, self.settings)
                )

            series = _build_series(
                frame,
                time_column=time_column,
                price_source=request.price_source,
                conditioning=conditioning_frame,
                conditioning_name=conditioning_meta.name if conditioning_meta else None,
            )

            # The schedule is validated against the series the models will actually see,
            # not against the frame's row count: log returns lose one bar at the front.
            schedule_indices = _validate_schedule(request.walk_forward, series.returns.size)

            adopted_statistic, adf_pvalue = models.adf_test(series.returns)
            if adf_pvalue > STATIONARITY_ALPHA:
                raise ValidationFailure(
                    f"The series selected by price_source='{request.price_source}' is not "
                    f"stationary: the augmented Dickey–Fuller statistic is "
                    f"{adopted_statistic:.4f} with a p-value of {adf_pvalue:.4f}, so the "
                    f"unit-root null is not rejected at {STATIONARITY_ALPHA:.0%}.",
                    hint="GARCH is a model of a stationary return series. If this column is "
                    "a price level, a running total or a cumulative volume, point "
                    "price_source at 'close' so log returns are derived from it.",
                )

            runs: list[_WalkForward] = []
            forward_paths: list[VolForwardPath] = []
            quantiles_by_model: dict[str, dict[str, float]] = {}
            scores: list[VolModelScore] = []
            step_seconds, step_source = _forward_step(series.times, parent.timeframe)

            for name in request.models:
                model = models.build_model(
                    name,
                    distribution=request.distribution,
                    mean_model=request.mean_model,
                    p=request.p,
                    o=request.o,
                    q=request.q,
                    lags=request.lags,
                )
                run = _walk_forward(model, series, request.walk_forward)
                runs.append(run)
                scores.append(_score(run, name.family))

                quantiles = evaluation.forecast_error_quantiles(
                    run.actual / run.forecast, ERROR_QUANTILES
                )
                quantiles_by_model[name.value] = quantiles
                forward_paths.append(
                    _forward_path(
                        model,
                        series,
                        request.walk_forward,
                        request.forecast_horizon,
                        step_seconds,
                        quantiles,
                    )
                )

            output = _output_frame(runs, series)
            report = _report(
                request=request,
                parent=parent,
                series=series,
                runs=runs,
                scores=scores,
                comparisons=_compare(runs),
                quantiles=quantiles_by_model,
                forward_paths=forward_paths,
                schedule_indices=schedule_indices,
                step_seconds=step_seconds,
                step_source=step_source,
                adf_statistic=adopted_statistic,
                adf_pvalue=adf_pvalue,
                conditioning_meta=conditioning_meta,
            )

            meta: DatasetMeta | None = None
            if request.write:
                meta = persist_dataset(
                    parent=parent,
                    frame=output,
                    data_dir=self.settings.data_dir,
                    catalog=self.catalog,
                    # `outputs`: the product of an analysis rather than an input to one.
                    # Its rows are forecasts, not bars, and a forecasting model that wanted
                    # to consume them would want them as features, not as prices.
                    category=DatasetCategory.OUTPUTS,
                    name=request.name
                    or f"{parent.name} (vol forecast · {_model_summary(request)})",
                    operation=MODULE_SOURCE,
                    detail=_lineage_detail(request, report, conditioning_meta),
                    extra_provenance={
                        "module_source": MODULE_SOURCE,
                        "distribution": request.distribution.value,
                        "mean_model": request.mean_model.value,
                        "models": ",".join(model.value for model in request.models),
                        "train_window": str(request.walk_forward.train_window),
                        "test_window": str(request.walk_forward.test_window),
                        "step_size": str(request.walk_forward.step_size),
                    },
                    extra_parents=(
                        [(conditioning_meta.dataset_id, {"role": "conditioning"})]
                        if conditioning_meta is not None
                        else None
                    ),
                )
            return {
                "dataset": meta.model_dump(mode="json") if meta is not None else None,
                "report": report.model_dump(mode="json"),
                "preview": _preview_of(output),
            }

        return await run_blocking(work)


# ------------------------------------------------------------------------ helpers


def _model_options() -> list[dict[str, Any]]:
    """The catalogue of models, in the shape the page renders.

    ``family`` drives the grouping and ``needs`` drives which controls are live, both read
    from the enum so a model added there appears here without a second edit.
    """
    labels = {
        VolModel.GARCH: "GARCH",
        VolModel.EGARCH: "EGARCH",
        VolModel.GJR_GARCH: "GJR-GARCH",
        VolModel.ARMA_EGARCH_XGBOOST: "ARMA-EGARCH-XGBoost",
        VolModel.TDNN_GJR_GARCH: "TDNN-GJR-GARCH",
    }
    notes = {
        VolModel.GARCH: "Symmetric, the baseline every other model is measured against.",
        VolModel.EGARCH: "Log-variance, so a shock never drives the variance negative.",
        VolModel.GJR_GARCH: "Adds a leverage term: a fall raises tomorrow's variance more "
        "than a rise of the same size.",
        VolModel.ARMA_EGARCH_XGBOOST: "Fits a gradient-boosted correction to the linear "
        "model's own residuals. Refits a second model at every step — the slowest option.",
        VolModel.TDNN_GJR_GARCH: "Feeds a tapped delay line to a small neural network and "
        "uses its output the same way. Refits a network at every step.",
    }
    return [
        {
            "id": model.value,
            "label": labels[model],
            "family": model.family,
            "needs": list(model.needs),
            "note": notes[model],
        }
        for model in VolModel
    ]


def _metric_options() -> list[dict[str, Any]]:
    """The metrics table's columns, with the direction that counts as better.

    Sent from here rather than decided in TypeScript for the reason every other verdict in
    this application is: marking the *highest* MQLIKE as best is a one-character mistake
    that inverts the conclusion of the whole report, and it should be impossible to make it
    in two places independently.

    ``is_ranking`` is not the same field as ``lower_is_better``. The R² has a direction
    (higher is better) and is still not a ranking, so a table that inferred "crown the
    maximum" from the direction alone would mark the largest R² in the column as the winner
    — contradicting the note printed beside it. Both facts go over the wire so the page
    does not have to reconstruct either.
    """
    labels = {
        VolMetric.MQLIKE: ("MQLIKE", "Patton's robust QLIKE on variance. Lower is better."),
        VolMetric.RMSE: ("RMSE", "Root mean squared error on variance. Lower is better."),
        VolMetric.MAE: ("MAE", "Mean absolute error on variance. Lower is better."),
        VolMetric.MZ_R2: (
            "MZ R²",
            "Mincer–Zarnowitz fit. A diagnostic, not a ranking — near zero is expected "
            "against a squared-return proxy.",
        ),
    }
    return [
        {
            "id": metric.value,
            "label": labels[metric][0],
            "note": labels[metric][1],
            "lower_is_better": VOL_METRIC_LOWER_IS_BETTER[metric],
            "is_ranking": VOL_METRIC_RANKING[metric],
        }
        for metric in VOL_SCALAR_METRICS
    ]


def _option_warnings(
    meta: DatasetMeta, train: int, test: int, usable: int, feasible: bool
) -> list[dict[str, str]]:
    """The reasons a run would be refused or degraded, so the page can say them early.

    The short-dataset message names both counts. "Only 60 bars" beside a schedule needing 80
    reads as a mismatch between two numbers the reader can see, and the reader would count the
    dataset's rows and find 60 — while the walk-forward is over 59. Both numbers, so the
    off-by-one is stated rather than discovered.
    """
    warnings: list[dict[str, str]] = []
    names = {column.name for column in meta.columns}
    if not feasible:
        warnings.append(
            {
                "level": "error",
                "code": "too_short",
                "message": f"Only {usable} usable bars — {meta.row_count} rows, less the first "
                f"difference that has no predecessor. A {train}-bar training window and a "
                f"{test}-bar test window need at least {train + test}. Fetch a longer series.",
            }
        )
    if "close" not in names:
        warnings.append(
            {
                "level": "error",
                "code": "no_close",
                "message": "No 'close' column, so log returns cannot be derived. Name a "
                "return column instead.",
            }
        )
    # There is deliberately no warning about a missing 'volume' column, which Module D
    # raises. No estimator here reads volume: Family A is a model of the return series
    # alone, and the hybrids' delay line carries lagged returns, lagged absolute returns
    # and the conditioning posteriors (DECISIONS §109). Warning that the hybrids "lose an
    # input" would send a user to fetch volume data that no model in this module consumes.
    if meta.timeframe is None:
        warnings.append(
            {
                "level": "warning",
                "code": "no_timeframe",
                "message": "No timeframe recorded, so the forward path's timestamps are "
                "projected from the spacing actually observed in the data.",
            }
        )
    return warnings


def _model_summary(request: VolForecastRequest) -> str:
    names = [model.value for model in request.models]
    if len(names) <= 2:
        return " + ".join(names)
    return f"{names[0]} + {len(names) - 1} more"


def _output_frame(runs: list[_WalkForward], series: _Series) -> pd.DataFrame:
    """The scored walk-forward span, one row per (model, bar), sorted by time then model.

    Sorted by **time first**, not by model, so the stored frame is a chronological series
    like every other dataset in the store. A frame grouped by model and restarting at the
    beginning of the scored span for each one is not monotonic, and the next module to read
    it would refuse it as unsorted — correctly, since it cannot tell an intentional
    interleave from a corrupted one.
    """
    frames: list[pd.DataFrame] = []
    for run in runs:
        times = series.times[run.indices - series.offset]
        forecast = run.forecast
        actual = run.actual
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(
                np.isfinite(forecast) & (forecast > 0.0) & np.isfinite(actual),
                actual / forecast,
                np.nan,
            )
        frames.append(
            pd.DataFrame(
                {
                    "open_time": times,
                    "model": run.model.value,
                    "forecast_variance": forecast,
                    "forecast_volatility": [_sqrt_or_nan(value) for value in forecast],
                    "actual_variance": actual,
                    "actual_volatility": [_sqrt_or_nan(value) for value in actual],
                    "error_variance": actual - forecast,
                    "ratio": ratio,
                    "finite": np.isfinite(forecast) & np.isfinite(actual),
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=list(VOL_FORECAST_COLUMNS))
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["open_time", "model"], kind="stable").reset_index(drop=True)
    return combined[list(VOL_FORECAST_COLUMNS)]


def _report(
    *,
    request: VolForecastRequest,
    parent: DatasetMeta,
    series: _Series,
    runs: list[_WalkForward],
    scores: list[VolModelScore],
    comparisons: _Comparisons,
    quantiles: dict[str, dict[str, float]],
    forward_paths: list[VolForwardPath],
    schedule_indices: list[int],
    step_seconds: float,
    step_source: str,
    adf_statistic: float,
    adf_pvalue: float,
    conditioning_meta: DatasetMeta | None,
) -> VolForecastReport:
    first, last = schedule_indices[0], schedule_indices[-1]
    warnings: list[str] = []
    if series.derived:
        warnings.append(
            "Log returns were derived from the close column as diff(log(close))."
        )
    if series.exog is not None:
        if series.exog_rows_dropped:
            warnings.append(
                f"{series.exog_rows_dropped} bars have no regime posterior to condition on "
                "(Module D's warm-up), so the hybrids fall back to the linear forecast "
                "there rather than being trained on a placeholder."
            )
        warnings.append(
            "Conditioning columns are lagged before use, so a regime known only at the "
            "close of a bar cannot inform that same bar's forecast."
        )
    unscored = [score.model.value for score in scores if score.unscored_reason]
    if unscored:
        warnings.append(
            f"{', '.join(unscored)} could not be scored — see the metrics table for why."
        )
    # A pair that could not be tested is stated with the count, beside the ones that were.
    # The alternative — a shorter comparison table and no note — reads as "these models were
    # compared and found equal", which is the opposite of what happened.
    warnings.extend(comparisons.gaps)
    if any(score.mz_r2 is not None and score.mz_r2 < 0.1 for score in scores):
        warnings.append(
            "A Mincer–Zarnowitz R² near zero is the expected result against a "
            f"{ACTUAL_PROXY} proxy, not evidence of a broken model: the proxy is an "
            "unbiased but very noisy estimate of the conditional variance, so most of its "
            "variation is unpredictable by construction. Rank on MQLIKE, which is robust "
            "to the proxy; read the R² as a calibration diagnostic only."
        )
    for path in forward_paths:
        if any(math.isnan(value) for value in path.variance):
            warnings.append(
                f"{path.model.value}'s forward path contains non-finite steps; the fan "
                "chart will have gaps rather than interpolated values."
            )
    if any(run.untrained for run in runs):
        warnings.append(
            "A hybrid fell back to its linear leg for want of enough usable history to "
            "train the correction. Its row in the table is the linear model's result."
        )
    if any(run.clamped for run in runs):
        warnings.append(
            f"A hybrid's learned correction hit its clamp on "
            f"{sum(run.clamped for run in runs)} steps. A correction pinned at the boundary "
            "is an extrapolation, not a confident view."
        )
    failures = [failure for run in runs for failure in run.failures]
    if failures:
        # The per-run list is capped at five, so it cannot be used as a count. The scores
        # carry the exact tally of steps that produced no usable forecast, which is what the
        # sentence is about: a run that failed 200 times must not read as having failed five.
        unusable = sum(score.n_nan + score.n_inf for score in scores)
        warnings.append(
            f"{unusable} walk-forward steps produced no usable forecast and were recorded as "
            f"non-finite rather than scored. First: {failures[0]}"
        )

    # ``series.times`` is already indexed the way ``schedule_indices`` is — both are sliced
    # from the same span of the frame, so position ``j`` of one is position ``j`` of the
    # other. Subtracting ``series.offset`` here as well would date every reported timestamp
    # one bar early whenever returns had to be derived from close (which is always, unless
    # the frame already carries a return column), putting the report's scored span a bar
    # behind the timestamps written into the frame it describes. ``_output_frame`` adds the
    # offset to the run's indices and subtracts it again, which is why the frame was right
    # and only the report was wrong — a difference invisible to every check that reads one
    # of the two alone.
    scored_times = series.times[list(schedule_indices)]
    return VolForecastReport(
        dataset_id=parent.dataset_id,
        dataset_name=parent.name,
        price_source=request.price_source,
        return_column=series.column,
        derived_returns=series.derived,
        distribution=request.distribution,
        mean_model=request.mean_model,
        p=request.p,
        o=request.o,
        q=request.q,
        lags=request.lags,
        walk_forward=request.walk_forward,
        forecast_horizon=request.forecast_horizon,
        rows=parent.row_count,
        return_rows=int(series.returns.size),
        first_scored_index=int(first),
        last_scored_index=int(last),
        first_scored_time=scored_times[0],
        last_scored_time=scored_times[-1],
        scale_factor=models.SCALE,
        adf_statistic=adf_statistic,
        adf_pvalue=adf_pvalue,
        stationary=adf_pvalue <= STATIONARITY_ALPHA,
        actual_proxy=ACTUAL_PROXY,
        scores=scores,
        comparisons=comparisons.rows,
        error_quantiles=quantiles,
        forward_paths=forward_paths,
        forward_step_seconds=step_seconds,
        forward_step_source=step_source,
        conditioning_dataset_id=conditioning_meta.dataset_id if conditioning_meta else None,
        conditioning_dataset_name=conditioning_meta.name if conditioning_meta else None,
        conditioning_columns=list(series.exog_columns),
        warnings=warnings,
    )


def _isoformat(value: datetime | None) -> str | None:
    """A timestamp as an ISO-8601 string, for the lineage row.

    The detail dict is stored as JSON **text** by ``Catalog.add_lineage``, which calls
    ``json.dumps`` with no ``default=`` fallback — and the standard library's encoder has no
    datetime branch, so a ``datetime`` left in the dict is a ``TypeError`` raised from
    inside the store. It would arrive after the frame has been written and its catalog row
    inserted, so the run would report a failure having already left an artifact behind.

    The report itself carries the object; this is only how it is written down.
    """
    return None if value is None else value.isoformat()


def _lineage_detail(
    request: VolForecastRequest,
    report: VolForecastReport,
    conditioning_meta: DatasetMeta | None,
) -> dict[str, Any]:
    """What goes into the lineage row for one forecast run.

    ``parameters_used`` is the whole request. A distribution or a lag order left out is a
    run that cannot be reproduced, and the difference between a research artifact and an
    unexplained column of numbers is exactly whether the knobs were written down.
    """
    return {
        "module_source": MODULE_SOURCE,
        "parameters_used": request.model_dump(mode="json"),
        "resolved_series": {
            "return_column": report.return_column,
            "derived": report.derived_returns,
            "rows": report.return_rows,
        },
        "walk_forward": {
            "train_window": request.walk_forward.train_window,
            "test_window": request.walk_forward.test_window,
            "step_size": request.walk_forward.step_size,
            "n_scored": request.walk_forward.n_scored,
            "first_scored_time": _isoformat(report.first_scored_time),
            "last_scored_time": _isoformat(report.last_scored_time),
        },
        "model_scores": {
            score.model.value: {
                "family": score.family,
                **{
                    metric.value: getattr(score, VOL_METRIC_FIELDS[metric])
                    for metric in VOL_SCALAR_METRICS
                },
                # `n_steps` / `n_finite` rather than a second key called `n_scored`. The
                # walk-forward block above already uses `n_scored` for the number of steps
                # the *protocol* produces, and that number is the same for every model; a
                # model-level `n_scored` meaning "steps that were finite" would sit under the
                # same name one level down, so a reader comparing the two would conclude the
                # model had converged on every step. These are the field names the report
                # itself carries on `VolModelScore`.
                "n_steps": score.n_steps,
                "n_finite": score.n_finite,
                "n_nan": score.n_nan,
                "n_inf": score.n_inf,
            }
            for score in report.scores
        },
        "diebold_mariano": [
            {
                "a": entry.model_a.value,
                "b": entry.model_b.value,
                "statistic": entry.statistic,
                "p_value": entry.p_value,
                "n": entry.n_effective,
                "better": entry.better.value if entry.better else None,
            }
            for entry in report.comparisons
        ],
        "stationarity": {
            "adf_statistic": report.adf_statistic,
            "adf_pvalue": report.adf_pvalue,
            "stationary": report.stationary,
            "alpha": STATIONARITY_ALPHA,
        },
        "actual_proxy": report.actual_proxy,
        "conditioning_dataset_id": conditioning_meta.dataset_id if conditioning_meta else None,
        "conditioning_columns": report.conditioning_columns,
    }


def _summary_message(payload: dict[str, Any]) -> str:
    """One line for the job list. Says what happened, not that something did."""
    report = payload.get("report") or {}
    dataset = payload.get("dataset")
    scores = report.get("scores") or []
    scored = [entry for entry in scores if entry.get("mqlike") is not None]
    best = min(scored, key=lambda entry: entry["mqlike"], default=None)
    headline = (
        f"best MQLIKE {best['mqlike']:.4f} ({best['model']})" if best else "no model scored"
    )
    where = f"saved as {dataset['name']}" if dataset else "preview only, nothing written"
    return f"{len(scores)} model(s) over {report.get('return_rows', 0)} bars · {headline} · {where}"
