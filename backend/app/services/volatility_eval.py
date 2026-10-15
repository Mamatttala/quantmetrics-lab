"""Module E — the scoring functions, with no model and no I/O in sight.

Every function here takes two float arrays and returns numbers. That is deliberate and it
is what makes the module testable against a calculator: the brief's own verification
requirement is a hand-computed MQLIKE, and a loss function buried inside a fitting loop
cannot be checked that way. Nothing here reads a dataset, imports arch, or knows what a
volatility model is.

**Everything is scored in variance, not in volatility.** MQLIKE, RMSE and MAE all take
``σ²``. Mixing the two — a QLIKE on variance beside an RMSE on standard deviation — produces
a table whose columns are not comparable and whose "best" markings differ by model for no
reason a reader can see. The report says which units it is in.

**The proxy is imperfect, and that is what QLIKE is for.** The realised variance is
approximated by a squared return, which is an unbiased but extremely noisy estimate of the
true conditional variance. Patton (2011) showed that of the common loss functions only
QLIKE and MSE are *robust* to this: they rank models correctly even though the proxy is
noisy. That is why MQLIKE is the headline metric here and why the Mincer–Zarnowitz R² is
reported as a diagnostic rather than used to rank anything — against a squared-return proxy
it is near zero even for a correctly specified model, because most of the variation in the
proxy is noise the model could not have predicted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: A forecast of exactly zero is *dropped* rather than floored. Flooring it is the tempting
#: move — QLIKE divides by the forecast, so zero is an infinity — but the floor does not
#: remove the infinity, it just moves it: lifting 0 to ``1e-300`` makes the ratio ``1e300``
#: and the step's loss ``1e295``, which dominates the mean for every other bar in the run.
#: A run whose headline MQLIKE is ``4.7e293`` is not a run with one bad step in it; it is a
#: run where one diverged recursion consumed the statistic.
#:
#: A variance forecast of zero is not a prediction that the market will be perfectly calm.
#: It is an optimiser at a boundary or a recursion that underflowed, and there is no honest
#: loss to assign it: the realised variance is positive by definition and no finite loss
#: expresses "predicted exactly zero". So the pair is dropped, which makes it a *counted*
#: step rather than a scored one — the same treatment the engine gives a step that failed to
#: fit, and the number the report prints as the scored-step count.
#:
#: Applied to all the metrics rather than only the ratio-based ones, so a model's RMSE and
#: its MQLIKE are read off the same pairs. Scoring RMSE on a step where QLIKE had none gives
#: a table whose columns are computed over different samples, with nothing on the page saying
#: so — the exact inconsistency the module docstring's "scored in variance" note exists to
#: prevent.
MIN_FORECAST_VARIANCE = 0.0

#: The fewest pairs the **statistics** are defined on. Diebold–Mariano estimates a long-run
#: variance and Mincer–Zarnowitz fits a slope; neither exists on a single observation — the
#: variance of one number is zero by construction, and DM's zero-variance branch would turn
#: that into a decisive difference from a sample that cannot support one.
#:
#: The mean-based losses (:func:`mqlike`, :func:`rmse`, :func:`mae`) are exact arithmetic at
#: any sample size and are guarded only against an empty one, so a hand-worked property such
#: as QLIKE's asymmetry can be demonstrated on a single pair. Where a *reported* metric needs
#: a floor, that floor belongs to the reporting decision and is the service's
#: ``MIN_SCORED_STEPS`` — 30 scored steps before any of these numbers is shown.
MIN_PAIRS = 2

#: The magnitude the Diebold–Mariano statistic is reported at when the loss differential has
#: no estimated variance at all — one model's per-step loss is the other's plus a constant.
#: The standardised statistic is unbounded there (the standard error is exactly zero), so the
#: true value is an infinity — and an infinity is not representable in the response, because
#: FastAPI renders JSON with ``allow_nan=False`` and would raise on the way out of the jobs
#: endpoint. The direction is carried by the sign and the magnitude is a stated convention;
#: the p-value is exactly zero, which is the limit the unbounded statistic takes.
DEGENERATE_STATISTIC = 1e12


@dataclass(frozen=True)
class MZResult:
    """The Mincer–Zarnowitz regression ``actual = intercept + slope · forecast``.

    A well-calibrated forecast has ``intercept = 0`` and ``slope = 1``. In practice, on a
    squared-return proxy, the slope is usually well below one and the R² is near zero — the
    regression is dominated by proxy noise, and that is a fact about the proxy rather than
    about the model. Reported, and read with that in mind.
    """

    intercept: float
    slope: float
    r2: float
    n: int


@dataclass(frozen=True)
class DMResult:
    """One pairwise Diebold–Mariano test.

    ``statistic`` is the Harvey–Laybourne–Newbold-corrected standardised mean loss
    differential; ``p_value`` is two-sided against the standard normal. ``n_effective`` is
    the number of loss-differential pairs the variance was estimated from, which is the
    number the test's power actually depends on.
    """

    statistic: float
    p_value: float
    n_effective: int
    mean_differential: float


def _aligned(actual: np.ndarray, forecast: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The scorable pairs of ``(actual, forecast)``.

    Two kinds of pair are dropped and neither is filled in any way. Filling is the one
    option that is definitely wrong here: it would make a model that failed to converge on
    half its steps score the same as one that converged on all of them.

    * **Non-finite.** A NaN is a step the model produced nothing for; an inf is a step it
      diverged on. ``mqlike``'s ``log`` and ``rmse``'s square both propagate them into the
      mean, so a single failure would erase the whole run's statistic.
    * **Non-positive.** See ``MIN_FORECAST_VARIANCE``: a zero or negative variance is a
      boundary artefact, and there is no loss value that expresses it honestly.
    """
    actual = np.asarray(actual, dtype=float).ravel()
    forecast = np.asarray(forecast, dtype=float).ravel()
    if actual.shape != forecast.shape:
        raise ValueError(
            f"actual and forecast must be the same length; got {actual.size} and "
            f"{forecast.size}."
        )
    keep = np.isfinite(actual) & np.isfinite(forecast) & (forecast > MIN_FORECAST_VARIANCE)
    return actual[keep], forecast[keep]


def _scaled_errors(actual: np.ndarray, forecast: np.ndarray) -> tuple[float, np.ndarray]:
    """The error in units of the largest magnitude present, plus that magnitude.

    ``(actual - forecast) ** 2`` overflows to ``inf`` when the forecast is huge: a variance
    of ``8.3e213`` — which a diverged arch recursion really does produce — squares to more
    than the largest double, and the ``RuntimeWarning`` numpy prints is the only sign that
    the RMSE it returns is not a root of anything.

    Dividing both series by the larger magnitude first keeps every intermediate bounded by
    two, so the square cannot overflow, and the scale is multiplied back at the end. The
    result is the same number the direct formula gives whenever the direct formula is
    representable; where it is not, this returns the finite value instead of ``inf`` — and
    ``inf`` cannot leave this process in any case, since the API renders JSON with
    ``allow_nan=False``.
    """
    scale = float(max(np.max(np.abs(actual)), np.max(np.abs(forecast))))
    if scale == 0.0:
        return 0.0, np.zeros_like(actual)
    return scale, actual / scale - forecast / scale


def mqlike(actual: np.ndarray, forecast: np.ndarray) -> float:
    """The QLIKE loss in its variance form.

    ``mean( σ²/σ̂² − log(σ²/σ̂²) − 1 )``

    Written against the **ratio** rather than as ``log σ̂² + σ²/σ̂² − 1``. The two differ by
    ``mean(log σ²)``, which is a constant across models on the same data — so they rank
    identically — but only this form is zero at a perfect forecast, which makes it the one
    that can be checked against a calculator and the one whose magnitude means something.

    This is Patton's robust form. It is asymmetric in the right direction: under-predicting
    the variance is penalised far more heavily than over-predicting it, which is the
    behaviour a risk model should be scored on.
    """
    actual, forecast = _aligned(actual, forecast)
    if actual.size == 0:
        raise ValueError("QLIKE needs at least one finite pair; got 0.")
    ratio = actual / forecast
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Root mean squared error, in **variance** units."""
    actual, forecast = _aligned(actual, forecast)
    if actual.size == 0:
        raise ValueError("RMSE needs at least one finite pair; got 0.")
    scale, errors = _scaled_errors(actual, forecast)
    return float(scale * np.sqrt(np.mean(errors**2)))


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Mean absolute error, in **variance** units."""
    actual, forecast = _aligned(actual, forecast)
    if actual.size == 0:
        raise ValueError("MAE needs at least one finite pair; got 0.")
    scale, errors = _scaled_errors(actual, forecast)
    return float(scale * np.mean(np.abs(errors)))


def mincer_zarnowitz(actual: np.ndarray, forecast: np.ndarray) -> MZResult:
    """Regress the realised variance on the forecast, and report the calibration.

    Ordinary least squares with an intercept, solved in closed form. A constant forecast
    makes the slope undefined — every candidate slope fits equally badly, and least squares
    would return whatever the numerical solver's rounding produced — so that case is
    reported as a zero slope with a zero R², which is the honest reading of "this forecast
    carries no information about the level".
    """
    actual, forecast = _aligned(actual, forecast)
    n = int(actual.size)
    if n < MIN_PAIRS:
        raise ValueError(f"Mincer–Zarnowitz needs at least {MIN_PAIRS} pairs; got {n}.")

    mean_forecast = float(np.mean(forecast))
    mean_actual = float(np.mean(actual))
    centred = forecast - mean_forecast
    denominator = float(np.dot(centred, centred))
    if denominator <= 0.0 or not math.isfinite(denominator):
        return MZResult(intercept=mean_actual, slope=0.0, r2=0.0, n=n)

    slope = float(np.dot(centred, actual - mean_actual) / denominator)
    intercept = mean_actual - slope * mean_forecast

    residual = actual - (intercept + slope * forecast)
    total = float(np.dot(actual - mean_actual, actual - mean_actual))
    r2 = 0.0 if total <= 0.0 else float(1.0 - float(np.dot(residual, residual)) / total)
    # A slope fitted on a finite sample can overshoot the variance decomposition by a
    # hair, giving an R² of -1e-16. Clamped at zero rather than reported as a negative
    # coefficient of determination, which no reader should have to reason about.
    return MZResult(
        intercept=intercept,
        slope=slope,
        r2=max(0.0, min(1.0, r2)),
        n=n,
    )


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    *,
    horizon: int = 1,
) -> DMResult:
    """Diebold–Mariano: is model A's loss different from model B's, beyond noise?

    The loss differential is ``d_t = L_A,t − L_B,t``. Under the null of equal predictive
    accuracy its mean is zero, and the test standardises that mean by a **heteroskedasticity-
    and autocorrelation-consistent** variance estimate: a Newey–West sum with a Bartlett
    kernel truncated at ``horizon − 1`` lags. For a one-step forecast that is just the
    sample variance of ``d``; for an ``h``-step forecast, overlapping forecast windows make
    the ``d`` series autocorrelated up to ``h − 1`` and ignoring that understates the
    variance and manufactures significance.

    Then the Harvey–Laybourne–Newbold small-sample correction, which multiplies the
    statistic by ``sqrt((N + 1 − 2h + h(h−1)/N) / N)``. It matters most exactly where these
    tests are usually run — a few hundred steps.

    statsmodels has no DM test, so this is written out. A version without the HAC variance
    and the HLN factor is two lines shorter and wrong in the direction that flatters
    whichever model the user hoped would win.

    ``mean_differential`` is ``mean(L_A) − mean(L_B)``. This function returns no verdict:
    ``_compare`` in the service reads the sign of that differential to name the better model
    and thresholds the p-value to decide whether the difference is worth naming at all.
    """
    a = np.asarray(loss_a, dtype=float).ravel()
    b = np.asarray(loss_b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(
            f"the two loss series must be the same length; got {a.size} and {b.size}."
        )
    keep = np.isfinite(a) & np.isfinite(b)
    d = a[keep] - b[keep]
    n = int(d.size)
    if n < MIN_PAIRS:
        raise ValueError(f"Diebold–Mariano needs at least {MIN_PAIRS} pairs; got {n}.")

    mean_d = float(np.mean(d))
    centred = d - mean_d

    # Bartlett-kernel long-run variance: gamma_0 + 2 * sum_{k=1}^{h-1} (1 - k/h) * gamma_k.
    # The truncation is at horizon - 1 because that is the overlap the forecast step
    # induces; nothing beyond it is autocorrelation the model created.
    gamma_0 = float(np.dot(centred, centred)) / n
    long_run = gamma_0
    for lag in range(1, max(1, horizon)):
        if lag >= n:
            break
        gamma_k = float(np.dot(centred[lag:], centred[:-lag])) / n
        long_run += 2.0 * (1.0 - lag / horizon) * gamma_k

    if long_run <= 0.0 or not math.isfinite(long_run):
        # The long-run variance is estimated at zero or below, which happens when one model's
        # per-step loss is the other's plus a constant. Two cases hide here and they read in
        # opposite directions, so they are separated rather than collapsed:
        #
        # * ``mean_d == 0`` — the two models scored identically on every step. There is no
        #   evidence of a difference and nothing to standardise: a zero statistic, a p-value
        #   of one. This is the branch the "identical losses" test pins.
        # * ``mean_d != 0`` — every step favoured the same model by the same amount, so the
        #   standard error is zero and the statistic is unbounded. Reporting "no evidence"
        #   here would invert the finding: the report turns ``significant`` and the sign of
        #   the differential into a sentence, and the sentence would read "no significant
        #   difference (B lower, on noise)" about a difference that has no noise in it at all.
        if mean_d == 0.0:
            return DMResult(statistic=0.0, p_value=1.0, n_effective=n, mean_differential=mean_d)
        return DMResult(
            statistic=math.copysign(DEGENERATE_STATISTIC, mean_d),
            p_value=0.0,
            n_effective=n,
            mean_differential=mean_d,
        )

    statistic = mean_d / math.sqrt(long_run / n)
    correction = math.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    statistic *= correction
    p_value = math.erfc(abs(statistic) / math.sqrt(2.0))

    return DMResult(
        statistic=float(statistic),
        p_value=float(p_value),
        n_effective=n,
        mean_differential=mean_d,
    )


def qlike_loss_series(actual: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """The per-step QLIKE loss, for feeding :func:`diebold_mariano`.

    The mean of this is exactly :func:`mqlike`. It exists so the two cannot disagree: a DM
    test run on a differently-defined loss than the one in the metrics table would answer a
    question nobody asked, and the discrepancy would be invisible because both are called
    "QLIKE".
    """
    actual, forecast = _aligned(actual, forecast)
    ratio = actual / forecast
    return ratio - np.log(ratio) - 1.0


def forecast_error_quantiles(
    ratios: np.ndarray,
    quantiles: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95),
) -> dict[str, float]:
    """Empirical quantiles of the walk-forward error ratio ``actual / forecast``.

    These are the fan chart's bands. They are **empirical** — read off the model's own
    realised walk-forward errors — and the chart says so, because the alternative is to
    derive bands from the fitted conditional distribution, which assumes the model is
    correctly specified. That assumption is precisely what the walk-forward exists to
    question, so using it to draw the uncertainty around the forecast would answer the
    question with the hypothesis.

    A ratio of 1.0 means the forecast was exactly right; the 0.95 quantile is the factor by
    which the model's own history says it typically under-predicts.

    The quantiles are read with ``np.quantile``'s default **linear interpolation**, so a band
    edge may fall between two observed ratios. The method is named here rather than left to
    the default because it is what decides the value on a small sample — the nearest-rank
    reading of ``{0.5 … 1.5}`` is the largest observation, 1.5, where linear interpolation
    gives 1.45 — and a band edge that moves because the quantile convention changed, with
    nothing else about the run different, is exactly the kind of silent change this module
    states its conventions to prevent.
    """
    values = np.asarray(ratios, dtype=float).ravel()
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return {}
    return {
        f"{q:.2f}": float(np.quantile(values, q, method="linear")) for q in quantiles
    }


#: The fan chart's band edges, as quantile keys, paired with a label. Two symmetric bands
#: rather than a single interval: one interval hides the asymmetry, and the asymmetry is
#: the interesting part — a volatility forecast is wrong much further to the upside than to
#: the downside, and a symmetric band would say the opposite.
FAN_BANDS: tuple[tuple[str, str, str], ...] = (
    ("0.05", "0.95", "5–95%"),
    ("0.25", "0.75", "25–75%"),
)


def fan_bands(quantiles: dict[str, float]) -> list[tuple[str, float, float]]:
    """The ``(label, low, high)`` bands the chart draws, from a quantile dict.

    Returns an empty list when the quantiles are unavailable, so a chart drawn from an
    unscored model shows a single line rather than a band invented from nothing.
    """
    bands: list[tuple[str, float, float]] = []
    for low_key, high_key, label in FAN_BANDS:
        low = quantiles.get(low_key)
        high = quantiles.get(high_key)
        if low is None or high is None:
            continue
        bands.append((label, low, high))
    return bands
