"""The scoring functions, checked against arithmetic worked out by hand.

This file exists because of one requirement in the specification — "hand-calculated MQLIKE
test" — and it is the right requirement to have. A loss function is the only part of a
forecasting module whose correctness cannot be established by looking at a chart: a QLIKE
that is off by a constant, or that silently uses the standard-deviation form instead of the
variance form, ranks slightly differently and produces a report that looks entirely
reasonable. The only way to know which function is implemented is to evaluate a case whose
answer is known.

The hand-worked values are written out longhand in the tests rather than pre-computed,
so a reader can check the arithmetic against the comment without running anything.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.services import volatility_eval as evaluation

# ------------------------------------------------------------------ MQLIKE


def test_mqlike_matches_a_hand_computed_value() -> None:
    """Two steps, worked out by hand.

    ``QLIKE = mean( σ²/σ̂² − log(σ²/σ̂²) − 1 )``

    Step 1: actual 1.0, forecast 1.0 → ratio 1.0 → ``1 − log 1 − 1 = 0``
    Step 2: actual 4.0, forecast 2.0 → ratio 2.0 → ``2 − log 2 − 1 = 1 − log 2``

    Mean of ``{0, 1 − log 2}`` is ``(1 − log 2) / 2``.

    ``log 2 = 0.69314718055994530942…`` so the mean is ``0.15342640972002734529…``.

    The value is asserted twice: once against the closed form, which documents the
    derivation, and once against the decimal literal, which would catch the closed form
    being changed to match a changed implementation.
    """
    actual = np.array([1.0, 4.0])
    forecast = np.array([1.0, 2.0])

    expected_closed_form = (0.0 + (2.0 - math.log(2.0) - 1.0)) / 2.0
    assert evaluation.mqlike(actual, forecast) == pytest.approx(expected_closed_form)
    assert evaluation.mqlike(actual, forecast) == pytest.approx(0.15342640972002735, abs=1e-15)


def test_mqlike_is_zero_at_a_perfect_forecast() -> None:
    """The property the variance form has and the ``log σ̂² + σ²/σ̂²`` form does not.

    It is the reason for choosing this form over the algebraically equivalent one: a
    reader can tell at a glance whether a number is good, and a test can assert a value
    rather than a difference between two runs.
    """
    actual = np.array([0.5, 1.0, 2.0, 7.25])
    assert evaluation.mqlike(actual, actual) == pytest.approx(0.0, abs=1e-15)


def test_mqlike_punishes_under_prediction_more_than_over_prediction() -> None:
    """Asymmetry, which is the whole reason a risk model is scored with QLIKE.

    Predicting half the true variance gives a ratio of 2 and a loss of ``1 − log 2``;
    predicting double gives a ratio of ½ and a loss of ``½ + log 2 − 1``. The second is
    smaller — under-predicting risk is the expensive error, and a metric that treated the
    two symmetrically would let a model that habitually under-forecasts volatility rank
    alongside one that does not.

    One pair is enough to state the property, and this asserts it on one: the mean losses
    are defined at any sample size, so the asymmetry is checked against the arithmetic
    rather than against a sample. The floor on how few steps may be *reported* is the
    service's, not this module's.
    """
    actual = np.array([1.0])
    under = evaluation.mqlike(actual, np.array([0.5]))
    over = evaluation.mqlike(actual, np.array([2.0]))
    assert under == pytest.approx(1.0 - math.log(2.0))
    assert over == pytest.approx(0.5 + math.log(2.0) - 1.0)
    assert under > over


def test_mqlike_drops_non_finite_pairs_rather_than_filling_them() -> None:
    """A model that failed on half its steps must not score like one that did not.

    Filling a NaN with the previous forecast, or with the mean, would produce a number for
    a step where the model produced nothing — and a run that converged on 60% of its steps
    would report a metric indistinguishable from one that converged on all of them.
    """
    actual = np.array([1.0, np.nan, 4.0])
    forecast = np.array([1.0, 2.0, 2.0])
    assert evaluation.mqlike(actual, forecast) == pytest.approx(
        evaluation.mqlike(np.array([1.0, 4.0]), np.array([1.0, 2.0]))
    )


def test_mqlike_refuses_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        evaluation.mqlike(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


def test_a_zero_forecast_is_dropped_rather_than_floored() -> None:
    """Flooring a zero forecast does not remove the infinity it creates — it moves it.

    Lifting ``0.0`` to ``1e-300`` makes the ratio ``1e300`` and that one step's loss ``1e295``,
    which is larger than every other step's loss put together by hundreds of orders of
    magnitude. The mean then describes the floor rather than the run. This is not
    hypothetical: this project's first end-to-end run reported an MQLIKE of ``4.7e293`` from a
    single diverged EGARCH step, and no number on that page meant anything afterwards.

    A variance forecast of exactly zero is an optimiser at a boundary or a recursion that
    underflowed, and no loss value expresses "predicted zero" against a realised variance that
    is positive by definition. So the pair is dropped — and the *service* counts it as a step
    that produced no forecast, which is where the user is told about it.
    """
    actual = np.array([1.0, 4.0, 9.0])
    forecast = np.array([1.0, 0.0, 3.0])

    assert evaluation.mqlike(actual, forecast) == pytest.approx(
        evaluation.mqlike(np.array([1.0, 9.0]), np.array([1.0, 3.0]))
    )
    assert evaluation.mqlike(actual, forecast) < 100.0


def test_every_metric_drops_the_same_pairs() -> None:
    """One alignment for the whole table, so its columns are comparable.

    A model's MQLIKE and its RMSE are read side by side. Scoring them on different samples —
    the ratio-based loss dropping a step the mean losses kept — puts two different sample sizes
    in one row with nothing on the page saying so.
    """
    actual = np.array([1.0, 4.0, 9.0, 16.0])
    forecast = np.array([1.0, 0.0, 3.0, np.nan])
    kept_actual = np.array([1.0, 9.0])
    kept_forecast = np.array([1.0, 3.0])

    assert evaluation.mqlike(actual, forecast) == pytest.approx(
        evaluation.mqlike(kept_actual, kept_forecast)
    )
    assert evaluation.rmse(actual, forecast) == pytest.approx(
        evaluation.rmse(kept_actual, kept_forecast)
    )
    assert evaluation.mae(actual, forecast) == pytest.approx(
        evaluation.mae(kept_actual, kept_forecast)
    )
    assert evaluation.mincer_zarnowitz(actual, forecast).n == 2


def test_the_error_metrics_stay_finite_for_a_forecast_too_large_to_square() -> None:
    """``8.3e213 ** 2`` is not a representable double, and the direct formula returns ``inf``.

    numpy says so with a ``RuntimeWarning`` nobody reads, and then hands back a root of
    nothing. An ``inf`` here cannot leave the process either: the API renders JSON with
    ``allow_nan=False`` and a browser's ``JSON.parse`` throws on ``Infinity``. Scaling by the
    largest magnitude before subtracting keeps every intermediate bounded by two.
    """
    actual = np.array([1e-5, 2e-5])
    forecast = np.array([8.3e213, 8.3e213])
    huge = 8.3e213

    assert evaluation.rmse(actual, forecast) == pytest.approx(huge, rel=1e-12)
    assert evaluation.mae(actual, forecast) == pytest.approx(huge, rel=1e-12)
    assert np.isfinite(evaluation.rmse(actual, forecast))

    # And the scaling is exact rather than approximate where the direct formula works: an
    # error series whose answer is known exactly comes back exactly. Both forecasts are
    # positive, which is not incidental — a negative variance forecast is dropped by the
    # alignment before either metric sees it.
    assert evaluation.rmse(np.array([3.0]), np.array([1.0])) == pytest.approx(2.0)
    assert evaluation.mae(np.array([3.0, 1.0]), np.array([1.0, 1.0])) == pytest.approx(1.0)


def test_the_error_metrics_are_zero_when_every_pair_is_identical() -> None:
    """The scale is zero there, and the scaled form must not divide by it."""
    actual = np.array([2.0, 3.0, 4.0])
    assert evaluation.rmse(actual, actual) == 0.0
    assert evaluation.mae(actual, actual) == 0.0


# ------------------------------------------------------------------ RMSE and MAE


def test_rmse_and_mae_match_hand_computed_values() -> None:
    """Errors of −1, +1, −2 → RMSE ``sqrt(6/3) = sqrt 2``, MAE ``4/3``."""
    actual = np.array([1.0, 2.0, 3.0])
    forecast = np.array([2.0, 1.0, 5.0])
    assert evaluation.rmse(actual, forecast) == pytest.approx(math.sqrt(2.0))
    assert evaluation.mae(actual, forecast) == pytest.approx(4.0 / 3.0)


# ------------------------------------------------------- Mincer–Zarnowitz


def test_mz_is_exact_for_a_perfect_forecast() -> None:
    """``actual = 0 + 1·forecast`` exactly, so the slope is 1, the intercept 0, R² 1."""
    actual = np.array([0.5, 1.0, 1.5, 2.0, 9.0])
    result = evaluation.mincer_zarnowitz(actual, actual)
    assert result.slope == pytest.approx(1.0)
    assert result.intercept == pytest.approx(0.0, abs=1e-12)
    assert result.r2 == pytest.approx(1.0)


def test_mz_reports_a_zero_slope_for_a_constant_forecast() -> None:
    """A forecast that never moves cannot be regressed on.

    Least squares would return whatever the solver's rounding produced for the slope, and
    a reader would take it for an estimate. A slope of zero with the intercept at the mean
    actual is the honest reading: this forecast carries no information about the level.
    """
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    forecast = np.full(4, 2.5)
    result = evaluation.mincer_zarnowitz(actual, forecast)
    assert result.slope == 0.0
    assert result.intercept == pytest.approx(2.5)
    assert result.r2 == 0.0


def test_mz_recovers_an_exact_affine_relationship() -> None:
    """``actual = 3 + 2·forecast`` over four points: slope 2, intercept 3, R² 1."""
    forecast = np.array([1.0, 2.0, 3.0, 4.0])
    actual = 3.0 + 2.0 * forecast
    result = evaluation.mincer_zarnowitz(actual, forecast)
    assert result.slope == pytest.approx(2.0)
    assert result.intercept == pytest.approx(3.0)
    assert result.r2 == pytest.approx(1.0)


def test_mz_r2_is_never_negative() -> None:
    """A fitted slope can overshoot the variance decomposition by a rounding hair.

    Clamped at zero: an R² of ``-1e-16`` is a number no reader should have to reason about,
    and reporting it invites the question of what a negative coefficient of determination
    means.
    """
    rng = np.random.default_rng(7)
    for _ in range(20):
        actual = rng.normal(1.0, 0.5, 12)
        forecast = rng.normal(1.0, 0.5, 12)
        assert evaluation.mincer_zarnowitz(actual, forecast).r2 >= 0.0


# ------------------------------------------------------- Diebold–Mariano


def test_dm_at_horizon_one_equals_the_paired_t_statistic() -> None:
    """The identity that makes the implementation checkable.

    With ``h = 1`` the Newey–West sum has no autocovariance terms and the
    Harvey–Laybourne–Newbold correction is exactly ``sqrt((N−1)/N)``, which turns the
    statistic into ``d̄·sqrt(N)·sqrt(N−1)/sqrt(Σ(d−d̄)²)`` — algebraically the paired
    t-statistic ``d̄ / (s/√N)`` with ``s² = Σ(d−d̄)²/(N−1)``.

    So the whole test is: implement the reference with a different algorithm and check the
    two agree. ``scipy.stats.ttest_1samp`` is that reference.
    """
    from scipy import stats

    rng = np.random.default_rng(11)
    loss_a = rng.gamma(2.0, 0.5, 200)
    loss_b = loss_a + rng.normal(0.02, 0.1, 200)

    result = evaluation.diebold_mariano(loss_a, loss_b, horizon=1)
    reference = stats.ttest_1samp(loss_a - loss_b, 0.0)

    assert result.n_effective == 200
    assert result.statistic == pytest.approx(float(reference.statistic), rel=1e-12)
    # The statistic is the identity being tested; the p-value deliberately is *not* the
    # t-distribution's. Diebold–Mariano is asymptotic and standardises against the normal,
    # so the two p-values differ in the third decimal at N=200 — agreeing here would mean
    # this module had quietly become a t-test.
    assert result.p_value != pytest.approx(float(reference.pvalue), rel=1e-6)


def test_dm_p_value_is_two_sided_against_the_normal() -> None:
    """``erfc(|t|/√2)`` is exactly ``2·(1 − Φ(|t|))``."""
    rng = np.random.default_rng(13)
    loss_a = rng.gamma(2.0, 0.5, 150)
    loss_b = loss_a + rng.normal(0.05, 0.2, 150)
    result = evaluation.diebold_mariano(loss_a, loss_b)
    from scipy import stats

    assert result.p_value == pytest.approx(
        2.0 * (1.0 - float(stats.norm.cdf(abs(result.statistic)))), abs=1e-12
    )


def test_dm_is_unchanged_by_a_constant_added_to_both_loss_series() -> None:
    """The differential is what is tested, so anything common to both cancels.

    A test that moved when both models were scored on the same shifted units would be
    testing the level of the loss rather than the difference between the models.
    """
    rng = np.random.default_rng(17)
    loss_a = rng.gamma(2.0, 0.5, 120)
    loss_b = rng.gamma(2.0, 0.5, 120)
    base = evaluation.diebold_mariano(loss_a, loss_b)
    shifted = evaluation.diebold_mariano(loss_a + 100.0, loss_b + 100.0)
    assert shifted.statistic == pytest.approx(base.statistic, rel=1e-12)
    assert shifted.p_value == pytest.approx(base.p_value, rel=1e-12)


def test_dm_per_step_losses_match_the_headline_mqlike() -> None:
    """A DM test run on a differently-defined loss than the table's would answer a
    question nobody asked, and both are called "QLIKE" so the discrepancy would be
    invisible."""
    rng = np.random.default_rng(19)
    actual = rng.gamma(3.0, 1.0, 100)
    forecast = actual * rng.uniform(0.5, 2.0, 100)
    series = evaluation.qlike_loss_series(actual, forecast)
    assert float(np.mean(series)) == pytest.approx(evaluation.mqlike(actual, forecast))


def test_dm_reports_no_evidence_when_the_models_scored_identically() -> None:
    """Two identical loss series have no differential to standardise.

    A zero statistic with a p-value of 1 is the reading — not an error, and not a
    division by a variance of zero.
    """
    losses = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    result = evaluation.diebold_mariano(losses, losses)
    assert result.statistic == 0.0
    assert result.p_value == 1.0
    assert result.mean_differential == 0.0


def test_dm_reports_a_decisive_difference_when_the_differential_has_no_variance() -> None:
    """A constant non-zero loss differential is the opposite of "no evidence", and the two
    zero-variance cases are kept apart.

    When one model's loss is the other's plus a constant, the differential has no variance at
    all, so the standard error is zero and the standardised statistic is unbounded. Two very
    different situations produce that: identical losses (differential identically zero) and a
    constant *non-zero* differential. Collapsing them into a single "no evidence" branch would
    print "no significant difference (B lower, on noise)" beside a difference that contains no
    noise — inverting what the report says about the pair.

    The magnitude is finite rather than infinite because ``inf`` cannot leave the API: FastAPI
    renders JSON with ``allow_nan=False`` and would raise on the way out of the jobs endpoint.

    The two loss series are built from dyadic values so the differential is *exactly* constant.
    ``linspace`` endpoints are not binary-exact: ``a − (a + 0.25)`` then carries a jitter of
    order ``1e-17``, whose variance is tiny but positive, which takes the ordinary branch and
    reports a statistic near ``1e17``. That case is a real one and it reads correctly — a huge
    statistic and a zero p-value — but it is not the branch this test is about.
    """
    loss_a = np.arange(60, dtype=float) / 2.0
    constant = np.full(60, 0.25)

    worse_b = evaluation.diebold_mariano(loss_a, loss_a + constant)
    assert worse_b.mean_differential == pytest.approx(-0.25)
    assert worse_b.statistic == pytest.approx(-evaluation.DEGENERATE_STATISTIC)
    assert worse_b.p_value == 0.0
    assert np.isfinite(worse_b.statistic)

    # The same difference the other way round: the sign follows the differential, which is
    # what `_compare` turns into the name of the better model.
    worse_a = evaluation.diebold_mariano(loss_a + constant, loss_a)
    assert worse_a.statistic == pytest.approx(evaluation.DEGENERATE_STATISTIC)
    assert worse_a.p_value == 0.0


def test_dm_detects_a_deliberate_difference() -> None:
    """A model whose loss is uniformly 0.5 higher must come out significantly worse.

    The direction matters as much as the magnitude: the sign of the statistic has to say
    *which* model lost, because the report turns it into a sentence.

    The differential is exactly constant here, so this is the zero-variance case — the
    statistic arrives at ``DEGENERATE_STATISTIC`` rather than at a t-like value — and what is
    asserted is the sign and the p-value, not the magnitude.
    """
    rng = np.random.default_rng(23)
    loss_good = rng.gamma(2.0, 0.5, 300)
    loss_bad = loss_good + 0.5
    result = evaluation.diebold_mariano(loss_good, loss_bad)
    assert result.mean_differential == pytest.approx(-0.5)
    assert result.statistic < 0.0
    assert result.p_value < 1e-9


# ------------------------------------------------------- forecast error quantiles


def test_forecast_error_quantiles_are_read_off_the_ratios() -> None:
    """The fan chart's bands are empirical, and this is what that means.

    A ratio of 1 is an exact forecast. Over eleven ratios from 0.5 to 1.5 the median is the
    sixth point, exactly 1.0, and the 0.95 quantile is read with linear interpolation — the
    convention ``forecast_error_quantiles`` documents — which puts it half a step past the
    tenth: ``1.4 + 0.5 · 0.1 = 1.45``. Bands derived from the fitted distribution instead
    would assume the model is correctly specified — the very thing the walk-forward exists
    to question.
    """
    ratios = np.linspace(0.5, 1.5, 11)
    quantiles = evaluation.forecast_error_quantiles(ratios, (0.5, 0.95))
    assert quantiles["0.50"] == pytest.approx(1.0)
    # Interpolated, not the largest observation (1.5). Asserting the interpolated value is
    # deliberate: it pins the quantile method, so switching to a nearest-rank reading would
    # fail here rather than silently widening every band on the page.
    assert quantiles["0.95"] == pytest.approx(1.45)


def test_forecast_error_quantiles_ignore_non_positive_ratios() -> None:
    """A ratio of zero or below has no meaning as a scale factor, and ``log`` of it would
    poison the loss series this exists to summarise."""
    ratios = np.array([np.nan, np.inf, -1.0, 0.0, 1.0, 2.0])
    quantiles = evaluation.forecast_error_quantiles(ratios, (0.5,))
    assert quantiles["0.50"] == pytest.approx(1.5)


def test_fan_bands_come_from_the_quantiles_and_are_empty_without_them() -> None:
    bands = evaluation.fan_bands({"0.05": 0.4, "0.95": 3.0, "0.25": 0.7, "0.75": 1.8})
    assert bands == [("5–95%", 0.4, 3.0), ("25–75%", 0.7, 1.8)]
    assert evaluation.fan_bands({}) == []
