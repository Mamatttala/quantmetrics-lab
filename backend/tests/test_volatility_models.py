"""The estimators and the walk-forward loop.

The centre of this file is one test. Everything a volatility module does can be made to
look right — a chart of a fitted conditional variance is smooth and plausible whatever
produced it — and there is exactly one property that separates a forecast from a curve
drawn through data that already knew the answer:

    **A forecast for bar ``t`` must not change when bars after ``t`` are added to the
    series.**

That is checked directly, by running the whole loop twice — once over the full series and
once over its first 80% — and requiring the two to agree on every bar they both scored. A
model fitted once on the whole series and then "forecast" backwards fails it immediately, and
so does an expanding window, which is the harder case: it looks causal, and its parameters
for bar ``t`` were still estimated partly from data after ``t``.

The rest of the file pins what is easy to get quietly wrong: the arch distribution names, the
scale change, the simulation's seeding, and the lag alignment of the hybrid feature builder —
which is off by one in the obvious implementation and produces a model that is merely
*worse* rather than visibly broken.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.core.errors import ValidationFailure
from app.models.enums import VolDistribution, VolMeanModel, VolModel
from app.models.volatility import WalkForwardSpec
from app.services import volatility_models as models
from app.services.volatility import (
    FORECAST_REFUSAL_TAG,
    MAX_FORECAST_VARIANCE_MULTIPLE,
    MIN_FORECAST_VARIANCE_RATIO,
    _compare,
    _forecast_refusal,
    _score,
    _Series,
    _validate_schedule,
    _walk_forward,
    _WalkForward,
)


def garch_series(rows: int = 400, *, seed: int = 5, omega: float = 2e-6) -> np.ndarray:
    """A series with genuine volatility clustering, so a GARCH has something to find.

    A pure i.i.d. normal series would let the optimiser sit on a boundary — the ARCH and
    GARCH terms cannot be separated when there is no persistence — and every model would then
    forecast the unconditional variance. Every test here would pass and mean nothing.
    """
    rng = np.random.default_rng(seed)
    returns = np.zeros(rows)
    variance = np.full(rows, omega / (1.0 - 0.10 - 0.85))
    for t in range(1, rows):
        variance[t] = omega + 0.10 * returns[t - 1] ** 2 + 0.85 * variance[t - 1]
        returns[t] = rng.normal(0.0, np.sqrt(variance[t]))
    return returns


def build(name: VolModel, **overrides: object) -> models.ArchModel | models.HybridModel:
    """One estimator, constructed the way the service constructs it."""
    kwargs: dict[str, object] = {
        "distribution": VolDistribution.NORMAL,
        "mean_model": VolMeanModel.CONSTANT,
        "p": 1,
        "o": 1,
        "q": 1,
        "lags": 3,
    }
    kwargs.update(overrides)
    return models.build_model(name, **kwargs)  # type: ignore[arg-type]


# ------------------------------------------------------------------ causality


def prefix_forecasts(
    name: VolModel, series: np.ndarray, spec: WalkForwardSpec
) -> tuple[np.ndarray, np.ndarray]:
    """One-step forecasts for every scored bar, as ``(bar index, variance)``.

    Drives the engine's own ``_walk_forward`` rather than a reimplementation of it: a
    reimplementation could pass this file while the code the application runs leaked.
    """
    times = pd.date_range("2024-01-01", periods=series.size, freq="h", tz="UTC")
    wrapped = _Series(
        returns=series, offset=0, column="log_return", derived=False, times=times
    )
    run = _walk_forward(build(name), wrapped, spec)
    return run.indices, run.forecast


def shared_forecasts(
    name: VolModel, series: np.ndarray, spec: WalkForwardSpec, fraction: float
) -> list[tuple[int, float, float]]:
    """``(index, full_run_value, truncated_run_value)`` for every bar scored in both runs."""
    full_index, full_forecast = prefix_forecasts(name, series, spec)
    short_index, short_forecast = prefix_forecasts(
        name, series[: int(series.size * fraction)], spec
    )
    full_by_index = dict(zip(full_index.tolist(), full_forecast.tolist(), strict=True))
    short_by_index = dict(zip(short_index.tolist(), short_forecast.tolist(), strict=True))
    return [
        (index, full_by_index[index], short_by_index[index])
        for index in sorted(set(full_by_index) & set(short_by_index))
    ]


def assert_prefixes_agree(pairs: list[tuple[int, float, float]], *, minimum: int) -> None:
    """Every bar scored in both runs carries the same forecast — or is refused in both.

    A refused step is a legitimate outcome since the service gained ``_forecast_refusal``: a
    window whose fit leaves the parameter space produces no forecast, and the step is recorded
    as non-finite rather than scored. That refusal is a function of the training window's own
    content and of the value the optimiser returned, both of which are identical for a shared
    origin — so two runs must refuse the *same* bars, which is a causality statement like any
    other. Dropping the refused bars from the comparison instead would quietly shrink it.

    ``pytest.approx`` cannot express ``nan == nan`` — comparing them fails — which is why the
    both-refused case is written out. One run refusing where the other did not still falls
    through to the assertion and is reported as the difference it is.
    """
    compared = 0
    for index, full_value, short_value in pairs:
        if math.isnan(full_value) or math.isnan(short_value):
            assert math.isnan(full_value) and math.isnan(short_value), (
                f"bar {index} was unusable in one run and scored in the other"
            )
            continue
        compared += 1
        assert full_value == pytest.approx(short_value, rel=1e-6, abs=1e-12), (
            f"bar {index} was forecast differently once the future was appended"
        )
    assert compared >= minimum, (
        f"only {compared} of {len(pairs)} shared bars produced a usable forecast in both "
        "runs, too few for the comparison to mean anything"
    )


@pytest.mark.parametrize("name", [VolModel.GARCH, VolModel.EGARCH, VolModel.GJR_GARCH])
def test_a_forecast_does_not_change_when_the_future_is_appended(name: VolModel) -> None:
    """The causality test, on the simple family.

    The 400-bar series is walked forward whole, and again over its first 320 bars. Every bar
    scored in both runs must carry the *same* one-step forecast, to within 1e-6.

    The window sizes are chosen so that three things hold at once. The truncated run must be
    feasible: ``150 + 150 <= 320``. The two scored regions must overlap — ``[250, 320)``, fourteen
    bars — or the intersection is empty and the test passes without comparing anything; the
    overlap is asserted below rather than assumed. And the train window must be long enough that
    the model is *identified* on it. At 100 bars EGARCH is not: measured on this fixture it
    returns a collapsed forecast at six of these fourteen origins, which the service refuses, so
    the comparison would run over eight usable bars and the test would be measuring the fixture
    rather than the walk-forward. At 150 bars all three models are usable at all fourteen.

    1e-6 rather than bit-equality: the optimiser is iterative, and a slice one bar longer can
    converge to a parameter vector differing in the twelfth decimal. A leak is not a rounding
    difference — it moves forecasts by percent.
    """
    series = garch_series(400)
    spec = WalkForwardSpec(train_window=150, test_window=150, step_size=5)

    pairs = shared_forecasts(name, series, spec, 0.8)
    assert len(pairs) > 10, (
        "the two runs must overlap on enough bars for the comparison to mean anything; "
        f"got {len(pairs)}"
    )
    assert_prefixes_agree(pairs, minimum=10)


def test_the_hybrid_is_causal_too() -> None:
    """The same property on a hybrid, where the risk is larger and the failure is quieter.

    A hybrid trains a second model on the linear leg's standardised residuals. Computing
    those residuals from a fit over the whole series is the natural thing to write — they are
    "in-sample" either way — and it makes the learner's target depend on bars after the
    forecast origin. The resulting model is *better* on the training span for exactly that
    reason, so no test of the output alone would flag it. Only this comparison does.

    The XGBoost hybrid rather than the TDNN one: both go through the same ``HybridModel.fit``,
    and forty refits of a neural network would make this file slow enough that someone would
    eventually mark it skipped.
    """
    series = garch_series(300)
    spec = WalkForwardSpec(train_window=100, test_window=100, step_size=5)

    pairs = shared_forecasts(VolModel.ARMA_EGARCH_XGBOOST, series, spec, 0.8)
    assert len(pairs) > 5, f"only {len(pairs)} bars were scored in both runs"
    assert_prefixes_agree(pairs, minimum=5)


def test_the_leaky_alternative_really_would_have_failed_the_test_above() -> None:
    """The control that gives the truncation test its teeth.

    A property test only means something if the thing it forbids is both reachable and
    detectable. The mistake the walk-forward exists to prevent is *one fit over the whole
    series, then every bar's "forecast" read off that fit* — and the value such an
    implementation would have reported for the first scored bar is computable, because it is
    the bar's in-sample conditional variance under the full-sample parameters.

    So it is computed, and required to differ from the causal forecast for the same bar. If
    the two agreed, the truncation test would pass for a leaky loop too, and would be
    decoration.
    """
    series = garch_series(400)
    spec = WalkForwardSpec(train_window=100, test_window=150, step_size=5)
    index = spec.schedule(series.size)[0]

    causal = build(VolModel.GARCH)
    causal.fit(series[index - spec.train_window : index])
    causal_value = float(causal.forecast_variance(1)[0])

    leaky = build(VolModel.GARCH)
    assert isinstance(leaky, models.ArchModel)
    leaky.fit(series)  # every bar, including the 150 the causal fit never saw
    # `conditional_volatility[i]` is the standard deviation of the i-th error computed from
    # information up to i-1 — under parameters estimated from all 400 bars. Scaled back to
    # decimal variance, this is exactly what a whole-sample fit would call bar `index`'s
    # forecast.
    leaky_value = (
        float(np.asarray(leaky._result.conditional_volatility)[index]) ** 2
        / models.SCALE_VARIANCE
    )

    assert causal_value != pytest.approx(leaky_value, rel=1e-4), (
        "the full-sample fit and the causal fit agree at the first scored bar, so the "
        "truncation test cannot distinguish them"
    )


def test_a_diverged_forecast_is_refused_with_the_numbers() -> None:
    """The ways a fit that *returned* a number still produced no forecast.

    Each case is a value arch really produced on this project's own data — the ``0.0``, the
    ``8.3e+213`` and the ``1e-127`` all came out of EGARCH fits over short windows whose
    optimiser reported convergence. What makes them refusable is not their size in the
    abstract but their size relative to the window they came from, so both bounds are checked
    against the window's own statistics rather than against a constant — and against
    *different* ones, because "beyond anything the window saw" and "orders of magnitude below
    the window's level" are different questions.
    """
    window_max = 4e-5
    window_mean = 1e-5

    # A plausible forecast is accepted, and the same number is judged against its own window:
    # a variance of 1e-5 is ordinary here and two orders of magnitude below the ceiling.
    assert _forecast_refusal(1e-5, window_max, window_mean) is None
    # Exactly at either bound is inside it. The comparisons are strict, and pinning that here
    # keeps a later change from silently making the guards exclusive.
    assert (
        _forecast_refusal(
            MAX_FORECAST_VARIANCE_MULTIPLE * window_max,
            window_max,
            window_mean,
        )
        is None
    )
    assert (
        _forecast_refusal(
            MIN_FORECAST_VARIANCE_RATIO * window_mean,
            window_max,
            window_mean,
        )
        is None
    )
    # The bounds belong to the window: the value refused below is fine on a larger series.
    assert _forecast_refusal(1e-5, 1.0, 0.5) is None
    # A window whose returns are all exactly zero carries no volatility at all, so *any*
    # forecast from it is unmoored and the window is refused by the first branch, before either
    # bound is consulted. Pinned because this is what a reader would expect the floor to be
    # doing, and it is not.
    zero_window = _forecast_refusal(1e-5, 0.0, 0.0)
    assert zero_window is not None and "all exactly zero" in zero_window
    # The floor divides by the window's mean, so it refuses to fire when that mean is zero. A
    # window with variation but no level cannot come from real returns; it is pinned here
    # because an edit that dropped the guard would raise ZeroDivisionError inside a running job
    # rather than refuse a forecast.
    assert _forecast_refusal(1e-5, 4e-5, 0.0) is None

    refused = {
        "non-finite": _forecast_refusal(float("nan"), window_max, window_mean),
        "zero": _forecast_refusal(0.0, window_max, window_mean),
        "negative": _forecast_refusal(-1e-6, window_max, window_mean),
        "far above the window": _forecast_refusal(8.3e213, window_max, window_mean),
        "just above the ceiling": _forecast_refusal(
            MAX_FORECAST_VARIANCE_MULTIPLE * window_max * 1.0001, window_max, window_mean
        ),
        "far below the window": _forecast_refusal(1e-127, window_max, window_mean),
        "just below the floor": _forecast_refusal(
            MIN_FORECAST_VARIANCE_RATIO * window_mean * 0.999, window_max, window_mean
        ),
    }
    for label, reason in refused.items():
        assert reason is not None, label
        assert reason.startswith(FORECAST_REFUSAL_TAG), label
    # The message has to be actionable on its own: it is what the report and the lineage row
    # carry as the reason a step was not scored.
    assert "8.3e+213" in refused["far above the window"]
    assert "4e-05" in refused["far above the window"]
    assert "1e-127" in refused["far below the window"]
    assert "30 orders of magnitude" in refused["far below the window"]


class _ExplodingModel:
    """A model that fits without complaint and forecasts a variance out of all proportion.

    Stands in for the fit this guard exists for — an EGARCH whose ``β`` pinned at ≈ 0.999 on a
    short window — without depending on the optimiser landing there on the day this runs.
    """

    name = VolModel.GARCH

    def fit(self, returns: np.ndarray, exog: np.ndarray | None) -> None:
        self.window_max = float(np.max(returns**2))

    def forecast_variance(self, horizon: int) -> np.ndarray:
        return np.array([MAX_FORECAST_VARIANCE_MULTIPLE * self.window_max * 1e6])


def test_a_diverged_step_is_counted_and_never_scored() -> None:
    """The refusal has to reach the report as a *step*, not as a number.

    A quantity that is refused but still counted as a step would make the run look like it
    converged on every bar; a quantity that is dropped without a count would make the scored
    step count disagree with the schedule. So: NaN in the array, the reason in ``failures``,
    and a metrics row that is unscored with a sentence naming the likely fix.
    """
    series = garch_series(200)
    spec = WalkForwardSpec(train_window=100, test_window=30, step_size=1)
    times = pd.date_range("2024-01-01", periods=series.size, freq="h", tz="UTC")
    wrapped = _Series(
        returns=series, offset=0, column="log_return", derived=False, times=times
    )

    run = _walk_forward(_ExplodingModel(), wrapped, spec)  # type: ignore[arg-type]

    assert run.forecast.size == 30
    assert np.isnan(run.forecast).all()
    assert run.failures and run.failures[0].startswith(FORECAST_REFUSAL_TAG)

    score = _score(run, VolModel.GARCH.family)
    assert score.mqlike is None
    assert score.n_nan == 30
    assert score.unscored_reason is not None
    assert "Only 0 of 30 steps" in score.unscored_reason
    assert "longer train_window" in score.unscored_reason


class _CollapsingModel:
    """A fit whose variance recursion has been driven toward zero rather than toward infinity.

    The other half of the same pathology, and the half the ceiling alone does not catch: a
    value of ``1e-127`` is finite, positive, and passes every "did the recursion produce a
    number" test while carrying no more information than ``8e213``. This is the shape the
    XGBoost hybrid's EGARCH leg took on the e2e driver's four-hour fixture, where three of its
    forty scored steps collapsed and the surviving thirty-seven produced an MQLIKE of
    ``1.87e127`` — a headline number that meant nothing, printed beside two models whose
    MQLIKEs were 0.88 and 0.91.
    """

    name = VolModel.EGARCH

    def fit(self, returns: np.ndarray, exog: np.ndarray | None) -> None:
        self.window_mean = float(np.mean(returns**2))

    def forecast_variance(self, horizon: int) -> np.ndarray:
        return np.array([MIN_FORECAST_VARIANCE_RATIO * self.window_mean * 1e-3])


def test_a_collapsed_forecast_is_refused_too() -> None:
    """The floor reaches the report the same way the ceiling does.

    One test rather than two because the *consequences* are what matter and they are
    identical: the step becomes a NaN, it is counted, and the model's row says why instead of
    averaging a number that has nothing to do with the market.
    """
    series = garch_series(200)
    spec = WalkForwardSpec(train_window=100, test_window=30, step_size=1)
    times = pd.date_range("2024-01-01", periods=series.size, freq="h", tz="UTC")
    wrapped = _Series(
        returns=series, offset=0, column="log_return", derived=False, times=times
    )

    run = _walk_forward(_CollapsingModel(), wrapped, spec)  # type: ignore[arg-type]

    assert np.isnan(run.forecast).all()
    assert run.failures and "orders of magnitude below" in run.failures[0]

    score = _score(run, VolModel.EGARCH.family)
    assert score.mqlike is None
    assert score.n_nan == 30


def test_a_pair_with_too_little_overlap_is_reported_rather_than_dropped() -> None:
    """The comparison that did not happen is a finding with both model names in it.

    ``_compare`` used to ``continue`` past this pair, so a two-model run could come back with
    no comparison and no reason — indistinguishable, to a reader, from a comparison that found
    the models equal.
    """

    def run(name: VolModel, forecast: np.ndarray) -> _WalkForward:
        return _WalkForward(
            model=name,
            indices=np.arange(forecast.size),
            forecast=forecast,
            actual=np.full(forecast.size, 1e-5),
            seconds=0.0,
            clamped=0,
            untrained=False,
        )

    size = 60
    good = np.full(size, 1e-5)
    # The second model produced something usable on five bars only — the shape a diverged
    # hybrid leaves behind.
    patchy = np.full(size, np.nan)
    patchy[:5] = 1e-5

    comparisons = _compare(
        [run(VolModel.GARCH, good), run(VolModel.EGARCH, patchy)]
    )
    assert comparisons.rows == []
    assert len(comparisons.gaps) == 1
    assert "garch" in comparisons.gaps[0] and "egarch" in comparisons.gaps[0]
    assert "5 steps" in comparisons.gaps[0]

    # And with enough overlap the ordinary path still produces a row.
    healthy = _compare([run(VolModel.GARCH, good), run(VolModel.GARCH, good)])
    assert len(healthy.rows) == 1
    assert healthy.gaps == []


# --------------------------------------------------------------- the design


def test_the_delayed_design_reads_only_the_past() -> None:
    """Row ``j`` must hold lags ``1..lags`` and nothing at or beyond ``j``.

    Checked by poisoning the future: multiply every return from index 100 onward by a
    thousand and require the first 100 rows of the design to be bit-identical.
    """
    series = garch_series(200)
    before = models.delayed_design(series, None, 4)

    poisoned = series.copy()
    poisoned[100:] *= 1000.0
    after = models.delayed_design(poisoned, None, 4)

    np.testing.assert_array_equal(before[:100], after[:100])


def test_the_delayed_design_puts_the_most_recent_return_in_the_first_block() -> None:
    """The alignment that is off by one in the obvious implementation.

    Row ``j``'s first two entries are ``r_{j-1}`` and ``|r_{j-1}|``. Getting this wrong — by
    using ``r_j`` or ``r_{j-2}`` — leaves a model that still trains and still forecasts, just
    worse, with nothing in the output to say so. Hence a direct assertion.
    """
    design = models.delayed_design(np.arange(1.0, 11.0), None, 2)
    assert design[5][0] == pytest.approx(5.0)  # r_{j-1}, j = 5
    assert design[5][1] == pytest.approx(5.0)  # |r_{j-1}|
    assert design[5][2] == pytest.approx(4.0)  # r_{j-2}
    assert np.isnan(design[0]).all()


def test_a_hybrid_carries_the_feature_vector_for_the_bar_being_forecast() -> None:
    """The hybrid's next-step features are the lags ending at the last fitted bar.

    The version with the mistake trains, forecasts a finite number, and looks reasonable.
    What is wrong is that the features describe the *previous* bar, so the correction is
    always one step stale — a real loss of skill with no symptom.
    """
    series = garch_series(200)
    model = build(VolModel.TDNN_GJR_GARCH, lags=2)
    assert isinstance(model, models.HybridModel)
    model.fit(series)

    features = model._next_features
    assert features is not None
    assert features[0][0] == pytest.approx(series[-1])  # lag 1 of the bar being forecast
    assert features[0][2] == pytest.approx(series[-2])  # its lag 2


# ------------------------------------------------------------- the estimators


@pytest.mark.parametrize("distribution", list(VolDistribution))
@pytest.mark.parametrize("name", [VolModel.GARCH, VolModel.EGARCH, VolModel.GJR_GARCH])
def test_every_model_fits_and_forecasts_under_every_distribution(
    name: VolModel, distribution: VolDistribution
) -> None:
    """The four distributions on the three simple models — twelve combinations.

    A request for ``skew_student`` that quietly fitted a normal would produce a
    correct-looking forecast under a mislabelled report, which is why the mapping is asserted
    separately below as well as exercised here.
    """
    model = build(name, distribution=distribution)
    model.fit(garch_series(300))
    forecast = model.forecast_variance(1)
    assert forecast.shape == (1,)
    assert np.isfinite(forecast).all()
    assert forecast[0] > 0.0


def test_the_arch_distribution_map_is_exactly_the_specified_names() -> None:
    """One table, four rows, asserted literally. This is a translation layer, and a
    translation checked only by "something fitted" is not checked."""
    assert models.ARCH_DISTRIBUTION == {
        VolDistribution.NORMAL: "normal",
        VolDistribution.STUDENTS_T: "t",
        VolDistribution.SKEW_STUDENT: "skewt",
        VolDistribution.GED: "ged",
    }
    assert set(models.ARCH_DISTRIBUTION_CLASS) == set(VolDistribution)


def test_every_model_produces_a_multi_step_path() -> None:
    """The forward path, including for the models arch cannot forecast analytically.

    EGARCH and GJR do not evolve in squares, so steps 2..h come from a simulation rather than
    a closed form. The path still has to exist, be positive and be finite — and be the same on
    every call. arch reads the ``random_state`` argument only for *bootstrap*, so this is the
    test that would catch the simulation silently drawing fresh shocks each time.
    """
    for name in VolModel:
        model = build(name, lags=2)
        model.fit(garch_series(300))
        first = model.forecast_variance(12)
        second = model.forecast_variance(12)
        assert first.shape == (12,), name
        assert np.isfinite(first).all(), name
        assert (first > 0.0).all(), name
        np.testing.assert_array_equal(first, second, err_msg=f"{name} is not reproducible")


def test_the_two_hybrids_are_deterministic_across_runs() -> None:
    """Two runs of the same request must give bit-identical numbers.

    Both hybrids contain a stochastic learner — a boosted ensemble and a neural network — and
    both are seeded. An unseeded model would make the reproducibility gate unpassable and
    would make the causality test flaky instead of failing.
    """
    series = garch_series(250)
    for name in (VolModel.ARMA_EGARCH_XGBOOST, VolModel.TDNN_GJR_GARCH):
        first = build(name, lags=3)
        second = build(name, lags=3)
        first.fit(series)
        second.fit(series)
        np.testing.assert_array_equal(
            first.forecast_variance(1), second.forecast_variance(1), err_msg=name.value
        )


def test_the_scale_change_is_a_conditioning_transform_not_a_model_choice() -> None:
    """Fitting at ×100 and dividing back must give what fitting at ×100 directly gives.

    arch's optimiser works on the scale of its input, and decimal returns are ~1e-4, which
    puts the likelihood surface far from where the default starting values were tuned. The
    claim that this is a conditioning transform rather than a modelling decision rests on the
    two answers agreeing — and the comparison is made against ``arch_model`` called directly,
    not against the wrapper itself, so the assertion is about the rescaling rather than about
    the wrapper agreeing with its own arithmetic.
    """
    from arch import arch_model

    series = garch_series(400)
    wrapper = build(VolModel.GARCH)
    assert isinstance(wrapper, models.ArchModel)
    wrapper.fit(series)

    raw = arch_model(
        series * models.SCALE,
        mean="Constant",
        vol="GARCH",
        p=1,
        o=0,
        q=1,
        rescale=False,
    )
    raw_variance = float(
        np.asarray(
            raw.fit(disp="off", show_warning=False).forecast(horizon=1, reindex=False).variance
        )[0][0]
    )

    assert wrapper.forecast_variance(1)[0] == pytest.approx(
        raw_variance / models.SCALE_VARIANCE, rel=1e-10
    )


def test_the_correction_is_clamped_and_the_clamp_is_reported() -> None:
    """``E[z²] = 1`` is a fact about a correctly specified model, not about a tree ensemble.

    A boosted learner extrapolating on features outside its training range will predict a
    correction of 30 without hesitation, and an unclamped product turns a reasonable variance
    forecast into a spike. The clamp bounds it — asserted here by reconstructing the hybrid's
    step-1 variance from the linear leg's — and ``correction_clamped`` counts how often it
    bound, so a run pinned at the boundary is visible rather than confident.
    """
    low, high = models.CORRECTION_CLAMP
    assert 0.0 < low < 1.0 < high

    model = build(VolModel.TDNN_GJR_GARCH, lags=2)
    assert isinstance(model, models.HybridModel)
    model.fit(garch_series(200))

    linear = float(model.linear.forecast_variance(1)[0])
    combined = float(model.forecast_variance(1)[0])
    assert low * linear <= combined <= high * linear
    assert model.correction_clamped >= 0


def test_standardised_residuals_have_unit_variance_by_construction() -> None:
    """``z = e/σ`` and ``E[z²] ≈ 1`` — the property the hybrid's multiplicative form relies on.

    If the residuals were not standardised, the learner would be predicting a quantity whose
    scale drifts with the volatility regime, and the "correction" would be a constant multiple
    of nothing. The band is wide because this is a sample moment of a heavy-tailed ratio, not
    an identity the estimator is constrained to satisfy exactly.
    """
    model = build(VolModel.GARCH)
    assert isinstance(model, models.ArchModel)
    model.fit(garch_series(400))

    z = model.standardised_residuals()
    finite = z[np.isfinite(z)]
    assert finite.size > 300
    assert float(np.mean(finite**2)) == pytest.approx(1.0, abs=0.2)


def test_the_ar_mean_option_is_exercised_on_both_families() -> None:
    """``mean_model="arma"`` is a control on the page and the only path where the two arrays
    a hybrid pairs are not the same length.

    arch estimates an AR(p) mean by holding back the first ``p`` observations from the fit
    sample, and it then writes the residuals into a **full-length** array: ``ARCHModel._fit``
    builds ``resid`` and ``conditional_volatility`` as ``np.full(self._y.shape, np.nan)`` and
    fills ``[first_obs:last_obs]``. So the standardised residual series is as long as the
    slice with ``p`` leading NaNs — which is what lets the hybrid's ``train_x`` and
    ``train_y`` line up and the ``isfinite`` mask drop exactly those rows.

    This asserts the padding instead of assuming it. If a future arch returned the shorter
    fit-sample array, the broadcast in ``HybridModel.fit``'s ``valid`` mask would raise, and
    the walk-forward's per-step ``except`` would swallow it into "the hybrid produced no
    finite forecasts" — a model reported as unscored for a reason no message would name.
    """
    series = garch_series(300)

    linear = build(VolModel.GARCH, mean_model=VolMeanModel.ARMA, p=2)
    assert isinstance(linear, models.ArchModel)
    linear.fit(series)
    assert linear.mean_label == "ar2"
    residuals = linear.standardised_residuals()
    assert residuals.shape == series.shape
    assert np.isnan(residuals[:2]).all()
    assert np.isfinite(residuals[2:]).all()
    assert float(linear.forecast_variance(1)[0]) > 0.0

    # The hybrid reads the same array through the same call, so it is the estimator that
    # would break first — and it would break silently, by falling back to its linear leg.
    hybrid = build(VolModel.ARMA_EGARCH_XGBOOST, mean_model=VolMeanModel.ARMA, p=2, lags=3)
    assert isinstance(hybrid, models.HybridModel)
    hybrid.fit(series)
    assert hybrid.untrained is False
    assert np.isfinite(hybrid.forecast_variance(1)).all()
    assert hybrid.forecast_variance(1)[0] > 0.0


# -------------------------------------------------------------------- refusals


def test_a_train_window_that_leaves_no_history_is_refused_with_the_numbers() -> None:
    spec = WalkForwardSpec(train_window=100, test_window=50)
    with pytest.raises(ValueError, match="150"):
        spec.schedule(120)


def test_a_step_larger_than_the_test_window_is_refused() -> None:
    with pytest.raises(ValueError, match="step_size"):
        WalkForwardSpec(train_window=100, test_window=50, step_size=60)


def test_the_schedule_is_the_last_test_window_bars_at_the_requested_step() -> None:
    """``test_window`` is a *bar count*, so the scored region is the last that many bars and
    the number of scored points is ``ceil(test_window / step_size)``."""
    spec = WalkForwardSpec(train_window=100, test_window=50, step_size=10)
    assert spec.schedule(400) == [350, 360, 370, 380, 390]
    assert spec.n_scored == 5


def test_an_impossible_schedule_is_a_validation_failure_not_a_crash() -> None:
    """The refusal the service raises, so the error class the API turns into a 422 is pinned
    here rather than only in the router tests."""
    with pytest.raises(ValidationFailure):
        _validate_schedule(WalkForwardSpec(train_window=300, test_window=200), 100)


def test_a_non_return_column_is_refused_by_the_stationarity_guard() -> None:
    """What the guard is actually for: a column that is not a return series at all.

    A price level is near-unit-root by construction, so a cumulative total, an open interest
    or a running volume fails this and is caught. The failure carries the statistic and the
    p-value rather than reading as "invalid input".
    """
    rng = np.random.default_rng(3)
    walk = np.cumsum(rng.normal(0.0, 1.0, 500)) + 100.0
    statistic, p_value = models.adf_test(walk)
    assert p_value > 0.05
    assert np.isfinite(statistic)


def test_a_genuine_return_series_passes_the_stationarity_guard() -> None:
    _, p_value = models.adf_test(garch_series(500))
    assert p_value < 0.05


def test_the_model_factory_covers_every_member_of_the_enum() -> None:
    """A model added to the enum without a factory branch would raise at request time rather
    than being caught here."""
    for name in VolModel:
        assert build(name).name is name


def test_a_hybrid_falls_back_to_its_linear_leg_without_enough_history() -> None:
    """Rather than training a second model on ten usable rows, which would fit noise and then
    report itself as a hybrid that had earned its cost.

    The fallback is recorded (``untrained``) so the report can say the row in the metrics table
    is the linear model's result, and the forecast is still produced.
    """
    model = build(VolModel.ARMA_EGARCH_XGBOOST, lags=35)
    assert isinstance(model, models.HybridModel)
    model.fit(garch_series(50))
    assert model.untrained is True
    assert model.forecast_variance(1)[0] > 0.0
