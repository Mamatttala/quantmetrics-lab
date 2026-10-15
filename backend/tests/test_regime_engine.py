"""Module D's statistical guarantees, tested against the properties the specification names.

The tests here are written around one idea: a regime model is easy to make *look* right.
The probabilities sum to one, the timeline is pretty, and every one of those things stays
true when the labels were computed with knowledge of the future. So almost nothing here
asserts that a number looks plausible. The tests assert properties that a leaking
implementation cannot have:

* **Truncation invariance.** Re-run on 80% of the series and the first 80% must be
  bit-identical. An implementation that peeks at later bars — a smoothed posterior, a
  whole-sample scaler, a refit schedule derived from the series length — fails this and
  fails almost nothing else.
* **The crisis override fires immediately.** Not "within a few bars": on the first bar the
  breach is measurable, with the minimum-duration rule set high enough that a
  rule-respecting implementation could not possibly have committed yet.
* **Anti-thrashing reduces switches.** Asserted as a comparison between two settings on
  the same data, because the absolute switch count of a random series means nothing.
* **Net Value-Add matches hand arithmetic.** Computed by hand in the test, not regenerated
  by the code under test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.models.enums import AssetClass, RegimeFeature, RegimeModel, RegimeState, Timeframe
from app.services import regime_engine
from app.services.regime_engine import RegimeSpec, commit_labels, net_value_add

# ---------------------------------------------------------------- synthetic data


def _prices(
    n: int,
    *,
    seed: int = 0,
    base_vol: float = 0.0015,
    spike: tuple[int, int, float] | None = None,
) -> pd.DataFrame:
    """Hourly bars with a controllable volatility profile.

    The volatility levels are chosen against the annualisation factor, not picked for
    looks. Hourly crypto bars annualise by ``24 * 365 = 8760``, so ``sqrt(8760) ≈ 93.6``:
    a per-bar standard deviation of 0.0015 gives an annualised volatility of about 0.14,
    comfortably under the 0.35 crisis threshold, and a 0.02 spike gives about 1.87,
    comfortably over it. A test that used a 1% per-bar move would sit at 0.94 and be in
    permanent crisis without anyone noticing.
    """
    rng = np.random.default_rng(seed)
    vol = np.full(n, base_vol)
    if spike is not None:
        start, stop, level = spike
        vol[start:stop] = level
    returns = rng.normal(0.0, 1.0, n) * vol
    close = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame(
        {
            "open_time": pd.date_range("2021-01-01", periods=n, freq="h", tz="UTC"),
            "close": close,
            "volume": np.exp(rng.normal(10.0, 0.3, n)),
        }
    )


def _spec(**overrides: object) -> RegimeSpec:
    base: dict[str, object] = {
        "model_type": RegimeModel.HMM,
        "n_states": 4,
        "features": (RegimeFeature.LOG_RETURN, RegimeFeature.REALIZED_VOLATILITY),
        "min_state_duration": 5,
        "switching_cost_penalty": 0.005,
        "crisis_threshold": 0.35,
        "override_probability": 0.90,
        "transition_confidence": 0.55,
        "commit_confidence": 0.60,
        "min_train_bars": 200,
        "refit_every": 200,
        "volatility_window": 24,
    }
    base.update(overrides)
    return RegimeSpec(**base)  # type: ignore[arg-type]


def _run(frame: pd.DataFrame, spec: RegimeSpec) -> regime_engine.RegimeOutcome:
    return regime_engine.detect(
        frame,
        time_column="open_time",
        spec=spec,
        timeframe=Timeframe.H1,
        asset_class=AssetClass.CRYPTO_SPOT,
    )


PROBABILITY_COLUMNS = [
    "prob_trending_up",
    "prob_trending_down",
    "prob_ranging",
    "prob_crisis",
]

#: The output frame's columns, in the order the specification names them. The time column
#: is `open_time` — the observation instant — not `close_time`, which it was misnamed as.
OUTPUT_COLUMNS = ["open_time", "dominant_state", *PROBABILITY_COLUMNS, "confidence"]


# ------------------------------------------------------------- output contract


def test_output_schema_is_exactly_what_the_specification_names() -> None:
    outcome = _run(_prices(520, seed=1), _spec())
    assert list(outcome.frame.columns) == OUTPUT_COLUMNS
    assert outcome.frame["open_time"].dt.tz is not None
    assert str(outcome.frame["open_time"].dt.tz) == "UTC"


def test_rows_before_the_first_fit_carry_no_label() -> None:
    """Warm-up rows are null, and are counted rather than quietly labelled.

    Labelling them would mean picking a state from a model that has not been fitted, or
    from one fitted on the whole series. Both are worse than an honest gap.
    """
    outcome = _run(_prices(520, seed=2), _spec())
    frame = outcome.frame
    warmup = outcome.report["warmup_bars"]
    assert warmup >= 24  # the volatility window
    assert frame["dominant_state"].iloc[:warmup].isna().all()
    assert frame[PROBABILITY_COLUMNS].iloc[:warmup].isna().all().all()
    assert frame["dominant_state"].iloc[warmup:].notna().any()


def test_probabilities_are_soft_and_sum_to_one_where_labelled() -> None:
    """Every labelled row is a probability vector, and the model is not a hard argmax.

    The specification forbids hard 0/1 labels. The precise claim that is testable is *not*
    "no probability is ever near zero or one" — a well-separated fit legitimately
    saturates, and asserting otherwise would be asserting that the model is never
    confident. It is "the output is a posterior, not a one-hot encoding": on most bars the
    mass is spread over more than one state, which is exactly what a hard-labelling
    implementation could not produce.
    """
    outcome = _run(_prices(520, seed=3), _spec())
    frame = outcome.frame
    labelled = frame[frame["dominant_state"].notna()]
    assert not labelled.empty
    probabilities = labelled[PROBABILITY_COLUMNS].to_numpy(dtype=float)

    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0.0).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-9)

    spread = (probabilities.max(axis=1) < 0.999).mean()
    assert spread > 0.10, (
        f"only {spread:.1%} of labelled bars carry any uncertainty — this looks like a "
        "hard argmax wearing a probability's name"
    )


def test_the_label_and_its_probabilities_never_contradict_each_other() -> None:
    """Each committed label is consistent with the row it sits on.

    Three separate guarantees, and it is worth being precise about which holds where, since
    the obvious general claim — "``dominant_state`` is always the argmax" — is **false by
    design**. Anti-thrashing means the committed label lags the argmax: the model can be
    most confident of Ranging while the label still reads Trending Up, because the
    challenger has not held ``commit_confidence`` for ``min_state_duration`` yet. That lag
    is the feature.

    * ``confidence`` is the probability of the state that was committed, not the row's
      maximum. On a lagging bar those differ, and reporting the maximum would claim
      confidence in a state the label does not name.
    * On a **crisis-override** bar — one where the volatility mask actually fired — the
      label is Crisis, the crisis column is at least the override probability, and crisis
      is that row's argmax. This is the property an earlier version violated: it relabelled
      the bar while leaving ``prob_crisis`` at ~1e-183, so the frame contradicted itself and
      any consumer reading the columns instead of the label got the opposite answer with no
      error raised. Crisis *held* after the breach is not asserted here: it is subject to
      the ordinary anti-thrashing rule, and its probability is allowed to decay.
    """
    spec = _spec(refit_every=200)
    frame = _prices(520, seed=31, base_vol=0.0015, spike=(400, 430, 0.02))
    outcome = _run(frame, spec)
    labelled = outcome.frame[outcome.frame["dominant_state"].notna()]
    assert not labelled.empty

    decided = labelled[labelled["dominant_state"] != RegimeState.TRANSITION.value]
    assert not decided.empty
    values = decided[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    positions = [
        PROBABILITY_COLUMNS.index(f"prob_{state}") for state in decided["dominant_state"]
    ]
    np.testing.assert_allclose(
        decided["confidence"].to_numpy(dtype=float),
        values[np.arange(len(values)), positions],
        atol=1e-12,
    )

    volatility = regime_engine.realized_volatility(
        regime_engine._log_returns(frame["close"].to_numpy(dtype=float)),
        spec.volatility_window,
        8760.0,
    )
    breaching = outcome.frame[regime_engine.crisis_mask_for(volatility, spec.crisis_threshold)]
    breaching = breaching[breaching["dominant_state"].notna()]
    assert not breaching.empty, "the synthetic spike produced no breaching bars"
    assert (breaching["dominant_state"] == RegimeState.CRISIS.value).all()
    assert (breaching["prob_crisis"] >= spec.override_probability).all()
    assert (
        breaching[PROBABILITY_COLUMNS].idxmax(axis=1).str.removeprefix("prob_")
        == RegimeState.CRISIS.value
    ).all()


def test_confidence_falls_when_the_committed_state_loses_support() -> None:
    """The gauge the interface draws must be able to read below its own warning line.

    If ``confidence`` were the row's maximum it could never sit under 55% while any state
    was doing well, and the "treat this as a gray zone" warning would be unreachable —
    a control that cannot fire. Reporting the committed state's own probability is what
    makes the warning mean something.
    """
    spec = _spec(min_state_duration=10, transition_confidence=0.55, commit_confidence=0.60)
    # Trending Up is committed and held; the model then swings to Ranging.
    named = np.array(
        [
            [0.90, 0.02, 0.06, 0.02],
            [0.02, 0.02, 0.94, 0.02],
        ]
    )
    labelling = commit_labels(named, spec=spec, crisis_mask=np.zeros(2, dtype=bool))
    assert labelling.states[0] is RegimeState.TRENDING_UP
    # Held, because the incumbent is only one bar old.
    assert labelling.states[1] is RegimeState.TRENDING_UP
    # And the reported confidence is Trending Up's own 0.02, not Ranging's 0.94.
    assert labelling.confidence[1] == pytest.approx(0.02)


def test_the_crisis_override_beats_the_models_own_ranking() -> None:
    """On a breaching bar, crisis becomes the argmax even if the model ranked it last."""
    named = np.array([[0.90, 0.05, 0.05, 0.00], [0.00, 0.00, 0.01, 0.00]])
    mask = np.array([False, True])
    adjusted = regime_engine.apply_crisis_override(named, mask, 0.90)

    # The untouched bar is returned unchanged.
    np.testing.assert_allclose(adjusted[0], named[0])
    # The breaching bar's crisis column is at least the override, and is now the argmax.
    assert adjusted[1][3] >= 0.90
    assert int(np.argmax(adjusted[1])) == 3
    # Mass is conserved and redistributed in proportion to what the model believed.
    np.testing.assert_allclose(adjusted[1].sum(), 1.0, atol=1e-12)
    assert adjusted[1][2] > adjusted[1][1]  # the model's 0.01 outweighs its 0.00


def test_the_crisis_override_leaves_a_breach_in_the_warmup_unlabelled_but_counted() -> None:
    """A breach before the first fitted bar is reported, not silently dropped.

    A row with a state and no probability would break the invariant that the state is the
    argmax of its own row, so warm-up breaches cannot be labelled. They are counted
    instead, because "the crisis detector was blind for the first 223 bars" is something
    the user needs to know about a risk control.
    """
    frame = _prices(520, seed=91, base_vol=0.0015, spike=(30, 60, 0.02))
    outcome = _run(frame, _spec(refit_every=200))
    assert outcome.report["crisis_bars_in_warmup"] > 0
    assert outcome.report["crisis_override_bars"] == 0
    assert outcome.frame["dominant_state"].iloc[: outcome.report["warmup_bars"]].isna().all()


# ----------------------------------------------------------- causality (Part 5.1)


@pytest.mark.parametrize("model", [RegimeModel.HMM, RegimeModel.GMM, RegimeModel.RULES])
def test_truncating_the_series_does_not_change_the_earlier_labels(
    model: RegimeModel,
) -> None:
    """The central no-look-ahead property, checked the way the specification asks.

    Run on the full series, then on the first 80%, and require the shared prefix to be
    identical. This is the test that fails for a smoothed posterior, for a whole-sample
    normalisation, and for a refit schedule whose stride is a fraction of the series
    length.
    """
    full = _prices(600, seed=11)
    cutoff = 480
    spec = _spec(model_type=model)

    whole = _run(full, spec).frame
    truncated = _run(full.iloc[:cutoff].copy(), spec).frame

    assert len(truncated) == cutoff
    np.testing.assert_allclose(
        whole[PROBABILITY_COLUMNS].to_numpy(dtype=float)[:cutoff],
        truncated[PROBABILITY_COLUMNS].to_numpy(dtype=float)[:cutoff],
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
        err_msg=f"{model.value}: the first {cutoff} bars changed when the series was "
        "truncated, which means some bar's label depends on data that followed it.",
    )
    assert (
        whole["dominant_state"].iloc[:cutoff].tolist()
        == truncated["dominant_state"].iloc[:cutoff].tolist()
    )
    np.testing.assert_allclose(
        whole["confidence"].to_numpy(dtype=float)[:cutoff],
        truncated["confidence"].to_numpy(dtype=float)[:cutoff],
        equal_nan=True,
    )


def test_refit_schedule_is_anchored_to_bar_index_not_series_length() -> None:
    """The schedule must not move when the series grows.

    Stated separately from the truncation test because it fails *for a different reason*
    and would otherwise be masked: a length-derived stride changes every refit point, so
    the truncation test would catch it, but this one names the cause.
    """
    spec = _spec()
    short = regime_engine._refit_schedule(800, spec, valid_from=23)
    long = regime_engine._refit_schedule(1600, spec, valid_from=23)
    assert [s.trained_through for s in short] == [223, 423, 623]
    # The longer series extends the same schedule rather than recomputing it.
    assert [s.trained_through for s in long][: len(short)] == [
        s.trained_through for s in short
    ]
    # Every segment trains from the end of the warm-up, never from row zero.
    assert {s.train_from for s in long} == {23}


def test_volatility_uses_a_trailing_window_only() -> None:
    """A prefix's last volatility equals the full run's value at that bar."""
    close = _prices(400, seed=12)["close"].to_numpy(dtype=float)
    returns = regime_engine._log_returns(close)
    whole = regime_engine.realized_volatility(returns, 24, 8760.0)
    prefix = regime_engine.realized_volatility(returns[:300], 24, 8760.0)
    np.testing.assert_allclose(whole[299], prefix[299], rtol=1e-12)
    assert np.isnan(whole[:23]).all()


# ------------------------------------------------------ anti-thrashing (Part 5.2)


def test_minimum_state_duration_reduces_switching_on_an_oscillating_series() -> None:
    """Same data, two settings, fewer switches for the patient one.

    An absolute switch count would be meaningless — it depends entirely on how noisy the
    synthetic series is — so the assertion is a comparison.
    """
    frame = _prices(600, seed=21, base_vol=0.0012)
    impatient = _run(frame, _spec(min_state_duration=1, refit_every=150))
    patient = _run(frame, _spec(min_state_duration=25, refit_every=150))
    assert patient.report["switch_count"] < impatient.report["switch_count"]


def test_an_ambiguous_bar_is_labelled_transition_not_a_guess() -> None:
    """Below the transition threshold no state has a mandate, and the label says so.

    Tested directly against the commit rule rather than by trying to synthesise a market
    that is genuinely 40% trending. A synthetic series that produced this posterior would
    be tuned until it did, and the test would then be asserting the tuning.
    """
    spec = _spec(transition_confidence=0.55, commit_confidence=0.60)
    ambiguous = np.array([[0.40, 0.35, 0.25, 0.00]])
    labelling = commit_labels(ambiguous, spec=spec, crisis_mask=np.array([False]))
    assert labelling.states[0] is RegimeState.TRANSITION
    # And the confidence reported is the model's own, not the commit threshold.
    assert labelling.confidence[0] == pytest.approx(0.40)


def test_a_confident_but_impatient_switch_is_refused() -> None:
    """Confidence alone is not enough; the incumbent must have been held long enough."""
    spec = _spec(min_state_duration=10, transition_confidence=0.55, commit_confidence=0.60)
    # Bar 1 is confidently Ranging, but Trending Up has only been held one bar.
    named = np.array(
        [
            [0.90, 0.05, 0.05, 0.00],
            [0.05, 0.05, 0.90, 0.00],
        ]
    )
    labelling = commit_labels(named, spec=spec, crisis_mask=np.zeros(2, dtype=bool))
    assert labelling.states[0] is RegimeState.TRENDING_UP
    assert labelling.states[1] is RegimeState.TRENDING_UP
    assert labelling.switches == 0


def test_a_confident_and_patient_switch_is_allowed() -> None:
    spec = _spec(min_state_duration=2, transition_confidence=0.55, commit_confidence=0.60)
    up = [0.90, 0.05, 0.05, 0.00]
    ranging = [0.05, 0.05, 0.90, 0.00]
    named = np.array([up, up, ranging, ranging])
    labelling = commit_labels(named, spec=spec, crisis_mask=np.zeros(4, dtype=bool))
    assert labelling.states[2] is RegimeState.RANGING
    assert labelling.switches == 1


# --------------------------------------------------------- transition (Part 5.4)


def test_transition_appears_when_no_state_holds_a_clear_majority() -> None:
    """A run of low-confidence bars must produce actual Transition labels."""
    spec = _spec(min_state_duration=5)
    soft = np.array([[0.30, 0.28, 0.25, 0.17]] * 8)
    labelling = commit_labels(soft, spec=spec, crisis_mask=np.zeros(8, dtype=bool))
    assert all(state is RegimeState.TRANSITION for state in labelling.states)
    assert labelling.switches == 0


# ----------------------------------------------------------- crisis (Part 5.3)


def test_crisis_override_fires_on_the_first_breaching_bar() -> None:
    """The override is not permitted to wait.

    ``min_state_duration`` is set to 20 so a rule-respecting implementation would need
    twenty bars before it could commit to anything; the crisis must appear on the first
    bar whose trailing volatility exceeds the threshold.
    """
    spike_start, spike_stop = 400, 430
    frame = _prices(520, seed=31, base_vol=0.0015, spike=(spike_start, spike_stop, 0.02))
    spec = _spec(min_state_duration=20, volatility_window=24, refit_every=200)
    outcome = _run(frame, spec)

    volatility = regime_engine.realized_volatility(
        regime_engine._log_returns(frame["close"].to_numpy(dtype=float)), 24, 8760.0
    )
    breaching = np.flatnonzero(volatility > spec.crisis_threshold)
    assert breaching.size > 0, "the synthetic spike did not breach the threshold"

    first = int(breaching[0])
    row = outcome.frame.iloc[first]
    assert row["dominant_state"] == RegimeState.CRISIS.value, (
        f"bar {first} breached the crisis threshold but was labelled "
        f"{row['dominant_state']!r} — the override did not bypass min_state_duration"
    )
    assert row["prob_crisis"] >= spec.override_probability
    assert outcome.report["crisis_override_bars"] > 0


def test_crisis_probability_overrides_a_confident_model() -> None:
    """Even when the model is sure of something else, crisis wins on a breach."""
    dangerous = np.array([[0.00, 0.00, 0.00, 0.05]])
    labelling = commit_labels(
        np.array([[0.95, 0.02, 0.02, 0.01]]), spec=_spec(), crisis_mask=np.array([True])
    )
    assert labelling.states[0] is RegimeState.CRISIS
    assert labelling.confidence[0] >= 0.90
    del dangerous


def test_crisis_is_not_declared_below_the_threshold() -> None:
    frame = _prices(520, seed=32, base_vol=0.0012)
    outcome = _run(frame, _spec(refit_every=200))
    assert outcome.report["crisis_override_bars"] == 0


def test_raising_the_threshold_removes_crisis_bars() -> None:
    """The threshold is a real control, not a decoration."""
    frame = _prices(520, seed=33, base_vol=0.0015, spike=(400, 430, 0.02))
    low = _run(frame, _spec(crisis_threshold=0.20, refit_every=200))
    high = _run(frame, _spec(crisis_threshold=50.0, refit_every=200))
    assert low.report["crisis_override_bars"] > 0
    assert high.report["crisis_override_bars"] == 0


# ------------------------------------------------------ net value-add (Part 5.5)


def test_net_value_add_matches_hand_arithmetic() -> None:
    """Worked by hand, on four bars, with the position lag built into the expectation.

    ``trending_up`` on bar 1 earns bar 2's return, not bar 1's. Getting that lag wrong
    makes the statistic look better and is the classic same-bar look-ahead.
    """
    returns = np.array([np.nan, 0.01, 0.02, -0.05])
    states = [None, RegimeState.TRENDING_UP, RegimeState.TRENDING_UP, RegimeState.RANGING]
    result = net_value_add(returns, states, switching_cost_penalty=0.01)

    # t=1: long, earns r[2] = +0.02, and is the first position so it counts as a change.
    # t=2: still long, earns r[3] = -0.05, no change.
    assert result["strategy_log_return"] == pytest.approx(0.02 - 0.05)
    assert result["benchmark_log_return"] == pytest.approx(0.02 - 0.05)
    assert result["return_improvement"] == pytest.approx(0.0)
    assert result["position_changes"] == 1
    assert result["bars_long"] == 2
    assert result["switching_cost"] == pytest.approx(0.01)
    assert result["net_value_add"] == pytest.approx(-0.01)
    assert result["bars_scored"] == 2


def test_net_value_add_counts_position_changes_not_state_changes() -> None:
    """The two counts differ, which is why they are not allowed to share a name.

    Moving from ``ranging`` to ``crisis`` is a state change that costs nothing here,
    because neither holds a position. A report whose cost is computed on one count while
    its headline ``switch_count`` shows the other cannot be checked by hand — which is
    exactly what a reader does with a number like this.
    """
    returns = np.array([np.nan, 0.01, 0.01, 0.01, 0.01])
    states = [
        None,
        RegimeState.RANGING,
        RegimeState.CRISIS,
        RegimeState.TRENDING_UP,
        RegimeState.TRENDING_UP,
    ]
    result = net_value_add(returns, states, switching_cost_penalty=0.01)
    assert result["position_changes"] == 1  # entering the long, not the two state moves
    assert result["bars_long"] == 1
    assert result["switching_cost"] == pytest.approx(0.01)


def test_a_strategy_that_never_goes_long_says_so() -> None:
    """``bars_long == 0`` is the tell that the improvement is just ``-benchmark``.

    The heuristic path produces this whenever volatility sits near the crisis threshold:
    every state's probability is scaled down until nothing clears the commit threshold, so
    no bar is ever ``trending_up`` and the Net Value-Add measures the benchmark rather than
    the model. Without this field the number looks like a scored evaluation.
    """
    returns = np.array([np.nan, -0.02, -0.05, -0.05, -0.05])
    states = [
        None,
        RegimeState.RANGING,
        RegimeState.TRANSITION,
        RegimeState.CRISIS,
        RegimeState.RANGING,
    ]
    result = net_value_add(returns, states, switching_cost_penalty=0.0)
    assert result["bars_long"] == 0
    assert result["position_changes"] == 0
    assert result["strategy_log_return"] == pytest.approx(0.0)
    assert result["benchmark_log_return"] == pytest.approx(-0.15)
    assert result["return_improvement"] == pytest.approx(
        -result["benchmark_log_return"]
    )


def test_net_value_add_credits_a_correct_avoidance() -> None:
    """Flat during a decline beats buy-and-hold, and the sign must reflect that."""
    returns = np.array([np.nan, 0.0, -0.10, -0.10])
    states = [None, RegimeState.CRISIS, RegimeState.CRISIS, RegimeState.CRISIS]
    result = net_value_add(returns, states, switching_cost_penalty=0.0)
    assert result["strategy_log_return"] == pytest.approx(0.0)
    assert result["benchmark_log_return"] == pytest.approx(-0.20)
    assert result["net_value_add"] == pytest.approx(0.20)


def test_net_value_add_ignores_warmup_rows() -> None:
    returns = np.array([np.nan, 0.05, 0.05])
    states = [None, None, RegimeState.TRENDING_UP]
    result = net_value_add(returns, states, switching_cost_penalty=0.0)
    assert result["bars_scored"] == 0
    assert result["net_value_add"] == pytest.approx(0.0)


def test_the_run_reports_an_in_sample_caveat() -> None:
    """The number is a descriptive statistic, and the output says so in the payload.

    Without this the figure is read as a backtest result, which it is not: the states
    were fitted on the same rows they are scored on.
    """
    outcome = _run(_prices(520, seed=41), _spec(refit_every=200))
    evaluation = outcome.report["net_value_add"]
    assert evaluation["in_sample"] is True
    assert "buy-and-hold" in evaluation["definition"]
    assert "t+1" in evaluation["definition"]


# ------------------------------------------------------------ report integrity


def test_report_carries_the_diagnostics_the_specification_asks_for() -> None:
    outcome = _run(_prices(520, seed=51), _spec(refit_every=200))
    report = outcome.report
    for key in (
        "bic_latest_refit",
        "aic_latest_refit",
        "switch_count",
        "state_durations",
        "transition_matrix",
        "net_value_add",
        "stationarity",
        "warmup_bars",
    ):
        assert key in report, key
    assert report["bic_latest_refit"] is not None
    assert report["refit_count"] == len(report["refit_points"]) > 0


def test_the_two_unlabelled_prefixes_are_reported_separately() -> None:
    """``warmup_bars`` and the null prefix are different lengths, and both are reported.

    The observation warm-up is only the first part of it: after the features exist, the
    first model still has to train on ``min_train_bars`` rows before it can label anything,
    and those rows carry no label either. Reporting the smaller number under a name that
    reads as "the rows before the first label" understates the gap several-fold — and it
    contradicted ``state_counts["warmup"]``, which counts the larger one.
    """
    report = _run(_prices(520, seed=61), _spec(refit_every=200)).report
    assert report["unlabelled_bars"] + report["labelled_rows"] == report["rows"]
    assert report["state_counts"]["warmup"] == report["unlabelled_bars"]
    assert report["warmup_bars"] < report["unlabelled_bars"]
    assert report["unlabelled_bars"] - report["warmup_bars"] == report["min_train_bars"]


def test_transition_matrix_rows_sum_to_one_or_zero() -> None:
    outcome = _run(_prices(520, seed=52), _spec(refit_every=200))
    labels = outcome.report["transition_matrix"]["labels"]
    values = outcome.report["transition_matrix"]["values"]
    assert labels == [state.value for state in RegimeState]
    for label, row in zip(labels, values, strict=True):
        total = sum(row)
        assert total == pytest.approx(1.0) or total == 0.0, label


def test_transition_matrix_describes_the_committed_labels_not_the_fitted_parameters() -> None:
    """Counted from the output frame, so the reported matrix is checkable by hand."""
    outcome = _run(_prices(520, seed=53), _spec(refit_every=200))
    labels = outcome.report["transition_matrix"]["labels"]
    values = outcome.report["transition_matrix"]["values"]
    states = outcome.frame["dominant_state"].dropna().tolist()
    index = {label: position for position, label in enumerate(labels)}

    expected = np.zeros((len(labels), len(labels)))
    for previous, current in zip(states[:-1], states[1:], strict=True):
        expected[index[previous], index[current]] += 1.0
    totals = expected.sum(axis=1, keepdims=True)
    expected = np.divide(expected, totals, out=np.zeros_like(expected), where=totals > 0)
    np.testing.assert_allclose(values, expected)


def test_it_does_not_change_the_source_frame() -> None:
    frame = _prices(520, seed=61)
    before = frame.copy(deep=True)
    _run(frame, _spec(refit_every=200))
    pd.testing.assert_frame_equal(frame, before)


def test_the_same_request_twice_produces_the_same_labels() -> None:
    """Reproducibility. EM finds a local optimum, so a random start would break this."""
    frame = _prices(520, seed=62)
    first = _run(frame, _spec(refit_every=200)).frame
    second = _run(frame, _spec(refit_every=200)).frame
    pd.testing.assert_frame_equal(first, second)


# -------------------------------------------------------------- input refusals


def test_unsorted_timestamps_are_refused() -> None:
    frame = _prices(400, seed=71)
    frame = frame.iloc[::-1].reset_index(drop=True)
    with pytest.raises(Exception, match="not sorted"):
        _run(frame, _spec())


def test_duplicate_timestamps_are_refused() -> None:
    frame = _prices(400, seed=72)
    frame.loc[10, "open_time"] = frame.loc[9, "open_time"]
    with pytest.raises(Exception, match="duplicate"):
        _run(frame, _spec())


def test_empty_timestamps_are_refused() -> None:
    frame = _prices(400, seed=73)
    frame.loc[5, "open_time"] = pd.NaT
    with pytest.raises(Exception, match="empty"):
        _run(frame, _spec())


def test_a_dataset_without_a_timeframe_is_refused_rather_than_guessed() -> None:
    """The annualisation factor is a silent multiplier if it is wrong.

    Volatility would be scaled by the wrong constant, the crisis threshold would fire at
    the wrong level, and nothing in the output would say so. Refusing is the only honest
    option.
    """
    with pytest.raises(regime_engine.ValidationFailure, match="timeframe"):
        regime_engine.annualisation_factor(None, AssetClass.CRYPTO_SPOT)


def test_a_missing_close_column_is_refused() -> None:
    frame = _prices(400, seed=74).drop(columns=["close"])
    with pytest.raises(regime_engine.ValidationFailure, match="close"):
        _run(frame, _spec())


def test_a_carried_indicator_that_is_absent_is_refused_with_a_direction() -> None:
    """The hybrid needs ADX or RSI; the refusal says where to get them."""
    frame = _prices(400, seed=75)
    with pytest.raises(regime_engine.ValidationFailure) as excinfo:
        _run(frame, _spec(model_type=RegimeModel.HYBRID, features=(RegimeFeature.ADX,)))
    assert "C5" in str(excinfo.value.details.get("hint", ""))


def test_too_little_data_is_refused_before_a_model_is_fitted() -> None:
    frame = _prices(120, seed=76)
    with pytest.raises(regime_engine.ValidationFailure, match="Not enough bars"):
        _run(frame, _spec(min_train_bars=250))


# -------------------------------------------------------------- annualisation


def test_annualisation_factor_follows_timeframe_and_asset_class() -> None:
    assert regime_engine.annualisation_factor(Timeframe.H1, AssetClass.CRYPTO_SPOT) == 24 * 365
    assert regime_engine.annualisation_factor(Timeframe.D1, AssetClass.CRYPTO_SPOT) == 365
    assert regime_engine.annualisation_factor(Timeframe.D1, AssetClass.EQUITY_SPOT) == 252
    weekly = regime_engine.annualisation_factor(Timeframe.W1, AssetClass.CRYPTO_SPOT)
    assert weekly == pytest.approx(365 / 7)


# ------------------------------------------------------------------- stationarity


def test_a_price_level_is_reported_as_non_stationary() -> None:
    """The warning the specification asks for, on the series that always trips it."""
    frame = _prices(600, seed=81)
    matrix = np.column_stack(
        [
            frame["close"].to_numpy(dtype=float),
            regime_engine._log_returns(frame["close"].to_numpy(dtype=float)),
        ]
    )
    warnings = regime_engine.stationarity_warnings(matrix, ["close", "log_return"])
    by_name = {entry["feature"]: entry for entry in warnings}
    assert by_name["close"]["stationary"] is False
    assert by_name["log_return"]["stationary"] is True
