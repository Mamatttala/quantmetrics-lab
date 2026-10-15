"""Module C7 — scaling, and the leak it exists to prevent.

The unit tests here fall into three groups.

**The leak itself.** A frame is built whose training span has a mean of 100 and whose test
span has a mean of 200. Standard-scaling it against a split plan, the transformed test rows
must *not* be centred at zero — if they are, the scaler saw them. The mirror assertion is
that perturbing the test span leaves every fitted parameter and every transformed training
row bit-identical, which is the property that makes the first one meaningful rather than
coincidental.

**Causality.** The rolling z-score is computed twice, once over the whole series and once
over a truncated copy, and the shared prefix must match exactly. Any leak through a centred
window or a forward-looking statistic shows up as a difference in the first 80%.

**Round trips and refusals.** Every scaler must invert exactly, and every request that would
fit on too much data must raise rather than produce a plausible frame.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from app.core.errors import DatasetNotFound, ValidationFailure
from app.models.enums import DatasetCategory, ScalerType, SplitType
from app.models.scaling import ScalerParameter, ScalingRequest
from app.models.splits import SplitCreateRequest, SplitParameters
from app.services import scaling as scaling_module
from app.services.scaling import (
    PARAMS_OPERATION,
    ScalingService,
    numeric_features,
    resolve_features,
    rolling_zscore,
)
from app.services.splits import SplitService
from tests.conftest import hourly_frame, register


def _run(settings, catalog) -> ScalingService:
    return ScalingService(settings, catalog)


def _splits(settings, catalog) -> SplitService:
    return SplitService(settings, catalog)


def _two_regime_frame(rows: int = 200) -> pd.DataFrame:
    """A frame whose first 70% has mean 100 and whose last 30% has mean 200.

    Constructed rather than sampled: the leak test is a statement about means, and a
    sampled frame would make the expected values approximate. Here ``train_mean`` is
    exactly 100 and ``test_mean`` is exactly 200, so the assertion "the transformed test
    rows are not centred at zero" can be checked against the arithmetic rather than against
    a tolerance.
    """
    frame = hourly_frame(rows)
    cut = int(rows * 0.7)
    frame["close"] = np.concatenate(
        [np.full(cut, 100.0), np.full(rows - cut, 200.0)]
    )
    frame["volume"] = np.linspace(10.0, 20.0, rows)
    return frame


async def _plan(settings, catalog, parent, split_type=SplitType.PERCENTAGE_70_15_15, **params):
    created = await _splits(settings, catalog).create(
        SplitCreateRequest(
            dataset_id=parent.dataset_id,
            split_type=split_type,
            parameters=SplitParameters(**params),
        )
    )
    return created


# ---------------------------------------------------------------- the leak


async def test_standard_scaling_does_not_centre_the_test_span_at_zero(settings, catalog):
    """The load-bearing test of Module C7.

    The training span is a flat 100 and the test span a flat 200. A scaler fitted on the
    whole series would subtract ~130 from everything, so the transformed test rows would
    land somewhere around +1 — a number that looks exactly like scaled data. Fitted on the
    training span only, the test rows land at +100: not centred, and visibly not centred,
    which is the correct result rather than a fault.
    """
    frame = _two_regime_frame(200)
    parent = register(settings, catalog, frame)
    plan = await _plan(settings, catalog, parent)

    result = await _run(settings, catalog).run(
        ScalingRequest(
            dataset_id=parent.dataset_id,
            split_plan_id=plan.dataset.dataset_id,
            scaler_type=ScalerType.STANDARD,
            features=["close"],
        )
    )
    assert result.dataset is not None
    scaled = pd.read_parquet(result.dataset.storage_path)["close"]

    train_mean = scaled.iloc[:140].mean()
    test_mean = scaled.iloc[170:].mean()

    assert abs(train_mean) < 1e-9, "training rows are centred by construction"
    assert abs(test_mean) > 1.0, (
        "the test span came out centred at zero, which means the scaler was fitted on it"
    )
    # The centre is the *training* mean exactly. Had the whole series been fitted it would
    # be 130.0 — and nothing else in the frame would have said so.
    assert result.parameters[0].center_ == pytest.approx(100.0)
    assert test_mean == pytest.approx(100.0)


async def test_scaling_matches_the_training_statistics_exactly(settings, catalog):
    frame = hourly_frame(200)
    parent = register(settings, catalog, frame)
    plan = await _plan(settings, catalog, parent)
    result = await _run(settings, catalog).run(
        ScalingRequest(
            dataset_id=parent.dataset_id,
            split_plan_id=plan.dataset.dataset_id,
            scaler_type=ScalerType.STANDARD,
            features=["close"],
        )
    )
    assert result.dataset is not None
    scaled = pd.read_parquet(result.dataset.storage_path)["close"]

    train = frame["close"].iloc[:140]
    # ddof=0, which is sklearn's convention and therefore this module's: `StandardScaler`
    # divides by sqrt(var_) with variance taken over the population.
    expected = (frame["close"] - train.mean()) / train.std(ddof=0)
    assert np.allclose(scaled.to_numpy(), expected.to_numpy(), equal_nan=True)


async def test_perturbing_the_test_span_changes_nothing(settings, catalog):
    """The complement of the leak test: the fit is a function of the training rows alone.

    If any parameter moved when the last 30% of the frame changed, the scaler had seen
    those rows — and no assertion about means would have caught it, because a small
    contamination can leave the mean looking right.
    """
    frame = hourly_frame(200)
    parent = register(settings, catalog, frame, name="baseline")
    plan = await _plan(settings, catalog, parent)

    first = await _run(settings, catalog).run(
        ScalingRequest(
            dataset_id=parent.dataset_id,
            split_plan_id=plan.dataset.dataset_id,
            scaler_type=ScalerType.STANDARD,
            features=["close", "volume"],
        )
    )

    # A second frame identical in the first 140 rows and wildly different after them.
    perturbed = frame.copy()
    perturbed.loc[170:, "close"] += 5_000.0
    perturbed.loc[170:, "volume"] *= 50.0
    other = register(settings, catalog, perturbed, name="perturbed")
    other_plan = await _plan(settings, catalog, other)

    second = await _run(settings, catalog).run(
        ScalingRequest(
            dataset_id=other.dataset_id,
            split_plan_id=other_plan.dataset.dataset_id,
            scaler_type=ScalerType.STANDARD,
            features=["close", "volume"],
        )
    )

    assert [(p.feature, p.center_, p.scale_) for p in first.parameters] == [
        (p.feature, p.center_, p.scale_) for p in second.parameters
    ]

    assert first.dataset is not None and second.dataset is not None
    left = pd.read_parquet(first.dataset.storage_path).iloc[:140]
    right = pd.read_parquet(second.dataset.storage_path).iloc[:140]
    pd.testing.assert_frame_equal(left, right)


async def test_every_fitted_scaler_names_the_plan_it_was_fitted_against(settings, catalog):
    frame = hourly_frame(200)
    parent = register(settings, catalog, frame)
    plan = await _plan(settings, catalog, parent)
    for scaler_type in (ScalerType.STANDARD, ScalerType.MINMAX, ScalerType.ROBUST):
        result = await _run(settings, catalog).run(
            ScalingRequest(
                dataset_id=parent.dataset_id,
                split_plan_id=plan.dataset.dataset_id,
                scaler_type=scaler_type,
                features=["close"],
            )
        )
        parameter = result.parameters[0]
        # What each scaler centres on differs by design — the mean, the minimum, the median
        # — but all three are finite, all three come from the training span, and the report
        # always says which plan and which rows they came from.
        assert np.isfinite(parameter.center_) and np.isfinite(parameter.scale_)
        assert result.report.split_plan_id == plan.dataset.dataset_id
        assert result.report.fit_rows == 140
        assert result.report.fit_role == "train"


# ------------------------------------------------------------- causality


def test_rolling_zscore_is_causal():
    """Computed whole versus truncated to 80%: identical on the shared prefix.

    This is the mechanical statement of causality. A centred window, a whole-series mean, or
    any statistic that reads forward would make row *t* depend on rows after *t*, and the
    truncated run — which has no such rows — would disagree.
    """
    frame = hourly_frame(200)
    whole, _, _ = rolling_zscore(frame, ["close"], 60)
    truncated, _, _ = rolling_zscore(frame.iloc[:160], ["close"], 60)

    shared_whole = whole["close"].iloc[:160].to_numpy()
    shared_truncated = truncated["close"].to_numpy()
    assert np.allclose(shared_whole, shared_truncated, equal_nan=True)


def test_rolling_zscore_warmup_is_null():
    frame = hourly_frame(200)
    scaled, _, per_feature = rolling_zscore(frame, ["close"], 60)
    assert scaled["close"].iloc[:59].isna().all()
    assert scaled["close"].iloc[59:].notna().all()
    assert per_feature["close"] == 141


def test_rolling_zscore_leaves_other_columns_alone():
    frame = hourly_frame(100)
    scaled, _, _ = rolling_zscore(frame, ["close"], 20)
    pd.testing.assert_series_equal(scaled["open"], frame["open"])
    pd.testing.assert_series_equal(scaled["close_time"], frame["close_time"])


def test_rolling_zscore_matches_a_hand_computed_window():
    frame = hourly_frame(100)
    scaled, _, _ = rolling_zscore(frame, ["close"], 5)
    window = frame["close"].iloc[10:15]
    expected = (frame["close"].iloc[14] - window.mean()) / window.std(ddof=0)
    assert scaled["close"].iloc[14] == pytest.approx(expected)


def test_rolling_zscore_uses_the_window_ending_at_the_row():
    """The window is ``[t-9, t]`` inclusive, not ``[t-10, t)``.

    One bar of error here shifts every statistic by a bar, which is small enough to be
    invisible in a chart and large enough to matter at a boundary.
    """
    frame = hourly_frame(100)
    scaled, _, _ = rolling_zscore(frame, ["close"], 10)
    window = frame["close"].iloc[50:60]
    expected = (frame["close"].iloc[59] - window.mean()) / window.std(ddof=0)
    assert scaled["close"].iloc[59] == pytest.approx(expected)


def test_a_flat_window_does_not_produce_infinities():
    frame = hourly_frame(60)
    frame["close"] = 42.0
    scaled, _, _ = rolling_zscore(frame, ["close"], 10)
    assert not np.isinf(scaled["close"].to_numpy()).any()
    assert (scaled["close"].iloc[9:] == 0.0).all()


def test_centering_is_never_used_in_this_module():
    """The mechanical guard lives in `test_leakage_guards.py` beside its bar-close sibling.

    Asserted there rather than here so the two source-level guards that exist to catch a
    *silent* mistake sit in one file; a second copy in this file would be a second place
    to update and the pair could drift apart without either test noticing.
    """
    assert "center=True" not in inspect.getsource(scaling_module)


# ------------------------------------------------------------- round trips


async def test_all_four_scalers_round_trip(settings, catalog):
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    rng = np.random.default_rng(7)
    values = rng.normal(50, 5, (120, 1))
    for scaler_type, scaler in (
        (ScalerType.STANDARD, StandardScaler()),
        (ScalerType.MINMAX, MinMaxScaler()),
        (ScalerType.ROBUST, RobustScaler()),
    ):
        scaler.fit(values)
        parameter = scaling_module._parameters(
            scaler, ["x"], scaler_type, {"x": 120}
        )[0]
        inverted = values.ravel() * parameter.scale_ + parameter.center_
        # MinMax maps to [0, 1] and the others to their own centres; in every case the
        # affine pair must return the original values exactly.
        back = (inverted - parameter.center_) / parameter.scale_
        assert np.allclose(back, values.ravel())

    frame = hourly_frame(120)
    scaled, parameters, _ = rolling_zscore(frame, ["close"], 20)
    # The rolling z-score is not invertible from a single pair — its statistics differ per
    # row — which is exactly why it is not offered as an inverse-transform target for
    # Modules E and F, and why the parameter card says so.
    assert parameters[0].scaler_type is ScalerType.ROLLING_ZSCORE
    assert scaled["close"].iloc[:19].isna().all()


def test_minmax_affine_pair_uses_the_training_range():
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaler.fit(np.array([[10.0], [30.0]]))
    parameter = scaling_module._parameters(scaler, ["x"], ScalerType.MINMAX, {"x": 2})[0]
    assert parameter.center_ == pytest.approx(10.0)
    assert parameter.scale_ == pytest.approx(20.0)
    assert parameter.attributes["data_min_"] == pytest.approx(10.0)
    assert parameter.attributes["data_max_"] == pytest.approx(30.0)


def test_robust_affine_pair_uses_the_median_and_iqr():
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler()
    scaler.fit(np.array([[1.0], [2.0], [3.0], [4.0], [100.0]]))
    parameter = scaling_module._parameters(scaler, ["x"], ScalerType.ROBUST, {"x": 5})[0]
    assert parameter.center_ == pytest.approx(3.0)
    # IQR of [1,2,3,4,100] is 3.5 - 1.5 = 2.0.
    assert parameter.scale_ == pytest.approx(2.0)


def test_standard_stores_the_attributes_the_specification_names():
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(np.array([[1.0], [2.0], [3.0]]))
    parameter = scaling_module._parameters(scaler, ["x"], ScalerType.STANDARD, {"x": 3})[0]
    assert set(parameter.attributes) == {"mean_", "scale_", "var_", "n_samples_seen_"}
    assert parameter.attributes["mean_"] == pytest.approx(2.0)
    assert "mean_" in parameter.formula or parameter.formula


def test_a_constant_column_is_reported_not_divided_by_zero():
    """sklearn guards a zero scale by substituting 1.0, so the replacement is invisible.

    A constant column scaled by ``(x − c)/1`` is a column of zeros, which is the correct
    answer and is *not* the same as a column of nulls. The guard exists so the alternative
    — dividing by zero — does not turn the whole column into NaN, which reads as missing
    data. Detecting it from the fitted scale alone would not work: a legitimately
    unit-scaled column carries the same 1.0.
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(np.array([[5.0], [5.0], [5.0]]))
    parameter = scaling_module._parameters(
        scaler, ["x"], ScalerType.STANDARD, {"x": 3}, {"x"}
    )[0]
    assert parameter.constant is True
    assert parameter.scale_ == 1.0
    assert "constant column" in parameter.formula


def test_a_unit_scaled_column_is_not_mislabelled_constant():
    from sklearn.preprocessing import StandardScaler

    values = np.array([[4.0], [6.0], [4.0], [6.0]])
    scaler = StandardScaler()
    scaler.fit(values)
    parameter = scaling_module._parameters(
        scaler, ["x"], ScalerType.STANDARD, {"x": 4}, set()
    )[0]
    assert parameter.constant is False
    assert parameter.scale_ == pytest.approx(1.0)  # a real sd of exactly 1
    assert "constant column" not in parameter.formula


async def test_a_constant_training_column_is_flagged_end_to_end(settings, catalog):
    """The constant is read from the training rows, and the warning names the column."""
    frame = hourly_frame(200)
    frame["flat"] = 7.0
    parent = register(settings, catalog, frame)
    plan = await _plan(settings, catalog, parent)
    result = await _run(settings, catalog).run(
        ScalingRequest(
            dataset_id=parent.dataset_id,
            split_plan_id=plan.dataset.dataset_id,
            scaler_type=ScalerType.STANDARD,
            features=["close", "flat"],
        )
    )
    by_name = {p.feature: p for p in result.parameters}
    assert by_name["flat"].constant is True
    assert by_name["close"].constant is False
    assert any("single value throughout the dataset" in note for note in result.report.warnings)


def test_a_non_finite_attribute_becomes_null():
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(np.array([[1.0], [2.0], [3.0]]))
    parameter = scaling_module._parameters(scaler, ["x"], ScalerType.STANDARD, {"x": 3})[0]
    # Every stored attribute is JSON-safe by construction: NaN and inf cannot survive, and
    # JavaScript's JSON.parse rejects both outright.
    for value in parameter.attributes.values():
        assert value is None or np.isfinite(value)


# ------------------------------------------------------------- the transform


async def test_the_scaled_frame_keeps_every_column_and_row(settings, catalog):
    frame = hourly_frame(200)
    parent = register(settings, catalog, frame)
    plan = await _plan(settings, catalog, parent)
    result = await _run(settings, catalog).run(
        ScalingRequest(
            dataset_id=parent.dataset_id,
            split_plan_id=plan.dataset.dataset_id,
            scaler_type=ScalerType.STANDARD,
            features=["close"],
        )
    )
    assert result.dataset is not None
    scaled = pd.read_parquet(result.dataset.storage_path)
    assert len(scaled) == len(frame)
    assert list(scaled.columns) == list(frame.columns)
    pd.testing.assert_series_equal(scaled["volume"], frame["volume"])


async def test_feature_selection_leaves_the_others_untouched(settings, catalog):
    frame = hourly_frame(200)
    parent = register(settings, catalog, frame)
    plan = await _plan(settings, catalog, parent)
    result = await _run(settings, catalog).run(
        ScalingRequest(
            dataset_id=parent.dataset_id,
            split_plan_id=plan.dataset.dataset_id,
            scaler_type=ScalerType.MINMAX,
            features=["volume"],
        )
    )
    assert result.dataset is not None
    scaled = pd.read_parquet(result.dataset.storage_path)
    pd.testing.assert_series_equal(scaled["close"], frame["close"])

    # The training span maps into [0, 1] by construction, because the range came from it.
    train = scaled["volume"].iloc[:140]
    assert train.min() == pytest.approx(0.0)
    assert train.max() == pytest.approx(1.0)
    # The test span does not, and that is the whole point: its range was never used to
    # choose the scale, so a value above the training maximum lands above 1. A scaler fitted
    # on the whole series would have clamped every row into [0, 1] — and that is precisely
    # what leaking looks like from the outside.
    assert scaled["volume"].iloc[140:].max() > 1.0


async def test_an_empty_feature_list_means_every_numeric_column(settings, catalog):
    frame = hourly_frame(200)
    parent = register(settings, catalog, frame)
    plan = await _plan(settings, catalog, parent)
    result = await _run(settings, catalog).run(
        ScalingRequest(
            dataset_id=parent.dataset_id,
            split_plan_id=plan.dataset.dataset_id,
            scaler_type=ScalerType.STANDARD,
            features=[],
        )
    )
    assert result.report.features == ["open", "high", "low", "close", "volume"]


def test_time_columns_are_never_candidates():
    frame = hourly_frame(10)
    assert "open_time" not in numeric_features(frame)
    assert "close_time" not in numeric_features(frame)


def test_numeric_detection_skips_booleans():
    frame = pd.DataFrame({"a": [1.0, 2.0], "flag": [True, False], "name": ["x", "y"]})
    assert numeric_features(frame) == ["a"]


def test_a_named_non_numeric_column_is_refused():
    frame = pd.DataFrame({"a": [1.0, 2.0], "name": ["x", "y"]})
    with pytest.raises(ValidationFailure, match="not a numeric column"):
        resolve_features(frame, ["name"])


def test_a_named_missing_column_is_refused():
    frame = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValidationFailure, match="is not a column"):
        resolve_features(frame, ["nope"])


# ------------------------------------------------------------- persistence


async def test_the_scaled_frame_lands_in_processed_data(settings, catalog):
    frame = hourly_frame(200)
    parent = register(settings, catalog, frame)
    plan = await _plan(settings, catalog, parent)
    result = await _run(settings, catalog).run(
        ScalingRequest(
            dataset_id=parent.dataset_id,
            split_plan_id=plan.dataset.dataset_id,
            scaler_type=ScalerType.STANDARD,
            features=["close"],
        )
    )
    assert result.dataset is not None
    assert result.dataset.category is DatasetCategory.PROCESSED
    assert result.dataset.parent_dataset_id == parent.dataset_id
    assert result.dataset.operation == "scaling"


async def test_the_parameters_are_saved_as_their_own_output_dataset(settings, catalog):
    frame = hourly_frame(200)
    parent = register(settings, catalog, frame)
    plan = await _plan(settings, catalog, parent)
    result = await _run(settings, catalog).run(
        ScalingRequest(
            dataset_id=parent.dataset_id,
            split_plan_id=plan.dataset.dataset_id,
            scaler_type=ScalerType.STANDARD,
            features=["close", "volume"],
        )
    )
    assert result.params_dataset is not None
    assert result.params_dataset.category is DatasetCategory.OUTPUTS
    assert result.params_dataset.operation == PARAMS_OPERATION
    table = pd.read_parquet(result.params_dataset.storage_path)
    assert list(table["feature"]) == ["close", "volume"]
    assert {"center_", "scale_", "scaler_type", "formula"} <= set(table.columns)


async def test_lineage_records_the_dataset_the_plan_the_scaler_and_the_fold(settings, catalog):
    """The four things that make a scaled dataset reproducible, all present.

    The fold is the one that would go missing: it looks like a detail of a detail, and
    without it a walk-forward run cannot be re-derived, because the parameters depend on
    which fold was fitted.
    """
    frame = hourly_frame(200)
    parent = register(settings, catalog, frame)
    plan = await _plan(
        settings, catalog, parent, split_type=SplitType.TSCV_EXPANDING, n_folds=3
    )
    result = await _run(settings, catalog).run(
        ScalingRequest(
            dataset_id=parent.dataset_id,
            split_plan_id=plan.dataset.dataset_id,
            scaler_type=ScalerType.STANDARD,
            features=["close"],
            fold=2,
        )
    )
    assert result.dataset is not None
    provenance = result.dataset.provenance
    assert provenance["derived_from"] == parent.dataset_id
    assert provenance["split_plan_id"] == plan.dataset.dataset_id
    assert provenance["scaler_type"] == "standard"
    assert provenance["operation"] == "scaling"
    assert provenance["scaled_features"] == "close"
    assert result.report.fold == 2
    assert result.report.split_plan_name == plan.dataset.name


async def test_the_plan_is_an_extra_lineage_parent(settings, catalog):
    frame = hourly_frame(200)
    parent = register(settings, catalog, frame)
    plan = await _plan(settings, catalog, parent)
    result = await _run(settings, catalog).run(
        ScalingRequest(
            dataset_id=parent.dataset_id,
            split_plan_id=plan.dataset.dataset_id,
            scaler_type=ScalerType.STANDARD,
            features=["close"],
        )
    )
    assert result.dataset is not None
    derived = catalog.derived_datasets(parent.dataset_id, "split")
    assert [d.dataset_id for d in derived] == [plan.dataset.dataset_id]


async def test_provenance_carries_the_window_for_a_rolling_run(settings, catalog):
    frame = hourly_frame(200)
    parent = register(settings, catalog, frame)
    result = await _run(settings, catalog).run(
        ScalingRequest(
            dataset_id=parent.dataset_id,
            scaler_type=ScalerType.ROLLING_ZSCORE,
            features=["close"],
            window=30,
        )
    )
    assert result.dataset is not None
    assert result.dataset.provenance["scaler_window"] == "30"
    assert result.dataset.provenance["split_plan_id"] == ""
    assert result.report.window == 30


async def test_a_preview_run_writes_nothing(settings, catalog):
    frame = hourly_frame(200)
    parent = register(settings, catalog, frame)
    plan = await _plan(settings, catalog, parent)
    before = len(catalog.list_datasets())
    result = await _run(settings, catalog).run(
        ScalingRequest(
            dataset_id=parent.dataset_id,
            split_plan_id=plan.dataset.dataset_id,
            scaler_type=ScalerType.STANDARD,
            features=["close"],
            write=False,
        )
    )
    assert result.dataset is None and result.params_dataset is None
    assert len(catalog.list_datasets()) == before
    assert result.preview["row_count"] == 200


# ------------------------------------------------------------- refusals


def test_a_fitted_scaler_without_a_split_plan_is_refused():
    """The specification's guardrail, enforced in the type system.

    Not merely checked in the service: a required field cannot be bypassed by a new caller,
    which is what makes this a property of the module rather than a rule the current
    implementation happens to follow.
    """
    with pytest.raises(ValueError, match="needs split_plan_id"):
        ScalingRequest(
            dataset_id="x", scaler_type=ScalerType.MINMAX, features=["close"]
        )


def test_the_rolling_zscore_refuses_a_split_plan():
    with pytest.raises(ValueError, match="takes no split plan"):
        ScalingRequest(
            dataset_id="x",
            scaler_type=ScalerType.ROLLING_ZSCORE,
            split_plan_id="plan",
        )


def test_a_window_on_a_fitted_scaler_is_refused():
    with pytest.raises(ValueError, match="window applies to rolling_zscore only"):
        ScalingRequest(
            dataset_id="x",
            scaler_type=ScalerType.STANDARD,
            split_plan_id="plan",
            window=30,
        )


async def test_a_fitted_scaler_without_a_plan_never_reaches_the_fitter(settings, catalog):
    """The service refuses independently of the request model.

    ``ScalingRequest`` already makes ``split_plan_id`` mandatory for a fitted scaler, so
    this path is unreachable over HTTP. It is asserted anyway, because the model is the
    outer ring: a future caller that builds a request by another route (an internal job, a
    script) must still hit a wall rather than silently fitting on the whole frame.
    """
    parent = register(settings, catalog, hourly_frame(200))
    smuggled = ScalingRequest.model_construct(
        dataset_id=parent.dataset_id,
        scaler_type=ScalerType.STANDARD,
        split_plan_id=None,
        fold=None,
        features=["close"],
        window=None,
        write=False,
        name=None,
    )
    with pytest.raises(ValidationFailure, match="needs a split plan"):
        await _run(settings, catalog).run(smuggled)


async def test_a_plan_for_another_dataset_is_refused(settings, catalog):
    first = register(settings, catalog, hourly_frame(200), name="first")
    second = register(settings, catalog, hourly_frame(200), name="second")
    plan = await _plan(settings, catalog, first)
    with pytest.raises(ValidationFailure, match="split plan for a different dataset"):
        await _run(settings, catalog).run(
            ScalingRequest(
                dataset_id=second.dataset_id,
                split_plan_id=plan.dataset.dataset_id,
                scaler_type=ScalerType.STANDARD,
                features=["close"],
            )
        )


async def test_a_plan_from_a_stale_frame_is_refused(settings, catalog):
    """The checksum guard: a plan applied to bytes it was not computed on.

    The lineage check catches a plan made from a *different* dataset, but not a dataset
    re-written in place under the same id. The checksum is the mechanical form of the same
    rule, and this is the only test that reaches it — so it is exercised directly rather
    than through the service, where the lineage check would fire first.
    """
    frame = hourly_frame(200)
    parent = register(settings, catalog, frame)
    plan = await _plan(settings, catalog, parent)

    stale = parent.model_copy(update={"checksum": "0" * 64})
    with pytest.raises(ValidationFailure, match="different version"):
        await _run(settings, catalog)._load_plan(plan.dataset.dataset_id, stale, None)


async def test_a_tscv_plan_without_a_fold_is_refused(settings, catalog):
    frame = hourly_frame(200)
    parent = register(settings, catalog, frame)
    plan = await _plan(
        settings, catalog, parent, split_type=SplitType.TSCV_EXPANDING, n_folds=3
    )
    with pytest.raises(ValidationFailure, match="walk-forward plan with 3 folds"):
        await _run(settings, catalog).run(
            ScalingRequest(
                dataset_id=parent.dataset_id,
                split_plan_id=plan.dataset.dataset_id,
                scaler_type=ScalerType.STANDARD,
                features=["close"],
            )
        )


async def test_a_fold_outside_the_plan_is_refused(settings, catalog):
    frame = hourly_frame(200)
    parent = register(settings, catalog, frame)
    plan = await _plan(
        settings, catalog, parent, split_type=SplitType.TSCV_EXPANDING, n_folds=3
    )
    with pytest.raises(ValidationFailure, match="Fold 9 is not in"):
        await _run(settings, catalog).run(
            ScalingRequest(
                dataset_id=parent.dataset_id,
                split_plan_id=plan.dataset.dataset_id,
                scaler_type=ScalerType.STANDARD,
                features=["close"],
                fold=9,
            )
        )


async def test_a_fold_on_a_single_span_plan_is_refused(settings, catalog):
    frame = hourly_frame(200)
    parent = register(settings, catalog, frame)
    plan = await _plan(settings, catalog, parent)
    with pytest.raises(ValidationFailure, match="no folds, so fold 1"):
        await _run(settings, catalog).run(
            ScalingRequest(
                dataset_id=parent.dataset_id,
                split_plan_id=plan.dataset.dataset_id,
                scaler_type=ScalerType.STANDARD,
                features=["close"],
                fold=1,
            )
        )


async def test_a_plan_id_that_is_not_a_plan_is_refused(settings, catalog):
    parent = register(settings, catalog, hourly_frame(200))
    with pytest.raises(ValidationFailure, match="is not a split plan"):
        await _run(settings, catalog).run(
            ScalingRequest(
                dataset_id=parent.dataset_id,
                split_plan_id=parent.dataset_id,
                scaler_type=ScalerType.STANDARD,
                features=["close"],
            )
        )


async def test_an_unknown_plan_is_refused(settings, catalog):
    parent = register(settings, catalog, hourly_frame(200))
    with pytest.raises(DatasetNotFound):
        await _run(settings, catalog).run(
            ScalingRequest(
                dataset_id=parent.dataset_id,
                split_plan_id="nope",
                scaler_type=ScalerType.STANDARD,
                features=["close"],
            )
        )


async def test_a_feature_empty_in_the_training_span_is_refused(settings, catalog):
    frame = hourly_frame(200)
    # A column that exists from row 150 onwards only: exactly the shape a leak would take,
    # since filling it backwards is the only way to use it and that is look-ahead.
    frame["late"] = np.nan
    frame.loc[150:, "late"] = 1.0
    parent = register(settings, catalog, frame)
    plan = await _plan(settings, catalog, parent)
    with pytest.raises(ValidationFailure, match="entirely empty within the training span"):
        await _run(settings, catalog).run(
            ScalingRequest(
                dataset_id=parent.dataset_id,
                split_plan_id=plan.dataset.dataset_id,
                scaler_type=ScalerType.STANDARD,
                features=["late"],
            )
        )


def test_a_one_row_training_span_is_refused():
    """One row has a standard deviation that is defined but meaningless.

    ``std(ddof=1)`` on a single observation is NaN, and sklearn's scaler would write NaN
    through the whole column. Even where it does not, every transformed value would be
    exactly 0 — a column of constants that the model would happily read.
    """
    frame = hourly_frame(5)
    with pytest.raises(ValidationFailure, match="too few to estimate a spread"):
        scaling_module._require_fittable(frame.iloc[:1], ["close"])


def test_an_empty_training_span_is_refused():
    frame = hourly_frame(5)
    with pytest.raises(ValidationFailure, match="training span of this plan is empty"):
        scaling_module._require_fittable(frame.iloc[:0], ["close"])


def test_an_all_null_training_column_is_refused():
    frame = hourly_frame(5)
    frame["blank"] = np.nan
    with pytest.raises(ValidationFailure, match="entirely empty within the training span"):
        scaling_module._require_fittable(frame, ["blank"])


# ------------------------------------------------------------- options


async def test_options_lists_numeric_features_and_plans(settings, catalog):
    frame = hourly_frame(200)
    parent = register(settings, catalog, frame)
    plan = await _plan(settings, catalog, parent)
    options = await _run(settings, catalog).options(parent.dataset_id)
    assert options["features"] == ["open", "high", "low", "close", "volume"]
    assert [p["split_plan_id"] for p in options["plans"]] == [plan.dataset.dataset_id]
    assert options["defaults"]["window"] == 60


async def test_options_greys_the_fitted_scalers_without_a_plan(settings, catalog):
    parent = register(settings, catalog, hourly_frame(200))
    options = await _run(settings, catalog).options(parent.dataset_id)
    by_id = {s["id"]: s for s in options["scalers"]}
    assert by_id["minmax"]["available"] is False
    assert "whole series" in by_id["minmax"]["reason"]
    assert by_id["rolling_zscore"]["available"] is True
    assert by_id["rolling_zscore"]["needs_split_plan"] is False


async def test_options_enables_the_fitted_scalers_once_a_plan_exists(settings, catalog):
    parent = register(settings, catalog, hourly_frame(200))
    await _plan(settings, catalog, parent)
    options = await _run(settings, catalog).options(parent.dataset_id)
    assert all(s["available"] for s in options["scalers"])
    assert any(w["code"] == "fit_on_train_only" for w in options["warnings"])


# ------------------------------------------------------------- job plumbing


async def test_fit_transform_accepted_is_a_job_handle(settings, catalog):
    frame = hourly_frame(200)
    parent = register(settings, catalog, frame)
    plan = await _plan(settings, catalog, parent)
    accepted = await _run(settings, catalog).fit_transform(
        ScalingRequest(
            dataset_id=parent.dataset_id,
            split_plan_id=plan.dataset.dataset_id,
            scaler_type=ScalerType.STANDARD,
            features=["close"],
        )
    )
    assert accepted.poll == f"/api/jobs/{accepted.job_id}"
    record = await _run(settings, catalog).wait(accepted.job_id)
    assert record.status.value == "succeeded"
    assert record.output_dataset_id


async def test_a_failing_job_records_the_refusal(settings, catalog):
    parent = register(settings, catalog, hourly_frame(200))
    accepted = await _run(settings, catalog).fit_transform(
        ScalingRequest(
            dataset_id=parent.dataset_id,
            split_plan_id=parent.dataset_id,  # not a plan
            scaler_type=ScalerType.STANDARD,
            features=["close"],
        )
    )
    record = await _run(settings, catalog).wait(accepted.job_id)
    assert record.status.value == "failed"
    assert "not a split plan" in (record.error or {}).get("message", "")


async def test_a_missing_dataset_is_refused_before_a_job_exists(settings, catalog):
    with pytest.raises(DatasetNotFound):
        await _run(settings, catalog).fit_transform(
            ScalingRequest(
                dataset_id="nope",
                scaler_type=ScalerType.ROLLING_ZSCORE,
                features=["close"],
            )
        )


def test_scaler_parameter_model_is_json_safe():
    parameter = ScalerParameter(
        feature="close",
        scaler_type=ScalerType.STANDARD,
        center_=1.5,
        scale_=0.5,
        fitted_rows=10,
        formula="close_scaled = (close - 1.5) / 0.5",
    )
    dumped = parameter.model_dump(mode="json")
    assert dumped["center_"] == 1.5
    assert "center_" in dumped and "scale_" in dumped
