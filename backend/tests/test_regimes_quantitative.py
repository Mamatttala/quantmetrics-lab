"""Phase 2 audit regressions: return conventions, occupancy, PELT and alignment."""

import itertools
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from app.core.errors import ValidationFailure
from app.services.regimes import metrics
from app.services.regimes.artifacts import Store
from app.services.regimes.contracts import Experiment, Inputs, ValidationRequest
from app.services.regimes.data import load
from app.services.regimes.evaluation import evaluate, validate
from app.services.regimes.experiments import execute
from app.services.regimes.structural import pelt
from tests.test_regimes_workflow import config_of
from tests.test_regimes_workflow import prepared as prepared_fixture

prepared = prepared_fixture


@pytest.mark.parametrize(
    "case", ["valid", "different_asset", "unrelated", "no_overlap", "unassigned"]
)
def test_f11_alignment_requires_matched_observations_and_valid_assignments(monkeypatch, case):
    from app.services.regimes import evaluation

    times = pd.Series(pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC"))
    reference = SimpleNamespace(
        meta=SimpleNamespace(dataset_id="reference"),
        times=times,
        x=np.arange(6)[:, None],
        contract={
            "asset": "BTCUSDT",
            "timeframe": "1h",
            "preprocessing_signature": "reference-fit",
        },
    )
    fold = SimpleNamespace(
        meta=SimpleNamespace(dataset_id="fold"),
        times=times + pd.Timedelta(hours=1),
        x=(100 + np.arange(6))[:, None],
        spans={"train": np.arange(4)},
        contract=reference.contract | {"preprocessing_signature": "fold-fit"},
    )
    if case == "different_asset":
        fold.contract["asset"] = "ETHUSDT"
    if case == "no_overlap":
        fold.times += pd.Timedelta(days=10)
    catalog = SimpleNamespace(
        ancestors=lambda identifier: [] if case == "unrelated" else ["source"]
    )

    def own_prediction(x):
        np.testing.assert_array_equal(x, reference.x)
        return {"states": np.array([-1, 0, 1, 0, 1, 0])}

    model = SimpleNamespace(count=2, predict=own_prediction)
    output = {"states": np.array([-1] * 6 if case == "unassigned" else [1, 0, 1, 0, -1, 1])}
    result = evaluation.fold_alignment(
        model, reference, SimpleNamespace(count=2), fold, output, catalog
    )
    if case == "valid":
        assert result["aligned_agreement"] == 1
        assert result["observations"] == 4
        assert result["mapping"] == {0: 1, 1: 0}
    else:
        assert "unavailable" in result
        assert "aligned_agreement" not in result


@pytest.mark.parametrize("mode", ["expanding", "rolling"])
@pytest.mark.parametrize("model_name", ["kmeans", "gaussian_hmm"])
async def test_f11_fold_scalers_never_enter_reference_model(
    settings, catalog, prepared, monkeypatch, mode, model_name
):
    from app.models.scaling import ScalingRequest
    from app.models.splits import SplitCreateRequest, SplitParameters
    from app.services.regimes import evaluation
    from app.services.regimes.estimators import Model
    from app.services.scaling import ScalingService
    from app.services.splits import SplitService

    original, _, _ = prepared
    config = config_of(prepared, model_name)
    store = Store(settings, catalog)
    identifier = execute(config, settings, catalog, store)["selected_experiment_id"]
    reference = load(catalog, settings, config.inputs)
    baseline = store.model(identifier).predict(reference.x)["states"]
    plan = await SplitService(settings, catalog).create(
        SplitCreateRequest(
            dataset_id=original.dataset_id,
            split_type=f"tscv_{mode}",
            parameters=SplitParameters(n_folds=3),
        )
    )
    datasets = {}
    from app.services.regimes.data import plan_rows

    for fold in sorted(
        {row.fold for row in plan_rows(catalog, settings, plan.dataset.dataset_id)[1]}
    ):
        scaled = await ScalingService(settings, catalog).run(
            ScalingRequest(
                dataset_id=original.dataset_id,
                split_plan_id=plan.dataset.dataset_id,
                fold=fold,
                scaler_type="standard",
                features=["signal", "second"],
            )
        )
        datasets[fold] = scaled.dataset.dataset_id

    original_reopen, original_predict, original_fit = (
        evaluation.reopen,
        Model.predict,
        evaluation.fit,
    )
    seen = []
    reference_calls = []

    def reopen_reference(*args, **kwargs):
        artifact, conf, data, model = original_reopen(*args, **kwargs)
        model._f11_reference = True
        return artifact, conf, data, model

    def guard_coordinates(self, x, *args, **kwargs):
        if getattr(self, "_f11_reference", False):
            # Full original history is also essential for HMM filtering context.
            np.testing.assert_array_equal(x, reference.x)
            reference_calls.append(len(x))
        return original_predict(self, x, *args, **kwargs)

    def capture_fold(conf, data):
        fitted = original_fit(conf, data)
        seen.append((data, fitted))
        return fitted

    monkeypatch.setattr(evaluation, "reopen", reopen_reference)
    monkeypatch.setattr(Model, "predict", guard_coordinates)
    monkeypatch.setattr(evaluation, "fit", capture_fold)
    result = validate(
        identifier,
        ValidationRequest(
            mode=mode,
            split_plan_id=plan.dataset.dataset_id,
            prepared_datasets=datasets,
        ),
        settings,
        catalog,
        store,
    )
    assert reference_calls
    assert len(seen) == len(result["folds"]) == 3
    for row, (data, fitted) in zip(result["folds"], seen, strict=True):
        assert (
            data.contract["preprocessing_signature"]
            != reference.contract["preprocessing_signature"]
        )
        assert not np.allclose(data.x, reference.x)
        train = data.spans["train"]
        expected = metrics.align(baseline[train], fitted.predict(data.x)["states"][train])
        alignment = row["state_alignment"]
        assert alignment["aligned_agreement"] == pytest.approx(expected["aligned_agreement"])
        assert alignment["adjusted_rand"] == pytest.approx(expected["adjusted_rand"])
        assert alignment["observations"] == len(train)
        assert (
            alignment["reference_preprocessing_signature"]
            == reference.contract["preprocessing_signature"]
        )
        assert alignment["fold_preprocessing_signature"] == data.contract["preprocessing_signature"]
        assert row["train_end"] < row["test_start"]


def independent_sse(x):
    # Two-pass, local-origin arithmetic and compensated sums, independent of PELT.
    total = 0.0
    for column in x.T:
        centered = [float(v - column[0]) for v in column]
        mean = math.fsum(centered) / len(centered)
        total += math.fsum((v - mean) ** 2 for v in centered)
    return total


def exhaustive_pelt(x, penalty, minimum):
    best = np.inf
    for flags in itertools.product([False, True], repeat=len(x) - 1):
        edges = [0, *[i + 1 for i, flag in enumerate(flags) if flag], len(x)]
        if min(np.diff(edges)) < minimum:
            continue
        value = math.fsum(
            independent_sse(x[a:b]) for a, b in zip(edges[:-1], edges[1:], strict=True)
        )
        best = min(best, value + penalty * (len(edges) - 2))
    return best


@pytest.mark.parametrize("offset", [0.0, 1e12, -1e12])
def test_f9_segment_sse_survives_large_translation(offset):
    x = np.array([0, 1, 2, 0, 1, 2], dtype=float)[:, None] + offset
    states, changes, score = pelt(x, penalty=7, minimum=len(x))
    assert score == pytest.approx(4.0, rel=1e-13, abs=1e-13)
    assert changes == []
    assert states.tolist() == [0] * len(x)


@pytest.mark.parametrize("minimum", [1, 2, 3])
@pytest.mark.parametrize("offset", [0.0, 1e12, -1e12])
def test_f9_shifted_segmentation_matches_exhaustive_partitions(minimum, offset):
    # Dyadic values remain exactly representable after the 1e12 shift.
    x = np.array(
        [
            [0, 1],
            [0.25, 0],
            [-0.5, 1],
            [5, -3],
            [5.5, -2],
            [4.75, -3],
            [-1, 2],
            [-0.5, 1],
            [-1.25, 2],
        ],
        dtype=float,
    )
    shifted = x + np.array([offset, -offset])
    penalty = 2.5
    states, changes, score = pelt(shifted, penalty, minimum)
    expected = exhaustive_pelt(shifted, penalty, minimum)
    # 128 eps per row permits accumulated binary64 moment/update rounding.
    tolerance = 128 * np.finfo(float).eps * len(x) * max(1, expected)
    assert abs(score - expected) <= tolerance
    original = pelt(x, penalty, minimum)
    np.testing.assert_array_equal(states, original[0])
    assert changes == original[1]
    assert abs(score - original[2]) <= tolerance


def test_f9_local_offsets_preserve_small_late_segment_variance():
    x = np.r_[np.zeros(4), 1e12 + np.array([0, 1, 2, 0, 1, 2])][:, None]
    states, changes, score = pelt(x, penalty=20, minimum=2)
    assert changes == [4]
    assert score == pytest.approx(24, rel=1e-13)
    np.testing.assert_array_equal(states, [0] * 4 + [1] * 6)


def test_f9_nondyadic_offsets_agree_with_representable_input_ground_truth():
    rng = np.random.default_rng(910)
    for _ in range(6):
        x = rng.normal(size=(10, 2))
        shifted = x + 1e12
        expected = exhaustive_pelt(shifted, 1.2, 2)
        score = pelt(shifted, 1.2, 2)[2]
        assert score == pytest.approx(expected, rel=1e-12, abs=1e-12)
        # Translation rounds each coordinate by <= q. Bound the SSE perturbation
        # using |centered error| <= 2q and |centered value| <= range per feature.
        q = np.spacing(1e12) / 2
        bound = len(x) * np.sum(4 * np.ptp(x, axis=0) * q + 4 * q * q)
        assert abs(score - pelt(x, 1.2, 2)[2]) <= bound + 1e-12


@pytest.mark.parametrize(
    "states, expected",
    [
        ([-1, -1, 0, 1, 1], [1 / 3, 2 / 3]),
        ([-1, -1], [0, 0]),
        ([], [0, 0]),
        ([0, 1, 1], [1 / 3, 2 / 3]),
        ([-1, 0, 1, 9], [0.5, 0.5]),
    ],
)
def test_f8_profiles_and_occupancy_share_assigned_denominator(states, expected):
    states = np.asarray(states, dtype=int)
    data, inputs = economic_case(np.zeros(len(states)), "simple")
    rows = metrics.profiles(data, inputs, states, np.arange(len(states)), 2)
    np.testing.assert_allclose([r["occupancy"] for r in rows], expected)
    np.testing.assert_allclose(metrics.occupancy(states, 2), expected)
    assert all(r["occupancy_denominator"] == sum(r["count"] for r in rows) for r in rows)
    assert metrics.occupancy_drift(np.array([-1, -1]), states, 2) is None


async def test_f8_saved_validation_profiles_match_assignments(
    settings, catalog, prepared, monkeypatch
):
    from app.services.regimes.estimators import Model

    config = config_of(prepared, "kmeans")
    store = Store(settings, catalog)
    original_predict = Model.predict

    def warmup(self, x, *args, **kwargs):
        result = original_predict(self, x, *args, **kwargs)
        result["states"] = np.resize([-1, -1, 0, 1, 1], len(x))
        return result

    monkeypatch.setattr(Model, "predict", warmup)
    identifier = execute(config, settings, catalog, store)["selected_experiment_id"]
    data = load(catalog, settings, config.inputs)
    states = store.model(identifier).predict(data.x)["states"]
    np.testing.assert_array_equal(store.frame(identifier)["state_id"], states)
    np.testing.assert_allclose(
        store.get(identifier, "experiment")["diagnostics"]["occupancy"],
        metrics.occupancy(states[data.spans["train"]], 2),
    )
    validation = validate(identifier, ValidationRequest(), settings, catalog, store)
    saved = store.get(validation["id"], "validation")
    expected = metrics.occupancy(states[data.spans["test"]], 2)
    np.testing.assert_allclose([p["occupancy"] for p in saved["test"]["profiles"]], expected)
    assert saved["test"]["occupancy_note"] == metrics.OCCUPANCY_NOTE
    import io
    import json
    import zipfile

    with zipfile.ZipFile(io.BytesIO(store.archive(validation["id"]))) as bundle:
        exported = json.loads(bundle.read(f"{validation['id']}/manifest.json"))
        assert exported["test"] == saved["test"]
        exported_states = pd.read_parquet(
            io.BytesIO(bundle.read(f"{identifier}/assignments.parquet"))
        )
        np.testing.assert_array_equal(exported_states["state_id"], states)


def economic_case(returns, return_type):
    values = np.asarray(returns, dtype=float)
    prepared = SimpleNamespace(market=pd.DataFrame({"r": values}), x=values[:, None])
    inputs = SimpleNamespace(
        return_column="r",
        return_type=return_type,
        price_column=None,
        adx_column=None,
        rsi_column=None,
        volatility_column=None,
        features=["r"],
    )
    return prepared, inputs


@pytest.mark.parametrize(
    "return_type, values",
    [
        ("log", [np.log(0.5), np.log(0.5)]),
        ("simple", [-0.5, -0.5]),
    ],
)
def test_f5_equity_and_drawdown_respect_return_type(return_type, values):
    prepared, inputs = economic_case(values, return_type)
    row = metrics.profiles(prepared, inputs, np.zeros(2, dtype=int), np.arange(2), 1)[0]
    assert row["maximum_drawdown"] == pytest.approx(-0.75)
    equity = metrics.equity_curve(np.asarray(values), return_type)
    np.testing.assert_allclose(equity, [1, 0.5, 0.25], rtol=1e-14)
    assert equity[-1] - 1 == pytest.approx(-0.75)
    assert row["return_type"] == return_type
    assert row["mean_return"] == pytest.approx(np.mean(values))


@pytest.mark.parametrize(
    "return_type, values, expected",
    [
        ("log", [-2.0, -2.0], np.exp(-4) - 1),
        ("simple", [-1.0, 0.1], -1.0),
        ("simple", [-1.1, 0.1], None),
        ("log", [np.nan, -0.1], None),
    ],
)
def test_f5_return_domains_and_missing_episode(return_type, values, expected):
    prepared, inputs = economic_case(values, return_type)
    row = metrics.profiles(prepared, inputs, np.zeros(2, dtype=int), np.arange(2), 1)[0]
    assert (
        row["maximum_drawdown"] == pytest.approx(expected)
        if expected is not None
        else row["maximum_drawdown"] is None
    )


def test_f5_legacy_ambiguous_returns_are_unavailable_without_blocking_profiles():
    prepared, inputs = economic_case([np.log(0.5)] * 2, None)
    row = metrics.profiles(prepared, inputs, np.zeros(2, dtype=int), np.arange(2), 1)[0]
    assert row["maximum_drawdown"] is None
    assert row["mean_return"] is None
    assert "return_type" in row["return_note"]
    assert row["count"] == 2
    # An old saved configuration remains readable; no convention is guessed.
    assert (
        Inputs(dataset_id="d", split_plan_id="s", features=["x"], return_column="r").return_type
        is None
    )


def test_f5_new_experiment_requires_explicit_return_type():
    config = Experiment(
        inputs=Inputs(dataset_id="d", split_plan_id="s", features=["x"], return_column="r")
    )
    with pytest.raises(ValidationFailure, match="return_type"):
        execute(config, None, None, None)


def test_f5_price_derived_returns_are_simple_and_do_not_join_state_episodes():
    prepared, inputs = economic_case([0, 0, 0, 0], "log")
    inputs.return_column = None
    inputs.price_column = "close"
    prepared.market["close"] = [100, 50, 25, 12.5]
    economics = metrics.economic_columns(prepared, inputs)
    assert economics["return_type"] == "simple"
    np.testing.assert_allclose(economics["return"][1:], [-0.5] * 3)
    rows = metrics.profiles(prepared, inputs, np.array([1, 0, 1, 0]), np.arange(4), 2)
    assert rows[0]["maximum_drawdown"] == pytest.approx(-0.5)


@pytest.mark.parametrize("return_type", ["simple", "log"])
def test_f5_return_declaration_round_trips(return_type):
    inputs = Inputs(
        dataset_id="d",
        split_plan_id="s",
        features=["x"],
        return_column="r",
        return_type=return_type,
    )
    assert Inputs.model_validate(inputs.model_dump()).return_type == return_type
    with pytest.raises(ValidationError):
        Inputs.model_validate(inputs.model_dump() | {"return_type": "infer"})


@pytest.mark.parametrize("return_type", ["simple", "log"])
async def test_f5_saved_evaluation_and_validation_keep_return_convention(
    settings, catalog, prepared, return_type
):
    config = config_of(prepared, "kmeans")
    contract = load(catalog, settings, config.inputs).contract
    config.inputs.return_column = "second"
    config.inputs.return_type = return_type
    store = Store(settings, catalog)
    identifier = execute(config, settings, catalog, store)["selected_experiment_id"]
    artifact = store.get(identifier, "experiment")
    assert artifact["config"]["inputs"]["return_type"] == return_type
    assert artifact["contract"] == contract
    result = evaluate(identifier, settings, catalog, store)
    assert result["charts"]["return_type"] == return_type
    assert f"use {return_type} returns" in result["economic_note"]
    validation = validate(identifier, ValidationRequest(), settings, catalog, store)
    assert all(row["return_type"] == return_type for row in validation["test"]["profiles"])
    assert store.get(validation["id"], "validation")["test"] == validation["test"]
