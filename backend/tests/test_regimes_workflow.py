"""C6/C7 integration, experiment persistence, D10 approval and causal D11 serving."""

import numpy as np
import pandas as pd
import pytest

from app.core.errors import ValidationFailure
from app.models.enums import ScalerType, SplitType
from app.models.scaling import ScalingRequest
from app.models.splits import SplitCreateRequest, SplitParameters
from app.services.regimes.artifacts import Store
from app.services.regimes.contracts import (
    Approval,
    Experiment,
    Inference,
    Inputs,
    LabelRequest,
    StateConfig,
    Target,
    ValidationRequest,
)
from app.services.regimes.data import load
from app.services.regimes.evaluation import approve, evaluate, label, validate
from app.services.regimes.experiments import execute
from app.services.regimes.inference import infer
from app.services.scaling import ScalingService
from app.services.splits import SplitService
from tests.conftest import hourly_frame, register


@pytest.fixture
async def prepared(settings, catalog):
    frame = hourly_frame(160)
    rng = np.random.default_rng(11)
    frame["signal"] = np.tile(np.r_[np.full(20, -1.0), np.full(20, 1.0)], 4) + rng.normal(
        0, 0.1, 160
    )
    frame["second"] = rng.normal(size=160)
    frame["regime_target"] = (frame.signal > 0).astype(int)
    frame["fast_ma"] = frame.close + 0.5 * frame.signal
    frame["slow_ma"] = frame.close
    frame["adx"] = 20 + 10 * np.abs(frame.signal)
    frame["plus_di"] = 20 + 5 * frame.signal
    frame["minus_di"] = 20 - 5 * frame.signal
    frame["volatility"] = np.abs(frame.second) * 0.01
    original = register(settings, catalog, frame)
    plan = await SplitService(settings, catalog).create(
        SplitCreateRequest(dataset_id=original.dataset_id, split_type=SplitType.PERCENTAGE_70_15_15)
    )
    scaled = await ScalingService(settings, catalog).run(
        ScalingRequest(
            dataset_id=original.dataset_id,
            split_plan_id=plan.dataset.dataset_id,
            scaler_type=ScalerType.STANDARD,
            features=["signal", "second"],
        )
    )
    return original, plan, scaled


def config_of(prepared, model="gaussian_hmm"):
    original, plan, scaled = prepared
    return Experiment(
        name="BTCUSDT reference",
        model=model,
        inputs=Inputs(
            dataset_id=scaled.dataset.dataset_id,
            split_plan_id=plan.dataset.dataset_id,
            features=["signal"],
            market_dataset_id=original.dataset_id,
        ),
        states=StateConfig(count=2, minimum_occupancy=0),
        parameters={"max_iter": 15} if model == "gaussian_hmm" else {},
    )


async def teacher_scaled(settings, catalog, prepared):
    """Actual C7 artifact fitted only through the hybrid teacher's 56-row window."""
    original, _, _ = prepared
    data = load(catalog, settings, config_of(prepared).inputs)
    times = data.times
    cutoff = data.spans["train"][len(data.spans["train"]) // 2 - 1]
    plan = await SplitService(settings, catalog).create(
        SplitCreateRequest(
            dataset_id=original.dataset_id,
            split_type="date_2way",
            parameters=SplitParameters(
                train_start=times.iloc[0], train_end=times.iloc[cutoff],
                test_start=times.iloc[cutoff + 1], test_end=times.iloc[-1],
            ),
        )
    )
    return await ScalingService(settings, catalog).run(
        ScalingRequest(
            dataset_id=original.dataset_id, split_plan_id=plan.dataset.dataset_id,
            scaler_type="standard", features=["signal", "second"],
        )
    )


@pytest.mark.parametrize(
    "model",
    [
        "gaussian_hmm",
        "multivariate_hmm",
        "student_t_hmm",
        "gmm_hmm",
        "kmeans",
        "gmm",
        "rules",
        "moving_average",
        "adx",
        "trend_volatility",
        "pelt",
        "cusum",
        "logistic",
        "xgboost",
        "lightgbm",
        "lstm",
        "gru",
        "hmm_xgboost",
    ],
)
async def test_each_model_executes_real_training_and_persisted_predictions(
    settings, catalog, prepared, model
):
    config = config_of(prepared, model)
    if model == "multivariate_hmm":
        config.inputs.features = ["signal", "second"]
    if "hmm" in model and model != "hmm_xgboost":
        config.parameters["max_iter"] = 12
    if model == "rules":
        config.parameters = {"feature": "signal", "lower": -0.5, "upper": 0.5}
    if model == "moving_average":
        config.inputs.features = ["fast_ma", "slow_ma"]
        config.parameters = {"fast_column": "fast_ma", "slow_column": "slow_ma"}
    if model == "adx":
        config.inputs.features = ["adx", "plus_di", "minus_di"]
        config.parameters = {
            "adx_column": "adx",
            "plus_di_column": "plus_di",
            "minus_di_column": "minus_di",
        }
    if model == "trend_volatility":
        config.inputs.features = ["signal", "volatility"]
        config.parameters = {"trend_column": "signal", "volatility_column": "volatility"}
    if model in {"logistic", "xgboost", "lightgbm", "lstm", "gru"}:
        config.target = Target(
            source="column", column="regime_target", objective="future", horizon=3
        )
        if model in {"lstm", "gru"}:
            config.parameters = {"epochs": 2, "hidden_size": 4, "sequence_length": 5}
        if model in {"xgboost", "lightgbm"}:
            config.parameters = {"n_estimators": 5}
    if model == "hmm_xgboost":
        safe = await teacher_scaled(settings, catalog, prepared)
        config.inputs.dataset_id = safe.dataset.dataset_id
        config.target = Target(source="teacher", objective="future", horizon=3)
        config.parameters = {"n_estimators": 5, "teacher_max_iter": 10}
    store = Store(settings, catalog)
    result = execute(config, settings, catalog, store)
    experiment = store.get(result["selected_experiment_id"], "experiment")
    fitted = store.model(experiment["id"])
    data = load(catalog, settings, config.inputs)
    saved = store.frame(experiment["id"])
    np.testing.assert_array_equal(saved.state_id, fitted.predict(data.x)["states"])
    assert experiment["contract"]["asset"] == "BTCUSDT"
    if model in {"lstm", "gru"}:
        assert len(fitted.estimator.history) == 2
        assert any("recurrent.weight" in key for key in fitted.estimator.weights)
    if config.target:
        assert fitted.diagnostics["last_training_label"] <= data.spans["train"][-1]
        assert fitted.diagnostics["last_training_origin"] + 3 <= data.spans["train"][-1]


async def test_end_to_end_validation_labels_approval_inference_and_prefix(
    settings, catalog, prepared
):
    store = Store(settings, catalog)
    config = config_of(prepared)
    result = execute(config, settings, catalog, store)
    identifier = result["selected_experiment_id"]
    result = evaluate(identifier, settings, catalog, store)
    assert result["partitions"]["test"]["observations"] == 24
    assert result["partitions"]["test"]["profiles"][0]["mean_return"] is not None
    label(
        identifier,
        LabelRequest(mode="manual", labels={"0": "Custom A", "1": "Custom B"}),
        settings,
        catalog,
        store,
    )
    validation = validate(identifier, ValidationRequest(), settings, catalog, store)
    approval = approve(
        identifier,
        Approval(validation_id=validation["id"], rationale="Validated chronological reference"),
        settings,
        catalog,
        store,
    )
    assert approval["labels"] == {"0": "Custom A", "1": "Custom B"}
    dataset = prepared[2].dataset.dataset_id
    full = infer(
        Inference(approval_id=approval["id"], dataset_id=dataset), settings, catalog, store
    )
    data = load(catalog, settings, config.inputs)
    boundary = str(data.times.iloc[135] + pd.Timedelta(minutes=30))
    short = infer(
        Inference(approval_id=approval["id"], dataset_id=dataset, as_of=boundary),
        settings,
        catalog,
        store,
    )
    long_frame, short_frame = store.frame(full["id"]), store.frame(short["id"])
    assert pd.Timestamp(short_frame.available_at.iloc[-1]) <= pd.Timestamp(boundary)
    assert pd.Timestamp(short_frame.timestamp.iloc[-1]) < data.times.iloc[135]
    np.testing.assert_allclose(
        short_frame.probability_0, long_frame.probability_0[: len(short_frame)]
    )
    np.testing.assert_array_equal(
        short_frame.confirmed_state_id, long_frame.confirmed_state_id[: len(short_frame)]
    )
    raw = infer(
        Inference(
            approval_id=approval["id"], dataset_id=prepared[0].dataset_id, input_mode="replay"
        ),
        settings,
        catalog,
        store,
    )
    np.testing.assert_allclose(store.frame(raw["id"]).probability_0, long_frame.probability_0)
    smooth = infer(
        Inference(approval_id=approval["id"], dataset_id=dataset, decoding="smooth"),
        settings,
        catalog,
        store,
    )
    assert smooth["snapshot"]["retrospective"] is True
    assert smooth["snapshot"]["live_current_regime"] is False
    # Change future features in a different source dataset; a historical origin cannot see them.
    from app.services.regimes.data import read

    _, changed = read(catalog, settings, prepared[0].dataset_id)
    changed.loc[135:, "signal"] += 10000
    changed_source = register(settings, catalog, changed)
    truncated = infer(
        Inference(
            approval_id=approval["id"],
            dataset_id=changed_source.dataset_id,
            input_mode="replay",
            as_of=boundary,
        ),
        settings,
        catalog,
        store,
    )
    np.testing.assert_allclose(
        store.frame(truncated["id"]).probability_0, short_frame.probability_0
    )


async def test_c8_pca_fitted_artifacts_replay_without_refitting(settings, catalog, prepared):
    from app.models.exploratory import PCARequest
    from app.services.exploratory import ExploratoryService

    original, plan, scaled = prepared
    pca_plan = await SplitService(settings, catalog).create(
        SplitCreateRequest(dataset_id=scaled.dataset.dataset_id, split_type="percentage_70_15_15")
    )
    result = await ExploratoryService(settings, catalog).run(
        PCARequest(
            dataset_id=scaled.dataset.dataset_id,
            features=["signal", "second"],
            scope="train_span",
            split_plan_id=pca_plan.dataset.dataset_id,
            n_components=1,
            standardizer="none",
            write_dataset=True,
        )
    )
    config = config_of(prepared)
    config.inputs.dataset_id = result["dataset"]["dataset_id"]
    config.inputs.split_plan_id = pca_plan.dataset.dataset_id
    config.inputs.features = ["pc_1"]
    store = Store(settings, catalog)
    experiment = execute(config, settings, catalog, store)["selected_experiment_id"]
    validation = validate(experiment, ValidationRequest(), settings, catalog, store)
    approval = approve(
        experiment,
        Approval(validation_id=validation["id"], rationale="C7 and PCA replay validation"),
        settings,
        catalog,
        store,
    )
    first = infer(
        Inference(approval_id=approval["id"], dataset_id=config.inputs.dataset_id),
        settings,
        catalog,
        store,
    )
    replayed = infer(
        Inference(approval_id=approval["id"], dataset_id=original.dataset_id, input_mode="replay"),
        settings,
        catalog,
        store,
    )
    np.testing.assert_allclose(
        store.frame(first["id"]).probability_0,
        store.frame(replayed["id"]).probability_0,
        atol=1e-10,
    )


async def test_full_dataset_scaling_refused_and_integrity_tampering_detected(
    settings, catalog, prepared
):
    original, plan, _ = prepared
    unsafe = await ScalingService(settings, catalog).run(
        ScalingRequest(
            dataset_id=original.dataset_id, scaler_type=ScalerType.STANDARD, features=["signal"]
        )
    )
    config = config_of(prepared)
    config.inputs.dataset_id = unsafe.dataset.dataset_id
    with pytest.raises(ValidationFailure, match="outside this training window"):
        load(catalog, settings, config.inputs)
    store = Store(settings, catalog)
    result = execute(config_of(prepared), settings, catalog, store)
    identifier = result["selected_experiment_id"]
    (store.path(identifier) / "model.bin").write_bytes(b"tampered")
    with pytest.raises(ValidationFailure, match="integrity"):
        store.model(identifier)


async def test_automatic_count_selection_and_incompatible_metrics(settings, catalog, prepared):
    config = config_of(prepared, "kmeans")
    config.states.mode = "automatic"
    config.states.candidates = [2, 3]
    config.states.criterion = "silhouette"
    store = Store(settings, catalog)
    result = execute(config, settings, catalog, store)
    assert len(result["trials"]) == 2
    scores = [row["score"] for row in result["trials"]]
    best = next(
        row for row in result["trials"] if row["artifact_id"] == result["selected_experiment_id"]
    )
    assert best["score"] == max(scores)
    config.states.criterion = "bic"
    with pytest.raises(ValidationFailure, match="incompatible"):
        execute(config, settings, catalog, store)
