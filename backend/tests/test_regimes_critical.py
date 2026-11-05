"""F1–F4 regressions for D ingestion, predictive approval and inference."""

import numpy as np
import pandas as pd
import pytest

from app.core.errors import ValidationFailure
from app.models.scaling import ScalingRequest
from app.models.splits import SplitCreateRequest
from app.services.datasets import DatasetService, persist_dataset
from app.services.merge import JoinSpec
from app.services.regimes.artifacts import Store
from app.services.regimes.contracts import Approval, Inference, Target, ValidationRequest
from app.services.regimes.data import load, preprocessing, read, replay
from app.services.regimes.estimators import fit
from app.services.regimes.evaluation import (
    approve,
    compare,
    evaluate,
    reopen,
    require_predictive_evidence,
    validate,
)
from app.services.regimes.experiments import execute
from app.services.regimes.inference import infer
from app.services.scaling import ScalingService
from app.services.splits import SplitService
from tests.conftest import register
from tests.test_regimes_workflow import config_of, teacher_scaled
from tests.test_regimes_workflow import prepared as prepared_fixture

prepared = prepared_fixture


@pytest.mark.parametrize("branch", ["primary", "secondary", "nested_secondary"])
async def test_f1_rejects_full_fit_on_every_merge_branch(settings, catalog, prepared, branch):
    original, _, _ = prepared
    unsafe = await ScalingService(settings, catalog).run(
        ScalingRequest(dataset_id=original.dataset_id, scaler_type="standard", features=["signal"])
    )
    service = DatasetService(settings=settings, catalog=catalog)
    primary, secondary = original.dataset_id, unsafe.dataset.dataset_id
    if branch == "primary":
        primary, secondary = secondary, primary
    elif branch == "nested_secondary":
        nested = await service.merge(primary, [JoinSpec(secondary, prefix="nested_")])
        secondary = nested["dataset"]["dataset_id"]
    merged = await service.merge(primary, [JoinSpec(secondary, prefix="other_")])
    config = config_of(prepared)
    config.inputs.dataset_id = merged["dataset"]["dataset_id"]
    with pytest.raises(ValidationFailure, match="outside this training window"):
        load(catalog, settings, config.inputs)


async def test_f1_valid_secondary_fit_is_recorded_and_not_flattened_for_replay(
    settings, catalog, prepared
):
    original, _, scaled = prepared
    config = config_of(prepared)
    single = load(catalog, settings, config.inputs)
    merged = await DatasetService(settings=settings, catalog=catalog).merge(
        original.dataset_id, [JoinSpec(scaled.dataset.dataset_id, prefix="other_")]
    )
    config.inputs.dataset_id = merged["dataset"]["dataset_id"]
    config.inputs.features = ["other_signal"]
    data = load(catalog, settings, config.inputs)
    assert [s["dataset_id"] for s in data.steps] == [scaled.dataset.dataset_id]
    np.testing.assert_allclose(data.x, single.x)
    with pytest.raises(ValidationFailure, match="branch"):
        replay(data.frame, data.steps)


async def test_f1_incomplete_or_cyclic_secondary_lineage_is_refused(settings, catalog, prepared):
    original, _, scaled = prepared
    catalog.add_lineage(original.dataset_id, scaled.dataset.dataset_id, "merge", {})
    with pytest.raises(ValidationFailure, match="cycle"):
        preprocessing(catalog, settings, scaled.dataset)


def hybrid_config(prepared):
    config = config_of(prepared, "hmm_xgboost")
    config.target = Target(source="teacher", objective="future", horizon=3)
    config.parameters = {"n_estimators": 5, "teacher_max_iter": 10}
    return config


async def test_f2_teacher_rejects_outer_train_fitted_preprocessing(settings, catalog, prepared):
    config = hybrid_config(prepared)
    data = load(catalog, settings, config.inputs)  # Valid for the outer training interval.
    assert data.steps[0]["fit_end"] == data.contract["train_end"]
    with pytest.raises(ValidationFailure, match="teacher"):
        execute(config, settings, catalog, Store(settings, catalog))


async def test_f2_accepts_actual_teacher_cutoff_and_causal_targets(settings, catalog, prepared):
    scaled = await teacher_scaled(settings, catalog, prepared)
    config = hybrid_config(prepared)
    config.inputs.dataset_id = scaled.dataset.dataset_id
    data = load(catalog, settings, config.inputs)
    model = fit(config, data)
    cutoff = model.diagnostics["teacher_fit_end_index"]
    assert data.times.iloc[cutoff].isoformat() == data.steps[0]["fit_end"]
    assert model.diagnostics["last_training_label"] <= data.spans["train"][-1]
    probability = model.teacher.predict(data.x)[1]
    changed = data.x.copy()
    changed[80:] += 10000
    np.testing.assert_allclose(probability[:80], model.teacher.predict(changed)[1][:80])


async def test_f2_legacy_unsafe_teacher_cannot_be_reopened(settings, catalog, prepared):
    config = hybrid_config(prepared)
    unsafe = load(catalog, settings, config.inputs)
    raw_config = config.model_copy(deep=True)
    raw_config.inputs.dataset_id = prepared[0].dataset_id
    model = fit(raw_config, load(catalog, settings, raw_config.inputs))
    store = Store(settings, catalog)
    legacy = store.save(
        "experiment",
        {
            "name": "Legacy teacher",
            "config": config.model_dump(mode="json"),
            "contract": unsafe.contract,
            "diagnostics": model.diagnostics,
            "model": model.name,
        },
        unsafe.meta,
        model=model,
    )
    with pytest.raises(ValidationFailure, match="teacher"):
        reopen(legacy["id"], settings, catalog, store)


@pytest.mark.parametrize("model", ["logistic", "hmm_xgboost"])
@pytest.mark.parametrize("horizon,lag,pairs", [(30, 0, 0), (23, 0, 1), (22, 1, 1), (22, 0, 2)])
async def test_f3_predictive_approval_requires_usable_pairs(
    settings, catalog, prepared, model, horizon, lag, pairs
):
    config = hybrid_config(prepared) if model == "hmm_xgboost" else config_of(prepared, model)
    if model == "hmm_xgboost":
        config.inputs.dataset_id = (
            await teacher_scaled(settings, catalog, prepared)
        ).dataset.dataset_id
    else:
        config.target = Target(source="column", column="regime_target", objective="future")
    config.target.horizon, config.target.availability_lag = horizon, lag
    store = Store(settings, catalog)
    identifier = execute(config, settings, catalog, store)["selected_experiment_id"]
    if pairs < 2:
        with pytest.raises(ValidationFailure, match="Insufficient predictive evidence"):
            validate(identifier, ValidationRequest(), settings, catalog, store)
        assert not store.list("validation")
        assert not store.list("approval")
    else:
        validation = validate(identifier, ValidationRequest(), settings, catalog, store)
        assert validation["test"]["predictive"]["observations"] == pairs
        approved = approve(
            identifier,
            Approval(validation_id=validation["id"], rationale="Reviewed two pairs"),
            settings,
            catalog,
            store,
        )
        assert approved["experiment_id"] == identifier


async def test_f3_legacy_validated_flag_does_not_override_empty_evidence(
    settings, catalog, prepared
):
    config = config_of(prepared, "logistic")
    config.target = Target(source="column", column="regime_target", objective="future", horizon=30)
    store = Store(settings, catalog)
    identifier = execute(config, settings, catalog, store)["selected_experiment_id"]
    result = evaluate(identifier, settings, catalog, store)
    legacy = store.save(
        "validation",
        {
            "name": "Legacy empty evidence",
            "experiment_id": identifier,
            "validated": True,
            "causal": True,
            "test": result["partitions"]["test"],
        },
        prepared[0],
    )
    with pytest.raises(ValidationFailure, match="Insufficient predictive evidence"):
        approve(
            identifier,
            Approval(validation_id=legacy["id"], rationale="Legacy approval"),
            settings,
            catalog,
            store,
        )
    assert not compare([identifier], settings, catalog, store)["rows"][0]["validated"]
    experiment = store.get(identifier, "experiment")
    legacy_approval = store.save(
        "approval",
        {
            "name": "Legacy approval",
            "experiment_id": identifier,
            "validation_id": legacy["id"],
            "contract": experiment["contract"],
            "model_checksum": experiment["files"]["model.bin"],
            "version": identifier,
            "labels": result["labels"],
            "objective": "future",
        },
        prepared[0],
    )
    with pytest.raises(ValidationFailure, match="Insufficient predictive evidence"):
        infer(
            Inference(approval_id=legacy_approval["id"], dataset_id=config.inputs.dataset_id),
            settings,
            catalog,
            store,
        )
    assert store.get(legacy_approval["id"])["id"] == legacy_approval["id"]


async def test_f3_negative_predictive_result_is_evidence_not_insufficiency(
    settings, catalog, prepared
):
    original, _, _ = prepared
    _, frame = read(catalog, settings, original.dataset_id)
    frame.loc[136:, "regime_target"] = 1 - frame.loc[136:, "regime_target"]
    source = register(settings, catalog, frame)
    plan = await SplitService(settings, catalog).create(
        SplitCreateRequest(dataset_id=source.dataset_id, split_type="percentage_70_15_15")
    )
    config = config_of(prepared, "logistic")
    config.inputs.dataset_id, config.inputs.split_plan_id = (
        source.dataset_id,
        plan.dataset.dataset_id,
    )
    config.target = Target(source="column", column="regime_target")
    store = Store(settings, catalog)
    identifier = execute(config, settings, catalog, store)["selected_experiment_id"]
    validation = validate(identifier, ValidationRequest(), settings, catalog, store)
    assert validation["validated"]
    assert validation["test"]["predictive"]["observations"] == 24
    assert validation["test"]["predictive"]["accuracy"] == 0


@pytest.mark.parametrize("family", ["supervised", "hybrid", "clustering", "hmm"])
def test_f3_every_predictive_fold_needs_evidence_but_unsupervised_does_not(family):
    from types import SimpleNamespace

    model = SimpleNamespace(family=family)
    test = {"predictive": {"log_loss_rows": 24}}
    folds = [{"fold": 2, "observations": 24, "predictive": {"log_loss_rows": 0}}]
    if family in {"supervised", "hybrid"}:
        with pytest.raises(ValidationFailure, match="fold 2"):
            require_predictive_evidence(model, test, folds)
    else:
        require_predictive_evidence(model, {}, folds)


@pytest.mark.parametrize("source", ["column", "rules"])
@pytest.mark.parametrize("missing", ["tail", "all", "absent", "outside_scoring"])
@pytest.mark.parametrize("mode", ["prepared", "replay"])
async def test_f4_predictions_survive_unavailable_monitoring_targets(
    settings, catalog, prepared, source, missing, mode
):
    config = config_of(prepared, "logistic")
    config.target = (
        Target(source="column", column="regime_target", objective="future", horizon=3)
        if source == "column"
        else Target(
            source="rules",
            rule_feature="volatility",
            lower=0.005,
            upper=0.012,
            objective="future",
            horizon=3,
        )
    )
    store = Store(settings, catalog)
    identifier = execute(config, settings, catalog, store)["selected_experiment_id"]
    validation = validate(identifier, ValidationRequest(), settings, catalog, store)
    approval = approve(
        identifier,
        Approval(validation_id=validation["id"], rationale="Predictive reference"),
        settings,
        catalog,
        store,
    )
    source_id = prepared[0].dataset_id if mode == "replay" else prepared[2].dataset.dataset_id
    request = Inference(approval_id=approval["id"], dataset_id=source_id, input_mode=mode)
    reference = infer(request, settings, catalog, store)
    parent, frame = read(catalog, settings, source_id)
    key = "regime_target" if source == "column" else "volatility"
    frame[key] = frame[key].astype(float)
    if missing == "absent":
        frame = frame.drop(columns=[key])
    elif missing == "all":
        frame[key] = np.nan
    elif missing == "tail":
        frame.loc[158:, key] = np.nan
    else:
        frame.loc[:114, key] = np.nan  # Outside labels 115..159 used by post-train scoring.
    changed = (
        register(settings, catalog, frame)
        if mode == "replay"
        else persist_dataset(
            parent=parent,
            frame=frame,
            data_dir=settings.data_dir,
            catalog=catalog,
            category=parent.category,
            detail={},
            operation="derive",
            name="Optional monitoring labels",
        )
    )
    actual = infer(
        request.model_copy(update={"dataset_id": changed.dataset_id}), settings, catalog, store
    )
    expected_frame, actual_frame = store.frame(reference["id"]), store.frame(actual["id"])
    for column in ["state_id", "confirmed_state_id", "probability_0"]:
        np.testing.assert_allclose(actual_frame[column], expected_frame[column])
    performance = actual["monitoring"]["performance"]
    assert performance["available"] is (missing not in {"all", "absent"})
    if performance["available"]:
        assert performance["metrics"]["observations"] == (43 if missing == "tail" else 45)
        assert performance["metrics"]["log_loss_rows"] == performance["metrics"]["observations"]


@pytest.mark.parametrize("missing_test", [False, True])
async def test_f4_training_uses_only_eligible_labels_and_f3_rejects_missing_test_evidence(
    settings, catalog, prepared, missing_test
):
    _, frame = read(catalog, settings, prepared[0].dataset_id)
    frame["regime_target"] = frame["regime_target"].astype(float)
    frame.loc[136 if missing_test else 158 :, "regime_target"] = np.nan
    source = register(settings, catalog, frame)
    plan = await SplitService(settings, catalog).create(
        SplitCreateRequest(dataset_id=source.dataset_id, split_type="percentage_70_15_15")
    )
    config = config_of(prepared, "logistic")
    config.inputs.dataset_id, config.inputs.split_plan_id = (
        source.dataset_id,
        plan.dataset.dataset_id,
    )
    config.target = Target(source="column", column="regime_target", objective="future", horizon=3)
    store = Store(settings, catalog)
    identifier = execute(config, settings, catalog, store)["selected_experiment_id"]
    if missing_test:
        with pytest.raises(ValidationFailure, match="Insufficient predictive evidence"):
            validate(identifier, ValidationRequest(), settings, catalog, store)
    else:
        validation = validate(identifier, ValidationRequest(), settings, catalog, store)
        assert validation["test"]["predictive"]["observations"] == 19


async def test_f4_inference_shorter_than_horizon_still_returns_predictions(
    settings, catalog, prepared
):
    config = config_of(prepared, "logistic")
    config.target = Target(source="column", column="regime_target", objective="future", horizon=3)
    store = Store(settings, catalog)
    identifier = execute(config, settings, catalog, store)["selected_experiment_id"]
    validation = validate(identifier, ValidationRequest(), settings, catalog, store)
    approval = approve(
        identifier,
        Approval(validation_id=validation["id"], rationale="Valid test"),
        settings,
        catalog,
        store,
    )
    data = load(catalog, settings, config.inputs)
    short = infer(
        Inference(
            approval_id=approval["id"],
            dataset_id=config.inputs.dataset_id,
            as_of=(data.times.iloc[113] + pd.Timedelta("1h")).isoformat(),
        ),
        settings,
        catalog,
        store,
    )
    assert short["observation_count"] == 2
    assert not short["monitoring"]["performance"]["available"]
    assert short["monitoring"]["performance"]["metrics"]["observations"] == 0


@pytest.mark.parametrize("eligible_missing", [False, True])
async def test_f4_required_training_labels_remain_strict(
    settings, catalog, prepared, eligible_missing
):
    config = config_of(prepared, "logistic")
    config.target = Target(source="column", column="regime_target", objective="future", horizon=3)
    data = load(catalog, settings, config.inputs)
    data.frame["regime_target"] = data.frame["regime_target"].astype("Int64")
    data.frame.loc[111 if eligible_missing else 0, "regime_target"] = pd.NA
    if eligible_missing:
        with pytest.raises(ValidationFailure, match="unavailable labels"):
            fit(config, data)
    else:
        model = fit(config, data)
        assert model.diagnostics["training_origins"] == 109
