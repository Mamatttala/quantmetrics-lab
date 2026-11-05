"""Phase 4: downstream contracts using actual existing upstream service outputs."""

import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from app.models.exploratory import PCARequest
from app.models.scaling import ScalingRequest
from app.models.splits import SplitCreateRequest, SplitParameters
from app.services.datasets import DatasetService
from app.services.exploratory import ExploratoryService
from app.services.features import FeatureRequest, Selection
from app.services.regimes.artifacts import Store
from app.services.regimes.contracts import Approval, Inference, Search, Target, ValidationRequest
from app.services.regimes.data import load, plan_rows
from app.services.regimes.evaluation import approve, validate
from app.services.regimes.experiments import execute
from app.services.regimes.inference import infer
from app.services.scaling import ScalingService
from app.services.splits import SplitService
from tests.test_regimes_workflow import config_of
from tests.test_regimes_workflow import prepared as prepared_fixture

prepared = prepared_fixture


@pytest.mark.parametrize("pca", [False, True])
async def test_phase4_actual_c5_selection_c6_c7_and_optional_c8_outputs(
    settings,
    catalog,
    prepared,
    pca,
):
    selected = ["normalized_high_low_range", "high_low_range"]
    generated = await DatasetService(settings, catalog).features(
        prepared[0].dataset_id,
        FeatureRequest(selections=tuple(Selection(name) for name in selected)),
    )
    assert generated["report"]["generated_columns"] == selected
    source = generated["dataset"]["dataset_id"]
    plan = await SplitService(settings, catalog).create(
        SplitCreateRequest(dataset_id=source, split_type="percentage_70_15_15")
    )
    scaled = await ScalingService(settings, catalog).run(
        ScalingRequest(
            dataset_id=source,
            split_plan_id=plan.dataset.dataset_id,
            features=selected,
            scaler_type="standard",
        )
    )
    config = config_of(prepared, "multivariate_hmm")
    config.inputs.dataset_id = scaled.dataset.dataset_id
    config.inputs.split_plan_id = plan.dataset.dataset_id
    config.inputs.features = selected
    config.parameters = {"max_iter": 6}
    if pca:
        pca_plan = await SplitService(settings, catalog).create(
            SplitCreateRequest(
                dataset_id=scaled.dataset.dataset_id, split_type="percentage_70_15_15"
            )
        )
        projection = await ExploratoryService(settings, catalog).run(
            PCARequest(
                dataset_id=scaled.dataset.dataset_id,
                features=selected,
                scope="train_span",
                split_plan_id=pca_plan.dataset.dataset_id,
                n_components=1,
                standardizer="none",
                write_dataset=True,
            )
        )
        config.inputs.dataset_id = projection["dataset"]["dataset_id"]
        config.inputs.split_plan_id = pca_plan.dataset.dataset_id
        config.inputs.features = ["pc_1"]
        config.model = "gaussian_hmm"
    data = load(catalog, settings, config.inputs)
    np.testing.assert_array_equal(data.x, data.frame[config.inputs.features].to_numpy())
    store = Store(settings, catalog)
    identifier = execute(config, settings, catalog, store)["selected_experiment_id"]
    model = store.model(identifier)
    assert model.features == config.inputs.features
    validation = validate(identifier, ValidationRequest(), settings, catalog, store)
    approval = approve(
        identifier,
        Approval(
            validation_id=validation["id"], rationale="Phase 4 upstream contract verification"
        ),
        settings,
        catalog,
        store,
    )
    native = infer(
        Inference(approval_id=approval["id"], dataset_id=config.inputs.dataset_id),
        settings,
        catalog,
        store,
    )
    replayed = infer(
        Inference(approval_id=approval["id"], dataset_id=source, input_mode="replay"),
        settings,
        catalog,
        store,
    )
    np.testing.assert_allclose(
        store.frame(native["id"]).probability_0,
        store.frame(replayed["id"]).probability_0,
        atol=1e-10,
    )
    with zipfile.ZipFile(io.BytesIO(store.archive(native["id"]))) as bundle:
        assert bundle.testzip() is None
        saved = json.loads(bundle.read(f"{identifier}/manifest.json"))
        assert saved["contract"]["features"] == config.inputs.features
        assert (
            saved["contract"]["preprocessing_signature"] == data.contract["preprocessing_signature"]
        )


@pytest.mark.parametrize("mode", ["expanding", "rolling"])
@pytest.mark.parametrize("model_name", ["xgboost", "hmm_xgboost"])
async def test_phase4_predictive_walk_forward_with_fold_fits_and_purged_targets(
    settings,
    catalog,
    prepared,
    mode,
    model_name,
):
    source = prepared[0].dataset_id
    config = config_of(prepared, model_name)
    # Original units are a supported explicit representation, with no fitted preprocessing.
    config.inputs.dataset_id = source
    hybrid = model_name == "hmm_xgboost"
    config.target = Target(
        source="teacher" if hybrid else "column",
        column=None if hybrid else "regime_target",
        objective="future",
        horizon=3,
        availability_lag=1,
    )
    config.parameters = {"n_estimators": 3, **({"teacher_max_iter": 6} if hybrid else {})}
    if not hybrid:
        config.search = Search(method="grid", space={"n_estimators": [2, 3]})
        config.states.criterion = "validation_log_loss"
    store = Store(settings, catalog)
    identifier = execute(config, settings, catalog, store)["selected_experiment_id"]
    plan = await SplitService(settings, catalog).create(
        SplitCreateRequest(
            dataset_id=source,
            split_type=f"tscv_{mode}",
            parameters=SplitParameters(n_folds=2, include_validation=not hybrid),
        )
    )
    rows = plan_rows(catalog, settings, plan.dataset.dataset_id)[1]
    reference = load(catalog, settings, config.inputs)
    datasets = {}
    for fold in sorted({row.fold for row in rows}):
        fit_plan_id = plan.dataset.dataset_id
        fit_fold = fold
        if hybrid:
            train = next(row for row in rows if row.fold == fold and row.role.value == "train")
            teacher_end = train.start_index + train.row_count // 2 - 1
            fit_plan = await SplitService(settings, catalog).create(
                SplitCreateRequest(
                    dataset_id=source,
                    split_type="date_2way",
                    parameters=SplitParameters(
                        train_start=train.start_time,
                        train_end=reference.times.iloc[teacher_end],
                        test_start=reference.times.iloc[teacher_end + 1],
                        test_end=reference.times.iloc[-1],
                    ),
                )
            )
            fit_plan_id, fit_fold = fit_plan.dataset.dataset_id, None
        scaled = await ScalingService(settings, catalog).run(
            ScalingRequest(
                dataset_id=source,
                split_plan_id=fit_plan_id,
                fold=fit_fold,
                features=config.inputs.features,
                scaler_type="standard",
            )
        )
        datasets[fold] = scaled.dataset.dataset_id
    before = {item["id"] for item in store.list("search")}
    result = validate(
        identifier,
        ValidationRequest(
            mode=mode, split_plan_id=plan.dataset.dataset_id, prepared_datasets=datasets
        ),
        settings,
        catalog,
        store,
    )
    assert len(result["folds"]) == 2
    signatures = []
    for row in result["folds"]:
        test = next(r for r in rows if r.fold == row["fold"] and r.role.value == "test")
        assert row["predictive"]["log_loss_rows"] == test.row_count - 4
        assert pd.Timestamp(row["train_end"]) < pd.Timestamp(row["test_start"])
        assert row["state_alignment"]["observations"] >= 2
        assert row["selection_scope"] == "fold-local training/validation"
        signatures.append(row["preprocessing_signature"])
    assert len(set(signatures)) == 2
    if not hybrid:
        searches = [s for s in store.list("search") if s["id"] not in before]
        assert len(searches) == 2
        for search in searches:
            assert search["config"]["inputs"]["split_plan_id"] == plan.dataset.dataset_id
            assert len(search["trials"]) == 2
            assert all(t["status"] == "succeeded" for t in search["trials"])
    approved = approve(
        identifier,
        Approval(validation_id=result["id"], rationale="Phase 4 predictive fold verification"),
        settings,
        catalog,
        store,
    )
    assert approved["validation_id"] == result["id"]
