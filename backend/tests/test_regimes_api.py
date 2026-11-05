"""HTTP workflow, project isolation, strict configuration and optimization/refitting."""

import time

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import create_app
from app.models.enums import SplitType
from app.models.splits import SplitCreateRequest, SplitParameters
from app.services.regimes.artifacts import Store
from app.services.regimes.contracts import ValidationRequest
from app.services.regimes.evaluation import validate
from app.services.regimes.experiments import execute
from app.services.splits import SplitService
from tests.test_regimes_workflow import config_of
from tests.test_regimes_workflow import prepared as prepared_fixture

prepared = prepared_fixture


def wait(client, response):
    assert response.status_code == 202, response.text
    for _ in range(400):
        result = client.get(response.json()["poll"]).json()
        if result["status"] != "running":
            assert result["status"] == "succeeded", result
            return result["result"]
        time.sleep(0.02)
    raise AssertionError("Regime job did not finish")


async def test_api_complete_workflow_and_project_isolation(
    settings, catalog, prepared, monkeypatch
):
    monkeypatch.setenv("QML_DATA_DIR", str(settings.data_dir))
    get_settings.cache_clear()
    try:
        with TestClient(create_app()) as client:
            config = config_of(prepared)
            options = client.get(
                "/api/regime/options", params={"dataset_id": config.inputs.dataset_id}
            )
            assert options.status_code == 200, options.text
            assert len(options.json()["models"]) == 18
            result = wait(
                client, client.post("/api/regime/experiments", json=config.model_dump(mode="json"))
            )
            identifier = result["selected_experiment_id"]
            response = client.get(f"/api/regime/experiments/{identifier}/evaluation")
            assert response.status_code == 200, response.text
            validated = wait(
                client, client.post(f"/api/regime/experiments/{identifier}/validation", json={})
            )
            approved = client.post(
                f"/api/regime/experiments/{identifier}/approval",
                json={
                    "validation_id": validated["id"],
                    "rationale": "Reviewed held-out BTCUSDT state behavior",
                },
            )
            assert approved.status_code == 201, approved.text
            inferred = wait(
                client,
                client.post(
                    "/api/regime/inference",
                    json={
                        "approval_id": approved.json()["id"],
                        "dataset_id": config.inputs.dataset_id,
                    },
                ),
            )
            assert inferred["snapshot"]["live_current_regime"] is True
            csv = client.get(f"/api/regime/artifacts/{inferred['id']}/export?format=csv")
            assert csv.status_code == 200
            assert "probability_0" in csv.text and "available_at" in csv.text
            exported = client.get(f"/api/regime/artifacts/{inferred['id']}/export").json()
            assert len(exported["history"]) == 48
            other = client.post("/api/projects", json={"name": "Isolated regime project"}).json()
            isolated = client.get(
                f"/api/regime/artifacts/{identifier}", headers={"X-Project-Id": other["project_id"]}
            )
            assert isolated.status_code == 404
            assert (
                client.post(
                    "/api/regime/inference",
                    json={
                        "approval_id": approved.json()["id"],
                        "dataset_id": config.inputs.dataset_id,
                        "unknown_parameter": True,
                    },
                ).status_code
                == 422
            )
            assert client.get("/api/regime/setup/options").status_code == 404
    finally:
        get_settings.cache_clear()


@pytest.mark.parametrize("method", ["grid", "random", "bayesian", "optuna"])
async def test_real_optimization_backends(settings, catalog, prepared, method):
    config = config_of(prepared, "kmeans")
    config.states.criterion = "silhouette"
    config.search.method = method
    config.search.space = {"n_init": [1, 2], "max_iter": [10, 20]}
    config.search.trials = 3
    result = execute(config, settings, catalog, Store(settings, catalog))
    assert len(result["trials"]) == (4 if method == "grid" else 3)
    assert all(row["status"] == "succeeded" for row in result["trials"])
    assert result["config"]["search"]["method"] == method


@pytest.mark.parametrize("mode", ["expanding", "rolling"])
async def test_walk_forward_reuses_c6_and_refuses_leaking_preprocessing(
    settings, catalog, prepared, mode
):
    original, _, _ = prepared
    config = config_of(prepared, "kmeans")
    store = Store(settings, catalog)
    experiment = execute(config, settings, catalog, store)["selected_experiment_id"]
    plan = await SplitService(settings, catalog).create(
        SplitCreateRequest(
            dataset_id=original.dataset_id,
            split_type=SplitType(f"tscv_{mode}"),
            parameters=SplitParameters(n_folds=3),
        )
    )
    from app.core.errors import ValidationFailure

    with pytest.raises(ValidationFailure, match="outside this training window"):
        validate(
            experiment,
            ValidationRequest(mode=mode, split_plan_id=plan.dataset.dataset_id),
            settings,
            catalog,
            store,
        )
    # Features in original units require no fitted preprocessing. D never refits a scaler.
    config.inputs.dataset_id = original.dataset_id
    unscaled = execute(config, settings, catalog, store)["selected_experiment_id"]
    result = validate(
        unscaled,
        ValidationRequest(mode=mode, split_plan_id=plan.dataset.dataset_id),
        settings,
        catalog,
        store,
    )
    assert len(result["folds"]) == 3
    assert all(row["train_end"] < row["test_start"] for row in result["folds"])
