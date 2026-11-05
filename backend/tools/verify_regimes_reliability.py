"""Phase 4 checks on saved BTCUSDT C6/C7 outputs; all D writes use an isolated catalog.

This is a verification driver, not a new model or preprocessing implementation.
Run from backend with the existing environment: python tools/verify_regimes_reliability.py.
"""

from __future__ import annotations

import io
import json
import shutil
import time
import zipfile
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from app.config import Settings
from app.core.errors import ValidationFailure
from app.services.datasets import persist_dataset
from app.services.regimes.artifacts import Store, clean
from app.services.regimes.contracts import (
    Approval,
    Experiment,
    Inference,
    Inputs,
    StateConfig,
    Target,
    ValidationRequest,
)
from app.services.regimes.data import contributing_lineage, load, plan_rows, read, replay
from app.services.regimes.evaluation import approve, evaluate, reopen, validate
from app.services.regimes.experiments import execute
from app.services.regimes.inference import infer
from app.storage.catalog import Catalog

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/module-d-phase4/reliability.json"


def refused(call, message):
    try:
        call()
    except ValidationFailure as exc:
        assert message in str(exc), str(exc)
        return str(exc)
    raise AssertionError(f"Expected refusal containing {message}")


def run():
    inventory = json.loads((REPORT.parent / "inventory.json").read_text(encoding="utf-8"))
    project = next(
        p
        for p in inventory
        if any(
            d["symbol"] == "BTCUSDT" and json.loads(d["provenance"]).get("operation") == "scaling"
            for d in p["datasets"]
        )
    )
    original_root = Path(project["catalog"]).parent
    # Refuse copying an active WAL snapshot rather than ignore committed WAL pages.
    wal = original_root / "catalog.db-wal"
    assert not wal.exists() or wal.stat().st_size == 0, "Source catalog has an active WAL"
    destination = ROOT / ".tmp/regime-reliability" / uuid4().hex
    shutil.copytree(original_root, destination)
    settings = Settings(data_dir=destination)
    catalog = Catalog(settings.catalog_path)
    store = Store(settings, catalog)
    datasets = catalog.list_datasets()
    scaled = next(d for d in datasets if d.operation == "scaling")
    raw_id = catalog.get_dataset(scaled.dataset_id).parent_dataset_id
    fixed_plan = next(
        d
        for d in datasets
        if d.operation == "split"
        and catalog.get_dataset(d.dataset_id).provenance["split_type"] == "percentage_60_20_20"
    )
    walk_plan = next(
        d
        for d in datasets
        if d.operation == "split"
        and catalog.get_dataset(d.dataset_id).provenance["split_type"] == "tscv_expanding"
    )
    inputs = Inputs(
        dataset_id=scaled.dataset_id,
        split_plan_id=fixed_plan.dataset_id,
        features=["close"],
        market_dataset_id=raw_id,
    )
    data = load(catalog, settings, inputs)
    raw_meta, raw_frame = read(catalog, settings, raw_id)
    np.testing.assert_allclose(replay(raw_frame, data.steps)["close"], data.x[:, 0], atol=1e-10)
    report = {
        "isolated_storage": str(destination),
        "source_id": raw_id,
        "source_checksum": raw_meta.checksum,
        "prepared_dataset_id": scaled.dataset_id,
        "observations": len(data.x),
        "timeframe": raw_meta.timeframe.value,
        "first_observation": str(data.times.iloc[0]),
        "last_observation": str(data.times.iloc[-1]),
        "split_plan_id": fixed_plan.dataset_id,
        "preprocessing": data.steps,
        "preprocessing_signature": data.contract["preprocessing_signature"],
        "selected_features": inputs.features,
        "partitions": {
            role: {
                "count": len(indices),
                "start": str(data.times.iloc[indices[0]]),
                "end": str(data.times.iloc[indices[-1]]),
            }
            for role, indices in data.spans.items()
            if len(indices)
        },
        "lineage_ids": contributing_lineage(catalog, scaled.dataset_id),
        "saved_c7_replay_matches": True,
        "models": [],
        "failures": [],
    }
    # All configurations are fixed before reading predictive results. They are smoke
    # configurations; test accuracy does not select a model or certify forecasting skill.
    cases = {
        "gaussian_hmm": {"max_iter": 12},
        "student_t_hmm": {"max_iter": 12},
        "gmm_hmm": {"max_iter": 12},
        "kmeans": {"n_init": 3},
        "gmm": {"max_iter": 30, "n_init": 2},
        "xgboost": {"n_estimators": 5},
        "lstm": {"epochs": 2, "hidden_size": 4, "sequence_length": 5},
        "pelt": {"penalty": 50, "min_segment_length": 30},
        "cusum": {"threshold": 50, "min_segment_length": 50},
        "hmm_xgboost": {"n_estimators": 5, "teacher_max_iter": 12},
    }
    for name, parameters in cases.items():
        start = time.perf_counter()
        config = Experiment(
            name=f"Phase 4 BTCUSDT {name}",
            inputs=inputs.model_copy(deep=True),
            model=name,
            parameters=parameters,
            states=StateConfig(count=2, minimum_occupancy=0),
        )
        if name in {"xgboost", "lstm"}:
            config.target = Target(
                source="rules",
                rule_feature="volume",
                lower=-0.5,
                upper=0.5,
                objective="future",
                horizon=3,
                availability_lag=1,
            )
        elif name == "hmm_xgboost":
            config.target = Target(
                source="teacher", objective="future", horizon=3, availability_lag=1
            )
        row = {"model": name, "configuration": config.model_dump(mode="json")}
        try:
            identifier = execute(config, settings, catalog, store)["selected_experiment_id"]
            artifact, _, reopened, fitted = reopen(identifier, settings, catalog, store)
            assert fitted.features == ["close"]
            np.testing.assert_array_equal(reopened.x, data.frame[["close"]].to_numpy())
            np.testing.assert_array_equal(
                store.frame(identifier).state_id, fitted.predict(data.x)["states"]
            )
            evaluated = evaluate(identifier, settings, catalog, store)
            row.update(
                experiment_id=identifier,
                diagnostics=artifact["diagnostics"],
                predictive=evaluated["partitions"]["test"]["predictive"],
            )
            if name == "pelt":
                row["live_approval_refusal"] = refused(
                    lambda identifier=identifier: validate(
                        identifier, ValidationRequest(), settings, catalog, store
                    ),
                    "retrospective segmentation",
                )
                row.update(status="passed", capability="offline fit/evaluation; live refused")
                continue
            validation = validate(identifier, ValidationRequest(), settings, catalog, store)
            if config.target:
                assert row["predictive"]["log_loss_rows"] == len(data.spans["test"]) - 4
                assert fitted.diagnostics["last_training_label"] <= data.spans["train"][-1]
            approval = approve(
                identifier,
                Approval(
                    validation_id=validation["id"],
                    rationale="Phase 4 contract verification; not an economic endorsement",
                ),
                settings,
                catalog,
                store,
            )
            request = Inference(approval_id=approval["id"], dataset_id=scaled.dataset_id)
            full = infer(request, settings, catalog, store)
            saved = store.frame(full["id"])
            assert len(saved) == len(data.x) - len(data.spans["train"])
            assert full["causal"] and not full["snapshot"]["retrospective"]
            prefix_end = int(data.spans["train"][-1]) + 30
            as_of = str(data.times.iloc[prefix_end] + pd.Timedelta(hours=12))
            short = infer(request.model_copy(update={"as_of": as_of}), settings, catalog, store)
            pd.testing.assert_frame_equal(store.frame(short["id"]), saved.iloc[:29])
            if fitted.family == "hmm":
                changed = raw_frame.copy()
                changed.loc[prefix_end:, "close"] *= 10
                source = persist_dataset(
                    parent=raw_meta,
                    frame=changed,
                    data_dir=settings.data_dir,
                    catalog=catalog,
                    category=raw_meta.category,
                    operation="edit",
                    detail={},
                    name="Phase 4 future perturbation",
                )
                altered = infer(
                    request.model_copy(
                        update={
                            "dataset_id": source.dataset_id,
                            "input_mode": "replay",
                            "as_of": as_of,
                        }
                    ),
                    settings,
                    catalog,
                    store,
                )
                pd.testing.assert_frame_equal(store.frame(altered["id"]), store.frame(short["id"]))
                row["future_perturbation_prefix_invariant"] = True
                for decoding in ["smooth", "fixed_lag", "viterbi"]:
                    retrospective = infer(
                        request.model_copy(update={"decoding": decoding}), settings, catalog, store
                    )
                    assert (
                        retrospective["snapshot"]["retrospective"] and not retrospective["causal"]
                    )
                row["retrospective_decoders_verified"] = ["smooth", "fixed_lag", "viterbi"]
            if fitted.family == "supervised":
                missing = data.frame.drop(columns=["volume"])
                source = persist_dataset(
                    parent=data.meta,
                    frame=missing,
                    data_dir=settings.data_dir,
                    catalog=catalog,
                    category=data.meta.category,
                    operation="edit",
                    detail={},
                    name="Phase 4 missing monitoring targets",
                )
                predicted = infer(
                    request.model_copy(update={"dataset_id": source.dataset_id}),
                    settings,
                    catalog,
                    store,
                )
                pd.testing.assert_frame_equal(store.frame(predicted["id"]), saved)
                assert not predicted["monitoring"]["performance"]["available"]
                row["predictions_without_targets"] = True
            with zipfile.ZipFile(io.BytesIO(store.archive(full["id"]))) as bundle:
                assert bundle.testzip() is None
                assert (
                    bundle.read(f"{identifier}/model.bin")
                    == (store.path(identifier) / "model.bin").read_bytes()
                )
            row.update(
                validation_id=validation["id"],
                approval_id=approval["id"],
                inference_id=full["id"],
                inference_rows=len(saved),
                inference_mode="filter",
                bundle_integrity=True,
                status="passed",
            )
            if name == "xgboost":
                folded = validate(
                    identifier,
                    ValidationRequest(mode="expanding", split_plan_id=walk_plan.dataset_id),
                    settings,
                    catalog,
                    store,
                )
                plan = plan_rows(catalog, settings, walk_plan.dataset_id)[1]
                for fold in folded["folds"]:
                    span = next(
                        r for r in plan if r.fold == fold["fold"] and r.role.value == "test"
                    )
                    assert fold["predictive"]["log_loss_rows"] == span.row_count - 4
                row["saved_expanding_plan_validation"] = folded
            if fitted.teacher is not None:
                cutoff = str(data.times.iloc[fitted.diagnostics["teacher_fit_end_index"]])
                assert all(pd.Timestamp(s["fit_end"]) <= pd.Timestamp(cutoff) for s in data.steps)
                row["teacher_cutoff"] = cutoff
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            report["failures"].append({"model": name, "error": row["error"]})
        finally:
            row["seconds"] = time.perf_counter() - start
            report["models"].append(row)
            REPORT.write_text(
                json.dumps(clean(report), indent=2, allow_nan=False), encoding="utf-8"
            )
            print(f"{name}: {row['status']}", flush=True)
    catalog.close()
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(run())
