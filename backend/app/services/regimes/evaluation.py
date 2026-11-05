"""D10 evaluation, interpretation, walk-forward validation, robustness and approval."""

from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd

from app.core.errors import ValidationFailure
from app.services.regimes.contracts import Experiment
from app.services.regimes.data import check_preprocessing, load, plan_rows
from app.services.regimes.estimators import fit
from app.services.regimes.metrics import (
    OCCUPANCY_NOTE,
    align,
    economic_columns,
    occupancy_drift,
    predictive,
    profiles,
    separation,
    statistical_labels,
    transitions,
)


def reopen(identifier, settings, catalog, store):
    artifact = store.get(identifier, "experiment")
    config = Experiment.model_validate(artifact["config"])
    prepared = load(catalog, settings, config.inputs)
    if prepared.contract != artifact["contract"]:
        raise ValidationFailure(
            "The experiment's upstream data or preprocessing contract has changed."
        )
    model = store.model(identifier)
    if model.teacher is not None:
        cutoff = model.diagnostics.get("teacher_fit_end_index")
        if cutoff is None or cutoff not in prepared.spans["train"]:
            raise ValidationFailure("The saved teacher training cutoff cannot be verified.")
        check_preprocessing(
            prepared.steps,
            prepared.times.iloc[prepared.spans["train"][0]],
            prepared.times.iloc[cutoff],
            context="teacher training",
        )
    return artifact, config, prepared, model


def summarize(model, config, prepared, indices, output):
    states = output["states"][indices]
    count = model.count or int(states.max() + 1)
    if count > 100:
        raise ValidationFailure(
            "This segmentation has more than 100 segments; "
            "increase the penalty or alert threshold for interactive analysis."
        )
    payload = {
        "observations": len(indices),
        "occupancy_note": OCCUPANCY_NOTE,
        "profiles": profiles(prepared, config.inputs, states, indices, count),
        "transitions": transitions(
            states,
            count,
            prepared.times.iloc[indices].astype(str).tolist(),
            model.estimator.transition if model.family == "hmm" else None,
        ),
        "predictive": predictive(model, prepared, config, indices, output),
    }
    if model.family in {"hmm", "clustering"}:
        payload["separation"] = separation(prepared.x[indices], states)
    if model.family == "hmm":
        # Conditional held-out likelihood uses only the causal preceding history.
        previous = None
        if indices[0] > 0:
            previous = model.estimator.predict(prepared.x[: indices[0]])[1][-1]
        payload["log_likelihood_per_observation"] = model.estimator.score(
            prepared.x[indices], previous
        ) / len(indices)
    elif model.name == "gmm":
        payload["log_likelihood_per_observation"] = float(
            model.estimator.score(prepared.x[indices])
        )
    return payload


def evaluate(identifier, settings, catalog, store):
    artifact, config, prepared, model = reopen(identifier, settings, catalog, store)
    output = model.predict(prepared.x)
    partitions = {}
    for role, indices in prepared.spans.items():
        if len(indices):
            partitions[role] = summarize(model, config, prepared, indices, output)
    train = prepared.spans["train"]
    # Label proposals are derived from training statistics only, never the test span.
    labels = statistical_labels(partitions["train"]["profiles"])
    economics = economic_columns(prepared, config.inputs)
    plot_indices = np.unique(
        np.linspace(0, len(prepared.x) - 1, min(2500, len(prepared.x))).astype(int)
    )
    charts = {
        "timestamps": prepared.times.iloc[plot_indices].astype(str).tolist(),
        "state_ids": output["states"][plot_indices].tolist(),
        "price": economics["price"][plot_indices].tolist() if "price" in economics else None,
        "returns": economics["return"][plot_indices].tolist() if "return" in economics else None,
        "return_type": economics["return_type"],
        "volatility": economics["volatility"][plot_indices].tolist()
        if "volatility" in economics
        else None,
        "probabilities": output["probabilities"][plot_indices].tolist()
        if output["probabilities"] is not None
        else None,
        "sampling": "Evenly sampled for display only; statistics and exports use every row.",
    }
    drift = {
        role: occupancy_drift(output["states"][train], output["states"][indices], model.count)
        if model.count
        else None
        for role, indices in prepared.spans.items()
        if role != "train" and len(indices)
    }
    return {
        "experiment_id": identifier,
        "model": model.name,
        "diagnostics": artifact["diagnostics"],
        "fitted_parameters": artifact.get("fitted_parameters", {}),
        "partitions": partitions,
        "labels": latest_labels(store, identifier, labels),
        "label_proposal": labels,
        "charts": charts,
        "state_distribution_drift": drift,
        "causal": model.name != "pelt",
        "economic_note": economics["return_note"]
        + " Contemporaneous state descriptions are not trading returns or evidence "
        "of predictability. Missing upstream ADX/RSI remains unavailable.",
        "structural_note": "Sequential segment IDs are not recurring probabilistic states."
        if model.family == "structural"
        else None,
    }


def latest_labels(store, identifier, default):
    for item in store.list("labels"):
        if item.get("experiment_id") == identifier:
            return item["labels"]
    return default


def label(identifier, request, settings, catalog, store):
    artifact, config, prepared, model = reopen(identifier, settings, catalog, store)
    output = model.predict(prepared.x)
    train = prepared.spans["train"]
    count = model.count or int(output["states"].max() + 1)
    state_profiles = profiles(prepared, config.inputs, output["states"][train], train, count)
    if request.mode == "manual":
        if set(request.labels) != {str(k) for k in range(count)} or any(
            not value.strip() for value in request.labels.values()
        ):
            raise ValidationFailure("Supply a nonempty label for every original state ID.")
        labels = request.labels
    else:
        labels = statistical_labels(state_profiles, request.template)
    return store.save(
        "labels",
        {
            "name": artifact["name"],
            "experiment_id": identifier,
            "labels": labels,
            "mode": request.mode,
            "template": request.template,
            "source": "training state statistics",
        },
        prepared.meta,
    )


def require_predictive_evidence(model, test, folds=()):
    """F3: retain the two-observation floor, measured in scorable OOS pairs.

    This is an operational minimum, not a statistical significance or accuracy gate.
    log_loss_rows counts realized targets in the fitted class vocabulary with a
    valid prediction; unknown targets remain reported separately.
    """
    if model.family not in {"supervised", "hybrid"}:
        return
    for scope, summary in [("saved test", test), *[(f"fold {f.get('fold')}", f) for f in folds]]:
        pairs = summary.get("predictive", {}).get("log_loss_rows", 0)
        if pairs < 2:
            raise ValidationFailure(
                f"Insufficient predictive evidence in {scope}: {pairs} usable out-of-sample "
                "prediction/target pairs; at least 2 are required. "
                "Use a longer test window with realized targets and a complete horizon."
            )


def fold_alignment(model, prepared, fitted, data, prediction, catalog):
    """F11: match observations, never feed another fit's coordinates to a model."""
    if any(prepared.contract[key] != data.contract[key] for key in ("asset", "timeframe")):
        return {"unavailable": "State alignment requires the same asset and timeframe."}
    # Common primary observations establish bar identity. Sharing only an auxiliary
    # merged parent does not. load() still validates preprocessing on every branch.
    shared = {prepared.meta.dataset_id, *catalog.ancestors(prepared.meta.dataset_id)}.intersection(
        {data.meta.dataset_id, *catalog.ancestors(data.meta.dataset_id)}
    )
    if not shared:
        return {"unavailable": "State alignment requires shared primary observation lineage."}
    train = data.spans["train"]
    positions = pd.Index(prepared.times).get_indexer(data.times.iloc[train])
    matched = positions >= 0
    if matched.sum() < 2:
        return {"unavailable": "State alignment needs at least two shared training timestamps."}
    # Predict the full chronological history in each model's own prepared space.
    # Slicing inputs here would also restart an HMM/recurrent model's history.
    reference = model.predict(prepared.x)["states"][positions[matched]]
    other = prediction["states"][train[matched]]
    valid = (reference >= 0) & (reference < model.count) & (other >= 0) & (other < fitted.count)
    if valid.sum() < 2:
        return {"unavailable": "State alignment needs at least two jointly assigned observations."}
    return {
        **align(reference[valid], other[valid]),
        "observations": int(valid.sum()),
        "reference_preprocessing_signature": prepared.contract["preprocessing_signature"],
        "fold_preprocessing_signature": data.contract["preprocessing_signature"],
        "method": "Maximum overlap at shared fold-training timestamps, each model in its own "
        "prepared representation with full chronological context. Retrospective label-switching "
        "diagnostic only; not an out-of-sample score or a deployment mapping.",
    }


def validate(identifier, request, settings, catalog, store, progress=None):
    artifact, config, prepared, model = reopen(identifier, settings, catalog, store)
    if model.name == "pelt":
        raise ValidationFailure(
            "PELT is retrospective segmentation. It cannot be approved as a causal live-state "
            "model; use CUSUM for sequential alerts."
        )
    begin = time.perf_counter()
    # Always validate the exact saved model that will be approved for deployment.
    result = evaluate(identifier, settings, catalog, store)
    folds = []
    if request.mode != "chronological":
        if not request.split_plan_id:
            raise ValidationFailure(
                "Walk-forward refitting consumes a saved expanding/rolling C6 plan."
            )
        plan, rows = plan_rows(catalog, settings, request.split_plan_id)
        if plan.provenance.get("split_type") != f"tscv_{request.mode}":
            raise ValidationFailure(
                "Selected C6 plan does not match the requested validation mode."
            )
        for fold in sorted({row.fold for row in rows}):
            if progress:
                progress(f"Refitting chronological fold {fold}")
            design = Experiment.model_validate(artifact.get("research_config", artifact["config"]))
            fold_config = design.model_copy(deep=True)
            fold_config.inputs.fold = fold
            fold_config.inputs.split_plan_id = request.split_plan_id
            fold_config.inputs.dataset_id = request.prepared_datasets.get(
                fold, config.inputs.dataset_id
            )
            data = load(catalog, settings, fold_config.inputs)
            if design.search.method != "manual" or design.states.mode == "automatic":
                # Re-select inside each fold; choices made on later original validation
                # data cannot be reused as though they were known at an earlier origin.
                from app.services.regimes.experiments import execute

                search = execute(fold_config, settings, catalog, store, progress)
                selected = store.get(search["selected_experiment_id"], "experiment")
                fitted = store.model(selected["id"])
                fold_config = Experiment.model_validate(selected["config"])
            else:
                fitted = fit(fold_config, data)
            prediction = fitted.predict(data.x)
            indices = data.spans["test"]
            metric = summarize(fitted, fold_config, data, indices, prediction)
            require_predictive_evidence(fitted, metric)
            train = data.spans["train"]
            metric.update(
                fold=fold,
                train_end=data.times.iloc[train[-1]].isoformat(),
                test_start=data.times.iloc[indices[0]].isoformat(),
                test_end=data.times.iloc[indices[-1]].isoformat(),
                preprocessing_signature=data.contract["preprocessing_signature"],
                selected_parameters=fitted.parameters,
                selected_state_count=fitted.count,
                selection_scope="fold-local training/validation",
                state_distribution_drift=occupancy_drift(
                    prediction["states"][train], prediction["states"][indices], fitted.count
                )
                if fitted.count
                else None,
            )
            # Mapping is a diagnostic of label switching; no economic names are forced.
            if fitted.count:
                metric["state_alignment"] = fold_alignment(
                    model, prepared, fitted, data, prediction, catalog
                )
            folds.append(metric)
    test = result["partitions"]["test"]
    require_predictive_evidence(model, test, folds)
    if test["observations"] < 2:
        raise ValidationFailure(
            "Approval requires at least two chronologically later observations."
        )
    return store.save(
        "validation",
        {
            "name": artifact["name"],
            "experiment_id": identifier,
            "request": request.model_dump(mode="json"),
            "validated": True,
            "causal": True,
            "test": test,
            "validation": result["partitions"].get("val"),
            "folds": folds,
            "state_distribution_drift": result["state_distribution_drift"],
            "seconds": time.perf_counter() - begin,
            "note": "Parameters were selected without test data. Validation records evidence; "
            "approval remains a researcher decision.",
        },
        prepared.meta,
    )


def robustness(identifier, request, settings, catalog, store, progress=None):
    artifact, config, prepared, reference = reopen(identifier, settings, catalog, store)
    if reference.family == "structural":
        raise ValidationFailure(
            "Hidden-state stability metrics do not apply to sequential segment numbering."
        )
    variants = [{"seed": seed} for seed in request.seeds]
    variants += [{"count": k} for k in request.counts]
    variants += [{"features": features} for features in request.feature_sets]
    variants += [{"parameters": p} for p in request.configurations]
    variants += [{"fraction": f} for f in request.training_fractions if f != 1]
    variants += [{"parameters": {"initialization": value}} for value in request.initializations]
    if len(variants) > 40:
        raise ValidationFailure("Robustness is limited to 40 independent refits per request.")
    indices = prepared.spans["val"] if len(prepared.spans["val"]) else prepared.spans["test"]
    baseline = reference.predict(prepared.x)["states"][indices]
    rows = []
    for i, variant in enumerate(variants):
        if progress:
            progress(f"Robustness refit {i + 1} of {len(variants)}")
        current = config.model_copy(deep=True)
        data = copy.copy(prepared)
        data.spans = dict(prepared.spans)
        try:
            if "seed" in variant:
                current.seed = variant["seed"]
            if "count" in variant:
                if not 2 <= variant["count"] <= 12:
                    raise ValidationFailure("State counts must be between 2 and 12.")
                current.states.count = variant["count"]
            if "parameters" in variant:
                current.parameters |= variant["parameters"]
            if "features" in variant:
                current.inputs.features = variant["features"]
                data = load(catalog, settings, current.inputs)
            if "fraction" in variant:
                fraction = variant["fraction"]
                if not 0 < fraction <= 1:
                    raise ValidationFailure("Training fractions must be in (0,1].")
                train = prepared.spans["train"]
                data.spans["train"] = train[-max(12, int(len(train) * fraction)) :]
                check_preprocessing(
                    data.steps, data.times.iloc[data.spans["train"][0]], data.times.iloc[train[-1]]
                )
            started = time.perf_counter()
            fitted = fit(current, data)
            states = fitted.predict(data.x)["states"][indices]
            valid = (states >= 0) & (baseline >= 0)
            scores = align(baseline[valid], states[valid])
            rows.append(
                {
                    "variant": variant,
                    "status": "succeeded",
                    **scores,
                    "seconds": time.perf_counter() - started,
                }
            )
        except (ValidationFailure, ValueError) as exc:
            rows.append({"variant": variant, "status": "unavailable", "reason": str(exc)})
    return store.save(
        "robustness",
        {
            "name": artifact["name"],
            "experiment_id": identifier,
            "rows": rows,
            "alignment": "Maximum overlap via Hungarian assignment; ARI is permutation invariant.",
        },
        prepared.meta,
    )


def approve(identifier, request, settings, catalog, store):
    artifact, config, prepared, model = reopen(identifier, settings, catalog, store)
    validation = store.get(request.validation_id, "validation")
    if (
        validation.get("experiment_id") != identifier
        or not validation.get("validated")
        or not validation.get("causal")
    ):
        raise ValidationFailure(
            "Approval requires a completed causal validation of this exact experiment."
        )
    result = evaluate(identifier, settings, catalog, store)
    # Recheck actual saved-model evidence, including pre-remediation validation records.
    require_predictive_evidence(model, result["partitions"]["test"], validation.get("folds", []))
    labels = result["labels"]
    return store.save(
        "approval",
        {
            "name": artifact["name"],
            "experiment_id": identifier,
            "validation_id": validation["id"],
            "rationale": request.rationale,
            "model": model.name,
            "labels": labels,
            "contract": prepared.contract,
            "model_checksum": artifact["files"]["model.bin"],
            "version": artifact["id"],
            "objective": config.target.objective if config.target else "current",
        },
        prepared.meta,
    )


def compare(identifiers, settings, catalog, store):
    rows = []
    contracts = []
    for identifier in identifiers:
        artifact, config, prepared, model = reopen(identifier, settings, catalog, store)
        result = evaluate(identifier, settings, catalog, store)
        test = result["partitions"]["test"]
        validations = [v for v in store.list("validation") if v.get("experiment_id") == identifier]
        robust = [r for r in store.list("robustness") if r.get("experiment_id") == identifier]
        stable = (
            [r["adjusted_rand"] for r in robust[0]["rows"] if r["status"] == "succeeded"]
            if robust
            else []
        )
        profiles_ = test["profiles"]
        rows.append(
            {
                "id": identifier,
                "name": artifact["name"],
                "model": model.name,
                "state_count": model.count,
                "bic": artifact["diagnostics"].get("bic"),
                "silhouette": test.get("separation", {}).get("silhouette"),
                "test_log_likelihood": test.get("log_likelihood_per_observation"),
                "test_accuracy": test["predictive"].get("accuracy"),
                "test_log_loss": test["predictive"].get("log_loss"),
                "switching_frequency": test["transitions"]["switching_frequency"],
                "economic_profiles_available": sum(p["mean_return"] is not None for p in profiles_),
                "stability_ari": float(np.mean(stable)) if stable else None,
                "seconds": artifact["diagnostics"].get("seconds"),
                "validated": any(
                    v.get("validated") and v.get("causal")
                    and (
                        model.family not in {"supervised", "hybrid"}
                        or all(
                            s.get("predictive", {}).get("log_loss_rows", 0) >= 2
                            for s in [test, *v.get("folds", [])]
                        )
                    )
                    for v in validations
                ),
            }
        )
        contracts.append((prepared.contract, config.target.model_dump() if config.target else None))
    return {
        "rows": rows,
        "comparable": all(value == contracts[0] for value in contracts),
        "note": "Side-by-side diagnostics; likelihood criteria are comparable only for the same "
        "observations and feature representation. Teacher agreement is not economic validity.",
    }
