"""D9 fitting, compatible candidate metrics and reproducible optimization."""

from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd

from app.core.errors import ValidationFailure
from app.services.regimes.artifacts import clean
from app.services.regimes.data import load
from app.services.regimes.estimators import fit, fitted_parameters
from app.services.regimes.metrics import OCCUPANCY_NOTE, occupancy, predictive, separation
from app.services.regimes.registry import MODELS, parameters


def assignments(prepared, model, output):
    frame = pd.DataFrame(
        {"timestamp": prepared.times.astype(str), "state_id": output["states"], "split": "unused"}
    )
    for role, indices in prepared.spans.items():
        frame.loc[indices, "split"] = role
    frame["estimate_scope"] = np.where(
        np.arange(len(frame)) > prepared.spans["train"][-1], "out_of_sample", "in_sample"
    )
    if model.name == "pelt":
        frame["estimate_scope"] = "retrospective_segmentation"
    if output["probabilities"] is not None:
        for k in range(model.count):
            frame[f"probability_{k}"] = output["probabilities"][:, k]
    if output["distances"] is not None:
        for k in range(model.count):
            frame[f"distance_{k}"] = output["distances"][:, k]
    frame["change_alert"] = False
    if output["alerts"]:
        frame.loc[output["alerts"], "change_alert"] = True
    return frame


def candidate(config, prepared):
    begin = time.perf_counter()
    model = fit(config, prepared)
    # PELT fits each split independently. No later observations enter training scores.
    output = model.predict(prepared.x)
    train = prepared.spans["train"]
    train_output = model.predict(prepared.x[train]) if model.family == "structural" else None
    states = train_output["states"] if train_output is not None else output["states"][train]
    diagnostics = dict(model.diagnostics)
    if model.family in {"hmm", "clustering"}:
        diagnostics.update(separation(prepared.x[train], states))
        distribution = occupancy(states, model.count)
        diagnostics["occupancy"] = distribution.tolist()
        diagnostics["occupancy_note"] = OCCUPANCY_NOTE
        if distribution.min() < config.states.minimum_occupancy:
            raise ValidationFailure(
                "A discovered state violates the configured minimum training occupancy."
            )
    validation = prepared.spans["val"]
    if len(validation) and model.family in {"supervised", "hybrid"}:
        diagnostics["validation"] = predictive(model, prepared, config, validation, output)
    diagnostics["seconds"] = time.perf_counter() - begin
    return model, output, diagnostics


def execute(config, settings, catalog, store, progress=None):
    if config.inputs.return_column and config.inputs.return_type is None:
        raise ValidationFailure("Declare return_type as simple or log for the return column.")
    if config.model not in MODELS:
        raise ValidationFailure("Unknown model family.")
    entry = MODELS[config.model]
    parameters(config.model, config.parameters)
    if config.states.mode == "automatic" and not entry["state_count"]:
        raise ValidationFailure("Automatic state count is not applicable to this model.")
    search = config.search
    optimizing = search.method != "manual" or config.states.mode == "automatic"
    if optimizing and config.states.criterion not in entry["criteria"]:
        raise ValidationFailure(f"{config.states.criterion} is incompatible with {config.model}.")
    if search.method != "manual" and not search.space:
        raise ValidationFailure("Choose a nonempty model-specific optimization space.")
    for key, values in search.space.items():
        if not values or key not in entry["parameters"] or len(values) > 30:
            raise ValidationFailure(f"Invalid search range for {key}.")
        for value in values:
            parameters(config.model, config.parameters | {key: value})
    prepared = load(catalog, settings, config.inputs)
    if (
        optimizing
        and config.states.criterion.startswith("validation_")
        and not len(prepared.spans["val"])
    ):
        raise ValidationFailure(
            "Predictive optimization requires a separate C6 validation span; "
            "test data cannot select parameters."
        )
    counts = (
        sorted(set(config.states.candidates))
        if config.states.mode == "automatic"
        else [config.states.count]
    )
    trials = []
    best = None
    best_value = np.inf
    created_ids = []

    def run_trial(count, values):
        nonlocal best, best_value
        current = config.model_copy(deep=True)
        current.parameters = config.parameters | values
        current.states.count = count
        current.states.mode = "fixed"
        current.search.method = "manual"
        begin = time.perf_counter()
        trial = {
            "number": len(trials) + 1,
            "count": count,
            "parameters": current.parameters,
            "status": "running",
        }
        if progress:
            progress(f"Fitting candidate {trial['number']} · {config.model} · {count} states")
        try:
            model, output, diagnostics = candidate(current, prepared)
            criterion = config.states.criterion
            score = diagnostics.get(criterion)
            if criterion.startswith("validation_"):
                score = diagnostics.get("validation", {}).get(criterion.removeprefix("validation_"))
            if optimizing and (score is None or not np.isfinite(score)):
                raise ValidationFailure(
                    "The requested selection metric is unavailable for this candidate."
                )
            value = float(score or 0)
            if criterion in {"silhouette", "calinski_harabasz", "validation_accuracy"}:
                value *= -1
            artifact = store.save(
                "experiment",
                {
                    "name": current.name,
                    "config": current.model_dump(mode="json"),
                    "research_config": config.model_dump(mode="json"),
                    "contract": prepared.contract,
                    "diagnostics": diagnostics,
                    "fitted_parameters": fitted_parameters(model),
                    "model": config.model,
                    "state_count": model.count,
                    "runtime_versions": runtime_versions(),
                },
                prepared.meta,
                model=model,
                frame=assignments(prepared, model, output),
            )
            created_ids.append(artifact["id"])
            trial.update(
                status="succeeded", score=score, artifact_id=artifact["id"], diagnostics=diagnostics
            )
            if best is None or value < best_value:
                best, best_value = artifact, value
        except (ValidationFailure, ValueError, np.linalg.LinAlgError) as exc:
            value = float("inf")
            trial.update(status="failed", error=str(exc))
        trial["seconds"] = time.perf_counter() - begin
        trials.append(trial)
        return value

    if search.method in {"bayesian", "optuna"}:
        try:
            import optuna
        except ImportError as exc:
            raise ValidationFailure("Install Optuna to run Bayesian optimization.") from exc
        # Both are explicit real TPE optimization; no random-search substitution.
        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=config.seed)
        )

        def objective(trial):
            count = trial.suggest_categorical("state_count", counts)
            values = {
                key: trial.suggest_categorical(key, options)
                for key, options in search.space.items()
            }
            return run_trial(count, values)

        study.optimize(objective, n_trials=search.trials)
    else:
        keys = list(search.space) if search.method != "manual" else []
        total = len(counts) * int(np.prod([len(search.space[key]) for key in keys], dtype=np.int64))
        if search.method == "grid" and total > 100:
            raise ValidationFailure(
                "Grid search exceeds the 100-fit execution limit. "
                "Narrow the space or use random/Bayesian search."
            )
        if search.method == "random":
            rng = np.random.default_rng(config.seed)
            visited = set()
            for _ in range(min(search.trials, total)):
                for _attempt in range(1000):
                    choice = (
                        int(rng.integers(len(counts))),
                        *(int(rng.integers(len(search.space[key]))) for key in keys),
                    )
                    if choice not in visited:
                        break
                visited.add(choice)
                run_trial(
                    counts[choice[0]],
                    {key: search.space[key][choice[i + 1]] for i, key in enumerate(keys)},
                )
        else:
            for count in counts:
                for values in itertools.product(*(search.space[key] for key in keys)):
                    run_trial(count, dict(zip(keys, values, strict=True)))
    if best is None:
        raise ValidationFailure(
            "No candidate completed successfully. "
            + "; ".join(t.get("error", "") for t in trials)[:3000]
        )
    return store.save(
        "search",
        {
            "name": config.name,
            "config": config.model_dump(mode="json"),
            "trials": clean(trials),
            "selected_experiment_id": best["id"],
            "experiment_ids": created_ids,
            "selection_scope": "training/validation only; D10 approval still required",
        },
        prepared.meta,
    )


def runtime_versions():
    import importlib.metadata
    import platform

    versions = {"python": platform.python_version(), "engine": "regimes-v1"}
    for name in ("numpy", "scipy", "scikit-learn", "xgboost", "lightgbm", "torch", "optuna"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions
