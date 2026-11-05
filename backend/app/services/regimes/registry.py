"""Capabilities and conditional forms share a single model registry."""

from __future__ import annotations

import math
from importlib.util import find_spec

from app.core.errors import ValidationFailure


def number(default, minimum=None, maximum=None):
    return {"type": "number", "default": default, "min": minimum, "max": maximum}


def choice(default, values):
    return {"type": "select", "default": default, "options": values}


def column():
    return {"type": "column", "default": ""}


ITER = {"max_iter": number(100, 1, 2000), "tol": number(0.0001, 1e-9, 1)}
INIT = {"initialization": choice("kmeans", ["kmeans", "random"])}
COV = {"covariance_type": choice("full", ["full", "diag", "spherical", "tied"])}
HMM = {
    **ITER,
    **INIT,
    **COV,
    "transition_stickiness": number(0.9, 0, 1),
    "start_probabilities": {"type": "json", "default": None},
    "transition_matrix": {"type": "json", "default": None},
}
TREE = {
    "n_estimators": number(100, 1, 2000),
    "max_depth": number(3, 1, 20),
    "learning_rate": number(0.05, 0.0001, 1),
}
RECURRENT = {
    "sequence_length": number(20, 2, 500),
    "hidden_size": number(32, 2, 256),
    "layers": number(1, 1, 4),
    "epochs": number(25, 1, 500),
    "learning_rate": number(0.001, 0.00001, 1),
    "batch_size": number(64, 1, 4096),
}


def spec(
    name,
    family,
    params,
    *,
    probabilities=False,
    count=True,
    dependency=None,
    decoding=False,
    criteria=None,
):
    return dict(
        name=name,
        family=family,
        parameters=params,
        probabilities=probabilities,
        state_count=count,
        dependency=dependency,
        decoding=decoding,
        criteria=criteria or [],
    )


LIKELIHOOD = ["bic", "aic"]
SEPARATION = ["silhouette", "davies_bouldin", "calinski_harabasz"]
PREDICTIVE = ["validation_log_loss", "validation_accuracy"]
MODELS = {
    "rules": spec(
        "Rule-Based Classification",
        "baseline",
        {"feature": column(), "lower": number(-0.001), "upper": number(0.001)},
        count=False,
    ),
    "moving_average": spec(
        "Moving Average Trend",
        "baseline",
        {"fast_column": column(), "slow_column": column(), "neutral_band": number(0.001, 0)},
        count=False,
    ),
    "adx": spec(
        "ADX Trend Classification",
        "baseline",
        {
            "adx_column": column(),
            "plus_di_column": column(),
            "minus_di_column": column(),
            "threshold": number(25, 0, 100),
        },
        count=False,
    ),
    "trend_volatility": spec(
        "Trend × Volatility",
        "baseline",
        {
            "trend_column": column(),
            "volatility_column": column(),
            "lower": number(-0.001),
            "upper": number(0.001),
            "volatility_threshold": number(0.02, 0),
        },
        count=False,
    ),
    "gaussian_hmm": spec(
        "Gaussian HMM", "hmm", HMM, probabilities=True, decoding=True, criteria=LIKELIHOOD
    ),
    "multivariate_hmm": spec(
        "Multivariate Gaussian HMM",
        "hmm",
        HMM,
        probabilities=True,
        decoding=True,
        criteria=LIKELIHOOD,
    ),
    "student_t_hmm": spec(
        "Student-t HMM",
        "hmm",
        {**HMM, "degrees_of_freedom": number(5, 2.01, 100)},
        probabilities=True,
        decoding=True,
        criteria=LIKELIHOOD,
    ),
    "gmm_hmm": spec(
        "GMM-HMM",
        "hmm",
        {**HMM, "mixtures": number(2, 1, 6)},
        probabilities=True,
        decoding=True,
        criteria=LIKELIHOOD,
    ),
    "kmeans": spec(
        "K-Means", "clustering", {**ITER, **INIT, "n_init": number(10, 1, 50)}, criteria=SEPARATION
    ),
    "gmm": spec(
        "Gaussian Mixture",
        "clustering",
        {**ITER, **INIT, **COV, "n_init": number(3, 1, 50)},
        probabilities=True,
        criteria=LIKELIHOOD + SEPARATION,
    ),
    "pelt": spec(
        "PELT — Offline Segmentation",
        "structural",
        {
            "cost": choice("l2", ["l2"]),
            "penalty": number(10, 0.0001),
            "min_segment_length": number(10, 2, 10000),
        },
        count=False,
    ),
    "cusum": spec(
        "CUSUM — Sequential Detection",
        "structural",
        {
            "threshold": number(5, 0.01),
            "drift": number(0.5, 0),
            "min_segment_length": number(5, 1, 10000),
        },
        count=False,
    ),
    "logistic": spec(
        "Logistic Regression",
        "supervised",
        {"C": number(1, 0.00001), "max_iter": number(500, 1, 10000)},
        probabilities=True,
        count=False,
        criteria=PREDICTIVE,
    ),
    "xgboost": spec(
        "XGBoost",
        "supervised",
        TREE,
        probabilities=True,
        count=False,
        dependency="xgboost",
        criteria=PREDICTIVE,
    ),
    "lightgbm": spec(
        "LightGBM",
        "supervised",
        TREE,
        probabilities=True,
        count=False,
        dependency="lightgbm",
        criteria=PREDICTIVE,
    ),
    "lstm": spec(
        "LSTM",
        "supervised",
        RECURRENT,
        probabilities=True,
        count=False,
        dependency="torch",
        criteria=PREDICTIVE,
    ),
    "gru": spec(
        "GRU",
        "supervised",
        RECURRENT,
        probabilities=True,
        count=False,
        dependency="torch",
        criteria=PREDICTIVE,
    ),
    "hmm_xgboost": spec(
        "HMM → XGBoost",
        "hybrid",
        {
            **TREE,
            "teacher_fraction": number(0.5, 0.2, 0.8),
            "teacher_max_iter": number(100, 1, 2000),
        },
        probabilities=True,
        dependency="xgboost",
        criteria=PREDICTIVE,
    ),
}


def catalog():
    return [
        {
            "id": key,
            **value,
            "available": not value["dependency"] or find_spec(value["dependency"]) is not None,
            "availability_reason": None
            if not value["dependency"] or find_spec(value["dependency"])
            else f"Install {value['dependency']} in the backend environment.",
        }
        for key, value in MODELS.items()
    ]


def parameters(model, supplied):
    if model not in MODELS:
        raise ValidationFailure(f"Unknown model: {model}.")
    entry = MODELS[model]
    if entry["dependency"] and find_spec(entry["dependency"]) is None:
        raise ValidationFailure(f"Install {entry['dependency']} to execute {entry['name']}.")
    fields = entry["parameters"]
    unknown = set(supplied) - set(fields)
    if unknown:
        raise ValidationFailure(f"Unsupported {model} settings: {', '.join(sorted(unknown))}.")
    result = {key: field["default"] for key, field in fields.items()} | supplied
    for key, value in result.items():
        field = fields[key]
        if field["type"] == "number":
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
            ):
                raise ValidationFailure(f"{key} must be numeric.")
            if isinstance(field["default"], int) and int(value) != value:
                raise ValidationFailure(f"{key} must be an integer.")
            if (
                field.get("min") is not None
                and value < field["min"]
                or field.get("max") is not None
                and value > field["max"]
            ):
                raise ValidationFailure(f"{key} is outside its supported range.")
        if field["type"] == "select" and value not in field["options"]:
            raise ValidationFailure(f"Unsupported {key}: {value}.")
        if field["type"] == "column" and not value:
            raise ValidationFailure(f"Choose an upstream {key}; generate indicators in C5.")
    return result
