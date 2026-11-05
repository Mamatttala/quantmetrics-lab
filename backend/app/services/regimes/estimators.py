"""Independent fitted estimators, causal targets and family-specific outputs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture

from app.core.errors import ValidationFailure
from app.services.regimes.data import check_preprocessing
from app.services.regimes.hmm import fit_hmm
from app.services.regimes.neural import fit_neural
from app.services.regimes.registry import MODELS
from app.services.regimes.registry import parameters as resolve_parameters
from app.services.regimes.structural import cusum, pelt


def rule_labels(values, lower, upper, *, allow_equal=False):
    if lower > upper or (lower == upper and not allow_equal):
        raise ValidationFailure("Lower rule threshold must be below the upper threshold.")
    return np.where(values < lower, 0, np.where(values > upper, 2, 1))


def baseline(x, features, name, p):
    def col(key):
        if p[key] not in features:
            raise ValidationFailure(f"{p[key]} must be in the selected upstream features.")
        return x[:, features.index(p[key])]

    if name == "rules":
        return rule_labels(col("feature"), p["lower"], p["upper"])
    if name == "moving_average":
        slow = col("slow_column")
        if (slow == 0).any():
            raise ValidationFailure(
                "Moving-average reference contains zero; use unscaled C5 indicators."
            )
        # A zero-width MA band keeps exact equality neutral, as the registry permits.
        return rule_labels(
            col("fast_column") / slow - 1, -p["neutral_band"], p["neutral_band"], allow_equal=True
        )
    if name == "adx":
        return np.where(
            col("adx_column") < p["threshold"],
            1,
            np.where(col("plus_di_column") > col("minus_di_column"), 2, 0),
        )
    return rule_labels(col("trend_column"), p["lower"], p["upper"]) * 2 + (
        col("volatility_column") > p["volatility_threshold"]
    ).astype(int)


def target_values(frame, target, *, allow_unavailable=False):
    if target.source == "teacher":
        raise ValidationFailure("Teacher targets are only available in HMM → XGBoost experiments.")
    key = target.column if target.source == "column" else target.rule_feature
    if key not in frame:
        if allow_unavailable:
            return np.full(len(frame), None, dtype=object)
        raise ValidationFailure(f"Target feature {key} is missing. Provide it upstream.")
    values = frame[key].to_numpy()
    missing = pd.isna(values)
    if target.source == "rules":
        numeric = pd.to_numeric(frame[key], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        missing = ~np.isfinite(numeric)
        values = rule_labels(numeric, target.lower, target.upper)
    elif pd.api.types.is_numeric_dtype(frame[key]):
        missing |= ~np.isfinite(frame[key].to_numpy(dtype=float, na_value=np.nan))
    missing |= np.array([str(value) in {"nan", "<NA>"} for value in values], dtype=bool)
    if missing.any() and not allow_unavailable:
        raise ValidationFailure("The target column contains unavailable labels.")
    return np.asarray(
        [None if absent else str(value) for value, absent in zip(values, missing, strict=True)],
        dtype=object,
    )


def training_pairs(indices, target):
    horizon = target.horizon if target.objective == "future" else 0
    origins = indices[indices + horizon + target.availability_lag <= indices[-1]]
    return origins, origins + horizon


@dataclass
class Model:
    name: str
    family: str
    features: list[str]
    parameters: dict
    count: int
    estimator: object = None
    teacher: object = None
    vocabulary: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    def predict(self, x, method="filter", lag=5):
        if method != "filter" and self.family != "hmm":
            raise ValidationFailure("Smoothing and Viterbi decoding apply only to HMMs.")
        output = {"probabilities": None, "distances": None, "alerts": [], "scores": None}
        if self.family == "hmm":
            states, probability = self.estimator.predict(x, method, lag)
            return {**output, "states": states, "probabilities": probability}
        if self.family == "baseline":
            return {**output, "states": baseline(x, self.features, self.name, self.parameters)}
        if self.name == "kmeans":
            return {
                **output,
                "states": self.estimator.predict(x),
                "distances": self.estimator.transform(x),
            }
        if self.name == "gmm":
            probability = self.estimator.predict_proba(x)
            return {**output, "states": probability.argmax(axis=1), "probabilities": probability}
        if self.name == "pelt":
            states, alerts, score = pelt(
                x, self.parameters["penalty"], int(self.parameters["min_segment_length"])
            )
            return {**output, "states": states, "alerts": alerts, "scores": score}
        if self.name == "cusum":
            states, alerts, scores = cusum(
                x,
                self.estimator["center"],
                self.estimator["scale"],
                self.parameters["threshold"],
                self.parameters["drift"],
                int(self.parameters["min_segment_length"]),
            )
            return {**output, "states": states, "alerts": alerts, "scores": scores}
        predicted = self.estimator.predict_proba(x)
        probability = np.zeros((len(x), self.count))
        for j, state in enumerate(self.estimator.classes_):
            probability[:, int(state)] = predicted[:, j]
        invalid = ~np.isfinite(predicted).all(axis=1)
        probability[~invalid] /= probability[~invalid].sum(axis=1, keepdims=True)
        states = np.nan_to_num(probability).argmax(axis=1)
        states[invalid] = -1
        probability[invalid] = np.nan
        return {**output, "states": states, "probabilities": probability}


def fit(config, prepared):
    p = resolve_parameters(config.model, config.parameters)
    family = MODELS[config.model]["family"]
    count = config.states.count
    train_indices = prepared.spans["train"]
    x = prepared.x[train_indices]
    if len(x) < count * 2:
        raise ValidationFailure(
            "Training observations are insufficient for the requested state count."
        )
    model = Model(config.model, family, config.inputs.features, p, count)
    if family == "baseline":
        model.count = 6 if config.model == "trend_volatility" else 3
        baseline(x, model.features, model.name, p)
    elif family == "hmm":
        if config.model == "gaussian_hmm" and x.shape[1] != 1:
            raise ValidationFailure(
                "Gaussian HMM uses one selected feature. "
                "Choose Multivariate Gaussian HMM for multiple features."
            )
        if config.model == "multivariate_hmm" and x.shape[1] < 2:
            raise ValidationFailure("Multivariate HMM requires at least two features.")
        kind = "student_t" if config.model == "student_t_hmm" else "gaussian"
        model.estimator = fit_hmm(x, count, p, config.seed, kind)
        model.diagnostics = model.estimator.diagnostics(len(x))
    elif config.model == "kmeans":
        model.estimator = KMeans(
            n_clusters=count,
            init="k-means++" if p["initialization"] == "kmeans" else "random",
            n_init=int(p["n_init"]),
            max_iter=int(p["max_iter"]),
            tol=p["tol"],
            random_state=config.seed,
        ).fit(x)
        model.diagnostics = {
            "inertia": model.estimator.inertia_,
            "iterations": model.estimator.n_iter_,
            "converged": model.estimator.n_iter_ < p["max_iter"],
        }
    elif config.model == "gmm":
        model.estimator = GaussianMixture(
            n_components=count,
            covariance_type=p["covariance_type"],
            init_params=p["initialization"],
            n_init=int(p["n_init"]),
            max_iter=int(p["max_iter"]),
            tol=p["tol"],
            random_state=config.seed,
            reg_covar=1e-6,
        ).fit(x)
        model.diagnostics = {
            "aic": model.estimator.aic(x),
            "bic": model.estimator.bic(x),
            "log_likelihood": model.estimator.score(x) * len(x),
            "iterations": model.estimator.n_iter_,
            "converged": bool(model.estimator.converged_),
        }
    elif family == "structural":
        model.count = 0  # IDs are sequential segments, not recurring regimes.
        model.estimator = {"center": x.mean(axis=0), "scale": np.maximum(x.std(axis=0), 1e-8)}
        model.diagnostics = {
            "output_kind": "offline_segments"
            if model.name == "pelt"
            else "sequential_change_alerts"
        }
    else:
        if config.target is None:
            raise ValidationFailure(
                "Supervised/hybrid models require an explicit target configuration."
            )
        if config.target.source == "column" and config.target.column in model.features:
            raise ValidationFailure("A label column cannot also be a predictive input feature.")
        indices, label_indices = training_pairs(train_indices, config.target)
        if family == "hybrid":
            if config.target.source != "teacher":
                raise ValidationFailure("HMM → XGBoost requires causal teacher targets.")
            cut = int(len(train_indices) * p["teacher_fraction"])
            if cut < count * 2 or len(x) - cut < 12:
                raise ValidationFailure(
                    "The hybrid needs separate teacher-fit and student-training windows."
                )
            teacher_params = resolve_parameters(
                "multivariate_hmm", {"max_iter": p["teacher_max_iter"]}
            )
            check_preprocessing(
                prepared.steps,
                prepared.times.iloc[train_indices[0]],
                prepared.times.iloc[train_indices[cut - 1]],
                context="teacher training",
            )
            model.teacher = fit_hmm(x[:cut], count, teacher_params, config.seed)
            teacher_states, _ = model.teacher.predict(
                prepared.x[train_indices[0] : train_indices[-1] + 1]
            )
            keep = indices >= train_indices[0] + cut
            indices, label_indices = indices[keep], label_indices[keep]
            labels = teacher_states[label_indices - train_indices[0]]
            model.vocabulary = [str(i) for i in range(count)]
            model.diagnostics["teacher_fit_end_index"] = int(train_indices[0] + cut - 1)
            model.diagnostics["target_meaning"] = (
                "Teacher agreement measures imitation, not independent economic validity."
            )
        else:
            # Only labels eligible for fitting are required; later realized labels
            # belong to evaluation and cannot prevent training on historical pairs.
            raw = target_values(prepared.frame.iloc[label_indices], config.target)
            model.vocabulary = sorted(set(raw))
            model.count = len(model.vocabulary)
            labels = np.array([model.vocabulary.index(value) for value in raw])
        if len(np.unique(labels)) < 2 or len(indices) < 8:
            raise ValidationFailure(
                "Training requires at least two observed target classes and eight labeled origins."
            )
        if config.model in {"lstm", "gru"}:
            model.estimator = fit_neural(config.model, prepared.x, indices, labels, p, config.seed)
            model.diagnostics["training_loss"] = model.estimator.history
        elif config.model == "logistic":
            model.estimator = LogisticRegression(
                C=p["C"], max_iter=int(p["max_iter"]), random_state=config.seed
            ).fit(prepared.x[indices], labels)
        else:
            # XGBoost requires contiguous class indices; the wrapper preserves teacher IDs.
            unique, encoded = np.unique(labels, return_inverse=True)
            if config.model == "lightgbm":
                from lightgbm import LGBMClassifier

                estimator = LGBMClassifier(
                    n_estimators=int(p["n_estimators"]),
                    max_depth=int(p["max_depth"]),
                    learning_rate=p["learning_rate"],
                    random_state=config.seed,
                    verbosity=-1,
                    n_jobs=1,
                )
            else:
                from xgboost import XGBClassifier

                estimator = XGBClassifier(
                    n_estimators=int(p["n_estimators"]),
                    max_depth=int(p["max_depth"]),
                    learning_rate=p["learning_rate"],
                    random_state=config.seed,
                    n_jobs=1,
                )
            estimator.fit(prepared.x[indices], encoded)
            model.estimator = Classes(estimator, unique)
        model.diagnostics.update(
            training_origins=len(indices),
            last_training_origin=int(indices[-1]),
            last_training_label=int(label_indices[-1]),
            objective=config.target.objective,
            target_horizon=config.target.horizon if config.target.objective == "future" else 0,
            class_vocabulary=model.vocabulary,
        )
    return model


@dataclass
class Classes:
    estimator: object
    classes_: np.ndarray

    def predict_proba(self, x):
        return self.estimator.predict_proba(x)


def fitted_parameters(model):
    """Readable numerical parameters alongside the complete serialized estimator."""
    if model.family == "hmm":
        hmm = model.estimator
        return {
            "initial_probabilities": hmm.start,
            "transition_matrix": hmm.transition,
            "emission_means": hmm.emission.means,
            "emission_scales": hmm.emission.covariances,
            "mixture_weights": hmm.emission.weights,
            "emission_kind": hmm.emission.kind,
        }
    if model.name == "kmeans":
        return {"cluster_centers": model.estimator.cluster_centers_}
    if model.name == "gmm":
        return {
            "means": model.estimator.means_,
            "covariances": model.estimator.covariances_,
            "weights": model.estimator.weights_,
        }
    if model.name == "logistic":
        return {"coefficients": model.estimator.coef_, "intercepts": model.estimator.intercept_}
    if model.name in {"lstm", "gru"}:
        return {
            "trained_weight_shapes": {
                key: list(value.shape) for key, value in model.estimator.weights.items()
            },
            "weight_storage": "Complete trained weights are saved in model.bin.",
        }
    if model.family in {"supervised", "hybrid"}:
        result = {"feature_importances": model.estimator.estimator.feature_importances_}
        if model.teacher:
            result["teacher_transition_matrix"] = model.teacher.transition
            result["teacher_emission_means"] = model.teacher.emission.means
        return result
    return {"rules": model.parameters} if model.family == "baseline" else model.estimator
