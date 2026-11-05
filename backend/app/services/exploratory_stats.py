"""Pure C8 preprocessing and compatibility entry points; no I/O.

Fit indices refer to source-row indices. Transforming never drops/reorders rows,
so lags retain observation positions across missing values.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.core.errors import ValidationFailure

STATELESS = frozenset(
    {"first_difference", "percent_change", "log_difference", "absolute_return", "squared_return"}
)
FITTED = frozenset({"standard_scale", "robust_scale", "minmax_scale", "winsorize_1_99"})
LIMITS = {
    "max_features": 250,
    "max_controls": 30,
    "max_rolling_window": 5000,
    "max_ccf_lag_abs": 120,
    "max_parallel_permutations": 200,
    "max_bootstrap_samples": 500,
    "max_pca_components": 100,
    "max_groups": 50,
    "max_rolling_points": 250_000,
    "min_obs": 30,
    "min_group_obs": 30,
}


def finite_frame(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return frame[features].astype(float).replace([np.inf, -np.inf], np.nan)


def stateless_transform(frame: pd.DataFrame, features: list[str], mode: str) -> pd.DataFrame:
    result = frame.copy()
    values = finite_frame(frame, features)
    if mode == "log_difference":
        if (values <= 0).any().any():
            raise ValidationFailure("log_difference requires strictly positive non-missing values.")
        values = pd.DataFrame(np.log(values), index=values.index, columns=features).diff()
    elif mode == "first_difference":
        values = values.diff()
    elif mode in {"percent_change", "absolute_return", "squared_return"}:
        values = values.pct_change(fill_method=None)
        if mode == "absolute_return":
            values = values.abs()
        elif mode == "squared_return":
            values = values**2
    result[features] = values.replace([np.inf, -np.inf], np.nan)
    return result


def preprocess(
    frame: pd.DataFrame,
    fit_index: pd.Index,
    features: list[str],
    *,
    pre_transform: str = "none",
    missing_policy: str = "pairwise_complete",
    constant_policy: str = "exclude_constants",
    constant_tolerance: float = 1e-12,
    stateless_done: bool = False,
) -> tuple[pd.DataFrame, list[str], dict[str, Any], dict[str, Any], list[str], list[str]]:
    """Fit on fit_index only, then apply fixed parameters to every source row."""
    result = frame.copy()
    result[features] = finite_frame(frame, features)
    if pre_transform in STATELESS and not stateless_done:
        result = stateless_transform(result, features, pre_transform)
    fit = finite_frame(result.loc[fit_index], features)
    original_missing = fit.isna().any(axis=1)
    parameters: dict[str, Any] = {
        "pre_transform": pre_transform,
        "missing_policy": missing_policy,
        "fit_rows": len(fit),
        "transforms": {},
        "imputation": {},
    }
    warnings: list[str] = []
    if pre_transform in FITTED:
        for feature in features:
            column = fit[feature].dropna()
            if column.empty:
                continue
            if pre_transform == "winsorize_1_99":
                low, high = float(column.quantile(0.01)), float(column.quantile(0.99))
                result[feature] = result[feature].clip(low, high)
                parameters["transforms"][feature] = {"low": low, "high": high}
            else:
                if pre_transform == "standard_scale":
                    center, scale = float(column.mean()), float(column.std(ddof=1))
                elif pre_transform == "robust_scale":
                    center = float(column.median())
                    scale = float(column.quantile(0.75) - column.quantile(0.25))
                else:
                    center, scale = float(column.min()), float(column.max() - column.min())
                if not np.isfinite(scale) or scale <= 0:
                    scale = 1.0
                result[feature] = (result[feature] - center) / scale
                parameters["transforms"][feature] = {"center": center, "scale": scale}
    elif pre_transform == "rank_transform":
        result[features] = np.nan
        result.loc[fit_index, features] = fit.rank(method="average", pct=True)
        parameters["rank_method"] = "average percentile rank within fit rows; diagnostics only"
        warnings.append("Rank transform is diagnostic-only; scores outside fit rows are undefined.")
    impute = (
        missing_policy
        if missing_policy in {"median_impute_scope", "mean_impute_scope"}
        else (
            pre_transform if pre_transform in {"median_impute_scope", "mean_impute_scope"} else None
        )
    )
    if impute:
        for feature in features:
            column = result.loc[fit_index, feature]
            value = column.median() if impute == "median_impute_scope" else column.mean()
            parameters["imputation"][feature] = float(value) if pd.notna(value) else None
            if pd.notna(value):
                result[feature] = result[feature].fillna(value)
    fitted = finite_frame(result.loc[fit_index], features)
    constants, near_zero = [], []
    for feature in features:
        values = fitted[feature].dropna().to_numpy()
        if len(values) == 0 or np.ptp(values) <= constant_tolerance * max(
            1.0, float(np.max(np.abs(values)))
        ):
            constants.append(feature)
        elif np.std(values) <= 1e-8 * max(1.0, float(np.max(np.abs(values)))):
            near_zero.append(feature)
    if constants and constant_policy == "refuse_if_constants":
        raise ValidationFailure("Constant/empty fit features: " + ", ".join(constants))
    excluded = constants if constant_policy == "exclude_constants" else []
    if constants:
        warnings.append(
            ("Excluded" if excluded else "Retained")
            + " constant/empty fit columns: "
            + ", ".join(constants)
        )
    if near_zero:
        warnings.append("Near-zero fit variance: " + ", ".join(near_zero))
    available = [feature for feature in features if feature not in excluded]
    remaining_missing = fitted[available].isna().any(axis=1)
    if missing_policy == "drop_any_missing" and available:
        invalid = result[available].isna().any(axis=1)
        result.loc[invalid, available] = np.nan
    summary = {
        "policy": missing_policy,
        "n_fit_rows": len(fit),
        "n_missing_rows": int(original_missing.sum()),
        "n_missing_after_policy": int(remaining_missing.sum()),
        "n_dropped_rows": int(remaining_missing.sum())
        if missing_policy == "drop_any_missing"
        else 0,
        "per_feature_missing": {name: int(fit[name].isna().sum()) for name in features},
        "imputation_parameters": parameters["imputation"],
    }
    return result, available, parameters, summary, excluded, warnings


def pair_statistics(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from app.services.exploratory_corr import pair_statistics as calculate

    return calculate(*args, **kwargs)


def vif_table(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
    from app.services.exploratory_corr import vif_table as calculate

    return calculate(*args, **kwargs)


def correlation_analysis(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from app.services.exploratory_corr import correlation_analysis as calculate

    return calculate(*args, **kwargs)


def static_pca(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from app.services.exploratory_pca import static_pca as calculate

    return calculate(*args, **kwargs)


def transform_pca(*args: Any, **kwargs: Any) -> pd.DataFrame:
    from app.services.exploratory_pca import transform_pca as calculate

    return calculate(*args, **kwargs)
