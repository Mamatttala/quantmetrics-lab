"""Pure dependence diagnostics. No file, catalog, or event-loop operations."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from app.core.errors import ValidationFailure
from app.services.exploratory_stats import LIMITS, finite_frame, preprocess, stateless_transform

CCF_WARNING = (
    "Positive-lag CCF uses future target values and must not be used as causal trading "
    "features. Create causal lagged features in C5."
)


def number(value: Any) -> float | None:
    return float(value) if np.isfinite(value) else None


def fisher_interval(
    r: float, n: int, confidence: float = 0.95
) -> tuple[float | None, float | None]:
    if n <= 3:
        return None, None
    if abs(r) >= 1 - 1e-15:
        return r, r
    delta = stats.norm.ppf((1 + confidence) / 2) / np.sqrt(n - 3)
    low, high = np.tanh(np.arctanh(r) + np.array([-delta, delta]))
    return float(low), float(high)


def adjust_pvalues(values: list[float | None], method: str) -> list[float | None]:
    indices = [i for i, p in enumerate(values) if p is not None]
    p = np.array([values[i] for i in indices], dtype=float)
    if method == "bonferroni":
        p = np.minimum(1, p * len(p))
    elif method in {"benjamini_hochberg", "fdr_bh"} and len(p):
        order = np.argsort(p, kind="stable")
        adjusted = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
        p[order] = np.minimum(1, adjusted)
    result: list[float | None] = [None] * len(values)
    for i, value in zip(indices, p, strict=True):
        result[i] = float(value)
    return result


def _coefficient(values: np.ndarray, method: str) -> tuple[float, float, int]:
    rank = 0
    if "spearman" in method:
        values = stats.rankdata(values, axis=0)
    if method.startswith("partial"):
        design = np.column_stack([np.ones(len(values)), values[:, 2:]])
        rank = int(np.linalg.matrix_rank(design)) - 1
        values = values[:, :2] - design @ np.linalg.lstsq(design, values[:, :2], rcond=None)[0]
    if np.any(np.std(values[:, :2], axis=0) <= np.finfo(float).eps):
        return float("nan"), float("nan"), rank
    if method == "kendall":
        result = stats.kendalltau(values[:, 0], values[:, 1])
        return float(result.statistic), float(result.pvalue), rank
    r = float(np.clip(np.corrcoef(values[:, :2].T)[0, 1], -1, 1))
    df = len(values) - rank - 2
    p = (
        0.0
        if abs(r) >= 1 - 1e-15
        else float(2 * stats.t.sf(abs(r) * np.sqrt(max(0, df) / max(1e-30, 1 - r * r)), df))
    )
    return r, p if df > 0 else float("nan"), rank


def pair_statistics(
    frame: pd.DataFrame,
    x: str,
    y: str,
    method: str = "pearson",
    *,
    controls: list[str] | None = None,
    min_obs: int = 30,
    compute_p_values: bool = True,
    compute_confidence_intervals: bool = True,
    confidence_level: float = 0.95,
    ci_method: str | None = None,
    bootstrap_samples: int = 200,
    random_state: int = 0,
) -> dict[str, Any]:
    controls = controls or []
    values = finite_frame(frame, [x, y, *controls]).dropna().to_numpy()
    n = len(values)
    result: dict[str, Any] = {
        "feature_a": x,
        "feature_b": y,
        "feature_x": x,
        "feature_y": y,
        "correlation": None,
        "abs_correlation": None,
        "p_value": None,
        "adjusted_p_value": None,
        "ci_low": None,
        "ci_high": None,
        "ci_method": "none",
        "n_used": n,
        "controls_rank": 0,
        "significant": False,
        "redundancy_flag": False,
    }
    if n < max(3, min_obs):
        return result
    r, p, rank = _coefficient(values, method)
    result["controls_rank"] = rank
    if n - rank <= 2 or not np.isfinite(r):
        return result
    result.update(
        correlation=r, abs_correlation=abs(r), p_value=number(p) if compute_p_values else None
    )
    if compute_confidence_intervals and ci_method != "none":
        if method in {"pearson", "partial_pearson"} and ci_method != "bootstrap":
            result["ci_low"], result["ci_high"] = fisher_interval(r, n - rank, confidence_level)
            result["ci_method"] = "fisher_z" if result["ci_low"] is not None else "none"
        elif ci_method == "bootstrap":
            rng = np.random.default_rng(random_state)
            draws = [
                _coefficient(values[rng.integers(n, size=n)], method)[0]
                for _ in range(bootstrap_samples)
            ]
            finite = np.array(draws)[np.isfinite(draws)]
            if len(finite) >= max(20, int(0.8 * bootstrap_samples)):
                low, high = np.quantile(
                    finite, [(1 - confidence_level) / 2, (1 + confidence_level) / 2]
                )
                result.update(ci_low=float(low), ci_high=float(high), ci_method="paired_bootstrap")
    return result


def matrix_report(frame: pd.DataFrame, features: list[str], **options: Any) -> dict[str, Any]:
    testing = options.pop("multiple_testing", "none")
    alpha = options.pop("alpha", 0.05)
    threshold = options.pop("redundancy_threshold", 0.9)
    pairs = [pair_statistics(frame, a, b, **options) for a, b in combinations(features, 2)]
    adjusted = adjust_pvalues([pair["p_value"] for pair in pairs], testing)
    matrix = np.full((len(features), len(features)), np.nan)
    for i, feature in enumerate(features):
        column = finite_frame(frame, [feature])[feature].dropna()
        if len(column) >= options.get("min_obs", 30) and column.std() > 0:
            matrix[i, i] = 1
    for pair, p in zip(pairs, adjusted, strict=True):
        pair["adjusted_p_value"] = p
        pair["significant"] = p is not None and p < alpha
        pair["redundancy_flag"] = (pair["abs_correlation"] or 0) >= threshold
        i, j = features.index(pair["feature_a"]), features.index(pair["feature_b"])
        matrix[i, j] = matrix[j, i] = (
            pair["correlation"] if pair["correlation"] is not None else np.nan
        )
    warnings = []
    if np.isfinite(matrix).all() and np.linalg.eigvalsh(matrix).min() < -1e-10:
        warnings.append(
            "Correlation matrix is not positive semi-definite; pairwise missingness can cause this."
        )
    rows = [[number(value) for value in row] for row in matrix]
    return {
        "features": features,
        "matrix": rows,
        "correlation_matrix": rows,
        "pairs": pairs,
        "pairwise": pairs,
        "warnings": warnings,
    }


def vif_table(
    frame: pd.DataFrame,
    features: list[str],
    *,
    include_intercept: bool = True,
    vif_threshold: float = 5.0,
    tolerance: float = 1e-12,
) -> list[dict[str, Any]]:
    values = finite_frame(frame, features).dropna().to_numpy()
    result = []
    for i, feature in enumerate(features):
        y = values[:, i]
        x = np.delete(values, i, axis=1)
        state, vif = "insufficient", None
        if len(y) > len(features):
            if np.ptp(y) <= tolerance:
                state = "constant"
            else:
                if include_intercept:
                    x = np.column_stack([np.ones(len(y)), x])
                residual = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
                total = np.sum((y - y.mean()) ** 2) if include_intercept else np.sum(y**2)
                ratio = float(np.sum(residual**2) / total)
                if ratio <= tolerance:
                    state, vif = "infinite", 1e30
                else:
                    vif = 1 / ratio
                    state = "high" if vif >= vif_threshold else "ok"
        result.append(
            {
                "feature": feature,
                "vif": vif,
                "n_used": len(y),
                "status": state,
                "tolerance": 0.0 if state == "infinite" else (1 / vif if vif else None),
                "condition_flag": state,
                "suggested_action": (
                    "Consider PCA or removing a redundant feature."
                    if state != "ok"
                    else "Retain; no multicollinearity flag."
                ),
            }
        )
    return result


def redundancy_groups(
    features: list[str],
    pairs: list[dict[str, Any]],
    threshold: float,
    vif: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    graph: dict[str, set[str]] = {feature: set() for feature in features}
    lookup = {}
    for pair in pairs:
        a, b, value = pair["feature_a"], pair["feature_b"], pair["abs_correlation"]
        lookup[frozenset((a, b))] = value
        if value is not None and value >= threshold:
            graph[a].add(b)
            graph[b].add(a)
    remaining = set(features)
    groups: list[dict[str, Any]] = []
    scores = {row["feature"]: row["vif"] for row in vif}
    while remaining:
        pending, members = [min(remaining)], set()
        while pending:
            item = pending.pop()
            if item in members:
                continue
            members.add(item)
            pending.extend(sorted(graph[item] - members))
        remaining -= members
        if len(members) < 2:
            continue
        ordered = sorted(members)
        values = [
            lookup[frozenset(pair)]
            for pair in combinations(ordered, 2)
            if lookup.get(frozenset(pair)) is not None
        ]
        representative = min(ordered, key=lambda name: (scores.get(name) or float("inf"), name))
        groups.append(
            {
                "group_id": len(groups) + 1,
                "members": ordered,
                "representative": representative,
                "min_pairwise_abs_correlation": min(values),
                "max_pairwise_abs_correlation": max(values),
                "average_pairwise_abs_correlation": float(np.mean(values)),
                "suggested_action": "Review representatives or PCA; no features removed.",
            }
        )
    return groups


def ccf_analysis(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    lag_min: int,
    lag_max: int,
    min_obs: int,
    method: str = "pearson",
    controls: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, summaries = [], []
    for feature in features:
        if feature == target:
            continue
        series = []
        for lag in range(lag_min, lag_max + 1):
            aligned = frame.copy()
            aligned[target] = frame[target].shift(-lag)
            pair = pair_statistics(
                aligned,
                feature,
                target,
                method,
                controls=controls,
                min_obs=min_obs,
                compute_confidence_intervals=False,
            )
            pair["lag"] = lag
            series.append(pair)
        valid = [row for row in series if row["correlation"] is not None]

        def best(sign: int, valid: list[dict[str, Any]] = valid) -> dict[str, Any] | None:
            candidates = [row for row in valid if row["lag"] * sign > 0]
            return max(candidates, key=lambda row: row["abs_correlation"]) if candidates else None

        positive, negative = best(1), best(-1)
        summaries.append(
            {
                "feature": feature,
                "best_positive_lag": positive["lag"] if positive else None,
                "best_negative_lag": negative["lag"] if negative else None,
                "max_absolute_predictive_correlation": positive["abs_correlation"]
                if positive
                else None,
            }
        )
        rows.extend(series)
    return rows, summaries


def rolling_analysis(
    frame: pd.DataFrame,
    features: list[str],
    *,
    time_column: str,
    window: int,
    min_periods: int = 30,
    step: int = 1,
    selection: str = "all_pairs",
    top_pairs: int = 10,
    selected_pairs: list[tuple[str, str]] | None = None,
    method: str = "pearson",
    controls: list[str] | None = None,
    pre_transform: str = "none",
    missing_policy: str = "pairwise_complete",
    stateless_done: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """At observation t use [t-window, t); even t itself cannot affect its output."""
    pairs = list(combinations(features, 2))
    if selection == "selected_pairs":
        pairs = selected_pairs or []
    elif selection == "top_n":
        full = matrix_report(frame, features, min_obs=min_periods)["pairs"]
        full.sort(
            key=lambda row: (-(row["abs_correlation"] or 0), row["feature_a"], row["feature_b"])
        )
        pairs = [(row["feature_a"], row["feature_b"]) for row in full[:top_pairs]]
    if len(pairs) * ((len(frame) + step - 1) // step) > LIMITS["max_rolling_points"]:
        raise ValidationFailure(
            "Rolling output exceeds max_rolling_points; increase step or select fewer pairs."
        )
    all_features = list(dict.fromkeys([*features, *(controls or [])]))
    transformed = (
        frame.copy() if stateless_done else stateless_transform(frame, all_features, pre_transform)
    )
    series, summaries = [], []
    simple = (
        method == "pearson"
        and pre_transform
        in {
            "none",
            "first_difference",
            "percent_change",
            "log_difference",
            "absolute_return",
            "squared_return",
        }
        and missing_policy in {"pairwise_complete", "leave_nan", "drop_any_missing"}
    )
    if missing_policy == "drop_any_missing":
        transformed.loc[transformed[all_features].isna().any(axis=1), all_features] = np.nan
    for a, b in pairs:
        if simple:
            x, y = transformed[a].shift(1), transformed[b].shift(1)
            correlations = x.rolling(window, min_periods=min_periods, center=False).corr(y)
            counts = (x.notna() & y.notna()).rolling(window, min_periods=1, center=False).sum()
        points: list[dict[str, Any]] = []
        for end in range(0, len(frame), step):
            if simple:
                corr, n = number(correlations.iloc[end]), int(counts.iloc[end])
            else:
                piece = transformed.iloc[max(0, end - window) : end]
                processed, _, _, _, _, _ = preprocess(
                    piece,
                    piece.index,
                    all_features,
                    pre_transform=pre_transform,
                    missing_policy=missing_policy,
                    constant_policy="warn_only",
                    stateless_done=True,
                )
                pair = pair_statistics(
                    processed,
                    a,
                    b,
                    method,
                    controls=controls,
                    min_obs=min_periods,
                    compute_p_values=False,
                    compute_confidence_intervals=False,
                )
                corr, n = pair["correlation"], pair["n_used"]
            points.append(
                {
                    "feature": f"{a} / {b}",
                    "feature_a": a,
                    "feature_b": b,
                    "time": str(frame[time_column].iloc[end]),
                    "correlation": corr,
                    "n_used": n,
                }
            )
        finite = [point["correlation"] for point in points if point["correlation"] is not None]
        std = float(np.std(finite, ddof=1)) if len(finite) > 1 else None
        summaries.append(
            {
                "feature_a": a,
                "feature_b": b,
                "mean_corr": float(np.mean(finite)) if finite else None,
                "std_corr": std,
                "min_corr": min(finite) if finite else None,
                "max_corr": max(finite) if finite else None,
                "latest_corr": points[-1]["correlation"] if points else None,
                "stability_score": 1 / (1 + std) if std is not None else None,
                "n_points": len(finite),
            }
        )
        if selection != "all_pairs":
            series.extend(points)
    # Time-major ordering is stable when later observations are removed.
    series.sort(key=lambda row: (row["time"], row["feature_a"], row["feature_b"]))
    return series, summaries


def correlation_analysis(
    frame: pd.DataFrame,
    features: list[str],
    *,
    method: str = "pearson",
    controls: list[str] | None = None,
    target: str | None = None,
    redundancy_threshold: float = 0.9,
    max_lag: int = 0,
    rolling_window: int | None = None,
    group_column: str | None = None,
    time_column: str | None = None,
    **options: Any,
) -> dict[str, Any]:
    """Static matrix and optional diagnostics; caller owns fit scope and preprocessing."""
    features = list(dict.fromkeys([*features, *([target] if target else [])]))
    rolling_only = bool(options.pop("rolling_only", False))
    pair_options = {
        key: value
        for key, value in options.items()
        if key
        in {
            "min_obs",
            "compute_p_values",
            "compute_confidence_intervals",
            "confidence_level",
            "ci_method",
            "bootstrap_samples",
            "random_state",
            "multiple_testing",
            "alpha",
        }
    }
    pair_options.update(method=method, controls=controls, redundancy_threshold=redundancy_threshold)
    if rolling_only:
        report: dict[str, Any] = {
            "features": features,
            "matrix": [],
            "correlation_matrix": [],
            "pairs": [],
            "pairwise": [],
            "warnings": [],
            "vif": [],
            "redundancy_groups": [],
        }
    else:
        report = matrix_report(frame, features, **pair_options)
        report["vif"] = (
            vif_table(
                frame,
                features,
                include_intercept=options.get("include_intercept", True),
                vif_threshold=options.get("vif_threshold", 5.0),
            )
            if options.get("calculate_vif", True)
            else []
        )
        groups = redundancy_groups(
            features, report["pairs"], redundancy_threshold, report["vif"]
        )
        report["redundancy_groups"] = (
            groups if options.get("enable_redundancy_groups", True) else []
        )
    report.update(
        method=method,
        controls=controls or [],
        redundancy_threshold=redundancy_threshold,
        conditional=[],
        ccf=[],
        ccf_summary=[],
        rolling=[],
        rolling_series=[],
        rolling_summary=[],
    )
    if controls and not rolling_only:
        complete = finite_frame(frame, [*features, *controls]).dropna()
        if len(complete):
            design = np.column_stack([np.ones(len(complete)), complete[controls].to_numpy()])
            if np.linalg.matrix_rank(design) < design.shape[1]:
                report["warnings"].append(
                    "Control features are rank-deficient; partial correlations use the "
                    "estimable control subspace."
                )
            strong = []
            for control in controls:
                for feature in features:
                    value = complete[control].corr(complete[feature])
                    if pd.notna(value) and abs(value) >= 0.95:
                        strong.append(f"{control}/{feature}={value:.3f}")
            if strong:
                report["warnings"].append(
                    "Controls are highly collinear with analyzed features: " + ", ".join(strong)
                )
    if group_column and not rolling_only:
        grouping = frame[group_column]
        if options.get("group_bins"):
            grouping = pd.cut(grouping, options["group_bins"], include_lowest=True)
        if grouping.nunique(dropna=False) > LIMITS["max_groups"]:
            raise ValidationFailure(
                "Too many groups; provide numeric bins or a coarser grouping column."
            )
        for group, part in frame.groupby(grouping, dropna=False, observed=True, sort=True):
            entry = matrix_report(part, features, **pair_options)
            entry.update(
                group=None if pd.isna(group) else str(group), rows=len(part), group_size=len(part)
            )
            if len(part) < options.get("group_min_obs", 30):
                entry["warnings"].append("Small group: fewer than group_min_obs observations.")
            report["conditional"].append(entry)
        report["warnings"].append(
            "Grouping by future-derived columns is diagnostic only; "
            "C8 cannot certify group provenance."
        )
    report["conditional_correlations"] = report["conditional"]
    if (max_lag or options.get("enable_ccf")) and not rolling_only:
        if not target:
            raise ValidationFailure("CCF requires target_feature.")
        report["ccf"], report["ccf_summary"] = ccf_analysis(
            frame,
            features,
            target,
            options.get("lag_min", -max_lag),
            options.get("lag_max", max_lag),
            options.get("ccf_min_obs", 30),
            method,
            controls,
        )
        report["warnings"].append(CCF_WARNING)
    if rolling_window:
        if not time_column:
            raise ValidationFailure("Rolling correlation requires an observation/time column.")
        report["rolling"], report["rolling_summary"] = rolling_analysis(
            options.get("rolling_frame", frame),
            options.get("rolling_features", features),
            time_column=time_column,
            window=rolling_window,
            min_periods=options.get("rolling_min_periods") or min(30, rolling_window),
            step=options.get("rolling_step", 1),
            selection=options.get("rolling_pair_selection", "all_pairs"),
            top_pairs=options.get("rolling_top_pairs", 10),
            selected_pairs=options.get("rolling_pairs"),
            method=method,
            controls=controls,
            pre_transform=options.get("pre_transform", "none"),
            missing_policy=options.get("missing_policy", "pairwise_complete"),
            stateless_done=options.get("rolling_stateless_done", False),
        )
        report["rolling_series"] = report["rolling"]
    report["ci_method"] = sorted({pair["ci_method"] for pair in report["pairs"]})
    report["multiple_testing_method"] = options.get("multiple_testing", "none")
    report["notes"] = (
        [
            "Rolling scope returns causal trailing-window summaries only; static diagnostics "
            "use a separate full/custom/train request.",
            "All-pairs mode omits row-level series; selected-pairs mode returns them.",
            "stability_score = 1 / (1 + sample std of finite rolling correlations).",
        ]
        if rolling_only
        else [
            "Inference assumes independent observations; no autocorrelation correction.",
            "Rank-method CIs are null unless paired bootstrap is enabled.",
            "VIF uses complete selected-feature rows under the requested missing policy.",
        ]
    )
    report["metadata"] = {
        "vif_threshold": options.get("vif_threshold", 5.0),
        "include_intercept": options.get("include_intercept", True),
        "controls": controls or [],
        "lag_convention": "corr(x_t, y_{t+L}); bars only",
        "rolling_excludes_current_row": True,
    }
    return report
