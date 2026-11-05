"""Deterministic static PCA and replay of its fitted parameters; no I/O."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.core.errors import ValidationFailure
from app.services.exploratory_stats import LIMITS, finite_frame


def static_pca(
    frame: pd.DataFrame,
    features: list[str],
    *,
    standardizer: str = "standard",
    selection: str = "manual",
    n_components: int | None = 2,
    variance_threshold: float = 0.95,
    eigenvalue_threshold: float = 1.0,
    parallel_permutations: int = 100,
    random_state: int = 0,
    whiten: bool = False,
    min_obs: int = 2,
    component_prefix: str = "pc_",
) -> dict[str, Any]:
    complete = finite_frame(frame, features).dropna()
    values = complete.to_numpy()
    n, p = values.shape
    if n < 2 or p < 1:
        raise ValidationFailure(
            "PCA needs at least two complete fit rows and a nonconstant feature."
        )
    limit = min(n - 1, p, LIMITS["max_pca_components"])
    low, high = values.min(axis=0), values.max(axis=0)
    center, scale = np.zeros(p), np.ones(p)
    if standardizer == "standard":
        center, scale = values.mean(axis=0), values.std(axis=0, ddof=1)
    elif standardizer == "robust":
        center = np.median(values, axis=0)
        scale = np.quantile(values, 0.75, axis=0) - np.quantile(values, 0.25, axis=0)
    elif standardizer == "minmax":
        center, scale = low, high - low
    scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
    scaled = (values - center) / scale
    pca_center = scaled.mean(axis=0)
    centered = scaled - pca_center
    try:
        _, singular, vectors = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise ValidationFailure(
            "PCA SVD did not converge; inspect feature magnitudes and constants."
        ) from exc
    order = np.argsort(-(singular**2), kind="stable")
    singular, vectors = singular[order], vectors[order]
    eigenvalues = singular**2 / (n - 1)
    total = eigenvalues.sum()
    if total <= np.finfo(float).eps:
        raise ValidationFailure("PCA fit rows have no usable variance after preprocessing.")
    ratios = eigenvalues / total
    cumulative = np.cumsum(ratios)
    cumulative[-1] = 1.0
    signs = []
    for vector in vectors:
        sign = -1 if vector[np.argmax(np.abs(vector))] < 0 else 1
        vector *= sign
        signs.append(sign)
    warnings: list[str] = []
    if n < min_obs:
        warnings.append(
            "PCA has fewer than min_obs complete fit rows; "
            "small-sample diagnostics may be unstable."
        )
    parallel_cutoffs: list[float] = []
    if selection == "manual":
        k = n_components if n_components is not None else min(2, limit)
        if not 1 <= k <= limit:
            raise ValidationFailure(
                f"n_components must be between 1 and {limit} for these fit rows/features."
            )
    elif selection == "variance_threshold":
        desired = int(np.searchsorted(cumulative, variance_threshold) + 1)
        k = min(desired, limit)
        if desired > limit:
            warnings.append("Component limit prevents reaching the requested variance threshold.")
    elif selection == "eigenvalue_gt_1":
        if standardizer != "standard":
            warnings.append("Kaiser criterion is intended for standardized features.")
        k = min(limit, max(1, int((eigenvalues > eigenvalue_threshold).sum())))
        if not (eigenvalues > eigenvalue_threshold).any():
            warnings.append(
                "No eigenvalues exceed the threshold; retain one component for inspection."
            )
    elif selection == "scree":
        x = np.linspace(0.0, 1.0, len(eigenvalues))
        spread = eigenvalues[0] - eigenvalues[-1]
        normalized = (eigenvalues - eigenvalues[-1]) / spread if spread > 0 else np.ones(len(x))
        k = min(limit, max(1, int(np.argmax(1 - x - normalized)) + 1))
    elif selection == "parallel":
        if not 1 <= parallel_permutations <= LIMITS["max_parallel_permutations"]:
            raise ValidationFailure("parallel_permutations exceeds the computational limit.")
        rng = np.random.default_rng(random_state)
        draws = []
        for _ in range(parallel_permutations):
            permuted = np.column_stack([rng.permutation(centered[:, j]) for j in range(p)])
            draws.append(np.linalg.svd(permuted, compute_uv=False) ** 2 / (n - 1))
        cutoff = np.quantile(draws, 0.95, axis=0)
        parallel_cutoffs = cutoff.tolist()
        # Retain a leading contiguous set, stopping at the first nonsignificant component.
        exceeds = eigenvalues > cutoff
        k = min(
            limit, max(1, next((i for i, keep in enumerate(exceeds) if not keep), len(exceeds)))
        )
    else:
        raise ValidationFailure("Unknown PCA component selection.")
    retained = vectors[:k].T
    if whiten and np.any(eigenvalues[:k] <= max(1e-15, eigenvalues[0] * 1e-12)):
        raise ValidationFailure(
            "Whitening requires positive-variance selected components; reduce n_components."
        )
    scores = centered @ retained
    if whiten:
        scores /= np.sqrt(eigenvalues[:k])
    reconstructed = (scores * np.sqrt(eigenvalues[:k]) if whiten else scores) @ retained.T
    error = float(np.max(np.abs(centered - reconstructed))) if k == min(p, n - 1) else None
    if error is not None and not np.allclose(centered, reconstructed, rtol=1e-9, atol=1e-9):
        raise ValidationFailure("Full-component PCA reconstruction failed numerical validation.")
    sample = np.unique(np.linspace(0, n - 1, min(n, 500), dtype=int))
    biplot = {
        "scores": scores[sample].tolist(),
        "loadings": retained.tolist(),
        "row_indices": complete.index[sample].tolist(),
    }
    names = [f"{component_prefix}{i + 1}" for i in range(k)]
    return {
        "features": features,
        "feature_order": features,
        "standardizer": standardizer,
        "n_components": k,
        "n_components_requested": n_components,
        "n_components_selected": k,
        "n_fit_rows": n,
        "fit_rows": n,
        "n_missing_rows": len(frame) - n,
        "component_selection": selection,
        "component_columns": names,
        "component_prefix": component_prefix,
        "center": center.tolist(),
        "scale": scale.tolist(),
        "min": low.tolist(),
        "max": high.tolist(),
        "pca_center": pca_center.tolist(),
        "whiten": whiten,
        "standardizer_parameters": {
            "center": center.tolist(),
            "scale": scale.tolist(),
            "pca_center": pca_center.tolist(),
        },
        "loadings": retained.tolist(),
        "eigenvalues": eigenvalues.tolist(),
        "explained_variance": eigenvalues.tolist(),
        "explained_variance_ratio": ratios.tolist(),
        "cumulative_explained_variance_ratio": cumulative.tolist(),
        "sign_flips": signs,
        "component_summary": [
            {
                "component": f"{component_prefix}{i + 1}",
                "eigenvalue": float(value),
                "explained_variance_ratio": float(ratios[i]),
                "cumulative_ratio": float(cumulative[i]),
                "selected": i < k,
            }
            for i, value in enumerate(eigenvalues)
        ],
        "biplot": biplot,
        "biplot_data": biplot,
        "reconstruction_error": error,
        "parallel_cutoffs": parallel_cutoffs,
        "variance_threshold": variance_threshold,
        "warnings": warnings,
        "notes": [
            "Static PCA; rolling PCA is deferred.",
            "Largest absolute loading is positive; first feature wins ties.",
        ],
        "metadata": {"random_state": random_state, "parallel_permutations": parallel_permutations},
    }


def transform_pca(frame: pd.DataFrame, parameters: dict[str, Any]) -> pd.DataFrame:
    values = finite_frame(frame, parameters["features"])
    valid = values.notna().all(axis=1)
    result = pd.DataFrame(np.nan, index=frame.index, columns=parameters["component_columns"])
    matrix = (values.loc[valid].to_numpy() - parameters["center"]) / parameters["scale"]
    scores = (matrix - parameters["pca_center"]) @ np.array(parameters["loadings"])
    if parameters.get("whiten"):
        scores /= np.sqrt(parameters["eigenvalues"][: parameters["n_components"]])
    result.loc[valid, :] = scores
    return result
