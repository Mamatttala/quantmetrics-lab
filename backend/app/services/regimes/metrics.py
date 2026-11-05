"""Model-compatible diagnostics, state economics, transitions and label alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon
from scipy.stats import kurtosis, skew
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    log_loss,
    silhouette_score,
)

from app.services.regimes.estimators import target_values

OCCUPANCY_NOTE = (
    "Occupancy is the fraction of assigned observations (0 <= state_id < state count). "
    "Unassigned/warm-up observations are excluded; an empty denominator yields zero occupancy."
)


def separation(x, states):
    valid = states >= 0
    x, states = x[valid], states[valid]
    if len(np.unique(states)) < 2 or len(np.unique(states)) >= len(states):
        return {
            "unavailable": "State separation needs at least two nonempty states "
            "and within-state replication."
        }
    return {
        "silhouette": float(
            silhouette_score(x, states, sample_size=min(3000, len(x)), random_state=0)
        ),
        "davies_bouldin": float(davies_bouldin_score(x, states)),
        "calinski_harabasz": float(calinski_harabasz_score(x, states)),
    }


def runs(states):
    if len(states) == 0:
        return []
    changes = np.r_[0, np.flatnonzero(states[1:] != states[:-1]) + 1, len(states)]
    return [
        {
            "state_id": int(states[start]),
            "start": int(start),
            "end": int(end - 1),
            "bars": int(end - start),
            "censored": bool(start == 0 or end == len(states)),
        }
        for start, end in zip(changes[:-1], changes[1:], strict=True)
    ]


def transitions(states, count, timestamps=None, fitted=None):
    matrix = np.zeros((count, count), dtype=int)
    events = []
    for i in range(1, len(states)):
        a, b = int(states[i - 1]), int(states[i])
        if 0 <= a < count and 0 <= b < count:
            matrix[a, b] += 1
            if a != b:
                events.append(
                    {
                        "index": i,
                        "from_state": a,
                        "to_state": b,
                        "timestamp": str(timestamps[i]) if timestamps is not None else None,
                    }
                )
    probability = np.divide(
        matrix,
        matrix.sum(axis=1, keepdims=True),
        out=np.zeros_like(matrix, dtype=float),
        where=matrix.sum(axis=1, keepdims=True) > 0,
    )
    durations = runs(states)
    expected_source = fitted if fitted is not None else probability
    expected = [
        float(1 / (1 - expected_source[k, k]))
        if expected_source[k, k] < 1 and (fitted is not None or matrix[k].sum())
        else None
        for k in range(count)
    ]
    return {
        "counts": matrix.tolist(),
        "probabilities": probability.tolist(),
        "fitted_probabilities": fitted.tolist() if fitted is not None else None,
        "expected_duration": expected,
        "expected_duration_source": "fitted HMM transition"
        if fitted is not None
        else "empirical transition",
        "empirical_duration": [
            float(np.mean([r["bars"] for r in durations if r["state_id"] == k]))
            if any(r["state_id"] == k for r in durations)
            else None
            for k in range(count)
        ],
        "switching_frequency": len(events) / max(1, len(states) - 1),
        "events": events,
        "durations": durations,
        "duration_note": "First and last runs are censored; "
        "empirical means include these observed portions.",
    }


def economic_columns(prepared, inputs):
    market = prepared.market
    result = {"return_type": None, "return_note": "Return observations are unavailable."}
    if inputs.return_column:
        if inputs.return_type is None:
            result["return_note"] = (
                "Return statistics unavailable: saved return_column has no explicit return_type. "
                "Create a new experiment declaring simple or log returns."
            )
        elif inputs.return_column in market:
            result["return"] = pd.to_numeric(
                market[inputs.return_column], errors="coerce"
            ).to_numpy()
            result["return_type"] = inputs.return_type
    if inputs.price_column and inputs.price_column in market:
        result["price"] = pd.to_numeric(market[inputs.price_column], errors="coerce").to_numpy()
        if not inputs.return_column:
            result["return"] = pd.Series(result["price"]).pct_change(fill_method=None).to_numpy()
            result["return_type"] = "simple"
    if result["return_type"] is not None:
        result["return_note"] = (
            f"Return statistics use {result['return_type']} returns; "
            "drawdown compounds within contiguous state episodes."
        )
    for key, column in (
        ("adx", inputs.adx_column),
        ("rsi", inputs.rsi_column),
        ("volatility", inputs.volatility_column),
    ):
        if column and column in market:
            result[key] = pd.to_numeric(market[column], errors="coerce").to_numpy()
    return result


def equity_curve(returns, return_type):
    """Unit initial capital, with an explicit return convention (F5)."""
    if return_type == "log":
        return np.exp(np.r_[0.0, np.cumsum(returns)])
    if return_type == "simple":
        return np.r_[1.0, np.cumprod(1 + returns)]
    raise ValueError("An explicit simple or log return_type is required.")


def profiles(prepared, inputs, states, indices, count):
    economics = economic_columns(prepared, inputs)
    durations = runs(states)
    distribution = occupancy(states, count)
    denominator = int(((states >= 0) & (states < count)).sum())
    output = []
    for state in range(count):
        mask = states == state
        feature_values = prepared.x[indices][mask]
        stats = {
            "state_id": state,
            "count": int(mask.sum()),
            "occupancy": float(distribution[state]),
            "occupancy_denominator": denominator,
            "feature_means": dict(
                zip(inputs.features, feature_values.mean(axis=0).tolist(), strict=True)
            )
            if len(feature_values)
            else {},
            "mean_return": None,
            "median_return": None,
            "return_volatility": None,
            "skewness": None,
            "kurtosis": None,
            "downside_deviation": None,
            "maximum_drawdown": None,
            "return_type": economics["return_type"],
            "return_note": economics["return_note"],
            "adx_mean": None,
            "rsi_mean": None,
            "mean_duration": float(
                np.mean([r["bars"] for r in durations if r["state_id"] == state])
            )
            if any(r["state_id"] == state for r in durations)
            else None,
        }
        if "return" in economics:
            values = economics["return"][indices][mask]
            values = values[np.isfinite(values)]
            if len(values):
                stats.update(
                    mean_return=float(values.mean()),
                    median_return=float(np.median(values)),
                    return_volatility=float(values.std(ddof=1)) if len(values) > 1 else None,
                    skewness=float(skew(values, bias=False))
                    if len(values) > 2 and values.std() > 0
                    else None,
                    kurtosis=float(kurtosis(values, bias=False))
                    if len(values) > 3 and values.std() > 0
                    else None,
                    downside_deviation=float(np.sqrt(np.mean(np.minimum(values, 0) ** 2))),
                )
                # Drawdown is measured within contiguous state episodes, not by
                # joining disjoint visits and pretending they were a continuous trade.
                drawdowns = []
                for run in durations:
                    if run["state_id"] != state:
                        continue
                    r = economics["return"][indices][run["start"] : run["end"] + 1]
                    if np.isfinite(r).all() and (
                        economics["return_type"] == "log" or (r >= -1).all()
                    ):
                        with np.errstate(over="ignore", invalid="ignore"):
                            equity = equity_curve(r, economics["return_type"])
                        if np.isfinite(equity).all():
                            drawdowns.append(
                                float((equity / np.maximum.accumulate(equity) - 1).min())
                            )
                stats["maximum_drawdown"] = min(drawdowns) if drawdowns else None
        for key in ("adx", "rsi"):
            if key in economics:
                values = economics[key][indices][mask]
                values = values[np.isfinite(values)]
                stats[f"{key}_mean"] = float(values.mean()) if len(values) else None
        output.append(stats)
    return output


def statistical_labels(state_profiles, template="trend"):
    vols = [p["return_volatility"] for p in state_profiles if p["return_volatility"] is not None]
    median_vol = float(np.median(vols)) if vols else None
    labels = {}
    for row in state_profiles:
        state = str(row["state_id"])
        mean, vol = row["mean_return"], row["return_volatility"]
        if mean is None or vol is None:
            labels[state] = f"State {state}"
            continue
        trend = "Bullish" if mean > 0.1 * vol else "Bearish" if mean < -0.1 * vol else "Sideways"
        volatility = "High Volatility" if vol > median_vol else "Low Volatility"
        labels[state] = (
            volatility
            if template == "volatility"
            else f"{trend} · {volatility}"
            if template == "trend_volatility"
            else "Stress"
            if template == "stress" and trend == "Bearish" and vol > median_vol
            else trend
        )
    return labels


def occupancy(states, count):
    counts = np.array([(states == k).sum() for k in range(count)])
    return counts / max(1, counts.sum())


def occupancy_drift(reference, observed, count):
    p, q = occupancy(reference, count), occupancy(observed, count)
    if not p.sum() or not q.sum():
        return None
    return float(jensenshannon(p, q, base=2) ** 2)


def align(reference, other):
    a, b = np.unique(reference), np.unique(other)
    overlap = np.array([[(reference == x)[other == y].sum() for y in b] for x in a])
    rows, cols = linear_sum_assignment(-overlap)
    mapping = {int(b[j]): int(a[i]) for i, j in zip(rows, cols, strict=True)}
    aligned = np.array([mapping.get(int(value), -1) for value in other])
    return {
        "adjusted_rand": float(adjusted_rand_score(reference, other)),
        "aligned_agreement": float((reference == aligned).mean()),
        "mapping": mapping,
    }


def predictive(model, prepared, config, indices, output):
    if config.target is None or model.family not in {"supervised", "hybrid"}:
        return {}
    target = config.target
    horizon = target.horizon if target.objective == "future" else 0
    valid = indices + horizon + target.availability_lag <= indices[-1] if len(indices) else []
    origins = indices[valid]
    label_indices = origins + horizon
    realized = np.ones(len(origins), dtype=bool)
    if model.teacher is not None:
        raw, _ = model.teacher.predict(prepared.x)
        y = raw[label_indices]
    else:
        raw = target_values(prepared.frame.iloc[label_indices], target, allow_unavailable=True)
        realized = pd.notna(raw)
        vocabulary = {v: i for i, v in enumerate(model.vocabulary)}
        if target.source == "column" and target.column in prepared.frame and (
            pd.api.types.is_numeric_dtype(prepared.frame[target.column])
        ):
            # A missing value can promote integer labels to floats in pandas.
            # Preserve old numeric vocabularies while leaving string classes exact.
            vocabulary = {v.removesuffix(".0"): i for v, i in vocabulary.items()}
            raw = np.array([v.removesuffix(".0") if v is not None else None for v in raw])
        y = np.array([vocabulary.get(value, -1) for value in raw])
    predicted = output["states"][origins]
    probability = output["probabilities"][origins]
    usable = realized & (predicted >= 0) & np.isfinite(probability).all(axis=1)
    y, predicted = y[usable], predicted[usable]
    if not len(y):
        return {"unavailable": "No complete prediction/target pairs in this span.",
                "observations": 0, "log_loss_rows": 0,
                "missing_target_count": int((~realized).sum())}
    probability = probability[usable]
    known = y >= 0
    loss = (
        float(log_loss(y[known], probability[known], labels=list(range(model.count))))
        if known.any()
        else None
    )
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "log_loss": loss,
        "unknown_target_count": int((~known).sum()),
        "log_loss_rows": int(known.sum()),
        "observations": len(y),
        "missing_target_count": int((~realized).sum()),
        "meaning": "teacher agreement"
        if model.teacher is not None
        else "declared target performance",
        "objective": target.objective,
        "horizon": horizon,
    }
