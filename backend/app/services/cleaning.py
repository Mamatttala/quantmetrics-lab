"""Cleaning: declared repairs to a dataset's values.

Three properties this module exists to hold.

1. **Nothing is repaired unless it was asked for by name.** Every policy defaults to
   `KEEP`, so a caller that forgets to send a field cannot quietly rewrite data. This
   is also why the missing-value policies are not a single "fill" switch.

2. **Every change is reported.** Each step returns what it did — rows removed, cells
   substituted, fences applied — and the report travels with the saved dataset. A
   cleaning step that cannot be described after the fact is indistinguishable from a
   corrupted dataset.

3. **The timestamp column is never filled or replaced.** Filling a gap in a price
   series is a documented modelling assumption the user is entitled to make. Inventing
   a bar *timestamp* is inventing an observation, and it would also change the bar
   spacing that the missing-bar detector measures — so a dataset could be "repaired"
   into failing a check it previously passed.

The whole-frame statistics that `winsorize` and `iqr` need are estimated over every
row, including rows later than the one being clamped. That is a look-ahead, and it is
recorded in DECISIONS.md rather than hidden: the fences are returned in the report and
stored in the dataset's provenance, so a backtest built on a cleaned dataset can be
audited for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from app.core.errors import ValidationFailure
from app.models.enums import DuplicatePolicy, MissingValuePolicy, OutlierPolicy

#: Aggregations that turn one bar into a coarser one. Only `resample` in `merge.py`
#: uses this table; it lives here because the OHLCV vocabulary is a property of the
#: dataset, not of the resampling step.
OHLC_AGGREGATIONS: dict[str, str] = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


@dataclass(frozen=True)
class CleaningPlan:
    """A complete description of what to repair. Defaults to repairing nothing."""

    duplicates: DuplicatePolicy = DuplicatePolicy.KEEP
    missing: MissingValuePolicy = MissingValuePolicy.KEEP
    outliers: OutlierPolicy = OutlierPolicy.KEEP
    #: Columns the outlier policy applies to. Empty means the policy cannot run, which
    #: is refused rather than applied to "everything": clamping every numeric column
    #: would silently mutilate price levels, which is the single most damaging thing a
    #: one-click outlier button can do.
    outlier_columns: tuple[str, ...] = ()
    lower_quantile: float = 0.01
    upper_quantile: float = 0.99
    iqr_multiplier: float = 1.5
    sort_by_time: bool = False


def clean_frame(
    frame: pd.DataFrame,
    plan: CleaningPlan,
    time_column: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply `plan` to `frame` and return the result with a report of what changed.

    Step order is fixed and reported, because it is observable: `sort` establishes the
    chronological order that `keep_first_by_time` and `keep_last_by_time` mean by
    "first" and "last" — on an unsorted frame those words describe the file's row order
    rather than time — then `duplicates` decides which rows exist, `missing` fills on
    that order, and `outliers` runs last so that a row dropped for being an outlier is
    not first interpolated into its neighbours.

    Every `_handle_*` returns `(frame, step_or_None)`, and returns `None` for a `KEEP`
    policy: a step that was not applied is not reported, so an empty `steps` list means
    "nothing was changed" rather than "nothing was considered". The frame is returned
    rather than mutated because a step that removes rows cannot report a removal it
    never performed — which is what happened while the frame travelled by mutation and
    one branch returned only its report.
    """
    report: dict[str, Any] = {
        "rows_before": int(len(frame)),
        "columns_before": [str(c) for c in frame.columns],
        "steps": [],
        "warnings": [],
    }
    result = frame.copy()

    if plan.sort_by_time:
        result, step = _sort_by_time(result, time_column)
    else:
        step = None

    result, duplicate_step = _handle_duplicates(result, plan, time_column)
    result, missing_step = _handle_missing(result, plan, time_column)
    result, outlier_step = _handle_outliers(result, plan)

    for candidate in (step, duplicate_step, missing_step, outlier_step):
        if candidate is not None:
            report["steps"].append(candidate)
            report["warnings"].extend(candidate.get("warnings", []))

    report["rows_after"] = int(len(result))
    report["rows_removed"] = report["rows_before"] - report["rows_after"]
    report["columns_after"] = [str(c) for c in result.columns]
    return result, report


# --------------------------------------------------------------- sort


def _sort_by_time(
    frame: pd.DataFrame, time_column: str | None
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    if time_column is None:
        raise ValidationFailure(
            "Cannot sort by time: the dataset has no recognisable timestamp column.",
            hint="Use a dataset with an open_time, create_time, or Date column.",
        )
    ordered = frame.sort_values(time_column, kind="stable")
    return ordered, {
        "step": "sort",
        "policy": "sort_by_time",
        # Stable, so rows sharing a timestamp keep the order they arrived in. An
        # unstable sort would reorder them arbitrarily and make the result depend on
        # the sort implementation rather than on the data.
        "detail": f"Sorted ascending by {time_column} (stable, so ties keep their order).",
        "warnings": [],
    }


# --------------------------------------------------------- duplicates


def _handle_duplicates(
    frame: pd.DataFrame,
    plan: CleaningPlan,
    time_column: str | None,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    policy = plan.duplicates
    if policy is DuplicatePolicy.KEEP:
        return frame, None

    if policy is DuplicatePolicy.DROP_EXACT:
        keep = ~frame.duplicated(keep="first")
        removed = int((~keep).sum())
        return frame.loc[keep], {
            "step": "duplicates",
            "policy": policy.value,
            "rows_removed": removed,
            "detail": (
                f"Removed {removed} row(s) identical to an earlier row in every column."
            ),
            "warnings": [],
        }

    if time_column is None:
        raise ValidationFailure(
            f"Cannot apply {policy.value}: the dataset has no recognisable timestamp column."
        )

    keep_rule: Literal["first", "last"] = (
        "first" if policy is DuplicatePolicy.KEEP_FIRST_BY_TIME else "last"
    )
    before = len(frame)
    deduped = frame.drop_duplicates(subset=[time_column], keep=keep_rule)
    removed = before - len(deduped)
    return deduped, {
        "step": "duplicates",
        "policy": policy.value,
        "rows_removed": removed,
        "detail": (
            f"Kept the {keep_rule} row for each of {removed} repeated "
            f"{time_column} value(s). Rows with the same timestamp but different "
            "values are not exact duplicates, so this is the repair that removes them."
        ),
        "warnings": [],
    }


# ------------------------------------------------------ missing values


def _numeric_targets(frame: pd.DataFrame, time_column: str | None) -> list[str]:
    """Columns a fill may touch: numeric, non-boolean, and never the timestamp."""
    targets: list[str] = []
    for name in frame.columns:
        if name == time_column:
            continue
        dtype = frame[name].dtype
        if pd.api.types.is_bool_dtype(dtype):
            # Interpolating a boolean, or zero-filling it, produces a value that is not
            # a boolean. `False` is the only defensible fill and choosing it silently
            # would be a decision about the data, not a repair of it.
            continue
        if pd.api.types.is_numeric_dtype(dtype):
            targets.append(str(name))
    return targets


def _handle_missing(
    frame: pd.DataFrame,
    plan: CleaningPlan,
    time_column: str | None,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    policy = plan.missing
    if policy is MissingValuePolicy.KEEP:
        return frame, None

    step: dict[str, Any] = {
        "step": "missing",
        "policy": policy.value,
        "filled": 0,
        "rows_removed": 0,
        "columns": [],
        "warnings": [],
    }

    if policy is MissingValuePolicy.DROP_ROWS:
        before = len(frame)
        cleaned = frame.dropna()
        step["rows_removed"] = before - len(cleaned)
        step["detail"] = (
            f"Removed {step['rows_removed']} row(s) containing at least one empty cell. "
            "A row is dropped for a hole anywhere in it, including columns unrelated to "
            "the one you were looking at."
        )
        return cleaned, step

    if policy is MissingValuePolicy.INTERPOLATE_TIME:
        return _interpolate(frame, plan, time_column, step)

    targets = _numeric_targets(frame, time_column)
    if not targets:
        step["warnings"].append(
            f"{policy.value} did nothing: the dataset has no numeric column other than "
            "the timestamp."
        )
        return frame, step

    before = int(frame[targets].isna().sum().sum())
    result = frame.copy()
    if policy is MissingValuePolicy.ZERO:
        result[targets] = result[targets].fillna(0.0)
        step["warnings"].append(
            "Empty cells were replaced with zero. On a price column a zero is a −100% "
            "return, not a neutral value — this is only safe for columns where zero is "
            "a real observation, such as an open-interest or volume field."
        )
    elif policy is MissingValuePolicy.FFILL:
        result[targets] = result[targets].ffill()
        step["warnings"].append(
            "Forward fill carries the last known value into the gap. A leading empty "
            "cell has nothing before it and is left empty."
        )
    else:
        result[targets] = result[targets].bfill()
        step["warnings"].append(
            "Backward fill carries the next value *backwards* into the gap, so a filled "
            "cell contains a value that had not been observed at that time. That is a "
            "look-ahead if the dataset feeds a backtest."
        )

    after = int(result[targets].isna().sum().sum())
    step["filled"] = before - after
    step["columns"] = targets
    step["detail"] = f"Filled {step['filled']} of {before} empty numeric cell(s)."
    return result, step


def _interpolate(
    frame: pd.DataFrame,
    plan: CleaningPlan,
    time_column: str | None,
    step: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if time_column is None:
        raise ValidationFailure(
            "Time-based interpolation needs a timestamp column to measure the gaps "
            "between known points, and this dataset has none.",
            hint="Use forward fill, backward fill, or row removal on a dataset "
            "without timestamps.",
        )
    targets = _numeric_targets(frame, time_column)
    if not targets:
        step["warnings"].append(
            "Time-based interpolation did nothing: no numeric column to interpolate."
        )
        return frame, step

    stamps = pd.to_datetime(frame[time_column], utc=True, errors="coerce")
    if bool(stamps.isna().any()):
        raise ValidationFailure(
            "Time-based interpolation refuses to run: at least one row has an "
            "unparseable timestamp, so the gap between two known points is unknown.",
            hint="Repair the timestamps first, or use forward fill.",
        )

    indexed = frame.set_index(stamps)
    before = int(indexed[targets].isna().sum().sum())
    # `limit_area="inside"` restricts every interpolation to the span between two known
    # observations. Without it pandas also extrapolates past the ends, which would
    # invent values beyond the data instead of between it.
    interpolated = indexed[targets].interpolate(method="time", limit_area="inside")
    result = indexed.copy()
    result[targets] = interpolated
    result = result.reset_index(drop=True)

    after = int(result[targets].isna().sum().sum())
    step["filled"] = before - after
    step["columns"] = targets
    step["detail"] = (
        f"Linearly interpolated {step['filled']} of {before} empty cell(s) between known "
        "points, weighted by the actual time gap rather than by row distance."
    )
    step["warnings"].append(
        "Interpolated cells are fabricated values that were never observed. Leading and "
        "trailing empty cells are deliberately left empty — filling them would be "
        "extrapolation past the data, not interpolation between it."
    )
    return result, step


# ----------------------------------------------------------- outliers


def _fences(
    series: pd.Series, plan: CleaningPlan, column: str
) -> tuple[float, float, str] | None:
    """Return (lower, upper, rule) for a column, or None when no fence is computable."""
    usable = series.dropna()
    if usable.empty:
        return None
    if plan.outliers is OutlierPolicy.IQR:
        q1 = float(usable.quantile(0.25))
        q3 = float(usable.quantile(0.75))
        spread = q3 - q1
        return (
            q1 - plan.iqr_multiplier * spread,
            q3 + plan.iqr_multiplier * spread,
            f"iqr({plan.iqr_multiplier:g})",
        )
    lower = float(usable.quantile(plan.lower_quantile))
    upper = float(usable.quantile(plan.upper_quantile))
    if not (lower < upper):
        return None
    return lower, upper, f"quantile({plan.lower_quantile:g},{plan.upper_quantile:g})"


def _handle_outliers(
    frame: pd.DataFrame, plan: CleaningPlan
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    policy = plan.outliers
    if policy is OutlierPolicy.KEEP:
        return frame, None

    step: dict[str, Any] = {
        "step": "outliers",
        "policy": policy.value,
        "columns": [],
        "fences": {},
        "rows_removed": 0,
        "cells_changed": 0,
        "warnings": [],
    }

    if not plan.outlier_columns:
        raise ValidationFailure(
            f"Outlier handling is set to {policy.value} but no column was selected.",
            hint="Choose the columns to treat. Selecting every numeric column would "
            "clamp price levels, which is almost never what you want — apply outlier "
            "handling to returns or spreads instead.",
        )

    missing = [c for c in plan.outlier_columns if c not in frame.columns]
    if missing:
        raise ValidationFailure(
            f"Outlier handling targets unknown column(s): {', '.join(sorted(missing))}."
        )
    non_numeric = [
        c
        for c in plan.outlier_columns
        if not pd.api.types.is_numeric_dtype(frame[c].dtype)
        or pd.api.types.is_bool_dtype(frame[c].dtype)
    ]
    if non_numeric:
        raise ValidationFailure(
            f"Outlier handling needs numeric columns; {', '.join(sorted(non_numeric))} "
            "is not one.",
            hint="Outlier detection on text or boolean data has no meaning.",
        )

    result = frame.copy()
    keep_mask = pd.Series(True, index=frame.index)
    for column in plan.outlier_columns:
        series = result[column]
        fence = _fences(series, plan, column)
        if fence is None:
            step["warnings"].append(
                f"{column}: no fence could be computed (the column is empty, constant, "
                "or too short), so it was left untouched."
            )
            continue
        lower, upper, rule = fence
        step["columns"].append(column)
        step["fences"][column] = {
            "lower": lower,
            "upper": upper,
            "rule": rule,
            # Counted before the repair so the report says how much data the fence
            # caught, not how much is left after it.
            "outside": int(((series < lower) | (series > upper)).sum()),
        }

        if policy is OutlierPolicy.REMOVE:
            keep_mask &= series.isna() | ((series >= lower) & (series <= upper))
        else:
            # `winsorize` and `iqr` both clamp. `isin`-style equality is not used here
            # because a float comparison on the fence value is exact by construction:
            # the fence is a value already present in the series or a linear combination
            # of two of them.
            changed = int(((series < lower) | (series > upper)).sum())
            step["cells_changed"] += changed
            result[column] = series.clip(lower=lower, upper=upper)

    if policy is OutlierPolicy.REMOVE:
        removed = int((~keep_mask).sum())
        step["rows_removed"] = removed
        result = result.loc[keep_mask]
        step["detail"] = (
            f"Removed {removed} row(s) falling outside the fence in "
            f"{len(step['columns'])} column(s). The result has holes in its timeline, "
            "which the missing-bar check will report."
        )
    else:
        step["detail"] = (
            f"Clamped {step['cells_changed']} cell(s) to the fence in "
            f"{len(step['columns'])} column(s)."
        )

    step["warnings"].append(
        "The fence was estimated over the whole dataset, including rows later than the "
        "ones it clamps. If this dataset feeds a backtest, that is a look-ahead — the "
        "fence values are recorded in the saved dataset's provenance so it can be "
        "audited."
    )
    return result, step
