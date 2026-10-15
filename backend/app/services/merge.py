"""Resampling, alignment, and merging.

The alignment rule is the whole point of this module, so it is stated once and applied
everywhere: **a joined value is only ever the most recent observation at or before the
bar being labelled.** Funding rates arrive every eight hours and metrics on their own
cadence; a kline at 03:00 may only know a funding rate published at 00:00, and that is
what it gets. Anything else — a forward fill, a nearest match — stamps a bar with a
number that did not exist when the bar closed, which is look-ahead. `AlignMethod`
therefore offers `backward` and `exact` and nothing else.

Only the *base* frame is resampled. A finer joined series does not need to be reduced
to the base's cadence first, because the as-of rule already collapses it: at each bar,
the last value at or before it is the same value a prior resample would have selected.
Resampling it as well would be redundant work that adds a second place for the
aggregation rule to be wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from app.core.errors import ValidationFailure
from app.models.enums import TIMEFRAME_SECONDS, AlignMethod, MergeHow, Timeframe
from app.services.cleaning import OHLC_AGGREGATIONS


@dataclass(frozen=True)
class JoinSpec:
    """One dataset to place onto the base timeline."""

    dataset_id: str
    how: MergeHow = MergeHow.LEFT
    method: AlignMethod = AlignMethod.BACKWARD
    #: Prepended to every column this join adds. Required when a column would otherwise
    #: collide — a merge that silently renames one of two `close` columns produces a
    #: frame whose meaning depends on join order.
    prefix: str = ""
    #: Maximum staleness accepted by `backward`, in seconds. None accepts a value of
    #: any age, which is how a funding rate quoted eight hours earlier legitimately
    #: reaches a one-hour bar.
    tolerance_seconds: int | None = None


def resample_frame(
    frame: pd.DataFrame,
    time_column: str,
    timeframe: Timeframe,
    source_timeframe: Timeframe | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate bars up to `timeframe`.

    Refuses to go the other way. Building four 15-minute bars from one hourly bar
    invents three open/high/low paths that no market produced, and every downstream
    statistic would be computed on fabricated structure.
    """
    target_seconds = TIMEFRAME_SECONDS[timeframe]
    report: dict[str, Any] = {
        "timeframe": timeframe.value,
        "rows_before": int(len(frame)),
        "aggregations": {},
        "warnings": [],
    }

    if source_timeframe is not None:
        source_seconds = TIMEFRAME_SECONDS[source_timeframe]
        if target_seconds < source_seconds:
            raise ValidationFailure(
                f"Cannot resample {source_timeframe.value} bars up to "
                f"{timeframe.value}: that would fabricate detail the source never had.",
                hint=f"Choose {source_timeframe.value} or a coarser timeframe.",
            )
        if target_seconds == source_seconds:
            report["detail"] = (
                f"Source is already {timeframe.value}; no resampling was applied."
            )
            report["rows_after"] = int(len(frame))
            return frame, report
    else:
        report["warnings"].append(
            "The base dataset declares no timeframe, so the request could not be "
            "checked against the source interval. If the target is finer than the "
            "source, the result contains fabricated bars."
        )

    stamps = pd.to_datetime(frame[time_column], utc=True, errors="coerce")
    if bool(stamps.isna().any()):
        raise ValidationFailure(
            f"Resampling needs a parseable timestamp in every row; {time_column} has "
            "at least one unparseable value.",
            hint="Run the Cleaning step first to repair or remove those rows.",
        )

    indexed = frame.drop(columns=[time_column]).set_index(stamps)
    indexed.index.name = time_column

    aggregations: dict[str, str] = {}
    for column in indexed.columns:
        name = str(column)
        if name in OHLC_AGGREGATIONS:
            aggregations[column] = OHLC_AGGREGATIONS[name]
        else:
            # `last`, not `mean`. Open interest, a funding rate, and a long/short ratio
            # are levels observed at an instant; averaging them over an hour reports a
            # value that was never observed at any time. Volume-like fields are the
            # exception and are named in `OHLC_AGGREGATIONS`.
            aggregations[column] = "last"

    # `label="left", closed="left"` matches how Binance and this application stamp bars:
    # the timestamp is the *opening* instant of the interval. pandas' defaults would
    # label the bar with its close, shifting every timestamp by one bar.
    step = pd.Timedelta(seconds=target_seconds)
    resampled = indexed.resample(step, label="left", closed="left").agg(aggregations)

    # An interval with no source rows has no observation and must not become a bar.
    # Counting the source rows per bin is the only test that answers that question:
    # `dropna(how="all")` does not, because `sum` over an empty group is 0 rather than
    # NaN, so a volume column keeps the row alive and the result carries a fabricated bar
    # with no prices and a zero volume. It is also too strong in the other direction — a
    # bin whose rows all happen to be empty is a real hole and must stay visible to the
    # missing-value detector, which `dropna(how="all")` would have hidden.
    occupied = indexed.resample(step, label="left", closed="left").size() > 0
    resampled = resampled.loc[occupied.to_numpy()]
    resampled = resampled.reset_index()
    resampled = resampled.sort_values(time_column, kind="stable").reset_index(drop=True)

    report["aggregations"] = {str(k): v for k, v in aggregations.items()}
    report["rows_after"] = int(len(resampled))
    report["detail"] = (
        f"Aggregated {report['rows_before']} row(s) into {report['rows_after']} "
        f"{timeframe.value} bar(s), labelled by interval open."
    )
    return resampled, report


def align_and_merge(
    base: pd.DataFrame,
    time_column: str,
    inputs: list[tuple[JoinSpec, pd.DataFrame, str, str]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Place each input onto `base`'s timeline and return the widened frame.

    `inputs` carries, per join: the spec, the frame, a human label for the report, and
    the name of that frame's own timestamp column.
    """
    result = base
    reports: list[dict[str, Any]] = []
    for spec, frame, label, right_time in inputs:
        result, report = _join_one(result, time_column, frame, right_time, spec, label)
        reports.append(report)
    return result, reports


def _join_one(
    left: pd.DataFrame,
    time_column: str,
    right: pd.DataFrame,
    right_time: str,
    spec: JoinSpec,
    label: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    report: dict[str, Any] = {
        "dataset_id": spec.dataset_id,
        "label": label,
        "method": spec.method.value,
        "how": spec.how.value,
        "prefix": spec.prefix,
        "tolerance_seconds": spec.tolerance_seconds,
        "columns_added": [],
        "rows": int(len(left)),
        "unmatched": 0,
        "warnings": [],
    }

    if right_time not in right.columns:
        raise ValidationFailure(
            f"{label} has no recognisable timestamp column, so it cannot be aligned "
            f"onto {time_column}.",
            hint="Only datasets with an open_time, create_time, calc_time, or Date "
            "column can be merged.",
        )

    if right.empty:
        raise ValidationFailure(f"{label} has no rows to merge.")

    stamps = pd.to_datetime(right[right_time], utc=True, errors="coerce")
    if bool(stamps.isna().any()):
        raise ValidationFailure(
            f"{label} has at least one unparseable timestamp in {right_time}.",
            hint="Run the Cleaning step on it first.",
        )

    repeated = int(stamps.duplicated(keep=False).sum())
    if repeated:
        raise ValidationFailure(
            f"{label} has {repeated} row(s) sharing a timestamp, so there is no single "
            "value to align onto each bar.",
            hint=f"Run Cleaning on {label} with a duplicate-timestamp policy first.",
        )

    # Prefix first, then rename the key. Doing it the other way round would collide with
    # any column in the right frame that already carries the base's key name, and pandas
    # tolerates duplicate column labels — the resulting frame would have two columns
    # called `open_time`, one of which is data, and nothing downstream could tell them
    # apart.
    prepared = right.copy()
    if spec.prefix:
        prepared = prepared.rename(
            columns={
                column: f"{spec.prefix}{column}"
                for column in prepared.columns
                if column != right_time
            }
        )
    prepared = prepared.rename(columns={right_time: time_column})

    # Every duplicate label is a defect here, the key column included. The exemption that
    # used to be applied to `time_column` was wrong: the key legitimately appears once in
    # each frame, but two columns *inside* `prepared` both called `open_time` means the
    # joined dataset has a data column wearing the timestamp's name, and nothing
    # downstream — a resample, a chart, a feature matrix — could tell them apart.
    duplicated_labels = [
        str(column) for column in prepared.columns[prepared.columns.duplicated()]
    ]
    if duplicated_labels:
        raise ValidationFailure(
            f"{label} would contribute two columns named "
            f"{', '.join(sorted(set(duplicated_labels)))}.",
            hint="Set a prefix for this join so its columns stay distinguishable.",
        )

    # The key itself is shared by design, so it is exempt here but not above.
    collisions = [
        str(column)
        for column in prepared.columns
        if column != time_column and column in left.columns
    ]
    if collisions:
        raise ValidationFailure(
            f"{label} adds column(s) that already exist: {', '.join(sorted(collisions))}.",
            hint="Set a prefix for this join so both columns keep their own name.",
        )

    # A sentinel that exists only on the right side turns "did this bar find a value?"
    # into a null test that does not depend on the joined columns' dtypes or on whether
    # a matched row happened to be empty.
    marker = "__qca_matched"
    prepared = prepared.copy()
    prepared[time_column] = stamps
    prepared[marker] = True

    left_sorted = left.copy()
    left_sorted[time_column] = pd.to_datetime(left[time_column], utc=True, errors="coerce")
    if bool(left_sorted[time_column].isna().any()):
        # merge_asof on an unparseable key does not raise — it sorts NaT to one end and
        # matches the wrong neighbour, so every aligned value in the result would be
        # silently attached to the wrong bar.
        raise ValidationFailure(
            f"The base dataset has at least one unparseable timestamp in {time_column}, "
            "so bars cannot be aligned onto it.",
            hint="Run the Cleaning step on the base dataset first, with a timestamp "
            "repair or row removal.",
        )
    left_sorted = left_sorted.sort_values(time_column, kind="stable")
    prepared = prepared.sort_values(time_column, kind="stable")

    if spec.method is AlignMethod.EXACT:
        base_repeats = int(left_sorted[time_column].duplicated(keep=False).sum())
        if base_repeats:
            raise ValidationFailure(
                f"An exact join needs one row per timestamp in the base dataset, which "
                f"has {base_repeats} row(s) sharing a timestamp.",
                hint="Every one of them would match every duplicate in the joined "
                "dataset, multiplying rows. Run Cleaning on the base dataset first.",
            )
        merged = pd.merge(left_sorted, prepared, on=time_column, how=spec.how.value)
        report["detail"] = (
            f"Exact join on {time_column}: only bars whose timestamp appears in both "
            "datasets survive."
        )
    else:
        tolerance = (
            pd.Timedelta(seconds=spec.tolerance_seconds)
            if spec.tolerance_seconds is not None
            else None
        )
        merged = pd.merge_asof(
            left_sorted,
            prepared,
            on=time_column,
            direction="backward",
            tolerance=tolerance,
        )
        report["detail"] = (
            f"As-of join: each bar takes the most recent {label} value at or before its "
            "timestamp"
            + (
                f", up to {spec.tolerance_seconds}s old."
                if spec.tolerance_seconds is not None
                else ", at any age."
            )
        )

    if spec.method is AlignMethod.BACKWARD and spec.how is MergeHow.INNER:
        # merge_asof has no `how`: it is always a left join. An inner join is therefore
        # the left join with the unmatched bars removed afterwards.
        matched_mask = merged[marker].notna()
        dropped = int((~matched_mask).sum())
        merged = merged.loc[matched_mask]
        report["rows"] = int(len(merged))
        if dropped:
            report["detail"] += f" {dropped} bar(s) with no match were dropped (inner join)."

    unmatched = int(merged[marker].isna().sum())
    report["unmatched"] = unmatched
    # The sentinel is dropped *before* the column list is read. Computed the other way
    # round it lands in `columns_added`, and the report tells the user the merge added a
    # column called `__qca_matched` that they will never see in the dataset.
    merged = merged.drop(columns=[marker]).reset_index(drop=True)
    report["rows"] = int(len(merged))
    report["columns_added"] = [
        str(column) for column in merged.columns if column not in left.columns
    ]

    if unmatched:
        report["warnings"].append(
            f"{unmatched} of {report['rows']} base bar(s) found no {label} value and "
            "carry an empty cell for every column this join adds."
        )
    return merged, report
