"""Cleaning transforms.

The property under test throughout is that a cleaning step is *declared and reported*.
So the assertions are as much about what a plan did **not** touch, and about what the
report says, as about the values that changed. A repair that cannot be described after
the fact is indistinguishable from a corrupted dataset.

Note on the outlier tests: a quantile fence at the default 1%/99% is *designed* to
catch about 2% of rows, so the counts are asserted against the report's own `outside`
figure rather than against a magic number. A test asserting "exactly one row changed"
would be asserting that the fence is broken.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.core.errors import ValidationFailure
from app.models.enums import DuplicatePolicy, MissingValuePolicy, OutlierPolicy
from app.services import cleaning
from tests.conftest import hourly_frame


def plan(**overrides: Any) -> cleaning.CleaningPlan:
    return cleaning.CleaningPlan(**overrides)


# --------------------------------------------------------------- the default


def test_the_default_plan_changes_nothing_at_all() -> None:
    """Every policy defaults to KEEP, so a caller that forgets a field — or a future code
    path that builds a plan without one — repairs nothing rather than repairing something
    nobody asked for."""
    frame = hourly_frame(50)
    frame.loc[10, "close"] = float("nan")
    frame.loc[11, "close"] = 10_000.0

    result, report = cleaning.clean_frame(frame, plan(), "open_time")

    pd.testing.assert_frame_equal(result, frame)
    assert report["rows_removed"] == 0
    # No step ran, so no step is reported. Reporting a no-op step would suggest something
    # was considered and decided, which is a different claim.
    assert report["steps"] == []
    assert report["warnings"] == []


def test_the_report_carries_the_shape_before_and_after() -> None:
    frame = hourly_frame(20)
    result, report = cleaning.clean_frame(
        frame, plan(missing=MissingValuePolicy.DROP_ROWS), "open_time"
    )
    assert report["rows_before"] == 20
    assert report["rows_after"] == 20
    assert report["rows_removed"] == 0
    assert report["columns_before"] == report["columns_after"]
    assert len(result) == 20


# -------------------------------------------------------------- duplicates


def test_drop_exact_removes_only_rows_identical_in_every_column() -> None:
    frame = hourly_frame(10)
    doubled = pd.concat([frame, frame.iloc[[3, 7]]], ignore_index=True)

    result, report = cleaning.clean_frame(
        doubled, plan(duplicates=DuplicatePolicy.DROP_EXACT), "open_time"
    )

    assert len(result) == 10
    assert report["rows_removed"] == 2
    assert report["steps"][0]["step"] == "duplicates"


def test_drop_exact_leaves_rows_that_differ_in_one_column() -> None:
    """A repeated timestamp is not an exact duplicate when the *values* differ. Treating
    it as one would delete a real correction made to a bar."""
    frame = hourly_frame(5)
    repeated = pd.concat([frame, frame.iloc[[2]]], ignore_index=True)
    repeated.loc[5, "close"] = 999.0

    result, _ = cleaning.clean_frame(
        repeated, plan(duplicates=DuplicatePolicy.DROP_EXACT), "open_time"
    )
    assert len(result) == 6


def test_keep_first_by_time_collapses_rows_that_share_a_timestamp() -> None:
    frame = hourly_frame(5)
    repeated = pd.concat([frame, frame.iloc[[2]]], ignore_index=True)
    repeated.loc[5, "close"] = 999.0

    result, report = cleaning.clean_frame(
        repeated, plan(duplicates=DuplicatePolicy.KEEP_FIRST_BY_TIME), "open_time"
    )

    assert len(result) == 5
    assert report["rows_removed"] == 1
    kept = result.loc[result["open_time"] == frame.loc[2, "open_time"], "close"].iloc[0]
    assert kept == frame.loc[2, "close"]


def test_keep_last_by_time_keeps_the_other_row() -> None:
    """The two policies must actually differ. A shared implementation that always kept
    the first row would pass the test above."""
    frame = hourly_frame(5)
    repeated = pd.concat([frame, frame.iloc[[2]]], ignore_index=True)
    repeated.loc[5, "close"] = 999.0

    result, _ = cleaning.clean_frame(
        repeated, plan(duplicates=DuplicatePolicy.KEEP_LAST_BY_TIME), "open_time"
    )
    kept = result.loc[result["open_time"] == frame.loc[2, "open_time"], "close"].iloc[0]
    assert kept == 999.0


def test_time_based_deduplication_needs_a_timestamp() -> None:
    frame = pd.DataFrame({"close": [1.0, 1.0]})
    with pytest.raises(ValidationFailure, match="no recognisable timestamp"):
        cleaning.clean_frame(frame, plan(duplicates=DuplicatePolicy.KEEP_LAST_BY_TIME), None)


def test_sorting_happens_before_deduplication_so_first_means_first_in_time() -> None:
    """`drop_duplicates(keep="first")` keeps the first row *in row order*.

    On an unsorted frame that is whichever row the file happened to list first, which is
    not what "keep the first observation for this timestamp" means. Sorting first is what
    makes the word "first" describe time.
    """
    frame = hourly_frame(3)
    shuffled = frame.iloc[[2, 0, 1]].copy()
    extra = frame.iloc[[1]].copy()
    extra["close"] = 777.0
    shuffled = pd.concat([shuffled, extra], ignore_index=True)

    result, _ = cleaning.clean_frame(
        shuffled,
        plan(duplicates=DuplicatePolicy.KEEP_FIRST_BY_TIME, sort_by_time=True),
        "open_time",
    )

    assert len(result) == 3
    kept = result.loc[result["open_time"] == frame.loc[1, "open_time"], "close"].iloc[0]
    assert kept == frame.loc[1, "close"], "the chronologically first row should win"


def test_sort_by_time_without_a_timestamp_is_refused() -> None:
    with pytest.raises(ValidationFailure, match="Cannot sort by time"):
        cleaning.clean_frame(pd.DataFrame({"close": [1.0]}), plan(sort_by_time=True), None)


def test_sort_by_time_is_stable_so_ties_keep_their_original_order() -> None:
    """An unstable sort would order same-timestamp rows arbitrarily, making the result
    depend on the sort implementation rather than on the data."""
    frame = hourly_frame(4)
    frame["marker"] = [0, 1, 2, 3]
    tie = frame.iloc[[1]].copy()
    tie["close"] = 555.0
    tie["marker"] = 99
    frame = pd.concat([frame, tie], ignore_index=True)

    result, _ = cleaning.clean_frame(frame, plan(sort_by_time=True), "open_time")
    same_stamp = result.loc[result["open_time"] == frame.loc[1, "open_time"], "marker"]
    assert list(same_stamp) == [1, 99]


# ---------------------------------------------------------- missing values


def test_forward_fill_carries_the_previous_value_into_the_gap() -> None:
    frame = hourly_frame(6)
    frame.loc[2, "close"] = float("nan")
    frame.loc[3, "close"] = float("nan")

    result, report = cleaning.clean_frame(
        frame, plan(missing=MissingValuePolicy.FFILL), "open_time"
    )

    assert result.loc[2, "close"] == frame.loc[1, "close"]
    assert result.loc[3, "close"] == frame.loc[1, "close"]
    step = report["steps"][0]
    assert step["filled"] == 2
    assert step["columns"] == ["open", "high", "low", "close", "volume"]


def test_forward_fill_leaves_a_leading_hole_empty() -> None:
    """Nothing precedes the first row, so there is no value to carry forward. Filling it
    from the future would be a backward fill wearing a forward fill's name."""
    frame = hourly_frame(5)
    frame.loc[0, "close"] = float("nan")

    result, _ = cleaning.clean_frame(frame, plan(missing=MissingValuePolicy.FFILL), "open_time")
    assert bool(pd.isna(result.loc[0, "close"]))


def test_backward_fill_fills_a_leading_hole_and_declares_the_look_ahead() -> None:
    """Backward fill is the mirror of forward fill, so it is the only fill that can
    repair a hole at the *start* of a series — and the only one that puts a value into a
    cell from a moment that had not happened yet."""
    frame = hourly_frame(5)
    frame.loc[0, "close"] = float("nan")

    result, report = cleaning.clean_frame(
        frame, plan(missing=MissingValuePolicy.BFILL), "open_time"
    )

    assert result.loc[0, "close"] == frame.loc[1, "close"]
    assert report["steps"][0]["filled"] == 1
    assert any("look-ahead" in warning for warning in report["warnings"])


def test_neither_fill_can_repair_a_both_ends_hole() -> None:
    """A hole that is leading *and* trailing has nothing on either side. Both fills must
    leave it empty rather than reach across the whole series for a value."""
    frame = hourly_frame(4)
    frame.loc[0, "close"] = float("nan")
    frame.loc[3, "close"] = float("nan")

    forward, _ = cleaning.clean_frame(frame, plan(missing=MissingValuePolicy.FFILL), "open_time")
    backward, _ = cleaning.clean_frame(frame, plan(missing=MissingValuePolicy.BFILL), "open_time")

    assert bool(pd.isna(forward.loc[0, "close"]))
    assert bool(pd.isna(backward.loc[3, "close"]))


def test_zero_fill_warns_that_a_zero_price_is_a_minus_one_hundred_percent_return() -> None:
    """Zero is a real observation for an open-interest or volume field and a catastrophic
    one for a price. That warning is the only thing between the user and a silently
    destroyed return series."""
    frame = hourly_frame(5)
    frame.loc[2, "close"] = float("nan")

    result, report = cleaning.clean_frame(
        frame, plan(missing=MissingValuePolicy.ZERO), "open_time"
    )

    assert result.loc[2, "close"] == 0.0
    assert any("−100%" in warning for warning in report["warnings"])


def test_drop_rows_removes_a_row_for_a_hole_anywhere_in_it() -> None:
    frame = hourly_frame(6)
    frame.loc[2, "close"] = float("nan")
    frame.loc[4, "volume"] = float("nan")

    result, report = cleaning.clean_frame(
        frame, plan(missing=MissingValuePolicy.DROP_ROWS), "open_time"
    )

    assert len(result) == 4
    assert report["rows_removed"] == 2
    assert "including columns unrelated" in report["steps"][0]["detail"]


def test_interpolation_is_weighted_by_elapsed_time_not_row_distance() -> None:
    """A hole's neighbours are often not evenly spaced.

    Here 04:00 sits three quarters of the way in *time* between 01:00 and 05:00, so the
    interpolated value is 400. Weighting the two known points equally — which is what
    interpolation on row distance does — would report 300, and that error is invisible in
    the output.
    """
    frame = pd.DataFrame(
        {
            "open_time": pd.to_datetime(
                ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 04:00", "2024-01-01 05:00"],
                utc=True,
            ),
            "close": [0.0, 100.0, float("nan"), 500.0],
        }
    )

    result, report = cleaning.clean_frame(
        frame, plan(missing=MissingValuePolicy.INTERPOLATE_TIME), "open_time"
    )

    assert result.loc[2, "close"] == pytest.approx(400.0)
    assert report["steps"][0]["filled"] == 1


def test_interpolation_leaves_leading_and_trailing_holes_empty() -> None:
    """`limit_area="inside"` is what enforces this. Without it pandas extrapolates past
    the ends, inventing values beyond the data instead of between it."""
    frame = pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC"),
            "close": [float("nan"), 100.0, float("nan"), 200.0, float("nan")],
        }
    )

    result, report = cleaning.clean_frame(
        frame, plan(missing=MissingValuePolicy.INTERPOLATE_TIME), "open_time"
    )

    assert result.loc[2, "close"] == pytest.approx(150.0)
    assert bool(pd.isna(result.loc[0, "close"])), "no value before the first observation"
    assert bool(pd.isna(result.loc[4, "close"])), "no value after the last observation"
    assert report["steps"][0]["filled"] == 1
    assert any("extrapolation" in warning for warning in report["warnings"])


def test_interpolation_refuses_an_unparseable_timestamp() -> None:
    """The gap between two known points is unknown when one of them has no clock, and the
    interpolation would silently treat it as zero seconds away."""
    frame = hourly_frame(5)
    frame.loc[2, "open_time"] = pd.NaT
    frame.loc[3, "close"] = float("nan")

    with pytest.raises(ValidationFailure, match="unparseable timestamp"):
        cleaning.clean_frame(
            frame, plan(missing=MissingValuePolicy.INTERPOLATE_TIME), "open_time"
        )


def test_interpolation_without_a_timestamp_is_refused_with_a_usable_alternative() -> None:
    frame = pd.DataFrame({"close": [1.0, float("nan"), 3.0]})
    with pytest.raises(ValidationFailure, match="needs a timestamp column"):
        cleaning.clean_frame(frame, plan(missing=MissingValuePolicy.INTERPOLATE_TIME), None)


def test_the_timestamp_column_is_never_filled_even_when_it_is_numeric() -> None:
    """An epoch-seconds timestamp column is numeric, so a fill would happily interpolate
    it. Inventing a bar timestamp is inventing an observation, and it would also change
    the bar spacing the missing-bar detector measures — so a dataset could be "repaired"
    into failing a check it previously passed.
    """
    frame = pd.DataFrame(
        {
            "open_time": [1_704_067_200.0, 1_704_070_800.0, np.nan, 1_704_078_000.0],
            "close": [1.0, 2.0, 3.0, float("nan")],
        }
    )

    result, report = cleaning.clean_frame(
        frame, plan(missing=MissingValuePolicy.ZERO), "open_time"
    )

    assert bool(pd.isna(result.loc[2, "open_time"])), "the timestamp hole must survive"
    assert result.loc[0, "open_time"] == 1_704_067_200.0
    assert result.loc[3, "close"] == 0.0, "the data column is still filled"
    assert report["steps"][0]["columns"] == ["close"]


def test_a_boolean_column_is_not_zero_filled() -> None:
    """Interpolating a boolean, or zero-filling it, produces a value that is not a
    boolean. `False` is the only defensible fill and choosing it silently would be a
    decision about the data, not a repair of it."""
    frame = hourly_frame(4)
    frame["regime"] = pd.array([True, False, None, True], dtype="boolean")

    result, report = cleaning.clean_frame(
        frame, plan(missing=MissingValuePolicy.ZERO), "open_time"
    )

    assert bool(pd.isna(result.loc[2, "regime"]))
    assert "regime" not in report["steps"][0]["columns"]


def test_a_fill_with_no_numeric_column_says_so_rather_than_reporting_success() -> None:
    frame = pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC"),
            "label": ["a", "b", "c"],
        }
    )
    _, report = cleaning.clean_frame(frame, plan(missing=MissingValuePolicy.FFILL), "open_time")
    assert report["steps"][0]["filled"] == 0
    assert any("no numeric column" in warning for warning in report["warnings"])


# --------------------------------------------------------------- outliers


def test_outlier_policies_default_to_touching_nothing() -> None:
    frame = hourly_frame(100)
    frame.loc[50, "volume"] = 10_000_000.0
    result, _ = cleaning.clean_frame(frame, plan(), "open_time")
    assert result.loc[50, "volume"] == 10_000_000.0


def test_an_outlier_policy_with_no_column_selected_is_refused() -> None:
    """The alternative — applying it to every numeric column — would clamp price levels,
    which is the single most damaging thing a one-click outlier button can do."""
    with pytest.raises(ValidationFailure, match="no column was selected"):
        cleaning.clean_frame(
            hourly_frame(100), plan(outliers=OutlierPolicy.WINSORIZE), "open_time"
        )


def test_an_outlier_policy_on_an_unknown_column_is_refused() -> None:
    with pytest.raises(ValidationFailure, match="unknown column"):
        cleaning.clean_frame(
            hourly_frame(50),
            plan(outliers=OutlierPolicy.WINSORIZE, outlier_columns=("nope",)),
            "open_time",
        )


def test_an_outlier_policy_on_a_text_column_is_refused() -> None:
    frame = hourly_frame(50)
    frame["venue"] = ["binance"] * 50
    with pytest.raises(ValidationFailure, match="needs numeric columns"):
        cleaning.clean_frame(
            frame,
            plan(outliers=OutlierPolicy.WINSORIZE, outlier_columns=("venue",)),
            "open_time",
        )


def test_winsorization_clamps_every_value_to_the_quantile_fence() -> None:
    frame = hourly_frame(1000)
    frame.loc[500, "volume"] = 1e9

    result, report = cleaning.clean_frame(
        frame,
        plan(outliers=OutlierPolicy.WINSORIZE, outlier_columns=("volume",)),
        "open_time",
    )

    step = report["steps"][0]
    fence = step["fences"]["volume"]
    assert fence["rule"] == "quantile(0.01,0.99)"
    assert result["volume"].max() == pytest.approx(fence["upper"])
    assert result["volume"].min() == pytest.approx(fence["lower"])
    assert result["volume"].max() < 1e9, "the extreme value is gone, not merely shrunk"
    # `outside` is counted before the repair, so it says how much data the fence caught
    # rather than how much is left after it. The two must agree with the cells changed.
    assert fence["outside"] == step["cells_changed"]
    assert fence["outside"] >= 1
    assert step["columns"] == ["volume"]


def test_the_outlier_fence_is_always_declared_a_look_ahead() -> None:
    """The fence is estimated over the whole frame, including rows later than the ones it
    clamps. That is a look-ahead, and the report has to say so — the reason the fences are
    written into the saved dataset's provenance is so a backtest built on it can be
    audited for exactly this."""
    _, report = cleaning.clean_frame(
        hourly_frame(200),
        plan(outliers=OutlierPolicy.WINSORIZE, outlier_columns=("close",)),
        "open_time",
    )
    assert any("look-ahead" in warning for warning in report["warnings"])


def test_iqr_uses_the_interquartile_rule() -> None:
    frame = hourly_frame(1000)
    frame.loc[500, "volume"] = 1e9

    result, report = cleaning.clean_frame(
        frame,
        plan(outliers=OutlierPolicy.IQR, outlier_columns=("volume",), iqr_multiplier=3.0),
        "open_time",
    )

    fence = report["steps"][0]["fences"]["volume"]
    assert fence["rule"] == "iqr(3)"
    assert result["volume"].max() <= fence["upper"]


def test_removing_extremes_drops_the_rows_and_reports_the_holes_it_leaves() -> None:
    frame = hourly_frame(500)
    frame.loc[250, "volume"] = 1e9

    result, report = cleaning.clean_frame(
        frame,
        plan(outliers=OutlierPolicy.REMOVE, outlier_columns=("volume",)),
        "open_time",
    )

    step = report["steps"][0]
    assert step["fences"]["volume"]["outside"] == step["rows_removed"] >= 1
    assert report["rows_removed"] == step["rows_removed"]
    assert len(result) == report["rows_after"]
    assert 1e9 not in set(result["volume"])
    assert "holes in its timeline" in step["detail"]


def test_remove_keeps_rows_that_are_empty_in_the_target_column() -> None:
    """A missing value is not an outlier. Dropping it here would silently perform a
    missing-value repair the user did not select.

    Note that `outside` counts only genuine deviations: a NaN fails every comparison
    against the fences, so it is neither clamped by the other policies nor counted here.
    """
    frame = hourly_frame(200)
    frame.loc[10, "volume"] = float("nan")

    result, report = cleaning.clean_frame(
        frame,
        plan(outliers=OutlierPolicy.REMOVE, outlier_columns=("volume",)),
        "open_time",
    )

    step = report["steps"][0]
    assert frame.loc[10, "open_time"] in set(result["open_time"])
    assert step["fences"]["volume"]["outside"] == step["rows_removed"]


def test_a_column_that_cannot_produce_a_fence_is_reported_not_clamped() -> None:
    """A constant column has quantile(0.99) == quantile(0.01). The fence degenerates to a
    single point and clamping to it would rewrite every row in the column."""
    frame = hourly_frame(100)
    frame["flat"] = 1.0

    result, report = cleaning.clean_frame(
        frame,
        plan(outliers=OutlierPolicy.WINSORIZE, outlier_columns=("flat",)),
        "open_time",
    )

    assert list(result["flat"]) == [1.0] * 100
    assert "flat" not in report["steps"][0]["fences"]
    assert any("no fence could be computed" in w for w in report["warnings"])


def test_an_empty_outlier_column_is_reported_rather_than_crashing() -> None:
    frame = hourly_frame(50)
    frame["blank"] = float("nan")

    _, report = cleaning.clean_frame(
        frame,
        plan(outliers=OutlierPolicy.WINSORIZE, outlier_columns=("blank",)),
        "open_time",
    )
    assert "blank" not in report["steps"][0]["fences"]


# -------------------------------------------------------------- step order


def test_every_selected_step_is_reported_in_the_documented_order() -> None:
    """Step order is observable in the result, so it is fixed rather than left to the
    order the options happened to be wired up in. `sort` precedes `duplicates` because it
    is what makes "first" and "last" mean first and last *in time*."""
    frame = hourly_frame(100)
    frame.loc[5, "close"] = float("nan")
    frame.loc[60, "volume"] = 1e9

    _, report = cleaning.clean_frame(
        frame,
        plan(
            sort_by_time=True,
            duplicates=DuplicatePolicy.KEEP_FIRST_BY_TIME,
            missing=MissingValuePolicy.FFILL,
            outliers=OutlierPolicy.REMOVE,
            outlier_columns=("volume",),
        ),
        "open_time",
    )

    assert [step["step"] for step in report["steps"]] == [
        "sort",
        "duplicates",
        "missing",
        "outliers",
    ]


def test_row_removals_from_separate_steps_add_up_to_the_total() -> None:
    """Each step reports its own removals, so the total has to be their sum. A step that
    forgot to update the running count would produce a report whose arithmetic does not
    close, which is exactly the kind of thing the report exists to make checkable."""
    frame = hourly_frame(300)
    frame.loc[10, "volume"] = float("nan")
    frame.loc[150, "volume"] = 1e9

    result, report = cleaning.clean_frame(
        frame,
        plan(
            missing=MissingValuePolicy.DROP_ROWS,
            outliers=OutlierPolicy.REMOVE,
            outlier_columns=("volume",),
        ),
        "open_time",
    )

    per_step = sum(step.get("rows_removed", 0) for step in report["steps"])
    assert report["rows_removed"] == per_step
    assert report["rows_before"] - report["rows_after"] == len(frame) - len(result)


# ------------------------------------------------------------- determinism


def test_cleaning_is_deterministic() -> None:
    frame = hourly_frame(300)
    frame.loc[10, "close"] = float("nan")
    frame.loc[200, "volume"] = 1e9
    full = plan(
        missing=MissingValuePolicy.FFILL,
        outliers=OutlierPolicy.WINSORIZE,
        outlier_columns=("volume",),
    )

    first, first_report = cleaning.clean_frame(frame, full, "open_time")
    second, second_report = cleaning.clean_frame(frame, full, "open_time")

    pd.testing.assert_frame_equal(first, second)
    assert first_report == second_report


def test_a_non_default_index_does_not_change_the_result() -> None:
    """Datasets arrive with any index — a CSV upload, a filtered frame. A cleaning step
    that indexed positionally into the frame would operate on the wrong rows."""
    frame = hourly_frame(60)
    frame.loc[10, "close"] = float("nan")

    plain, _ = cleaning.clean_frame(frame, plan(missing=MissingValuePolicy.FFILL), "open_time")

    shifted = frame.copy()
    shifted.index = np.arange(1000, 1060)
    offset, _ = cleaning.clean_frame(shifted, plan(missing=MissingValuePolicy.FFILL), "open_time")

    assert list(plain["close"]) == list(offset["close"])
