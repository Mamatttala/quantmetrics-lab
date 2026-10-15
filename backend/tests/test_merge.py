"""Resampling, alignment, and merging.

The alignment rule is the only thing here that can silently corrupt a research dataset,
so the tests around it are written to fail if a value ever reaches a bar from the future.
That failure mode is invisible in the output — the merged frame looks perfectly well
formed — so it has to be pinned by assertion rather than by inspection.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from app.config import Settings
from app.core.errors import ValidationFailure
from app.models.enums import AlignMethod, MergeHow, OutlierPolicy, Timeframe
from app.services import merge
from app.services.cleaning import CleaningPlan
from app.services.datasets import DatasetService
from app.services.merge import JoinSpec
from app.storage.catalog import Catalog
from tests.conftest import hourly_frame, register


def funding_frame(times: list[str], rates: list[float]) -> pd.DataFrame:
    """A funding-rate dataset on its own cadence, as Binance delivers it."""
    return pd.DataFrame(
        {
            "calc_time": pd.to_datetime(times, utc=True),
            "last_funding_rate": rates,
        }
    )


def times_of(frame: pd.DataFrame, column: str = "open_time") -> list[pd.Timestamp]:
    return list(pd.to_datetime(frame[column], utc=True))


def at(frame: pd.DataFrame, column: str, moment: str) -> Any:
    """The value of `column` on the bar stamped `moment`."""
    stamps = pd.to_datetime(frame["open_time"], utc=True)
    selected = frame.loc[stamps == pd.Timestamp(moment, tz="UTC"), column]
    assert len(selected) == 1, f"expected exactly one bar at {moment}, found {len(selected)}"
    return selected.iloc[0]


# =============================================================== resampling


def test_resampling_refuses_to_build_finer_bars_than_the_source() -> None:
    """Four 15-minute bars built from one hourly bar would invent three open/high/low
    paths no market produced, and every statistic downstream would be computed on that
    fabricated structure."""
    with pytest.raises(ValidationFailure, match="fabricate detail"):
        merge.resample_frame(hourly_frame(48), "open_time", Timeframe.M15, Timeframe.H1)


def test_resampling_to_the_source_timeframe_is_a_no_op() -> None:
    frame = hourly_frame(24)
    result, report = merge.resample_frame(frame, "open_time", Timeframe.H1, Timeframe.H1)

    pd.testing.assert_frame_equal(result, frame)
    assert "already 1h" in report["detail"]
    assert report["rows_after"] == 24


def test_resampling_a_finer_timeframe_up_is_allowed() -> None:
    frame = hourly_frame(8)
    result, _ = merge.resample_frame(frame, "open_time", Timeframe.H4, Timeframe.M15)
    assert len(result) == 2


def test_ohlc_columns_aggregate_by_their_own_rule() -> None:
    """`close` must be the last close in the window and `high` the maximum, not the mean
    of anything. Averaging prices reports a level that was never traded."""
    frame = hourly_frame(4)
    result, report = merge.resample_frame(frame, "open_time", Timeframe.H4, Timeframe.H1)

    assert len(result) == 1
    assert at(result, "open", "2024-01-01 00:00") == frame.loc[0, "open"]
    assert at(result, "close", "2024-01-01 00:00") == frame.loc[3, "close"]
    assert at(result, "high", "2024-01-01 00:00") == frame["high"].max()
    assert at(result, "low", "2024-01-01 00:00") == frame["low"].min()
    assert at(result, "volume", "2024-01-01 00:00") == frame["volume"].sum()
    assert report["aggregations"]["high"] == "max"


def test_a_level_column_takes_the_last_value_not_the_mean() -> None:
    """Open interest, a funding rate, and a long/short ratio are levels observed at an
    instant. Averaging them over four hours reports a number that was never observed at
    any time — and it looks entirely plausible, which is what makes it dangerous."""
    frame = hourly_frame(4)
    frame["open_interest"] = [10.0, 20.0, 30.0, 40.0]

    result, _ = merge.resample_frame(frame, "open_time", Timeframe.H4, Timeframe.H1)

    assert at(result, "open_interest", "2024-01-01 00:00") == 40.0
    assert at(result, "open_interest", "2024-01-01 00:00") != frame["open_interest"].mean()


def test_resampled_bars_are_labelled_by_interval_open_not_close() -> None:
    """pandas' defaults label a bar with its close, which would shift every timestamp in
    the dataset forward by one bar and misalign it against everything fetched from a
    provider that stamps bars at their open."""
    frame = hourly_frame(8)
    result, _ = merge.resample_frame(frame, "open_time", Timeframe.H4, Timeframe.H1)

    assert times_of(result) == [
        pd.Timestamp("2024-01-01 00:00", tz="UTC"),
        pd.Timestamp("2024-01-01 04:00", tz="UTC"),
    ]


def test_an_interval_with_no_source_rows_is_dropped_rather_than_filled() -> None:
    """Filling it would invent a bar. Dropping it keeps the hole visible to the
    missing-bar detector, which is the check whose whole job is to report this."""
    frame = hourly_frame(12)
    # Destroy the whole 04:00–08:00 window.
    frame = frame.drop(index=[4, 5, 6, 7]).reset_index(drop=True)

    result, _ = merge.resample_frame(frame, "open_time", Timeframe.H4, Timeframe.H1)

    assert times_of(result) == [
        pd.Timestamp("2024-01-01 00:00", tz="UTC"),
        pd.Timestamp("2024-01-01 08:00", tz="UTC"),
    ]


def test_an_empty_interval_does_not_survive_as_a_bar_with_zero_volume() -> None:
    """`sum` over an empty group is 0, not NaN, so `dropna(how="all")` does not remove an
    empty bin: the volume column keeps the row alive and the resample emits a bar with no
    prices and a zero volume.

    Zero volume is not "no trading" — it is a fabricated observation, and worse, the
    missing-bar detector cannot report a hole that has been papered over with a bar.
    """
    frame = hourly_frame(12).drop(index=[4, 5, 6, 7]).reset_index(drop=True)

    result, _ = merge.resample_frame(frame, "open_time", Timeframe.H4, Timeframe.H1)

    assert list(times_of(result)) == [
        pd.Timestamp("2024-01-01 00:00", tz="UTC"),
        pd.Timestamp("2024-01-01 08:00", tz="UTC"),
    ]
    assert list(result["volume"]) == [
        frame["volume"].iloc[0:4].sum(),
        frame["volume"].iloc[4:8].sum(),
    ]


def test_an_interval_whose_rows_are_all_empty_is_kept() -> None:
    """The opposite mistake, and the reason an explicit occupancy count is used rather
    than `dropna(how="all")`: rows existed here, so the interval is a real observation
    with missing values. Dropping it would hide a hole from the missing-value detector,
    which is the check whose entire job is to report exactly that.
    """
    frame = hourly_frame(8)
    frame.loc[4:7, frame.columns.drop("open_time")] = float("nan")

    result, _ = merge.resample_frame(frame, "open_time", Timeframe.H4, Timeframe.H1)

    assert len(result) == 2
    assert bool(pd.isna(at(result, "close", "2024-01-01 04:00")))


def test_a_partial_trailing_interval_is_kept_and_aggregates_only_what_exists() -> None:
    """A six-hour series resampled to four hours has a final window holding two bars. It
    is a real (if incomplete) bar and dropping it would silently truncate the dataset."""
    frame = hourly_frame(6)
    result, _ = merge.resample_frame(frame, "open_time", Timeframe.H4, Timeframe.H1)

    assert len(result) == 2
    assert at(result, "volume", "2024-01-01 04:00") == frame["volume"].iloc[4:].sum()


def test_resampling_warns_when_the_source_timeframe_is_unset() -> None:
    """Without a declared source interval the up/down check cannot run, so a request that
    would fabricate bars is allowed through with a warning rather than silently."""
    _, report = merge.resample_frame(hourly_frame(8), "open_time", Timeframe.M15, None)
    assert any("declares no timeframe" in w for w in report["warnings"])


def test_resampling_refuses_an_unparseable_timestamp() -> None:
    frame = hourly_frame(8)
    frame.loc[3, "open_time"] = pd.NaT
    with pytest.raises(ValidationFailure, match="parseable timestamp"):
        merge.resample_frame(frame, "open_time", Timeframe.H4, Timeframe.H1)


def test_resampling_reports_its_row_counts() -> None:
    _, report = merge.resample_frame(hourly_frame(8), "open_time", Timeframe.H4, Timeframe.H1)
    assert report["rows_before"] == 8
    assert report["rows_after"] == 2
    assert report["timeframe"] == "4h"


# ================================================================ alignment


def test_as_of_alignment_uses_the_most_recent_value_at_or_before_the_bar() -> None:
    """The funding rate at 04:00 was not known when the 01:00 bar closed. Every bar
    before 04:00 must carry the 00:00 rate."""
    base = hourly_frame(6)
    funding = funding_frame(["2024-01-01 00:00", "2024-01-01 04:00"], [0.01, 0.02])

    merged, reports = merge.align_and_merge(
        base,
        "open_time",
        [(JoinSpec("F1", prefix="f_"), funding, "funding", "calc_time")],
    )

    assert at(merged, "f_last_funding_rate", "2024-01-01 00:00") == 0.01
    assert at(merged, "f_last_funding_rate", "2024-01-01 03:00") == 0.01
    assert at(merged, "f_last_funding_rate", "2024-01-01 04:00") == 0.02
    assert at(merged, "f_last_funding_rate", "2024-01-01 05:00") == 0.02
    assert reports[0]["unmatched"] == 0


def test_no_bar_ever_receives_a_value_from_the_future() -> None:
    """Stated as its own test because it is the failure this module exists to prevent, and
    because it is invisible in the output: a merged frame built with a forward fill looks
    exactly as well formed as a correct one.

    Here the funding series has a value at 04:00 and nothing before it. An as-of join must
    leave the first four bars empty rather than reaching forward to it.
    """
    base = hourly_frame(6)
    funding = funding_frame(["2024-01-01 04:00"], [0.02])

    merged, _ = merge.align_and_merge(
        base, "open_time", [(JoinSpec("F1", prefix="f_"), funding, "funding", "calc_time")]
    )

    for moment in ("2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 02:00", "2024-01-01 03:00"):
        assert bool(pd.isna(at(merged, "f_last_funding_rate", moment))), (
            f"the {moment} bar took a value published at 04:00"
        )
    assert at(merged, "f_last_funding_rate", "2024-01-01 04:00") == 0.02


def test_a_tolerance_excludes_values_that_are_too_stale() -> None:
    base = hourly_frame(6)
    funding = funding_frame(["2024-01-01 00:00"], [0.01])

    merged, reports = merge.align_and_merge(
        base,
        "open_time",
        [(JoinSpec("F1", prefix="f_", tolerance_seconds=7200), funding, "funding", "calc_time")],
    )

    # Ages at each bar: 0s, 1h, 2h, 3h, 4h, 5h. The tolerance is inclusive, so 02:00 still
    # matches and 03:00 does not.
    assert at(merged, "f_last_funding_rate", "2024-01-01 02:00") == 0.01
    assert bool(pd.isna(at(merged, "f_last_funding_rate", "2024-01-01 03:00")))
    assert reports[0]["unmatched"] == 3


def test_an_unmatched_bar_warning_names_how_many_bars_are_affected() -> None:
    base = hourly_frame(6)
    funding = funding_frame(["2024-01-01 04:00"], [0.02])

    _, reports = merge.align_and_merge(
        base, "open_time", [(JoinSpec("F1", prefix="f_"), funding, "funding", "calc_time")]
    )

    assert reports[0]["unmatched"] == 4
    assert any("4 of 6 base bar(s)" in w for w in reports[0]["warnings"])


def test_an_inner_join_drops_the_bars_that_found_nothing() -> None:
    """`merge_asof` is always a left join, so an inner join has to be the left join with
    the unmatched bars removed afterwards."""
    base = hourly_frame(6)
    funding = funding_frame(["2024-01-01 02:00"], [0.01])

    merged, reports = merge.align_and_merge(
        base,
        "open_time",
        [(JoinSpec("F1", prefix="f_", how=MergeHow.INNER), funding, "funding", "calc_time")],
    )

    assert times_of(merged) == [
        pd.Timestamp("2024-01-01 02:00", tz="UTC"),
        pd.Timestamp("2024-01-01 03:00", tz="UTC"),
        pd.Timestamp("2024-01-01 04:00", tz="UTC"),
        pd.Timestamp("2024-01-01 05:00", tz="UTC"),
    ]
    assert reports[0]["unmatched"] == 0
    assert reports[0]["rows"] == 4


def test_an_exact_join_keeps_only_timestamps_present_in_both() -> None:
    base = hourly_frame(6)
    funding = funding_frame(["2024-01-01 00:00", "2024-01-01 04:00"], [0.01, 0.02])

    merged, reports = merge.align_and_merge(
        base,
        "open_time",
        [
            (
                JoinSpec("F1", prefix="f_", method=AlignMethod.EXACT, how=MergeHow.INNER),
                funding,
                "funding",
                "calc_time",
            )
        ],
    )

    assert times_of(merged) == [
        pd.Timestamp("2024-01-01 00:00", tz="UTC"),
        pd.Timestamp("2024-01-01 04:00", tz="UTC"),
    ]
    assert reports[0]["unmatched"] == 0


def test_an_exact_join_reports_the_bars_it_could_not_match() -> None:
    base = hourly_frame(6)
    funding = funding_frame(["2024-01-01 00:00", "2024-01-01 04:00"], [0.01, 0.02])

    merged, reports = merge.align_and_merge(
        base,
        "open_time",
        [
            (
                JoinSpec("F1", prefix="f_", method=AlignMethod.EXACT),
                funding,
                "funding",
                "calc_time",
            )
        ],
    )

    assert len(merged) == 6, "a left exact join keeps every base bar"
    assert reports[0]["unmatched"] == 4


def test_an_exact_join_refuses_a_base_with_duplicated_timestamps() -> None:
    """Each duplicate on either side would match every duplicate on the other, and the
    merged frame would have more rows than the base — a row multiplication no one asked
    for and that no downstream check looks for."""
    base = hourly_frame(4)
    base = pd.concat([base, base.iloc[[1]]], ignore_index=True)
    funding = funding_frame(["2024-01-01 00:00"], [0.01])

    with pytest.raises(ValidationFailure, match="one row per timestamp in the base"):
        merge.align_and_merge(
            base,
            "open_time",
            [
                (
                    JoinSpec("F1", prefix="f_", method=AlignMethod.EXACT),
                    funding,
                    "funding",
                    "calc_time",
                )
            ],
        )


def test_a_right_frame_with_duplicated_timestamps_is_refused() -> None:
    """There is no single funding rate to align onto a bar if two rows claim the same
    time. Silently picking one would make the result depend on row order."""
    base = hourly_frame(4)
    funding = funding_frame(["2024-01-01 00:00", "2024-01-01 00:00"], [0.01, 0.02])

    with pytest.raises(ValidationFailure, match="sharing a timestamp"):
        merge.align_and_merge(
            base, "open_time", [(JoinSpec("F1", prefix="f_"), funding, "funding", "calc_time")]
        )


def test_a_right_frame_without_a_timestamp_column_is_refused() -> None:
    base = hourly_frame(4)
    funding = pd.DataFrame({"last_funding_rate": [0.01]})

    with pytest.raises(ValidationFailure, match="no recognisable timestamp"):
        merge.align_and_merge(
            base, "open_time", [(JoinSpec("F1"), funding, "funding", "")]
        )


def test_an_empty_right_frame_is_refused() -> None:
    """Joining it would add a column of nulls dressed up as a successful merge."""
    base = hourly_frame(4)
    empty = pd.DataFrame({"calc_time": pd.to_datetime([], utc=True), "rate": []})

    with pytest.raises(ValidationFailure, match="no rows to merge"):
        merge.align_and_merge(
            base, "open_time", [(JoinSpec("F1", prefix="f_"), empty, "funding", "calc_time")]
        )


def test_an_unparseable_right_timestamp_is_refused() -> None:
    base = hourly_frame(4)
    funding = funding_frame(["2024-01-01 00:00"], [0.01])
    funding.loc[0, "calc_time"] = pd.NaT

    with pytest.raises(ValidationFailure, match="unparseable timestamp"):
        merge.align_and_merge(
            base, "open_time", [(JoinSpec("F1", prefix="f_"), funding, "funding", "calc_time")]
        )


def test_an_unparseable_base_timestamp_is_refused() -> None:
    """`merge_asof` does not raise on an unparseable key — it sorts NaT to one end and
    matches the wrong neighbour, so every aligned value lands on the wrong bar."""
    base = hourly_frame(4)
    base.loc[2, "open_time"] = pd.NaT
    funding = funding_frame(["2024-01-01 00:00"], [0.01])

    with pytest.raises(ValidationFailure, match="unparseable timestamp in open_time"):
        merge.align_and_merge(
            base, "open_time", [(JoinSpec("F1", prefix="f_"), funding, "funding", "calc_time")]
        )


# ================================================================== collisions


def test_a_column_that_already_exists_is_refused_rather_than_overwritten() -> None:
    """Both datasets are crypto klines and both have a `close`. Letting the second one win
    would produce a frame whose meaning depends on which dataset the user picked first."""
    base = hourly_frame(4)
    other = hourly_frame(4)

    with pytest.raises(ValidationFailure, match="already exist: close"):
        merge.align_and_merge(
            base, "open_time", [(JoinSpec("F2"), other, "more klines", "open_time")]
        )


def test_the_collision_error_says_what_to_do_about_it() -> None:
    base = hourly_frame(4)
    with pytest.raises(ValidationFailure) as raised:
        merge.align_and_merge(
            base, "open_time", [(JoinSpec("F2"), hourly_frame(4), "more klines", "open_time")]
        )
    assert "prefix" in (raised.value.details.get("hint") or "")


def test_a_prefix_lets_two_datasets_keep_their_own_close_column() -> None:
    base = hourly_frame(4)
    other = hourly_frame(4)

    merged, reports = merge.align_and_merge(
        base, "open_time", [(JoinSpec("F2", prefix="eth_"), other, "eth klines", "open_time")]
    )

    assert "close" in merged.columns and "eth_close" in merged.columns
    # The base's own close is untouched.
    assert at(merged, "close", "2024-01-01 00:00") == base.loc[0, "close"]
    assert "eth_close" in reports[0]["columns_added"]


def test_a_joined_dataset_carrying_the_bases_key_name_does_not_produce_two_of_them() -> None:
    """Prefixing happens *before* the key is renamed. The other way round, a right frame
    with its own `open_time` column would end up with two columns of that name — and
    pandas tolerates duplicate labels, so the merged frame would carry a data column
    wearing the timestamp's name with nothing to distinguish them."""
    base = hourly_frame(4)
    other = funding_frame(["2024-01-01 00:00"], [0.01])
    other["open_time"] = [111]  # its own column, unrelated to the key

    merged, _ = merge.align_and_merge(
        base, "open_time", [(JoinSpec("F1", prefix="f_"), other, "funding", "calc_time")]
    )

    assert len(set(map(str, merged.columns))) == len(merged.columns), "duplicate labels"
    assert "f_open_time" in merged.columns
    # The key column holds the base's timeline, not the joined frame's stray column.
    assert times_of(merged) == times_of(base)


def test_a_stray_key_named_column_without_a_prefix_is_refused() -> None:
    base = hourly_frame(4)
    other = funding_frame(["2024-01-01 00:00"], [0.01])
    other["open_time"] = [111]

    with pytest.raises(ValidationFailure, match="two columns named open_time"):
        merge.align_and_merge(
            base, "open_time", [(JoinSpec("F1"), other, "funding", "calc_time")]
        )


def test_the_internal_match_sentinel_never_reaches_the_result_or_the_report() -> None:
    """It exists to turn "did this bar find a value?" into a null test. Leaking it would
    put a column called `__qca_matched` in the saved dataset and in the list of columns
    the merge claims to have added."""
    base = hourly_frame(4)
    funding = funding_frame(["2024-01-01 00:00"], [0.01])

    merged, reports = merge.align_and_merge(
        base, "open_time", [(JoinSpec("F1", prefix="f_"), funding, "funding", "calc_time")]
    )

    assert not any("__qca" in str(column) for column in merged.columns)
    assert not any("__qca" in name for name in reports[0]["columns_added"])


# ============================================================== frame shape


def test_an_as_of_left_join_never_changes_the_row_count() -> None:
    """Right-side timestamps are unique by construction, so every base bar matches at most
    one joined row. A row count that grew would mean a silent many-to-many join."""
    base = hourly_frame(50)
    funding = funding_frame(
        ["2024-01-01 00:00", "2024-01-01 12:00", "2024-01-02 00:00"], [0.01, 0.02, 0.03]
    )

    merged, _ = merge.align_and_merge(
        base, "open_time", [(JoinSpec("F1", prefix="f_"), funding, "funding", "calc_time")]
    )

    assert len(merged) == len(base)


def test_the_merged_frame_stays_ordered_by_time() -> None:
    """The join sorts both sides to do its work. Returning a sorted frame is what keeps
    the result consistent with every other dataset in the store, all of which are
    time-ordered by construction."""
    base = hourly_frame(20)
    funding = funding_frame(["2024-01-01 05:00", "2024-01-01 10:00"], [0.01, 0.02])

    merged, _ = merge.align_and_merge(
        base, "open_time", [(JoinSpec("F1", prefix="f_"), funding, "funding", "calc_time")]
    )

    stamps = times_of(merged)
    assert stamps == sorted(stamps)


def test_a_base_with_an_unsorted_timeline_is_still_joined_correctly() -> None:
    """The base is sorted internally, so a frame that arrived shuffled does not corrupt
    the alignment. `merge_asof` requires sorted keys and does not verify them."""
    base = hourly_frame(6).iloc[[3, 1, 5, 0, 4, 2]].reset_index(drop=True)
    funding = funding_frame(["2024-01-01 00:00", "2024-01-01 04:00"], [0.01, 0.02])

    merged, _ = merge.align_and_merge(
        base, "open_time", [(JoinSpec("F1", prefix="f_"), funding, "funding", "calc_time")]
    )

    assert at(merged, "f_last_funding_rate", "2024-01-01 03:00") == 0.01
    assert at(merged, "f_last_funding_rate", "2024-01-01 04:00") == 0.02


def test_several_joins_accumulate_onto_one_base() -> None:
    base = hourly_frame(6)
    funding = funding_frame(["2024-01-01 00:00"], [0.01])
    metrics = pd.DataFrame(
        {
            "create_time": pd.to_datetime(["2024-01-01 01:00"], utc=True),
            "sum_open_interest": [1234.0],
        }
    )

    merged, reports = merge.align_and_merge(
        base,
        "open_time",
        [
            (JoinSpec("F1", prefix="f_"), funding, "funding", "calc_time"),
            (JoinSpec("M1", prefix="m_"), metrics, "metrics", "create_time"),
        ],
    )

    assert len(reports) == 2
    assert at(merged, "f_last_funding_rate", "2024-01-01 05:00") == 0.01
    assert at(merged, "m_sum_open_interest", "2024-01-01 05:00") == 1234.0
    assert at(merged, "m_sum_open_interest", "2024-01-01 00:00") is not None
    assert bool(pd.isna(at(merged, "m_sum_open_interest", "2024-01-01 00:00")))


def test_a_third_join_cannot_silently_shadow_a_column_added_by_the_second() -> None:
    """Collisions are checked against the frame as it stands at that point in the chain,
    not against the original base. Otherwise the second dataset's `close` would be
    overwritten by the third's without a word."""
    base = hourly_frame(6)
    first = hourly_frame(6)
    second = hourly_frame(6)

    with pytest.raises(ValidationFailure, match="already exist: eth_close"):
        merge.align_and_merge(
            base,
            "open_time",
            [
                (JoinSpec("F2", prefix="eth_"), first, "eth", "open_time"),
                (JoinSpec("F3", prefix="eth_"), second, "eth again", "open_time"),
            ],
        )


def test_merging_is_deterministic() -> None:
    base = hourly_frame(40)
    funding = funding_frame(["2024-01-01 00:00", "2024-01-01 20:00"], [0.01, 0.02])
    spec = JoinSpec("F1", prefix="f_")

    first, _ = merge.align_and_merge(base, "open_time", [(spec, funding, "funding", "calc_time")])
    second, _ = merge.align_and_merge(base, "open_time", [(spec, funding, "funding", "calc_time")])

    pd.testing.assert_frame_equal(first, second)


# ==================================================== through the service layer


async def test_cleaning_writes_a_processed_child_and_records_the_fences(
    settings: Settings, catalog: Catalog
) -> None:
    """Provenance has to answer "what is this file" after the fact.

    The fences cannot be reconstructed from the cleaned values — the data they were
    computed from is gone — so a cleaned dataset that does not record them is
    unauditable, and the look-ahead in the outlier step is invisible.
    """
    frame = hourly_frame(300)
    frame.loc[150, "volume"] = 1e9
    parent = register(settings, catalog, frame)
    service = DatasetService(settings=settings, catalog=catalog)

    result = await service.clean(
        parent.dataset_id,
        CleaningPlan(outliers=OutlierPolicy.WINSORIZE, outlier_columns=("volume",)),
    )

    child = result["dataset"]
    assert child is not None
    assert child["category"] == "processed_data"
    assert child["parent_dataset_id"] == parent.dataset_id
    assert child["provenance"]["operation"] == "clean"
    assert "outlier_fence[volume]" in child["provenance"]
    assert "quantile(0.01,0.99)" in child["provenance"]["outlier_fence[volume]"]
    assert result["report"]["rows_after"] == 300
    assert len(result["preview"]["head"]) == 5
    assert len(result["preview"]["tail"]) == 5
    assert result["preview"]["row_count"] == 300


async def test_cleaning_leaves_the_parent_untouched(
    settings: Settings, catalog: Catalog
) -> None:
    """The redesign of a mutable repair into a new immutable version is the whole reason
    the cleaning step is safe to experiment with."""
    parent = register(settings, catalog, hourly_frame(100))
    service = DatasetService(settings=settings, catalog=catalog)

    await service.clean(
        parent.dataset_id,
        CleaningPlan(outliers=OutlierPolicy.WINSORIZE, outlier_columns=("volume",)),
    )

    reloaded = catalog.get_dataset(parent.dataset_id)
    assert reloaded is not None
    assert reloaded.checksum == parent.checksum


async def test_a_cleaning_preview_stores_nothing(settings: Settings, catalog: Catalog) -> None:
    """This is what makes "what would this do?" answerable before the user commits, which
    matters most for the one step that can silently discard rows."""
    parent = register(settings, catalog, hourly_frame(100))
    service = DatasetService(settings=settings, catalog=catalog)

    result = await service.clean(
        parent.dataset_id,
        CleaningPlan(outliers=OutlierPolicy.REMOVE, outlier_columns=("volume",)),
        write=False,
    )

    assert result["dataset"] is None
    assert result["report"]["rows_removed"] >= 1
    assert [row.dataset_id for row in catalog.list_datasets()] == [parent.dataset_id]


async def test_a_merge_links_the_child_to_every_input(
    settings: Settings, catalog: Catalog
) -> None:
    """A merge has several parents. `parent_dataset_id` records the base — the timeline —
    while every input is recorded in the lineage table and in provenance, so "what went
    into this file" survives the inspection."""
    base = register(settings, catalog, hourly_frame(48), name="klines")
    funding = register(
        settings, catalog, funding_frame(["2024-01-01 00:00"], [0.01]), name="funding"
    )
    service = DatasetService(settings=settings, catalog=catalog)

    result = await service.merge(
        base.dataset_id, [JoinSpec(funding.dataset_id, prefix="f_")]
    )

    child = result["dataset"]
    assert child is not None
    assert child["parent_dataset_id"] == base.dataset_id
    assert child["category"] == "processed_data"
    assert funding.dataset_id in child["provenance"]["merge_inputs"]
    assert catalog.ancestors(child["dataset_id"]) == [base.dataset_id]
    assert any("f_last_funding_rate" in c["name"] for c in child["columns"])
    assert result["report"]["rows_after"] == 48


async def test_a_merge_preview_stores_nothing(settings: Settings, catalog: Catalog) -> None:
    base = register(settings, catalog, hourly_frame(24), name="klines")
    funding = register(
        settings, catalog, funding_frame(["2024-01-01 00:00"], [0.01]), name="funding"
    )
    service = DatasetService(settings=settings, catalog=catalog)

    before = {row.dataset_id for row in catalog.list_datasets()}
    result = await service.merge(
        base.dataset_id, [JoinSpec(funding.dataset_id, prefix="f_")], write=False
    )

    assert result["dataset"] is None
    assert len(result["preview"]["head"]) == 5
    assert {row.dataset_id for row in catalog.list_datasets()} == before


async def test_merging_a_dataset_with_itself_is_refused(
    settings: Settings, catalog: Catalog
) -> None:
    """Every column would collide with itself, and the as-of join would silently produce
    a frame of duplicated values."""
    base = register(settings, catalog, hourly_frame(24), name="klines")
    service = DatasetService(settings=settings, catalog=catalog)

    with pytest.raises(ValidationFailure, match="cannot be merged with itself"):
        await service.merge(base.dataset_id, [JoinSpec(base.dataset_id, prefix="x_")])


async def test_merging_nothing_is_refused(settings: Settings, catalog: Catalog) -> None:
    """No joins *and* no timeframe is a request that asks for nothing.

    The guard is on the pair rather than on the joins alone: resampling is a supported
    operation on its own (see the test below), so an empty join list is only a defect
    when there is also no target timeframe.
    """
    base = register(settings, catalog, hourly_frame(24), name="klines")
    service = DatasetService(settings=settings, catalog=catalog)

    with pytest.raises(ValidationFailure, match="no target timeframe"):
        await service.merge(base.dataset_id, [])


async def test_a_merge_with_no_joins_resamples_the_base(
    settings: Settings, catalog: Catalog
) -> None:
    """Resampling alone is the first half of what this step is named for.

    Turning 1-minute bars into 1-hour bars is a data-preparation decision in its own
    right, so it must not require an unrelated dataset to be attached to the request.
    """
    base = register(settings, catalog, hourly_frame(24), name="klines", timeframe=Timeframe.H1)
    service = DatasetService(settings=settings, catalog=catalog)

    result = await service.merge(base.dataset_id, [], timeframe=Timeframe.H4, write=False)

    assert result["report"]["joins"] == []
    assert result["report"]["rows_after"] == 6, "24 hourly bars are 6 four-hour bars"
    assert result["dataset"] is None, "a preview stores nothing"


async def test_a_resampled_child_is_named_for_what_happened(
    settings: Settings, catalog: Catalog
) -> None:
    """A resample-only child called "(merged)" would name an input that does not exist,
    and its provenance would carry empty merge_keys that read as lost inputs."""
    base = register(settings, catalog, hourly_frame(24), name="klines", timeframe=Timeframe.H1)
    service = DatasetService(settings=settings, catalog=catalog)

    result = await service.merge(base.dataset_id, [], timeframe=Timeframe.H4)

    child = result["dataset"]
    assert child is not None
    assert child["name"] == "klines (resampled)"
    assert child["category"] == "processed_data"
    assert child["parent_dataset_id"] == base.dataset_id
    assert "merge_inputs" not in child["provenance"]


async def test_a_merge_resamples_the_base_to_the_requested_timeframe(
    settings: Settings, catalog: Catalog
) -> None:
    base = register(settings, catalog, hourly_frame(24), name="klines", timeframe=Timeframe.H1)
    funding = register(
        settings, catalog, funding_frame(["2024-01-01 00:00"], [0.01]), name="funding"
    )
    service = DatasetService(settings=settings, catalog=catalog)

    result = await service.merge(
        base.dataset_id,
        [JoinSpec(funding.dataset_id, prefix="f_")],
        timeframe=Timeframe.H4,
    )

    assert result["report"]["resample"]["rows_after"] == 6
    assert result["dataset"] is not None
    assert result["dataset"]["timeframe"] == "4h"
    assert result["preview"]["row_count"] == 6


async def test_a_merge_can_rename_and_drop_columns_before_saving(
    settings: Settings, catalog: Catalog
) -> None:
    """The column vocabulary is the grid's own, so "remove this column" means the same
    thing in the merge step as it does in Data Management."""
    from app.services.datasets import ColumnOp

    base = register(settings, catalog, hourly_frame(24), name="klines")
    funding = register(
        settings, catalog, funding_frame(["2024-01-01 00:00"], [0.01]), name="funding"
    )
    service = DatasetService(settings=settings, catalog=catalog)

    result = await service.merge(
        base.dataset_id,
        [JoinSpec(funding.dataset_id, prefix="f_")],
        column_ops=[
            ColumnOp(op="drop", column="close_time"),
            ColumnOp(op="rename", column="f_last_funding_rate", new_name="funding"),
        ],
    )

    names = [column["name"] for column in result["dataset"]["columns"]]
    assert "close_time" not in names
    assert "funding" in names
    assert "f_last_funding_rate" not in names
    assert result["report"]["column_ops"] == ["drop", "rename"]


async def test_a_merge_of_an_already_merged_dataset_records_the_full_chain(
    settings: Settings, catalog: Catalog
) -> None:
    base = register(settings, catalog, hourly_frame(24), name="klines")
    funding = register(
        settings, catalog, funding_frame(["2024-01-01 00:00"], [0.01]), name="funding"
    )
    service = DatasetService(settings=settings, catalog=catalog)

    first = await service.merge(base.dataset_id, [JoinSpec(funding.dataset_id, prefix="f_")])
    assert first["dataset"] is not None
    second = await service.merge(
        first["dataset"]["dataset_id"], [JoinSpec(funding.dataset_id, prefix="g_")]
    )

    assert second["dataset"] is not None
    assert catalog.ancestors(second["dataset"]["dataset_id"]) == [
        first["dataset"]["dataset_id"],
        base.dataset_id,
    ]


def test_resample_then_align_matches_aligning_the_finer_source_directly() -> None:
    """Resampling the base and then as-of joining must give the same answer as as-of
    joining the fine-grained source onto the original bars and grouping afterwards.

    This is why only the base is resampled: the as-of rule already collapses a finer
    series to one value per bar, so resampling the joined side too would add a second
    place for the aggregation rule to be wrong without changing the result.
    """
    base = hourly_frame(24)
    funding = funding_frame(
        [f"2024-01-01 {hour:02d}:00" for hour in range(0, 24, 4)],
        [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
    )
    spec = JoinSpec("F1", prefix="f_")

    coarse, _ = merge.resample_frame(base, "open_time", Timeframe.H4, Timeframe.H1)
    coarsened, _ = merge.align_and_merge(
        coarse, "open_time", [(spec, funding, "funding", "calc_time")]
    )

    fine, _ = merge.align_and_merge(
        base, "open_time", [(spec, funding, "funding", "calc_time")]
    )
    # The 4h bar opening at 04:00 carries the same funding value in both routes.
    for moment in ("2024-01-01 00:00", "2024-01-01 04:00", "2024-01-01 08:00"):
        assert at(coarsened, "f_last_funding_rate", moment) == at(
            fine, "f_last_funding_rate", moment
        )


def test_resampled_ohlc_is_consistent_with_the_underlying_bars() -> None:
    """A structural check on the aggregate: the 4h high can never be below the 4h open or
    close, because it is the maximum of hourly highs that all bound their own bars."""
    frame = hourly_frame(96)
    result, _ = merge.resample_frame(frame, "open_time", Timeframe.H4, Timeframe.H1)

    assert (result["high"] >= result[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (result["low"] <= result[["open", "close"]].min(axis=1) + 1e-9).all()
    assert np.isfinite(result["volume"]).all()
