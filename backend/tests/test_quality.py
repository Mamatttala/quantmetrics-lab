"""Quality detectors.

Two things are being tested here beyond "does it find the defect": that a clean
dataset is reported clean (no false positives), and that a check which *cannot* run
says so rather than returning zero. "0 duplicate timestamps" and "duplicates were not
checked" are different claims, and conflating them would let a broken dataset pass.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.config import Settings
from app.models.enums import AssetClass, Timeframe
from app.services import quality
from app.storage.catalog import Catalog
from tests.conftest import hourly_frame, register


def counts(report: dict) -> dict:
    return report["counts"]


def test_time_resolution_prefers_the_first_exact_candidate() -> None:
    assert quality.find_time_column_in(["Date", "date"]) == "Date"


def test_ohlcv_resolution_accepts_one_case_insensitive_alias() -> None:
    assert quality.resolve_ohlcv(["OPEN"]) == {"open": "OPEN"}


def test_ohlcv_resolution_rejects_case_collisions() -> None:
    with pytest.raises(ValueError, match="Ambiguous column resolution for 'open'") as error:
        quality.resolve_ohlcv(["Open", "open"])

    assert "found ['Open', 'open']" in str(error.value)
    assert "Please rename to disambiguate." in str(error.value)


def test_ohlcv_resolution_maps_capitalised_equity_columns() -> None:
    names = ["Date", "Open", "High", "Low", "Close", "Volume"]
    assert quality.resolve_ohlcv(names) == {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }


def test_a_clean_series_produces_no_markers() -> None:
    """A count that is present and zero means "checked, nothing found". An absent count
    means "not checked". Both are reported, so this asserts the distinction rather than
    an empty dict — the frontend renders the two cases differently."""
    report = quality.analyse(hourly_frame(300), Timeframe.H1)
    assert report["markers"] == []
    assert counts(report) == {
        "duplicate_timestamp": 0,
        "timestamp_out_of_order": 0,
        "missing_value": 0,
        "invalid_dtype": 0,
        "invalid_numeric": 0,
        "missing_bar": 0,
        "non_positive_price": 0,
        "high_below_low": 0,
        "extreme_return": 0,
    }
    assert report["skipped"] == []
    assert report["notes"] == []
    assert report["time_column"] == "open_time"


def test_out_of_order_timestamps_are_found() -> None:
    """A frame whose clock goes backwards is not merely untidy: every return, lag, and
    rolling window computed on it reads the past as if it were the future."""
    frame = hourly_frame(10)
    frame.loc[6, "open_time"] = frame.loc[2, "open_time"]
    report = quality.analyse(frame, Timeframe.H1)

    assert counts(report)["timestamp_out_of_order"] >= 1
    flagged = [m for m in report["markers"] if m["kind"] == "timestamp_out_of_order"]
    # The marker sits on the row that steps back, not on the row it stepped behind.
    assert flagged[0]["time"] == int(frame.loc[6, "open_time"].timestamp())


def test_an_ordered_series_reports_no_order_violations() -> None:
    report = quality.analyse(hourly_frame(50), Timeframe.H1)
    assert counts(report)["timestamp_out_of_order"] == 0


def test_missing_values_are_counted_per_row_and_per_column() -> None:
    frame = hourly_frame(10)
    frame.loc[3, "close"] = float("nan")
    frame.loc[3, "volume"] = float("nan")
    frame.loc[7, "low"] = float("nan")

    report = quality.analyse(frame, Timeframe.H1)

    assert counts(report)["missing_value"] == 2, "two rows carry a hole, not three cells"
    detail = next(m for m in report["markers"] if m["kind"] == "missing_value")
    assert "2 empty cell(s)" in detail["detail"]
    per_column = {c["name"]: c["missing"] for c in report["columns"]}
    assert per_column["close"] == 1
    assert per_column["volume"] == 1
    assert per_column["low"] == 1
    assert per_column["open"] == 0


def test_a_text_column_contaminated_by_non_numbers_is_reported() -> None:
    """The CSV-upload case: a numeric column whose values arrived as text, with a few
    entries that are not numbers at all."""
    frame = hourly_frame(10)
    frame["oi_change"] = ["1.5", "2.0", "n/a", "3.0", "4.0", "5.0", "6.0", "7.0", "8.0", "9.0"]
    report = quality.analyse(frame, Timeframe.H1)

    assert counts(report)["invalid_numeric"] == 1
    # A finding, not a skip: the check ran and this is what it counted. Reporting it
    # under `skipped` made the panel say "not checked" about a check that had just
    # produced a number.
    assert any("oi_change" in note for note in report["notes"])
    assert not any("oi_change" in reason for reason in report["skipped"])


def test_a_genuine_text_column_is_not_reported_as_invalid() -> None:
    """A column of labels is mostly non-numeric, which is what keeps it out of the
    invalid-numeric count. Flagging it would make the check useless on any dataset with
    a notes or symbol column."""
    frame = hourly_frame(10)
    frame["venue"] = ["binance", "coinbase", "kraken", "okx", "bybit"] * 2
    report = quality.analyse(frame, Timeframe.H1)

    assert counts(report)["invalid_numeric"] == 0
    assert counts(report)["invalid_dtype"] == 0


def test_a_binary_column_is_reported_as_an_unsupported_type() -> None:
    """Not a corrupt value — a valid Parquet type this application cannot build a
    feature from, which is what "invalid data type" has to mean for a canonical
    Parquet store."""
    frame = hourly_frame(10)
    frame["blob"] = [b"\x00\x01"] * 10
    report = quality.analyse(frame, Timeframe.H1)

    assert counts(report)["invalid_dtype"] == 1
    assert any("blob" in reason for reason in report["skipped"])


def test_duplicate_timestamps_are_found() -> None:
    frame = hourly_frame(10)
    frame.loc[5, "open_time"] = frame.loc[4, "open_time"]
    report = quality.analyse(frame, Timeframe.H1)
    assert counts(report)["duplicate_timestamp"] == 1
    kinds = {m["kind"] for m in report["markers"]}
    assert "duplicate_timestamp" in kinds


def test_a_gap_is_reported_with_its_size() -> None:
    frame = hourly_frame(10)
    # Drop three rows, creating a four-hour hole between bar 3 and bar 4.
    frame = frame.drop(index=[4, 5, 6]).reset_index(drop=True)
    report = quality.analyse(frame, Timeframe.H1)
    assert counts(report)["missing_bar"] == 1
    detail = next(m["detail"] for m in report["markers"] if m["kind"] == "missing_bar")
    assert "3 bar(s)" in detail


def test_missing_bar_check_is_skipped_when_the_timeframe_is_unset() -> None:
    """Without a declared timeframe there is no expected spacing, so the check cannot
    run. Reporting zero gaps would be a false clean bill of health."""
    report = quality.analyse(hourly_frame(10), None)
    assert "missing_bar" not in counts(report)
    assert any("timeframe is unset" in reason for reason in report["skipped"])


def test_non_positive_prices_are_found() -> None:
    frame = hourly_frame(10)
    frame.loc[3, "close"] = 0.0
    frame.loc[7, "low"] = -1.0
    report = quality.analyse(frame, Timeframe.H1)
    assert counts(report)["non_positive_price"] == 2


def test_high_below_low_is_found() -> None:
    frame = hourly_frame(10)
    frame.loc[2, "high"] = frame.loc[2, "low"] - 1.0
    report = quality.analyse(frame, Timeframe.H1)
    assert counts(report)["high_below_low"] == 1


def test_extreme_return_is_found_by_robust_z_score() -> None:
    """A single 50-sigma shock. Standard deviation would be inflated by the very
    observation being looked for, which is why MAD is used."""
    frame = hourly_frame(400)
    frame.loc[200, "close"] = frame.loc[199, "close"] * 1.5
    report = quality.analyse(frame, Timeframe.H1)
    assert counts(report)["extreme_return"] >= 1
    # The bar *after* the jump carries the return, so it is the one flagged.
    flagged = [m for m in report["markers"] if m["kind"] == "extreme_return"]
    assert flagged[0]["time"] == int(frame.loc[200, "open_time"].timestamp())


def test_extreme_return_check_is_skipped_below_the_minimum_sample() -> None:
    report = quality.analyse(hourly_frame(10), Timeframe.H1)
    assert "extreme_return" not in counts(report)
    assert any("at least" in reason for reason in report["skipped"])


def test_degenerate_returns_are_reported_rather_than_silently_clean() -> None:
    """A constant price series has MAD = 0. The robust scale is undefined, so no
    threshold exists; saying "not checked" is the honest answer."""
    frame = hourly_frame(100)
    frame["close"] = 100.0
    frame["open"] = 100.0
    frame["high"] = 100.0
    frame["low"] = 100.0
    report = quality.analyse(frame, Timeframe.H1)
    assert "extreme_return" not in counts(report)
    assert any("degenerate" in reason for reason in report["skipped"])


def test_an_unparseable_timestamp_does_not_crash_and_is_not_dated() -> None:
    """`NaT.astype("int64")` yields the int64 sentinel, not NaN.

    Left unhandled it became a real-looking timestamp in 1677; left unguarded the
    `int()` conversion raised. Either way, a defective row with a broken clock is
    exactly what these checks exist to surface, so it must be reported as undated
    rather than dropped or misplaced.
    """
    frame = hourly_frame(100)
    frame.loc[5, "open_time"] = pd.NaT
    frame.loc[5, "close"] = -1.0  # also a defect, on the row with the broken clock

    report = quality.analyse(frame, Timeframe.H1)

    assert counts(report)["non_positive_price"] == 1
    undated = [m for m in report["markers"] if m["kind"] == "non_positive_price"]
    assert undated[0]["time"] is None


def test_markers_are_sorted_by_time_with_undated_last() -> None:
    frame = hourly_frame(100)
    frame.loc[90, "close"] = -1.0
    frame.loc[5, "open_time"] = pd.NaT
    frame.loc[5, "low"] = -2.0

    report = quality.analyse(frame, Timeframe.H1)
    times = [m["time"] for m in report["markers"]]

    dated = [t for t in times if t is not None]
    assert dated == sorted(dated)
    # Undated markers go last: at the top of the list they would read as the oldest
    # events rather than as events with no known date.
    assert times[-1] is None


def test_a_non_default_index_does_not_misalign_markers() -> None:
    """Datasets can arrive with any index — a CSV upload, a filtered frame.

    Indexing positional timestamp arrays with a label would read the wrong row, and
    the resulting marker would point confidently at an unrelated bar.
    """
    frame = hourly_frame(120)
    frame.index = np.arange(1000, 1120)
    frame.loc[1005, "close"] = -1.0

    report = quality.analyse(frame, Timeframe.H1)
    flagged = [m for m in report["markers"] if m["kind"] == "non_positive_price"]
    assert len(flagged) == 1
    assert flagged[0]["time"] == int(frame.loc[1005, "open_time"].timestamp())


def test_a_frame_with_no_timestamp_column_reports_everything_as_skipped() -> None:
    frame = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
    report = quality.analyse(frame, Timeframe.H1)
    assert report["markers"] == []
    assert report["time_column"] == ""
    assert report["skipped"]


def test_a_frame_with_no_price_columns_skips_price_checks() -> None:
    frame = pd.DataFrame(
        {"open_time": pd.date_range("2024-01-01", periods=60, freq="1h", tz="UTC")}
    )
    report = quality.analyse(frame, Timeframe.H1)
    assert "non_positive_price" not in counts(report)
    assert any("no OHLC columns" in reason for reason in report["skipped"])


def test_marker_time_is_unix_seconds_not_milliseconds() -> None:
    """lightweight-charts takes seconds. Milliseconds would place every marker in the
    year 57000 and off the visible axis entirely."""
    frame = hourly_frame(10)
    frame.loc[0, "close"] = 0.0
    report = quality.analyse(frame, Timeframe.H1)
    stamp = report["markers"][0]["time"]
    assert stamp is not None
    assert stamp == pytest.approx(1704067200, abs=1), "2024-01-01T00:00:00Z"


def test_analyse_is_deterministic() -> None:
    frame = hourly_frame(300)
    frame.loc[150, "close"] = frame.loc[149, "close"] * 2
    first = quality.analyse(frame, Timeframe.H1)
    second = quality.analyse(frame, Timeframe.H1)
    assert first == second


def test_missing_bar_check_is_skipped_for_equity_instruments() -> None:
    """An equity daily series has no weekend bars, and an intraday one has no
    overnight bars.

    Judging those against a single bar length would flag every Friday-to-Monday jump
    as a defect — a couple of hundred false "missing bar" markers on a two-year daily
    series, each indistinguishable from a real hole. Telling a market closure from a
    gap needs a trading calendar this application does not model, so the check reports
    itself as not run rather than inventing one.
    """
    frame = hourly_frame(300)
    # A three-day hole, which *is* a defect for crypto and would be found there.
    frame = frame.drop(index=list(range(100, 172))).reset_index(drop=True)

    equity = quality.analyse(frame, Timeframe.H1, asset_class=AssetClass.EQUITY_SPOT)
    assert "missing_bar" not in counts(equity)
    assert any("session calendar" in reason for reason in equity["skipped"])

    # Same frame, continuously-traded market: the hole is reported.
    crypto = quality.analyse(frame, Timeframe.H1, asset_class=AssetClass.CRYPTO_SPOT)
    assert counts(crypto)["missing_bar"] == 1


def test_missing_bar_check_reports_itself_when_there_are_too_few_rows() -> None:
    """Two rows cannot show a gap, and a count of zero would claim they were checked."""
    report = quality.analyse(hourly_frame(2), Timeframe.H1, asset_class=AssetClass.CRYPTO_SPOT)
    assert "missing_bar" not in counts(report)
    assert any("more than 2 dated rows" in reason for reason in report["skipped"])


async def test_the_quality_service_passes_the_asset_class_through(
    settings: Settings, catalog: Catalog
) -> None:
    """The continuous/equity distinction lives in `analyse`, so the service has to hand
    it the asset class. Without that the parameter defaults to `None`, every dataset is
    treated as continuously traded, and an equity series reports a defect per session
    boundary — a bug that would only be visible on real equity data, not in the unit
    tests above.
    """
    from app.services.datasets import DatasetService

    frame = hourly_frame(300)
    frame = frame.drop(index=list(range(100, 172))).reset_index(drop=True)
    equity = register(settings, catalog, frame, asset_class=AssetClass.EQUITY_SPOT)
    crypto = register(settings, catalog, frame, asset_class=AssetClass.CRYPTO_SPOT)

    service = DatasetService(settings=settings, catalog=catalog)
    equity_report = await service.quality(equity.dataset_id)
    crypto_report = await service.quality(crypto.dataset_id)

    assert "missing_bar" not in equity_report["counts"]
    assert crypto_report["counts"]["missing_bar"] == 1
