"""Data-quality checks.

These are computed here rather than in the browser because the specification is
explicit that the frontend must not implement statistical mathematics, and because a
MAD-based outlier score is exactly the kind of thing that must be identical between
two runs of the app.

Detectors are deliberately conservative: each one flags a condition that is
unambiguously wrong or unambiguously unusual, and the UI presents them as markers to
inspect, never as rows to delete. Silently dropping data is how a research pipeline
loses the very observations that matter.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from app.models.enums import TIMEFRAME_SECONDS, AssetClass, Timeframe

#: Robust z-score beyond which a return is marked extreme. 8 is deliberately far out:
#: on a normal sample this flags roughly one observation in a trillion, so a hit means
#: either a genuine shock or a data error, and is worth looking at either way. A
#: threshold like 3 would mark several percent of every real series and be useless.
EXTREME_RETURN_Z = 8.0

#: Minimum observations before MAD-based scoring is meaningful. Below this the
#: estimate is dominated by noise, so the check is skipped rather than reported.
MIN_OBS_FOR_Z = 50

#: Markets whose instruments trade without sessions. The missing-bar detector is only
#: valid for these — see `analyse` for why an equity frame cannot use it.
CONTINUOUS_MARKETS = frozenset({AssetClass.CRYPTO_SPOT, AssetClass.CRYPTO_FUTURES})


#: Types the rest of the application can compute on, chart, or edit. A column outside
#: this set is not an error in the file — Parquet carries many valid types — but it is
#: not something a feature matrix or a candlestick can be built from, and saying so is
#: more useful than silently ignoring it.
def _is_supported_dtype(series: pd.Series) -> bool:
    dtype = series.dtype
    if (
        pd.api.types.is_numeric_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
        or pd.api.types.is_datetime64_any_dtype(dtype)
    ):
        return True
    if pd.api.types.is_object_dtype(dtype):
        # `object` is pandas' escape hatch, and Arrow maps strings, binary blobs,
        # decimals, dates, and nested lists all onto it. The dtype alone therefore
        # cannot tell a text column from a binary one, so the values are inspected.
        return all(isinstance(value, str) for value in series.dropna())
    return False


#: The column names this application will accept as "when was this row observed", in
#: priority order. A module-level constant rather than a literal inside the function below
#: because C6 has to resolve the same column from the catalog *before* it reads the file —
#: a split needs one column from a five-million-row Parquet, not the whole frame — and two
#: copies of this list would eventually disagree about which column is time.
TIME_COLUMN_CANDIDATES: tuple[str, ...] = (
    "open_time",
    "create_time",
    "calc_time",
    "Date",
    "date",
    "timestamp",
)


def find_time_column(frame: pd.DataFrame) -> str | None:
    """First column that looks like the observation timestamp, or None.

    Public because the cleaning and merge steps must agree with the quality checks
    about *which* column is time. Two implementations would eventually disagree, and
    the failure would be a gap detector measuring one column while a resample grouped
    by another.
    """
    return find_time_column_in(frame.columns)


def find_time_column_in(names: pd.Index | list[str] | tuple[str, ...]) -> str | None:
    """`find_time_column` for a bare list of column names.

    The candlestick viewer reads its schema from the catalog before it opens the
    Parquet file, so it has names but no frame. Exact names are checked first in the
    declared priority order. A case-insensitive fallback is accepted only when one
    column matches; silently choosing one side of a case collision would make the
    timeline depend on dictionary insertion order.
    """
    return _resolve_column_name(names, TIME_COLUMN_CANDIDATES)


#: The OHLCV vocabulary, mapped to the fields the candlestick viewer and the price
#: checks ask for. Equity frames from Yahoo arrive capitalised (`Open`, `Close`, ...),
#: so an exact lower-case membership test reports "no OHLC columns" for a file that
#: plainly has them.
OHLCV_FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


def resolve_ohlcv(names: pd.Index | list[str] | tuple[str, ...]) -> dict[str, str]:
    """Map the lower-case OHLCV field names onto the dataset's actual column names.

    Returns only the fields that are present. Case-insensitive aliases such as `Open`
    are accepted when unambiguous, but a field with both `Open` and `open` is refused
    instead of allowing the later dictionary entry to win.
    """
    resolved: dict[str, str] = {}
    for field in OHLCV_FIELDS:
        # OHLCV fields have one canonical lower-case candidate rather than the explicit
        # `Date`/`date` priority pair. Therefore an exact `open` does not disambiguate a
        # second `Open` alias: both names still claim the same semantic field.
        actual = _resolve_column_name(names, (field,), detect_exact_collisions=True)
        if actual is not None:
            resolved[field] = actual
    return resolved


def _resolve_column_name(
    names: pd.Index | list[str] | tuple[str, ...],
    candidates: tuple[str, ...],
    *,
    detect_exact_collisions: bool = False,
) -> str | None:
    """Resolve one semantic column with exact priority and safe aliases.

    The exact pass is separate from the fallback pass on purpose. For timestamps the
    candidate tuple contains both `Date` and `date`, so a frame carrying both must use
    the first exact candidate rather than being rejected as an alias collision. OHLCV
    has only one canonical candidate per field; its caller enables collision detection
    even when that canonical spelling is present beside a differently-cased alias.
    """
    actual_names = [str(name) for name in names]

    for candidate in candidates:
        exact_matches = [name for name in actual_names if name == candidate]
        if exact_matches:
            if detect_exact_collisions:
                aliases = [name for name in actual_names if name.lower() == candidate.lower()]
                if len(aliases) > 1:
                    raise ValueError(
                        f"Ambiguous column resolution for '{candidate}': found {aliases}. "
                        "Please rename to disambiguate."
                    )
            return exact_matches[0]

    for candidate in candidates:
        matches = [name for name in actual_names if name.lower() == candidate.lower()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous column resolution for '{candidate}': found {matches}. "
                "Please rename to disambiguate."
            )
    return None


def has_intraday_pattern(frame: pd.DataFrame, time_column: str | None) -> bool | None:
    """Whether the bars fall at more than one time of day, read off the timestamps.

    ``True`` means the data records a clock time, ``False`` means every bar sits at the
    same time of day, and ``None`` means the question could not be answered — there is no
    usable time column. Availability asks this of the data rather than of the declared
    timeframe, because an uploaded file declares none, and a feature that is genuinely
    well defined in such a file should not be greyed out on a technicality.
    """
    if time_column is None or time_column not in frame.columns:
        return None
    try:
        index = pd.DatetimeIndex(pd.to_datetime(frame[time_column]))
    except (TypeError, ValueError):
        return None
    # `pd.to_datetime` maps what it cannot parse to NaT rather than raising, so an
    # unparseable column arrives here as an all-NaT index rather than as an exception.
    if index.empty or not bool(np.asarray(index.notna()).any()):
        return None
    times = index[index.notna()]
    seconds_of_day = times.hour * 3600 + times.minute * 60 + times.second
    return bool(pd.Index(seconds_of_day).nunique() > 1)


def _column_report(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Per-column view of what a cleaning step would be working with."""
    return [
        {
            "name": str(column),
            "dtype": str(frame[column].dtype),
            "missing": int(frame[column].isna().sum()),
            "supported": _is_supported_dtype(frame[column]),
        }
        for column in frame.columns
    ]


def _time_seconds(series: pd.Series) -> np.ndarray:
    """Convert a datetime column to UNIX seconds, with NaN for unusable values.

    Elapsed-since-epoch subtraction is used rather than `astype("int64")` for a
    specific reason: `NaT.astype("int64")` does not produce NaN, it produces the int64
    sentinel (-9223372036854775808), which divides down to a plausible-looking
    timestamp in 1677. Subtraction propagates NaT to NaN directly, so an unparseable
    row is reported as undated rather than as having occurred three centuries ago.
    """
    stamps = pd.to_datetime(series, utc=True, errors="coerce")
    elapsed = stamps - pd.Timestamp("1970-01-01", tz="UTC")
    # `to_numpy` before `floor`: `np.floor` is typed as returning an ndarray, so calling
    # `.to_numpy` on its result does not typecheck — and the conversion has to happen
    # somewhere, because dividing a timedelta Series yields a timedelta, not a number.
    seconds = (elapsed / pd.Timedelta(seconds=1)).to_numpy(dtype="float64")
    return np.floor(seconds)


def _epoch(value: Any) -> int | None:
    """UNIX seconds for a marker, or None when the row has no usable timestamp.

    Returns None rather than raising or substituting a zero: a row with an
    unparseable timestamp is exactly the kind of row these checks exist to surface,
    and stamping it 1970-01-01 would plant a marker at the start of the chart that
    corresponds to nothing.
    """
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return int(number)


def analyse(
    frame: pd.DataFrame,
    timeframe: Timeframe | None,
    *,
    asset_class: AssetClass | None = None,
) -> dict[str, Any]:
    """Run every applicable check and return markers plus per-kind counts.

    `asset_class` decides whether the missing-bar detector is applicable at all.
    Passing `None` means "assume continuous trading", which is what the detector's
    callers did before the parameter existed.
    """
    markers: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    skipped: list[str] = []
    notes: list[str] = []
    unsupported: list[str] = []
    unparsed: list[str] = []

    time_column = find_time_column(frame)
    if time_column is None:
        return {
            "time_column": "",
            "markers": [],
            "counts": {},
            "skipped": ["All checks: no recognisable timestamp column."],
            "notes": [],
            "columns": _column_report(frame),
        }

    times = _time_seconds(frame[time_column])
    valid_time = ~np.isnan(times)
    valid_positions = np.nonzero(valid_time)[0]

    # ---------------------------------------------------------- duplicates
    if valid_time.any():
        seen: set[int] = set()
        duplicate_positions: list[int] = []
        for position, stamp in enumerate(times):
            if np.isnan(stamp):
                continue
            value = int(stamp)
            if value in seen:
                duplicate_positions.append(position)
            else:
                seen.add(value)
        for position in duplicate_positions[:200]:
            markers.append(
                {
                    "time": _epoch(times[position]),
                    "kind": "duplicate_timestamp",
                    "detail": "More than one row shares this timestamp.",
                }
            )
        counts["duplicate_timestamp"] = len(duplicate_positions)
    else:
        skipped.append("Duplicate timestamps: no parseable timestamps.")

    # ------------------------------------------------------- timestamp order
    dated = times[valid_time]
    if dated.size > 1:
        # Rows, not values: `np.diff` on the dated subsequence finds where time steps
        # backwards, and the marker is placed on the row that goes back — the row that
        # would be read as an earlier observation than the one above it.
        backwards = np.nonzero(np.diff(dated) < 0)[0]
        counts["timestamp_out_of_order"] = int(backwards.size)
        for index in backwards[:200]:
            position = int(valid_positions[index + 1])
            markers.append(
                {
                    "time": _epoch(times[position]),
                    "kind": "timestamp_out_of_order",
                    "detail": (
                        "This row's timestamp is earlier than the row before it. Every "
                        "lag, return, and rolling window computed on this frame reads "
                        "the past as if it were the future."
                    ),
                }
            )
    else:
        skipped.append("Timestamp order: needs more than one dated row.")

    # ------------------------------------------------------- missing values
    if len(frame.columns):
        nulls = frame.isna()
        per_row = nulls.sum(axis=1).to_numpy()
        counts["missing_value"] = int((per_row > 0).sum())
        for position in np.nonzero(per_row > 0)[0][:200]:
            markers.append(
                {
                    "time": _epoch(times[position]),
                    "kind": "missing_value",
                    "detail": f"{int(per_row[position])} empty cell(s) in this row.",
                }
            )

    # ---------------------------------------------------------- column types
    #
    # "Invalid data types" and "invalid numeric values" are interpreted against what
    # this application can actually compute on. A Parquet column is well formed by
    # construction, so the interesting failure is not a corrupt cell but a column whose
    # *type* is outside the vocabulary the rest of the app understands, and a text
    # column that is carrying numbers with a few stragglers in it — which is what a CSV
    # upload with an "n/a" in a numeric column produces.
    #
    # Both counts are initialised to zero rather than left absent. This module's
    # contract is that a count of zero means "checked, nothing found" and an absent
    # count means "not checked"; a key that only appears when it is non-zero would
    # collapse the two at exactly the moment someone is reading the report.
    counts["invalid_dtype"] = 0
    counts["invalid_numeric"] = 0
    for column in frame.columns:
        series = frame[column]
        dtype = series.dtype
        name = str(column)
        if not _is_supported_dtype(series):
            counts["invalid_dtype"] += 1
            unsupported.append(f"{name} ({dtype})")
            continue
        if pd.api.types.is_object_dtype(dtype):
            non_null = series.dropna()
            if len(non_null):
                converted = pd.to_numeric(non_null, errors="coerce")
                parsed = int(converted.notna().sum())
                # A column that is *mostly* numeric is a numeric column with junk in it;
                # one that is mostly text is a text column. Only the first is a defect,
                # and requiring a majority is what keeps a genuine notes column from
                # being reported as broken.
                if 0 < parsed < len(non_null) and parsed >= len(non_null) / 2:
                    bad = int(len(non_null) - parsed)
                    counts["invalid_numeric"] += bad
                    unparsed.append(f"{name} ({bad} of {len(non_null)})")

    if unsupported:
        skipped.append(
            "Column types: "
            + ", ".join(unsupported)
            + " is outside the numeric/boolean/timestamp/text vocabulary this "
            "application computes on."
        )
    if unparsed:
        # A *note*, not a skip. Both are sentences the user has to read, but they are
        # different claims and the UI labels them differently: `skipped` means "this
        # check did not run", and the numeric-values check did run — it counted the
        # values that are not numbers, and that count is in `invalid_numeric`. Filed
        # under `skipped` the panel read "not checked: Numeric values: level (2 of 12)"
        # — a report contradicting itself on the same line, which is exactly the kind
        # of sentence that teaches a user to stop reading the caveats.
        notes.append(
            "Numeric values: "
            + ", ".join(unparsed)
            + " — these columns are stored as text and a minority of their values are "
            "not numbers."
        )

    # ------------------------------------------------------------ missing bars
    #
    # Only meaningful where the instrument trades continuously. An equity frame has a
    # session calendar: a daily series has no Saturday or Sunday bar, and an intraday
    # series has no overnight bars. Comparing consecutive timestamps against one bar
    # length would therefore flag every weekend and every close as a defect — a couple
    # of hundred false "missing bar" markers on a two-year daily series, each of which
    # looks exactly like a genuine hole. Distinguishing the two needs a trading
    # calendar this application does not have, and inventing one (a weekday filter, a
    # hard-coded holiday list) would be a fabricated methodology that silently
    # misclassifies the holidays it does not know about. The check is reported as not
    # run instead, with the reason, which is the honest answer.
    if asset_class is not None and asset_class not in CONTINUOUS_MARKETS:
        skipped.append(
            "Missing bars: no exchange session calendar is modelled, so a market "
            "closure cannot be told apart from a gap in the data. Checked on "
            "continuously-traded instruments only."
        )
    elif timeframe is None:
        skipped.append(
            "Missing bars: dataset timeframe is unset, so bar spacing cannot be checked."
        )
    elif int(valid_time.sum()) <= 2:
        skipped.append(
            "Missing bars: needs more than 2 dated rows, dataset has "
            f"{int(valid_time.sum())}."
        )
    else:
        expected = TIMEFRAME_SECONDS[timeframe]
        ordered = np.sort(times[valid_time])
        deltas = np.diff(ordered)
        gaps = np.nonzero(deltas > expected * 1.5)[0]
        for index in gaps[:200]:
            missing = int(deltas[index] // expected) - 1
            markers.append(
                {
                    # The marker sits on the last bar *before* the hole, so the absent
                    # bars come after it.
                    "time": _epoch(ordered[index]),
                    "kind": "missing_bar",
                    "detail": f"{missing} bar(s) absent after this point.",
                }
            )
        counts["missing_bar"] = int(len(gaps))

    # ------------------------------------------------------- price sanity
    # Resolved case-insensitively: Yahoo equity frames are stored with `Open`,
    # `High`, `Low` and `Close`, and an exact lower-case test reported "no OHLC
    # columns present" for a file that plainly has them.
    ohlcv = resolve_ohlcv(frame.columns)
    price_columns = [ohlcv[f] for f in ("open", "high", "low", "close") if f in ohlcv]
    if price_columns:
        prices = frame[price_columns].apply(pd.to_numeric, errors="coerce")
        non_positive = (prices <= 0).any(axis=1)
        for position in np.nonzero(non_positive.to_numpy())[0][:200]:
            markers.append(
                {
                    "time": _epoch(times[position]),
                    "kind": "non_positive_price",
                    "detail": "A price is zero or negative.",
                }
            )
        counts["non_positive_price"] = int(non_positive.sum())

        if "high" in ohlcv and "low" in ohlcv:
            inverted = prices[ohlcv["high"]] < prices[ohlcv["low"]]
            for position in np.nonzero(inverted.to_numpy())[0][:200]:
                markers.append(
                    {
                        "time": _epoch(times[position]),
                        "kind": "high_below_low",
                        "detail": "high is below low.",
                    }
                )
            counts["high_below_low"] = int(inverted.sum())
    else:
        skipped.append("Price sanity: no OHLC columns present.")
        skipped.append("High/low ordering: no OHLC columns present.")

    # ------------------------------------------------------- extreme returns
    if "close" in ohlcv:
        close = pd.to_numeric(frame[ohlcv["close"]], errors="coerce")
        log_returns = np.log(close.where(close > 0)).diff()
        # Work on positional numpy arrays rather than the Series index. A dataset can
        # arrive with a non-default index (a CSV upload, a filtered frame), and
        # indexing `times` with a label would then silently read the wrong row — or
        # raise. Positions are unambiguous.
        raw = log_returns.to_numpy(dtype="float64")
        usable = np.isfinite(raw)
        positions = np.nonzero(usable)[0]
        sample = raw[positions]

        if len(sample) >= MIN_OBS_FOR_Z:
            median = float(np.median(sample))
            # MAD rather than standard deviation: a single 50-sigma crash inflates the
            # standard deviation enough to hide itself, which is precisely the
            # observation we are looking for.
            mad = float(np.median(np.abs(sample - median)))
            if mad > 0:
                # 1.4826 scales MAD to be a consistent estimator of sigma for normal
                # data, so the threshold reads in familiar standard-deviation units.
                scale = 1.4826 * mad
                z = np.abs(sample - median) / scale
                flagged = np.nonzero(z > EXTREME_RETURN_Z)[0]
                for index in flagged[:200]:
                    row_position = int(positions[index])
                    markers.append(
                        {
                            "time": _epoch(times[row_position]),
                            "kind": "extreme_return",
                            "detail": (
                                f"Log return {float(sample[index]):+.4f} is "
                                f"{float(z[index]):.1f} robust SD from the median."
                            ),
                        }
                    )
                counts["extreme_return"] = int(flagged.size)
            else:
                skipped.append(
                    "Extreme returns: returns are degenerate (MAD is zero), so no "
                    "robust scale can be estimated."
                )
        else:
            skipped.append(
                f"Extreme returns: needs at least {MIN_OBS_FOR_Z} observations, "
                f"dataset has {len(sample)}."
            )
    else:
        skipped.append("Extreme returns: no close column present.")

    # Markers with no usable timestamp sort last. Ordering them first would put them
    # at the start of the chart viewer's list, where they would read as the oldest
    # events rather than as undated ones.
    markers.sort(key=lambda m: (m["time"] is None, m["time"] or 0))
    return {
        "time_column": time_column,
        "markers": markers,
        "counts": counts,
        "skipped": skipped,
        # Detail behind a count that has no per-row marker — what a check found, as
        # opposed to `skipped`, which says a check could not run at all.
        "notes": notes,
        # Per-column detail, so the Cleaning page can offer to fill the columns that
        # actually have holes rather than presenting every column as equally suspect.
        "columns": _column_report(frame),
    }


def now_utc() -> datetime:
    return datetime.now(UTC)
