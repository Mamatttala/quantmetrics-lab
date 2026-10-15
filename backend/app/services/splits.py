"""Module C6 — chronological splits.

A split is the decision about which rows a model may ever see. Getting it wrong does not
produce an error; it produces a backtest that is too good, and the frame it ran on looks
exactly like a correct one. Two ways to get it wrong are handled here explicitly.

**The boundary is a boundary, not a fence.** Bars either side of a cut are not
independent: a feature computed over a trailing window at the end of the training span
still sees bars that belong to the test span, and a label with a forward horizon at the
end of training still reads into it. So a boundary is not just a cut index — it is a cut
index plus whatever has to be removed from the earlier side for the two spans to be
genuinely separate. See ``_apply_boundary``.

**A plan is only valid against the frame it was computed on.** The plan stores row indices
because indices are unambiguous, and it stores the source dataset's checksum so that
applying it to a different frame is refused rather than silently mis-sliced.

Nothing here writes a new price frame. A plan is a description, so it is stored as an
``outputs`` table and the dataset it describes is not touched at all.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from app.config import Settings, get_settings
from app.core.blocking import run_blocking
from app.core.errors import DatasetNotFound, ValidationFailure
from app.models.dataset import DatasetMeta
from app.models.enums import DatasetCategory, SplitRole, SplitType
from app.models.splits import (
    PERCENTAGES,
    SplitCreateRequest,
    SplitParameters,
    SplitPlanCreated,
    SplitPlanRow,
    SplitPlanSummary,
    SplitReport,
)
from app.services import quality as quality_checks
from app.services.datasets import DatasetService, persist_dataset, resolved_path
from app.storage import parquet_store
from app.storage.catalog import Catalog

#: The lineage/operation verb for a split plan. Also what marks a plan table as a plan
#: rather than a frame of bars, so the two modules that could mistake one for data can
#: tell them apart.
OPERATION = "split"

#: The columns of the stored plan table, in order. The specification lists
#: `split_id, role, fold, start_time, end_time, row_count`; `start_index`, `end_index` and
#: `dataset_checksum` ride along, and the reasons are in the model docstrings. The order
#: here is what the Parquet writer uses, so the file reads left to right as the spec does.
PLAN_COLUMNS: tuple[str, ...] = (
    "split_id",
    "fold",
    "role",
    "start_index",
    "end_index",
    "start_time",
    "end_time",
    "row_count",
    "dataset_checksum",
)


# --------------------------------------------------------------------------- geometry


@dataclass(frozen=True)
class Span:
    """A half-open row range ``[start, end)`` with a role and a fold."""

    fold: int
    role: SplitRole
    start: int
    end: int

    @property
    def rows(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class Boundary:
    """One cut, and what had to be removed from the span ending at it.

    ``gap`` and ``purge`` are reported separately and are deliberately not additive. The
    embargo removes ``[c - gap, c)``; the purge removes every row whose label reaches the
    cut, ``[c - horizon, c)``. The two sets overlap, so the span ends at ``c - max(gap,
    horizon)`` and ``purge`` counts only the rows the purge removed *beyond* the embargo —
    which is frequently zero. Reporting them as two independent counts is the difference
    between a plan that reads "twice as safe" and one that says what it did.
    """

    fold: int
    role: SplitRole
    cut: int
    gap: int
    purge: int
    end: int

    def payload(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "role": self.role.value,
            "cut_index": self.cut,
            "gap_rows_removed": self.gap,
            "purge_rows_removed": self.purge,
            "effective_end_index": self.end,
        }


def _horizon(params: SplitParameters) -> int:
    """How many bars forward a label looks, or 0 when no purge was asked for.

    The label horizon is a property of the *labelling rule*, not of the split — it is
    meaningless unless the caller has said a purge is wanted, so it is read through the
    flag rather than stored unconditionally.
    """
    if not params.purge_flag:
        return 0
    return params.label_horizon_bars or 0


def _apply_boundary(
    span: Span, cut: int, *, gap: int, horizon: int
) -> tuple[Span, Boundary]:
    """Shorten a span that ends at `cut`, and record what the shortening removed.

    The slack is ``max(gap, horizon)`` rather than ``gap + horizon``: a row removed by the
    embargo cannot also be removed by the purge, and adding the two would push the
    training span further back than either rule asks for — silently discarding usable
    history, which is a real cost on a short series.
    """
    slack = max(gap, horizon)
    length = span.end - span.start
    removed = min(slack, length)
    end = span.end - removed
    return (
        Span(fold=span.fold, role=span.role, start=span.start, end=end),
        Boundary(
            fold=span.fold,
            role=span.role,
            cut=cut,
            # Clamped to the span so the two counts always sum to the rows actually lost.
            gap=min(gap, length),
            purge=max(0, min(horizon, length) - gap),
            end=end,
        ),
    )


def _percentage_spans(
    n_rows: int, split_type: SplitType, params: SplitParameters
) -> tuple[list[Span], list[Boundary]]:
    """70/15/15, 60/20/20, or the two-way reading of either.

    The cut indices are ``floor(N·p)``, so the three spans sum to exactly ``N`` — no row
    is silently dropped and none is counted twice. Rounding is not corrected for, because
    there is no drift to correct: the third span absorbs the remainder by construction.
    """
    first, second, _ = PERCENTAGES[split_type]
    train_val_cut = math.floor(n_rows * first)
    val_test_cut = math.floor(n_rows * (first + second))

    gap, horizon = params.gap_size, _horizon(params)
    spans: list[Span] = []
    boundaries: list[Boundary] = []

    train = Span(fold=0, role=SplitRole.TRAIN, start=0, end=train_val_cut)
    train, boundary = _apply_boundary(train, train_val_cut, gap=gap, horizon=horizon)
    spans.append(train)
    boundaries.append(boundary)

    middle = Span(fold=0, role=SplitRole.VAL, start=train_val_cut, end=val_test_cut)
    middle, boundary = _apply_boundary(middle, val_test_cut, gap=gap, horizon=horizon)
    spans.append(middle)
    boundaries.append(boundary)

    spans.append(Span(fold=0, role=SplitRole.TEST, start=val_test_cut, end=n_rows))
    return spans, boundaries


def _tscv_spans(
    n_rows: int, params: SplitParameters, *, rolling: bool
) -> tuple[list[Span], list[Boundary]]:
    """Walk-forward folds, expanding or rolling.

    ``n_folds + 1`` equal blocks; fold *k* tests block *k*, so the first block is only
    ever training data and the last fold's test span reaches the end of the series (it
    absorbs the remainder, so no tail row is left untested).

    The rolling window is **one block**, which is the only length derivable from the
    controls the specification offers — it gives `n_folds`, `gap_size` and `purge_flag`
    and no window length at all. Defining it as the block immediately before the cut makes
    every fold train on the same number of bars (``block − slack``), which is what makes it
    rolling rather than merely shorter, and the report states both numbers so the choice is
    visible rather than implied.
    """
    folds = params.n_folds
    block = n_rows // (folds + 1)
    gap, horizon = params.gap_size, _horizon(params)

    spans: list[Span] = []
    boundaries: list[Boundary] = []
    for fold in range(1, folds + 1):
        cut = fold * block
        # The last fold takes the remainder, so the final block is at least as long as
        # the others and the series has no untested tail.
        end = n_rows if fold == folds else (fold + 1) * block
        start = cut - block if rolling else 0
        train = Span(fold=fold, role=SplitRole.TRAIN, start=start, end=cut)
        train, boundary = _apply_boundary(train, cut, gap=gap, horizon=horizon)
        spans.append(train)
        boundaries.append(boundary)
        spans.append(Span(fold=fold, role=SplitRole.TEST, start=cut, end=end))
    return spans, boundaries


def _date_span(
    times: pd.Series, start: datetime, end: datetime, *, fold: int, role: SplitRole
) -> Span:
    """Resolve a date pair to the rows whose observation instant falls inside it.

    Both ends are inclusive, which is what a date picker means: a span ending on the 31st
    contains the bar stamped on the 31st. `searchsorted` gives that directly — ``left`` on
    the start so a bar exactly at it is included, ``right`` on the end so a bar exactly at
    it is too — and it does so without materialising a mask over five million rows.
    """
    return Span(
        fold=fold,
        role=role,
        start=int(times.searchsorted(start, side="left")),
        end=int(times.searchsorted(end, side="right")),
    )


def build_spans(
    n_rows: int, times: pd.Series, split_type: SplitType, params: SplitParameters
) -> tuple[list[Span], list[Boundary]]:
    """The whole geometry of a plan, from the row count and the observation instants.

    Date splits ignore gap and purge, and the API refuses a request that sets them: the
    user named the boundary, so shrinking it silently would be worse than having no knob.
    """
    if split_type in PERCENTAGES:
        return _percentage_spans(n_rows, split_type, params)
    if split_type.is_tscv:
        return _tscv_spans(n_rows, params, rolling=split_type is SplitType.TSCV_ROLLING)

    pairs: list[tuple[SplitRole, str, str]] = [
        (SplitRole.TRAIN, "train_start", "train_end"),
        (SplitRole.TEST, "test_start", "test_end"),
    ]
    if split_type is SplitType.DATE_3WAY:
        pairs.insert(1, (SplitRole.VAL, "val_start", "val_end"))

    spans: list[Span] = []
    for role, start_key, end_key in pairs:
        start = getattr(params, start_key)
        end = getattr(params, end_key)
        # The request validator guarantees both are set for this split type. Refused
        # rather than asserted: this function is also reachable directly from the tests,
        # and a bare `assert` disappears under `-O`, which would turn a clear refusal into
        # an AttributeError on None.
        if start is None or end is None:
            raise ValidationFailure(
                f"A {split_type.value} split needs {start_key} and {end_key}."
            )
        spans.append(_date_span(times, start, end, fold=0, role=role))
    return spans, []


# --------------------------------------------------------------------------- service


class SplitService:
    """Plans splits. Public methods are coroutines."""

    def __init__(self, settings: Settings | None = None, catalog: Catalog | None = None) -> None:
        self.settings = settings or get_settings()
        self.catalog = catalog or Catalog(self.settings.catalog_path)
        self.datasets = DatasetService(self.settings, self.catalog)

    # ------------------------------------------------------------------ creation

    async def create(self, request: SplitCreateRequest) -> SplitPlanCreated:
        parent = await self.datasets.get(request.dataset_id)
        path = resolved_path(parent, self.settings)
        time_column = _time_column_of(parent)

        def work() -> SplitPlanCreated:
            # One column, not the frame. A plan is arithmetic over the timeline; reading
            # every price column of a five-million-row file to compute it would be work
            # whose result is thrown away.
            frame = parquet_store.read_dataset(path, columns=[time_column])
            times = _observation_times(frame[time_column], parent)
            _require_strictly_increasing(times, parent)

            n_rows = int(len(times))
            spans, boundaries = build_spans(n_rows, times, request.split_type, request.parameters)
            _require_usable(spans, n_rows, request, time_column)

            plan_id = parquet_store.new_dataset_id()
            rows = [
                _to_row(plan_id, span, times, parent.checksum)
                for span in spans
                if span.rows > 0
            ]
            stored = persist_dataset(
                parent=parent,
                frame=_plan_frame(rows),
                data_dir=self.settings.data_dir,
                catalog=self.catalog,
                category=DatasetCategory.OUTPUTS,
                name=request.name or f"{parent.name} (split)",
                operation=OPERATION,
                detail={
                    "split_type": request.split_type.value,
                    "parameters": request.parameters.model_dump(mode="json"),
                    "spans": [row.model_dump(mode="json") for row in rows],
                    "boundaries": [boundary.payload() for boundary in boundaries],
                },
                extra_provenance={
                    "split_type": request.split_type.value,
                    "split_parameters": json.dumps(
                        request.parameters.model_dump(mode="json"), default=str
                    ),
                    # The number of distinct fold labels, not "one past the highest". A
                    # percentage plan labels its three spans fold 0 and a walk-forward plan
                    # labels its folds 1..n, so `max(fold) + 1` reads 1 for the first (right,
                    # by accident) and n+1 for the second — and C7 builds its fold selector
                    # from this number, so it offered a fold the plan does not contain and the
                    # run that picked it was refused.
                    "split_folds": str(len({row.fold for row in rows})),
                    "split_time_column": time_column,
                    "split_dataset_checksum": parent.checksum,
                },
                dataset_id=plan_id,
            )
            return SplitPlanCreated(
                dataset=stored,
                plan=rows,
                report=_report(
                    request, time_column, n_rows, spans, boundaries, rows, parent
                ),
            )

        return await run_blocking(work)

    # ------------------------------------------------------------------- reading

    async def plans(self, dataset_id: str) -> list[SplitPlanSummary]:
        """Every plan already made from this dataset, newest first."""
        await self.datasets.get(dataset_id)
        return await run_blocking(plans_for, self.catalog, dataset_id)

    async def load_plan(self, split_plan_id: str) -> tuple[DatasetMeta, list[SplitPlanRow]]:
        """A stored plan and its rows. Read by C7, which scales against it."""
        meta = await run_blocking(self.catalog.get_dataset, split_plan_id)
        if meta is None:
            raise DatasetNotFound(f"No split plan with id {split_plan_id}.")
        if meta.operation != OPERATION:
            raise ValidationFailure(
                f"'{meta.name}' is not a split plan; it is a "
                f"{meta.operation or 'dataset'}.",
                hint="Choose a plan from the list of plans made for this dataset.",
            )
        path = resolved_path(meta, self.settings)

        def work() -> list[SplitPlanRow]:
            frame = parquet_store.read_dataset(path)
            # `to_dict` types its keys as `Hashable` because a frame's labels need not be
            # strings; this one's are, because `_plan_frame` wrote them. Narrowed here
            # rather than loosening `_from_record`, which reads exact column names.
            return [
                _from_record({str(key): value for key, value in record.items()})
                for record in frame.to_dict(orient="records")
            ]

        return meta, await run_blocking(work)


# --------------------------------------------------------------------------- helpers


def plans_for(catalog: Catalog, dataset_id: str) -> list[SplitPlanSummary]:
    """Every plan made from one dataset, newest first. Synchronous by design.

    A module-level function rather than a method, because C7's options endpoint needs the
    plan list *inside* its own ``run_blocking`` call — it reads the frame and the plans in
    one blocking pass, and awaiting between them would hand the event loop back mid-read
    for no benefit. A public method here would have to be a coroutine
    (``tests/test_async_rule.py`` enforces that for every service entry point, and the rule
    is right: it is what keeps blocking work inside the one sanctioned chokepoint), so the
    sync form lives beside ``build_spans`` instead.
    """
    metas = catalog.derived_datasets(dataset_id, OPERATION)
    return [_summarise(meta) for meta in metas]


def _time_column_of(meta: DatasetMeta) -> str:
    """Which column the plan is laid out along, resolved from the catalog alone.

    The same candidate list the quality checks and the chart use, read from the dataset's
    recorded schema rather than from the file — so the file can be opened once, with one
    column projected.

    This is the *observation* instant — the moment a bar opens — and never the instant it
    closes. The two differ by exactly one bar length, which is the size of the look-ahead
    error the wrong choice would introduce, and it would be invisible: every span would
    still be contiguous, ordered and non-overlapping. `tests/test_leakage_guards.py`
    asserts the bar-close column is not in the candidate list at all.
    """
    present = {column.name for column in meta.columns}
    for candidate in quality_checks.TIME_COLUMN_CANDIDATES:
        if candidate in present:
            return candidate
    raise ValidationFailure(
        f"'{meta.name}' has no recognisable timestamp column, so there is no timeline to "
        "split.",
        hint="Expected one of: " + ", ".join(quality_checks.TIME_COLUMN_CANDIDATES) + ".",
    )


def _observation_times(column: pd.Series, meta: DatasetMeta) -> pd.Series:
    """The timeline as instants, whatever the column was stored as.

    An uploaded CSV keeps its timestamps as *text*: ``pandas.read_csv`` types nothing, so a
    file whose dates look like dates is stored and read back as a ``string`` column. Only
    the fetched datasets are born with a ``datetime64`` one, which is why this is easy to
    miss — every frame this module was written against already had the right dtype.

    Reading the text straight through would be worse than it sounds. ``is_monotonic`` on
    strings compares them lexicographically, and that agrees with the clock only while
    every row is spelled identically — one row at a different offset or width and the
    check passes on a frame whose spans are scattered. The spans would still be contiguous
    in row terms, still ordered, still non-overlapping, and would cover the wrong bars.
    So the column is coerced once, here, before any arithmetic touches it, and the
    comparison, the ``searchsorted`` in ``_date_span`` and the plan's own timestamps are
    all then made of real instants.

    This is the same coercion B3 and B4 apply (``cleaning``, ``merge``), and it fails the
    same way: a value that will not parse is refused rather than dropped, because a frame
    whose timeline has a hole in it cannot be cut on.
    """
    stamps = pd.to_datetime(column, utc=True, errors="coerce")
    if bool(stamps.isna().any()):
        raise ValidationFailure(
            f"Splitting needs a parseable timestamp in every row; {column.name} has at "
            "least one unparseable value.",
            hint="Repair it in B3 Cleaning, then split the cleaned dataset.",
        )
    return stamps


def _require_strictly_increasing(times: pd.Series, meta: DatasetMeta) -> None:
    """The timeline must be strictly increasing before anything is cut on it.

    Every span here is a row range, so a frame that is out of order would produce a plan
    that is contiguous in *row* terms and scattered in *time* terms — the spans would look
    right in the table and cover the wrong bars. A repeated instant is refused for the
    same reason: two rows at one timestamp make "the bar before the cut" ambiguous.
    """
    if not times.is_monotonic_increasing:
        raise ValidationFailure(
            f"'{meta.name}' is not sorted by {times.name}, so a split would cut rows by "
            "position while claiming to cut them by time.",
            hint="Sort it in B3 Cleaning (Sort by time) and split the cleaned dataset.",
        )
    if not times.is_unique:
        raise ValidationFailure(
            f"'{meta.name}' has repeated timestamps in {times.name}, so which of two rows "
            "at one instant belongs to a span is undefined.",
            hint="Deduplicate it in B3 Cleaning, keeping the first or last row per "
            "timestamp.",
        )


def _require_usable(
    spans: list[Span], n_rows: int, request: SplitCreateRequest, time_column: str
) -> None:
    """Refuse a plan with a span too short to be one.

    Judged before anything is written rather than reported afterwards: an empty test span
    is not a small problem with the plan, it is a plan that cannot be used, and storing it
    would leave a plausible-looking artifact that fails much later.

    The fold-count check comes **first**, and deliberately. ``n_folds`` folds need
    ``n_folds + 1`` blocks to be at least two bars each, so a series that is too short
    produces a *block* of one bar or none — and every symptom below it (an empty test span,
    a one-row training span) is a consequence of that one fact. Reporting the symptom would
    send the user to adjust the wrong control.
    """
    if n_rows == 0:
        raise ValidationFailure("The dataset has no rows to split.")
    labelled = f"{n_rows} rows of {time_column}"
    if request.split_type.is_tscv:
        folds = request.parameters.n_folds
        if n_rows // (folds + 1) < 2:
            raise ValidationFailure(
                f"{folds} folds need at least {2 * (folds + 1)} rows ({labelled}); each "
                f"block would be {n_rows // (folds + 1)} bar(s), and a fold has to train "
                "on one block and test on another.",
                hint="Use fewer folds, or a longer dataset.",
            )
    for span in spans:
        if span.role is SplitRole.TEST and span.rows == 0:
            raise ValidationFailure(
                f"This split leaves the test span empty ({labelled}).",
                hint="Use a shorter training span, or a longer dataset.",
            )
    train = [span for span in spans if span.role is SplitRole.TRAIN]
    if train and min(span.rows for span in train) < 2:
        smallest = min(span.rows for span in train)
        raise ValidationFailure(
            f"This split leaves a training span of {smallest} row(s) ({labelled}), which "
            "is too short to fit anything on.",
            hint="Reduce gap_size or label_horizon_bars, use fewer folds, or use a "
            "longer dataset.",
        )


def _to_row(plan_id: str, span: Span, times: pd.Series, checksum: str) -> SplitPlanRow:
    return SplitPlanRow(
        split_id=plan_id,
        fold=span.fold,
        role=span.role,
        start_index=span.start,
        end_index=span.end,
        start_time=times.iloc[span.start].to_pydatetime(),
        end_time=times.iloc[span.end - 1].to_pydatetime(),
        row_count=span.rows,
        dataset_checksum=checksum,
    )


def _plan_frame(rows: list[SplitPlanRow]) -> pd.DataFrame:
    """The plan as a frame, in `PLAN_COLUMNS` order.

    Built column by column with explicit pandas types rather than from the dicts, because
    the frame is what gets written to Parquet: an index column that landed as `object`
    would make the plan unreadable by `searchsorted`-free integer slicing downstream, and
    a timestamp column that landed tz-naive would be reinterpreted as local time on read.
    """
    return pd.DataFrame(
        {
            "split_id": pd.Series([row.split_id for row in rows], dtype="object"),
            "fold": pd.Series([row.fold for row in rows], dtype="int64"),
            "role": pd.Series([row.role.value for row in rows], dtype="object"),
            "start_index": pd.Series([row.start_index for row in rows], dtype="int64"),
            "end_index": pd.Series([row.end_index for row in rows], dtype="int64"),
            "start_time": pd.Series(
                pd.to_datetime([row.start_time for row in rows], utc=True)
            ),
            "end_time": pd.Series(pd.to_datetime([row.end_time for row in rows], utc=True)),
            "row_count": pd.Series([row.row_count for row in rows], dtype="int64"),
            "dataset_checksum": pd.Series(
                [row.dataset_checksum for row in rows], dtype="object"
            ),
        }
    )[list(PLAN_COLUMNS)]


def _from_record(record: dict[str, Any]) -> SplitPlanRow:
    return SplitPlanRow(
        split_id=str(record["split_id"]),
        fold=int(record["fold"]),
        role=SplitRole(str(record["role"])),
        start_index=int(record["start_index"]),
        end_index=int(record["end_index"]),
        start_time=record["start_time"],
        end_time=record["end_time"],
        row_count=int(record["row_count"]),
        dataset_checksum=str(record["dataset_checksum"]),
    )


def _split_type_of(meta: DatasetMeta) -> SplitType:
    """The split type a plan was built with, or a refusal.

    Not defaulted. A dataset whose operation is ``split`` but whose provenance carries no
    recognisable ``split_type`` was not written by this module, and guessing a scheme for
    it would put a plan in the C7 dropdown labelled with a geometry it may not have — and
    the whole point of the label is that the scaler is fitted against that geometry.
    """
    raw = meta.provenance.get("split_type", "")
    try:
        return SplitType(raw)
    except ValueError:
        raise ValidationFailure(
            f"'{meta.name}' is marked as a split plan but records no usable split type "
            f"(found {raw!r}).",
            hint="Recreate the plan from the dataset it was made from.",
        ) from None


def _summarise(meta: DatasetMeta) -> SplitPlanSummary:
    """One plan's stored provenance, back into the shape the UI lists.

    The parameters are read back out of the JSON string they were written as, so the
    dropdown in C7 can say what each plan *is* without reading its Parquet file.
    """
    raw = meta.provenance.get("split_parameters") or "{}"
    try:
        parameters: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError:  # pragma: no cover - only reachable on a hand-edited row
        parameters = {}
    return SplitPlanSummary(
        split_plan_id=meta.dataset_id,
        name=meta.name,
        split_type=_split_type_of(meta),
        folds=int(meta.provenance.get("split_folds", "1") or 1),
        rows=meta.row_count,
        created_at=meta.created_at,
        parameters=parameters,
    )


def _report(
    request: SplitCreateRequest,
    time_column: str,
    n_rows: int,
    spans: list[Span],
    boundaries: list[Boundary],
    rows: list[SplitPlanRow],
    parent: DatasetMeta,
) -> SplitReport:
    by_role: dict[str, int] = {}
    for span in spans:
        by_role[span.role.value] = by_role.get(span.role.value, 0) + span.rows
    tested = {
        index
        for span in spans
        if span.role is SplitRole.TEST
        for index in range(span.start, span.end)
    }
    return SplitReport(
        split_type=request.split_type,
        time_column=time_column,
        rows=n_rows,
        dataset_checksum=parent.checksum,
        parameters=request.parameters.model_dump(mode="json"),
        folds=len({row.fold for row in rows}),
        rows_by_role=by_role,
        rows_tested=len(tested),
        rows_never_tested=n_rows - len(tested),
        boundaries=[boundary.payload() for boundary in boundaries],
        warnings=_warnings(request, spans, boundaries, time_column, n_rows),
    )


def _warnings(
    request: SplitCreateRequest,
    spans: list[Span],
    boundaries: list[Boundary],
    time_column: str,
    n_rows: int,
) -> list[str]:
    notes: list[str] = []
    params = request.parameters
    if params.purge_flag and not any(boundary.purge for boundary in boundaries):
        notes.append(
            f"purge_flag removed no rows: the label horizon "
            f"({params.label_horizon_bars} bars) is not longer than gap_size "
            f"({params.gap_size} bars), so the embargo had already dropped every row the "
            "purge would have. The two rules overlap rather than adding up."
        )
    smallest_test = min(
        (span.rows for span in spans if span.role is SplitRole.TEST), default=0
    )
    if 0 < smallest_test < 2:
        notes.append(
            f"The shortest test span is {smallest_test} bar(s). A single bar has no "
            "distribution to score against, so the fold will report a number that means "
            "very little."
        )
    if request.split_type.is_tscv:
        block = n_rows // (params.n_folds + 1)
        scheme = (
            "everything before it (expanding)"
            if request.split_type is SplitType.TSCV_EXPANDING
            else "the single block before it (rolling)"
        )
        notes.append(
            f"Blocks are {block} bars: the series is divided into {params.n_folds + 1} "
            f"equal parts and fold k tests part k, training on {scheme}. The first block "
            "is never tested."
        )
    if any(boundary.gap for boundary in boundaries):
        first = boundaries[0]
        notes.append(
            f"{first.gap} bar(s) of embargo sit between the spans at row {first.cut}; "
            "those rows belong to no span, so no feature window reaches across the cut."
        )
    return notes
