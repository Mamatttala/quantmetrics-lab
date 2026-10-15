"""Module C6 — chronological splits over HTTP.

A split is a *description* of a dataset, not a change to it, so nothing here produces a
new price frame. What it produces is a plan: a table of spans, each saying which rows it
covers and what they are for. C7 scales against it and Modules E/F train on it.

Two shapes in this module are deliberately richer than the specification's sketch, and
both for the same reason — a plan has to remain *applicable* after it is written:

* **Row indices travel with the timestamps.** The specification asks for `{fold, role,
  start_index, end_index}` in the TSCV case and for start/end times in the plan schema.
  Storing both means a plan is never reconstructed by searching a timestamp column, which
  would silently pick the wrong row on a frame with a repeated or missing bar.
* **The plan records the checksum of the dataset it was computed on.** A span expressed in
  row indices is only meaningful against one exact frame. Without the checksum a plan could
  be applied to a different dataset whose rows *happen* to line up and nothing would say so.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from app.models.dataset import DatasetMeta
from app.models.enums import SplitRole, SplitType

#: The split types whose boundaries the application chooses. Gap and purge are offered for
#: these and refused for the date splits, where the user names the boundary themselves —
#: silently shrinking a span the user typed would be worse than having no knob at all.
DERIVED_BOUNDARY_TYPES: frozenset[SplitType] = frozenset(
    {
        SplitType.PERCENTAGE_70_15_15,
        SplitType.PERCENTAGE_60_20_20,
        SplitType.TSCV_EXPANDING,
        SplitType.TSCV_ROLLING,
    }
)

#: The train/validation/test proportions each percentage member stands for.
PERCENTAGES: dict[SplitType, tuple[float, float, float]] = {
    SplitType.PERCENTAGE_70_15_15: (0.70, 0.15, 0.15),
    SplitType.PERCENTAGE_60_20_20: (0.60, 0.20, 0.20),
}


class SplitParameters(BaseModel):
    """Everything a split needs, for every split type.

    One flat object rather than six per-type objects. The request is stored whole in the
    plan's lineage, and a union of six types would make the stored document's *shape*
    depend on the split type — so a reader asking "what gap did this plan use?" would have
    to know which type it was before knowing where to look.
    """

    #: Date-split boundaries. All six are required for `date_3way`, four for `date_2way`,
    #: and none for the others.
    train_start: datetime | None = None
    train_end: datetime | None = None
    val_start: datetime | None = None
    val_end: datetime | None = None
    test_start: datetime | None = None
    test_end: datetime | None = None

    #: Walk-forward configuration. `n_folds` bounds match the specification's slider.
    n_folds: int = Field(default=5, ge=2, le=10)

    #: Bars dropped between one span's end and the next span's start — the *embargo*. It
    #: exists because a feature computed over a trailing window at the end of the training
    #: span still sees bars that overlap the beginning of the test span.
    gap_size: int = Field(default=0, ge=0, le=20)

    #: Whether to drop training rows whose *label* reaches into the next span.
    purge_flag: bool = False

    #: How many bars forward a label looks. Required when `purge_flag` is set, and
    #: deliberately not defaulted: the purged row count is exactly this number (minus the
    #: gap), so a silent default would be a silent methodology choice.
    label_horizon_bars: int | None = Field(default=None, ge=1, le=10_000)

    @field_validator("train_start", "train_end", "val_start", "val_end", "test_start", "test_end")
    @classmethod
    def _require_zone(cls, value: datetime | None) -> datetime | None:
        """Refuse a zone-less timestamp rather than guessing a zone for it.

        `2024-01-01T00:00` means a different instant in every timezone, and a boundary
        resolved one zone away silently moves bars between train and test. `DECISIONS §23`
        resolves a zone-less *value in a dataset* as UTC; a zone-less *request* is a
        different situation, because the client knows the zone and simply did not say.
        """
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError(
                "Boundaries must carry a UTC offset, e.g. 2024-01-01T00:00:00Z. A "
                "zone-less timestamp would be read in the server's local zone, which "
                "would move bars between spans without saying so."
            )
        return value


class SplitCreateRequest(BaseModel):
    """One request to plan a split."""

    dataset_id: str
    split_type: SplitType
    parameters: SplitParameters = Field(default_factory=SplitParameters)

    #: Optional name for the stored plan. Defaults to the source name plus the split type.
    name: str | None = None

    @model_validator(mode="after")
    def _check(self) -> SplitCreateRequest:
        params = self.parameters
        needed = _REQUIRED_DATES[self.split_type]
        missing = [key for key in needed if getattr(params, key) is None]
        if missing:
            raise ValueError(
                f"A {self.split_type.value} split needs {', '.join(needed)}; "
                f"missing {', '.join(missing)}."
            )

        if needed:
            _check_date_order(params)
            # A no-op default is fine; a set value on a split that ignores it is not.
            if params.gap_size > 0 or params.purge_flag:
                raise ValueError(
                    "gap_size and purge_flag apply to the percentage and walk-forward "
                    "splits, where this application chooses the boundary. A date split "
                    "names its boundary directly — move the date instead."
                )

        if params.purge_flag and params.label_horizon_bars is None:
            raise ValueError(
                "purge_flag needs label_horizon_bars: how many bars forward a label "
                "looks. Without it the purge has no size and would remove nothing."
            )
        return self


#: Which date fields each split type requires. Written as a mapping rather than as
#: branches inside the validator so the error message and the check cannot drift apart.
_REQUIRED_DATES: dict[SplitType, tuple[str, ...]] = {
    SplitType.DATE_3WAY: (
        "train_start",
        "train_end",
        "val_start",
        "val_end",
        "test_start",
        "test_end",
    ),
    SplitType.DATE_2WAY: ("train_start", "train_end", "test_start", "test_end"),
    SplitType.PERCENTAGE_70_15_15: (),
    SplitType.PERCENTAGE_60_20_20: (),
    SplitType.TSCV_EXPANDING: (),
    SplitType.TSCV_ROLLING: (),
}

#: Every date boundary, in the order a plan reads left to right. The validation walks the
#: *present* members of this list rather than a fixed chain of pairs, which matters: a fixed
#: chain runs train_end → val_start → val_end → test_start, and a 2-way split has no
#: validation dates, so the train_end → test_start link would be skipped entirely and two
#: spans sharing a bar would pass validation.
_DATE_ORDER = (
    "train_start",
    "train_end",
    "val_start",
    "val_end",
    "test_start",
    "test_end",
)


def _check_date_order(params: SplitParameters) -> None:
    """Every boundary must be strictly after the one before it.

    Strictly, not merely non-decreasing. Two spans that share an instant share a bar, and
    a bar in both train and test is a leak with no visible symptom — the frame is the same
    either way. A 2-way split simply has four boundaries instead of six; the comparison is
    between consecutive *given* dates, so train_end is checked against test_start.
    """
    present = [
        (key, value)
        for key in _DATE_ORDER
        if (value := getattr(params, key)) is not None
    ]
    for (earlier_key, earlier), (later_key, later) in zip(present, present[1:], strict=False):
        if later <= earlier:
            raise ValueError(
                f"{later_key} must be after {earlier_key}; got {later.isoformat()} <= "
                f"{earlier.isoformat()}. Spans must not share a bar."
            )


class SplitPlanRow(BaseModel):
    """One span in a plan — one row of the stored plan table.

    ``end_index`` is **exclusive** and ``end_time`` is **inclusive**: the index pair
    slices, its timestamp pair reads. Both are needed and they are not the same
    convention, so each says which it is. ``end_time`` is the timestamp of the *last bar
    in the span*, which is what a person reading the table expects "End" to mean.
    """

    split_id: str
    fold: int = Field(ge=0)
    role: SplitRole
    start_index: int = Field(ge=0)
    end_index: int = Field(ge=0)
    start_time: datetime
    end_time: datetime
    row_count: int = Field(ge=0)
    dataset_checksum: str


class SplitReport(BaseModel):
    """What the plan did, in enough detail to check it without re-deriving it.

    ``boundaries`` is the part that matters most. The specification presents the gap and
    the purge as two independent knobs, and they are not additive: the embargo drops
    `[c - gap, c)` and the purge drops `[c - h, c)`, so together the training span ends at
    `c - max(gap, h)` and one of the two frequently removes nothing. Reporting the rows
    each rule actually removed is what keeps a `gap_size: 5, purge_flag: true` plan from
    reading as "twice as safe".
    """

    split_type: SplitType
    time_column: str
    rows: int
    dataset_checksum: str
    parameters: dict[str, Any]
    folds: int
    rows_by_role: dict[str, int]
    rows_tested: int
    rows_never_tested: int
    boundaries: list[dict[str, Any]]
    warnings: list[str]


class SplitPlanCreated(BaseModel):
    """The 201 body: the stored plan dataset, its rows, and the report."""

    dataset: DatasetMeta
    plan: list[SplitPlanRow]
    report: SplitReport


class SplitPlanSummary(BaseModel):
    """One existing plan, as listed in C6's panel and C7's dropdown."""

    split_plan_id: str
    name: str
    split_type: SplitType
    folds: int
    rows: int
    created_at: datetime
    parameters: dict[str, Any]
