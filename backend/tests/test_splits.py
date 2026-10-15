"""Module C6 — chronological splits, arithmetic and plan geometry.

Two families of test here, and they catch different mistakes.

**Geometry.** A split is correct if its spans are exactly the rows they claim: disjoint,
ordered, summing to the whole series, and with each fold's training span ending before its
test span begins. These are cheap to assert and they fail loudly on an off-by-one, which is
the failure mode a plan has — a plan that is wrong by one row still looks like a plan.

**Refusals.** The failures that matter are the ones that would otherwise be silent: an
unsorted timeline, a span with no rows in it, a plan applied to a different frame. Each is
asserted to raise rather than to produce a plausible-looking plan.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.core.errors import DatasetNotFound, ValidationFailure
from app.models.enums import DatasetCategory, SplitRole, SplitType
from app.models.splits import SplitCreateRequest, SplitParameters
from app.services.splits import (
    OPERATION,
    Span,
    SplitService,
    _apply_boundary,
    _time_column_of,
    build_spans,
    plans_for,
)
from tests.conftest import hourly_frame, register

UTC_START = datetime(2024, 1, 1, tzinfo=UTC)


def _params(**kwargs: object) -> SplitParameters:
    return SplitParameters(**kwargs)


def _service(settings: object, catalog: object) -> SplitService:
    return SplitService(settings, catalog)  # type: ignore[arg-type]


# ----------------------------------------------------------------------- geometry


def test_percentage_spans_are_disjoint_ordered_and_total():
    spans, boundaries = build_spans(
        100, hourly_frame(100)["open_time"], SplitType.PERCENTAGE_70_15_15, _params()
    )
    assert [(s.role, s.start, s.end) for s in spans] == [
        (SplitRole.TRAIN, 0, 70),
        (SplitRole.VAL, 70, 85),
        (SplitRole.TEST, 85, 100),
    ]
    # No gap and no purge, so the two boundaries exist and removed nothing. They are
    # reported anyway — a boundary that removed zero rows is still the place where a
    # feature window could have reached across, and that is worth saying explicitly.
    assert [(b.cut, b.gap, b.purge) for b in boundaries] == [(70, 0, 0), (85, 0, 0)]
    covered = [index for span in spans for index in range(span.start, span.end)]
    assert covered == list(range(100))


def test_percentage_remainder_lands_in_the_test_span():
    # 101 rows at 70/15/15: floor(70.7)=70, floor(85.85)=85, so the last span is 16 rows and
    # no row is dropped. The alternative — rounding each span to 15.15 — would leave a
    # remainder that belongs to nobody.
    spans, _ = build_spans(
        101, hourly_frame(101)["open_time"], SplitType.PERCENTAGE_70_15_15, _params()
    )
    assert sum(span.rows for span in spans) == 101
    assert spans[-1].end == 101


def test_sixty_twenty_twenty_uses_its_own_ratio():
    spans, _ = build_spans(
        100, hourly_frame(100)["open_time"], SplitType.PERCENTAGE_60_20_20, _params()
    )
    assert [(s.role, s.start, s.end) for s in spans] == [
        (SplitRole.TRAIN, 0, 60),
        (SplitRole.VAL, 60, 80),
        (SplitRole.TEST, 80, 100),
    ]


def test_tscv_expanding_folds_train_on_everything_before():
    spans, _ = build_spans(
        100, hourly_frame(100)["open_time"], SplitType.TSCV_EXPANDING, _params(n_folds=4)
    )
    assert len(spans) == 8
    for fold in range(1, 5):
        train = next(s for s in spans if s.fold == fold and s.role is SplitRole.TRAIN)
        test = next(s for s in spans if s.fold == fold and s.role is SplitRole.TEST)
        assert train.start == 0
        # `end` is exclusive, so equality here means the two spans are adjacent with no row
        # between them — which is what gap_size 0 asks for.
        assert train.end <= test.start
        assert test.start == fold * 20


def test_tscv_rolling_folds_train_on_one_block():
    spans, _ = build_spans(
        100, hourly_frame(100)["open_time"], SplitType.TSCV_ROLLING, _params(n_folds=4)
    )
    for fold in range(1, 5):
        train = next(s for s in spans if s.fold == fold and s.role is SplitRole.TRAIN)
        assert train.start == (fold - 1) * 20
        assert train.rows == 20


def test_tscv_last_fold_absorbs_the_remainder():
    # 103 rows, 4 folds → block 20, so fold 4's test would end at 100 and rows 100..102
    # would never be tested. They are given to the last fold instead.
    spans, _ = build_spans(
        103, hourly_frame(103)["open_time"], SplitType.TSCV_EXPANDING, _params(n_folds=4)
    )
    last = max(spans, key=lambda s: s.end)
    assert last.end == 103


def test_tscv_test_spans_are_disjoint_and_cover_everything_after_the_first_block():
    spans, _ = build_spans(
        100, hourly_frame(100)["open_time"], SplitType.TSCV_EXPANDING, _params(n_folds=4)
    )
    tested = [
        index
        for span in spans
        if span.role is SplitRole.TEST
        for index in range(span.start, span.end)
    ]
    assert tested == list(range(20, 100))


def test_every_fold_trains_before_it_tests():
    for split_type in (SplitType.TSCV_EXPANDING, SplitType.TSCV_ROLLING):
        spans, _ = build_spans(
            120, hourly_frame(120)["open_time"], split_type, _params(n_folds=5)
        )
        for fold in range(1, 6):
            train = next(s for s in spans if s.fold == fold and s.role is SplitRole.TRAIN)
            test = next(s for s in spans if s.fold == fold and s.role is SplitRole.TEST)
            assert train.end <= test.start, f"{split_type} fold {fold} trains into its test"


def test_date_split_is_inclusive_at_both_ends():
    frame = hourly_frame(48)
    times = frame["open_time"]
    spans, _ = build_spans(
        48,
        times,
        SplitType.DATE_2WAY,
        _params(
            train_start=times.iloc[0].to_pydatetime(),
            train_end=times.iloc[23].to_pydatetime(),
            test_start=times.iloc[24].to_pydatetime(),
            test_end=times.iloc[47].to_pydatetime(),
        ),
    )
    assert [(s.role, s.start, s.end) for s in spans] == [
        (SplitRole.TRAIN, 0, 24),
        (SplitRole.TEST, 24, 48),
    ]


def test_date_split_before_the_first_bar_yields_an_empty_span():
    times = hourly_frame(48)["open_time"]
    spans, _ = build_spans(
        48,
        times,
        SplitType.DATE_2WAY,
        _params(
            train_start=datetime(2023, 1, 1, tzinfo=UTC),
            train_end=datetime(2023, 6, 1, tzinfo=UTC),
            test_start=times.iloc[0].to_pydatetime(),
            test_end=times.iloc[47].to_pydatetime(),
        ),
    )
    assert spans[0].rows == 0


# ------------------------------------------------------------------- boundaries


def test_gap_shortens_the_training_span():
    span, boundary = _apply_boundary(
        Span(fold=0, role=SplitRole.TRAIN, start=0, end=70), 70, gap=5, horizon=0
    )
    assert span.end == 65
    assert boundary.gap == 5
    assert boundary.purge == 0


def test_purge_shortens_by_the_label_horizon():
    span, boundary = _apply_boundary(
        Span(fold=0, role=SplitRole.TRAIN, start=0, end=70), 70, gap=0, horizon=5
    )
    assert span.end == 65
    assert boundary.gap == 0
    assert boundary.purge == 5


def test_gap_and_purge_are_not_additive():
    # The embargo removes [65, 70); the purge removes [65, 70) too. The span ends at 65
    # either way — NOT at 60 — and the purge is reported as having removed nothing.
    span, boundary = _apply_boundary(
        Span(fold=0, role=SplitRole.TRAIN, start=0, end=70), 70, gap=5, horizon=5
    )
    assert span.end == 65
    assert boundary.gap == 5
    assert boundary.purge == 0


def test_purge_beyond_the_gap_removes_the_difference():
    span, boundary = _apply_boundary(
        Span(fold=0, role=SplitRole.TRAIN, start=0, end=70), 70, gap=2, horizon=7
    )
    assert span.end == 63
    assert boundary.gap == 2
    assert boundary.purge == 5


def test_boundary_counts_never_exceed_the_span():
    # A 3-row span asked to drop 20 rows loses 3, and says so. Reporting 20 would claim
    # rows were removed that never existed.
    span, boundary = _apply_boundary(
        Span(fold=0, role=SplitRole.TRAIN, start=0, end=3), 3, gap=20, horizon=0
    )
    assert span.rows == 0
    assert boundary.gap == 3
    assert boundary.end == 0


def test_purge_moves_the_tscv_training_span_back():
    spans, boundaries = build_spans(
        100,
        hourly_frame(100)["open_time"],
        SplitType.TSCV_EXPANDING,
        _params(n_folds=4, purge_flag=True, label_horizon_bars=5),
    )
    train = next(s for s in spans if s.fold == 1 and s.role is SplitRole.TRAIN)
    test = next(s for s in spans if s.fold == 1 and s.role is SplitRole.TEST)
    assert test.start == 20
    assert train.end == 15
    assert boundaries[0].purge == 5


def test_purge_with_an_equal_gap_removes_nothing_extra():
    _, boundaries = build_spans(
        100,
        hourly_frame(100)["open_time"],
        SplitType.TSCV_EXPANDING,
        _params(n_folds=4, gap_size=5, purge_flag=True, label_horizon_bars=5),
    )
    assert boundaries[0].gap == 5
    assert boundaries[0].purge == 0


def test_date_splits_ignore_gap_and_purge():
    # build_spans itself never applies them to a date split; the request model is what
    # refuses them. Asserted here so the two cannot drift.
    times = hourly_frame(48)["open_time"]
    spans, boundaries = build_spans(
        48,
        times,
        SplitType.DATE_2WAY,
        _params(
            train_start=times.iloc[0].to_pydatetime(),
            train_end=times.iloc[23].to_pydatetime(),
            test_start=times.iloc[24].to_pydatetime(),
            test_end=times.iloc[47].to_pydatetime(),
            gap_size=5,
        ),
    )
    assert boundaries == []
    assert spans[0].rows == 24


# ------------------------------------------------------------------- refusals


def test_date_split_without_dates_is_refused_by_the_request():
    with pytest.raises(ValueError, match="train_start"):
        SplitCreateRequest(dataset_id="x", split_type=SplitType.DATE_2WAY)


def test_gap_on_a_date_split_is_refused():
    with pytest.raises(ValueError, match="apply to the percentage"):
        SplitCreateRequest(
            dataset_id="x",
            split_type=SplitType.DATE_2WAY,
            parameters=_params(
                train_start=UTC_START,
                train_end=datetime(2024, 1, 2, tzinfo=UTC),
                test_start=datetime(2024, 1, 3, tzinfo=UTC),
                test_end=datetime(2024, 1, 4, tzinfo=UTC),
                gap_size=3,
            ),
        )


def test_purge_without_a_label_horizon_is_refused():
    with pytest.raises(ValueError, match="label_horizon_bars"):
        SplitCreateRequest(
            dataset_id="x",
            split_type=SplitType.PERCENTAGE_70_15_15,
            parameters=_params(purge_flag=True),
        )


def test_overlapping_dates_are_refused():
    # A 2-way split has no validation dates, so the only adjacency between the two spans is
    # train_end → test_start. A fixed six-link chain would skip it and accept this.
    with pytest.raises(ValueError, match="test_start must be after train_end"):
        SplitCreateRequest(
            dataset_id="x",
            split_type=SplitType.DATE_2WAY,
            parameters=_params(
                train_start=UTC_START,
                train_end=datetime(2024, 1, 5, tzinfo=UTC),
                test_start=datetime(2024, 1, 5, tzinfo=UTC),
                test_end=datetime(2024, 1, 9, tzinfo=UTC),
            ),
        )


def test_a_three_way_split_checks_every_adjacent_pair():
    with pytest.raises(ValueError, match="test_start must be after val_end"):
        SplitCreateRequest(
            dataset_id="x",
            split_type=SplitType.DATE_3WAY,
            parameters=_params(
                train_start=UTC_START,
                train_end=datetime(2024, 1, 5, tzinfo=UTC),
                val_start=datetime(2024, 1, 6, tzinfo=UTC),
                val_end=datetime(2024, 1, 9, tzinfo=UTC),
                test_start=datetime(2024, 1, 9, tzinfo=UTC),
                test_end=datetime(2024, 1, 12, tzinfo=UTC),
            ),
        )


def test_a_zoneless_boundary_is_refused():
    with pytest.raises(ValueError, match="UTC offset"):
        SplitParameters(train_start=datetime(2024, 1, 1))


def test_a_nanosecond_free_split_type_is_refused():
    with pytest.raises(ValueError):
        SplitCreateRequest(dataset_id="x", split_type="half_and_half")  # type: ignore[arg-type]


# --------------------------------------------------------------------- service


async def test_create_writes_a_plan_in_outputs(settings, catalog):
    parent = register(settings, catalog, hourly_frame(200))
    created = await _service(settings, catalog).create(
        SplitCreateRequest(dataset_id=parent.dataset_id, split_type=SplitType.PERCENTAGE_70_15_15)
    )
    assert created.dataset.category is DatasetCategory.OUTPUTS
    assert created.dataset.operation == OPERATION
    assert created.dataset.parent_dataset_id == parent.dataset_id
    assert [row.role for row in created.plan] == [
        SplitRole.TRAIN,
        SplitRole.VAL,
        SplitRole.TEST,
    ]
    assert [row.row_count for row in created.plan] == [140, 30, 30]
    assert all(row.split_id == created.dataset.dataset_id for row in created.plan)


async def test_create_does_not_touch_the_source_dataset(settings, catalog):
    parent = register(settings, catalog, hourly_frame(200))
    before = parent.checksum
    await _service(settings, catalog).create(
        SplitCreateRequest(dataset_id=parent.dataset_id, split_type=SplitType.PERCENTAGE_70_15_15)
    )
    after = catalog.get_dataset(parent.dataset_id)
    assert after is not None
    assert after.checksum == before
    assert after.row_count == parent.row_count


async def test_plan_records_the_source_checksum(settings, catalog):
    parent = register(settings, catalog, hourly_frame(200))
    created = await _service(settings, catalog).create(
        SplitCreateRequest(dataset_id=parent.dataset_id, split_type=SplitType.TSCV_EXPANDING)
    )
    assert created.report.dataset_checksum == parent.checksum
    assert {row.dataset_checksum for row in created.plan} == {parent.checksum}


async def test_plan_start_time_is_the_observation_time_not_the_close(settings, catalog):
    frame = hourly_frame(200)
    parent = register(settings, catalog, frame)
    created = await _service(settings, catalog).create(
        SplitCreateRequest(dataset_id=parent.dataset_id, split_type=SplitType.PERCENTAGE_70_15_15)
    )
    train = created.plan[0]
    assert train.start_time == frame["open_time"].iloc[0].to_pydatetime()
    assert train.end_time == frame["open_time"].iloc[139].to_pydatetime()


async def test_report_counts_every_role_and_the_untested_rows(settings, catalog):
    parent = register(settings, catalog, hourly_frame(100))
    created = await _service(settings, catalog).create(
        SplitCreateRequest(
            dataset_id=parent.dataset_id,
            split_type=SplitType.TSCV_EXPANDING,
            parameters=_params(n_folds=4),
        )
    )
    report = created.report
    assert report.rows == 100
    assert report.rows_by_role["train"] > 0
    assert report.rows_tested == 80
    assert report.rows_never_tested == 20
    assert report.folds == 4


async def test_report_gives_the_row_boundaries(settings, catalog):
    parent = register(settings, catalog, hourly_frame(100))
    created = await _service(settings, catalog).create(
        SplitCreateRequest(
            dataset_id=parent.dataset_id,
            split_type=SplitType.TSCV_EXPANDING,
            parameters=_params(n_folds=4, gap_size=2),
        )
    )
    first = created.report.boundaries[0]
    assert first["cut_index"] == 20
    assert first["gap_rows_removed"] == 2
    assert first["effective_end_index"] == 18


async def test_purge_is_reported_when_it_removes_nothing(settings, catalog):
    parent = register(settings, catalog, hourly_frame(100))
    created = await _service(settings, catalog).create(
        SplitCreateRequest(
            dataset_id=parent.dataset_id,
            split_type=SplitType.TSCV_EXPANDING,
            parameters=_params(
                n_folds=4, gap_size=5, purge_flag=True, label_horizon_bars=5
            ),
        )
    )
    assert any("purge_flag removed no rows" in note for note in created.report.warnings)


async def test_unsorted_dataset_is_refused(settings, catalog):
    frame = hourly_frame(200)
    shuffled = frame.iloc[::-1].reset_index(drop=True)
    parent = register(settings, catalog, shuffled, name="reversed")
    with pytest.raises(ValidationFailure, match="not sorted"):
        await _service(settings, catalog).create(
            SplitCreateRequest(
                dataset_id=parent.dataset_id, split_type=SplitType.PERCENTAGE_70_15_15
            )
        )


async def test_repeated_timestamps_are_refused(settings, catalog):
    frame = hourly_frame(200)
    frame.loc[10, "open_time"] = frame.loc[11, "open_time"]
    parent = register(settings, catalog, frame, name="duplicated instant")
    with pytest.raises(ValidationFailure, match="repeated timestamps"):
        await _service(settings, catalog).create(
            SplitCreateRequest(
                dataset_id=parent.dataset_id, split_type=SplitType.PERCENTAGE_70_15_15
            )
        )


async def test_a_text_timestamp_column_is_split_on_its_instants(settings, catalog):
    """An uploaded CSV's time column is text, and a plan still has to be cut on the clock.

    ``pandas.read_csv`` types nothing, so a file whose dates look like dates arrives as a
    ``string`` column — every frame in this file is built with a real ``datetime64`` one, so
    this is the shape the module was written against and the one no other test covers. Both
    readers of the column are exercised: the row-index arithmetic the percentage split uses,
    and ``searchsorted`` in the date split, which compares a string column to a datetime and
    cannot silently succeed.
    """
    frame = hourly_frame(200)
    text = frame.assign(open_time=frame["open_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
    assert text["open_time"].dtype == object
    parent = register(settings, catalog, text, name="uploaded csv")
    service = _service(settings, catalog)

    created = await service.create(
        SplitCreateRequest(
            dataset_id=parent.dataset_id, split_type=SplitType.PERCENTAGE_70_15_15
        )
    )
    first, last = created.plan[0], created.plan[-1]
    assert (first.start_index, first.end_index) == (0, 140)
    # The instants, not the text: a plan whose timestamps were the strings it read would
    # compare and sort, and would then be written to Parquet as text and read back by C7
    # as something that is not a time at all.
    assert first.start_time == frame["open_time"].iloc[0].to_pydatetime()
    assert last.end_time == frame["open_time"].iloc[199].to_pydatetime()

    dated = await service.create(
        SplitCreateRequest(
            dataset_id=parent.dataset_id,
            split_type=SplitType.DATE_2WAY,
            parameters=_params(
                train_start=datetime(2024, 1, 1, tzinfo=UTC),
                train_end=datetime(2024, 1, 4, tzinfo=UTC),
                test_start=datetime(2024, 1, 5, tzinfo=UTC),
                test_end=datetime(2024, 1, 8, tzinfo=UTC),
            ),
        )
    )
    # Both ends inclusive, which is what a date picker means: the 1st through the 4th at
    # 00:00 is 73 hourly bars, and so is the 5th through the 8th.
    assert [(span.role.value, span.start_index, span.end_index) for span in dated.plan] == [
        ("train", 0, 73),
        ("test", 96, 169),
    ]


async def test_an_unparseable_timestamp_is_refused(settings, catalog):
    """Refused, not dropped. A hole in the timeline is a frame that cannot be cut on."""
    frame = hourly_frame(50)
    text = frame.assign(open_time=frame["open_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
    text.loc[7, "open_time"] = "not a date"
    parent = register(settings, catalog, text, name="one bad stamp")
    with pytest.raises(ValidationFailure, match="parseable timestamp"):
        await _service(settings, catalog).create(
            SplitCreateRequest(
                dataset_id=parent.dataset_id, split_type=SplitType.PERCENTAGE_70_15_15
            )
        )


async def test_too_few_rows_for_the_fold_count_is_refused(settings, catalog):
    # 10 folds need 11 blocks of at least 2 bars = 22 rows. The message names the fold
    # count rather than the first span that came out short, because the fold count is the
    # control the user has to change.
    parent = register(settings, catalog, hourly_frame(20), name="tiny")
    with pytest.raises(ValidationFailure, match="10 folds need at least 22 rows"):
        await _service(settings, catalog).create(
            SplitCreateRequest(
                dataset_id=parent.dataset_id,
                split_type=SplitType.TSCV_EXPANDING,
                parameters=_params(n_folds=10),
            )
        )


async def test_an_empty_test_span_is_refused(settings, catalog):
    # 48 hourly bars from 2024-01-01, so the test window falls entirely after the last bar.
    # The plan would be a train span and nothing to score it against.
    parent = register(settings, catalog, hourly_frame(48), name="dated")
    with pytest.raises(ValidationFailure, match="test span empty"):
        await _service(settings, catalog).create(
            SplitCreateRequest(
                dataset_id=parent.dataset_id,
                split_type=SplitType.DATE_2WAY,
                parameters=_params(
                    train_start=datetime(2024, 1, 1, tzinfo=UTC),
                    train_end=datetime(2024, 1, 1, 12, tzinfo=UTC),
                    test_start=datetime(2025, 1, 1, tzinfo=UTC),
                    test_end=datetime(2025, 6, 1, tzinfo=UTC),
                ),
            )
        )


async def test_an_empty_training_span_is_refused(settings, catalog):
    # The mirror case: the training window falls entirely before the first bar, so there is
    # nothing to fit on and a model would be trained on no history at all.
    parent = register(settings, catalog, hourly_frame(48), name="dated")
    with pytest.raises(ValidationFailure, match="training span of 0 row"):
        await _service(settings, catalog).create(
            SplitCreateRequest(
                dataset_id=parent.dataset_id,
                split_type=SplitType.DATE_2WAY,
                parameters=_params(
                    train_start=datetime(2020, 1, 1, tzinfo=UTC),
                    train_end=datetime(2023, 12, 31, tzinfo=UTC),
                    test_start=datetime(2024, 1, 1, tzinfo=UTC),
                    test_end=datetime(2024, 6, 1, tzinfo=UTC),
                ),
            )
        )


async def test_a_missing_dataset_is_a_not_found(settings, catalog):
    with pytest.raises(DatasetNotFound):
        await _service(settings, catalog).create(
            SplitCreateRequest(dataset_id="nope", split_type=SplitType.PERCENTAGE_70_15_15)
        )


async def test_plans_lists_only_this_datasets_plans(settings, catalog):
    first = register(settings, catalog, hourly_frame(200), name="first")
    second = register(settings, catalog, hourly_frame(200), name="second")
    service = _service(settings, catalog)
    await service.create(
        SplitCreateRequest(dataset_id=first.dataset_id, split_type=SplitType.PERCENTAGE_70_15_15)
    )
    await service.create(
        SplitCreateRequest(dataset_id=first.dataset_id, split_type=SplitType.TSCV_EXPANDING)
    )
    assert len(await service.plans(first.dataset_id)) == 2
    assert await service.plans(second.dataset_id) == []


async def test_a_listed_plan_carries_its_type_and_parameters(settings, catalog):
    parent = register(settings, catalog, hourly_frame(100))
    await _service(settings, catalog).create(
        SplitCreateRequest(
            dataset_id=parent.dataset_id,
            split_type=SplitType.TSCV_EXPANDING,
            parameters=_params(n_folds=3, gap_size=2),
        )
    )
    summary = (await _service(settings, catalog).plans(parent.dataset_id))[0]
    assert summary.split_type is SplitType.TSCV_EXPANDING
    assert summary.parameters["n_folds"] == 3
    assert summary.parameters["gap_size"] == 2
    # The folds the plan *holds*, which is what the label in C7's dropdown says and what its
    # fold selector is built from — so it has to be the same number the report gives, and the
    # same number the ask was for. Asserted against `n_folds` rather than a bare 3 or 4: this
    # line previously read 4, which was `max(fold) + 1` over folds labelled 1..3, and it made
    # the page offer a fourth fold that no span in the plan had.
    assert summary.folds == 3 == summary.parameters["n_folds"]


async def test_load_plan_round_trips_every_row(settings, catalog):
    parent = register(settings, catalog, hourly_frame(100))
    service = _service(settings, catalog)
    created = await service.create(
        SplitCreateRequest(
            dataset_id=parent.dataset_id,
            split_type=SplitType.TSCV_EXPANDING,
            parameters=_params(n_folds=3),
        )
    )
    meta, rows = await service.load_plan(created.dataset.dataset_id)
    assert meta.dataset_id == created.dataset.dataset_id
    assert rows == created.plan


async def test_load_plan_refuses_a_dataset_that_is_not_a_plan(settings, catalog):
    parent = register(settings, catalog, hourly_frame(100))
    with pytest.raises(ValidationFailure, match="is not a split plan"):
        await _service(settings, catalog).load_plan(parent.dataset_id)


async def test_load_plan_refuses_an_unknown_id(settings, catalog):
    with pytest.raises(DatasetNotFound):
        await _service(settings, catalog).load_plan("nope")


async def test_time_column_ignores_the_bar_close(settings, catalog):
    """The candidate list excludes `close_time`, and the split follows it.

    Splitting on bar-close instants dates every span one bar late, which moves the last bar
    of training into the test span — a leak with no visible symptom, since the frame is
    unchanged either way.
    """
    parent = register(settings, catalog, hourly_frame(50))
    assert "close_time" in {column.name for column in parent.columns}
    assert _time_column_of(parent) == "open_time"


async def test_a_dataset_with_no_time_column_is_refused(settings, catalog):
    import pandas as pd

    frame = pd.DataFrame({"close": [1.0, 2.0, 3.0], "volume": [1.0, 1.0, 1.0]})
    parent = register(settings, catalog, frame, name="no timeline", timeframe=None)
    with pytest.raises(ValidationFailure, match="no recognisable timestamp"):
        await _service(settings, catalog).create(
            SplitCreateRequest(
                dataset_id=parent.dataset_id, split_type=SplitType.PERCENTAGE_70_15_15
            )
        )


# ------------------------------------------------------------------- lineage


async def test_the_plan_is_lineage_linked_to_its_parent(settings, catalog):
    parent = register(settings, catalog, hourly_frame(100))
    created = await _service(settings, catalog).create(
        SplitCreateRequest(dataset_id=parent.dataset_id, split_type=SplitType.PERCENTAGE_70_15_15)
    )
    # Recorded twice, on purpose: `parent_dataset_id` is the single authoritative parent
    # (`ancestors` walks it), and the lineage row carries the operation and the detail.
    assert created.dataset.parent_dataset_id == parent.dataset_id
    assert catalog.ancestors(created.dataset.dataset_id) == [parent.dataset_id]
    derived = catalog.derived_datasets(parent.dataset_id, OPERATION)
    assert [d.dataset_id for d in derived] == [created.dataset.dataset_id]


async def test_provenance_records_the_parameters(settings, catalog):
    parent = register(settings, catalog, hourly_frame(100))
    created = await _service(settings, catalog).create(
        SplitCreateRequest(
            dataset_id=parent.dataset_id,
            split_type=SplitType.TSCV_ROLLING,
            parameters=_params(n_folds=3, gap_size=1),
        )
    )
    provenance = created.dataset.provenance
    assert provenance["split_type"] == "tscv_rolling"
    assert '"n_folds": 3' in provenance["split_parameters"]
    assert provenance["split_time_column"] == "open_time"
    assert provenance["split_dataset_checksum"] == parent.checksum


async def test_the_stored_plan_table_has_the_specified_columns(settings, catalog):
    parent = register(settings, catalog, hourly_frame(100))
    created = await _service(settings, catalog).create(
        SplitCreateRequest(dataset_id=parent.dataset_id, split_type=SplitType.PERCENTAGE_70_15_15)
    )
    names = {column.name for column in created.dataset.columns}
    assert {"split_id", "role", "fold", "start_time", "end_time", "row_count"} <= names
    assert {"start_index", "end_index", "dataset_checksum"} <= names


def test_plans_for_returns_nothing_for_an_unknown_dataset(catalog):
    assert plans_for(catalog, "nope") == []


async def test_the_plan_gets_an_id_of_its_own(settings, catalog):
    parent = register(settings, catalog, hourly_frame(100))
    created = await _service(settings, catalog).create(
        SplitCreateRequest(dataset_id=parent.dataset_id, split_type=SplitType.PERCENTAGE_70_15_15)
    )
    assert created.dataset.dataset_id != parent.dataset_id
    assert created.dataset.dataset_id
