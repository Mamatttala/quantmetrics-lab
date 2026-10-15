"""Server-side paging, sorting, and the SQL-identifier guard.

The grid's contract is that the backend sends exactly one window and the browser
never holds the file. The tests that matter are therefore about *coverage*: that the
windows tile the dataset exactly, with no gap and no row served twice. A paging bug
that dropped or duplicated rows would be invisible in the UI and would quietly
corrupt any statistic computed downstream.
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.core.errors import ValidationFailure
from app.models.enums import DatasetCategory
from app.services.datasets import CellEdit, ColumnOp, DatasetService
from app.storage import duckdb_engine, parquet_store
from app.storage.catalog import Catalog
from tests.conftest import hourly_frame, register

PAGE = 100


@pytest.fixture
def service(settings: Settings, catalog: Catalog) -> DatasetService:
    return DatasetService(settings=settings, catalog=catalog)


@pytest.fixture
def dataset_id(settings: Settings, catalog: Catalog) -> str:
    """500 hourly rows — exactly five pages at the grid's default size."""
    return register(settings, catalog, hourly_frame(500)).dataset_id


async def test_pages_tile_the_dataset_exactly(
    service: DatasetService, dataset_id: str
) -> None:
    seen: list[int] = []
    offset = 0
    total = None

    while True:
        payload = await service.page(dataset_id, offset, PAGE)
        total = payload["total"]
        if not payload["rows"]:
            break
        seen.extend(range(payload["offset"], payload["offset"] + len(payload["rows"])))
        offset += PAGE

    assert total == 500
    assert len(seen) == 500
    # No row served twice, no row skipped. This is the assertion the whole design
    # exists to make possible.
    assert len(set(seen)) == 500
    assert seen == list(range(500))


async def test_page_beyond_the_end_returns_empty_not_an_error(
    service: DatasetService, dataset_id: str
) -> None:
    payload = await service.page(dataset_id, 10_000, PAGE)
    assert payload["rows"] == []
    assert payload["total"] == 500


async def test_limit_is_capped_at_the_configured_maximum(
    service: DatasetService, dataset_id: str
) -> None:
    """A browser asking for the whole file must be told what it actually got.

    The echoed `limit` is the contract: the frontend computes page counts from it, so
    silently honouring a smaller limit than requested would make the pager lie.
    """
    payload = await service.page(dataset_id, 0, 10_000_000)
    assert payload["limit"] == service.settings.max_page_size
    assert len(payload["rows"]) == service.settings.max_page_size


async def test_sort_is_applied_across_the_whole_dataset_not_the_page(
    settings: Settings, catalog: Catalog, service: DatasetService
) -> None:
    """The first row of page 1 of a descending sort must be the global maximum.

    Sorting only the loaded page would produce a plausible-looking ordering that is
    simply wrong — the failure mode this test exists to prevent.
    """
    import numpy as np

    frame = hourly_frame(300)
    frame["close"] = np.arange(300, dtype="float64")
    dataset_id = register(settings, catalog, frame).dataset_id

    desc = await service.page(dataset_id, 0, PAGE, "close:desc")
    assert desc["rows"][0]["close"] == 299.0
    assert desc["rows"][-1]["close"] == 200.0

    asc = await service.page(dataset_id, 0, PAGE, "close:asc")
    assert asc["rows"][0]["close"] == 0.0
    assert asc["rows"][-1]["close"] == 99.0


async def test_sort_pages_do_not_overlap(
    service: DatasetService, dataset_id: str
) -> None:
    first = await service.page(dataset_id, 0, PAGE, "close:desc")
    second = await service.page(dataset_id, PAGE, PAGE, "close:desc")
    first_values = [row["close"] for row in first["rows"]]
    second_values = [row["close"] for row in second["rows"]]
    assert set(first_values).isdisjoint(second_values)
    assert min(first_values) >= max(second_values)


async def test_unknown_sort_column_is_rejected(
    service: DatasetService, dataset_id: str
) -> None:
    with pytest.raises(ValidationFailure, match="unknown column"):
        await service.page(dataset_id, 0, PAGE, "close; DROP TABLE datasets")


async def test_bad_sort_direction_is_rejected(
    service: DatasetService, dataset_id: str
) -> None:
    with pytest.raises(ValidationFailure, match="asc.*desc"):
        await service.page(dataset_id, 0, PAGE, "close:sideways")


async def test_negative_offset_and_zero_limit_are_rejected(
    service: DatasetService, dataset_id: str
) -> None:
    with pytest.raises(ValidationFailure):
        await service.page(dataset_id, -1, PAGE)
    with pytest.raises(ValidationFailure):
        await service.page(dataset_id, 0, 0)


def test_identifier_quoting_handles_hostile_column_names(
    settings: Settings, catalog: Catalog
) -> None:
    """Column names come from provider payloads and user uploads.

    They are validated against the real schema *and* quoted, so a column legitimately
    named `select` — or one containing a quote — works rather than breaking or
    injecting.
    """
    import pandas as pd

    frame = pd.DataFrame({'weird "name"': [1.0, 2.0], "select": [3.0, 4.0]})
    written = parquet_store.write_dataset(
        frame, settings.data_dir, DatasetCategory.UPLOADED, parquet_store.new_dataset_id()
    )
    rows, total, names = duckdb_engine.page(
        written.storage_path, 0, 10, 'weird "name":desc'
    )
    assert total == 2
    assert names == ['weird "name"', "select"]
    assert rows[0]['weird "name"'] == 2.0


async def test_preview_head_and_tail_do_not_overlap(
    service: DatasetService, dataset_id: str
) -> None:
    payload = await service.preview(dataset_id, 5, 5)
    head_times = {row["open_time"] for row in payload["head"]}
    tail_times = {row["open_time"] for row in payload["tail"]}
    assert payload["row_count"] == 500
    assert len(head_times) == 5
    assert len(tail_times) == 5
    assert head_times.isdisjoint(tail_times)


async def test_derived_column_ops_and_edits_compose(
    service: DatasetService, dataset_id: str
) -> None:
    """Column operations are applied before cell edits, and the edit addresses the
    *new* name. Anything else would make an edit to a renamed column land nowhere."""
    child = await service.derive(
        dataset_id,
        name="edited",
        operation="test",
        edits=[CellEdit(row=0, column="volume_x2", value=42.0)],
        column_ops=[
            ColumnOp(op="rename", column="volume", new_name="volume_x2"),
            ColumnOp(op="drop", column="close_time"),
            ColumnOp(op="add", column="note", new_name="note"),
        ],
    )
    names = [c.name for c in child.columns]
    assert "volume_x2" in names
    assert "volume" not in names
    assert "close_time" not in names
    assert names[-1] == "note"
    assert child.category == DatasetCategory.PROCESSED


async def test_column_op_on_a_missing_column_is_refused(
    service: DatasetService, dataset_id: str
) -> None:
    with pytest.raises(ValidationFailure, match="missing column"):
        await service.derive(
            dataset_id,
            column_ops=[ColumnOp(op="drop", column="not_a_column")],
        )


async def test_rename_onto_an_existing_name_is_refused(
    service: DatasetService, dataset_id: str
) -> None:
    with pytest.raises(ValidationFailure, match="already exists"):
        await service.derive(
            dataset_id,
            column_ops=[ColumnOp(op="rename", column="volume", new_name="close")],
        )


async def test_a_column_can_be_dropped_and_another_renamed_into_its_name(
    service: DatasetService, dataset_id: str
) -> None:
    """Replacing a redundant column with a renamed one — drop `close_time`, rename
    `volume` to `close_time` — is a normal tidy-up, and the two operations interact:
    the rename is refused while the old name is still occupied. Applying them in the
    order they were submitted makes that save impossible to express."""
    child = await service.derive(
        dataset_id,
        column_ops=[
            ColumnOp(op="rename", column="volume", new_name="close_time"),
            ColumnOp(op="drop", column="close_time"),
        ],
    )
    names = [c.name for c in child.columns]
    assert names.count("close_time") == 1
    assert "volume" not in names


async def test_two_renames_can_shift_names_along_in_one_save(
    service: DatasetService, dataset_id: str
) -> None:
    """Renaming `close`->`settle` and `volume`->`close` in one save, which shifts a
    name along the way a user does when they add a column to the middle of a schema.

    The grid allows this: it checks the new name against the columns on screen, and
    by the time the second rename is made the first has already freed `close`. Applied
    one rename at a time the backend refused the second — `close` was still occupied by
    the column whose own rename had not run yet — so the user got "a column named
    'close' already exists" while looking at a grid with no such column on it. Renames
    are applied as one simultaneous mapping, so the pair is expressible and the order
    the two arrive in cannot matter.
    """
    child = await service.derive(
        dataset_id,
        column_ops=[
            ColumnOp(op="rename", column="close", new_name="settle"),
            ColumnOp(op="rename", column="volume", new_name="close"),
        ],
    )
    names = [c.name for c in child.columns]
    assert names.count("close") == 1
    assert names.count("settle") == 1
    assert "volume" not in names


async def test_shifting_renames_does_not_depend_on_the_order_they_arrive_in(
    service: DatasetService, dataset_id: str
) -> None:
    """The same argument as the reorder test above: a diff has no canonical order."""
    forwards = await service.derive(
        dataset_id,
        column_ops=[
            ColumnOp(op="rename", column="close", new_name="settle"),
            ColumnOp(op="rename", column="volume", new_name="close"),
        ],
    )
    backwards = await service.derive(
        dataset_id,
        column_ops=[
            ColumnOp(op="rename", column="volume", new_name="close"),
            ColumnOp(op="rename", column="close", new_name="settle"),
        ],
    )
    assert [c.name for c in forwards.columns] == [c.name for c in backwards.columns]


async def test_two_renames_onto_the_same_name_are_refused(
    service: DatasetService, dataset_id: str
) -> None:
    """Simultaneous renames must still not be able to produce a duplicate. Checking
    the names one at a time would let both through, because neither target is occupied
    until the other lands."""
    with pytest.raises(ValidationFailure, match="already exists"):
        await service.derive(
            dataset_id,
            column_ops=[
                ColumnOp(op="rename", column="close", new_name="merged"),
                ColumnOp(op="rename", column="volume", new_name="merged"),
            ],
        )


async def test_renaming_a_column_to_its_own_name_is_a_no_op(
    service: DatasetService, dataset_id: str
) -> None:
    """A rename that changes nothing must not be mistaken for a collision with
    itself."""
    parent = await service.get(dataset_id)
    child = await service.derive(
        dataset_id,
        column_ops=[ColumnOp(op="rename", column="close", new_name="close")],
    )
    assert [c.name for c in child.columns] == [c.name for c in parent.columns]


async def test_a_dropped_name_can_be_reused_by_an_added_column(
    service: DatasetService, dataset_id: str
) -> None:
    child = await service.derive(
        dataset_id,
        column_ops=[
            ColumnOp(op="drop", column="volume"),
            ColumnOp(op="add", column="volume", new_name="volume"),
        ],
    )
    names = [c.name for c in child.columns]
    assert names.count("volume") == 1
    assert names[-1] == "volume", "the added column is appended, not slotted into place"


async def test_columns_can_be_moved_to_the_screen_order(
    service: DatasetService, dataset_id: str
) -> None:
    """Dragging a column in the grid must survive the save.

    The grid has always *shown* the new order, so a save that quietly dropped it left
    the user looking at one schema and storing another — a difference nothing in the
    UI would ever reveal.
    """
    parent = await service.get(dataset_id)
    before = [c.name for c in parent.columns]
    wanted = [before[3], before[0], *before[1:3], *before[4:]]

    child = await service.derive(
        dataset_id,
        column_ops=[
            ColumnOp(op="reorder", column=name, position=index)
            for index, name in enumerate(wanted)
        ],
    )
    assert [c.name for c in child.columns] == wanted


async def test_a_reorder_does_not_depend_on_the_order_the_ops_arrive_in(
    service: DatasetService, dataset_id: str
) -> None:
    """These operations are a diff computed by the UI, so the same diff can be
    serialised in any order. Applying them as a sequence of moves would make the
    result depend on that serialisation — a contract that holds for exactly one
    caller is one the next caller breaks."""
    parent = await service.get(dataset_id)
    before = [c.name for c in parent.columns]
    wanted = [before[-1], *before[:-1]]

    forwards = await service.derive(
        dataset_id,
        column_ops=[
            ColumnOp(op="reorder", column=name, position=index)
            for index, name in enumerate(wanted)
        ],
    )
    backwards = await service.derive(
        dataset_id,
        column_ops=[
            ColumnOp(op="reorder", column=name, position=index)
            for index, name in reversed(list(enumerate(wanted)))
        ],
    )
    assert [c.name for c in forwards.columns] == wanted
    assert [c.name for c in backwards.columns] == wanted


async def test_a_reorder_names_the_column_as_it_is_after_a_rename(
    service: DatasetService, dataset_id: str
) -> None:
    """A reorder carries a place in the final schema, so it has to be resolvable
    there — which means the rename must have happened first."""
    parent = await service.get(dataset_id)
    before = [c.name for c in parent.columns]
    # What survives the rename and the drop: the first five columns, plus the renamed
    # one under its new name.
    wanted = ["close_time", *before[:5]]

    child = await service.derive(
        dataset_id,
        column_ops=[
            ColumnOp(op="rename", column="volume", new_name="close_time"),
            ColumnOp(op="drop", column="close_time"),
            *[
                ColumnOp(op="reorder", column=name, position=index)
                for index, name in enumerate(wanted)
            ],
        ],
    )
    assert [c.name for c in child.columns] == wanted


async def test_reordering_a_missing_column_is_refused(
    service: DatasetService, dataset_id: str
) -> None:
    with pytest.raises(ValidationFailure, match="reorder missing column"):
        await service.derive(
            dataset_id, column_ops=[ColumnOp(op="reorder", column="nope", position=0)]
        )


async def test_a_reorder_without_a_position_is_refused(
    service: DatasetService, dataset_id: str
) -> None:
    with pytest.raises(ValidationFailure, match="non-negative position"):
        await service.derive(
            dataset_id, column_ops=[ColumnOp(op="reorder", column="volume")]
        )


async def test_a_reorder_that_only_lists_some_columns_moves_those_to_the_front(
    service: DatasetService, dataset_id: str
) -> None:
    """The frontend sends a total order, but a partial one is not an error: the
    unmentioned columns keep their relative places behind the ones that were moved."""
    child = await service.derive(
        dataset_id, column_ops=[ColumnOp(op="reorder", column="volume", position=0)]
    )
    names = [c.name for c in child.columns]
    assert names[0] == "volume"
    assert "volume" not in names[1:]


async def test_edit_outside_the_row_range_is_refused(
    service: DatasetService, dataset_id: str
) -> None:
    """A paste that runs past the end of the dataset must fail loudly.

    Clamping or dropping the excess would let a user believe a value was saved when
    it was discarded.
    """
    with pytest.raises(ValidationFailure, match="outside the dataset"):
        await service.derive(
            dataset_id, edits=[CellEdit(row=999_999, column="close", value=1.0)]
        )


async def test_empty_derive_is_refused(
    service: DatasetService, dataset_id: str
) -> None:
    with pytest.raises(ValidationFailure, match="Nothing to save"):
        await service.derive(dataset_id)


async def test_unsorted_pages_are_in_the_file_row_order_that_edits_address(
    settings: Settings, catalog: Catalog, service: DatasetService
) -> None:
    """Positional cell edits are only sound because these two orders agree.

    The grid sends an edit as ``offset + index-within-page``, and
    ``_apply_cell_edits`` resolves that number with ``frame.iat[row, ...]`` against
    ``pandas.read_parquet``. The row the user actually clicked, though, came from
    DuckDB. If DuckDB's unordered scan disagreed with pandas about row order, an edit
    would silently land on a different row than the one on screen — corruption with no
    error attached to it.

    Two readers agreeing is not something to assume: applying a sort is precisely the
    case where they diverge, which is why the grid locks editing while sorted. What
    this test pins is the other half — that with no sort, the page the browser sees is
    the file order ``iat`` addresses.
    """
    import pandas as pd

    frame = hourly_frame(250)
    # A column whose value is its own row number: any reordering shows up as a
    # mismatch instead of hiding behind values that happen to be equal.
    frame["row_id"] = range(250)
    written = register(settings, catalog, frame)

    from_disk = pd.read_parquet(written.storage_path)["row_id"].tolist()
    assert from_disk == list(range(250))

    # Two consecutive pages, so the arithmetic is checked across a page boundary and
    # not just within one window.
    paged: list[int] = []
    for offset in (0, PAGE):
        payload = await service.page(written.dataset_id, offset, PAGE)
        paged.extend(int(row["row_id"]) for row in payload["rows"])

    assert paged == from_disk[: 2 * PAGE]
