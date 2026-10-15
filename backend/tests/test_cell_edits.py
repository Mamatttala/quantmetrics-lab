"""Cell edits: what a value is allowed to do to a column's type.

The grid's editor is a text box, so an edit frequently arrives as a string even for a
numeric column. Two failure modes are possible and both are silent, which is why they
are pinned here:

* the column is *widened* — a `num_trades` column that has quietly become `object`
  breaks every numeric operation built on it and nothing in the grid would show it;
* the value is *coerced* the wrong way — `"false"` into a boolean column is `True`
  under numpy's "non-empty string" rule, which inverts the value without complaint.

Refusing an edit the column cannot represent is the third outcome, and the only
honest one.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.config import Settings
from app.core.errors import ValidationFailure
from app.services.datasets import CellEdit, DatasetService
from app.storage.catalog import Catalog
from tests.conftest import register


@pytest.fixture
def service(settings: Settings, catalog: Catalog) -> DatasetService:
    return DatasetService(settings=settings, catalog=catalog)


@pytest.fixture
def dataset_id(settings: Settings, catalog: Catalog) -> str:
    """A frame carrying one column of each type an edit has to respect."""
    frame = pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-01", periods=20, freq="1h", tz="UTC"),
            "close": [100.0 + i for i in range(20)],
            "num_trades": [1000 + i for i in range(20)],
            "settled": [True] * 20,
            "note": ["original"] * 20,
        }
    )
    return register(settings, catalog, frame).dataset_id


async def _dtypes(service: DatasetService, dataset_id: str) -> dict[str, str]:
    meta = await service.get(dataset_id)
    return {c.name: c.dtype for c in meta.columns}


async def _cell(service: DatasetService, dataset_id: str, row: int, column: str):
    page = await service.page(dataset_id, row, 1)
    return page["rows"][0][column]


async def test_a_numeric_string_lands_as_a_number_not_as_text(
    service: DatasetService, dataset_id: str
) -> None:
    """The common case: the editor hands back "123.5" for a float column."""
    child = await service.derive(
        dataset_id, edits=[CellEdit(row=2, column="close", value="123.5")]
    )
    assert await _cell(service, child.dataset_id, 2, "close") == 123.5


async def test_an_edit_never_widens_the_column(
    service: DatasetService, dataset_id: str
) -> None:
    """A column whose dtype moved would break every numeric operation built on it,
    and no cell in the grid would look any different afterwards."""
    before = await _dtypes(service, dataset_id)
    child = await service.derive(
        dataset_id,
        edits=[
            CellEdit(row=0, column="close", value="101.25"),
            CellEdit(row=1, column="num_trades", value="2001"),
            CellEdit(row=2, column="settled", value="false"),
            CellEdit(row=3, column="note", value="edited"),
        ],
    )
    assert await _dtypes(service, child.dataset_id) == before


async def test_unparseable_text_in_a_numeric_column_is_refused(
    service: DatasetService, dataset_id: str
) -> None:
    """Not a 500: numpy's `ValueError` is caught and reported as something the user
    can act on."""
    with pytest.raises(ValidationFailure, match="close"):
        await service.derive(dataset_id, edits=[CellEdit(row=0, column="close", value="abc")])


async def test_a_fraction_in_an_integer_column_is_refused(
    service: DatasetService, dataset_id: str
) -> None:
    """`pd.Series([1.5], dtype="int64")` truncates to 1 without saying so, so a price
    typed into a trade-count column would silently become a plausible wrong number."""
    with pytest.raises(ValidationFailure, match="whole number"):
        await service.derive(
            dataset_id, edits=[CellEdit(row=0, column="num_trades", value=1.5)]
        )


async def test_a_blank_in_an_integer_column_is_refused(
    service: DatasetService, dataset_id: str
) -> None:
    """The grid's "Clear cell" and "Cut" both send nulls. `int64` has no missing
    value, so accepting one would change the column's type for every row."""
    with pytest.raises(ValidationFailure, match="num_trades"):
        await service.derive(
            dataset_id, edits=[CellEdit(row=0, column="num_trades", value=None)]
        )


async def test_the_word_false_does_not_become_true(
    service: DatasetService, dataset_id: str
) -> None:
    """numpy reads any non-empty string as truthy, so the generic conversion path
    would invert this value rather than reject it."""
    child = await service.derive(
        dataset_id, edits=[CellEdit(row=4, column="settled", value="false")]
    )
    assert await _cell(service, child.dataset_id, 4, "settled") is False


async def test_a_blank_in_a_float_column_is_stored_as_a_null(
    service: DatasetService, dataset_id: str
) -> None:
    """The one way to blank a cell that is representable, and the page must show it
    as null rather than as zero."""
    child = await service.derive(
        dataset_id, edits=[CellEdit(row=5, column="close", value=None)]
    )
    assert await _cell(service, child.dataset_id, 5, "close") is None


async def test_a_timestamp_string_is_converted_to_the_column_zone(
    service: DatasetService, dataset_id: str
) -> None:
    child = await service.derive(
        dataset_id,
        edits=[CellEdit(row=6, column="open_time", value="2024-06-01T12:00:00+05:00")],
    )
    stored = await _cell(service, child.dataset_id, 6, "open_time")
    assert stored.startswith("2024-06-01T07:00:00")


async def test_an_unparseable_timestamp_is_refused(
    service: DatasetService, dataset_id: str
) -> None:
    with pytest.raises(ValidationFailure, match="open_time"):
        await service.derive(
            dataset_id, edits=[CellEdit(row=6, column="open_time", value="not a date")]
        )


async def test_a_valid_edit_is_passed_through_unchanged(
    service: DatasetService, dataset_id: str
) -> None:
    """The conversion must not be the thing that alters a value — including for an
    object column, where anything goes."""
    child = await service.derive(
        dataset_id,
        edits=[
            CellEdit(row=7, column="close", value=999.5),
            CellEdit(row=7, column="num_trades", value=12345),
            CellEdit(row=7, column="note", value="free text"),
        ],
    )
    assert await _cell(service, child.dataset_id, 7, "close") == 999.5
    assert await _cell(service, child.dataset_id, 7, "num_trades") == 12345
    assert await _cell(service, child.dataset_id, 7, "note") == "free text"
