"""Lineage and immutability: what "raw datasets are permanently read-only" buys us.

The specification's guarantee is only meaningful if it is enforced and observable, so
these tests check both halves: that the parent's bytes are unchanged after a derive,
and that the derived dataset records where it came from.

The last three sections extend the same two questions to C6 and C7, where they are not
about a cell edit any more. A split plan is an artifact *about* a dataset rather than a
copy of it, and a scaled frame is produced from two inputs at once — the source frame and
the plan that says which rows the parameters may be estimated from. Both facts are only
worth anything if the store records them, so they are asserted from the catalog rather
than from the service's return value.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.config import Settings
from app.core.errors import ImmutableDataset
from app.models.dataset import DatasetMeta
from app.models.enums import DatasetCategory, ScalerType, SplitType
from app.models.scaling import ScalingRequest
from app.models.splits import SplitCreateRequest, SplitPlanCreated
from app.services.datasets import CellEdit, DatasetService
from app.services.scaling import MODULE_SOURCE as SCALING_OPERATION
from app.services.scaling import PARAMS_OPERATION, ScalingService
from app.services.splits import OPERATION as SPLIT_OPERATION
from app.services.splits import SplitService
from app.storage import parquet_store
from app.storage.catalog import Catalog
from tests.conftest import hourly_frame, register


@pytest.fixture
def service(settings: Settings, catalog: Catalog) -> DatasetService:
    return DatasetService(settings=settings, catalog=catalog)


@pytest.fixture
def parent(settings: Settings, catalog: Catalog):
    return register(settings, catalog, hourly_frame(120), name="parent")


async def test_derive_leaves_the_parent_checksum_unchanged(
    service: DatasetService, parent, settings: Settings
) -> None:
    before = parquet_store.verify_checksum(
        settings.data_dir / "raw_data" / f"{parent.dataset_id}.parquet"
    )

    child = await service.derive(
        parent.dataset_id,
        name="child",
        edits=[CellEdit(row=0, column="close", value=12345.0)],
    )

    after = parquet_store.verify_checksum(
        settings.data_dir / "raw_data" / f"{parent.dataset_id}.parquet"
    )
    assert before == after, "a derive must not touch the parent's bytes"
    assert before == parent.checksum
    assert child.checksum != parent.checksum


async def test_derive_records_the_parent_and_the_operation(
    service: DatasetService, parent, catalog: Catalog
) -> None:
    child = await service.derive(
        parent.dataset_id,
        name="child",
        operation="edit",
        edits=[CellEdit(row=3, column="close", value=1.0)],
    )
    assert child.parent_dataset_id == parent.dataset_id
    assert child.provenance["derived_from"] == parent.dataset_id
    assert child.provenance["operation"] == "edit"

    # `ancestors` returns dataset ids, oldest-last.
    assert catalog.ancestors(child.dataset_id) == [parent.dataset_id]


async def test_the_edit_actually_lands_in_the_child(
    service: DatasetService, parent
) -> None:
    child = await service.derive(
        parent.dataset_id,
        edits=[CellEdit(row=7, column="close", value=999.5)],
    )
    page = await service.page(child.dataset_id, 7, 1)
    assert page["rows"][0]["close"] == 999.5


async def test_deriving_from_a_child_extends_the_chain(
    service: DatasetService, parent
) -> None:
    child = await service.derive(
        parent.dataset_id, edits=[CellEdit(row=0, column="close", value=1.0)]
    )
    grandchild = await service.derive(
        child.dataset_id, edits=[CellEdit(row=0, column="close", value=2.0)]
    )
    assert grandchild.parent_dataset_id == child.dataset_id
    assert grandchild.parent_dataset_id != parent.dataset_id


async def test_deleting_a_parent_with_children_is_refused(
    service: DatasetService, parent
) -> None:
    """Allowing it would leave the child claiming provenance from a dataset that no
    longer exists — a lineage chain that cannot be verified is worse than none."""
    await service.derive(
        parent.dataset_id, edits=[CellEdit(row=0, column="close", value=1.0)]
    )
    with pytest.raises(ImmutableDataset, match="derived dataset"):
        await service.delete(parent.dataset_id)


async def test_deleting_a_childless_dataset_removes_row_and_file(
    service: DatasetService, parent, settings: Settings, catalog: Catalog
) -> None:
    path = settings.data_dir / "raw_data" / f"{parent.dataset_id}.parquet"
    assert path.exists()

    await service.delete(parent.dataset_id)

    assert not path.exists()
    assert catalog.get_dataset(parent.dataset_id) is None


async def test_derived_dataset_lands_in_processed_data(
    service: DatasetService, parent
) -> None:
    """Category is a statement about provenance. An edited frame does not have the
    provenance its parent claimed, so it must not inherit the parent's bucket."""
    from app.models.enums import DatasetCategory

    child = await service.derive(
        parent.dataset_id, edits=[CellEdit(row=0, column="close", value=1.0)]
    )
    assert child.category == DatasetCategory.PROCESSED


async def test_derived_dataset_inherits_symbol_and_timeframe(
    service: DatasetService, parent
) -> None:
    child = await service.derive(
        parent.dataset_id, edits=[CellEdit(row=0, column="close", value=1.0)]
    )
    assert child.symbol == parent.symbol
    assert child.timeframe == parent.timeframe
    assert child.asset_class == parent.asset_class


async def test_row_count_survives_a_derive(service: DatasetService, parent) -> None:
    """An edit changes a value, not the shape. A row count that moved would mean the
    column operations mutated more than they were asked to."""
    child = await service.derive(
        parent.dataset_id, edits=[CellEdit(row=0, column="close", value=1.0)]
    )
    assert child.row_count == parent.row_count


# ------------------------------------------------------- C6 · the plan's own lineage


@pytest.fixture
def splits(settings: Settings, catalog: Catalog) -> SplitService:
    return SplitService(settings=settings, catalog=catalog)


@pytest.fixture
def scaling(settings: Settings, catalog: Catalog) -> ScalingService:
    return ScalingService(settings=settings, catalog=catalog)


async def test_a_plan_is_a_child_of_the_dataset_it_describes(
    splits: SplitService, parent, catalog: Catalog
) -> None:
    """A plan names row ranges in someone else's frame, so the two facts that matter are
    that it points at that frame and that it can prove which bytes it was computed on —
    without the second, `start_index` is a claim about a dataset that may have changed."""
    created = await splits.create(
        SplitCreateRequest(dataset_id=parent.dataset_id, split_type=SplitType.PERCENTAGE_70_15_15)
    )
    plan = created.dataset

    assert plan.parent_dataset_id == parent.dataset_id
    assert plan.category == DatasetCategory.OUTPUTS
    assert plan.provenance["operation"] == SPLIT_OPERATION
    assert plan.provenance["split_dataset_checksum"] == parent.checksum
    assert catalog.ancestors(plan.dataset_id) == [parent.dataset_id]
    derived = catalog.derived_datasets(parent.dataset_id, SPLIT_OPERATION)
    assert plan.dataset_id in {row.dataset_id for row in derived}


async def test_the_plan_carries_its_own_id_as_a_column(splits: SplitService, parent) -> None:
    """One identifier for one object. The plan table is exportable, and a copy of it that
    could not be traced back to the row it came from would be a plan with no provenance —
    which is the whole reason the id is written inside the frame as well."""
    created = await splits.create(
        SplitCreateRequest(dataset_id=parent.dataset_id, split_type=SplitType.PERCENTAGE_70_15_15)
    )
    stored = pd.read_parquet(created.dataset.storage_path)

    assert "split_id" in stored.columns
    assert set(stored["split_id"]) == {created.dataset.dataset_id}
    # And the same id appears on every row it returned, so the response and the file agree.
    assert {row.split_id for row in created.plan} == {created.dataset.dataset_id}


async def test_planning_leaves_the_source_and_the_store_unchanged(
    splits: SplitService, parent, settings: Settings, catalog: Catalog
) -> None:
    """C6 computes a plan; it does not cut the frame. A second plan over the same dataset
    therefore starts from exactly the same bytes, and the source is still one dataset."""
    path = parquet_store.dataset_path(
        settings.data_dir, DatasetCategory.RAW, parent.dataset_id
    )
    before = parquet_store.verify_checksum(path)
    columns_before = [column.name for column in parent.columns]

    first = await splits.create(
        SplitCreateRequest(dataset_id=parent.dataset_id, split_type=SplitType.PERCENTAGE_70_15_15)
    )
    second = await splits.create(
        SplitCreateRequest(dataset_id=parent.dataset_id, split_type=SplitType.PERCENTAGE_60_20_20)
    )

    assert parquet_store.verify_checksum(path) == before == parent.checksum
    reloaded = catalog.get_dataset(parent.dataset_id)
    assert reloaded is not None
    # A plan must not have grown a column on the frame it plans: the row ranges live in the
    # plan, and a `role` column on the source would be a second, drifting source of truth.
    assert [column.name for column in reloaded.columns] == columns_before
    # Two plans are two datasets, not one overwritten.
    assert first.dataset.dataset_id != second.dataset.dataset_id


# -------------------------------------- C7 · two parents on one frame


async def _planned(
    splits: SplitService,
    parent: DatasetMeta,
    split_type: SplitType = SplitType.PERCENTAGE_70_15_15,
) -> SplitPlanCreated:
    return await splits.create(
        SplitCreateRequest(dataset_id=parent.dataset_id, split_type=split_type)
    )


async def test_a_scaled_frame_records_the_plan_as_a_second_parent(
    splits: SplitService, scaling: ScalingService, parent, catalog: Catalog
) -> None:
    """The scaled frame has two inputs, and the one a reader would not guess is the plan:
    it is not merged into the numbers, it decides which rows produced them. `ancestors`
    follows the single parent column, so the plan is found on the lineage table instead —
    which is exactly why the extra input is recorded there and not only in provenance."""
    plan = await _planned(splits, parent)
    result = await scaling.run(
        ScalingRequest(
            dataset_id=parent.dataset_id,
            split_plan_id=plan.dataset.dataset_id,
            scaler_type=ScalerType.STANDARD,
            features=["close"],
        )
    )
    assert result.dataset is not None

    edges = {
        (row["parent_id"], row["operation"])
        for row in catalog.conn.execute(
            "SELECT parent_id, operation FROM lineage WHERE child_id = ?",
            (result.dataset.dataset_id,),
        ).fetchall()
    }
    assert edges == {
        (parent.dataset_id, SCALING_OPERATION),
        (plan.dataset.dataset_id, SCALING_OPERATION),
    }
    assert result.dataset.provenance["split_plan_id"] == plan.dataset.dataset_id
    assert result.dataset.provenance["scaled_features"] == "close"


async def test_the_parameters_are_their_own_dataset_beside_the_frame(
    splits: SplitService, scaling: ScalingService, parent, catalog: Catalog
) -> None:
    """The fitted parameters are what makes a prediction invertible, so they are stored as
    a dataset like everything else: checksummed, linked, exportable. They hang off the
    *scaled frame*, not off the source, because they describe the frame they produced."""
    plan = await _planned(splits, parent)
    result = await scaling.run(
        ScalingRequest(
            dataset_id=parent.dataset_id,
            split_plan_id=plan.dataset.dataset_id,
            scaler_type=ScalerType.STANDARD,
            features=["close"],
        )
    )
    params = result.params_dataset
    assert params is not None
    assert params.category == DatasetCategory.OUTPUTS
    assert params.provenance["operation"] == PARAMS_OPERATION
    assert params.parent_dataset_id == result.dataset.dataset_id
    assert params.provenance["split_plan_id"] == plan.dataset.dataset_id
    assert catalog.ancestors(params.dataset_id) == [
        result.dataset.dataset_id,
        parent.dataset_id,
    ]

    table = pd.read_parquet(params.storage_path)
    # One row per rescaled feature, and the inversion pair the report shows.
    assert list(table["feature"]) == ["close"]
    assert {"center_", "scale_"} <= set(table.columns)


async def test_scaling_leaves_both_of_its_inputs_untouched(
    splits: SplitService, scaling: ScalingService, parent, settings: Settings
) -> None:
    """Neither input is rewritten. The scaled frame is a new dataset, and the plan that
    said which rows to fit on is still the plan — a run that edited it would make every
    earlier run's provenance a lie."""
    source_path = parquet_store.dataset_path(
        settings.data_dir, DatasetCategory.RAW, parent.dataset_id
    )
    plan = await _planned(splits, parent)
    plan_path = parquet_store.dataset_path(
        settings.data_dir, DatasetCategory.OUTPUTS, plan.dataset.dataset_id
    )
    plan_before = parquet_store.verify_checksum(plan_path)

    result = await scaling.run(
        ScalingRequest(
            dataset_id=parent.dataset_id,
            split_plan_id=plan.dataset.dataset_id,
            scaler_type=ScalerType.STANDARD,
            features=["close"],
        )
    )

    assert parquet_store.verify_checksum(source_path) == parent.checksum
    assert parquet_store.verify_checksum(plan_path) == plan_before
    assert result.dataset is not None
    assert result.dataset.dataset_id not in {parent.dataset_id, plan.dataset.dataset_id}
