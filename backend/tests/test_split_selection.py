"""Split inputs remain references to observations throughout downstream selection."""

from pathlib import Path

import pandas as pd
import pytest

from app.core.errors import ValidationFailure
from app.models.enums import DatasetCategory, ScalerType, SplitType
from app.models.scaling import ScalingRequest
from app.models.splits import SplitCreateRequest, SplitParameters
from app.services.datasets import DatasetService, persist_dataset
from app.services.scaling import ScalingService
from app.services.splits import SplitService, migrate_legacy_partitions
from app.services.volatility import VolatilityService
from app.storage import parquet_store
from app.storage.catalog import Catalog
from tests.conftest import hourly_frame, register


@pytest.mark.parametrize("split_type", [SplitType.TSCV_EXPANDING, SplitType.TSCV_ROLLING])
@pytest.mark.parametrize("gap,horizon", [(0, 0), (3, 1), (2, 5)])
@pytest.mark.parametrize("include_validation", [None, False, True])
async def test_walk_forward_validation_is_optional_and_boundaries_are_purged(
    settings, catalog, split_type, gap, horizon, include_validation
):
    parent = register(settings, catalog, hourly_frame(203))
    parameters = {
        "n_folds": 3, "gap_size": gap, "purge_flag": horizon > 0,
        "label_horizon_bars": horizon or None,
    }
    if include_validation is not None:
        parameters["include_validation"] = include_validation
    result = await SplitService(settings, catalog).create(SplitCreateRequest(
        dataset_id=parent.dataset_id, split_type=split_type,
        parameters=SplitParameters(**parameters),
    ))
    block = 40 if include_validation else 50
    expected_roles = ["train", "val", "test"] if include_validation else ["train", "test"]
    for fold in range(1, 4):
        spans = [row for row in result.plan if row.fold == fold]
        assert [row.role.value for row in spans] == expected_roles
        assert all(row.row_count > 0 for row in spans)
        for previous, following in zip(spans, spans[1:], strict=False):
            assert following.start_index - previous.end_index == max(gap, horizon)
            assert previous.end_time < following.start_time
        assert spans[0].end_index == fold * block - max(gap, horizon)
        assert spans[0].start_index == (
            (fold - 1) * block if split_type is SplitType.TSCV_ROLLING else 0
        )
        assert spans[-1].start_index == (fold + bool(include_validation)) * block
    assert result.plan[-1].end_index == parent.row_count
    enabled = bool(include_validation)
    assert result.report.parameters["include_validation"] is enabled
    summaries = await SplitService(settings, catalog).plans(parent.dataset_id)
    assert summaries[0].parameters["include_validation"] is enabled
    assert len(result.report.boundaries) == 3 * (2 if enabled else 1)


@pytest.mark.parametrize("split_type", [SplitType.TSCV_EXPANDING, SplitType.TSCV_ROLLING])
async def test_validation_requires_one_more_block_only_when_enabled(settings, catalog, split_type):
    parent = register(settings, catalog, hourly_frame(8))
    service = SplitService(settings, catalog)
    result = await service.create(SplitCreateRequest(
        dataset_id=parent.dataset_id, split_type=split_type,
        parameters=SplitParameters(n_folds=3),
    ))
    assert len(result.plan) == 6
    with pytest.raises(ValidationFailure, match="3 folds need at least 10 rows"):
        await service.create(SplitCreateRequest(
            dataset_id=parent.dataset_id, split_type=split_type,
            parameters=SplitParameters(n_folds=3, include_validation=True),
        ))


async def test_empty_validation_is_refused_before_persisting(settings, catalog):
    parent = register(settings, catalog, hourly_frame(100))
    with pytest.raises(ValidationFailure, match="validation span empty"):
        await SplitService(settings, catalog).create(SplitCreateRequest(
            dataset_id=parent.dataset_id, split_type=SplitType.PERCENTAGE_70_15_15,
            parameters=SplitParameters(gap_size=15),
        ))
    assert not catalog.list_datasets(DatasetCategory.OUTPUTS)


@pytest.mark.parametrize("service_type", [ScalingService, VolatilityService])
async def test_downstream_options_resolve_split_to_source(settings, catalog, service_type):
    parent = register(settings, catalog, hourly_frame(200))
    result = await SplitService(settings, catalog).create(SplitCreateRequest(
        dataset_id=parent.dataset_id, split_type=SplitType.TSCV_EXPANDING,
    ))
    service = service_type(settings, catalog)
    selected_options = await service.options(result.dataset.dataset_id)
    assert selected_options == await service.options(parent.dataset_id)


@pytest.mark.parametrize("split_type", [SplitType.TSCV_EXPANDING, SplitType.TSCV_ROLLING])
async def test_selected_split_scales_only_its_training_rows(settings, catalog, split_type):
    frame = hourly_frame(200)
    frame["close"] = list(range(200))
    parent = register(settings, catalog, frame)
    plan = await SplitService(settings, catalog).create(SplitCreateRequest(
        dataset_id=parent.dataset_id, split_type=split_type,
        parameters=SplitParameters(n_folds=3, gap_size=2, include_validation=True),
    ))
    result = await ScalingService(settings, catalog).run(ScalingRequest(
        dataset_id=plan.dataset.dataset_id, split_plan_id=plan.dataset.dataset_id,
        scaler_type=ScalerType.STANDARD, fold=2, features=["close"],
    ))
    train = next(row for row in plan.plan if row.fold == 2 and row.role.value == "train")
    assert result.report.dataset_id == parent.dataset_id
    assert result.report.rows == 200
    assert result.report.fit_rows == train.row_count
    training_mean = frame.iloc[train.start_index:train.end_index]["close"].mean()
    assert result.parameters[0].center_ == training_mean
    assert result.dataset.parent_dataset_id == parent.dataset_id
    assert result.dataset.provenance["split_plan_id"] == plan.dataset.dataset_id
    assert result.dataset.dataset_id in {
        row.dataset_id for row in catalog.derived_datasets(plan.dataset.dataset_id, "scaling")
    }


@pytest.mark.parametrize("split_type", [SplitType.PERCENTAGE_70_15_15, SplitType.TSCV_EXPANDING])
async def test_saved_partition_reopens_with_its_plan_and_uses_source_training_rows(
    settings, catalog, split_type,
):
    frame = hourly_frame(200)
    frame["close"] = list(range(200))
    parent = register(settings, catalog, frame)
    plan = await SplitService(settings, catalog).create(SplitCreateRequest(
        dataset_id=parent.dataset_id, split_type=split_type,
        parameters=SplitParameters(n_folds=3),
    ))
    partition = next(p for p in plan.partitions if p.provenance["split_role"] == "test")
    # A fresh catalog/service reproduces entering C7 after the splitting session ended.
    reopened_catalog = Catalog(settings.catalog_path)
    reopened_catalog.initialise()
    service = ScalingService(settings, reopened_catalog)
    options = await service.options(partition.dataset_id)
    assert options["defaults"]["split_plan_id"] == plan.dataset.dataset_id
    assert options["plans"][0]["split_plan_id"] == plan.dataset.dataset_id
    assert all(scaler["available"] for scaler in options["scalers"])
    fold = int(partition.provenance["split_fold"]) if split_type.is_tscv else None
    result = await service.run(ScalingRequest(
        dataset_id=partition.dataset_id, scaler_type=ScalerType.STANDARD,
        split_plan_id=plan.dataset.dataset_id, fold=fold, features=["close"],
    ))
    train = next(row for row in plan.plan if row.role.value == "train" and (
        fold is None or row.fold == fold
    ))
    training = frame["close"].iloc[train.start_index:train.end_index]
    observed = pd.read_parquet(partition.storage_path)
    scaled = pd.read_parquet(result.dataset.storage_path)
    pd.testing.assert_series_equal(
        scaled["close"], (observed["close"] - training.mean()) / training.std(ddof=0),
    )
    assert result.report.rows == partition.row_count
    assert result.report.fit_rows == train.row_count
    assert result.report.fit_role == "train"
    assert result.report.fit_dataset_id == parent.dataset_id
    assert result.dataset.parent_dataset_id == partition.dataset_id
    reopened_catalog.close()


async def test_stale_plan_cannot_resolve_to_different_source_bytes(settings, catalog):
    parent = register(settings, catalog, hourly_frame(200))
    plan = await SplitService(settings, catalog).create(SplitCreateRequest(
        dataset_id=parent.dataset_id, split_type=SplitType.PERCENTAGE_70_15_15,
    ))
    catalog.conn.execute(
        "UPDATE datasets SET checksum = 'changed' WHERE dataset_id = ?", (parent.dataset_id,)
    )
    catalog.conn.commit()
    with pytest.raises(ValidationFailure, match="different version"):
        await DatasetService(settings, catalog).resolve_frame(plan.dataset.dataset_id)


async def test_split_plan_metadata_upgrade_preserves_files_and_lineage(settings, catalog):
    parent = register(settings, catalog, hourly_frame(200))
    plan = await SplitService(settings, catalog).create(SplitCreateRequest(
        dataset_id=parent.dataset_id, split_type=SplitType.PERCENTAGE_70_15_15,
    ))
    # A real old-style Outputs artifact, retaining the original plan ID in its rows.
    legacy = persist_dataset(
        parent=parent, frame=parquet_store.read_dataset(Path(plan.dataset.storage_path)),
        data_dir=settings.data_dir, catalog=catalog, category=DatasetCategory.OUTPUTS,
        name="Legacy plan", operation="split", detail={},
        extra_provenance=plan.dataset.provenance,
    )
    ordinary = register(settings, catalog, hourly_frame(10), category=DatasetCategory.OUTPUTS)
    before = parquet_store.read_dataset(Path(legacy.storage_path))
    catalog.initialise()
    catalog.initialise()  # Idempotent on every subsequent startup.
    migrated = catalog.get_dataset(legacy.dataset_id)
    assert migrated.category == DatasetCategory.OUTPUTS
    assert migrated.storage_path == legacy.storage_path
    assert migrated.checksum == legacy.checksum
    assert catalog.ancestors(migrated.dataset_id) == [parent.dataset_id]
    assert parquet_store.read_dataset(Path(migrated.storage_path)).equals(before)
    assert catalog.get_dataset(ordinary.dataset_id).category == DatasetCategory.OUTPUTS
    split_ids = {
        row.dataset_id for row in catalog.list_datasets(DatasetCategory.OUTPUTS)
        if row.operation == "split"
    }
    assert migrated.dataset_id in split_ids
    resolved = await DatasetService(settings, catalog).resolve_frame(migrated.dataset_id)
    assert resolved.dataset_id == parent.dataset_id
    assert resolved.checksum == parent.checksum


async def test_legacy_boundary_only_plan_backfills_observation_partitions(settings, catalog):
    parent = register(settings, catalog, hourly_frame(200))
    created = await SplitService(settings, catalog).create(SplitCreateRequest(
        dataset_id=parent.dataset_id, split_type=SplitType.PERCENTAGE_70_15_15,
    ))
    service = DatasetService(settings, catalog)
    for partition in created.partitions:
        await service.delete(partition.dataset_id)
    catalog.conn.execute(
        "UPDATE datasets SET provenance = json_remove(provenance, "
        "'$.partitions_materialised') WHERE dataset_id = ?",
        (created.dataset.dataset_id,),
    )
    catalog.conn.commit()

    assert migrate_legacy_partitions(settings, catalog) == 3
    restored = catalog.derived_datasets(created.dataset.dataset_id, "split_partition")
    assert {row.provenance["split_role"] for row in restored} == {"train", "val", "test"}
    assert sum(row.row_count for row in restored) == parent.row_count
    assert migrate_legacy_partitions(settings, catalog) == 0
