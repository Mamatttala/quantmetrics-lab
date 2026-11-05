"""C8 arithmetic, causal scope, persistence and observation-time contracts."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from app.core.errors import ValidationFailure
from app.models.enums import DatasetCategory, SplitType
from app.models.exploratory import CorrelationRequest, PCARequest
from app.models.splits import SplitCreateRequest
from app.services.exploratory import ExploratoryService
from app.services.exploratory_stats import (
    correlation_analysis,
    pair_statistics,
    static_pca,
    transform_pca,
    vif_table,
)
from app.services.splits import SplitService
from app.storage import parquet_store
from tests.conftest import register


def sample(rows=160):
    rng = np.random.default_rng(818)
    x = rng.normal(size=rows)
    times = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "open_time": times,
            "close_time": times + pd.Timedelta(days=500),
            "x": x,
            "y": 0.8 * x + rng.normal(size=rows) * 0.5,
            "z": rng.normal(size=rows),
            "group": ["a", "b"] * (rows // 2),
        }
    )


def test_correlation_correctness_and_pairwise_missing():
    frame = sample()
    frame.loc[0, "x"] = np.nan
    frame.loc[1, "y"] = np.inf
    result = correlation_analysis(frame, ["x", "y", "z"])
    pair = result["pairwise"][0]
    expected = stats.pearsonr(frame.x.iloc[2:], frame.y.iloc[2:])
    assert pair["n_used"] == 158
    assert pair["correlation"] == pytest.approx(expected.statistic)
    assert pair["p_value"] == pytest.approx(expected.pvalue)
    assert pair["ci_low"] < pair["correlation"] < pair["ci_high"]
    assert result["matrix"][0][1] == result["matrix"][1][0]


@pytest.mark.parametrize("method", ["spearman", "kendall"])
def test_rank_correlations(method):
    frame = sample()
    pair = pair_statistics(frame, "x", "y", method)
    expected = (
        stats.spearmanr(frame.x, frame.y)
        if method == "spearman"
        else stats.kendalltau(frame.x, frame.y)
    )
    assert pair["correlation"] == pytest.approx(expected.statistic)
    assert pair["p_value"] == pytest.approx(expected.pvalue)
    assert pair["ci_low"] is None


def test_partial_correlation_controls_and_conditional_groups():
    frame = sample()
    frame["x"] += 12 * frame.z
    frame["y"] += 12 * frame.z
    result = correlation_analysis(
        frame, ["x", "y"], method="partial_pearson", controls=["z"], group_column="group"
    )
    design = np.column_stack([np.ones(len(frame)), frame.z])
    residuals = (
        frame[["x", "y"]].to_numpy()
        - design @ np.linalg.lstsq(design, frame[["x", "y"]], rcond=None)[0]
    )
    assert result["pairwise"][0]["correlation"] == pytest.approx(np.corrcoef(residuals.T)[0, 1])
    assert result["pairwise"][0]["controls_rank"] == 1
    assert len(result["conditional"]) == 2
    ranked = correlation_analysis(frame, ["x", "y"], method="partial_spearman", controls=["z"])
    assert ranked["pairwise"][0]["correlation"] is not None


def test_vif_correctness_exact_redundancy_and_constant():
    frame = sample()
    r = frame.x.corr(frame.y)
    result = vif_table(frame, ["x", "y"])
    assert all(row["vif"] == pytest.approx(1 / (1 - r**2)) for row in result)
    frame["duplicate"] = frame.x * 2
    assert vif_table(frame, ["x", "duplicate"])[0]["status"] == "infinite"
    frame["constant"] = 2.0
    assert vif_table(frame, ["constant", "y"])[0]["status"] == "constant"
    group = correlation_analysis(frame, ["x", "duplicate"])["redundancy_groups"][0]
    assert group["members"] == ["duplicate", "x"]
    assert group["representative"] == "duplicate"


def test_ccf_lag_direction_and_rolling_prefix_invariance():
    frame = sample()
    frame["y"] = frame.x.shift(2)
    result = correlation_analysis(
        frame,
        ["x"],
        target="y",
        max_lag=4,
        rolling_window=10,
        time_column="open_time",
        rolling_pair_selection="selected_pairs",
        rolling_pairs=[("x", "y")],
    )
    best = max(result["ccf"], key=lambda row: row["correlation"])
    assert best["lag"] == 2
    assert best["correlation"] == pytest.approx(1)
    prefix = correlation_analysis(
        frame.iloc[:100],
        ["x"],
        target="y",
        rolling_window=10,
        time_column="open_time",
        rolling_pair_selection="selected_pairs",
        rolling_pairs=[("x", "y")],
    )
    assert result["rolling"][:100] == prefix["rolling"]
    modified = frame.copy()
    modified.loc[50:, "y"] += 1000
    changed = correlation_analysis(
        modified,
        ["x"],
        target="y",
        rolling_window=10,
        time_column="open_time",
        rolling_pair_selection="selected_pairs",
        rolling_pairs=[("x", "y")],
    )
    # Even the current row must not reach its own diagnostic.
    assert result["rolling"][:51] == changed["rolling"][:51]


def test_pca_determinism_sign_and_reconstruction():
    frame = sample()
    first = static_pca(frame, ["x", "y", "z"], n_components=3)
    assert first == static_pca(frame, ["x", "y", "z"], n_components=3)
    vectors = np.array(first["loadings"])
    assert all(vector[np.argmax(np.abs(vector))] >= 0 for vector in vectors.T)
    np.testing.assert_allclose(vectors.T @ vectors, np.eye(3), atol=1e-14)
    transformed = transform_pca(frame, first).to_numpy()
    reconstructed = (transformed @ vectors.T + first["pca_center"]) * first["scale"] + first[
        "center"
    ]
    np.testing.assert_allclose(reconstructed, frame[["x", "y", "z"]], atol=1e-14)
    np.testing.assert_allclose(transformed.var(axis=0, ddof=1), first["eigenvalues"])
    assert first["cumulative_explained_variance_ratio"][-1] == 1


def test_sign_tie_uses_first_feature():
    frame = pd.DataFrame({"x": [-2.0, -1.0, 1.0, 2.0], "y": [2.0, 1.0, -1.0, -2.0]})
    result = static_pca(frame, ["x", "y"], n_components=1)
    vector = np.array(result["loadings"])[:, 0]
    assert vector[np.argmax(abs(vector))] > 0
    assert result == static_pca(frame, ["x", "y"], n_components=1)


@pytest.mark.parametrize("standardizer", ["none", "standard", "robust", "minmax"])
def test_component_threshold_and_scalers(standardizer):
    result = static_pca(
        sample(),
        ["x", "y", "z"],
        standardizer=standardizer,
        selection="variance_threshold",
        variance_threshold=0.9,
    )
    k = result["n_components"]
    assert result["cumulative_explained_variance_ratio"][k - 1] >= 0.9
    if k > 1:
        assert result["cumulative_explained_variance_ratio"][k - 2] < 0.9


async def test_full_sample_save_lineage_immutability_and_diagnostics(settings, catalog):
    source = register(settings, catalog, sample())
    path = settings.data_dir / source.storage_path
    before = path.read_bytes()
    service = ExploratoryService(settings, catalog)
    request = PCARequest(dataset_id=source.dataset_id, features=["x", "y", "z"])
    diagnostics = await service.run(request)
    assert diagnostics["dataset"] is None and diagnostics["params_dataset"] is None
    assert len(catalog.list_datasets()) == 1
    saved = await service.run(
        request.model_copy(
            update={
                "write_dataset": True,
                "carry_columns": ["group"],
                "keep_original_features": True,
            }
        )
    )
    assert saved["loadings"] == diagnostics["loadings"]
    assert saved["fit_scope"] == "full_dataset"
    assert saved["causal_for_downstream_training"] is False
    assert path.read_bytes() == before
    assert catalog.get_dataset(source.dataset_id).checksum == source.checksum
    for key, operation in [
        ("dataset", "pca_dimensionality_reduction"),
        ("params_dataset", "pca_parameters"),
    ]:
        meta = catalog.get_dataset(saved[key]["dataset_id"])
        assert meta.operation == operation
        assert meta.parent_dataset_id == source.dataset_id
        assert meta.provenance["causal_for_downstream_training"] == "false"
        assert json.loads(meta.provenance["pca_parameters"])["loadings"] == saved["loadings"]
        edges = catalog.conn.execute(
            "SELECT parent_id, operation FROM lineage WHERE child_id = ?", (meta.dataset_id,)
        ).fetchall()
        assert (source.dataset_id, operation) in [tuple(row) for row in edges]
    output = parquet_store.read_dataset(settings.data_dir / saved["dataset"]["storage_path"])
    assert set(output.columns) == {"open_time", "pc_1", "pc_2", "x", "y", "z", "group"}
    pd.testing.assert_series_equal(output.open_time, sample().open_time)
    assert len(catalog.list_datasets()) == 3


async def test_train_scope_fits_only_train_and_links_plan(settings, catalog):
    frame = sample()
    source = register(settings, catalog, frame)
    plan = await SplitService(settings, catalog).create(
        SplitCreateRequest(dataset_id=source.dataset_id, split_type=SplitType.PERCENTAGE_70_15_15)
    )
    request = PCARequest(
        dataset_id=source.dataset_id,
        features=["x", "y", "z"],
        scope="train_span",
        split_plan_id=plan.dataset.dataset_id,
        write_dataset=True,
    )
    result = await ExploratoryService(settings, catalog).run(request)
    train = next(row for row in plan.plan if row.role.value == "train")
    expected = static_pca(frame.iloc[train.start_index : train.end_index], request.features)
    assert result["center"] == expected["center"]
    assert result["loadings"] == expected["loadings"]
    assert result["n_fit_rows"] == train.row_count
    assert result["causal_for_downstream_training"] is True
    assert result["dataset"]["row_count"] == len(frame)
    for key in ["dataset", "params_dataset"]:
        rows = catalog.conn.execute(
            "SELECT parent_id FROM lineage WHERE child_id = ?", (result[key]["dataset_id"],)
        ).fetchall()
        assert {row[0] for row in rows} == {source.dataset_id, plan.dataset.dataset_id}


async def test_strict_refuses_noncaual_save_but_allows_diagnostics(settings, catalog):
    source = register(settings, catalog, sample())
    request = PCARequest(dataset_id=source.dataset_id, features=["x", "y"], strict_causal=True)
    service = ExploratoryService(settings, catalog)
    await service.run(request)
    with pytest.raises(ValidationFailure, match="Strict causal"):
        await service.run(request.model_copy(update={"write_dataset": True}))
    assert len(catalog.list_datasets()) == 1


async def test_saved_pca_uses_canonical_names_and_always_writes_parameters(settings, catalog):
    source = register(settings, catalog, sample(), name="friendly source")
    service = ExploratoryService(settings, catalog)
    request = PCARequest(
        dataset_id=source.dataset_id,
        features=["x", "y"],
        write_dataset=True,
    )
    first = await service.run(request)
    second = await service.run(request)
    assert first["dataset"]["name"] != second["dataset"]["name"]
    assert "pca_full_dataset" in first["dataset"]["name"]
    assert first["params_dataset"]["operation"] == "pca_parameters"
    assert first["parameter_dataset_id"] == first["params_dataset"]["dataset_id"]

    custom = await service.run(request.model_copy(update={"name": "custom PCA"}))
    assert custom["dataset"]["name"] == "custom PCA"
    with pytest.raises(ValidationFailure, match="already exists"):
        await service.run(request.model_copy(update={"name": "custom PCA"}))


async def test_custom_range_uses_open_time_never_close_time(settings, catalog):
    frame = sample()
    source = register(settings, catalog, frame)
    service = ExploratoryService(settings, catalog)
    result = await service.run(
        PCARequest(
            dataset_id=source.dataset_id,
            features=["x", "y"],
            scope="custom_date_range",
            start_time=frame.open_time.iloc[10],
            end_time=frame.open_time.iloc[29],
        )
    )
    assert result["n_fit_rows"] == 20
    assert result["time_column"] == "open_time"
    assert result["causal_for_downstream_training"] is False
    assert not {"open_time", "close_time"} & set(
        (await service.options(source.dataset_id))["features"]
    )
    no_time = register(settings, catalog, frame.drop(columns="open_time"))
    # No observation timeline still permits ordinary full-sample diagnostics.
    await service.run(PCARequest(dataset_id=no_time.dataset_id, features=["x", "y"]))
    with pytest.raises(ValidationFailure, match="observation/time"):
        await service.run(
            PCARequest(dataset_id=no_time.dataset_id, features=["x", "y"], write_dataset=True)
        )


async def test_all_categories_and_split_tables_are_directly_explorable(settings, catalog):
    service = ExploratoryService(settings, catalog)
    for category in DatasetCategory:
        source = register(settings, catalog, sample(), category=category)
        result = await service.run(
            CorrelationRequest(dataset_id=source.dataset_id, features=["x", "y"])
        )
        assert result["dataset_id"] == source.dataset_id
    options = await service.options()
    assert {row["category"] for row in options["datasets"]} == {c.value for c in DatasetCategory}


async def test_missing_pca_rows_preserve_alignment(settings, catalog):
    frame = sample()
    frame.loc[4, "x"] = np.nan
    frame.loc[14, "y"] = np.inf
    source = register(settings, catalog, frame)
    result = await ExploratoryService(settings, catalog).run(
        PCARequest(dataset_id=source.dataset_id, features=["x", "y"], write_dataset=True)
    )
    assert result["n_fit_rows"] == len(frame) - 2
    stored = parquet_store.read_dataset(settings.data_dir / result["dataset"]["storage_path"])
    assert stored.loc[[4, 14], ["pc_1", "pc_2"]].isna().all().all()
    pd.testing.assert_series_equal(stored.open_time, frame.open_time)


async def test_walk_forward_requires_single_fold_and_wrong_parent_refused(settings, catalog):
    source = register(settings, catalog, sample())
    plan = await SplitService(settings, catalog).create(
        SplitCreateRequest(dataset_id=source.dataset_id, split_type=SplitType.TSCV_EXPANDING)
    )
    request = PCARequest(
        dataset_id=source.dataset_id,
        features=["x", "y"],
        scope="train_span",
        split_plan_id=plan.dataset.dataset_id,
    )
    service = ExploratoryService(settings, catalog)
    with pytest.raises(ValidationFailure, match="fold"):
        await service.run(request)
    result = await service.run(request.model_copy(update={"fold": 1}))
    assert result["causal_for_downstream_training"] is True
    other = register(settings, catalog, sample())
    with pytest.raises(ValidationFailure, match="selected dataset"):
        await service.run(request.model_copy(update={"dataset_id": other.dataset_id, "fold": 1}))
