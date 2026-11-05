"""C8 fitted scope and rolling prefix invariance through the real service."""

import numpy as np
import pandas as pd
import pytest

from app.models.exploratory import CorrelationRequest, PCARequest
from app.models.splits import SplitCreateRequest
from app.services.exploratory import ExploratoryService
from app.services.exploratory_corr import correlation_analysis
from app.services.exploratory_stats import preprocess
from app.services.splits import SplitService
from app.storage import parquet_store
from tests.conftest import register
from tests.test_exploratory import sample


@pytest.mark.parametrize(
    "transform", ["standard_scale", "winsorize_1_99", "robust_scale", "minmax_scale"]
)
@pytest.mark.parametrize("missing", ["median_impute_scope", "mean_impute_scope"])
def test_fitted_preprocessing_cannot_read_test_rows(transform, missing):
    frame = sample()
    frame.loc[10, "x"] = np.nan
    changed = frame.copy()
    changed.loc[100:, ["x", "y"]] = 10000.0
    before = preprocess(
        frame, frame.index[:100], ["x", "y"], pre_transform=transform, missing_policy=missing
    )
    after = preprocess(
        changed, changed.index[:100], ["x", "y"], pre_transform=transform, missing_policy=missing
    )
    assert before[2] == after[2]
    pd.testing.assert_frame_equal(before[0].iloc[:100], after[0].iloc[:100])


@pytest.mark.parametrize(
    "transform", ["none", "standard_scale", "winsorize_1_99", "rank_transform", "first_difference"]
)
async def test_rolling_truncation_including_fitted_preprocessing(settings, catalog, transform):
    frame = sample()
    frame.loc[15, "x"] = np.nan
    service = ExploratoryService(settings, catalog)
    full, truncated = (
        register(settings, catalog, frame),
        register(settings, catalog, frame.iloc[:128]),
    )
    kwargs = dict(
        features=["x", "y"],
        scope="rolling",
        rolling_window=40,
        rolling_step=3,
        rolling_pair_selection="selected_pairs",
        rolling_pairs=[("x", "y")],
        pre_transform=transform,
        missing_policy="mean_impute_scope",
    )
    before = await service.run(CorrelationRequest(dataset_id=full.dataset_id, **kwargs))
    after = await service.run(CorrelationRequest(dataset_id=truncated.dataset_id, **kwargs))
    assert before["rolling_series"][: len(after["rolling_series"])] == after["rolling_series"]
    assert before["causal_for_downstream_training"] is True


async def test_train_pca_parameters_and_scores_ignore_test_perturbations(settings, catalog):
    frame = sample()
    frame.loc[3, "x"] = np.nan
    changed = frame.copy()
    changed.loc[120:, ["x", "y"]] += 5000
    results, outputs = [], []
    service = ExploratoryService(settings, catalog)
    for i, data in enumerate([frame, changed]):
        source = register(settings, catalog, data, name=f"source-{i}")
        split = await SplitService(settings, catalog).create(
            SplitCreateRequest(dataset_id=source.dataset_id, split_type="percentage_70_15_15")
        )
        result = await service.run(
            PCARequest(
                dataset_id=source.dataset_id,
                features=["x", "y"],
                scope="train_span",
                split_plan_id=split.dataset.dataset_id,
                write_dataset=True,
                pre_transform="winsorize_1_99",
                missing_policy="median_impute_scope",
            )
        )
        results.append(result)
        outputs.append(
            parquet_store.read_dataset(settings.data_dir / result["dataset"]["storage_path"])
        )
    for key in ["center", "loadings", "preprocessing_parameters", "eigenvalues"]:
        assert results[0][key] == results[1][key]
    pd.testing.assert_frame_equal(outputs[0].iloc[:112], outputs[1].iloc[:112])
    assert results[0]["causal_for_downstream_training"] is True


def test_conditional_sizes_ccf_and_non_psd():
    frame = sample()
    frame["target"] = frame.x.shift(4)
    report = correlation_analysis(
        frame, ["x", "y"], target="target", max_lag=6, group_column="group", group_min_obs=100
    )
    assert report["ccf_summary"][0]["best_positive_lag"] == 4
    assert any("future target" in warning for warning in report["warnings"])
    assert all(group["group_size"] == 80 and group["warnings"] for group in report["conditional"])
    # Pairwise missing observations yield r_ab=1, r_ac=1, r_bc=-1.
    a = [1, 2, 3, 1, 2, 3, np.nan, np.nan, np.nan]
    b = [1, 2, 3, np.nan, np.nan, np.nan, 1, 2, 3]
    c = [np.nan, np.nan, np.nan, 1, 2, 3, 3, 2, 1]
    report = correlation_analysis(
        pd.DataFrame({"a": a, "b": b, "c": c}), ["a", "b", "c"], min_obs=3
    )
    assert any("positive semi-definite" in warning for warning in report["warnings"])


async def test_full_sample_and_custom_saves_label_and_default_destination(settings, catalog):
    frame = sample()
    source = register(settings, catalog, frame)
    service = ExploratoryService(settings, catalog)
    for scope in ["full_dataset", "custom_date_range"]:
        dates = (
            {"date_start": frame.open_time.iloc[10], "date_end": frame.open_time.iloc[90]}
            if scope == "custom_date_range"
            else {}
        )
        result = await service.run(
            PCARequest(
                dataset_id=source.dataset_id,
                features=["x", "y"],
                scope=scope,
                write_dataset=True,
                new_dataset_name=scope,
                **dates,
            )
        )
        assert result["dataset"]["category"] == "outputs"
        assert result["causal_for_downstream_training"] is False
        assert result["output_dataset_id"] and result["parameter_dataset_id"]
        assert result["warnings"]
