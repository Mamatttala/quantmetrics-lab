"""Hand-calculated and numerical oracles for the pure C8 engines."""

import numpy as np
import pandas as pd
import pytest

from app.services.exploratory_corr import (
    adjust_pvalues,
    correlation_analysis,
    fisher_interval,
    pair_statistics,
    vif_table,
)
from app.services.exploratory_pca import static_pca, transform_pca


@pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
def test_perfect_dependence(method):
    frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.0, 4.0, 6.0, 8.0]})
    result = pair_statistics(frame, "x", "y", method, min_obs=3)
    assert result["correlation"] == pytest.approx(1.0, abs=1e-12)
    if method == "kendall":
        # The exact two-sided n=4 test has only 1/12 resolution.
        assert result["p_value"] == pytest.approx(1 / 12)
    else:
        assert result["p_value"] < 0.05
    frame["y"] = [4.0, 3.0, 2.0, 1.0]
    assert pair_statistics(frame, "x", "y", method, min_obs=3)["correlation"] == pytest.approx(-1.0)


def test_spearman_nonlinear_and_minimum():
    frame = pd.DataFrame({"x": np.arange(40.0), "y": np.arange(40.0) ** 3})
    assert pair_statistics(frame, "x", "y", "spearman")["correlation"] == pytest.approx(1.0)
    assert pair_statistics(frame.iloc[:10], "x", "y")["correlation"] is None


def test_fisher_and_bh_oracles():
    assert fisher_interval(0.5, 100) == pytest.approx((0.336643304409091, 0.634139840748545))
    assert adjust_pvalues([0.01, 0.04, 0.03, 0.002], "benjamini_hochberg") == pytest.approx(
        [0.02, 0.04, 0.04, 0.008]
    )
    assert adjust_pvalues([None, 0.04], "bonferroni") == [None, 0.04]


def test_pca_rank_one_and_reconstruction():
    frame = pd.DataFrame({"x": np.arange(40.0), "y": np.arange(40.0) * 2})
    result = static_pca(frame, ["x", "y"], n_components=2)
    assert result["explained_variance_ratio"][0] == pytest.approx(1.0)
    assert result["cumulative_explained_variance_ratio"][-1] == 1.0
    scores = transform_pca(frame, result)
    restored = (scores.to_numpy() @ np.array(result["loadings"]).T + result["pca_center"]) * result[
        "scale"
    ] + result["center"]
    np.testing.assert_allclose(restored, frame, atol=1e-12)


def test_vif_independence_and_intercept_contract():
    rng = np.random.default_rng(812)
    frame = pd.DataFrame({"x": rng.normal(size=5000), "y": rng.normal(size=5000)})
    assert all(row["vif"] == pytest.approx(1.0, abs=0.01) for row in vif_table(frame, ["x", "y"]))

    affine = pd.DataFrame({"x": np.arange(1.0, 80.0)})
    affine["y"] = affine["x"] + 20
    with_intercept = vif_table(affine, ["x", "y"], include_intercept=True)[0]["vif"]
    without_intercept = vif_table(affine, ["x", "y"], include_intercept=False)[0]["vif"]
    assert with_intercept == 1e30
    assert without_intercept is not None and without_intercept < with_intercept


def test_partial_correlation_removes_common_driver_and_warns_on_collinearity():
    rng = np.random.default_rng(99)
    z = rng.normal(size=4000)
    frame = pd.DataFrame(
        {"x": z + rng.normal(size=len(z)), "y": z + rng.normal(size=len(z)), "z": z}
    )
    result = correlation_analysis(frame, ["x", "y"], method="partial_pearson", controls=["z"])
    assert result["pairwise"][0]["correlation"] == pytest.approx(0.0, abs=0.04)

    frame["almost_x"] = frame["x"] * 1.00001
    warned = correlation_analysis(
        frame, ["x", "y"], method="partial_pearson", controls=["almost_x"]
    )
    assert any("highly collinear" in warning for warning in warned["warnings"])


def test_kaiser_warns_without_standardization():
    result = static_pca(
        pd.DataFrame({"x": np.arange(40.0), "y": np.arange(40.0) ** 2}),
        ["x", "y"],
        standardizer="none",
        selection="eigenvalue_gt_1",
    )
    assert any("Kaiser" in warning for warning in result["warnings"])
