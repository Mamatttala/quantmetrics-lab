"""Numerical ground truth and causal invariants for the rebuilt D module."""

import itertools

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, multivariate_t

from app.services.regimes.hmm import HMM, Emissions, fit_hmm, forward
from app.services.regimes.inference import confirm
from app.services.regimes.metrics import align, transitions
from app.services.regimes.registry import parameters
from app.services.regimes.structural import cusum, pelt


def test_forward_matches_enumerated_paths():
    probability = np.array([[0.3, 0.7], [0.8, 0.2], [0.4, 0.6]])
    start = np.array([0.6, 0.4])
    transition = np.array([[0.85, 0.15], [0.2, 0.8]])
    total = 0
    for path in itertools.product(range(2), repeat=3):
        joint = start[path[0]] * probability[0, path[0]]
        for t in range(1, 3):
            joint *= transition[path[t - 1], path[t]] * probability[t, path[t]]
        total += joint
    assert np.exp(logsumexp(forward(np.log(probability), start, transition)[-1])) == pytest.approx(
        total
    )


@pytest.mark.parametrize("kind", ["gaussian", "student_t"])
def test_emissions_match_scipy(kind):
    means = np.array([[[0.2, -0.1]]])
    cov = np.array([[[[2, 0.3], [0.3, 0.8]]]])
    x = np.array([[0.1, 0.8], [10, -3]])
    emission = Emissions(means, cov, np.ones((1, 1)), kind, "full", 5)
    expected = (
        multivariate_normal.logpdf(x, means[0, 0], cov[0, 0])
        if kind == "gaussian"
        else multivariate_t.logpdf(x, means[0, 0], cov[0, 0], df=5)
    )
    np.testing.assert_allclose(emission.log_probability(x)[:, 0], expected, atol=1e-12)


@pytest.mark.parametrize("kind", ["gaussian", "student_t"])
@pytest.mark.parametrize("covariance", ["full", "diag", "spherical", "tied"])
def test_hmm_fits_reproducibly_and_filter_is_prefix_invariant(kind, covariance):
    rng = np.random.default_rng(12)
    x = np.r_[rng.normal(-2, 0.3, (45, 2)), rng.normal(2, 0.4, (45, 2))]
    p = parameters(
        "student_t_hmm" if kind == "student_t" else "multivariate_hmm",
        {"max_iter": 30, "covariance_type": covariance},
    )
    model = fit_hmm(x[:65], 2, p, 4, kind)
    clone = fit_hmm(x[:65], 2, p, 4, kind)
    np.testing.assert_allclose(model.emission.means, clone.emission.means)
    states, probabilities = model.predict(x)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    np.testing.assert_allclose(probabilities[:75], model.predict(x[:75])[1], atol=1e-12)
    changed = x.copy()
    changed[75:] += 1000
    np.testing.assert_allclose(probabilities[:75], model.predict(changed)[1][:75], atol=1e-12)
    np.testing.assert_allclose(model.transition.sum(axis=1), 1)
    assert np.isfinite(model.history).all()


def test_smoothing_and_viterbi_agree_with_bruteforce():
    emission = Emissions(
        np.array([[[-0.6]], [[0.6]]]), np.ones((2, 1, 1, 1)), np.ones((2, 1)), "gaussian", "full"
    )
    model = HMM(np.array([0.5, 0.5]), np.array([[0.95, 0.05], [0.1, 0.9]]), emission, [], True)
    x = np.array([[-0.1], [0.1], [1.2], [1.7]])
    logs = emission.log_probability(x)
    paths = list(itertools.product(range(2), repeat=len(x)))
    weights = np.array(
        [
            model.start[path[0]]
            * np.exp(sum(logs[t, path[t]] for t in range(len(x))))
            * np.prod([model.transition[path[t - 1], path[t]] for t in range(1, len(x))])
            for path in paths
        ]
    )
    expected = np.array(
        [
            [weights[[path[t] == k for path in paths]].sum() / weights.sum() for k in range(2)]
            for t in range(len(x))
        ]
    )
    np.testing.assert_allclose(model.predict(x, "smooth")[1], expected)
    assert tuple(model.predict(x, "viterbi")[0]) == paths[weights.argmax()]
    np.testing.assert_allclose(model.predict(x, "fixed_lag", lag=10)[1], expected)
    assert not np.allclose(model.predict(x)[1], expected)


def test_pelt_matches_exhaustive_valid_partitions():
    rng = np.random.default_rng(33)
    for _ in range(8):
        x = rng.normal(size=(10, 2))
        penalty = 1.2
        states, changes, score = pelt(x, penalty, 2)
        best = np.inf
        for flags in itertools.product([False, True], repeat=9):
            edges = [0, *[i + 1 for i, flag in enumerate(flags) if flag], 10]
            if min(np.diff(edges)) < 2:
                continue
            cost = sum(
                ((x[a:b] - x[a:b].mean(axis=0)) ** 2).sum()
                for a, b in zip(edges[:-1], edges[1:], strict=True)
            ) + penalty * (len(edges) - 2)
            best = min(best, cost)
        assert score == pytest.approx(best)


def test_cusum_prefix_and_transition_confirmation():
    x = np.r_[np.zeros((20, 1)), np.full((20, 1), 5.0)]
    states, alerts, _ = cusum(x, np.zeros(1), np.ones(1))
    np.testing.assert_array_equal(states[:28], cusum(x[:28], np.zeros(1), np.ones(1))[0])
    assert alerts[0] >= 20
    raw = np.array([0, 1, 0, 1, 1, 1, 0])
    confirmed, events = confirm(raw, None, minimum=3)
    assert confirmed.tolist() == [0, 0, 0, 0, 0, 1, 1]
    assert events[0]["index"] == 5
    assert events[0]["confirmation_delay_bars"] == 2
    np.testing.assert_array_equal(confirmed[:5], confirm(raw[:5], None, minimum=3)[0])


def test_label_switching_and_transition_estimates():
    reference = np.array([0, 0, 1, 1, 1, 0])
    assert align(reference, 1 - reference)["aligned_agreement"] == 1
    result = transitions(reference, 2)
    np.testing.assert_allclose(result["probabilities"], [[0.5, 0.5], [1 / 3, 2 / 3]])
    assert result["expected_duration"] == pytest.approx([2, 3])
