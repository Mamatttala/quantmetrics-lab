"""Log-space HMM with Gaussian, fixed-df Student-t and Gaussian-mixture emissions.

Training uses forward/backward EM. Serving defaults to the forward recursion alone.
Student-t covariances are scale matrices, not the distribution's finite-sample covariance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln, logsumexp
from sklearn.cluster import KMeans

from app.core.errors import ValidationFailure

EPS = 1e-300
RIDGE = 1e-6


def forward(emission, start, transition):
    alpha = np.empty_like(emission)
    alpha[0] = np.log(np.maximum(start, EPS)) + emission[0]
    log_a = np.log(np.maximum(transition, EPS))
    for t in range(1, len(emission)):
        alpha[t] = emission[t] + logsumexp(alpha[t - 1, :, None] + log_a, axis=0)
    return alpha


def backward(emission, transition):
    beta = np.zeros_like(emission)
    log_a = np.log(np.maximum(transition, EPS))
    for t in range(len(emission) - 2, -1, -1):
        beta[t] = logsumexp(log_a + emission[t + 1] + beta[t + 1], axis=1)
    return beta


def normalize(log_values):
    return np.exp(log_values - logsumexp(log_values, axis=1, keepdims=True))


def stochastic(value, shape, name):
    result = np.asarray(value, dtype=float)
    if (
        result.shape != shape
        or not np.isfinite(result).all()
        or (result < 0).any()
        or not np.allclose(result.sum(axis=-1), 1)
    ):
        raise ValidationFailure(
            f"{name} must have shape {shape}, nonnegative entries and rows summing to one."
        )
    return result


@dataclass
class Emissions:
    means: np.ndarray
    covariances: np.ndarray
    weights: np.ndarray
    kind: str
    covariance_type: str
    df: float = 5

    def components(self, x):
        k, m, d = self.means.shape
        logp = np.empty((len(x), k, m))
        latent = np.ones_like(logp)
        for state in range(k):
            for component in range(m):
                cov = self.covariances[state, component]
                delta = x - self.means[state, component]
                sign, logdet = np.linalg.slogdet(cov)
                if sign <= 0:
                    raise ValidationFailure("Emission covariance is not positive definite.")
                mahal = np.einsum("ij,ij->i", delta, np.linalg.solve(cov, delta.T).T)
                if self.kind == "student_t":
                    logp[:, state, component] = (
                        gammaln((self.df + d) / 2)
                        - gammaln(self.df / 2)
                        - 0.5 * (d * np.log(self.df * np.pi) + logdet)
                        - (self.df + d) / 2 * np.log1p(mahal / self.df)
                    )
                    latent[:, state, component] = (self.df + d) / (self.df + mahal)
                else:
                    logp[:, state, component] = -0.5 * (d * np.log(2 * np.pi) + logdet + mahal)
        return logp + np.log(np.maximum(self.weights, EPS))[None, :, :], latent

    def log_probability(self, x):
        return logsumexp(self.components(x)[0], axis=2)

    def update(self, x, gamma):
        components, latent = self.components(x)
        responsibilities = gamma[:, :, None] * np.exp(
            components - logsumexp(components, axis=2, keepdims=True)
        )
        mass = responsibilities.sum(axis=0) + 1e-12
        weighted = responsibilities * latent
        self.means = np.einsum("nkm,nd->kmd", weighted, x) / np.maximum(
            weighted.sum(axis=0)[:, :, None], 1e-12
        )
        for k in range(self.means.shape[0]):
            for m in range(self.means.shape[1]):
                delta = x - self.means[k, m]
                self.covariances[k, m] = (delta.T * weighted[:, k, m]) @ delta / mass[k, m]
        if self.covariance_type == "tied":
            pooled = np.einsum("km,kmij->ij", mass, self.covariances) / mass.sum()
            self.covariances[:] = pooled
        elif self.covariance_type in {"diag", "spherical"}:
            for k in range(self.means.shape[0]):
                for m in range(self.means.shape[1]):
                    diagonal = np.diag(self.covariances[k, m])
                    if self.covariance_type == "spherical":
                        diagonal = np.full(len(diagonal), diagonal.mean())
                    self.covariances[k, m] = np.diag(diagonal)
        self.covariances += RIDGE * np.eye(x.shape[1])
        self.weights = mass / mass.sum(axis=1, keepdims=True)

    def parameter_count(self):
        k, m, d = self.means.shape
        cov = {
            "full": k * m * d * (d + 1) // 2,
            "diag": k * m * d,
            "spherical": k * m,
            "tied": d * (d + 1) // 2,
        }[self.covariance_type]
        return k * m * d + cov + k * (m - 1)


@dataclass
class HMM:
    start: np.ndarray
    transition: np.ndarray
    emission: Emissions
    history: list[float]
    converged: bool

    @property
    def n_states(self):
        return len(self.start)

    def score(self, x, previous=None):
        start = self.start if previous is None else previous @ self.transition
        return float(
            logsumexp(forward(self.emission.log_probability(x), start, self.transition)[-1])
        )

    def predict(self, x, method="filter", lag=5, previous=None):
        logp = self.emission.log_probability(x)
        start = self.start if previous is None else previous @ self.transition
        alpha = forward(logp, start, self.transition)
        if method == "viterbi":
            score = np.empty_like(logp)
            parents = np.zeros(logp.shape, dtype=int)
            score[0] = np.log(np.maximum(start, EPS)) + logp[0]
            for t in range(1, len(x)):
                pair = score[t - 1, :, None] + np.log(np.maximum(self.transition, EPS))
                parents[t] = pair.argmax(axis=0)
                score[t] = pair.max(axis=0) + logp[t]
            states = np.empty(len(x), dtype=int)
            states[-1] = score[-1].argmax()
            for t in range(len(x) - 2, -1, -1):
                states[t] = parents[t + 1, states[t + 1]]
            return states, None
        if method == "smooth":
            alpha = alpha + backward(logp, self.transition)
        elif method == "fixed_lag":
            # At row t, observations through min(t+lag,T) are used: retrospective.
            alpha = alpha + np.array(
                [
                    backward(logp[t : min(t + lag + 1, len(x))], self.transition)[0]
                    for t in range(len(x))
                ]
            )
        elif method != "filter":
            raise ValidationFailure(f"Unknown HMM decoder {method}.")
        probability = normalize(alpha)
        return probability.argmax(axis=1), probability

    def diagnostics(self, n):
        k = self.n_states
        count = k - 1 + k * (k - 1) + self.emission.parameter_count()
        likelihood = self.history[-1]
        return {
            "log_likelihood": likelihood,
            "aic": 2 * count - 2 * likelihood,
            "bic": np.log(n) * count - 2 * likelihood,
            "parameter_count": count,
            "iterations": len(self.history) - 1,
            "converged": self.converged,
            "log_likelihood_history": self.history,
            "emission": self.emission.kind,
            "degrees_of_freedom": self.emission.df if self.emission.kind == "student_t" else None,
        }


def fit_hmm(x, count, parameters, seed, kind="gaussian"):
    n, d = x.shape
    mixtures = int(parameters.get("mixtures", 1))
    if n < count * mixtures * 2:
        raise ValidationFailure(
            "Training window is too small for the requested states and mixtures."
        )
    rng = np.random.default_rng(seed)
    if parameters["initialization"] == "kmeans":
        centers = (
            KMeans(n_clusters=count * mixtures, random_state=seed, n_init=5).fit(x).cluster_centers_
        )
    else:
        centers = x[rng.choice(n, count * mixtures, replace=False)]
    cov = np.atleast_2d(np.cov(x, rowvar=False, ddof=0)) + RIDGE * np.eye(d)
    covariance_type = parameters["covariance_type"]
    if covariance_type == "diag":
        cov = np.diag(np.diag(cov))
    elif covariance_type == "spherical":
        cov = np.eye(d) * np.trace(cov) / d
    emission = Emissions(
        centers.reshape(count, mixtures, d),
        np.tile(cov, (count, mixtures, 1, 1)),
        np.full((count, mixtures), 1 / mixtures),
        kind,
        covariance_type,
        float(parameters.get("degrees_of_freedom", 5)),
    )
    sticky = float(parameters["transition_stickiness"])
    transition = np.full((count, count), (1 - sticky) / (count - 1))
    np.fill_diagonal(transition, sticky)
    if parameters.get("transition_matrix") is not None:
        transition = stochastic(
            parameters["transition_matrix"], (count, count), "Transition matrix"
        )
    start = np.full(count, 1 / count)
    if parameters.get("start_probabilities") is not None:
        start = stochastic(parameters["start_probabilities"], (count,), "Initial probabilities")
    model = HMM(start, transition, emission, [], False)
    for _ in range(int(parameters["max_iter"])):
        logp = emission.log_probability(x)
        alpha = forward(logp, model.start, model.transition)
        beta = backward(logp, model.transition)
        likelihood = float(logsumexp(alpha[-1]))
        model.history.append(likelihood)
        if len(model.history) > 1 and abs(likelihood - model.history[-2]) < parameters["tol"]:
            model.converged = True
            break
        gamma = normalize(alpha + beta)
        xi = (
            np.exp(
                alpha[:-1, :, None]
                + np.log(np.maximum(model.transition, EPS))[None, :, :]
                + logp[1:, None, :]
                + beta[1:, None, :]
                - likelihood
            ).sum(axis=0)
            + 1e-12
        )
        model.start = gamma[0] / gamma[0].sum()
        model.transition = xi / xi.sum(axis=1, keepdims=True)
        emission.update(x, gamma)
    model.history.append(model.score(x))
    return model
