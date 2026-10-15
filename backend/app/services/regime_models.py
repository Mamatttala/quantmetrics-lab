"""The statistical core of Module D.

Two fitted models live here — a Gaussian hidden Markov model and a Gaussian mixture —
plus the rule that gives their latent states names. Nothing in this module touches
pandas, FastAPI, or the storage layer; it takes arrays and returns arrays and numbers,
which is what makes the causality property testable without standing up a server.

**Why the HMM is implemented here rather than imported.**

The specification names ``hmmlearn``, and the obvious reading is that we should call it.
We do not, for a reason that is about correctness rather than packaging.

A regime label is only usable in a backtest if it was knowable at the bar it is stamped
on. The quantity that answers "what state is the market in, given what has happened so
far" is the *filtered* posterior ``P(s_t | x_1..x_t)``, and it comes from the forward
recursion alone. What HMM libraries expose — ``hmmlearn``'s ``predict_proba`` included —
is the *smoothed* posterior ``P(s_t | x_1..x_T)``, from forward-backward, which conditions
on bars that had not printed yet when ``t`` was live. Smoothed state estimates are the
single most common way a regime study acquires look-ahead bias, and they are invisible in
the output: the frame looks identical either way, and the backtest simply performs
better than reality.

So the forward pass has to be written regardless of which library is installed, and once
it is, the library is contributing a fit we can also write in eighty lines we control.
The specification's own verification requirement — truncate the series to 80%, refit, and
demand the first 80% be unchanged — cannot pass on smoothed output from any library, and
does pass on this, provided the refit schedule is anchored to absolute bar index rather
than to series length. That anchoring is enforced by the caller, not here.

**Parameter estimation does use smoothing, and that is not a contradiction.** The
Baum-Welch E-step needs ``xi``, which needs the backward pass, over the *training* window.
Estimating parameters from a completed window is what fitting means; the constraint is
that the window must end at or before the bar being labelled, which the walk-forward
driver in ``regime_engine`` guarantees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from app.core.errors import ValidationFailure
from app.models.enums import REGIME_PROBABILITY_COLUMNS, RegimeState

#: Ridge added to every fitted covariance. A state that has collapsed onto a handful of
#: near-identical bars otherwise produces a singular covariance, and the log-density then
#: goes to +inf and takes the whole recursion with it. The value is small enough to leave
#: well-conditioned fits untouched and large enough to keep the Cholesky factorisation
#: valid when they are not.
_COVARIANCE_RIDGE = 1e-6

#: Floor for a probability before taking its logarithm. A state whose responsibility
#: underflows to zero would send ``log(0)`` to -inf and poison the M-step averages with
#: NaN; clamping keeps the state alive at negligible cost instead of deleting it.
_LOG_FLOOR = 1e-300


def _log_emissions(x: np.ndarray, means: np.ndarray, covars: np.ndarray) -> np.ndarray:
    """Log Gaussian density of every observation under every state.

    Returns a ``(T, K)`` array. Computed in log space because the density of a
    high-dimensional observation under a tight covariance is far below the smallest
    representable float, and ``log(0) * anything`` is NaN.
    """
    n_obs, n_features = x.shape
    n_states = means.shape[0]
    out = np.empty((n_obs, n_states), dtype=float)
    normaliser = n_features * math.log(2.0 * math.pi)

    for k in range(n_states):
        covar = covars[k]
        sign, log_det = np.linalg.slogdet(covar)
        if sign <= 0:
            raise ValidationFailure(
                "A fitted state has a non-positive-definite covariance matrix.",
                hint="This usually means a feature is constant across the series. "
                "Drop it, or select one that varies.",
            )
        centred = x - means[k]
        # solve() rather than inv(): one triangular solve per state instead of forming
        # an inverse we would immediately discard.
        whitened = np.linalg.solve(covar, centred.T).T
        mahalanobis = np.einsum("td,td->t", centred, whitened)
        out[:, k] = -0.5 * (normaliser + log_det + mahalanobis)

    return out


def _forward_log(
    log_emission: np.ndarray, log_weights: np.ndarray, log_transition: np.ndarray
) -> np.ndarray:
    """The forward recursion, in log space.

    ``log_alpha[t, k] = log P(s_t = k, x_1..x_t)``. This is the half of the algorithm
    that is allowed to see the future, and it is the half we serve.
    """
    n_obs = log_emission.shape[0]
    log_alpha = np.empty_like(log_emission)
    log_alpha[0] = log_weights + log_emission[0]
    for t in range(1, n_obs):
        log_alpha[t] = (
            logsumexp(log_alpha[t - 1][:, None] + log_transition, axis=0) + log_emission[t]
        )
    return log_alpha


def _backward_log(log_emission: np.ndarray, log_transition: np.ndarray) -> np.ndarray:
    """The backward recursion, in log space: ``log P(x_{t+1}..x_T | s_t = k)``.

    Used only to fit parameters, never to produce an output column.
    """
    n_obs = log_emission.shape[0]
    log_beta = np.zeros_like(log_emission)
    for t in range(n_obs - 2, -1, -1):
        log_beta[t] = logsumexp(
            log_transition + (log_emission[t + 1] + log_beta[t + 1])[None, :], axis=1
        )
    return log_beta


@dataclass(frozen=True)
class FittedGaussianHMM:
    """A fitted Gaussian HMM. Immutable once produced."""

    weights: np.ndarray  # (K,)   initial-state distribution
    transition: np.ndarray  # (K, K) row-stochastic
    means: np.ndarray  # (K, d)
    covars: np.ndarray  # (K, d, d)
    log_likelihood: float
    n_iter: int
    converged: bool

    @property
    def n_states(self) -> int:
        return int(self.means.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.means.shape[1])

    @property
    def n_parameters(self) -> int:
        """Free parameters, by the standard count.

        ``(K-1)`` for the initial distribution, ``K(K-1)`` for the transition rows (each
        row sums to one), ``K*d`` for the means, and ``K*d(d+1)/2`` for the symmetric
        covariances. Getting this wrong changes BIC by a constant factor of the penalty
        and silently biases model selection, so it is stated once, here.
        """
        k, d = self.n_states, self.n_features
        return (k - 1) + k * (k - 1) + k * d + k * d * (d + 1) // 2

    def bic(self, n_obs: int) -> float:
        return -2.0 * self.log_likelihood + self.n_parameters * math.log(n_obs)

    def aic(self, n_obs: int) -> float:
        return -2.0 * self.log_likelihood + 2.0 * self.n_parameters


def _covariance_of(block: np.ndarray, fallback: np.ndarray, ridge: float) -> np.ndarray:
    """Sample covariance of a block, regularised, with a fallback for degenerate blocks.

    A quantile group can contain one row (covariance undefined) or several identical
    rows (covariance zero). Neither is an error worth refusing over, because it only
    affects the *initial* parameter guess, which EM then moves away from.
    """
    if block.shape[0] < 2:
        return fallback.copy()
    covar = np.atleast_2d(np.cov(block.T, ddof=0)).astype(float, copy=True)
    covar += ridge * np.eye(block.shape[1])
    return covar


def _initial_parameters(
    x: np.ndarray, n_states: int, ridge: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A deterministic starting point, from quantiles of the first observation.

    Deliberately not random. EM finds a local optimum, so a random start makes the same
    request return different regime labels on different runs, and a research artifact
    that is not reproducible is not usable. Splitting the ordered first feature into K
    equal groups gives separated, ordered initial means — the direction that matters most
    for a regime fit, since the first feature is a return and the states are meant to
    differ by drift.
    """
    n_obs, n_features = x.shape
    order = np.argsort(x[:, 0], kind="stable")
    groups = np.array_split(order, n_states)

    means = np.stack([x[group].mean(axis=0) for group in groups])
    overall = np.atleast_2d(np.cov(x.T, ddof=0)).astype(float) + ridge * np.eye(n_features)
    covars = np.stack(
        [_covariance_of(x[group], overall, ridge) for group in groups]
    )

    weights = np.full(n_states, 1.0 / n_states)
    # Persistence is the empirically correct prior for a regime model: states are
    # expected to last, and starting from near-uniform transitions makes the first
    # E-step almost uninformative and the fit slow.
    transition = np.full((n_states, n_states), 0.1 / max(n_states - 1, 1))
    np.fill_diagonal(transition, 0.9)
    return weights, transition, means, covars


def fit_gaussian_hmm(
    x: np.ndarray,
    n_states: int,
    *,
    max_iter: int = 200,
    tol: float = 1e-5,
    ridge: float = _COVARIANCE_RIDGE,
) -> FittedGaussianHMM:
    """Fit by Baum-Welch (EM). Deterministic: no seed is taken and none is needed."""
    if x.ndim != 2:
        raise ValueError("Expected a 2-D observation matrix.")
    n_obs, n_features = x.shape
    if n_obs < n_states * 2:
        raise ValidationFailure(
            f"Too few training bars ({n_obs}) to fit {n_states} states.",
            hint="Each state needs several observations to estimate a covariance. "
            "Widen the training window or reduce the number of states.",
        )
    if not np.isfinite(x).all():
        raise ValidationFailure("Training window contains non-finite observations.")

    weights, transition, means, covars = _initial_parameters(x, n_states, ridge)
    identity = np.eye(n_features)
    previous = -np.inf
    converged = False
    n_iter = 0
    log_likelihood = -np.inf

    for iteration in range(1, max_iter + 1):
        n_iter = iteration
        log_emission = _log_emissions(x, means, covars)
        log_weights = np.log(np.maximum(weights, _LOG_FLOOR))
        log_transition = np.log(np.maximum(transition, _LOG_FLOOR))

        log_alpha = _forward_log(log_emission, log_weights, log_transition)
        log_beta = _backward_log(log_emission, log_transition)

        log_likelihood = float(logsumexp(log_alpha[-1]))
        if not np.isfinite(log_likelihood):
            raise ValidationFailure(
                "The likelihood diverged while fitting.",
                hint="This usually means a feature has an extreme scale. Consider "
                "standardising it before detection.",
            )

        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        log_xi = (
            log_alpha[:-1, :, None]
            + log_transition[None, :, :]
            + (log_emission[1:] + log_beta[1:])[:, None, :]
        )
        log_xi -= logsumexp(log_xi, axis=(1, 2), keepdims=True)
        xi = np.exp(log_xi)

        # ---- M-step
        occupancy = gamma.sum(axis=0)
        safe_occupancy = np.maximum(occupancy, _LOG_FLOOR)

        weights = gamma[0] / max(gamma[0].sum(), _LOG_FLOOR)

        transition_counts = xi.sum(axis=0)
        row_totals = np.maximum(gamma[:-1].sum(axis=0), _LOG_FLOOR)
        transition = transition_counts / row_totals[:, None]
        # A state that was never occupied keeps its previous row rather than acquiring a
        # row of NaN. It stays in the model and can be picked up by a later E-step.
        dead = ~np.isfinite(transition).all(axis=1) | (transition.sum(axis=1) <= 0.0)
        if dead.any():
            transition[dead] = 1.0 / n_states
        transition /= transition.sum(axis=1, keepdims=True)

        means = (gamma.T @ x) / safe_occupancy[:, None]

        for k in range(n_states):
            centred = x - means[k]
            weighted = centred * gamma[:, k : k + 1]
            covars[k] = (weighted.T @ centred) / max(occupancy[k], _LOG_FLOOR) + ridge * identity

        if abs(log_likelihood - previous) < tol:
            converged = True
            break
        previous = log_likelihood

    return FittedGaussianHMM(
        weights=weights,
        transition=transition,
        means=means,
        covars=covars,
        log_likelihood=log_likelihood,
        n_iter=n_iter,
        converged=converged,
    )


def filtered_posteriors(x: np.ndarray, model: FittedGaussianHMM) -> np.ndarray:
    """``P(s_t | x_1..x_t)`` for every ``t`` — the causal state estimate.

    This is the function the module exists to expose. Everything printed as a regime
    probability comes from here and nowhere else.
    """
    log_emission = _log_emissions(x, model.means, model.covars)
    log_alpha = _forward_log(
        log_emission,
        np.log(np.maximum(model.weights, _LOG_FLOOR)),
        np.log(np.maximum(model.transition, _LOG_FLOOR)),
    )
    log_gamma = log_alpha - logsumexp(log_alpha, axis=1, keepdims=True)
    return np.exp(log_gamma)


def hmm_state_sequence(x: np.ndarray, model: FittedGaussianHMM) -> np.ndarray:
    """The most likely state at each bar under the *filtered* posterior.

    Deliberately not Viterbi. Viterbi finds the single most likely *path*, which is a
    different and non-causal object: the most likely path through ``t`` can change when
    bar ``t+1`` prints. The per-bar argmax of the filtered posterior is what a live
    system could have produced.
    """
    return np.argmax(filtered_posteriors(x, model), axis=1)


# ------------------------------------------------------------------ Gaussian mixture


@dataclass(frozen=True)
class FittedGaussianMixture:
    """A fitted GMM, kept in the same shape as the HMM so callers can treat them alike."""

    means: np.ndarray  # (K, d)
    covars: np.ndarray  # (K, d, d)
    weights: np.ndarray  # (K,)
    log_likelihood: float
    n_iter: int
    converged: bool

    @property
    def n_states(self) -> int:
        return int(self.means.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.means.shape[1])

    @property
    def n_parameters(self) -> int:
        k, d = self.n_states, self.n_features
        return (k - 1) + k * d + k * d * (d + 1) // 2

    def bic(self, n_obs: int) -> float:
        return -2.0 * self.log_likelihood + self.n_parameters * math.log(n_obs)

    def aic(self, n_obs: int) -> float:
        return -2.0 * self.log_likelihood + 2.0 * self.n_parameters


def fit_gaussian_mixture(
    x: np.ndarray, n_states: int, *, seed: int = 0, max_iter: int = 200
) -> FittedGaussianMixture:
    """Fit a Gaussian mixture, without the temporal part of the HMM.

    A GMM has no transition structure, so its posterior at ``t`` depends only on ``x_t``
    given the fitted parameters — the "filtering" question does not arise, and the
    walk-forward refit is what keeps the parameters honest.

    ``random_state`` is pinned: without it, ``GaussianMixture``'s k-means initialisation
    is seeded from the global RNG and two runs of the same request disagree. Same reason
    the HMM start is deterministic.
    """
    from sklearn.mixture import GaussianMixture

    model = GaussianMixture(
        n_components=n_states,
        covariance_type="full",
        random_state=seed,
        max_iter=max_iter,
        reg_covar=_COVARIANCE_RIDGE,
        init_params="k-means++",
    )
    model.fit(x)
    return FittedGaussianMixture(
        means=np.asarray(model.means_, dtype=float),
        covars=np.asarray(model.covariances_, dtype=float),
        weights=np.asarray(model.weights_, dtype=float),
        log_likelihood=float(model.lower_bound_ * x.shape[0]),
        n_iter=int(model.n_iter_),
        converged=bool(model.converged_),
    )


def mixture_posteriors(x: np.ndarray, model: FittedGaussianMixture) -> np.ndarray:
    """Responsibility of each state for each observation. Row-stochastic."""
    log_emission = _log_emissions(x, model.means, model.covars)
    log_joint = log_emission + np.log(np.maximum(model.weights, _LOG_FLOOR))[None, :]
    return np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))


# ------------------------------------------------------------------- state naming


@dataclass(frozen=True)
class StateNaming:
    """The map from latent index to named regime, plus what could not be mapped."""

    mapping: dict[int, RegimeState]
    unrepresented: tuple[RegimeState, ...]
    notes: tuple[str, ...]


def name_states(
    means: np.ndarray, covars: np.ndarray, *, return_index: int = 0
) -> StateNaming:
    """Give latent states names, from the fitted parameters and nothing else.

    A regime model has no idea which of its clusters is "the bull market" — the indices
    are arbitrary and change between fits. The naming therefore has to be a stated rule,
    and the rule has to read only the fitted parameters, which are themselves a function
    of the training window alone.

    The rule, in order: the highest-variance state is Crisis; of what remains, the
    highest mean return is Trending Up, the lowest is Trending Down, and the quietest is
    Ranging. Every assignment removes its state from the pool, so two names can never
    land on one state — the failure mode that produces a probability column that is
    always zero.

    With fewer than four states some regimes are simply absent, and the caller is told
    which. Reporting "Ranging: 0.0" for a two-state fit would imply the model looked for
    ranging behaviour and did not find it, when it never looked.
    """
    n_states = means.shape[0]
    mean_return = means[:, return_index]
    volatility = np.sqrt(np.maximum(covars[:, return_index, return_index], 0.0))

    mapping: dict[int, RegimeState] = {}
    notes: list[str] = []
    claimed: set[int] = set()

    crisis = int(np.argmax(volatility))
    mapping[crisis] = RegimeState.CRISIS
    claimed.add(crisis)

    remaining = [i for i in range(n_states) if i not in claimed]
    if remaining:
        up = max(remaining, key=lambda i: float(mean_return[i]))
        mapping[up] = RegimeState.TRENDING_UP
        claimed.add(up)

    remaining = [i for i in range(n_states) if i not in claimed]
    if remaining:
        down = min(remaining, key=lambda i: float(mean_return[i]))
        mapping[down] = RegimeState.TRENDING_DOWN
        claimed.add(down)

    remaining = [i for i in range(n_states) if i not in claimed]
    if remaining:
        # Quietest of what is left. Ranging is defined by the absence of both drift and
        # variance, and variance is the part the model can actually separate.
        ranging = min(
            remaining,
            key=lambda i: (float(volatility[i]), abs(float(mean_return[i]))),
        )
        mapping[ranging] = RegimeState.RANGING
        claimed.add(ranging)

    for index in [i for i in range(n_states) if i not in claimed]:
        # Fifth and further states have no name of their own in the schema. They are
        # folded by drift sign, and folding is recorded rather than hidden, because the
        # named columns then sum to more than the raw posterior for those bars.
        folded = (
            RegimeState.TRENDING_UP if mean_return[index] >= 0.0 else RegimeState.TRENDING_DOWN
        )
        mapping[index] = folded
        notes.append(
            f"State {index} has no regime of its own and was folded into "
            f"{folded.value} by its mean return."
        )

    represented = set(mapping.values())
    unrepresented = tuple(
        state for state in REGIME_PROBABILITY_COLUMNS if state not in represented
    )

    return StateNaming(
        mapping=mapping,
        unrepresented=unrepresented,
        notes=tuple(notes),
    )


def transition_matrix_from_states(states: np.ndarray, n_states: int) -> list[list[float]]:
    """Empirical transition counts of a *committed* label sequence, row-normalised.

    Rows for states that never occur come back as zeros rather than uniform, so an
    absent state is visibly absent instead of looking equiprobable.
    """
    counts = np.zeros((n_states, n_states), dtype=float)
    for previous, current in zip(states[:-1], states[1:], strict=True):
        counts[int(previous), int(current)] += 1.0
    totals = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalised = np.where(totals > 0.0, counts / totals, 0.0)
    return [[float(value) for value in row] for row in normalised]
