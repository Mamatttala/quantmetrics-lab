"""Module E — the estimators, behind one protocol.

Five models, two families, one interface::

    model.fit(returns, exog)        # returns is exactly the history it may see
    model.forecast_variance(h)      # σ̂² for the next h bars, in decimal variance

The interface is the leakage defence. ``fit`` is handed an **array slice**, never a frame
and never an index into one, so there is no code path by which a model can reach a bar it
was not given. Everything above this module — the walk-forward loop in
``app/services/volatility.py`` — is then a matter of choosing the slice correctly, and the
truncation test in ``tests/test_volatility_models.py`` checks that it does.

**Family A** is arch: GARCH, EGARCH and GJR-GARCH, one estimator each, all four
distributions.

**Family B** is a linear arch leg multiplied by a machine-learned correction::

    σ̂²_{t+1} = σ²_linear,t+1 · ẑ̂²_{t+1}

where ``ẑ²`` is predicted from a tapped delay line of *past* returns and (when a
conditioning dataset is supplied) regime posteriors. The multiplicative form is not
decoration. Predicting the variance directly makes the target dominated by a handful of
enormous squared returns — the learner spends its capacity ranking those and gets the level
wrong everywhere else. Standardised squared residuals are scale-free by construction
(``E[z²] = 1``), so the learner only has to say *how wrong the linear model is right now*,
and the linear leg keeps carrying the level. That is also why the correction is clamped:
``E[z²] = 1`` is a fact about a correctly specified model, and a tree ensemble extrapolating
on out-of-range features will happily predict 40.

**The correction is a one-step-ahead object.** It answers "is the next bar more or less
volatile than the linear model thinks"; it has no definition at horizon 5, and iterating it
would be inventing a dynamic the model was never trained on. So the forward path applies it
to step 1 and leaves steps 2..h as the linear model's. The report says so.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Protocol

import numpy as np
from arch import arch_model
from arch.univariate.distribution import (
    Distribution,
    GeneralizedError,
    Normal,
    SkewStudent,
    StudentsT,
)
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from app.models.enums import VolDistribution, VolMeanModel, VolModel

#: arch is fitted on percentage returns. Its optimiser works on the scale of the data, and
#: decimal returns are ~1e-4, which puts the likelihood surface far from the region where
#: the default starting values and convergence tolerances were tuned. Scaling is exact and
#: reversible: ``σ²_decimal = σ²_percent / SCALE²``. Nothing about the fitted model changes
#: — the parameters are equivariant under it — which is why this is a conditioning
#: transform rather than a modelling choice, and why the report states the factor.
SCALE = 100.0
SCALE_VARIANCE = SCALE * SCALE

#: arch's own names for the two variance recursions this module drives. ``"GARCH"`` covers
#: both the plain GARCH (``o=0``) and the GJR-GARCH (``o=1``).
#:
#: Named as a type and not only as a string because ``arch_model(vol=...)`` is annotated
#: with the literal union: a ``str`` attribute fails the call site, and widening the
#: annotation to ``str`` would silence that by throwing away the check.
ArchVolName = Literal["GARCH", "EGARCH"]

#: arch's names for the four distributions, typed for the same reason — ``arch_model``
#: accepts only these spellings (plus aliases this module never uses).
ArchDistributionName = Literal["normal", "t", "skewt", "ged"]

#: The specification's distribution names mapped to arch's. One table, in the one place
#: that is obviously a translation, so a rename on either side is a one-line change.
ARCH_DISTRIBUTION: dict[VolDistribution, ArchDistributionName] = {
    VolDistribution.NORMAL: "normal",
    VolDistribution.STUDENTS_T: "t",
    VolDistribution.SKEW_STUDENT: "skewt",
    VolDistribution.GED: "ged",
}

#: The same four distributions as *classes*, for the one job the name map cannot do: seeding
#: the simulation that produces a multi-step path for EGARCH and GJR.
#:
#: It has to be done this way. ``arch_model(dist="t")`` builds its own generator and never
#: exposes a seed for it, and the ``random_state`` argument to ``forecast`` is read only by
#: the bootstrap method — not by simulation. So a model fitted by name alone draws fresh
#: shocks on every call, and two identical requests return two different fan charts. Setting a
#: seeded instance on the fitted model is the supported way to fix that; a forward path that
#: moves between runs is not a research artifact.
ARCH_DISTRIBUTION_CLASS: dict[VolDistribution, type[Distribution]] = {
    VolDistribution.NORMAL: Normal,
    VolDistribution.STUDENTS_T: StudentsT,
    VolDistribution.SKEW_STUDENT: SkewStudent,
    VolDistribution.GED: GeneralizedError,
}


#: The band the machine-learned correction is held inside. ``E[z²] = 1``; a correction
#: outside this band is not a view about the next bar, it is an extrapolation, and the
#: number of times the clamp binds is reported so a run that is silently pinned is visible.
CORRECTION_CLAMP = (0.1, 10.0)

#: XGBoost hyperparameters. Deliberately shallow and heavily regularised: the training set
#: is a few hundred rows of an extremely noisy target, and a deep forest on that data
#: memorises the training slice and produces a correction that is pure noise at the
#: forecast origin. ``n_jobs=1`` because a per-step refit inside a walk-forward is a loop of
#: small fits, where thread coordination costs more than it saves — and because XGBoost's
#: histogram method is not guaranteed bit-identical across thread counts.
XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5.0,
    "reg_lambda": 1.0,
    "n_jobs": 1,
    "verbosity": 0,
}

#: The TDNN's shape. Two hidden layers, the wider one first — a standard tapered
#: time-delay net. Small on purpose, for the same reason XGB_PARAMS is shallow.
TDNN_HIDDEN = (32, 16)

#: The seed every stochastic learner is fixed to. Determinism is a requirement of the
#: project, not a nicety: a run that cannot be reproduced cannot be checked, and the
#: causality test compares two runs of the same code.
SEED = 20240101


def _seeded_distribution(distribution: VolDistribution) -> Distribution:
    """A fresh, seeded distribution instance.

    Fresh rather than reused: the generator is a *stream*, so handing the same instance to a
    second forecast would draw the next block of shocks and return a different answer. Each
    call that simulates gets a new instance at the start of the stream.
    """
    return ARCH_DISTRIBUTION_CLASS[distribution](seed=SEED)


class VolatilityModel(Protocol):
    """What the walk-forward loop needs from an estimator.

    Structural rather than nominal: ``arch`` is a third-party class hierarchy and the
    hybrids are compositions, so there is no common base to inherit. The loop depends on
    these two methods and nothing else.
    """

    name: VolModel

    def fit(self, returns: np.ndarray, exog: np.ndarray | None = None) -> None:
        """Estimate from ``returns`` — every element of which is in the past of the bar
        this model will next be asked to forecast."""

    def forecast_variance(self, horizon: int) -> np.ndarray:
        """``σ̂²`` for the next ``horizon`` bars, in decimal variance units."""


def _as_float_1d(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).ravel()
    if array.size < 2:
        raise ValueError("A volatility model needs at least two observations.")
    return array


def delayed_design(
    returns: np.ndarray,
    exog: np.ndarray | None,
    lags: int,
) -> np.ndarray:
    """The tapped delay line: row ``j`` holds lags ``1..lags`` of the series, all in the
    past of ``j``.

    Returns an array of shape ``(n, lags * (2 + n_exog))``. Rows ``0 .. lags-1`` are NaN —
    there is no history to fill them — and the caller drops them, which is why this returns
    an array rather than a mask: the valid rows are exactly the finite ones, and making
    that the caller's decision in two places is how the two come to disagree.

    Each lag contributes the return **and its absolute value**. The signed return carries
    the leverage effect — a large negative return raises future volatility more than a
    large positive one of the same size — and the absolute value carries the magnitude
    without the sign, which is what a tree splits on cleanly. Handing over only the signed
    value makes every split on magnitude cost a pair of thresholds.

    **Causality is structural here.** Row ``j`` reads indices ``j-lags .. j-1``. There is no
    offset parameter and no default that could shift it.
    """
    series = _as_float_1d(returns)
    n = series.size
    blocks: list[np.ndarray] = []
    for lag in range(1, lags + 1):
        shifted = np.full(n, np.nan)
        shifted[lag:] = series[:-lag]
        blocks.append(shifted)
        blocks.append(np.abs(shifted))
    if exog is not None:
        matrix = np.asarray(exog, dtype=float)
        if matrix.ndim != 2:
            raise ValueError("exog must be a 2-D array of shape (n, k).")
        if matrix.shape[0] != n:
            raise ValueError(
                f"exog has {matrix.shape[0]} rows but the series has {n}; the lagged "
                "features must be aligned with the returns they describe."
            )
        for lag in range(1, lags + 1):
            shifted = np.full((n, matrix.shape[1]), np.nan)
            shifted[lag:] = matrix[:-lag]
            blocks.extend(shifted[:, column] for column in range(matrix.shape[1]))
    return np.column_stack(blocks)


class ArchModel:
    """Family A — one arch estimator, wrapped so that the outside world never sees the
    scale change or arch's own calling convention.

    ``vol`` is arch's name for the variance process: ``"GARCH"`` covers both the plain
    GARCH (``o=0``) and the GJR-GARCH (``o=1``), because GJR *is* a GARCH with an
    asymmetry term — the specification lists them separately and so does the enum, but
    underneath they are one recursion.
    """

    def __init__(
        self,
        name: VolModel,
        *,
        vol: ArchVolName,
        p: int,
        o: int,
        q: int,
        distribution: VolDistribution,
        mean_model: VolMeanModel,
    ) -> None:
        self.name = name
        self.vol = vol
        self.p = p
        self.o = o
        self.q = q
        self.distribution = distribution
        self.mean_model = mean_model
        self._result: Any = None

    # ------------------------------------------------------------------ fitting

    def fit(self, returns: np.ndarray, exog: np.ndarray | None = None) -> None:
        """Estimate on ``returns``.

        ``exog`` is accepted and ignored. The signature is shared with the hybrids because
        the walk-forward loop is written once against the protocol; a conditional
        exogenous variable in the *mean* equation is a different model (arch's ``ARX``) and
        the specification does not ask for one, so accepting the argument and doing nothing
        with it is the honest reading. It is stated here rather than left for a reader to
        discover by grepping for its absence.
        """
        series = _as_float_1d(returns) * SCALE
        mean: Literal["Constant", "AR"] = "Constant"
        lags = 0
        if self.mean_model is VolMeanModel.ARMA:
            # arch has no ARMA mean for a univariate series alongside a variance equation.
            # This is an AR(p) mean estimated jointly with the variance — which is what the
            # spec's "ARMA-EGARCH" reduces to here, and the report says `ar{p}` rather than
            # "ARMA" so the artifact does not claim more than it estimated. Adding a
            # separate MA part on top of an EGARCH(q) is not identifiable: the q lagged
            # variance terms already carry the residual autocorrelation.
            mean = "AR"
            lags = self.p
        model = arch_model(
            series,
            mean=mean,
            lags=lags,
            vol=self.vol,
            p=self.p,
            o=self.o,
            q=self.q,
            dist=ARCH_DISTRIBUTION[self.distribution],
            rescale=False,
        )
        # Set the seeded instance the name map cannot express. See ARCH_DISTRIBUTION_CLASS.
        model.distribution = _seeded_distribution(self.distribution)
        self._result = model.fit(disp="off", show_warning=False)

    def _require_fit(self) -> Any:
        if self._result is None:
            raise RuntimeError("forecast_variance called before fit.")
        return self._result

    # --------------------------------------------------------------- forecasting

    @property
    def mean_label(self) -> str:
        """What the mean equation actually is, for the report."""
        return f"ar{self.p}" if self.mean_model is VolMeanModel.ARMA else "constant"

    def _analytic_beyond_one_step(self) -> bool:
        """Whether arch can forecast this process analytically past step 1.

        It cannot for the processes that do not evolve in squares — EGARCH and anything with
        an asymmetry term, which includes GJR. For those, steps 2..h are produced by
        simulation, and the simulation is seeded so the answer is the same on every run.
        """
        return self.vol == "GARCH" and self.o == 0

    def forecast_variance(self, horizon: int) -> np.ndarray:
        result = self._require_fit()
        method = "analytic" if (horizon <= 1 or self._analytic_beyond_one_step()) else "simulation"
        kwargs: dict[str, Any] = {"horizon": horizon, "reindex": False, "method": method}
        if method == "simulation":
            # Re-seeded here rather than once at fit time. The generator is a stream, so a
            # second `forecast` on the same fitted model would draw the *next* block of
            # shocks and return a different path; replacing the instance rewinds it. This is
            # what makes the determinism test pass, and a fan chart that moves between two
            # identical requests would be a chart nobody could cite.
            result.model.distribution = _seeded_distribution(self.distribution)
            kwargs["simulations"] = 1000
        forecast = result.forecast(**kwargs)
        variance = np.asarray(forecast.variance, dtype=float)[0]
        return variance / SCALE_VARIANCE

    # ------------------------------------------------------------------ internals

    def standardised_residuals(self) -> np.ndarray:
        """In-sample ``z = e_t / σ_t``, each computed from information up to ``t-1``.

        arch aligns ``conditional_volatility[t]`` with the variance of the ``t``-th error
        *computed from time ``t-1``*, so this ratio is in-sample in the only sense that
        matters: every element of it would have been available at the moment it is stamped
        with. It is the target the hybrids learn.
        """
        result = self._require_fit()
        residual = np.asarray(result.resid, dtype=float).ravel()
        volatility = np.asarray(result.conditional_volatility, dtype=float).ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(volatility > 0.0, residual / volatility, np.nan)


class CorrectionLearner(Protocol):
    """The second leg of a hybrid: predicts ``ẑ²`` from lagged features."""

    def fit(self, features: np.ndarray, target: np.ndarray) -> None: ...

    def predict(self, features: np.ndarray) -> np.ndarray: ...


class XGBoostCorrection:
    """A gradient-boosted correction. Seeded, shallow, single-threaded."""

    def __init__(self) -> None:
        self._model: XGBRegressor | None = None

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        self._model = XGBRegressor(random_state=SEED, **XGB_PARAMS)
        # `base_score` is left at the library default rather than set to mean(target). The
        # target's mean is ~1 by construction, and a model whose starting point is the
        # training mean of a heavy-tailed ratio is a model whose first trees fit noise.
        self._model.fit(features, target)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("predict called before fit.")
        return np.asarray(self._model.predict(features), dtype=float).ravel()


class TDNNCorrection:
    """A time-delay neural network correction.

    The input is exactly the tapped delay line the name promises. Standardisation is inside
    the pipeline rather than done to the array beforehand, because a scaler fitted on the
    whole series would be a second, quieter way to leak the future — and here it is fitted
    on the same slice the network is, per step, which is why this is a ``Pipeline`` and not
    a manual transform.
    """

    def __init__(self) -> None:
        self._model: Pipeline | None = None

    def fit(self, features: np.ndarray, target: np.ndarray) -> None:
        self._model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "net",
                    MLPRegressor(
                        hidden_layer_sizes=TDNN_HIDDEN,
                        activation="relu",
                        solver="adam",
                        max_iter=2000,
                        early_stopping=False,
                        random_state=SEED,
                    ),
                ),
            ]
        )
        self._model.fit(features, target)

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("predict called before fit.")
        return np.asarray(self._model.predict(features), dtype=float).ravel()


class HybridModel:
    """Family B — an arch leg times a learned one-step correction.

    The learner is trained on the linear leg's **in-sample** standardised residuals, using
    features built only from lags. Both halves of that sentence matter:

    * the target is ``z²``, not ``r²`` — scale-free, and it describes the linear model's
      error rather than the market's level, which is the part the linear model has already
      got right;
    * the features reach backwards only, so the pair ``(X_j, z²_j)`` is a statement about a
      bar that had already happened when the model was fitted.
    """

    def __init__(
        self,
        name: VolModel,
        *,
        linear: ArchModel,
        learner: CorrectionLearner,
        lags: int,
    ) -> None:
        self.name = name
        self.linear = linear
        self.learner = learner
        self.lags = lags
        self._next_features: np.ndarray | None = None
        self._clamped = 0
        self._untrained = False

    @property
    def mean_label(self) -> str:
        return self.linear.mean_label

    def fit(self, returns: np.ndarray, exog: np.ndarray | None = None) -> None:
        self.linear.fit(returns, exog)
        self._clamped = 0
        self._untrained = False

        series = _as_float_1d(returns)
        # The design is built over the series **extended by one placeholder**. Row `m` of the
        # result is the feature vector for the bar immediately after the slice ends — the bar
        # the walk-forward is about to ask for — and its lags are the last `lags`
        # observations, all of which are inside the slice. Building it over the slice alone
        # and taking the last row would produce the features for the *previous* bar, a
        # one-lag offset that no test of the forecast would catch and that silently dulls
        # every hybrid.
        if exog is None:
            extended_exog = None
        else:
            matrix = np.asarray(exog, dtype=float)
            extended_exog = np.vstack([matrix, matrix[-1:]])
        design = delayed_design(np.append(series, np.nan), extended_exog, self.lags)
        target_all = self.linear.standardised_residuals() ** 2

        self._next_features = design[-1:]
        train_x = design[:-1]
        train_y = target_all

        valid = np.isfinite(train_x).all(axis=1) & np.isfinite(train_y)
        if int(valid.sum()) < 20:
            # Not enough usable history to train a second model. Falling back to the linear
            # forecast is the honest response — it is exactly what the hybrid reduces to
            # when the correction carries no information — and it is recorded, because a
            # hybrid that silently is not one would be reported as having earned its cost.
            self._untrained = True
            return

        self.learner.fit(train_x[valid], train_y[valid])

    def _correction(self) -> float:
        if self._untrained or self._next_features is None:
            return 1.0
        if not np.isfinite(self._next_features).all():
            return 1.0
        predicted = float(self.learner.predict(self._next_features)[0])
        if not math.isfinite(predicted):
            return 1.0
        low, high = CORRECTION_CLAMP
        clamped = min(max(predicted, low), high)
        if clamped != predicted:
            self._clamped += 1
        return clamped

    def forecast_variance(self, horizon: int) -> np.ndarray:
        linear = self.linear.forecast_variance(horizon)
        if horizon >= 1:
            # Step 1 only. See the module docstring: the correction has no definition
            # beyond the bar it was trained to predict, and iterating it would be inventing
            # a dynamic.
            linear = linear.copy()
            linear[0] = linear[0] * self._correction()
        return linear

    @property
    def correction_clamped(self) -> int:
        """How many times the clamp bound on the last fit. Reported so a run pinned at the
        boundary is visible rather than looking like a confident correction."""
        return self._clamped

    @property
    def untrained(self) -> bool:
        """Whether the learner was skipped for want of usable history, leaving the linear
        forecast to stand alone."""
        return self._untrained


def build_model(
    name: VolModel,
    *,
    distribution: VolDistribution,
    mean_model: VolMeanModel,
    p: int,
    o: int,
    q: int,
    lags: int,
) -> ArchModel | HybridModel:
    """Construct one estimator from a request. The only place the enum is mapped to code.

    A single factory rather than a dispatch inside the walk-forward loop, so the loop has
    no idea which family it is running — it calls ``fit`` and ``forecast_variance`` and the
    cost difference is the only thing that distinguishes them.
    """
    if name is VolModel.GARCH:
        return ArchModel(
            name, vol="GARCH", p=p, o=0, q=q,
            distribution=distribution, mean_model=mean_model,
        )
    if name is VolModel.EGARCH:
        # EGARCH needs at least one asymmetry order to be the model the name refers to;
        # with o=0 it is a symmetric EARCH, which is a different and much less useful
        # process. The request validator refuses o=0 for this model.
        return ArchModel(
            name, vol="EGARCH", p=p, o=o, q=q,
            distribution=distribution, mean_model=mean_model,
        )
    if name is VolModel.GJR_GARCH:
        # GJR is defined by exactly one asymmetry term: the recursion has a single
        # indicator for "the last return was negative". o=1 is not a tuning choice.
        return ArchModel(
            name, vol="GARCH", p=p, o=1, q=q,
            distribution=distribution, mean_model=mean_model,
        )
    if name is VolModel.ARMA_EGARCH_XGBOOST:
        return HybridModel(
            name,
            linear=ArchModel(
                name, vol="EGARCH", p=p, o=max(1, o), q=q,
                distribution=distribution, mean_model=mean_model,
            ),
            learner=XGBoostCorrection(),
            lags=lags,
        )
    if name is VolModel.TDNN_GJR_GARCH:
        return HybridModel(
            name,
            linear=ArchModel(
                name, vol="GARCH", p=p, o=1, q=q,
                distribution=distribution, mean_model=mean_model,
            ),
            learner=TDNNCorrection(),
            lags=lags,
        )
    raise ValueError(f"Unknown model {name!r}.")


def adf_test(returns: np.ndarray) -> tuple[float, float]:
    """The augmented Dickey–Fuller statistic and p-value for a return series.

    Run on **returns**, never on prices. A price level is near-unit-root by construction —
    that is what a price is — so testing it would refuse every dataset ever loaded, which
    is a guardrail that refuses everything and therefore protects nothing. What the check
    is actually for is the case where the user has pointed ``price_source`` at a column
    that is not a return series at all: a running total, an open interest, a cumulative
    volume. Those fail, and the failure is reported with the numbers.

    statsmodels' critical values are for the standard regression with a constant; the
    default ``autolag="AIC"`` chooses the lag truncation, which is the part a hand-rolled
    version gets wrong.
    """
    from statsmodels.tsa.stattools import adfuller

    series = _as_float_1d(returns)
    finite = series[np.isfinite(series)]
    if finite.size < 20:
        raise ValueError("The series is too short to test for stationarity.")
    # result_object=False keeps the tuple return and suppresses statsmodels' FutureWarning
    # about the 0.16 switch to an ADFullerResult; left to raise, that warning is an error
    # under the suite's error::FutureWarning filter and would fail the whole forecast job.
    statistic, p_value, *_ = adfuller(finite, autolag="AIC", result_object=False)
    return float(statistic), float(p_value)
