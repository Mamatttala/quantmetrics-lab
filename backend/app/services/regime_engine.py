"""Module D's driver: observations in, labelled regimes out.

Everything here is organised around one contract, and the contract is the reason the
file is shaped the way it is:

    A label stamped on bar *t* is a function of bars ``0..t`` and nothing later.

Three separate things could break that, and each is handled in one place rather than
scattered:

1. **The observation vector.** Every feature is built from a trailing window. There is no
   whole-sample mean, standard deviation, or quantile anywhere in this module, because a
   normalisation fitted across the full series leaks the future into every row it touches.
2. **The parameters.** The model is refitted on an expanding window at a schedule anchored
   to *absolute bar index*. That anchoring is load-bearing: if the refit stride were
   derived from the series length, truncating the series would move every refit point and
   the first 80% of a re-run would differ from the first 80% of the original — which is
   precisely the failure the specification's causality test is designed to detect.
3. **The state names.** A HMM's state indices are arbitrary, so "which cluster is the
   bull market" is decided from the fitted parameters. Those parameters come from a
   training window, so the naming is recomputed at every refit and applied only to the
   bars that refit is allowed to label. Naming once from a fit on the whole series would
   smuggle the future in through the labels while every probability looked clean.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.core.errors import ValidationFailure
from app.models.enums import (
    REGIME_PROBABILITY_COLUMNS,
    AssetClass,
    RegimeFeature,
    RegimeModel,
    RegimeState,
    Timeframe,
)
from app.services import regime_models

#: Trading days per year, by asset class. Crypto trades every day; equities do not, and
#: using 365 for both scales every equity volatility by sqrt(365/252) — about 20% — which
#: is a silent multiplicative error rather than a visible one.
_DAYS_PER_YEAR: dict[str, int] = {"crypto": 365, "equity": 252}

#: Bars per day for each supported timeframe. Weekly is expressed as a fraction of a day
#: so the same multiplication produces bars-per-year for every member.
_BARS_PER_DAY: dict[Timeframe, float] = {
    Timeframe.M1: 1440.0,
    Timeframe.M5: 288.0,
    Timeframe.M15: 96.0,
    Timeframe.M30: 48.0,
    Timeframe.H1: 24.0,
    Timeframe.H4: 6.0,
    Timeframe.D1: 1.0,
    Timeframe.W1: 1.0 / 7.0,
}

_CRYPTO_CLASSES = frozenset({AssetClass.CRYPTO_SPOT, AssetClass.CRYPTO_FUTURES})

#: The names a state can be committed to. Transition is deliberately not here: it is
#: derived from the shape of these, not a competing hypothesis.
_CANONICAL = REGIME_PROBABILITY_COLUMNS


def annualisation_factor(timeframe: Timeframe | None, asset_class: AssetClass) -> float:
    """Periods per year for annualising a per-bar volatility.

    Refuses rather than guessing when the dataset declares no timeframe. A wrong factor
    scales every volatility by a constant the user cannot see, and the crisis threshold
    then fires at the wrong level — the error would be attributed to the model.
    """
    if timeframe is None:
        raise ValidationFailure(
            "This dataset declares no timeframe, so volatility cannot be annualised.",
            hint="Re-fetch it with a timeframe, or resample it in B4 so the bar length "
            "is recorded.",
        )
    family = "crypto" if asset_class in _CRYPTO_CLASSES else "equity"
    return _BARS_PER_DAY[timeframe] * _DAYS_PER_YEAR[family]


def _log_returns(close: np.ndarray) -> np.ndarray:
    """Bar log-returns. The first entry is NaN: it has no predecessor, and inventing a
    zero there would make the first bar look like a quiet one."""
    out = np.full(close.shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[1:] = np.log(close[1:] / close[:-1])
    return out


def _trailing_std(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing sample standard deviation, ``ddof=1``, ending at each bar.

    Uses a strided view rather than ``pandas.rolling`` so the window is unambiguously
    backward-looking by construction. ``min_periods`` and ``center`` are both one keyword
    away from a look-ahead and neither is expressible here.
    """
    n = values.shape[0]
    out = np.full(n, np.nan)
    if n < window:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(values, window)
    with np.errstate(invalid="ignore"):
        out[window - 1 :] = windows.std(axis=1, ddof=1)
    return out


def realized_volatility(
    returns: np.ndarray, window: int, periods_per_year: float
) -> np.ndarray:
    """Annualised trailing realized volatility, in decimal (0.35 = 35%)."""
    return _trailing_std(returns, window) * math.sqrt(periods_per_year)


def _volume_change(volume: np.ndarray) -> np.ndarray:
    """Log change in volume. Log, so a doubling and a halving are symmetric."""
    out = np.full(volume.shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[1:] = np.log(volume[1:] / volume[:-1])
    return out


@dataclass(frozen=True)
class RegimeSpec:
    """The modelling decisions for one run, resolved and immutable.

    Mirrors the request model but deliberately free of pydantic, so the engine can be
    tested without an HTTP layer and reused by Module E without importing FastAPI.
    """

    model_type: RegimeModel
    n_states: int
    features: tuple[RegimeFeature, ...]
    min_state_duration: int
    switching_cost_penalty: float
    crisis_threshold: float
    override_probability: float
    transition_confidence: float
    commit_confidence: float
    min_train_bars: int
    refit_every: int
    volatility_window: int


@dataclass(frozen=True)
class RegimeOutcome:
    """The output dataset and the report that explains it."""

    frame: pd.DataFrame
    report: dict[str, Any]


# ------------------------------------------------------------------ observations


def _observation_matrix(
    frame: pd.DataFrame,
    spec: RegimeSpec,
    *,
    close: np.ndarray,
    volume: np.ndarray | None,
    periods_per_year: float,
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    """Assemble the fitted observation columns, in the order the request listed them.

    ``ADX`` and ``RSI`` are *carried*, never computed. The feature catalogue already
    defines both, names them by window, documents their formula, and warns that they are
    recursive; a second implementation here would be a second formula for one indicator
    and the two would drift. A request naming them is refused when the columns are
    absent, with the module that can produce them named in the hint.
    """
    computed: dict[str, np.ndarray] = {}
    returns = _log_returns(close)
    computed[RegimeFeature.LOG_RETURN] = returns
    computed[RegimeFeature.REALIZED_VOLATILITY] = realized_volatility(
        returns, spec.volatility_window, periods_per_year
    )
    if volume is not None:
        computed[RegimeFeature.VOLUME_CHANGE] = _volume_change(volume)

    columns: list[str] = []
    for feature in spec.features:
        if feature not in computed:
            raise ValidationFailure(
                f"'{feature.value}' is not carried by this dataset.",
                hint="The hybrid model reads indicator columns rather than recomputing "
                "them. Produce them in C5 Feature Engineering first, then merge that "
                "output in B4.",
            )
        columns.append(feature.value)

    matrix = np.column_stack([computed[name] for name in columns])
    return matrix, columns, computed


def _first_complete_row(matrix: np.ndarray) -> int:
    """Index of the first row with no NaN. Everything before it is warm-up."""
    finite = np.isfinite(matrix).all(axis=1)
    if not finite.any():
        raise ValidationFailure(
            "No bar has a complete set of observations.",
            hint="Every feature is still in its warm-up after the last row. Use a "
            "longer dataset or a shorter volatility window.",
        )
    return int(np.argmax(finite))


# ------------------------------------------------------------------ walk-forward


@dataclass(frozen=True)
class _Segment:
    """One refit, the rows that trained it, and the rows it may label.

    ``train_from`` is carried explicitly rather than assumed to be zero. The first rows of
    the observation matrix are warm-up NaNs — the volatility window has not filled yet —
    and a training slice that started at row 0 would hand them to EM, which then refuses
    the fit. The training window is ``[train_from, trained_through)`` and the labelled
    window is ``[trained_through, labels_to)``; the two never overlap, which is what makes
    each label a function of strictly earlier bars.
    """

    train_from: int
    trained_through: int
    labels_from: int
    labels_to: int


def _refit_schedule(n_obs: int, spec: RegimeSpec, valid_from: int) -> list[_Segment]:
    """Refit points at fixed absolute indices, in bars from the start of the series.

    The stride is a bar count, never a fraction of ``n_obs``. A schedule like "refit every
    10% of the series" would move every point when the series changes length, so a
    truncated re-run would produce different labels over the shared prefix and the
    causality test would fail on a correct implementation.
    """
    segments: list[_Segment] = []
    first = valid_from + spec.min_train_bars
    if first >= n_obs:
        raise ValidationFailure(
            f"Not enough bars to train: {n_obs - valid_from} usable, "
            f"{spec.min_train_bars} required before the first label.",
            hint="Lower the minimum training window, or supply a longer dataset.",
        )
    start = first
    while start < n_obs:
        stop = min(start + spec.refit_every, n_obs)
        segments.append(
            _Segment(
                train_from=valid_from,
                trained_through=start,
                labels_from=start,
                labels_to=stop,
            )
        )
        start = stop
    return segments


def _fit_and_label_hmm(
    matrix: np.ndarray, spec: RegimeSpec, segments: list[_Segment]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Fit at each refit point, filter forward, and name states per segment."""
    n_obs, n_features = matrix.shape
    named = np.full((n_obs, len(_CANONICAL)), np.nan)
    diagnostics: list[dict[str, Any]] = []

    for segment in segments:
        training = matrix[segment.train_from : segment.trained_through]
        model = regime_models.fit_gaussian_hmm(training, spec.n_states)
        naming = regime_models.name_states(model.means, model.covars)

        window = matrix[segment.labels_from : segment.labels_to]
        posteriors = regime_models.filtered_posteriors(window, model)
        named[segment.labels_from : segment.labels_to] = _collapse_to_names(posteriors, naming)

        diagnostics.append(
            {
                "trained_through_bar": segment.trained_through,
                "training_bars": segment.trained_through - segment.train_from,
                "labels_bars": [segment.labels_from, segment.labels_to],
                "log_likelihood": model.log_likelihood,
                "bic": model.bic(segment.trained_through - segment.train_from),
                "aic": model.aic(segment.trained_through - segment.train_from),
                "n_parameters": model.n_parameters,
                "n_iter": model.n_iter,
                "converged": model.converged,
                "state_naming": {str(k): v.value for k, v in sorted(naming.mapping.items())},
                "states_without_a_regime": list(naming.unrepresented),
                "naming_notes": list(naming.notes),
            }
        )

    return named, diagnostics


def _fit_and_label_mixture(
    matrix: np.ndarray, spec: RegimeSpec, segments: list[_Segment]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """The GMM path. Same schedule and same per-segment naming as the HMM."""
    n_obs = matrix.shape[0]
    named = np.full((n_obs, len(_CANONICAL)), np.nan)
    diagnostics: list[dict[str, Any]] = []

    for segment in segments:
        training = matrix[segment.train_from : segment.trained_through]
        model = regime_models.fit_gaussian_mixture(training, spec.n_states)
        naming = regime_models.name_states(model.means, model.covars)

        window = matrix[segment.labels_from : segment.labels_to]
        named[segment.labels_from : segment.labels_to] = _collapse_to_names(
            regime_models.mixture_posteriors(window, model), naming
        )
        diagnostics.append(
            {
                "trained_through_bar": segment.trained_through,
                "training_bars": segment.trained_through - segment.train_from,
                "labels_bars": [segment.labels_from, segment.labels_to],
                "log_likelihood": model.log_likelihood,
                "bic": model.bic(segment.trained_through - segment.train_from),
                "aic": model.aic(segment.trained_through - segment.train_from),
                "n_parameters": model.n_parameters,
                "n_iter": model.n_iter,
                "converged": model.converged,
                "state_naming": {str(k): v.value for k, v in sorted(naming.mapping.items())},
                "states_without_a_regime": list(naming.unrepresented),
                "naming_notes": list(naming.notes),
            }
        )

    return named, diagnostics


def _collapse_to_names(
    posteriors: np.ndarray, naming: regime_models.StateNaming
) -> np.ndarray:
    """Sum latent-state posterior mass into the four named regimes.

    Every latent state maps to exactly one name, so the four columns still sum to one.
    That is why the map folds extra states rather than dropping them: dropping would
    silently destroy probability mass and make ``confidence`` read low on bars where the
    model was in fact certain.
    """
    out = np.zeros((posteriors.shape[0], len(_CANONICAL)))
    for index, state in naming.mapping.items():
        column = _CANONICAL.index(state)
        out[:, column] += posteriors[:, index]
    return out


def _fit_and_label_hybrid(
    matrix: np.ndarray,
    indicator_columns: np.ndarray,
    spec: RegimeSpec,
    segments: list[_Segment],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """HMM labels, reproduced by a gradient-boosted classifier on faster indicators.

    The HMM lags: a regime is only confirmed once enough bars of it have printed. A
    classifier trained to reproduce the HMM's own labels from indicators that respond
    sooner can anticipate the switch.

    **The labels must not be the HMM's smoothed output**, and they are not — they are the
    filtered labels from the same walk-forward schedule, so bar ``t`` was labelled from
    bars ``0..t``. Training on smoothed labels would teach the classifier the future and
    the resulting probabilities would look excellent in-sample and be worthless live.
    Each classifier is also trained only on bars strictly before the segment it predicts,
    so the fit sees the same information the HMM fit did.

    The specification names LightGBM. LightGBM is not installed in this environment and
    cannot be assumed; XGBoost is, and is the same family of model — gradient-boosted
    decision trees with the same inductive bias for this task. The substitution is
    recorded rather than silent. See DECISIONS §75.
    """
    from xgboost import XGBClassifier

    n_obs = matrix.shape[0]
    named = np.full((n_obs, len(_CANONICAL)), np.nan)
    diagnostics: list[dict[str, Any]] = []

    for segment in segments:
        training = matrix[segment.train_from : segment.trained_through]
        hmm = regime_models.fit_gaussian_hmm(training, spec.n_states)
        naming = regime_models.name_states(hmm.means, hmm.covars)

        # Labels the classifier imitates: the causal, filtered, named state of each
        # training bar. Assigned through the naming derived from the same fit.
        training_named = _collapse_to_names(
            regime_models.filtered_posteriors(training, hmm), naming
        )
        labels = np.argmax(training_named, axis=1)

        features = np.column_stack(
            [indicator_columns[segment.train_from : segment.trained_through], training]
        )
        window = matrix[segment.labels_from : segment.labels_to]
        window_features = np.column_stack(
            [indicator_columns[segment.labels_from : segment.labels_to], window]
        )

        present = np.unique(labels)
        if present.size < 2:
            # One class in the training window: a classifier has nothing to separate.
            # Falling back to the HMM's own posterior is honest; forcing a fit would
            # produce a constant, overconfident model.
            named[segment.labels_from : segment.labels_to] = _collapse_to_names(
                regime_models.filtered_posteriors(
                    matrix[segment.labels_from : segment.labels_to], hmm
                ),
                naming,
            )
            diagnostics.append(
                {
                    "trained_through_bar": segment.trained_through,
                "training_bars": segment.trained_through - segment.train_from,
                    "labels_bars": [segment.labels_from, segment.labels_to],
                    "fallback": "single state present in training window",
                    "state_naming": {str(k): v.value for k, v in sorted(naming.mapping.items())},
                }
            )
            continue

        classifier = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=0,
            n_jobs=1,
            objective="multi:softprob",
        )
        classifier.fit(features, labels)

        probabilities = classifier.predict_proba(window_features)
        aligned = np.zeros((probabilities.shape[0], len(_CANONICAL)))
        # predict_proba returns columns for the classes actually present, in sorted
        # order. Mapping through `classes_` rather than assuming all K are present keeps
        # a training window that never saw, say, a crisis state from shifting every
        # column left by one.
        for position, label in enumerate(classifier.classes_):
            aligned[:, int(label)] = probabilities[:, position]
        aligned = np.clip(aligned, 0.0, None)
        totals = aligned.sum(axis=1, keepdims=True)
        aligned = np.divide(aligned, totals, out=np.zeros_like(aligned), where=totals > 0.0)
        named[segment.labels_from : segment.labels_to] = aligned

        diagnostics.append(
            {
                "trained_through_bar": segment.trained_through,
                "training_bars": segment.trained_through - segment.train_from,
                "labels_bars": [segment.labels_from, segment.labels_to],
                "classes_present": [int(c) for c in classifier.classes_],
                "state_naming": {str(k): v.value for k, v in sorted(naming.mapping.items())},
                "estimator": "xgboost.XGBClassifier",
            }
        )

    return named, diagnostics


# --------------------------------------------------------------- rules heuristic


def _rules_probabilities(
    returns: np.ndarray, volatility: np.ndarray, spec: RegimeSpec
) -> np.ndarray:
    """A closed-form heuristic, published so it can be argued with.

    This is not a fitted model and the report says so. It exists because the
    specification asks for a rules-based path, and a rules-based path that emitted a hard
    0/1 label would violate the soft-label rule.

    For each bar, with ``s = mean(trailing return) / realized volatility`` over the same
    volatility window:

    * ``crisis``   — ``volatility > crisis_threshold`` → 1.0
    * ``up``       — otherwise ``logistic(s)`` scaled by how far volatility sits below
      the threshold
    * ``down``     — the mirror, ``logistic(-s)``
    * ``ranging``  — the remainder

    The construction is deliberately simple and its weakness is stated: it has no memory,
    so it will chatter far more than a fitted model. It is offered for diagnosis, not as
    a substitute for one.
    """
    n_obs = returns.shape[0]
    out = np.zeros((n_obs, len(_CANONICAL)))
    momentum = np.full(n_obs, np.nan)
    window = max(2, spec.volatility_window)
    trailing_mean = np.convolve(
        np.nan_to_num(returns, nan=0.0), np.ones(window) / window, mode="full"
    )[:n_obs]
    momentum = trailing_mean

    for t in range(n_obs):
        vol = volatility[t]
        if not np.isfinite(vol) or vol <= 0.0:
            out[t] = np.nan
            continue
        if vol > spec.crisis_threshold:
            out[t, _CANONICAL.index(RegimeState.CRISIS)] = 1.0
            continue
        drift = momentum[t] / vol
        up = 1.0 / (1.0 + math.exp(-float(drift)))
        down = 1.0 - up
        quiet = max(0.0, 1.0 - vol / spec.crisis_threshold)
        out[t, _CANONICAL.index(RegimeState.TRENDING_UP)] = up * quiet
        out[t, _CANONICAL.index(RegimeState.TRENDING_DOWN)] = down * quiet
        out[t, _CANONICAL.index(RegimeState.RANGING)] = 1.0 - quiet
    return out


# ------------------------------------------------------------------- labelling


@dataclass(frozen=True)
class Labelling:
    """Committed labels, the confidence attached to each, and the posterior behind them.

    ``named`` is the revised posterior the labels were actually read off — the input
    ``named`` matrix with any crisis override written into it. It is returned rather than
    recomputed so the frame a caller writes to disk is built from the same array the labels
    were chosen from, and cannot drift from it.
    """

    states: list[RegimeState | None]
    confidence: np.ndarray
    switches: int
    override_bars: int
    named: np.ndarray


def apply_crisis_override(
    named: np.ndarray, crisis_mask: np.ndarray, override_probability: float
) -> np.ndarray:
    """Write the crisis override into the probabilities, not just into the label.

    The specification asks for the breach to produce "Crisis with >90% probability", and
    an earlier version of this module did not do that: it relabelled the bar and left
    ``prob_crisis`` at whatever the model had said, which on a sudden spike is around
    1e-183. The output was then self-contradictory in two ways at once — the bar was
    labelled Crisis with a crisis probability of zero, and ``argmax`` of the probability
    columns disagreed with ``dominant_state``. Any consumer that read the columns instead
    of the label got the opposite answer, silently.

    The fix is to treat the override as what it is: evidence. A realized-volatility breach
    is a fact about the market that the model's likelihood did not anticipate, so it
    revises the posterior — the crisis column rises to at least ``override_probability``
    and the remaining mass is redistributed across the others *in proportion to what the
    model already believed*. Rows still sum to one and the crisis column is the argmax.

    Nothing else about the frame moves: a bar with no breach is returned untouched.

    This is called by :func:`commit_labels` rather than by the caller. That is deliberate.
    When the two were separate steps, a caller that passed a ``crisis_mask`` without having
    applied this first got a frame whose ``dominant_state`` said Crisis while its own
    ``prob_crisis`` column said ~0 — a contradiction the type system cannot catch and the
    tests only caught by accident. Keeping the revision inside the labelling call makes
    that state unreachable rather than merely discouraged.
    """
    out = named.copy()
    crisis_at = _CANONICAL.index(RegimeState.CRISIS)
    others = [index for index in range(len(_CANONICAL)) if index != crisis_at]

    for t in np.flatnonzero(crisis_mask):
        row = out[t]
        if not np.isfinite(row).all():
            # A breach during the warm-up. There is no posterior to revise, and inventing
            # one would put a probability vector on a bar the model has not reached. The
            # count of these is reported separately so the gap is visible rather than
            # silent.
            continue
        target = max(float(row[crisis_at]), float(override_probability))
        remaining = 1.0 - target
        other_total = float(row[others].sum())
        if other_total > 0.0:
            row[others] = row[others] * (remaining / other_total)
        else:
            row[others] = remaining / len(others)
        # Written last so the crisis figure is exact rather than ``1.0 - something``;
        # ``prob_crisis >= override_probability`` is a property the specification states
        # and a caller may assert, and it should not fail on a rounding error.
        row[crisis_at] = target
        out[t] = row
    return out


def commit_labels(
    named: np.ndarray,
    *,
    spec: RegimeSpec,
    crisis_mask: np.ndarray,
) -> Labelling:
    """Turn soft probabilities into a committed label sequence, under anti-thrashing.

    The model's beliefs are reported as they are. The one thing this function changes about
    the probabilities is the crisis revision, and it does that by calling
    ``apply_crisis_override`` itself rather than trusting a caller to have done it — see the
    note there for why the two steps are not separable. Everything else a user reads in
    ``prob_trending_up`` is what the model believes, not what the filter allowed it to say.

    The rules, in the order they are applied to each bar:

    * **Crisis overrides everything.** A bar whose trailing volatility breached the
      threshold is Crisis, and the minimum-duration rule is deliberately bypassed. That is
      the point of the rule: risk control that waits five bars to confirm a crash has
      already failed.
    * **A switch needs confidence *and* patience.** The committed state changes only when
      the candidate holds at least ``commit_confidence`` *and* the incumbent has been
      held for ``min_state_duration`` bars.
    * **Below ``transition_confidence`` the bar is Transition.** No state has a mandate,
      so the honest label is the gray zone rather than the runner-up.
    """
    named = apply_crisis_override(named, crisis_mask, spec.override_probability)
    n_obs = named.shape[0]
    states: list[RegimeState | None] = [None] * n_obs
    confidence = np.full(n_obs, np.nan)
    crisis_at = _CANONICAL.index(RegimeState.CRISIS)

    committed: RegimeState | None = None
    #: Bars the incumbent has occupied, *including* the bar it was committed on. Starting
    #: at zero would make `min_state_duration` mean one bar longer than it says: a state
    #: committed at t and replaced at t+1 would be reported as "held 0 bars", so the knob
    #: would silently be off by one against its own label in the UI.
    held = 0
    switches = 0
    override_bars = 0

    for t in range(n_obs):
        row = named[t]
        if not np.isfinite(row).all():
            continue  # warm-up

        top_index = int(np.argmax(row))
        top_state = _CANONICAL[top_index]
        top_confidence = float(row[top_index])

        if crisis_mask[t]:
            # The override has already been written into the probabilities by
            # `apply_crisis_override`, so the row's own argmax is Crisis. This branch
            # exists for the *bypass*: it commits on this bar regardless of
            # `min_state_duration` or `commit_confidence`.
            if committed is not RegimeState.CRISIS:
                switches += 1 if committed is not None else 0
            committed = RegimeState.CRISIS
            held = 1
            override_bars += 1
            states[t] = RegimeState.CRISIS
            confidence[t] = float(row[crisis_at])
            continue

        if committed is None:
            if top_confidence >= spec.commit_confidence:
                committed = top_state
                held = 1
                states[t] = committed
            else:
                # No state has ever been committed, so there is no incumbent whose
                # probability could be reported. The weak leader's is the only figure
                # available, and it is labelled as such by the Transition state.
                states[t] = RegimeState.TRANSITION
                confidence[t] = top_confidence
                continue

        elif top_state is committed:
            held += 1
            states[t] = (
                RegimeState.TRANSITION
                if top_confidence < spec.transition_confidence
                else committed
            )

        elif top_confidence >= spec.commit_confidence and held >= spec.min_state_duration:
            committed = top_state
            held = 1
            switches += 1
            states[t] = committed

        elif top_confidence < spec.transition_confidence:
            held += 1
            states[t] = RegimeState.TRANSITION

        else:
            held += 1
            states[t] = committed

        # The confidence reported is the probability of the **committed state**, not the
        # largest probability in the row. The two differ on exactly the bars that matter:
        # while the incumbent is being held against a challenger that has not yet earned a
        # switch, the row's maximum can belong to the challenger. Reporting that figure
        # would tell the user "70% confident it is Trending Up" when the 0.70 was Ranging's
        # — and it would make the interface's own "below 55%, treat as a gray zone"
        # warning unreachable, because the maximum is high by construction whenever any
        # state is doing well.
        confidence[t] = float(row[_CANONICAL.index(committed)])

    return Labelling(
        states=states,
        confidence=confidence,
        switches=switches,
        override_bars=override_bars,
        named=named,
    )


def crisis_mask_for(volatility: np.ndarray, threshold: float) -> np.ndarray:
    """Bars where the realized-volatility override fires."""
    with np.errstate(invalid="ignore"):
        return np.isfinite(volatility) & (volatility > threshold)


# ------------------------------------------------------------------ evaluation


def net_value_add(
    returns: np.ndarray,
    states: list[RegimeState | None],
    *,
    switching_cost_penalty: float,
) -> dict[str, Any]:
    """Return improvement over buy-and-hold, less the cost of switching into it.

    The specification asks for "(return improvement) − (switching_cost × switch_count)"
    without ever saying what strategy the improvement is measured against, and a metric
    with an unstated benchmark is not a metric. The definition used here is stated in
    full, is deliberately the simplest one that can be defended, and is written into the
    output's provenance so a reader never has to guess what was measured:

    * **Position.** Long when the committed state is ``trending_up``; flat otherwise.
      Flat in ``ranging`` because ranging is the mean-reverting state and a momentum
      position is the wrong trade there; flat in ``crisis`` because that is the entire
      purpose of detecting it.
    * **Timing.** The state is known at the close of bar ``t``, so the position earns the
      return of bar ``t+1``. Applying it to bar ``t``'s own return is the classic
      same-bar look-ahead and is what makes so many regime studies unfalsifiable.
    * **Benchmark.** Buy-and-hold over the same rows, so the two are compared on
      identical data.

    The count that the cost is charged on is reported as ``position_changes``, not
    ``switch_count``. The report has a top-level ``switch_count`` already, and it counts
    *committed state* changes — a different number, because a change from ``ranging`` to
    ``crisis`` moves no position and costs nothing under this rule. Two fields with one
    name is how a reader ends up checking the arithmetic by hand and finding it wrong.

    ``bars_long`` is reported because a run that never committed to ``trending_up`` has a
    ``return_improvement`` of exactly ``-benchmark`` and a Net Value-Add that says nothing
    about the model at all. That is a real outcome — the heuristic path produces it whenever
    volatility sits near the crisis threshold — and it has to be visible as an outcome
    rather than look like a scored evaluation.

    **This is an in-sample descriptive statistic, not a backtest.** The states were
    fitted on these rows, so the number is an upper bound on what the rule could have
    earned. It is reported to compare two parameter settings, never to claim an edge.
    """
    total = min(len(returns), len(states))
    strategy = 0.0
    benchmark = 0.0
    changes = 0
    bars_long = 0
    previous_position = 0.0
    counted = 0

    for t in range(total - 1):
        state = states[t]
        if state is None:
            continue
        forward_return = returns[t + 1]
        if not np.isfinite(forward_return):
            continue
        position = 1.0 if state is RegimeState.TRENDING_UP else 0.0
        if position != previous_position:
            changes += 1
            previous_position = position
        if position:
            bars_long += 1
        strategy += position * float(forward_return)
        benchmark += float(forward_return)
        counted += 1

    cost = switching_cost_penalty * changes
    improvement = strategy - benchmark
    return {
        "definition": "long in trending_up, flat otherwise; position set at t earns the "
        "return of t+1; benchmark is buy-and-hold over the same bars",
        "in_sample": True,
        "bars_scored": counted,
        "bars_long": bars_long,
        "strategy_log_return": strategy,
        "benchmark_log_return": benchmark,
        "return_improvement": improvement,
        "position_changes": changes,
        "switching_cost_penalty": switching_cost_penalty,
        "switching_cost": cost,
        "net_value_add": improvement - cost,
    }


# ------------------------------------------------------------------ stationarity


def stationarity_warnings(matrix: np.ndarray, columns: list[str]) -> list[dict[str, Any]]:
    """Augmented Dickey-Fuller on each observation, reported rather than enforced.

    A regime model fitted on a non-stationary series is fitting a trend: the states end
    up ordered by *when* a bar occurred rather than by what the market was doing, and the
    model will assign a state it has never seen before to any future bar. That is worth
    saying out loud.

    It is a warning and not a refusal because the test has low power on short series and
    the caller may know something we do not. The specification asks for a warning; a
    refusal here would be us overruling a defensible choice.
    """
    from statsmodels.tsa.stattools import adfuller

    warnings: list[dict[str, Any]] = []
    for index, column in enumerate(columns):
        series = matrix[:, index]
        series = series[np.isfinite(series)]
        if series.size < 20 or np.allclose(series, series[0]):
            warnings.append(
                {
                    "feature": column,
                    "stationary": None,
                    "detail": "Too short or constant to test.",
                }
            )
            continue
        try:
            # result_object=False pins the tuple return and suppresses statsmodels'
            # FutureWarning about it changing to an ADFullerResult in 0.16; left to be
            # raised, that warning is an error under the suite's error::FutureWarning
            # filter and would be swallowed by the catch below as "did not converge".
            statistic, p_value = adfuller(series, autolag="AIC", result_object=False)[:2]
        except Exception:  # noqa: BLE001 - a diagnostic must never fail the run
            warnings.append(
                {"feature": column, "stationary": None, "detail": "Test did not converge."}
            )
            continue
        stationary = bool(p_value <= 0.05)
        warnings.append(
            {
                "feature": column,
                "stationary": stationary,
                "adf_statistic": float(statistic),
                "p_value": float(p_value),
                "detail": ""
                if stationary
                else "Not stationary at 5%: the model may be separating this series by "
                "time rather than by regime. A price level is the usual cause; a return "
                "or a volatility is the usual fix.",
            }
        )
    return warnings


# ------------------------------------------------------------------- driver


def detect(
    frame: pd.DataFrame,
    *,
    time_column: str,
    spec: RegimeSpec,
    timeframe: Timeframe | None,
    asset_class: AssetClass,
) -> RegimeOutcome:
    """Run the whole pipeline and return the output frame plus its report."""
    if time_column not in frame.columns:
        raise ValidationFailure(f"Dataset has no '{time_column}' column.")
    if "close" not in frame.columns:
        raise ValidationFailure(
            "Regime detection needs a 'close' column.",
            hint="This does not look like price data. Module D reads OHLCV bars.",
        )

    timestamps = frame[time_column]
    if timestamps.isna().any():
        raise ValidationFailure("The time column contains empty values.")
    if not timestamps.is_monotonic_increasing:
        raise ValidationFailure(
            "The time column is not sorted ascending.",
            hint="Run B3 Cleaning with 'sort by time' before detecting regimes; a "
            "misordered series makes every trailing window wrong.",
        )
    if bool(timestamps.duplicated().any()):
        raise ValidationFailure(
            "The time column contains duplicate timestamps.",
            hint="Two bars sharing a timestamp double-count one interval. Repair it in "
            "B3 Cleaning first.",
        )

    close = frame["close"].to_numpy(dtype=float)
    volume = frame["volume"].to_numpy(dtype=float) if "volume" in frame.columns else None
    periods_per_year = annualisation_factor(timeframe, asset_class)

    matrix, columns, computed = _observation_matrix(
        frame, spec, close=close, volume=volume, periods_per_year=periods_per_year
    )
    valid_from = _first_complete_row(matrix)

    volatility = computed[RegimeFeature.REALIZED_VOLATILITY]
    crisis = crisis_mask_for(volatility, spec.crisis_threshold)
    warnings = stationarity_warnings(matrix[valid_from:], columns)

    n_obs = matrix.shape[0]
    named = np.full((n_obs, len(_CANONICAL)), np.nan)
    model_diagnostics: list[dict[str, Any]] = []
    method = "fitted"

    if spec.model_type is RegimeModel.RULES:
        # No refit schedule: this path has no parameters to estimate.
        named = _rules_probabilities(_compute_returns(close), volatility, spec)
        named[:valid_from] = np.nan
        method = "heuristic"
    else:
        segments = _refit_schedule(n_obs, spec, valid_from)
        if spec.model_type is RegimeModel.HMM:
            named, model_diagnostics = _fit_and_label_hmm(matrix, spec, segments)
        elif spec.model_type is RegimeModel.GMM:
            named, model_diagnostics = _fit_and_label_mixture(matrix, spec, segments)
        else:
            if RegimeFeature.ADX not in spec.features and RegimeFeature.RSI not in spec.features:
                raise ValidationFailure(
                    "The hybrid model needs at least one carried indicator.",
                    hint="Select adx or rsi — the point of the hybrid is to predict the "
                    "HMM from faster-moving indicators.",
                )
            indicators = np.column_stack(
                [frame[feature.value].to_numpy(dtype=float) for feature in spec.features
                 if feature in (RegimeFeature.ADX, RegimeFeature.RSI)]
            )
            base_features = tuple(
                feature
                for feature in spec.features
                if feature not in (RegimeFeature.ADX, RegimeFeature.RSI)
            )
            base_spec = dataclasses.replace(spec, features=base_features)
            named, model_diagnostics = _fit_and_label_hybrid(
                matrix[:, [columns.index(f.value) for f in base_features]],
                indicators,
                base_spec,
                segments,
            )

    # Counted before the labelling call, which is where the override is applied. A breach in
    # the warm-up has no posterior to revise, so it is counted rather than labelled — a row
    # with a state and no probability would break the invariant that the state is the argmax
    # of its own row. Counted rather than ignored, because "the risk control was blind for the
    # first 273 bars" is something the user needs to know about a crisis detector.
    warmup_breaches = int(np.sum(crisis & ~np.isfinite(named).all(axis=1)))

    # `commit_labels` applies the crisis override itself and hands back the revised
    # posterior, so the frame below is built from the same array the labels were read off.
    # Revising a separate copy here is what let `dominant_state` and `prob_crisis` disagree.
    labelling = commit_labels(named, spec=spec, crisis_mask=crisis)
    named = labelling.named

    evaluation = net_value_add(
        _compute_returns(close),
        labelling.states,
        switching_cost_penalty=spec.switching_cost_penalty,
    )
    # The column is named for what it holds. `frame[time_column]` is the *observation*
    # instant of the bar — `open_time` on a Binance frame — not the instant the bar closed.
    # Naming it for the bar's close made the label a lie, and a consumer that joined on the
    # name rather than reading the values would have aligned every regime with the following
    # bar: off by one, silently, in a frame that looks entirely normal. See DECISIONS §14,
    # §86, and the guard in `tests/test_leakage_guards.py` that keeps this name honest — the
    # guard is a source-level one, so this comment deliberately does not spell the old name.
    out = pd.DataFrame(
        {
            "open_time": pd.to_datetime(frame[time_column], utc=True),
            "dominant_state": [
                state.value if state is not None else None for state in labelling.states
            ],
            "prob_trending_up": named[:, _CANONICAL.index(RegimeState.TRENDING_UP)],
            "prob_trending_down": named[:, _CANONICAL.index(RegimeState.TRENDING_DOWN)],
            "prob_ranging": named[:, _CANONICAL.index(RegimeState.RANGING)],
            "prob_crisis": named[:, _CANONICAL.index(RegimeState.CRISIS)],
            "confidence": labelling.confidence,
        }
    )

    matrix_labels, matrix_values = _committed_transition_matrix(labelling.states)

    labelled_rows = int(np.isfinite(named).all(axis=1).sum())
    report: dict[str, Any] = {
        "method": method,
        "model_type": spec.model_type.value,
        "n_states_requested": spec.n_states,
        "features": columns,
        "periods_per_year": periods_per_year,
        "volatility_window": spec.volatility_window,
        "crisis_threshold": spec.crisis_threshold,
        "min_state_duration": spec.min_state_duration,
        "transition_confidence": spec.transition_confidence,
        "commit_confidence": spec.commit_confidence,
        "min_train_bars": spec.min_train_bars,
        "refit_every": spec.refit_every,
        # Two different prefixes, and they are not interchangeable. `warmup_bars` is how
        # many bars the observation window eats before a feature vector exists at all;
        # `unlabelled_bars` is the whole null prefix, which also includes the bars the
        # first model was trained on and therefore never labelled. Reporting only the
        # first made `warmup_bars` read as "the rows before the first label", which on a
        # 200-row frame with a 60-bar training window is wrong by a factor of three and a
        # half — and it disagreed with `state_counts["warmup"]`, which counts the second.
        "warmup_bars": valid_from,
        "unlabelled_bars": n_obs - labelled_rows,
        "rows": n_obs,
        "labelled_rows": labelled_rows,
        "refit_points": model_diagnostics,
        "refit_count": len(model_diagnostics),
        "switch_count": labelling.switches,
        "crisis_override_bars": labelling.override_bars,
        "crisis_bars_in_warmup": warmup_breaches,
        "stationarity": warnings,
        "net_value_add": evaluation,
        "state_counts": _state_counts(labelling.states),
        "state_durations": _run_lengths(labelling.states),
        "transition_matrix": {"labels": matrix_labels, "values": matrix_values},
    }
    latest = model_diagnostics[-1] if model_diagnostics else {}
    report["bic_latest_refit"] = latest.get("bic")
    report["aic_latest_refit"] = latest.get("aic")
    return RegimeOutcome(frame=out, report=report)


def _compute_returns(close: np.ndarray) -> np.ndarray:
    return _log_returns(close)


def _state_counts(states: list[RegimeState | None]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for state in states:
        key = state.value if state is not None else "warmup"
        counts[key] = counts.get(key, 0) + 1
    return counts


#: The order the transition matrix is indexed in — every named state, Transition included,
#: because a matrix that omitted Transition would redistribute its mass into the rows of
#: the states it was entered from and report transitions that never happened.
_MATRIX_ORDER: tuple[RegimeState, ...] = tuple(RegimeState)


def _run_lengths(states: list[RegimeState | None]) -> dict[str, dict[str, float]]:
    """How long each committed state lasts, in bars.

    Mean *and* longest, because the two answer different questions. A mean of 12 bars
    sounds healthy until you see the longest run is 400: that distribution is one long
    regime plus a swarm of one-bar flickers, and the anti-thrashing rule has not done its
    job. The mean alone hides exactly that shape.
    """
    runs: dict[str, list[int]] = {}
    current: RegimeState | None = None
    length = 0
    for state in states:
        if state is current:
            length += 1
            continue
        if current is not None:
            runs.setdefault(current.value, []).append(length)
        current = state
        length = 1
    if current is not None:
        runs.setdefault(current.value, []).append(length)

    return {
        name: {
            "runs": len(lengths),
            "mean_bars": sum(lengths) / len(lengths),
            "longest_bars": max(lengths),
        }
        for name, lengths in sorted(runs.items())
    }


def _committed_transition_matrix(
    states: list[RegimeState | None],
) -> tuple[list[str], list[list[float]]]:
    """Row-normalised transition counts between *committed* states.

    This is the matrix the user reads, and it is deliberately not the fitted HMM's
    ``transition`` matrix. The fitted one is a parameter of the model: it says how the
    latent states move, before the crisis override and before the minimum-duration rule
    have had their say. The matrix below is what the output columns actually do, which is
    the thing that can be checked against the frame.
    """
    index = {state: position for position, state in enumerate(_MATRIX_ORDER)}
    labelled = np.array(
        [index[state] for state in states if state is not None], dtype=int
    )
    matrix = regime_models.transition_matrix_from_states(labelled, len(_MATRIX_ORDER))
    return [state.value for state in _MATRIX_ORDER], matrix
