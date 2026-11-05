"""Strict, versioned research and inference contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Inputs(Contract):
    dataset_id: str
    split_plan_id: str
    features: list[str] = Field(min_length=1, max_length=100)
    fold: int = Field(default=0, ge=0)
    # Economics must be measured in original units, independently of the model inputs.
    market_dataset_id: str | None = None
    return_column: str | None = None
    # None keeps legacy manifests readable; new explicit return columns must declare it.
    return_type: Literal["simple", "log"] | None = None
    price_column: str | None = "close"
    adx_column: str | None = None
    rsi_column: str | None = None
    volatility_column: str | None = None

    @model_validator(mode="after")
    def unique(self):
        if len(set(self.features)) != len(self.features):
            raise ValueError("Feature names must be unique and ordered.")
        return self


class Target(Contract):
    source: Literal["column", "rules", "teacher"] = "rules"
    column: str | None = None
    objective: Literal["current", "future"] = "current"
    horizon: int = Field(default=1, ge=1, le=1000)
    availability_lag: int = Field(default=0, ge=0, le=1000)
    rule_feature: str | None = None
    lower: float = -0.001
    upper: float = 0.001

    @model_validator(mode="after")
    def valid(self):
        if self.source == "column" and not self.column:
            raise ValueError("A supervised target column must be specified.")
        if self.source == "rules" and not self.rule_feature:
            raise ValueError("Select the upstream feature used to generate rule labels.")
        if self.lower >= self.upper:
            raise ValueError("Lower label threshold must be below upper threshold.")
        return self


class StateConfig(Contract):
    mode: Literal["fixed", "automatic"] = "fixed"
    count: int = Field(default=3, ge=2, le=12)
    candidates: list[int] = Field(default_factory=lambda: [2, 3, 4], max_length=11)
    template: Literal[
        "statistical", "two", "bull_bear_sideways", "four", "trend_volatility", "custom"
    ] = "statistical"
    minimum_occupancy: float = Field(default=0.01, ge=0, lt=0.5)
    criterion: Literal[
        "bic",
        "aic",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "validation_log_loss",
        "validation_accuracy",
    ] = "bic"

    @model_validator(mode="after")
    def valid(self):
        if not self.candidates or any(k < 2 or k > 12 for k in self.candidates):
            raise ValueError("Candidate counts must be between 2 and 12.")
        return self


class Search(Contract):
    method: Literal["manual", "grid", "random", "bayesian", "optuna"] = "manual"
    space: dict[str, list[Any]] = Field(default_factory=dict)
    trials: int = Field(default=12, ge=1, le=100)


class Experiment(Contract):
    schema_version: Literal[1] = 1
    name: str = Field(default="Regime experiment", min_length=1, max_length=120)
    inputs: Inputs
    model: str = "gaussian_hmm"
    parameters: dict[str, Any] = Field(default_factory=dict)
    states: StateConfig = Field(default_factory=StateConfig)
    target: Target | None = None
    search: Search = Field(default_factory=Search)
    seed: int = Field(default=42, ge=0, le=2147483647)


class LabelRequest(Contract):
    mode: Literal["automatic", "template", "manual"] = "automatic"
    labels: dict[str, str] = Field(default_factory=dict)
    template: Literal["trend", "trend_volatility", "volatility", "stress"] = "trend"


class ValidationRequest(Contract):
    mode: Literal["chronological", "expanding", "rolling"] = "chronological"
    # Rolling/expanding validation consumes a saved C6 walk-forward plan.
    split_plan_id: str | None = None
    prepared_datasets: dict[int, str] = Field(default_factory=dict)


class RobustnessRequest(Contract):
    seeds: list[int] = Field(default_factory=lambda: [7, 42, 123], min_length=1, max_length=10)
    counts: list[int] = Field(default_factory=list, max_length=8)
    feature_sets: list[list[str]] = Field(default_factory=list, max_length=8)
    configurations: list[dict[str, Any]] = Field(default_factory=list, max_length=8)
    training_fractions: list[float] = Field(default_factory=lambda: [1.0], max_length=8)
    initializations: list[Literal["random", "kmeans"]] = Field(default_factory=list)


class Approval(Contract):
    rationale: str = Field(min_length=5, max_length=4000)
    validation_id: str


class Inference(Contract):
    approval_id: str
    dataset_id: str
    as_of: str | None = None
    input_mode: Literal["prepared", "replay"] = "prepared"
    decoding: Literal["filter", "smooth", "fixed_lag", "viterbi"] = "filter"
    lag: int = Field(default=5, ge=1, le=100)
    minimum_persistence: int = Field(default=3, ge=1, le=1000)
    probability_threshold: float = Field(default=0.55, ge=0, le=1)
    hysteresis: float = Field(default=0.1, ge=0, le=1)
    low_confidence: float = Field(default=0.5, ge=0, le=1)
    drift_threshold: float = Field(default=0.2, gt=0, le=1)
    cusum_threshold: float | None = Field(default=None, gt=0)


class Comparison(Contract):
    experiment_ids: list[str] = Field(min_length=1, max_length=20)
