"""C8 requests: exploration defaults to the entire selected dataset."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


class ExploratoryScope(StrEnum):
    FULL_DATASET = "full_dataset"
    CUSTOM_DATE_RANGE = "custom_date_range"
    TRAIN_SPAN = "train_span"
    ROLLING = "rolling"


class MissingPolicy(StrEnum):
    PAIRWISE_COMPLETE = "pairwise_complete"
    DROP_ANY_MISSING = "drop_any_missing"
    MEDIAN_IMPUTE_SCOPE = "median_impute_scope"
    MEAN_IMPUTE_SCOPE = "mean_impute_scope"
    LEAVE_NAN = "leave_nan"


class CorrMethod(StrEnum):
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    KENDALL = "kendall"
    PARTIAL_PEARSON = "partial_pearson"
    PARTIAL_SPEARMAN = "partial_spearman"


class PCAStandardizer(StrEnum):
    NONE = "none"
    STANDARD = "standard"
    ROBUST = "robust"
    MINMAX = "minmax"


class PCAComponentSelection(StrEnum):
    MANUAL = "manual"
    VARIANCE_THRESHOLD = "variance_threshold"
    EIGENVALUE_GT_1 = "eigenvalue_gt_1"
    SCREE = "scree"
    PARALLEL = "parallel"


class ScopeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    features: list[str] = Field(min_length=1, max_length=250)
    scope: ExploratoryScope = ExploratoryScope.FULL_DATASET
    start_time: datetime | None = Field(
        default=None, validation_alias=AliasChoices("date_start", "start_time")
    )
    end_time: datetime | None = Field(
        default=None, validation_alias=AliasChoices("date_end", "end_time")
    )
    split_plan_id: str | None = None
    fold: int | None = Field(default=None, ge=0)
    pre_transform: Literal[
        "none",
        "first_difference",
        "percent_change",
        "log_difference",
        "absolute_return",
        "squared_return",
        "rank_transform",
        "winsorize_1_99",
        "robust_scale",
        "standard_scale",
        "minmax_scale",
        "median_impute_scope",
        "mean_impute_scope",
    ] = "none"
    missing_policy: MissingPolicy = MissingPolicy.PAIRWISE_COMPLETE
    constant_policy: Literal["exclude_constants", "warn_only", "refuse_if_constants"] = (
        "exclude_constants"
    )
    min_obs: int = Field(default=30, ge=3)
    strict_causal: bool = Field(
        default=False, validation_alias=AliasChoices("strict_causal_mode", "strict_causal")
    )

    @model_validator(mode="after")
    def validate_scope(self) -> ScopeRequest:
        if len(self.features) != len(set(self.features)):
            raise ValueError("Select each feature only once.")
        if self.scope == "custom_date_range":
            if self.start_time is None or self.end_time is None:
                raise ValueError("Custom date range requires start_time and end_time.")
            self.start_time = (
                self.start_time.replace(tzinfo=UTC)
                if self.start_time.tzinfo is None
                else self.start_time.astimezone(UTC)
            )
            self.end_time = (
                self.end_time.replace(tzinfo=UTC)
                if self.end_time.tzinfo is None
                else self.end_time.astimezone(UTC)
            )
            if self.start_time > self.end_time:
                raise ValueError("start_time must not be after end_time.")
        elif self.start_time is not None or self.end_time is not None:
            raise ValueError("Dates apply only to custom date range scope.")
        if self.scope == "train_span":
            if not self.split_plan_id:
                raise ValueError("Train-span scope requires a split plan.")
        elif self.split_plan_id is not None or self.fold is not None:
            raise ValueError("Split plan and fold apply only to train-span scope.")
        return self


class CorrelationRequest(ScopeRequest):
    method: CorrMethod = CorrMethod.PEARSON
    target: str | None = Field(
        default=None, validation_alias=AliasChoices("target_feature", "target")
    )
    controls: list[str] = Field(
        default_factory=list,
        max_length=30,
        validation_alias=AliasChoices("control_features", "controls"),
    )
    group_column: str | None = Field(
        default=None, validation_alias=AliasChoices("group_by_column", "group_column")
    )
    group_min_obs: int = Field(default=30, ge=3)
    group_bins: list[float] | None = Field(default=None, min_length=2, max_length=51)
    alpha: float = Field(default=0.05, gt=0, lt=1)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    multiple_testing: Literal["none", "bonferroni", "benjamini_hochberg", "fdr_bh"] = "none"
    compute_p_values: bool = True
    compute_confidence_intervals: bool = True
    ci_method: Literal["fisher_z", "bootstrap", "none"] | None = None
    bootstrap_samples: int = Field(default=200, ge=20, le=500)
    random_state: int = Field(default=0, ge=0)
    calculate_vif: bool = True
    vif_threshold: float = Field(default=5.0, gt=1)
    include_intercept: bool = True
    enable_redundancy_groups: bool = True
    redundancy_threshold: float = Field(default=0.9, gt=0, le=1)
    max_lag: int = Field(default=0, ge=0, le=120)
    enable_ccf: bool = False
    lag_min: int = Field(default=-10, ge=-120, le=120)
    lag_max: int = Field(default=10, ge=-120, le=120)
    ccf_min_obs: int = Field(default=30, ge=3)
    enable_rolling: bool = False
    rolling_window: int | None = Field(default=None, ge=3, le=5000)
    rolling_step: int = Field(default=1, ge=1)
    rolling_min_periods: int | None = Field(default=None, ge=3)
    rolling_pair_selection: Literal["all_pairs", "top_n", "selected_pairs"] = "all_pairs"
    rolling_top_pairs: int = Field(default=10, ge=1, le=100)
    rolling_pairs: list[tuple[str, str]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_correlation(self) -> CorrelationRequest:
        names = set(self.features) | ({self.target} if self.target else set())
        if len(names) < 2:
            raise ValueError("Correlation needs at least two distinct features, including target.")
        if self.method.startswith("partial") != bool(self.controls):
            raise ValueError("Select control features for partial correlation only.")
        if len(self.controls) != len(set(self.controls)) or set(self.controls) & names:
            raise ValueError("Controls must be unique and separate from features and target.")
        if len(names) > 250:
            raise ValueError("At most 250 features, including target, are supported.")
        if self.max_lag:
            self.enable_ccf, self.lag_min, self.lag_max = True, -self.max_lag, self.max_lag
        if self.enable_ccf and not self.target:
            raise ValueError("Lead-lag diagnostics require target_feature.")
        if self.lag_min > self.lag_max:
            raise ValueError("lag_min must not exceed lag_max.")
        if self.scope == "rolling":
            if self.enable_ccf:
                raise ValueError("Rolling scope cannot be combined with lead-lag CCF.")
            if self.group_column:
                raise ValueError("Rolling scope cannot be combined with conditional groups.")
            if self.rolling_pair_selection == "top_n":
                raise ValueError(
                    "Full-sample top-N pair selection is not causal. Use all_pairs or "
                    "selected_pairs in rolling scope."
                )
            if "calculate_vif" in self.model_fields_set and self.calculate_vif:
                raise ValueError("Rolling scope cannot be combined with static VIF.")
            if (
                "enable_redundancy_groups" in self.model_fields_set
                and self.enable_redundancy_groups
            ):
                raise ValueError("Rolling scope cannot be combined with redundancy grouping.")
            self.enable_rolling = True
            self.calculate_vif = False
            self.enable_redundancy_groups = False
            self.rolling_window = self.rolling_window or 60
            self.rolling_min_periods = self.rolling_min_periods or min(
                self.min_obs, self.rolling_window
            )
            if self.rolling_min_periods > self.rolling_window:
                raise ValueError("rolling_min_periods must not exceed rolling_window.")
            if self.rolling_pair_selection == "selected_pairs" and not self.rolling_pairs:
                raise ValueError("Select at least one rolling pair.")
        elif self.enable_rolling or self.rolling_window is not None:
            raise ValueError("Rolling correlation requires scope='rolling'.")
        for a, b in self.rolling_pairs:
            if a == b or a not in names or b not in names:
                raise ValueError("Rolling pairs must contain two distinct selected features.")
        if self.group_bins and (
            not self.group_column
            or any(b <= a for a, b in zip(self.group_bins, self.group_bins[1:], strict=False))
        ):
            raise ValueError(
                "Numeric group bins require a grouping column and increasing boundaries."
            )
        return self


class PCARequest(ScopeRequest):
    standardizer: PCAStandardizer = PCAStandardizer.STANDARD
    component_selection: PCAComponentSelection = PCAComponentSelection.MANUAL
    n_components: int | None = Field(default=2, ge=1, le=100)
    variance_threshold: float = Field(default=0.95, gt=0, le=1)
    write_dataset: bool = False
    eigenvalue_threshold: float = Field(default=1.0, ge=0)
    parallel_permutations: int = Field(default=100, ge=1, le=200)
    random_state: int = Field(default=0, ge=0)
    whiten: bool = False
    destination: Literal["outputs", "processed_data"] | None = None
    target: str | None = Field(
        default=None, validation_alias=AliasChoices("target_feature", "target")
    )
    keep_original_features: bool = Field(
        default=False,
        validation_alias=AliasChoices("include_original_features", "keep_original_features"),
    )
    include_target_feature: bool = False
    include_carried_columns: bool = False
    carry_columns: list[str] = Field(default_factory=list)
    name: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        validation_alias=AliasChoices("new_dataset_name", "name"),
    )
    component_prefix: str = Field(
        default="pc_", min_length=1, max_length=40, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$"
    )

    @model_validator(mode="after")
    def validate_pca(self) -> PCARequest:
        if len(self.features) < 2:
            raise ValueError("PCA requires at least two selected numeric features.")
        if self.scope == "rolling":
            raise ValueError("Rolling PCA is deferred; choose a static fitting scope.")
        if self.write_dataset and self.pre_transform == "rank_transform":
            raise ValueError(
                "rank_transform is diagnostic-only; create causal rank features in C5."
            )
        return self
