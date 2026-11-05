"""Module C7 — scaling and normalisation over HTTP.

Fitted scalers accept an optional split plan. Without one, they estimate parameters from
the entire selected dataset. With one, they fit only its training rows, which is the scope
to use for model evaluation. Walk-forward plans require a fold. Rolling z-score estimates
statistics per row from a trailing window and takes neither a plan nor a fold.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from app.models.dataset import DatasetMeta
from app.models.enums import JobStatus, ScalerType

#: The default trailing window, in bars, for the rolling z-score. Sixty bars is roughly
#: three months of hourly data; the field is required rather than defaulted to be silent,
#: because a window this short is a statement about how fast the relationship is assumed
#: to change and the report states it back.
DEFAULT_WINDOW = 60


class ScalingRequest(BaseModel):
    """Parameters for one scaling run.

    ``write`` mirrors the regime request: a run may be previewed without being stored, and
    the preview is computed the same way either way, so a preview cannot disagree with the
    artifact it would have produced.
    """

    dataset_id: str
    scaler_type: ScalerType

    #: Optional for fitted scalers: absent means fit the entire selected dataset.
    #: A supplied plan restricts fitting to its training rows. Rolling takes no plan.
    split_plan_id: str | None = None

    #: Which fold of a walk-forward plan to fit against. Required for a TSCV plan, refused
    #: for the others: a percentage or date plan has exactly one train span, so a fold
    #: would be a number with no meaning.
    fold: int | None = Field(default=None, ge=0)

    #: Columns to rescale. Empty means "every numeric column", which is resolved
    #: server-side so the page and the request agree on what "all" contains.
    features: list[str] = Field(default_factory=list)

    #: Trailing window in bars for ``rolling_zscore``. Ignored by the other three, and
    #: refused rather than ignored if explicitly set for them — an option that is accepted
    #: and quietly dropped is how a user comes to believe a scaler behaved causally when
    #: it did not.
    window: int | None = Field(default=None, ge=2, le=10_000)

    write: bool = True
    name: str | None = None

    @model_validator(mode="after")
    def _check(self) -> ScalingRequest:
        if self.split_plan_id is not None and not self.split_plan_id.strip():
            raise ValueError("split_plan_id must be a saved plan ID or null.")
        if self.fold is not None and self.split_plan_id is None:
            raise ValueError("fold requires a split_plan_id.")
        if not self.scaler_type.is_fitted and self.split_plan_id is not None:
            raise ValueError(
                "rolling_zscore estimates its statistics from a trailing window at each "
                "row, so it is causal by construction and takes no split plan. Applying "
                "one would have no effect, and accepting it would suggest the plan was "
                "doing something."
            )
        if self.scaler_type is not ScalerType.ROLLING_ZSCORE and self.window is not None:
            raise ValueError(
                f"window applies to rolling_zscore only; {self.scaler_type.value} has no "
                "trailing window. Set it on a rolling run, or drop it."
            )
        return self

    @property
    def effective_window(self) -> int:
        """The window this run will actually use. One reading of the default, so the
        report and the computation cannot disagree about it."""
        return self.window or DEFAULT_WINDOW


class ScalerParameter(BaseModel):
    """One feature's fitted parameters, as a row of the stored table.

    The affine pair is the part that matters downstream. Every fitted scaler here is
    affine — ``x_scaled = (x − center) / scale`` — so Modules E and F can invert any of
    them with ``x = x_scaled · scale + center`` without knowing which one was used:

    * standard → ``center = mean_``, ``scale = scale_`` (the sample standard deviation)
    * min-max  → ``center = data_min_``, ``scale = data_range_``
    * robust   → ``center = median_``, ``scale = IQR``

    The raw sklearn attributes are stored beside it so the JSON is readable against the
    library's own names, which is what the specification asks for (``mean_``, ``scale_``,
    ``min_``, ``max_``). ``center_``/``scale_`` are the *interpretation*; the others are
    the *provenance*. Both are kept, because either alone leaves a reader unable to check
    the other.
    """

    feature: str
    scaler_type: ScalerType

    #: The inversion pair. Named with trailing underscores because they are the fitted
    #: quantities, like sklearn's own.
    center_: float
    scale_: float

    #: Whether a constant column was left alone. ``scale_`` of 0 or 1 with no variance is
    #: the honest report of a column with nothing to divide by — the alternative, a
    #: division by zero, produces NaN for the whole column and looks like missing data.
    constant: bool = False

    #: Fitted on this many rows. Less than the span's row count whenever the feature has a
    #: warm-up of leading nulls, and reported so the difference is visible.
    fitted_rows: int = Field(ge=0)

    #: The raw fitted attributes, keyed by sklearn's names.
    attributes: dict[str, float | None] = Field(default_factory=dict)

    #: The inversion, written out with this feature's numbers substituted.
    formula: str


class ScalingReport(BaseModel):
    """What the run did, and against exactly what."""

    dataset_id: str
    dataset_name: str
    scaler_type: ScalerType

    #: The plan and fold, or null for full-dataset fitting and rolling z-score.
    split_plan_id: str | None
    split_plan_name: str | None
    fold: int | None

    #: Resolved fit scope (train, full_dataset, or trailing window), count and bounds.
    fit_role: str | None
    fit_dataset_id: str
    fit_dataset_name: str
    fit_rows: int
    fit_start_time: Any = None
    fit_end_time: Any = None

    rows: int
    features: list[str]
    window: int | None
    rows_used_for_fit: int

    #: Per-feature row counts, so a feature with a long warm-up is visible rather than
    #: silently fitted on a shorter span than its neighbours.
    per_feature_rows: dict[str, int] = Field(default_factory=dict)

    warnings: list[str] = Field(default_factory=list)


class ScalingAccepted(BaseModel):
    """What ``POST /api/scaling/fit_transform`` returns immediately.

    Same contract as ``RegimeRunAccepted``: the work has not happened, and ``poll`` is the
    only way to learn whether it succeeded.
    """

    job_id: str
    status: JobStatus
    poll: str


class ScalingCreated(BaseModel):
    """The terminal payload of a scaling job — what ``GET /api/jobs/{id}`` carries."""

    dataset: DatasetMeta | None
    params_dataset: DatasetMeta | None
    report: ScalingReport
    parameters: list[ScalerParameter]
    preview: dict[str, Any]
