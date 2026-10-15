"""Module C7 — scaling and normalisation.

The whole module exists to make one mistake impossible, and the mistake is worth naming
because nothing about the wrong answer looks wrong:

    A scaler fitted on the entire series has already seen the test span. Every training row
    is then expressed in units that were chosen using the rows the model will be scored on —
    the mean it subtracts is partly the future's mean, the range it divides by is partly the
    future's range. The transformed frame is the same shape, the same dtypes, the same row
    count. It looks exactly like a correctly scaled frame, and the backtest that follows is
    quietly too good.

So there is no code path here that fits on a whole frame. ``_fit`` receives the training
rows and nothing else; the transform is applied to the whole frame afterwards, which is the
correct direction — transforming test rows with training statistics is the entire point.

Two consequences fall out of that, and both are refusals rather than defaults:

* A fitted scaler **requires** a split plan (``ScalerType.needs_split_plan``). The three
  fitted members cannot be run without one, at the model layer and again in the service.
* A walk-forward plan requires a **fold**, because it holds one train/test pair per fold.
  Fitting on the union of every fold's training rows would compute fold 1's statistics from
  rows lying after fold 1's test span — look-ahead relative to fold 1, and hidden inside what
  looks like the conservative choice.

The one scaler that needs neither is the rolling z-score, and it is not lucky: it estimates
its mean and standard deviation from the trailing window at each row, so it cannot contain
information from a row that has not happened yet. Causality there is a property of the
arithmetic, and the verification suite asserts it by computing the series twice — once whole,
once truncated to 80% — and requiring the shared prefix to be identical.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from app.config import Settings, get_settings
from app.core.blocking import run_blocking
from app.core.errors import ValidationFailure
from app.models.dataset import DatasetMeta
from app.models.enums import DatasetCategory, JobStatus, ScalerType, SplitRole
from app.models.jobs import JobRecord
from app.models.scaling import (
    ScalerParameter,
    ScalingAccepted,
    ScalingCreated,
    ScalingReport,
    ScalingRequest,
)
from app.models.splits import SplitPlanRow
from app.services.datasets import DatasetService, _preview_of, persist_dataset, resolved_path
from app.services.jobs import JobService, failure_payload, spawn
from app.services.splits import SplitService, plans_for
from app.storage import parquet_store
from app.storage.catalog import Catalog

MODULE_SOURCE = "scaling"

JOB_KIND = "scaling_fit_transform"

#: The lineage verb for the parameter table. Distinct from the scaled frame's operation so
#: that a reader of the lineage graph can tell "this frame was scaled" from "these are the
#: numbers it was scaled with" without opening either file.
PARAMS_OPERATION = "scaler_params"

#: Columns that are never candidates for scaling, whatever their dtype. A timestamp
#: subtracted from a mean is not a smaller timestamp, it is a number with no meaning, and
#: scaling one would destroy the axis every downstream module joins on.
_NEVER_SCALED: frozenset[str] = frozenset(
    {"open_time", "close_time", "create_time", "calc_time", "date", "Date", "timestamp"}
)

#: Roll a fitted scaler's attributes into the one affine pair that inverts it.
#: ``(center_attribute, scale_attribute)`` per scaler type, read from the sklearn object's
#: own fitted attribute names — the same names the specification asks to be stored.
_AFFINE: dict[ScalerType, tuple[str, str]] = {
    ScalerType.STANDARD: ("mean_", "scale_"),
    ScalerType.MINMAX: ("data_min_", "data_range_"),
    ScalerType.ROBUST: ("center_", "scale_"),
}

#: Which raw fitted attributes are stored beside the affine pair, per scaler type. The
#: specification names mean_, scale_, min_ and max_; they are all kept, because the affine
#: pair is this application's interpretation of them and a reader should be able to check
#: the interpretation against the library's own numbers.
_RAW_ATTRIBUTES: dict[ScalerType, tuple[str, ...]] = {
    ScalerType.STANDARD: ("mean_", "scale_", "var_", "n_samples_seen_"),
    ScalerType.MINMAX: ("data_min_", "data_max_", "data_range_", "min_", "scale_"),
    ScalerType.ROBUST: ("center_", "scale_"),
}


# ------------------------------------------------------------------------- inventory


def numeric_features(frame: pd.DataFrame) -> list[str]:
    """The columns that can be rescaled, in frame order.

    Numeric dtypes only, and the four time columns are excluded even when they are numeric
    (an epoch column reads as int64). Both rules live here so the page's checkboxes, the
    ``features: []`` default and the refusal for an explicitly named bad column all consult
    the same list.
    """
    features: list[str] = []
    for name in frame.columns:
        if name in _NEVER_SCALED:
            continue
        if pd.api.types.is_numeric_dtype(frame[name]) and not pd.api.types.is_bool_dtype(
            frame[name]
        ):
            features.append(str(name))
    return features


def resolve_features(
    frame: pd.DataFrame, requested: list[str], *, available: list[str] | None = None
) -> list[str]:
    """Turn a request's feature list into the columns to scale, or refuse.

    An empty list means "all of them" — the page's default — and is resolved here rather
    than in the browser so the request and the report agree on what "all" contained at the
    moment it ran. A named column that is absent, or present but not numeric, is a refusal
    rather than a silent skip: the caller asked for it, and quietly not scaling a feature the
    model expects to be scaled is a defect that surfaces as a mysteriously dominant feature
    much later.
    """
    allowed = available if available is not None else numeric_features(frame)
    if not requested:
        if not allowed:
            raise ValidationFailure(
                "This dataset has no numeric column to scale.",
                hint="Feature columns are numeric by definition; run C5 Feature Engineering "
                "first if this is still a raw price frame.",
            )
        return list(allowed)

    present = set(frame.columns)
    resolved: list[str] = []
    for name in requested:
        if name not in present:
            raise ValidationFailure(
                f"'{name}' is not a column of this dataset.",
                hint="Available numeric columns: " + ", ".join(allowed) + ".",
            )
        if name not in allowed:
            raise ValidationFailure(
                f"'{name}' is not a numeric column, so it cannot be scaled.",
                hint="Numeric columns: " + ", ".join(allowed) + ".",
            )
        if name not in resolved:
            resolved.append(name)
    return resolved


def _rows_of_role(
    plan: list[SplitPlanRow], role: SplitRole, fold: int | None
) -> list[SplitPlanRow]:
    """The plan rows for one role, narrowed to one fold when the plan has folds."""
    rows = [
        row
        for row in plan
        if row.role is role and (fold is None or row.fold == fold)
    ]
    return sorted(rows, key=lambda row: (row.fold, row.start_index))


# -------------------------------------------------------------------------- the fits


def _fit(train: pd.DataFrame, features: list[str], scaler_type: ScalerType) -> Any:
    """Fit one sklearn scaler over the training rows and nothing else.

    ``train`` is the only frame this function ever sees. It is passed in already sliced so
    that the slicing decision lives at the call site, where the plan is in scope, and not
    here, where a default of "the frame you were given" would be one refactor away from
    being the whole series.

    `MinMaxScaler`, `StandardScaler`, `RobustScaler` — named explicitly rather than reached
    through a registry, because the three differ in what they assume about a distribution
    and the choice between them is a modelling decision the user made, not a lookup.
    """
    values = train[features].to_numpy(dtype="float64")
    if scaler_type is ScalerType.MINMAX:
        scaler = MinMaxScaler()
    elif scaler_type is ScalerType.STANDARD:
        scaler = StandardScaler()
    elif scaler_type is ScalerType.ROBUST:
        scaler = RobustScaler()
    else:  # pragma: no cover - guarded by the caller, which never routes rolling here
        raise ValidationFailure(f"{scaler_type.value} is not a fitted scaler.")
    # `fit` skips NaN, so a feature's leading warm-up nulls do not corrupt its statistics —
    # but a column that is *entirely* null within the training span leaves sklearn with
    # nothing to fit, which is refused below rather than producing NaN parameters.
    scaler.fit(values)
    return scaler


def _attribute(raw: Any, index: int) -> float | None:
    """One stored sklearn attribute, as a finite float or null.

    Two shapes have to be handled, and getting this wrong is an ``IndexError`` rather than a
    wrong number — which is at least loud. ``mean_``, ``scale_``, ``var_`` and ``data_min_``
    are per-feature arrays; ``n_samples_seen_`` is a *scalar* in current sklearn, one count
    for the whole fit. Indexing a scalar'd array at position 1 of a two-feature fit raises.

    Non-finite becomes null rather than being written out: JSON has no NaN or Infinity and
    JavaScript's ``JSON.parse`` rejects both outright, so a single ``inf`` would break the
    whole parameters response rather than one cell.
    """
    if raw is None:
        return None
    values = np.atleast_1d(raw).astype("float64")
    value = float(values[index]) if index < len(values) else float(values[0])
    return value if math.isfinite(value) else None


def _parameters(
    scaler: Any,
    features: list[str],
    scaler_type: ScalerType,
    per_feature_rows: dict[str, int],
    constant: set[str] | None = None,
) -> list[ScalerParameter]:
    """Every fitted scaler, reduced to the same affine pair.

    One shape for all three, so that inverting a scaled column is one line whichever scaler
    produced it, and so Modules E and F never have to branch on the type to undo it. The
    raw attributes ride along in ``attributes`` so the reduction can be checked.

    ``constant`` names the features that held one value throughout the training span. It is
    passed in from the caller, which has the training frame, rather than inferred from the
    fitted attributes — because sklearn *guards* against a zero scale by substituting 1.0
    (``_handle_zeros_in_scale``), so the fitted scale of a constant column and a genuinely
    unit-scaled one are the same number. Inferring it would mislabel the second as the
    first. This is the honest definition anyway: constant is a property of the data.
    """
    center_name, scale_name = _AFFINE[scaler_type]
    centers = np.atleast_1d(getattr(scaler, center_name)).astype("float64")
    scales = np.atleast_1d(getattr(scaler, scale_name)).astype("float64")
    flat = constant or set()

    parameters: list[ScalerParameter] = []
    for index, feature in enumerate(features):
        center = float(centers[index])
        scale = float(scales[index])
        # A constant column has nothing to divide by, so it is only shifted. sklearn has
        # already substituted a scale of 1.0 — dividing by it is what makes the transformed
        # column a column of zeros, which is correct and is reported as such.
        is_constant = feature in flat or not math.isfinite(scale) or scale == 0.0
        if is_constant:
            scale = 1.0
        attributes: dict[str, float | None] = {}
        for name in _RAW_ATTRIBUTES[scaler_type]:
            attributes[name] = _attribute(getattr(scaler, name, None), index)
        parameters.append(
            ScalerParameter(
                feature=feature,
                scaler_type=scaler_type,
                center_=center,
                scale_=scale,
                constant=is_constant,
                fitted_rows=per_feature_rows.get(feature, 0),
                attributes=attributes,
                formula=(
                    f"{feature}_scaled = ({feature} - {center!r}) / {scale!r}"
                    if not is_constant
                    else f"{feature}_scaled = {feature} - {center!r} (constant column: "
                    "nothing to divide by, so it is only shifted)"
                ),
            )
        )
    return parameters


def _apply(frame: pd.DataFrame, parameters: list[ScalerParameter]) -> pd.DataFrame:
    """Transform the whole frame with parameters fitted on the training span.

    The whole frame, deliberately: transforming the test rows is not a leak, it is the
    point. What would leak is *estimating* on them, and that has already happened — or
    rather, has already been prevented — by the time this runs.

    Every other column is carried through untouched, and no row is dropped. A scaling step
    that changed the row count would silently invalidate the split plan it was performed
    against, because the plan addresses rows by index.
    """
    scaled = frame.copy()
    for parameter in parameters:
        column = pd.to_numeric(scaled[parameter.feature], errors="coerce").astype("float64")
        scaled[parameter.feature] = (column - parameter.center_) / parameter.scale_
    return scaled


def rolling_zscore(
    frame: pd.DataFrame, features: list[str], window: int
) -> tuple[pd.DataFrame, list[ScalerParameter], dict[str, int]]:
    """Per-row standardisation over a trailing window. Causal by construction.

    ``(x - mean_of_last_w) / std_of_last_w``, computed at every row from the *preceding*
    ``window`` bars, so row *t* is expressed only in terms of rows at or before *t*.

    ``min_periods=window`` — a row with fewer than a full window behind it is left null
    rather than standardised against a partial window. A two-bar "window" produces a
    z-score of ±1 by construction, which is not a small inaccuracy: it is a number that
    means something different from every other number in the column, and null is the honest
    way to say so. The cost is that the first ``window - 1`` rows of each scaled column are
    null, which the report states.

    ``center=False`` is stated rather than left to the default. It *is* the default, and
    that is exactly why it is written: a centred rolling window reads half a window into the
    future, so a reader must be able to see at a glance that this does not. The verification
    suite asserts the token never appears in this module.

    ``ddof=0`` — the *population* standard deviation, not pandas' default of ``ddof=1`` —
    so that "standard deviation" means one thing in this module. sklearn's fitted scalers
    divide by a ddof=0 scale, and a rolling z-score that divided by a ddof=1 one would differ
    from a standard-scaled column by a factor of ``sqrt(w/(w-1))``: 0.8% at the default
    window, and larger as the window shrinks. Two definitions of the same word in one module
    is how a "why do these two columns not match" question becomes unanswerable.
    """
    scaled = frame.copy()
    per_feature: dict[str, int] = {}
    parameters: list[ScalerParameter] = []

    for feature in features:
        column = pd.to_numeric(scaled[feature], errors="coerce").astype("float64")
        rolling = column.rolling(window=window, min_periods=window, center=False)
        mean = rolling.mean()
        std = rolling.std(ddof=0)
        # A flat stretch inside a trending series has zero dispersion within its window.
        # Dividing by it gives ±inf or NaN on a row where the value is perfectly defined,
        # so the z-score is written as 0 — the row is exactly at the window's mean, which
        # is what a zero deviation means. Documented because it is a choice, not an
        # artefact: the alternative convention is to leave those rows null.
        safe_std = std.where(std > 0, 1.0)
        scaled[feature] = ((column - mean) / safe_std).where(mean.notna(), np.nan)

        usable = int(mean.notna().sum())
        per_feature[feature] = usable
        # The *last* window's statistics, not one number for the run: there is no single
        # `center_` here, because it differs at every row. Reporting the most recent window
        # is what a reader wants to see, and the formula says plainly that it is per-row.
        # A trailing NaN (which happens when the last bar itself is null) is skipped rather
        # than written out, so the card never shows a NaN where a number is expected.
        last_mean = mean.dropna()
        last_std = std.dropna()
        parameters.append(
            ScalerParameter(
                feature=feature,
                scaler_type=ScalerType.ROLLING_ZSCORE,
                center_=float(last_mean.iloc[-1]) if len(last_mean) else 0.0,
                scale_=float(last_std.iloc[-1]) if len(last_std) else 1.0,
                constant=False,
                fitted_rows=usable,
                attributes={
                    "window": float(window),
                    "usable_rows": float(usable),
                    "last_window_mean": float(last_mean.iloc[-1]) if len(last_mean) else None,
                    "last_window_std": float(last_std.iloc[-1]) if len(last_std) else None,
                },
                formula=(
                    f"{feature}_scaled[t] = ({feature}[t] - mean({feature}[t-{window}:t])) "
                    f"/ std({feature}[t-{window}:t]), the trailing {window} bars only"
                ),
            )
        )
    return scaled, parameters, per_feature


# --------------------------------------------------------------------------- service


class ScalingService:
    """Runs Module C7. Public methods are coroutines."""

    def __init__(self, settings: Settings | None = None, catalog: Catalog | None = None) -> None:
        self.settings = settings or get_settings()
        self.catalog = catalog or Catalog(self.settings.catalog_path)
        self.datasets = DatasetService(self.settings, self.catalog)
        self.splits = SplitService(self.settings, self.catalog)
        self.jobs = JobService(self.settings, self.catalog)

    # --------------------------------------------------------------------- entry

    async def fit_transform(self, request: ScalingRequest) -> ScalingAccepted:
        """Accept the work and return a handle to it.

        The dataset is resolved here so an unknown id is an immediate 404 rather than a job
        that fails a minute later. The plan is *not* resolved here: reading it means opening
        a Parquet file, which is the work being deferred. A bad plan id therefore surfaces
        through the job's ``error`` field, and the driver waits for it.
        """
        parent = await self.datasets.get(request.dataset_id)
        job_id = await self.jobs.create(
            JOB_KIND,
            dataset_id=parent.dataset_id,
            params=request.model_dump(mode="json"),
        )
        spawn(self._execute(job_id, request, parent))
        return ScalingAccepted(
            job_id=job_id, status=JobStatus.RUNNING, poll=f"/api/jobs/{job_id}"
        )

    async def run(self, request: ScalingRequest) -> ScalingCreated:
        """Compute inline and return the payload, without scheduling a job.

        For the drivers and the tests, which want the result rather than a handle.
        """
        parent = await self.datasets.get(request.dataset_id)
        return await self._compute(request, parent)

    async def job(self, job_id: str) -> JobRecord:
        return await self.jobs.get(job_id)

    async def wait(
        self, job_id: str, *, timeout: float = 900.0, interval: float = 0.05
    ) -> JobRecord:
        """Block until a job reaches a terminal state. For tests and scripted drivers."""
        waited = 0.0
        while waited < timeout:
            record = await self.jobs.get(job_id)
            if record.status is not JobStatus.RUNNING:
                return record
            await asyncio.sleep(interval)
            waited += interval
        raise TimeoutError(f"Job {job_id} did not finish within {timeout:.0f}s.")

    # ------------------------------------------------------------------- options

    async def options(self, dataset_id: str) -> dict[str, Any]:
        """Everything the C7 page needs, in one call.

        The feature list is read from the frame rather than from the catalog: the catalog
        records dtypes as Arrow strings, and deciding "is this numeric" from an Arrow type
        string would be a second implementation of a rule pandas already answers — and the
        two would eventually disagree about, say, a decimal or a nullable integer.
        """
        meta = await self.datasets.get(dataset_id)
        path = resolved_path(meta, self.settings)

        def work() -> dict[str, Any]:
            frame = parquet_store.read_dataset(path, columns=None)
            available = numeric_features(frame)
            plans = plans_for(self.catalog, dataset_id)

            scalers: list[dict[str, Any]] = []
            for scaler in ScalerType:
                reason: str | None = None
                if scaler.needs_split_plan and not plans:
                    reason = (
                        "No split plan exists for this dataset, and "
                        f"{scaler.value} is fitted on a stretch of history. Make one in C6 "
                        "Chronological Splits first — fitting on the whole series would "
                        "carry the test span's statistics into every training row."
                    )
                elif not available:
                    reason = "This dataset has no numeric column to scale."
                scalers.append(
                    {
                        "id": scaler.value,
                        "needs_split_plan": scaler.needs_split_plan,
                        "available": reason is None,
                        "reason": reason,
                    }
                )
            return {
                "dataset_id": meta.dataset_id,
                "name": meta.name,
                "rows": meta.row_count,
                "features": available,
                "defaults": {
                    "features": available,
                    "window": 60,
                    "scaler_type": ScalerType.STANDARD.value,
                },
                "scalers": scalers,
                "plans": [plan.model_dump(mode="json") for plan in plans],
                "warnings": _option_warnings(meta, available, plans),
            }

        return await run_blocking(work)

    # -------------------------------------------------------------------- worker

    async def _execute(
        self, job_id: str, request: ScalingRequest, parent: DatasetMeta
    ) -> None:
        """The background task: run, then record the terminal state.

        Failures are written to the job rather than raised — there is no request left to
        attach a 422 to, and the poller has to learn the outcome down the same channel as
        the result. ``Exception`` and not ``BaseException``: a ``CancelledError`` means the
        server is shutting down and must not be swallowed.
        """
        await self.jobs.start(job_id)
        try:
            payload = await self._compute(request, parent)
        except Exception as exc:  # noqa: BLE001 - the job record is the error channel
            await self.jobs.fail(job_id, failure_payload(exc), message=str(exc))
            return
        dataset = payload.dataset
        await self.jobs.succeed(
            job_id,
            result=payload.model_dump(mode="json"),
            message=_summary_message(payload),
            output_dataset_id=dataset.dataset_id if dataset else None,
        )

    async def _compute(
        self, request: ScalingRequest, parent: DatasetMeta
    ) -> ScalingCreated:
        path = resolved_path(parent, self.settings)

        # The plan is read before the frame, so a request naming a plan that does not exist
        # fails without reading a five-million-row Parquet file first.
        plan_meta: DatasetMeta | None = None
        plan: list[SplitPlanRow] = []
        if request.split_plan_id is not None:
            plan_meta, plan = await self._load_plan(
                request.split_plan_id, parent, request.fold
            )

        def work() -> ScalingCreated:
            frame = parquet_store.read_dataset(path)
            available = numeric_features(frame)
            features = resolve_features(frame, request.features, available=available)

            parameters: list[ScalerParameter]
            per_feature: dict[str, int]
            fit_rows = 0
            fit_role: str | None = None
            fit_start = None
            fit_end = None

            if request.scaler_type is ScalerType.ROLLING_ZSCORE:
                window = request.effective_window
                scaled, parameters, per_feature = rolling_zscore(frame, features, window)
                fit_rows = len(frame)
                fit_role = "trailing window (no fit, no plan)"
            else:
                window = None
                # Refused rather than asserted. `ScalingRequest` requires the plan for every
                # fitted scaler, so this is unreachable through HTTP — but a bare `assert`
                # disappears under `-O`, and this function is reachable directly from the
                # tests, where silent whole-series fitting is exactly the bug the module
                # exists to prevent.
                if plan_meta is None:
                    raise ValidationFailure(
                        f"{request.scaler_type.value} is fitted on a stretch of history, so "
                        "it needs a split plan to say which rows it may be fitted on."
                    )
                train = _rows_of_role(plan, SplitRole.TRAIN, request.fold)
                if not train:
                    raise ValidationFailure(
                        "This split plan has no training span to fit on.",
                        hint="Recreate the plan in C6, or choose another fold.",
                    )
                fit_rows = sum(row.row_count for row in train)
                fit_role = SplitRole.TRAIN.value
                fit_start = train[0].start_time
                fit_end = train[-1].end_time
                # The slice, and the whole of the golden rule: `train_rows` is built from
                # the plan's training spans and `frame` is never handed to the fitter.
                train_rows = pd.concat(
                    [frame.iloc[row.start_index : row.end_index] for row in train],
                    axis=0,
                )
                _require_fittable(train_rows, features)
                scaler = _fit(train_rows, features, request.scaler_type)
                per_feature = {
                    feature: int(train_rows[feature].notna().sum()) for feature in features
                }
                flat = {
                    feature
                    for feature in features
                    if train_rows[feature].nunique(dropna=True) <= 1
                }
                parameters = _parameters(
                    scaler, features, request.scaler_type, per_feature, flat
                )
                scaled = _apply(frame, parameters)

            report = ScalingReport(
                dataset_id=parent.dataset_id,
                dataset_name=parent.name,
                scaler_type=request.scaler_type,
                split_plan_id=plan_meta.dataset_id if plan_meta else None,
                split_plan_name=plan_meta.name if plan_meta else None,
                fold=request.fold,
                fit_role=fit_role,
                fit_rows=fit_rows,
                fit_start_time=fit_start,
                fit_end_time=fit_end,
                rows=len(frame),
                features=features,
                window=window,
                rows_used_for_fit=fit_rows,
                per_feature_rows=per_feature,
                warnings=_run_warnings(request, plan_meta, plan, frame, features, per_feature),
            )

            stored: DatasetMeta | None = None
            params_meta: DatasetMeta | None = None
            if request.write:
                stored = persist_dataset(
                    parent=parent,
                    frame=scaled,
                    data_dir=self.settings.data_dir,
                    catalog=self.catalog,
                    # `processed_data`, not `outputs`: this is still a frame of bars and it
                    # feeds the next module, exactly like a C5 feature dataset. The regime
                    # labels that go to `outputs` are a different kind of object.
                    category=DatasetCategory.PROCESSED,
                    name=request.name
                    or f"{parent.name} ({request.scaler_type.value} scaled)",
                    operation=MODULE_SOURCE,
                    detail=_lineage_detail(request, report, parameters),
                    extra_provenance={
                        "module_source": MODULE_SOURCE,
                        "scaler_type": request.scaler_type.value,
                        "split_plan_id": plan_meta.dataset_id if plan_meta else "",
                        "scaled_features": ",".join(features),
                        "scaler_window": str(window) if window is not None else "",
                    },
                    extra_parents=(
                        [(plan_meta.dataset_id, {"role": "split_plan"})]
                        if plan_meta is not None
                        else None
                    ),
                )
                # The parameters land in a dataset of their own — confirmed with the user.
                # Not a JSON file beside the frame: this way they are checksummed, linked
                # into the lineage, visible in the Explorer, and exportable as JSON through
                # the existing `/export?fmt=json` route. One source of truth, and the format
                # the specification asked for is still obtainable from it.
                params_meta = persist_dataset(
                    parent=stored,
                    frame=_params_frame(parameters),
                    data_dir=self.settings.data_dir,
                    catalog=self.catalog,
                    category=DatasetCategory.OUTPUTS,
                    name=f"{stored.name} — scaler parameters",
                    operation=PARAMS_OPERATION,
                    detail={
                        "module_source": MODULE_SOURCE,
                        "scaler_type": request.scaler_type.value,
                        "scaled_dataset_id": stored.dataset_id,
                        "parameters": [p.model_dump(mode="json") for p in parameters],
                    },
                    extra_provenance={
                        "module_source": MODULE_SOURCE,
                        "scaler_type": request.scaler_type.value,
                        "scaled_dataset_id": stored.dataset_id,
                        "split_plan_id": plan_meta.dataset_id if plan_meta else "",
                    },
                )

            return ScalingCreated(
                dataset=stored,
                params_dataset=params_meta,
                report=report,
                parameters=parameters,
                preview=_preview_of(scaled),
            )

        return await run_blocking(work)

    async def _load_plan(
        self, split_plan_id: str, parent: DatasetMeta, fold: int | None
    ) -> tuple[DatasetMeta, list[SplitPlanRow]]:
        """The plan this run will be fitted against, checked against the dataset.

        Three refusals, each of which would otherwise be silent:

        * The plan must belong to *this* dataset. A plan is a set of row indices, so one
          made from a different frame can apply cleanly to the wrong rows — and the spans
          would look plausible, because the row counts are similar.
        * The plan must have been computed on the same bytes. The checksum comparison is
          the mechanical form of the same rule, and it catches the case the lineage cannot:
          a dataset re-written in place.
        * A walk-forward plan needs a **fold**. It holds one train/test pair per fold, and
          fitting on the union of every fold's training rows would compute early folds'
          statistics from rows lying after their test spans — the conservative-looking
          choice, and look-ahead.
        """
        plan_meta, plan = await self.splits.load_plan(split_plan_id)

        if plan_meta.parent_dataset_id != parent.dataset_id:
            raise ValidationFailure(
                f"'{plan_meta.name}' is a split plan for a different dataset, so its row "
                "indices address different bars.",
                hint="Choose a plan made from this dataset.",
            )
        if not plan:
            raise ValidationFailure(f"'{plan_meta.name}' contains no spans.")
        checked = plan[0].dataset_checksum
        if checked != parent.checksum:
            raise ValidationFailure(
                f"'{plan_meta.name}' was computed on a different version of "
                f"'{parent.name}' (checksum {checked[:12]}…, current "
                f"{parent.checksum[:12]}…), so its row indices may no longer line up.",
                hint="Recreate the plan in C6 against the current dataset.",
            )

        folds = sorted({row.fold for row in plan})
        if _is_tscv(plan_meta):
            if fold is None:
                raise ValidationFailure(
                    f"'{plan_meta.name}' is a walk-forward plan with {len(folds)} folds, "
                    "so it holds one training span per fold. Fitting on all of them "
                    "together would compute early folds' statistics from rows that lie "
                    "after their test spans.",
                    hint="Set fold to choose which fold to fit against.",
                )
            if fold not in folds:
                raise ValidationFailure(
                    f"Fold {fold} is not in '{plan_meta.name}' "
                    f"(folds {folds[0]}–{folds[-1]})."
                )
        elif fold is not None:
            raise ValidationFailure(
                f"'{plan_meta.name}' is a single-span plan with no folds, so fold "
                f"{fold} has no meaning.",
                hint="Drop fold, or choose a walk-forward plan.",
            )
        return plan_meta, plan


# --------------------------------------------------------------------------- helpers


def _is_tscv(plan_meta: DatasetMeta) -> bool:
    from app.models.enums import SplitType

    raw = plan_meta.provenance.get("split_type", "")
    try:
        return SplitType(raw).is_tscv
    except ValueError:
        return False


def _require_fittable(train_rows: pd.DataFrame, features: list[str]) -> None:
    """Refuse a training span that cannot produce parameters.

    Three ways it cannot, and each produces a *plausible* output if not caught: an empty
    span fits nothing and sklearn raises a bare ValueError; an all-null column makes the
    fitted statistics NaN, which then makes the transformed column NaN and reads as missing
    data; and a span of one row has a standard deviation that is defined but meaningless, so
    every transformed value is exactly 0.
    """
    if len(train_rows) == 0:
        raise ValidationFailure("The training span of this plan is empty.")
    if len(train_rows) < 2:
        raise ValidationFailure(
            f"The training span holds {len(train_rows)} row(s), which is too few to "
            "estimate a spread.",
            hint="Choose a fold with a longer training span, or use a longer dataset.",
        )
    blanked = [name for name in features if train_rows[name].notna().sum() == 0]
    if blanked:
        raise ValidationFailure(
            "These features are entirely empty within the training span, so there is "
            f"nothing to fit: {', '.join(blanked)}.",
            hint="They may be filled only later in the series, which is exactly the "
            "information a scaler must not be given. Drop them, or clean them in B3.",
        )


def _params_frame(parameters: list[ScalerParameter]) -> pd.DataFrame:
    """The parameter table, one row per feature.

    Flat columns with the affine pair as first-class columns, because that is what Modules
    E and F read: they need ``center_`` and ``scale_`` to invert a prediction, and making
    them parse a nested document to find it would be the kind of friction that ends with a
    second implementation of the inversion somewhere else.
    """
    return pd.DataFrame(
        {
            "feature": pd.Series([p.feature for p in parameters], dtype="object"),
            "scaler_type": pd.Series(
                [p.scaler_type.value for p in parameters], dtype="object"
            ),
            "center_": pd.Series([p.center_ for p in parameters], dtype="float64"),
            "scale_": pd.Series([p.scale_ for p in parameters], dtype="float64"),
            "constant": pd.Series([p.constant for p in parameters], dtype="bool"),
            "fitted_rows": pd.Series([p.fitted_rows for p in parameters], dtype="int64"),
            "formula": pd.Series([p.formula for p in parameters], dtype="object"),
        }
    )


def _lineage_detail(
    request: ScalingRequest,
    report: ScalingReport,
    parameters: list[ScalerParameter],
) -> dict[str, Any]:
    """What the lineage row records for one scaling run.

    The whole request, not a summary. A scaled dataset is only reproducible if the scaler,
    the plan it was fitted against, *and* the fold within that plan are all written down —
    and the fold is the one that would be silently lost, because it looks like a detail of
    a detail.
    """
    return {
        "module_source": MODULE_SOURCE,
        "parameters_used": request.model_dump(mode="json"),
        "resolved": {
            "features": report.features,
            "window": report.window,
            "fit_role": report.fit_role,
            "fit_rows": report.fit_rows,
            "fit_start_time": _iso(report.fit_start_time),
            "fit_end_time": _iso(report.fit_end_time),
            "rows": report.rows,
            "per_feature_rows": report.per_feature_rows,
        },
        "scaler_parameters": [p.model_dump(mode="json") for p in parameters],
    }


def _iso(value: Any) -> str | None:
    return value.isoformat() if hasattr(value, "isoformat") else None


def _option_warnings(
    meta: DatasetMeta, available: list[str], plans: list[Any]
) -> list[dict[str, str]]:
    """What the page should say before Run is pressed.

    ``error`` blocks the run, ``warning`` does not — the same convention Module D uses, and
    applied by the frontend so the two modules cannot drift into disagreeing about what a
    level means.
    """
    warnings: list[dict[str, str]] = []
    if not available:
        warnings.append(
            {
                "level": "error",
                "code": "no_numeric",
                "message": "This dataset has no numeric column to scale.",
            }
        )
    if not plans:
        warnings.append(
            {
                "level": "warning",
                "code": "no_split_plan",
                "message": "No split plan exists for this dataset, so the three fitted "
                "scalers are unavailable. Make one in C6 first. The rolling z-score needs "
                "no plan — it standardises each row against its own trailing window.",
            }
        )
    else:
        warnings.append(
            {
                "level": "warning",
                "code": "fit_on_train_only",
                "message": "The scaler will be fitted on the training span of the chosen "
                "plan and applied to every row. The transformed test rows will not be "
                "centred at zero — that is the correct result, not a fault.",
            }
        )
    if meta.row_count < 120:
        warnings.append(
            {
                "level": "warning",
                "code": "short_dataset",
                "message": f"Only {meta.row_count} rows. The rolling z-score's default "
                "60-bar window leaves the first 59 rows of every scaled column null.",
            }
        )
    return warnings


def _run_warnings(
    request: ScalingRequest,
    plan_meta: DatasetMeta | None,
    plan: list[SplitPlanRow],
    frame: pd.DataFrame,
    features: list[str],
    per_feature: dict[str, int],
) -> list[str]:
    """What the finished run should say about itself. Plain sentences, no levels."""
    notes: list[str] = []
    if request.scaler_type is ScalerType.ROLLING_ZSCORE:
        window = request.effective_window
        notes.append(
            f"Each row was standardised against the {window} bars before it, so the first "
            f"{window - 1} rows of each scaled column are null. Nothing was fitted, so no "
            "split plan was needed and none was used."
        )
    elif plan_meta is not None:
        train = _rows_of_role(plan, SplitRole.TRAIN, request.fold)
        if train:
            where = (
                f"fold {request.fold}" if request.fold is not None else "the training span"
            )
            notes.append(
                f"Fitted on {where} of '{plan_meta.name}' only: "
                f"{sum(row.row_count for row in train)} rows from "
                f"{_iso(train[0].start_time)} to {_iso(train[-1].end_time)}. Every other "
                "row — validation, test, and every later fold — was transformed with those "
                "numbers and did not contribute to them."
            )
        if request.fold is None:
            notes.append(
                "No fold was given, so this plan has a single train span and the fit is "
                "against all of it."
            )
    if per_feature:
        most = max(per_feature.values())
        thin = sorted(name for name, count in per_feature.items() if count < most)
        if thin:
            notes.append(
                "Some features had fewer usable rows in the training span than others: "
                + ", ".join(thin)
                + f" (fewest {min(per_feature.values())} of {most}). A shorter fit is not "
                "wrong, but those parameters rest on less evidence than their neighbours'."
            )
    constants = sorted(
        name for name in features if frame[name].nunique(dropna=True) <= 1
    )
    if constants:
        notes.append(
            "These features hold a single value throughout the dataset, so scaling them "
            f"cannot change them and the model cannot use them: {', '.join(constants)}."
        )
    return notes


def _summary_message(payload: ScalingCreated) -> str:
    """One line for the job list. Says what happened, not that something did."""
    report = payload.report
    where = (
        f"saved as {payload.dataset.name}" if payload.dataset else "preview only, nothing written"
    )
    fitted = (
        f"fitted on {report.fit_rows} rows"
        if report.scaler_type.needs_split_plan
        else f"trailing {report.window}-bar window"
    )
    return (
        f"{where}; {report.scaler_type.value} on {len(report.features)} feature(s), {fitted}"
    )
