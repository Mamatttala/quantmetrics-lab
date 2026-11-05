"""C8 scope selection, background jobs and reusable catalog artifacts."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from app.config import Settings, get_project_settings
from app.core.blocking import run_blocking
from app.core.errors import ValidationFailure
from app.models.dataset import DatasetMeta
from app.models.enums import DatasetCategory, JobStatus, SplitRole
from app.models.exploratory import CorrelationRequest, PCARequest, ScopeRequest
from app.models.splits import SplitPlanRow
from app.services.datasets import DatasetService, persist_dataset, resolved_path
from app.services.exploratory_stats import (
    LIMITS,
    STATELESS,
    correlation_analysis,
    preprocess,
    stateless_transform,
    static_pca,
    transform_pca,
)
from app.services.jobs import JobService, failure_payload, spawn
from app.services.quality import TIME_COLUMN_CANDIDATES
from app.services.splits import (
    SplitService,
    _observation_times,
    _require_strictly_increasing,
    plans_for,
)
from app.storage import parquet_store
from app.storage.catalog import Catalog

CAUSAL_NOTE = (
    "The causal flag describes only C8 fitting; it does not certify the upstream or "
    "downstream pipeline."
)
NONCAUSAL_NOTE = (
    "Full-sample/custom-range fitting is allowed for exploration. These fitted parameters "
    "are not causally safe for downstream training."
)


def observation_column(meta: DatasetMeta) -> str | None:
    names = {column.name.lower(): column.name for column in meta.columns}
    declared = meta.provenance.get("observation_column") or meta.provenance.get("time_column")
    if declared and declared.lower() != "close_time" and declared.lower() in names:
        return names[declared.lower()]
    return next((names[c.lower()] for c in TIME_COLUMN_CANDIDATES if c.lower() in names), None)


def numeric_columns(meta: DatasetMeta, settings: Settings) -> list[str]:
    """Read the Parquet footer only; options never scan the dataset's rows."""
    schema = pq.read_schema(resolved_path(meta, settings))
    excluded = {c.lower() for c in TIME_COLUMN_CANDIDATES} | {"close_time"}
    time_column = observation_column(meta)
    if time_column:
        excluded.add(time_column.lower())
    return [
        field.name
        for field in schema
        if field.name.lower() not in excluded
        and (
            pa.types.is_integer(field.type)
            or pa.types.is_floating(field.type)
            or pa.types.is_decimal(field.type)
        )
    ]


class ExploratoryService:
    def __init__(self, settings: Settings | None = None, catalog: Catalog | None = None) -> None:
        self.settings = settings or get_project_settings()
        self.catalog = catalog or Catalog(self.settings.catalog_path)
        self.datasets = DatasetService(self.settings, self.catalog)
        self.jobs = JobService(self.settings, self.catalog)
        self.splits = SplitService(self.settings, self.catalog)

    async def options(self, dataset_id: str | None = None) -> dict[str, Any]:
        meta = await self.datasets.get(dataset_id) if dataset_id else None

        def work() -> dict[str, Any]:
            features = numeric_columns(meta, self.settings) if meta else []
            groups: dict[str, list[str]] = {}
            for feature in features:
                group = feature_group(feature, meta)
                groups.setdefault(group, []).append(feature)
            return {
                "datasets": [d.model_dump(mode="json") for d in self.catalog.list_datasets()],
                "dataset_id": dataset_id,
                "features": features,
                "numeric_columns": features,
                "feature_groups": groups,
                "dataset_name": meta.name if meta else None,
                "row_count": meta.row_count if meta else None,
                "column_count": len(meta.columns) if meta else None,
                "has_close_time": any(c.name.lower() == "close_time" for c in meta.columns)
                if meta
                else False,
                "target_candidates": features,
                "group_by_candidates": [
                    c.name
                    for c in meta.columns
                    if c.name != observation_column(meta) and c.name.lower() != "close_time"
                ]
                if meta
                else [],
                "ignored_columns": [c.name for c in meta.columns if c.name not in features]
                if meta
                else [],
                "computational_limits": LIMITS,
                "supported_dataset_categories": [category.value for category in DatasetCategory],
                "dataset_naming": "canonical_auto",
                "default_parameters": {
                    "scope": "full_dataset",
                    "strict_causal_mode": False,
                    "min_obs": 30,
                    "group_min_obs": 30,
                    "missing_policy": "pairwise_complete",
                    "standardizer": "standard",
                },
                "columns": [c.name for c in meta.columns] if meta else [],
                "time_column": observation_column(meta) if meta else None,
                "plans": [p.model_dump(mode="json") for p in plans_for(self.catalog, dataset_id)]
                if dataset_id
                else [],
                "methods": [
                    "pearson",
                    "spearman",
                    "kendall",
                    "partial_pearson",
                    "partial_spearman",
                ],
                "warnings": [CAUSAL_NOTE],
            }

        return await run_blocking(work)

    async def _validate(self, request: CorrelationRequest | PCARequest) -> DatasetMeta:
        # Do not resolve split selections to another frame: C8 explores the selected table.
        parent = await self.datasets.get(request.dataset_id)

        def work() -> None:
            available = numeric_columns(parent, self.settings)
            chosen = list(request.features)
            if isinstance(request, CorrelationRequest):
                chosen.extend(request.controls)
                if request.target:
                    chosen.append(request.target)
                if request.group_column and request.group_column not in {
                    c.name for c in parent.columns
                }:
                    raise ValidationFailure("The grouping column is not in this dataset.")
                if request.group_bins and request.group_column not in available:
                    raise ValidationFailure("Numeric group bins require a numeric grouping column.")
            else:
                unknown = set(request.carry_columns) - {c.name for c in parent.columns}
                if unknown:
                    raise ValidationFailure(
                        "Unknown carried columns: " + ", ".join(sorted(unknown))
                    )
                if request.target:
                    chosen.append(request.target)
                if (
                    request.write_dataset
                    and request.strict_causal
                    and request.scope != "train_span"
                ):
                    raise ValidationFailure(
                        "Strict causal mode requires train-span fitting to save PCA."
                    )
            invalid = set(chosen) - set(available)
            if invalid:
                raise ValidationFailure(
                    "Choose numeric, non-time features: " + ", ".join(sorted(invalid))
                )
            needs_time = (
                request.scope != "full_dataset"
                or (isinstance(request, PCARequest) and request.write_dataset)
                or request.pre_transform in STATELESS
                or (
                    isinstance(request, CorrelationRequest)
                    and (request.enable_ccf or request.enable_rolling)
                )
            )
            if needs_time and observation_column(parent) is None:
                raise ValidationFailure(
                    "This operation needs an observation/time column (normally "
                    "open_time). close_time is never used as the timeline."
                )
            if isinstance(request, PCARequest) and request.write_dataset:
                name = (request.name or "").strip()
                if name and any(meta.name == name for meta in self.catalog.list_datasets()):
                    raise ValidationFailure(
                        "A dataset with this name already exists; choose a new unique name."
                    )

        await run_blocking(work)
        return parent

    async def submit(self, request: CorrelationRequest | PCARequest) -> dict[str, Any]:
        parent = await self._validate(request)
        kind = "exploratory_pca" if isinstance(request, PCARequest) else "exploratory_correlation"
        job_id = await self.jobs.create(
            kind, dataset_id=parent.dataset_id, params=request.model_dump(mode="json")
        )
        spawn(self._execute(job_id, request, parent))
        return {"job_id": job_id, "status": JobStatus.RUNNING.value, "poll": f"/api/jobs/{job_id}"}

    async def run(self, request: CorrelationRequest | PCARequest) -> dict[str, Any]:
        """In-process entry for tests; shares all HTTP validation and worker code."""
        return await self._compute(request, await self._validate(request))

    async def _execute(
        self, job_id: str, request: CorrelationRequest | PCARequest, parent: DatasetMeta
    ) -> None:
        await self.jobs.start(job_id)
        try:
            result = await self._compute(request, parent)
            if isinstance(request, PCARequest) and request.write_dataset:
                result["job_id"] = job_id
            dataset = result.get("dataset")
            await self.jobs.succeed(
                job_id, result=result, output_dataset_id=dataset["dataset_id"] if dataset else None
            )
        except Exception as exc:
            await self.jobs.fail(job_id, failure_payload(exc), message=str(exc))

    async def _compute(
        self, request: CorrelationRequest | PCARequest, parent: DatasetMeta
    ) -> dict[str, Any]:
        plan_meta: DatasetMeta | None = None
        plan: list[SplitPlanRow] = []
        if request.scope == "train_span":
            if request.split_plan_id is None:
                raise ValidationFailure("Train-span scope requires a split plan.")
            plan_meta, plan = await self.splits.load_plan(request.split_plan_id)
            if plan_meta.parent_dataset_id != parent.dataset_id or not plan:
                raise ValidationFailure(
                    "Choose a nonempty split plan made from the selected dataset."
                )
            if any(row.dataset_checksum != parent.checksum for row in plan):
                raise ValidationFailure(
                    "The split plan source checksum no longer matches this dataset."
                )
            folds = sorted({row.fold for row in plan})
            walk_forward = plan_meta.provenance.get("split_type", "").startswith("tscv")
            if walk_forward and request.fold not in folds:
                raise ValidationFailure(
                    "Select one available walk-forward fold; folds cannot be pooled."
                )
            if not walk_forward and request.fold is not None:
                raise ValidationFailure("Single-span plans do not take a fold.")

        def work() -> dict[str, Any]:
            frame = parquet_store.read_dataset(resolved_path(parent, self.settings))
            time_column = observation_column(parent)
            fit = scope_frame(frame, parent, request, plan, time_column)
            temporal = request.pre_transform in STATELESS or (
                isinstance(request, CorrelationRequest)
                and (request.enable_ccf or request.enable_rolling)
            )
            if temporal:
                if not time_column:
                    raise ValidationFailure(
                        "Temporal diagnostics require an observation/time column."
                    )
                _require_strictly_increasing(_observation_times(frame[time_column], parent), parent)
            chosen = list(request.features)
            if isinstance(request, CorrelationRequest):
                chosen = list(
                    dict.fromkeys(
                        [*chosen, *request.controls, *([request.target] if request.target else [])]
                    )
                )
            if isinstance(request, CorrelationRequest) and request.scope == "rolling":
                transformed = frame.copy()
                available = chosen
                preprocessing = {
                    "pre_transform": request.pre_transform,
                    "missing_policy": request.missing_policy,
                    "rolling": "Parameters are fitted independently inside each preceding window.",
                }
                raw = frame[chosen].replace([np.inf, -np.inf], np.nan)
                missing = {
                    "policy": request.missing_policy,
                    "n_fit_rows": 0,
                    "n_missing_rows": int(raw.isna().any(axis=1).sum()),
                    "n_missing_after_policy": None,
                    "n_dropped_rows": 0,
                    "per_feature_missing": {
                        name: int(raw[name].isna().sum()) for name in chosen
                    },
                    "imputation_parameters": {},
                }
                excluded: list[str] = []
                warnings: list[str] = []
            else:
                transformed, available, preprocessing, missing, excluded, warnings = preprocess(
                    frame,
                    fit.index,
                    chosen,
                    pre_transform=request.pre_transform,
                    missing_policy=request.missing_policy,
                    constant_policy=request.constant_policy,
                )
            features = [name for name in request.features if name in available]
            if (
                isinstance(request, CorrelationRequest)
                and request.target in available
                and request.target not in features
            ):
                features.append(request.target)
            if len(features) < 2 and request.scope != "rolling":
                raise ValidationFailure(
                    "Fewer than two usable numeric features remain after constant exclusion."
                )
            controls = (
                [name for name in request.controls if name in available]
                if isinstance(request, CorrelationRequest)
                else []
            )
            fit_processed = transformed.loc[fit.index]
            causal = request.scope in {"train_span", "rolling"}
            common = {
                "dataset_id": parent.dataset_id,
                "dataset_name": parent.name,
                "fit_scope": request.scope,
                "scope": request.scope,
                "causal_for_downstream_training": causal,
                "split_plan_id": request.split_plan_id,
                "fold": request.fold,
                "time_column": time_column,
                "n_scope_rows": len(fit),
                "fit_rows": len(fit),
                "transform_rows": len(frame),
                "n_transform_rows": len(frame),
                "excluded_constants": excluded,
                "missing_summary": missing,
                "preprocessing_parameters": preprocessing,
                "scope_start": str(fit[time_column].iloc[0]) if time_column else None,
                "scope_end": str(fit[time_column].iloc[-1]) if time_column else None,
            }
            if isinstance(request, CorrelationRequest):
                options = request.model_dump()
                for key in (
                    "features",
                    "method",
                    "controls",
                    "redundancy_threshold",
                    "target",
                    "max_lag",
                    "rolling_window",
                    "group_column",
                ):
                    options.pop(key, None)
                rolling_source = stateless_transform(frame, chosen, request.pre_transform).loc[
                    fit.index
                ]
                # Stateless transforms must not be recomputed after the scope boundary.
                options["rolling_frame"] = rolling_source
                options["rolling_stateless_done"] = True
                options["rolling_features"] = list(
                    dict.fromkeys(
                        [*request.features, *([request.target] if request.target else [])]
                    )
                )
                rolling_causal = request.scope == "rolling"
                if request.scope == "rolling":
                    features = list(
                        dict.fromkeys(
                            [*request.features, *([request.target] if request.target else [])]
                        )
                    )
                    controls = request.controls
                    fit_processed = frame.loc[fit.index]
                    options["rolling_only"] = True
                result = correlation_analysis(
                    fit_processed,
                    features,
                    method=request.method,
                    controls=controls,
                    redundancy_threshold=request.redundancy_threshold,
                    target=request.target,
                    rolling_window=request.rolling_window if request.enable_rolling else None,
                    group_column=request.group_column,
                    time_column=time_column,
                    **options,
                )
                result.update(common)
                result["rolling_fit_scope"] = "rolling" if request.enable_rolling else None
                result["rolling_causal_for_downstream_training"] = (
                    request.enable_rolling and rolling_causal
                )
                if request.scope == "rolling":
                    result["causal_for_downstream_training"] = True
                    result["static_diagnostics_causal_for_downstream_training"] = None
                if request.pre_transform == "rank_transform":
                    result["warnings"].append("Rank transform is diagnostic-only in this MVP.")
                result["warnings"].extend(
                    warnings + [CAUSAL_NOTE] + ([] if causal else [NONCAUSAL_NOTE])
                )
                if request.pre_transform == "none" and any(
                    feature_group(name, parent) == "price" for name in features
                ):
                    result["warnings"].append(
                        "Price levels may correlate through trend; "
                        "consider differences or returns. "
                        "Non-stationarity can make relationships unstable."
                    )
                result["metadata"]["causal_flag_meaning"] = CAUSAL_NOTE
                return result
            result = static_pca(
                fit_processed,
                features,
                standardizer=request.standardizer,
                selection=request.component_selection,
                n_components=request.n_components,
                variance_threshold=request.variance_threshold,
                eigenvalue_threshold=request.eigenvalue_threshold,
                parallel_permutations=request.parallel_permutations,
                random_state=request.random_state,
                whiten=request.whiten,
                min_obs=request.min_obs,
                component_prefix=request.component_prefix,
            )
            result.update(common)
            result["fit_rows"] = result["n_fit_rows"]
            result["warnings"].extend(
                warnings + [CAUSAL_NOTE] + ([] if causal else [NONCAUSAL_NOTE])
            )
            result["metadata"]["causal_flag_meaning"] = CAUSAL_NOTE
            result["warnings"].append(
                "PCA always mean-centers after the selected standardizer. "
                "Saved scores transform all source rows with these fixed parameters."
            )
            if time_column:
                result["biplot"]["times"] = [
                    str(frame.loc[i, time_column]) for i in result["biplot"]["row_indices"]
                ]
            result["dataset"], result["params_dataset"] = None, None
            scores = transform_pca(transformed, result)
            result["n_nan_scores"] = int(scores.isna().any(axis=1).sum())
            result["missing_summary"]["n_nan_scores"] = result["n_nan_scores"]
            result["scores"] = (
                scores.head(500)
                .astype(object)
                .where(scores.head(500).notna(), None)
                .to_dict(orient="records")
            )
            if request.write_dataset:
                destination = request.destination or ("processed_data" if causal else "outputs")
                if not causal and destination == "processed_data":
                    result["warnings"].append(
                        "Non-causal PCA saved to processed_data; "
                        "C8 causal_for_downstream_training remains false."
                    )
                carried = (
                    [name for name in frame.columns if name not in request.features]
                    if request.include_carried_columns
                    else []
                )
                columns = list(
                    dict.fromkeys(
                        [
                            *([time_column] if time_column else []),
                            *(request.features if request.keep_original_features else []),
                            *request.carry_columns,
                            *carried,
                            *(
                                [request.target]
                                if request.include_target_feature and request.target
                                else []
                            ),
                        ]
                    )
                )
                conflicts = set(columns) & set(scores.columns)
                if conflicts:
                    raise ValidationFailure(
                        "Carried/original columns conflict with PCA names: "
                        + ", ".join(sorted(conflicts))
                    )
                transformed = pd.concat([frame[columns], scores], axis=1)
                parameters = {
                    key: value
                    for key, value in result.items()
                    if key not in {"biplot", "biplot_data", "scores", "dataset", "params_dataset"}
                }
                provenance = {
                    "module_source": "exploratory",
                    "fit_scope": request.scope,
                    "causal_for_downstream_training": json.dumps(causal),
                    "split_plan_id": request.split_plan_id or "",
                    "time_column": time_column or "",
                    "causal_flag_meaning": CAUSAL_NOTE,
                    "pca_parameters": json.dumps(parameters, allow_nan=False),
                }
                extra_parents = (
                    [(plan_meta.dataset_id, {"role": "split_plan", **common})] if plan_meta else []
                )
                output_id, params_id = (
                    parquet_store.new_dataset_id(),
                    parquet_store.new_dataset_id(),
                )
                provenance["pca_parameters_dataset_id"] = params_id
                stored = persist_dataset(
                    parent=parent,
                    frame=transformed,
                    data_dir=self.settings.data_dir,
                    catalog=self.catalog,
                    category=DatasetCategory(destination),
                    name=request.name,
                    operation="pca_dimensionality_reduction",
                    detail=parameters,
                    extra_provenance=provenance,
                    extra_parents=extra_parents,
                    dataset_id=output_id,
                )
                parameter_frame = pd.DataFrame(
                    {
                        "feature": features,
                        **{
                            key: parameters[key]
                            for key in ("center", "scale", "min", "max", "pca_center")
                        },
                        **{
                            f"loading_pc_{i + 1}": np.array(parameters["loadings"])[:, i]
                            for i in range(parameters["n_components"])
                        },
                    }
                )
                params = persist_dataset(
                    parent=parent,
                    frame=parameter_frame,
                    data_dir=self.settings.data_dir,
                    catalog=self.catalog,
                    category=DatasetCategory.OUTPUTS,
                    name=None,
                    operation="pca_parameters",
                    detail={**parameters, "transformed_dataset_id": output_id},
                    extra_provenance={**provenance, "transformed_dataset_id": output_id},
                    extra_parents=extra_parents,
                    dataset_id=params_id,
                )
                result["dataset"] = stored.model_dump(mode="json")
                result["params_dataset"] = params.model_dump(mode="json")
                result.update(
                    output_dataset_id=stored.dataset_id,
                    output_dataset_name=stored.name,
                    output_category=stored.category.value,
                    parameter_dataset_id=params.dataset_id,
                    parameter_dataset_name=params.name,
                    parent_dataset_id=parent.dataset_id,
                    status="succeeded",
                )
                result["diagnostics"] = {
                    key: value
                    for key, value in result.items()
                    if key not in {"dataset", "params_dataset", "scores"}
                }
            return result

        try:
            return await run_blocking(work)
        except np.linalg.LinAlgError as exc:
            raise ValidationFailure(
                "Numerical decomposition failed; inspect feature scale and collinearity."
            ) from exc


def scope_frame(
    frame: pd.DataFrame,
    parent: DatasetMeta,
    request: ScopeRequest,
    plan: list[SplitPlanRow],
    time_column: str | None,
) -> pd.DataFrame:
    if request.scope in {"full_dataset", "rolling"}:
        result = frame
    else:
        if time_column is None:
            raise ValidationFailure("This scope requires an observation/time column.")
        times = _observation_times(frame[time_column], parent)
        if request.scope == "custom_date_range":
            if request.start_time is None or request.end_time is None:
                raise ValidationFailure("Custom date range requires both boundaries.")
            start, end = pd.Timestamp(request.start_time), pd.Timestamp(request.end_time)
            start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
            end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
            result = frame.loc[(times >= start) & (times <= end)]
        else:
            _require_strictly_increasing(times, parent)
            spans = [
                row
                for row in plan
                if row.role == SplitRole.TRAIN
                and (request.fold is None or row.fold == request.fold)
            ]
            if len(spans) != 1:
                raise ValidationFailure("Select a plan/fold with exactly one training span.")
            span = spans[0]
            if not 0 <= span.start_index < span.end_index <= len(frame):
                raise ValidationFailure("The training span falls outside the source rows.")
            if times.iloc[span.start_index] != pd.Timestamp(span.start_time) or times.iloc[
                span.end_index - 1
            ] != pd.Timestamp(span.end_time):
                raise ValidationFailure(
                    "The training span does not match the observation timeline."
                )
            result = frame.iloc[span.start_index : span.end_index]
    if result.empty:
        raise ValidationFailure("The selected analysis scope contains no rows.")
    return result


def feature_group(name: str, meta: DatasetMeta | None) -> str:
    """Use recorded feature groups when present, then conservative name hints."""
    if meta:
        try:
            groups = json.loads(meta.provenance.get("feature_groups", "{}"))
            for group, members in groups.items():
                if name in members:
                    return str(group)
        except (ValueError, AttributeError, TypeError):
            pass
    lower = name.lower()
    for group, tokens in {
        "pca": ("pc_",),
        "regime": ("regime", "state", "trend"),
        "volatility_forecast": ("forecast", "garch"),
        "returns": ("return", "ret_"),
        "volatility": ("volatility", "vol_", "atr"),
        "volume": ("volume", "quote_volume"),
        "exogenous": ("funding", "interest", "exog"),
        "indicators": ("rsi", "macd", "sma", "ema"),
        "time": ("hour", "weekday", "month"),
        "price": ("open", "close", "high", "low", "price"),
    }.items():
        if any(token in lower for token in tokens):
            return group
    return "other"
