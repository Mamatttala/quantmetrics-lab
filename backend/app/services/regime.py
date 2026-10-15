"""Module D over HTTP: validate, run in the background, persist as an output.

The orchestration here is thin by design. The statistics live in ``regime_engine`` and
``regime_models``; the durability lives in the catalog. What this module decides is the
three things neither of those can:

1. **What is refused, and when.** A dataset that does not exist is a 404 on the `POST`,
   not a job that fails ten seconds later — the caller should not have to poll to learn
   they mistyped an id. Everything else about the data — unsorted timestamps, a missing
   close column, a hybrid model with no indicators — is only discoverable after reading
   the file, so it surfaces through the job's ``error`` field.
2. **What is recorded.** A regime frame is uninterpretable without the thresholds and
   window lengths that produced it, so the whole request goes into lineage as
   ``parameters_used`` alongside the fit diagnostics.
3. **Where the result lands.** ``outputs``, not ``processed_data`` — see
   ``persist_dataset``.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

import numpy as np

from app.config import Settings, get_settings
from app.core.blocking import run_blocking
from app.core.errors import ValidationFailure
from app.models.dataset import DatasetMeta
from app.models.enums import DatasetCategory, JobStatus, RegimeFeature
from app.models.jobs import JobRecord
from app.models.regime import RegimeDetectRequest, RegimeRunAccepted
from app.services import quality as quality_checks
from app.services import regime_engine
from app.services.datasets import DatasetService, _preview_of, persist_dataset, resolved_path
from app.services.jobs import JobService, failure_payload, spawn
from app.services.regime_engine import RegimeSpec
from app.storage import parquet_store
from app.storage.catalog import Catalog

#: Written into lineage so a saved regime dataset can be traced back to the module that
#: produced it without reading the operation string.
MODULE_SOURCE = "regime"

JOB_KIND = "regime_detect"

#: The source column each observation is built from. Kept beside the enum rather than
#: inside it so the vocabulary stays a vocabulary — `enums.py` should not need to know
#: that `log_return` comes from `close`.
_FEATURE_SOURCES: dict[RegimeFeature, tuple[str, ...]] = {
    RegimeFeature.LOG_RETURN: ("close",),
    RegimeFeature.REALIZED_VOLATILITY: ("close",),
    RegimeFeature.VOLUME_CHANGE: ("volume",),
    RegimeFeature.ADX: ("adx",),
    RegimeFeature.RSI: ("rsi",),
}

#: Observations this module computes versus columns it merely reads. The distinction is
#: what lets the UI grey out `adx` with "produce it in C5" while greying out
#: `volume_change` with "this dataset has no volume" — different problems, different fixes.
_CARRIED: frozenset[RegimeFeature] = frozenset({RegimeFeature.ADX, RegimeFeature.RSI})

_FEATURE_LABELS: dict[RegimeFeature, str] = {
    RegimeFeature.LOG_RETURN: "Log return",
    RegimeFeature.REALIZED_VOLATILITY: "Realized volatility",
    RegimeFeature.VOLUME_CHANGE: "Volume change",
    RegimeFeature.ADX: "ADX (carried from C5)",
    RegimeFeature.RSI: "RSI (carried from C5)",
}

_MODEL_NEEDS: dict[str, tuple[str, ...]] = {
    "hmm": (),
    "gmm": (),
    "hybrid": ("adx", "rsi"),
    "rules": (),
}


def _annualisation_report(meta: DatasetMeta) -> dict[str, Any]:
    """Whether a volatility can be annualised for this dataset, and by what factor.

    Reported rather than raised: the config panel wants to *show* that the factor is
    unknown and why, and a 422 here would blank the whole form over a field the user may
    not even need if they never look at the annualised number.
    """
    timeframe_label = meta.timeframe.value if meta.timeframe else None
    if meta.timeframe is None:
        return {
            "available": False,
            "periods_per_year": None,
            "timeframe": None,
            "reason": "This dataset declares no timeframe, so a per-bar volatility cannot "
            "be annualised, and the crisis threshold — which is stated in annualised "
            "terms — has no meaning. Re-fetch it with a timeframe, or resample it in B4.",
        }
    try:
        factor = regime_engine.annualisation_factor(meta.timeframe, meta.asset_class)
    except ValidationFailure as exc:  # pragma: no cover - guarded by the branch above
        return {
            "available": False,
            "periods_per_year": None,
            "timeframe": timeframe_label,
            "reason": exc.message,
        }
    family = "365 days" if meta.asset_class.value.startswith("crypto") else "252 days"
    return {
        "available": True,
        "periods_per_year": factor,
        "timeframe": timeframe_label,
        "reason": f"{timeframe_label} bars over {family} per year.",
    }


def _model_options(available_features: list[str]) -> list[dict[str, Any]]:
    """Which model types can run, and what each is missing if it cannot.

    ``rules`` has no feature requirement at all, which is the point of it: it is the one
    path that still produces a crisis label on a dataset with nothing but a close column.
    """
    options: list[dict[str, Any]] = []
    for model in ("hmm", "gmm", "hybrid", "rules"):
        needed = _MODEL_NEEDS[model]
        missing = [name for name in needed if name not in available_features]
        options.append(
            {
                "id": model,
                "available": not missing,
                "requires": list(needed),
                "reason": None
                if not missing
                else f"Needs {' and '.join(missing)} — produce it in C5 Feature "
                "Engineering and merge it in (B4) first.",
            }
        )
    return options


def _option_warnings(
    meta: DatasetMeta, annualisation: dict[str, Any], available_features: list[str]
) -> list[dict[str, str]]:
    """Everything the user should know before pressing Run.

    Each entry carries a ``level``: ``error`` blocks the run, ``warning`` does not. The
    distinction is applied by the frontend, so a short dataset is a caution the user can
    overrule while a missing timeframe is a refusal they cannot.
    """
    warnings: list[dict[str, str]] = []
    if not annualisation["available"]:
        warnings.append(
            {"level": "error", "code": "no_timeframe", "message": str(annualisation["reason"])}
        )
    if "close" not in {column.name for column in meta.columns}:
        warnings.append(
            {
                "level": "error",
                "code": "no_close",
                "message": "This dataset has no 'close' column. Module D reads OHLCV bars.",
            }
        )
    # The first label needs `min_train_bars` usable rows before it, so a series shorter
    # than that produces a frame of nothing but nulls — a result that looks like a
    # success and contains no labels at all.
    default_train = 250
    if meta.row_count <= default_train:
        warnings.append(
            {
                "level": "warning",
                "code": "short_dataset",
                "message": f"Only {meta.row_count} rows. The first label needs "
                f"{default_train} bars of training data before it, so this dataset will "
                "produce no labelled rows at the default settings. Lower the minimum "
                "training window, or fetch a longer series.",
            }
        )
    if "volume" not in {column.name for column in meta.columns}:
        warnings.append(
            {
                "level": "warning",
                "code": "no_volume",
                "message": "No 'volume' column, so volume change cannot be selected.",
            }
        )
    if not any(name in available_features for name in ("adx", "rsi")):
        warnings.append(
            {
                "level": "warning",
                "code": "no_indicators",
                "message": "No ADX or RSI column, so the hybrid model is unavailable.",
            }
        )
    return warnings



def _spec_from(request: RegimeDetectRequest) -> RegimeSpec:
    """The request's scalars, resolved. Kept as one function so the engine never sees a
    pydantic model and the request's own validation is the only validation."""
    return RegimeSpec(
        model_type=request.model_type,
        n_states=request.n_states,
        features=tuple(request.features),
        min_state_duration=request.min_state_duration,
        switching_cost_penalty=request.switching_cost_penalty,
        crisis_threshold=request.crisis_threshold,
        override_probability=request.override_probability,
        transition_confidence=request.transition_confidence,
        commit_confidence=request.commit_confidence,
        min_train_bars=request.min_train_bars,
        refit_every=request.refit_every,
        volatility_window=request.volatility_window,
    )


def jsonable(value: Any) -> Any:
    """A JSON-safe copy of a report.

    Three conversions, each of which has bitten this application or would have:

    * **NumPy scalars become Python ones.** ``json.dumps`` cannot serialise ``np.int64``,
      and the catalog's fallback would stringify it — turning a BIC of ``-4821.7`` into
      ``"-4821.7"`` and a bar count into ``"250"``. The frontend would then format a
      string with a thousands separator and produce nonsense.
    * **Non-finite floats become null.** JSON has no NaN or Infinity, and JavaScript's
      ``JSON.parse`` rejects both outright. A single ``inf`` anywhere in the diagnostics
      would break the whole response, not just that field.
    * **Arrays become lists**, which also applies the two rules above element-wise.
    """
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class RegimeService:
    """Runs Module D. Public methods are coroutines."""

    def __init__(self, settings: Settings | None = None, catalog: Catalog | None = None) -> None:
        self.settings = settings or get_settings()
        self.catalog = catalog or Catalog(self.settings.catalog_path)
        self.datasets = DatasetService(self.settings, self.catalog)
        self.jobs = JobService(self.settings, self.catalog)

    # ------------------------------------------------------------------- entry

    async def detect(self, request: RegimeDetectRequest) -> RegimeRunAccepted:
        """Accept the work and return a handle to it.

        The dataset is resolved here, before the job is created, so an unknown id is an
        immediate 404 with the id in the message. Doing it inside the task would return
        202 for a dataset that does not exist and leave the user watching a spinner for a
        failure that was knowable instantly.
        """
        parent = await self.datasets.get(request.dataset_id)
        job_id = await self.jobs.create(
            JOB_KIND,
            dataset_id=parent.dataset_id,
            params=request.model_dump(mode="json"),
        )
        spawn(self._execute(job_id, request, parent))
        return RegimeRunAccepted(
            job_id=job_id, status=JobStatus.RUNNING, poll=f"/api/jobs/{job_id}"
        )

    async def run(self, request: RegimeDetectRequest) -> dict[str, Any]:
        """Compute a detection inline and return the payload, without scheduling a job.

        Used by the verification drivers, which want the result rather than a handle, and
        by the background task, which is running already.
        """
        parent = await self.datasets.get(request.dataset_id)
        return await self._compute(request, parent)

    async def job(self, job_id: str) -> JobRecord:
        return await self.jobs.get(job_id)

    async def options(self, dataset_id: str) -> dict[str, Any]:
        """What this dataset can actually support, decided here and not in the browser.

        The same rule the feature catalogue follows: one function answers "is this
        available, and if not why", so a disabled control and a refusal from the run
        endpoint can never disagree about the reason. Deriving it twice — once in
        TypeScript for the greyed-out state, once in Python for the error — is how a UI
        ends up offering a checkbox the backend then rejects.
        """
        meta = await self.datasets.get(dataset_id)

        def work() -> dict[str, Any]:
            present = {column.name for column in meta.columns}
            empty = set(meta.unpopulated_fields)

            features: list[dict[str, Any]] = []
            for feature in RegimeFeature:
                needed = _FEATURE_SOURCES[feature]
                missing = [name for name in needed if name not in present]
                blanked = [name for name in needed if name in empty]
                if missing:
                    reason: str | None = (
                        f"This dataset has no '{missing[0]}' column."
                        if feature in _CARRIED
                        else f"Needs a '{missing[0]}' column."
                    )
                elif blanked:
                    reason = (
                        f"Column '{blanked[0]}' is present but entirely empty. Merge a "
                        "real one in (B4) first."
                    )
                else:
                    reason = None
                features.append(
                    {
                        "id": feature.value,
                        "label": _FEATURE_LABELS[feature],
                        "kind": "carried" if feature in _CARRIED else "computed",
                        "needs": list(needed),
                        "available": reason is None,
                        "reason": reason,
                    }
                )

            annualisation = _annualisation_report(meta)
            available_features = [f["id"] for f in features if f["available"]]
            return {
                "dataset_id": meta.dataset_id,
                "name": meta.name,
                "rows": meta.row_count,
                "timeframe": meta.timeframe.value if meta.timeframe else None,
                "asset_class": meta.asset_class.value,
                "columns": [column.name for column in meta.columns],
                "features": features,
                "available_features": available_features,
                "annualisation": annualisation,
                "models": _model_options(available_features),
                "defaults": {
                    "n_states": 4,
                    "min_state_duration": 5,
                    "switching_cost_penalty": 0.005,
                    "crisis_threshold": 0.35,
                    "min_train_bars": 250,
                    "refit_every": 250,
                    "volatility_window": 24,
                },
                "warnings": _option_warnings(meta, annualisation, available_features),
            }

        return await run_blocking(work)

    async def wait(
        self, job_id: str, *, timeout: float = 900.0, interval: float = 0.05
    ) -> JobRecord:
        """Block until a job reaches a terminal state. For tests and scripted drivers.

        Bounded by ``timeout`` rather than looping forever: a job whose task was lost would
        otherwise hang the caller with no diagnostic, which is a worse failure than a
        timeout that names itself.
        """
        waited = 0.0
        while waited < timeout:
            record = await self.jobs.get(job_id)
            if record.status is not JobStatus.RUNNING:
                return record
            await asyncio.sleep(interval)
            waited += interval
        raise TimeoutError(f"Job {job_id} did not finish within {timeout:.0f}s.")

    # ------------------------------------------------------------------ worker

    async def _execute(
        self, job_id: str, request: RegimeDetectRequest, parent: DatasetMeta
    ) -> None:
        """The background task: run, then record the terminal state.

        Failures are written to the job rather than raised. There is no request left to
        carry a 422, and an exception escaping into the event loop would be logged to a
        console the user is not reading while the job sat at RUNNING forever.

        ``Exception`` and not ``BaseException``: a ``CancelledError`` means the server is
        shutting down, and swallowing it would leave a task that refuses to die.
        """
        await self.jobs.start(job_id)
        try:
            payload = await self._compute(request, parent)
        except Exception as exc:  # noqa: BLE001 - the job record is the error channel
            await self.jobs.fail(job_id, failure_payload(exc), message=str(exc))
            return
        dataset = payload.get("dataset") or {}
        await self.jobs.succeed(
            job_id,
            result=payload,
            message=_summary_message(payload),
            output_dataset_id=dataset.get("dataset_id"),
        )

    async def _compute(
        self, request: RegimeDetectRequest, parent: DatasetMeta
    ) -> dict[str, Any]:
        path = resolved_path(parent, self.settings)
        spec = _spec_from(request)

        def work() -> dict[str, Any]:
            frame = parquet_store.read_dataset(path)
            time_column = quality_checks.find_time_column(frame)
            if time_column is None:
                raise ValidationFailure(
                    "This dataset has no time column to stamp regimes onto.",
                    hint="Regime rows are labelled by bar, so the output has to carry the "
                    "same timeline as the input.",
                )
            outcome = regime_engine.detect(
                frame,
                time_column=time_column,
                spec=spec,
                timeframe=parent.timeframe,
                asset_class=parent.asset_class,
            )
            report = jsonable(outcome.report)

            meta: DatasetMeta | None = None
            if request.write:
                meta = persist_dataset(
                    parent=parent,
                    frame=outcome.frame,
                    data_dir=self.settings.data_dir,
                    catalog=self.catalog,
                    # `outputs`, because this is the product of an analysis rather than
                    # an input to one.
                    category=DatasetCategory.OUTPUTS,
                    name=request.name or f"{parent.name} (regime · {request.model_type.value})",
                    operation=MODULE_SOURCE,
                    detail=_lineage_detail(request, report),
                    extra_provenance={
                        "module_source": MODULE_SOURCE,
                        "regime_model": request.model_type.value,
                        "crisis_threshold": str(request.crisis_threshold),
                        "min_state_duration": str(request.min_state_duration),
                        "net_value_add": str(
                            report["net_value_add"]["net_value_add"]
                        ),
                    },
                )
            return {
                "dataset": meta.model_dump(mode="json") if meta is not None else None,
                "report": report,
                "preview": _preview_of(outcome.frame),
            }

        return await run_blocking(work)


def _lineage_detail(request: RegimeDetectRequest, report: dict[str, Any]) -> dict[str, Any]:
    """What goes into the lineage row for one regime run.

    ``parameters_used`` is the whole request, not a summary of it. A threshold left out is
    a run that cannot be reproduced, and the difference between a research artifact and an
    unexplained column of labels is exactly whether the knobs were written down.
    """
    durations = report.get("state_durations") or {}
    labelled = sum(entry["runs"] for entry in durations.values())
    average_duration = (
        report["labelled_rows"] / labelled if labelled else None
    )
    return {
        "module_source": MODULE_SOURCE,
        "parameters_used": request.model_dump(mode="json"),
        "features": report["features"],
        "resolved_windows": {
            "volatility_window": report["volatility_window"],
            "min_train_bars": request.min_train_bars,
            "refit_every": request.refit_every,
            "refit_count": report["refit_count"],
        },
        "model_diagnostics": {
            # Named for what it is. With an expanding window there is one fit per refit
            # point and therefore one BIC per refit point; reporting "the" BIC without
            # saying which fit it came from would invite comparing across runs that had
            # different numbers of refits.
            "bic_latest_refit": report.get("bic_latest_refit"),
            "aic_latest_refit": report.get("aic_latest_refit"),
            "switch_count": report["switch_count"],
            "average_state_duration_bars": average_duration,
            "net_value_add": report["net_value_add"]["net_value_add"],
            "net_value_add_definition": report["net_value_add"]["definition"],
            "in_sample": report["net_value_add"]["in_sample"],
            "crisis_override_bars": report["crisis_override_bars"],
        },
        "warmup_bars": report["warmup_bars"],
        "state_counts": report["state_counts"],
        "non_stationary_features": [
            entry["feature"]
            for entry in report.get("stationarity", [])
            if entry.get("stationary") is False
        ],
    }


def _summary_message(payload: dict[str, Any]) -> str:
    """One line for the job list. Says what happened, not that something did."""
    report = payload.get("report") or {}
    dataset = payload.get("dataset")
    nva = (report.get("net_value_add") or {}).get("net_value_add")
    where = f"saved as {dataset['name']}" if dataset else "preview only, nothing written"
    if nva is None:
        return where
    return f"{where}; net value-add {nva:+.4f} (in-sample)"
