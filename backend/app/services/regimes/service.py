"""Project-scoped coordination using the application's existing background jobs."""

from __future__ import annotations

import asyncio

from app.config import get_project_settings
from app.core.blocking import run_blocking
from app.services.jobs import JobService, failure_payload, spawn
from app.services.regimes import evaluation, experiments, inference
from app.services.regimes.artifacts import Store, clean
from app.services.regimes.data import preprocessing, read
from app.services.regimes.registry import catalog as model_catalog
from app.storage.catalog import Catalog


class RegimeService:
    def __init__(self, settings=None, catalog=None):
        self.settings = settings or get_project_settings()
        self.catalog = catalog or Catalog(self.settings.catalog_path)
        self.store = Store(self.settings, self.catalog)
        self.jobs = JobService(self.settings, self.catalog)

    def options(self, dataset_id=None):
        from importlib.util import find_spec

        from app.services.scaling import numeric_features

        rows = self.catalog.list_datasets()
        plans = [row.model_dump(mode="json") for row in rows if row.operation == "split"]
        features, columns, steps = [], [], []
        if dataset_id:
            meta, frame = read(self.catalog, self.settings, dataset_id)
            features, columns = numeric_features(frame), list(frame.columns)
            lineage = {dataset_id, *self.catalog.ancestors(dataset_id)}
            plans = [row for row in plans if row["parent_dataset_id"] in lineage]
            steps = preprocessing(self.catalog, self.settings, meta)
        return {
            "models": model_catalog(),
            "datasets": [
                row.model_dump(mode="json") for row in rows if row.artifact_kind == "observations"
            ],
            "plans": plans,
            "features": features,
            "columns": columns,
            "preprocessing": steps,
            "optimization": {
                "manual": True,
                "grid": True,
                "random": True,
                "bayesian": find_spec("optuna") is not None,
                "optuna": find_spec("optuna") is not None,
            },
        }

    async def submit(self, kind, payload, identifier=None):
        job_id = await self.jobs.create(f"regime_{kind}", params=payload.model_dump(mode="json"))
        loop = asyncio.get_running_loop()

        def progress(message):
            asyncio.run_coroutine_threadsafe(
                self.jobs.progress(job_id, message=message), loop
            ).result()

        def work():
            if kind == "fit":
                return experiments.execute(
                    payload, self.settings, self.catalog, self.store, progress
                )
            if kind == "validation":
                return evaluation.validate(
                    identifier, payload, self.settings, self.catalog, self.store, progress
                )
            if kind == "robustness":
                return evaluation.robustness(
                    identifier, payload, self.settings, self.catalog, self.store, progress
                )
            return inference.infer(payload, self.settings, self.catalog, self.store)

        async def run():
            try:
                await self.jobs.start(job_id)
                result = await run_blocking(work)
                await self.jobs.succeed(
                    job_id,
                    result=clean(result),
                    output_dataset_id=result["id"],
                    message=f"{kind.capitalize()} complete",
                )
            except Exception as exc:
                await self.jobs.fail(job_id, failure_payload(exc))

        spawn(run())
        return {"job_id": job_id, "status": "running", "poll": f"/api/jobs/{job_id}"}
