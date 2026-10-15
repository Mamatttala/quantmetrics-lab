"""The background-job registry.

Some work in this application cannot finish inside a request. A Gaussian HMM walk-forward
refits at every stride point, and each refit runs Baum-Welch to convergence; on a
multi-year hourly series that is tens of seconds to minutes of CPU. Holding the HTTP
request open for it would occupy a browser connection, defeat every proxy timeout, and
give the user a spinner with no way to tell "working" from "hung".

So the request is split in two:

* ``POST`` validates what it can immediately, records a job, hands the work to a task,
  and returns an id.
* ``GET /api/jobs/{id}`` reports where that job has got to.

**The state lives in SQLite, not in a dictionary.** An in-memory registry would be
simpler and would lose every in-flight job on reload — and, worse, would lose the
*finished* ones, which for a research tool are the artifact: the diagnostics that make a
saved regime dataset interpretable are in that record. Persisting costs one table and buys
survival across the reload that happens constantly during development.

**Failures are delivered, not raised.** By the time the work runs there is no request left
to attach a 422 to, so a refusal is written into the job's ``error`` field and reaches the
caller down the same channel as the result. A job that is not ``FAILED`` did not fail; a
caller must branch on ``status``, never on the mere presence of an error field.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any

from app.config import Settings, get_settings
from app.core.blocking import run_blocking
from app.core.errors import JobNotFound, QuantmetricsError
from app.models.enums import JobStatus
from app.models.jobs import JobRecord
from app.storage import parquet_store
from app.storage.catalog import Catalog

#: Strong references to running tasks. `asyncio.create_task` returns a task the event loop
#: only holds weakly, so a task nothing references can be garbage-collected mid-flight and
#: the job then sits at RUNNING forever with no trace of why. The set is the reference; the
#: done-callback is what keeps it from growing without bound.
_TASKS: set[asyncio.Task[None]] = set()


def spawn(coro: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
    """Run `coro` in the background, keeping it alive until it finishes."""
    task = asyncio.create_task(coro)
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return task


def running_jobs() -> int:
    """How many tasks are in flight. Used by tests to assert nothing is left dangling."""
    return sum(1 for task in _TASKS if not task.done())


def _new_job_id() -> str:
    """A ULID, from the same generator datasets use.

    Reusing it rather than writing a second id scheme keeps one alphabet in the codebase
    and one ordering guarantee: ULIDs sort by creation time, so the jobs table's primary
    key is already in the order a listing wants.
    """
    return parquet_store.new_dataset_id()


class JobService:
    """Create, advance, and read background jobs. Public methods are coroutines."""

    def __init__(self, settings: Settings | None = None, catalog: Catalog | None = None) -> None:
        self.settings = settings or get_settings()
        self.catalog = catalog or Catalog(self.settings.catalog_path)

    async def create(
        self,
        kind: str,
        *,
        dataset_id: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> str:
        job_id = _new_job_id()
        await run_blocking(
            self.catalog.create_job, job_id, kind, dataset_id=dataset_id, params=params
        )
        return job_id

    async def start(self, job_id: str) -> None:
        await run_blocking(self.catalog.start_job, job_id)

    async def succeed(
        self,
        job_id: str,
        *,
        result: dict[str, Any] | None = None,
        message: str | None = None,
        output_dataset_id: str | None = None,
    ) -> None:
        await run_blocking(
            self.catalog.finish_job,
            job_id,
            status=JobStatus.SUCCEEDED,
            result=result,
            message=message,
            output_dataset_id=output_dataset_id,
        )

    async def fail(
        self, job_id: str, error: dict[str, Any], *, message: str | None = None
    ) -> None:
        await run_blocking(
            self.catalog.finish_job,
            job_id,
            status=JobStatus.FAILED,
            error=error,
            message=message,
        )

    async def get(self, job_id: str) -> JobRecord:
        row = await run_blocking(self.catalog.get_job, job_id)
        if row is None:
            raise JobNotFound(
                f"No job with id {job_id}.",
                hint="Job ids are returned by the endpoint that started the work and are "
                "not reused.",
            )
        return JobRecord(**row)

    async def recent(self, limit: int = 50) -> list[JobRecord]:
        rows = await run_blocking(self.catalog.list_jobs, limit)
        return [JobRecord(**row) for row in rows]


def failure_payload(exc: BaseException) -> dict[str, Any]:
    """The error envelope to store on a failed job.

    An unexpected exception is reported as ``internal_error`` with its type named and its
    message carried through, rather than as a bare string. A traceback would leak local
    paths into a field the UI renders, and swallowing the message entirely would leave the
    user with "something went wrong" and nothing to search for.
    """
    if isinstance(exc, QuantmetricsError):
        payload = exc.to_payload()
        return dict(payload["error"])
    return {
        "code": "internal_error",
        "message": f"{type(exc).__name__}: {exc}",
        "details": {},
    }
