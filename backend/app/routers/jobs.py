"""The job registry over HTTP.

One resource, read-only. There is no POST here — jobs are created by whichever endpoint
owns the work, and a generic "create a job" route would be an invitation to put work
somewhere nothing records what it was for.

There is deliberately no DELETE. A job row is the only surviving record of why an output
dataset looks the way it does; making it removable would put a destructive button next to
the one piece of provenance that cannot be reconstructed from the Parquet file.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from app.services.jobs import JobService

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("")
async def list_jobs(
    limit: int = Query(default=25, ge=1, le=200),
) -> list[dict[str, Any]]:
    """Recent jobs, newest first.

    ``limit`` is bounded rather than unbounded because the list is rendered whole; an
    unbounded query would let a long-lived installation produce a response the browser
    cannot draw.
    """
    records = await JobService().recent(limit)
    return [record.model_dump(mode="json") for record in records]


@router.get("/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    """One job, in whatever state it has reached.

    A job that does not exist is a 404, which is distinguishable from a job that is still
    running — the latter returns 200 with ``status: "running"``. Collapsing those two into
    one response is how a mistyped id becomes an infinite spinner.

    When ``status`` is ``"failed"`` the refusal is in ``error``, as the same
    ``{"code", "message", "details"}`` envelope every other endpoint uses. The work ran
    after the request ended, so there was no 422 left to attach it to.
    """
    return (await JobService().get(job_id)).model_dump(mode="json")
