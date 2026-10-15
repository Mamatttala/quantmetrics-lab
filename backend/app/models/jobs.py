"""The job record, as the polling endpoint reports it.

This lives outside ``models/regime.py`` because nothing about it is about regimes. Any
future slow module — a walk-forward backtest, a cross-validation sweep — will hand its
work to the same registry and poll the same endpoint, and a caller should not have to
import a regime module to type the response of ``GET /api/jobs/{job_id}``.
"""

from __future__ import annotations

from pydantic import BaseModel

from app.models.enums import JobStatus


class JobRecord(BaseModel):
    """One job, in whatever state it has reached.

    ``result`` and ``error`` are mutually exclusive by construction: ``finish_job`` writes
    one or the other, and a job that has neither is still running. That is why ``status``
    is authoritative and the caller should branch on it rather than on which field is
    populated.
    """

    job_id: str
    kind: str
    status: JobStatus
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    dataset_id: str | None = None
    output_dataset_id: str | None = None
    message: str | None = None
    error: dict[str, object] | None = None
    result: dict[str, object] | None = None
