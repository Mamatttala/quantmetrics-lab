"""Module D over HTTP.

Two endpoints, and the split between them is the point of the module.

``GET /api/regime/options`` answers "what can this dataset support" and is cheap — it
reads the catalog, not the Parquet file. ``POST /api/regime/detect`` accepts work that may
take minutes and returns a job handle rather than a result, because the fit cannot finish
inside a request and pretending otherwise would hold a connection open until something
timed out.

The result is fetched from ``GET /api/jobs/{job_id}``, which lives in the jobs router
rather than here: the registry is generic, and a second polling endpoint under
``/api/regime`` would be a second implementation of the same lookup.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, status

from app.models.regime import RegimeDetectRequest
from app.services.regime import RegimeService

router = APIRouter(prefix="/regime", tags=["regime"])


@router.get("/options")
async def regime_options(
    dataset_id: str = Query(description="The dataset the options are evaluated against."),
) -> dict[str, Any]:
    """Which models and observations this dataset can support, and why not for the rest.

    Availability is computed server-side in one function, exactly as the feature catalogue
    does it. The frontend renders the verdict it is given — ``available`` plus ``reason`` —
    and never re-derives it, so a control greyed out in the UI and a refusal from the run
    endpoint cannot disagree.
    """
    return await RegimeService().options(dataset_id)


@router.post("/detect", status_code=status.HTTP_202_ACCEPTED)
async def detect_regimes(request: RegimeDetectRequest) -> dict[str, Any]:
    """Accept a detection run and return the job that will carry it out.

    **202, not 200.** The work has not happened. A caller that renders this response as a
    result will show an empty dashboard; the ``poll`` field exists so there is an obvious
    next step rather than a guess.

    A dataset that does not exist is refused here, with a 404, before any job is created —
    see ``RegimeService.detect``.
    """
    accepted = await RegimeService().detect(request)
    return accepted.model_dump(mode="json")
