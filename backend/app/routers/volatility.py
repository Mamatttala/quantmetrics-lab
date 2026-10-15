"""Module E over HTTP.

Two endpoints, split the way Module D's are: ``GET /api/volatility/options`` answers "what
can this dataset support" from the catalog and is cheap, and ``POST
/api/volatility/forecast`` accepts work that may take minutes and returns a job handle
rather than a result.

The result is fetched from ``GET /api/jobs/{job_id}``, which lives in the jobs router
rather than here: the registry is generic, and a second polling endpoint under
``/api/volatility`` would be a second implementation of the same lookup.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, status

from app.models.volatility import VolForecastRequest
from app.services.volatility import VolatilityService

router = APIRouter(prefix="/volatility", tags=["volatility"])


@router.get("/options")
async def volatility_options(
    dataset_id: str = Query(description="The dataset the options are evaluated against."),
) -> dict[str, Any]:
    """Which sources, models and schedules this dataset can support, and why not for the rest.

    The default schedule is derived from the row count rather than fixed, so a short
    dataset opens on a configuration that can actually run instead of on the refusal it
    would have produced.
    """
    return await VolatilityService().options(dataset_id)


@router.post("/forecast", status_code=status.HTTP_202_ACCEPTED)
async def forecast_volatility(request: VolForecastRequest) -> dict[str, Any]:
    """Accept a forecasting run and return the job that will carry it out.

    **202, not 200.** A walk-forward refits at every scored step, and a hybrid refits a
    second model inside that loop; the work cannot finish inside a request, and pretending
    otherwise would hold the connection open until something timed out. The ``poll`` field
    exists so there is an obvious next step rather than a guess.

    A dataset that does not exist is refused here, with a 404, before any job is created.
    An impossible walk-forward schedule or a non-stationary return series is refused from
    inside the job, with the computed numbers, because neither is knowable until the file
    has been read.
    """
    accepted = await VolatilityService().forecast(request)
    return accepted.model_dump(mode="json")
