"""Module C7 over HTTP.

The same two-endpoint split as Module D, and for the same reason. ``GET /api/scaling/options``
answers "what can this dataset be scaled with" from the catalog and the plan list and is
cheap; ``POST /api/scaling/fit_transform`` reads a whole Parquet file, writes a whole one
back, and may take minutes, so it returns a job handle.

The result is fetched from ``GET /api/jobs/{job_id}``, which lives in the jobs router — the
registry is generic and a second polling endpoint would be a second implementation of one
lookup.

**The one thing this module will not do.** There is no request shape that fits a scaler on
the whole series. ``minmax``, ``standard`` and ``robust`` require ``split_plan_id`` at the
model layer, so a request without one is rejected by pydantic before the handler runs and
the service could never see it. That is deliberate: a validation rule in the service is one
refactor away from being bypassed by a new caller, while a required field is not.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, status

from app.models.scaling import ScalingRequest
from app.services.scaling import ScalingService

router = APIRouter(prefix="/scaling", tags=["scaling"])


@router.get("/options")
async def scaling_options(
    dataset_id: str = Query(description="The dataset the options are evaluated against."),
) -> dict[str, Any]:
    """Numeric features, available scalers, and the plans this dataset has.

    One call rather than three, because the page cannot render any of them independently:
    which scalers are available depends on whether a plan exists, and the feature list comes
    from the frame. Availability is decided here and echoed — the page greys a control with
    the reason it is given, so a control presented as available and a refusal from the run
    endpoint cannot disagree.
    """
    return await ScalingService().options(dataset_id)


@router.post("/fit_transform", status_code=status.HTTP_202_ACCEPTED)
async def fit_transform(request: ScalingRequest) -> dict[str, Any]:
    """Accept a scaling run and return the job that will carry it out.

    **202, not 201.** The work has not happened. A caller that renders this response as a
    result will show an empty frame with no parameters on it; the ``poll`` field is the
    only way to learn whether it succeeded or why it did not.

    Refusals the handler makes itself are 404s and 422s raised before any job exists — a
    dataset that does not exist, or a request that names a scaler which must be fitted
    without a plan. Everything that can only be discovered after reading the files — a plan
    belonging to another dataset, a fold outside it, features that are empty within the
    training span — arrives through the job's ``error`` field, because by then there is no
    request left to attach a status to.
    """
    accepted = await ScalingService().fit_transform(request)
    return accepted.model_dump(mode="json")
