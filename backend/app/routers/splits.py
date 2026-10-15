"""Module C6 over HTTP.

One write endpoint and one read endpoint, and the difference between them is the module's
whole cost profile.

``POST /api/splits/create`` is **synchronous and 201**, unlike ``/api/regime/detect``. The
architecture mandate behind the 202 there is that heavy work must not run on the event loop;
that is satisfied here by ``run_blocking``, which is this application's one sanctioned
blocking chokepoint (``tests/test_async_rule.py`` enforces it). Creating a plan projects a
single timestamp column and writes a handful of rows — the work is milliseconds, and a job
handle would put a polling loop on the page for it.

``GET /api/splits/plans`` is the read side, and it is what C7's dropdown is built from.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, status

from app.models.splits import SplitCreateRequest, SplitPlanSummary
from app.services.splits import SplitService

router = APIRouter(prefix="/splits", tags=["splits"])


@router.post("/create", status_code=status.HTTP_201_CREATED)
async def create_split(request: SplitCreateRequest) -> dict[str, Any]:
    """Compute a split plan over a dataset and store it.

    **201, not 202.** The plan exists by the time this returns: the response carries the
    stored dataset, all of its spans, and the report. A caller that renders it sees the
    finished object, because it is finished.

    Nothing in the source dataset is modified — a plan describes a frame, it does not
    slice one — so there is no new price table to go looking for. The plan lands in the
    ``outputs`` library, and its ``split_id`` is the id of that stored dataset: one
    identifier, not two.

    Refusals are 422 (``ValidationFailure``): a dataset that is not strictly ordered by its
    observation time, a split that would leave a span empty or a train span too short to
    fit on, and a request that sets ``gap_size`` or ``purge_flag`` on a date split, where
    the user names the boundary directly.
    """
    created = await SplitService().create(request)
    return created.model_dump(mode="json")


@router.get("/plans")
async def list_plans(
    dataset_id: str = Query(description="The dataset the plans were made from."),
) -> list[SplitPlanSummary]:
    """Every plan already made from this dataset, newest first.

    Read from the lineage table rather than by scanning the ``outputs`` library, so the
    list is exactly the plans derived from *this* dataset and not every plan in the store.
    """
    return await SplitService().plans(dataset_id)
