"""The feature catalogue over HTTP.

One endpoint. Availability is a property of a dataset — a daily dataset cannot support a
4-hour window, an equity dataset cannot supply a funding rate — so the catalogue is always
served *for* a dataset rather than in the abstract, and the frontend renders the verdict it
is given instead of re-deriving it.

The engine's compute endpoint lives on the dataset path
(``POST /api/datasets/{id}/features``) because it produces a dataset; the catalogue lives
here because it produces a description. That split is why there is no
``GET /api/datasets/{id}/features``: it would return exactly this payload under a second
name, and two paths for one resource is how they drift apart.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from app.services.datasets import DatasetService

router = APIRouter(prefix="/features", tags=["features"])


@router.get("/catalog")
async def feature_catalog(
    dataset_id: str = Query(description="The dataset availability is evaluated against."),
) -> dict[str, Any]:
    """Every feature, grouped, with the formula the code actually implements.

    The payload carries each entry's ``formula`` and ``detail`` verbatim. The UI shows
    those strings in the info icon and never restates them, so the description of a
    calculation cannot drift from the calculation.
    """
    return await DatasetService().feature_catalog(dataset_id)
