"""Read-only options are immediate; numerical analyses return pollable jobs."""

from typing import Any

from fastapi import APIRouter, status

from app.models.exploratory import CorrelationRequest, PCARequest
from app.services.exploratory import ExploratoryService

router = APIRouter(prefix="/exploratory", tags=["exploratory"])


@router.get("/options")
async def options(dataset_id: str | None = None) -> dict[str, Any]:
    return await ExploratoryService().options(dataset_id)


@router.post("/correlation", status_code=status.HTTP_202_ACCEPTED)
async def correlation(request: CorrelationRequest) -> dict[str, Any]:
    return await ExploratoryService().submit(request)


@router.post("/pca", status_code=status.HTTP_202_ACCEPTED)
async def pca(request: PCARequest) -> dict[str, Any]:
    return await ExploratoryService().submit(request)
