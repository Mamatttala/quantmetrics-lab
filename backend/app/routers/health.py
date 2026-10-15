"""Liveness and capability probe.

The frontend calls this on boot to fail loudly when the backend is down, rather
than rendering an empty shell that looks like "no datasets".
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import get_settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    version: str
    data_dir: str


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version="0.1.0",
        data_dir=str(settings.data_dir),
    )
