"""Lifecycle, CORS, and router wiring.

Run with:  uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.core.errors import register_exception_handlers
from app.routers import (
    datasets,
    features,
    fetch,
    health,
    jobs,
    regime,
    scaling,
    splits,
    volatility,
)

API_PREFIX = "/api"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    settings.ensure_dirs()
    # The catalog schema is created eagerly: every route assumes it exists, and a
    # missing table discovered mid-request surfaces as a confusing 500.
    from app.storage.catalog import Catalog

    Catalog(settings.catalog_path).initialise()
    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Quantmetrics Lab",
        version="0.1.0",
        summary="Local quantitative finance research workstation.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)
    app.include_router(health.router, prefix=API_PREFIX)
    app.include_router(datasets.router, prefix=API_PREFIX)
    app.include_router(features.router, prefix=API_PREFIX)
    app.include_router(fetch.router, prefix=API_PREFIX)
    app.include_router(regime.router, prefix=API_PREFIX)
    app.include_router(splits.router, prefix=API_PREFIX)
    app.include_router(scaling.router, prefix=API_PREFIX)
    app.include_router(volatility.router, prefix=API_PREFIX)
    app.include_router(jobs.router, prefix=API_PREFIX)

    return app


app = create_app()
