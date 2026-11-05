"""Lifecycle, CORS, and router wiring.

Run with:  uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.core.errors import register_exception_handlers
from app.routers import (
    datasets,
    exploratory,
    features,
    fetch,
    health,
    jobs,
    projects,
    regimes,
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

    catalog = Catalog(settings.catalog_path)
    try:
        catalog.initialise()
    finally:
        catalog.close()
    from app.storage.projects import ProjectStore

    ProjectStore(settings).initialise()
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
    app.include_router(projects.router, prefix=API_PREFIX)
    for router in (
        health.router, datasets.router, features.router, fetch.router,
        regimes.router,
        splits.router, scaling.router, exploratory.router,
        volatility.router, jobs.router,
    ):
        app.include_router(
            router, prefix=API_PREFIX, dependencies=[Depends(projects.project_scope)]
        )

    return app


app = create_app()
