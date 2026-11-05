"""Project selection is request-local, never a server-wide active-project flag."""

from collections.abc import AsyncIterator

from fastapi import APIRouter, Request

from app.config import get_settings, project_settings
from app.core.blocking import run_blocking
from app.models.project import CreateProject, Project
from app.storage.projects import DEFAULT_PROJECT_ID, ProjectStore

router = APIRouter(prefix="/projects", tags=["projects"])


async def project_scope(request: Request) -> AsyncIterator[None]:
    # Query selection supports browser downloads, which cannot add custom headers.
    project_id = request.headers.get("X-Project-Id")
    if project_id is None:
        project_id = request.query_params.get("project_id", DEFAULT_PROJECT_ID)
    settings = get_settings()
    if project_id != DEFAULT_PROJECT_ID:
        settings = await run_blocking(ProjectStore(settings).resolve, project_id)
    token = project_settings.set(settings)
    try:
        yield
    finally:
        project_settings.reset(token)


@router.get("", response_model=list[Project])
async def list_projects() -> list[Project]:
    return await run_blocking(ProjectStore(get_settings()).list)


@router.post("", response_model=Project, status_code=201)
async def create_project(body: CreateProject) -> Project:
    return await run_blocking(ProjectStore(get_settings()).create, body.name)
