"""Project registry and isolated on-disk catalogs.

The default project uses the original data directory so existing datasets, lineage,
and jobs stay intact. New projects own projects/<opaque id>/ beneath that directory.
Display names never become filesystem paths.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from uuid import uuid4

from app.config import Settings
from app.core.errors import ProjectNotFound, ValidationFailure
from app.models.project import CreateProject, Project
from app.storage.catalog import Catalog

DEFAULT_PROJECT_ID = "default"


class ProjectStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.path = settings.data_dir / "projects.db"

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def initialise(self) -> None:
        self.settings.ensure_dirs()
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS projects ("
                "project_id TEXT PRIMARY KEY, name TEXT NOT NULL, "
                "name_key TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT OR IGNORE INTO projects VALUES (?, ?, ?, ?)",
                (DEFAULT_PROJECT_ID, "Default project", "default project",
                 datetime.now(UTC).isoformat()),
            )

    def list(self) -> list[Project]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT project_id, name, created_at FROM projects "
                "ORDER BY created_at, project_id"
            ).fetchall()
        return [Project(**dict(row)) for row in rows]

    def create(self, name: str) -> Project:
        name = CreateProject(name=name).name
        project = Project(project_id=uuid4().hex, name=name, created_at=datetime.now(UTC))
        try:
            with closing(self._connect()) as connection, connection:
                connection.execute(
                    "INSERT INTO projects VALUES (?, ?, ?, ?)",
                    (project.project_id, name, name.casefold(), project.created_at.isoformat()),
                )
                # Publish the registry row only after the new catalog is ready.
                self._initialise_catalog(self._settings_for_id(project.project_id))
        except sqlite3.IntegrityError as exc:
            raise ValidationFailure("A project with this name already exists.") from exc
        return project

    def _settings_for_id(self, project_id: str) -> Settings:
        if project_id == DEFAULT_PROJECT_ID:
            return self.settings
        root = self.settings.data_dir.resolve()
        directory = (root / "projects" / project_id).resolve()
        if not directory.is_relative_to(root / "projects"):
            raise ProjectNotFound("This project is not available.")
        return self.settings.model_copy(update={"data_dir": directory})

    @staticmethod
    def _initialise_catalog(settings: Settings) -> None:
        settings.ensure_dirs()
        catalog = Catalog(settings.catalog_path)
        try:
            catalog.initialise()
            from app.services.splits import migrate_legacy_partitions

            migrate_legacy_partitions(settings, catalog)
        finally:
            catalog.close()

    def resolve(self, project_id: str) -> Settings:
        with closing(self._connect()) as connection:
            found = connection.execute(
                "SELECT project_id FROM projects WHERE project_id = ?", (project_id,)
            ).fetchone()
        if found is None:
            raise ProjectNotFound(
                "This project is not available.", hint="Select a project from Data Explorer."
            )
        settings = self._settings_for_id(project_id)
        # Applies additive catalog migrations when reopening after an upgrade.
        self._initialise_catalog(settings)
        return settings
