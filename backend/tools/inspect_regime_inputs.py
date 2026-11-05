"""Read-only inventory of local prepared BTCUSDT inputs for Module D verification."""

import json

from app.config import Settings
from app.storage.catalog import Catalog
from app.storage.projects import ProjectStore

settings = Settings()
projects = ProjectStore(settings)
for project in projects.list():
    project_settings = projects._settings_for_id(project.project_id)
    catalog = Catalog(project_settings.catalog_path)
    rows = [row for row in catalog.list_datasets() if row.symbol == "BTCUSDT"]
    print(
        json.dumps(
            {
                "project": project.name,
                "project_id": project.project_id,
                "datasets": [
                    {
                        "id": row.dataset_id,
                        "name": row.name,
                        "rows": row.row_count,
                        "operation": row.operation,
                        "parent": row.parent_dataset_id,
                        "timeframe": row.timeframe,
                    }
                    for row in rows[:25]
                ],
            },
            indent=2,
        )
    )
    catalog.close()
