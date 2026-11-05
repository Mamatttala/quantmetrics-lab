"""Isolated real-BTCUSDT browser smoke test of D9 -> D10 -> D11.

Copies existing observations into .tmp; never changes the user's project datasets.
Uses existing C6/C7 services to prepare the fixture, then drives the actual UI/API.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from uuid import uuid4

from app.config import Settings
from app.models.enums import DatasetCategory
from app.models.scaling import ScalingRequest
from app.models.splits import SplitCreateRequest
from app.services.regimes.data import read
from app.services.scaling import ScalingService
from app.services.splits import SplitService
from app.storage import parquet_store
from app.storage.catalog import Catalog
from app.storage.projects import ProjectStore

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = uuid4().hex
ARTIFACTS = ROOT / "artifacts" / "regime-rebuild" / RUN_ID
REVIEW = ROOT / ".tmp" / "regime-review" / RUN_ID
WEB_DIST = Path(os.environ.get("QM_REGIME_WEB_DIST", str(ROOT / "frontend/dist")))
URL = "http://127.0.0.1:8018"


async def seed():
    settings = Settings()
    projects = ProjectStore(settings)
    selected = None
    for project in projects.list():
        project_settings = projects._settings_for_id(project.project_id)
        source_catalog = Catalog(project_settings.catalog_path)
        candidates = [
            row
            for row in source_catalog.list_datasets()
            if row.symbol == "BTCUSDT"
            and row.timeframe
            and row.timeframe.value == "1d"
            and row.operation is None
        ]
        if candidates:
            source = max(candidates, key=lambda row: row.row_count)
            selected = read(source_catalog, project_settings, source.dataset_id)
            source_catalog.close()
            break
        source_catalog.close()
    if selected is None:
        raise RuntimeError("No local BTCUSDT daily observations are available.")
    parent, frame = selected
    fixture = Settings(data_dir=REVIEW)
    fixture.ensure_dirs()
    ProjectStore(fixture).initialise()
    catalog = Catalog(fixture.catalog_path)
    catalog.initialise()
    identifier = parquet_store.new_dataset_id()
    written = parquet_store.write_dataset(frame, fixture.data_dir, DatasetCategory.RAW, identifier)
    copied = parent.model_copy(
        update={
            "dataset_id": identifier,
            "name": "BTCUSDT daily - isolated real-data verification",
            "storage_path": str(written.storage_path),
            "checksum": written.checksum,
            "parent_dataset_id": None,
        }
    )
    catalog.insert_dataset(copied)
    plan = await SplitService(fixture, catalog).create(
        SplitCreateRequest(dataset_id=identifier, split_type="percentage_70_15_15")
    )
    scaled = await ScalingService(fixture, catalog).run(
        ScalingRequest(
            dataset_id=identifier,
            split_plan_id=plan.dataset.dataset_id,
            features=["close"],
            scaler_type="standard",
        )
    )
    catalog.close()
    return {
        "source_id": parent.dataset_id,
        "rows": len(frame),
        "dataset_id": scaled.dataset.dataset_id,
        "plan_id": plan.dataset.dataset_id,
    }


def serve():
    import uvicorn
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    from app.main import create_app

    app = create_app()
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        return FileResponse(WEB_DIST / "index.html")

    uvicorn.run(app, host="127.0.0.1", port=8018, log_level="warning")


def main():
    fixture = asyncio.run(seed())
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "QML_DATA_DIR": str(REVIEW), "PYTHONIOENCODING": "utf-8"}
    log = (ARTIFACTS / "server.log").open("w", encoding="utf-8")
    server = subprocess.Popen(
        [sys.executable, __file__, "--serve"],
        env=env,
        stdout=log,
        stderr=log,
        cwd=ROOT / "backend",
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    try:
        for _ in range(80):
            try:
                with urllib.request.urlopen(URL + "/api/health", timeout=1):
                    break
            except OSError:
                time.sleep(0.25)
        from playwright.sync_api import expect, sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel="msedge", headless=True)
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(URL + "/regime/discovery")
            page.wait_for_timeout(1500)
            page.screenshot(path=str(ARTIFACTS / "initial-page.png"))
            page.get_by_label("Prepared dataset", exact=True).select_option(fixture["dataset_id"])
            page.get_by_label("C6 chronological split plan", exact=True).select_option(
                fixture["plan_id"]
            )
            page.get_by_label("close", exact=True).check()
            page.get_by_label("States / clusters", exact=True).fill("2")
            page.get_by_label("Minimum training occupancy", exact=True).fill("0")
            page.get_by_label("max iter", exact=True).fill("30")
            page.get_by_label("Experiment name", exact=True).fill("BTCUSDT real-data reference")
            page.get_by_role("button", name="Fit model & save artifacts", exact=True).click()
            link = page.get_by_role("link", name="Evaluate in D10 →", exact=True)
            link.wait_for(timeout=90000)
            experiment = link.get_attribute("href").split("experiment=")[1]
            page.locator("#results").scroll_into_view_if_needed()
            page.screenshot(path=str(ARTIFACTS / "d9-results.png"))
            link.click()
            page.get_by_role("button", name="Run chronological validation", exact=True).click()
            approval_button = page.get_by_role("button", name="Approve validated model for D11")
            page.get_by_label("Approval rationale").fill(
                "Real-data smoke verification of chronological fit and preprocessing replay."
            )
            expect(approval_button).to_be_enabled(timeout=90000)
            page.locator("#analysis").scroll_into_view_if_needed()
            page.screenshot(path=str(ARTIFACTS / "d10-analysis.png"))
            approval_button.click()
            deployment = page.get_by_role("link", name="Deploy in D11 →", exact=True)
            deployment.wait_for(timeout=30000)
            approval_id = deployment.get_attribute("href").split("approval=")[1]
            deployment.click()
            page.get_by_label("Market observations", exact=True).select_option(
                fixture["dataset_id"]
            )
            page.get_by_role("button", name="Run approved model", exact=True).click()
            page.locator(".regime-current").wait_for(timeout=90000)
            page.locator("#dashboard").scroll_into_view_if_needed()
            page.screenshot(path=str(ARTIFACTS / "d11-dashboard.png"))
            response = page.request.get(URL + "/api/regime/artifacts?kind=inference").json()[0]
            assert response["snapshot"]["live_current_regime"]
            assert response["snapshot"]["asset"] == "BTCUSDT"
            assert response["observation_count"] > 0
            # A real chart must use original price units even though C7 scaled close.
            history = page.request.get(
                URL + f"/api/regime/artifacts/{response['id']}/history"
            ).json()
            assert history["rows"][0]["price"] > 1000
            assert page.locator(".regime-subnav a").count() == 8
            page.locator(".regime-subnav").get_by_role(
                "link", name="Current Regime Dashboard", exact=True
            ).click()
            expect(page.locator('.regime-subnav a[aria-current="page"]')).to_have_count(1)
            assert f"approval={approval_id}" in page.url
            bundle = page.request.get(URL + f"/api/regime/artifacts/{response['id']}/bundle")
            assert bundle.ok and bundle.body().startswith(b"PK")
            page.get_by_role("button", name="Choose theme", exact=True).click()
            page.get_by_role("menuitemradio", name="Light", exact=True).click()
            page.screenshot(path=str(ARTIFACTS / "d11-light.png"))
            page.set_viewport_size({"width": 1000, "height": 850})
            page.get_by_role("button", name="Collapse sidebar", exact=True).click()
            page.locator("#dashboard").scroll_into_view_if_needed()
            page.screenshot(path=str(ARTIFACTS / "d11-compact.png"))
            assert not errors, errors
            report = {
                "fixture": fixture,
                "artifact_directory": str(ARTIFACTS),
                "experiment_id": experiment,
                "approval_id": approval_id,
                "inference_id": response["id"],
                "snapshot": response["snapshot"],
                "browser_errors": errors,
                "status": "passed",
            }
            (ARTIFACTS / "verification.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            print(json.dumps(report, indent=2))
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=15)
        log.close()


if __name__ == "__main__":
    serve() if "--serve" in sys.argv else main()
