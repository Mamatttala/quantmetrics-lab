"""Check the feature picker and compact Run controls with the real catalogue.

Run with Vite running: backend/.venv/Scripts/python.exe backend/tools/verify_features_ui.py
API calls use an in-memory dataset; no stored data is changed.
"""

import json
import os
import re
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.models.enums import AssetClass, Timeframe
from app.services.feature_catalog import DatasetContext, availability, catalogue, spec_payload

URL = os.environ.get("QML_LAYOUT_CHECK_URL", "http://127.0.0.1:5173")
COLUMNS = ["open_time", "open", "high", "low", "close", "volume"]
CONTEXT = DatasetContext(AssetClass.CRYPTO_SPOT, Timeframe.H1, frozenset(COLUMNS))
SPECS = [spec_payload(spec, availability(spec, CONTEXT)) for spec in catalogue()]
GROUPS = list(dict.fromkeys(spec["group"] for spec in SPECS))
DATASET = {
    "dataset_id": "feature-ui",
    "name": "BTC hourly",
    "category": "raw_data",
    "source": "binance",
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "row_count": 1000,
    "start_timestamp": "2026-01-01T00:00:00Z",
    "end_timestamp": "2026-02-11T15:00:00Z",
    "created_at": "2026-02-12T00:00:00Z",
    "parent_dataset_id": None,
    "operation": None,
    "asset_class": "crypto_spot",
    "market_type": "spot",
    "timezone": "UTC",
    "columns": [{"name": name, "dtype": "float64"} for name in COLUMNS],
    "checksum": "fixture",
    "storage_path": "fixture",
    "unpopulated_fields": [],
    "provenance": {},
}
CATALOG = {
    "dataset_id": DATASET["dataset_id"],
    "asset_class": "crypto_spot",
    "timeframe": "1h",
    "timezone": "UTC",
    "columns": COLUMNS,
    "unpopulated_fields": [],
    "has_time": True,
    "groups": [
        {"group": group, "features": [spec for spec in SPECS if spec["group"] == group]}
        for group in GROUPS
    ],
    "available_count": sum(spec["available"] for spec in SPECS),
    "total_count": len(SPECS),
}


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors, previews = [], []
        page.on("pageerror", lambda error: errors.append(str(error)))

        def mock_api(route):
            path = urlparse(route.request.url).path
            if route.request.method == "POST":
                body = json.loads(route.request.post_data)
                assert path == "/api/datasets/feature-ui/features" and body["write"] is False
                previews.append(body)
                route.fulfill(
                    status=400,
                    json={"error": {"code": "fixture", "message": "Preview request verified."}},
                )
                return
            payload = {
                "/api/projects": [
                    {
                        "project_id": "default",
                        "name": "Feature design check",
                        "created_at": "2026-01-01T00:00:00Z",
                    }
                ],
                "/api/health": {"status": "ok", "version": "test", "data_dir": "fixture"},
                "/api/datasets": [DATASET],
                "/api/datasets/feature-ui": DATASET,
                "/api/features/catalog": CATALOG,
            }[path]
            route.fulfill(json=payload)

        page.route(re.compile(r"^https?://[^/]+/api/"), mock_api)
        page.goto(f"{URL}/features/feature-ui")
        expect(page.locator(".feature__row")).to_have_count(len(SPECS))
        expect(page.locator(".feature__group")).to_have_count(len(GROUPS))
        expect(page.locator(".feature__windows")).to_have_count(0)
        assert (
            len(
                page.locator(".feature")
                .first.evaluate("el => getComputedStyle(el).gridTemplateColumns")
                .split()
            )
            >= 2
        )
        print(
            f"PASS: {len(SPECS)} features in {len(GROUPS)} categories form a responsive card grid"
        )

        spec = next(
            spec
            for spec in SPECS
            if spec["available"]
            and spec["param"]
            and spec["param"]["custom"]
            and spec["param"]["kind"] == "bars"
        )
        card = page.locator(".feature__row").filter(has=page.get_by_text(spec["label"], exact=True))
        card.locator(".feature__pick input").check()
        expect(card).to_have_class(re.compile("feature__row--selected"))
        token = "137"
        card.get_by_role("textbox", name=f"Custom window for {spec['label']}").fill(token)
        add = card.get_by_role("button", name=f"Add custom window for {spec['label']}")
        box = add.bounding_box()
        assert box["width"] < 70 and box["height"] == 28, box
        add.click()
        remove = card.get_by_role("button", name=f"Remove window {token} from {spec['label']}")
        expect(remove).to_be_visible()
        remove.click()
        expect(remove).to_have_count(0)
        card.locator(".feature__pick input").uncheck()
        expect(card.locator(".feature__windows")).to_have_count(0)
        card.locator(".feature__pick input").check()
        expect(card.locator(".feature__windows")).to_be_visible()
        card.get_by_role("button", name=f"Formula for {spec['label']}").click()
        expect(page.locator(".description-popup")).to_be_visible()
        page.mouse.click(0, 0)
        print(
            "PASS: feature selection, compact custom Add/remove controls "
            "and click-open formula help"
        )

        actions = page.locator(".feature__run > button")
        expect(actions).to_have_count(3)
        boxes = [actions.nth(i).bounding_box() for i in range(3)]
        assert all(box["width"] == 160 and box["height"] == 36 for box in boxes), boxes
        assert len({box["y"] for box in boxes}) == 1, boxes
        assert all(abs(boxes[i + 1]["x"] - boxes[i]["x"] - 170) < 1 for i in range(2))
        page.get_by_role("button", name="Preview", exact=True).click()
        expect(page.get_by_text("Preview request verified.", exact=True)).to_be_visible()
        assert previews and any(
            item["feature_id"] == spec["id"] for item in previews[0]["selections"]
        )
        print(
            "PASS: Run actions are 160 x 36px in one row, spaced 10px; Preview sends the selection"
        )

        page.locator(".feature__group").first.scroll_into_view_if_needed()
        desktop = Path(tempfile.gettempdir()) / "quantmetrics-features-desktop.png"
        page.screenshot(path=str(desktop))
        print(f"Screenshot: {desktop}")
        for size in [1920, 1440, 1024, 768, 390, 320]:
            page.set_viewport_size({"width": size, "height": 1000})
            assert page.locator(".main").evaluate("el => el.scrollWidth <= el.clientWidth + 1"), (
                size
            )
            assert page.locator(".feature__row").evaluate_all(
                "nodes => nodes.every(el => el.scrollWidth <= el.clientWidth + 1)"
            ), size
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), size
            print(f"PASS: page and feature cards fit {size}px")
        assert not errors, errors
        browser.close()


if __name__ == "__main__":
    main()
