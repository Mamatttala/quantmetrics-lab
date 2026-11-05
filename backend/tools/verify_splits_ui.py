"""Exercise the split workflow in Chromium against an isolated real backend.

Requires Vite on port 5173. All API calls go to a temporary TestClient catalog.
"""

import gc
import os
import re
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from playwright.sync_api import expect, sync_playwright

from app.config import get_settings
from app.main import create_app
from tests.conftest import hourly_frame
from tests.test_regime_api import register


def main():
    with tempfile.TemporaryDirectory(
        prefix="qml-split-ui-", ignore_cleanup_errors=True
    ) as temporary:
        os.environ["QML_DATA_DIR"] = temporary
        get_settings.cache_clear()
        with TestClient(create_app()) as client, sync_playwright() as playwright:
            source = register(Path(temporary), hourly_frame(1000), name="Split UI source")
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            errors, created, runs = [], [], []
            page.on("pageerror", lambda error: errors.append(str(error)))

            def forward_api(route):
                url = urlparse(route.request.url)
                response = client.request(
                    route.request.method,
                    url.path + (f"?{url.query}" if url.query else ""),
                    content=route.request.post_data,
                    headers={"Content-Type": "application/json"},
                )
                if url.path == "/api/splits/create" and response.status_code == 201:
                    created.append(response.json())
                if url.path == "/api/scaling/fit_transform":
                    runs.append(route.request.post_data_json)
                if response.status_code >= 400:
                    errors.append(f"{url.path}: {response.status_code} {response.text}")
                route.fulfill(
                    status=response.status_code,
                    body=response.content,
                    content_type=response.headers.get("content-type", "application/json"),
                )

            page.route(re.compile(r"^https?://[^/]+/api/"), forward_api)
            page.goto(f"http://127.0.0.1:5173/splits/{source}")
            for scheme in ["tscv_expanding", "tscv_rolling"]:
                page.locator(f'input[value="{scheme}"]').check()
                validation = page.get_by_role("checkbox", name="Include validation", exact=True)
                expect(validation).not_to_be_checked()
                for enabled in [False, True, False]:
                    validation.set_checked(enabled)
                    page.locator("#split-name").fill(f"{scheme} validation={enabled}")
                    with page.expect_response(
                        lambda response: response.url.endswith("/api/splits/create")
                    ):
                        page.get_by_role("button", name="Generate Split Plan", exact=True).click()
                    plan = created[-1]
                    expect(page.locator(".result__bar--ok").first).to_contain_text(
                        plan["dataset"]["dataset_id"]
                    )
                    expect(page.locator(".timeline__bar--train")).to_have_count(5)
                    expect(page.locator(".timeline__bar--val")).to_have_count(5 if enabled else 0)
                    expect(page.locator(".timeline__swatch--val")).to_have_count(
                        1 if enabled else 0
                    )
                    roles = ["train", "val", "test"] if enabled else ["train", "test"]
                    for lane in page.locator(".timeline__lane").all():
                        boxes = [
                            lane.locator(f".timeline__bar--{role}").bounding_box() for role in roles
                        ]
                        assert all(box["width"] > 0 for box in boxes)
                        for previous, following in zip(boxes, boxes[1:], strict=False):
                            assert previous["x"] + previous["width"] <= following["x"] + 1
                    assert plan["dataset"]["category"] == "outputs"
                    assert all(
                        item["category"] == "processed_data" for item in plan["partitions"]
                    )
                    assert plan["report"]["parameters"]["include_validation"] is enabled
                    print(f"PASS: {scheme}, validation={enabled}, correct spans and legend")

            group = page.locator(".explorer__group").filter(
                has=page.get_by_text("Outputs", exact=True)
            )
            expect(group.locator(".explorer__row")).to_have_count(6)
            selected = created[1]["dataset"]["dataset_id"]  # Older opted-in plan, not the newest.
            page.goto("http://127.0.0.1:5173/scaling")
            expect(page.locator('#scaling-dataset optgroup[label="Splits"] option')).to_have_count(
                6
            )
            page.locator("#scaling-dataset").select_option(selected)
            expect(page.locator("#scaling-plan")).to_have_value(selected)
            expect(page.locator("#scaling-fold")).to_be_visible()
            expect(
                page.locator(".columnpick__name").filter(has_text=re.compile("^close$"))
            ).to_be_visible()
            assert (
                not page.locator(".columnpick__name")
                .filter(has_text=re.compile("^(fold|row_count|start_index)$"))
                .count()
            )
            page.locator("#scaling-fold").select_option("2")
            page.locator(".actions > button").first.click()
            expect(page.get_by_role("heading", name="Result · standard", exact=True)).to_be_visible(
                timeout=30000
            )
            assert runs[-1]["dataset_id"] == source
            assert runs[-1]["split_plan_id"] == selected and runs[-1]["fold"] == 2
            print(
                "PASS: selecting an older split chooses that plan, "
                "loads source features, and completes scaling"
            )

            for route_name, select_id, heading in [
                ("regime", "regime-dataset", "Source"),
                ("volatility", "vol-dataset", "Return series"),
            ]:
                page.goto(f"http://127.0.0.1:5173/{route_name}")
                selector = page.locator(f"#{select_id}")
                expect(selector.locator('optgroup[label="Splits"] option')).to_have_count(6)
                selector.select_option(selected)
                expect(page.get_by_role("heading", name=heading, exact=True)).to_be_visible()
                expect(selector).to_have_value(selected)
                expect(selector.locator('optgroup[label="Splits"] option')).to_have_count(6)
                assert not page.locator(".result__bar--error").count()
                print(f"PASS: {route_name} loads source options from the Splits selection")

            assert not errors, errors
            browser.close()
        get_settings.cache_clear()
        gc.collect()


if __name__ == "__main__":
    main()
