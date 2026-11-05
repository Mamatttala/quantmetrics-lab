"""C8 HTTP driver; run against a live API, without market-data dependencies.

    python tools/e2e_exploratory.py
    python tools/e2e_exploratory.py --base http://127.0.0.1:8018/api --catalog PATH

Creates a small uploaded dataset, a split plan, and two pairs of PCA artifacts.
The optional catalog check verifies both lineage edges directly in SQLite.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sqlite3
import sys
import time
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError


def call(base: str, path: str, payload: dict[str, Any] | None = None) -> tuple[int, Any]:
    request = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status, json.loads(response.read())


def run(base: str, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    status, accepted = call(base, f"/exploratory/{endpoint}", payload)
    assert status == 202, (status, accepted)
    assert accepted["poll"] == f"/api/jobs/{accepted['job_id']}"
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        _, job = call(base, f"/jobs/{accepted['job_id']}")
        if job["status"] == "succeeded":
            return dict(job["result"])
        assert job["status"] == "running", job
        time.sleep(0.1)
    raise TimeoutError("C8 job did not finish")


def upload(base: str) -> str:
    rows = [
        "open_time,close_time,x,y,z,group,close,log_return,realized_volatility,"
        "lag_target,controlled_missing,regime_state"
    ]
    rng = random.Random(184)
    xs = [rng.gauss(0, 1) for _ in range(180)]
    for i in range(180):
        stamp = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(hours=i)
        x = xs[i]
        y = x + rng.gauss(0, 0.005)
        lag_target = xs[i - 3] if i >= 3 else ""
        missing = x if i % 17 else ""
        rows.append(
            f"{stamp.isoformat()},{(stamp + timedelta(days=99)).isoformat()},"
            f"{x},{y},{rng.gauss(0, 1)},{'a' if i % 2 else 'b'},"
            f"{100 * math.exp(x * 0.01)},{x * 0.01},{abs(x * 0.01)},"
            f"{lag_target},{missing},{i % 2}"
        )
    boundary = "----qml-c8-e2e"
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        'filename="c8-synthetic.csv"\r\nContent-Type: text/csv\r\n\r\n'
        + "\n".join(rows)
        + f"\r\n--{boundary}--\r\n"
    ).encode()
    request = urllib.request.Request(
        base + "/datasets/upload",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return str(json.loads(response.read())["dataset_id"])


def verify_saved(
    base: str,
    result: dict[str, Any],
    source: str,
    causal: bool,
    catalog: Path | None = None,
    plan: str | None = None,
) -> None:
    assert result["causal_for_downstream_training"] is causal
    _, listed = call(base, "/datasets")
    ids = {item["dataset_id"] for item in listed}
    for key, operation in [
        ("dataset", "pca_dimensionality_reduction"),
        ("params_dataset", "pca_parameters"),
    ]:
        dataset_id = result[key]["dataset_id"]
        assert dataset_id in ids
        _, meta = call(base, f"/datasets/{dataset_id}")
        assert meta["operation"] == operation
        assert meta["parent_dataset_id"] == source
        assert json.loads(meta["provenance"]["causal_for_downstream_training"]) is causal
        parameters = json.loads(meta["provenance"]["pca_parameters"])
        assert parameters["n_fit_rows"] == result["n_fit_rows"]
        assert parameters["loadings"] == result["loadings"]
        if plan:
            assert meta["provenance"]["split_plan_id"] == plan
        if catalog:
            with sqlite3.connect(f"{catalog.resolve().as_uri()}?mode=ro", uri=True) as connection:
                edges = connection.execute(
                    "SELECT parent_id FROM lineage WHERE child_id = ?", (dataset_id,)
                ).fetchall()
            assert {row[0] for row in edges} == {source, *([plan] if plan else [])}
    columns = {column["name"] for column in result["dataset"]["columns"]}
    assert columns == {"open_time", "pc_1", "pc_2", "group"}, columns
    assert result["dataset"]["row_count"] == 180
    assert result["params_dataset"]["category"] == "outputs"
    _, page = call(base, f"/datasets/{result['dataset']['dataset_id']}/page?offset=0&limit=5")
    assert len(page["rows"]) == 5
    assert all(row["pc_1"] is not None for row in page["rows"])


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8000/api")
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--web", help="Also check C8 in a headless browser at this frontend URL")
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    base = args.base.rstrip("/")
    run_id = str(time.time_ns())
    source = upload(base)
    _, before = call(base, f"/datasets/{source}")
    _, listed_before = call(base, "/datasets")
    request = {"dataset_id": source, "features": ["x", "y", "z"]}
    _, options = call(base, f"/exploratory/options?dataset_id={source}")
    assert {"x", "y", "z", "lag_target"} <= set(options["features"])
    correlation = run(
        base,
        "correlation",
        {
            **request,
            "target": "lag_target",
            "max_lag": 5,
            "group_column": "group",
        },
    )
    assert len(correlation["pairwise"]) == 6
    assert len(correlation["matrix"]) == 4 and len(correlation["vif"]) == 4
    assert any({"x", "y"} <= set(group["members"]) for group in correlation["redundancy_groups"])
    assert next(row for row in correlation["vif"] if row["feature"] == "x")["vif"] > 5
    assert correlation["ccf_summary"][0]["best_positive_lag"] == 3
    assert any("future target" in warning for warning in correlation["warnings"])
    assert correlation["ccf"] and correlation["conditional"]
    print("PASS: correlation diagnostics returned through a job")
    rolling = run(
        base,
        "correlation",
        {
            **request,
            "scope": "rolling",
            "rolling_window": 40,
            "rolling_pair_selection": "selected_pairs",
            "rolling_pairs": [["x", "y"]],
        },
    )
    assert rolling["causal_for_downstream_training"] is True
    assert rolling["matrix"] == rolling["vif"] == rolling["ccf"] == []
    assert rolling["rolling"] and rolling["rolling_summary"][0]["n_points"] > 0
    print("PASS: rolling scope returned causal series without static diagnostics")
    diagnostics = run(base, "pca", request)
    assert diagnostics["dataset"] is None and diagnostics["params_dataset"] is None
    _, listed_after = call(base, "/datasets")
    assert len(listed_before) == len(listed_after)
    print("PASS: PCA diagnostics displayed without creating datasets")
    saved = run(
        base,
        "pca",
        {
            **request,
            "write_dataset": True,
            "carry_columns": ["group"],
            "name": f"{source}_{run_id}_full_pca",
        },
    )
    assert saved["loadings"] == diagnostics["loadings"]
    verify_saved(base, saved, source, False, args.catalog)
    print("PASS: full-sample PCA saved, cataloged, linked and labeled non-causal")
    # Make a plan so the optional training-span check runs even in an empty catalog.
    status, split = call(
        base, "/splits/create", {"dataset_id": source, "split_type": "percentage_70_15_15"}
    )
    assert status == 201, split
    plan = split["dataset"]["dataset_id"]
    train = run(
        base,
        "pca",
        {
            **request,
            "scope": "train_span",
            "split_plan_id": plan,
            "write_dataset": True,
            "strict_causal": True,
            "carry_columns": ["group"],
            "name": f"{source}_{run_id}_train_pca",
        },
    )
    verify_saved(base, train, source, True, args.catalog, plan)
    assert train["n_fit_rows"] == next(
        row["row_count"] for row in split["plan"] if row["role"] == "train"
    )
    print("PASS: train-span PCA saved, labeled causal and linked to source and plan")
    try:
        call(
            base,
            "/exploratory/pca",
            {
                **request,
                "write_dataset": True,
                "strict_causal_mode": True,
                "new_dataset_name": f"{source}_strict_pca",
            },
        )
    except HTTPError as exc:
        assert exc.code == 422
        assert "Strict causal" in exc.read().decode()
    else:
        raise AssertionError("Strict mode accepted a full-sample PCA save")
    print("PASS: strict-mode full-sample save refused with 422")
    _, after = call(base, f"/datasets/{source}")
    assert before == after
    print("PASS: source metadata and checksum unchanged; open_time preserved")
    if args.web:
        browser_checks(base, args.web, source, args.screenshot)
    print(f"C8 e2e passed. Source dataset: {source}")
    print("FAILURES: 0")
    return 0


def browser_checks(base: str, web: str, source: str, screenshot: Path | None) -> None:
    """Optional focused browser pass, also usable with an isolated API/catalog."""
    from playwright.sync_api import expect, sync_playwright

    with sync_playwright() as playwright:
        edge = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
        browser = playwright.chromium.launch(
            headless=True, **({"executable_path": str(edge)} if edge.exists() else {})
        )
        page = browser.new_page(viewport={"width": 1680, "height": 1050})
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))

        def proxy(route):
            path = route.request.url.split("/api/", 1)[1]
            route.fulfill(response=route.fetch(url=base + "/" + path))

        page.route(web.rstrip("/") + "/api/**", proxy)
        try:
            page.goto(web.rstrip("/") + "/exploratory/" + source)
            expect(page.get_by_role("button", name="Run Correlation", exact=True)).to_be_enabled(
                timeout=60000
            )
            links = page.locator(".sidebar__nav a").all_text_contents()
            c7 = next(i for i, label in enumerate(links) if "C7" in label)
            assert "C8" in links[c7 + 1]
            expect(page.locator("#exploratory-dataset")).to_have_value(source)
            assert page.locator("#exploratory-dataset option").count() > 1
            expect(page.get_by_label("Analysis scope", exact=True)).to_have_value("full_dataset")
            page.get_by_role("button", name="Run Correlation", exact=True).click()
            expect(page.get_by_test_id("correlation-results")).to_be_visible(timeout=60000)
            page.get_by_role("button", name="Run PCA Diagnostics", exact=True).click()
            expect(page.get_by_test_id("pca-results")).to_be_visible(timeout=60000)
            expect(page.get_by_test_id("correlation-results")).to_be_attached()
            expect(page.get_by_test_id("pca-saved")).to_have_count(0)
            page.get_by_role("button", name="Run Correlation", exact=True).click()
            expect(page.get_by_role("button", name="Run Correlation", exact=True)).to_be_enabled()
            expect(page.get_by_test_id("pca-results")).to_be_attached()
            page.get_by_label("Strict causal mode for saving (default off)", exact=True).check()
            expect(page.get_by_role("button", name="Run PCA and Save Dataset")).to_be_disabled()
            expect(page.get_by_role("button", name="Run PCA Diagnostics")).to_be_enabled()
            page.get_by_label("Strict causal mode for saving (default off)", exact=True).uncheck()
            page.get_by_label("Saved dataset name (optional)").fill(f"{source}_browser_full")
            page.get_by_role("button", name="Run PCA and Save Dataset", exact=True).click()
            expect(page.get_by_test_id("pca-saved")).to_be_visible(timeout=60000)
            expect(page.get_by_test_id("correlation-results")).to_be_attached()
            expect(page.get_by_test_id("pca-saved")).to_contain_text("Non-causal fit")
            # Metadata links lead to normal Data Explorer datasets, not private C8 storage.
            banner_links = page.get_by_test_id("pca-saved").locator("a")
            assert banner_links.count() == 2
            saved_path = banner_links.first.get_attribute("href")
            assert saved_path and re.fullmatch(r"/data/[A-Za-z0-9]+", saved_path)
            page.get_by_label("Grouping column (optional)", exact=True).select_option("group")
            page.get_by_role("button", name="Run Correlation", exact=True).click()
            expect(
                page.get_by_role("tab", name="Conditional Correlation", exact=True)
            ).to_be_visible(timeout=60000)
            page.get_by_role("tab", name="Conditional Correlation", exact=True).click()
            expect(page.get_by_label("Conditional group", exact=True)).to_be_visible()
            page.get_by_label("Analysis scope", exact=True).select_option("train_span")
            page.get_by_label("Saved dataset name (optional)").fill(f"{source}_browser_train")
            page.get_by_role("button", name="Run PCA and Save Dataset", exact=True).click()
            expect(page.get_by_test_id("pca-saved")).to_contain_text(
                "Train-span C8 fit", timeout=60000
            )
            if screenshot:
                page.get_by_test_id("pca-results").scroll_into_view_if_needed()
                page.screenshot(path=str(screenshot))
            page.get_by_role("button", name="Clear Results", exact=True).click()
            expect(page.get_by_test_id("pca-results")).to_have_count(0)
            expect(page.get_by_test_id("correlation-results")).to_have_count(0)
            assert not errors, errors
            print("PASS: browser charts, independent results, strict mode, saving and clear")
        except Exception:
            print("Browser errors:", errors)
            print("Browser page:", page.locator("body").inner_text()[:3000])
            if screenshot:
                page.screenshot(path=str(screenshot))
            raise
        finally:
            browser.close()


if __name__ == "__main__":
    raise SystemExit(main())
