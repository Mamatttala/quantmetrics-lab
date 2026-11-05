"""Browser regression check for pane resizing and responsive content.

Run with the frontend Vite dev server on port 5173:
    backend/.venv/Scripts/python.exe backend/tools/verify_layout.py

API reads are mocked; this check does not need or modify a backend dataset.
Requires the same Playwright installation as verify_theme.py.
"""

import os
import re
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

URL = os.environ.get("QML_LAYOUT_CHECK_URL", "http://127.0.0.1:5173")


def mock_api(route):
    path = urlparse(route.request.url).path
    payload = {
        "/api/projects": [
            {"project_id": "default", "name": "Layout check", "created_at": "2026-01-01T00:00:00Z"}
        ],
        "/api/health": {"status": "ok", "version": "test", "data_dir": "test"},
        "/api/datasets": [],
    }.get(path, [])
    assert route.request.method == "GET", "Layout checks must only read API data"
    route.fulfill(json=payload)


def width(page, selector):
    return page.locator(selector).first.evaluate("el => el.getBoundingClientRect().width")


def near(actual, expected):
    assert abs(actual - expected) < 2, (actual, expected)


def drag(page, name, delta):
    box = page.get_by_role("separator", name=name).bounding_box()
    x, y = box["x"] + box["width"] / 2, 180
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + delta, y, steps=8)
    page.mouse.up()


def fits(page):
    assert page.locator(".main").evaluate("el => el.scrollWidth <= el.clientWidth + 1"), (
        "Main content overflows horizontally"
    )
    near(width(page, ".page"), page.locator(".main").evaluate("el => el.clientWidth"))
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), (
        "Document overflows horizontally"
    )


def charts_fit(page):
    try:
        page.wait_for_function(
            """() => {
      const plot = document.querySelector('.js-plotly-plot');
      const candle = document.querySelector('.tv-lightweight-charts');
      return plot?._fullLayout && candle
        && Math.abs(plot._fullLayout.width - plot.parentElement.clientWidth) < 2
        && Math.abs(candle.clientWidth - candle.parentElement.clientWidth) < 2;
        }""",
            timeout=10000,
        )
    except Exception:
        print(
            page.evaluate("""() => ({
          plot: document.querySelector('.js-plotly-plot')?._fullLayout?.width,
          plotParent: document.querySelector('.js-plotly-plot')?.parentElement.clientWidth,
          candle: document.querySelector('.tv-lightweight-charts')?.clientWidth,
          candleParent: document.querySelector('.tv-lightweight-charts')?.parentElement.clientWidth,
        })""")
        )
        raise


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 2560, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.route(re.compile(r"^https?://[^/]+/api/"), mock_api)
        page.goto(f"{URL}/fetch")
        page.get_by_label("Symbol", exact=True).wait_for()
        near(width(page, ".sidebar"), 220)
        near(width(page, ".explorer"), 280)
        fits(page)
        initial = width(page, ".page")
        assert initial > 1400, "Page still has the old width cap"
        page.get_by_role("button", name="Collapse sidebar", exact=True).click()
        near(width(page, ".page"), initial + 168)
        page.get_by_role("button", name="Hide", exact=True).click()
        near(width(page, ".page"), initial + 168 + 262)
        page.get_by_role("button", name="Expand sidebar", exact=True).click()
        near(width(page, ".page"), initial + 262)
        page.get_by_role("button", name="Show Data Explorer", exact=True).click()
        near(width(page, ".page"), initial)
        print("PASS: either/both hidden panes give their space to the full-width page")

        for name, selector, key, start in (
            ("Resize Sidebar Menu", ".sidebar", "qca.ui.sidebarWidth", 220),
            ("Resize Data Explorer", ".explorer", "qca.ui.explorerWidth", 280),
        ):
            before = width(page, ".main")
            drag(page, name, 80)
            near(width(page, selector), start + 80)
            near(width(page, ".main"), before - 80)
            drag(page, name, 700)
            near(width(page, selector), 500)
            drag(page, name, -450)
            near(width(page, selector), 200)
            drag(page, name, 130)
            near(width(page, selector), 330)
            page.reload()
            page.get_by_label("Symbol", exact=True).wait_for()
            near(width(page, selector), 330)
            assert page.evaluate("key => localStorage.getItem(key)", key) == "330"
        print("PASS: both real borders track dragging, clamp to 200-500px and persist")

        # Mount the production chart components within a real page card. Vite imports
        # let this exercise chart layout without training a forecasting model.
        page.evaluate("""async () => {
          const [{default: React}, {default: ReactDOM}, {ResponsivePlot},
            {Candles}, {ThemeProvider}] =
            await Promise.all([
              import('/node_modules/.vite/deps/react.js'),
              import('/node_modules/.vite/deps/react-dom_client.js'),
              import('/src/charts/ResponsivePlot.tsx'),
              import('/src/charts/Candles.tsx'),
              import('/src/theme.tsx'),
            ]);
          const card = document.createElement('section');
          card.className = 'section layout-chart-fixture';
          document.querySelector('.page').append(card);
          const h = React.createElement;
          window.layoutChartRoot = ReactDOM.createRoot(card);
          window.layoutChartRoot.render(h(ThemeProvider, null,
            h(ResponsivePlot, {
              data: [{x: [1,2,3], y: [1,3,2], type: 'scatter'}],
              layout: {autosize: true}, config: {displayModeBar: false},
              style: {width: '100%', height: '240px'},
            }),
            h(Candles, {bars: [
              {time: 1767225600, open: 100, high: 110, low: 90, close: 105, volume: 100},
              {time: 1767312000, open: 105, high: 120, low: 100, close: 115, volume: 120},
            ], markers: [], height: 240}),
          ));
        }""")
        charts_fit(page)
        old_card = width(page, ".layout-chart-fixture")
        page.get_by_role("button", name="Collapse sidebar", exact=True).click()
        charts_fit(page)
        near(width(page, ".layout-chart-fixture"), old_card + 278)
        page.get_by_role("button", name="Hide", exact=True).click()
        charts_fit(page)
        near(width(page, ".layout-chart-fixture"), old_card + 278 + 312)
        page.get_by_role("button", name="Expand sidebar", exact=True).click()
        page.get_by_role("button", name="Show Data Explorer", exact=True).click()
        drag(page, "Resize Sidebar Menu", -80)
        charts_fit(page)
        drag(page, "Resize Data Explorer", -80)
        charts_fit(page)
        print("PASS: cards, Plotly and candlestick canvases follow pane toggles and drags")

        for size in [1920, 1440, 1180, 1024, 960, 768, 640, 390, 320]:
            page.set_viewport_size({"width": size, "height": 900})
            charts_fit(page)
            fits(page)
            page.get_by_role("button", name="Hide", exact=True).click()
            charts_fit(page)
            fits(page)
            if size <= 640:
                assert page.locator(".main").bounding_box()["height"] == 900
            page.get_by_role("button", name="Show Data Explorer", exact=True).click()
            print(f"PASS: {size}px viewport with explorer visible/hidden")

        assert not errors, errors
        browser.close()


if __name__ == "__main__":
    main()
