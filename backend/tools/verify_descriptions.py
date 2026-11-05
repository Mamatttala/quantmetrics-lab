"""Check clickable descriptions against the Vite dev server, using mocked API reads.

Run: backend/.venv/Scripts/python.exe backend/tools/verify_descriptions.py
Requires Playwright/Chromium as used by verify_layout.py.
"""

import os
import re

from playwright.sync_api import expect, sync_playwright
from verify_layout import mock_api

URL = os.environ.get("QML_LAYOUT_CHECK_URL", "http://127.0.0.1:5173")
ROUTES = [
    "fetch",
    "view",
    "data",
    "cleaning",
    "resample",
    "features",
    "splits",
    "scaling",
    "regime",
    "volatility",
]


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.route(re.compile(r"^https?://[^/]+/api/"), mock_api)
        for route in ROUTES:
            page.goto(f"{URL}/{route}")
            title = page.locator(".page__title")
            title.wait_for()
            page.mouse.move(0, 0)
            expect(page.locator(".description-popup")).to_have_count(0)
            assert page.locator(".description-source").evaluate_all(
                "nodes => nodes.length > 0 && "
                "nodes.every(node => node.getBoundingClientRect().height === 0)"
            )
            title.hover()
            expect(page.locator(".description-popup")).to_have_count(0)
            info = title.get_by_role("button")
            info.hover()
            expect(page.locator(".description-popup")).to_have_count(0)
            info.click()
            expect(page.locator(".description-popup")).to_be_visible()
            page.mouse.move(0, 0)
            expect(page.locator(".description-popup")).to_be_visible()
            page.mouse.click(0, 0)
            expect(page.locator(".description-popup")).to_have_count(0)
            print(
                f"PASS: /{route} help opens on i click, stays open on pointer leave, "
                "closes on outside click"
            )

        # Exercise the production reusable controls without requiring model training.
        page.evaluate("""async () => {
          // Use the same Vite module URL as the controls, including any HMR timestamp,
          // so the injected provider shares their React context.
          const controlSource = await (await fetch('/src/components/PolicyGroup.tsx')).text();
          const descriptionUrl = controlSource.split('"').find(
            part => part.includes('/Description.tsx'));
          const [{default: React}, {default: ReactDOM}, {Description, DescriptionProvider},
            {PolicyGroup}, {InfoTip}] = await Promise.all([
              import('/node_modules/.vite/deps/react.js'),
              import('/node_modules/.vite/deps/react-dom_client.js'),
              import(descriptionUrl),
              import('/src/components/PolicyGroup.tsx'),
              import('/src/components/InfoTip.tsx'),
            ]);
          const h = React.createElement;
          const card = document.createElement('div');
          card.className = 'section';
          document.querySelector('.page').append(card);
          window.descriptionCheck = ReactDOM.createRoot(card);
          window.descriptionCheck.render(h(DescriptionProvider, null,
            h(PolicyGroup, {name: 'check-policy', legend: 'Policy check',
              hint: 'Group explanation.', value: 'keep', onChange: () => {},
              options: [{value: 'keep', label: 'Keep test value',
                detail: 'Keep the original observations.'},
                {value: 'fill', label: 'Fill test value', detail: 'Use the preceding observation.',
                caution: 'Check for missing bars.'}]}),
            h('div', {className: 'field'},
              h('label', {htmlFor: 'check-window'}, 'Window check'),
              h('input', {id: 'check-window', type: 'number', defaultValue: 10}),
              h(Description, null, 'Number of observations.')),
            h('fieldset', {disabled: true}, h('div', {className: 'feature__row'},
              h('label', null, h('input', {type: 'checkbox', disabled: true}), 'Feature check'),
              h(InfoTip, {label: 'Formula check', formula: 'x + y',
                detail: 'Add the two inputs.'}))),
          ));
        }""")
        option = page.get_by_role("radio", name="Fill test value", exact=True)
        info = page.get_by_role("button", name="About Fill test value", exact=True)
        try:
            info.wait_for(timeout=5000)
        except Exception:
            print("Browser errors:", errors)
            print(page.locator(".page").inner_text())
            print(
                page.locator(".description-toggle").evaluate_all(
                    "nodes => nodes.map(node => node.outerHTML)"
                )
            )
            raise
        info.click()
        expect(page.locator(".description-popup")).to_contain_text("Use the preceding observation.")
        expect(page.locator(".description-popup")).to_contain_text("Check for missing bars.")
        expect(option).not_to_be_checked()
        before = option.bounding_box()
        page.mouse.move(0, 0)
        expect(page.locator(".description-popup")).to_be_visible()
        assert before == option.bounding_box(), "Showing help must not move controls"

        page.get_by_role("button", name="Formula check", exact=True).click()
        expect(page.locator(".description-popup")).to_have_count(1)
        expect(page.locator(".description-popup")).to_contain_text("x + y")
        page.mouse.click(0, 0)
        window_info = page.get_by_role("button", name="About Window check", exact=True)
        window_info.focus()
        expect(page.locator(".description-popup")).to_have_count(0)
        window_info.press("Enter")
        expect(page.locator(".description-popup")).to_contain_text("Number of observations.")
        page.keyboard.press("Escape")
        expect(page.locator(".description-popup")).to_have_count(0)
        expect(window_info).to_be_focused()
        print(
            "PASS: option selection unchanged; disabled fieldset help, "
            "keyboard activation and Escape work"
        )

        for size in [1440, 768, 390, 320]:
            page.set_viewport_size({"width": size, "height": 900})
            info.click()
            popup = page.locator(".description-popup")
            expect(popup).to_be_visible()
            box = popup.bounding_box()
            assert box["x"] >= 0 and box["x"] + box["width"] <= size
            assert box["y"] >= 0 and box["y"] + box["height"] <= 900
            page.mouse.click(0, 0)
            expect(popup).to_have_count(0)
            print(f"PASS: description popup fits the {size}px viewport")

        assert not errors, errors
        browser.close()


if __name__ == "__main__":
    main()
