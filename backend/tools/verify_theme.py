r"""Runtime check for the light/dark theme toggle.

Loads the dev server in a real Chromium via Playwright, flips the theme, and asserts
that the document attribute, the localStorage preference, and the computed surface
colours all change together. Exits non-zero on the first failed assertion.

Requirements (one time):
    cd backend
    .\.venv\Scripts\python.exe -m pip install playwright
    .\.venv\Scripts\python.exe -m playwright install chromium

Run with the frontend dev server up:
    cd backend
    .\.venv\Scripts\python.exe tools\verify_theme.py
"""

from __future__ import annotations

import sys

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:5173/"
DARK_BODY = "rgb(17, 17, 17)"          # --bg-main #111111
LIGHT_BODY = "rgb(255, 255, 255)"      # --bg-main #ffffff
DARK_SIDEBAR = "rgb(26, 26, 26)"       # --bg-sidebar #1a1a1a
LIGHT_SIDEBAR = "rgb(248, 249, 250)"   # --bg-sidebar #f8f9fa

failures: list[str] = []


def check(label: str, condition: bool, detail: object = "") -> None:
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(label)


def theme(page) -> str | None:
    return page.evaluate("document.documentElement.getAttribute('data-theme')")


def css(page, selector: str, prop: str) -> str:
    return page.eval_on_selector(selector, f"el => getComputedStyle(el).{prop}")


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(URL, wait_until="domcontentloaded")
        page.wait_for_selector(".theme-toggle", timeout=30_000)
        page.wait_for_timeout(500)

        print("initial (dark) state")
        check("default theme is dark", theme(page) == "dark", theme(page))
        check("body background is #111111", css(page, "body", "backgroundColor") == DARK_BODY,
              css(page, "body", "backgroundColor"))
        check("sidebar background is #1a1a1a", css(page, ".sidebar", "backgroundColor") == DARK_SIDEBAR,
              css(page, ".sidebar", "backgroundColor"))
        check("two toggles are present (sidebar + explorer)",
              page.locator(".theme-toggle").count() >= 2,
              page.locator(".theme-toggle").count())

        page.locator(".theme-toggle").first.click()
        page.wait_for_timeout(600)

        print("after switching to light")
        check("theme attribute is light", theme(page) == "light", theme(page))
        check("body background is #ffffff", css(page, "body", "backgroundColor") == LIGHT_BODY,
              css(page, "body", "backgroundColor"))
        check("sidebar background is #f8f9fa", css(page, ".sidebar", "backgroundColor") == LIGHT_SIDEBAR,
              css(page, ".sidebar", "backgroundColor"))
        check("localStorage persisted light", page.evaluate("localStorage.getItem('theme')") == "light",
              page.evaluate("localStorage.getItem('theme')"))
        check("explorer background flipped",
              css(page, ".explorer", "backgroundColor") == "rgb(241, 245, 249)",
              css(page, ".explorer", "backgroundColor"))

        page.reload(wait_until="domcontentloaded")
        page.wait_for_selector(".theme-toggle", timeout=30_000)
        page.wait_for_timeout(500)
        print("after reload")
        check("light theme survived reload", theme(page) == "light", theme(page))
        check("body still light after reload", css(page, "body", "backgroundColor") == LIGHT_BODY,
              css(page, "body", "backgroundColor"))

        page.locator(".theme-toggle").first.click()
        page.wait_for_timeout(600)
        print("back to dark")
        check("theme attribute is dark again", theme(page) == "dark", theme(page))
        check("body background is #111111 again", css(page, "body", "backgroundColor") == DARK_BODY,
              css(page, "body", "backgroundColor"))

        browser.close()

    if failures:
        print(f"\n{len(failures)} check(s) FAILED: {failures}")
        return 1
    print("\nall theme checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())