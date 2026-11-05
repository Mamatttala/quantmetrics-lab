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

import json
import os
import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = os.environ.get("QML_THEME_CHECK_URL", "http://127.0.0.1:5173/")
DARK_BODY = "rgb(17, 17, 17)"          # --bg-main #111111
LIGHT_BODY = "rgb(255, 255, 255)"      # --bg-main #ffffff
DARK_SIDEBAR = "rgb(26, 26, 26)"       # --bg-sidebar #1a1a1a
LIGHT_SIDEBAR = "rgb(248, 249, 250)"   # --bg-sidebar #f8f9fa

failures: list[str] = []

THEMES = [
    ("light", "Light"), ("dark", "Dark"), ("finance-light", "Finance Light"),
    ("nord", "Nord"), ("simple-dark", "Simple Dark"),
]
PREVIEWS = {
    "light": ["rgb(255, 255, 255)", "rgb(255, 255, 255)", "rgb(184, 134, 11)", "rgb(15, 23, 42)"],
    "dark": ["rgb(17, 17, 17)", "rgb(28, 28, 28)", "rgb(212, 168, 67)", "rgb(240, 240, 240)"],
    "finance-light": [
        "rgb(255, 255, 255)", "rgb(246, 248, 250)", "rgb(9, 105, 218)", "rgb(31, 35, 40)"
    ],
    "nord": ["rgb(46, 52, 64)", "rgb(59, 66, 82)", "rgb(136, 192, 208)", "rgb(236, 239, 244)"],
    "simple-dark": [
        "rgb(15, 17, 21)", "rgb(22, 26, 32)", "rgb(47, 129, 247)", "rgb(230, 237, 243)"
    ],
}


def select_theme(page, name: str) -> None:
    page.get_by_role("button", name="Choose theme", exact=True).click()
    page.get_by_role("menuitemradio", name=name, exact=True).click()
    page.wait_for_timeout(350)  # Existing color transitions finish before computed-color checks.


def verify_legacy_tokens(page, theme_id: str) -> None:
    fixture = (
        Path(__file__).resolve().parents[2]
        / "frontend/src/__tests__/fixtures/legacy-palettes.json"
    )
    original = json.loads(fixture.read_text(encoding="utf-8"))[theme_id]

    def resolve(value: str) -> str:
        return re.sub(r"var\((--[\w-]+)\)", lambda match: resolve(original[match[1]]), value)

    actual = page.evaluate(
        "(names) => Object.fromEntries(names.map(name => [name, "
        "getComputedStyle(document.documentElement).getPropertyValue(name).trim()]))",
        list(original),
    )
    changed = [name for name, value in original.items()
               if re.sub(r"\s+", "", actual[name]) != re.sub(r"\s+", "", resolve(value))]
    check(f"{theme_id}: all original tokens unchanged", not changed, changed)


def verify_options(page) -> None:
    page.get_by_role("button", name="Choose theme", exact=True).click()
    menu = page.get_by_role("menu", name="Appearance")
    check("exactly five options", menu.get_by_role("menuitemradio").count() == 5)
    for theme_id, name in THEMES:
        option = menu.get_by_role("menuitemradio", name=name, exact=True)
        colors = option.locator(".theme-preview i").evaluate_all(
            "nodes => nodes.map(node => getComputedStyle(node).backgroundColor)"
        )
        check(f"{theme_id}: preview colors", colors == PREVIEWS[theme_id], colors)
        check(f"{theme_id}: selected indicator",
              option.get_attribute("aria-checked") == str(theme(page) == theme_id).lower())
    menu.press("Escape")


def check(label: str, condition: bool, detail: object = "") -> None:
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f"  ({detail})" if detail else ""))
    if not condition:
        failures.append(label)


def theme(page) -> str | None:
    return page.evaluate("document.documentElement.getAttribute('data-theme')")


def css(page, selector: str, prop: str) -> str:
    return page.eval_on_selector(selector, f"el => getComputedStyle(el).{prop}")


def css_var(page, name: str) -> str:
    return page.evaluate(
        "(name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim()",
        name,
    )


def stylesheet_text(page) -> str:
    return page.evaluate(
        """
        () => Array.from(document.styleSheets).flatMap((sheet) => {
          try {
            return Array.from(sheet.cssRules, (rule) => rule.cssText);
          } catch {
            return [];
          }
        }).join("\\n")
        """
    )


def add_theme_fixtures(page) -> None:
    page.evaluate(
        """
        () => {
          const select = document.createElement("select");
          select.id = "theme-verification-select";
          select.append(new Option("verification", "verification"));
          select.style.position = "fixed";
          select.style.left = "-10000px";
          select.style.top = "-10000px";
          document.body.append(select);

          const modal = document.createElement("div");
          modal.id = "theme-verification-modal";
          modal.className = "modal";
          modal.style.position = "fixed";
          modal.style.left = "-10000px";
          modal.style.top = "-10000px";
          document.body.append(modal);
        }
        """
    )


def check_theme_tokens(page, label: str) -> None:
    for name in ("--accent-dim", "--tooltip-shadow", "--modal-shadow"):
        value = css_var(page, name)
        check(f"{label} {name} is defined", bool(value), value)


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.goto(URL, wait_until="domcontentloaded")
        page.wait_for_selector(".theme-toggle", timeout=30_000)
        page.wait_for_timeout(500)
        add_theme_fixtures(page)
        stylesheet = stylesheet_text(page)

        print("initial (dark) state")
        check("default theme is dark", theme(page) == "dark", theme(page))
        verify_legacy_tokens(page, "dark")
        verify_options(page)
        check_theme_tokens(page, "dark")
        check("body background is #111111", css(page, "body", "backgroundColor") == DARK_BODY,
              css(page, "body", "backgroundColor"))
        check(
            "sidebar background is #1a1a1a",
            css(page, ".sidebar", "backgroundColor") == DARK_SIDEBAR,
              css(page, ".sidebar", "backgroundColor"))
        check("one theme toggle is present in the page toolbar",
              page.locator(".theme-toggle").count() == 1
              and page.locator(".shell__toolbar .theme-toggle").count() == 1,
              page.locator(".theme-toggle").count())

        select_theme(page, "Light")

        print("after switching to light")
        check("theme attribute is light", theme(page) == "light", theme(page))
        verify_legacy_tokens(page, "light")
        check_theme_tokens(page, "light")
        check("body background is #ffffff", css(page, "body", "backgroundColor") == LIGHT_BODY,
              css(page, "body", "backgroundColor"))
        check(
            "sidebar background is #f8f9fa",
            css(page, ".sidebar", "backgroundColor") == LIGHT_SIDEBAR,
              css(page, ".sidebar", "backgroundColor"))
        check(
            "localStorage persisted light",
            page.evaluate("localStorage.getItem('theme')") == "light",
              page.evaluate("localStorage.getItem('theme')"))
        check("explorer background flipped",
              css(page, ".explorer", "backgroundColor") == "rgb(241, 245, 249)",
              css(page, ".explorer", "backgroundColor"))
        check("select arrow uses the theme token",
              ".select-arrow" in stylesheet and "stroke: var(--select-arrow)" in stylesheet,
              "global select arrow rule")
        check("select arrow has no hardcoded stroke color",
              "stroke='%23d4a843'" not in stylesheet and "stroke: #d4a843" not in stylesheet,
              "no fixed SVG stroke")
        check("select arrow is rendered",
              css(page, "#theme-verification-select", "backgroundImage") != "none",
              css(page, "#theme-verification-select", "backgroundImage"))
        check("modal uses the light shadow token",
              css(page, "#theme-verification-modal", "boxShadow") != "none",
              css(page, "#theme-verification-modal", "boxShadow"))

        page.reload(wait_until="domcontentloaded")
        page.wait_for_selector(".theme-toggle", timeout=30_000)
        page.wait_for_timeout(500)
        print("after reload")
        check("light theme survived reload", theme(page) == "light", theme(page))
        check("body still light after reload", css(page, "body", "backgroundColor") == LIGHT_BODY,
              css(page, "body", "backgroundColor"))

        select_theme(page, "Dark")
        print("back to dark")
        check("theme attribute is dark again", theme(page) == "dark", theme(page))
        check("body background is #111111 again", css(page, "body", "backgroundColor") == DARK_BODY,
              css(page, "body", "backgroundColor"))

        for theme_id, name in THEMES[2:]:
            select_theme(page, name)
            check(f"{theme_id}: applies immediately", theme(page) == theme_id)
            check(f"{theme_id}: background",
                  css(page, "body", "backgroundColor") == PREVIEWS[theme_id][0])
            check(f"{theme_id}: text", css(page, "body", "color") == PREVIEWS[theme_id][3])
            check(f"{theme_id}: persists",
                  page.evaluate("localStorage.getItem('theme')") == theme_id)
            verify_options(page)
            page.reload(wait_until="domcontentloaded")
            page.wait_for_selector(".theme-toggle")
            check(f"{theme_id}: restored on reload", theme(page) == theme_id)

        browser.close()

    if failures:
        print(f"\n{len(failures)} check(s) FAILED: {failures}")
        return 1
    print("\nall theme checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
