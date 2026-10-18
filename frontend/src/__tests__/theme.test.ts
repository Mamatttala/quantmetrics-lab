import { createElement } from "react";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { ThemeProvider, useTheme } from "@/theme";

const tokensCss = readFileSync(resolve(process.cwd(), "src/styles/tokens.css"), "utf8");
const globalCss = readFileSync(resolve(process.cwd(), "src/styles/global.css"), "utf8");

function ThemeControls() {
  const { toggle } = useTheme();

  return createElement(
    "div",
    null,
    createElement(
      "select",
      { "aria-label": "Theme select", defaultValue: "one" },
      createElement("option", { value: "one" }, "One"),
    ),
    createElement("div", { className: "modal", "data-testid": "theme-modal" }),
    createElement("button", { type: "button", onClick: toggle }, "Switch theme"),
  );
}

function renderThemeControls() {
  return render(
    createElement(
      ThemeProvider,
      null,
      createElement(ThemeControls),
    ),
  );
}

describe("light theme controls", () => {
  beforeEach(() => {
    window.localStorage.clear();
    document.documentElement.removeAttribute("data-theme");

    const style = document.createElement("style");
    style.dataset.themeTest = "true";
    style.textContent = `${tokensCss}\n${globalCss}`;
    document.head.append(style);
  });

  afterEach(() => {
    document.head.querySelectorAll('style[data-theme-test="true"]').forEach((style) => {
      style.remove();
    });
  });

  it("uses the light accent for select arrows and the light modal shadow", () => {
    renderThemeControls();

    fireEvent.click(screen.getByRole("button", { name: "Switch theme" }));

    expect(document.documentElement.getAttribute("data-theme")).toBe("light");
    expect(screen.getByRole("combobox")).toBeTruthy();
    expect(screen.getByTestId("theme-modal")).toBeTruthy();
    expect(tokensCss).toContain("--accent-dim: #b8941f;");
    expect(tokensCss).toContain("--tooltip-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);");
    expect(tokensCss).toContain("--modal-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);");
    expect(globalCss).toContain("var(--accent-dim)");
    expect(globalCss).toContain("stroke: var(--accent-dim);");
    expect(globalCss).toContain("box-shadow: var(--modal-shadow);");
  });
});
