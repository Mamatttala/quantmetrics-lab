import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ThemeToggle } from "@/components/ThemeToggle";
import { THEMES, ThemeProvider, applyTheme, isLightTheme, useTheme } from "@/theme";
import legacy from "./fixtures/legacy-palettes.json";

const tokensCss = readFileSync(resolve("src/styles/tokens.css"), "utf8");
const themesCss = readFileSync(resolve("src/styles/themes.css"), "utf8");
const html = readFileSync(resolve("index.html"), "utf8");
const bootScript = html.match(/<script>([\s\S]*?)<\/script>/)?.[1] ?? "";

function declarations(block: string): Record<string, string> {
  return Object.fromEntries([...block.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)]
    .map((match) => [match[1], match[2]?.trim()]));
}

function OriginalControls() {
  const { theme, toggle } = useTheme();
  return <>
    <output>{theme}</output>
    <button onClick={toggle}>Original toggle</button>
    <button onClick={() => { toggle(); toggle(); }}>Toggle twice</button>
  </>;
}

function renderSelector() {
  return render(<ThemeProvider><ThemeToggle /></ThemeProvider>);
}

beforeEach(() => {
  window.localStorage.clear();
  document.documentElement.removeAttribute("data-theme");
  document.documentElement.removeAttribute("style");
});
afterEach(() => vi.restoreAllMocks());

describe("supported themes and legacy compatibility", () => {
  it("offers exactly the five required IDs and retains the original schemes", () => {
    expect(THEMES.map((entry) => entry.id)).toEqual([
      "light", "dark", "finance-light", "nord", "simple-dark",
    ]);
    expect(THEMES.filter(({ id }) => isLightTheme(id)).map(({ id }) => id))
      .toEqual(["light", "finance-light"]);
  });

  it("preserves every original Light and Dark token and adds no overriding root values", () => {
    const dark = declarations(tokensCss.match(/:root\s*\{([\s\S]*?)\}/)?.[1] ?? "");
    const light = {
      ...dark,
      ...declarations(tokensCss.match(/\[data-theme="light"\]\s*\{([\s\S]*?)\}/)?.[1] ?? ""),
    };
    expect(dark).toEqual(legacy.dark);
    expect(light).toEqual(legacy.light);
    const additions = declarations(themesCss.match(/:root\s*\{([\s\S]*?)\}/)?.[1] ?? "");
    expect(Object.keys(additions).filter((name) => name in dark)).toEqual([]);
  });

  it.each([null, "unknown", "system"])("keeps Dark as the default for stored value %s", (saved) => {
    if (saved !== null) localStorage.setItem("theme", saved);
    const matchMedia = vi.fn();
    vi.stubGlobal("matchMedia", matchMedia);
    try {
      new Function(bootScript)();
      expect(document.documentElement.dataset.theme).toBe("dark");
      renderSelector();
      expect(document.documentElement.dataset.theme).toBe("dark");
      expect(matchMedia).not.toHaveBeenCalled();
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("retains the original Light/Dark toggle behavior, including batched toggles", () => {
    render(<ThemeProvider><OriginalControls /></ThemeProvider>);
    fireEvent.click(screen.getByText("Original toggle"));
    expect(document.documentElement.dataset.theme).toBe("light");
    fireEvent.click(screen.getByText("Toggle twice"));
    expect(document.documentElement.dataset.theme).toBe("light");
    fireEvent.click(screen.getByText("Original toggle"));
    expect(document.documentElement.dataset.theme).toBe("dark");
  });

  it.each(THEMES)("restores $id before first paint and after provider mount", ({ id }) => {
    localStorage.setItem("theme", id);
    new Function(bootScript)();
    expect(document.documentElement.dataset.theme).toBe(id);
    renderSelector();
    expect(document.documentElement.dataset.theme).toBe(id);
    expect(localStorage.getItem("theme")).toBe(id);
    expect(localStorage.getItem("app-theme")).toBeNull();
  });
});

describe("appearance selector", () => {
  it("renders all five names, four-color previews, and one selected indicator", () => {
    renderSelector();
    fireEvent.click(screen.getByRole("button", { name: "Choose theme" }));
    const menu = screen.getByRole("menu", { name: "Appearance" });
    expect(within(menu).getAllByRole("menuitemradio")).toHaveLength(5);
    for (const entry of THEMES) {
      const option = within(menu).getByRole("menuitemradio", { name: entry.name });
      const preview = option.querySelector(".theme-preview");
      expect(preview?.getAttribute("data-theme")).toBe(entry.id);
      expect(preview?.children).toHaveLength(4);
      expect(option.getAttribute("aria-checked")).toBe(String(entry.id === "dark"));
      expect(option.querySelector(".theme-menu__selected") !== null).toBe(entry.id === "dark");
    }
  });

  it.each(THEMES)("applies and persists $id immediately, then restores it on remount", ({ id, name }) => {
    const view = renderSelector();
    fireEvent.click(screen.getByRole("button", { name: "Choose theme" }));
    fireEvent.click(screen.getByRole("menuitemradio", { name }));
    expect(document.documentElement.dataset.theme).toBe(id);
    expect(localStorage.getItem("theme")).toBe(id);
    expect(screen.queryByRole("menu")).toBeNull();
    view.unmount();
    applyTheme("dark");
    renderSelector();
    expect(document.documentElement.dataset.theme).toBe(id);
    fireEvent.click(screen.getByRole("button", { name: "Choose theme" }));
    expect(screen.getByRole("menuitemradio", { name }).getAttribute("aria-checked")).toBe("true");
  });

  it("supports arrow keys, Home/End, Escape, and outside dismissal", () => {
    renderSelector();
    const trigger = screen.getByRole("button", { name: "Choose theme" });
    fireEvent.click(trigger);
    const menu = screen.getByRole("menu");
    expect(document.activeElement).toBe(screen.getByRole("menuitemradio", { name: "Dark" }));
    fireEvent.keyDown(menu, { key: "ArrowDown" });
    expect(document.activeElement).toBe(screen.getByRole("menuitemradio", { name: "Finance Light" }));
    fireEvent.keyDown(menu, { key: "End" });
    expect(document.activeElement).toBe(screen.getByRole("menuitemradio", { name: "Simple Dark" }));
    fireEvent.keyDown(menu, { key: "Home" });
    expect(document.activeElement).toBe(screen.getByRole("menuitemradio", { name: "Light" }));
    fireEvent.keyDown(menu, { key: "Escape" });
    expect(screen.queryByRole("menu")).toBeNull();
    expect(document.activeElement).toBe(trigger);
    fireEvent.click(trigger);
    fireEvent.pointerDown(document.body);
    expect(screen.queryByRole("menu")).toBeNull();
  });

  it("still applies a selection when localStorage writes are unavailable", () => {
    renderSelector();
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => { throw new Error("disabled"); });
    fireEvent.click(screen.getByRole("button", { name: "Choose theme" }));
    fireEvent.click(screen.getByRole("menuitemradio", { name: "Nord" }));
    expect(document.documentElement.dataset.theme).toBe("nord");
  });
});
