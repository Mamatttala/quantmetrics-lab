import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

export type Theme = "dark" | "light";

/** localStorage key. The spec's `theme` key, not a namespaced one, so an existing
 *  preference carries over. */
const STORAGE_KEY = "theme";

function readStoredTheme(): Theme {
  try {
    return window.localStorage.getItem(STORAGE_KEY) === "light" ? "light" : "dark";
  } catch {
    // localStorage throws in private windows; the default is dark.
    return "dark";
  }
}

/** Apply a theme to the document. Idempotent, so it is safe to call from the
 *  toggle and again from the mount effect. */
export function applyTheme(theme: Theme): void {
  document.documentElement.setAttribute("data-theme", theme);
}

interface ThemeContextValue {
  theme: Theme;
  toggle: () => void;
}

const ThemeContext = createContext<ThemeContextValue>({ theme: "dark", toggle: () => {} });

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<Theme>(readStoredTheme);

  // The attribute is set synchronously in `toggle` as well, so a chart that reads
  // the resolved CSS variables during its own effect sees the new theme rather
  // than waiting for an ancestor effect to run.
  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  const toggle = useCallback(() => {
    setTheme((value) => {
      const next: Theme = value === "light" ? "dark" : "light";
      applyTheme(next);
      try {
        window.localStorage.setItem(STORAGE_KEY, next);
      } catch {
        /* preference not persisted; not worth surfacing */
      }
      return next;
    });
  }, []);

  const value = useMemo(() => ({ theme, toggle }), [theme, toggle]);
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeContextValue {
  return useContext(ThemeContext);
}

function cssVar(name: string, fallback: string): string {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value === "" ? fallback : value;
}

export interface ChartTheme {
  paper: string;
  plot: string;
  font: string;
  axis: string;
  legend: string;
  grid: string;
  zero: string;
  line: string;
  gold: string;
}

/**
 * Chart colours read from the live theme tokens, so the Plotly figures match the
 * CSS palette in both themes without a second hardcoded table. Call during render
 * (the attribute is already set by the provider).
 */
export function chartTheme(): ChartTheme {
  return {
    paper: cssVar("--bg-chart", "#0b0f19"),
    plot: cssVar("--bg-chart", "#0b0f19"),
    font: cssVar("--text-code", "#cbd5e1"),
    axis: cssVar("--chart-axis", "#64748b"),
    legend: cssVar("--text-secondary", "#a0a0a0"),
    grid: cssVar("--chart-grid", "#1e293b"),
    zero: cssVar("--chart-zero", "#334155"),
    line: cssVar("--border-card", "#2a2a2a"),
    gold: cssVar("--accent-gold", "#d4a843"),
  };
}

/** Chart chrome for lightweight-charts, from the same tokens. */
export function candleTheme() {
  return {
    bg: cssVar("--bg-chart", "#0b0f19"),
    text: cssVar("--text-primary", "#f0f0f0"),
    muted: cssVar("--chart-axis", "#64748b"),
    border: cssVar("--border-card", "#2a2a2a"),
    grid: cssVar("--chart-grid", "#1e293b"),
    label: cssVar("--bg-hover", "#222222"),
    accent: cssVar("--accent-gold", "#d4a843"),
    // Candle semantics, not chrome: up is green, down is red in both themes.
    up: "#4ade80",
    down: "#ef4444",
  };
}