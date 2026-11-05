import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

export const THEMES = [
  { id: "light", name: "Light", scheme: "light" },
  { id: "dark", name: "Dark", scheme: "dark" },
  { id: "finance-light", name: "Finance Light", scheme: "light" },
  { id: "nord", name: "Nord", scheme: "dark" },
  { id: "simple-dark", name: "Simple Dark", scheme: "dark" },
] as const;

export type Theme = (typeof THEMES)[number]["id"];

export function isTheme(value: unknown): value is Theme {
  return THEMES.some(({ id }) => id === value);
}

export function isLightTheme(theme: Theme): boolean {
  return THEMES.some((entry) => entry.id === theme && entry.scheme === "light");
}

/** localStorage key. The spec's `theme` key, not a namespaced one, so an existing
 *  preference carries over. */
const STORAGE_KEY = "theme";

function readStoredTheme(): Theme {
  try {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    return isTheme(saved) ? saved : "dark";
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

function persistTheme(theme: Theme): void {
  applyTheme(theme);
  try {
    window.localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    /* preference not persisted; not worth surfacing */
  }
}

interface ThemeContextValue {
  theme: Theme;
  toggle: () => void;
  setTheme: (theme: Theme) => void;
}

const ThemeContext = createContext<ThemeContextValue>({
  theme: "dark", toggle: () => {}, setTheme: () => {},
});

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>(readStoredTheme);

  // The attribute is set synchronously in `toggle` as well, so a chart that reads
  // the resolved CSS variables during its own effect sees the new theme rather
  // than waiting for an ancestor effect to run.
  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  const setTheme = useCallback((next: Theme) => {
    persistTheme(next);
    setThemeState(next);
  }, []);

  // Preserve the original Light/Dark toggle contract for existing consumers.
  const toggle = useCallback(() => {
    setThemeState((current) => {
      const next = current === "light" ? "dark" : "light";
      persistTheme(next);
      return next;
    });
  }, []);

  const value = useMemo(() => ({ theme, toggle, setTheme }), [theme, toggle, setTheme]);
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme(): ThemeContextValue {
  return useContext(ThemeContext);
}

export function themeColor(name: string, fallback = ""): string {
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
  series: string[];
  neutral: string;
  heatmapLow: string;
  heatmapHigh: string;
  bandOuterOpacity: number;
  bandInnerOpacity: number;
}

/**
 * Chart colours read from the live theme tokens, so the Plotly figures match the
 * CSS palette in both themes without a second hardcoded table. Call during render
 * (the attribute is already set by the provider).
 */
export function chartTheme(): ChartTheme {
  return {
    paper: themeColor("--bg-chart", "#0b0f19"),
    plot: themeColor("--bg-chart", "#0b0f19"),
    font: themeColor("--text-code", "#cbd5e1"),
    axis: themeColor("--chart-axis", "#64748b"),
    legend: themeColor("--text-secondary", "#a0a0a0"),
    grid: themeColor("--chart-grid", "#1e293b"),
    zero: themeColor("--chart-zero", "#334155"),
    line: themeColor("--border-card", "#2a2a2a"),
    gold: themeColor("--accent-gold", "#d4a843"),
    series: [
      themeColor("--chart-series-1", "#d4a843"),
      themeColor("--chart-series-2", "#8b5cf6"),
      themeColor("--chart-series-3", "#4ade80"),
      themeColor("--chart-series-4", "#60a5fa"),
      themeColor("--chart-series-5", "#f59e0b"),
    ],
    neutral: themeColor("--chart-neutral", "#888888"),
    heatmapLow: themeColor("--chart-heatmap-low", themeColor("--bg-chart", "#0b0f19")),
    heatmapHigh: themeColor("--chart-heatmap-high", themeColor("--accent-gold", "#d4a843")),
    bandOuterOpacity: Number(themeColor("--chart-band-outer-opacity", "0.16")),
    bandInnerOpacity: Number(themeColor("--chart-band-inner-opacity", "0.24")),
  };
}

/** Chart chrome for lightweight-charts, from the same tokens. */
export function candleTheme() {
  return {
    bg: themeColor("--bg-chart", "#0b0f19"),
    text: themeColor("--text-primary", "#f0f0f0"),
    muted: themeColor("--chart-axis", "#64748b"),
    border: themeColor("--border-card", "#2a2a2a"),
    grid: themeColor("--chart-grid", "#1e293b"),
    label: themeColor("--bg-hover", "#222222"),
    accent: themeColor("--accent-gold", "#d4a843"),
    up: themeColor("--color-positive", "#4ade80"),
    down: themeColor("--color-negative", "#ef4444"),
    volume: themeColor("--chart-volume", "rgba(136,136,136,0.5)"),
    volumeUp: themeColor("--chart-volume-up", "rgba(74,222,128,0.42)"),
    volumeDown: themeColor("--chart-volume-down", "rgba(239,68,68,0.42)"),
  };
}
