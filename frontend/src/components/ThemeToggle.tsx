import { useTheme } from "@/theme";

/**
 * Sun/moon theme switch. It lives in the Data Explorer header, which is the app's
 * top-right control strip. The icon shows the theme the click will move to:
 * dark shows a sun, light shows a moon.
 */
export function ThemeToggle() {
  const { theme, toggle } = useTheme();
  const isLight = theme === "light";
  const label = isLight ? "Switch to dark theme" : "Switch to light theme";

  return (
    <button
      type="button"
      className="theme-toggle"
      onClick={toggle}
      title={label}
      aria-label={label}
      aria-pressed={isLight}
    >
      <span aria-hidden="true">{isLight ? "\u{1F319}" : "\u2600\uFE0F"}</span>
    </button>
  );
}