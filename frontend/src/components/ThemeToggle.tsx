import { useEffect, useRef, useState } from "react";
import { THEMES, useTheme } from "@/theme";

/** The existing sun button opens the application's single appearance selector. */
export function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const options = useRef<(HTMLButtonElement | null)[]>([]);

  useEffect(() => {
    if (!open) return;
    options.current[THEMES.findIndex(({ id }) => id === theme)]?.focus();
    const dismiss = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", dismiss);
    return () => document.removeEventListener("pointerdown", dismiss);
  }, [open, theme]);

  return (
    <div className="theme-picker" ref={root}>
    <button
      ref={trigger}
      type="button"
      className="theme-toggle"
      onClick={() => setOpen((value) => !value)}
      title="Choose theme"
      aria-label="Choose theme"
      aria-haspopup="menu"
      aria-expanded={open}
      aria-controls={open ? "theme-menu" : undefined}
    >
      <svg
        width="18"
        height="18"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        aria-hidden="true"
        focusable="false"
      >
        <circle cx="12" cy="12" r="4" />
        <path d="M12 2v2m0 16v2M2 12h2m16 0h2M4.93 4.93l1.42 1.42m11.3 11.3 1.42 1.42M4.93 19.07l1.42-1.42m11.3-11.3 1.42-1.42" />
      </svg>
    </button>
    {open && (
      <div id="theme-menu" className="theme-menu" role="menu" aria-label="Appearance"
        onKeyDown={(event) => {
          const current = options.current.indexOf(document.activeElement as HTMLButtonElement);
          let next: number | null = null;
          if (event.key === "ArrowDown") next = (current + 1) % THEMES.length;
          if (event.key === "ArrowUp") next = (current + THEMES.length - 1) % THEMES.length;
          if (event.key === "Home") next = 0;
          if (event.key === "End") next = THEMES.length - 1;
          if (next !== null) {
            event.preventDefault();
            options.current[next]?.focus();
          }
          if (event.key === "Escape") {
            event.preventDefault();
            setOpen(false);
            trigger.current?.focus();
          }
          if (event.key === "Tab") setOpen(false);
        }}>
        <div className="theme-menu__title">Appearance</div>
        {THEMES.map((entry, index) => (
          <button key={entry.id} type="button" role="menuitemradio"
            ref={(element) => { options.current[index] = element; }}
            className="theme-menu__option"
            aria-label={entry.name}
            aria-checked={theme === entry.id}
            tabIndex={-1}
            onClick={() => {
              setTheme(entry.id);
              setOpen(false);
              trigger.current?.focus();
            }}>
            <span className="theme-preview" data-theme={entry.id} aria-hidden="true">
              <i className="theme-preview__background" />
              <i className="theme-preview__surface" />
              <i className="theme-preview__accent" />
              <i className="theme-preview__text" />
            </span>
            <span>{entry.name}</span>
            {theme === entry.id && <span className="theme-menu__selected" aria-hidden="true">✓</span>}
          </button>
        ))}
      </div>
    )}
    </div>
  );
}
