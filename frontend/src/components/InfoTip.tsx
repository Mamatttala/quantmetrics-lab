import { useEffect, useId, useRef, useState } from "react";

interface InfoTipProps {
  /** Accessible name. Read aloud in place of the icon, so it must say what is
   *  described — "Formula for Simple moving average", not "info". */
  label: string;
  /** The catalogue's own statement of the formula, shown verbatim and first. */
  formula: string;
  detail: string;
  caution?: string | null;
  /** Source columns and resolved window, appended as a short factual footer. */
  meta?: string[];
}

/**
 * The purple ℹ️ beside a feature.
 *
 * Deliberately a real `<button>` opening a real panel rather than a `title` attribute:
 * a native tooltip never appears on touch, is not reachable by keyboard, cannot be
 * scrolled when it is long, and vanishes the moment the pointer moves — which is a poor
 * way to deliver the one piece of text that explains what a column actually computes.
 *
 * The formula shown here is the backend catalogue's string, passed through untouched.
 * Restating it in the frontend would create a second description of the same
 * mathematics, and the two would drift.
 *
 * An entry carrying a `caution` marks its icon rather than only its panel. The panel is
 * the only place the caveat is written, and an icon that looks identical whether or not
 * there is one underneath it is a caveat nobody opens.
 */
export function InfoTip({ label, formula, detail, caution, meta }: InfoTipProps) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLSpanElement>(null);
  const panelId = useId();

  useEffect(() => {
    if (!open) return;
    const close = () => setOpen(false);
    const onPointerDown = (event: MouseEvent) => {
      if (root.current !== null && !root.current.contains(event.target as Node)) {
        close();
      }
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") close();
    };
    // Capture phase for scroll: the panel is absolutely positioned, so a scroll in any
    // ancestor would slide the anchor out from under it while it stayed on screen.
    // One named handler for all three, because `removeEventListener` matches on
    // identity — an inline arrow in the cleanup removes nothing and leaks a listener
    // per open.
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    window.addEventListener("scroll", close, true);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("scroll", close, true);
    };
  }, [open]);

  return (
    <span className="infotip" ref={root}>
      <button
        type="button"
        className={
          caution === undefined || caution === null
            ? "infotip__button"
            : "infotip__button infotip__button--caution"
        }
        aria-label={label}
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => setOpen((value) => !value)}
      >
        {/* The glyph is decorative; the button carries the name. */}
        <span aria-hidden="true">i</span>
      </button>

      {open && (
        <span className="infotip__panel" id={panelId} role="note">
          <span className="infotip__formula mono">{formula}</span>
          <span className="infotip__detail">{detail}</span>
          {caution !== undefined && caution !== null && (
            <span className="infotip__caution">{caution}</span>
          )}
          {meta !== undefined && meta.length > 0 && (
            <span className="infotip__meta muted">{meta.join(" · ")}</span>
          )}
        </span>
      )}
    </span>
  );
}
