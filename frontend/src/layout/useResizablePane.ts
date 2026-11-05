import { useCallback, useEffect, useRef, useState } from "react";
import type { KeyboardEvent, PointerEvent } from "react";

const MIN_WIDTH = 200;
const MAX_WIDTH = 500;
const clampWidth = (width: number) => Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, width));

/** Shared drag limits, persistence and cleanup for the two left panes. */
export function useResizablePane(key: string, defaultWidth: number, disabled = false) {
  const [width, setWidth] = useState(() => {
    try {
      const stored = window.localStorage.getItem(key);
      const parsed = stored === null ? NaN : Number(stored);
      return Number.isFinite(parsed) ? clampWidth(parsed) : defaultWidth;
    } catch {
      return defaultWidth;
    }
  });
  const stopResize = useRef<(() => void) | null>(null);

  useEffect(() => {
    try {
      window.localStorage.setItem(key, String(width));
    } catch {
      // A blocked preference store must not prevent resizing.
    }
  }, [key, width]);

  useEffect(() => {
    if (disabled) stopResize.current?.();
    return () => stopResize.current?.();
  }, [disabled]);

  const onPointerDown = useCallback((event: PointerEvent<HTMLDivElement>) => {
    if (disabled || event.button !== 0) return;
    event.preventDefault();
    stopResize.current?.();
    const startX = event.clientX;
    // Use the rendered width so a responsive CSS cap never creates a dead zone.
    const startWidth = event.currentTarget.parentElement?.getBoundingClientRect().width || width;
    const pointerId = event.pointerId;
    const onMove = (move: globalThis.PointerEvent) => {
      if (move.pointerId === pointerId) setWidth(clampWidth(startWidth + move.clientX - startX));
    };
    const stop = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onEnd);
      window.removeEventListener("pointercancel", onEnd);
      window.removeEventListener("blur", stop);
      document.body.classList.remove("pane-resizing");
      stopResize.current = null;
    };
    const onEnd = (end: globalThis.PointerEvent) => {
      if (end.pointerId === pointerId) stop();
    };
    stopResize.current = stop;
    document.body.classList.add("pane-resizing");
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onEnd);
    window.addEventListener("pointercancel", onEnd);
    window.addEventListener("blur", stop);
  }, [disabled, width]);

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (disabled) return;
    const step = event.shiftKey ? 50 : 10;
    const next = { ArrowLeft: width - step, ArrowRight: width + step, Home: MIN_WIDTH, End: MAX_WIDTH }[event.key];
    if (next === undefined) return;
    event.preventDefault();
    setWidth(clampWidth(next));
  };

  return {
    width,
    resizeHandleProps: {
      role: "separator",
      "aria-orientation": "vertical" as const,
      "aria-valuemin": MIN_WIDTH,
      "aria-valuemax": MAX_WIDTH,
      "aria-valuenow": width,
      tabIndex: 0,
      onPointerDown,
      onKeyDown,
    },
  };
}
