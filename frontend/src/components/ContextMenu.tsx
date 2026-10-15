import { useEffect, useRef } from "react";

export interface MenuItem {
  label: string;
  onSelect: () => void;
  danger?: boolean;
  disabled?: boolean;
  separatorBefore?: boolean;
}

interface ContextMenuProps {
  x: number;
  y: number;
  items: MenuItem[];
  onClose: () => void;
}

/**
 * A context menu for ag-Grid Community.
 *
 * ag-Grid's built-in context menu is an Enterprise feature, so this renders its own
 * absolutely-positioned list. It closes on outside click, on Escape, and on scroll —
 * a menu that stays put while the grid scrolls under it would act on the wrong row.
 */
export function ContextMenu({ x, y, items, onClose }: ContextMenuProps) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onPointerDown = (event: MouseEvent) => {
      if (ref.current !== null && !ref.current.contains(event.target as Node)) {
        onClose();
      }
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    window.addEventListener("scroll", onClose, true);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("scroll", onClose, true);
    };
  }, [onClose]);

  // Keep the menu inside the viewport. A menu opened near the right or bottom edge
  // would otherwise render partly offscreen, with its items unreachable.
  useEffect(() => {
    const node = ref.current;
    if (node === null) return;
    const rect = node.getBoundingClientRect();
    const overflowX = rect.right - window.innerWidth;
    const overflowY = rect.bottom - window.innerHeight;
    if (overflowX > 0) node.style.left = `${Math.max(0, x - rect.width)}px`;
    if (overflowY > 0) node.style.top = `${Math.max(0, y - rect.height)}px`;
  }, [x, y]);

  return (
    <div
      ref={ref}
      className="ctxmenu"
      style={{ left: x, top: y }}
      role="menu"
      onContextMenu={(e) => e.preventDefault()}
    >
      {items.map((item, index) => (
        <div key={`${item.label}-${index}`}>
          {item.separatorBefore === true && <div className="ctxmenu__sep" />}
          <button
            type="button"
            role="menuitem"
            className={item.danger === true ? "ctxmenu__item ctxmenu__item--danger" : "ctxmenu__item"}
            disabled={item.disabled === true}
            onClick={() => {
              item.onSelect();
              onClose();
            }}
          >
            {item.label}
          </button>
        </div>
      ))}
    </div>
  );
}
