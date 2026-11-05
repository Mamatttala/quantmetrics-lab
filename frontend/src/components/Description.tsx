import { createContext, useCallback, useContext, useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { HTMLAttributes, ReactNode } from "react";
import { createPortal } from "react-dom";

type Entry = { id: string; content: ReactNode; label: string };
type Group = { anchor: HTMLElement; entries: Map<string, Entry>; namedInput: HTMLInputElement | null };
type Register = (source: HTMLElement, content: ReactNode, anchor?: string, label?: string) => () => void;
interface DescriptionContextValue {
  register: Register;
  groupFor: (id: string) => Group | undefined;
  toggle: (id: string, trigger: HTMLElement) => void;
  active: Group | null;
  popupId: string;
  revision: number;
}
const DescriptionContext = createContext<DescriptionContextValue | null>(null);

function findAnchor(source: HTMLElement, selector?: string): HTMLElement | null {
  const parent = source.parentElement?.parentElement;
  if (!parent) return null;
  if (selector) return parent.closest<HTMLElement>(selector);
  if (source.classList.contains("infotip__description")) {
    return parent.closest<HTMLElement>(".feature__row, .regime__annualisation, .report__row, th") ?? parent;
  }
  if (source.classList.contains("page__lead")) return parent.closest("h1") ?? parent.querySelector("h1");
  if (source.classList.contains("policy__hint")) return parent.closest("fieldset")?.querySelector("legend") ?? parent;
  return parent.closest<HTMLElement>(
    ".policy__option, .vol__radio, .feature__tick, .feature__row, .field, .report__row, .regime__gauge, .row, .policy__aside, .split__dates, .section",
  ) ?? parent;
}

function itemLabel(anchor: HTMLElement): string {
  const label = anchor.matches("h1, legend") ? anchor : anchor.querySelector(
    ".policy__label, .feature__label, .report__key, .section__title, label, strong",
  ) ?? anchor;
  const copy = label.cloneNode(true) as HTMLElement;
  copy.querySelectorAll(".description-control").forEach(node => node.remove());
  return `About ${copy.textContent?.replace(/\s+/g, " ").trim() || "this item"}`;
}

/** One click-open description popup, shared across all pages and scrolling panes. */
export function DescriptionProvider({ children }: { children: ReactNode }) {
  const groups = useRef(new Map<HTMLElement, Group>());
  const byId = useRef(new Map<string, Group>());
  const [active, setActive] = useState<Group | null>(null);
  const activeRef = useRef<Group | null>(null);
  const trigger = useRef<HTMLElement | null>(null);
  const [revision, setRevision] = useState(0);
  const popupId = useId();
  const popup = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState<{ left: number; top: number } | null>(null);

  const close = useCallback(() => {
    activeRef.current = null;
    setActive(null);
  }, []);

  const register = useCallback<Register>((source, content, selector, label) => {
    const anchor = findAnchor(source, selector);
    if (!anchor) return () => {};
    let group = groups.current.get(anchor);
    if (!group) {
      const input = anchor.matches("label") ? anchor.querySelector<HTMLInputElement>("input") : null;
      group = {
        anchor, entries: new Map<string, Entry>(),
        namedInput: input && !input.hasAttribute("aria-label") && !input.hasAttribute("aria-labelledby") ? input : null,
      };
    }
    // An information control inside a label must not become part of the input's name.
    group.namedInput?.setAttribute("aria-label", itemLabel(anchor).slice("About ".length));
    group.entries.set(source.id, { id: source.id, content, label: label ?? itemLabel(anchor) });
    groups.current.set(anchor, group);
    byId.current.set(source.id, group);
    setRevision(value => value + 1);

    return () => {
      group.entries.delete(source.id);
      byId.current.delete(source.id);
      // A description may be replaced in the same React commit. Keep its group so
      // the open popup updates instead of closing while the user edits a setting.
      queueMicrotask(() => {
        if (!group.entries.size) {
          groups.current.delete(anchor);
          group.namedInput?.removeAttribute("aria-label");
          if (activeRef.current === group) close();
        }
        setRevision(value => value + 1);
      });
    };
  }, [close]);

  const groupFor = useCallback((id: string) => byId.current.get(id), []);
  const toggle = useCallback((id: string, button: HTMLElement) => {
    const group = byId.current.get(id);
    if (!group) return;
    if (activeRef.current === group) { close(); return; }
    trigger.current = button;
    activeRef.current = group;
    setPosition(null);
    setActive(group);
  }, [close]);

  useEffect(() => {
    const inside = (target: EventTarget | null) => target instanceof Node
      && (popup.current?.contains(target) || trigger.current?.contains(target));
    const onOutside = (event: Event) => { if (!inside(event.target)) close(); };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape" && activeRef.current) {
        close();
        trigger.current?.focus();
      }
    };
    const onScroll = (event: Event) => {
      if (!(event.target instanceof Node && popup.current?.contains(event.target))) close();
    };
    document.addEventListener("pointerdown", onOutside, true);
    document.addEventListener("click", onOutside, true);
    document.addEventListener("keydown", onKey);
    window.addEventListener("scroll", onScroll, true);
    window.addEventListener("resize", close);
    window.addEventListener("blur", close);
    return () => {
      document.removeEventListener("pointerdown", onOutside, true);
      document.removeEventListener("click", onOutside, true);
      document.removeEventListener("keydown", onKey);
      window.removeEventListener("scroll", onScroll, true);
      window.removeEventListener("resize", close);
      window.removeEventListener("blur", close);
    };
  }, [close]);

  useLayoutEffect(() => {
    if (!active || !popup.current || !trigger.current) return;
    const box = trigger.current.getBoundingClientRect();
    const panel = popup.current.getBoundingClientRect();
    const margin = 12;
    const below = box.bottom + 8;
    const top = below + panel.height <= window.innerHeight - margin ? below : box.top - panel.height - 8;
    setPosition({
      left: Math.max(margin, Math.min(box.left, window.innerWidth - panel.width - margin)),
      top: Math.max(margin, Math.min(top, window.innerHeight - panel.height - margin)),
    });
  }, [active, revision]);

  const context = useMemo(() => ({ register, groupFor, toggle, active, popupId, revision }),
    [register, groupFor, toggle, active, popupId, revision]);
  const entries = active ? [...active.entries.values()] : [];
  return <DescriptionContext.Provider value={context}>
    {children}
    {active && entries.length > 0 && createPortal(
      <div
        ref={popup}
        id={popupId}
        role="region"
        aria-label={entries[0]?.label ?? "Description"}
        className="description-popup"
        style={{ left: position?.left ?? 0, top: position?.top ?? 0, visibility: position ? "visible" : "hidden" }}
      >
        {entries.map(entry => <div className="description-popup__part" key={entry.id}>{entry.content}</div>)}
      </div>, document.body,
    )}
  </DescriptionContext.Provider>;
}

interface DescriptionProps extends HTMLAttributes<HTMLSpanElement> {
  anchor?: string;
  label?: string;
}

/** The icon opens help without selecting its surrounding label or submitting a form. */
export function Description({ children, className = "", anchor, label, ...props }: DescriptionProps) {
  const id = useId();
  const source = useRef<HTMLSpanElement>(null);
  const context = useContext(DescriptionContext);
  const register = context?.register;
  useLayoutEffect(() => {
    if (source.current && register) return register(source.current, children, anchor, label);
  }, [register, children, anchor, label]);
  const group = context?.groupFor(id);
  const first = group?.entries.values().next().value;
  const primary = first?.id === id;
  const open = !!group && context?.active === group;

  return <span className={`description-control${primary ? "" : " description-control--hidden"}`}>
    {primary && <span
      role="button"
      tabIndex={0}
      className="description-toggle"
      aria-label={first.label}
      aria-expanded={open}
      aria-controls={open ? context?.popupId : undefined}
      onClick={event => {
        event.preventDefault();
        event.stopPropagation();
        context?.toggle(id, event.currentTarget);
      }}
      onKeyDown={event => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        event.stopPropagation();
        context?.toggle(id, event.currentTarget);
      }}
    ><span aria-hidden="true">i</span></span>}
    <span {...props} ref={source} id={id} hidden className={`description-source ${className}`}>{children}</span>
  </span>;
}
