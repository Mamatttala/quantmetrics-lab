import { useState } from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { DataExplorer } from "./DataExplorer";
import { Sidebar } from "./Sidebar";

function Panes() {
  const [collapsed, setCollapsed] = useState(false);
  const [hidden, setHidden] = useState(false);
  return <MemoryRouter>
    <Sidebar collapsed={collapsed} onToggleCollapsed={() => setCollapsed(!collapsed)} />
    {hidden
      ? <button onClick={() => setHidden(false)}>Show explorer</button>
      : <DataExplorer project={null} projectControls={null} onHide={() => setHidden(true)} />}
  </MemoryRouter>;
}

beforeEach(() => {
  window.localStorage.clear();
  // jsdom does not yet implement PointerEvent.
  vi.stubGlobal("PointerEvent", class extends MouseEvent {
    pointerId: number;
    constructor(type: string, init: PointerEventInit = {}) {
      super(type, init);
      this.pointerId = init.pointerId ?? 1;
    }
  });
});

afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe.each([
  { name: "Sidebar Menu", key: "qca.ui.sidebarWidth", initial: 220 },
  { name: "Data Explorer", key: "qca.ui.explorerWidth", initial: 280 },
])("$name resizing", ({ name, key, initial }) => {
  it("follows the pointer, clamps the width, and restores it after remount", () => {
    const view = render(<Panes />);
    const handle = screen.getByRole("separator", { name: `Resize ${name}` });
    const pane = handle.parentElement!;
    fireEvent.pointerDown(handle, { clientX: 200, button: 0 });
    fireEvent.pointerMove(window, { clientX: 260 });
    expect(pane.style.width).toBe(`${initial + 60}px`);
    fireEvent.pointerMove(window, { clientX: 1500 });
    expect(pane.style.width).toBe("500px");
    fireEvent.pointerMove(window, { clientX: -1500 });
    expect(pane.style.width).toBe("200px");
    fireEvent.pointerMove(window, { clientX: 260 });
    fireEvent.pointerUp(window);
    fireEvent.pointerMove(window, { clientX: 300 });
    expect(pane.style.width).toBe(`${initial + 60}px`);
    expect(window.localStorage.getItem(key)).toBe(String(initial + 60));
    expect(document.body.classList.contains("pane-resizing")).toBe(false);
    view.unmount();
    render(<Panes />);
    expect(screen.getByRole("separator", { name: `Resize ${name}` }).parentElement!.style.width)
      .toBe(`${initial + 60}px`);
  });

  it("starts at the rendered width and cleans up a cancelled or unmounted drag", () => {
    const view = render(<Panes />);
    const handle = screen.getByRole("separator", { name: `Resize ${name}` });
    const pane = handle.parentElement!;
    vi.spyOn(pane, "getBoundingClientRect").mockReturnValue({ width: 240 } as DOMRect);
    fireEvent.pointerDown(handle, { clientX: 240, button: 0 });
    fireEvent.pointerMove(window, { clientX: 230 });
    expect(pane.style.width).toBe("230px");
    fireEvent.pointerCancel(window);
    fireEvent.pointerMove(window, { clientX: 400 });
    expect(pane.style.width).toBe("230px");
    fireEvent.pointerDown(handle, { clientX: 240, button: 0 });
    view.unmount();
    expect(document.body.classList.contains("pane-resizing")).toBe(false);
    fireEvent.pointerMove(window, { clientX: 400 });
    expect(window.localStorage.getItem(key)).toBe("230");
  });

  it("supports keyboard resizing and safely handles invalid stored widths", () => {
    window.localStorage.setItem(key, "invalid");
    render(<Panes />);
    const handle = screen.getByRole("separator", { name: `Resize ${name}` });
    expect(handle.getAttribute("aria-valuenow")).toBe(String(initial));
    fireEvent.keyDown(handle, { key: "ArrowRight" });
    expect(handle.parentElement!.style.width).toBe(`${initial + 10}px`);
    fireEvent.keyDown(handle, { key: "End" });
    expect(handle.parentElement!.style.width).toBe("500px");
    fireEvent.keyDown(handle, { key: "Home" });
    expect(handle.parentElement!.style.width).toBe("200px");
  });
});

it("restores pane widths after collapsing the sidebar and hiding the explorer", () => {
  render(<Panes />);
  fireEvent.keyDown(screen.getByRole("separator", { name: "Resize Sidebar Menu" }), { key: "ArrowRight" });
  fireEvent.keyDown(screen.getByRole("separator", { name: "Resize Data Explorer" }), { key: "ArrowRight" });
  fireEvent.click(screen.getByRole("button", { name: "Collapse sidebar" }));
  fireEvent.click(screen.getByRole("button", { name: "Hide" }));
  expect(screen.queryByRole("separator")).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "Expand sidebar" }));
  fireEvent.click(screen.getByRole("button", { name: "Show explorer" }));
  expect(screen.getByRole("separator", { name: "Resize Sidebar Menu" }).parentElement!.style.width).toBe("230px");
  expect(screen.getByRole("separator", { name: "Resize Data Explorer" }).parentElement!.style.width).toBe("290px");
});
