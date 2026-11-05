import { StrictMode, useState } from "react";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { Description, DescriptionProvider } from "./Description";
import { InfoTip } from "./InfoTip";
import { PageHeader } from "./PageHeader";
import { PolicyGroup } from "./PolicyGroup";

function Fixture() {
  const [value, setValue] = useState("keep");
  return <StrictMode><DescriptionProvider>
    <PageHeader title="Cleaning" lead="Prepare stored bars." />
    <PolicyGroup name="fill" legend="Missing values" hint="Choose how missing values are handled."
      options={[
        { value: "keep", label: "Keep", detail: "Keep the original values." },
        { value: "fill", label: "Fill", detail: "Use the preceding value.", caution: "Check long gaps." },
      ]}
      value={value} onChange={setValue} />
    <div className="field">
      <label htmlFor="window">Window</label><input id="window" defaultValue="5" />
      <Description className="field__note">Number of bars in the window.</Description>
    </div>
    <fieldset disabled><div className="feature__row">
      <label><input type="checkbox" disabled />Unavailable feature</label>
      <InfoTip label="Formula for feature" formula="x + y" detail="Adds two inputs." caution="Needs both inputs." />
    </div></fieldset>
    <p role="status">Loading datasets…</p>
    <p role="alert">A dataset is required.</p>
  </DescriptionProvider></StrictMode>;
}

describe("clickable descriptions", () => {
  it("opens only on clicking i, combines details and cautions, and stays open after leaving", () => {
    render(<Fixture />);
    const info = screen.getByRole("button", { name: "About Fill" });
    expect(screen.getByText("Prepare stored bars.").closest("[hidden]")).toBeTruthy();
    fireEvent.pointerOver(info);
    fireEvent.focusIn(info);
    expect(screen.queryByRole("region")).toBeNull();
    fireEvent.click(info);
    const popup = screen.getByRole("region", { name: "About Fill" });
    expect(popup.textContent).toContain("Use the preceding value.");
    expect(popup.textContent).toContain("Check long gaps.");
    expect(info.getAttribute("aria-expanded")).toBe("true");
    expect(info.getAttribute("aria-controls")).toBe(popup.id);
    fireEvent.pointerOut(info, { relatedTarget: document.body });
    expect(screen.getByRole("region")).toBeTruthy();
    fireEvent.click(popup);
    expect(screen.getByRole("region")).toBeTruthy();
    fireEvent.click(document.body);
    expect(screen.queryByRole("region")).toBeNull();
    expect(info.getAttribute("aria-expanded")).toBe("false");
  });

  it("does not select an option when its information icon is clicked", () => {
    render(<Fixture />);
    fireEvent.click(screen.getByRole("button", { name: "About Fill" }));
    const option = screen.getByRole("radio", { name: "Fill" }) as HTMLInputElement;
    expect(option.checked).toBe(false);
    expect((screen.getByRole("radio", { name: "Keep" }) as HTMLInputElement).checked).toBe(true);
    fireEvent.click(option);
    expect(option.checked).toBe(true);
    expect(screen.queryByRole("region")).toBeNull();
    expect(screen.getByRole("status").textContent).toBe("Loading datasets…");
    expect(screen.getByRole("alert").textContent).toBe("A dataset is required.");
  });

  it("switches to another description and toggles the current icon closed", () => {
    render(<Fixture />);
    fireEvent.click(screen.getByRole("button", { name: "About Fill" }));
    const next = screen.getByRole("button", { name: "About Window" });
    fireEvent.click(next);
    expect(screen.getAllByRole("region")).toHaveLength(1);
    expect(screen.getByRole("region").textContent).toBe("Number of bars in the window.");
    fireEvent.click(next);
    expect(screen.queryByRole("region")).toBeNull();
  });

  it("supports Enter, Space and Escape, including help within disabled fieldsets", () => {
    render(<Fixture />);
    const info = screen.getByRole("button", { name: "Formula for feature" });
    fireEvent.keyDown(info, { key: "Enter" });
    expect(screen.getByRole("region").textContent).toContain("x + yAdds two inputs.Needs both inputs.");
    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByRole("region")).toBeNull();
    expect(document.activeElement).toBe(info);
    fireEvent.keyDown(info, { key: " " });
    expect(screen.getByRole("region")).toBeTruthy();
    fireEvent.pointerDown(document.body);
    expect(screen.queryByRole("region")).toBeNull();
  });

  it("updates open descriptions and removes them when their page unmounts", async () => {
    function Dynamic({ detail, visible }: { detail: string; visible: boolean }) {
      return <DescriptionProvider>{visible && <div className="field">
        <label htmlFor="source">Source</label><input id="source" />
        <Description>{detail}</Description>
      </div>}</DescriptionProvider>;
    }
    const view = render(<Dynamic detail="First dataset" visible />);
    fireEvent.click(screen.getByRole("button", { name: "About Source" }));
    view.rerender(<Dynamic detail="Second dataset" visible />);
    await waitFor(() => expect(screen.getByRole("region").textContent).toBe("Second dataset"));
    await act(async () => view.rerender(<Dynamic detail="Second dataset" visible={false} />));
    expect(screen.queryByRole("region")).toBeNull();
  });
});
