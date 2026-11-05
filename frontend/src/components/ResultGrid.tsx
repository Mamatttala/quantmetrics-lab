import { useMemo } from "react";
import { AllCommunityModule, ModuleRegistry } from "ag-grid-community";
import type { ColDef } from "ag-grid-community";
import { AgGridReact } from "ag-grid-react";
import { isLightTheme, useTheme } from "@/theme";

ModuleRegistry.registerModules([AllCommunityModule]);

/** Read-only diagnostics share DataGrid's installed grid and theme, without persistence. */
export function ResultGrid<T extends object>({ caption, rows, columns }: {
  caption: string; rows: T[]; columns: [keyof T, string][];
}) {
  const { theme } = useTheme();
  const definitions = useMemo<ColDef<T>[]>(() => columns.map(([key, label]) => ({
    colId: String(key), headerName: label, valueGetter: ({ data }) => data?.[key],
    valueFormatter: ({ value }: { value: unknown }) => value === null || value === undefined ? "—"
      : Array.isArray(value) ? value.join(", ") : typeof value === "number" && !Number.isInteger(value)
        ? value.toPrecision(5) : String(value),
  })), [columns]);
  return <section aria-label={caption}><h3>{caption}</h3>
    <div className={`ag-theme-quartz${isLightTheme(theme) ? "" : "-dark"}`} style={{ height: 390, width: "100%" }}>
      <AgGridReact<T> theme="legacy" rowData={rows} columnDefs={definitions}
        defaultColDef={{ sortable: true, filter: true, resizable: true, minWidth: 130, flex: 1 }}
        pagination paginationPageSize={50} paginationPageSizeSelector={[25, 50, 100]}
        animateRows={false} />
    </div>
  </section>;
}
