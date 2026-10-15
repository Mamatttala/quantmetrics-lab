import {
  AgGridReact,
} from "ag-grid-react";
import {
  AllCommunityModule,
  ModuleRegistry,
} from "ag-grid-community";
import type {
  CellClassParams,
  CellContextMenuEvent,
  CellValueChangedEvent,
  ColDef,
  ColumnHeaderContextMenuEvent,
  SortChangedEvent,
} from "ag-grid-community";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type MouseEvent as ReactMouseEvent,
} from "react";

import "ag-grid-community/styles/ag-grid.css";
import "ag-grid-community/styles/ag-theme-quartz.css";

// ag-Grid v33 stopped registering its own modules. A grid created without this renders an
// empty container and logs "error #272 — No AG Grid modules are registered!" — and the
// failure is quiet in the way that matters most: the toolbar above the grid is ordinary
// React and goes on reporting "0–33 of 33", so the page looks like a working grid that was
// handed no data rather than like a grid that never started. Registration is global and
// idempotent, so it is done once at module scope beside the stylesheets it belongs with,
// which is also before any `AgGridReact` in the tree can mount.
ModuleRegistry.registerModules([AllCommunityModule]);

import { ApiError, api } from "@/api/client";
import type { PageResponse } from "@/api/types";

import { ContextMenu, type MenuItem } from "./ContextMenu";
import { useTheme } from "@/theme";
import {
  coerce,
  parse,
  readFromClipboard,
  rectContains,
  serialise,
  writeToClipboard,
  type CellRect,
} from "./clipboard";

export interface GridColumn {
  name: string;
  /** The parent's name for this column, or `null` if it was added in this session.
   *  This is what lets column operations be computed as a diff against the parent
   *  schema instead of replayed as a log of clicks. */
  origin: string | null;
  dtype: string;
  hidden: boolean;
  /** Added in this editing session, so it exists only in derived data. */
  isNew: boolean;
}

export interface PendingEdit {
  row: number;
  column: string;
  value: unknown;
}

interface DataGridProps {
  datasetId: string;
  columns: GridColumn[];
  /** Keyed `${row}|${column}` so lookups during render are O(1). */
  pendingEdits: Record<string, unknown>;
  onEdit: (edit: PendingEdit) => void;
  onBulkEdit: (edits: PendingEdit[]) => void;
  onRenameColumn: (from: string, to: string) => void;
  onDropColumn: (column: string) => void;
  onHideColumn: (column: string) => void;
  onAddColumn: () => void;
  onReorderColumns: (order: string[]) => void;
  onNotice?: (message: string, kind: "ok" | "error") => void;
}

const PAGE_SIZES = [50, 100, 250, 500];

/** Arrow/Parquet numeric type prefixes. Used for right-alignment. */
const NUMERIC_DTYPE = /^(double|float|int|uint|decimal)/;

/** Arrow/Parquet timestamp type prefix, e.g. `timestamp[ns, tz=UTC]`. */
const TIMESTAMP_DTYPE = /^timestamp/;

type Row = Record<string, unknown>;

/**
 * What `coerce` should treat a pasted cell as, taken from the column's declared Arrow
 * type rather than from the value currently sitting in that one row.
 *
 * Reading the sample off the target cell looks equivalent and is not: a null there —
 * which on a partly-populated column is the common case — leaves a pasted `1.5` as the
 * string `"1.5"`, and a column that silently turns to `object` dtype breaks every
 * arithmetic feature built on it. The declared type is always the better answer when
 * it says something; `null` means "no opinion, use the cell".
 */
function declaredSample(dtype: string): unknown {
  if (NUMERIC_DTYPE.test(dtype)) return 0;
  // A Date instance routes `coerce` to its timestamp branch, which normalises to ISO.
  if (TIMESTAMP_DTYPE.test(dtype)) return new Date(0);
  return null;
}

/**
 * The editable data grid.
 *
 * **Pagination.** Each page is fetched from the server and only that page is held in
 * the grid. The spec requires that the browser never receives a whole Parquet file,
 * and this satisfies it directly: a 5M-row dataset costs the same browser memory as a
 * 50-row one. ag-Grid still virtualises the DOM within the page, so 500 rows is not
 * 500 mounted cells. (ag-Grid's `infinite` row model would give smooth cross-page
 * scrolling and is a drop-in change here if that becomes worth the extra parts.)
 *
 * **Sorting is server-side.** Sorting only the loaded page would present one page's
 * ordering as if it were the dataset's — wrong in a way that looks right.
 *
 * **Selection.** ag-Grid Community has no cell-range selection (Enterprise only), so
 * the range is tracked here as an anchor plus a focus cell, set by click and
 * shift-click. That range drives copy/cut/paste, which are likewise hand-rolled.
 */
export function DataGrid({
  datasetId,
  columns,
  pendingEdits,
  onEdit,
  onBulkEdit,
  onRenameColumn,
  onDropColumn,
  onHideColumn,
  onAddColumn,
  onReorderColumns,
  onNotice,
}: DataGridProps) {
  const gridRef = useRef<AgGridReact<Row>>(null);
  // ag-Grid's `ColumnHeaderContextMenuEvent` carries no DOM event, so pointer
  // coordinates cannot be read from it. They are recorded during the capture phase on
  // the wrapper below, which is reached before ag-Grid's own bubble-phase listener,
  // giving both context menus one reliable source of position.
  const pointer = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const { theme } = useTheme();
  const [rows, setRows] = useState<Row[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [limit, setLimit] = useState(100);
  const [sort, setSort] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [menu, setMenu] = useState<{ x: number; y: number; items: MenuItem[] } | null>(null);
  const [anchor, setAnchor] = useState<{ row: number; column: string } | null>(null);
  const [focus, setFocus] = useState<{ row: number; column: string } | null>(null);

  // Kept in refs as well as state because `load` must not be rebuilt every time the
  // selection moves — that would refetch the page on every click.
  const sortRef = useRef<string | null>(null);
  const limitRef = useRef(100);

  const load = useCallback(
    async (nextOffset: number, nextLimit: number, nextSort: string | null) => {
      setLoading(true);
      try {
        const page: PageResponse = await api.page(
          datasetId,
          nextOffset,
          nextLimit,
          nextSort ?? undefined,
        );
        setRows(page.rows);
        setTotal(page.total);
        setOffset(page.offset);
        setLimit(page.limit);
        limitRef.current = page.limit;
      } catch (cause) {
        onNotice?.(cause instanceof ApiError ? cause.message : String(cause), "error");
      } finally {
        setLoading(false);
      }
    },
    [datasetId, onNotice],
  );

  // Reload on dataset change and reset paging — carrying an offset across datasets
  // would show an arbitrary window of the new one.
  useEffect(() => {
    sortRef.current = null;
    setSort(null);
    setOffset(0);
    setAnchor(null);
    setFocus(null);
    void load(0, limitRef.current, null);
  }, [datasetId, load]);

  const visibleColumns = useMemo(() => columns.filter((c) => !c.hidden), [columns]);

  // Declared Arrow type per column, for `pasteAtSelection`'s coercion.
  const dtypes = useMemo(
    () => new Map<string, string>(columns.map((c) => [c.name, c.dtype])),
    [columns],
  );

  // Row-addressed writes are only sound while the grid shows the dataset in its own
  // row order. `/page` sorts on the server, and the backend resolves an edit's row
  // number against the file's own order — so with a sort applied, the row on screen
  // is not the row that would be written, and an edit or paste would silently land
  // elsewhere. Editing and the mutating clipboard commands are therefore closed while
  // a sort is active. Copying is unaffected: it only reads.
  const editable = sort === null;

  const columnDefs = useMemo<ColDef<Row>[]>(
    () =>
      visibleColumns.map((column) => {
        // Built up imperatively rather than returning one object literal: with
        // `exactOptionalPropertyTypes` a property that is *present and undefined* is
        // not the same as an absent one, and neither `type` nor `cellClass` is
        // declared to accept `undefined`.
        const def: ColDef<Row> = {
          field: column.name,
          headerName: column.name,
          editable,
          resizable: true,
          sortable: true,
          minWidth: 110,
          headerTooltip: `${column.name} · ${column.dtype}${
            column.isNew ? " · added in this session, not yet saved" : ""
          }`,
        };
        if (NUMERIC_DTYPE.test(column.dtype)) def.type = "numericColumn";
        if (column.isNew) def.cellClass = "cell--new";
        return def;
      }),
    [visibleColumns, editable],
  );

  // Pending edits are overlaid at render time rather than mutating `rows`, so the
  // grid always shows exactly what will be saved and the fetched page stays pristine.
  const displayRows = useMemo(
    () =>
      rows.map((row, index) => {
        const absolute = offset + index;
        let patched: Row | null = null;
        for (const column of columns) {
          const key = `${absolute}|${column.name}`;
          if (key in pendingEdits) {
            patched ??= { ...row };
            patched[column.name] = pendingEdits[key];
          }
        }
        // `__index` addresses the row within the page for ag-Grid's row id.
        if (patched === null) return { ...row, __index: index };
        patched.__index = index;
        return patched;
      }),
    [rows, offset, columns, pendingEdits],
  );

  const getRowId = useCallback(
    (params: { data: Row }) => String(offset + Number(params.data.__index ?? 0)),
    [offset],
  );

  // ---------------------------------------------------------------- selection

  const selection: CellRect | null = useMemo(() => {
    if (anchor === null || focus === null) return null;
    const names = visibleColumns.map((c) => c.name);
    const a = names.indexOf(anchor.column);
    const b = names.indexOf(focus.column);
    if (a === -1 || b === -1) return null;
    return {
      startRow: Math.min(anchor.row, focus.row),
      endRow: Math.max(anchor.row, focus.row),
      columns: names.slice(Math.min(a, b), Math.max(a, b) + 1),
    };
  }, [anchor, focus, visibleColumns]);

  const cellClassRules = useMemo(
    () => ({
      "cell--selected": (params: CellClassParams<Row>) =>
        selection !== null &&
        params.rowIndex !== null &&
        params.column !== null &&
        rectContains(selection, offset + params.rowIndex, params.column.getColId()),
      "cell--dirty": (params: CellClassParams<Row>) =>
        params.rowIndex !== null &&
        params.column !== null &&
        `${offset + params.rowIndex}|${params.column.getColId()}` in pendingEdits,
    }),
    [selection, offset, pendingEdits],
  );

  // --------------------------------------------------------------- clipboard

  const copySelection = useCallback(
    async (cut: boolean) => {
      if (selection === null) {
        onNotice?.("Select cells first — click a cell, then shift-click to extend.", "error");
        return;
      }
      const names = selection.columns;
      const block: unknown[][] = [];
      for (let row = selection.startRow; row <= selection.endRow; row += 1) {
        const index = row - offset;
        block.push(
          names.map((name) => {
            const key = `${row}|${name}`;
            if (key in pendingEdits) return pendingEdits[key];
            return rows[index]?.[name] ?? null;
          }),
        );
      }
      const copied = await writeToClipboard(serialise(block));
      if (!copied) {
        onNotice?.("The browser blocked clipboard access.", "error");
        return;
      }
      if (cut) {
        const cleared: PendingEdit[] = [];
        for (let row = selection.startRow; row <= selection.endRow; row += 1) {
          for (const name of names) cleared.push({ row, column: name, value: null });
        }
        onBulkEdit(cleared);
        onNotice?.(`Cut ${cleared.length} cell(s) to the clipboard.`, "ok");
      } else {
        onNotice?.(`Copied ${block.length} row(s) × ${names.length} column(s).`, "ok");
      }
    },
    [selection, offset, rows, pendingEdits, onBulkEdit, onNotice],
  );

  const pasteAtSelection = useCallback(async () => {
    const text = await readFromClipboard();
    if (text === null) {
      onNotice?.("The browser blocked clipboard access.", "error");
      return;
    }
    if (anchor === null) {
      onNotice?.("Click the cell to paste into first.", "error");
      return;
    }
    const block = parse(text);
    if (block.length === 0) return;

    const names = visibleColumns.map((c) => c.name);
    const startColumn = names.indexOf(anchor.column);
    if (startColumn === -1) return;

    const edits: PendingEdit[] = [];
    let clipped = 0;

    for (let r = 0; r < block.length; r += 1) {
      const absoluteRow = anchor.row + r;
      if (absoluteRow >= total) {
        // Counted per cell, like the overflow below, so the notice's total is a cell
        // count rather than a mix of cells and rows.
        for (let rest = r; rest < block.length; rest += 1) {
          clipped += block[rest]?.length ?? 0;
        }
        break;
      }
      const sourceRow = block[r] ?? [];
      for (let c = 0; c < sourceRow.length; c += 1) {
        const name = names[startColumn + c];
        if (name === undefined) {
          // Past the last column. Counted and reported rather than dropped silently.
          clipped += 1;
          continue;
        }
        const index = absoluteRow - offset;
        const declared = declaredSample(dtypes.get(name) ?? "");
        const sample =
          declared !== null
            ? declared
            : (pendingEdits[`${absoluteRow}|${name}`] ?? rows[index]?.[name]);
        edits.push({ row: absoluteRow, column: name, value: coerce(sourceRow[c] ?? "", sample) });
      }
    }

    if (edits.length > 0) onBulkEdit(edits);
    onNotice?.(
      clipped > 0
        ? `Pasted ${edits.length} cell(s); ${clipped} fell outside the dataset and were skipped.`
        : `Pasted ${edits.length} cell(s).`,
      clipped > 0 ? "error" : "ok",
    );
  }, [anchor, visibleColumns, dtypes, total, offset, rows, pendingEdits, onBulkEdit, onNotice]);

  // ---------------------------------------------------------------- handlers

  const handleCellValueChanged = useCallback(
    (event: CellValueChangedEvent<Row>) => {
      const field = event.colDef.field;
      if (event.rowIndex === null || field === undefined) return;
      onEdit({ row: offset + event.rowIndex, column: field, value: event.newValue });
    },
    [offset, onEdit],
  );

  const handleSortChanged = useCallback(
    (event: SortChangedEvent<Row>) => {
      const sorted = event.api.getColumnState().filter((c) => c.sort != null);
      const first = sorted[0];
      const next = first === undefined ? null : `${first.colId}:${first.sort ?? "asc"}`;
      sortRef.current = next;
      setSort(next);
      setAnchor(null);
      setFocus(null);
      void load(0, limitRef.current, next);
    },
    [load],
  );

  const handleColumnMoved = useCallback(() => {
    const state = gridRef.current?.api?.getColumnState() ?? [];
    const dragged = state
      .map((c) => c.colId)
      .filter((id): id is string => typeof id === "string");

    // The grid only knows about the columns it is showing, so a hidden one has no
    // place in the order it reports. It keeps the slot it already holds in the model
    // rather than being appended: hiding is a view action, and the order is now
    // something Save writes down, so appending would record a rearrangement the user
    // never made — on the next drag after a hide, and only then.
    let next = 0;
    onReorderColumns(
      columns.map((column) =>
        column.hidden ? column.name : (dragged[next++] ?? column.name),
      ),
    );
  }, [columns, onReorderColumns]);

  const renamePrompt = useCallback(
    (column: string) => {
      const next = window.prompt(`Rename "${column}" to:`, column);
      if (next !== null && next.trim() !== "" && next.trim() !== column) {
        onRenameColumn(column, next.trim());
      }
    },
    [onRenameColumn],
  );

  // Select the whole of one column across the page currently loaded. The server owns
  // the rest of the dataset, so selecting rows that are not in memory would make Copy
  // emit nulls for them; the notice says which window was selected instead.
  const selectEntireColumn = useCallback(
    (column: string) => {
      if (rows.length === 0) {
        onNotice?.("Nothing to select on this page.", "error");
        return;
      }
      const last = offset + rows.length - 1;
      setAnchor({ row: offset, column });
      setFocus({ row: last, column });
      onNotice?.(`Selected rows ${offset.toLocaleString()}–${last.toLocaleString()} of "${column}".`, "ok");
    },
    [offset, rows.length, onNotice],
  );

  const openCellMenu = useCallback(
    (event: CellContextMenuEvent<Row>) => {
      // The browser's own context menu has to be cancelled here. ag-Grid only does it
      // for the cell path through `contextMenuSvc`, which is the Enterprise context
      // menu module and is therefore absent in Community — so without this, right-click
      // opens the browser menu *and* this one, and the browser menu wins the paint.
      // (`preventDefaultOnContextMenu` on the grid covers the header path, which goes
      // through a different handler.)
      event.event?.preventDefault();

      const { x, y } = pointer.current;
      const column = event.colDef.field;
      if (column === undefined) return;
      const rowIndex = event.rowIndex;
      setMenu({
        x,
        y,
        items: [
          { label: "Rename column…", onSelect: () => renamePrompt(column) },
          { label: "Remove column", danger: true, onSelect: () => onDropColumn(column) },
          { label: "Select all", onSelect: () => selectEntireColumn(column) },
          { label: "Hide column", onSelect: () => onHideColumn(column) },
          { label: "Add empty column", separatorBefore: true, onSelect: () => onAddColumn() },
          {
            label: "Copy",
            separatorBefore: true,
            disabled: selection === null,
            onSelect: () => void copySelection(false),
          },
          {
            label: "Cut",
            disabled: selection === null || !editable,
            onSelect: () => void copySelection(true),
          },
          {
            label: "Paste",
            disabled: anchor === null || !editable,
            onSelect: () => void pasteAtSelection(),
          },
          ...(rowIndex === null || !editable
            ? []
            : [
                {
                  label: `Clear cell (row ${(offset + rowIndex).toLocaleString()})`,
                  separatorBefore: true,
                  onSelect: () => onEdit({ row: offset + rowIndex, column, value: null }),
                },
              ]),
        ],
      });
    },
    [
      selection,
      anchor,
      offset,
      editable,
      copySelection,
      pasteAtSelection,
      renamePrompt,
      selectEntireColumn,
      onHideColumn,
      onDropColumn,
      onAddColumn,
      onEdit,
    ],
  );

  const openHeaderMenu = useCallback(
    (event: ColumnHeaderContextMenuEvent<Row>) => {
      // `column` is `Column | ProvidedColumnGroup`. Only a real column has an id to
      // act on; a right-click on a group header (or the padding group) has nothing to
      // rename, hide, or drop, so it is ignored rather than guessed at.
      const target = event.column;
      if (!target.isColumn) return;
      const column = target.getColId();
      const { x, y } = pointer.current;
      setMenu({
        x,
        y,
        items: [
          { label: "Rename column…", onSelect: () => renamePrompt(column) },
          { label: "Remove column", danger: true, onSelect: () => onDropColumn(column) },
          { label: "Select all", onSelect: () => selectEntireColumn(column) },
          {
            label: "Cut",
            separatorBefore: true,
            disabled: selection === null || !editable,
            onSelect: () => void copySelection(true),
          },
          {
            label: "Copy",
            disabled: selection === null,
            onSelect: () => void copySelection(false),
          },
          {
            label: "Paste",
            disabled: anchor === null || !editable,
            onSelect: () => void pasteAtSelection(),
          },
          { label: "Hide column", separatorBefore: true, onSelect: () => onHideColumn(column) },
          { label: "Add empty column", onSelect: () => onAddColumn() },
        ],
      });
    },
    [
      renamePrompt,
      selectEntireColumn,
      selection,
      anchor,
      editable,
      copySelection,
      pasteAtSelection,
      onHideColumn,
      onDropColumn,
      onAddColumn,
    ],
  );

  const capturePointer = useCallback((event: ReactMouseEvent<HTMLDivElement>) => {
    pointer.current = { x: event.clientX, y: event.clientY };
  }, []);

  // Bound to the grid wrapper rather than `document`, so the shortcut does not fire
  // while the user is typing in a form field elsewhere on the page.
  const onKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>) => {
      if (!(event.ctrlKey || event.metaKey)) return;

      // The grid's own cell editor is a text box, and it lives inside this wrapper, so
      // the wrapper sees its keystrokes on the way up. Without this the shortcut would
      // steal Ctrl+C and Ctrl+V from the editor the user is typing in: copying selected
      // text would instead copy the cell range, and pasting would scatter the clipboard
      // across cells rather than inserting it at the caret.
      const target = event.target;
      if (
        target instanceof HTMLElement &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.isContentEditable)
      ) {
        return;
      }

      const key = event.key.toLowerCase();
      if (key === "c") {
        event.preventDefault();
        void copySelection(false);
        return;
      }
      // Cut and paste both write, so they follow the same rule as cell editing.
      if (!editable) return;
      if (key === "x") {
        event.preventDefault();
        void copySelection(true);
      } else if (key === "v") {
        event.preventDefault();
        void pasteAtSelection();
      }
    },
    [copySelection, pasteAtSelection, editable],
  );

  const clearSort = useCallback(() => {
    // Clears the grid's own sort indicators. `applyColumnState` dispatches
    // `sortChanged`, which `handleSortChanged` already responds to by reloading
    // unsorted — so clearing goes through exactly the same path as a header click and
    // there is no second code path to keep in step. The explicit reset after it makes
    // the control still work if that event does not fire; the duplicate read is
    // identical to the one the event triggers, so there is nothing to reconcile.
    gridRef.current?.api?.applyColumnState({ defaultState: { sort: null } });
    sortRef.current = null;
    setSort(null);
    setAnchor(null);
    setFocus(null);
    void load(0, limitRef.current, null);
  }, [load]);

  const pageCount = Math.max(1, Math.ceil(total / limit));
  const currentPage = Math.floor(offset / limit) + 1;

  const goTo = useCallback(
    (page: number) => {
      const clamped = Math.min(Math.max(1, page), pageCount);
      setAnchor(null);
      setFocus(null);
      void load((clamped - 1) * limitRef.current, limitRef.current, sortRef.current);
    },
    [load, pageCount],
  );

  return (
    // `tabIndex={-1}` makes the wrapper focusable so the Ctrl+C/X/V shortcuts have
    // somewhere to land. This project has no linter configured, so the stale
    // `eslint-disable` directive that used to sit here was removed rather than kept:
    // it named a rule that nothing in this repository can enforce.
    <div
      className="grid"
      onKeyDown={onKeyDown}
      onContextMenuCapture={capturePointer}
      tabIndex={-1}
    >
      <div className="grid__toolbar">
        <button
          type="button"
          className="btn btn--sm btn--util"
          onClick={() => goTo(currentPage - 1)}
          disabled={currentPage <= 1 || loading}
        >
          ‹ Prev
        </button>
        <span className="mono muted">
          page {currentPage} / {pageCount}
        </span>
        <button
          type="button"
          className="btn btn--sm btn--util"
          onClick={() => goTo(currentPage + 1)}
          disabled={currentPage >= pageCount || loading}
        >
          Next ›
        </button>

        <span className="mono muted" style={{ marginLeft: 10 }}>
          rows
        </span>
        <select
          aria-label="Rows per page"
          value={limit}
          style={{ width: 84 }}
          onChange={(e) => {
            const next = Number(e.target.value);
            limitRef.current = next;
            setLimit(next);
            setAnchor(null);
            setFocus(null);
            void load(0, next, sortRef.current);
          }}
        >
          {PAGE_SIZES.map((size) => (
            <option key={size} value={size}>
              {size}
            </option>
          ))}
        </select>

        <span className="mono muted" style={{ marginLeft: "auto" }}>
          {loading
            ? "loading…"
            : `${offset.toLocaleString()}–${Math.min(offset + rows.length, total).toLocaleString()} of ${total.toLocaleString()}`}
        </span>
        {sort !== null && <span className="mono muted">sorted by {sort}</span>}
        {sort !== null && (
          <button type="button" className="btn btn--sm btn--util" onClick={clearSort}>
            Clear sort
          </button>
        )}
        {!editable && (
          <span className="mono" style={{ color: "var(--warn)" }}>
            editing locked while sorted — clear the sort to edit
          </span>
        )}
      </div>

      <div className={`ag-theme-quartz${theme === "dark" ? "-dark" : ""} grid__body`}>
        <AgGridReact<Row>
          ref={gridRef}
          // v33 defaults to the JS theming API; `legacy` opts back into the CSS
          // themes that the project's design tokens override.
          theme="legacy"
          // Cancels the browser's default context menu for the *header* path, which
          // routes through `HeaderCtrl.handleContextMenuMouseEvent` rather than through
          // the cell handler below. ag-Grid defaults this to false, so without it a
          // right-click on a column header shows the browser menu over this one.
          preventDefaultOnContextMenu
          // ag-Grid multi-sorts on shift-click by default, and this grid cannot honour
          // it: `/page` takes a single `sort=column:dir` (§17), so only one key ever
          // reaches the server. `handleSortChanged` then sends `sorted[0]` — the first
          // sorted column in *display* order — while ag-Grid orders its own rows by
          // *sort priority*. Shift-click a column left of the current one and the two
          // disagree: the server picks which 100 rows to send by one key, the grid
          // displays them by another, and the result is a page that is sorted by
          // neither and looks sorted by both. Suppressing multi-sort removes the
          // disagreement rather than trying to reconcile it, and matches what the
          // server can actually do.
          suppressMultiSort
          rowData={displayRows}
          columnDefs={columnDefs}
          getRowId={getRowId}
          defaultColDef={{ editable, resizable: true, cellClassRules }}
          onCellValueChanged={handleCellValueChanged}
          onSortChanged={handleSortChanged}
          onColumnMoved={handleColumnMoved}
          onCellContextMenu={openCellMenu}
          onColumnHeaderContextMenu={openHeaderMenu}
          onCellClicked={(event) => {
            if (event.rowIndex === null) return;
            const cell = { row: offset + event.rowIndex, column: event.column.getColId() };
            // `event.event` is typed as a DOM `Event`, which has no `shiftKey` — the
            // narrowing is what makes the modifier readable at all.
            const native = event.event;
            if (native instanceof MouseEvent && native.shiftKey) {
              setFocus(cell);
            } else {
              setAnchor(cell);
              setFocus(cell);
            }
          }}
          suppressCellFocus={false}
          rowBuffer={20}
          animateRows={false}
          stopEditingWhenCellsLoseFocus
        />
      </div>

      {menu !== null && (
        <ContextMenu x={menu.x} y={menu.y} items={menu.items} onClose={() => setMenu(null)} />
      )}
    </div>
  );
}
