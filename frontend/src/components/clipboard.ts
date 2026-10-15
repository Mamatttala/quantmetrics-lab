/**
 * Clipboard handling for the data grid.
 *
 * ag-Grid's clipboard integration is an Enterprise feature, so this implements the
 * cut/copy/paste behaviour directly. Two deliberate choices:
 *
 * * Cells are joined with TAB and rows with newline — the format spreadsheets and
 *   ag-Grid itself both use, so copy/paste interoperates with Excel.
 * * A paste never silently truncates. Pasting a block larger than the selection is
 *   clipped to what fits and *reported*, because a paste that quietly drops half its
 *   rows is indistinguishable from one that worked.
 */

export interface CellRef {
  row: number;
  column: string;
}

export interface CellRect {
  startRow: number;
  endRow: number;
  columns: string[];
}

export function rectContains(rect: CellRect, row: number, column: string): boolean {
  return (
    row >= Math.min(rect.startRow, rect.endRow) &&
    row <= Math.max(rect.startRow, rect.endRow) &&
    rect.columns.includes(column)
  );
}

export function serialise(rows: unknown[][]): string {
  return rows
    .map((row) =>
      row
        .map((cell) => {
          if (cell === null || cell === undefined) return "";
          const text = String(cell);
          // Quote anything containing a delimiter so the structure survives a round
          // trip. The comma is in this list because `parse` accepts it as a
          // delimiter: without it, copying a cell holding "1,234" and pasting it back
          // would split the value across two columns.
          return /[\t\n",]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
        })
        .join("\t"),
    )
    .join("\n");
}

/** Parse TSV/CSV text into a rectangular block of strings. */
export function parse(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let quoted = false;

  for (let i = 0; i < text.length; i += 1) {
    // `charAt` rather than `text[i]`: under `noUncheckedIndexedAccess` indexing yields
    // `string | undefined`, and `field += char` would then be a type error. `charAt`
    // returns `""` past the end, which the loop bound already excludes.
    const char = text.charAt(i);
    if (quoted) {
      if (char === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i += 1;
        } else {
          quoted = false;
        }
      } else {
        field += char;
      }
      continue;
    }
    if (char === '"' && field === "") {
      // A quote only opens a quoted field at the *start* of one, which is the CSV
      // rule. Treating it as a quote anywhere would make a stray `"` mid-value
      // swallow the rest of the row.
      quoted = true;
    } else if (char === "\t" || char === ",") {
      row.push(field);
      field = "";
    } else if (char === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else if (char !== "\r") {
      field += char;
    }
  }
  row.push(field);
  rows.push(row);

  // Drop trailing blank rows. `parse("")` yields exactly one, and `pasteAtSelection`
  // treats an empty block as "nothing to paste"; a final newline in the copied text
  // produces one too, and writing it would clear a row the user never selected.
  while (rows.length > 0 && rows[rows.length - 1]?.every((cell) => cell === "")) {
    rows.pop();
  }
  return rows;
}

/**
 * Coerce pasted text toward the target column's existing type.
 *
 * A grid whose `close` column is numeric must not silently acquire the string
 * "1.5", which would turn the column to `object` dtype downstream and break every
 * arithmetic feature built on it. Anything unparseable is passed through as text so
 * the user sees what they actually pasted rather than a mangled value.
 */
export function coerce(raw: string, sample: unknown): unknown {
  if (raw === "") return null;
  if (typeof sample === "number") {
    const parsed = Number(raw);
    return Number.isFinite(parsed) ? parsed : raw;
  }
  if (typeof sample === "boolean") {
    if (raw.toLowerCase() === "true") return true;
    if (raw.toLowerCase() === "false") return false;
    return raw;
  }
  if (sample instanceof Date) {
    const parsed = new Date(raw);
    return Number.isNaN(parsed.getTime()) ? raw : parsed.toISOString();
  }
  return raw;
}

export async function writeToClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // The async Clipboard API is unavailable over plain HTTP on some browsers and
    // can be blocked by permissions policy. Reporting the failure beats pretending
    // the copy succeeded.
    return false;
  }
}

export async function readFromClipboard(): Promise<string | null> {
  try {
    return await navigator.clipboard.readText();
  } catch {
    return null;
  }
}
