import type { ColumnMeta } from "@/api/types";

/**
 * JupyterLab-style head/tail window: the first five and last five rows of a dataset,
 * with a row explaining how many are not shown.
 *
 * Extracted from Data Management because the Cleaning and Resample & Merge pages show
 * the same thing — the preview of a dry run, which is the only chance the user gets to
 * see what a plan would do before it is written. Three copies of it would drift, and
 * the one that drifted would be the one being trusted at the moment of committing.
 */

/** Columns whose values should be laid out right-aligned. */
const NUMERIC = /^(double|float|int|uint|decimal)/;

export function isNumericDtype(dtype: string): boolean {
  return NUMERIC.test(dtype);
}

export function formatCell(value: unknown): string {
  if (value === null || value === undefined) return "null";
  if (typeof value === "number") {
    return Number.isInteger(value) ? value.toLocaleString() : String(value);
  }
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

interface HeadTailTableProps {
  columns: ColumnMeta[];
  head: Record<string, unknown>[];
  tail: Record<string, unknown>[];
  rowCount: number;
}

export function HeadTailTable({ columns, head, tail, rowCount }: HeadTailTableProps) {
  const cellClass = (dtype: string, value: unknown): string =>
    [
      isNumericDtype(dtype) ? "preview__numeric" : "",
      value === null || value === undefined ? "is-null" : "",
    ]
      .filter(Boolean)
      .join(" ");

  return (
    <div className="preview">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column.name} title={column.dtype}>
                {column.name}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {head.map((row, index) => (
            <tr key={`head-${index}`}>
              {columns.map((column) => {
                const value = row[column.name];
                return (
                  <td key={column.name} className={cellClass(column.dtype, value)}>
                    {formatCell(value)}
                  </td>
                );
              })}
            </tr>
          ))}

          {/* Only meaningful when head and tail do not already overlap. The count makes
              clear how much is not being shown. */}
          {rowCount > head.length + tail.length && (
            <tr className="preview__gap">
              <td colSpan={columns.length}>
                ⋮ {(rowCount - head.length - tail.length).toLocaleString()} rows not shown ⋮
              </td>
            </tr>
          )}

          {tail.map((row, index) => (
            <tr key={`tail-${index}`}>
              {columns.map((column) => {
                const value = row[column.name];
                return (
                  <td key={column.name} className={cellClass(column.dtype, value)}>
                    {formatCell(value)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
