import type { SplitPlanRow, SplitRole } from "@/api/types";

/**
 * The plan as a timeline — one lane per fold, one bar per span.
 *
 * Bars are positioned by **row index**, not by date, and that is the whole point of the
 * component. A Gantt drawn on wall-clock time renders an hourly series and a daily series
 * with the same picture, and it cannot show a gap at all: an embargo that removes five bars
 * from an hourly series is five hours wide and invisible, while on a daily series it is five
 * days. Positioned by index, every bar is proportional to the rows it actually holds, and
 * any rows no span covers — the embargo, the purge, the first walk-forward block — appear as
 * empty space exactly where they are. A gap the user asked for and a gap they did not ask
 * for look the same here, which is the honest rendering: both are rows no model will see.
 *
 * Not a Plotly figure, deliberately. Plotly is several megabytes and is currently reached
 * only through the lazily-loaded D1 route; a timeline is a stack of absolutely positioned
 * divs, and using a charting library for it would pull that library into this page's chunk
 * to draw rectangles.
 *
 * The labels are the roles as the plan records them. Nothing here is re-derived: a span's
 * width is its own `row_count`, and its offset is its own `start_index`.
 */

interface TimelineProps {
  rows: SplitPlanRow[];
  /** The dataset's row count — the full width every bar is positioned against. */
  totalRows: number;
}

const ROLE_LABELS: Record<SplitRole, string> = {
  train: "Train",
  val: "Validation",
  test: "Test",
};

/** Fold number → the spans in it, in the plan's own order. A percentage or date plan puts
 *  every span in fold 0; a walk-forward plan has one lane per fold instead of one lane
 *  holding every bar twice over. */
function lanes(rows: SplitPlanRow[]): { fold: number; spans: SplitPlanRow[] }[] {
  const byFold = new Map<number, SplitPlanRow[]>();
  for (const row of rows) {
    const found = byFold.get(row.fold);
    if (found === undefined) byFold.set(row.fold, [row]);
    else found.push(row);
  }
  return [...byFold.entries()]
    .sort(([a], [b]) => a - b)
    .map(([fold, spans]) => ({ fold, spans }));
}

function percent(value: number, total: number): number {
  if (total <= 0) return 0;
  return (value / total) * 100;
}

export function Timeline({ rows, totalRows }: TimelineProps) {
  if (rows.length === 0 || totalRows <= 0) {
    return (
      <div className="timeline timeline--empty muted">
        Nothing to draw: the plan holds no spans.
      </div>
    );
  }

  const grouped = lanes(rows);
  const multiFold = grouped.length > 1;

  return (
    <div className="timeline">
      <div className="timeline__legend">
        {(["train", "val", "test"] as const).map((role) => (
          <span className="timeline__legend-item" key={role}>
            <span className={`timeline__swatch timeline__swatch--${role}`} />
            {ROLE_LABELS[role]}
          </span>
        ))}
        <span className="timeline__legend-item muted">
          {totalRows.toLocaleString()} rows · bar width is proportional to rows, so an
          embargo or a purge shows as the gap it is
        </span>
      </div>

      <div className="timeline__body">
        {grouped.map(({ fold, spans }) => (
          <div className="timeline__lane" key={fold}>
            <span className="timeline__lane-label mono">
              {multiFold ? `fold ${fold}` : "all rows"}
            </span>
            <div className="timeline__track">
              {spans.map((span) => {
                const left = percent(span.start_index, totalRows);
                const width = percent(span.row_count, totalRows);
                return (
                  <div
                    key={`${span.fold}-${span.role}-${span.start_index}`}
                    className={`timeline__bar timeline__bar--${span.role}`}
                    style={{ left: `${left}%`, width: `${width}%` }}
                    title={
                      `${ROLE_LABELS[span.role]} · rows ${span.start_index}–` +
                      `${span.end_index - 1} (${span.row_count.toLocaleString()}) · ` +
                      `${span.start_time} → ${span.end_time}`
                    }
                  >
                    <span className="timeline__bar-text">
                      {ROLE_LABELS[span.role]} · {span.row_count.toLocaleString()}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
