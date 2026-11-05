import { CATEGORY_LABELS, CATEGORY_ORDER, type DatasetSummary } from "@/api/types";

/** Group downstream inputs while distinguishing plan spans from observation rows. */
export function DatasetOptions({ datasets }: { datasets: DatasetSummary[] }) {
  return CATEGORY_ORDER.map((category) => {
    const rows = datasets.filter((row) => row.category === category);
    if (rows.length === 0) return null;
    return (
      <optgroup key={category} label={CATEGORY_LABELS[category]}>
        {rows.map((row) => (
          <option key={row.dataset_id} value={row.dataset_id}>
            {row.name} · {row.row_count.toLocaleString()} {row.artifact_kind === "split_plan" ? "spans" : "rows"}
            {row.timeframe === null ? "" : ` · ${row.timeframe}`}
          </option>
        ))}
      </optgroup>
    );
  });
}
