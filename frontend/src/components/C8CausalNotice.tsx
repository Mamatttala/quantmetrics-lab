import { useEffect, useState } from "react";
import { api } from "@/api/client";

/** A C8 fit label is advisory; downstream selection remains available. */
export function C8CausalNotice({ datasetId }: { datasetId: string | undefined }) {
  const [nonCausal, setNonCausal] = useState(false);
  useEffect(() => {
    let active = true;
    setNonCausal(false);
    if (datasetId) void api.getDataset(datasetId).then((meta) => {
      if (active) setNonCausal(meta.provenance.causal_for_downstream_training === "false");
    }).catch(() => {});
    return () => { active = false; };
  }, [datasetId]);
  return nonCausal ? <p role="status" style={{ color: "var(--warn)", borderLeft: "3px solid var(--warn)", padding: 10 }}>
    Non-causal C8 fit — selection is allowed. PCA/preprocessing used exploratory fit rows.
    The C8 flag describes its own fitting only, not the entire pipeline.
  </p> : null;
}
