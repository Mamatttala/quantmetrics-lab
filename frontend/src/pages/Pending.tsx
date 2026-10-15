import type { ReactNode } from "react";

import { PageHeader } from "@/components/PageHeader";
import { entryById } from "@/layout/registry";

interface PendingProps {
  entryId: string;
  title: string;
  lead: ReactNode;
  /** Which milestone delivers this page, so an empty route is never a mystery. */
  milestone: string;
}

/**
 * Placeholder for a route that is registered but not yet built.
 *
 * This renders an explicit "not implemented" statement rather than a mock UI. A
 * convincing-looking mock is worse than a blank page here: it makes an unimplemented
 * feature indistinguishable from a broken one.
 */
export function Pending({ entryId, title, lead, milestone }: PendingProps) {
  const entry = entryById(entryId);
  return (
    <div className="page">
      <PageHeader title={title} lead={lead} />
      <div className="result__bar">
        <span>{entry?.badge ?? "—"}</span>
        <span>·</span>
        <span>not implemented</span>
        <span>·</span>
        <span>arrives in {milestone}</span>
      </div>
    </div>
  );
}
