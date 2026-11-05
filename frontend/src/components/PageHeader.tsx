import { Description } from "@/components/Description";
import type { ReactNode } from "react";

interface PageHeaderProps {
  title: string;
  /** One muted sentence. Wrap emphasized words in <strong> or <code>. */
  lead: ReactNode;
  children?: ReactNode;
}

/** The header pattern every page uses, so titles and leads align across routes. */
export function PageHeader({ title, lead, children }: PageHeaderProps) {
  return (
    <header>
      <h1 className="page__title">
        {title}
        <Description className="page__lead">{lead}</Description>
      </h1>
      {children}
    </header>
  );
}
