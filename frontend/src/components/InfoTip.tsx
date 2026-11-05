import { Description } from "./Description";

interface InfoTipProps {
  label: string;
  formula: string;
  detail: string;
  caution?: string | null;
  meta?: string[];
}

/** Formulas use the same click-open information control as other descriptions. */
export function InfoTip({ label, formula, detail, caution, meta }: InfoTipProps) {
  return (
    <Description className="infotip__description" label={label}>
      <span className="infotip__formula mono">{formula}</span>
      <span className="infotip__detail">{detail}</span>
      {caution && <span className="infotip__caution">{caution}</span>}
      {meta && meta.length > 0 && <span className="infotip__meta muted">{meta.join(" · ")}</span>}
    </Description>
  );
}
