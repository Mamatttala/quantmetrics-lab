import { cloneElement, isValidElement, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { NavLink, useLocation } from "react-router-dom";
import { waitForJob, failureMessage } from "@/api/jobs";
import type { Accepted, Artifact, Parameter } from "@/api/regimes";
import { ResultGrid } from "@/components/ResultGrid";
import { D_SECTIONS } from "@/layout/regimeNavigation";
import "./regimes.css";

export function Navigation({ step }: { step: "discovery" | "evaluation" | "inference" }) {
  const location = useLocation();
  useEffect(() => {
    if (location.hash) document.getElementById(location.hash.slice(1))?.scrollIntoView({ block: "start" });
  }, [location.hash]);
  return <nav className="regime-steps" aria-label="Regime research workflow">
    {Object.entries(D_SECTIONS).map(([key, item]) => <NavLink key={key} to={`/regime/${key}`} className={key === step ? "regime-step active" : "regime-step"}>
      <b>{item.badge}</b><span>{item.label}</span></NavLink>)}
  </nav>;
}
export function Section({ id, title, children, action }: { id: string; title: string; children: ReactNode; action?: ReactNode }) {
  return <section id={id} className="regime-section panel"><div className="regime-section-head"><h2>{title}</h2>{action}</div>{children}</section>;
}
export function Empty({ children }: { children: ReactNode }) { return <div className="regime-empty">{children}</div>; }
export function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  const control = isValidElement<{ "aria-label"?: string }>(children) && typeof children.type === "string"
    ? cloneElement(children, { "aria-label": label }) : children;
  return <label className="regime-field"><span>{label}</span>{control}{hint && <small>{hint}</small>}</label>;
}
export function format(value: unknown): string {
  if (value === null || value === undefined) return "Unavailable";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toPrecision(5);
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}
export function Details({ value, title = "Computed details" }: { value: unknown; title?: string }) {
  return <details className="regime-details"><summary>{title}</summary><pre>{JSON.stringify(value, null, 2)}</pre></details>;
}
export function Metrics({ values }: { values: Record<string, unknown> }) {
  return <dl className="regime-metrics">{Object.entries(values).filter(([, v]) => typeof v !== "object" || v === null).map(([key, value]) => <div key={key}><dt>{key.replaceAll("_", " ")}</dt><dd>{format(value)}</dd></div>)}</dl>;
}
export function Table({ rows, caption, keys }: { rows: Record<string, unknown>[]; caption: string; keys?: string[] }) {
  if (!rows.length) return <Empty>No {caption.toLowerCase()} are available yet.</Empty>;
  const columns: [string, string][] = (keys ?? Object.keys(rows[0] ?? {})).map(key => [key, key.replaceAll("_", " ")]);
  return <ResultGrid caption={caption} rows={rows} columns={columns} />;
}
export function Parameters({ fields, values, columns, onChange }: { fields: Record<string, Parameter>; values: Record<string, unknown>; columns: string[]; onChange: (key: string, value: unknown) => void }) {
  return <div className="regime-form">{Object.entries(fields).map(([key, field]) => <Field key={key} label={key.replaceAll("_", " ")}>
    {field.type === "column" || field.type === "select" ? <select value={String(values[key] ?? field.default ?? "")} onChange={e => onChange(key, e.target.value)}>
      {field.type === "column" && <option value="">Choose upstream feature</option>}{(field.type === "column" ? columns : field.options ?? []).map(option => <option key={option} value={option}>{option}</option>)}
    </select> : field.type === "json" ? <JsonInput value={values[key] ?? field.default} onChange={value => onChange(key, value)} />
      : <input type="number" step="any" min={field.min ?? undefined} max={field.max ?? undefined} value={String(values[key] ?? field.default)} onChange={e => onChange(key, Number(e.target.value))} />}
  </Field>)}</div>;
}
export function JsonInput({ value, onChange }: { value: unknown; onChange: (value: unknown) => void }) {
  const [text, setText] = useState(JSON.stringify(value, null, 2));
  const [invalid, setInvalid] = useState(false);
  useEffect(() => { setText(JSON.stringify(value, null, 2)); }, [value]);
  return <><textarea value={text} rows={3} aria-invalid={invalid} onChange={e => {
    setText(e.target.value);
    try { onChange(JSON.parse(e.target.value)); setInvalid(false); } catch { setInvalid(true); }
  }} />{invalid && <small role="alert">Enter valid JSON before submitting.</small>}</>;
}
export function useTask() {
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);
  useEffect(() => () => abort.current?.abort(), []);
  async function run<T>(action: (signal: AbortSignal, progress: (message: string) => void) => Promise<T | Accepted>, done: (value: T) => void | Promise<void>) {
    setBusy(true); setError(null); setStatus("Submitting…");
    abort.current?.abort(); abort.current = new AbortController();
    try {
      let result: T | Accepted = await action(abort.current.signal, setStatus);
      if (result && typeof result === "object" && "job_id" in result) {
        const job = await waitForJob<T>((result as Accepted).job_id, { signal: abort.current.signal, onPoll: row => setStatus(row.message ?? "Computing…") });
        if (job.status === "failed" || !job.result) throw new Error(failureMessage(job));
        result = job.result;
      }
      if (!abort.current.signal.aborted) { await done(result as T); setStatus("Complete"); }
    } catch (cause) { if (!abort.current?.signal.aborted) setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { if (!abort.current?.signal.aborted) setBusy(false); }
  }
  return { busy, status, error, run, setError };
}
export function Status({ task }: { task: ReturnType<typeof useTask> }) {
  return <>{task.error && <div role="alert" className="result__bar result__bar--error">{task.error}</div>}{task.busy && <div role="status" className="regime-status">{task.status}</div>}</>;
}
export function ArtifactSelect({ label, value, items, onChange }: { label: string; value: string; items: Artifact[]; onChange: (id: string) => void }) {
  return <Field label={label}><select value={value} onChange={e => onChange(e.target.value)}><option value="">Select…</option>{items.map(item => <option key={item.id} value={item.id} disabled={item.integrity_error}>{item.name} · {item.model ?? item.kind} · {item.id.slice(-6)}</option>)}</select></Field>;
}
