import { useState } from "react";
import type { Project } from "@/api/types";

interface ProjectPickerProps {
  projects: Project[];
  project: Project | null;
  loading: boolean;
  error: string | null;
  onSelect: (project: Project) => void;
  onCreate: (name: string) => Promise<void>;
  onRetry: () => void;
}

export function ProjectPicker({ projects, project, loading, error, onSelect, onCreate, onRetry }: ProjectPickerProps) {
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("");
  const [saving, setSaving] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);

  return (
    <div className="project-picker">
      <label className="project-picker__label" htmlFor="active-project">Project</label>
      <select
        id="active-project"
        className="project-picker__select"
        value={project?.project_id ?? ""}
        disabled={loading || saving || projects.length === 0}
        onChange={(event) => {
          const selected = projects.find((row) => row.project_id === event.target.value);
          if (selected && selected.project_id !== project?.project_id) onSelect(selected);
        }}
      >
        {!project && <option value="">{loading ? "Loading projects…" : "Select a project"}</option>}
        {projects.map((row) => <option key={row.project_id} value={row.project_id}>{row.name}</option>)}
      </select>
      {!creating ? (
        <button type="button" className="explorer__btn" disabled={loading} onClick={() => setCreating(true)}>
          + New project
        </button>
      ) : (
        <form className="project-picker__form" onSubmit={async (event) => {
          event.preventDefault();
          if (!name.trim() || saving) return;
          setSaving(true);
          setCreateError(null);
          try {
            await onCreate(name.trim());
            setName("");
            setCreating(false);
          } catch (cause) {
            setCreateError(cause instanceof Error ? cause.message : String(cause));
          } finally {
            setSaving(false);
          }
        }}>
          <label className="project-picker__label" htmlFor="new-project-name">Project name</label>
          <input id="new-project-name" autoFocus required maxLength={80} value={name}
            disabled={saving} placeholder="e.g. Bitcoin research"
            onChange={(event) => setName(event.target.value)} />
          <div className="project-picker__buttons">
            <button type="submit" className="explorer__btn" disabled={saving || !name.trim()}>
              {saving ? "Creating…" : "Create project"}
            </button>
            <button type="button" className="explorer__btn" disabled={saving} onClick={() => {
              setCreating(false); setName(""); setCreateError(null);
            }}>Cancel</button>
          </div>
          {createError && <p className="project-picker__error" role="alert">{createError}</p>}
        </form>
      )}
      {error && <div className="project-picker__error" role="alert">
        {error}
        <button type="button" className="explorer__btn" onClick={onRetry}>Retry</button>
      </div>}
    </div>
  );
}
