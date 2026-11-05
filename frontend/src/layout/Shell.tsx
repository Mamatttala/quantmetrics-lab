import { useEffect, useState } from "react";
import { Outlet } from "react-router-dom";

import { ApiError, api } from "@/api/client";
import { ThemeToggle } from "@/components/ThemeToggle";
import { DescriptionProvider } from "@/components/Description";

import { DataExplorer } from "./DataExplorer";
import { Sidebar } from "./Sidebar";
import { ProjectPicker } from "./ProjectPicker";
import { useProjects } from "./useProjects";
import { usePersistedFlag } from "./usePersistedFlag";

/**
 * Three-pane shell, left to right: Sidebar | Data Explorer | Main.
 *
 * Both pane states persist under the specified keys so the layout survives a
 * reload. When the explorer is hidden it collapses to zero width and a thin
 * handle remains in the gutter to restore it — without the handle a hidden
 * explorer would be unrecoverable except by clearing localStorage.
 */
export function Shell() {
  const { projects, project, loading, error, selectProject, createProject, refreshProjects } = useProjects();
  const [collapsed, setCollapsed] = usePersistedFlag("qca.ui.sidebar", false);
  const [explorerHidden, setExplorerHidden] = usePersistedFlag("qca.ui.explorer", false);
  const [backendDown, setBackendDown] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .health()
      .then(() => {
        if (!cancelled) setBackendDown(null);
      })
      .catch((cause: unknown) => {
        if (cancelled) return;
        setBackendDown(cause instanceof ApiError ? cause.message : String(cause));
      });
    return () => {
      cancelled = true;
    };
  }, [project?.project_id]);

  return (
    <DescriptionProvider>
      <div className={explorerHidden ? "shell shell--explorer-hidden" : "shell"}>
        <Sidebar collapsed={collapsed} onToggleCollapsed={() => setCollapsed(!collapsed)} />

        {explorerHidden ? (
          <button
            type="button"
            className="explorer-handle"
            onClick={() => setExplorerHidden(false)}
            title="Show Data Explorer"
            aria-label="Show Data Explorer"
          >
            {"‹"}
          </button>
        ) : (
          <DataExplorer
            key={project?.project_id ?? "loading"}
            project={project}
            projectControls={<ProjectPicker
              projects={projects} project={project} loading={loading} error={error}
              onSelect={selectProject} onCreate={createProject}
              onRetry={() => void refreshProjects()}
            />}
            onHide={() => setExplorerHidden(true)}
          />
        )}

        {/* Section cards collapse when their head is clicked (redesign §3.2). One
         * delegated handler covers every page: sections render heads with the
         * `.section__head` class and the toggle is a class flip plus a CSS rule,
         * so no page needs its own state. Clicks on controls inside a head (the
         * count, a policy toggle) must not fold the section, hence the guard. */}
        <main
          className="shell__pane main"
          onClick={(event) => {
            const target = event.target as HTMLElement;
            if (target.closest("button, a, input, select, textarea, label")) return;
            const head = target.closest(".section__head");
            if (head === null) return;
            head.closest(".section")?.classList.toggle("section--collapsed");
          }}
        >
          <div className="shell__toolbar">
            {project && <span className="shell__project-name" title={project.name}>{project.name}</span>}
            <ThemeToggle />
          </div>
          {backendDown !== null && (
            <div className="page">
              <div className="result__bar result__bar--error">
                <span>{backendDown}</span>
              </div>
            </div>
          )}
          {project ? <Outlet key={project.project_id} /> : (
            <div className="page muted">{loading ? "Loading projects…" : "Select or create a project in Data Explorer."}</div>
          )}
        </main>
      </div>
    </DescriptionProvider>
  );
}
