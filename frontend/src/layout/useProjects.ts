import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

import { api } from "@/api/client";
import { getActiveProjectId, setActiveProjectId } from "@/api/projectSelection";
import type { Project } from "@/api/types";

export function useProjects() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [project, setProject] = useState<Project | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const loadSequence = useRef(0);
  const navigate = useNavigate();
  const navigateRef = useRef(navigate);
  navigateRef.current = navigate;

  const selectProject = useCallback((selected: Project) => {
    setActiveProjectId(selected.project_id);
    setProject(selected);
    // Clear dataset routes and remount page state before accepting new input.
    navigateRef.current("/data");
  }, []);

  const refreshProjects = useCallback(async () => {
    const sequence = ++loadSequence.current;
    setLoading(true);
    setError(null);
    try {
      const rows = await api.listProjects();
      if (sequence !== loadSequence.current) return;
      setProjects(rows);
      const selected = rows.find((row) => row.project_id === getActiveProjectId());
      if (selected) {
        // Pin the selection to this tab, including the first load of a new tab.
        setActiveProjectId(selected.project_id);
        setProject(selected);
      } else if (rows[0]) {
        selectProject(rows[0]);
      } else {
        throw new Error("No projects are available. Create a project to get started.");
      }
    } catch (cause) {
      if (sequence === loadSequence.current) {
        setError(cause instanceof Error ? cause.message : String(cause));
      }
    } finally {
      if (sequence === loadSequence.current) setLoading(false);
    }
  }, [selectProject]);

  useEffect(() => {
    void refreshProjects();
    return () => { loadSequence.current += 1; };
  }, [refreshProjects]);

  const createProject = async (name: string) => {
    const created = await api.createProject(name);
    setProjects((rows) => [...rows, created]);
    setError(null);
    selectProject(created);
  };

  return { projects, project, loading, error, selectProject, createProject, refreshProjects };
}
