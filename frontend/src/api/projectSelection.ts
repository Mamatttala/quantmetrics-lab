/** Keep this tab's project stable while other tabs select their own projects. */
const STORAGE_KEY = "qca.project";

function readSelection(): string {
  try {
    return window.sessionStorage.getItem(STORAGE_KEY)
      ?? window.localStorage.getItem(STORAGE_KEY)
      ?? "default";
  } catch {
    return "default";
  }
}

let projectId = readSelection();
let revision = 0;

export function getActiveProjectId(): string {
  return projectId;
}

export function getProjectRevision(): number {
  return revision;
}

export function setActiveProjectId(id: string): void {
  if (id !== projectId) revision += 1;
  projectId = id;
  try {
    window.sessionStorage.setItem(STORAGE_KEY, id);
    window.localStorage.setItem(STORAGE_KEY, id);
  } catch {
    // The tab can still select projects when browser storage is unavailable.
  }
}
