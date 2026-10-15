/**
 * Persisted UI state.
 *
 * localStorage is wrapped in try/catch on every access: it throws rather than
 * returns null in private windows and when site data is blocked. A layout
 * preference is not worth a blank screen, so a failure silently degrades to the
 * in-memory default.
 */

import { useCallback, useState } from "react";

function read(key: string, fallback: boolean): boolean {
  try {
    const raw = window.localStorage.getItem(key);
    if (raw === null) return fallback;
    return raw === "true";
  } catch {
    return fallback;
  }
}

function write(key: string, value: boolean): void {
  try {
    window.localStorage.setItem(key, String(value));
  } catch {
    /* preference not persisted; not worth surfacing */
  }
}

export function usePersistedFlag(
  key: string,
  fallback: boolean,
): [boolean, (next: boolean) => void] {
  const [value, setValue] = useState<boolean>(() => read(key, fallback));
  const set = useCallback(
    (next: boolean) => {
      setValue(next);
      write(key, next);
    },
    [key],
  );
  return [value, set];
}
