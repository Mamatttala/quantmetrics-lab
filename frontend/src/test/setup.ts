import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

const stored = new Map<string, string>();
const memoryStorage: Storage = {
  get length() { return stored.size; },
  clear() { stored.clear(); },
  getItem(key) { return stored.get(key) ?? null; },
  key(index) { return [...stored.keys()][index] ?? null; },
  removeItem(key) { stored.delete(key); },
  setItem(key, value) { stored.set(key, String(value)); },
};

// Node 26 exposes an incomplete global localStorage unless it is launched with a
// file flag. Tests need browser semantics, so install a deterministic in-memory store.
Object.defineProperty(window, "localStorage", { configurable: true, value: memoryStorage });
Object.defineProperty(globalThis, "localStorage", { configurable: true, value: memoryStorage });

afterEach(() => {
  cleanup();
});
