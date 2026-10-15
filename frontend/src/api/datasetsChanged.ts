/**
 * A change signal for the dataset catalog.
 *
 * The Data Explorer is mounted once by the shell and does not remount when a fetch,
 * derive, or upload happens in the main pane, so without a signal it would keep
 * showing a stale list until the user pressed Refresh. A subscriber set is
 * deliberately the smallest thing that closes that gap: this is not a state
 * container, holds no data, and should not grow into one.
 */

type Listener = () => void;

const listeners = new Set<Listener>();

/** Announce that the catalog changed. Listeners are called synchronously. */
export function notifyDatasetsChanged(): void {
  // Iterating a copy so a listener that unsubscribes while being notified cannot
  // disturb the iteration.
  for (const listener of [...listeners]) listener();
}

/** Subscribe to catalog changes. Returns the unsubscribe function. */
export function subscribeToDatasets(listener: Listener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}
