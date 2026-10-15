/**
 * Navigation registry — the sidebar's single source of truth.
 *
 * Badge codes are keyed by registry id and hardcoded: they are stable references used
 * in documentation and conversation ("A1 is broken"), so they must not drift when items
 * are reordered or relabelled.
 *
 * One deliberate exception is recorded here. C5/C6/C7 were guessed when Part C had not
 * been specified, and the guess was wrong: the specification numbers C5 = Features
 * engineering, C6 = Chronological Splits, C7 = Scaling. None of the three had ever
 * shipped, so the codes were corrected before anything referred to them and are frozen
 * from here. This is the only time a badge has moved; see DECISIONS §42.
 *
 * Part D is now specified: it is the regime/trend detector, one entry, badge D1. It is
 * filed under its own heading rather than appended to C because it consumes a dataset the
 * way C does but answers a different question — C produces columns for a model, D produces
 * a state series for a strategy — and the sidebar is how the user tells those apart.
 *
 * Part E is specified too: volatility forecasting, one entry, badge E1. It gets its own
 * heading for the same reason D did, and one more: it is the first entry in the registry
 * whose subject is a bar that has not happened yet. Everything before it describes or
 * transforms data that exists; nothing in this sidebar's C or D groups may be read as a
 * prediction, and the heading is where that distinction is visible.
 */

export interface NavEntry {
  /** Stable registry id. Never renumbered — routing and docs reference it. */
  id: string;
  /** Short code shown in the collapsed rail and beside the label. */
  badge: string;
  label: string;
  path: string;
}

export interface NavGroup {
  heading: string;
  entries: NavEntry[];
}

export const NAV_GROUPS: NavGroup[] = [
  {
    heading: "A · Data Collection",
    entries: [
      { id: "raw_data_fetch", badge: "A1", label: "Raw Data Fetch", path: "/fetch" },
      { id: "raw_data_view", badge: "A2", label: "Raw Data View", path: "/view" },
    ],
  },
  {
    heading: "B · Preparation",
    entries: [
      { id: "cleaning", badge: "B3", label: "Cleaning", path: "/cleaning" },
      { id: "resample_merge", badge: "B4", label: "Resample & Merge", path: "/resample" },
    ],
  },
  {
    heading: "C · Preprocessing",
    entries: [
      {
        id: "feature_selection",
        badge: "C5",
        label: "Features Engineering",
        path: "/features",
      },
      {
        id: "chronological_splits",
        badge: "C6",
        label: "Chronological Splits",
        path: "/splits",
      },
      { id: "scaling", badge: "C7", label: "Scaling", path: "/scaling" },
    ],
  },
  {
    heading: "D · Regime Detection",
    entries: [
      {
        id: "regime_detection",
        badge: "D1",
        label: "Regime & Trend",
        path: "/regime",
      },
    ],
  },
  {
    heading: "E · Volatility Forecasting",
    entries: [
      {
        id: "volatility_forecast",
        badge: "E1",
        label: "Volatility Forecast",
        path: "/volatility",
      },
    ],
  },
];

/** Pages reachable outside the lettered registry. */
export const UTILITY_NAV: NavEntry[] = [
  { id: "data_management", badge: "DM", label: "Data Management", path: "/data" },
];

export const ALL_ENTRIES: NavEntry[] = [
  ...UTILITY_NAV,
  ...NAV_GROUPS.flatMap((g) => g.entries),
];

export function entryById(id: string): NavEntry | undefined {
  return ALL_ENTRIES.find((e) => e.id === id);
}
