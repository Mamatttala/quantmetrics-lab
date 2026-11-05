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
      { id: "exploratory", badge: "C8", label: "Exploratory Analysis", path: "/exploratory" },
    ],
  },
  {
    heading: "D · Regime Detection",
    entries: [
      {
        id: "regime_discovery",
        badge: "D9",
        label: "Model Selection & State Discovery",
        path: "/regime/discovery",
      },
      {
        id: "regime_evaluation",
        badge: "D10",
        label: "Evaluation & Interpretation",
        path: "/regime/evaluation",
      },
      {
        id: "regime_inference",
        badge: "D11",
        label: "Inference & Current Regime",
        path: "/regime/inference",
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
