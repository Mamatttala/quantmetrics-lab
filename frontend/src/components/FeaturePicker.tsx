import { useState } from "react";

import type { FeatureGroupPayload, FeatureSpec, FeatureWindow } from "@/api/types";
import { FEATURE_GROUP_LABELS } from "@/api/types";

import { InfoTip } from "./InfoTip";

interface FeaturePickerProps {
  group: FeatureGroupPayload;
  /** Ids of the features that are ticked. */
  picked: ReadonlySet<string>;
  /** Chosen windows per feature id. Parameterless features have no entry. */
  windows: Readonly<Record<string, readonly string[]>>;
  onTogglePick: (spec: FeatureSpec) => void;
  onSetWindows: (featureId: string, windows: string[]) => void;
  /** Tick or untick every feature in this group that can be ticked. */
  onToggleGroup: (group: FeatureGroupPayload, on: boolean) => void;
  busy: boolean;
}

/** A window the catalogue offers, keyed by token, so a chosen custom token can be told
 *  apart from a tick. */
function tickMap(spec: FeatureSpec): Map<string, FeatureWindow> {
  return new Map(spec.windows.map((window) => [window.token, window]));
}

/**
 * The window to tick when a feature is selected but no window has been chosen yet.
 *
 * The catalogue's own default, unless this dataset cannot resolve it — on a daily
 * dataset a feature that defaults to an hour-long window has to fall back to one that
 * exists, or the first tick of a checkbox would produce a request the server refuses.
 * The catalogue promises at least one available window whenever it calls a feature
 * available, so an available feature always has an answer here.
 */
export function firstWindow(spec: FeatureSpec): string[] {
  if (spec.param === null) return [];
  const chosen =
    spec.windows.find((window) => window.token === spec.param?.default && window.available) ??
    spec.windows.find((window) => window.available);
  return chosen === undefined ? [] : [chosen.token];
}

/**
 * One section of the feature catalogue: a heading, a select-all control, and a row per
 * feature.
 *
 * Every row renders the same three things — what the feature is, what it produces, and
 * what it needs — because the two failure modes this page has to prevent are both
 * invisible otherwise. Selecting `hour_of_day` on daily bars yields a column of one
 * repeated value; selecting `sma` five times with neighbouring windows yields five
 * columns that are almost the same column. The first is prevented by disabling the
 * option with its reason on screen; the second is reported after the run by the
 * backend's collinearity scan, since it is a property of the data and not of the
 * catalogue.
 *
 * Unavailable entries are shown disabled rather than hidden: a feature that vanishes
 * from the list on one dataset and appears on another teaches the user that the list is
 * unreliable, and gives them no way to find out why.
 */
export function FeaturePicker({
  group,
  picked,
  windows,
  onTogglePick,
  onSetWindows,
  onToggleGroup,
  busy,
}: FeaturePickerProps) {
  const tickable = group.features.filter((spec) => spec.available);
  const allOn = tickable.length > 0 && tickable.every((spec) => picked.has(spec.id));

  return (
    <section className="section feature__group">
      <div className="section__head">
        <h3 className="section__title">{FEATURE_GROUP_LABELS[group.group]}</h3>
        <span className="muted feature__count">
          {tickable.length} of {group.features.length} available
        </span>
        <div className="actions__spacer" />
        <button
          type="button"
          className="btn btn--ghost"
          disabled={busy || tickable.length === 0}
          onClick={() => onToggleGroup(group, !allOn)}
        >
          {allOn ? "Clear group" : "Select all"}
        </button>
      </div>

      <div className="feature">
        {group.features.map((spec) => (
          <FeatureRow
            key={spec.id}
            spec={spec}
            picked={picked.has(spec.id)}
            windows={windows[spec.id] ?? []}
            onTogglePick={onTogglePick}
            onSetWindows={onSetWindows}
            busy={busy}
          />
        ))}
      </div>
    </section>
  );
}

interface FeatureRowProps {
  spec: FeatureSpec;
  picked: boolean;
  windows: readonly string[];
  onTogglePick: (spec: FeatureSpec) => void;
  onSetWindows: (featureId: string, windows: string[]) => void;
  busy: boolean;
}

function FeatureRow({
  spec,
  picked,
  windows,
  onTogglePick,
  onSetWindows,
  busy,
}: FeatureRowProps) {
  const [draft, setDraft] = useState("");
  const ticks = tickMap(spec);
  // A custom token is one the chosen list holds that the catalogue does not offer. It is
  // kept as a chip with a remove button, because the user typed it and must be able to
  // take it back.
  const custom = windows.filter((token) => !ticks.has(token));
  const off = !spec.available;

  const setWindow = (token: string, on: boolean) => {
    const next = on
      ? [...windows.filter((value) => value !== token), token]
      : windows.filter((value) => value !== token);
    onSetWindows(spec.id, next);
  };

  const addCustom = () => {
    const token = draft.trim();
    if (token === "" || windows.includes(token)) {
      setDraft("");
      return;
    }
    onSetWindows(spec.id, [...windows, token]);
    setDraft("");
  };

  const produces = spec.columns
    .map((template) =>
      spec.param === null
        ? template
        : template.replace("{arg}", picked && windows.length ? windows.join("|") : "{arg}"),
    )
    .join(", ");

  const meta = [
    spec.needs.length > 0 ? `needs ${spec.needs.join(", ")}` : "",
    spec.needs_price ? "uses the price source" : "",
    spec.kind === "carried" ? "carried as-is, nothing is computed" : "",
  ].filter((entry) => entry !== "");

  return (
    <div className={`feature__row${off ? " feature__row--off" : ""}`}>
      <label className="feature__pick">
        <input
          type="checkbox"
          checked={picked}
          disabled={busy || off}
          onChange={() => onTogglePick(spec)}
        />
        <span className="feature__label">{spec.label}</span>
      </label>

      <InfoTip
        label={`Formula for ${spec.label}`}
        formula={spec.formula}
        detail={spec.detail}
        caution={spec.caution}
        meta={
          spec.param === null && spec.warmup_bars !== null
            ? [...meta, `${spec.warmup_bars} warm-up bars`]
            : meta
        }
      />

      <span className="feature__produces mono muted">{produces}</span>

      {off && spec.reason !== null && (
        <span className="feature__reason">{spec.reason}</span>
      )}

      {!off && spec.param !== null && (
        <div className="feature__windows">
          {spec.windows.map((window) => (
            <label
              key={window.token}
              className={`feature__tick${window.available ? "" : " feature__tick--off"}`}
            >
              <input
                type="checkbox"
                checked={windows.includes(window.token)}
                disabled={busy || !window.available}
                onChange={(event) => setWindow(window.token, event.target.checked)}
              />
              <span className="mono">{window.token}</span>
              {window.bars !== null && spec.param?.kind === "duration" && (
                <span className="muted feature__bars">= {window.bars} bars</span>
              )}
            </label>
          ))}

          {spec.param.custom && (
            <span className="feature__custom">
              <input
                type="text"
                className="feature__input mono"
                value={draft}
                placeholder={spec.param.kind === "duration" ? "6h" : "37"}
                aria-label={`Custom window for ${spec.label}`}
                disabled={busy}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    event.preventDefault();
                    addCustom();
                  }
                }}
              />
              <button
                type="button"
                className="btn btn--ghost"
                disabled={busy || draft.trim() === ""}
                onClick={addCustom}
              >
                Add another
              </button>
            </span>
          )}

          {custom.map((token) => (
            <button
              key={token}
              type="button"
              className="feature__chip"
              disabled={busy}
              aria-label={`Remove window ${token} from ${spec.label}`}
              onClick={() => setWindow(token, false)}
            >
              <span className="mono">{token}</span>
              <span aria-hidden="true"> ×</span>
            </button>
          ))}

          {/* A tick that cannot be resolved on this dataset says why, in place. The
              custom box is not checked here: resolving an arbitrary token is the
              backend's job, and a second implementation of that arithmetic in the
              browser is exactly the drift the catalogue exists to prevent. */}
          {spec.windows
            .filter((window) => !window.available && window.reason !== null)
            .map((window) => (
              <span className="feature__reason" key={`why-${window.token}`}>
                {window.token}: {window.reason}
              </span>
            ))}
        </div>
      )}
    </div>
  );
}
