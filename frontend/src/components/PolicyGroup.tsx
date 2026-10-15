import type { ReactNode } from "react";

import type { PolicyOption } from "@/api/types";

/**
 * A radio group of methodological choices, each rendered with the sentence that
 * explains it.
 *
 * The reason this is a component rather than inline markup on each page: every option
 * here is a decision about the data, and several of them are wrong in ways that are
 * invisible in the frame afterwards. Backward fill fabricates a value that had not been
 * published yet; `zero` turns a missing price into a −100% return; winsorization
 * estimates its fence over rows that come *later* than the ones it clamps. A menu of
 * bare labels ("FFill", "Bfill", "Zero") presents all three as equivalent repairs,
 * because the only thing distinguishing them is the consequence — so the consequence
 * has to be on screen next to the radio button, not in a tooltip or a document.
 *
 * `caution` is not a warning that the option is unavailable. Each of these is a
 * legitimate choice for the right dataset; the text says which dataset that is.
 */
interface PolicyGroupProps<T extends string> {
  /** Radio input group name. Must be unique on the page — two groups sharing a name
   *  would silently make one of them unselectable. */
  name: string;
  legend: string;
  /** What the group is for, shown under the legend. */
  hint?: string;
  options: readonly PolicyOption<T>[];
  value: T;
  onChange: (value: T) => void;
  disabled?: boolean;
  /** Rendered inside the group, under the options — for a control that only applies
   *  to certain choices in this group. */
  children?: ReactNode;
}

export function PolicyGroup<T extends string>({
  name,
  legend,
  hint,
  options,
  value,
  onChange,
  disabled = false,
  children,
}: PolicyGroupProps<T>) {
  return (
    <fieldset className="policy" disabled={disabled}>
      <legend className="policy__legend">{legend}</legend>
      {hint !== undefined && <p className="policy__hint muted">{hint}</p>}

      {options.map((option) => {
        const selected = option.value === value;
        return (
          <label
            className={`policy__option${selected ? " policy__option--on" : ""}`}
            key={option.value}
          >
            <input
              type="radio"
              name={name}
              value={option.value}
              checked={selected}
              onChange={() => onChange(option.value)}
            />
            <span className="policy__body">
              <span className="policy__label">{option.label}</span>
              <span className="policy__detail muted">{option.detail}</span>
              {option.caution !== undefined && (
                <span className="policy__caution">{option.caution}</span>
              )}
            </span>
          </label>
        );
      })}

      {children}
    </fieldset>
  );
}
