import { Description } from "@/components/Description";
import type { ReactNode } from "react";

import type { PolicyOption } from "@/api/types";

/** Methodological choices with details and cautions behind an information control. */
interface PolicyGroupProps<T extends string> {
  /** Radio input group name. Must be unique on the page — two groups sharing a name
   *  would silently make one of them unselectable. */
  name: string;
  legend: string;
  /** What the group is for, available from the legend's information control. */
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
      <legend className="policy__legend">
        {legend}
        {hint !== undefined && <Description className="policy__hint muted">{hint}</Description>}
      </legend>

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
              <Description className="policy__detail muted">{option.detail}</Description>
              {option.caution !== undefined && (
                <Description className="policy__caution">{option.caution}</Description>
              )}
            </span>
          </label>
        );
      })}

      {children}
    </fieldset>
  );
}
