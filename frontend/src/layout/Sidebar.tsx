import { NavLink } from "react-router-dom";

import { ThemeToggle } from "@/components/ThemeToggle";

import { NAV_GROUPS, UTILITY_NAV, type NavEntry } from "./registry";

interface SidebarProps {
  collapsed: boolean;
  onToggleCollapsed: () => void;
}

function NavRow({ entry, collapsed }: { entry: NavEntry; collapsed: boolean }) {
  return (
    <NavLink
      to={entry.path}
      // Native tooltip carries the full label in the collapsed rail, where the
      // text is hidden and the badge alone would be ambiguous.
      title={collapsed ? `${entry.badge} · ${entry.label}` : entry.label}
      className={({ isActive }) =>
        isActive ? "navitem navitem--active" : "navitem"
      }
    >
      <span className="navitem__badge">{entry.badge}</span>
      <span className="navitem__label">{entry.label}</span>
    </NavLink>
  );
}

export function Sidebar({ collapsed, onToggleCollapsed }: SidebarProps) {
  return (
    <aside className={collapsed ? "sidebar sidebar--collapsed" : "sidebar"}>
      <div className="sidebar__header">
        <span className="sidebar__brand">Quantmetrics Lab</span>
        <div className="sidebar__header-actions">
          <ThemeToggle />
          <button
            type="button"
            className="sidebar__collapse"
            onClick={onToggleCollapsed}
            aria-expanded={!collapsed}
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
          {collapsed ? "›" : "‹"}
          </button>
        </div>
      </div>

      <nav className="sidebar__nav">
        {UTILITY_NAV.map((entry) => (
          <NavRow key={entry.id} entry={entry} collapsed={collapsed} />
        ))}

        {NAV_GROUPS.map((group) => (
          <div className="sidebar__group" key={group.heading}>
            {!collapsed && <div className="sidebar__heading">{group.heading}</div>}
            {group.entries.map((entry) => (
              <NavRow key={entry.id} entry={entry} collapsed={collapsed} />
            ))}
          </div>
        ))}
      </nav>
    </aside>
  );
}
