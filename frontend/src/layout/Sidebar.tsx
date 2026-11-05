import { Link, NavLink, useLocation } from "react-router-dom";
import { Fragment } from "react";
import { D_SECTIONS } from "./regimeNavigation";

import { NAV_GROUPS, UTILITY_NAV, type NavEntry } from "./registry";
import { useResizablePane } from "./useResizablePane";

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
  const location = useLocation();
  const { width, resizeHandleProps } = useResizablePane("qca.ui.sidebarWidth", 220, collapsed);
  return (
    <aside
      className={collapsed ? "sidebar sidebar--collapsed" : "sidebar"}
      style={collapsed ? undefined : { width }}
    >
      <div className="sidebar__header">
        <span className="sidebar__brand">Quantmetrics Lab</span>
        <div className="sidebar__header-actions">
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
              <Fragment key={entry.id}>
                <NavRow entry={entry} collapsed={collapsed} />
                {!collapsed && Object.entries(D_SECTIONS).map(([key, group]) => entry.path === `/regime/${key}` && location.pathname === entry.path ?
                  <div key={key} className="regime-subnav">{group.sections.map(([id, label]) =>
                    <Link key={id} to={`${entry.path}${location.search}#${id}`} aria-current={location.hash === `#${id}` ? "page" : undefined}>{label}</Link>
                  )}</div> : null)}
              </Fragment>
            ))}
          </div>
        ))}
      </nav>
      {!collapsed && <div
        {...resizeHandleProps}
        className="pane-resize sidebar__resize"
        aria-label="Resize Sidebar Menu"
      />}
    </aside>
  );
}
