# Frontend UI Redesign Specification for Quantitative Finance Web App

## 1. Context and Objective

Redesign the frontend UI of an existing single-user quantitative finance and data analysis web application. The application currently uses a dark theme. The goal is to implement a cohesive, modern, and highly functional design system across all pages.

**Crucial Constraint:** Do not rely on specific page names (e.g., Monte Carlo, Generator, etc.). Apply this design system universally to all existing pages, forms, data views, and charts. Focus purely on the visual design, component architecture, layout, and interactivity patterns.

## 2. Global Design Tokens

### 2.1 Color Palette

Implement the following CSS custom properties globally:

**Backgrounds:**

- App/Canvas Background: `#161616` (Near black)
- Sidebar/Explorer Background: `#1a1a1a`
- Main Panel/Card Background: `#1c1c1c`
- Card Header Background: `#202020`
- Input Field Background: `#1a1a1a`
- Dark Chart Background: `#161616`
- Light Chart Background (for specific diagnostic plots): `#ffffff`

**Borders:**

- Subtle Dividers: `#2a2a2a`
- Panel/Card Borders: `#2a2a2a`
- Input Borders: `#333333`
- Input Focus Border: `#d4a843`

**Accent & Semantic Colors:**

- Primary Accent (Gold): `#d4a843`
- Accent Hover: `#e8c060`
- Accent Dim (Backgrounds): `rgba(212, 168, 67, 0.15)`
- Success/Positive: `#4ade80`
- Warning: `#fbbf24`
- Error/Negative: `#ef4444`
- Info: `#60a5fa`

**Text Colors:**

- Primary Text: `#f0f0f0`
- Secondary Text (Labels): `#cccccc`
- Muted Text (Descriptions/Axis): `#888888`
- Disabled/Placeholder Text: `#5a5a5a`
- Dark Text (for light charts): `#1a1a1a`

**Chart Specific Colors:**

- Selected Path / Best Path: `#2dd4bf` (Teal) or `#4ade80` (Green)
- Worst Path: `#f87171` (Salmon Red)
- Max Run-up Path: `#60a5fa` (Light Blue)
- Max Drawdown Path: `#f59e0b` (Amber)
- Path Cloud / Non-selected paths: `rgba(136, 136, 136, 0.2)`
- 10-90% Confidence Band: `rgba(136, 136, 136, 0.15)`
- Histogram Bars (Dark): `rgba(99, 102, 241, 0.7)` (Indigo)
- Histogram Bars (Gold): `rgba(212, 168, 67, 0.7)`
- Reference PDF Curve: `#22d3ee` (Cyan)
- KDE Curve: `#ffffff` (White)
- ECDF Curve: `#38bdf8` (Sky Blue)
- Mean Line: `#4ade80` (Green, dashed)
- Median Line: `#f87171` (Red, dotted)
- QQ Plot Theoretical Line: `#0d9488` (Teal)
- QQ Plot 45-degree Line: `#dc2626` (Red, dashed)
- QQ Plot Quantile Dots: `#7c3aed` (Purple)

**UI Elements:**

- Slider Track: `#3a3a3a`
- Slider Fill/Thumb: `#8b5cf6` (Purple)
- Scrollbar Track: `#1a1a1a`
- Scrollbar Thumb: `#3a3a3a` (Hover: `#d4a843`)
- JSON Syntax - Keys: `#60a5fa`, Strings: `#4ade80`, Numbers: `#f59e0b`

### 2.2 Typography

Import 'Inter' (UI) and 'JetBrains Mono' (Code/Data) from Google Fonts.

**Type Scale:**

- App Title: 18px, Bold (700), UPPERCASE, Gold color.
- Page Titles: 22px, Bold (700), Capitalize, Primary Text.
- Section/Card Headers: 15px, SemiBold (600), Capitalize, Primary Text.
- Subsection Headers: 13px, SemiBold (600), UPPERCASE, Gold color.
- Form Labels: 12px, Medium (500), Capitalize, Secondary Text.
- Form Inputs/Values: 13px, Regular (400), None, Primary Text.
- Placeholders: 13px, Regular (400), Italic, Disabled Text.
- Button Text (Primary): 13px, SemiBold (600), UPPERCASE, Dark Text.
- Button Text (Secondary): 12px, Medium (500), UPPERCASE, Gold Text.
- Sidebar Nav Items: 13px, Medium (500), Capitalize.
- Sidebar Shortcut Letters: 14px, Bold (700), UPPERCASE, Gold.
- Explorer Tree Items: 12px, Regular (400), None (preserve case for IDs).
- Status/Feedback Text: 12px, Regular (400), Monospace for IDs/hashes.
- Chart Titles: 14px, Bold (700).
- Chart Axis Labels: 11px, Regular (400), Capitalize, Muted Text.
- Chart Tick Values: 10px, Regular (400), Muted Text.
- Chart Legends: 11px, Regular (400).
- Descriptions/Help Text: 12px, Regular (400), Italic, Muted Text.
- Badges (Core/Required/Optional): 10px, Medium (500), UPPERCASE.
- Code/JSON/IDs: 11-12px, Monospace.

**Capitalization Rules:**

- UPPERCASE: App title, explorer headers, subsection titles, button text, badges, shortcut letters.
- Capitalize: Page titles, section titles, form labels, nav items, chart axis labels.
- None (preserve original case): Input values, placeholders, tree items (IDs/hashes), status values, JSON/code blocks, description text.
- Always uppercase acronyms (e.g., MDD, AR1, PDF, JSON, ID).

### 2.3 Spacing and Radii

- Spacing: xs(4px), sm(8px), md(12px), lg(16px), xl(20px), 2xl(24px).
- Border Radii: sm(4px), md(6px), lg(8px), xl(12px).

## 3. Layout Architecture

### 3.1 Three-Column Structure

1. **Main Navigation Sidebar (Left):** ~220px wide. Collapsible to ~50px (showing only shortcut letter squares). Background: Sidebar Background.
2. **Explorer Panel (Middle):** ~280px wide. Collapsible. Background: Sidebar Background.
3. **Main Content Area (Right):** Flex: 1, takes remaining space. Background: App Background.

### 3.2 Collapsible Behavior

- **Sidebars:** Toggle buttons (`<` / `>`) at the top. Smooth transition.
- **Section Cards:** Every major section in the main content area must be a collapsible card. Header bar contains the title and a chevron icon (▼/▶). Clicking the header toggles the body with a smooth 0.3s height/opacity animation. Default state: expanded.
- **Tree Nodes:** Expandable/collapsible with chevrons. 16px indentation per level.
- **Multi-step Form Steps:** Each step is an independently collapsible card.

### 3.3 Responsiveness

- Form rows using 2-column grids (`1fr 1fr`) must collapse to a single column on screens narrower than 1200px.
- Chart grids (2-column) must collapse to 1-column on narrow screens.
- Charts must be responsive (`width: 100%`, maintaining aspect ratio).

## 4. Component Library

### 4.1 Buttons

- **Primary (Gold Filled):** Background `#d4a843`, Text `#1a1a1a`, no border, radius 6px, padding 12px 24px. Hover: Background `#e6bc5a`, slight upward translate, gold box-shadow.
- **Secondary (Gold Outline):** Background transparent, Text `#d4a843`, border 1px solid `#d4a843`, radius 6px, padding 12px 24px. Hover: Background `rgba(212,168,67,0.1)`.
- **Small Buttons (Refresh, etc.):** Padding 6px 14px, font-size 11px.
- **Icon Buttons (Chart toolbars):** Transparent background, border 1px solid `#3a3a3a`, radius 4px, padding 4px 8px, color `#888888`. Hover: border and color become gold.
- **Button Rows:** Large primary button takes `flex: 1`, smaller buttons sit to the right with a 12px gap.

### 4.2 Form Inputs

- Background `#2a2a2a`, border 1px solid `#3a3a3a`, radius 6px, padding 10px 12px, color `#f0f0f0`, font-size 13px.
- Focus state: border-color `#d4a843`, box-shadow `0 0 0 2px rgba(212,168,67,0.15)`.
- Dropdowns: Same styling, custom gold chevron arrow.
- Checkboxes: Custom 14x14px. Unchecked: border `#3a3a3a`. Checked: background `#d4a843`, dark checkmark.

### 4.3 Cards and Panels

- Background `#1e1e1e`, border 1px solid `#333333`, radius 8px, padding 20px, margin-bottom 16px.
- **Flat design:** No box shadows on panels, no gradients.
- **Card Header (for collapsible cards):** Background `#252525`, bottom border 1px solid `#333333`, padding 12px 20px, top-radius 8px.

### 4.4 Sidebar Navigation

- Items: radius 6px, padding 10px 14px, margin 2px 8px, left-border 3px solid transparent.
- Active state: left-border-color `#d4a843`, background `#2a2520`, text `#f0f0f0`.
- Hover state: background `#2a2a2a`.

### 4.5 Explorer Tree

- Container: Background `#1a1a1a`, border 1px solid `#333333`, radius 8px, padding 12px.
- Header: UPPERCASE, 11px bold, letter-spacing 1.5px, color `#888888`, with a "Refresh" button aligned right.
- Nodes: Hover background `#2a2a2a`, radius 4px. Selected node: background `#2a2520`, text `#f0f0f0`.

### 4.6 Badges

- Padding 2px 8px, radius 4px, font-size 10px, UPPERCASE.
- **Core:** Border/Text `#d4a843`, bg `rgba(212,168,67,0.15)`.
- **Required:** Border/Text `#ef4444` (use `#f87171` for text), bg `rgba(239,68,68,0.15)`.
- **Optional:** Border/Text `#888888` (use `#555555` for border), bg `rgba(136,136,136,0.1)`.

### 4.7 Status and Info Bars

- **Status Bar:** Background `#1a1a1a`, border 1px solid `#333333`, radius 6px, padding 10px 16px, monospace 12px. Success: green border/text. Error: red border/text. Idle: gray border/text.
- **Pipe-Separated Stats Bar:** Background `#1e1e1e`, border 1px solid `#333333`, radius 6px, padding 10px 16px, monospace 12px. Format: `key=value | key=value`. Keys slightly lighter than values.
- **Metadata Block:** Background `#1a1a1a`, border 1px solid `#333333`, radius 6px, padding 12px 16px. Labels muted, values monospace.

### 4.8 Code and JSON Blocks

- Background `#161616`, border 1px solid `#333333`, radius 6px, padding 16px, monospace 12px.
- Syntax highlighting: Keys blue, strings green, numbers amber.

### 4.9 Sliders and Bin Controls

- **Sliders:** Track `#3a3a3a`, fill/thumb `#8b5cf6` (purple). Thumb is a 14px circle. Hover scales thumb to 1.2. Placed next to a small number input (60px wide, right-aligned).
- **Bin Controls ( - [value] + ):** Minus and Plus buttons have background `#2a2a2a`, border `#3a3a3a`, radius 4px, color gold. Hover: gold border and dim gold background. Number input in the middle is centered, 50px wide.

### 4.10 Hover Mode Toggles

- Two buttons side-by-side (e.g., "Hover: Concise" / "Hover: Verbose").
- Active: background `#2a2a2a`, border 1px solid gold, text gold.
- Inactive: transparent background, border `#3a3a3a`, text muted.

## 5. Data Visualization and Charts

### 5.1 Chart Containers and Toolbars

- **Dark Theme (Default):** Background `#161616`, border 1px solid `#333333`, radius 8px, padding 16px.
- **Light Theme (Diagnostics/QQ Plots):** Background `#ffffff`, border 1px solid `#e0e0e0`, radius 8px, padding 16px. Uses dark text (`#1a1a1a`) and gray axes (`#666666`).
- **Toolbar (Top-Right):** Icons for camera, zoom-in, zoom-out, pan, reset, home, lasso. Color `#666666`, size 14px. Hover: gold.

### 5.2 Chart Grid and Axes

- Grid lines: `#2a2a2a`, very subtle.
- Axis labels: Muted text, 11px, Capitalize.
- Tick values: `#666666`, 10px.

### 5.3 Specific Chart Types

- **Path Cloud (Fan Chart):** Dense fan of non-selected paths in gray with low opacity. 10-90% band shaded in subtle gray. Max run-up (dotted blue), Max drawdown (dashed orange), Worst path (solid red), Best/Selected path (solid teal/green). Legend horizontal at top-center.
- **Histogram + PDF/KDE:** Histogram bars in indigo or gold (opacity 0.7). PDF curve in cyan (linewidth 2px). KDE curve in white. Mean line (green dashed), Median line (red dotted).
- **ECDF Chart:** Sky blue curve, linewidth 2px, step-function style.
- **QQ Plots:** Theoretical line in teal. 45-degree reference line in red dashed. Quantile annotation dots in purple (1%, 5%, 50%, 95%, 99%) with small text labels.
- **Overlaid PDF (Light Theme):** Sample histogram in light teal (opacity 0.3). Reference curve red, Comparison curve blue.

### 5.4 Chart Interactivity

- **Tooltips:** Background `#2a2a2a`, border `#3a3a3a`, radius 6px, padding 8px 12px. Title 12px bold. Rows 11px monospace with colored line swatches matching the series.
- **Zoom/Pan:** Mouse wheel to zoom, drag to pan. Double-click to reset.
- **Crosshairs:**
  - Style 1 (Dark charts): Gold `+` crosshair, linewidth 1px.
  - Style 2 (Light charts/Annotation mode): Gray circle ~30px with "T" icon inside, gold border.
- **Selection:** Clicking a path in a fan chart highlights it (turns green/teal) and dims others.
- **Legend Toggle:** Clicking legend items shows/hides the respective series.

## 6. Global CSS Rules and Utilities

```css
/* Base Resets */
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background-color: #0d0d0d;
  color: #f0f0f0;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  font-size: 13px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}

/* Scrollbars */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #1a1a1a; }
::-webkit-scrollbar-thumb { background: #3a3a3a; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #d4a843; }

/* Selection and Focus */
::selection { background: rgba(212, 168, 67, 0.3); color: #f0f0f0; }
*:focus-visible { outline: 2px solid #d4a843; outline-offset: 2px; }

/* Transitions */
button, input, select, textarea, a { transition: all 0.2s ease; }

/* Collapsible Animations */
.collapsible-body {
  overflow: hidden;
  transition: max-height 0.3s ease, opacity 0.3s ease;
}
.collapsible-body.collapsed { max-height: 0; opacity: 0; }

```

## 7. Spacing & Radii

```css
:root {
  --space-xs: 4px;
  --space-sm: 8px;
  --space-md: 12px;
  --space-lg: 16px;
  --space-xl: 20px;
  --space-2xl: 24px;

  --radius-sm: 4px;
  --radius-md: 6px;
  --radius-lg: 8px;
  --radius-xl: 12px;
}
```

## Font stacks

Body/UI: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif
Code/IDs/JSON: 'JetBrains Mono', 'Fira Code', 'Consolas', monospace

### Type scale

Token,Size,Weight,Letter-Spacing,Transform,Use
--type-app-title,18px,700,0.5px,uppercase,App title
--type-page-title,22px,700,0.3px,capitalize,Page headings
--type-section-title,15px,600,0.2px,capitalize,Card/section headers
--type-subsection,13px,600,0.3px,uppercase,"Sub-headers, labels like ""Validation Tests"""
--type-chart-title,14px,700,0,none,Chart titles (dark: #e8e8e8; light: #1a1a1a)
--type-form-label,12px,500,0.2px,capitalize,Input labels
--type-form-value,13px,400,0,none,Input text
--type-placeholder,13px,400,0,none,"Placeholder (italic, #5a5a5a)"
--type-btn-primary,13px,600,0.5px,uppercase,Primary button text
--type-btn-secondary,12px,500,0.5px,uppercase,Secondary button text
--type-nav-item,13px,500,0.3px,capitalize,Sidebar nav labels
--type-nav-shortcut,14px,700,0,uppercase,Sidebar shortcut letters
--type-explorer-header,11px,700,1.5px,uppercase,Explorer panel headers
--type-tree-item,12px,400,0,none,Tree node labels
--type-tree-item-selected,12px,500,0,none,Selected tree node
--type-status,12px,400,0,none,Status/feedback text (monospace for IDs)
--type-chart-axis-label,11px,400,0.5px,capitalize,Axis labels
--type-chart-tick,10px,400,0,none,Axis tick values
--type-chart-legend,11px,400,0,none,Legend text
--type-description,12px,400,0,none,"Help/description text (italic, #888888)"
--type-badge,10px,500,0.5px,uppercase,Badges (Core/Required/Optional)
--type-code,11–12px,400,0,none,"JSON, hashes, IDs (monospace)"

## 8. Color Palette (CSS Variables)

```css
:root {
  /* Backgrounds */
  --bg-app: #161616;
  --bg-sidebar: #1a1a1a;
  --bg-panel: #1c1c1c;
  --bg-input: #1a1a1a;
  --bg-chart: #161616;
  --bg-chart-light: #ffffff;

  /* Borders */
  --border-subtle: #2a2a2a;
  --border-panel: #2a2a2a;
  --border-input: #333333;
  --border-input-focus: #d4a843;

  /* Accent — Gold */
  --accent: #d4a843;
  --accent-hover: #e8c060;
  --accent-dim: rgba(212, 168, 67, 0.15);
  --control: #a855f7;

  /* Text */
  --text-primary: #f0f0f0;
  --text-secondary: #a0a0a0;
  --text-muted: #a0a0a0;
  --text-disabled: #666666;
  --text-dark: #1a1a1a;

  /* Semantic */
  --success: #4ade80;
  --warning: #fbbf24;
  --error: #ef4444;
  --info: #60a5fa;

  /* Chart colors */
  --chart-path-selected: #4ade80;
  --chart-path-best: #2dd4bf;
  --chart-path-worst: #f87171;
  --chart-path-runup: #60a5fa;
  --chart-path-drawdown: #f59e0b;
  --chart-path-cloud: rgba(136, 136, 136, 0.2);
  --chart-band: rgba(136, 136, 136, 0.15);
  --chart-histogram: rgba(99, 102, 241, 0.7);
  --chart-histogram-gold: rgba(212, 168, 67, 0.7);
  --chart-pdf: #22d3ee;
  --chart-kde: #ffffff;
  --chart-ecdf: #38bdf8;
  --chart-mean: #4ade80;
  --chart-median: #f87171;
  --chart-qq-line: #0d9488;
  --chart-qq-45deg: #dc2626;
  --chart-qq-dots: #7c3aed;
  --chart-ref-curve: #dc2626;
  --chart-comp-curve: #2563eb;
  --chart-hist-light: rgba(45, 212, 191, 0.3);

  /* Slider */
  --slider-track: #3a3a3a;
  --slider-fill: #8b5cf6;
  --slider-thumb: #a78bfa;
  --slider-thumb-hover: #c4b5fd;

  /* Scrollbar */
  --scroll-track: #1a1a1a;
  --scroll-thumb: #3a3a3a;
  --scroll-thumb-hover: #d4a843;

  /* JSON syntax */
  --json-key: #60a5fa;
  --json-string: #4ade80;
  --json-number: #f59e0b;
}
```
