# South India Open Data Integration Engine — Design System (DESIGN.md)

## 1. Overview & Aesthetic Intent
The design system for the **South India Open Data Integration Engine with EV Charging Coverage Explorer** is engineered for high data density, maximum legibility, and uncompromising statistical honesty. It delivers a modern, high-contrast data-product aesthetic built for academic defense and production inspection.

> **Stitch Fallback Note**: In accordance with project instructions (*"If Stitch fails, fall back to a clean Tailwind design and say so in README"*), the Stitch generation timed out at the remote backend. The design system therefore utilizes a deterministic, accessible Tailwind CSS implementation coupled with Plotly charts and vanilla JavaScript.

---

## 2. Color Palette & Semantics

The palette operates on a sleek, dark-mode foundation with purpose-driven semantic accents:

### Surfaces & Backgrounds
| Token | Hex | Tailwind Class | Usage |
|---|---|---|---|
| `canvas-base` | `#020617` | `bg-slate-950` | Primary application canvas background |
| `surface-card` | `#0f172a` | `bg-slate-900` | Metric cards, table containers, modal trays |
| `surface-subtle` | `#1e293b` | `bg-slate-800` | Table headers, dropdown selects, active tabs |
| `border-subtle` | `#1e293b` | `border-slate-800` | Hairline dividers, card outlines |
| `border-focus` | `#334155` | `border-slate-700` | Input focus rings, card hover outlines |

### Semantic Brand & Status Accents
| Token | Hex | Tailwind Class | Usage |
|---|---|---|---|
| `accent-emerald` | `#10b981` | `text-emerald-400`, `bg-emerald-950` | Verified data badge, reconciled audit status, real data pulse |
| `accent-indigo` | `#6366f1` | `text-indigo-400`, `bg-indigo-600` | Primary buttons, active navigation, graph nodes |
| `accent-amber` | `#f59e0b` | `text-amber-400`, `bg-amber-950` | Missing data warnings, review queue uncertainty band |
| `accent-rose` | `#f43f5e` | `text-rose-400`, `bg-rose-950` | Suspected errors, unresolved candidate matches |
| `accent-sky` | `#38bdf8` | `text-sky-400`, `bg-sky-950` | Secondary links, data.gov.in external portals |

---

## 3. Typography Hierarchy

The type system pairs modern geometric sans-serif **Inter** for UI clarity with **JetBrains Mono** for data hashes, coordinates, counts, and code.

| Style Role | Font Family | Size | Weight | Line Height | Tracking | Usage |
|---|---|---|---|---|---|---|
| `display-hero` | Inter | 24px (1.5rem) | 700 (Bold) | 1.2 | -0.025em | Main dashboard headers |
| `headline-title` | Inter | 18px (1.125rem) | 600 (Semibold) | 1.3 | -0.015em | Section titles, modal headers |
| `body-primary` | Inter | 14px (0.875rem) | 400 (Regular) | 1.5 | 0.0em | Narrative descriptions, labels |
| `caption-subtle` | Inter | 12px (0.75rem) | 500 (Medium) | 1.4 | +0.01em | Provenance strips, table metadata |
| `data-mono` | JetBrains Mono | 12px (0.75rem) | 400 (Regular) | 1.4 | 0.0em | Counts, resource IDs, pincodes |
| `code-hash` | JetBrains Mono | 11px (0.6875rem) | 400 (Regular) | 1.2 | +0.02em | SHA-256 checksums, URL queries |

---

## 4. Spacing & Grid System

- **Base Rhythm**: 4px / 8px incremental scale (`0.25rem` = 4px, `0.5rem` = 8px, `1rem` = 16px, `1.5rem` = 24px, `2rem` = 32px).
- **Responsive Breakpoints**:
  - `mobile`: `< 768px` (single-column card stacks, horizontal table scroll).
  - `tablet`: `768px - 1024px` (2-column metric cards, collapsible nav).
  - `desktop`: `> 1024px` (4-column metric grids, full table expanders, split-view graph visualizers).
- **Elevation Philosophy**: Zero artificial drop shadows. Visual depth is established purely through planar tone stepping (`slate-950` -> `slate-900` -> `slate-800`) and 1px crisp borders.

---

## 5. Mandatory UI Elements

### 5.1 Permanent Real Data Banner
Fixed at the top of every screen:
```html
<header class="bg-emerald-950/80 border-b border-emerald-500/30 px-6 py-2.5 flex items-center justify-between text-xs font-mono text-emerald-300">
    <div class="flex items-center space-x-2">
        <span class="inline-block w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
        <span class="font-bold text-emerald-200">REAL DATA</span>
        <span>—</span>
        <span>https://data.gov.in/ (Official Open Government Data)</span>
    </div>
    <div>Scope: TN, KA, KL, AP, TG, PY</div>
</header>
```

### 5.2 Provenance Strip
Every card, table, or visual chart includes a provenance footer stating:
- **Ministry / Source**: e.g., *Department of Posts* or *Ministry of Rural Development*
- **Resource ID & Hash**: Tracing directly to `manifest.json`
- **Missing Dataset State**: If data is missing (e.g. Telangana in e-Vahan 2023), render `<span class="text-amber-400 font-mono">Not available: dataset missing</span>`.

---

## 6. Seven Screen Architecture
1. **Screen 1: Overview Dashboard**
   - KPI metrics, active territory status, database connectivity, table count summary.
2. **Screen 2: Data Sources & Cleaning**
   - Dataset catalog with live links, SHA-256 checksums, pipeline transformation logs, summary reconciliation audits, and column profiling.
3. **Screen 3: Dataset Integration & Graph**
   - Interactive NetworkX-driven integration mart graph (nodes & edges), grain hierarchy visualizer, district-to-state upward aggregation schema.
4. **Screen 4: Entity Matching & Review Queue**
   - ML model vs 3 baselines comparison, candidate matching score inspection, live interactive review queue allowing students to label candidates (1, 2, 3, none, unsure) and save to `data/labels/hand_labels.csv`.
5. **Screen 5: EV Charging Coverage Explorer**
   - State-grain side-by-side comparison (PCS 2024 vs OMC 2025 with definition warnings), EV share, chargers per 1,000 EVs (with date-gap warnings and missing Telangana state notice), 16 highway corridors.
6. **Screen 6: Analytics & Suspected Error Reports**
   - Tabular view of all 5 suspected portal anomalies, data quality completeness metrics, release package zip export.
7. **Screen 7: Model Evaluation & Benchmarks**
   - Precision, recall, F1-scores, state-grouped cross-validation, Wilson confidence intervals, and active-learning curves.
