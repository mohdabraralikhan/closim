// G11D — production pattern presentation (technical sheet).
//
// Renders the complete production pattern as a deterministic SVG string:
// cut/sewing boundaries, allowances, notches, grainlines, folds, drills,
// internal lines, annotations, label regions, and dimensions. The SVG is
// pure data (no DOM needed) so export adapters (G11E) reuse it directly.
//
// Pattern y-up is flipped to SVG y-down; panels lay out left-to-right in
// document order with a fixed gap. Nothing here edits the document.

import {
  PatternCadError,
  validatePatternDocument,
  type EntityId,
  type PatternDocument,
} from "../pattern/cad.js";
import { formatLength, type LengthUnit } from "./constraints.js";
import type { Vec2 } from "./geom.js";
import { getLoop, getPanel, getPoint, getSegment, resolveSegment, sampleLoopLocal } from "./queries.js";
import {
  allowanceBoundary,
  cutBoundary,
  type ProductionSet,
} from "./production.js";
import { notchStatus, notchTicks } from "./markings.js";

export interface TechSheetOptions {
  showAllowance?: boolean;
  showNotches?: boolean;
  showGrainlines?: boolean;
  showFolds?: boolean;
  showDrills?: boolean;
  showInternals?: boolean;
  showAnnotations?: boolean;
  showLabels?: boolean;
  showDimensions?: boolean;
  /** Entity ids to highlight (panels, segments, production ids). */
  selectedIds?: string[];
  title?: string;
  units?: LengthUnit;
  /** Pixels per metre. Default 500. */
  scale?: number;
  marginPx?: number;
  /** Gap between panels in metres. Default 0.1. */
  panelGapM?: number;
}

export interface TechSheet {
  svg: string;
  widthPx: number;
  heightPx: number;
  panelIds: string[];
  /** Selected ids that actually exist in the sheet. */
  selectionHits: string[];
}

const DEF = {
  showAllowance: true, showNotches: true, showGrainlines: true, showFolds: true,
  showDrills: true, showInternals: true, showAnnotations: true, showLabels: true,
  showDimensions: true,
};

function esc(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function fmt(n: number): string {
  const r = Math.round(n * 1000) / 1000;
  return Object.is(r, -0) ? "0" : String(r);
}

interface PlacedPanel {
  panelId: string;
  name: string;
  originX: number; // layout offset in metres (pattern space)
  minY: number;
  maxY: number;
  minX: number;
  maxX: number;
}

export function renderTechSheet(
  doc: PatternDocument,
  set: ProductionSet,
  opts: TechSheetOptions = {},
): TechSheet {
  const validation = validatePatternDocument(doc);
  if (!validation.valid) {
    throw new PatternCadError("invalid-document", "cannot present an invalid pattern");
  }
  const o = { ...DEF, ...opts };
  const units: LengthUnit = opts.units ?? "mm";
  const scale = opts.scale ?? 500;
  const marginPx = opts.marginPx ?? 40;
  const gapM = opts.panelGapM ?? 0.1;
  const selected = new Set(opts.selectedIds ?? []);
  const selectionHits: string[] = [];
  /** Selection suffix for inside an existing class attribute (" selected" or ""). */
  const sel = (id: string): string => {
    if (!selected.has(id)) return "";
    selectionHits.push(id);
    return " selected";
  };

  // Layout: panels left-to-right in document order.
  const placed: PlacedPanel[] = [];
  {
    let cursorX = 0;
    for (const panel of doc.panels) {
      const outer = panel.boundaryLoops.find((l) => l.role === "outer");
      if (!outer) continue;
      const pts = sampleLoopLocal(doc, panel.id, outer.id);
      if (pts.length === 0) continue;
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      for (const p of pts) {
        if (p[0] < minX) minX = p[0];
        if (p[0] > maxX) maxX = p[0];
        if (p[1] < minY) minY = p[1];
        if (p[1] > maxY) maxY = p[1];
      }
      placed.push({ panelId: panel.id, name: panel.name, originX: cursorX - minX, minX, maxX, minY, maxY });
      cursorX += maxX - minX + gapM;
    }
  }
  if (placed.length === 0) throw new PatternCadError("invalid-document", "no presentable panels");
  const totalW = placed.reduce((s, p) => Math.max(s, p.originX + p.maxX), 0);
  const topY = Math.max(...placed.map((p) => p.maxY));
  const botY = Math.min(...placed.map((p) => p.minY));
  const widthPx = Math.ceil(totalW * scale + marginPx * 2);
  const heightPx = Math.ceil((topY - botY) * scale + marginPx * 2 + (opts.title ? 28 : 0));

  const X = (panel: PlacedPanel, x: number): number => marginPx + (panel.originX + x) * scale;
  const Y = (y: number): number =>
    marginPx + (opts.title ? 28 : 0) + (topY - y) * scale;
  const pts = (panel: PlacedPanel, ring: Vec2[]): string =>
    ring.map((p) => `${fmt(X(panel, p[0]))},${fmt(Y(p[1]))}`).join(" ");

  const dim = (metres: number): string => {
    const v = formatLength(metres, units);
    return `${(Math.round(v * 10) / 10).toFixed(1)} ${units}`;
  };

  const parts: string[] = [];
  parts.push(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${widthPx}" height="${heightPx}" viewBox="0 0 ${widthPx} ${heightPx}" font-family="monospace">`,
    `<style>.cut{fill:none;stroke:#111;stroke-width:2}.sew{fill:none;stroke:#111;stroke-width:1}.allow{fill:none;stroke:#2563eb;stroke-width:1.2;stroke-dasharray:7 4}.notch{stroke:#b91c1c;stroke-width:2}.grain{stroke:#047857;stroke-width:1.6}.fold{stroke:#7c3aed;stroke-width:1.4;stroke-dasharray:12 4 3 4}.drill{stroke:#b45309;stroke-width:1.4;fill:none}.internal{stroke:#374151;stroke-width:1;fill:none}.dim{fill:#111;font-size:10px}.lbl{fill:#111;font-size:11px}.panelname{fill:#111;font-size:12px;font-weight:bold}.note{fill:#6b7280;font-size:10px}.labelbox{fill:none;stroke:#111}.selected{stroke:#dc2626 !important;stroke-width:3 !important}</style>`,
  );
  if (opts.title) parts.push(`<text x="${marginPx}" y="20" font-size="14">${esc(opts.title)}</text>`);

  for (const panel of placed) {
    const full = getPanel(doc, panel.panelId);
    parts.push(`<g id="panel-${esc(panel.panelId)}">`);
    parts.push(
      `<text class="panelname${sel(panel.panelId)}" x="${fmt(X(panel, panel.minX))}" y="${fmt(Y(panel.maxY) - 6)}">${esc(full.name)}</text>`,
    );
    const outer = full.boundaryLoops.find((l) => l.role === "outer")!;
    const sewing = sampleLoopLocal(doc, panel.panelId, outer.id);

    // Cut boundary: allowance ring when a cut line says so, else sewing.
    const cut = set.cutLines.find((c) => c.panelId === panel.panelId && c.loopId === outer.id);
    let cutRing: Vec2[] | null = null;
    if (cut) {
      try {
        cutRing = cutBoundary(doc, set, cut.id).ring;
      } catch {
        cutRing = null;
      }
    }
    if (cutRing && cut!.source === "allowance") {
      parts.push(`<polygon class="sew" points="${pts(panel, sewing)}"/>`);
      parts.push(`<polygon class="cut${sel(cut!.id)}" points="${pts(panel, cutRing)}"/>`);
    } else {
      parts.push(`<polygon class="cut${sel(panel.panelId)}" points="${pts(panel, sewing)}"/>`);
      if (o.showAllowance) {
        const allowance = set.allowances.find((a) => a.panelId === panel.panelId && a.loopId === outer.id);
        if (allowance) {
          try {
            const derived = allowanceBoundary(doc, set, panel.panelId, outer.id);
            if (derived.ring.length >= 3) {
              parts.push(`<polygon class="allow${sel(allowance.id)}" points="${pts(panel, derived.ring)}"/>`);
            }
          } catch {
            // validation layer owns the error; presentation skips
          }
        }
      }
    }

    // Dimensions on every outer edge.
    if (o.showDimensions) {
      for (const segmentId of outer.segmentIds) {
        const segment = getSegment(doc, segmentId, panel.panelId);
        const r = resolveSegment(doc, segmentId, panel.panelId);
        const mx = (r.start[0] + r.end[0]) / 2;
        const my = (r.start[1] + r.end[1]) / 2;
        const len = segment.kind === "line"
          ? Math.hypot(r.end[0] - r.start[0], r.end[1] - r.start[1])
          : Math.abs(r.arc!.sweep) * r.arc!.radius;
        parts.push(
          `<text class="dim${sel(segmentId)}" x="${fmt(X(panel, mx))}" y="${fmt(Y(my))}" text-anchor="middle">${esc(dim(len))}</text>`,
        );
      }
    }

    // Notches.
    if (o.showNotches) {
      for (const notch of set.notches.filter((e) => e.panelId === panel.panelId)) {
        if (notchStatus(doc, notch) !== "ok") continue;
        try {
          const { ticks } = notchTicks(doc, notch);
          for (const tick of ticks) {
            parts.push(
              `<line class="notch${sel(notch.id)}" x1="${fmt(X(panel, tick.from[0]))}" y1="${fmt(Y(tick.from[1]))}" x2="${fmt(X(panel, tick.to[0]))}" y2="${fmt(Y(tick.to[1]))}"/>`,
            );
          }
        } catch {
          continue;
        }
      }
    }

    // Grainlines with arrowheads both ends.
    if (o.showGrainlines) {
      for (const grain of set.grainlines.filter((e) => e.panelId === panel.panelId)) {
        const dx = grain.to[0] - grain.from[0], dy = grain.to[1] - grain.from[1];
        const len = Math.hypot(dx, dy);
        if (!(len > 0)) continue;
        const ux = dx / len, uy = dy / len;
        const ah = 0.008; // arrowhead size in metres
        const arrow = (tip: Vec2, s: number): string =>
          `${fmt(X(panel, tip[0]))},${fmt(Y(tip[1]))} ` +
          `${fmt(X(panel, tip[0] - s * (ux * ah - uy * ah * 0.5)) )},${fmt(Y(tip[1] - s * (uy * ah + ux * ah * 0.5)))} ` +
          `${fmt(X(panel, tip[0] - s * (ux * ah + uy * ah * 0.5)))},${fmt(Y(tip[1] - s * (uy * ah - ux * ah * 0.5)))}`;
        parts.push(
          `<line class="grain${sel(grain.id)}" x1="${fmt(X(panel, grain.from[0]))}" y1="${fmt(Y(grain.from[1]))}" x2="${fmt(X(panel, grain.to[0]))}" y2="${fmt(Y(grain.to[1]))}"/>`,
          `<polygon class="grain" points="${arrow(grain.to, 1)}"/>`,
          `<polygon class="grain" points="${arrow(grain.from, -1)}"/>`,
        );
      }
    }

    // Folds.
    if (o.showFolds) {
      for (const fold of set.folds.filter((e) => e.panelId === panel.panelId)) {
        parts.push(
          `<line class="fold${sel(fold.id)}" x1="${fmt(X(panel, fold.a[0]))}" y1="${fmt(Y(fold.a[1]))}" x2="${fmt(X(panel, fold.b[0]))}" y2="${fmt(Y(fold.b[1]))}"/>`,
          `<text class="note" x="${fmt(X(panel, (fold.a[0] + fold.b[0]) / 2))}" y="${fmt(Y((fold.a[1] + fold.b[1]) / 2))}">${esc(fold.direction)} ${esc(fold.foldType)}</text>`,
        );
      }
    }

    // Drills.
    if (o.showDrills) {
      for (const drill of set.drills.filter((e) => e.panelId === panel.panelId)) {
        const cx = X(panel, drill.pos[0]), cy = Y(drill.pos[1]);
        const rPx = Math.max(2.5, drill.radiusM * scale);
        if (drill.mark === "circle") {
          parts.push(`<circle class="drill${sel(drill.id)}" cx="${fmt(cx)}" cy="${fmt(cy)}" r="${fmt(rPx)}"/>`);
        } else if (drill.mark === "cross") {
          parts.push(
            `<line class="drill${sel(drill.id)}" x1="${fmt(cx - rPx)}" y1="${fmt(cy)}" x2="${fmt(cx + rPx)}" y2="${fmt(cy)}"/>`,
            `<line class="drill" x1="${fmt(cx)}" y1="${fmt(cy - rPx)}" x2="${fmt(cx)}" y2="${fmt(cy + rPx)}"/>`,
          );
        } else {
          parts.push(`<circle class="drill${sel(drill.id)}" cx="${fmt(cx)}" cy="${fmt(cy)}" r="2" fill="#b45309"/>`);
        }
      }
    }

    // Internal lines.
    if (o.showInternals) {
      for (const line of set.internals.filter((e) => e.panelId === panel.panelId)) {
        parts.push(`<polyline class="internal${sel(line.id)}" points="${pts(panel, line.points)}"/>`);
        if (line.label) {
          const mid = line.points[Math.floor(line.points.length / 2)];
          parts.push(`<text class="note" x="${fmt(X(panel, mid[0]))}" y="${fmt(Y(mid[1]))}">${esc(line.label)}</text>`);
        }
      }
    }

    // Annotations.
    if (o.showAnnotations) {
      for (const ann of set.annotations.filter((e) => e.panelId === panel.panelId)) {
        parts.push(
          `<circle cx="${fmt(X(panel, ann.pos[0]))}" cy="${fmt(Y(ann.pos[1]))}" r="2" fill="#6b7280"/>`,
          `<text class="note${sel(ann.id)}" x="${fmt(X(panel, ann.pos[0]) + 5)}" y="${fmt(Y(ann.pos[1]))}">${esc(ann.note)}</text>`,
        );
      }
    }

    // Label regions.
    if (o.showLabels) {
      for (const region of set.labelRegions.filter((e) => e.panelId === panel.panelId)) {
        const x = X(panel, region.min[0]), y = Y(region.max[1]);
        const w = (region.max[0] - region.min[0]) * scale;
        const h = (region.max[1] - region.min[1]) * scale;
        const entries = Object.entries(region.fields);
        parts.push(`<rect class="labelbox${sel(region.id)}" x="${fmt(x)}" y="${fmt(y)}" width="${fmt(w)}" height="${fmt(h)}"/>`);        entries.forEach(([k, v], i) => {
          parts.push(`<text class="lbl" x="${fmt(x + 4)}" y="${fmt(y + 16 + i * 14)}">${esc(k)}: ${esc(v)}</text>`);
        });
      }
    }

    parts.push(`</g>`);
  }

  return {
    svg: parts.join("\n"),
    widthPx,
    heightPx,
    panelIds: placed.map((p) => p.panelId),
    selectionHits,
  };
}
