// G13C — true-scale SVG pattern export from the canonical export IR.
//
// SVG coordinates preserve real-world dimensions through explicit unit
// handling: 1 user unit == 1 millimetre by contract, a width/height pair in
// real millimetres, and an `mm`-declared metadata block (data-units +
// application/cut/seam units) so downstream tooling cannot misread scale.
// Arcs are emitted as real SVG arc commands (A rx ry ...), never flattened.
// Pattern y-up is flipped to SVG y-down; pieces lay out left-to-right.
//
// Never "fit to page": the SVG is true scale at 100%; consumers scale only
// through an explicit configuration (not implemented here by design).

import type { Vec2 } from "./geom.js";
import {
  convertM,
  requireExportUnits,
  type ExportIR,
  type ExportPanelIR,
  type ExportUnits,
} from "./export-ir.js";

export interface SvgExportOptions {
  /** Declared output units for the metadata block (default mm). */
  units?: ExportUnits;
  /** Gap between pieces in canonical metres. Default 0.1. */
  gapM?: number;
  /** SVG title (defaults to the IR garment name). */
  title?: string;
}

export interface SvgExportResult {
  svg: string;
  warnings: string[];
  widthUnits: number;
  heightUnits: number;
  units: ExportUnits;
}

function esc(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function fmt(n: number): string {
  const r = Math.round(n * 1000) / 1000;
  return Object.is(r, -0) ? "0" : String(r);
}

function bboxOfPanel(panel: ExportPanelIR): { min: Vec2; max: Vec2 } {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  const scan = (ring: readonly { a: Vec2; b: Vec2; kind: string; center?: Vec2; radiusM?: number }[]): void => {
    for (const e of ring) {
      for (const p of [e.a, e.b]) {
        if (p[0] < minX) minX = p[0];
        if (p[0] > maxX) maxX = p[0];
        if (p[1] < minY) minY = p[1];
        if (p[1] > maxY) maxY = p[1];
      }
      if (e.kind === "arc" && e.center && e.radiusM !== undefined) {
        const c = e.center, r = e.radiusM;
        if (c[0] - r < minX) minX = c[0] - r;
        if (c[0] + r > maxX) maxX = c[0] + r;
        if (c[1] - r < minY) minY = c[1] - r;
        if (c[1] + r > maxY) maxY = c[1] + r;
      }
    }
  };
  scan(panel.sewingEdges);
  if (panel.allowanceEdges) scan(panel.allowanceEdges);
  if (panel.cutEdges) scan(panel.cutEdges);
  for (const h of panel.holes) scan(h);
  for (const g of panel.grainlines) scan([{ kind: "line", a: g.from, b: g.to }]);
  for (const f of panel.folds) scan([{ kind: "line", a: f.a, b: f.b }]);
  for (const i of panel.internals) for (const e of i.edges) scan([{ kind: "line", a: e.a, b: e.b }]);
  if (!Number.isFinite(minX)) return { min: [0, 0], max: [0, 0] };
  return { min: [minX, minY], max: [maxX, maxY] };
}

/** SVG path for a closed ring: M/L for lines, A for arcs (sweep flag: CCW-positive becomes 0 in y-down). */
function ringToPath(ring: readonly { a: Vec2; b: Vec2; kind: string; center?: Vec2; radiusM?: number; a0Rad?: number; sweepRad?: number }[], X: (x: number) => number, Y: (y: number) => number, u: (m: number) => number): string {
  const parts: string[] = [];
  const first = ring[0];
  parts.push(`M ${fmt(X(first.a[0]))} ${fmt(Y(first.a[1]))}`);
  for (const e of ring) {
    if (e.kind === "arc" && e.center && e.radiusM !== undefined && e.sweepRad !== undefined) {
      const rx = fmt(u(e.radiusM)), ry = fmt(u(e.radiusM));
      // y-down flip mirrors orientation: CCW sweep (positive) -> sweep-flag 0.
      const sweepFlag = e.sweepRad > 0 ? 0 : 1;
      parts.push(`A ${rx} ${ry} 0 ${Math.abs(e.sweepRad) > Math.PI ? 1 : 0} ${sweepFlag} ${fmt(X(e.b[0]))} ${fmt(Y(e.b[1]))}`);
    } else {
      parts.push(`L ${fmt(X(e.b[0]))} ${fmt(Y(e.b[1]))}`);
    }
  }
  parts.push("Z");
  return parts.join(" ");
}

/**
 * Render the IR to a true-scale SVG. Deterministic: identical inputs give
 * identical bytes.
 */
export function exportIRToSVG(ir: ExportIR, opts: SvgExportOptions = {}): SvgExportResult {
  const units = requireExportUnits(opts.units ?? "mm", "SVG export");
  if (!ir || !Array.isArray(ir.panels) || ir.panels.length === 0) {
    throw new Error("SVG export needs at least one panel");
  }
  const gapM = opts.gapM ?? 0.1;
  const warnings: string[] = [];
  const u = (metres: number): number => convertM(metres, units);

  // Global layout: left-to-right, y-down after flip. We compute total bounds
  // in user units (mm), with the pattern's max-Y at the top margin.
  const bbs = ir.panels.map(bboxOfPanel);
  let cursorX = 0;
  const origins: number[] = [];
  for (const bb of bbs) {
    origins.push(cursorX - bb.min[0]);
    cursorX += bb.max[0] - bb.min[0] + gapM;
  }
  const totalWM = cursorX - gapM;
  const minY = Math.min(...bbs.map((bb) => bb.min[1]));
  const maxY = Math.max(...bbs.map((bb) => bb.max[1]));
  const margin = u(0.02);
  const heightU = u(maxY - minY) + margin * 2 + u(0.012); // header band for labels
  const widthU = u(totalWM) + margin * 2;
  const headerH = u(0.012);
  const X = (ox: number) => (x: number): number => margin + u(ox + x);
  const Y = (y: number): number => margin + headerH + u(maxY - y);

  const parts: string[] = [];
  parts.push(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${fmt(widthU)}mm" height="${fmt(heightU)}mm" viewBox="0 0 ${fmt(widthU)} ${fmt(heightU)}" font-family="monospace">`,
  );
  parts.push(
    `<metadata data-format="closim-pattern-svg" data-units="mm" data-true-scale="1" ` +
    `data-style="${esc(ir.style.garmentName)}" data-revision="${ir.style.revision}" ` +
    `data-cut-units="${units}" data-app="${esc("closim G13")}"/>`,
  );
  parts.push(
    `<style>.cut{fill:none;stroke:#111;stroke-width:0.4}.sew{fill:none;stroke:#64748b;stroke-width:0.25}` +
    `.allow{fill:none;stroke:#2563eb;stroke-width:0.3;stroke-dasharray:4 2.5}` +
    `.notch{stroke:#b91c1c;stroke-width:0.35}.grain{stroke:#047857;stroke-width:0.35}` +
    `.fold{stroke:#7c3aed;stroke-width:0.3;stroke-dasharray:6 2 1 2}.drill{stroke:#b45309;stroke-width:0.3;fill:none}` +
    `.internal{stroke:#374151;stroke-width:0.25;fill:none}.constr{stroke:#9ca3af;stroke-width:0.15}` +
    `.dim{fill:#111;font-size:2.2px}.lbl{fill:#111;font-size:2.6px}.note{fill:#6b7280;font-size:2.2px}` +
    `.panelname{fill:#111;font-size:3px;font-weight:bold}.labelbox{fill:none;stroke:#111;stroke-width:0.2}</style>`,
  );
  if (opts.title ?? ir.style.garmentName) {
    parts.push(`<text class="panelname" x="${fmt(margin)}" y="${fmt(margin + 2.4)}">${esc(opts.title ?? ir.style.garmentName)} — true scale, 1 unit = 1 mm</text>`);
  }

  for (let pi = 0; pi < ir.panels.length; pi++) {
    const panel = ir.panels[pi];
    const ox = origins[pi];
    const xi = X(ox);
    parts.push(`<g id="panel-${esc(panel.panelId)}" data-panel-name="${esc(panel.name)}" data-cut-quantity="${panel.cutQuantity}">`);
    const cutRing = panel.cutEdges ?? panel.sewingEdges;
    parts.push(`<path class="cut" d="${ringToPath(cutRing, xi, Y, u)}"/>`);
    for (const hole of panel.holes) parts.push(`<path class="cut" d="${ringToPath(hole, xi, Y, u)}"/>`);
    if (panel.cutEdges && panel.cutSource === "allowance") {
      parts.push(`<path class="sew" d="${ringToPath(panel.sewingEdges, xi, Y, u)}"/>`);
    }
    if (panel.allowanceEdges && !(panel.cutEdges && panel.cutSource === "allowance")) {
      parts.push(`<path class="allow" d="${ringToPath(panel.allowanceEdges, xi, Y, u)}"/>`);
    }
    for (const notch of panel.notches) {
      for (const t of notch.ticks) {
        parts.push(`<line class="notch" x1="${fmt(xi(t.from[0]))}" y1="${fmt(Y(t.from[1]))}" x2="${fmt(xi(t.to[0]))}" y2="${fmt(Y(t.to[1]))}"/>`);
      }
    }
    for (const grain of panel.grainlines) {
      parts.push(`<line class="grain" x1="${fmt(xi(grain.from[0]))}" y1="${fmt(Y(grain.from[1]))}" x2="${fmt(xi(grain.to[0]))}" y2="${fmt(Y(grain.to[1]))}"/>`);
      // Arrowhead at the "to" end (8 mm at true scale).
      const dx = grain.to[0] - grain.from[0], dy = grain.to[1] - grain.from[1];
      const len = Math.hypot(dx, dy);
      if (len > 0) {
        const ux = dx / len, uy = dy / len;
        const s = 0.008;
        const p1: Vec2 = [grain.to[0] - s * (ux - uy * 0.5), grain.to[1] - s * (uy + ux * 0.5)];
        const p2: Vec2 = [grain.to[0] - s * (ux + uy * 0.5), grain.to[1] - s * (uy - ux * 0.5)];
        parts.push(`<polygon class="grain" points="${fmt(xi(grain.to[0]))},${fmt(Y(grain.to[1]))} ${fmt(xi(p1[0]))},${fmt(Y(p1[1]))} ${fmt(xi(p2[0]))},${fmt(Y(p2[1]))}"/>`);
      }
    }
    for (const fold of panel.folds) {
      parts.push(`<line class="fold" x1="${fmt(xi(fold.a[0]))}" y1="${fmt(Y(fold.a[1]))}" x2="${fmt(xi(fold.b[0]))}" y2="${fmt(Y(fold.b[1]))}"/>`);
    }
    for (const drill of panel.drills) {
      const rPx = Math.max(1, u(drill.radiusM));
      if (drill.mark === "circle") {
        parts.push(`<circle class="drill" cx="${fmt(xi(drill.pos[0]))}" cy="${fmt(Y(drill.pos[1]))}" r="${fmt(rPx)}"/>`);
      } else if (drill.mark === "cross") {
        parts.push(
          `<line class="drill" x1="${fmt(xi(drill.pos[0] - drill.radiusM))}" y1="${fmt(Y(drill.pos[1]))}" x2="${fmt(xi(drill.pos[0] + drill.radiusM))}" y2="${fmt(Y(drill.pos[1]))}"/>`,
          `<line class="drill" x1="${fmt(xi(drill.pos[0]))}" y1="${fmt(Y(drill.pos[1] - drill.radiusM))}" x2="${fmt(xi(drill.pos[0]))}" y2="${fmt(Y(drill.pos[1] + drill.radiusM))}"/>`,
        );
      } else {
        parts.push(`<circle class="drill" cx="${fmt(xi(drill.pos[0]))}" cy="${fmt(Y(drill.pos[1]))}" r="${fmt(Math.max(0.6, rPx))}"/>`);
      }
    }
    for (const internal of panel.internals) {
      const d = internal.edges.map((e, i2) => `${i2 === 0 ? `M ${fmt(xi(e.a[0]))} ${fmt(Y(e.a[1]))}` : ""} L ${fmt(xi(e.b[0]))} ${fmt(Y(e.b[1]))}`).join(" ");
      parts.push(`<path class="internal" d="${d}"/>`);
    }
    for (const ce of panel.constructionEdges) {
      parts.push(`<line class="constr" x1="${fmt(xi(ce.a[0]))}" y1="${fmt(Y(ce.a[1]))}" x2="${fmt(xi(ce.b[0]))}" y2="${fmt(Y(ce.b[1]))}"/>`);
    }
    for (const ann of panel.annotations) {
      parts.push(`<text class="note" x="${fmt(xi(ann.pos[0]) + 1)}" y="${fmt(Y(ann.pos[1]))}">${esc(ann.note)}</text>`);
    }
    for (const region of panel.labelRegions) {
      const x = xi(region.min[0]), y = Y(region.max[1]);
      const w = u(region.max[0] - region.min[0]), h = u(region.max[1] - region.min[1]);
      parts.push(`<rect class="labelbox" x="${fmt(x)}" y="${fmt(y)}" width="${fmt(w)}" height="${fmt(h)}"/>`);
      Object.entries(region.fields).forEach(([k, v], i) => {
        parts.push(`<text class="lbl" x="${fmt(x + 1)}" y="${fmt(y + 3.2 + i * 3)}">${esc(k)}: ${esc(v)}</text>`);
      });
    }
    if (units !== "mm") {
      // Dimension text is informational; geometry itself is always true scale.
      for (const dim of panel.dimensions) {
        parts.push(`<text class="dim" x="${fmt(xi(dim.mid[0]))}" y="${fmt(Y(dim.mid[1]))}">${fmt(convertM(dim.lengthM, units))} ${esc(units)}</text>`);
      }
    }
    parts.push(
      `<text class="panelname" x="${fmt(xi(bb0(panel).min[0]))}" y="${fmt(Y(bb0(panel).max[1]) - 1.2)}">${esc(panel.name)} ×${panel.cutQuantity}${panel.mirrorPair ? ` (mirror of ${esc(panel.mirrorPair)})` : ""}</text>`,
    );
    parts.push(`</g>`);
  }

  parts.push("</svg>");
  if (units !== "mm") {
    warnings.push(`SVG user units are millimetres by contract; dimension text is in '${units}' — geometry remains true scale`);
  }
  warnings.push("SVG is true scale at 100%: never rescale before printing without a calibration check");
  return {
    svg: parts.join("\n"),
    warnings,
    widthUnits: widthU,
    heightUnits: heightU,
    units: "mm",
  };
}

function bb0(panel: ExportPanelIR): { min: Vec2; max: Vec2 } {
  return bboxOfPanel(panel);
}
