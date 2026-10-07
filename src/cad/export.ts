// G11E — production export architecture and QA.
//
// Pipeline:
//
//   Native document (PatternDocument + ProductionSet + Seam[])
//     -> productionReadiness gate (must be READY_FOR_EXPORT, never warnings)
//     -> ExportIR (deterministic intermediate representation, metres)
//     -> format adapter (JSON round-trippable / DXF R12 minimal)
//
// Format choice (per-track instruction to inspect first): the repo has no
// export layer at all. DXF R12 (LINE/CIRCLE/TEXT on named layers) is the
// minimal genuinely-manufacturing-readable path every cutting/plotting tool
// accepts; JSON carries the lossless round-trip. Full AAMA/DXF, PDF, and
// nesting stay out of scope and are reported as unsupported, not faked.

import {
  PatternCadError,
  validatePatternDocument,
  type EntityId,
  type PatternDocument,
} from "../pattern/cad.js";
import type { Vec2 } from "./geom.js";
import { getPanel, measurePanel, resolveSegment, sampleLoopLocal } from "./queries.js";
import type { Seam } from "../garment/sewing.js";
import {
  allowanceBoundary,
  cutBoundary,
  notchFrame,
  type OffsetIssue,
  type ProductionSet,
} from "./production.js";
import { crossGrainline, notchStatus, notchTicks } from "./markings.js";
import {
  productionReadiness,
  seamSideLengths,
  type ReadinessReport,
  type ReadinessTolerances,
} from "./readiness.js";

export const EXPORT_IR_VERSION = 1;

export interface ExportNotchIR {
  id: string;
  kind: string;
  pos: Vec2;
  ticks: Array<{ from: Vec2; to: Vec2 }>;
  depthM: number;
}

export interface ExportPanelIR {
  panelId: string;
  name: string;
  materialId: string;
  sewing: Vec2[];
  holes: Vec2[][];
  cut: Vec2[] | null;
  cutSource: "sewing" | "allowance" | null;
  allowance: Vec2[] | null;
  allowanceIssues: OffsetIssue[];
  notches: ExportNotchIR[];
  grainlines: Array<{ id: string; from: Vec2; to: Vec2 }>;
  crossGrains: Array<{ from: Vec2; to: Vec2 }>;
  folds: Array<{ id: string; a: Vec2; b: Vec2; direction: string; foldType: string }>;
  drills: Array<{ id: string; pos: Vec2; mark: string; radiusM: number }>;
  internals: Array<{ id: string; points: Vec2[]; kind: string; label?: string }>;
  annotations: Array<{ id: string; pos: Vec2; note: string }>;
  labelRegions: Array<{ id: string; min: Vec2; max: Vec2; fields: Record<string, string> }>;
  dimensions: Array<{ segmentId: string; lengthM: number; mid: Vec2 }>;
  cutQuantity: number;
  areaM2: number;
  /** True when any sampled ring came from authored arcs (DXF facets them). */
  facetedArcs: boolean;
}

export interface ExportSeamIR {
  id: string;
  sideA: { panelId: string; loopId: string; segmentIds: string[] };
  sideB: { panelId: string; loopId: string; segmentIds: string[] };
  stitchCount: number;
  lengthAM: number;
  lengthBM: number;
}

export interface ExportIR {
  format: "closim-production-ir";
  version: typeof EXPORT_IR_VERSION;
  units: "m";
  garment: { documentId: string; name: string };
  panels: ExportPanelIR[];
  seams: ExportSeamIR[];
  readiness: { state: string; errorCount: number; warningCount: number };
}

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

/** Build the deterministic intermediate representation (no validation gate here). */
export function buildExportIR(
  doc: PatternDocument,
  seams: readonly Seam[],
  set: ProductionSet,
  opts: { garmentName?: string; readiness?: ReadinessReport } = {},
): ExportIR {
  const kernel = validatePatternDocument(doc);
  if (!kernel.valid) {
    throw new PatternCadError("invalid-document", "cannot build export IR from an invalid pattern");
  }
  const panels: ExportPanelIR[] = doc.panels.map((panel) => {
    const outer = panel.boundaryLoops.find((l) => l.role === "outer")!;
    const facetedArcs =
      outer.segmentIds.some((id) => doc.segments.find((s) => s.id === id)?.kind === "arc") ||
      panel.boundaryLoops
        .filter((l) => l.role === "hole")
        .some((l) => l.segmentIds.some((id) => doc.segments.find((s) => s.id === id)?.kind === "arc"));
    const sewing = sampleLoopLocal(doc, panel.id, outer.id);
    const holes = panel.boundaryLoops
      .filter((l) => l.role === "hole")
      .map((l) => sampleLoopLocal(doc, panel.id, l.id));
    const cut = set.cutLines.find((c) => c.panelId === panel.id && c.loopId === outer.id) ?? null;
    let cutRing: Vec2[] | null = null;
    let cutSource: ExportPanelIR["cutSource"] = null;
    if (cut) {
      cutSource = cut.source;
      try {
        cutRing = cutBoundary(doc, set, cut.id).ring;
      } catch {
        cutRing = null;
      }
    }
    const allowance = set.allowances.find((a) => a.panelId === panel.id && a.loopId === outer.id) ?? null;
    let allowanceRing: Vec2[] | null = null;
    let allowanceIssues: OffsetIssue[] = [];
    if (allowance) {
      try {
        const derived = allowanceBoundary(doc, set, panel.id, outer.id);
        allowanceRing = derived.ring.length >= 3 ? derived.ring : null;
        allowanceIssues = derived.issues;
      } catch {
        allowanceRing = null;
      }
    }
    const notches: ExportNotchIR[] = [];
    for (const notch of set.notches.filter((e) => e.panelId === panel.id)) {
      if (notchStatus(doc, notch) !== "ok") continue;
      try {
        const { ticks } = notchTicks(doc, notch);
        const frame = notchFrame(doc, notch);
        notches.push({ id: notch.id, kind: notch.kind, pos: frame.pos, ticks, depthM: notch.depthM });
      } catch {
        continue;
      }
    }
    const meta = set.panelMeta.find((m) => m.panelId === panel.id) ?? null;
    let areaM2 = 0;
    try {
      areaM2 = measurePanel(doc, panel.id).area;
    } catch {
      areaM2 = 0;
    }
    return {
      panelId: panel.id,
      name: panel.name,
      materialId: panel.materialId,
      sewing,
      holes,
      cut: cutRing,
      cutSource,
      allowance: allowanceRing,
      allowanceIssues,
      notches,
      grainlines: set.grainlines
        .filter((e) => e.panelId === panel.id)
        .map((e) => ({ id: e.id, from: [...e.from] as Vec2, to: [...e.to] as Vec2 })),
      crossGrains: set.grainlines
        .filter((e) => e.panelId === panel.id)
        .map((e) => crossGrainline({ id: e.id, panelId: e.panelId, from: e.from, to: e.to })),
      folds: set.folds
        .filter((e) => e.panelId === panel.id)
        .map((e) => ({ id: e.id, a: [...e.a] as Vec2, b: [...e.b] as Vec2, direction: e.direction, foldType: e.foldType })),
      drills: set.drills
        .filter((e) => e.panelId === panel.id)
        .map((e) => ({ id: e.id, pos: [...e.pos] as Vec2, mark: e.mark, radiusM: e.radiusM })),
      internals: set.internals
        .filter((e) => e.panelId === panel.id)
        .map((e) => ({
          id: e.id, points: e.points.map((p) => [...p] as Vec2), kind: e.kind,
          ...(e.label !== undefined ? { label: e.label } : {}),
        })),
      annotations: set.annotations
        .filter((e) => e.panelId === panel.id)
        .map((e) => ({ id: e.id, pos: [...e.pos] as Vec2, note: e.note })),
      labelRegions: set.labelRegions
        .filter((e) => e.panelId === panel.id)
        .map((e) => ({ id: e.id, min: [...e.min] as Vec2, max: [...e.max] as Vec2, fields: { ...e.fields } })),
      dimensions: outer.segmentIds.map((segmentId) => {
        const r = resolveSegment(doc, segmentId, panel.id);
        const mid: Vec2 = [(r.start[0] + r.end[0]) / 2, (r.start[1] + r.end[1]) / 2];
        const lengthM = r.arc
          ? Math.abs(r.arc.sweep) * r.arc.radius
          : Math.hypot(r.end[0] - r.start[0], r.end[1] - r.start[1]);
        return { segmentId, lengthM, mid };
      }),
      cutQuantity: meta?.cutQuantity ?? 0,
      areaM2,
      facetedArcs,
    };
  });
  // Fill dimensions without the placeholder above (kept simple + explicit).
  for (const p of panels) {
    const panel = getPanel(doc, p.panelId);
    const outer = panel.boundaryLoops.find((l) => l.role === "outer")!;
    p.dimensions = outer.segmentIds.map((segmentId) => {
      const r = resolveSegment(doc, segmentId, panel.id);
      const mid: Vec2 = [(r.start[0] + r.end[0]) / 2, (r.start[1] + r.end[1]) / 2];
      const lengthM = r.arc
        ? Math.abs(r.arc.sweep) * r.arc.radius
        : Math.hypot(r.end[0] - r.start[0], r.end[1] - r.start[1]);
      return { segmentId, lengthM, mid };
    });
  }
  const readiness = opts.readiness ?? null;
  return {
    format: "closim-production-ir",
    version: EXPORT_IR_VERSION,
    units: "m",
    garment: { documentId: doc.id, name: doc.name },
    panels,
    seams: seams.map((seam) => {
      let lengthAM = 0, lengthBM = 0;
      try {
        ({ lengthAM, lengthBM } = seamSideLengths(doc, seam));
      } catch {
        lengthAM = 0;
        lengthBM = 0;
      }
      return {
        id: seam.id,
        sideA: clone(seam.sideA),
        sideB: clone(seam.sideB),
        stitchCount: seam.stitchCount,
        lengthAM,
        lengthBM,
      };
    }),
    readiness: readiness
      ? { state: readiness.state, errorCount: readiness.errorCount, warningCount: readiness.warningCount }
      : { state: "UNCHECKED", errorCount: 0, warningCount: 0 },
  };
}

// ---------------------------------------------------------------------------
// Canonical JSON (lossless round-trip)
// ---------------------------------------------------------------------------

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value !== null && typeof value === "object") {
    const record = value as Record<string, unknown>;
    return `{${Object.keys(record).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`).join(",")}}`;
  }
  if (typeof value === "number" && Object.is(value, -0)) return "0";
  return JSON.stringify(value);
}

export function exportIRToJSON(ir: ExportIR): string {
  return canonicalJson(ir);
}

export function importIRFromJSON(json: string): ExportIR {
  let parsed: unknown;
  try {
    parsed = JSON.parse(json);
  } catch {
    throw new PatternCadError("invalid-document", "export JSON does not parse");
  }
  const ir = parsed as ExportIR;
  if (!ir || ir.format !== "closim-production-ir" || ir.version !== EXPORT_IR_VERSION || !Array.isArray(ir.panels)) {
    throw new PatternCadError("invalid-document", "export JSON has the wrong shape or version");
  }
  return clone(ir);
}

/** Structural comparison for round-trip QA (exact; IR numbers are deterministic). */
export function compareIR(a: ExportIR, b: ExportIR): string[] {
  const diffs: string[] = [];
  const walk = (path: string, x: unknown, y: unknown): void => {
    if (JSON.stringify(x) === JSON.stringify(y)) return;
    if (Array.isArray(x) && Array.isArray(y)) {
      if (x.length !== y.length) {
        diffs.push(`${path}: length ${x.length} vs ${y.length}`);
        return;
      }
      x.forEach((v, i) => walk(`${path}[${i}]`, v, y[i]));
      return;
    }
    if (x !== null && y !== null && typeof x === "object" && typeof y === "object") {
      const keys = new Set([...Object.keys(x as object), ...Object.keys(y as object)]);
      for (const k of [...keys].sort()) {
        walk(`${path}.${k}`, (x as Record<string, unknown>)[k], (y as Record<string, unknown>)[k]);
      }
      return;
    }
    diffs.push(`${path}: ${JSON.stringify(x)} vs ${JSON.stringify(y)}`);
  };
  walk("$", a, b);
  return diffs;
}

// ---------------------------------------------------------------------------
// DXF R12 minimal writer (LINE / CIRCLE / TEXT on named layers, millimetres)
// ---------------------------------------------------------------------------

const DXF_LAYERS = ["CUT", "SEW", "ALLOWANCE", "NOTCH", "GRAIN", "FOLD", "DRILL", "INTERNAL", "LABEL", "DIM"] as const;

function dxfEscape(text: string): string {
  return text.replace(/[\r\n]+/g, " ");
}

function fmtMM(metres: number): string {
  return (metres * 1000).toFixed(4);
}

export interface DXFExport {
  dxf: string;
  warnings: string[];
  entityCount: number;
}

/**
 * Minimal DXF R12 subset writer. Curves are faceted polylines (reported);
 * multi-field labels flatten to one TEXT per field (reported); everything
 * else maps 1:1 onto layers. Panels lay out left-to-right like the tech
 * sheet so pieces never overlap.
 */
export function exportIRToDXF(ir: ExportIR, gapM = 0.1): DXFExport {
  const warnings: string[] = [];
  const lines: string[] = [];
  let entityCount = 0;
  const line = (x1: number, y1: number, x2: number, y2: number, layer: string): void => {
    lines.push(
      "0", "LINE", "8", layer,
      "10", fmtMM(x1), "20", fmtMM(y1), "30", "0.0000",
      "11", fmtMM(x2), "21", fmtMM(y2), "31", "0.0000",
    );
    entityCount++;
  };
  const circle = (x: number, y: number, rM: number, layer: string): void => {
    lines.push("0", "CIRCLE", "8", layer, "10", fmtMM(x), "20", fmtMM(y), "30", "0.0000", "40", fmtMM(rM));
    entityCount++;
  };
  const text = (x: number, y: number, heightM: number, value: string, layer: string): void => {
    lines.push(
      "0", "TEXT", "8", layer,
      "10", fmtMM(x), "20", fmtMM(y), "30", "0.0000",
      "40", fmtMM(heightM), "1", dxfEscape(value),
    );
    entityCount++;
  };
  const polyline = (ring: Vec2[], ox: number, layer: string, close = true): void => {
    for (let i = 0; i < ring.length; i++) {
      const a = ring[i], b = ring[(i + 1) % ring.length];
      if (!close && i === ring.length - 1) break;
      line(a[0] + ox, a[1], b[0] + ox, b[1], layer);
    }
  };
  let faceted = false;
  let cursorX = 0;
  for (const panel of ir.panels) {
    // Panel-local bbox for layout.
    let minX = Infinity, maxX = -Infinity;
    for (const p of panel.sewing) {
      if (p[0] < minX) minX = p[0];
      if (p[0] > maxX) maxX = p[0];
    }
    if (!Number.isFinite(minX)) continue;
    const ox = cursorX - minX;
    const cutRing = panel.cut ?? panel.sewing;
    polyline(cutRing, ox, "CUT");
    if (panel.cut && panel.cutSource === "allowance") polyline(panel.sewing, ox, "SEW");
    if (panel.allowance && !(panel.cut && panel.cutSource === "allowance")) polyline(panel.allowance, ox, "ALLOWANCE");
    for (const hole of panel.holes) polyline(hole, ox, "CUT");
    for (const notch of panel.notches) {
      for (const tick of notch.ticks) {
        line(tick.from[0] + ox, tick.from[1], tick.to[0] + ox, tick.to[1], "NOTCH");
      }
      text(notch.pos[0] + ox, notch.pos[1], 0.003, `notch:${notch.id}`, "NOTCH");
    }
    for (const grain of panel.grainlines) {
      line(grain.from[0] + ox, grain.from[1], grain.to[0] + ox, grain.to[1], "GRAIN");
    }
    for (const fold of panel.folds) {
      line(fold.a[0] + ox, fold.a[1], fold.b[0] + ox, fold.b[1], "FOLD");
      text((fold.a[0] + fold.b[0]) / 2 + ox, (fold.a[1] + fold.b[1]) / 2, 0.003, `${fold.direction} ${fold.foldType}`, "FOLD");
    }
    for (const drill of panel.drills) {
      if (drill.mark === "circle") circle(drill.pos[0] + ox, drill.pos[1], drill.radiusM, "DRILL");
      else if (drill.mark === "cross") {
        line(drill.pos[0] - drill.radiusM + ox, drill.pos[1], drill.pos[0] + drill.radiusM + ox, drill.pos[1], "DRILL");
        line(drill.pos[0] + ox, drill.pos[1] - drill.radiusM, drill.pos[0] + ox, drill.pos[1] + drill.radiusM, "DRILL");
      } else circle(drill.pos[0] + ox, drill.pos[1], 0.001, "DRILL");
    }
    for (const internal of panel.internals) {
      polyline(internal.points, ox, "INTERNAL", false);
      if (internal.label) text(internal.points[0][0] + ox, internal.points[0][1], 0.003, internal.label, "INTERNAL");
    }
    for (const ann of panel.annotations) {
      circle(ann.pos[0] + ox, ann.pos[1], 0.001, "LABEL");
      text(ann.pos[0] + ox, ann.pos[1], 0.0025, ann.note, "LABEL");
    }
    for (const region of panel.labelRegions) {
      const corners: Vec2[] = [region.min, [region.max[0], region.min[1]], region.max, [region.min[0], region.max[1]]];
      polyline(corners, ox, "LABEL");
      warnings.push(`label fields flattened to TEXT for region '${region.id}'`);
      Object.entries(region.fields).forEach(([k, v], i) => {
        text(region.min[0] + ox, region.max[1] - 0.005 * (i + 1), 0.0035, `${k}: ${v}`, "LABEL");
      });
    }
    for (const dim of panel.dimensions) {
      text(dim.mid[0] + ox, dim.mid[1], 0.0025, `${(dim.lengthM * 1000).toFixed(1)} mm`, "DIM");
    }
    text(minX + ox, panel.sewing.reduce((s, p) => Math.max(s, p[1]), -Infinity), 0.004, `${panel.name} x${panel.cutQuantity}`, "LABEL");
    if (panel.allowanceIssues.length > 0 || panel.facetedArcs) faceted = true;
    cursorX += maxX - minX + gapM;
  }
  if (faceted) warnings.push("arc edges faceted to sagitta-bounded polylines in DXF output");
  warnings.push("DXF subset: R12 LINE/CIRCLE/TEXT only; no SPLINE, HATCH, MTEXT, or paper-space layout");
  const header = [
    "0", "SECTION", "2", "HEADER",
    "9", "$INSUNITS", "70", "4",
    "0", "ENDSEC",
    "0", "SECTION", "2", "TABLES",
    "0", "TABLE", "2", "LAYER", "70", String(DXF_LAYERS.length),
  ];
  for (const layer of DXF_LAYERS) {
    header.push("0", "LAYER", "2", layer, "70", "0", "62", "7", "6", "CONTINUOUS");
  }
  header.push("0", "ENDTAB", "0", "ENDSEC", "0", "SECTION", "2", "ENTITIES");
  const footer = ["0", "ENDSEC", "0", "EOF"];
  return { dxf: [...header, ...lines, ...footer].join("\n") + "\n", warnings, entityCount };
}

// ---------------------------------------------------------------------------
// Gated production package (the only path to an external file)
// ---------------------------------------------------------------------------

export interface ProductionPackage {
  ir: ExportIR;
  readiness: ReadinessReport;
  json: string;
  dxf: DXFExport;
}

/**
 * Validate -> gate -> IR -> adapters. Throws unless the report is
 * READY_FOR_EXPORT: warnings never silently become files.
 */
export function exportProductionPackage(
  doc: PatternDocument,
  seams: readonly Seam[],
  set: import("./production.js").ProductionSet,
  opts: { garmentName?: string; tolerances?: ReadinessTolerances } = {},
): ProductionPackage {
  const readiness = productionReadiness(doc, seams, set, opts.tolerances);
  if (readiness.state !== "READY_FOR_EXPORT") {
    const first = readiness.diagnostics[0];
    throw new PatternCadError(
      "invalid-document",
      `export refused (state=${readiness.state}): ${first.severity} ${first.code}: ${first.message}`,
      first.entityId,
    );
  }
  const ir = buildExportIR(doc, seams, set, { garmentName: opts.garmentName, readiness });
  // Seam lengths in the IR come from the authoritative correspondence.
  ir.seams = seams.map((seam) => {
    const row = readiness.seams.find((r) => r.seamId === seam.id)!;
    return {
      id: seam.id,
      sideA: clone(seam.sideA),
      sideB: clone(seam.sideB),
      stitchCount: seam.stitchCount,
      lengthAM: row.lengthAM,
      lengthBM: row.lengthBM,
    };
  });
  return { ir, readiness, json: exportIRToJSON(ir), dxf: exportIRToDXF(ir) };
}

export type { EntityId };
