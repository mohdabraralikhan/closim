// G13A — canonical production export intermediate representation (IR v2).
//
// Pipeline (the ONLY sanctioned path to any external file):
//
//   Native garment document (PatternDocument + ProductionSet + Seam[])
//     -> production gate (READY_FOR_EXPORT, or explicit mode override)
//     -> ExportIR2 (deterministic, canonical metres)
//     -> format adapter (DXF / SVG / PDF / package)
//
// Critical rule: the IR is DERIVED. The native document remains the single
// source of truth; the IR must never be round-tripped back into editing.
// Importers of IR JSON exist only for QA comparisons.
//
// Geometry contract: every boundary ring is stored as EXACT edge records
// (line: endpoints; arc: centre/radius/angles/sweep) — never flattened into
// polylines by the IR. Sampling is a presentation decision owned by each
// adapter, which must declare it (DXF adapters emit ARC entities; SVG emits
// real arc commands; PDF tiles sample with a declared sagitta bound).
//
// Units contract: the canonical unit is metres (same as the pattern kernel).
// Every adapter converts explicitly at export time and must declare the
// emitted unit both structurally (DXF $INSUNITS, SVG user-unit + metadata)
// and textually. Ambiguous export configs (unit absent from its declared
// table, or a profile without a unit mapping) throw instead of guessing.
//
// Determinism: identical native document + identical export configuration
// produces identical JSON bytes. No timestamps, no randomness, no absolute
// paths, sorted map keys, -0 normalized.

import {
  PatternCadError,
  validatePatternDocument,
  type PatternDocument,
} from "../pattern/cad.js";
import type { Vec2 } from "./geom.js";
import { signedArea } from "./geom.js";
import { formatLength, parseLengthToM, type LengthUnit } from "./constraints.js";
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

// ---------------------------------------------------------------------------
// Schema versioning (independent from the native document format)
// ---------------------------------------------------------------------------

/** Bump on any IR shape change; adapters pin the versions they accept. */
export const EXPORT_IR_VERSION = 2;
export const EXPORT_IR_FORMAT = "closim-production-ir";

// ---------------------------------------------------------------------------
// Units
// ---------------------------------------------------------------------------

/** Units a file adapter may emit; the IR itself is always canonical metres. */
export type ExportUnits = LengthUnit;

const MM_PER_EXPORT_UNIT: Record<ExportUnits, number> = { mm: 1, cm: 10, m: 1000, in: 25.4 };

/**
 * Explicit unit conversion at export time. `unit` must be declared by the
 * caller's export configuration; ambiguity (unknown unit) throws rather
 * than guessing.
 */
export function convertM(valueM: number, unit: ExportUnits): number {
  const f = MM_PER_EXPORT_UNIT[unit];
  if (!(f > 0)) throw new PatternCadError("invalid-transform", `unknown export unit '${String(unit)}'`);
  if (!Number.isFinite(valueM)) throw new PatternCadError("invalid-transform", "cannot convert a non-finite length");
  return valueM * 1000 / f;
}

/** Validate a user-provided units declaration: unknown or missing = error. */
export function requireExportUnits(unit: ExportUnits | undefined, context: string): ExportUnits {
  if (unit === undefined) {
    throw new PatternCadError("invalid-transform", `${context}: units are undeclared; refuse to guess`);
  }
  if (!(unit in MM_PER_EXPORT_UNIT)) {
    throw new PatternCadError("invalid-transform", `${context}: unknown unit '${String(unit)}'`);
  }
  return unit;
}

/** Metres -> declared unit, formatted deterministically. */
export function formatInUnits(valueM: number, unit: ExportUnits, decimals: number): string {
  return convertM(valueM, unit).toFixed(decimals);
}

export type { LengthUnit };

// ---------------------------------------------------------------------------
// Exact geometry records
// ---------------------------------------------------------------------------

export interface ExportLineEdge {
  kind: "line";
  a: Vec2;
  b: Vec2;
}

export interface ExportArcEdge {
  kind: "arc";
  center: Vec2;
  radiusM: number;
  /** Start angle (radians) measured from the centre, y-up. */
  a0Rad: number;
  /** Signed sweep in radians (positive CCW), strictly within (-2π, 2π). */
  sweepRad: number;
  a: Vec2;
  b: Vec2;
}

export type ExportEdge = ExportLineEdge | ExportArcEdge;

function lineEdge(a: Vec2, b: Vec2): ExportLineEdge {
  return { kind: "line", a: [...a] as Vec2, b: [...b] as Vec2 };
}

function arcEdge(center: Vec2, radius: number, a0: number, sweep: number, a: Vec2, b: Vec2): ExportArcEdge {
  return {
    kind: "arc",
    center: [...center] as Vec2,
    radiusM: radius,
    a0Rad: a0,
    sweepRad: sweep,
    a: [...a] as Vec2,
    b: [...b] as Vec2,
  };
}

/** Closed ring of exact edges (implicit closure: last edge ends at the first edge's start). */
export type ExportRing = ExportEdge[];

function ringPoints(ring: ExportRing): Vec2[] {
  const pts: Vec2[] = [];
  for (const e of ring) pts.push(e.a);
  return pts;
}

// ---------------------------------------------------------------------------
// IR entities
// ---------------------------------------------------------------------------

export interface ExportNotchIR {
  id: string;
  kind: string;
  segmentId: string;
  t: number;
  pos: Vec2;
  ticks: Array<{ from: Vec2; to: Vec2 }>;
  depthM: number;
}

export interface ExportPanelIR {
  panelId: string;
  name: string;
  materialId: string;
  /** Sewing boundary as exact edges (closed ring). */
  sewingEdges: ExportRing;
  holes: ExportRing[];
  /** Cut boundary as exact edges; null when no cut line is authored. */
  cutEdges: ExportRing | null;
  cutSource: "sewing" | "allowance" | null;
  /** Derived allowance ring (exact edges) when an allowance is authored. */
  allowanceEdges: ExportRing | null;
  allowanceIssues: OffsetIssue[];
  notches: ExportNotchIR[];
  grainlines: Array<{ id: string; from: Vec2; to: Vec2 }>;
  crossGrains: Array<{ from: Vec2; to: Vec2 }>;
  folds: Array<{ id: string; a: Vec2; b: Vec2; direction: string; foldType: string }>;
  drills: Array<{ id: string; pos: Vec2; mark: string; radiusM: number }>;
  internals: Array<{ id: string; edges: ExportEdge[]; kind: string; label?: string }>;
  annotations: Array<{ id: string; pos: Vec2; note: string }>;
  labelRegions: Array<{ id: string; min: Vec2; max: Vec2; fields: Record<string, string> }>;
  dimensions: Array<{ segmentId: string; lengthM: number; mid: Vec2 }>;
  /** Authored construction segments resolved to exact edges (kernel construction lines). */
  constructionEdges: ExportEdge[];
  cutQuantity: number;
  areaM2: number;
  /** Authored metadata (section, mirror pair) from the production set. */
  section?: string;
  mirrorPair?: string;
}

export interface ExportSeamIR {
  id: string;
  sideA: { panelId: string; loopId: string; segmentIds: string[] };
  sideB: { panelId: string; loopId: string; segmentIds: string[] };
  stitchCount: number;
  lengthAM: number;
  lengthBM: number;
}

export interface ExportSizeIR {
  sizeId: string;
  label: string;
  isBase: boolean;
  measurements: Record<string, number>;
}

/**
 * Per-size grading record: one entry per size (master deltas when the base
 * size IS the master; per-size deltas otherwise).
 */
export interface ExportGradingIR {
  present: boolean;
  baseSizeId: string;
  sizes: ExportSizeIR[];
  rules: Array<{
    id: string;
    gradingPointId: string;
    mode: "per-size" | "transition";
    deltas: Record<string, [number, number]>;
    masterPosition: [number, number];
    gradedPosition: [number, number];
  }>;
}

export interface ExportStyleIR {
  garmentName: string;
  styleId: string;
  revision: number;
  sizeSetId?: string;
  sizeSetName?: string;
}

export interface ExportIR {
  format: typeof EXPORT_IR_FORMAT;
  version: typeof EXPORT_IR_VERSION;
  /** Canonical units of every `...M` / `*M2` field. Adapters convert. */
  units: "m";
  style: ExportStyleIR;
  garment: { documentId: string; panelCount: number; seamCount: number };
  panels: ExportPanelIR[];
  seams: ExportSeamIR[];
  grading: ExportGradingIR;
  readiness: { state: string; errorCount: number; warningCount: number };
}

// ---------------------------------------------------------------------------
// Grading input
// ---------------------------------------------------------------------------

/** Optional grading context derived from a G12 grading document. */
export interface ExportGradingContext {
  sizes: Array<{ sizeId: string; label: string; isBase: boolean; measurements: Record<string, number> }>;
  baseSizeId: string;
  sizeSetId?: string;
  sizeSetName?: string;
  /** Per-size derivation reports keyed by sizeId (the audit surface). */
  ruleApplications: Record<string, Array<{
    ruleId: string;
    gradingPointId: string;
    mode: "per-size" | "transition";
    delta: [number, number];
    masterPosition: [number, number];
    gradedPosition: [number, number];
  }>>;
}

// ---------------------------------------------------------------------------
// Gate
// ---------------------------------------------------------------------------

export type ExportGateMode = "strict" | "allow-warnings";

export interface ExportGateDecision {
  ok: boolean;
  state: string;
  blocked: boolean;
  reason?: string;
  readiness: ReadinessReport;
}

/**
 * Run production readiness and decide whether export may proceed.
 * strict: only READY_FOR_EXPORT passes. allow-warnings: WARNINGS passes with
 * an explicit decision record (warnings stay in the IR); INVALID never passes.
 */
export function exportGate(
  doc: PatternDocument,
  seams: readonly Seam[],
  set: ProductionSet,
  mode: ExportGateMode = "strict",
  tolerances?: ReadinessTolerances,
): ExportGateDecision {
  const readiness = productionReadiness(doc, seams, set, tolerances);
  if (readiness.state === "READY_FOR_EXPORT") return { ok: true, state: readiness.state, blocked: false, readiness };
  if (readiness.state === "WARNINGS" && mode === "allow-warnings") {
    return { ok: true, state: readiness.state, blocked: false, reason: "warnings accepted explicitly", readiness };
  }
  const first = readiness.diagnostics[0];
  return {
    ok: false,
    state: readiness.state,
    blocked: true,
    reason: first ? `${first.severity} ${first.code}: ${first.message}` : "validation failed",
    readiness,
  };
}

// ---------------------------------------------------------------------------
// IR construction
// ---------------------------------------------------------------------------

export interface BuildExportIROptions {
  garmentName?: string;
  styleId?: string;
  revision?: number;
  grading?: ExportGradingContext;
  readiness?: ReadinessReport;
}

/** Resolve one loop to exact edges (lines exact, arcs centre/radius/angles). */
function loopToEdges(doc: PatternDocument, panelId: string, loop: { segmentIds: string[] }): ExportRing {
  const edges: ExportRing = [];
  for (const segmentId of loop.segmentIds) {
    const segment = doc.segments.find((s) => s.id === segmentId);
    if (!segment || segment.panelId !== panelId) {
      throw new PatternCadError("missing-reference", `boundary segment '${segmentId}' missing for panel '${panelId}'`, segmentId);
    }
    const start = doc.points.find((p) => p.id === segment.startPointId);
    const end = doc.points.find((p) => p.id === segment.endPointId);
    if (!start || !end) throw new PatternCadError("missing-reference", `segment '${segmentId}' endpoints missing`, segmentId);
    const a: Vec2 = [start.x, start.y];
    const b: Vec2 = [end.x, end.y];
    if (segment.kind === "line") {
      edges.push(lineEdge(a, b));
    } else {
      const center = doc.points.find((p) => p.id === segment.centerPointId);
      if (!center) throw new PatternCadError("missing-reference", `arc '${segmentId}' centre missing`, segmentId);
      const radius = Math.hypot(a[0] - center.x, a[1] - center.y);
      edges.push(arcEdge([center.x, center.y], radius, Math.atan2(a[1] - center.y, a[0] - center.x), segment.sweepRad, a, b));
    }
  }
  return edges;
}

function panelBBox(rings: ExportRing[], extra: Vec2[]): { min: Vec2; max: Vec2 } | null {
  const pts: Vec2[] = [];
  for (const ring of rings) {
    for (const e of ring) {
      pts.push(e.a, e.b);
      if (e.kind === "arc") {
        pts.push([e.center[0] - e.radiusM, e.center[1] - e.radiusM]);
        pts.push([e.center[0] + e.radiusM, e.center[1] + e.radiusM]);
      }
    }
  }
  pts.push(...extra);
  if (pts.length === 0) return null;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of pts) {
    if (p[0] < minX) minX = p[0];
    if (p[1] < minY) minY = p[1];
    if (p[0] > maxX) maxX = p[0];
    if (p[1] > maxY) maxY = p[1];
  }
  return { min: [minX, minY], max: [maxX, maxY] };
}

/**
 * Build the deterministic IR. Validates the kernel first; run
 * `exportGate` (or pass `readiness`) before calling when you need the gate.
 */
export function buildExportIR(
  doc: PatternDocument,
  seams: readonly Seam[],
  set: ProductionSet,
  opts: BuildExportIROptions = {},
): ExportIR {
  const kernel = validatePatternDocument(doc);
  if (!kernel.valid) {
    throw new PatternCadError("invalid-document", "cannot build export IR from an invalid pattern");
  }
  const panels: ExportPanelIR[] = doc.panels.map((panel) => {
    const outer = panel.boundaryLoops.find((l) => l.role === "outer");
    if (!outer) throw new PatternCadError("open-boundary", `panel '${panel.id}' has no outer loop`, panel.id);
    const sewingEdges = loopToEdges(doc, panel.id, outer);
    const holes = panel.boundaryLoops
      .filter((l) => l.role === "hole")
      .map((l) => loopToEdges(doc, panel.id, l));
    const cut = set.cutLines.find((c) => c.panelId === panel.id && c.loopId === outer.id) ?? null;
    let cutEdges: ExportRing | null = null;
    let cutSource: ExportPanelIR["cutSource"] = null;
    if (cut) {
      cutSource = cut.source;
      try {
        const { ring } = cutBoundary(doc, set, cut.id);
        if (ring.length >= 3) {
          cutEdges = [];
          for (let i = 0; i < ring.length; i++) {
            cutEdges.push(lineEdge(ring[i], ring[(i + 1) % ring.length]));
          }
        }
      } catch {
        cutEdges = null;
      }
    }
    const allowance = set.allowances.find((a) => a.panelId === panel.id && a.loopId === outer.id) ?? null;
    let allowanceEdges: ExportRing | null = null;
    let allowanceIssues: OffsetIssue[] = [];
    if (allowance) {
      try {
        const derived = allowanceBoundary(doc, set, panel.id, outer.id);
        allowanceIssues = derived.issues;
        if (derived.ring.length >= 3) {
          allowanceEdges = [];
          for (let i = 0; i < derived.ring.length; i++) {
            allowanceEdges.push(lineEdge(derived.ring[i], derived.ring[(i + 1) % derived.ring.length]));
          }
        }
      } catch {
        allowanceEdges = null;
      }
    }
    const notches: ExportNotchIR[] = [];
    for (const notch of set.notches.filter((e) => e.panelId === panel.id)) {
      if (notchStatus(doc, notch) !== "ok") continue;
      try {
        const { ticks, frame } = notchTicks(doc, notch);
        const frame2 = frame ?? notchFrame(doc, notch);
        notches.push({
          id: notch.id, kind: notch.kind, segmentId: notch.segmentId, t: notch.t,
          pos: frame2.pos, ticks, depthM: notch.depthM,
        });
      } catch {
        continue;
      }
    }
    const meta = set.panelMeta.find((m) => m.panelId === panel.id) ?? null;
    let areaM2 = 0;
    try {
      const pts = ringPoints(sewingEdges);
      areaM2 = Math.abs(signedArea(pts));
    } catch {
      areaM2 = 0;
    }
    const constructionEdges: ExportEdge[] = [];
    for (const segmentId of panel.constructionSegmentIds) {
      const segment = doc.segments.find((s) => s.id === segmentId);
      if (!segment) continue;
      const start = doc.points.find((p) => p.id === segment.startPointId);
      const end = doc.points.find((p) => p.id === segment.endPointId);
      if (!start || !end) continue;
      constructionEdges.push(lineEdge([start.x, start.y], [end.x, end.y]));
    }
    return {
      panelId: panel.id,
      name: panel.name,
      materialId: panel.materialId,
      sewingEdges,
      holes,
      cutEdges,
      cutSource,
      allowanceEdges,
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
        .map((e) => {
          const edges: ExportEdge[] = [];
          for (let i = 0; i < e.points.length - 1; i++) {
            edges.push(lineEdge(e.points[i], e.points[i + 1]));
          }
          return {
            id: e.id, edges, kind: e.kind,
            ...(e.label !== undefined ? { label: e.label } : {}),
          };
        }),
      annotations: set.annotations
        .filter((e) => e.panelId === panel.id)
        .map((e) => ({ id: e.id, pos: [...e.pos] as Vec2, note: e.note })),
      labelRegions: set.labelRegions
        .filter((e) => e.panelId === panel.id)
        .map((e) => ({ id: e.id, min: [...e.min] as Vec2, max: [...e.max] as Vec2, fields: { ...e.fields } })),
      dimensions: outer.segmentIds.map((segmentId) => {
        const edges = sewingEdges;
        const idx = outer.segmentIds.indexOf(segmentId);
        const edge = edges[idx];
        const mid: Vec2 = edge ? [(edge.a[0] + edge.b[0]) / 2, (edge.a[1] + edge.b[1]) / 2] : [0, 0];
        const lengthM = edge
          ? (edge.kind === "line"
            ? Math.hypot(edge.b[0] - edge.a[0], edge.b[1] - edge.a[1])
            : edge.radiusM * Math.abs(edge.sweepRad))
          : 0;
        return { segmentId, lengthM, mid };
      }),
      constructionEdges,
      cutQuantity: meta?.cutQuantity ?? 0,
      areaM2,
      ...(meta?.section !== undefined ? { section: meta.section } : {}),
      ...(meta?.mirrorPair !== undefined ? { mirrorPair: meta.mirrorPair } : {}),
    };
  });

  // Layout-independent per-panel bbox sanity (kept for adapter reuse).
  for (const p of panels) {
    const bb = panelBBox([p.sewingEdges], []);
    if (bb && (!Number.isFinite(bb.min[0]) || !Number.isFinite(bb.max[1]))) {
      throw new PatternCadError("invalid-document", `panel '${p.panelId}' has a non-finite bbox`, p.panelId);
    }
  }

  const readiness = opts.readiness ?? null;
  const grading = opts.grading ?? null;
  const seamRows = readiness ? readiness.seams : null;
  return {
    format: EXPORT_IR_FORMAT,
    version: EXPORT_IR_VERSION,
    units: "m",
    style: {
      garmentName: opts.garmentName ?? doc.name,
      styleId: opts.styleId ?? doc.id,
      revision: opts.revision ?? 1,
      ...(grading?.sizeSetId !== undefined ? { sizeSetId: grading.sizeSetId } : {}),
      ...(grading?.sizeSetName !== undefined ? { sizeSetName: grading.sizeSetName } : {}),
    },
    garment: { documentId: doc.id, panelCount: panels.length, seamCount: seams.length },
    panels,
    seams: seams.map((seam) => {
      const row = seamRows?.find((r) => r.seamId === seam.id);
      let lengthAM = 0, lengthBM = 0;
      if (row) {
        lengthAM = row.lengthAM;
        lengthBM = row.lengthBM;
      } else {
        try {
          ({ lengthAM, lengthBM } = seamSideLengths(doc, seam));
        } catch {
          lengthAM = 0;
          lengthBM = 0;
        }
      }
      return {
        id: seam.id,
        sideA: JSON.parse(JSON.stringify(seam.sideA)) as ExportSeamIR["sideA"],
        sideB: JSON.parse(JSON.stringify(seam.sideB)) as ExportSeamIR["sideB"],
        stitchCount: seam.stitchCount,
        lengthAM,
        lengthBM,
      };
    }),
    grading: grading
      ? {
        present: true,
        baseSizeId: grading.baseSizeId,
        sizes: grading.sizes.map((s) => ({
          sizeId: s.sizeId, label: s.label, isBase: s.isBase, measurements: { ...s.measurements },
        })),
        rules: Object.entries(grading.ruleApplications)
          .flatMap(([sizeId, rules]) =>
            rules.map((r) => ({
              id: r.ruleId,
              gradingPointId: r.gradingPointId,
              mode: r.mode,
              deltas: { [sizeId]: [r.delta[0], r.delta[1]] as [number, number] },
              masterPosition: r.masterPosition,
              gradedPosition: r.gradedPosition,
            }))),
      }
      : { present: false, baseSizeId: "", sizes: [], rules: [] },
    readiness: readiness
      ? { state: readiness.state, errorCount: readiness.errorCount, warningCount: readiness.warningCount }
      : { state: "UNCHECKED", errorCount: 0, warningCount: 0 },
  };
}

// ---------------------------------------------------------------------------
// Canonical JSON (lossless round-trip for QA)
// ---------------------------------------------------------------------------

export function canonicalJson(value: unknown): string {
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
  if (!ir || ir.format !== EXPORT_IR_FORMAT || ir.version !== EXPORT_IR_VERSION || !Array.isArray(ir.panels)) {
    throw new PatternCadError("invalid-document", "export JSON has the wrong shape or version");
  }
  return JSON.parse(JSON.stringify(ir)) as ExportIR;
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

export { formatLength, parseLengthToM };
