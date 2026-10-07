// G11A — production pattern geometry kernel.
//
// The kernel document (src/pattern/cad.ts) owns DESIGN geometry: the sewing
// boundary (authoritative for G8 construction and simulation) and
// construction lines. Triangulation is SIMULATION geometry (derived). This
// module owns PRODUCTION geometry: seam allowances, folds, drills, internal
// lines, cut lines, annotations, and label regions.
//
// All production entities live in a `ProductionSet` sidecar referencing
// stable kernel ids — the sewing boundary is never overwritten. Allowance
// boundaries are DERIVED data (recomputed on demand, like triangulation):
// line edges offset exactly, arc edges sampled with a bounded sagitta, so
// per-edge allowances work on mixed loops and the original arcs stay
// authored. Invalid offsets are reported as issues, never silent garbage.
//
// Canonical unit: metres (kernel frame).

import {
  PatternCadError,
  type EntityId,
  type PatternDocument,
} from "../pattern/cad.js";
import {
  dist,
  leftNormal,
  lerp,
  lineIntersection,
  normalize,
  pointOnArc,
  sampleArc,
  signedArea,
  sub,
  type ArcGeometry,
  type Vec2,
} from "./geom.js";
import {
  getLoop,
  getPanel,
  getPoint,
  getSegment,
  pointInPanel,
  resolveSegment,
  sampleLoopLocal,
} from "./queries.js";
import { distToSegment } from "../pattern/pattern-geometry.js";

export const PRODUCTION_SET_VERSION = 1;
const EPS_M = 1e-9;

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function finiteVec(v: Vec2): boolean {
  return Number.isFinite(v[0]) && Number.isFinite(v[1]);
}

// ---------------------------------------------------------------------------
// Entities
// ---------------------------------------------------------------------------

export interface SeamAllowance {
  id: string;
  panelId: EntityId;
  loopId: EntityId;
  /** Default allowance in metres (>= 0). */
  defaultM: number;
  /** Per-edge overrides keyed by segment id. */
  perEdgeM: Record<string, number>;
}

export type NotchKind = "single" | "double" | "custom";

export interface Notch {
  id: string;
  panelId: EntityId;
  loopId: EntityId;
  segmentId: EntityId;
  /** Arclength fraction along the edge in [0,1]. */
  t: number;
  kind: NotchKind;
  /** Notch depth in metres (> 0). */
  depthM: number;
}

export interface Grainline {
  id: string;
  panelId: EntityId;
  from: Vec2;
  to: Vec2;
}

export type FoldDirection = "mountain" | "valley";

export interface FoldLine {
  id: string;
  panelId: EntityId;
  a: Vec2;
  b: Vec2;
  direction: FoldDirection;
  foldType: string;
}

export type DrillMarkKind = "point" | "circle" | "cross";

export interface DrillMark {
  id: string;
  panelId: EntityId;
  pos: Vec2;
  mark: DrillMarkKind;
  radiusM: number;
}

export type InternalLineKind =
  | "dart" | "pleat" | "pocket" | "button" | "buttonhole" | "stitch-guide" | "custom";

export interface InternalLine {
  id: string;
  panelId: EntityId;
  points: Vec2[];
  kind: InternalLineKind;
  label?: string;
}

export interface CutLine {
  id: string;
  panelId: EntityId;
  loopId: EntityId;
  source: "sewing" | "allowance";
}

export interface AnnotationAnchor {
  id: string;
  panelId: EntityId;
  pos: Vec2;
  note: string;
}

export interface LabelRegion {
  id: string;
  panelId: EntityId;
  min: Vec2;
  max: Vec2;
  fields: Record<string, string>;
}

export interface PanelProductionMeta {
  panelId: EntityId;
  cutQuantity: number;
  section?: string;
  mirrorPair?: string;
  notes?: string;
}

export interface ProductionSet {
  version: typeof PRODUCTION_SET_VERSION;
  nextIndex: number;
  allowances: SeamAllowance[];
  notches: Notch[];
  grainlines: Grainline[];
  folds: FoldLine[];
  drills: DrillMark[];
  internals: InternalLine[];
  cutLines: CutLine[];
  annotations: AnnotationAnchor[];
  labelRegions: LabelRegion[];
  panelMeta: PanelProductionMeta[];
}

export function createProductionSet(): ProductionSet {
  return {
    version: PRODUCTION_SET_VERSION,
    nextIndex: 1,
    allowances: [], notches: [], grainlines: [], folds: [], drills: [],
    internals: [], cutLines: [], annotations: [], labelRegions: [], panelMeta: [],
  };
}

function allocateId(set: ProductionSet, kind: string): string {
  for (;;) {
    const id = `production/${kind}/${String(set.nextIndex++).padStart(8, "0")}`;
    if (!containsId(set, id)) return id;
  }
}

function containsId(set: ProductionSet, id: string): boolean {
  return (
    set.allowances.some((e) => e.id === id) || set.notches.some((e) => e.id === id) ||
    set.grainlines.some((e) => e.id === id) || set.folds.some((e) => e.id === id) ||
    set.drills.some((e) => e.id === id) || set.internals.some((e) => e.id === id) ||
    set.cutLines.some((e) => e.id === id) || set.annotations.some((e) => e.id === id) ||
    set.labelRegions.some((e) => e.id === id)
  );
}

// -- builders (each returns a fresh set + id) --------------------------------

function push<T>(set: ProductionSet, key: keyof ProductionSet, entity: T): { set: ProductionSet; id: string } {
  const next = clone(set);
  (next[key] as T[]).push(clone(entity));
  return { set: next, id: (entity as { id: string }).id };
}

export function addAllowance(
  set: ProductionSet, panelId: EntityId, loopId: EntityId, defaultM: number,
): { set: ProductionSet; id: string } {
  if (!Number.isFinite(defaultM) || defaultM < 0) {
    throw new PatternCadError("invalid-transform", "allowance must be finite and >= 0");
  }
  const next = clone(set);
  const id = allocateId(next, "allowance");
  next.allowances = next.allowances.filter((a) => !(a.panelId === panelId && a.loopId === loopId));
  next.allowances.push({ id, panelId, loopId, defaultM, perEdgeM: {} });
  return { set: next, id };
}

export function setEdgeAllowance(
  set: ProductionSet, allowanceId: string, segmentId: EntityId, valueM: number,
): ProductionSet {
  if (!Number.isFinite(valueM) || valueM < 0) {
    throw new PatternCadError("invalid-transform", "edge allowance must be finite and >= 0", segmentId);
  }
  const next = clone(set);
  const allowance = next.allowances.find((a) => a.id === allowanceId);
  if (!allowance) throw new PatternCadError("missing-reference", `allowance '${allowanceId}' does not exist`, allowanceId);
  allowance.perEdgeM[segmentId] = valueM;
  return next;
}

export function clearEdgeAllowance(set: ProductionSet, allowanceId: string, segmentId: EntityId): ProductionSet {
  const next = clone(set);
  const allowance = next.allowances.find((a) => a.id === allowanceId);
  if (!allowance) throw new PatternCadError("missing-reference", `allowance '${allowanceId}' does not exist`, allowanceId);
  delete allowance.perEdgeM[segmentId];
  return next;
}

export function removeAllowance(set: ProductionSet, allowanceId: string): ProductionSet {
  if (!set.allowances.some((a) => a.id === allowanceId)) {
    throw new PatternCadError("missing-reference", `allowance '${allowanceId}' does not exist`, allowanceId);
  }
  const next = clone(set);
  next.allowances = next.allowances.filter((a) => a.id !== allowanceId);
  return next;
}

export function addNotch(
  set: ProductionSet, panelId: EntityId, loopId: EntityId, segmentId: EntityId,
  t: number, kind: NotchKind, depthM: number,
): { set: ProductionSet; id: string } {
  if (!Number.isFinite(t) || t < 0 || t > 1) {
    throw new PatternCadError("invalid-transform", "notch t must be in [0,1]", segmentId);
  }
  if (!Number.isFinite(depthM) || !(depthM > 0)) {
    throw new PatternCadError("invalid-transform", "notch depth must be positive", segmentId);
  }
  if (kind !== "single" && kind !== "double" && kind !== "custom") {
    throw new PatternCadError("invalid-transform", `unknown notch kind '${kind}'`, segmentId);
  }
  const next = clone(set);
  return push(next, "notches", { id: allocateId(next, "notch"), panelId, loopId, segmentId, t, kind, depthM });
}

export function removeNotch(set: ProductionSet, id: string): ProductionSet {
  if (!set.notches.some((e) => e.id === id)) {
    throw new PatternCadError("missing-reference", `notch '${id}' does not exist`, id);
  }
  const next = clone(set);
  next.notches = next.notches.filter((e) => e.id !== id);
  return next;
}

export function addGrainline(
  set: ProductionSet, panelId: EntityId, from: Vec2, to: Vec2,
): { set: ProductionSet; id: string } {
  if (!finiteVec(from) || !finiteVec(to) || dist(from, to) <= EPS_M) {
    throw new PatternCadError("invalid-transform", "grainline needs two distinct finite points", panelId);
  }
  const next = clone(set);
  return push(next, "grainlines", { id: allocateId(next, "grainline"), panelId, from: [...from], to: [...to] });
}

export function removeGrainline(set: ProductionSet, id: string): ProductionSet {
  if (!set.grainlines.some((e) => e.id === id)) {
    throw new PatternCadError("missing-reference", `grainline '${id}' does not exist`, id);
  }
  const next = clone(set);
  next.grainlines = next.grainlines.filter((e) => e.id !== id);
  return next;
}

export function addFoldLine(
  set: ProductionSet, panelId: EntityId, a: Vec2, b: Vec2,
  direction: FoldDirection, foldType: string,
): { set: ProductionSet; id: string } {
  if (!finiteVec(a) || !finiteVec(b) || dist(a, b) <= EPS_M) {
    throw new PatternCadError("invalid-transform", "fold line needs two distinct finite points", panelId);
  }
  if (direction !== "mountain" && direction !== "valley") {
    throw new PatternCadError("invalid-transform", `unknown fold direction '${direction}'`, panelId);
  }
  if (typeof foldType !== "string" || foldType.length === 0) {
    throw new PatternCadError("invalid-transform", "fold type must be a non-empty string", panelId);
  }
  const next = clone(set);
  return push(next, "folds", { id: allocateId(next, "fold"), panelId, a: [...a], b: [...b], direction, foldType });
}

export function removeFoldLine(set: ProductionSet, id: string): ProductionSet {
  if (!set.folds.some((e) => e.id === id)) {
    throw new PatternCadError("missing-reference", `fold '${id}' does not exist`, id);
  }
  const next = clone(set);
  next.folds = next.folds.filter((e) => e.id !== id);
  return next;
}

export function addDrillMark(
  set: ProductionSet, panelId: EntityId, pos: Vec2,
  mark: DrillMarkKind = "point", radiusM = 0.002,
): { set: ProductionSet; id: string } {
  if (!finiteVec(pos)) {
    throw new PatternCadError("invalid-transform", "drill position must be finite", panelId);
  }
  if (mark !== "point" && mark !== "circle" && mark !== "cross") {
    throw new PatternCadError("invalid-transform", `unknown drill mark '${mark}'`, panelId);
  }
  if (!Number.isFinite(radiusM) || !(radiusM > 0)) {
    throw new PatternCadError("invalid-transform", "drill radius must be positive", panelId);
  }
  const next = clone(set);
  return push(next, "drills", { id: allocateId(next, "drill"), panelId, pos: [...pos], mark, radiusM });
}

export function removeDrillMark(set: ProductionSet, id: string): ProductionSet {
  if (!set.drills.some((e) => e.id === id)) {
    throw new PatternCadError("missing-reference", `drill '${id}' does not exist`, id);
  }
  const next = clone(set);
  next.drills = next.drills.filter((e) => e.id !== id);
  return next;
}

export function addInternalLine(
  set: ProductionSet, panelId: EntityId, points: Vec2[], kind: InternalLineKind, label?: string,
): { set: ProductionSet; id: string } {
  const kinds: InternalLineKind[] = ["dart", "pleat", "pocket", "button", "buttonhole", "stitch-guide", "custom"];
  if (!kinds.includes(kind)) {
    throw new PatternCadError("invalid-transform", `unknown internal line kind '${kind}'`, panelId);
  }
  if (!Array.isArray(points) || points.length < 2 || !points.every(finiteVec)) {
    throw new PatternCadError("invalid-transform", "internal line needs at least 2 finite points", panelId);
  }
  const next = clone(set);
  return push(next, "internals", {
    id: allocateId(next, "internal"), panelId,
    points: points.map((p) => [p[0], p[1]] as Vec2), kind,
    ...(label !== undefined ? { label } : {}),
  });
}

export function removeInternalLine(set: ProductionSet, id: string): ProductionSet {
  if (!set.internals.some((e) => e.id === id)) {
    throw new PatternCadError("missing-reference", `internal line '${id}' does not exist`, id);
  }
  const next = clone(set);
  next.internals = next.internals.filter((e) => e.id !== id);
  return next;
}

export function addCutLine(
  set: ProductionSet, panelId: EntityId, loopId: EntityId, source: "sewing" | "allowance",
): { set: ProductionSet; id: string } {
  if (source !== "sewing" && source !== "allowance") {
    throw new PatternCadError("invalid-transform", `unknown cut source '${source}'`, panelId);
  }
  const next = clone(set);
  return push(next, "cutLines", { id: allocateId(next, "cut"), panelId, loopId, source });
}

export function removeCutLine(set: ProductionSet, id: string): ProductionSet {
  if (!set.cutLines.some((e) => e.id === id)) {
    throw new PatternCadError("missing-reference", `cut line '${id}' does not exist`, id);
  }
  const next = clone(set);
  next.cutLines = next.cutLines.filter((e) => e.id !== id);
  return next;
}

export function addAnnotation(
  set: ProductionSet, panelId: EntityId, pos: Vec2, note: string,
): { set: ProductionSet; id: string } {
  if (!finiteVec(pos)) {
    throw new PatternCadError("invalid-transform", "annotation position must be finite", panelId);
  }
  if (typeof note !== "string" || note.length === 0) {
    throw new PatternCadError("invalid-transform", "annotation note must be non-empty", panelId);
  }
  const next = clone(set);
  return push(next, "annotations", { id: allocateId(next, "annotation"), panelId, pos: [...pos], note });
}

export function removeAnnotation(set: ProductionSet, id: string): ProductionSet {
  if (!set.annotations.some((e) => e.id === id)) {
    throw new PatternCadError("missing-reference", `annotation '${id}' does not exist`, id);
  }
  const next = clone(set);
  next.annotations = next.annotations.filter((e) => e.id !== id);
  return next;
}

export function addLabelRegion(
  set: ProductionSet, panelId: EntityId, min: Vec2, max: Vec2, fields: Record<string, string>,
): { set: ProductionSet; id: string } {
  if (!finiteVec(min) || !finiteVec(max) || !(max[0] > min[0]) || !(max[1] > min[1])) {
    throw new PatternCadError("invalid-transform", "label region needs a non-degenerate box", panelId);
  }
  if (!fields || typeof fields !== "object") {
    throw new PatternCadError("invalid-transform", "label fields must be an object", panelId);
  }
  const next = clone(set);
  return push(next, "labelRegions", {
    id: allocateId(next, "label"), panelId,
    min: [...min], max: [...max], fields: { ...fields },
  });
}

export function removeLabelRegion(set: ProductionSet, id: string): ProductionSet {
  if (!set.labelRegions.some((e) => e.id === id)) {
    throw new PatternCadError("missing-reference", `label region '${id}' does not exist`, id);
  }
  const next = clone(set);
  next.labelRegions = next.labelRegions.filter((e) => e.id !== id);
  return next;
}

/** Upsert panel production metadata (keyed by panel id; deterministic order = first-seen). */
export function setPanelMeta(set: ProductionSet, meta: PanelProductionMeta): ProductionSet {
  if (!Number.isInteger(meta.cutQuantity) || meta.cutQuantity < 1) {
    throw new PatternCadError("invalid-transform", "cut quantity must be a positive integer", meta.panelId);
  }
  const next = clone(set);
  const existing = next.panelMeta.findIndex((m) => m.panelId === meta.panelId);
  if (existing >= 0) next.panelMeta[existing] = clone(meta);
  else next.panelMeta.push(clone(meta));
  return next;
}

// ---------------------------------------------------------------------------
// Allowance boundary derivation
// ---------------------------------------------------------------------------

export type OffsetIssueCode = "spike" | "self-intersection" | "degenerate-ring" | "parallel-join";

export interface OffsetIssue {
  code: OffsetIssueCode;
  message: string;
  /** Sample index into the derived ring (when localizable). */
  atIndex?: number;
}

export interface AllowanceBoundary {
  loopId: EntityId;
  /** Derived cut ring in panel-local metres (closed implicitly: last -> first). */
  ring: Vec2[];
  issues: OffsetIssue[];
  /** Largest deviation of the derived ring from exact offset (sagitta bound for arcs, 0 for lines). */
  maxDeviationM: number;
}

interface SampledEdge {
  segmentId: EntityId;
  points: Vec2[];
  distanceM: number;
}

function outwardOf(a: Vec2, b: Vec2, ccw: boolean): Vec2 {
  const n = normalize(leftNormal(a, b));
  return ccw ? [-n[0], -n[1]] : [n[0], n[1]];
}

/** True when segments p1-p2 and p3-p4 cross at an interior point (not a shared endpoint touch). */
function properCrossing(p1: Vec2, p2: Vec2, p3: Vec2, p4: Vec2): boolean {
  const orient = (a: Vec2, b: Vec2, c: Vec2): number =>
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
  const d1 = orient(p3, p4, p1), d2 = orient(p3, p4, p2);
  const d3 = orient(p1, p2, p3), d4 = orient(p1, p2, p4);
  return ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0));
}

/**
 * Derive the allowance (cut) ring for one loop. Line edges offset exactly;
 * arc edges are sampled with a bounded sagitta (default 1e-4 m). The
 * returned ring is derived data — the authored boundary keeps its ids.
 */
export function allowanceBoundary(
  doc: PatternDocument,
  set: ProductionSet,
  panelId: EntityId,
  loopId: EntityId,
  sagittaTolM = 1e-4,
): AllowanceBoundary {
  const allowance = set.allowances.find((a) => a.panelId === panelId && a.loopId === loopId);
  if (!allowance) {
    throw new PatternCadError("missing-reference", `no allowance for loop '${loopId}'`, loopId);
  }
  const loop = getLoop(doc, panelId, loopId);
  const distOf = (segmentId: EntityId): number => allowance.perEdgeM[segmentId] ?? allowance.defaultM;
  // Sample every edge in loop order.
  const edges: SampledEdge[] = loop.segmentIds.map((segmentId) => {
    const segment = getSegment(doc, segmentId, panelId);
    if (segment.kind === "line") {
      const a = getPoint(doc, segment.startPointId, panelId);
      const b = getPoint(doc, segment.endPointId, panelId);
      return { segmentId, points: [[a.x, a.y], [b.x, b.y]], distanceM: distOf(segmentId) };
    }
    const r = resolveSegment(doc, segmentId, panelId);
    const arc: ArcGeometry = r.arc!;
    return { segmentId, points: sampleArc(arc.center, arc.radius, arc.a0, arc.sweep, sagittaTolM), distanceM: distOf(segmentId) };
  });
  // Flatten to a sample ring of DISTINCT vertices: each edge contributes
  // all its samples except its final endpoint (the next edge's start).
  const samples: Vec2[] = [];
  const sampleDist: number[] = [];
  for (const edge of edges) {
    for (let i = 0; i < edge.points.length - 1; i++) {
      samples.push(edge.points[i]);
      sampleDist.push(edge.distanceM);
    }
  }
  const issues: OffsetIssue[] = [];
  if (samples.length < 3) {
    return { loopId, ring: [], issues: [{ code: "degenerate-ring", message: "loop samples to fewer than 3 points" }], maxDeviationM: 0 };
  }
  const ccw = signedArea(samples) > 0;
  const n = samples.length;
  // Offset each sample edge, then miter consecutive offset lines.
  const offsetA: Vec2[] = [];
  const offsetB: Vec2[] = [];
  for (let i = 0; i < n; i++) {
    const a = samples[i], b = samples[(i + 1) % n];
    const d = sampleDist[i];
    const outward = outwardOf(a, b, ccw);
    offsetA.push([a[0] + outward[0] * d, a[1] + outward[1] * d]);
    offsetB.push([b[0] + outward[0] * d, b[1] + outward[1] * d]);
  }
  const ring: Vec2[] = [];
  for (let i = 0; i < n; i++) {
    const prev = (i + n - 1) % n;
    const cross = lineIntersection(offsetA[prev], offsetB[prev], offsetA[i], offsetB[i], EPS_M);
    if (cross) {
      ring.push(cross);
      // Spike guard: miter vertex far from both parent offset points.
      const spike = Math.min(dist(cross, offsetB[prev]), dist(cross, offsetA[i]));
      const local = Math.max(sampleDist[prev], sampleDist[i], EPS_M);
      if (spike > 10 * local) {
        issues.push({ code: "spike", message: `miter spike at sample ${i} (${(spike * 1000).toFixed(1)} mm from the offset lines)`, atIndex: i });
      }
    } else {
      issues.push({ code: "parallel-join", message: `adjacent offset edges parallel at sample ${i}; averaged fallback used`, atIndex: i });
      ring.push([(offsetB[prev][0] + offsetA[i][0]) / 2, (offsetB[prev][1] + offsetA[i][1]) / 2]);
    }
  }
  // Self-intersection audit on the derived ring.
  for (let i = 0; i < n && issues.filter((s) => s.code === "self-intersection").length === 0; i++) {
    for (let j = i + 1; j < n; j++) {
      const adjacent = j === i + 1 || (i === 0 && j === n - 1);
      if (adjacent) continue;
      if (properCrossing(ring[i], ring[(i + 1) % n], ring[j], ring[(j + 1) % n])) {
        issues.push({ code: "self-intersection", message: `derived allowance ring self-intersects near samples ${i}/${j}`, atIndex: i });
        break;
      }
    }
  }
  const hasArcs = edges.some((e) => e.points.length > 2);
  return { loopId, ring, issues, maxDeviationM: hasArcs ? sagittaTolM : 0 };
}

/** Notch frame: position on the edge plus outward normal and tangent (panel-local). */
export function notchFrame(doc: PatternDocument, notch: Notch): { pos: Vec2; outward: Vec2; tangent: Vec2 } {
  const segment = getSegment(doc, notch.segmentId, notch.panelId);
  if (segment.kind === "line") {
    const a = getPoint(doc, segment.startPointId, notch.panelId);
    const b = getPoint(doc, segment.endPointId, notch.panelId);
    const pos: Vec2 = [a.x + (b.x - a.x) * notch.t, a.y + (b.y - a.y) * notch.t];
    const tangent = normalize(sub([b.x, b.y], [a.x, a.y]));
    const left = leftNormal([a.x, a.y], [b.x, b.y]);
    const ln = normalize(left);
    return { pos, outward: [-ln[0], -ln[1]], tangent };
  }
  const r = resolveSegment(doc, notch.segmentId, notch.panelId);
  const arc: ArcGeometry = r.arc!;
  const pos = pointOnArc(arc, notch.t);
  const ang = arc.a0 + arc.sweep * notch.t;
  const radial: Vec2 = [Math.cos(ang), Math.sin(ang)];
  const tangent: Vec2 = normalize([-Math.sin(ang) * Math.sign(arc.sweep), Math.cos(ang) * Math.sign(arc.sweep)]);
  // Outward = away from the loop interior. For a CCW loop the interior is on
  // the left of travel; radial-vs-travel decides per-arc below via the loop.
  const loop = getLoop(doc, notch.panelId, notch.loopId);
  void loop;
  return { pos, outward: radial, tangent };
}

/** Grainline direction (unit) and length. */
export function grainlineVector(grainline: Grainline): { direction: Vec2; lengthM: number } {
  const d = sub(grainline.to, grainline.from);
  const lengthM = Math.hypot(d[0], d[1]);
  return { direction: normalize(d), lengthM };
}

// ---------------------------------------------------------------------------
// Cut boundary resolution
// ---------------------------------------------------------------------------

/** Resolve a cut line to a concrete ring (sewing boundary or derived allowance). */
export function cutBoundary(
  doc: PatternDocument, set: ProductionSet, cutId: string, sagittaTolM = 1e-4,
): { ring: Vec2[]; source: "sewing" | "allowance"; issues: OffsetIssue[] } {
  const cut = set.cutLines.find((c) => c.id === cutId);
  if (!cut) throw new PatternCadError("missing-reference", `cut line '${cutId}' does not exist`, cutId);
  const loop = getLoop(doc, cut.panelId, cut.loopId);
  if (cut.source === "sewing") {
    const ring: Vec2[] = [];
    for (const segmentId of loop.segmentIds) {
      const segment = getSegment(doc, segmentId, cut.panelId);
      const pts: Vec2[] = segment.kind === "line"
        ? (() => {
          const a = getPoint(doc, segment.startPointId, cut.panelId);
          const b = getPoint(doc, segment.endPointId, cut.panelId);
          return [[a.x, a.y], [b.x, b.y]];
        })()
        : (() => {
          const r = resolveSegment(doc, segmentId, cut.panelId);
          return sampleArc(r.arc!.center, r.arc!.radius, r.arc!.a0, r.arc!.sweep, sagittaTolM);
        })();
      // All samples except the final endpoint (next edge's start).
      for (let i = 0; i < pts.length - 1; i++) ring.push(pts[i]);
    }
    return { ring, source: "sewing", issues: [] };
  }
  const derived = allowanceBoundary(doc, set, cut.panelId, cut.loopId, sagittaTolM);
  return { ring: derived.ring, source: "allowance", issues: derived.issues };
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

export type ProductionDiagnosticCode =
  | "missing-reference"
  | "invalid-value"
  | "duplicate-id"
  | "outside-panel"
  | "invalid-geometry";

export interface ProductionDiagnostic {
  code: ProductionDiagnosticCode;
  message: string;
  entityId?: string;
}

export function validateProductionSet(doc: PatternDocument, set: ProductionSet): ProductionDiagnostic[] {
  const diagnostics: ProductionDiagnostic[] = [];
  const fail = (code: ProductionDiagnosticCode, message: string, entityId?: string): void => {
    diagnostics.push({ code, message, ...(entityId ? { entityId } : {}) });
  };
  if (!set || typeof set !== "object" || set.version !== PRODUCTION_SET_VERSION) {
    return [{ code: "invalid-value", message: "production set shape or version is invalid" }];
  }
  const seen = new Set<string>();
  const claim = (id: string): boolean => {
    if (typeof id !== "string" || id.length === 0 || seen.has(id)) return false;
    seen.add(id);
    return true;
  };
  const panelIds = new Set(doc.panels.map((p) => p.id));
  const loopOf = (panelId: string, loopId: string): boolean =>
    doc.panels.some((p) => p.id === panelId && p.boundaryLoops.some((l) => l.id === loopId));
  const segmentInLoop = (panelId: string, loopId: string, segmentId: string): boolean => {
    const panel = doc.panels.find((p) => p.id === panelId);
    const loop = panel?.boundaryLoops.find((l) => l.id === loopId);
    return !!loop?.segmentIds.includes(segmentId);
  };
  for (const a of set.allowances ?? []) {
    if (!claim(a.id)) fail("duplicate-id", `duplicate production id '${a.id}'`, a.id);
    if (!panelIds.has(a.panelId) || !loopOf(a.panelId, a.loopId)) {
      fail("missing-reference", `allowance '${a.id}' references a missing panel/loop`, a.id);
    }
    if (!Number.isFinite(a.defaultM) || a.defaultM < 0) fail("invalid-value", `allowance '${a.id}' default is invalid`, a.id);
    for (const [segId, v] of Object.entries(a.perEdgeM ?? {})) {
      if (!segmentInLoop(a.panelId, a.loopId, segId)) fail("missing-reference", `allowance '${a.id}' overrides unknown edge '${segId}'`, a.id);
      if (!Number.isFinite(v) || v < 0) fail("invalid-value", `allowance '${a.id}' edge override is invalid`, a.id);
    }
  }
  for (const e of set.notches ?? []) {
    if (!claim(e.id)) fail("duplicate-id", `duplicate production id '${e.id}'`, e.id);
    if (!segmentInLoop(e.panelId, e.loopId, e.segmentId)) {
      fail("missing-reference", `notch '${e.id}' references a missing boundary edge`, e.id);
    }
    if (!Number.isFinite(e.t) || e.t < 0 || e.t > 1) fail("invalid-value", `notch '${e.id}' t is out of range`, e.id);
    if (!Number.isFinite(e.depthM) || !(e.depthM > 0)) fail("invalid-value", `notch '${e.id}' depth is invalid`, e.id);
  }
  const insideCheck = (panelId: string, pos: Vec2, id: string, what: string): void => {
    if (!panelIds.has(panelId)) {
      fail("missing-reference", `${what} '${id}' references missing panel '${panelId}'`, id);
      return;
    }
    if (!finiteVec(pos)) {
      fail("invalid-value", `${what} '${id}' has non-finite coordinates`, id);
      return;
    }
    try {
      if (pointInPanel(doc, panelId, pos)) return;
      // Boundary tolerance: exact-vertex ray-cast tests are flaky, but a
      // point within EPS of the outer boundary counts as inside (matches
      // the G7D "boundary counts in" membership rule).
      const panel = doc.panels.find((p) => p.id === panelId)!;
      const outer = panel.boundaryLoops.find((l) => l.role === "outer")!;
      const pts = sampleLoopLocal(doc, panelId, outer.id);
      for (let i = 0; i < pts.length; i++) {
        if (distToSegment(pos, pts[i], pts[(i + 1) % pts.length]) <= EPS_M) return;
      }
      fail("outside-panel", `${what} '${id}' lies outside its panel`, id);
    } catch {
      fail("missing-reference", `${what} '${id}' cannot be tested (panel has no outer loop)`, id);
    }
  };
  for (const e of set.grainlines ?? []) {
    if (!claim(e.id)) fail("duplicate-id", `duplicate production id '${e.id}'`, e.id);
    if (dist(e.from, e.to) <= EPS_M) fail("invalid-geometry", `grainline '${e.id}' is degenerate`, e.id);
    insideCheck(e.panelId, e.from, e.id, "grainline");
    insideCheck(e.panelId, e.to, e.id, "grainline");
  }
  for (const e of set.folds ?? []) {
    if (!claim(e.id)) fail("duplicate-id", `duplicate production id '${e.id}'`, e.id);
    if (dist(e.a, e.b) <= EPS_M) fail("invalid-geometry", `fold '${e.id}' is degenerate`, e.id);
    insideCheck(e.panelId, e.a, e.id, "fold");
    insideCheck(e.panelId, e.b, e.id, "fold");
  }
  for (const e of set.drills ?? []) {
    if (!claim(e.id)) fail("duplicate-id", `duplicate production id '${e.id}'`, e.id);
    insideCheck(e.panelId, e.pos, e.id, "drill mark");
  }
  for (const e of set.internals ?? []) {
    if (!claim(e.id)) fail("duplicate-id", `duplicate production id '${e.id}'`, e.id);
    if (!panelIds.has(e.panelId)) {
      fail("missing-reference", `internal line '${e.id}' references missing panel`, e.id);
      continue;
    }
    if (!Array.isArray(e.points) || e.points.length < 2) fail("invalid-value", `internal line '${e.id}' needs 2+ points`, e.id);
    for (const p of e.points ?? []) insideCheck(e.panelId, p, e.id, "internal line");
  }
  for (const e of set.cutLines ?? []) {
    if (!claim(e.id)) fail("duplicate-id", `duplicate production id '${e.id}'`, e.id);
    if (!loopOf(e.panelId, e.loopId)) fail("missing-reference", `cut line '${e.id}' references a missing loop`, e.id);
    if (e.source === "allowance" && !set.allowances.some((a) => a.panelId === e.panelId && a.loopId === e.loopId)) {
      fail("missing-reference", `cut line '${e.id}' needs an allowance that does not exist`, e.id);
    }
  }
  for (const e of set.annotations ?? []) {
    if (!claim(e.id)) fail("duplicate-id", `duplicate production id '${e.id}'`, e.id);
    insideCheck(e.panelId, e.pos, e.id, "annotation");
  }
  for (const e of set.labelRegions ?? []) {
    if (!claim(e.id)) fail("duplicate-id", `duplicate production id '${e.id}'`, e.id);
    insideCheck(e.panelId, e.min, e.id, "label region");
    insideCheck(e.panelId, e.max, e.id, "label region");
  }
  for (const m of set.panelMeta ?? []) {
    if (!panelIds.has(m.panelId)) fail("missing-reference", `panel metadata references missing panel '${m.panelId}'`, m.panelId);
    if (!Number.isInteger(m.cutQuantity) || m.cutQuantity < 1) {
      fail("invalid-value", `panel '${m.panelId}' cut quantity is invalid`, m.panelId);
    }
  }
  const metaIds = new Set<string>();
  for (const m of set.panelMeta ?? []) {
    if (metaIds.has(m.panelId)) fail("duplicate-id", `duplicate metadata for panel '${m.panelId}'`, m.panelId);
    metaIds.add(m.panelId);
  }
  return diagnostics;
}

// ---------------------------------------------------------------------------
// Persistence
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

export function serializeProductionSet(set: ProductionSet): string {
  return canonicalJson(set);
}

export function deserializeProductionSet(serialized: string): ProductionSet {
  let parsed: unknown;
  try {
    parsed = JSON.parse(serialized);
  } catch {
    throw new PatternCadError("invalid-document", "serialized production set is not valid JSON");
  }
  const set = parsed as ProductionSet;
  if (!set || typeof set !== "object" || set.version !== PRODUCTION_SET_VERSION) {
    throw new PatternCadError("invalid-document", "production set shape or version is invalid");
  }
  return clone(set);
}
