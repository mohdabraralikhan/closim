// G16A — darts, pleats, and gathers as persistent construction features.
//
// Architecture: features are PARAMETER records in a ConstructionSet
// sidecar; `deriveConstruction` turns them into pattern geometry on a fresh
// document clone. The authored polygon is never destructively rewritten and
// the semantics (intake, apex, depth, correspondence) survive every edit:
//
//   Construction Feature ──► Derived Pattern Geometry (+ seams + markings)
//
// Closed darts become sewable G8B seams (the solver never learns what a
// dart is). Intake/depth validation reports failures per feature instead of
// producing corrupt geometry.

import {
  PatternCadError,
  createPoint,
  movePoint,
  splitBoundarySegment,
  validatePatternDocument,
  type EntityId,
  type PatternDocument,
  type PatternSegment,
} from "../pattern/cad.js";
import { dist, lerp, segmentIntersection, type Vec2 } from "../cad/geom.js";
import {
  getLoop,
  getPanel,
  getPoint,
  getSegment,
} from "../cad/queries.js";
import { divideSegment } from "../cad/draft.js";
import type { Seam } from "../garment/sewing.js";

export const CONSTRUCTION_SET_VERSION = 1;
const EPS_M = 1e-9;

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function finiteVec(v: Vec2): boolean {
  return Number.isFinite(v[0]) && Number.isFinite(v[1]);
}

/** Replicates the kernel allocateId algorithm (same precedent as G9A ops). */
function allocateEntityId(doc: PatternDocument, kind: string): EntityId {
  const occupied = new Set<string>([
    doc.id,
    ...doc.panels.map((item) => item.id),
    ...doc.points.map((item) => item.id),
    ...doc.segments.map((item) => item.id),
    ...doc.panels.flatMap((panel) => [
      ...panel.boundaryLoops.map((loop) => loop.id),
      ...panel.dimensions.map((dimension) => dimension.id),
      ...panel.constraints.map((constraint) => constraint.id),
    ]),
  ]);
  for (;;) {
    const suffix = String(doc.nextEntityIndex++).padStart(8, "0");
    const id = `${doc.id}/${kind}/${suffix}`;
    if (!occupied.has(id)) return id;
  }
}

// ---------------------------------------------------------------------------
// Feature model
// ---------------------------------------------------------------------------

interface FeatureBase {
  id: string;
  panelId: EntityId;
  note?: string;
}

export interface DartFeature extends FeatureBase {
  kind: "dart";
  loopId: EntityId;
  /** Boundary edge the intake opens on. */
  edgeSegmentId: EntityId;
  /** Intake endpoints as fractions along the edge (0 < tA < tB < 1). */
  tA: number;
  tB: number;
  /** Dart apex in panel-local coordinates (interior point). */
  apex: Vec2;
  /** Open V vs sewn-closed (closed emits a leg-to-leg seam). */
  state: "open" | "closed";
}

export type PleatType = "knife" | "box" | "inverted";

export interface PleatFeature extends FeatureBase {
  kind: "pleat";
  loopId: EntityId;
  edgeSegmentId: EntityId;
  /** Placement fraction along the edge. */
  t: number;
  pleatType: PleatType;
  /** Fold depth in metres (> 0). */
  depthM: number;
  /** +1 folds to the left of travel, -1 to the right. */
  direction: 1 | -1;
}

export interface GatherFeature extends FeatureBase {
  kind: "gather";
  sourceLoopId: EntityId;
  sourceSegmentId: EntityId;
  /** Receiving edge (absent = free gather to a target length). */
  recvPanelId?: EntityId;
  recvLoopId?: EntityId;
  recvSegmentId?: EntityId;
  /** Target gathered length in metres (used when no receiving edge). */
  targetLengthM?: number;
  /** Notch marks to emit along the source edge. */
  notchCount: number;
}

export type ConstructionFeature = DartFeature | PleatFeature | GatherFeature;

export interface ConstructionSet {
  version: typeof CONSTRUCTION_SET_VERSION;
  nextIndex: number;
  features: ConstructionFeature[];
}

export function createConstructionSet(): ConstructionSet {
  return { version: CONSTRUCTION_SET_VERSION, nextIndex: 1, features: [] };
}

function allocateId(set: ConstructionSet, kind: string): string {
  return `construction/${kind}/${String(set.nextIndex++).padStart(8, "0")}`;
}

function pushFeature<T extends ConstructionFeature>(set: ConstructionSet, feature: T): { set: ConstructionSet; id: string } {
  const next = clone(set);
  next.features.push(clone(feature));
  return { set: next, id: feature.id };
}

export function addDart(
  set: ConstructionSet,
  panelId: EntityId,
  loopId: EntityId,
  edgeSegmentId: EntityId,
  tA: number,
  tB: number,
  apex: Vec2,
  state: "open" | "closed" = "open",
  note?: string,
): { set: ConstructionSet; id: string } {
  if (![tA, tB].every((t) => Number.isFinite(t)) || !(tA > 0) || !(tB < 1) || !(tA < tB)) {
    throw new PatternCadError("invalid-transform", "dart intake needs 0 < tA < tB < 1", edgeSegmentId);
  }
  if (!finiteVec(apex)) throw new PatternCadError("invalid-transform", "dart apex must be finite", panelId);
  if (state !== "open" && state !== "closed") throw new PatternCadError("invalid-transform", "dart state must be open or closed", panelId);
  const next = clone(set);
  const feature: DartFeature = {
    kind: "dart", id: allocateId(next, "dart"), panelId, loopId, edgeSegmentId, tA, tB,
    apex: [apex[0], apex[1]], state, ...(note ? { note } : {}),
  };
  return pushFeature(next, feature);
}

export function addPleat(
  set: ConstructionSet,
  panelId: EntityId,
  loopId: EntityId,
  edgeSegmentId: EntityId,
  t: number,
  pleatType: PleatType,
  depthM: number,
  direction: 1 | -1 = 1,
  note?: string,
): { set: ConstructionSet; id: string } {
  if (!Number.isFinite(t) || t <= 0 || t >= 1) {
    throw new PatternCadError("invalid-transform", "pleat placement needs t in (0,1)", edgeSegmentId);
  }
  if (pleatType !== "knife" && pleatType !== "box" && pleatType !== "inverted") {
    throw new PatternCadError("invalid-transform", `unknown pleat type '${pleatType}'`, edgeSegmentId);
  }
  if (!Number.isFinite(depthM) || !(depthM > 0)) {
    throw new PatternCadError("invalid-transform", "pleat depth must be positive", edgeSegmentId);
  }
  if (direction !== 1 && direction !== -1) {
    throw new PatternCadError("invalid-transform", "pleat direction must be +1 or -1", edgeSegmentId);
  }
  const next = clone(set);
  const feature: PleatFeature = {
    kind: "pleat", id: allocateId(next, "pleat"), panelId, loopId, edgeSegmentId, t,
    pleatType, depthM, direction, ...(note ? { note } : {}),
  };
  return pushFeature(next, feature);
}

export function addGather(
  set: ConstructionSet,
  sourcePanelId: EntityId,
  sourceLoopId: EntityId,
  sourceSegmentId: EntityId,
  opts: {
    recvPanelId?: EntityId;
    recvLoopId?: EntityId;
    recvSegmentId?: EntityId;
    targetLengthM?: number;
    notchCount?: number;
    note?: string;
  } = {},
): { set: ConstructionSet; id: string } {
  const notchCount = opts.notchCount ?? 3;
  if (!Number.isInteger(notchCount) || notchCount < 2) {
    throw new PatternCadError("invalid-transform", "gather needs at least 2 notches", sourceSegmentId);
  }
  const hasRecv = !!(opts.recvPanelId && opts.recvLoopId && opts.recvSegmentId);
  if (!hasRecv && (opts.targetLengthM === undefined || !(opts.targetLengthM > 0) || !Number.isFinite(opts.targetLengthM))) {
    throw new PatternCadError("invalid-transform", "gather needs a receiving edge or a positive target length", sourceSegmentId);
  }
  const next = clone(set);
  const feature: GatherFeature = {
    kind: "gather", id: allocateId(next, "gather"), panelId: sourcePanelId,
    sourceLoopId, sourceSegmentId, notchCount,
    ...(hasRecv ? { recvPanelId: opts.recvPanelId, recvLoopId: opts.recvLoopId, recvSegmentId: opts.recvSegmentId } : {}),
    ...(opts.targetLengthM !== undefined ? { targetLengthM: opts.targetLengthM } : {}),
    ...(opts.note ? { note: opts.note } : {}),
  };
  return pushFeature(next, feature);
}

export function removeFeature(set: ConstructionSet, id: string): ConstructionSet {
  if (!set.features.some((f) => f.id === id)) {
    throw new PatternCadError("missing-reference", `construction feature '${id}' does not exist`, id);
  }
  const next = clone(set);
  next.features = next.features.filter((f) => f.id !== id);
  return next;
}

export function updateFeature<T extends ConstructionFeature>(
  set: ConstructionSet, id: string, patch: Partial<T>,
): ConstructionSet {
  const next = clone(set);
  const index = next.features.findIndex((f) => f.id === id);
  if (index < 0) throw new PatternCadError("missing-reference", `construction feature '${id}' does not exist`, id);
  const merged = { ...next.features[index], ...clone(patch), id, kind: next.features[index].kind };
  next.features[index] = merged as ConstructionFeature;
  return next;
}

/** Move a dart's intake to a new edge (dart transfer/rotation around the apex). */
export function transferDart(
  set: ConstructionSet, id: string, edgeSegmentId: EntityId, tA: number, tB: number,
): ConstructionSet {
  const feature = set.features.find((f) => f.id === id);
  if (!feature || feature.kind !== "dart") {
    throw new PatternCadError("missing-reference", `dart '${id}' does not exist`, id);
  }
  if (!(tA > 0) || !(tB < 1) || !(tA < tB)) {
    throw new PatternCadError("invalid-transform", "dart intake needs 0 < tA < tB < 1", edgeSegmentId);
  }
  return updateFeature<DartFeature>(set, id, { edgeSegmentId, tA, tB });
}

// ---------------------------------------------------------------------------
// Validation (per-feature status; broken references never throw here)
// ---------------------------------------------------------------------------

export interface FeatureStatus {
  id: string;
  kind: string;
  ok: boolean;
  reason?: string;
}

function segmentEndpoints(doc: PatternDocument, segment: PatternSegment): [Vec2, Vec2] {
  const a = getPoint(doc, segment.startPointId, segment.panelId);
  const b = getPoint(doc, segment.endPointId, segment.panelId);
  return [[a.x, a.y], [b.x, b.y]];
}

function edgeLength(doc: PatternDocument, segment: PatternSegment): number {
  const [a, b] = segmentEndpoints(doc, segment);
  return dist(a, b);
}

function lookup(doc: PatternDocument, panelId: EntityId, loopId: EntityId, segmentId: EntityId):
  { panel: unknown; loop: unknown; segment: PatternSegment } | { error: string } {
  try {
    const panel = getPanel(doc, panelId);
    const loop = getLoop(doc, panelId, loopId);
    const segment = getSegment(doc, segmentId, panelId);
    if (!loop.segmentIds.includes(segmentId)) return { error: `segment '${segmentId}' is not in loop '${loopId}'` };
    void panel;
    return { panel, loop, segment };
  } catch (error) {
    return { error: error instanceof Error ? error.message : String(error) };
  }
}

export function validateFeatures(doc: PatternDocument, set: ConstructionSet): FeatureStatus[] {
  return set.features.map((feature) => {
    if (feature.kind === "dart") {
      const found = lookup(doc, feature.panelId, feature.loopId, feature.edgeSegmentId);
      if ("error" in found) return { id: feature.id, kind: feature.kind, ok: false, reason: found.error };
      if (found.segment.kind !== "line") {
        return { id: feature.id, kind: feature.kind, ok: false, reason: "dart intake needs a line edge" };
      }
      const [a, b] = segmentEndpoints(doc, found.segment);
      const intake = dist(a, b) * (feature.tB - feature.tA);
      if (!(intake > EPS_M)) return { id: feature.id, kind: feature.kind, ok: false, reason: "dart intake is degenerate" };
      if (dist(feature.apex, a) <= EPS_M || dist(feature.apex, b) <= EPS_M) {
        return { id: feature.id, kind: feature.kind, ok: false, reason: "dart apex coincides with the intake edge" };
      }
      // Legs must meet the loop only at the intake endpoints.
      const pA = lerp(a, b, feature.tA), pB = lerp(a, b, feature.tB);
      for (const [legStart, legEnd] of [[pA, feature.apex], [feature.apex, pB]] as Array<[Vec2, Vec2]>) {
        for (const otherId of (found.loop as { segmentIds: string[] }).segmentIds) {
          if (otherId === feature.edgeSegmentId) continue;
          const other = getSegment(doc, otherId, feature.panelId);
          if (other.kind !== "line") continue;
          const [c, d] = segmentEndpoints(doc, other);
          const hit = segmentIntersection(legStart, legEnd, c, d, EPS_M);
          if (hit && dist(hit.pos, legStart) > EPS_M && dist(hit.pos, legEnd) > EPS_M) {
            return { id: feature.id, kind: feature.kind, ok: false, reason: "dart leg crosses the panel boundary" };
          }
        }
      }
      return { id: feature.id, kind: feature.kind, ok: true };
    }
    if (feature.kind === "pleat") {
      const found = lookup(doc, feature.panelId, feature.loopId, feature.edgeSegmentId);
      if ("error" in found) return { id: feature.id, kind: feature.kind, ok: false, reason: found.error };
      if (!(feature.depthM > 0)) return { id: feature.id, kind: feature.kind, ok: false, reason: "pleat depth must be positive" };
      return { id: feature.id, kind: feature.kind, ok: true };
    }
    const found = lookup(doc, feature.panelId, feature.sourceLoopId, feature.sourceSegmentId);
    if ("error" in found) return { id: feature.id, kind: feature.kind, ok: false, reason: found.error };
    if (feature.recvSegmentId && feature.recvPanelId && feature.recvLoopId) {
      const recv = lookup(doc, feature.recvPanelId, feature.recvLoopId, feature.recvSegmentId);
      if ("error" in recv) return { id: feature.id, kind: feature.kind, ok: false, reason: `receiving edge: ${recv.error}` };
      if (!("error" in found) && edgeLength(doc, (recv as { segment: PatternSegment }).segment) >= edgeLength(doc, found.segment)) {
        return { id: feature.id, kind: feature.kind, ok: false, reason: "gather source must be longer than the receiving edge" };
      }
    }
    return { id: feature.id, kind: feature.kind, ok: true };
  });
}

// ---------------------------------------------------------------------------
// Derivation
// ---------------------------------------------------------------------------

export interface DerivedNotch {
  panelId: EntityId;
  loopId: EntityId;
  segmentId: EntityId;
  t: number;
  kind: "single" | "double";
  depthM: number;
}

export interface DerivedFold {
  panelId: EntityId;
  a: Vec2;
  b: Vec2;
  label: string;
}

export interface DerivationReport {
  document: PatternDocument;
  /** Seams to append to the garment seam graph (dart closures). */
  seams: Seam[];
  /** Fold lines for the production set (G11A addFoldLine inputs). */
  folds: DerivedFold[];
  /** Notches for the production set (G11A addNotch inputs). */
  notches: DerivedNotch[];
  applied: string[];
  failed: Array<{ id: string; reason: string }>;
}

/**
 * Apply every feature in stored order to a fresh clone. Broken features are
 * skipped and reported (never silent, never corrupting). The input document
 * is never mutated.
 */
export function deriveConstruction(doc: PatternDocument, set: ConstructionSet): DerivationReport {
  const statuses = new Map(validateFeatures(doc, set).map((s) => [s.id, s]));
  let next = clone(doc);
  const seams: Seam[] = [];
  const folds: DerivedFold[] = [];
  const notches: DerivedNotch[] = [];
  const applied: string[] = [];
  const failed: Array<{ id: string; reason: string }> = [];

  for (const feature of set.features) {
    const status = statuses.get(feature.id)!;
    if (!status.ok) {
      failed.push({ id: feature.id, reason: status.reason ?? "invalid" });
      continue;
    }
    try {
      if (feature.kind === "dart") next = deriveDart(next, feature, seams, folds);
      else if (feature.kind === "pleat") next = derivePleat(next, feature, notches);
      else next = deriveGather(next, feature, notches);
      applied.push(feature.id);
    } catch (error) {
      failed.push({ id: feature.id, reason: error instanceof Error ? error.message : String(error) });
    }
  }
  const validation = validatePatternDocument(next);
  if (!validation.valid) {
    const first = validation.diagnostics[0];
    throw new PatternCadError(first.code, `construction derivation produced invalid geometry: ${first.message}`, first.entityId);
  }
  return { document: next, seams, folds, notches, applied, failed };
}

function splitEdgeAt(
  doc: PatternDocument, panelId: EntityId, loopId: EntityId, segmentId: EntityId, t: number,
): { document: PatternDocument; pointId: EntityId } {
  const r = splitBoundarySegment(doc, panelId, loopId, segmentId, t);
  return { document: r.document, pointId: r.pointId };
}

function deriveDart(
  doc: PatternDocument, feature: DartFeature, seams: Seam[], folds: DerivedFold[],
): PatternDocument {
  // Split at tB first, then at tA/tB within the head piece (params stay valid).
  let next = doc;
  const first = splitEdgeAt(next, feature.panelId, feature.loopId, feature.edgeSegmentId, feature.tB);
  next = first.document;
  const loop = getLoop(next, feature.panelId, feature.loopId);
  const headId = loop.segmentIds.find((id) => {
    const s = getSegment(next, id, feature.panelId);
    return s.startPointId !== first.pointId && s.endPointId === first.pointId;
  })!;
  const tAinHead = feature.tA / feature.tB;
  const second = splitEdgeAt(next, feature.panelId, feature.loopId, headId, tAinHead);
  next = second.document;
  // Middle piece runs tA-point -> tB-point along the edge.
  const loopAfter = getLoop(next, feature.panelId, feature.loopId);
  const middleId = loopAfter.segmentIds.find((id) => {
    const s = getSegment(next, id, feature.panelId);
    return s.startPointId === second.pointId && s.endPointId === first.pointId;
  })!;
  const apexCreated = createPoint(next, feature.panelId, [feature.apex[0], feature.apex[1]], "boundary");
  next = apexCreated.document;
  const apexId = apexCreated.pointId;
  // Replace the middle edge piece with two legs: tA -> apex -> tB.
  const legAId = allocateEntityId(next, "segment");
  const legBId = allocateEntityId(next, "segment");
  const middle = getSegment(next, middleId, feature.panelId);
  next.segments = next.segments.filter((s) => s.id !== middleId);
  next.segments.push(
    { id: legAId, panelId: feature.panelId, role: "boundary", kind: "line", startPointId: middle.startPointId, endPointId: apexId },
    { id: legBId, panelId: feature.panelId, role: "boundary", kind: "line", startPointId: apexId, endPointId: middle.endPointId },
  );
  const targetLoop = getPanel(next, feature.panelId).boundaryLoops.find((l) => l.id === feature.loopId)!;
  targetLoop.segmentIds.splice(targetLoop.segmentIds.indexOf(middleId), 1, legAId, legBId);
  // Legs become press lines; a closed dart additionally becomes a seam.
  const apexPos: Vec2 = [feature.apex[0], feature.apex[1]];
  const tA = getPoint(next, middle.startPointId, feature.panelId);
  const tB = getPoint(next, middle.endPointId, feature.panelId);
  folds.push(
    { panelId: feature.panelId, a: [tA.x, tA.y], b: apexPos, label: `dart ${feature.id} fold` },
    { panelId: feature.panelId, a: apexPos, b: [tB.x, tB.y], label: `dart ${feature.id} fold` },
  );
  if (feature.state === "closed") {
    const intake = Math.hypot(tB.x - tA.x, tB.y - tA.y);
    seams.push({
      id: `seam/${feature.id}`,
      sideA: { panelId: feature.panelId, loopId: feature.loopId, segmentIds: [legAId], reversed: false },
      sideB: { panelId: feature.panelId, loopId: feature.loopId, segmentIds: [legBId], reversed: true },
      stitchCount: Math.max(2, Math.round(intake / 0.01) + 1),
      metadata: { kind: "dart-closure" },
    });
  }
  return next;
}

function derivePleat(
  doc: PatternDocument, feature: PleatFeature, notches: DerivedNotch[],
): PatternDocument {
  const segment = getSegment(doc, feature.edgeSegmentId, feature.panelId);
  if (segment.kind !== "line") {
    throw new PatternCadError("unsupported-operation", "pleats need a line edge", feature.edgeSegmentId);
  }
  const [a, b] = segmentEndpoints(doc, segment);
  const edgeLen = dist(a, b);
  const folds = feature.pleatType === "knife" ? 1 : 2;
  const intake = folds * 2 * feature.depthM;
  // Extend the edge endpoint outward along its direction by the intake.
  const dir: Vec2 = [(b[0] - a[0]) / edgeLen, (b[1] - a[1]) / edgeLen];
  let next = movePoint(doc, feature.panelId, segment.endPointId, [b[0] + dir[0] * intake, b[1] + dir[1] * intake]);
  // Subdivide the extended edge into fold sections and mark each fold.
  const total = edgeLen + intake;
  const sectionLen = feature.depthM;
  const marks: number[] = [];
  if (feature.pleatType === "knife") {
    marks.push(edgeLen / total, (edgeLen + sectionLen) / total);
  } else {
    // Box/inverted: mirrored fold pair around the placement point.
    marks.push(
      Math.max(0.02, feature.t - (2 * sectionLen) / total),
      Math.max(0.03, feature.t - sectionLen / total),
      Math.min(0.97, feature.t + sectionLen / total),
      Math.min(0.98, feature.t + (2 * sectionLen) / total),
    );
  }
  // Subdivide at each mark. Marks are fractions of the ORIGINAL extended
  // edge; each split applies to the current tail piece, so the parameter is
  // rebased: rel = (t - consumed) / (1 - consumed). Fold points are recorded
  // and resolved to their CURRENT segments only after all splits (earlier
  // tail pieces are consumed by later splits).
  let edgeId = feature.edgeSegmentId;
  const foldPointIds: string[] = [];
  let consumed = 0;
  for (const t of [...marks].sort((x, y) => x - y)) {
    if (!(t > 0.01) || !(t < 0.99) || !(t > consumed + 0.005)) continue;
    const rel = (t - consumed) / (1 - consumed);
    if (!(rel > 0.005) || !(rel < 0.995)) continue;
    const split = splitEdgeAt(next, feature.panelId, feature.loopId, edgeId, rel);
    next = split.document;
    consumed = t;
    foldPointIds.push(split.pointId);
    const loopNow = getLoop(next, feature.panelId, feature.loopId);
    const tail = loopNow.segmentIds.find((id) => getSegment(next, id, feature.panelId).startPointId === split.pointId);
    if (!tail) continue;
    edgeId = tail;
  }
  for (const pid of foldPointIds) {
    const segId = getLoop(next, feature.panelId, feature.loopId).segmentIds
      .find((id) => getSegment(next, id, feature.panelId).startPointId === pid);
    if (!segId) continue;
    notches.push({
      panelId: feature.panelId,
      loopId: feature.loopId,
      segmentId: segId,
      t: 0.02, // just past the fold point (strictly interior)
      kind: "single" as const,
      depthM: 0.004,
    });
  }
  return next;
}

function deriveGather(
  doc: PatternDocument, feature: GatherFeature, notches: DerivedNotch[],
): PatternDocument {
  const divided = divideSegment(doc, feature.panelId, feature.sourceSegmentId, feature.notchCount + 1);
  notches.push(
    ...divided.pointIds.map((pid, i): DerivedNotch => ({
      panelId: feature.panelId,
      loopId: feature.sourceLoopId,
      segmentId: feature.sourceSegmentId,
      t: (i + 1) / (feature.notchCount + 1),
      kind: i % 2 === 0 ? ("single" as const) : ("double" as const),
      depthM: 0.004,
    })),
  );
  return divided.document;
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

export function serializeConstructionSet(set: ConstructionSet): string {
  return canonicalJson(set);
}

export function deserializeConstructionSet(serialized: string): ConstructionSet {
  let parsed: unknown;
  try {
    parsed = JSON.parse(serialized);
  } catch {
    throw new PatternCadError("invalid-document", "serialized construction set is not valid JSON");
  }
  const set = parsed as ConstructionSet;
  if (!set || typeof set !== "object" || set.version !== CONSTRUCTION_SET_VERSION || !Array.isArray(set.features)) {
    throw new PatternCadError("invalid-document", "construction set shape or version is invalid");
  }
  return clone(set);
}
