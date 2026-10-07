// G9B — garment drafting tools over the G8A PatternDocument.
//
// Scope relative to G9A (src/cad/ops.ts): ops.ts owns point/segment editing
// verbs (move, delete-point, merge, trim-by-reference, extend, mirror,
// split-segment). This module owns *drafting construction*: creating
// geometry from geometric relationships (angle, perpendicular, parallel,
// midpoint, division, intersection), loop-level offsetting, panel-level
// split/join/arrange, and explicit seam-invalidation reporting.
//
// Conventions (aligned with cad.ts / ops.ts):
//   - Immutable ops on panel-local coordinates; fresh documents returned.
//   - Preconditions throw PatternCadError with cad.ts diagnostic codes.
//   - Construction geometry never becomes boundary geometry implicitly:
//     callers choose the target explicitly.
//   - Determinism: kernel creators allocate ids; sequences replay identically.

import {
  PatternCadError,
  createBoundaryLine,
  createBoundaryLoop,
  createConstructionLine,
  createPanel,
  createPoint,
  deletePanel,
  movePoint,
  validatePatternDocument,
  type EntityId,
  type PatternDocument,
  type PatternSegment,
} from "../pattern/cad.js";
import {
  dist,
  leftNormal,
  lerp,
  lineIntersection,
  normalize,
  pointOnArc,
  reflectAcrossLine,
  rotateAround,
  segmentIntersection,
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
  intersectSegments,
  resolveSegment,
} from "./queries.js";

const EPS_M = 1e-9;
const MIN_AREA_M2 = 1e-12;

function finite(value: number): boolean {
  return Number.isFinite(value);
}

function finiteVec(v: Vec2): boolean {
  return finite(v[0]) && finite(v[1]);
}

function cloneDoc(doc: PatternDocument): PatternDocument {
  return JSON.parse(JSON.stringify(doc)) as PatternDocument;
}

export type DraftTarget =
  | { role: "construction" }
  | { role: "boundary"; loopId: EntityId };

/** Minimal structural seam view for invalidation reporting (G8B Seam satisfies this). */
export interface SeamRef {
  id: string;
  sideA: { panelId: string };
  sideB: { panelId: string };
}

/** Resolve a Vec2-or-point-id into panel-local coordinates (creating the point when given coordinates). */
function resolveOrCreatePoint(
  doc: PatternDocument,
  panelId: EntityId,
  at: Vec2 | EntityId,
  role: "boundary" | "construction",
): { document: PatternDocument; pointId: EntityId; pos: Vec2 } {
  if (typeof at === "string") {
    const p = getPoint(doc, at, panelId);
    return { document: doc, pointId: p.id, pos: [p.x, p.y] };
  }
  if (!finiteVec(at)) {
    throw new PatternCadError("invalid-transform", "drafting position must be finite", panelId);
  }
  const created = createPoint(doc, panelId, [at[0], at[1]], role);
  return { document: created.document, pointId: created.pointId, pos: [at[0], at[1]] };
}

function lineEndpoints(doc: PatternDocument, segment: PatternSegment): [Vec2, Vec2] {
  if (segment.kind !== "line") {
    throw new PatternCadError("unsupported-operation", "drafting relation needs a line segment", segment.id);
  }
  const a = getPoint(doc, segment.startPointId, segment.panelId);
  const b = getPoint(doc, segment.endPointId, segment.panelId);
  return [[a.x, a.y], [b.x, b.y]];
}

function newLine(
  doc: PatternDocument,
  panelId: EntityId,
  fromId: EntityId,
  toId: EntityId,
  target: DraftTarget,
): { document: PatternDocument; segmentId: EntityId } {
  if (target.role === "construction") {
    return createConstructionLine(doc, panelId, fromId, toId);
  }
  const loop = getLoop(doc, panelId, target.loopId);
  void loop;
  return createBoundaryLine(doc, panelId, target.loopId, fromId, toId);
}

// ---------------------------------------------------------------------------
// Construction: lines from relationships
// ---------------------------------------------------------------------------

/** Straight line between two positions (existing points or coordinates). */
export function draftLine(
  doc: PatternDocument,
  panelId: EntityId,
  from: Vec2 | EntityId,
  to: Vec2 | EntityId,
  target: DraftTarget = { role: "construction" },
): { document: PatternDocument; segmentId: EntityId } {
  getPanel(doc, panelId);
  const role = target.role;
  const r1 = resolveOrCreatePoint(doc, panelId, from, role);
  const r2 = resolveOrCreatePoint(r1.document, panelId, to, role);
  if (dist(r1.pos, r2.pos) <= EPS_M) {
    throw new PatternCadError("zero-length-edge", "drafted line has zero length", panelId);
  }
  return newLine(r2.document, panelId, r1.pointId, r2.pointId, target);
}

/** Line of explicit length starting at an origin along an absolute angle. */
export function lineAtAngle(
  doc: PatternDocument,
  panelId: EntityId,
  origin: Vec2 | EntityId,
  angleRad: number,
  lengthM: number,
  target: DraftTarget = { role: "construction" },
): { document: PatternDocument; segmentId: EntityId } {
  if (!finite(angleRad) || !finite(lengthM) || !(lengthM > EPS_M)) {
    throw new PatternCadError("invalid-transform", "line angle must be finite and length positive", panelId);
  }
  const r = resolveOrCreatePoint(doc, panelId, origin, target.role);
  const end: Vec2 = [r.pos[0] + lengthM * Math.cos(angleRad), r.pos[1] + lengthM * Math.sin(angleRad)];
  return draftLine(r.document, panelId, r.pointId, end, target);
}

/** Perpendicular line through the point at parameter t of a line segment. */
export function perpendicularAt(
  doc: PatternDocument,
  panelId: EntityId,
  segmentId: EntityId,
  t: number,
  lengthM: number,
  target: DraftTarget = { role: "construction" },
): { document: PatternDocument; segmentId: EntityId } {
  const segment = getSegment(doc, segmentId, panelId);
  if (!finite(t) || t < 0 || t > 1 || !finite(lengthM) || !(lengthM > EPS_M)) {
    throw new PatternCadError("invalid-transform", "perpendicular needs t in [0,1] and positive length", segmentId);
  }
  const [a, b] = lineEndpoints(doc, segment);
  const base: Vec2 = lerp(a, b, t);
  const n = normalize(leftNormal(a, b));
  const half = lengthM / 2;
  return draftLine(doc, panelId,
    [base[0] - n[0] * half, base[1] - n[1] * half],
    [base[0] + n[0] * half, base[1] + n[1] * half],
    target);
}

/** Line parallel to a line segment through a point (length defaults to the source length). */
export function parallelThrough(
  doc: PatternDocument,
  panelId: EntityId,
  segmentId: EntityId,
  through: Vec2 | EntityId,
  lengthM?: number,
  target: DraftTarget = { role: "construction" },
): { document: PatternDocument; segmentId: EntityId } {
  const segment = getSegment(doc, segmentId, panelId);
  const [a, b] = lineEndpoints(doc, segment);
  const dir = normalize(sub(b, a));
  const r = resolveOrCreatePoint(doc, panelId, through, target.role);
  const total = lengthM ?? dist(a, b);
  if (!finite(total) || !(total > EPS_M)) {
    throw new PatternCadError("invalid-transform", "parallel line needs a positive length", segmentId);
  }
  const half = total / 2;
  return draftLine(r.document, panelId,
    [r.pos[0] - dir[0] * half, r.pos[1] - dir[1] * half],
    [r.pos[0] + dir[0] * half, r.pos[1] + dir[1] * half],
    target);
}

/** Construction point at the midpoint of a line segment. */
export function midpointPoint(
  doc: PatternDocument,
  panelId: EntityId,
  segmentId: EntityId,
): { document: PatternDocument; pointId: EntityId } {
  const segment = getSegment(doc, segmentId, panelId);
  const [a, b] = lineEndpoints(doc, segment);
  return createPoint(doc, panelId, [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], "construction");
}

/** Divide a segment into n equal parts with n-1 construction points (lines and arcs). */
export function divideSegment(
  doc: PatternDocument,
  panelId: EntityId,
  segmentId: EntityId,
  divisions: number,
): { document: PatternDocument; pointIds: EntityId[] } {
  if (!Number.isInteger(divisions) || divisions < 2) {
    throw new PatternCadError("unsupported-operation", "divide needs an integer >= 2", segmentId);
  }
  const segment = getSegment(doc, segmentId, panelId);
  let next = doc;
  const pointIds: EntityId[] = [];
  if (segment.kind === "line") {
    const [a, b] = lineEndpoints(doc, segment);
    for (let i = 1; i < divisions; i++) {
      const p = lerp(a, b, i / divisions);
      const created = createPoint(next, panelId, p, "construction");
      next = created.document;
      pointIds.push(created.pointId);
    }
    return { document: next, pointIds };
  }
  const resolved = resolveSegment(doc, segmentId, panelId);
  const arc: ArcGeometry = resolved.arc!;
  for (let i = 1; i < divisions; i++) {
    const created = createPoint(next, panelId, pointOnArc(arc, i / divisions), "construction");
    next = created.document;
    pointIds.push(created.pointId);
  }
  return { document: next, pointIds };
}

/** Construction point at the intersection of two same-panel segments. */
export function intersectionPoint(
  doc: PatternDocument,
  panelId: EntityId,
  segmentAId: EntityId,
  segmentBId: EntityId,
): { document: PatternDocument; pointId: EntityId } {
  const segA = getSegment(doc, segmentAId, panelId);
  const segB = getSegment(doc, segmentBId, panelId);
  void segA;
  void segB;
  const hits = intersectSegments(doc, segmentAId, segmentBId);
  if (hits.length === 0) {
    throw new PatternCadError(
      "unsupported-operation",
      `segments '${segmentAId}' and '${segmentBId}' do not intersect`,
      segmentAId,
    );
  }
  return createPoint(doc, panelId, hits[0], "construction");
}

// ---------------------------------------------------------------------------
// Trim / extend by explicit distance or cutter
// ---------------------------------------------------------------------------

/** Move one endpoint of a line segment along its direction by a distance (positive lengthens). */
export function extendSegmentByDistance(
  doc: PatternDocument,
  panelId: EntityId,
  segmentId: EntityId,
  end: "start" | "end",
  distanceM: number,
): PatternDocument {
  const segment = getSegment(doc, segmentId, panelId);
  if (!finite(distanceM) || Math.abs(distanceM) <= EPS_M) {
    throw new PatternCadError("invalid-transform", "extension distance must be finite and nonzero", segmentId);
  }
  const [a, b] = lineEndpoints(doc, segment);
  const dir = end === "end" ? normalize(sub(b, a)) : normalize(sub(a, b));
  const anchor = end === "end" ? b : a;
  const moved: Vec2 = [anchor[0] + dir[0] * distanceM, anchor[1] + dir[1] * distanceM];
  const pointId = end === "end" ? segment.endPointId : segment.startPointId;
  return movePoint(doc, panelId, pointId, moved);
}

/**
 * Trim a line segment to its intersection with a cutter segment, keeping the
 * selected end fixed and moving the other endpoint to the crossing.
 */
export function trimSegmentToSegment(
  doc: PatternDocument,
  panelId: EntityId,
  segmentId: EntityId,
  cutterId: EntityId,
  keepEnd: "start" | "end",
): PatternDocument {
  const segment = getSegment(doc, segmentId, panelId);
  const cutter = getSegment(doc, cutterId, panelId);
  const [a, b] = lineEndpoints(doc, segment);
  const [c, d] = lineEndpoints(doc, cutter);
  const hit = segmentIntersection(a, b, c, d, EPS_M);
  if (!hit) {
    throw new PatternCadError(
      "unsupported-operation",
      `segment '${segmentId}' does not cross cutter '${cutterId}'`,
      segmentId,
    );
  }
  const movingId = keepEnd === "start" ? segment.endPointId : segment.startPointId;
  return movePoint(doc, panelId, movingId, hit.pos);
}

// ---------------------------------------------------------------------------
// Loop offset (line loops; new panel)
// ---------------------------------------------------------------------------

function outwardNormal(a: Vec2, b: Vec2, ccw: boolean): Vec2 {
  const n = normalize(leftNormal(a, b));
  return ccw ? [-n[0], -n[1]] : n;
}

/**
 * Offset a line-only outer loop outward (positive distance) into a NEW panel.
 * Miter joins; arcs and holes are rejected explicitly. The result is gated
 * through document validation — an offset that self-intersects throws
 * instead of producing garbage.
 */
export function offsetLoop(
  doc: PatternDocument,
  panelId: EntityId,
  loopId: EntityId,
  distanceM: number,
  newName?: string,
): { document: PatternDocument; panelId: EntityId } {
  const panel = getPanel(doc, panelId);
  const loop = getLoop(doc, panelId, loopId);
  if (!finite(distanceM) || Math.abs(distanceM) <= EPS_M) {
    throw new PatternCadError("invalid-transform", "offset distance must be finite and nonzero", loopId);
  }
  const resolved = loop.segmentIds.map((id) => getSegment(doc, id, panelId));
  if (resolved.some((s) => s.kind !== "line")) {
    throw new PatternCadError("unsupported-operation", "loop offset supports line segments only", loopId);
  }
  if (panel.boundaryLoops.some((l) => l.role === "hole")) {
    throw new PatternCadError("unsupported-operation", "loop offset does not support panels with holes", panelId);
  }
  const pts: Vec2[] = resolved.map((s) => {
    const p = getPoint(doc, s.startPointId, panelId);
    return [p.x, p.y] as Vec2;
  });
  const ccw = signedArea(pts) > 0;
  const n = pts.length;
  const out: Vec2[] = [];
  for (let i = 0; i < n; i++) {
    const prev = pts[(i + n - 1) % n], cur = pts[i], nextP = pts[(i + 1) % n];
    const n1 = outwardNormal(prev, cur, ccw);
    const n2 = outwardNormal(cur, nextP, ccw);
    const l1a: Vec2 = [prev[0] + n1[0] * distanceM, prev[1] + n1[1] * distanceM];
    const l1b: Vec2 = [cur[0] + n1[0] * distanceM, cur[1] + n1[1] * distanceM];
    const l2a: Vec2 = [cur[0] + n2[0] * distanceM, cur[1] + n2[1] * distanceM];
    const l2b: Vec2 = [nextP[0] + n2[0] * distanceM, nextP[1] + n2[1] * distanceM];
    const cross = lineIntersection(l1a, l1b, l2a, l2b, EPS_M);
    if (cross) {
      out.push(cross);
    } else {
      // Collinear neighbours: translate along the averaged normal.
      out.push([cur[0] + ((n1[0] + n2[0]) / 2) * distanceM, cur[1] + ((n1[1] + n2[1]) / 2) * distanceM]);
    }
  }
  let next = buildPanelFromRing(doc, newName ?? `${panel.name} offset`, panel.materialId, panel.grainAngleRad, out);
  const validation = validatePatternDocument(next.document);
  if (!validation.valid) {
    const first = validation.diagnostics[0];
    throw new PatternCadError(first.code, `offset produced invalid geometry: ${first.message}`, first.entityId);
  }
  return next;
}

/** Build a fresh panel from a closed ring (fresh ids; orientation recorded from geometry). */
export function buildPanelFromRing(
  doc: PatternDocument,
  name: string,
  materialId: string,
  grainAngleRad: number,
  ring: Vec2[],
): { document: PatternDocument; panelId: EntityId } {
  if (ring.length < 3) {
    throw new PatternCadError("degenerate-panel", "panel ring needs at least 3 points");
  }
  if (Math.abs(signedArea(ring)) <= MIN_AREA_M2) {
    throw new PatternCadError("degenerate-panel", "panel ring has zero area");
  }
  let next = doc;
  const created = createPanel(next, name, materialId);
  next = created.document;
  const panelId = created.panelId;
  const orientation = signedArea(ring) > 0 ? "ccw" : "cw";
  const loop = createBoundaryLoop(next, panelId, "outer", orientation);
  next = loop.document;
  const pointIds: EntityId[] = [];
  for (const p of ring) {
    if (!finiteVec(p)) {
      throw new PatternCadError("invalid-transform", "panel ring must be finite", panelId);
    }
    const c = createPoint(next, panelId, [p[0], p[1]], "boundary");
    next = c.document;
    pointIds.push(c.pointId);
  }
  for (let i = 0; i < pointIds.length; i++) {
    const r = createBoundaryLine(next, panelId, loop.loopId, pointIds[i], pointIds[(i + 1) % pointIds.length]);
    next = r.document;
  }
  const panel = getPanel(next, panelId);
  panel.grainAngleRad = grainAngleRad;
  return { document: next, panelId };
}

// ---------------------------------------------------------------------------
// Panel-level geometry transforms (move actual points, not the transform)
// ---------------------------------------------------------------------------

/** Translate every point of a panel in local coordinates. */
export function translatePanelGeometry(doc: PatternDocument, panelId: EntityId, delta: Vec2): PatternDocument {
  getPanel(doc, panelId);
  if (!finiteVec(delta)) {
    throw new PatternCadError("invalid-transform", "translation must be finite", panelId);
  }
  const next = cloneDoc(doc);
  for (const point of next.points) {
    if (point.panelId !== panelId) continue;
    point.x += delta[0];
    point.y += delta[1];
  }
  return next;
}

/** Rotate every point of a panel around a local pivot (arc centers rotate consistently; sweeps kept). */
export function rotatePanelGeometry(
  doc: PatternDocument,
  panelId: EntityId,
  pivotLocal: Vec2,
  angleRad: number,
): PatternDocument {
  getPanel(doc, panelId);
  if (!finiteVec(pivotLocal) || !finite(angleRad)) {
    throw new PatternCadError("invalid-transform", "rotation pivot and angle must be finite", panelId);
  }
  const next = cloneDoc(doc);
  for (const point of next.points) {
    if (point.panelId !== panelId) continue;
    const r = rotateAround([point.x, point.y], pivotLocal, angleRad);
    point.x = r[0];
    point.y = r[1];
  }
  return next;
}

/** Mirror every point of a panel across a LOCAL line (sweeps negated, orientations flipped). */
export function mirrorPanelAcrossLine(
  doc: PatternDocument,
  panelId: EntityId,
  lineA: Vec2,
  lineB: Vec2,
): PatternDocument {
  getPanel(doc, panelId);
  if (!finiteVec(lineA) || !finiteVec(lineB)) {
    throw new PatternCadError("invalid-transform", "mirror line must be finite", panelId);
  }
  if (dist(lineA, lineB) <= EPS_M) {
    throw new PatternCadError("invalid-transform", "mirror line is degenerate", panelId);
  }
  const next = cloneDoc(doc);
  for (const point of next.points) {
    if (point.panelId !== panelId) continue;
    const r = reflectAcrossLine([point.x, point.y], lineA, lineB);
    point.x = r[0];
    point.y = r[1];
  }
  for (const segment of next.segments) {
    if (segment.panelId !== panelId) continue;
    if (segment.kind === "arc") segment.sweepRad = -segment.sweepRad;
  }
  const panel = getPanel(next, panelId);
  for (const loop of panel.boundaryLoops) {
    loop.orientation = loop.orientation === "ccw" ? "cw" : "ccw";
  }
  return next;
}

/** Translate several panels' geometry in one deterministic pass (document order of the moves array). */
export function arrangePanels(
  doc: PatternDocument,
  moves: Array<{ panelId: EntityId; delta: Vec2 }>,
): PatternDocument {
  let next = doc;
  for (const move of moves) {
    next = translatePanelGeometry(next, move.panelId, move.delta);
  }
  return next;
}

// ---------------------------------------------------------------------------
// Split / join with explicit seam-invalidation reporting
// ---------------------------------------------------------------------------

/** Seam ids whose sideA/sideB reference any of the given panels. */
export function invalidatedSeamIdsForPanels(
  seams: ReadonlyArray<SeamRef>,
  panelIds: ReadonlyArray<EntityId>,
): string[] {
  const dead = new Set(panelIds);
  return seams
    .filter((s) => dead.has(s.sideA.panelId) || dead.has(s.sideB.panelId))
    .map((s) => s.id);
}

/**
 * Split a panel's line-only outer loop (no holes) by an infinite cutter line.
 * Produces two fresh panels and deletes the original. All seams touching the
 * original panel are reported invalid — their loop/segment references die
 * with it. Never silently reconnects geometry.
 */
export function splitPanelByLine(
  doc: PatternDocument,
  panelId: EntityId,
  lineA: Vec2,
  lineB: Vec2,
  seams: ReadonlyArray<SeamRef> = [],
): { document: PatternDocument; panelIds: [EntityId, EntityId]; invalidatedSeamIds: string[] } {
  const panel = getPanel(doc, panelId);
  if (!finiteVec(lineA) || !finiteVec(lineB) || dist(lineA, lineB) <= EPS_M) {
    throw new PatternCadError("invalid-transform", "cutter line is degenerate", panelId);
  }
  const outer = panel.boundaryLoops.find((l) => l.role === "outer");
  if (!outer) throw new PatternCadError("open-boundary", "panel has no outer loop", panelId);
  if (panel.boundaryLoops.some((l) => l.role === "hole")) {
    throw new PatternCadError("unsupported-operation", "panel split does not support holes", panelId);
  }
  for (const id of outer.segmentIds) {
    if (getSegment(doc, id, panelId).kind !== "line") {
      throw new PatternCadError("unsupported-operation", "panel split supports line loops only", id);
    }
  }
  const ring: Vec2[] = outer.segmentIds.map((id) => {
    const s = getSegment(doc, id, panelId);
    const p = getPoint(doc, s.startPointId, panelId);
    return [p.x, p.y] as Vec2;
  });
  const dx = lineB[0] - lineA[0], dy = lineB[1] - lineA[1];
  const len = Math.hypot(dx, dy);
  const sd = ring.map((p) => ((p[0] - lineA[0]) * dy - (p[1] - lineA[1]) * dx) / len);
  if (sd.every((v) => Math.abs(v) <= EPS_M)) {
    throw new PatternCadError("unsupported-operation", "cutter lies along the panel boundary", panelId);
  }
  if (sd.every((v) => v > -EPS_M) || sd.every((v) => v < EPS_M)) {
    throw new PatternCadError("unsupported-operation", "cutter misses the panel", panelId);
  }
  // Walk edges, collecting crossings in loop order.
  const augmented: Vec2[] = [];
  const crossingAt: number[] = []; // augmented indices that are crossings
  const n = ring.length;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    augmented.push(ring[i]);
    const si = sd[i], sj = sd[j];
    if (Math.abs(sj) <= EPS_M) {
      // Touching vertex j: it is itself a crossing; record when emitted next round.
      continue;
    }
    if (si * sj < 0) {
      const t = si / (si - sj);
      augmented.push(lerp(ring[i], ring[j], t));
      crossingAt.push(augmented.length - 1);
    }
  }
  // Vertices within eps of the line are crossings too.
  const augmentedCross = new Set(crossingAt);
  {
    let k = 0;
    const perEdge: number[] = [];
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      perEdge.push(k); // augmented index of ring[i]
      k++;
      if (!(Math.abs(sd[j]) <= EPS_M) && sd[i] * sd[j] < 0) k++;
    }
    for (let i = 0; i < n; i++) {
      if (Math.abs(sd[i]) <= EPS_M) augmentedCross.add(perEdge[i]);
    }
  }
  const crossings = [...augmentedCross].sort((x, y) => x - y);
  if (crossings.length !== 2) {
    throw new PatternCadError(
      "unsupported-operation",
      `cutter produces ${crossings.length} boundary crossings; only clean 2-crossing splits are supported`,
      panelId,
    );
  }
  const [c0, c1] = crossings;
  const chainA = augmented.slice(c0, c1 + 1);
  const chainB = [...augmented.slice(c1), ...augmented.slice(0, c0 + 1)];
  for (const [label, chain] of [["first", chainA], ["second", chainB]] as const) {
    if (chain.length < 3 || Math.abs(signedArea(chain)) <= MIN_AREA_M2) {
      throw new PatternCadError("degenerate-panel", `split ${label} piece is degenerate`, panelId);
    }
  }
  let next = doc;
  const a = buildPanelFromRing(next, `${panel.name} A`, panel.materialId, panel.grainAngleRad, chainA);
  next = a.document;
  const b = buildPanelFromRing(next, `${panel.name} B`, panel.materialId, panel.grainAngleRad, chainB);
  next = b.document;
  next = deletePanel(next, panelId);
  const validation = validatePatternDocument(next);
  if (!validation.valid) {
    const first = validation.diagnostics[0];
    throw new PatternCadError(first.code, `split produced invalid geometry: ${first.message}`, first.entityId);
  }
  return { document: next, panelIds: [a.panelId, b.panelId], invalidatedSeamIds: invalidatedSeamIdsForPanels(seams, [panelId]) };
}

/**
 * Join two line-only panels sharing one coincident, oppositely-oriented edge.
 * Produces one fresh panel and deletes both originals. Seams touching either
 * original are reported invalid.
 */
export function joinPanelsAtSharedEdge(
  doc: PatternDocument,
  panelAId: EntityId,
  panelBId: EntityId,
  seams: ReadonlyArray<SeamRef> = [],
): { document: PatternDocument; panelId: EntityId; invalidatedSeamIds: string[] } {
  if (panelAId === panelBId) {
    throw new PatternCadError("unsupported-operation", "cannot join a panel to itself", panelAId);
  }
  const panelA = getPanel(doc, panelAId);
  const panelB = getPanel(doc, panelBId);
  for (const [p, id] of [[panelA, panelAId], [panelB, panelBId]] as const) {
    const outer = p.boundaryLoops.find((l) => l.role === "outer");
    if (!outer) throw new PatternCadError("open-boundary", "panel has no outer loop", id);
    if (p.boundaryLoops.some((l) => l.role === "hole")) {
      throw new PatternCadError("unsupported-operation", "panel join does not support holes", id);
    }
    for (const sid of outer.segmentIds) {
      if (getSegment(doc, sid, id).kind !== "line") {
        throw new PatternCadError("unsupported-operation", "panel join supports line loops only", sid);
      }
    }
  }
  const ringOf = (pid: EntityId): Vec2[] => {
    const p = getPanel(doc, pid);
    const outer = p.boundaryLoops.find((l) => l.role === "outer")!;
    return outer.segmentIds.map((sid) => {
      const s = getSegment(doc, sid, pid);
      const pt = getPoint(doc, s.startPointId, pid);
      return [pt.x, pt.y] as Vec2;
    });
  };
  const ra = ringOf(panelAId), rb = ringOf(panelBId);
  let match: { i: number; j: number } | null = null;
  for (let i = 0; i < ra.length && !match; i++) {
    const a0 = ra[i], a1 = ra[(i + 1) % ra.length];
    for (let j = 0; j < rb.length && !match; j++) {
      const b0 = rb[j], b1 = rb[(j + 1) % rb.length];
      if (dist(a0, b1) <= EPS_M && dist(a1, b0) <= EPS_M) match = { i, j };
    }
  }
  if (!match) {
    throw new PatternCadError("unsupported-operation", "panels share no coincident reversed edge", panelBId);
  }
  const { i, j } = match;
  const chainA: Vec2[] = [];
  for (let k = 0; k < ra.length; k++) chainA.push(ra[(i + 1 + k) % ra.length]);
  const chainB: Vec2[] = [];
  for (let k = 0; k < rb.length; k++) chainB.push(rb[(j + 1 + k) % rb.length]);
  // Both chains run shared-edge-end -> shared-edge-start and share both
  // endpoints (chainA[0]≈chainB[last], chainA[last]≈chainB[0]). Keep chainA
  // whole and only chainB's strict interior so no junction duplicates.
  const ring = [...chainA, ...chainB.slice(1, -1)];
  if (ring.length < 3 || Math.abs(signedArea(ring)) <= MIN_AREA_M2) {
    throw new PatternCadError("degenerate-panel", "joined panel is degenerate", panelAId);
  }
  let next = buildPanelFromRing(doc, `${panelA.name}+${panelB.name}`, panelA.materialId, panelA.grainAngleRad, ring).document;
  const built = next.panels[next.panels.length - 1];
  next = deletePanel(next, panelAId);
  next = deletePanel(next, panelBId);
  const validation = validatePatternDocument(next);
  if (!validation.valid) {
    const first = validation.diagnostics[0];
    throw new PatternCadError(first.code, `join produced invalid geometry: ${first.message}`, first.entityId);
  }
  return {
    document: next,
    panelId: built.id,
    invalidatedSeamIds: invalidatedSeamIdsForPanels(seams, [panelAId, panelBId]),
  };
}
