// G9A — advanced editing operations over the G8A PatternDocument.
//
// Scope relative to src/pattern/cad.ts: cad.ts owns the document model,
// basic creation/moves/transforms/split/reverse/duplicate/delete-panel,
// validation and serialization. This module adds the G9A editing surface
// cad.ts does not provide — move segment, delete vertex, merge, trim,
// extend, offset, mirror, arc splitting, polyline construction — without
// modifying cad.ts.
//
// Conventions (aligned with cad.ts):
//   - Operations are immutable: input documents are never mutated; each op
//     returns a fresh document (or { document, ...result }). CadSession
//     records the transition for undo/redo.
//   - Geometry is in PANEL-LOCAL coordinates (the same frame cad.ts stores,
//     measures, and validates in). UI callers convert workspace positions
//     with globalPointToPanelLocal first.
//   - Preconditions throw PatternCadError with cad.ts's diagnostic codes.
//     Nothing here repairs geometry silently, and nothing here validates the
//     resulting document (validatePatternDocument remains the gate).
//   - Shared-vertex semantics: moving/trimming an endpoint point moves it
//     for every segment that references it; that is the defined behavior of
//     a shared-vertex CAD model, and validation reports any arc-radius
//     inconsistency such moves can create on neighbor arcs.
//
// Determinism: id allocation replicates cad.ts's allocateId algorithm
// exactly (document.id + kind + zero-padded nextEntityIndex with occupied
// checks), so operation sequences always produce identical ids.

import {
  PatternCadError,
  createConstructionLine,
  createPoint,
  duplicatePanel,
  globalToLocal,
  splitBoundarySegment,
  type BoundaryLoop,
  type CircularArcSegment,
  type EntityId,
  type LineSegment,
  type PatternDocument,
  type PatternSegment,
} from "../pattern/cad.js";
import {
  angleOnArc,
  collinear,
  dist,
  dot,
  leftNormal,
  lineIntersection,
  lerp,
  normalizeAngle,
  paramAlong,
  pointOnArc,
  reflectAcrossLine,
  sub,
  type Vec2,
} from "./geom.js";
import {
  getLoop,
  getPanel,
  getPoint,
  getSegment,
  intersectSegments,
  pointIsReferenced,
  pointReferences,
  resolveSegment,
  type ResolvedSegment,
} from "./queries.js";

const EPS_M = 1e-9;
const TAU = 2 * Math.PI;

function finite(value: number): boolean {
  return Number.isFinite(value);
}

function cloneDoc(doc: PatternDocument): PatternDocument {
  return JSON.parse(JSON.stringify(doc)) as PatternDocument;
}

/**
 * Allocate an entity id exactly the way cad.ts's private allocateId does:
 * `${document.id}/${kind}/${nextEntityIndex padded to 8}`, skipping ids
 * already occupied anywhere in the document.
 */
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
// Creation: construction polyline
// ---------------------------------------------------------------------------

/**
 * Create an open construction polyline: one point per position plus one
 * construction line between consecutive points. Positions are panel-local.
 */
export function createConstructionPolyline(
  doc: PatternDocument,
  panelId: EntityId,
  positions: readonly Vec2[],
): { document: PatternDocument; pointIds: EntityId[]; segmentIds: EntityId[] } {
  if (positions.length < 2) {
    throw new PatternCadError("unsupported-operation", `polyline needs at least 2 positions, got ${positions.length}`, panelId);
  }
  for (const p of positions) {
    if (!finite(p[0]) || !finite(p[1])) {
      throw new PatternCadError("invalid-transform", `polyline position (${p[0]}, ${p[1]}) is not finite`, panelId);
    }
  }
  let next = doc;
  const pointIds: EntityId[] = [];
  for (const pos of positions) {
    const created = createPoint(next, panelId, pos, "construction");
    next = created.document;
    pointIds.push(created.pointId);
  }
  const segmentIds: EntityId[] = [];
  for (let i = 0; i + 1 < pointIds.length; i++) {
    const created = createConstructionLine(next, panelId, pointIds[i], pointIds[i + 1]);
    next = created.document;
    segmentIds.push(created.segmentId);
  }
  return { document: next, pointIds, segmentIds };
}

// ---------------------------------------------------------------------------
// Move segment
// ---------------------------------------------------------------------------

/**
 * Translate one segment by a panel-local delta: both endpoints move, and an
 * arc's center moves with them (the bulge/sweep are untouched, so the arc
 * keeps its shape). Endpoints shared with neighboring segments move with
 * them — shared-vertex semantics, same as cad.ts movePoint.
 */
export function moveSegmentBy(
  doc: PatternDocument,
  panelId: EntityId,
  segmentId: EntityId,
  delta: Vec2,
): PatternDocument {
  if (!finite(delta[0]) || !finite(delta[1])) {
    throw new PatternCadError("invalid-transform", `segment delta must be finite, got (${delta[0]}, ${delta[1]})`, segmentId);
  }
  const segment = getSegment(doc, segmentId, panelId);
  const next = cloneDoc(doc);
  const shift = (pointId: EntityId): void => {
    const p = getPoint(next, pointId, panelId);
    p.x += delta[0];
    p.y += delta[1];
  };
  shift(segment.startPointId);
  shift(segment.endPointId);
  if (segment.kind === "arc") shift(segment.centerPointId);
  return next;
}

// ---------------------------------------------------------------------------
// Point deletion
// ---------------------------------------------------------------------------

/**
 * Safety gate before removing a point entity: the only segment references it
 * may have are the ones this operation is about to remove, and dimensions /
 * constraints must not reference it (their removal would be a silent
 * invalidation — G9A reports instead).
 */
function assertPointRemovable(doc: PatternDocument, pointId: EntityId, removableSegmentIds: readonly EntityId[]): void {
  const refs = pointReferences(doc, pointId);
  const blocking = refs.segments.filter((id) => !removableSegmentIds.includes(id));
  if (blocking.length > 0) {
    throw new PatternCadError(
      "unsupported-operation",
      `point '${pointId}' is also referenced by segment(s) ${blocking.join(", ")}; deleting it would invalidate them`,
      pointId,
    );
  }
  if (refs.dimensions.length > 0 || refs.constraints.length > 0) {
    const ids = [...refs.dimensions, ...refs.constraints];
    throw new PatternCadError(
      "unsupported-operation",
      `point '${pointId}' is referenced by dimension/constraint ${ids.join(", ")}; delete those first`,
      pointId,
    );
  }
}

/**
 * Delete a point.
 *
 * - Construction/orphan point: its incident segments are removed with it,
 *   and construction endpoints orphaned by the removal are cleaned up too.
 *   Points that are arc centers, or referenced by dimensions/constraints,
 *   are rejected with an explicit report.
 * - Boundary vertex of degree 2: the two incident segments merge into one
 *   that keeps the first segment's id and loop slot. line+line merges
 *   directly (deleting a vertex is supposed to remove that corner);
 *   arc+arc merges when both arcs share a circle and sweep direction.
 *   line+arc mixes, degree != 2, and points referenced by other entities are
 *   rejected with PatternCadError — never silently rewired.
 */
export function deletePoint(doc: PatternDocument, panelId: EntityId, pointId: EntityId): PatternDocument {
  const panel = getPanel(doc, panelId);
  getPoint(doc, pointId, panelId); // precondition: exists in this panel

  // Locate the point inside boundary loops.
  const loopHits: Array<{ loop: BoundaryLoop; segs: PatternSegment[] }> = [];
  for (const loop of panel.boundaryLoops) {
    const segs = loop.segmentIds
      .map((id) => getSegment(doc, id, panelId))
      .filter((s) => s.startPointId === pointId || s.endPointId === pointId);
    if (segs.length > 0) loopHits.push({ loop, segs });
  }

  if (loopHits.length === 0) {
    // ---- construction / orphan point ----
    const refs = pointReferences(doc, pointId);
    const foreign = refs.segments.filter((id) => getSegment(doc, id).panelId !== panelId);
    if (foreign.length > 0) {
      throw new PatternCadError(
        "unsupported-operation",
        `point '${pointId}' is referenced by foreign-panel segment(s) ${foreign.join(", ")}`,
        pointId,
      );
    }
    const endpointSegs: PatternSegment[] = [];
    const centerSegs: PatternSegment[] = [];
    for (const id of refs.segments) {
      const s = getSegment(doc, id);
      if (s.startPointId === pointId || s.endPointId === pointId) endpointSegs.push(s);
      else if (s.kind === "arc" && s.centerPointId === pointId) centerSegs.push(s);
    }
    if (centerSegs.length > 0) {
      throw new PatternCadError(
        "unsupported-operation",
        `point '${pointId}' is the center of arc(s) ${centerSegs.map((s) => s.id).join(", ")}; deleting it would invalidate them`,
        pointId,
      );
    }
    assertPointRemovable(doc, pointId, endpointSegs.map((s) => s.id));

    const next = cloneDoc(doc);
    const removedFars: EntityId[] = [];
    for (const seg of endpointSegs) {
      const other = seg.startPointId === pointId ? seg.endPointId : seg.startPointId;
      removedFars.push(other);
      next.segments = next.segments.filter((s) => s.id !== seg.id);
      const owner = getPanel(next, seg.panelId);
      owner.constructionSegmentIds = owner.constructionSegmentIds.filter((id) => id !== seg.id);
    }
    next.points = next.points.filter((p) => p.id !== pointId);
    // Cascade: construction endpoints orphaned by the removal.
    for (const far of new Set(removedFars)) {
      const farPoint = next.points.find((p) => p.id === far);
      if (!farPoint || farPoint.role !== "construction") continue;
      if (pointIsReferenced(next, far)) continue;
      next.points = next.points.filter((p) => p.id !== far);
    }
    return next;
  }

  // ---- boundary vertex ----
  if (loopHits.length > 1) {
    throw new PatternCadError(
      "unsupported-operation",
      `point '${pointId}' lies in ${loopHits.length} boundary loops; deletion is ambiguous`,
      pointId,
    );
  }
  const { loop, segs } = loopHits[0];
  if (segs.length !== 2) {
    throw new PatternCadError(
      "unsupported-operation",
      `point '${pointId}' has loop degree ${segs.length} (deletion needs exactly 2)`,
      pointId,
    );
  }
  if (loop.segmentIds.length <= 2) {
    throw new PatternCadError(
      "unsupported-operation",
      `loop '${loop.id}' has only ${loop.segmentIds.length} segments; deleting '${pointId}' would collapse it`,
      pointId,
    );
  }
  if (segs[0].kind === "line" && segs[1].kind === "line" && loop.segmentIds.length <= 3) {
    throw new PatternCadError(
      "unsupported-operation",
      `deleting '${pointId}' would leave loop '${loop.id}' with ${loop.segmentIds.length - 1} line segments (< 3); the loop would collapse`,
      pointId,
    );
  }
  const idxs = [loop.segmentIds.indexOf(segs[0].id), loop.segmentIds.indexOf(segs[1].id)];
  const n = loop.segmentIds.length;
  const adjacent = idxs[1] === (idxs[0] + 1) % n || idxs[0] === (idxs[1] + 1) % n;
  if (!adjacent) {
    throw new PatternCadError(
      "unsupported-operation",
      `segments ${segs[0].id}, ${segs[1].id} are not adjacent in loop '${loop.id}'`,
      pointId,
    );
  }
  assertPointRemovable(doc, pointId, [segs[0].id, segs[1].id]);

  const chain = chainEndpoints(segs[0], segs[1], pointId);
  const merged = buildMergedSegment(doc, segs[0], segs[1], chain.startId, chain.endId);

  const next = cloneDoc(doc);
  const seg0 = getSegment(next, segs[0].id, panelId);
  const seg1 = getSegment(next, segs[1].id, panelId);
  // Replace seg0's record in place (keeps store order = loop slot).
  const storeIdx = next.segments.findIndex((s) => s.id === seg0.id);
  next.segments[storeIdx] = merged;
  next.segments = next.segments.filter((s) => s.id !== seg1.id);
  next.points = next.points.filter((p) => p.id !== pointId);
  const nextPanel = getPanel(next, panelId);
  const nextLoop = nextPanel.boundaryLoops.find((l) => l.id === loop.id)!;
  nextLoop.segmentIds = nextLoop.segmentIds.filter((id) => id !== seg1.id);

  // Clean up seg1's arc center when it was a distinct, now-unreferenced point.
  if (seg1.kind === "arc") {
    const centerId = seg1.centerPointId;
    const stillUsed = next.segments.some((s) => s.kind === "arc" && s.centerPointId === centerId);
    if (!stillUsed && !pointIsReferenced(next, centerId)) {
      next.points = next.points.filter((p) => p.id !== centerId);
    }
  }
  return next;
}

/**
 * Chain orientation for two segments meeting at point `s`: which endpoint of
 * each segment is the "far" one, expressed as the merged segment's
 * (startId, endId). Only the two consistent layouts are accepted; anything
 * else means the chain is broken through `s` and is reported, not repaired.
 */
function chainEndpoints(
  segA: PatternSegment, segB: PatternSegment, s: EntityId,
): { startId: EntityId; endId: EntityId } {
  if (segA.endPointId === s && segB.startPointId === s) {
    return { startId: segA.startPointId, endId: segB.endPointId };
  }
  if (segA.startPointId === s && segB.endPointId === s) {
    return { startId: segB.startPointId, endId: segA.endPointId };
  }
  throw new PatternCadError(
    "unsupported-operation",
    `segments ${segA.id} and ${segB.id} do not form a consistent chain through point '${s}'`,
    s,
  );
}

/** Build the merged segment geometry (line+line direct; arc+arc same circle). */
function buildMergedSegment(
  doc: PatternDocument,
  segA: PatternSegment,
  segB: PatternSegment,
  startId: EntityId,
  endId: EntityId,
): PatternSegment {
  if (segA.kind === "line" && segB.kind === "line") {
    return { ...segA, startPointId: startId, endPointId: endId } satisfies LineSegment;
  }
  if (segA.kind === "arc" && segB.kind === "arc") {
    const cA = getPoint(doc, segA.centerPointId, segA.panelId);
    const cB = getPoint(doc, segB.centerPointId, segB.panelId);
    const centerDist = dist([cA.x, cA.y], [cB.x, cB.y]);
    const rA = radiusOf(doc, segA);
    if (centerDist > Math.max(EPS_M, rA * 1e-9)) {
      throw new PatternCadError(
        "unsupported-operation",
        `arcs ${segA.id} and ${segB.id} have different centers (${centerDist} apart); cannot merge`,
        segA.id,
      );
    }
    if (Math.sign(segA.sweepRad) !== Math.sign(segB.sweepRad)) {
      throw new PatternCadError(
        "unsupported-operation",
        `arcs ${segA.id} and ${segB.id} sweep in opposite directions; merging would retrace the boundary`,
        segA.id,
      );
    }
    const sum = segA.sweepRad + segB.sweepRad;
    if (!(Math.abs(sum) > 1e-12 && Math.abs(sum) < TAU)) {
      throw new PatternCadError(
        "unsupported-operation",
        `merged arc sweep ${sum} is degenerate (must be nonzero and < 2pi)`,
        segA.id,
      );
    }
    // Keep segA's id/center; the merged record takes segA's loop slot.
    return {
      ...segA,
      startPointId: startId,
      endPointId: endId,
      sweepRad: sum,
    } satisfies CircularArcSegment;
  }
  throw new PatternCadError(
    "unsupported-operation",
    `cannot merge line segment with arc segment (${segA.id}, ${segB.id}); mixed primitives are not a single editable edge`,
    segA.id,
  );
}

function radiusOf(doc: PatternDocument, seg: PatternSegment & { kind: "arc" }): number {
  const s = getPoint(doc, seg.startPointId, seg.panelId);
  const c = getPoint(doc, seg.centerPointId, seg.panelId);
  return dist([s.x, s.y], [c.x, c.y]);
}

// ---------------------------------------------------------------------------
// Merge segments
// ---------------------------------------------------------------------------

/**
 * Merge two adjacent segments of one loop (or two construction segments)
 * into a single primitive, when a single primitive represents the result
 * WITHOUT changing geometry:
 *   - two collinear lines -> one line;
 *   - two arcs on one circle with the same sweep direction -> one arc;
 *   - anything else -> PatternCadError("unsupported-operation").
 * The merged segment keeps `segAId`'s id and loop slot; `segBId` and the
 * shared point are removed (points referenced by dimensions/constraints or
 * other entities are rejected up front, never silently invalidated).
 */
export function mergeSegments(
  doc: PatternDocument,
  panelId: EntityId,
  loopId: EntityId | null,
  segAId: EntityId,
  segBId: EntityId,
): PatternDocument {
  if (segAId === segBId) {
    throw new PatternCadError("unsupported-operation", "cannot merge a segment with itself", segAId);
  }
  const segA = getSegment(doc, segAId, panelId);
  const segB = getSegment(doc, segBId, panelId);
  if (segA.role !== segB.role) {
    throw new PatternCadError(
      "unsupported-operation",
      `cannot merge ${segA.role} segment ${segA.id} with ${segB.role} segment ${segB.id}`,
      segAId,
    );
  }

  const shared = [segA.startPointId, segA.endPointId].filter((x) => x === segB.startPointId || x === segB.endPointId);
  if (shared.length !== 1) {
    throw new PatternCadError("unsupported-operation", "segments must share exactly one endpoint", segAId);
  }
  const s = shared[0];
  const farA = segA.startPointId === s ? segA.endPointId : segA.startPointId;
  const farB = segB.startPointId === s ? segB.endPointId : segB.startPointId;
  if (farA === farB) {
    throw new PatternCadError("unsupported-operation", "segments share both endpoints", segAId);
  }

  // Boundary: locate + adjacency + collapse guard.
  let loop: BoundaryLoop | null = null;
  if (loopId !== null) {
    loop = getLoop(doc, panelId, loopId);
    const iA = loop.segmentIds.indexOf(segA.id);
    const iB = loop.segmentIds.indexOf(segB.id);
    if (iA < 0 || iB < 0) {
      throw new PatternCadError("missing-reference", `segments are not both in loop '${loopId}'`, loopId);
    }
    const n = loop.segmentIds.length;
    if (!(iB === (iA + 1) % n || iA === (iB + 1) % n)) {
      throw new PatternCadError("unsupported-operation", `segments ${segA.id}, ${segB.id} are not adjacent in loop '${loopId}'`, segAId);
    }
    if (n <= 2) {
      throw new PatternCadError("unsupported-operation", `loop '${loopId}' has only ${n} segments; merging would collapse it`, loopId);
    }
    if (segA.kind === "line" && segB.kind === "line" && n <= 3) {
      throw new PatternCadError(
        "unsupported-operation",
        `merging lines would leave loop '${loopId}' with ${n - 1} line segments (< 3); the loop would collapse`,
        loopId,
      );
    }
  }

  // Shared-point degree: no other segment endpoint may use it.
  const incident = doc.segments.filter(
    (sg) => sg.startPointId === s || sg.endPointId === s,
  );
  if (incident.length !== 2) {
    throw new PatternCadError(
      "unsupported-operation",
      `shared point '${s}' has segment degree ${incident.length} (merge needs exactly 2)`,
      s,
    );
  }
  assertPointRemovable(doc, s, [segA.id, segB.id]);

  const chain = chainEndpoints(segA, segB, s);

  // Shape preservation checks (merge must not move geometry).
  if (segA.kind === "line" && segB.kind === "line") {
    const p0 = getPoint(doc, chain.startId, panelId);
    const p1 = getPoint(doc, s, panelId);
    const p2 = getPoint(doc, chain.endId, panelId);
    if (!collinear([p0.x, p0.y], [p1.x, p1.y], [p2.x, p2.y], 1e-9)) {
      throw new PatternCadError(
        "unsupported-operation",
        `lines through '${s}' are not collinear; merging would change the shape (use deletePoint to remove the vertex)`,
        s,
      );
    }
  }

  const merged = buildMergedSegment(doc, segA, segB, chain.startId, chain.endId);

  const next = cloneDoc(doc);
  const storeIdx = next.segments.findIndex((sg) => sg.id === segA.id);
  next.segments[storeIdx] = merged;
  next.segments = next.segments.filter((sg) => sg.id !== segB.id);
  next.points = next.points.filter((p) => p.id !== s);

  if (loop) {
    const nextPanel = getPanel(next, panelId);
    const nextLoop = nextPanel.boundaryLoops.find((l) => l.id === loop!.id)!;
    nextLoop.segmentIds = nextLoop.segmentIds.filter((id) => id !== segB.id);
  } else {
    const owner = getPanel(next, panelId);
    owner.constructionSegmentIds = owner.constructionSegmentIds.filter((id) => id !== segB.id);
  }

  // Drop segB's arc center when it was a distinct, now-unreferenced point.
  if (segB.kind === "arc" && merged.kind === "arc") {
    const centerId = segB.centerPointId;
    if (centerId !== merged.centerPointId && !next.segments.some((sg) => sg.kind === "arc" && sg.centerPointId === centerId)) {
      if (!pointIsReferenced(next, centerId)) {
        next.points = next.points.filter((p) => p.id !== centerId);
      }
    }
  }
  return next;
}

// ---------------------------------------------------------------------------
// Trim / extend
// ---------------------------------------------------------------------------

/**
 * Trim a segment back to its nearest intersection with `cutterId`, moving
 * the endpoint that is nearest to the chosen intersection (the `reference`
 * position selects which intersection when several exist).
 *
 * Works on boundary and construction segments alike — the endpoint point
 * id stays the same, so loops keep their chain. Line and arc targets are
 * supported; the moved arc endpoint stays on its circle because the
 * intersection lies on the arc itself (only the sweep shrinks).
 * No intersection => PatternCadError (the cad.ts code union has no
 * dedicated "no-intersection" code, so "unsupported-operation" carries the
 * detail in its message).
 */
export function trimSegment(
  doc: PatternDocument,
  panelId: EntityId,
  targetId: EntityId,
  cutterId: EntityId,
  reference: Vec2,
): PatternDocument {
  if (targetId === cutterId) {
    throw new PatternCadError("unsupported-operation", "cannot trim a segment by itself", targetId);
  }
  getSegment(doc, targetId, panelId); // precondition: exists in this panel
  getSegment(doc, cutterId, panelId);
  if (!finite(reference[0]) || !finite(reference[1])) {
    throw new PatternCadError("invalid-transform", "trim reference must be a finite position", targetId);
  }
  const hits = intersectSegments(doc, targetId, cutterId, "local");
  if (hits.length === 0) {
    throw new PatternCadError(
      "unsupported-operation",
      `segments ${targetId} and ${cutterId} do not intersect; nothing to trim`,
      targetId,
    );
  }
  let best = hits[0];
  for (const h of hits) {
    if (dist(h, reference) < dist(best, reference)) best = h;
  }
  const r = resolveSegment(doc, targetId);
  const dStart = dist(best, r.start);
  const dEnd = dist(best, r.end);
  const which: "start" | "end" = dStart <= dEnd ? "start" : "end";
  const keepPos = which === "start" ? r.end : r.start;
  if (dist(best, keepPos) <= EPS_M) {
    throw new PatternCadError("zero-length-edge", `trim would collapse ${targetId} onto its kept endpoint`, targetId);
  }
  return moveEndpointTo(doc, panelId, targetId, which, best);
}

/**
 * Extend a line segment by `distance` beyond the endpoint nearest
 * `reference` (arcs are rejected: extending one would change its radius).
 */
export function extendLineBy(
  doc: PatternDocument,
  panelId: EntityId,
  segmentId: EntityId,
  distance: number,
  reference: Vec2,
): PatternDocument {
  const segment = getSegment(doc, segmentId, panelId);
  if (segment.kind !== "line") {
    throw new PatternCadError("unsupported-operation", "extendLineBy supports line segments only", segmentId);
  }
  if (!(distance > 0) || !finite(distance)) {
    throw new PatternCadError("invalid-transform", `extension distance must be > 0, got ${distance}`, segmentId);
  }
  const r = resolveSegment(doc, segmentId);
  const atEnd = dist(reference, r.end) <= dist(reference, r.start);
  const end = atEnd ? r.end : r.start;
  const other = atEnd ? r.start : r.end;
  const dir = [end[0] - other[0], end[1] - other[1]] as Vec2;
  const l = Math.hypot(dir[0], dir[1]);
  if (l <= EPS_M) throw new PatternCadError("zero-length-edge", `segment ${segmentId} has zero length`, segmentId);
  const next: Vec2 = [end[0] + (dir[0] / l) * distance, end[1] + (dir[1] / l) * distance];
  return moveEndpointTo(doc, panelId, segmentId, atEnd ? "end" : "start", next);
}

/**
 * Extend a line segment until it meets `cutterId`, on whichever side lies
 * beyond an endpoint. Intersections on the segment itself are rejected —
 * extend never trims. The cutter may be a line or an arc.
 */
export function extendLineTo(
  doc: PatternDocument,
  panelId: EntityId,
  targetId: EntityId,
  cutterId: EntityId,
  reference: Vec2,
): PatternDocument {
  if (targetId === cutterId) {
    throw new PatternCadError("unsupported-operation", "cannot extend a segment to itself", targetId);
  }
  const target = getSegment(doc, targetId, panelId);
  if (target.kind !== "line") {
    throw new PatternCadError("unsupported-operation", "extendLineTo supports line targets only", targetId);
  }
  getSegment(doc, cutterId, panelId);
  // The target line is INFINITE here (that is what extending means); the
  // cutter stays closed. Intersections on the target segment itself are
  // filtered out below so extend can never trim.
  const hits = infiniteLineHits(
    resolveSegment(doc, targetId),
    resolveSegment(doc, cutterId),
  );
  const r = resolveSegment(doc, targetId);
  const beyond: Array<{ pos: Vec2; side: "start" | "end" }> = [];
  for (const h of hits) {
    const dx = r.end[0] - r.start[0];
    const dy = r.end[1] - r.start[1];
    const d2 = dx * dx + dy * dy;
    if (d2 === 0) throw new PatternCadError("zero-length-edge", `segment ${targetId} has zero length`, targetId);
    const t = ((h[0] - r.start[0]) * dx + (h[1] - r.start[1]) * dy) / d2;
    if (t > 1 + 1e-9) beyond.push({ pos: h, side: "end" });
    else if (t < -1e-9) beyond.push({ pos: h, side: "start" });
  }
  if (beyond.length === 0) {
    throw new PatternCadError(
      "unsupported-operation",
      `segment ${targetId} has no intersection with ${cutterId} beyond its endpoints`,
      targetId,
    );
  }
  let best = beyond[0];
  for (const c of beyond) {
    if (dist(c.pos, reference) < dist(best.pos, reference)) best = c;
  }
  return moveEndpointTo(doc, panelId, targetId, best.side, best.pos);
}

/**
 * Move one endpoint of a segment to a local position (shared-vertex: the
 * point moves for every referencing segment). Arc endpoints update their
 * sweep so the arc still starts/ends at the moved point along the same
 * direction; the position must remain on the arc's circle (as guaranteed by
 * trim/extend intersections), otherwise "invalid-arc" is thrown.
 */
function moveEndpointTo(
  doc: PatternDocument,
  panelId: EntityId,
  segmentId: EntityId,
  which: "start" | "end",
  pos: Vec2,
): PatternDocument {
  const segment = getSegment(doc, segmentId, panelId);
  const pointId = which === "start" ? segment.startPointId : segment.endPointId;
  const otherId = which === "start" ? segment.endPointId : segment.startPointId;
  const other = getPoint(doc, otherId, panelId);
  if (dist(pos, [other.x, other.y]) <= EPS_M) {
    throw new PatternCadError("zero-length-edge", `endpoint target coincides with the other endpoint of ${segmentId}`, segmentId);
  }

  // Arc sweep bookkeeping needs the ORIGINAL endpoint angles, captured
  // before the point moves.
  let oldSweep = 0;
  let fixedAng = 0; // end angle (when moving start) — fixed by construction
  let oldEndAng = 0; // end angle (when moving end) — the OLD one
  let centerPos: Vec2 = [0, 0];
  if (segment.kind === "arc") {
    const c = getPoint(doc, segment.centerPointId, panelId);
    const s0 = getPoint(doc, segment.startPointId, panelId);
    const e0 = getPoint(doc, segment.endPointId, panelId);
    centerPos = [c.x, c.y];
    oldSweep = segment.sweepRad;
    fixedAng = Math.atan2(e0.y - c.y, e0.x - c.x);
    oldEndAng = fixedAng;
    const r0 = dist([s0.x, s0.y], centerPos);
    const rNew = dist(pos, centerPos);
    if (Math.abs(rNew - r0) > Math.max(EPS_M, r0 * 1e-6)) {
      throw new PatternCadError(
        "invalid-arc",
        `endpoint of arc ${segmentId} moved off its circle (radius ${r0} -> ${rNew})`,
        segmentId,
      );
    }
  }

  const next = cloneDoc(doc);
  const seg = getSegment(next, segmentId, panelId);
  const p = getPoint(next, pointId, panelId);
  p.x = pos[0];
  p.y = pos[1];

  if (seg.kind === "arc") {
    const newAng = Math.atan2(p.y - centerPos[1], p.x - centerPos[0]);
    let newSweep: number;
    if (which === "start") {
      // End angle is fixed: sweep must run from the new start to the same
      // end angle, keeping the old direction.
      newSweep = fixedAng - newAng;
      while (Math.abs(newSweep) >= TAU) newSweep -= Math.sign(newSweep) * TAU;
      if (newSweep * oldSweep < 0) newSweep += Math.sign(oldSweep) * TAU;
    } else {
      // Start angle is fixed; the end moved: adjust by the end's angular
      // change, keeping direction and the (< 2pi) bound.
      newSweep = oldSweep + normalizeAngle(newAng - oldEndAng);
    }
    if (!(Math.abs(newSweep) > 1e-12 && Math.abs(newSweep) < TAU) || Math.sign(newSweep) !== Math.sign(oldSweep)) {
      throw new PatternCadError(
        "unsupported-operation",
        `trim/extend would degenerate arc ${segmentId} (sweep ${oldSweep} -> ${newSweep})`,
        segmentId,
      );
    }
    seg.sweepRad = newSweep;
  }
  return next;
}

/**
 * Intersections of the target's supporting LINE (infinite in both
 * directions) with a cutter treated as CLOSED (segment or arc).
 * Used by extendLineTo: hits may lie beyond the target's endpoints, but
 * must lie on the cutter. Deterministic order: by parameter along the
 * target line, then coordinates.
 */
function infiniteLineHits(target: ResolvedSegment, cutter: ResolvedSegment): Vec2[] {
  const angEps = 1e-12;
  if (cutter.segment.kind === "line") {
    const hit = lineIntersection(target.start, target.end, cutter.start, cutter.end, 1e-12);
    if (!hit) return [];
    const t = paramAlong(hit, cutter.start, cutter.end);
    return t >= -1e-9 && t <= 1 + 1e-9 ? [hit] : [];
  }
  // Cutter is an arc: solve line vs circle with UNCLAMPED target params.
  const arc = cutter.arc!;
  const d = sub(target.end, target.start);
  const f = sub(target.start, arc.center);
  const A = dot(d, d);
  if (A === 0) return [];
  const B = 2 * dot(f, d);
  const C = dot(f, f) - arc.radius * arc.radius;
  const disc = B * B - 4 * A * C;
  if (disc < 0) return [];
  const sq = Math.sqrt(disc);
  const roots = sq === 0 ? [-B / (2 * A)] : [(-B - sq) / (2 * A), (-B + sq) / (2 * A)];
  const out: Vec2[] = [];
  for (const t of roots) {
    const pos = lerp(target.start, target.end, t);
    const ang = Math.atan2(pos[1] - arc.center[1], pos[0] - arc.center[0]);
    if (!angleOnArc(arc, ang, angEps)) continue;
    if (!out.some((o) => dist(o, pos) <= 1e-9)) out.push(pos);
  }
  out.sort((p, q) => {
    const tp = paramAlong(p, target.start, target.end);
    const tq = paramAlong(q, target.start, target.end);
    return tp - tq || p[0] - q[0] || p[1] - q[1];
  });
  return out;
}

// ---------------------------------------------------------------------------
// Offset
// ---------------------------------------------------------------------------

/**
 * Create a construction line parallel to a line segment at `distance`
 * (positive = left of start->end) with two fresh construction points.
 * Returns the created segment id through the result object.
 * Arc offsetting belongs to G9B's curve-offset work.
 */
export function offsetConstructionLine(
  doc: PatternDocument,
  panelId: EntityId,
  segmentId: EntityId,
  distance: number,
): { document: PatternDocument; segmentId: EntityId; pointIds: [EntityId, EntityId] } {
  const segment = getSegment(doc, segmentId, panelId);
  if (segment.kind !== "line") {
    throw new PatternCadError("unsupported-operation", "offset supports line segments only (curve offset arrives with G9B)", segmentId);
  }
  if (!finite(distance) || distance === 0) {
    throw new PatternCadError("invalid-transform", `offset distance must be finite and nonzero, got ${distance}`, segmentId);
  }
  const r = resolveSegment(doc, segmentId);
  const n = leftNormal(r.start, r.end);
  const len = Math.hypot(n[0], n[1]);
  if (len <= EPS_M) throw new PatternCadError("zero-length-edge", `segment ${segmentId} has zero length`, segmentId);
  const a: Vec2 = [r.start[0] + (n[0] / len) * distance, r.start[1] + (n[1] / len) * distance];
  const b: Vec2 = [r.end[0] + (n[0] / len) * distance, r.end[1] + (n[1] / len) * distance];

  let next = doc;
  const pA = createPoint(next, panelId, a, "construction");
  next = pA.document;
  const pB = createPoint(next, panelId, b, "construction");
  next = pB.document;
  const line = createConstructionLine(next, panelId, pA.pointId, pB.pointId);
  return { document: line.document, segmentId: line.segmentId, pointIds: [pA.pointId, pB.pointId] };
}

// ---------------------------------------------------------------------------
// Mirror
// ---------------------------------------------------------------------------

/**
 * Mirror a panel in place across a workspace-space line.
 *
 * Reflection is baked into the panel's local point coordinates (cad.ts's
 * Transform2D documents that reflection is not a transform operation), the
 * transform itself is untouched, and all ids are preserved:
 *   - every local point is reflected across the mirror line mapped into
 *     panel-local space (affine conjugation of a reflection is a reflection);
 *   - arc sweeps negate (reflection reverses orientation);
 *   - every loop's `orientation` field flips so it stays consistent with
 *     the reflected geometry (validatePatternDocument checks this).
 * Dimensions and constraints keep their references — measures update to the
 * mirrored positions automatically.
 */
export function mirrorPanel(
  doc: PatternDocument,
  panelId: EntityId,
  mirrorLineA: Vec2,
  mirrorLineB: Vec2,
): PatternDocument {
  const panel = getPanel(doc, panelId);
  if (!finite(mirrorLineA[0]) || !finite(mirrorLineA[1]) || !finite(mirrorLineB[0]) || !finite(mirrorLineB[1])) {
    throw new PatternCadError("invalid-transform", "mirror line must be finite", panelId);
  }
  if (dist(mirrorLineA, mirrorLineB) <= EPS_M) {
    throw new PatternCadError("invalid-transform", "mirror line is degenerate", panelId);
  }
  const localA = globalToLocal(panel, mirrorLineA);
  const localB = globalToLocal(panel, mirrorLineB);

  const next = cloneDoc(doc);
  const nextPanel = getPanel(next, panelId);
  for (const point of next.points) {
    if (point.panelId !== panelId) continue;
    const r = reflectAcrossLine([point.x, point.y], localA, localB);
    point.x = r[0];
    point.y = r[1];
  }
  for (const segment of next.segments) {
    if (segment.panelId !== panelId) continue;
    if (segment.kind === "arc") segment.sweepRad = -segment.sweepRad;
  }
  for (const loop of nextPanel.boundaryLoops) {
    loop.orientation = loop.orientation === "ccw" ? "cw" : "ccw";
  }
  return next;
}

/**
 * Mirror a panel into a NEW panel (the drafting-standard left/right pair):
 * duplicate with no offset, mirror the copy across the same workspace line,
 * and name it "<source> mirror". Returns the new panel id.
 */
export function mirrorPanelCopy(
  doc: PatternDocument,
  panelId: EntityId,
  mirrorLineA: Vec2,
  mirrorLineB: Vec2,
): { document: PatternDocument; panelId: EntityId } {
  const source = getPanel(doc, panelId);
  const dup = duplicatePanel(doc, panelId, [0, 0]);
  const mirrored = mirrorPanel(dup.document, dup.panelId, mirrorLineA, mirrorLineB);
  getPanel(mirrored, dup.panelId).name = `${source.name} mirror`;
  return { document: mirrored, panelId: dup.panelId };
}

// ---------------------------------------------------------------------------
// Split (lines delegate to cad.ts; arcs are new)
// ---------------------------------------------------------------------------

/**
 * Split a boundary segment at parameter t in (0,1).
 * Line segments delegate to cad.ts splitBoundarySegment; arc segments are
 * split into two arcs sharing the same center point, with sweeps t*s and
 * (1-t)*s. Returns the inserted point and the two resulting segment ids.
 */
export function splitSegment(
  doc: PatternDocument,
  panelId: EntityId,
  loopId: EntityId,
  segmentId: EntityId,
  t: number,
): { document: PatternDocument; pointId: EntityId; segmentIds: [EntityId, EntityId] } {
  if (!finite(t) || !(t > 0 && t < 1)) {
    throw new PatternCadError("unsupported-operation", `split parameter must be strictly inside (0,1), got ${t}`, segmentId);
  }
  const segment = getSegment(doc, segmentId, panelId);
  if (segment.kind === "line") {
    return splitBoundarySegment(doc, panelId, loopId, segmentId, t);
  }
  const loop = getLoop(doc, panelId, loopId);
  if (!loop.segmentIds.includes(segmentId)) {
    throw new PatternCadError("missing-reference", `segment '${segmentId}' is not in loop '${loopId}'`, segmentId);
  }

  const r = resolveSegment(doc, segmentId, panelId);
  const arc = r.arc!;
  const mid = pointOnArc(arc, t);
  if (Math.abs(arc.sweep * t) <= 1e-9 || Math.abs(arc.sweep * (1 - t)) <= 1e-9) {
    throw new PatternCadError(
      "unsupported-operation",
      `split at t=${t} would leave a sub-arc with sweep <= validation epsilon`,
      segmentId,
    );
  }

  let next = doc;
  const created = createPoint(next, panelId, mid, "boundary");
  next = created.document; // fresh clone from cad.ts — safe to mutate below
  const firstId = allocateEntityId(next, "segment");
  const secondId = allocateEntityId(next, "segment");
  const old = getSegment(next, segmentId, panelId) as PatternSegment & { kind: "arc" };
  const first: PatternSegment = {
    ...old,
    id: firstId,
    startPointId: old.startPointId,
    endPointId: created.pointId,
    sweepRad: old.sweepRad * t,
  };
  const second: PatternSegment = {
    ...old,
    id: secondId,
    startPointId: created.pointId,
    endPointId: old.endPointId,
    sweepRad: old.sweepRad * (1 - t),
  };
  next.segments = next.segments.filter((s) => s.id !== segmentId);
  next.segments.push(first, second);
  const panel = getPanel(next, panelId);
  const loop2 = panel.boundaryLoops.find((l) => l.id === loopId)!;
  const index = loop2.segmentIds.indexOf(segmentId);
  loop2.segmentIds.splice(index, 1, firstId, secondId);
  return { document: next, pointId: created.pointId, segmentIds: [firstId, secondId] };
}
