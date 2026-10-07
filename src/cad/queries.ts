// G9A — geometry queries over the G8A PatternDocument.
//
// Everything here is READ-ONLY with respect to the document and returns
// structured values or throws PatternCadError for missing references (the
// same precondition style as src/pattern/cad.ts).
//
// Space conventions (critical):
//   - "local"  = raw panel coordinates. PatternDocument stores and validates
//     geometry in local space, so all editing math (trim/extend/merge/split)
//     runs here, exactly matching validatePatternDocument's view.
//   - "global" = workspace coordinates after the panel transform. UI-facing
//     queries (hit-test, box-select, snapping) use this. Lines map exactly
//     through the affine transform; arcs only stay circles under uniform
//     scale, so global arc queries use deterministic sagitta-bounded
//     polylines (error <= the requested tolerance).
//
// Determinism: lookups scan stored arrays in order; every sorted result uses
// explicit (rank, distance, id) tie-breaks; sampling counts depend only on
// (radius, sweep, tolerance).

import {
  PatternCadError,
  globalToLocal,
  localToGlobal,
  type BoundaryLoop,
  type EntityId,
  type PatternDocument,
  type PatternPanel,
  type PatternPoint,
  type PatternSegment,
} from "../pattern/cad.js";
import {
  arcSegmentIntersections,
  arcArcIntersections,
  bbox as geomBbox,
  dist,
  nearestOnArc,
  paramAlong,
  pointInPolygon,
  projectOnSegment,
  sampleArc,
  signedArea,
  sub,
  type ArcGeometry,
  type Vec2,
} from "./geom.js";

export type { Vec2 };
export type Space = "local" | "global";

const DEFAULT_SAGITTA_M = 1e-5; // matches cad.ts validate default

// ---------------------------------------------------------------------------
// Finders (mirror cad.ts's private finders, exported for G9 consumers)
// ---------------------------------------------------------------------------

export function getPanel(doc: PatternDocument, panelId: EntityId): PatternPanel {
  const panel = doc.panels.find((p) => p.id === panelId);
  if (!panel) throw new PatternCadError("missing-reference", `panel '${panelId}' does not exist`, panelId);
  return panel;
}

export function getPoint(doc: PatternDocument, pointId: EntityId, panelId?: EntityId): PatternPoint {
  const point = doc.points.find((p) => p.id === pointId);
  if (!point || (panelId !== undefined && point.panelId !== panelId)) {
    throw new PatternCadError("missing-reference", `point '${pointId}' does not exist in the requested panel`, pointId);
  }
  return point;
}

export function getSegment(doc: PatternDocument, segmentId: EntityId, panelId?: EntityId): PatternSegment {
  const segment = doc.segments.find((s) => s.id === segmentId);
  if (!segment || (panelId !== undefined && segment.panelId !== panelId)) {
    throw new PatternCadError("missing-reference", `segment '${segmentId}' does not exist in the requested panel`, segmentId);
  }
  return segment;
}

export function getLoop(doc: PatternDocument, panelId: EntityId, loopId: EntityId): BoundaryLoop {
  const panel = getPanel(doc, panelId);
  const loop = panel.boundaryLoops.find((l) => l.id === loopId);
  if (!loop) throw new PatternCadError("missing-reference", `boundary loop '${loopId}' does not exist`, loopId);
  return loop;
}

export function findLoopByRole(panel: PatternPanel, role: "outer" | "hole"): BoundaryLoop {
  const loop = panel.boundaryLoops.find((l) => l.role === role);
  if (!loop) throw new PatternCadError("open-boundary", `panel '${panel.id}' has no ${role} loop`, panel.id);
  return loop;
}

// ---------------------------------------------------------------------------
// Segment resolution (local space)
// ---------------------------------------------------------------------------

export interface ResolvedSegment {
  segment: PatternSegment;
  panel: PatternPanel;
  start: Vec2;
  end: Vec2;
  /** Present for kind === "arc" (center from centerPointId, radius from start). */
  arc?: ArcGeometry;
  /** Exact length in local units. */
  length: number;
}

/** Resolve a segment to coordinates in its OWN panel's local space. */
export function resolveSegment(doc: PatternDocument, segmentId: EntityId, panelId?: EntityId): ResolvedSegment {
  const segment = getSegment(doc, segmentId, panelId);
  const start = getPoint(doc, segment.startPointId, segment.panelId);
  const end = getPoint(doc, segment.endPointId, segment.panelId);
  const a: Vec2 = [start.x, start.y];
  const b: Vec2 = [end.x, end.y];
  const panel = getPanel(doc, segment.panelId);
  if (segment.kind === "line") {
    return { segment, panel, start: a, end: b, length: dist(a, b) };
  }
  const center = getPoint(doc, segment.centerPointId, segment.panelId);
  const c: Vec2 = [center.x, center.y];
  const radius = dist(a, c);
  const arc: ArcGeometry = {
    center: c,
    radius,
    a0: Math.atan2(a[1] - c[1], a[0] - c[0]),
    sweep: segment.sweepRad,
  };
  return {
    segment, panel, start: a, end: b, arc,
    length: radius * Math.abs(segment.sweepRad),
  };
}

/** Ordered resolved segments of a loop (chain order as stored). */
export function resolveLoop(doc: PatternDocument, panelId: EntityId, loopId: EntityId): ResolvedSegment[] {
  const loop = getLoop(doc, panelId, loopId);
  return loop.segmentIds.map((sid) => resolveSegment(doc, sid, panelId));
}

// ---------------------------------------------------------------------------
// Global-space mapping
// ---------------------------------------------------------------------------

/** Panel-local -> workspace position for one point entity. */
export function pointToGlobal(doc: PatternDocument, pointId: EntityId): Vec2 {
  const point = getPoint(doc, pointId);
  const panel = getPanel(doc, point.panelId);
  return localToGlobal(panel, [point.x, point.y]);
}

/** Workspace -> panel-local position. */
export function globalPointToPanelLocal(doc: PatternDocument, panelId: EntityId, globalPos: Vec2): Vec2 {
  return globalToLocal(getPanel(doc, panelId), globalPos);
}

export type SegmentShapeGlobal =
  | { kind: "line"; a: Vec2; b: Vec2 }
  /**
   * Arc in workspace space. Under non-uniform panel scale the true image is
   * an ellipse; `points` are the authoritative sampled polyline (sagitta
   * bounded), while center/radius describe the circle only when `isotropic`.
   */
  | { kind: "arc"; points: Vec2[]; center: Vec2; radius: number; isotropic: boolean };

/** Workspace-space shape of a segment (deterministic sampling for arcs). */
export function segmentShapeGlobal(
  doc: PatternDocument, segmentId: EntityId, sagittaTolM = DEFAULT_SAGITTA_M,
): SegmentShapeGlobal {
  const r = resolveSegment(doc, segmentId);
  const panel = r.panel;
  if (r.segment.kind === "line") {
    return { kind: "line", a: localToGlobal(panel, r.start), b: localToGlobal(panel, r.end) };
  }
  const arc = r.arc!;
  const samples = sampleArc(arc.center, arc.radius, arc.a0, arc.sweep, sagittaTolM)
    .map((p) => localToGlobal(panel, p));
  const sx = panel.transform.scale[0];
  const sy = panel.transform.scale[1];
  const isotropic = Math.abs(sx - sy) <= 1e-12 * Math.max(Math.abs(sx), Math.abs(sy));
  const centerG = localToGlobal(panel, arc.center);
  const radiusG = arc.radius * (isotropic ? sx : (sx + sy) / 2);
  return { kind: "arc", points: samples, center: centerG, radius: radiusG, isotropic };
}

// ---------------------------------------------------------------------------
// Nearest-point queries
// ---------------------------------------------------------------------------

export interface NearestResult {
  pos: Vec2;
  distance: number;
  /** Parameter along the segment: t in [0,1] (line), fraction of sweep (arc). */
  t: number;
}

/** Nearest point on a segment to `p` in the requested space. */
export function nearestOnSegment(
  doc: PatternDocument, segmentId: EntityId, p: Vec2, space: Space = "local",
): NearestResult {
  const r = resolveSegment(doc, segmentId);
  if (space === "local") {
    if (r.segment.kind === "line") {
      const proj = projectOnSegment(p, r.start, r.end);
      return { pos: proj.pos, distance: proj.distance, t: proj.t };
    }
    const arc = r.arc!;
    const hit = nearestOnArc(arc, p);
    const ang = Math.atan2(hit[1] - arc.center[1], hit[0] - arc.center[0]);
    let rel = ang - arc.a0;
    // Fold into the sweep's direction.
    if (arc.sweep > 0) {
      while (rel < -1e-12) rel += 2 * Math.PI;
      while (rel > arc.sweep + 2 * Math.PI) rel -= 2 * Math.PI;
    } else {
      while (rel > 1e-12) rel -= 2 * Math.PI;
      while (rel < arc.sweep - 2 * Math.PI) rel += 2 * Math.PI;
    }
    const t = arc.sweep === 0 ? 0 : rel / arc.sweep;
    return { pos: hit, distance: dist(hit, p), t: t < 0 ? 0 : t > 1 ? 1 : t };
  }
  // Global space: lines exact, arcs via the authoritative polyline.
  const shape = segmentShapeGlobal(doc, segmentId);
  if (shape.kind === "line") {
    const proj = projectOnSegment(p, shape.a, shape.b);
    return { pos: proj.pos, distance: proj.distance, t: proj.t };
  }
  let best = { pos: shape.points[0], distance: Infinity, t: 0 };
  for (let i = 0; i + 1 < shape.points.length; i++) {
    const proj = projectOnSegment(p, shape.points[i], shape.points[i + 1]);
    const t = (i + proj.t) / (shape.points.length - 1);
    if (proj.distance < best.distance) best = { pos: proj.pos, distance: proj.distance, t };
  }
  return best;
}

// ---------------------------------------------------------------------------
// Intersections
// ---------------------------------------------------------------------------

/**
 * Intersections between two segments.
 * - Same panel: exact local-space math (line-line, line-arc, arc-arc),
 *   results mapped to the requested space.
 * - Different panels: workspace space is the only shared frame; both
 *   segments must be lines (arcs under panel transforms are not guaranteed
 *   circular) — otherwise PatternCadError("unsupported-operation").
 * Deterministic order: by parameter along `segAId`, then coordinates.
 */
export function intersectSegments(
  doc: PatternDocument,
  segAId: EntityId,
  segBId: EntityId,
  space: Space = "local",
): Vec2[] {
  if (segAId === segBId) {
    throw new PatternCadError("unsupported-operation", "cannot intersect a segment with itself", segAId);
  }
  const a = resolveSegment(doc, segAId);
  const b = resolveSegment(doc, segBId);
  const samePanel = a.segment.panelId === b.segment.panelId;
  const angEps = 1e-12;

  let hitsLocal: Vec2[];
  if (samePanel) {
    hitsLocal = localHits(a, b, angEps);
  } else {
    if (space !== "global") {
      throw new PatternCadError(
        "unsupported-operation",
        `cross-panel intersection requires global space (panels ${a.segment.panelId} vs ${b.segment.panelId})`,
        segAId,
      );
    }
    if (a.segment.kind !== "line" || b.segment.kind !== "line") {
      throw new PatternCadError(
        "unsupported-operation",
        "cross-panel intersection supports line segments only",
        segAId,
      );
    }
    const ga = segmentShapeGlobal(doc, segAId) as { kind: "line"; a: Vec2; b: Vec2 };
    const gb = segmentShapeGlobal(doc, segBId) as { kind: "line"; a: Vec2; b: Vec2 };
    const hit = lineHit(ga.a, ga.b, gb.a, gb.b, angEps);
    return hit ? [hit] : [];
  }

  if (space === "local") return hitsLocal;
  const panel = a.panel;
  return hitsLocal.map((p) => localToGlobal(panel, p));
}

function lineHit(a1: Vec2, a2: Vec2, b1: Vec2, b2: Vec2, angEps: number): Vec2 | null {
  const d1 = sub(a2, a1);
  const d2 = sub(b2, b1);
  const den = d1[0] * d2[1] - d1[1] * d2[0];
  const denScale = Math.hypot(d1[0], d1[1]) * Math.hypot(d2[0], d2[1]);
  if (denScale === 0 || Math.abs(den) <= angEps * denScale) return null;
  const t = ((b1[0] - a1[0]) * d2[1] - (b1[1] - a1[1]) * d2[0]) / den;
  const u = ((b1[0] - a1[0]) * d1[1] - (b1[1] - a1[1]) * d1[0]) / den;
  if (t < -1e-9 || t > 1 + 1e-9 || u < -1e-9 || u > 1 + 1e-9) return null;
  return [a1[0] + d1[0] * t, a1[1] + d1[1] * t];
}

function localHits(a: ResolvedSegment, b: ResolvedSegment, angEps: number): Vec2[] {
  if (a.segment.kind === "line" && b.segment.kind === "line") {
    const hit = lineHit(a.start, a.end, b.start, b.end, angEps);
    return hit ? [hit] : [];
  }
  if (a.segment.kind === "arc" && b.segment.kind === "line") {
    return arcSegmentIntersections(a.arc!, b.start, b.end, angEps).map((h) => h.pos);
  }
  if (a.segment.kind === "line" && b.segment.kind === "arc") {
    return arcSegmentIntersections(b.arc!, a.start, a.end, angEps).map((h) => h.pos);
  }
  return arcArcIntersections(a.arc!, b.arc!, angEps);
}

// ---------------------------------------------------------------------------
// Measurements (local space — same frame cad.ts measures in)
// ---------------------------------------------------------------------------

export interface LoopMeasurement {
  /** Exact perimeter: line lengths + arc lengths. */
  perimeter: number;
  /**
   * Exact signed area: chord shoelace + circular-segment corrections
   * r^2/2 * (theta - sin theta) per arc. Positive = CCW.
   */
  area: number;
  /** Sign of the sampled polygon area (matches validatePatternDocument). */
  winding: "ccw" | "cw" | null;
  bbox: { min: Vec2; max: Vec2 } | null;
}

/** Deterministic open polyline of a loop in local space (arcs sampled). */
export function sampleLoopLocal(
  doc: PatternDocument, panelId: EntityId, loopId: EntityId, sagittaTolM = DEFAULT_SAGITTA_M,
): Vec2[] {
  const segments = resolveLoop(doc, panelId, loopId);
  const pts: Vec2[] = [];
  for (const r of segments) {
    if (r.segment.kind === "line") {
      pts.push(r.start);
    } else {
      const arc = r.arc!;
      const s = sampleArc(arc.center, arc.radius, arc.a0, arc.sweep, sagittaTolM);
      pts.push(...s.slice(0, -1)); // drop the duplicate join with the next segment
    }
  }
  return pts;
}

export function measureLoop(
  doc: PatternDocument, panelId: EntityId, loopId: EntityId, sagittaTolM = DEFAULT_SAGITTA_M,
): LoopMeasurement {
  const segments = resolveLoop(doc, panelId, loopId);
  let perimeter = 0;
  let area2 = 0; // 2 * area accumulator
  const pts: Vec2[] = [];
  for (const r of segments) {
    perimeter += r.length;
    // Chord contribution.
    area2 += r.start[0] * r.end[1] - r.end[0] * r.start[1];
    if (r.segment.kind === "arc" && r.arc) {
      const th = r.arc.sweep;
      area2 += r.arc.radius * r.arc.radius * (th - Math.sin(th));
    }
    pts.push(r.start, r.end);
  }
  const sampled = sampleLoopLocal(doc, panelId, loopId, sagittaTolM);
  const sampleSign = sampled.length >= 3 ? Math.sign(signedArea(sampled)) : 0;
  return {
    perimeter,
    area: area2 / 2,
    winding: sampleSign === 0 ? null : sampleSign > 0 ? "ccw" : "cw",
    bbox: geomBbox(pts),
  };
}

export interface PanelMeasurement extends LoopMeasurement {
  /** |outer area| minus |hole areas| (exact). */
  netArea: number;
  holeCount: number;
}

export function measurePanel(doc: PatternDocument, panelId: EntityId): PanelMeasurement {
  const panel = getPanel(doc, panelId);
  const outer = findLoopByRole(panel, "outer");
  const base = measureLoop(doc, panelId, outer.id);
  let holeArea = 0;
  let holeCount = 0;
  for (const loop of panel.boundaryLoops) {
    if (loop.role !== "hole") continue;
    holeCount++;
    holeArea += Math.abs(measureLoop(doc, panelId, loop.id).area);
  }
  return { ...base, netArea: Math.abs(base.area) - holeArea, holeCount };
}

/** Workspace-space bounding box of a panel (or the whole document). */
export function bboxGlobal(doc: PatternDocument, panelId?: EntityId): { min: Vec2; max: Vec2 } | null {
  const panels = panelId ? [getPanel(doc, panelId)] : doc.panels;
  const pts: Vec2[] = [];
  for (const panel of panels) {
    for (const point of doc.points) {
      if (point.panelId !== panel.id) continue;
      pts.push(localToGlobal(panel, [point.x, point.y]));
    }
  }
  return geomBbox(pts);
}

// ---------------------------------------------------------------------------
// Reference queries (used by merge/delete safety checks)
// ---------------------------------------------------------------------------

export interface PointReferences {
  segments: EntityId[];
  dimensions: EntityId[];
  constraints: EntityId[];
}

/** Everything that references a point by id (deterministic store order). */
export function pointReferences(doc: PatternDocument, pointId: EntityId): PointReferences {
  const segments = doc.segments
    .filter(
      (s) =>
        s.startPointId === pointId ||
        s.endPointId === pointId ||
        (s.kind === "arc" && s.centerPointId === pointId),
    )
    .map((s) => s.id);
  const dimensions: EntityId[] = [];
  const constraints: EntityId[] = [];
  for (const panel of doc.panels) {
    for (const d of panel.dimensions) {
      if (d.pointAId === pointId || d.pointBId === pointId) dimensions.push(d.id);
    }
    for (const c of panel.constraints) {
      if (c.pointAId === pointId || c.pointBId === pointId) constraints.push(c.id);
    }
  }
  return { segments, dimensions, constraints };
}

/** True when a point id appears anywhere as a reference. */
export function pointIsReferenced(doc: PatternDocument, pointId: EntityId): boolean {
  const r = pointReferences(doc, pointId);
  return r.segments.length + r.dimensions.length + r.constraints.length > 0;
}

// ---------------------------------------------------------------------------
// Hit-testing + selection geometry (workspace space)
// ---------------------------------------------------------------------------

export type HitKind = "point" | "segment" | "panel";

export interface Hit {
  entityId: EntityId;
  kind: HitKind;
  panelId: EntityId;
  /** Distance from the query position to the entity's surface (0 for inside-panel). */
  distance: number;
  /** Closest position on the entity (query position for panels). */
  pos: Vec2;
}

const KIND_RANK: Record<HitKind, number> = { point: 0, segment: 1, panel: 2 };

/**
 * Entities within `toleranceM` of a workspace position.
 * Deterministic ordering: kind rank (point < segment < panel), distance,
 * then entity id. Vertex hits therefore win over the edges that share them.
 */
export function hitTest(
  doc: PatternDocument, globalPos: Vec2, toleranceM: number, sagittaTolM = DEFAULT_SAGITTA_M,
): Hit[] {
  const hits: Hit[] = [];
  for (const point of doc.points) {
    const g = pointToGlobal(doc, point.id);
    const d = dist(g, globalPos);
    if (d <= toleranceM) hits.push({ entityId: point.id, kind: "point", panelId: point.panelId, distance: d, pos: g });
  }
  for (const segment of doc.segments) {
    const near = nearestOnSegment(doc, segment.id, globalPos, "global");
    if (near.distance <= toleranceM) {
      hits.push({ entityId: segment.id, kind: "segment", panelId: segment.panelId, distance: near.distance, pos: near.pos });
    }
  }
  for (const panel of doc.panels) {
    if (pointInPanelGlobal(doc, panel, globalPos)) {
      hits.push({ entityId: panel.id, kind: "panel", panelId: panel.id, distance: 0, pos: globalPos });
    }
  }
  hits.sort(
    (x, y) =>
      KIND_RANK[x.kind] - KIND_RANK[y.kind] ||
      x.distance - y.distance ||
      (x.entityId < y.entityId ? -1 : x.entityId > y.entityId ? 1 : 0),
  );
  return hits;
}

/** Nearest entity of any kind within tolerance (null when none). */
export function nearestHit(
  doc: PatternDocument, globalPos: Vec2, toleranceM: number, sagittaTolM = DEFAULT_SAGITTA_M,
): Hit | null {
  const hits = hitTest(doc, globalPos, toleranceM, sagittaTolM);
  return hits.length > 0 ? hits[0] : null;
}

/** Inside-the-panel test in workspace space (boundary counts as inside). */
export function pointInPanelGlobal(doc: PatternDocument, panel: PatternPanel, globalPos: Vec2): boolean {
  const outer = panel.boundaryLoops.find((l) => l.role === "outer");
  if (!outer) return false;
  let local: Vec2;
  try {
    local = globalToLocal(panel, globalPos);
  } catch {
    return false; // degenerate transform
  }
  return pointInPanelLocal(doc, panel, local);
}

function pointInPanelLocal(doc: PatternDocument, panel: PatternPanel, localPos: Vec2): boolean {
  const outer = panel.boundaryLoops.find((l) => l.role === "outer");
  if (!outer) return false;
  const outerPts = sampleLoopLocal(doc, panel.id, outer.id);
  if (outerPts.length < 3) return false;
  if (!pointInPolygon(localPos, outerPts)) return false;
  for (const loop of panel.boundaryLoops) {
    if (loop.role !== "hole") continue;
    const holePts = sampleLoopLocal(doc, panel.id, loop.id);
    if (holePts.length >= 3 && pointInPolygon(localPos, holePts)) return false;
  }
  return true;
}

/** Inside test in LOCAL panel space (boundary counts as inside). */
export function pointInPanel(doc: PatternDocument, panelId: EntityId, localPos: Vec2): boolean {
  return pointInPanelLocal(doc, getPanel(doc, panelId), localPos);
}

export interface BoxSelectOptions {
  /** "contain" = fully inside the box; "overlap" = intersects it (default). */
  mode?: "contain" | "overlap";
  sagittaTolM?: number;
}

/**
 * Entities interacting with an axis-aligned workspace box.
 * Deterministic: points, then segments, then panels, each in store order.
 */
export function boxSelect(
  doc: PatternDocument, min: Vec2, max: Vec2, opts: BoxSelectOptions = {},
): EntityId[] {
  const mode = opts.mode ?? "overlap";
  const sagittaTolM = opts.sagittaTolM ?? DEFAULT_SAGITTA_M;
  const lo: Vec2 = [Math.min(min[0], max[0]), Math.min(min[1], max[1])];
  const hi: Vec2 = [Math.max(min[0], max[0]), Math.max(min[1], max[1])];
  const inBox = (p: Vec2): boolean => p[0] >= lo[0] && p[0] <= hi[0] && p[1] >= lo[1] && p[1] <= hi[1];
  const boxAsLoop: Vec2[] = [[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]];

  const out: EntityId[] = [];
  for (const point of doc.points) {
    if (inBox(pointToGlobal(doc, point.id))) out.push(point.id);
  }
  for (const segment of doc.segments) {
    const pts = segmentShapePointsGlobal(doc, segment.id, sagittaTolM);
    if (mode === "contain") {
      if (pts.every(inBox)) out.push(segment.id);
    } else if (pts.some(inBox) || segmentIntersectsBox(pts, boxAsLoop)) {
      out.push(segment.id);
    }
  }
  for (const panel of doc.panels) {
    const corners: Vec2[] = [lo, [hi[0], lo[1]], hi, [lo[0], hi[1]]];
    if (mode === "contain") {
      const allIn = corners.every((c) => pointInPanelGlobal(doc, panel, c));
      // "Contain" for a panel means the box lies fully inside it (the panel
      // is the selected region) — rare; also accept exact-boundary boxes.
      if (allIn) out.push(panel.id);
    } else {
      // Overlap: any panel corner inside the box, or box corner on panel.
      const panelPts = doc.points.filter((p) => p.panelId === panel.id);
      const anyPanelPointInBox = panelPts.some((p) => inBox(pointToGlobal(doc, p.id)));
      const anyBoxCornerInPanel = corners.some((c) => pointInPanelGlobal(doc, panel, c));
      if (anyPanelPointInBox || anyBoxCornerInPanel) out.push(panel.id);
    }
  }
  return out;
}

function segmentShapePointsGlobal(doc: PatternDocument, segmentId: EntityId, sagittaTolM: number): Vec2[] {
  const shape = segmentShapeGlobal(doc, segmentId, sagittaTolM);
  return shape.kind === "line" ? [shape.a, shape.b] : shape.points;
}

function segmentIntersectsBox(pts: Vec2[], box: Vec2[]): boolean {
  for (let i = 0; i + 1 < pts.length; i++) {
    for (let j = 0; j < 4; j++) {
      const b0 = box[j];
      const b1 = box[(j + 1) % 4];
      if (segmentCross(pts[i], pts[i + 1], b0, b1)) return true;
    }
  }
  return false;
}

function segmentCross(p1: Vec2, p2: Vec2, p3: Vec2, p4: Vec2): boolean {
  const d1 = sub(p2, p1);
  const d2 = sub(p4, p3);
  const den = d1[0] * d2[1] - d1[1] * d2[0];
  const denScale = Math.hypot(d1[0], d1[1]) * Math.hypot(d2[0], d2[1]);
  if (denScale === 0 || Math.abs(den) <= 1e-12 * denScale) return false;
  const t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / den;
  const u = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / den;
  return t > 0 && t < 1 && u > 0 && u < 1;
}

// ---------------------------------------------------------------------------
// Raw parameter helpers used by trim/extend
// ---------------------------------------------------------------------------

/** Unclamped parameter of a point along the segment's start->end direction. */
export function segmentParam(r: ResolvedSegment, p: Vec2): number {
  if (r.segment.kind === "line") return paramAlong(p, r.start, r.end);
  // Arc: fraction of sweep (may fall outside [0,1] — callers treat that as
  // "beyond the endpoint").
  const arc = r.arc!;
  const ang = Math.atan2(p[1] - arc.center[1], p[0] - arc.center[0]);
  let rel = ang - arc.a0;
  const TAU = 2 * Math.PI;
  if (arc.sweep > 0) {
    while (rel < 0) rel += TAU;
    while (rel > TAU) rel -= TAU;
  } else {
    while (rel > 0) rel -= TAU;
    while (rel < -TAU) rel += TAU;
  }
  return arc.sweep === 0 ? 0 : rel / arc.sweep;
}
