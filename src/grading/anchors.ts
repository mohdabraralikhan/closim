// Grading anchor resolution against a pattern document. All lookups use stable
// entity IDs; array positions are never used as references.
import type { EntityId, PatternDocument, PatternPanel, PatternSegment, PatternPoint } from "../pattern/cad.js";
import type { Seam } from "../garment/sewing.js";
import type { GradingAnchor, Vec2 } from "./types.js";
import { GradingError } from "./types.js";

export function findPanel(document: PatternDocument, panelId: EntityId): PatternPanel {
  const panel = document.panels.find((candidate) => candidate.id === panelId);
  if (!panel) throw new GradingError("unknown-entity", `panel '${panelId}' does not exist`, panelId);
  return panel;
}

export function findPoint(document: PatternDocument, pointId: EntityId): PatternPoint {
  const point = document.points.find((candidate) => candidate.id === pointId);
  if (!point) throw new GradingError("unknown-entity", `point '${pointId}' does not exist`, pointId);
  return point;
}

export function findSegment(document: PatternDocument, segmentId: EntityId): PatternSegment {
  const segment = document.segments.find((candidate) => candidate.id === segmentId);
  if (!segment) throw new GradingError("unknown-entity", `segment '${segmentId}' does not exist`, segmentId);
  return segment;
}

/** Evaluate a position along a segment at parameter t, in panel-local coordinates. */
export function evaluateSegmentPoint(document: PatternDocument, segmentId: EntityId, t: number): Vec2 {
  const segment = findSegment(document, segmentId);
  const start = findPoint(document, segment.startPointId);
  const end = findPoint(document, segment.endPointId);
  if (segment.kind === "line") {
    return [start.x + (end.x - start.x) * t, start.y + (end.y - start.y) * t];
  }
  const center = findPoint(document, segment.centerPointId);
  const radius = Math.hypot(start.x - center.x, start.y - center.y);
  const angle = Math.atan2(start.y - center.y, start.x - center.x) + segment.sweepRad * t;
  return [center.x + radius * Math.cos(angle), center.y + radius * Math.sin(angle)];
}

function cornerSharedPoint(document: PatternDocument, panelId: EntityId, segmentIdA: EntityId, segmentIdB: EntityId): PatternPoint {
  const a = findSegment(document, segmentIdA);
  const b = findSegment(document, segmentIdB);
  if (a.panelId !== panelId || b.panelId !== panelId) {
    throw new GradingError("invalid-grading-point", "corner segments must belong to the anchored panel", panelId);
  }
  const shared = [a.startPointId, a.endPointId].filter((id) => id === b.startPointId || id === b.endPointId);
  const unique = [...new Set(shared)];
  if (unique.length !== 1) {
    throw new GradingError("invalid-grading-point", `segments '${segmentIdA}' and '${segmentIdB}' do not share exactly one endpoint`, segmentIdA);
  }
  return findPoint(document, unique[0]);
}

/**
 * Resolve a grading anchor to the pattern points it displaces plus its
 * evaluated position (panel-local). Throws GradingError on unresolvable
 * references ("unknown-entity") or structurally invalid anchors
 * ("invalid-grading-point").
 */
export function resolveAnchor(
  document: PatternDocument,
  anchor: GradingAnchor,
  seams: readonly Seam[] = [],
): { pointIds: EntityId[]; position: Vec2; displacement: "vertex" | "evaluated" } {
  switch (anchor.kind) {
    case "point": {
      findPanel(document, anchor.panelId);
      const point = findPoint(document, anchor.pointId);
      if (point.panelId !== anchor.panelId) {
        throw new GradingError("invalid-grading-point", `point '${anchor.pointId}' does not belong to panel '${anchor.panelId}'`, anchor.pointId);
      }
      return { pointIds: [point.id], position: [point.x, point.y], displacement: "vertex" };
    }
    case "corner": {
      findPanel(document, anchor.panelId);
      const point = cornerSharedPoint(document, anchor.panelId, anchor.segmentIdA, anchor.segmentIdB);
      return { pointIds: [point.id], position: [point.x, point.y], displacement: "vertex" };
    }
    case "edge-relative": {
      findPanel(document, anchor.panelId);
      const segment = findSegment(document, anchor.segmentId);
      if (segment.panelId !== anchor.panelId) {
        throw new GradingError("invalid-grading-point", `segment '${anchor.segmentId}' does not belong to panel '${anchor.panelId}'`, anchor.segmentId);
      }
      if (!Number.isFinite(anchor.t) || anchor.t < 0 || anchor.t > 1) {
        throw new GradingError("invalid-grading-point", `edge-relative parameter ${anchor.t} must be finite within [0,1]`, anchor.segmentId);
      }
      if (anchor.t === 0) {
        const point = findPoint(document, segment.startPointId);
        return { pointIds: [point.id], position: [point.x, point.y], displacement: "vertex" };
      }
      if (anchor.t === 1) {
        const point = findPoint(document, segment.endPointId);
        return { pointIds: [point.id], position: [point.x, point.y], displacement: "vertex" };
      }
      // Intermediate anchor: the derived document keeps master topology, so
      // this anchor contributes an evaluated position (moving with its graded
      // endpoints) rather than a vertex displacement.
      return { pointIds: [], position: evaluateSegmentPoint(document, anchor.segmentId, anchor.t), displacement: "evaluated" };
    }
    case "seam-point": {
      const seam = seams.find((candidate) => candidate.id === anchor.seamId);
      if (!seam) throw new GradingError("unknown-entity", `seam '${anchor.seamId}' does not exist in the grading seam context`, anchor.seamId);
      const side = anchor.side === "a" ? seam.sideA : seam.sideB;
      if (!side.segmentIds.includes(anchor.segmentId)) {
        throw new GradingError(
          "invalid-grading-point",
          `segment '${anchor.segmentId}' is not part of seam '${anchor.seamId}' side ${anchor.side}`,
          anchor.segmentId,
        );
      }
      const segment = findSegment(document, anchor.segmentId);
      const pointId = anchor.endpoint === "start" ? segment.startPointId : segment.endPointId;
      const point = findPoint(document, pointId);
      return { pointIds: [point.id], position: [point.x, point.y], displacement: "vertex" };
    }
  }
}
