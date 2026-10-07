// G9A — explicit, deterministic snapping.
//
// Snapping is NEVER applied implicitly: core operations take raw coordinates
// and callers (G9D tools) decide when to consult snapPosition(). Priority is
// fixed and documented, so the same document + same candidate position always
// produce the same result:
//
//   vertex > intersection > midpoint > on-segment > grid > none
//
// At equal priority the candidate closest to the raw position wins, with the
// entity id as the final tie-break. All distances are workspace (global)
// meters; arcs are handled through their deterministic sagitta-bounded
// polylines so non-uniform panel scale cannot skew the result.

import type { EntityId, PatternDocument } from "../pattern/cad.js";
import { PatternCadError } from "../pattern/cad.js";
import { dist, type Vec2 } from "./geom.js";
import {
  intersectSegments,
  nearestOnSegment,
  pointToGlobal,
  segmentShapeGlobal,
} from "./queries.js";

export type SnapKind = "vertex" | "intersection" | "midpoint" | "on-segment" | "grid" | "none";

export interface SnapResult {
  /** Workspace position to use (raw input when kind === "none"). */
  pos: Vec2;
  kind: SnapKind;
  /** Entities that justify the snap (1 for vertex/midpoint/on-segment, 2 for intersection, 0 for grid/none). */
  sourceIds: EntityId[];
}

export interface SnapOptions {
  /** Snap capture radius in workspace meters. */
  toleranceM: number;
  /** Grid step in meters; omit or 0 disables grid snapping. */
  gridM?: number;
  /** Restrict candidate entities to these panels (default: all). */
  panelIds?: readonly EntityId[];
  /** Sagitta tolerance for arc sampling (m). Default 1e-5. */
  sagittaTolM?: number;
}

const KIND_PRIORITY: Record<SnapKind, number> = {
  vertex: 0,
  intersection: 1,
  midpoint: 2,
  "on-segment": 3,
  grid: 4,
  none: 5,
};

interface Candidate {
  kind: SnapKind;
  pos: Vec2;
  sourceIds: EntityId[];
  distance: number;
}

/**
 * Snap a raw workspace position to nearby geometry.
 * Returns kind "none" (and the raw position) when nothing is within
 * tolerance and grid snapping is disabled or lands on the raw point.
 */
export function snapPosition(
  doc: PatternDocument, raw: Vec2, opts: SnapOptions,
): SnapResult {
  if (!(opts.toleranceM > 0)) throw new Error(`snap: toleranceM must be > 0, got ${opts.toleranceM}`);
  const sagittaTolM = opts.sagittaTolM ?? 1e-5;
  const inScope = (panelId: EntityId): boolean =>
    opts.panelIds === undefined || opts.panelIds.includes(panelId);

  const candidates: Candidate[] = [];

  // --- vertex (points: endpoints + arc centers + construction) ---
  for (const point of doc.points) {
    if (!inScope(point.panelId)) continue;
    const g = pointToGlobal(doc, point.id);
    const d = dist(g, raw);
    if (d <= opts.toleranceM) candidates.push({ kind: "vertex", pos: g, sourceIds: [point.id], distance: d });
  }

  // --- segment-based candidates (only segments close enough to matter) ---
  const nearSegmentIds: EntityId[] = [];
  for (const segment of doc.segments) {
    if (!inScope(segment.panelId)) continue;
    const near = nearestOnSegment(doc, segment.id, raw, "global");
    if (near.distance <= opts.toleranceM) nearSegmentIds.push(segment.id);
  }

  // --- midpoint ---
  for (const id of nearSegmentIds) {
    const shape = segmentShapeGlobal(doc, id, sagittaTolM);
    const mid: Vec2 = shape.kind === "line"
      ? [(shape.a[0] + shape.b[0]) / 2, (shape.a[1] + shape.b[1]) / 2]
      : shape.points[Math.floor((shape.points.length - 1) / 2)];
    const d = dist(mid, raw);
    if (d <= opts.toleranceM) candidates.push({ kind: "midpoint", pos: mid, sourceIds: [id], distance: d });
  }

  // --- on-segment (projection) ---
  for (const id of nearSegmentIds) {
    const near = nearestOnSegment(doc, id, raw, "global");
    if (near.distance <= opts.toleranceM) {
      candidates.push({ kind: "on-segment", pos: near.pos, sourceIds: [id], distance: near.distance });
    }
  }

  // --- intersection (pairwise among nearby segments; unsupported pairs skip) ---
  for (let i = 0; i < nearSegmentIds.length; i++) {
    for (let j = i + 1; j < nearSegmentIds.length; j++) {
      let hits: Vec2[];
      try {
        hits = intersectSegments(doc, nearSegmentIds[i], nearSegmentIds[j], "global");
      } catch (error) {
        if (error instanceof PatternCadError && error.code === "unsupported-operation") continue;
        throw error;
      }
      for (const hit of hits) {
        const d = dist(hit, raw);
        if (d <= opts.toleranceM) {
          candidates.push({
            kind: "intersection",
            pos: hit,
            sourceIds: [nearSegmentIds[i], nearSegmentIds[j]],
            distance: d,
          });
        }
      }
    }
  }

  // --- grid ---
  const gridM = opts.gridM ?? 0;
  if (gridM > 0) {
    const gx = Math.round(raw[0] / gridM) * gridM;
    const gy = Math.round(raw[1] / gridM) * gridM;
    const g: Vec2 = [gx, gy];
    const d = dist(g, raw);
    if (d <= opts.toleranceM) candidates.push({ kind: "grid", pos: g, sourceIds: [], distance: d });
  }

  if (candidates.length === 0) {
    return { pos: raw, kind: "none", sourceIds: [] };
  }

  // Fixed priority, then distance, then a stable source-id comparison so two
  // equally good candidates always resolve the same way.
  candidates.sort((a, b) => {
    const pa = KIND_PRIORITY[a.kind];
    const pb = KIND_PRIORITY[b.kind];
    if (pa !== pb) return pa - pb;
    if (a.distance !== b.distance) return a.distance - b.distance;
    const ka = a.sourceIds.join(",");
    const kb = b.sourceIds.join(",");
    return ka < kb ? -1 : ka > kb ? 1 : 0;
  });
  const best = candidates[0];
  return { pos: best.pos, kind: best.kind, sourceIds: best.sourceIds };
}
