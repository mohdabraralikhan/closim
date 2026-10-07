// Shared fixtures for the G9A cad test suite. Everything is built through
// the public G8A PatternDocument API (src/pattern/cad.ts) so the tests
// exercise exactly the document model the production pipeline consumes.

import {
  createBoundaryArc,
  createBoundaryLine,
  createBoundaryLoop,
  createPanel,
  createPatternDocument,
  createPoint,
  type EntityId,
  type PatternDocument,
} from "../../src/pattern/cad.js";
import type { Vec2 } from "../../src/cad/geom.js";

export interface RectFixture {
  document: PatternDocument;
  panelId: EntityId;
  loopId: EntityId;
  points: { bl: EntityId; br: EntityId; tr: EntityId; tl: EntityId };
  segments: { bottom: EntityId; right: EntityId; top: EntityId; left: EntityId };
}

/** CCW rectangle: bottom-left origin, width w, height h (panel-local == global). */
export function rectFixture(
  w = 0.4,
  h = 0.3,
  origin: Vec2 = [0, 0],
  docName = "test-doc",
): RectFixture {
  let document = createPatternDocument(docName);
  const panel = createPanel(document, "rect");
  document = panel.document;
  const loop = createBoundaryLoop(document, panel.panelId, "outer", "ccw");
  document = loop.document;

  const mk = (pos: Vec2): EntityId => {
    const r = createPoint(document, panel.panelId, pos, "boundary");
    document = r.document;
    return r.pointId;
  };
  const bl = mk([origin[0], origin[1]]);
  const br = mk([origin[0] + w, origin[1]]);
  const tr = mk([origin[0] + w, origin[1] + h]);
  const tl = mk([origin[0], origin[1] + h]);

  const link = (a: EntityId, b: EntityId): EntityId => {
    const r = createBoundaryLine(document, panel.panelId, loop.loopId, a, b);
    document = r.document;
    return r.segmentId;
  };
  const bottom = link(bl, br);
  const right = link(br, tr);
  const top = link(tr, tl);
  const left = link(tl, bl);

  return {
    document,
    panelId: panel.panelId,
    loopId: loop.loopId,
    points: { bl, br, tr, tl },
    segments: { bottom, right, top, left },
  };
}

/** CCW rectangle with a CW rectangular hole. */
export function rectWithHoleFixture(): RectFixture & { holeLoopId: EntityId; holePoints: EntityId[]; holeSegments: EntityId[] } {
  const base = rectFixture(0.4, 0.3);
  let document = base.document;
  const hole = createBoundaryLoop(document, base.panelId, "hole", "cw");
  document = hole.document;
  const mk = (pos: Vec2): EntityId => {
    const r = createPoint(document, base.panelId, pos, "boundary");
    document = r.document;
    return r.pointId;
  };
  // CW ordering of an inner rectangle.
  const p0 = mk([0.1, 0.05]);
  const p1 = mk([0.1, 0.25]);
  const p2 = mk([0.3, 0.25]);
  const p3 = mk([0.3, 0.05]);
  const link = (a: EntityId, b: EntityId): EntityId => {
    const r = createBoundaryLine(document, base.panelId, hole.loopId, a, b);
    document = r.document;
    return r.segmentId;
  };
  const s0 = link(p0, p1);
  const s1 = link(p1, p2);
  const s2 = link(p2, p3);
  const s3 = link(p3, p0);
  return {
    ...base,
    document,
    holeLoopId: hole.loopId,
    holePoints: [p0, p1, p2, p3],
    holeSegments: [s0, s1, s2, s3],
  };
}

/**
 * Generic closed polygon panel from ordered CCW/CW points.
 * Returns per-vertex point ids and per-edge segment ids in loop order.
 */
export function polygonFixture(
  points: Vec2[],
  orientation: "ccw" | "cw" = "ccw",
  name = "poly",
): { document: PatternDocument; panelId: EntityId; loopId: EntityId; pointIds: EntityId[]; segmentIds: EntityId[] } {
  let document = createPatternDocument("test-doc");
  const panel = createPanel(document, name);
  document = panel.document;
  const loop = createBoundaryLoop(document, panel.panelId, "outer", orientation);
  document = loop.document;
  const pointIds = points.map((pos) => {
    const r = createPoint(document, panel.panelId, pos, "boundary");
    document = r.document;
    return r.pointId;
  });
  const segmentIds = pointIds.map((id, i) => {
    const r = createBoundaryLine(document, panel.panelId, loop.loopId, id, pointIds[(i + 1) % pointIds.length]);
    document = r.document;
    return r.segmentId;
  });
  return { document, panelId: panel.panelId, loopId: loop.loopId, pointIds, segmentIds };
}

export interface CircleFixture {
  document: PatternDocument;
  panelId: EntityId;
  loopId: EntityId;
  points: { right: EntityId; left: EntityId; center: EntityId };
  segments: { topHalf: EntityId; bottomHalf: EntityId };
}

/**
 * Circular panel built from two semicircular arcs (a valid 2-segment loop:
 * the two halves share endpoints and enclose pi r^2). CCW.
 */
export function circleFixture(r = 0.1): CircleFixture {
  let document = createPatternDocument("test-doc");
  const panel = createPanel(document, "circle");
  document = panel.document;
  const loop = createBoundaryLoop(document, panel.panelId, "outer", "ccw");
  document = loop.document;
  const mk = (pos: Vec2): EntityId => {
    const res = createPoint(document, panel.panelId, pos, "boundary");
    document = res.document;
    return res.pointId;
  };
  const right = mk([r, 0]);
  const left = mk([-r, 0]);
  const center = mk([0, 0]);
  // Right -> left over the top (CCW 0..pi), left -> right under it (pi..2pi).
  const topHalf = createBoundaryArc(document, panel.panelId, loop.loopId, right, left, center, Math.PI);
  document = topHalf.document;
  const bottomHalf = createBoundaryArc(document, panel.panelId, loop.loopId, left, right, center, Math.PI);
  document = bottomHalf.document;
  return {
    document,
    panelId: panel.panelId,
    loopId: loop.loopId,
    points: { right, left, center },
    segments: { topHalf: topHalf.segmentId, bottomHalf: bottomHalf.segmentId },
  };
}

export interface DShapeFixture {
  document: PatternDocument;
  panelId: EntityId;
  loopId: EntityId;
  points: { a: EntityId; b: EntityId; center: EntityId };
  segments: { line: EntityId; arc: EntityId };
}

/**
 * Semicircular "D" panel: diameter from A(-r,0) to B(r,0) plus an arc
 * B -> A over the top (CCW sweep = +pi). Area = pi*r^2/2, perimeter = 2r + pi*r.
 */
export function dShapeFixture(r = 1): DShapeFixture {
  let document = createPatternDocument("test-doc");
  const panel = createPanel(document, "d-shape");
  document = panel.document;
  const loop = createBoundaryLoop(document, panel.panelId, "outer", "ccw");
  document = loop.document;
  const mk = (pos: Vec2): EntityId => {
    const r0 = createPoint(document, panel.panelId, pos, "boundary");
    document = r0.document;
    return r0.pointId;
  };
  const a = mk([-r, 0]);
  const b = mk([r, 0]);
  const center = mk([0, 0]);
  const lineRes = createBoundaryLine(document, panel.panelId, loop.loopId, a, b);
  document = lineRes.document;
  const arcRes = createBoundaryArc(document, panel.panelId, loop.loopId, b, a, center, Math.PI);
  document = arcRes.document;
  return {
    document,
    panelId: panel.panelId,
    loopId: loop.loopId,
    points: { a, b, center },
    segments: { line: lineRes.segmentId, arc: arcRes.segmentId },
  };
}
