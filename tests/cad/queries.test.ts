// G9A document queries: finders, segment resolution, global mapping,
// nearest/intersection queries, exact measurements, hit-test + box-select.
// Uses the G8A PatternDocument built through tests/cad/fixtures.
import { describe, it, expect } from "vitest";
import {
  PatternCadError,
  createConstructionLine,
  createDistanceDimension,
  createPoint,
  movePanel,
  rotatePanel,
  type PatternDocument,
} from "../../src/pattern/cad.js";
import {
  bboxGlobal,
  boxSelect,
  findLoopByRole,
  getLoop,
  getPanel,
  getPoint,
  getSegment,
  hitTest,
  intersectSegments,
  measureLoop,
  measurePanel,
  nearestHit,
  nearestOnSegment,
  pointInPanel,
  pointInPanelGlobal,
  pointIsReferenced,
  pointReferences,
  pointToGlobal,
  resolveLoop,
  resolveSegment,
  sampleLoopLocal,
  segmentShapeGlobal,
  segmentParam,
} from "../../src/cad/queries.js";
import { dist } from "../../src/cad/geom.js";
import { dShapeFixture, rectFixture, rectWithHoleFixture } from "./fixtures.js";

const W = 0.4;
const H = 0.3;

describe("finders", () => {
  it("resolves panels/points/segments/loops by id", () => {
    const f = rectFixture();
    expect(getPanel(f.document, f.panelId).name).toBe("rect");
    expect(getPoint(f.document, f.points.bl, f.panelId).x).toBe(0);
    expect(getSegment(f.document, f.segments.bottom, f.panelId).kind).toBe("line");
    expect(getLoop(f.document, f.panelId, f.loopId).role).toBe("outer");
    expect(findLoopByRole(getPanel(f.document, f.panelId), "outer").id).toBe(f.loopId);
  });

  it("throws PatternCadError with missing-reference for unknown ids", () => {
    const f = rectFixture();
    expect(() => getPanel(f.document, "nope")).toThrowError(PatternCadError);
    expect(() => getPoint(f.document, "nope")).toThrowError(/missing-reference/);
    expect(() => getSegment(f.document, "nope")).toThrowError(/missing-reference/);
    expect(() => getLoop(f.document, f.panelId, "nope")).toThrowError(/missing-reference/);
    // Wrong panel scoping also rejects.
    expect(() => getPoint(f.document, f.points.bl, "other-panel")).toThrowError(/missing-reference/);
  });
});

describe("resolveSegment", () => {
  it("resolves line segments exactly", () => {
    const f = rectFixture();
    const r = resolveSegment(f.document, f.segments.bottom);
    expect(r.start).toEqual([0, 0]);
    expect(r.end).toEqual([W, 0]);
    expect(r.arc).toBeUndefined();
    expect(r.length).toBeCloseTo(W, 12);
  });

  it("resolves arc segments to center/radius/a0/sweep with exact length", () => {
    const f = dShapeFixture(1);
    const r = resolveSegment(f.document, f.segments.arc);
    expect(r.segment.kind).toBe("arc");
    expect(r.arc!.center[0]).toBeCloseTo(0, 12);
    expect(r.arc!.radius).toBeCloseTo(1, 12);
    expect(r.arc!.a0).toBeCloseTo(0, 12); // starts at (1, 0)
    expect(r.arc!.sweep).toBeCloseTo(Math.PI, 12);
    expect(r.length).toBeCloseTo(Math.PI, 12);
  });

  it("resolveLoop preserves chain order", () => {
    const f = rectFixture();
    const loop = resolveLoop(f.document, f.panelId, f.loopId);
    expect(loop.length).toBe(4);
    expect(loop[0].segment.id).toBe(f.segments.bottom);
    // Chain: each segment's start equals the previous end.
    for (let i = 0; i < loop.length; i++) {
      const a = loop[i];
      const b = loop[(i + 1) % loop.length];
      expect(a.end).toEqual(b.start);
    }
  });
});

describe("global mapping", () => {
  it("pointToGlobal applies translation", () => {
    const f = rectFixture();
    const moved = movePanel(f.document, f.panelId, [1, 0.5]);
    const g = pointToGlobal(moved, f.points.tr);
    expect(g[0]).toBeCloseTo(W + 1, 12);
    expect(g[1]).toBeCloseTo(H + 0.5, 12);
  });

  it("pointToGlobal applies rotation about the origin", () => {
    const f = rectFixture();
    const rotated = rotatePanel(f.document, f.panelId, Math.PI / 2, [0, 0]);
    const g = pointToGlobal(rotated, f.points.br); // (0.4, 0) -> (0, 0.4)
    expect(g[0]).toBeCloseTo(0, 9);
    expect(g[1]).toBeCloseTo(W, 9);
  });

  it("segmentShapeGlobal maps lines exactly and flags arc isotropy", () => {
    const f = rectFixture();
    const shape = segmentShapeGlobal(f.document, f.segments.bottom);
    expect(shape.kind).toBe("line");

    const d = dShapeFixture(1);
    const iso = segmentShapeGlobal(d.document, d.segments.arc);
    expect(iso.kind).toBe("arc");
    if (iso.kind === "arc") expect(iso.isotropic).toBe(true);

    // Non-uniform layout scale stretches the workspace image of the bottom
    // edge (0.4m local -> 0.8m global) and makes arcs non-isotropic.
    const scaled = layoutScale(f, 2, 1);
    const shape2 = segmentShapeGlobal(scaled, f.segments.bottom);
    expect(shape2.kind).toBe("line");
    if (shape2.kind === "line") {
      expect(shape2.b[0] - shape2.a[0]).toBeCloseTo(W * 2, 12);
    }
    const dScaledDoc = layoutScale(d, 2, 1);
    const arcShape = segmentShapeGlobal(dScaledDoc, d.segments.arc);
    expect(arcShape.kind === "arc" && arcShape.isotropic).toBe(false);
  });

  it("bboxGlobal spans transformed points", () => {
    const f = rectFixture();
    const moved = movePanel(f.document, f.panelId, [10, 0]);
    const b = bboxGlobal(moved, f.panelId);
    expect(b!.min[0]).toBeCloseTo(10, 12);
    expect(b!.max[0]).toBeCloseTo(10 + W, 12);
    expect(b!.max[1]).toBeCloseTo(H, 12);
  });
});

function layoutScale(f: { document: PatternDocument; panelId: string }, sx: number, sy: number): PatternDocument {
  const doc = JSON.parse(JSON.stringify(f.document)) as PatternDocument;
  const panel = getPanel(doc, f.panelId);
  panel.transform.scale = [sx, sy];
  return doc;
}

describe("nearestOnSegment", () => {
  it("local line: exact projection with parameter", () => {
    const f = rectFixture();
    const n = nearestOnSegment(f.document, f.segments.bottom, [0.1, 0.05], "local");
    expect(n.pos[0]).toBeCloseTo(0.1, 12);
    expect(n.pos[1]).toBeCloseTo(0, 12);
    expect(n.distance).toBeCloseTo(0.05, 12);
    expect(n.t).toBeCloseTo(0.25, 12);
  });

  it("local arc: projection lands on the circle", () => {
    const f = dShapeFixture(1);
    const n = nearestOnSegment(f.document, f.segments.arc, [0, 0.5], "local");
    expect(Math.hypot(n.pos[0], n.pos[1])).toBeCloseTo(1, 9);
    expect(n.pos[1]).toBeGreaterThan(0);
    expect(n.t).toBeGreaterThanOrEqual(0);
    expect(n.t).toBeLessThanOrEqual(1);
  });

  it("global space respects the panel transform", () => {
    const f = rectFixture();
    const moved = movePanel(f.document, f.panelId, [5, 0]);
    const n = nearestOnSegment(moved, f.segments.bottom, [5.2, 1], "global");
    expect(n.pos[1]).toBeCloseTo(0, 12);
    expect(n.distance).toBeCloseTo(1, 12);
  });
});

describe("intersectSegments", () => {
  it("line x line in local space", () => {
    const f = rectFixture();
    // Two construction lines forming a controlled X.
    const a = constructionLine(f.document, f.panelId, [0.1, -0.1], [0.3, 0.5]);
    const hitsParallel = intersectSegments(a.document, f.segments.bottom, f.segments.top, "local");
    expect(hitsParallel).toEqual([]); // bottom/top are parallel & disjoint

    const hitsX = intersectSegments(a.document, a.segmentId, a.otherSegmentId, "local");
    expect(hitsX.length).toBe(1);
    expect(hitsX[0][0]).toBeCloseTo(0.2, 9);
    expect(hitsX[0][1]).toBeCloseTo(0.2, 9);
  });

  it("line x arc: diameter chord crosses the arc twice", () => {
    const f = dShapeFixture(1);
    const chord = constructionLine(f.document, f.panelId, [-2, 0], [2, 0]);
    const hits = intersectSegments(chord.document, f.segments.arc, chord.segmentId, "local");
    expect(hits.length).toBe(2);
    expect(hits.map((h) => Math.hypot(h[0], h[1])).every((r) => Math.abs(r - 1) < 1e-9)).toBe(true);
  });

  it("same segment and cross-panel guards", () => {
    const f = rectFixture();
    expect(() => intersectSegments(f.document, f.segments.bottom, f.segments.bottom)).toThrowError(/itself/);

    const other = rectFixture(W, H, [1, 0], "other-doc");
    const merged: PatternDocument = {
      ...f.document,
      panels: [...f.document.panels, ...other.document.panels],
      points: [...f.document.points, ...other.document.points],
      segments: [...f.document.segments, ...other.document.segments],
    };
    // Cross-panel intersections need global space.
    expect(() => intersectSegments(merged, f.segments.bottom, other.segments.bottom, "local")).toThrowError(
      /global space/,
    );
    const globalHits = intersectSegments(merged, f.segments.bottom, other.segments.bottom, "global");
    expect(globalHits).toEqual([]); // parallel lines 1m apart
  });
});

function constructionLine(
  doc: PatternDocument,
  panelId: string,
  a: [number, number],
  b: [number, number],
): { document: PatternDocument; segmentId: string; otherSegmentId: string } {
  const p1 = createPoint(doc, panelId, a, "construction");
  const p2 = createPoint(p1.document, panelId, b, "construction");
  const seg = createConstructionLine(p2.document, panelId, p1.pointId, p2.pointId);
  // A second line crossing the first through its midpoint (different slope).
  const mid: [number, number] = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  const q1 = createPoint(seg.document, panelId, [mid[0] - 0.1, mid[1] + 0.3], "construction");
  const q2 = createPoint(q1.document, panelId, [mid[0] + 0.1, mid[1] - 0.3], "construction");
  const seg2 = createConstructionLine(q2.document, panelId, q1.pointId, q2.pointId);
  return { document: seg2.document, segmentId: seg.segmentId, otherSegmentId: seg2.segmentId };
}

describe("measurements", () => {
  it("rectangle: exact perimeter, area, winding, bbox", () => {
    const f = rectFixture();
    const m = measureLoop(f.document, f.panelId, f.loopId);
    expect(m.perimeter).toBeCloseTo(2 * (W + H), 12);
    expect(m.area).toBeCloseTo(W * H, 12);
    expect(m.winding).toBe("ccw");
    expect(m.bbox!.max[0]).toBeCloseTo(W, 12);
    expect(m.bbox!.max[1]).toBeCloseTo(H, 12);
  });

  it("D-shape: exact area/perimeter including the circular segment", () => {
    const f = dShapeFixture(1);
    const m = measureLoop(f.document, f.panelId, f.loopId);
    // Diameter + semicircle: area = pi r^2 / 2, perimeter = 2r + pi r.
    expect(m.area).toBeCloseTo(Math.PI / 2, 9);
    expect(m.perimeter).toBeCloseTo(2 + Math.PI, 9);
    expect(m.winding).toBe("ccw");
  });

  it("hole subtraction yields net area", () => {
    const f = rectWithHoleFixture();
    const m = measurePanel(f.document, f.panelId);
    expect(Math.abs(m.area)).toBeCloseTo(W * H, 12);
    expect(m.holeCount).toBe(1);
    expect(m.netArea).toBeCloseTo(W * H - 0.2 * 0.2, 12);
  });

  it("sampleLoopLocal is an open polyline (no duplicated join)", () => {
    const f = rectFixture();
    const pts = sampleLoopLocal(f.document, f.panelId, f.loopId);
    expect(pts.length).toBe(4);
    expect(pts[0]).toEqual([0, 0]);
  });
});

describe("point references", () => {
  it("finds segment, center, dimension and constraint references", () => {
    const f = rectFixture();
    const dim = createDistanceDimension(f.document, f.panelId, f.points.bl, f.points.br, "width");
    const refs = pointReferences(dim.document, f.points.bl);
    expect(refs.segments).toContain(f.segments.bottom);
    expect(refs.segments).toContain(f.segments.left);
    expect(refs.dimensions).toEqual([dim.dimensionId]);
    expect(pointIsReferenced(dim.document, f.points.bl)).toBe(true);

    const d = dShapeFixture(1);
    const centerRefs = pointReferences(d.document, d.points.center);
    expect(centerRefs.segments).toEqual([d.segments.arc]); // center reference
    expect(pointIsReferenced(d.document, d.points.center)).toBe(true);
    expect(pointIsReferenced(f.document, f.points.tl)).toBe(true); // segment refs only
  });
});

describe("hit-test + box-select", () => {
  it("vertex beats segment beats panel, deterministically", () => {
    const f = rectFixture();
    const nearCorner = hitTest(f.document, [-0.001, -0.001], 0.05);
    expect(nearCorner[0].entityId).toBe(f.points.bl);
    expect(nearCorner[0].kind).toBe("point");
    const kinds = nearCorner.map((h) => h.kind);
    expect(kinds.indexOf("point")).toBeLessThanOrEqual(kinds.lastIndexOf("segment"));

    const midEdge = hitTest(f.document, [W / 2, 0.002], 0.02);
    expect(midEdge[0].kind).toBe("segment");
    expect(midEdge[0].entityId).toBe(f.segments.bottom);

    const inside = hitTest(f.document, [W / 2, H / 2], 0.01);
    expect(inside.some((h) => h.kind === "panel" && h.entityId === f.panelId)).toBe(true);
  });

  it("hit-test never hits beyond tolerance and repeats identically", () => {
    const f = rectFixture();
    expect(hitTest(f.document, [10, 10], 0.05)).toEqual([]);
    const a = hitTest(f.document, [0.05, 0.01], 0.05);
    const b = hitTest(f.document, [0.05, 0.01], 0.05);
    expect(JSON.stringify(a)).toBe(JSON.stringify(b));
  });

  it("nearestHit returns the top-ranked hit or null", () => {
    const f = rectFixture();
    expect(nearestHit(f.document, [0, 0], 0.01)!.entityId).toBe(f.points.bl);
    expect(nearestHit(f.document, [5, 5], 0.01)).toBeNull();
  });

  it("pointInPanel respects holes; global variant respects transform", () => {
    const f = rectWithHoleFixture();
    // Panel region outside the hole (hole spans 0.1..0.3 x 0.05..0.25).
    const inPanel: [number, number] = [0.05, 0.02];
    const inHole: [number, number] = [0.2, 0.15];
    expect(pointInPanel(f.document, f.panelId, inPanel)).toBe(true);
    expect(pointInPanel(f.document, f.panelId, inHole)).toBe(false); // inside hole
    expect(pointInPanelGlobal(f.document, getPanel(f.document, f.panelId), inPanel)).toBe(true);
    expect(pointInPanelGlobal(f.document, getPanel(f.document, f.panelId), inHole)).toBe(false);

    const moved = movePanel(f.document, f.panelId, [2, 0]);
    // The old position is now outside the moved panel...
    expect(pointInPanelGlobal(moved, getPanel(moved, f.panelId), inPanel)).toBe(false);
    // ...while the same local spot at the new offset is inside the panel
    // (and outside the hole).
    expect(pointInPanelGlobal(moved, getPanel(moved, f.panelId), [2 + inPanel[0], inPanel[1]])).toBe(true);
  });

  it("boxSelect picks points and segments overlapping the box", () => {
    const f = rectFixture();
    const inBox = boxSelect(f.document, [-0.01, -0.01], [0.05, 0.05]);
    expect(inBox).toContain(f.points.bl);
    expect(inBox).not.toContain(f.points.tr);
    expect(inBox).toContain(f.segments.bottom);
    expect(inBox).not.toContain(f.segments.top);
    expect(inBox).toContain(f.panelId); // panel overlaps the box

    const far = boxSelect(f.document, [5, 5], [6, 6]);
    expect(far).toEqual([]);
  });
});

describe("segmentParam", () => {
  it("is unclamped for lines and wraps for arcs", () => {
    const f = rectFixture();
    const r = resolveSegment(f.document, f.segments.bottom);
    expect(segmentParam(r, [W * 0.25, 0])).toBeCloseTo(0.25, 12);
    expect(segmentParam(r, [-W, 0])).toBeCloseTo(-1, 12);
    expect(segmentParam(r, [W * 2, 0])).toBeCloseTo(2, 12);

    const d = dShapeFixture(1);
    const a = resolveSegment(d.document, d.segments.arc);
    const t0 = segmentParam(a, [1, 0]);
    expect(t0).toBeCloseTo(0, 9);
    const t1 = segmentParam(a, [-1, 0]);
    expect(Math.min(Math.abs(t1 - 0), Math.abs(t1 - 1))).toBeLessThan(1e-9);
    const mid = segmentParam(a, [0, 1]);
    expect(mid).toBeCloseTo(0.5, 9);
  });
});

describe("dist sanity", () => {
  it("dist matches measurement expectations", () => {
    expect(dist([0, 0], [3, 4])).toBeCloseTo(5, 12);
  });
});
