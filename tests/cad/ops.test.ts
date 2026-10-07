// G9A editing ops part 1: creation, move segment, delete point, merge.
// Every mutation asserts a still-valid document (validatePatternDocument)
// or an explicit PatternCadError — never silent damage.
import { describe, it, expect } from "vitest";
import * as cad from "../../src/pattern/cad.js";
import {
  createConstructionLine,
  createDistanceDimension,
  createPoint,
  validatePatternDocument,
  type PatternDocument,
} from "../../src/pattern/cad.js";
import {
  createConstructionPolyline,
  deletePoint,
  mergeSegments,
  moveSegmentBy,
} from "../../src/cad/ops.js";
import { measureLoop, measurePanel, resolveSegment } from "../../src/cad/queries.js";
import { dShapeFixture, polygonFixture, rectFixture } from "./fixtures.js";

const W = 0.4;
const H = 0.3;

function expectValid(doc: PatternDocument): void {
  const result = validatePatternDocument(doc);
  expect(result.diagnostics).toEqual([]);
  expect(result.valid).toBe(true);
}

function addConstructionLine(
  doc: PatternDocument,
  panelId: string,
  a: [number, number],
  b: [number, number],
): { document: PatternDocument; segmentId: string } {
  const p1 = createPoint(doc, panelId, a, "construction");
  const p2 = createPoint(p1.document, panelId, b, "construction");
  const seg = createConstructionLine(p2.document, panelId, p1.pointId, p2.pointId);
  return { document: seg.document, segmentId: seg.segmentId };
}

// ---------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------

describe("createConstructionPolyline", () => {
  it("creates a chained polyline and keeps the document valid", () => {
    const f = rectFixture();
    const res = createConstructionPolyline(f.document, f.panelId, [
      [0.05, -0.05], [0.2, -0.1], [0.35, -0.05],
    ]);
    expect(res.pointIds.length).toBe(3);
    expect(res.segmentIds.length).toBe(2);
    expectValid(res.document);
    expect(res.document.points.filter((p) => p.role === "construction").length).toBe(3);
  });

  it("rejects fewer than 2 positions and non-finite input", () => {
    const f = rectFixture();
    expect(() => createConstructionPolyline(f.document, f.panelId, [[0, 0]])).toThrowError(
      /at least 2 positions/,
    );
    expect(() =>
      createConstructionPolyline(f.document, f.panelId, [[0, 0], [NaN, 1]]),
    ).toThrowError(/not finite/);
  });
});

// ---------------------------------------------------------------------------
// Move segment
// ---------------------------------------------------------------------------

describe("moveSegmentBy", () => {
  it("translates both endpoints of a boundary segment (shared vertices move too)", () => {
    const f = rectFixture();
    const moved = moveSegmentBy(f.document, f.panelId, f.segments.bottom, [0, 0.05]);
    expectValid(moved);
    const bottom = resolveSegment(moved, f.segments.bottom);
    expect(bottom.start).toEqual([0, 0.05]);
    expect(bottom.end).toEqual([W, 0.05]);
    const left = resolveSegment(moved, f.segments.left);
    expect(left.end).toEqual([0, 0.05]); // shared vertex followed
    expect(Math.abs(measurePanel(moved, f.panelId).area)).toBeCloseTo(W * (H - 0.05), 12);
  });

  it("moves an arc's center with its endpoints", () => {
    const d = dShapeFixture(1);
    const moved = moveSegmentBy(d.document, d.panelId, d.segments.arc, [1, 1]);
    expectValid(moved);
    const arc = resolveSegment(moved, d.segments.arc);
    expect(arc.arc!.center).toEqual([1, 1]);
    expect(arc.arc!.radius).toBeCloseTo(1, 12);
    expect(Math.abs(arc.arc!.sweep)).toBeCloseTo(Math.PI, 12);
    expect(arc.start[0]).toBeCloseTo(2, 12); // B moved with the segment
    expect(arc.end[0]).toBeCloseTo(0, 12); // A moved too
  });

  it("rejects non-finite deltas and missing segments", () => {
    const f = rectFixture();
    expect(() => moveSegmentBy(f.document, f.panelId, f.segments.bottom, [NaN, 0])).toThrowError(
      /finite/,
    );
    expect(() => moveSegmentBy(f.document, f.panelId, "missing", [0, 1])).toThrowError(
      /missing-reference/,
    );
  });
});

// ---------------------------------------------------------------------------
// Delete point
// ---------------------------------------------------------------------------

describe("deletePoint", () => {
  it("deletes a construction point with its segments and orphaned endpoints", () => {
    const f = rectFixture();
    const poly = createConstructionPolyline(f.document, f.panelId, [
      [0.05, -0.05], [0.2, -0.1], [0.35, -0.05],
    ]);
    expect(poly.document.segments.filter((s) => s.role === "construction").length).toBe(2);

    const after = deletePoint(poly.document, f.panelId, poly.pointIds[1]); // middle
    expectValid(after);
    // Middle + both now-orphaned endpoints are gone (nothing referenced them).
    expect(after.segments.filter((s) => s.role === "construction").length).toBe(0);
    expect(after.points.filter((p) => p.role === "construction").length).toBe(0);
    expect(after.segments.filter((s) => s.role === "boundary").length).toBe(4);
  });

  it("keeps a polyline's far endpoint when only one segment dies", () => {
    const f = rectFixture();
    const poly = createConstructionPolyline(f.document, f.panelId, [
      [0.05, -0.05], [0.2, -0.1], [0.35, -0.05],
    ]);
    const after = deletePoint(poly.document, f.panelId, poly.pointIds[0]);
    expectValid(after);
    expect(after.segments.filter((s) => s.role === "construction").length).toBe(1);
    expect(after.points.some((p) => p.id === poly.pointIds[2])).toBe(true);
    expect(after.points.some((p) => p.id === poly.pointIds[1])).toBe(true);
    expect(after.points.some((p) => p.id === poly.pointIds[0])).toBe(false);
  });

  it("deletes a boundary vertex, merging its two edges into a triangle", () => {
    const f = rectFixture();
    const before = Math.abs(measurePanel(f.document, f.panelId).area);
    const after = deletePoint(f.document, f.panelId, f.points.br);
    expectValid(after);
    expect(after.points.length).toBe(3);
    expect(after.segments.length).toBe(3);
    expect(after.points.some((p) => p.id === f.points.br)).toBe(false);
    // The merged edge keeps the FIRST loop slot's id (bottom) and spans bl -> tr.
    const merged = resolveSegment(after, f.segments.bottom);
    expect(merged.start).toEqual([0, 0]);
    expect(merged.end).toEqual([W, H]);
    expect(after.segments.some((s) => s.id === f.segments.right)).toBe(false);
    expect(Math.abs(measurePanel(after, f.panelId).area)).toBeCloseTo((W * H) / 2, 12);
    expect(before).toBeCloseTo(W * H, 12);
    const loop = after.panels[0].boundaryLoops[0];
    expect(loop.segmentIds.length).toBe(3);
    expect(loop.segmentIds.includes(f.segments.right)).toBe(false);
  });

  it("merges two same-circle arcs when the vertex between them dies", () => {
    const f = arcPairPie();
    const after = deletePoint(f.document, f.panelId, f.points.b);
    expectValid(after);
    expect(after.segments.length).toBe(2);
    const merged = resolveSegment(after, f.segments.arc1);
    expect(merged.segment.kind).toBe("arc");
    expect(merged.arc!.sweep).toBeCloseTo(Math.PI / 2, 9);
    expect(after.points.some((p) => p.id === f.points.b)).toBe(false);
    // arc2's DISTINCT center point became unreferenced and was cleaned up.
    expect(after.points.some((p) => p.id === f.points.center2)).toBe(false);
    expect(after.points.some((p) => p.id === f.points.center1)).toBe(true);
    expect(measureLoop(after, f.panelId, f.loopId).area).toBeCloseTo(f.areaBefore, 9);
  });

  it("refuses points protected by dimensions or arc centers", () => {
    const f = rectFixture();
    const dim = createDistanceDimension(f.document, f.panelId, f.points.bl, f.points.br, "w");
    expect(() => deletePoint(dim.document, f.panelId, f.points.bl)).toThrowError(/dimension/);

    const d = dShapeFixture(1);
    expect(() => deletePoint(d.document, d.panelId, d.points.center)).toThrowError(/center of arc/);
  });

  it("refuses mixed line/arc junctions and collapsing loops", () => {
    // The D-shape has only TWO segments, so the loop-collapse guard fires
    // before primitive mixing is even considered.
    const d = dShapeFixture(1);
    expect(() => deletePoint(d.document, d.panelId, d.points.a)).toThrowError(/collapse/);
    expect(() => deletePoint(d.document, d.panelId, d.points.b)).toThrowError(/collapse/);

    // Mixed junction with a >= 3 segment loop: line + arc sharing ONE
    // endpoint cannot merge into a single primitive.
    const pie = arcPairPie();
    expect(() => deletePoint(pie.document, pie.panelId, pie.points.a)).toThrowError(
      /mixed|cannot merge/i,
    );

    const tri = polygonFixture([[0, 0], [1, 0], [0, 1]]);
    expect(() => deletePoint(tri.document, tri.panelId, tri.pointIds[0])).toThrowError(/collapse/);
  });

  it("rejects missing points", () => {
    const f = rectFixture();
    expect(() => deletePoint(f.document, f.panelId, "nope")).toThrowError(/missing-reference/);
    expect(() => deletePoint(f.document, "wrong-panel", f.points.bl)).toThrowError(/missing-reference/);
  });
});

// ---------------------------------------------------------------------------
// Merge
// ---------------------------------------------------------------------------

describe("mergeSegments", () => {
  it("merges two collinear construction lines into one", () => {
    const f = rectFixture();
    // Two construction lines sharing the SAME point entity (p2) — merging
    // is only defined for shared vertices, not coincident coordinates.
    const p1 = createPoint(f.document, f.panelId, [0, -0.1], "construction");
    const p2 = createPoint(p1.document, f.panelId, [0.5, -0.1], "construction");
    const p3 = createPoint(p2.document, f.panelId, [1, -0.1], "construction");
    const l1 = createConstructionLine(p3.document, f.panelId, p1.pointId, p2.pointId);
    const l2 = createConstructionLine(l1.document, f.panelId, p2.pointId, p3.pointId);
    const shared = p2.pointId;
    const merged = mergeSegments(l2.document, f.panelId, null, l1.segmentId, l2.segmentId);
    expectValid(merged);
    const line = resolveSegment(merged, l1.segmentId);
    expect(line.start).toEqual([0, -0.1]);
    expect(line.end).toEqual([1, -0.1]);
    expect(merged.segments.some((s) => s.id === l2.segmentId)).toBe(false);
    expect(merged.points.some((p) => p.id === shared)).toBe(false);
    // The construction registry lost the second id and kept the first.
    const panel = merged.panels.find((p) => p.id === f.panelId)!;
    expect(panel.constructionSegmentIds).toContain(l1.segmentId);
    expect(panel.constructionSegmentIds).not.toContain(l2.segmentId);
  });

  it("merges collinear boundary edges without changing the area", () => {
    // Pentagon whose bottom edge has a redundant collinear vertex at (0.2, 0).
    const p = polygonFixture([[0, 0], [0.2, 0], [0.4, 0], [0.4, 0.3], [0, 0.3]]);
    const before = Math.abs(measurePanel(p.document, p.panelId).area);
    const merged = mergeSegments(p.document, p.panelId, p.loopId, p.segmentIds[0], p.segmentIds[1]);
    expectValid(merged);
    expect(merged.segments.length).toBe(4);
    expect(merged.points.length).toBe(4);
    expect(Math.abs(measurePanel(merged, p.panelId).area)).toBeCloseTo(before, 12);
    const bottom = resolveSegment(merged, p.segmentIds[0]);
    expect(bottom.start).toEqual([0, 0]);
    expect(bottom.end).toEqual([0.4, 0]);
  });

  it("rejects non-collinear boundary lines (shape must not change silently)", () => {
    const f = rectFixture();
    expect(() =>
      mergeSegments(f.document, f.panelId, f.loopId, f.segments.bottom, f.segments.right),
    ).toThrowError(/not collinear/);
    // The document is untouched by the rejection.
    expect(f.document.segments.length).toBe(4);
  });

  it("merges two same-circle arcs and drops the orphaned center point", () => {
    const f = arcPairPie();
    const merged = mergeSegments(f.document, f.panelId, f.loopId, f.segments.arc1, f.segments.arc2);
    expectValid(merged);
    expect(merged.segments.length).toBe(2);
    const arc = resolveSegment(merged, f.segments.arc1);
    expect(arc.segment.kind).toBe("arc");
    expect(arc.arc!.sweep).toBeCloseTo(Math.PI / 2, 9);
    expect(arc.start).toEqual([1, 0]);
    expect(arc.end[1]).toBeCloseTo(1, 9);
    expect(merged.points.some((p) => p.id === f.points.center2)).toBe(false);
    expect(measureLoop(merged, f.panelId, f.loopId).area).toBeCloseTo(f.areaBefore, 9);
  });

  it("rejects opposite-sweep arcs, mixed primitives, and protected points", () => {
    const f = reverseArcPair();
    expect(() =>
      mergeSegments(f.document, f.panelId, f.loopId, f.segments.arc1, f.segments.arc2),
    ).toThrowError(/opposite directions/);

    // Mixed primitives sharing exactly ONE endpoint (the D-shape's line and
    // arc share both, so the pie fixture provides a single shared vertex).
    const pie = arcPairPie();
    expect(() =>
      mergeSegments(pie.document, pie.panelId, pie.loopId, pie.segments.chord, pie.segments.arc1),
    ).toThrowError(/mixed/i);

    const r = rectFixture();
    const dim = createDistanceDimension(r.document, r.panelId, r.points.br, r.points.tr, "side");
    expect(() =>
      mergeSegments(dim.document, r.panelId, r.loopId, r.segments.bottom, r.segments.right),
    ).toThrowError(/collinear|dimension/); // right: non-collinear OR protected, both report

    expect(() => mergeSegments(r.document, r.panelId, r.loopId, r.segments.bottom, r.segments.bottom)).toThrowError(
      /itself/,
    );
    expect(() => mergeSegments(r.document, r.panelId, r.loopId, r.segments.bottom, "missing")).toThrowError(
      /missing-reference/,
    );
  });

  it("refuses merges that would leave a line-only loop with fewer than 3 edges", () => {
    const tri = polygonFixture([[0, 0], [1, 0], [0, 1]]);
    expect(() =>
      mergeSegments(tri.document, tri.panelId, tri.loopId, tri.segmentIds[0], tri.segmentIds[1]),
    ).toThrowError(/collapse/);
  });
});

// ---------------------------------------------------------------------------
// Fixtures local to this file
// ---------------------------------------------------------------------------

interface PieFixture {
  document: PatternDocument;
  panelId: string;
  loopId: string;
  points: { a: string; b: string; c: string; center1: string; center2: string };
  segments: { arc1: string; arc2: string; chord: string };
  areaBefore: number;
}

/**
 * Pie slice: arc(A->B, +45 degrees) + arc(B->C, +45 degrees) on the unit
 * circle with DISTINCT center entities, closed by the chord C->A. CCW.
 */
function arcPairPie(): PieFixture {
  let document = cad.createPatternDocument("test-doc");
  const panel = cad.createPanel(document, "pie");
  document = panel.document;
  const loop = cad.createBoundaryLoop(document, panel.panelId, "outer", "ccw");
  document = loop.document;
  const mk = (pos: [number, number], role: "boundary" | "construction" = "boundary"): string => {
    const res = cad.createPoint(document, panel.panelId, pos, role);
    document = res.document;
    return res.pointId;
  };
  const a = mk([1, 0]);
  const b = mk([Math.SQRT1_2, Math.SQRT1_2]);
  const c = mk([0, 1]);
  const center1 = mk([0, 0]);
  const center2 = mk([0, 0]);
  const arc1 = cad.createBoundaryArc(
    document, panel.panelId, loop.loopId, a, b, center1, Math.PI / 4,
  );
  document = arc1.document;
  const arc2 = cad.createBoundaryArc(
    document, panel.panelId, loop.loopId, b, c, center2, Math.PI / 4,
  );
  document = arc2.document;
  const chord = cad.createBoundaryLine(document, panel.panelId, loop.loopId, c, a);
  document = chord.document;
  const areaBefore = measureLoop(document, panel.panelId, loop.loopId).area;
  return {
    document,
    panelId: panel.panelId,
    loopId: loop.loopId,
    points: { a, b, c, center1, center2 },
    segments: { arc1: arc1.segmentId, arc2: arc2.segmentId, chord: chord.segmentId },
    areaBefore,
  };
}

interface ReverseArcFixture {
  document: PatternDocument;
  panelId: string;
  loopId: string;
  segments: { arc1: string; arc2: string };
}

/**
 * Two arcs sharing a point with OPPOSITE sweeps (arc2 backtracks over arc1's
 * territory). The document itself is intentionally degenerate — this fixture
 * only exists to prove mergeSegments REJECTS it up front.
 */
function reverseArcPair(): ReverseArcFixture {
  let document = cad.createPatternDocument("test-doc");
  const panel = cad.createPanel(document, "rev-arcs");
  document = panel.document;
  const loop = cad.createBoundaryLoop(document, panel.panelId, "outer", "ccw");
  document = loop.document;
  const mk = (pos: [number, number]): string => {
    const res = cad.createPoint(document, panel.panelId, pos, "boundary");
    document = res.document;
    return res.pointId;
  };
  const a = mk([1, 0]);
  const b = mk([Math.SQRT1_2, Math.SQRT1_2]);
  const c = mk([-Math.SQRT1_2, Math.SQRT1_2]); // NOT on the arc(a->b) circle path
  const center = mk([0, 0]);
  const arc1 = cad.createBoundaryArc(
    document, panel.panelId, loop.loopId, a, b, center, Math.PI / 4,
  );
  document = arc1.document;
  // Opposite direction: negative sweep from b.
  const arc2 = cad.createBoundaryArc(
    document, panel.panelId, loop.loopId, b, c, center, -Math.PI / 4,
  );
  document = arc2.document;
  const chord = cad.createBoundaryLine(document, panel.panelId, loop.loopId, c, a);
  document = chord.document;
  void chord;
  return {
    document,
    panelId: panel.panelId,
    loopId: loop.loopId,
    segments: { arc1: arc1.segmentId, arc2: arc2.segmentId },
  };
}
