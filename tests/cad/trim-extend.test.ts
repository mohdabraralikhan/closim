// G9A trim / extend / offset:
//   - trim moves the nearest endpoint to an intersection, chosen by the
//     caller's reference point when several exist;
//   - extend never trims (on-segment intersections are rejected);
//   - offset creates parallel construction geometry.
// Arc targets keep their radius: the endpoint slides on the circle and only
// the sweep changes.
import { describe, it, expect } from "vitest";
import {
  createConstructionLine,
  createPoint,
  validatePatternDocument,
  type PatternDocument,
} from "../../src/pattern/cad.js";
import {
  extendLineBy,
  extendLineTo,
  offsetConstructionLine,
  trimSegment,
} from "../../src/cad/ops.js";
import { measurePanel, resolveSegment } from "../../src/cad/queries.js";
import { dist } from "../../src/cad/geom.js";
import { circleFixture, dShapeFixture, rectFixture } from "./fixtures.js";

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

describe("trimSegment", () => {
  it("trims a line to its intersection with a cutter, pulling shared vertices", () => {
    const f = rectFixture();
    const cutter = addConstructionLine(f.document, f.panelId, [0.3, -0.1], [0.3, 0.4]);
    const trimmed = trimSegment(cutter.document, f.panelId, f.segments.bottom, cutter.segmentId, [0.35, 0]);
    expectValid(trimmed);
    // The bottom edge's END point moved onto the intersection; the shared
    // right-edge vertex followed (shared-vertex semantics).
    const bottom = resolveSegment(trimmed, f.segments.bottom);
    expect(bottom.end[0]).toBeCloseTo(0.3, 12);
    expect(bottom.end[1]).toBeCloseTo(0, 12);
    expect(bottom.start).toEqual([0, 0]);
    const right = resolveSegment(trimmed, f.segments.right);
    expect(right.start[0]).toBeCloseTo(0.3, 12);
    // Area: trapezoid (0,0) (0.3,0) (0.4,0.3) (0,0.3) = 0.105.
    expect(Math.abs(measurePanel(trimmed, f.panelId).area)).toBeCloseTo(0.105, 12);
  });

  it("reference point selects which intersection of an arc target is used", () => {
    const d = dShapeFixture(1);
    // Horizontal cutter at y = 0.5 crosses the upper semicircle twice.
    const cutter = addConstructionLine(d.document, d.panelId, [-2, 0.5], [2, 0.5]);

    // Reference near the LEFT crossing -> arc END (point a) moves there.
    const trimLeft = trimSegment(
      cutter.document, d.panelId, d.segments.arc, cutter.segmentId, [-1, 0.5],
    );
    expectValid(trimLeft);
    const arcLeft = resolveSegment(trimLeft, d.segments.arc);
    expect(arcLeft.end[0]).toBeCloseTo(-Math.sqrt(3) / 2, 9);
    expect(arcLeft.end[1]).toBeCloseTo(0.5, 9);
    expect(dist(arcLeft.end, arcLeft.arc!.center)).toBeCloseTo(1, 9); // still on circle
    expect(arcLeft.arc!.sweep).toBeGreaterThan(0);
    expect(arcLeft.arc!.sweep).toBeLessThan(Math.PI); // shrank

    // Reference near the RIGHT crossing -> arc START moves there instead.
    const trimRight = trimSegment(
      cutter.document, d.panelId, d.segments.arc, cutter.segmentId, [1, 0.5],
    );
    expectValid(trimRight);
    const arcRight = resolveSegment(trimRight, d.segments.arc);
    expect(arcRight.start[0]).toBeCloseTo(Math.sqrt(3) / 2, 9);
    expect(arcRight.start[1]).toBeCloseTo(0.5, 9);
    expect(dist(arcRight.start, arcRight.arc!.center)).toBeCloseTo(1, 9);
    expect(arcRight.arc!.sweep).toBeCloseTo(Math.PI - Math.PI / 6, 9); // 180 -> 150 degrees
  });

  it("trims construction geometry and rejects missing intersections", () => {
    const f = rectFixture();
    const target = addConstructionLine(f.document, f.panelId, [0, -0.2], [1, -0.2]);
    // Cutter at x = 0.7: the intersection is clearly nearer the END, so the
    // reference at x = 0.9 selects the end for the move.
    const cutter = addConstructionLine(target.document, f.panelId, [0.7, -0.3], [0.7, -0.1]);
    const trimmed = trimSegment(cutter.document, f.panelId, target.segmentId, cutter.segmentId, [0.9, -0.2]);
    expectValid(trimmed);
    const line = resolveSegment(trimmed, target.segmentId);
    expect(line.end[0]).toBeCloseTo(0.7, 12);
    expect(line.start[0]).toBeCloseTo(0, 12);

    // A cutter that never meets the target is reported, not ignored.
    const far = addConstructionLine(trimmed, f.panelId, [5, -0.3], [5, -0.1]);
    expect(() =>
      trimSegment(far.document, f.panelId, target.segmentId, far.segmentId, [0.9, -0.2]),
    ).toThrowError(/do not intersect/);
  });

  it("rejects self-trim, foreign cutters, and non-finite references", () => {
    const f = rectFixture();
    expect(() => trimSegment(f.document, f.panelId, f.segments.bottom, f.segments.bottom, [0, 0]))
      .toThrowError(/itself/);
    expect(() => trimSegment(f.document, f.panelId, f.segments.bottom, "missing", [0, 0]))
      .toThrowError(/missing-reference/);
    expect(() => trimSegment(f.document, f.panelId, f.segments.bottom, f.segments.top, [NaN, 0]))
      .toThrowError(/finite/);
  });
});

describe("extendLineBy", () => {
  it("lengthens the endpoint nearest the reference by the given distance", () => {
    const f = rectFixture();
    const ext = extendLineBy(f.document, f.panelId, f.segments.bottom, 0.1, [0.4, 0]);
    expectValid(ext);
    const bottom = resolveSegment(ext, f.segments.bottom);
    expect(bottom.end[0]).toBeCloseTo(W + 0.1, 12);
    expect(bottom.start).toEqual([0, 0]);
    // The shared bottom-right vertex moved: right edge starts further out.
    const right = resolveSegment(ext, f.segments.right);
    expect(right.start[0]).toBeCloseTo(W + 0.1, 12);

    // Reference near the START extends the other side.
    const extA = extendLineBy(f.document, f.panelId, f.segments.bottom, 0.1, [0, 0]);
    expectValid(extA);
    expect(resolveSegment(extA, f.segments.bottom).start[0]).toBeCloseTo(-0.1, 12);
  });

  it("rejects arcs, bad distances and missing segments", () => {
    const d = dShapeFixture(1);
    expect(() => extendLineBy(d.document, d.panelId, d.segments.arc, 0.1, [1, 0]))
      .toThrowError(/line segments only/);
    const f = rectFixture();
    expect(() => extendLineBy(f.document, f.panelId, f.segments.bottom, 0, [0, 0]))
      .toThrowError(/> 0/);
    expect(() => extendLineBy(f.document, f.panelId, f.segments.bottom, -1, [0, 0]))
      .toThrowError(/> 0/);
    expect(() => extendLineBy(f.document, f.panelId, "missing", 0.1, [0, 0]))
      .toThrowError(/missing-reference/);
  });
});

describe("extendLineTo", () => {
  it("extends to an intersection strictly beyond an endpoint", () => {
    const f = rectFixture();
    // Target sticks out past x = 0.5; cutter crosses at x = 0.6? Use a
    // short construction target inside the panel instead for clarity.
    const target = addConstructionLine(f.document, f.panelId, [0.1, -0.2], [0.3, -0.2]);
    const cutter = addConstructionLine(target.document, f.panelId, [0.6, -0.3], [0.6, -0.1]);
    const ext = extendLineTo(cutter.document, f.panelId, target.segmentId, cutter.segmentId, [1, -0.2]);
    expectValid(ext);
    expect(resolveSegment(ext, target.segmentId).end[0]).toBeCloseTo(0.6, 12);

    // Reference on the other side picks the other intersection when both
    // are beyond the endpoints (circle cutter around a short segment).
    const circle = circleFixture(0.1);
    const inner = addConstructionLine(circle.document, circle.panelId, [-0.05, 0], [0.05, 0]);
    const topArc = circle.segments.topHalf;
    const extEnd = extendLineTo(
      inner.document, circle.panelId, inner.segmentId, topArc, [1, 0],
    );
    // The top half alone only crosses at (+/-0.1, 0) — the +x crossing is
    // beyond the end at x = 0.05.
    expect(resolveSegment(extEnd, inner.segmentId).end[0]).toBeCloseTo(0.1, 9);

    const extStart = extendLineTo(
      inner.document, circle.panelId, inner.segmentId, topArc, [-1, 0],
    );
    expect(resolveSegment(extStart, inner.segmentId).start[0]).toBeCloseTo(-0.1, 9);
  });

  it("rejects intersections that lie ON the segment (extend must never trim)", () => {
    const f = rectFixture();
    const target = addConstructionLine(f.document, f.panelId, [0.1, -0.2], [0.5, -0.2]);
    const cutter = addConstructionLine(target.document, f.panelId, [0.3, -0.3], [0.3, -0.1]);
    expect(() =>
      extendLineTo(cutter.document, f.panelId, target.segmentId, cutter.segmentId, [1, -0.2]),
    ).toThrowError(/beyond its endpoints/);
  });

  it("rejects arc targets and missing entities", () => {
    const d = dShapeFixture(1);
    const cutter = addConstructionLine(d.document, d.panelId, [-2, 0.5], [2, 0.5]);
    expect(() =>
      extendLineTo(cutter.document, d.panelId, d.segments.arc, cutter.segmentId, [1, 0.5]),
    ).toThrowError(/line targets only/);
    expect(() =>
      extendLineTo(cutter.document, d.panelId, d.segments.line, "missing", [1, 0.5]),
    ).toThrowError(/missing-reference/);
  });
});

describe("offsetConstructionLine", () => {
  it("creates a parallel construction line at the signed distance", () => {
    const f = rectFixture();
    const off = offsetConstructionLine(f.document, f.panelId, f.segments.bottom, 0.05);
    expectValid(off.document);
    expect(off.pointIds.length).toBe(2);
    const line = resolveSegment(off.document, off.segmentId);
    // Left normal of (0,0)->(0.4,0) is +y, so +0.05 sits above the edge.
    expect(line.start[1]).toBeCloseTo(0.05, 12);
    expect(line.end[1]).toBeCloseTo(0.05, 12);
    expect(line.start[0]).toBeCloseTo(0, 12);
    expect(line.end[0]).toBeCloseTo(W, 12);
    const panel = off.document.panels.find((p) => p.id === f.panelId)!;
    expect(panel.constructionSegmentIds).toContain(off.segmentId);

    // Negative distance goes the other way.
    const offNeg = offsetConstructionLine(f.document, f.panelId, f.segments.bottom, -0.05);
    expect(resolveSegment(offNeg.document, offNeg.segmentId).start[1]).toBeCloseTo(-0.05, 12);
  });

  it("rejects arcs, zero/non-finite distances, missing segments", () => {
    const d = dShapeFixture(1);
    expect(() => offsetConstructionLine(d.document, d.panelId, d.segments.arc, 0.05))
      .toThrowError(/line segments only/);
    const f = rectFixture();
    expect(() => offsetConstructionLine(f.document, f.panelId, f.segments.bottom, 0))
      .toThrowError(/nonzero/);
    expect(() => offsetConstructionLine(f.document, f.panelId, f.segments.bottom, NaN))
      .toThrowError(/nonzero/);
    expect(() => offsetConstructionLine(f.document, f.panelId, "missing", 0.05))
      .toThrowError(/missing-reference/);
  });
});
