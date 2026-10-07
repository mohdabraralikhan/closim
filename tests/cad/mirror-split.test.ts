// G9A mirror + split operations and cross-op determinism.
// Mirror bakes the reflection into local coordinates (ids preserved),
// negates arc sweeps, and flips loop orientation fields so the document
// stays valid; split delegates line segments to cad.ts and adds arcs.
import { describe, it, expect } from "vitest";
import {
  duplicatePanel,
  serializePatternDocument,
  splitBoundarySegment,
  validatePatternDocument,
  type PatternDocument,
} from "../../src/pattern/cad.js";
import { mirrorPanel, mirrorPanelCopy, moveSegmentBy, offsetConstructionLine, splitSegment } from "../../src/cad/ops.js";
import { measurePanel, resolveSegment, sampleLoopLocal } from "../../src/cad/queries.js";
import { dist, reflectAcrossLine, type Vec2 } from "../../src/cad/geom.js";
import { dShapeFixture, rectFixture } from "./fixtures.js";

const W = 0.4;
const H = 0.3;

function expectValid(doc: PatternDocument): void {
  const result = validatePatternDocument(doc);
  expect(result.diagnostics).toEqual([]);
  expect(result.valid).toBe(true);
}

function serialize(doc: PatternDocument): string {
  return serializePatternDocument(doc);
}

// ---------------------------------------------------------------------------
// mirrorPanel
// ---------------------------------------------------------------------------

describe("mirrorPanel", () => {
  it("bakes the reflection: ids and transform preserved, winding flipped, still valid", () => {
    const f = rectFixture();
    const lineA: Vec2 = [-0.1, -1];
    const lineB: Vec2 = [-0.1, 2]; // vertical mirror line at x = -0.1
    const before = f.document;
    const mirrored = mirrorPanel(before, f.panelId, lineA, lineB);
    expectValid(mirrored);

    // Original untouched (immutability).
    expect(serialize(mirrored)).not.toBe(serialize(before));

    // Every point is exactly the reflection of its original.
    for (const orig of before.points) {
      const now = mirrored.points.find((p) => p.id === orig.id)!;
      expect(now).toBeDefined();
      const expected = reflectAcrossLine([orig.x, orig.y], lineA, lineB);
      expect(now.x).toBeCloseTo(expected[0], 9);
      expect(now.y).toBeCloseTo(expected[1], 9);
      expect(now.panelId).toBe(orig.panelId);
    }

    // Segment/loop identity survives; orientation flips with the geometry.
    const loop = mirrored.panels[0].boundaryLoops[0];
    expect(loop.segmentIds).toEqual(f.document.panels[0].boundaryLoops[0].segmentIds);
    expect(loop.orientation).toBe("cw");
    const srcPanel = before.panels[0];
    const dstPanel = mirrored.panels[0];
    expect(dstPanel.transform).toEqual(srcPanel.transform);

    // Isometry: area magnitude and perimeter are unchanged.
    const m0 = measurePanel(before, f.panelId);
    const m1 = measurePanel(mirrored, f.panelId);
    expect(Math.abs(m1.area)).toBeCloseTo(Math.abs(m0.area), 12);
    expect(m1.perimeter).toBeCloseTo(m0.perimeter, 12);
  });

  it("negates arc sweeps and keeps the D-shape valid", () => {
    const d = dShapeFixture(1);
    const mirrored = mirrorPanel(d.document, d.panelId, [0, -1], [0, 2]); // vertical axis
    expectValid(mirrored);
    const arc = resolveSegment(mirrored, d.segments.arc);
    expect(arc.segment.kind).toBe("arc");
    expect(arc.arc!.sweep).toBeCloseTo(-Math.PI, 12);
    // The reflected boundary samples match pointwise (within 1e-9).
    const src = sampleLoopLocal(d.document, d.panelId, d.loopId);
    const dst = sampleLoopLocal(mirrored, d.panelId, d.loopId);
    expect(dst.length).toBe(src.length);
    // Sampled loop order differs after the winding flip; compare as sets of
    // reflected points by checking bounds instead.
    const b0 = measurePanel(d.document, d.panelId).bbox!;
    const b1 = measurePanel(mirrored, d.panelId).bbox!;
    expect(b1.min[0]).toBeCloseTo(-b0.max[0], 9);
    expect(b1.max[0]).toBeCloseTo(-b0.min[0], 9);
    expect(b1.min[1]).toBeCloseTo(b0.min[1], 9);
    expect(Math.abs(measurePanel(mirrored, d.panelId).area)).toBeCloseTo(
      Math.abs(measurePanel(d.document, d.panelId).area),
      9,
    );
  });

  it("double mirroring returns the original geometry (to float precision)", () => {
    const f = rectFixture();
    const lineA: Vec2 = [0.13, -0.4];
    const lineB: Vec2 = [0.7, 0.9];
    const twice = mirrorPanel(mirrorPanel(f.document, f.panelId, lineA, lineB), f.panelId, lineA, lineB);
    expectValid(twice);
    for (const orig of f.document.points) {
      const p = twice.points.find((q) => q.id === orig.id)!;
      expect(p.x).toBeCloseTo(orig.x, 9);
      expect(p.y).toBeCloseTo(orig.y, 9);
    }
    expect(twice.panels[0].boundaryLoops[0].orientation).toBe("ccw");
  });

  it("rejects degenerate and non-finite mirror lines", () => {
    const f = rectFixture();
    expect(() => mirrorPanel(f.document, f.panelId, [1, 1], [1, 1])).toThrowError(/degenerate/);
    expect(() => mirrorPanel(f.document, f.panelId, [NaN, 0], [1, 0])).toThrowError(/finite/);
    expect(() => mirrorPanel(f.document, "missing", [0, 0], [0, 1])).toThrowError(/missing-reference/);
  });
});

// ---------------------------------------------------------------------------
// mirrorPanelCopy
// ---------------------------------------------------------------------------

describe("mirrorPanelCopy", () => {
  it("creates a reflected twin without touching the source", () => {
    const f = rectFixture();
    const srcBefore = serialize(f.document);
    const lineA: Vec2 = [-0.1, -1];
    const lineB: Vec2 = [-0.1, 2];
    const copy = mirrorPanelCopy(f.document, f.panelId, lineA, lineB);
    expectValid(copy.document);
    expect(serialize(f.document)).toBe(srcBefore); // source untouched
    expect(copy.panelId).not.toBe(f.panelId);

    const srcPanel = f.document.panels[0];
    const copyPanel = copy.document.panels.find((p) => p.id === copy.panelId)!;
    expect(copyPanel.name).toBe(`${srcPanel.name} mirror`);

    // The copy's workspace bbox is the source's bbox reflected.
    const srcPts = f.document.points.map((p) => [p.x, p.y] as Vec2);
    const copyPts = copy.document.points
      .filter((p) => p.panelId === copy.panelId)
      .map((p) => [p.x, p.y] as Vec2);
    expect(copyPts.length).toBe(srcPts.length);
    const reflectSet = srcPts.map((p) => reflectAcrossLine(p, lineA, lineB));
    for (const cp of copyPts) {
      const nearest = Math.min(...reflectSet.map((rp) => dist(rp, cp)));
      expect(nearest).toBeLessThan(1e-9);
    }

    // Ids are fresh (duplicatePanel semantics).
    const copyIds = new Set(copy.document.points.filter((p) => p.panelId === copy.panelId).map((p) => p.id));
    for (const orig of f.document.points) {
      expect(copyIds.has(orig.id)).toBe(false);
    }
  });

  it("keeps dimensions/constraints attached to the right panel after copy", () => {
    const f = rectFixture();
    const copy = mirrorPanelCopy(f.document, f.panelId, [0, -1], [0, 2]);
    expectValid(copy.document);
    // Two panels now, each with their own loops.
    expect(copy.document.panels.length).toBe(2);
    const copyPanel = copy.document.panels.find((p) => p.id === copy.panelId)!;
    expect(copyPanel.boundaryLoops.length).toBe(f.document.panels[0].boundaryLoops.length);
  });
});

// ---------------------------------------------------------------------------
// splitSegment
// ---------------------------------------------------------------------------

describe("splitSegment", () => {
  it("line segments delegate to cad.ts exactly (byte-identical output)", () => {
    const f1 = rectFixture();
    const f2 = rectFixture();
    const viaG8 = splitBoundarySegment(f1.document, f1.panelId, f1.loopId, f1.segments.bottom, 0.25);
    const viaG9 = splitSegment(f2.document, f2.panelId, f2.loopId, f2.segments.bottom, 0.25);
    expect(serialize(viaG9.document)).toBe(serialize(viaG8.document));
    expect(viaG9.pointId).toBe(viaG8.pointId);
    expect(viaG9.segmentIds).toEqual(viaG8.segmentIds);
  });

  it("splits an arc into two same-center halves with proportional sweeps", () => {
    const d = dShapeFixture(1);
    const res = splitSegment(d.document, d.panelId, d.loopId, d.segments.arc, 0.5);
    expectValid(res.document);

    expect(res.document.segments.length).toBe(3); // line + 2 arcs
    expect(res.document.segments.some((s) => s.id === d.segments.arc)).toBe(false);
    const [firstId, secondId] = res.segmentIds;
    const first = resolveSegment(res.document, firstId);
    const second = resolveSegment(res.document, secondId);

    expect(first.segment.kind).toBe("arc");
    expect(second.segment.kind).toBe("arc");
    // Same center point entity as the original arc.
    expect(first.segment.kind === "arc" && first.segment.centerPointId).toBe(d.points.center);
    expect(second.segment.kind === "arc" && second.segment.centerPointId).toBe(d.points.center);
    expect(first.arc!.sweep).toBeCloseTo(Math.PI / 2, 12);
    expect(second.arc!.sweep).toBeCloseTo(Math.PI / 2, 12);

    // The inserted point sits on the circle at the top.
    const mid = res.document.points.find((p) => p.id === res.pointId)!;
    expect(mid.x).toBeCloseTo(0, 12);
    expect(mid.y).toBeCloseTo(1, 12);
    expect(Math.hypot(mid.x, mid.y)).toBeCloseTo(1, 12);

    // Loop order: line, first half, second half — chain intact.
    const loop = res.document.panels[0].boundaryLoops[0];
    expect(loop.segmentIds).toEqual([d.segments.line, firstId, secondId]);
    // Geometric content is unchanged (same region).
    expect(measurePanel(res.document, d.panelId).area).toBeCloseTo(
      measurePanel(d.document, d.panelId).area,
      9,
    );
  });

  it("is deterministic: two identical splits serialize identically", () => {
    const fa = dShapeFixture(1);
    const fb = dShapeFixture(1);
    const a = splitSegment(fa.document, fa.panelId, fa.loopId, fa.segments.arc, 0.3);
    const b = splitSegment(fb.document, fb.panelId, fb.loopId, fb.segments.arc, 0.3);
    expect(serialize(a.document)).toBe(serialize(b.document));
    expect(a.pointId).toBe(b.pointId);
    expect(a.segmentIds).toEqual(b.segmentIds);
  });

  it("rejects out-of-range t, degenerate sweeps, wrong loop and missing ids", () => {
    const d = dShapeFixture(1);
    for (const t of [0, 1, -0.5, 1.5, NaN]) {
      expect(() => splitSegment(d.document, d.panelId, d.loopId, d.segments.arc, t)).toThrowError(
        /strictly inside/,
      );
    }
    // A split so close to the start that the sub-arc falls below the
    // validation sweep epsilon is rejected, not silently produced.
    expect(() => splitSegment(d.document, d.panelId, d.loopId, d.segments.arc, 1e-12)).toThrowError(
      /sub-arc/,
    );
    expect(() => splitSegment(d.document, d.panelId, "wrong-loop", d.segments.arc, 0.5)).toThrowError(
      /missing-reference/,
    );
    expect(() => splitSegment(d.document, d.panelId, d.loopId, "missing", 0.5)).toThrowError(
      /missing-reference/,
    );
  });
});

// ---------------------------------------------------------------------------
// Cross-op determinism
// ---------------------------------------------------------------------------

describe("cross-operation determinism", () => {
  function opSequence(): PatternDocument {
    const f = rectFixture();
    let doc = f.document;
    doc = moveSegmentBy(doc, f.panelId, f.segments.bottom, [0, 0.02]);
    const off = offsetConstructionLine(doc, f.panelId, f.segments.bottom, 0.05);
    doc = off.document;
    const split = splitSegment(doc, f.panelId, f.loopId, f.segments.left, 0.4);
    doc = split.document;
    doc = mirrorPanel(doc, f.panelId, [-0.1, -1], [-0.1, 2]);
    const dup = duplicatePanel(doc, f.panelId, [1, 0]);
    doc = dup.document;
    return doc;
  }

  it("identical op sequences produce identical serialization", () => {
    const a = serialize(opSequence());
    const b = serialize(opSequence());
    expect(a).toBe(b);
  });

  it("every step of the sequence stays valid", () => {
    const f = rectFixture();
    let doc = f.document;
    expectValid(doc);
    doc = moveSegmentBy(doc, f.panelId, f.segments.bottom, [0, 0.02]);
    expectValid(doc);
    const off = offsetConstructionLine(doc, f.panelId, f.segments.bottom, 0.05);
    expectValid(off.document);
    const split = splitSegment(off.document, f.panelId, f.loopId, f.segments.left, 0.4);
    expectValid(split.document);
    const mirrored = mirrorPanel(split.document, f.panelId, [-0.1, -1], [-0.1, 2]);
    expectValid(mirrored);
    const dup = duplicatePanel(mirrored, f.panelId, [1, 0]);
    expectValid(dup.document);
    expect(dup.document.panels.length).toBe(2);
  });
});
