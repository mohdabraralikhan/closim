import { describe, expect, it } from "vitest";
import {
  createBoundaryLine,
  createBoundaryLoop,
  createPanel,
  createPatternDocument,
  createPoint,
} from "../../src/pattern/cad.js";
import {
  buildAssemblyGraph,
  resolveStitchPairs,
  validateSeams,
  type Seam,
  type SeamSide,
} from "../../src/garment/sewing.js";

function makeRect(document: ReturnType<typeof createPatternDocument>, name: string, width: number, height: number) {
  const p = createPanel(document, name, "cotton");
  document = p.document;
  const l = createBoundaryLoop(document, p.panelId, "outer");
  document = l.document;
  const points: string[] = [];
  for (const xy of [[0, 0], [width, 0], [width, height], [0, height]] as [number, number][]) {
    const result = createPoint(document, p.panelId, xy);
    document = result.document;
    points.push(result.pointId);
  }
  const segments: string[] = [];
  for (let i = 0; i < 4; i++) {
    const result = createBoundaryLine(document, p.panelId, l.loopId, points[i], points[(i + 1) % 4]);
    document = result.document;
    segments.push(result.segmentId);
  }
  return { document, panelId: p.panelId, loopId: l.loopId, segments };
}

function twoPanels(widthA = 1, widthB = 1) {
  let document = createPatternDocument("sewing", "Two panel test");
  const a = makeRect(document, "A", widthA, 1); document = a.document;
  const b = makeRect(document, "B", widthB, 1); document = b.document;
  const side = (panelId: string, loopId: string, segmentIds: string[], reversed = false): SeamSide => ({ panelId, loopId, segmentIds, reversed });
  const seam: Seam = {
    id: "seam:side",
    sideA: side(a.panelId, a.loopId, [a.segments[0]]),
    sideB: side(b.panelId, b.loopId, [b.segments[2]], true),
    stitchCount: 5,
    groupId: "side-seams",
  };
  return { document, a, b, seam };
}

describe("G8B sewing and construction", () => {
  it("resolves equal-length, reversed two-panel seams to deterministic stitch pairs", () => {
    const { document, seam } = twoPanels();
    expect(validateSeams(document, [seam]).valid).toBe(true);
    const pairs = resolveStitchPairs(document, seam);
    expect(pairs).toHaveLength(5);
    expect(pairs.map((pair) => pair.index)).toEqual([0, 1, 2, 3, 4]);
    expect(pairs.map((pair) => pair.tA)).toEqual([0, 0.25, 0.5, 0.75, 1]);
    expect(pairs[0].pointA).toEqual([0, 0]);
    expect(pairs[0].pointB).toEqual([0, 1]);
    expect(resolveStitchPairs(document, seam)).toEqual(pairs);
  });

  it("resamples unequal seam lengths independently while preserving normalized correspondence", () => {
    const { document, a, b, seam } = twoPanels(1, 2);
    seam.sideB.segmentIds = [b.segments[2]];
    const pairs = resolveStitchPairs(document, seam);
    expect(pairs).toHaveLength(5);
    expect(pairs[2].pointA[0]).toBeCloseTo(0.5);
    expect(pairs[2].pointB[0]).toBeCloseTo(1);
    expect(pairs[2].pointB[1]).toBeCloseTo(1);
    expect(a.panelId).not.toBe(b.panelId);
  });

  it("accepts ordered segmented seams and reverses the resampled path without changing CAD IDs", () => {
    const { document, a, seam } = twoPanels();
    seam.sideA.segmentIds = [a.segments[0], a.segments[1]];
    seam.sideA.reversed = true;
    expect(validateSeams(document, [seam]).valid).toBe(true);
    const pairs = resolveStitchPairs(document, seam);
    expect(pairs[0].pointA).toEqual([1, 1]);
    expect(pairs.at(-1)?.pointA).toEqual([0, 0]);
    expect(seam.sideA.segmentIds).toEqual([a.segments[0], a.segments[1]]);
  });

  it("builds a stable panel-boundary-seam-stitch assembly graph", () => {
    const { document, seam, a, b } = twoPanels();
    const graph = buildAssemblyGraph(document, [seam]);
    expect(graph.panelIds).toEqual([a.panelId, b.panelId]);
    expect(graph.seams).toHaveLength(1);
    expect(graph.seams[0]).toMatchObject({ seamId: seam.id, panelA: a.panelId, panelB: b.panelId });
    expect(graph.seams[0].stitchPairs).toEqual(resolveStitchPairs(document, seam));
  });

  it("reports missing panels, loops, segments, and non-contiguous references", () => {
    const { document, seam } = twoPanels();
    const badPanel: Seam = { ...seam, id: "bad-panel", sideA: { ...seam.sideA, panelId: "missing-panel" } };
    const badLoop: Seam = { ...seam, id: "bad-loop", sideB: { ...seam.sideB, loopId: "missing-loop" } };
    const badSegment: Seam = { ...seam, id: "bad-segment", sideB: { ...seam.sideB, segmentIds: ["missing-segment"] } };
    const codes = validateSeams(document, [badPanel, badLoop, badSegment]).diagnostics.map((diagnostic) => diagnostic.code);
    expect(codes).toContain("missing-panel");
    expect(codes).toContain("missing-loop");
    expect(codes).toContain("missing-segment");
    expect(codes).toContain("invalid-chain");
  });

  it("rejects duplicate seam definitions and invalid stitch counts", () => {
    const { document, seam } = twoPanels();
    const duplicate = { ...seam, id: "seam:duplicate" };
    expect(validateSeams(document, [seam, duplicate]).diagnostics.map((d) => d.code)).toContain("duplicate-seam");
    expect(validateSeams(document, [{ ...seam, stitchCount: 1 }]).diagnostics.map((d) => d.code)).toContain("invalid-stitch-count");
  });
});
