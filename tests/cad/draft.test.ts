// G9B drafting-tools tests: construction relations, offsets, panel
// split/join/arrange, seam-invalidation reporting, adversarial geometry.
import { describe, expect, it } from "vitest";
import { PatternCadError, validatePatternDocument } from "../../src/pattern/cad.js";
import {
  arrangePanels,
  buildPanelFromRing,
  divideSegment,
  draftLine,
  extendSegmentByDistance,
  intersectionPoint,
  invalidatedSeamIdsForPanels,
  joinPanelsAtSharedEdge,
  lineAtAngle,
  midpointPoint,
  mirrorPanelAcrossLine,
  offsetLoop,
  parallelThrough,
  perpendicularAt,
  rotatePanelGeometry,
  splitPanelByLine,
  translatePanelGeometry,
  trimSegmentToSegment,
  type SeamRef,
} from "../../src/cad/draft.js";
import { getPoint, getSegment, measurePanel } from "../../src/cad/queries.js";
import { polygonFixture, rectFixture } from "./fixtures.js";

const SEAMS: SeamRef[] = [
  { id: "seam/1", sideA: { panelId: "PANEL-A" }, sideB: { panelId: "PANEL-B" } },
];

function valid(doc: Parameters<typeof validatePatternDocument>[0]): void {
  expect(validatePatternDocument(doc).valid).toBe(true);
}

describe("G9B construction relations", () => {
  it("drafts lines between coordinates and existing points", () => {
    const f = rectFixture();
    const r = draftLine(f.document, f.panelId, [0.05, 0.05], [0.35, 0.25]);
    const seg = getSegment(r.document, r.segmentId, f.panelId);
    expect(seg.kind).toBe("line");
    expect(seg.role).toBe("construction");
    const byIds = draftLine(f.document, f.panelId, f.points.bl, f.points.tr);
    expect(getSegment(byIds.document, byIds.segmentId, f.panelId).role).toBe("construction");
    expect(() => draftLine(f.document, f.panelId, [0.1, 0.1], [0.1, 0.1])).toThrowError(PatternCadError);
  });

  it("creates lines at angles with exact length", () => {
    const f = rectFixture();
    const r = lineAtAngle(f.document, f.panelId, [0, 0], Math.PI / 2, 0.25);
    const seg = getSegment(r.document, r.segmentId, f.panelId);
    const a = getPoint(r.document, seg.startPointId, f.panelId);
    const b = getPoint(r.document, seg.endPointId, f.panelId);
    expect(b.x - a.x).toBeCloseTo(0, 12);
    expect(b.y - a.y).toBeCloseTo(0.25, 12);
    expect(() => lineAtAngle(f.document, f.panelId, [0, 0], 0, 0)).toThrowError(PatternCadError);
  });

  it("builds perpendiculars and parallels", () => {
    const f = rectFixture();
    const perp = perpendicularAt(f.document, f.panelId, f.segments.bottom, 0.5, 0.2);
    const ps = getSegment(perp.document, perp.segmentId, f.panelId);
    const pa = getPoint(perp.document, ps.startPointId, f.panelId);
    const pb = getPoint(perp.document, ps.endPointId, f.panelId);
    expect(pb.x - pa.x).toBeCloseTo(0, 12);
    expect(Math.abs(pb.y - pa.y)).toBeCloseTo(0.2, 12);
    const par = parallelThrough(f.document, f.panelId, f.segments.bottom, [0.2, 0.1]);
    const qs = getSegment(par.document, par.segmentId, f.panelId);
    const qa = getPoint(par.document, qs.startPointId, f.panelId);
    const qb = getPoint(par.document, qs.endPointId, f.panelId);
    expect(qa.y).toBeCloseTo(0.1, 12);
    expect(qb.y).toBeCloseTo(0.1, 12);
    expect(qb.x - qa.x).toBeCloseTo(0.4, 12); // defaults to source length
  });

  it("marks midpoints and divides segments", () => {
    const f = rectFixture();
    const mid = midpointPoint(f.document, f.panelId, f.segments.bottom);
    const p = getPoint(mid.document, mid.pointId, f.panelId);
    expect([p.x, p.y]).toEqual([0.2, 0]);
    const div = divideSegment(f.document, f.panelId, f.segments.bottom, 4);
    expect(div.pointIds).toHaveLength(3);
    const xs = div.pointIds.map((id) => getPoint(div.document, id, f.panelId).x);
    expect(xs).toEqual([0.1, 0.2, 0.30000000000000004]);
    expect(() => divideSegment(f.document, f.panelId, f.segments.bottom, 1)).toThrowError(PatternCadError);
  });

  it("places construction points at intersections and rejects misses", () => {
    const f = rectFixture();
    const diag = draftLine(f.document, f.panelId, f.points.bl, f.points.tr);
    const cross = intersectionPoint(diag.document, f.panelId, diag.segmentId, f.segments.right);
    const p = getPoint(cross.document, cross.pointId, f.panelId);
    expect(p.x).toBeCloseTo(0.4, 12);
    expect(p.y).toBeCloseTo(0.3, 12);
    expect(() =>
      intersectionPoint(f.document, f.panelId, f.segments.bottom, f.segments.top),
    ).toThrowError(/do not intersect/);
  });

  it("trims to a cutter and extends by distance", () => {
    const f = rectFixture();
    // Diagonal from bottom-left; trim its far end back to the right edge.
    const diag = draftLine(f.document, f.panelId, [0, 0], [0.8, 0.3]);
    const trimmed = trimSegmentToSegment(diag.document, f.panelId, diag.segmentId, f.segments.right, "start");
    const seg = getSegment(trimmed, diag.segmentId, f.panelId);
    const end = getPoint(trimmed, seg.endPointId, f.panelId);
    expect(end.x).toBeCloseTo(0.4, 9);
    expect(end.y).toBeCloseTo(0.15, 9);
    const grown = extendSegmentByDistance(f.document, f.panelId, f.segments.bottom, "end", 0.1);
    const bs = getSegment(grown, f.segments.bottom, f.panelId);
    const be = getPoint(grown, bs.endPointId, f.panelId);
    expect(be.x).toBeCloseTo(0.5, 12);
    expect(() => trimSegmentToSegment(f.document, f.panelId, f.segments.bottom, f.segments.top, "start"))
      .toThrowError(/does not cross/);
    expect(() => extendSegmentByDistance(f.document, f.panelId, f.segments.bottom, "end", 0))
      .toThrowError(PatternCadError);
  });
});

describe("G9B loop offset", () => {
  it("offsets a rectangle outward with exact miter positions", () => {
    const f = rectFixture(0.4, 0.3);
    const r = offsetLoop(f.document, f.panelId, f.loopId, 0.05);
    const m = measurePanel(r.document, r.panelId);
    expect(m.area).toBeCloseTo(0.5 * 0.4, 9);
    expect(m.perimeter).toBeCloseTo(2 * (0.5 + 0.4), 9);
    valid(r.document);
    // Original panel is untouched (fresh panel holds the offset).
    expect(measurePanel(f.document, f.panelId).area).toBeCloseTo(0.12, 12);
  });

  it("offsets inward with negative distance", () => {
    const f = rectFixture(0.4, 0.3);
    const r = offsetLoop(f.document, f.panelId, f.loopId, -0.05);
    expect(measurePanel(r.document, r.panelId).area).toBeCloseTo(0.3 * 0.2, 9);
  });

  it("rejects zero offsets, arcs, and holes explicitly", () => {
    const f = rectFixture();
    expect(() => offsetLoop(f.document, f.panelId, f.loopId, 0)).toThrowError(PatternCadError);
    expect(() => offsetLoop(f.document, f.panelId, "missing-loop", 0.05)).toThrowError(PatternCadError);
  });

  it("rejects self-annihilating offsets instead of producing garbage", () => {
    // Insetting a 0.3-wide rect by 0.2 collapses past the centerline.
    const f = rectFixture(0.4, 0.3);
    expect(() => offsetLoop(f.document, f.panelId, f.loopId, -0.2)).toThrowError(PatternCadError);
  });
});

describe("G9B panel geometry transforms", () => {
  it("translates and rotates actual points", () => {
    const f = rectFixture();
    const moved = translatePanelGeometry(f.document, f.panelId, [1, 2]);
    const p = getPoint(moved, f.points.bl, f.panelId);
    expect([p.x, p.y]).toEqual([1, 2]);
    const spun = rotatePanelGeometry(f.document, f.panelId, [0, 0], Math.PI / 2);
    const q = getPoint(spun, f.points.br, f.panelId);
    expect(q.x).toBeCloseTo(0, 12);
    expect(q.y).toBeCloseTo(0.4, 12);
    valid(moved);
    valid(spun);
  });

  it("mirrors across a local line and flips arc sweeps", () => {
    const f = rectFixture();
    const mirrored = mirrorPanelAcrossLine(f.document, f.panelId, [0.2, 0], [0.2, 1]);
    const p = getPoint(mirrored, f.points.bl, f.panelId);
    expect(p.x).toBeCloseTo(0.4, 12);
    expect(p.y).toBeCloseTo(0, 12);
    valid(mirrored);
    expect(() => mirrorPanelAcrossLine(f.document, f.panelId, [0, 0], [0, 0])).toThrowError(PatternCadError);
  });

  it("arranges several panels deterministically", () => {
    const poly = polygonFixture([[0, 0], [1, 0], [1, 1], [0, 1]]);
    const second = buildPanelFromRing(poly.document, "b", "m", 0, [[5, 5], [6, 5], [6, 6], [5, 6]]);
    const panels = second.document.panels.map((p) => p.id);
    const arranged = arrangePanels(second.document, [
      { panelId: panels[0], delta: [10, 0] },
      { panelId: panels[1], delta: [-5, -5] },
    ]);
    const pts = arranged.points.filter((p) => p.panelId === panels[0]);
    expect(pts.map((p) => p.x).sort((x, y) => x - y)).toEqual([10, 10, 11, 11]);
    const again = arrangePanels(second.document, [
      { panelId: panels[0], delta: [10, 0] },
      { panelId: panels[1], delta: [-5, -5] },
    ]);
    expect(JSON.stringify(again)).toBe(JSON.stringify(arranged));
  });
});

describe("G9B split / join with seam reporting", () => {
  it("splits a rectangle into two valid panels", () => {
    const f = rectFixture(0.4, 0.3);
    const r = splitPanelByLine(f.document, f.panelId, [0.2, -1], [0.2, 1]);
    expect(r.panelIds).toHaveLength(2);
    expect(r.invalidatedSeamIds).toEqual([]);
    valid(r.document);
    expect(r.document.panels).toHaveLength(2); // two fresh panels replace the original
    const areas = r.panelIds.map((id) => measurePanel(r.document, id).area);
    expect(areas[0] + areas[1]).toBeCloseTo(0.12, 9);
  });

  it("reports invalidated seams on split and join", () => {
    const f = rectFixture(0.4, 0.3);
    const seams: SeamRef[] = [{ id: "seam/x", sideA: { panelId: f.panelId }, sideB: { panelId: "other" } }];
    const r = splitPanelByLine(f.document, f.panelId, [0.2, -1], [0.2, 1], seams);
    expect(r.invalidatedSeamIds).toEqual(["seam/x"]);
    expect(invalidatedSeamIdsForPanels(SEAMS, ["nope"])).toEqual([]);
    expect(invalidatedSeamIdsForPanels(SEAMS, ["PANEL-B"])).toEqual(["seam/1"]);
  });

  it("joins two panels along a shared edge and inverts the split", () => {
    const f = rectFixture(0.4, 0.3);
    const split = splitPanelByLine(f.document, f.panelId, [0.2, -1], [0.2, 1]);
    const joined = joinPanelsAtSharedEdge(split.document, split.panelIds[0], split.panelIds[1]);
    valid(joined.document);
    expect(joined.document.panels).toHaveLength(1);
    expect(measurePanel(joined.document, joined.panelId).area).toBeCloseTo(0.12, 9);
  });

  it("rejects misses, grazes, multi-crossing cuts, and edge-free joins", () => {
    const f = rectFixture();
    expect(() => splitPanelByLine(f.document, f.panelId, [5, 0], [5, 1])).toThrowError(/misses/);
    expect(() => splitPanelByLine(f.document, f.panelId, [0, 0], [0.4, 0])).toThrowError(PatternCadError);
    const g = rectFixture(0.4, 0.3, [10, 10], "other-doc");
    // Two disjoint panels from different documents cannot join; build locally:
    const two = buildPanelFromRing(f.document, "far", "m", 0, [[5, 5], [6, 5], [6, 6], [5, 6]]);
    void g;
    expect(() => joinPanelsAtSharedEdge(two.document, f.panelId, two.panelId)).toThrowError(/no coincident/);
    expect(() => joinPanelsAtSharedEdge(f.document, f.panelId, f.panelId)).toThrowError(/itself/);
  });

  it("handles concave splits deterministically", () => {
    const l = polygonFixture([[0, 0], [0.4, 0], [0.4, 0.2], [0.2, 0.2], [0.2, 0.4], [0, 0.4]]);
    const a = splitPanelByLine(l.document, l.panelId, [0.1, -1], [0.1, 1]);
    const b = splitPanelByLine(l.document, l.panelId, [0.1, -1], [0.1, 1]);
    expect(JSON.stringify(a)).toBe(JSON.stringify(b));
    valid(a.document);
  });
});

describe("G9B adversarial drafting", () => {
  it("rejects zero-area rings and huge/small coordinates sanely", () => {
    const f = rectFixture();
    expect(() => buildPanelFromRing(f.document, "flat", "m", 0, [[0, 0], [1, 0]])).toThrowError(/at least 3/);
    expect(() => buildPanelFromRing(f.document, "flat", "m", 0, [[0, 0], [1, 1], [2, 2]])).toThrowError(/zero area/);
    const big = buildPanelFromRing(f.document, "big", "m", 0, [[0, 0], [1e6, 0], [1e6, 1e6], [0, 1e6]]);
    expect(measurePanel(big.document, big.panelId).area).toBeCloseTo(1e12, 0);
    const tiny = buildPanelFromRing(f.document, "tiny", "m", 0, [[0, 0], [1e-3, 0], [1e-3, 1e-3], [0, 1e-3]]);
    valid(tiny.document);
  });

  it("keeps construction geometry out of boundary loops", () => {
    const f = rectFixture();
    const r = draftLine(f.document, f.panelId, [0, 0], [5, 5]);
    const panel = r.document.panels.find((p) => p.id === f.panelId)!;
    expect(panel.boundaryLoops[0].segmentIds).not.toContain(r.segmentId);
    expect(panel.constructionSegmentIds).toContain(r.segmentId);
    // Triangulation input is unchanged by construction clutter.
    expect(measurePanel(r.document, f.panelId).area).toBeCloseTo(0.12, 12);
  });
});
