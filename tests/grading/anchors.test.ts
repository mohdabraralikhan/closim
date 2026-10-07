import { describe, expect, it } from "vitest";
import { resolveAnchor, evaluateSegmentPoint, GradingError } from "../../src/grading/index.js";
import { buildArcPanelDocument, buildGradingFixture, buildTwoPanelDocument, SEAM_ID } from "./fixtures.js";

describe("G12A anchors: point and corner", () => {
  it("resolves a point anchor to its vertex and position", () => {
    const fixture = buildTwoPanelDocument();
    const r = resolveAnchor(fixture.document, { kind: "point", panelId: fixture.ids.frontPanel, pointId: fixture.ids.points.C });
    expect(r.displacement).toBe("vertex");
    expect(r.pointIds).toEqual([fixture.ids.points.C]);
    expect(r.position).toEqual([0.4, 0.6]);
  });

  it("resolves a corner anchor to the shared endpoint of two segments", () => {
    const fixture = buildTwoPanelDocument();
    const r = resolveAnchor(fixture.document, {
      kind: "corner", panelId: fixture.ids.frontPanel, segmentIdA: fixture.ids.segments.ab, segmentIdB: fixture.ids.segments.bc,
    });
    expect(r.pointIds).toEqual([fixture.ids.points.B]);
    expect(r.position).toEqual([0.4, 0]);
  });

  it("rejects corners whose segments do not share exactly one endpoint", () => {
    const fixture = buildTwoPanelDocument();
    expect(() => resolveAnchor(fixture.document, {
      kind: "corner", panelId: fixture.ids.frontPanel, segmentIdA: fixture.ids.segments.ab, segmentIdB: fixture.ids.segments.cd,
    })).toThrowError(GradingError);
  });

  it("rejects points that belong to another panel", () => {
    const fixture = buildTwoPanelDocument();
    expect(() => resolveAnchor(fixture.document, { kind: "point", panelId: fixture.ids.backPanel, pointId: fixture.ids.points.A }))
      .toThrowError(/does not belong/);
  });
});

describe("G12A anchors: edge-relative", () => {
  it("maps t=0 and t=1 to the segment endpoints as vertex anchors", () => {
    const fixture = buildTwoPanelDocument();
    const start = resolveAnchor(fixture.document, { kind: "edge-relative", panelId: fixture.ids.frontPanel, segmentId: fixture.ids.segments.bc, t: 0 });
    const end = resolveAnchor(fixture.document, { kind: "edge-relative", panelId: fixture.ids.frontPanel, segmentId: fixture.ids.segments.bc, t: 1 });
    expect(start.pointIds).toEqual([fixture.ids.points.B]);
    expect(end.pointIds).toEqual([fixture.ids.points.C]);
  });

  it("evaluates intermediate positions on straight edges without vertex displacement", () => {
    const fixture = buildTwoPanelDocument();
    const r = resolveAnchor(fixture.document, { kind: "edge-relative", panelId: fixture.ids.frontPanel, segmentId: fixture.ids.segments.cd, t: 0.5 });
    expect(r.displacement).toBe("evaluated");
    expect(r.pointIds).toEqual([]);
    expect(r.position[0]).toBeCloseTo(0.2, 12);
    expect(r.position[1]).toBeCloseTo(0.6, 12);
  });

  it("evaluates intermediate positions on curved (arc) edges", () => {
    const arc = buildArcPanelDocument();
    const mid = evaluateSegmentPoint(arc.document, arc.ids.segments.arc, 0.5);
    expect(mid[0]).toBeCloseTo(0.8, 12);
    expect(mid[1]).toBeCloseTo(0.4, 12);
    const r = resolveAnchor(arc.document, { kind: "edge-relative", panelId: arc.ids.panel, segmentId: arc.ids.segments.arc, t: 0.25 });
    expect(r.displacement).toBe("evaluated");
    expect(r.position[0]).toBeCloseTo(0.4 + 0.4 * Math.cos(-Math.PI / 4), 12);
    expect(r.position[1]).toBeCloseTo(0.4 + 0.4 * Math.sin(-Math.PI / 4), 12);
  });

  it("rejects out-of-range parameters and unknown entities", () => {
    const fixture = buildTwoPanelDocument();
    expect(() => resolveAnchor(fixture.document, { kind: "edge-relative", panelId: fixture.ids.frontPanel, segmentId: fixture.ids.segments.bc, t: 1.5 }))
      .toThrowError(GradingError);
    expect(() => resolveAnchor(fixture.document, { kind: "edge-relative", panelId: fixture.ids.frontPanel, segmentId: "doc/g12fixture/segment/999", t: 0.5 }))
      .toThrowError(/does not exist/);
  });
});

describe("G12A anchors: seam-related points", () => {
  it("resolves through the seam context to the underlying pattern point", () => {
    const fixture = buildGradingFixture();
    const r = resolveAnchor(fixture.doc.master.document, {
      kind: "seam-point", seamId: SEAM_ID, side: "b", segmentId: fixture.ids.segments.he, endpoint: "start",
    }, fixture.doc.seams);
    expect(r.pointIds).toEqual([fixture.ids.points.H]);
    expect(r.position).toEqual([0.4, 0.6]);
  });

  it("rejects unknown seams and segments outside the seam side", () => {
    const fixture = buildGradingFixture();
    expect(() => resolveAnchor(fixture.doc.master.document, {
      kind: "seam-point", seamId: "seam/none", side: "a", segmentId: fixture.ids.segments.bc, endpoint: "start",
    }, fixture.doc.seams)).toThrowError(/seam 'seam\/none' does not exist/);
    expect(() => resolveAnchor(fixture.doc.master.document, {
      kind: "seam-point", seamId: SEAM_ID, side: "a", segmentId: fixture.ids.segments.gh, endpoint: "start",
    }, fixture.doc.seams)).toThrowError(/not part of seam/);
  });
});
