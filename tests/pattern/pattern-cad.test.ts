import { describe, expect, it } from "vitest";
import {
  createBoundaryArc,
  createBoundaryLine,
  createBoundaryLoop,
  createCoincidentConstraint,
  createConstructionLine,
  createDistanceDimension,
  createPanel,
  createPatternDocument,
  createPoint,
  deletePanel,
  deserializePatternDocument,
  duplicatePanel,
  globalToLocal,
  insertBoundaryPoint,
  localToGlobal,
  measureDimension,
  measureDistance,
  movePanel,
  movePoint,
  reverseBoundaryOrientation,
  rotatePanel,
  scalePanel,
  serializePatternDocument,
  splitBoundarySegment,
  transformPanelAroundPivot,
  triangulateCadPanel,
  validatePatternDocument,
  type EntityId,
  type PatternDocument,
} from "../../src/pattern/cad.js";

function rectangle(id = "rect", x0 = 0, y0 = 0, width = 1, height = 1) {
  let document = createPatternDocument(id, "Rectangle");
  const panelResult = createPanel(document, "Front", "cotton"); document = panelResult.document;
  const panelId = panelResult.panelId;
  const loopResult = createBoundaryLoop(document, panelId, "outer"); document = loopResult.document;
  const loopId = loopResult.loopId;
  const points: EntityId[] = [];
  for (const p of [[x0, y0], [x0 + width, y0], [x0 + width, y0 + height], [x0, y0 + height]] as [number, number][]) {
    const created = createPoint(document, panelId, p);
    document = created.document;
    points.push(created.pointId);
  }
  const segments: EntityId[] = [];
  for (let i = 0; i < points.length; i++) {
    const created = createBoundaryLine(document, panelId, loopId, points[i], points[(i + 1) % points.length]);
    document = created.document;
    segments.push(created.segmentId);
  }
  return { document, panelId, loopId, points, segments };
}

function issueCodes(document: PatternDocument): string[] {
  return validatePatternDocument(document).diagnostics.map((diagnostic) => diagnostic.code);
}

describe("G8A editable CAD entities and transforms", () => {
  it("allocates stable IDs and creates separate boundary and construction geometry", () => {
    const rect = rectangle("stable");
    let document = rect.document;
    const a = createPoint(document, rect.panelId, [0.2, 0.2], "construction"); document = a.document;
    const b = createPoint(document, rect.panelId, [0.8, 0.8], "construction"); document = b.document;
    const line = createConstructionLine(document, rect.panelId, a.pointId, b.pointId); document = line.document;
    expect(document.points.find((p) => p.id === a.pointId)?.role).toBe("construction");
    expect(document.segments.find((s) => s.id === line.segmentId)?.role).toBe("construction");
    expect(document.panels[0].constructionSegmentIds).toEqual([line.segmentId]);
    expect(validatePatternDocument(document).valid).toBe(true);
    expect(triangulateCadPanel(document, rect.panelId).triangles).toHaveLength(6);
    expect(rect.document.segments).toHaveLength(4); // pure operations do not mutate prior states
    expect(rect.points[0]).toMatch(/stable\/point\/\d{8}/);
  });

  it("moves points and panels while converting local and global coordinates", () => {
    const rect = rectangle();
    let document = movePanel(rect.document, rect.panelId, [3, -2]);
    const panel = document.panels[0];
    expect(localToGlobal(panel, [0.25, 0.75])).toEqual([3.25, -1.25]);
    expect(globalToLocal(panel, [3.25, -1.25])).toEqual([0.25, 0.75]);
    document = movePoint(document, rect.panelId, rect.points[0], [4, -2], "global");
    expect(document.points.find((p) => p.id === rect.points[0])).toMatchObject({ x: 1, y: 0 });
    expect(validatePatternDocument(document).valid).toBe(false); // only the corner moved; no silent repair
  });

  it("rotates around a pivot and supports explicit combined pivot transforms", () => {
    const rect = rectangle();
    const pivot: [number, number] = [0, 0];
    const rotated = rotatePanel(rect.document, rect.panelId, Math.PI / 2, pivot);
    const panel = rotated.panels[0];
    const corner = localToGlobal(panel, [1, 0]);
    expect(corner[0]).toBeCloseTo(0, 12);
    expect(corner[1]).toBeCloseTo(1, 12);
    const moved = transformPanelAroundPivot(rotated, rect.panelId, pivot, [2, 3], -Math.PI / 2);
    expect(localToGlobal(moved.panels[0], [1, 0])).toEqual([3, 3]);
  });

  it("scales in layout space or changes pattern size only with explicit intent", () => {
    const rect = rectangle();
    expect(() => scalePanel(rect.document, rect.panelId, 2, 2, undefined as never)).toThrow(/explicit intent/);
    expect(() => scalePanel(rect.document, rect.panelId, -1, 1, { intent: "layout-only" })).toThrow(/positive finite/);
    const layout = scalePanel(rect.document, rect.panelId, 2, 3, { intent: "layout-only", pivotGlobal: [0, 0] });
    expect(localToGlobal(layout.panels[0], [1, 1])).toEqual([2, 3]);
    const dim = createDistanceDimension(rect.document, rect.panelId, rect.points[0], rect.points[1]);
    const sized = scalePanel(dim.document, rect.panelId, 2, 1, { intent: "pattern-size-adjustment" });
    expect(measureDimension(sized, rect.panelId, dim.dimensionId)).toBeCloseTo(2, 12);
    expect(triangulateCadPanel(sized, rect.panelId).vertices[2]).toBe(2);
  });

  it("measures dimensions and stores explicit geometric constraints", () => {
    const rect = rectangle("measure", 0, 0, 3, 4);
    const dimension = createDistanceDimension(rect.document, rect.panelId, rect.points[0], rect.points[2], "diagonal");
    const constraint = createCoincidentConstraint(dimension.document, rect.panelId, rect.points[0], rect.points[0]);
    expect(measureDistance(constraint.document, rect.panelId, rect.points[0], rect.points[2])).toBe(5);
    expect(measureDimension(constraint.document, rect.panelId, dimension.dimensionId)).toBe(5);
    expect(constraint.document.panels[0].constraints[0].kind).toBe("coincident");
  });
});

describe("G8A boundary editing", () => {
  it("inserts a boundary point by splitting a line and preserves the loop", () => {
    const rect = rectangle();
    const inserted = insertBoundaryPoint(rect.document, rect.panelId, rect.loopId, rect.segments[0], [0.5, 0]);
    expect(inserted.document.panels[0].boundaryLoops[0].segmentIds).toEqual([
      ...inserted.segmentIds,
      rect.segments[1], rect.segments[2], rect.segments[3],
    ]);
    expect(validatePatternDocument(inserted.document).valid).toBe(true);
    expect(triangulateCadPanel(inserted.document, rect.panelId).vertices.length / 2).toBe(4); // collinear CAD point is not a mesh vertex
    expect(() => insertBoundaryPoint(rect.document, rect.panelId, rect.loopId, rect.segments[0], [0.5, 0.1])).toThrow(/must lie/);
  });

  it("splits a line at a deterministic parameter and reverses the authored winding", () => {
    const rect = rectangle();
    const split = splitBoundarySegment(rect.document, rect.panelId, rect.loopId, rect.segments[0], 0.25);
    expect(split.document.points.find((p) => p.id === split.pointId)).toMatchObject({ x: 0.25, y: 0 });
    expect(validatePatternDocument(split.document).valid).toBe(true);
    const reversed = reverseBoundaryOrientation(rect.document, rect.panelId, rect.loopId);
    expect(reversed.panels[0].boundaryLoops[0].orientation).toBe("cw");
    expect(reversed.panels[0].boundaryLoops[0].segmentIds).toEqual([...rect.segments].reverse());
    expect(validatePatternDocument(reversed).valid).toBe(true);
    expect(triangulateCadPanel(reversed, rect.panelId).triangles).toHaveLength(6);
  });

  it("duplicates panels with remapped entity references and deletes without damaging the source", () => {
    const rect = rectangle("duplicate");
    const dimension = createDistanceDimension(rect.document, rect.panelId, rect.points[0], rect.points[1]);
    const constructionPoint = createPoint(dimension.document, rect.panelId, [0.2, 0.3], "construction");
    const constructionPointB = createPoint(constructionPoint.document, rect.panelId, [0.7, 0.3], "construction");
    const construction = createConstructionLine(constructionPointB.document, rect.panelId, constructionPoint.pointId, constructionPointB.pointId);
    const duplicate = duplicatePanel(construction.document, rect.panelId, [3, 0]);
    const copy = duplicate.document.panels[1];
    expect(copy.id).not.toBe(rect.panelId);
    expect(copy.boundaryLoops[0].segmentIds.every((id) => !rect.segments.includes(id))).toBe(true);
    expect(validatePatternDocument(duplicate.document).valid).toBe(true);
    expect(triangulateCadPanel(duplicate.document, copy.id).triangles).toHaveLength(6);
    const deleted = deletePanel(duplicate.document, copy.id);
    expect(deleted.panels).toHaveLength(1);
    expect(validatePatternDocument(deleted).valid).toBe(true);
    expect(construction.document.panels).toHaveLength(1);
  });

  it("preserves a circular arc as CAD geometry and approximates only for triangulation", () => {
    let document = createPatternDocument("arc-panel");
    const panel = createPanel(document, "D-shape"); document = panel.document;
    const panelId = panel.panelId;
    const loop = createBoundaryLoop(document, panelId, "outer"); document = loop.document;
    const loopId = loop.loopId;
    const p0 = createPoint(document, panelId, [1, 0]); document = p0.document;
    const p1 = createPoint(document, panelId, [-1, 0]); document = p1.document;
    const center = createPoint(document, panelId, [0, 0], "construction"); document = center.document;
    const arc = createBoundaryArc(document, panelId, loopId, p0.pointId, p1.pointId, center.pointId, Math.PI); document = arc.document;
    const line = createBoundaryLine(document, panelId, loopId, p1.pointId, p0.pointId); document = line.document;
    expect(document.segments.find((segment) => segment.id === arc.segmentId)?.kind).toBe("arc");
    expect(validatePatternDocument(document).valid).toBe(true);
    const triangulated = triangulateCadPanel(document, panelId, { sagittaTol: 0.002 });
    expect(triangulated.vertices.length / 2).toBeGreaterThan(4);
    expect(document.segments).toHaveLength(2); // sampled vertices are derived, not authored
    expect(triangulated.triangles.length).toBeGreaterThan(0);
  });
});

describe("G8A validation and deterministic persistence", () => {
  it("returns structured diagnostics for open loops, zero edges, repeated points, and invalid winding", () => {
    const rect = rectangle("validation");
    const open = structuredClone(rect.document);
    open.panels[0].boundaryLoops[0].segmentIds.pop();
    expect(issueCodes(open)).toContain("open-boundary");

    const zero = structuredClone(rect.document);
    const edge = zero.segments.find((segment) => segment.id === rect.segments[0])!;
    edge.endPointId = edge.startPointId;
    expect(issueCodes(zero)).toContain("zero-length-edge");

    const repeated = structuredClone(rect.document);
    const second = repeated.segments.find((segment) => segment.id === rect.segments[1])!;
    second.startPointId = rect.points[1];
    const duplicatePoint = createPoint(repeated, rect.panelId, [1, 0]);
    const last = duplicatePoint.document.segments.find((segment) => segment.id === rect.segments[1])!;
    last.startPointId = duplicatePoint.pointId;
    expect(issueCodes(duplicatePoint.document)).toContain("duplicate-consecutive-points");

    const winding = structuredClone(rect.document);
    winding.panels[0].boundaryLoops[0].orientation = "cw";
    expect(issueCodes(winding)).toContain("invalid-winding");
  });

  it("reports bow-tie self-intersection and degenerate area instead of repairing either", () => {
    let bow = createPatternDocument("self-crossing");
    const panel = createPanel(bow, "Star"); bow = panel.document;
    const loop = createBoundaryLoop(bow, panel.panelId, "outer"); bow = loop.document;
    const pentagon = Array.from({ length: 5 }, (_, i) => {
      const angle = -Math.PI / 2 + i * (2 * Math.PI / 5);
      return [Math.cos(angle), Math.sin(angle)] as [number, number];
    });
    const pts: EntityId[] = [];
    for (const position of [0, 2, 4, 1, 3].map((index) => pentagon[index])) {
      const point = createPoint(bow, panel.panelId, position); bow = point.document; pts.push(point.pointId);
    }
    for (let i = 0; i < pts.length; i++) {
      bow = createBoundaryLine(bow, panel.panelId, loop.loopId, pts[i], pts[(i + 1) % pts.length]).document;
    }
    expect(issueCodes(bow)).toContain("self-intersection");
    expect(() => triangulateCadPanel(bow, panel.panelId)).toThrow();

    const degenerate = rectangle("tiny-invalid", 0, 0, 1e-7, 1e-7);
    expect(issueCodes(degenerate.document)).toContain("degenerate-panel");
  });

  it("rejects duplicate IDs and broken entity references", () => {
    const rect = rectangle("ids");
    const invalid = structuredClone(rect.document);
    invalid.points[1].id = invalid.points[0].id;
    invalid.segments[0].endPointId = "missing-point";
    expect(issueCodes(invalid)).toContain("duplicate-id");
    expect(issueCodes(invalid)).toContain("missing-reference");
  });

  it("accepts small, large, narrow, concave, collinear, and reversed-winding valid patterns", () => {
    expect(validatePatternDocument(rectangle("small", 0, 0, 1e-4, 1e-4).document).valid).toBe(true);
    expect(validatePatternDocument(rectangle("large", 0, 0, 1e6, 1e6).document).valid).toBe(true);
    expect(validatePatternDocument(rectangle("narrow", 0, 0, 1e-5, 10).document).valid).toBe(true);
    const reversed = rectangle("reversed");
    expect(validatePatternDocument(reverseBoundaryOrientation(reversed.document, reversed.panelId, reversed.loopId)).valid).toBe(true);
    const collinear = rectangle("collinear");
    const inserted = insertBoundaryPoint(collinear.document, collinear.panelId, collinear.loopId, collinear.segments[0], [0.5, 0]);
    expect(validatePatternDocument(inserted.document).valid).toBe(true);
    expect(triangulateCadPanel(inserted.document, collinear.panelId).vertices.length / 2).toBe(4);

    let concave = createPatternDocument("concave");
    const panel = createPanel(concave, "L panel"); concave = panel.document;
    const loop = createBoundaryLoop(concave, panel.panelId, "outer"); concave = loop.document;
    const coords: [number, number][] = [[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]];
    const pointIds: string[] = [];
    for (const position of coords) {
      const point = createPoint(concave, panel.panelId, position); concave = point.document; pointIds.push(point.pointId);
    }
    for (let i = 0; i < pointIds.length; i++) {
      concave = createBoundaryLine(concave, panel.panelId, loop.loopId, pointIds[i], pointIds[(i + 1) % pointIds.length]).document;
    }
    expect(validatePatternDocument(concave).valid).toBe(true);
    expect(triangulateCadPanel(concave, panel.panelId).triangles).toHaveLength(12);

    const nearCoincident = rectangle("near-coincident", 0, 0, 5e-10, 1);
    expect(issueCodes(nearCoincident.document)).toContain("zero-length-edge");
  });

  it("reports malformed circular arcs and preserves authored entities without repair", () => {
    let document = createPatternDocument("invalid-arc");
    const panel = createPanel(document, "bad arc"); document = panel.document;
    const loop = createBoundaryLoop(document, panel.panelId, "outer"); document = loop.document;
    const start = createPoint(document, panel.panelId, [1, 0]); document = start.document;
    const end = createPoint(document, panel.panelId, [0, 2]); document = end.document;
    const center = createPoint(document, panel.panelId, [0, 0], "construction"); document = center.document;
    const arc = createBoundaryArc(document, panel.panelId, loop.loopId, start.pointId, end.pointId, center.pointId, Math.PI / 2); document = arc.document;
    const close = createBoundaryLine(document, panel.panelId, loop.loopId, end.pointId, start.pointId); document = close.document;
    expect(issueCodes(document)).toContain("invalid-arc");
    expect(document.segments.find((item) => item.id === arc.segmentId)).toMatchObject({ kind: "arc", sweepRad: Math.PI / 2 });
    expect(close.segmentId).toBeTruthy();
  });

  it("splits an arc at a deterministic parameter and inserts points on arcs", () => {
    let document = createPatternDocument("arc-split");
    const panel = createPanel(document, "D-shape"); document = panel.document;
    const panelId = panel.panelId;
    const loop = createBoundaryLoop(document, panelId, "outer"); document = loop.document;
    const loopId = loop.loopId;
    const p0 = createPoint(document, panelId, [1, 0]); document = p0.document;
    const p1 = createPoint(document, panelId, [-1, 0]); document = p1.document;
    const center = createPoint(document, panelId, [0, 0], "construction"); document = center.document;
    const arc = createBoundaryArc(document, panelId, loopId, p0.pointId, p1.pointId, center.pointId, Math.PI); document = arc.document;
    const line = createBoundaryLine(document, panelId, loopId, p1.pointId, p0.pointId); document = line.document;

    // Split at t=0.25: midpoint of the quarter sweep lands at 45 degrees.
    const split = splitBoundarySegment(document, panelId, loopId, arc.segmentId, 0.25);
    const mid = split.document.points.find((p) => p.id === split.pointId)!;
    expect(mid.x).toBeCloseTo(Math.cos(Math.PI / 4), 12);
    expect(mid.y).toBeCloseTo(Math.sin(Math.PI / 4), 12);
    const children = split.segmentIds.map((id) => split.document.segments.find((s) => s.id === id)!);
    expect(children.map((s) => s.kind)).toEqual(["arc", "arc"]);
    expect((children[0] as { sweepRad: number }).sweepRad).toBeCloseTo(Math.PI * 0.25, 12);
    expect((children[1] as { sweepRad: number }).sweepRad).toBeCloseTo(Math.PI * 0.75, 12);
    expect(validatePatternDocument(split.document).valid).toBe(true);
    // NOTE: curved panels need an explicit coarse sagittaTol; dense default
    // sampling (~350-gon here) stalls the G7D ear-clipper (pre-existing
    // triangulator limit, also true for the unsplit D-shape). Same applies below.
    expect(triangulateCadPanel(split.document, panelId, { sagittaTol: 0.002 }).triangles.length).toBeGreaterThan(0);

    // Insert a point lying on the arc (top of the semicircle).
    const inserted = insertBoundaryPoint(document, panelId, loopId, arc.segmentId, [0, 1]);
    const insertedPoint = inserted.document.points.find((p) => p.id === inserted.pointId)!;
    expect(insertedPoint.x).toBeCloseTo(0, 12);
    expect(insertedPoint.y).toBeCloseTo(1, 12);
    expect(validatePatternDocument(inserted.document).valid).toBe(true);

    // Off-arc positions and arc endpoints are rejected without mutation.
    expect(() => insertBoundaryPoint(document, panelId, loopId, arc.segmentId, [0, 0.5])).toThrow(/must lie on the boundary arc/);
    expect(() => insertBoundaryPoint(document, panelId, loopId, arc.segmentId, [1, 0])).toThrow(/strictly inside/);
    expect(() => splitBoundarySegment(document, panelId, loopId, arc.segmentId, 0)).toThrow(/strictly inside/);
    expect(() => splitBoundarySegment(document, panelId, loopId, arc.segmentId, 1)).toThrow(/strictly inside/);

    // Deterministic: same split twice yields identical documents.
    const again = splitBoundarySegment(document, panelId, loopId, arc.segmentId, 0.25);
    expect(serializePatternDocument(again.document)).toBe(serializePatternDocument(split.document));
    expect(document.segments).toHaveLength(2); // pure operations do not mutate prior states
  });

  it("uses deterministic IDs, entity ordering, and canonical serialization across reconstruction", () => {
    const first = rectangle("same-document");
    const second = rectangle("same-document");
    expect(first.document).toEqual(second.document);
    expect(serializePatternDocument(first.document)).toBe(serializePatternDocument(second.document));
    const restored = deserializePatternDocument(serializePatternDocument(first.document));
    expect(serializePatternDocument(restored)).toBe(serializePatternDocument(first.document));
    expect(Array.from(triangulateCadPanel(restored, first.panelId).triangles)).toEqual(
      Array.from(triangulateCadPanel(first.document, first.panelId).triangles),
    );
    const withCounter = structuredClone(first.document);
    withCounter.nextEntityIndex = 0;
    expect(issueCodes(withCounter)).toContain("invalid-document");
  });
});
