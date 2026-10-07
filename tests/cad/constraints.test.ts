// G9C tests: constraint model, projection solver, measurements, units,
// persistence, and interaction with undo/redo.
import { describe, expect, it } from "vitest";
import {
  serializePatternDocument,
  deserializePatternDocument,
  createPoint,
  type PatternDocument,
} from "../../src/pattern/cad.js";
import {
  addAngleConstraint,
  addDistanceConstraint,
  addEqualLengthConstraint,
  addFixedLengthConstraint,
  addHorizontalConstraint,
  addParallelConstraint,
  addPerpendicularConstraint,
  addVerticalConstraint,
  constraintResidual,
  createConstraintSet,
  deserializeConstraintSet,
  deserializeConstrainedDocument,
  formatLength,
  measureEdgeLength,
  measurePanelArea,
  measurePanelPerimeter,
  measurePointDistance,
  measureSeamLength,
  measureSegmentAngle,
  parseLengthToM,
  removeConstraint,
  serializeConstraintSet,
  serializeConstrainedDocument,
  setConstraintEnabled,
  solveConstraints,
  validateConstraintSet,
} from "../../src/cad/constraints.js";
import { CadSession } from "../../src/cad/history.js";
import { draftLine } from "../../src/cad/draft.js";
import { getPoint } from "../../src/cad/queries.js";
import { rectFixture } from "./fixtures.js";

describe("G9C constraint model", () => {
  it("creates stable ids and validates references", () => {
    const f = rectFixture();
    let set = createConstraintSet();
    const a = addDistanceConstraint(set, f.panelId, f.points.bl, f.points.br, 0.4);
    set = a.set;
    expect(a.id).toBe("constraint/00000001");
    expect(validateConstraintSet(f.document, set)).toEqual([]);
    const bad = addDistanceConstraint(set, f.panelId, f.points.bl, "missing", 0.4);
    expect(validateConstraintSet(f.document, bad.set).map((d) => d.code)).toContain("missing-reference");
    const neg = addDistanceConstraint(set, f.panelId, f.points.bl, f.points.br, -1);
    expect(validateConstraintSet(f.document, neg.set).map((d) => d.code)).toContain("invalid-target");
  });

  it("disables and removes constraints", () => {
    const f = rectFixture();
    let set = createConstraintSet();
    const a = addFixedLengthConstraint(set, f.panelId, f.segments.bottom, 0.4);
    set = setConstraintEnabled(a.set, a.id, false);
    expect(set.constraints[0].enabled).toBe(false);
    const solved = solveConstraints(f.document, set);
    expect(solved.satisfied).toBe(true); // nothing active: vacuously satisfied
    set = removeConstraint(set, a.id);
    expect(set.constraints).toEqual([]);
    expect(() => removeConstraint(set, a.id)).toThrowError(/does not exist/);
  });
});

describe("G9C projection solver", () => {
  it("drives an exact distance", () => {
    const f = rectFixture();
    let set = createConstraintSet();
    const a = addDistanceConstraint(set, f.panelId, f.points.bl, f.points.br, 0.5);
    const solved = solveConstraints(f.document, a.set);
    expect(solved.satisfied).toBe(true);
    expect(solved.residuals[a.id]).toBeLessThan(1e-9);
    const p = getPoint(solved.document, f.points.bl, f.panelId);
    const q = getPoint(solved.document, f.points.br, f.panelId);
    expect(Math.hypot(q.x - p.x, q.y - p.y)).toBeCloseTo(0.5, 9);
    // Symmetric correction preserves the midpoint.
    expect((p.x + q.x) / 2).toBeCloseTo(0.2, 9);
  });

  it("solves multiple simultaneous constraints", () => {
    const f = rectFixture();
    let set = createConstraintSet();
    const h = addHorizontalConstraint(set, f.panelId, f.points.bl, f.points.br);
    set = h.set;
    const v = addVerticalConstraint(set, f.panelId, f.points.bl, f.points.tl);
    set = v.set;
    const d = addDistanceConstraint(set, f.panelId, f.points.bl, f.points.tr, Math.hypot(0.4, 0.3));
    set = d.set;
    const solved = solveConstraints(f.document, set);
    expect(solved.satisfied).toBe(true);
    expect(solved.unsatisfied).toEqual([]);
  });

  it("reports contradictory constraints as unsatisfied, never as garbage", () => {
    const f = rectFixture();
    let set = createConstraintSet();
    const a = addDistanceConstraint(set, f.panelId, f.points.bl, f.points.br, 0.5);
    set = a.set;
    const b = addDistanceConstraint(set, f.panelId, f.points.bl, f.points.br, 0.1);
    set = b.set;
    const solved = solveConstraints(f.document, set, { maxIterations: 50 });
    expect(solved.satisfied).toBe(false);
    // Sequential projection ends each sweep on b: a stays violated, b holds.
    // The contradiction is reported, not papered over.
    expect(solved.unsatisfied).toEqual([a.id]);
    // Geometry stays finite and sane.
    for (const p of solved.document.points) {
      expect(Number.isFinite(p.x)).toBe(true);
      expect(Number.isFinite(p.y)).toBe(true);
    }
  });

  it("equalizes lengths and fixes edge length", () => {
    const f = rectFixture();
    let set = createConstraintSet();
    const e = addEqualLengthConstraint(set, f.panelId, f.segments.bottom, f.segments.right);
    const solved = solveConstraints(f.document, e.set);
    expect(solved.satisfied).toBe(true);
    expect(measureEdgeLength(solved.document, f.panelId, f.segments.bottom))
      .toBeCloseTo(measureEdgeLength(solved.document, f.panelId, f.segments.right), 9);
    let set2 = createConstraintSet();
    const x = addFixedLengthConstraint(set2, f.panelId, f.segments.bottom, 1);
    set2 = x.set;
    const solved2 = solveConstraints(f.document, set2);
    expect(measureEdgeLength(solved2.document, f.panelId, f.segments.bottom)).toBeCloseTo(1, 9);
  });

  it("enforces parallel, perpendicular, and angle", () => {
    const f = rectFixture();
    const diag = draftLine(f.document, f.panelId, [0, 0], [1, 1]);
    let set = createConstraintSet();
    const p = addParallelConstraint(set, f.panelId, diag.segmentId, f.segments.bottom);
    // Note: bottom is horizontal; diagonal must rotate to horizontal.
    const solved = solveConstraints(diag.document, p.set);
    expect(solved.satisfied).toBe(true);
    expect(measureSegmentAngle(solved.document, f.panelId, diag.segmentId, f.segments.bottom))
      .toBeLessThan(1e-6);
    let set2 = createConstraintSet();
    const q = addPerpendicularConstraint(set2, f.panelId, diag.segmentId, f.segments.bottom);
    const solved2 = solveConstraints(diag.document, q.set);
    expect(solved2.satisfied).toBe(true);
    expect(Math.abs(measureSegmentAngle(solved2.document, f.panelId, diag.segmentId, f.segments.bottom) - Math.PI / 2))
      .toBeLessThan(1e-6);
    let set3 = createConstraintSet();
    const r = addAngleConstraint(set3, f.panelId, f.segments.bottom, diag.segmentId, Math.PI / 4);
    const solved3 = solveConstraints(diag.document, r.set);
    expect(solved3.satisfied).toBe(true);
    expect(Math.abs(measureSegmentAngle(solved3.document, f.panelId, f.segments.bottom, diag.segmentId) - Math.PI / 4))
      .toBeLessThan(1e-6);
  });

  it("leaves coincident points unsatisfied but finite", () => {
    const f = rectFixture();
    let set = createConstraintSet();
    // Two coincident construction points: no defined direction to separate them.
    const c1 = createPoint(f.document, f.panelId, [0.05, 0.05], "construction");
    const c2 = createPoint(c1.document, f.panelId, [0.05, 0.05], "construction");
    const a = addDistanceConstraint(set, f.panelId, c1.pointId, c2.pointId, 0.5);
    const solved = solveConstraints(c2.document, a.set, { maxIterations: 5 });
    expect(solved.satisfied).toBe(false);
    expect(solved.unsatisfied).toEqual([a.id]);
  });

  it("is deterministic across repeated solves", () => {
    const f = rectFixture();
    let set = createConstraintSet();
    const a = addDistanceConstraint(set, f.panelId, f.points.bl, f.points.tr, 0.55);
    set = a.set;
    const b = addHorizontalConstraint(set, f.panelId, f.points.bl, f.points.br);
    set = b.set;
    const s1 = solveConstraints(f.document, set);
    const s2 = solveConstraints(f.document, set);
    expect(JSON.stringify(s1.document)).toBe(JSON.stringify(s2.document));
    expect(s1.residuals).toEqual(s2.residuals);
  });

  it("plays through undo/redo sessions", () => {
    const f = rectFixture();
    const session = new CadSession(f.document);
    let set = createConstraintSet();
    const a = addDistanceConstraint(set, f.panelId, f.points.bl, f.points.br, 0.5);
    set = a.set;
    const solved = solveConstraints(session.document, set);
    session.run("solve distance", () => solved.document);
    expect(measurePointDistance(session.document, f.panelId, f.points.bl, f.points.br)).toBeCloseTo(0.5, 9);
    session.undo();
    expect(measurePointDistance(session.document, f.panelId, f.points.bl, f.points.br)).toBeCloseTo(0.4, 12);
    session.redo();
    expect(measurePointDistance(session.document, f.panelId, f.points.bl, f.points.br)).toBeCloseTo(0.5, 9);
  });
});

describe("G9C measurements and units", () => {
  it("measures distances, edges, angles, areas, perimeters", () => {
    const f = rectFixture(0.4, 0.3);
    expect(measurePointDistance(f.document, f.panelId, f.points.bl, f.points.tr)).toBeCloseTo(0.5, 12);
    expect(measureEdgeLength(f.document, f.panelId, f.segments.bottom)).toBeCloseTo(0.4, 12);
    expect(measureSegmentAngle(f.document, f.panelId, f.segments.bottom, f.segments.right)).toBeCloseTo(Math.PI / 2, 12);
    expect(measurePanelArea(f.document, f.panelId)).toBeCloseTo(0.12, 12);
    expect(measurePanelPerimeter(f.document, f.panelId)).toBeCloseTo(1.4, 12);
  });

  it("measures seam length along G8B correspondence", () => {
    const f = rectFixture(0.4, 0.3);
    const len = measureSeamLength(f.document, {
      id: "seam/m",
      sideA: { panelId: f.panelId, loopId: f.loopId, segmentIds: [f.segments.bottom], reversed: false },
      sideB: { panelId: f.panelId, loopId: f.loopId, segmentIds: [f.segments.top], reversed: false },
      stitchCount: 9,
    });
    expect(len).toBeCloseTo(0.4, 9);
  });

  it("converts units without touching storage", () => {
    expect(formatLength(0.5, "mm")).toBeCloseTo(500, 9);
    expect(formatLength(0.5, "cm")).toBeCloseTo(50, 9);
    expect(formatLength(0.5, "in")).toBeCloseTo(500 / 25.4, 9);
    expect(parseLengthToM(500, "mm")).toBeCloseTo(0.5, 12);
    expect(parseLengthToM(50, "cm")).toBeCloseTo(0.5, 12);
    expect(() => formatLength(NaN, "mm")).toThrowError(/finite/);
  });

  it("reports residuals for inspection", () => {
    const f = rectFixture();
    let set = createConstraintSet();
    const a = addDistanceConstraint(set, f.panelId, f.points.bl, f.points.br, 0.4);
    set = a.set;
    expect(constraintResidual(f.document, set.constraints[0])).toBeLessThan(1e-12);
  });
});

describe("G9C persistence", () => {
  it("round-trips the sidecar byte-identically", () => {
    const f = rectFixture();
    let set = createConstraintSet();
    const a = addDistanceConstraint(set, f.panelId, f.points.bl, f.points.br, 0.5);
    set = a.set;
    const s0 = serializeConstraintSet(set);
    expect(serializeConstraintSet(deserializeConstraintSet(s0))).toBe(s0);
  });

  it("binds pattern and constraints in one envelope", () => {
    const f = rectFixture();
    let set = createConstraintSet();
    const a = addDistanceConstraint(set, f.panelId, f.points.bl, f.points.br, 0.5);
    set = a.set;
    const patternJson = serializePatternDocument(f.document);
    const env = serializeConstrainedDocument(patternJson, set);
    const back = deserializeConstrainedDocument(env);
    expect(deserializePatternDocument(back.documentJson)).toEqual(f.document);
    expect(serializeConstraintSet(back.set)).toBe(serializeConstraintSet(set));
    expect(serializeConstrainedDocument(patternJson, set)).toBe(env);
  });

  it("rejects malformed envelopes", () => {
    expect(() => deserializeConstraintSet("nope")).toThrowError(/JSON/);
    expect(() => deserializeConstraintSet('{"version":2,"constraints":[]}')).toThrowError(/version/);
    expect(() => deserializeConstrainedDocument('{"pattern":1}')).toThrowError(/envelope/);
  });

  it("survives save/load/edit cycles with stable references", () => {
    const f = rectFixture();
    let set = createConstraintSet();
    const a = addDistanceConstraint(set, f.panelId, f.points.bl, f.points.br, 0.5);
    set = a.set;
    // Save.
    const saved = serializeConstrainedDocument(serializePatternDocument(f.document), set);
    // Load.
    const loaded = deserializeConstrainedDocument(saved);
    const doc: PatternDocument = deserializePatternDocument(loaded.documentJson);
    // Edit under constraint, then solve: references still resolve.
    const solved = solveConstraints(doc, loaded.set);
    expect(solved.satisfied).toBe(true);
    expect(measurePointDistance(solved.document, f.panelId, f.points.bl, f.points.br)).toBeCloseTo(0.5, 9);
  });
});
