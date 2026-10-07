// G12B grade-rule engine tests: delta composition precedence (own rule vs
// mirror binding vs panel adjustments), transition accumulation, seam and
// direction diagnostics, curved-boundary grading, production-set derivation
// and master-pattern integrity under regeneration.
import { describe, expect, it } from "vitest";
import {
  movePoint,
  validatePatternDocument,
  type PatternDocument,
} from "../../src/pattern/cad.js";
import {
  addAllowance,
  addGrainline,
  addNotch,
  createProductionSet,
  removeGrainline,
} from "../../src/cad/production.js";
import {
  DEFAULT_SEAM_TOLERANCE_M,
  activeOrderedSizes,
  addMarkingAnchor,
  addMirrorBinding,
  addPanelAdjustment,
  addRule,
  addSize,
  addGradingPoint,
  assignProduction,
  checkRuleConsistency,
  createGradingDocument,
  createGradingPoint,
  createMarkingAnchor,
  createMasterPattern,
  createMirrorBinding,
  createPanelAdjustment,
  createRule,
  createRuleTable,
  createSize,
  createSizeSet,
  deriveProductionSet,
  deriveSize,
  effectiveDeltaForPoint,
  findGraded,
  isStale,
  regenerateAll,
  removePanelAdjustment,
  ruleDeltaForSize,
  setRuleDelta,
  setSizeActive,
  validateGradingDocument,
  GradingError,
  type GradingDocument,
} from "../../src/grading/index.js";
import {
  addCoreGradingPoints,
  addCoreRules,
  addCoreSizes,
  buildArcPanelDocument,
  buildGradingFixture,
  SEAM_ID,
} from "./fixtures.js";

function coreDocument(): { doc: GradingDocument; ids: ReturnType<typeof buildGradingFixture>["ids"] } {
  const fixture = buildGradingFixture();
  let doc = addCoreSizes(fixture.doc);
  doc = addCoreGradingPoints(doc, fixture.ids);
  doc = addCoreRules(doc);
  return { doc, ids: fixture.ids };
}

function pointPosition(doc: PatternDocument, pointId: string): [number, number] {
  const point = doc.points.find((candidate) => candidate.id === pointId);
  if (!point) throw new Error(`point ${pointId} missing in derived document`);
  return [point.x, point.y];
}

function expectPointAt(doc: PatternDocument, pointId: string, expected: [number, number]): void {
  const [x, y] = pointPosition(doc, pointId);
  expect(x).toBeCloseTo(expected[0], 10);
  expect(y).toBeCloseTo(expected[1], 10);
}

function gradingError(fn: () => unknown): GradingError {
  try {
    fn();
  } catch (error) {
    if (error instanceof GradingError) return error;
    throw error;
  }
  throw new Error("expected GradingError");
}

// ---------------------------------------------------------------------------
// Delta propagation
// ---------------------------------------------------------------------------

describe("delta propagation", () => {
  it("accumulates transition deltas linearly across five sizes", () => {
    const fixture = buildGradingFixture();
    let doc = fixture.doc;
    for (const label of ["1", "2", "3", "4", "5"]) {
      doc = addSize(doc, createSize({ id: `size/g${label}`, label }));
    }
    doc = addGradingPoint(doc, createGradingPoint("gp/hem-side", {
      kind: "corner", panelId: fixture.ids.frontPanel,
      segmentIdA: fixture.ids.segments.ab, segmentIdB: fixture.ids.segments.bc,
    }));
    doc = addRule(doc, createRule("rule/hem", "gp/hem-side", "transition", {
      "size/g1": [0.008, 0], "size/g2": [0.008, 0], "size/g3": [0.008, 0],
      "size/g4": [0.008, 0], "size/g5": [0.008, 0],
    }));

    const sizes = activeOrderedSizes(doc.sizeSet);
    const rule = doc.ruleTable.rules.find((candidate) => candidate.id === "rule/hem")!;
    expect(ruleDeltaForSize(rule, sizes, "size/g4")).toEqual([0.032, 0]);
    expect(ruleDeltaForSize(rule, sizes, "size/g5")).toEqual([0.04, 0]);

    for (let k = 1; k <= 5; k++) {
      const { graded, report } = deriveSize(doc, `size/g${k}`);
      expectPointAt(graded.document, fixture.ids.points.B, [0.4 + 0.008 * k, 0]);
      expect(report.applied).toHaveLength(1);
      expect(report.applied[0]).toMatchObject({
        gradingPointId: "gp/hem-side",
        ruleId: "rule/hem",
        mode: "transition",
        delta: [0.008 * k, 0],
      });
    }
  });

  it("applies asymmetric per-size deltas verbatim", () => {
    const { doc, ids } = coreDocument();
    const s = deriveSize(doc, "size/s").graded;
    const m = deriveSize(doc, "size/m").graded;
    const l = deriveSize(doc, "size/l").graded;
    expectPointAt(s.document, ids.points.C, [0.4, 0.59]);
    expectPointAt(m.document, ids.points.C, [0.4, 0.58]);
    expectPointAt(l.document, ids.points.C, [0.4, 0.57]);
  });

  it("skips inactive sizes and refuses to derive them", () => {
    const { doc } = coreDocument();
    const inactive = setSizeActive(doc, "size/m", false);
    const regenerated = regenerateAll(inactive).document;
    expect(regenerated.graded.map((graded) => graded.sizeId).sort()).toEqual(["size/l", "size/s"]);
    const error = gradingError(() => deriveSize(inactive, "size/m"));
    expect(error.code).toBe("unknown-size");
    expect(error.message).toMatch(/inactive/i);
  });
});

// ---------------------------------------------------------------------------
// Mirror bindings
// ---------------------------------------------------------------------------

// The core fixture's gp/cb seam-point owns H, so mirror tests run on a
// dedicated neck-only document where H (the mirror counterpart of C across
// the side seam x=0.4) is free.
function mirrorDocument(): { doc: GradingDocument; ids: ReturnType<typeof buildGradingFixture>["ids"] } {
  const fixture = buildGradingFixture();
  let doc = addCoreSizes(fixture.doc);
  doc = addGradingPoint(doc, createGradingPoint("gp/neck", { kind: "point", panelId: fixture.ids.frontPanel, pointId: fixture.ids.points.C }));
  doc = addRule(doc, createRule("rule/neck", "gp/neck", "per-size", {
    "size/s": [0, -0.01], "size/m": [0, -0.02], "size/l": [0, -0.03],
  }));
  return { doc, ids: fixture.ids };
}

function withBackShoulderMirror(doc: GradingDocument, ids: ReturnType<typeof buildGradingFixture>["ids"]): GradingDocument {
  let next = addGradingPoint(doc, createGradingPoint("gp/back-shoulder", {
    kind: "point", panelId: ids.backPanel, pointId: ids.points.H,
  }));
  next = addMirrorBinding(next, createMirrorBinding("mirror/back-shoulder", "gp/back-shoulder", "gp/neck", "y"));
  return next;
}

describe("mirror bindings", () => {
  it("mirrors the source rule delta with the axis component negated", () => {
    const { doc, ids } = mirrorDocument();
    const mirrored = withBackShoulderMirror(doc, ids);
    const { graded, report } = deriveSize(mirrored, "size/m");
    expectPointAt(graded.document, ids.points.H, [0.4, 0.58]);
    const row = report.applied.find((application) => application.gradingPointId === "gp/back-shoulder")!;
    expect(row.mirrorOfId).toBe("gp/neck");
    expect(row.ruleId).toBe("rule/neck");
    expect(row.mode).toBe("per-size");
    expect(row.delta).toEqual([0, -0.02]);
    expect(effectiveDeltaForPoint(mirrored, "gp/back-shoulder", "size/m")).toEqual([0, -0.02]);
  });

  it("rejects a rule on a mirror-bound grading point", () => {
    const { doc, ids } = mirrorDocument();
    const mirrored = withBackShoulderMirror(doc, ids);
    const error = gradingError(() => addRule(mirrored, createRule("rule/back", "gp/back-shoulder", "per-size", {
      "size/s": [0, 0], "size/m": [0, 0], "size/l": [0, 0],
    })));
    expect(error.code).toBe("conflicting-rule");
  });

  it("rejects mirror chains", () => {
    const { doc, ids } = mirrorDocument();
    const mirrored = withBackShoulderMirror(doc, ids);
    const chained = addGradingPoint(mirrored, createGradingPoint("gp/hand", {
      kind: "point", panelId: ids.backPanel, pointId: ids.points.G,
    }));
    const error = gradingError(() =>
      addMirrorBinding(chained, createMirrorBinding("mirror/hand", "gp/hand", "gp/back-shoulder", "y")));
    expect(error.code).toBe("conflicting-rule");
    expect(error.message).toMatch(/chain/i);
  });

  it("rejects self-mirroring", () => {
    const { doc, ids } = mirrorDocument();
    const withPoint = addGradingPoint(doc, createGradingPoint("gp/back-shoulder", {
      kind: "point", panelId: ids.backPanel, pointId: ids.points.F,
    }));
    const error = gradingError(() =>
      addMirrorBinding(withPoint, createMirrorBinding("mirror/self", "gp/back-shoulder", "gp/back-shoulder", "y")));
    expect(error.code).toBe("conflicting-rule");
  });

  it("flags a mirror source without a rule at validation time", () => {
    const { doc, ids } = coreDocument();
    let next = addGradingPoint(doc, createGradingPoint("gp/loose", {
      kind: "point", panelId: ids.frontPanel, pointId: ids.points.D,
    }));
    next = addGradingPoint(next, createGradingPoint("gp/hand", {
      kind: "point", panelId: ids.backPanel, pointId: ids.points.G,
    }));
    next = addMirrorBinding(next, createMirrorBinding("mirror/hand", "gp/hand", "gp/loose", "y"));
    const diagnostics = validateGradingDocument(next);
    const flagged = diagnostics.find((diagnostic) =>
      diagnostic.code === "invalid-rule" && diagnostic.message.includes("gp/loose"));
    expect(flagged).toBeDefined();
  });
});

// ---------------------------------------------------------------------------
// Panel adjustments
// ---------------------------------------------------------------------------

describe("panel adjustments", () => {
  it("composes with point deltas for every grading point on the panel and records provenance", () => {
    const { doc, ids } = coreDocument();
    const adjusted = addPanelAdjustment(
      doc,
      createPanelAdjustment("adj/front-m", ids.frontPanel, "size/m", [0.005, 0.01]),
    );
    const { graded, report } = deriveSize(adjusted, "size/m");
    const neck = report.applied.find((application) => application.gradingPointId === "gp/neck")!;
    expect(neck.delta).toEqual([0.005, -0.01]);
    expect(neck.panelAdjustmentIds).toEqual(["adj/front-m"]);
    const hem = report.applied.find((application) => application.gradingPointId === "gp/hem-side")!;
    expect(hem.delta).toEqual([0.025, 0.01]);
    expectPointAt(graded.document, ids.points.B, [0.425, 0.01]);
    const cb = report.applied.find((application) => application.gradingPointId === "gp/cb")!;
    expect(cb.delta).toEqual([-0.01, 0]);
    expect(cb.panelAdjustmentIds).toBeUndefined();

    const restored = removePanelAdjustment(adjusted, "adj/front-m");
    const backToBase = deriveSize(restored, "size/m");
    const neckBase = backToBase.report.applied.find((application) => application.gradingPointId === "gp/neck")!;
    expect(neckBase.delta).toEqual([0, -0.02]);
    expect(neckBase.panelAdjustmentIds).toBeUndefined();
  });

  it("rejects a second adjustment for the same panel and size", () => {
    const { doc, ids } = coreDocument();
    const adjusted = addPanelAdjustment(
      doc,
      createPanelAdjustment("adj/front-m", ids.frontPanel, "size/m", [0.005, 0.01]),
    );
    const error = gradingError(() => addPanelAdjustment(
      adjusted,
      createPanelAdjustment("adj/front-m-2", ids.frontPanel, "size/m", [0.001, 0]),
    ));
    expect(error.code).toBe("conflicting-rule");
  });
});

// ---------------------------------------------------------------------------
// Quality diagnostics: seams and grade direction
// ---------------------------------------------------------------------------

function twoSizeHemDocument(): { doc: GradingDocument; ids: ReturnType<typeof buildGradingFixture>["ids"] } {
  const fixture = buildGradingFixture();
  let doc = addSize(fixture.doc, createSize({ id: "size/s", label: "S" }));
  doc = addSize(doc, createSize({ id: "size/m", label: "M" }));
  doc = addGradingPoint(doc, createGradingPoint("gp/hem-side", {
    kind: "corner", panelId: fixture.ids.frontPanel,
    segmentIdA: fixture.ids.segments.ab, segmentIdB: fixture.ids.segments.bc,
  }));
  doc = addRule(doc, createRule("rule/hem", "gp/hem-side", "transition", {
    "size/s": [0.01, 0], "size/m": [0.01, 0],
  }));
  return { doc, ids: fixture.ids };
}

describe("quality diagnostics", () => {
  it("keeps matched seams inside tolerance for small grades", () => {
    const { doc } = twoSizeHemDocument();
    const { report } = deriveSize(doc, "size/m");
    expect(report.seams).toHaveLength(1);
    expect(report.seams[0].seamId).toBe(SEAM_ID);
    expect(report.seams[0].withinTolerance).toBe(true);
    expect(report.seams[0].diffM).toBeLessThanOrEqual(DEFAULT_SEAM_TOLERANCE_M);
    expect(report.diagnostics.filter((diagnostic) => diagnostic.code === "seam-mismatch")).toHaveLength(0);
  });

  it("reports a seam mismatch when one side grades away from the other", () => {
    const { doc } = twoSizeHemDocument();
    const desynced = setRuleDelta(doc, "rule/hem", "size/m", [0.01, 0.05]);
    const { report } = deriveSize(desynced, "size/m");
    const row = report.seams[0];
    expect(row.withinTolerance).toBe(false);
    expect(row.diffM).toBeGreaterThan(DEFAULT_SEAM_TOLERANCE_M);
    expect(row.lengthAM).toBeCloseTo(0.55036, 3);
    expect(row.lengthBM).toBeCloseTo(0.6, 3);
    expect(report.diagnostics.some((diagnostic) => diagnostic.code === "seam-mismatch")).toBe(true);
  });

  it("flags transition direction flips as diagnostics without blocking derivation", () => {
    const { doc, ids } = coreDocument();
    const flipped = setRuleDelta(doc, "rule/hem", "size/m", [-0.01, 0]);
    const diagnostics = checkRuleConsistency(flipped);
    expect(diagnostics).toHaveLength(1);
    expect(diagnostics[0]).toMatchObject({ code: "inconsistent-direction", ruleId: "rule/hem", sizeId: "size/l" });
    const { graded } = deriveSize(flipped, "size/l");
    expectPointAt(graded.document, ids.points.B, [0.41, 0]);
  });
});

// ---------------------------------------------------------------------------
// Curved boundaries
// ---------------------------------------------------------------------------

function arcGradingDocument(): { doc: GradingDocument; ids: ReturnType<typeof buildArcPanelDocument>["ids"] } {
  const arc = buildArcPanelDocument();
  let doc = createGradingDocument({
    id: "grading/arc",
    name: "Arc grading",
    master: createMasterPattern("master/arc", "Arc master", arc.document),
    sizeSet: createSizeSet("sizeset/arc", "Arc sizes"),
    ruleTable: createRuleTable("ruletable/arc", "Arc rules"),
    seams: [],
  });
  doc = addSize(doc, createSize({ id: "size/s", label: "S" }));
  doc = addSize(doc, createSize({ id: "size/m", label: "M" }));
  doc = addGradingPoint(doc, createGradingPoint("gp/arc-b", { kind: "point", panelId: arc.ids.panel, pointId: arc.ids.points.B }));
  doc = addGradingPoint(doc, createGradingPoint("gp/arc-c", { kind: "point", panelId: arc.ids.panel, pointId: arc.ids.points.C }));
  doc = addGradingPoint(doc, createGradingPoint("gp/arc-mid", {
    kind: "edge-relative", panelId: arc.ids.panel, segmentId: arc.ids.segments.arc, t: 0.5,
  }));
  doc = addRule(doc, createRule("rule/arc-b", "gp/arc-b", "per-size", { "size/s": [0, -0.01], "size/m": [0, -0.02] }));
  doc = addRule(doc, createRule("rule/arc-c", "gp/arc-c", "per-size", { "size/s": [0, 0.01], "size/m": [0, 0.02] }));
  return { doc, ids: arc.ids };
}

describe("curved boundaries", () => {
  it("grades a radial arc consistently and follows the graded curve for evaluated points", () => {
    const { doc, ids } = arcGradingDocument();
    const { graded, report } = deriveSize(doc, "size/s");
    expect(validatePatternDocument(graded.document).valid).toBe(true);
    expectPointAt(graded.document, ids.points.B, [0.4, -0.01]);
    expectPointAt(graded.document, ids.points.C, [0.4, 0.81]);
    const midpoint = report.evaluated.find((evaluation) => evaluation.gradingPointId === "gp/arc-mid")!;
    expect(midpoint.displacement).toBe("evaluated");
    expect(midpoint.masterPosition[0]).toBeCloseTo(0.8, 10);
    expect(midpoint.masterPosition[1]).toBeCloseTo(0.4, 10);
    expect(midpoint.gradedPosition[0]).toBeCloseTo(0.81, 10);
    expect(midpoint.gradedPosition[1]).toBeCloseTo(0.4, 10);
  });

  it("rejects radial grades that break arc radius consistency", () => {
    const { doc } = arcGradingDocument();
    const asymmetric = setRuleDelta(doc, "rule/arc-c", "size/s", [0, 0.02]);
    const error = gradingError(() => deriveSize(asymmetric, "size/s"));
    expect(error.code).toBe("invalid-document");
    expect(error.message).toMatch(/radii/i);
  });
});

// ---------------------------------------------------------------------------
// Production-set derivation
// ---------------------------------------------------------------------------

interface ProductionHarness {
  doc: GradingDocument;
  ids: ReturnType<typeof buildGradingFixture>["ids"];
  set: ReturnType<typeof createProductionSet>;
  anchoredGrainlineId: string;
  insideGrainlineId: string;
  outsideGrainlineId: string;
}

function productionHarness(): ProductionHarness {
  const fixture = buildGradingFixture();
  const ids = fixture.ids;
  let doc = addSize(fixture.doc, createSize({ id: "size/one", label: "1" }));
  doc = addGradingPoint(doc, createGradingPoint("gp/top", { kind: "point", panelId: ids.frontPanel, pointId: ids.points.C }));
  doc = addGradingPoint(doc, createGradingPoint("gp/mid", {
    kind: "edge-relative", panelId: ids.frontPanel, segmentId: ids.segments.cd, t: 0.5,
  }));
  doc = addRule(doc, createRule("rule/top", "gp/top", "transition", { "size/one": [0, -0.3] }));

  let set = createProductionSet();
  set = addAllowance(set, ids.frontPanel, ids.frontLoop, 0.01).set;
  set = addNotch(set, ids.frontPanel, ids.frontLoop, ids.segments.cd, 0.5, "single", 0.005).set;
  const anchored = addGrainline(set, ids.frontPanel, [0.2, 0.5], [0.2, 0.55]);
  set = anchored.set;
  const inside = addGrainline(set, ids.frontPanel, [0.1, 0.1], [0.1, 0.15]);
  set = inside.set;
  // Inside the master panel; after the top edge grades down to y=0.3 both
  // points sit above the tilted edge and fall outside the derived panel.
  const outside = addGrainline(set, ids.frontPanel, [0.35, 0.55], [0.35, 0.57]);
  set = outside.set;
  doc = assignProduction(doc, set);
  doc = addMarkingAnchor(doc, createMarkingAnchor("ma/top", ids.frontPanel, "grainline", anchored.id, "gp/top"));
  return { doc, ids, set, anchoredGrainlineId: anchored.id, insideGrainlineId: inside.id, outsideGrainlineId: outside.id };
}

describe("production-set derivation", () => {
  it("copies ID/t-referenced entities verbatim, displaces anchored markings and flags outside-panel markings", () => {
    const harness = productionHarness();
    const { graded } = deriveSize(harness.doc, "size/one");
    const derived = deriveProductionSet(harness.doc, graded.document, "size/one");

    expect(derived.set.allowances).toHaveLength(1);
    expect(derived.set.allowances[0]).toEqual(harness.set.allowances[0]);
    expect(derived.set.notches).toHaveLength(1);
    expect(derived.set.notches[0]).toEqual(harness.set.notches[0]);

    const anchored = derived.set.grainlines.find((grainline) => grainline.id === harness.anchoredGrainlineId)!;
    expect(anchored.from[0]).toBeCloseTo(0.2, 10);
    expect(anchored.from[1]).toBeCloseTo(0.2, 10);
    expect(anchored.to[1]).toBeCloseTo(0.25, 10);
    const inside = derived.set.grainlines.find((grainline) => grainline.id === harness.insideGrainlineId)!;
    expect(inside.from).toEqual([0.1, 0.1]);
    expect(inside.to).toEqual([0.1, 0.15]);
    expect(derived.set.grainlines).toHaveLength(3);

    const outsideIssues = derived.issues.filter((issue) =>
      issue.code === "outside-panel" && issue.entityId === harness.outsideGrainlineId);
    expect(outsideIssues).toHaveLength(2);
    expect(derived.issues.filter((issue) => issue.code === "unknown-marking-anchor")).toHaveLength(0);
  });

  it("reports orphan marking anchors when queried directly after the bound entity disappears", () => {
    const harness = productionHarness();
    const { graded } = deriveSize(harness.doc, "size/one");
    const stripped = assignProduction(harness.doc, removeGrainline(harness.set, harness.anchoredGrainlineId));
    // Direct query path: document-level validation would block a full derive,
    // but deriveProductionSet is callable on its own and must surface the
    // dangling anchor instead of skipping it silently.
    const derived = deriveProductionSet(stripped, graded.document, "size/one");
    const orphan = derived.issues.find((issue) =>
      issue.code === "unknown-marking-anchor" && issue.markingAnchorId === "ma/top");
    expect(orphan).toBeDefined();
    expect(orphan?.entityId).toBe(harness.anchoredGrainlineId);
  });

  it("requires a production set", () => {
    const fixture = buildGradingFixture();
    const error = gradingError(() =>
      deriveProductionSet(fixture.doc, fixture.document, "size/s"));
    expect(error.code).toBe("invalid-argument");
  });
});

// ---------------------------------------------------------------------------
// Master integrity and regeneration
// ---------------------------------------------------------------------------

describe("master integrity and regeneration", () => {
  it("never mutates the master pattern during derivation", () => {
    const { doc } = coreDocument();
    const before = JSON.stringify(doc.master.document);
    deriveSize(doc, "size/m");
    deriveSize(doc, "size/l");
    expect(JSON.stringify(doc.master.document)).toBe(before);
  });

  it("regenerates from an edited master and replaces the derived cache", () => {
    const { doc, ids } = coreDocument();
    const first = regenerateAll(doc).document;
    const oldGraded = findGraded(first, "size/m")!;

    const edited = {
      ...first,
      master: {
        ...first.master,
        document: movePoint(first.master.document, ids.frontPanel, ids.points.B, [0.45, 0]),
      },
    };
    expect(isStale(edited, oldGraded)).toBe(true);

    const second = regenerateAll(edited).document;
    expect(second.graded).toHaveLength(3);
    const regenerated = findGraded(second, "size/m")!;
    expectPointAt(regenerated.document, ids.points.B, [0.47, 0]);
    expect(regenerated.sourceFingerprint).not.toBe(oldGraded.sourceFingerprint);
  });
});
