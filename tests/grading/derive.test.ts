import { describe, expect, it } from "vitest";
import { deletePanel, movePoint, serializePatternDocument, type PatternDocument } from "../../src/pattern/cad.js";
import {
  addGradingPoint,
  addRule,
  addSize,
  createGradingDocument,
  createGradingPoint,
  createRule,
  createRuleTable,
  createSize,
  createSizeSet,
  deriveSize,
  fingerprintGradingDocument,
  findGraded,
  insertSize,
  isStale,
  regenerateAll,
  removeSize,
  setRuleDelta,
  setSizeActive,
  upsertGraded,
  validateGradingDocument,
  GradingError,
  parseGradingDocument,
  serializeGradingDocument,
  serializeGradingValue,
  type GradingDocument,
} from "../../src/grading/index.js";
import {
  addCoreGradingPoints,
  addCoreRules,
  addCoreSizes,
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

function deriveError(fn: () => unknown): GradingError {
  try {
    fn();
  } catch (error) {
    if (error instanceof GradingError) return error;
    throw error;
  }
  throw new Error("expected GradingError");
}

function expectPointAt(doc: PatternDocument, pointId: string, expected: [number, number]): void {
  const [x, y] = pointPosition(doc, pointId);
  expect(x).toBeCloseTo(expected[0], 10);
  expect(y).toBeCloseTo(expected[1], 10);
}

describe("G12A derivation: sizes", () => {
  it("grades a single size with transition semantics relative to the master", () => {
    const { doc, ids } = coreDocument();
    let single = createGradingDocument({
      id: "grading/single", name: "Single", master: doc.master,
      sizeSet: createSizeSet("sizeset/single", "One size"),
      ruleTable: createRuleTable("ruletable/single", "Rules"),
      seams: doc.seams,
    });
    single = addSize(single, createSize({ id: "size/one", label: "ONE" }));
    single = addGradingPoint(single, createGradingPoint("gp/neck", { kind: "point", panelId: ids.frontPanel, pointId: ids.points.C }));
    single = addRule(single, createRule("rule/neck", "gp/neck", "transition", { "size/one": [0.05, -0.01] }));
    const { graded, report } = deriveSize(single, "size/one");
    expect(report.applied).toHaveLength(1);
    expect(report.applied[0].delta).toEqual([0.05, -0.01]);
    expectPointAt(graded.document, ids.points.C, [0.45, 0.59]);
  });

  it("grades multiple sizes deterministically, accumulating transitions", () => {
    const { doc, ids } = coreDocument();
    const small = deriveSize(doc, "size/s").graded;
    const medium = deriveSize(doc, "size/m").graded;
    const large = deriveSize(doc, "size/l").graded;

    expectPointAt(small.document, ids.points.C, [0.4, 0.59]);
    expectPointAt(medium.document, ids.points.C, [0.4, 0.58]);
    expectPointAt(large.document, ids.points.C, [0.4, 0.57]);

    // transition hem: accumulates along the ordered active sizes
    expectPointAt(small.document, ids.points.B, [0.41, 0]);
    expectPointAt(medium.document, ids.points.B, [0.42, 0]);
    expectPointAt(large.document, ids.points.B, [0.43, 0]);

    // per-size centre-back seam endpoint
    expectPointAt(medium.document, ids.points.H, [0.39, 0.6]);
    expectPointAt(large.document, ids.points.H, [0.385, 0.6]);
  });

  it("moves only the anchored points; unrelated geometry stays at master positions", () => {
    const { doc, ids } = coreDocument();
    const medium = deriveSize(doc, "size/m").graded;
    for (const id of [ids.points.A, ids.points.D, ids.points.E, ids.points.F, ids.points.G]) {
      const master = doc.master.document.points.find((p) => p.id === id)!;
      expect(pointPosition(medium.document, id)).toEqual([master.x, master.y]);
    }
    expect(medium.document.panels).toHaveLength(2);
    expect(medium.document.id).toBe(doc.master.document.id);
  });

  it("reports evaluated-only anchors (edge midpoints) moving with graded endpoints plus their rule delta", () => {
    const { doc } = coreDocument();
    const { report } = deriveSize(doc, "size/m");
    expect(report.evaluated).toHaveLength(1);
    const underarm = report.evaluated[0];
    expect(underarm.gradingPointId).toBe("gp/underarm");
    // derived cd midpoint: C moved to (0.4,0.58), D stays (0,0.6) -> (0.2,0.59); + accumulated [0,0.01]
    expect(underarm.gradedPosition[0]).toBeCloseTo(0.2, 12);
    expect(underarm.gradedPosition[1]).toBeCloseTo(0.6, 12);
  });
});

describe("G12A derivation: rule failures", () => {
  it("throws on a missing rule in strict mode and reports a diagnostic in lenient mode", () => {
    const { doc, ids } = coreDocument();
    const broken = addGradingPoint(doc, createGradingPoint("gp/extra", { kind: "point", panelId: ids.frontPanel, pointId: ids.points.D }));
    expect(deriveError(() => deriveSize(broken, "size/m")).code).toBe("missing-rule");
    const lenient = deriveSize(broken, "size/m", { strict: false });
    expect(lenient.report.diagnostics.map((d) => d.code)).toContain("missing-rule");
    expectPointAt(lenient.graded.document, ids.points.D, [0, 0.6]);
  });

  it("throws when a per-size rule lacks the target size key", () => {
    const { doc } = coreDocument();
    const partial: GradingDocument = {
      ...doc,
      ruleTable: { ...doc.ruleTable, rules: doc.ruleTable.rules.map((r) => r.id === "rule/neck" ? { ...r, deltas: { "size/s": r.deltas["size/s"] } } : r) },
    };
    expect(deriveError(() => deriveSize(partial, "size/m")).code).toBe("missing-rule");
  });

  it("throws when a transition chain has a hole at a middle size", () => {
    const { doc } = coreDocument();
    const holed: GradingDocument = {
      ...doc,
      ruleTable: { ...doc.ruleTable, rules: doc.ruleTable.rules.map((r) => r.id === "rule/hem" ? { ...r, deltas: { "size/s": r.deltas["size/s"], "size/l": r.deltas["size/l"] } } : r) },
    };
    expect(deriveError(() => deriveSize(holed, "size/l")).code).toBe("missing-rule");
  });

  it("flags duplicate rules for one grading point as ambiguous", () => {
    const { doc } = coreDocument();
    const duplicated: GradingDocument = {
      ...doc,
      ruleTable: {
        ...doc.ruleTable,
        rules: [...doc.ruleTable.rules, createRule("rule/neck-shadow", "gp/neck", "per-size", { "size/m": [1, 1] })],
      },
    };
    expect(validateGradingDocument(duplicated).map((d) => d.code)).toContain("duplicate-rule");
    expect(deriveError(() => deriveSize(duplicated, "size/m")).code).toBe("invalid-document");
  });
});

describe("G12A derivation: invalid and deleted references", () => {
  it("detects grading points whose source entities were deleted from the master", () => {
    const { doc } = coreDocument();
    const masterBefore = serializePatternDocument(doc.master.document);
    const severed: GradingDocument = {
      ...doc,
      master: { ...doc.master, document: deletePanel(doc.master.document, doc.master.document.panels[0].id) },
    };
    const codes = validateGradingDocument(severed).map((d) => d.code);
    expect(codes).toContain("unknown-entity");
    expect(deriveError(() => deriveSize(severed, "size/m")).code).toBe("invalid-document");
    expect(serializePatternDocument(doc.master.document)).toBe(masterBefore);
  });

  it("rejects derived geometry that self-intersects after a huge grade delta", () => {
    const { doc } = coreDocument();
    let huge = addSize(doc, createSize({ id: "size/huge", label: "HUGE" }));
    huge = setRuleDelta(huge, "rule/neck", "size/huge", [0, 0]);
    huge = setRuleDelta(huge, "rule/hem", "size/huge", [0, 10]);
    huge = setRuleDelta(huge, "rule/cb", "size/huge", [0, 0]);
    huge = setRuleDelta(huge, "rule/underarm", "size/huge", [0, 0]);
    const error = deriveError(() => deriveSize(huge, "size/huge"));
    expect(error.code).toBe("invalid-document");
    expect(error.message).toMatch(/self-intersection|invalid/);
  });

  it("refuses to grade unknown or inactive sizes", () => {
    const { doc } = coreDocument();
    expect(deriveError(() => deriveSize(doc, "size/none")).code).toBe("unknown-size");
    const withInactive = setSizeActive(doc, "size/l", false);
    expect(deriveError(() => deriveSize(withInactive, "size/l")).code).toBe("unknown-size");
  });
});

describe("G12A derivation: size insertion and deletion", () => {
  it("inserts a middle size and keeps accumulation consistent", () => {
    const { doc, ids } = coreDocument();
    let inserted = insertSize(doc, createSize({ id: "size/s2", label: "S+" }), 1);
    inserted = setRuleDelta(inserted, "rule/neck", "size/s2", [0, -0.005]);
    inserted = setRuleDelta(inserted, "rule/hem", "size/s2", [0.005, 0]);
    inserted = setRuleDelta(inserted, "rule/cb", "size/s2", [-0.0025, 0]);
    inserted = setRuleDelta(inserted, "rule/underarm", "size/s2", [0, 0.0025]);

    const s2 = deriveSize(inserted, "size/s2").graded;
    expectPointAt(s2.document, ids.points.B, [0.415, 0]);
    // Size N->N+1 transitions re-chain across the insertion: later sizes
    // inherit the inserted step unless their rules are retuned.
    const medium = deriveSize(inserted, "size/m").graded;
    expectPointAt(medium.document, ids.points.B, [0.425, 0]);
    const large = deriveSize(inserted, "size/l").graded;
    expectPointAt(large.document, ids.points.B, [0.435, 0]);
  });

  it("survives size deletion and stops grading removed sizes", () => {
    const { doc } = coreDocument();
    const trimmed = removeSize(doc, "size/l");
    const medium = deriveSize(trimmed, "size/m").graded;
    expect(medium.document.id).toBe(doc.master.document.id);
    expect(deriveError(() => deriveSize(trimmed, "size/l")).code).toBe("unknown-size");
  });
});

describe("G12A: master invariance and determinism", () => {
  it("never mutates the master pattern when deriving (byte-for-byte)", () => {
    const { doc } = coreDocument();
    const before = serializePatternDocument(doc.master.document);
    const fingerprintBefore = fingerprintGradingDocument(doc);
    deriveSize(doc, "size/s");
    deriveSize(doc, "size/m");
    deriveSize(doc, "size/l");
    expect(serializePatternDocument(doc.master.document)).toBe(before);
    expect(fingerprintGradingDocument(doc)).toBe(fingerprintBefore);
  });

  it("regenerates the full size set with byte-identical output", () => {
    const { doc } = coreDocument();
    const first = regenerateAll(doc);
    const second = regenerateAll(doc);
    expect(first.document.graded.map((g) => g.sizeId)).toEqual(["size/s", "size/m", "size/l"]);
    expect(serializeGradingValue(first.document)).toBe(serializeGradingValue(second.document));
    expect(serializeGradingValue(first.reports)).toBe(serializeGradingValue(second.reports));
  });

  it("keeps entity IDs stable across derived sizes", () => {
    const { doc, ids } = coreDocument();
    const { graded } = deriveSize(doc, "size/l");
    expect(graded.document.points.map((p) => p.id)).toEqual(doc.master.document.points.map((p) => p.id));
    expectPointAt(graded.document, ids.points.C, [0.4, 0.57]);
  });
});

describe("G12A: persistence and invalidation", () => {
  it("round-trips save/load byte-for-byte including the derived cache", () => {
    const { doc } = coreDocument();
    const populated = regenerateAll(doc).document;
    const serialized = serializeGradingDocument(populated);
    const parsed = parseGradingDocument(serialized);
    expect(serializeGradingDocument(parsed)).toBe(serialized);
    expect(parsed.graded).toHaveLength(3);
    expect(parsed.sizeSet.baseSizeId).toBe("size/s");
    expect(parsed.gradingPoints.map((gp) => gp.id)).toEqual(populated.gradingPoints.map((gp) => gp.id));
    expect(isStale(parsed, parsed.graded[1])).toBe(false);
  });

  it("invalidates derived geometry when the master changes and clears staleness on regeneration", () => {
    const { doc, ids } = coreDocument();
    const populated = regenerateAll(doc).document;
    const gradedM = findGraded(populated, "size/m")!;
    expect(isStale(populated, gradedM)).toBe(false);

    const movedMaster = movePoint(populated.master.document, ids.frontPanel, ids.points.A, [0.05, 0.05], "local");
    const edited: GradingDocument = { ...populated, master: { ...populated.master, document: movedMaster } };
    expect(fingerprintGradingDocument(edited)).not.toBe(fingerprintGradingDocument(populated));
    expect(isStale(edited, gradedM)).toBe(true);

    const regenerated = regenerateAll(edited).document;
    const newGradedM = findGraded(regenerated, "size/m")!;
    expect(isStale(regenerated, newGradedM)).toBe(false);
    expect(newGradedM.sourceFingerprint).not.toBe(gradedM.sourceFingerprint);
  });

  it("invalidates derived geometry when a rule changes and re-derivation clears it", () => {
    const { doc, ids } = coreDocument();
    const populated = regenerateAll(doc).document;
    const gradedM = findGraded(populated, "size/m")!;
    const retuned = setRuleDelta(populated, "rule/neck", "size/m", [0, -0.025]);
    expect(isStale(retuned, gradedM)).toBe(true);
    const regenerated = upsertGraded(retuned, deriveSize(retuned, "size/m").graded);
    expect(isStale(regenerated, findGraded(regenerated, "size/m")!)).toBe(false);
    expectPointAt(findGraded(regenerated, "size/m")!.document, ids.points.C, [0.4, 0.575]);
  });

  it("serializes entity values deterministically regardless of key insertion order", () => {
    const a = serializeGradingValue({ b: 1, a: [1, 2], c: { z: -0, y: "x" } });
    const b = serializeGradingValue({ c: { y: "x", z: 0 }, a: [1, 2], b: 1 });
    expect(a).toBe(b);
    expect(a).toBe('{"a":[1,2],"b":1,"c":{"y":"x","z":0}}');
  });

  it("covers the seam context through persistence", () => {
    const { doc } = coreDocument();
    expect(doc.seams).toHaveLength(1);
    expect(doc.seams[0].id).toBe(SEAM_ID);
    expect(doc.seams[0].sideB.reversed).toBe(true);
  });
});
