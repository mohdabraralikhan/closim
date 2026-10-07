// G12E — adversarial grading QA. Attacks reference integrity, size-set
// bookkeeping, geometry, persistence and the export pipeline. Production
// code is under test, never modified: every attack documents either the
// defense that held or a defect (recorded in G12E_FINDINGS.md).
import { describe, expect, it } from "vitest";
import type { PatternDocument } from "../../src/pattern/cad.js";
import {
  addAllowance,
  addAnnotation,
  addDrillMark,
  addGrainline,
  addNotch,
  createProductionSet,
  notchFrame,
  setPanelMeta,
} from "../../src/cad/production.js";
import { productionReadiness } from "../../src/cad/readiness.js";
import {
  canonicalJson,
  createGradingPoint,
  createMarkingAnchor,
  createMirrorBinding,
  createRule,
  createSize,
  deriveProductionSet,
  deriveSize,
  addGradingPoint,
  addMarkingAnchor,
  addMirrorBinding,
  addPanelAdjustment,
  addRule,
  addSize,
  assignProduction,
  fingerprintGradingDocument,
  insertSize,
  isStale,
  moveSize,
  parseGradingDocument,
  regenerateAll,
  removeGradingPoint,
  removeSize,
  serializeGradingDocument,
  setRuleDelta,
  setSizeActive,
  validateGradingDocument,
  GradingError,
  type GradingDocument,
} from "../../src/grading/index.js";
import { createSizeView, defaultSizeViewOptions, documentForSize } from "../../src/grading/presentation.js";
import {
  SEAM_ID,
  buildGradingFixture,
  type TwoPanelFixture,
} from "./fixtures.js";

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

/** Canonical master byte snapshot (immune to object key insertion order). */
function masterSnapshot(doc: GradingDocument): string {
  return canonicalJson(doc.master.document);
}

/** Simulates corrupted persistence: deep clone, apply hostile mutation. */
function corrupted(document: GradingDocument, mutate: (draft: GradingDocument) => void): GradingDocument {
  const draft = JSON.parse(JSON.stringify(document)) as GradingDocument;
  mutate(draft);
  return draft;
}

/** Hem-only document: n sizes, one corner grading point on ab∩bc, one rule. */
function hemDocument(
  deltas: Array<[number, number]>,
  mode: "transition" | "per-size" = "transition",
): { doc: GradingDocument; ids: TwoPanelFixture["ids"] } {
  const fixture = buildGradingFixture();
  let doc = fixture.doc;
  deltas.forEach((_, index) => {
    doc = addSize(doc, createSize({ id: `size/g${index + 1}`, label: String(index + 1) }));
  });
  doc = addGradingPoint(doc, createGradingPoint("gp/hem-side", {
    kind: "corner", panelId: fixture.ids.frontPanel,
    segmentIdA: fixture.ids.segments.ab, segmentIdB: fixture.ids.segments.bc,
  }));
  const table: Record<string, [number, number]> = {};
  deltas.forEach((delta, index) => {
    table[`size/g${index + 1}`] = delta;
  });
  doc = addRule(doc, createRule("rule/hem", "gp/hem-side", mode, table));
  return { doc, ids: fixture.ids };
}

/**
 * Seam-consistent production harness: every seam endpoint grading keeps the
 * side-seam sides vertical (whole pattern shifts by [0.01k, 0]), so derived
 * sizes stay seam-true and can legitimately reach READY_FOR_EXPORT.
 * Markings: 2 allowances, 1 notch, 2 grainlines (front one anchored), 1 drill,
 * 1 annotation, panel meta for both panels.
 */
function productionDocument(): {
  doc: GradingDocument;
  ids: TwoPanelFixture["ids"];
  seam: ReturnType<typeof buildGradingFixture>["seam"];
  grainlineIds: { front: string; back: string };
  notchId: string;
} {
  const fixture = buildGradingFixture();
  const { ids, seam } = fixture;
  let doc = addSize(fixture.doc, createSize({ id: "size/s", label: "S" }));
  doc = addSize(doc, createSize({ id: "size/m", label: "M" }));
  doc = addSize(doc, createSize({ id: "size/l", label: "L" }));
  doc = addGradingPoint(doc, createGradingPoint("gp/hem-side", {
    kind: "corner", panelId: ids.frontPanel, segmentIdA: ids.segments.ab, segmentIdB: ids.segments.bc,
  }));
  doc = addGradingPoint(doc, createGradingPoint("gp/neck", { kind: "point", panelId: ids.frontPanel, pointId: ids.points.C }));
  doc = addGradingPoint(doc, createGradingPoint("gp/cb", { kind: "seam-point", seamId: SEAM_ID, side: "b", segmentId: ids.segments.he, endpoint: "start" }));
  doc = addGradingPoint(doc, createGradingPoint("gp/he-end", { kind: "seam-point", seamId: SEAM_ID, side: "b", segmentId: ids.segments.he, endpoint: "end" }));
  doc = addRule(doc, createRule("rule/hem", "gp/hem-side", "transition", { "size/s": [0.01, 0], "size/m": [0.01, 0], "size/l": [0.01, 0] }));
  doc = addRule(doc, createRule("rule/neck", "gp/neck", "per-size", { "size/s": [0.01, 0], "size/m": [0.02, 0], "size/l": [0.03, 0] }));
  doc = addRule(doc, createRule("rule/cb", "gp/cb", "per-size", { "size/s": [0.01, 0], "size/m": [0.02, 0], "size/l": [0.03, 0] }));
  doc = addRule(doc, createRule("rule/he-end", "gp/he-end", "per-size", { "size/s": [0.01, 0], "size/m": [0.02, 0], "size/l": [0.03, 0] }));

  let set = createProductionSet();
  set = addAllowance(set, ids.frontPanel, ids.frontLoop, 0.02).set;
  set = addAllowance(set, ids.backPanel, ids.backLoop, 0.02).set;
  const notch = addNotch(set, ids.frontPanel, ids.frontLoop, ids.segments.ab, 0.5, "single", 0.005);
  set = notch.set;
  const frontGrainline = addGrainline(set, ids.frontPanel, [0.15, 0.25], [0.25, 0.35]);
  set = frontGrainline.set;
  const backGrainline = addGrainline(set, ids.backPanel, [0.55, 0.25], [0.65, 0.35]);
  set = backGrainline.set;
  set = addDrillMark(set, ids.frontPanel, [0.2, 0.3]).set;
  set = addAnnotation(set, ids.frontPanel, [0.3, 0.5], "cut single ply").set;
  set = setPanelMeta(set, { panelId: ids.frontPanel, cutQuantity: 2 });
  set = setPanelMeta(set, { panelId: ids.backPanel, cutQuantity: 1 });

  doc = assignProduction(doc, set);
  doc = addMarkingAnchor(doc, createMarkingAnchor("ma/front-grainline", ids.frontPanel, "grainline", frontGrainline.id, "gp/hem-side"));
  return { doc, ids, seam, grainlineIds: { front: frontGrainline.id, back: backGrainline.id }, notchId: notch.id };
}

// ---------------------------------------------------------------------------
// Reference integrity attacks
// ---------------------------------------------------------------------------

describe("reference integrity attacks", () => {
  it("removing a grading point cascades its rule, mirrors and marking anchors, then flags stale caches", () => {
    let { doc, ids } = productionDocument();
    const regenerated = regenerateAll(doc).document;
    const cached = regenerated.graded.find((graded) => graded.sizeId === "size/m")!;
    expect(isStale(regenerated, cached)).toBe(false);

    doc = removeGradingPoint(regenerated, "gp/neck");
    expect(doc.ruleTable.rules.some((rule) => rule.gradingPointId === "gp/neck")).toBe(false);
    expect(doc.markingAnchors ?? []).toHaveLength(1);
    expect(validateGradingDocument(doc)).toHaveLength(0);
    expect(isStale(doc, cached)).toBe(true);
    const redone = regenerateAll(doc).document;
    expectPointAt(redone.graded.find((graded) => graded.sizeId === "size/m")!.document, ids.points.C, [0.4, 0.6]);
  });

  it("rejects a duplicated grading point ID and two grading points displacing one master point", () => {
    const fixture = buildGradingFixture();
    let doc = addSize(fixture.doc, createSize({ id: "size/s", label: "S" }));
    doc = addGradingPoint(doc, createGradingPoint("gp/neck", { kind: "point", panelId: fixture.ids.frontPanel, pointId: fixture.ids.points.C }));
    const duplicate = gradingError(() =>
      addGradingPoint(doc, createGradingPoint("gp/neck", { kind: "point", panelId: fixture.ids.frontPanel, pointId: fixture.ids.points.C })));
    expect(duplicate.code).toBe("duplicate-id");

    const withCorner = addGradingPoint(doc, createGradingPoint("gp/hem-side", {
      kind: "corner", panelId: fixture.ids.frontPanel,
      segmentIdA: fixture.ids.segments.ab, segmentIdB: fixture.ids.segments.bc,
    }));
    const conflict = gradingError(() =>
      addGradingPoint(withCorner, createGradingPoint("gp/d", { kind: "point", panelId: fixture.ids.frontPanel, pointId: fixture.ids.points.B })));
    expect(conflict.code).toBe("conflicting-anchor");
  });

  it("flags duplicated rules injected through corrupted persistence and refuses derivation", () => {
    const fixture = buildGradingFixture();
    let doc = addSize(fixture.doc, createSize({ id: "size/s", label: "S" }));
    doc = addGradingPoint(doc, createGradingPoint("gp/hem-side", {
      kind: "corner", panelId: fixture.ids.frontPanel,
      segmentIdA: fixture.ids.segments.ab, segmentIdB: fixture.ids.segments.bc,
    }));
    doc = addRule(doc, createRule("rule/hem", "gp/hem-side", "per-size", { "size/s": [0.01, 0] }));
    const hostile = corrupted(doc, (draft) => {
      draft.ruleTable.rules.push({
        id: "rule/hem-impostor", gradingPointId: "gp/hem-side", mode: "per-size", deltas: { "size/s": [0.5, 0.5] },
      });
    });
    const diagnostics = validateGradingDocument(hostile);
    expect(diagnostics.some((d) => d.code === "duplicate-rule" && /more than one rule/.test(d.message))).toBe(true);
    expect(gradingError(() => deriveSize(hostile, "size/s")).code).toBe("invalid-document");
  });

  it("flags a rule on a mirror-bound grading point injected through corruption", () => {
    const fixture = buildGradingFixture();
    let doc = addSize(fixture.doc, createSize({ id: "size/s", label: "S" }));
    doc = addGradingPoint(doc, createGradingPoint("gp/neck", { kind: "point", panelId: fixture.ids.frontPanel, pointId: fixture.ids.points.C }));
    doc = addRule(doc, createRule("rule/neck", "gp/neck", "per-size", { "size/s": [0, -0.01] }));
    doc = addGradingPoint(doc, createGradingPoint("gp/back-shoulder", { kind: "point", panelId: fixture.ids.backPanel, pointId: fixture.ids.points.H }));
    doc = addMirrorBinding(doc, createMirrorBinding("mirror/back-shoulder", "gp/back-shoulder", "gp/neck", "y"));
    const hostile = corrupted(doc, (draft) => {
      draft.ruleTable.rules.push({
        id: "rule/back-shoulder", gradingPointId: "gp/back-shoulder", mode: "per-size", deltas: { "size/s": [0, 0.02] },
      });
    });
    expect(validateGradingDocument(hostile).some((d) => d.code === "conflicting-rule")).toBe(true);
    expect(gradingError(() => deriveSize(hostile, "size/s")).code).toBe("invalid-document");
  });

  it("flags orphaned rules that reference deleted grading points", () => {
    const fixture = buildGradingFixture();
    let doc = addSize(fixture.doc, createSize({ id: "size/s", label: "S" }));
    doc = addGradingPoint(doc, createGradingPoint("gp/hem-side", {
      kind: "corner", panelId: fixture.ids.frontPanel,
      segmentIdA: fixture.ids.segments.ab, segmentIdB: fixture.ids.segments.bc,
    }));
    doc = addRule(doc, createRule("rule/hem", "gp/hem-side", "per-size", { "size/s": [0.01, 0] }));
    const hostile = corrupted(doc, (draft) => {
      draft.gradingPoints = draft.gradingPoints.filter((point) => point.id !== "gp/hem-side");
    });
    expect(validateGradingDocument(hostile).some((d) => d.code === "unknown-entity" && /rule 'rule\/hem'/.test(d.message))).toBe(true);
    expect(gradingError(() => deriveSize(hostile, "size/s")).code).toBe("invalid-document");
  });

  it("flags rule deltas targeting unknown sizes", () => {
    const { doc } = hemDocument([[0.01, 0]], "per-size");
    const hostile = corrupted(doc, (draft) => {
      draft.ruleTable.rules[0].deltas["size/ghost"] = [0, 0];
    });
    expect(validateGradingDocument(hostile).some((d) => d.code === "unknown-size" && d.sizeId === "size/ghost")).toBe(true);
    expect(gradingError(() => deriveSize(hostile, "size/g1")).code).toBe("invalid-document");
  });

  it("rejects non-finite deltas injected into a rule table", () => {
    const { doc } = hemDocument([[0.01, 0]], "per-size");
    for (const delta of [[Number.NaN, 0], [Infinity, 1]] as Array<[number, number]>) {
      const hostile = corrupted(doc, (draft) => {
        draft.ruleTable.rules[0].deltas["size/g1"] = delta;
      });
      expect(validateGradingDocument(hostile).some((d) => d.code === "invalid-rule")).toBe(true);
      expect(gradingError(() => deriveSize(hostile, "size/g1")).code).toBe("invalid-document");
    }
  });

  it("flags duplicate panel adjustments for one panel-size pair", () => {
    const fixture = buildGradingFixture();
    let doc = addSize(fixture.doc, createSize({ id: "size/s", label: "S" }));
    doc = addGradingPoint(doc, createGradingPoint("gp/hem-side", {
      kind: "corner", panelId: fixture.ids.frontPanel,
      segmentIdA: fixture.ids.segments.ab, segmentIdB: fixture.ids.segments.bc,
    }));
    doc = addRule(doc, createRule("rule/hem", "gp/hem-side", "per-size", { "size/s": [0.01, 0] }));
    doc = addPanelAdjustment(doc, { id: "pa/front-s", panelId: fixture.ids.frontPanel, sizeId: "size/s", delta: [0.005, 0] });
    const hostile = corrupted(doc, (draft) => {
      (draft.ruleTable.panelAdjustments ?? []).push({ id: "pa/front-s-2", panelId: fixture.ids.frontPanel, sizeId: "size/s", delta: [0.01, 0] });
    });
    expect(validateGradingDocument(hostile).some((d) => d.code === "conflicting-rule" && /more than one adjustment/.test(d.message))).toBe(true);
    expect(gradingError(() => deriveSize(hostile, "size/s")).code).toBe("invalid-document");
  });
});

// ---------------------------------------------------------------------------
// Size-set attacks
// ---------------------------------------------------------------------------

describe("size-set attacks", () => {
  it("strict derivation aborts on a rule-less vertex grading point; non-strict reports and leaves it at master", () => {
    const fixture = buildGradingFixture();
    let doc = addSize(fixture.doc, createSize({ id: "size/g1", label: "1" }));
    doc = addGradingPoint(doc, createGradingPoint("gp/hem-side", {
      kind: "corner", panelId: fixture.ids.frontPanel,
      segmentIdA: fixture.ids.segments.ab, segmentIdB: fixture.ids.segments.bc,
    }));
    doc = addRule(doc, createRule("rule/hem", "gp/hem-side", "per-size", { "size/g1": [0.01, 0] }));
    doc = addGradingPoint(doc, createGradingPoint("gp/free", { kind: "point", panelId: fixture.ids.frontPanel, pointId: fixture.ids.points.D }));

    const strict = gradingError(() => deriveSize(doc, "size/g1"));
    expect(strict.code).toBe("missing-rule");
    expect(strict.message).toMatch(/gp\/free/);

    const lenient = deriveSize(doc, "size/g1", { strict: false });
    expectPointAt(lenient.graded.document, fixture.ids.points.D, [0, 0.6]);
    expectPointAt(lenient.graded.document, fixture.ids.points.B, [0.41, 0]);
    expect(lenient.report.diagnostics.some((d) => d.code === "missing-rule" && d.gradingPointId === "gp/free")).toBe(true);
  });

  it("a rule missing the target size delta is a hard error even in non-strict mode", () => {
    const fixture = buildGradingFixture();
    let doc = addSize(fixture.doc, createSize({ id: "size/g1", label: "1" }));
    doc = addSize(doc, createSize({ id: "size/g2", label: "2" }));
    doc = addGradingPoint(doc, createGradingPoint("gp/hem-side", {
      kind: "corner", panelId: fixture.ids.frontPanel,
      segmentIdA: fixture.ids.segments.ab, segmentIdB: fixture.ids.segments.bc,
    }));
    doc = addRule(doc, createRule("rule/hem", "gp/hem-side", "per-size", { "size/g1": [0.01, 0] }));
    const error = gradingError(() => deriveSize(doc, "size/g2", { strict: false }));
    expect(error.code).toBe("missing-rule");
    expect(error.message).toMatch(/size\/g2/);
  });

  it("removing a size cleans rule deltas, adjustments and derived entries, and replaces the base size", () => {
    let { doc, ids } = productionDocument();
    doc = regenerateAll(doc).document;
    doc = removeSize(doc, "size/s");
    expect(doc.sizeSet.baseSizeId).toBe("size/m");
    for (const rule of doc.ruleTable.rules) {
      expect(rule.deltas["size/s"]).toBeUndefined();
    }
    expect(doc.graded.some((graded) => graded.sizeId === "size/s")).toBe(false);
    expect(validateGradingDocument(doc)).toHaveLength(0);
    for (const [sizeId, x] of [["size/m", 0.41], ["size/l", 0.42]] as Array<[string, number]>) {
      const { graded } = deriveSize(doc, sizeId);
      expectPointAt(graded.document, ids.points.B, [x, 0]);
    }
  });

  it("inserting a size without completing its rule deltas fails loudly, then derives once completed", () => {
    let { doc, ids } = hemDocument([[0.01, 0], [0.01, 0], [0.01, 0]]);
    const before = fingerprintGradingDocument(doc);
    doc = insertSize(doc, createSize({ id: "size/gx", label: "X" }), 1);
    expect(fingerprintGradingDocument(doc)).not.toBe(before);
    expect(gradingError(() => deriveSize(doc, "size/gx")).code).toBe("missing-rule");
    doc = setRuleDelta(doc, "rule/hem", "size/gx", [0.005, 0]);
    const { graded } = deriveSize(doc, "size/gx");
    expectPointAt(graded.document, ids.points.B, [0.415, 0]);
  });

  it("inactive sizes are excluded from transition accumulation and refused as targets", () => {
    const { doc, ids } = hemDocument([[0.01, 0], [0.01, 0], [0.01, 0]]);
    const doc2 = setSizeActive(doc, "size/g2", false);
    const error = gradingError(() => deriveSize(doc2, "size/g2"));
    expect(error.code).toBe("unknown-size");
    expect(error.message).toMatch(/inactive/i);
    const { graded } = deriveSize(doc2, "size/g3");
    expectPointAt(graded.document, ids.points.B, [0.42, 0]);
  });

  it("transition accumulation follows size-set order, not label order, after a reorder", () => {
    const { doc, ids } = hemDocument([[0.008, 0], [0.008, 0], [0.008, 0], [0.008, 0], [0.008, 0]]);
    const reordered = moveSize(doc, "size/g5", 0);
    expectPointAt(deriveSize(reordered, "size/g1").graded.document, ids.points.B, [0.416, 0]);
    expectPointAt(deriveSize(reordered, "size/g5").graded.document, ids.points.B, [0.408, 0]);
  });
});

// ---------------------------------------------------------------------------
// Geometry attacks
// ---------------------------------------------------------------------------

describe("geometry attacks", () => {
  it("rejects a grade delta that folds the panel into a bowtie, leaving the master untouched", () => {
    const { doc, ids } = hemDocument([[0.01, 0], [0.01, 0], [0.01, 0]], "per-size");
    const before = masterSnapshot(doc);
    const doc2 = setRuleDelta(doc, "rule/hem", "size/g3", [-0.2, 1.2]);
    const error = gradingError(() => deriveSize(doc2, "size/g3"));
    expect(error.code).toBe("invalid-document");
    expect(error.message).toMatch(/self-intersect/i);
    expectPointAt(doc2.master.document, ids.points.B, [0.4, 0]);
    expect(masterSnapshot(doc2)).toBe(before);
  });

  it("rejects a balanced bowtie whose signed area cancels to zero", () => {
    const { doc } = hemDocument([[0.01, 0], [0.01, 0], [0.01, 0]], "per-size");
    const doc2 = setRuleDelta(doc, "rule/hem", "size/g3", [-0.2, 0.9]);
    const error = gradingError(() => deriveSize(doc2, "size/g3"));
    expect(error.code).toBe("invalid-document");
    expect(error.message).toMatch(/degenerate-panel|self-intersect/i);
  });

  it("rejects a grade delta that collapses a boundary edge to zero length", () => {
    const { doc, ids } = hemDocument([[0.01, 0], [0.01, 0], [0.01, 0]], "per-size");
    const doc2 = setRuleDelta(doc, "rule/hem", "size/g3", [-0.4, 0]);
    const error = gradingError(() => deriveSize(doc2, "size/g3"));
    expect(error.code).toBe("invalid-document");
    expect(error.message).toMatch(/degenerate|zero-length/i);
    expectPointAt(doc2.master.document, ids.points.B, [0.4, 0]);
  });

  it("keeps sub-micron grade deltas accurate with no measurable drift over ten sizes", () => {
    const { doc, ids } = hemDocument(
      Array.from({ length: 10 }, () => [1e-6, 0] as [number, number]),
    );
    for (let k = 1; k <= 10; k++) {
      const { graded } = deriveSize(doc, `size/g${k}`);
      const x = pointPosition(graded.document, ids.points.B)[0];
      expect(Math.abs(x - (0.4 + k * 1e-6))).toBeLessThan(1e-12);
    }
  });

  it("reports direction-flipping transition rules as diagnostics while still deriving exact positions", () => {
    const { doc, ids } = hemDocument([[0.01, 0], [-0.01, 0], [0.01, 0]]);
    const { graded, report } = deriveSize(doc, "size/g3");
    expectPointAt(graded.document, ids.points.B, [0.41, 0]);
    expect(report.diagnostics.filter((d) => d.code === "inconsistent-direction").length).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// Production sidecar attacks
// ---------------------------------------------------------------------------

describe("production sidecar attacks", () => {
  it("rejects a negative seam allowance injected into the production sidecar", () => {
    const { doc } = productionDocument();
    const hostile = corrupted(doc, (draft) => {
      draft.production!.allowances[0].defaultM = -0.01;
    });
    const diagnostics = validateGradingDocument(hostile);
    expect(diagnostics.some((d) => d.code === "invalid-document" && /production set: .*allowance/.test(d.message))).toBe(true);
    expect(gradingError(() => deriveSize(hostile, "size/s")).code).toBe("invalid-document");
  });

  it("surfaces orphaned marking anchors as explicit issues instead of silent drops", () => {
    const { doc, ids } = productionDocument();
    const hostile = corrupted(doc, (draft) => {
      draft.production!.grainlines = [];
    });
    expect(validateGradingDocument(hostile).some((d) =>
      d.code === "unknown-entity" && /ma\/front-grainline/.test(d.message))).toBe(true);
    expect(gradingError(() => deriveSize(hostile, "size/s")).code).toBe("invalid-document");
    const derived = deriveProductionSet(hostile, hostile.master.document, "size/s");
    expect(derived.issues.some((issue) => issue.code === "unknown-marking-anchor")).toBe(true);
  });

  it("keeps notch identity and arclength fraction stable, resolving against graded geometry", () => {
    const { doc, ids, notchId } = productionDocument();
    const { graded } = deriveSize(doc, "size/l");
    const derived = deriveProductionSet(doc, graded.document, "size/l");
    const notch = derived.set.notches.find((candidate) => candidate.id === notchId)!;
    expect(notch.t).toBe(0.5);
    expect(notch.segmentId).toBe(ids.segments.ab);
    const frame = notchFrame(graded.document, notch);
    expect(frame.pos[0]).toBeCloseTo(0.215, 10);
    expect(frame.pos[1]).toBeCloseTo(0, 10);
  });

  it("preserves every production metadata collection through per-size derivation", () => {
    const { doc, grainlineIds } = productionDocument();
    const { graded } = deriveSize(doc, "size/l");
    const derived = deriveProductionSet(doc, graded.document, "size/l");
    expect(derived.set.allowances).toHaveLength(2);
    expect(derived.set.notches).toHaveLength(1);
    expect(derived.set.grainlines).toHaveLength(2);
    expect(derived.set.drills).toHaveLength(1);
    expect(derived.set.annotations).toHaveLength(1);
    expect(derived.set.panelMeta.map((meta) => meta.cutQuantity).sort()).toEqual([1, 2]);
    const front = derived.set.grainlines.find((grainline) => grainline.id === grainlineIds.front)!;
    expect(front.from).toEqual([0.18, 0.25]);
    expect(front.to).toEqual([0.28, 0.35]);
    const back = derived.set.grainlines.find((grainline) => grainline.id === grainlineIds.back)!;
    expect(back.from).toEqual([0.55, 0.25]);
    expect(derived.issues).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// Persistence attacks
// ---------------------------------------------------------------------------

describe("persistence attacks", () => {
  it("regenerating five times yields byte-identical derived output", () => {
    const fixture = buildGradingFixture();
    let doc = addSize(fixture.doc, createSize({ id: "size/s", label: "S" }));
    doc = addGradingPoint(doc, createGradingPoint("gp/hem-side", {
      kind: "corner", panelId: fixture.ids.frontPanel,
      segmentIdA: fixture.ids.segments.ab, segmentIdB: fixture.ids.segments.bc,
    }));
    doc = addRule(doc, createRule("rule/hem", "gp/hem-side", "transition", { "size/s": [0.0123456, 0] }));
    const outputs: string[] = [];
    for (let attempt = 0; attempt < 5; attempt++) {
      const { document: regenerated } = regenerateAll(doc);
      outputs.push(canonicalJson(regenerated.graded));
    }
    for (const output of outputs) expect(output).toBe(outputs[0]);
  });

  it("save/reload round-trip reproduces derived output byte-for-byte", () => {
    const { doc } = productionDocument();
    const regenerated = regenerateAll(doc).document;
    const serialized = serializeGradingDocument(regenerated);
    const parsed = parseGradingDocument(serialized);
    expect(fingerprintGradingDocument(parsed)).toBe(fingerprintGradingDocument(regenerated));
    const fresh = deriveSize(regenerated, "size/m").graded;
    const reloaded = deriveSize(parsed, "size/m").graded;
    expect(canonicalJson(reloaded)).toBe(canonicalJson(fresh));
  });

  it("refuses to load tampered persistence", () => {
    const { doc } = productionDocument();
    const serialized = serializeGradingDocument(regenerateAll(doc).document);

    const wrongMasterSchema = JSON.parse(serialized) as GradingDocument;
    wrongMasterSchema.master.document.schemaVersion = 2 as never;
    expect(() => parseGradingDocument(JSON.stringify(wrongMasterSchema))).toThrow(GradingError);

    const duplicatePoint = JSON.parse(serialized) as GradingDocument;
    duplicatePoint.gradingPoints.push(JSON.parse(JSON.stringify(duplicatePoint.gradingPoints[0])) as never);
    expect(() => parseGradingDocument(JSON.stringify(duplicatePoint))).toThrow(/duplicated/);
  });
});

// ---------------------------------------------------------------------------
// Export pipeline (G11 gate) attacks
// ---------------------------------------------------------------------------

describe("export pipeline attacks", () => {
  it("every generated size enters the G11 export gate with full metadata and no errors", () => {
    const { doc, ids, seam } = productionDocument();
    const regenerated = regenerateAll(doc).document;
    for (const sizeId of ["size/s", "size/m", "size/l"]) {
      const graded = regenerated.graded.find((candidate) => candidate.sizeId === sizeId)!;
      const derived = deriveProductionSet(regenerated, graded.document, sizeId);
      expect(derived.issues).toHaveLength(0);
      const report = productionReadiness(graded.document, [seam], derived.set);
      expect(report.errorCount).toBe(0);
      expect(report.state).toBe("READY_FOR_EXPORT");
      expect(report.panels.map((row) => row.panelId).sort()).toEqual([ids.backPanel, ids.frontPanel].sort());
    }
  });

  it("the G11 gate rejects over-graded sizes whose anchored markings leave the panel", () => {
    const { doc, seam } = productionDocument();
    const doc2 = setRuleDelta(doc, "rule/hem", "size/s", [-0.3, 0]);
    const doc3 = setRuleDelta(doc2, "rule/neck", "size/s", [-0.3, 0]);
    const doc4 = setRuleDelta(doc3, "rule/cb", "size/s", [-0.3, 0]);
    const doc5 = setRuleDelta(doc4, "rule/he-end", "size/s", [-0.3, 0]);
    const { graded } = deriveSize(doc5, "size/s");
    const derived = deriveProductionSet(doc5, graded.document, "size/s");
    expect(derived.issues.some((issue) => issue.code === "outside-panel")).toBe(true);
    const report = productionReadiness(graded.document, [seam], derived.set);
    expect(report.state).toBe("INVALID");
    expect(report.diagnostics.some((d) => /outside-panel/.test(d.code))).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Master-pattern integrity under the full workflow
// ---------------------------------------------------------------------------

describe("master-pattern integrity", () => {
  it("the master pattern survives the full grading workflow byte-for-byte", () => {
    let { doc } = productionDocument();
    const before = masterSnapshot(doc);
    const fingerprintBefore = fingerprintGradingDocument(doc);

    doc = regenerateAll(doc).document;
    expect(masterSnapshot(doc)).toBe(before);
    for (const sizeId of ["size/s", "size/m", "size/l"]) {
      const graded = doc.graded.find((candidate) => candidate.sizeId === sizeId)!;
      deriveProductionSet(doc, graded.document, sizeId);
    }
    createSizeView(doc, { ...defaultSizeViewOptions(), mode: "overlay" });
    documentForSize(doc, "size/l");
    expect(masterSnapshot(doc)).toBe(before);

    doc = insertSize(doc, createSize({ id: "size/xl", label: "XL" }), 3);
    for (const [ruleId, delta] of [
      ["rule/hem", [0.04, 0]], ["rule/neck", [0.04, 0]], ["rule/cb", [0.04, 0]], ["rule/he-end", [0.04, 0]],
    ] as Array<[string, [number, number]]>) {
      doc = setRuleDelta(doc, ruleId, "size/xl", delta);
    }
    expect(deriveSize(doc, "size/xl").graded.sizeId).toBe("size/xl");
    expect(masterSnapshot(doc)).toBe(before);

    doc = removeSize(doc, "size/s");
    expect(validateGradingDocument(doc)).toHaveLength(0);
    expect(masterSnapshot(doc)).toBe(before);

    doc = parseGradingDocument(serializeGradingDocument(doc));
    expect(masterSnapshot(doc)).toBe(before);
    expect(fingerprintGradingDocument(doc)).not.toBe(fingerprintBefore);

    doc = regenerateAll(doc).document;
    doc = regenerateAll(doc).document;
    expect(masterSnapshot(doc)).toBe(before);
    expect(doc.graded.map((graded) => graded.sizeId).sort()).toEqual(["size/l", "size/m", "size/xl"]);
  });
});
