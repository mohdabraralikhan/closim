// G12 FINAL INTEGRATION — commercial multi-size product.
//
// The existing production-ready 4-panel top (G11 layout, 5 seams) becomes a
// graded product: XS–XXL derived from the master size M via body-width and
// body-length grade rules, with a full production sidecar (allowances,
// grainlines, notches, cut lines, fold, anchored drill, dart, annotation,
// label, panel metadata). The 16-step commercial acceptance workflow is
// exercised end to end; the master pattern stays authoritative throughout.
import { describe, expect, it } from "vitest";
import { createPatternDocument, movePoint, type PatternDocument } from "../../src/pattern/cad.js";
import { addRectPanel } from "../../src/garment/tshirt.js";
import type { Seam } from "../../src/garment/sewing.js";
import {
  addAllowance,
  addAnnotation,
  addCutLine,
  addDrillMark,
  addFoldLine,
  addGrainline,
  addInternalLine,
  addLabelRegion,
  addNotch,
  createProductionSet,
  setPanelMeta,
} from "../../src/cad/production.js";
import { buildLabelFields, centeredGrainline } from "../../src/cad/markings.js";
import { productionReadiness } from "../../src/cad/readiness.js";
import { compareIR, exportProductionPackage, importIRFromJSON } from "../../src/cad/export.js";
import {
  addGradingPoint,
  addMarkingAnchor,
  addMeasurementDefinition,
  addRule,
  addSize,
  assignMeasurement,
  assignProduction,
  canonicalJson,
  createGradingDocument,
  createGradingPoint,
  createMarkingAnchor,
  createMasterPattern,
  createMeasurementDefinition,
  createRule,
  createRuleTable,
  createSize,
  createSizeSet,
  deriveProductionSet,
  deriveSize,
  isStale,
  parseGradingDocument,
  regenerateAll,
  serializeGradingDocument,
  setBaseSize,
  validateGradingDocument,
  type GradingDocument,
} from "../../src/grading/index.js";
import { compareSizes, createSizeView, defaultSizeViewOptions, documentForSize } from "../../src/grading/presentation.js";

interface SizeSpec {
  id: string;
  label: string;
  /** Target offset of the front-right edge from the master (chest grows by 2·dx). */
  dx: number;
  /** Target hem offset from the master, negative = longer garment (length = 0.62 − dy). */
  dy: number;
  chest: number;
  length: number;
}

// Target offsets per size relative to master M. Transition rules consume
// consecutive differences (STEP below), so the cumulative grade at each size
// lands exactly on its target and the base size M stays on the master.
const SIZES: SizeSpec[] = [
  { id: "size/xs", label: "XS", dx: -0.03, dy: 0.04, chest: 0.86, length: 0.58 },
  { id: "size/s", label: "S", dx: -0.015, dy: 0.02, chest: 0.89, length: 0.6 },
  { id: "size/m", label: "M", dx: 0, dy: 0, chest: 0.92, length: 0.62 },
  { id: "size/l", label: "L", dx: 0.015, dy: -0.02, chest: 0.95, length: 0.64 },
  { id: "size/xl", label: "XL", dx: 0.03, dy: -0.04, chest: 0.98, length: 0.66 },
  { id: "size/xxl", label: "XXL", dx: 0.045, dy: -0.06, chest: 1.01, length: 0.68 },
];

const STEP: Array<{ dx: number; dy: number }> = SIZES.map((spec, index) =>
  index === 0
    ? { dx: spec.dx, dy: spec.dy }
    : { dx: spec.dx - SIZES[index - 1].dx, dy: spec.dy - SIZES[index - 1].dy },
);

const stepTable = (fn: (step: { dx: number; dy: number }) => [number, number]): Record<string, [number, number]> =>
  Object.fromEntries(STEP.map((step, index) => [SIZES[index].id, fn(step)]));

function panels() {
  let document = createPatternDocument("g12-top", "Graded production top");
  const front = addRectPanel(document, "front", [0, 0], 0.46, 0.62);
  document = front.document;
  const back = addRectPanel(document, "back", [0.6, 0], 0.46, 0.62);
  document = back.document;
  const sleeveL = addRectPanel(document, "sleeve-left", [1.3, 0], 0.3, 0.62);
  document = sleeveL.document;
  const sleeveR = addRectPanel(document, "sleeve-right", [1.7, 0], 0.3, 0.62);
  document = sleeveR.document;
  return { document, front: front.refs, back: back.refs, sleeveL: sleeveL.refs, sleeveR: sleeveR.refs };
}

function seams(p: ReturnType<typeof panels>): Seam[] {
  const side = (r: { panelId: string; loopId: string; segmentIds: string[] }, k: number, reversed = false) => ({
    panelId: r.panelId, loopId: r.loopId, segmentIds: [r.segmentIds[k]], reversed,
  });
  return [
    { id: "seam/shoulder", sideA: side(p.front, 2), sideB: side(p.back, 2, true), stitchCount: 7 },
    { id: "seam/left", sideA: side(p.front, 3), sideB: side(p.back, 3, true), stitchCount: 7 },
    { id: "seam/right", sideA: side(p.front, 1), sideB: side(p.back, 1, true), stitchCount: 7 },
    { id: "seam/sleeve-l", sideA: side(p.front, 3), sideB: side(p.sleeveL, 1), stitchCount: 5 },
    { id: "seam/sleeve-r", sideA: side(p.front, 1), sideB: side(p.sleeveR, 3, true), stitchCount: 5 },
  ];
}

/** Master edit: widen the body by 1 cm (front edges right, back edges left). */
function widenMaster(document: PatternDocument, p: ReturnType<typeof panels>): PatternDocument {
  let moved = movePoint(document, p.front.panelId, p.front.pointIds[1], [0.47, 0], "local");
  moved = movePoint(moved, p.front.panelId, p.front.pointIds[2], [0.47, 0.62], "local");
  moved = movePoint(moved, p.back.panelId, p.back.pointIds[0], [0.59, 0], "local");
  moved = movePoint(moved, p.back.panelId, p.back.pointIds[3], [0.59, 0.62], "local");
  return moved;
}

function buildCommercialDocument(): {
  doc: GradingDocument;
  p: ReturnType<typeof panels>;
  seamList: Seam[];
  drillId: string;
  frontGrainlineId: string;
} {
  const p = panels();
  const seamList = seams(p);
  let doc = createGradingDocument({
    id: "grading/top",
    name: "Graded production top",
    master: createMasterPattern("master/top", "Production top master", p.document),
    sizeSet: createSizeSet("sizeset/top", "Top sizes XS-XXL"),
    ruleTable: createRuleTable("ruletable/top", "Top grade rules"),
    seams: seamList,
  });

  // Size table + base size.
  for (const spec of SIZES) {
    doc = addSize(doc, createSize({ id: spec.id, label: spec.label, displayName: spec.label === "M" ? "Medium" : spec.label }));
  }
  doc = setBaseSize(doc, "size/m");

  // Body measurement catalogue + per-size values (canonical metres).
  doc = addMeasurementDefinition(doc, createMeasurementDefinition({ id: "chest", name: "Chest circumference", unit: "m", ordering: "increasing" }));
  doc = addMeasurementDefinition(doc, createMeasurementDefinition({ id: "center-back-length", name: "Center back length", unit: "m", ordering: "increasing" }));
  SIZES.forEach((spec) => {
    doc = assignMeasurement(doc, spec.id, "chest", spec.chest, { unit: "m", source: "size chart" });
    doc = assignMeasurement(doc, spec.id, "center-back-length", spec.length, { unit: "m", source: "size chart" });
  });

  // Grading points on the master (rect segment convention: 0 bottom, 1 right, 2 top, 3 left).
  doc = addGradingPoint(doc, createGradingPoint("gp/front-hem-right", { kind: "corner", panelId: p.front.panelId, segmentIdA: p.front.segmentIds[0], segmentIdB: p.front.segmentIds[1] }));
  doc = addGradingPoint(doc, createGradingPoint("gp/front-top-right", { kind: "corner", panelId: p.front.panelId, segmentIdA: p.front.segmentIds[1], segmentIdB: p.front.segmentIds[2] }));
  doc = addGradingPoint(doc, createGradingPoint("gp/front-hem-left", { kind: "corner", panelId: p.front.panelId, segmentIdA: p.front.segmentIds[0], segmentIdB: p.front.segmentIds[3] }));
  doc = addGradingPoint(doc, createGradingPoint("gp/back-hem-left", { kind: "corner", panelId: p.back.panelId, segmentIdA: p.back.segmentIds[0], segmentIdB: p.back.segmentIds[3] }));
  doc = addGradingPoint(doc, createGradingPoint("gp/back-hem-right", { kind: "corner", panelId: p.back.panelId, segmentIdA: p.back.segmentIds[0], segmentIdB: p.back.segmentIds[1] }));
  doc = addGradingPoint(doc, createGradingPoint("gp/back-top-left", { kind: "corner", panelId: p.back.panelId, segmentIdA: p.back.segmentIds[2], segmentIdB: p.back.segmentIds[3] }));
  doc = addGradingPoint(doc, createGradingPoint("gp/sleeve-l-top", { kind: "corner", panelId: p.sleeveL.panelId, segmentIdA: p.sleeveL.segmentIds[1], segmentIdB: p.sleeveL.segmentIds[2] }));
  doc = addGradingPoint(doc, createGradingPoint("gp/sleeve-r-top", { kind: "corner", panelId: p.sleeveR.panelId, segmentIdA: p.sleeveR.segmentIds[2], segmentIdB: p.sleeveR.segmentIds[3] }));

  // Transition grade rules (per-step deltas over the STEP table; hem moves
  // opposite the sleeve top so all five seam sides stay equal for every size).
  doc = addRule(doc, createRule("rule/hem-right", "gp/front-hem-right", "transition", stepTable((s) => [s.dx, s.dy])));
  doc = addRule(doc, createRule("rule/hem-left", "gp/front-hem-left", "transition", stepTable((s) => [0, s.dy])));
  doc = addRule(doc, createRule("rule/back-hem-left", "gp/back-hem-left", "transition", stepTable((s) => [-s.dx, s.dy])));
  doc = addRule(doc, createRule("rule/back-hem-right", "gp/back-hem-right", "transition", stepTable((s) => [0, s.dy])));
  doc = addRule(doc, createRule("rule/top-right", "gp/front-top-right", "transition", stepTable((s) => [s.dx, 0])));
  doc = addRule(doc, createRule("rule/back-top", "gp/back-top-left", "transition", stepTable((s) => [-s.dx, 0])));
  doc = addRule(doc, createRule("rule/sleeve-l", "gp/sleeve-l-top", "transition", stepTable((s) => [0, -s.dy])));
  doc = addRule(doc, createRule("rule/sleeve-r", "gp/sleeve-r-top", "transition", stepTable((s) => [0, -s.dy])));

  // Production sidecar on the master.
  let set = createProductionSet();
  let frontGrainlineId = "";
  for (const r of [p.front, p.back, p.sleeveL, p.sleeveR]) {
    set = addAllowance(set, r.panelId, r.loopId, 0.01).set;
    const g = centeredGrainline(p.document, r.panelId);
    const grain = addGrainline(set, r.panelId, g.from, g.to);
    set = grain.set;
    if (r === p.front) frontGrainlineId = grain.id;
    set = addNotch(set, r.panelId, r.loopId, r.segmentIds[0], 0.5, "single", 0.005).set;
    set = addCutLine(set, r.panelId, r.loopId, "sewing").set;
  }
  set = addFoldLine(set, p.front.panelId, [0.1, 0.1], [0.1, 0.5], "valley", "pleat").set;
  const drill = addDrillMark(set, p.front.panelId, [0.23, 0.31], "circle");
  set = drill.set;
  set = addInternalLine(set, p.front.panelId, [[0.1, 0.1], [0.2, 0.2]], "dart", "front dart").set;
  set = addAnnotation(set, p.front.panelId, [0.23, 0.4], "match pocket").set;
  const labelFields = buildLabelFields({
    garmentName: "Graded production top", panelName: "front", panelNumber: 1,
    size: "M", cutQuantity: 2, material: "cotton",
  });
  set = addLabelRegion(set, p.front.panelId, [0.05, 0.05], [0.41, 0.14], labelFields).set;
  set = setPanelMeta(set, { panelId: p.front.panelId, cutQuantity: 2, section: "body" });
  set = setPanelMeta(set, { panelId: p.back.panelId, cutQuantity: 2, section: "body" });
  set = setPanelMeta(set, { panelId: p.sleeveL.panelId, cutQuantity: 1, section: "sleeve", mirrorPair: "sleeve-right" });
  set = setPanelMeta(set, { panelId: p.sleeveR.panelId, cutQuantity: 1, section: "sleeve", mirrorPair: "sleeve-left" });
  set = addCutLine(set, p.front.panelId, p.front.loopId, "allowance").set;

  doc = assignProduction(doc, set);
  // Anchored markings follow their grading point; everything else stays verbatim.
  doc = addMarkingAnchor(doc, createMarkingAnchor("ma/front-drill", p.front.panelId, "drill", drill.id, "gp/front-hem-left"));
  doc = addMarkingAnchor(doc, createMarkingAnchor("ma/front-grainline", p.front.panelId, "grainline", frontGrainlineId, "gp/front-top-right"));

  return { doc, p, seamList, drillId: drill.id, frontGrainlineId };
}

function pointPosition(doc: PatternDocument, pointId: string): [number, number] {
  const point = doc.points.find((candidate) => candidate.id === pointId);
  if (!point) throw new Error(`point ${pointId} missing`);
  return [point.x, point.y];
}

describe("G12 final integration — commercial multi-size product", () => {
  it("runs the full 16-step commercial workflow on the production-ready top", () => {
    const workflowStarted = Date.now();

    // 1-6. Open garment; master size M; size table; body measurements;
    // grading points; grade rules.
    const { doc, p, seamList, drillId, frontGrainlineId } = buildCommercialDocument();
    expect(doc.sizeSet.baseSizeId).toBe("size/m");
    expect(doc.sizeSet.sizes.map((size) => size.label)).toEqual(["XS", "S", "M", "L", "XL", "XXL"]);
    expect(doc.sizeSet.measurementDefinitions.map((definition) => definition.id)).toEqual(["chest", "center-back-length"]);
    expect(doc.gradingPoints).toHaveLength(8);
    expect(doc.ruleTable.rules).toHaveLength(8);

    // 7. Generate sizes.
    const generationStarted = Date.now();
    const generated = regenerateAll(doc).document;
    const generationMs = Date.now() - generationStarted;
    expect(generated.graded.map((graded) => graded.sizeId).sort()).toEqual(SIZES.map((s) => s.id).sort());

    // 8. Validate every size (document, geometry, seams).
    expect(validateGradingDocument(generated)).toHaveLength(0);
    SIZES.forEach((spec) => {
      const graded = generated.graded.find((candidate) => candidate.sizeId === spec.id)!;
      expect(isStale(generated, graded)).toBe(false);
      const hemRight = pointPosition(graded.document, p.front.pointIds[1]);
      expect(hemRight[0]).toBeCloseTo(0.46 + spec.dx, 10);
      expect(hemRight[1]).toBeCloseTo(spec.dy, 10);
      const sleeveTop = pointPosition(graded.document, p.sleeveL.pointIds[2]);
      expect(sleeveTop[1]).toBeCloseTo(0.62 - spec.dy, 10);
    });

    // 9. Inspect nested sizes (all six + master, overlay order = size order).
    const view = createSizeView(generated, { ...defaultSizeViewOptions(), mode: "overlay" });
    expect(view.layers.map((layer) => layer.sizeId)).toEqual(SIZES.map((s) => s.id));
    expect(view.layers.every((layer) => layer.visible && !layer.invalid)).toBe(true);
    expect(view.masterLayer).not.toBeNull();

    // 10. Inspect grading differences (M vs XXL).
    const comparison = compareSizes(generated, "size/m", "size/xxl");
    expect(comparison.maxPointDistanceM).toBeCloseTo(Math.hypot(0.045, 0.06), 10);
    const chest = comparison.measurementDeltas.find((delta) => delta.measurementId === "chest")!;
    expect(chest.valueAM).toBeCloseTo(0.92, 10);
    expect(chest.valueBM).toBeCloseTo(1.01, 10);
    expect(chest.deltaM).toBeCloseTo(0.09, 10);
    for (const seamDelta of comparison.seamLengthDeltas) {
      // Size-to-size growth: shoulder follows the width grade, sides and
      // sleeve attachments follow the length grade.
      const expected = seamDelta.seamId === "seam/shoulder" ? 0.045 : 0.06;
      expect(seamDelta.diffM).toBeCloseTo(expected, 10);
    }

    // 11. Return to master.
    expect(documentForSize(generated, "master")).toBe(generated.master.document);

    // 12. Modify the master pattern (widen body 1 cm) — sizes stay derived.
    const widenedMaster = widenMaster(generated.master.document, p);
    const edited = { ...generated, master: { ...generated.master, document: widenedMaster } };
    expect(isStale(edited, generated.graded[0])).toBe(true);
    expect(JSON.stringify(edited.master.document)).not.toBe(JSON.stringify(generated.master.document));

    // 13-14. Regenerate the size set and validate again: every graded size is
    // the widened master + its grade (deltas survive the master edit).
    const regenerated = regenerateAll(edited).document;
    expect(validateGradingDocument(regenerated)).toHaveLength(0);
    const xxl = regenerated.graded.find((candidate) => candidate.sizeId === "size/xxl")!;
    const hemRightXXL = pointPosition(xxl.document, p.front.pointIds[1]);
    expect(hemRightXXL[0]).toBeCloseTo(0.47 + 0.045, 10);
    expect(hemRightXXL[1]).toBeCloseTo(-0.06, 10);
    expect(pointPosition(xxl.document, p.front.pointIds[2])).toEqual([
      expect.closeTo(0.47 + 0.045, 10),
      expect.closeTo(0.62, 10),
    ]);
    expect(pointPosition(xxl.document, p.back.pointIds[0])).toEqual([
      expect.closeTo(0.59 - 0.045, 10),
      expect.closeTo(-0.06, 10),
    ]);
    expect(pointPosition(xxl.document, p.back.pointIds[3])).toEqual([
      expect.closeTo(0.59 - 0.045, 10),
      expect.closeTo(0.62, 10),
    ]);
    // The base size M tracks the edited master exactly (no grade residue).
    const m = regenerated.graded.find((candidate) => candidate.sizeId === "size/m")!;
    expect(pointPosition(m.document, p.front.pointIds[1])[0]).toBeCloseTo(0.47, 10);
    expect(pointPosition(m.document, p.back.pointIds[0])[0]).toBeCloseTo(0.59, 10);

    // 15. Send every size through production validation.
    for (const spec of SIZES) {
      const graded = regenerated.graded.find((candidate) => candidate.sizeId === spec.id)!;
      const derived = deriveProductionSet(regenerated, graded.document, spec.id);
      expect(derived.issues).toEqual([]);
      expect(derived.set.allowances).toHaveLength(4);
      expect(derived.set.notches).toHaveLength(4);
      expect(derived.set.grainlines).toHaveLength(4);
      expect(derived.set.drills).toHaveLength(1);
      expect(derived.set.panelMeta).toHaveLength(4);
      const anchoredDrill = derived.set.drills.find((d) => d.id === drillId)!;
      expect(anchoredDrill.pos[1]).toBeCloseTo(0.31 + spec.dy, 10);
      const anchoredGrainline = derived.set.grainlines.find((g) => g.id === frontGrainlineId)!;
      expect(anchoredGrainline.from[0]).toBeCloseTo(centeredGrainline(p.document, p.front.panelId).from[0] + spec.dx, 10);
      const readiness = productionReadiness(graded.document, seamList, derived.set);
      expect(readiness.errorCount).toBe(0);
      expect(readiness.state).toBe("READY_FOR_EXPORT");
      expect(readiness.seams.every((row) => row.withinTolerance)).toBe(true);
      expect(readiness.panels).toHaveLength(4);
      expect(readiness.totalCutAreaM2).toBeGreaterThan(0);
    }

    // 16. Export the selected size (XXL) and the complete size set.
    const pkg = exportProductionPackage(xxl.document, seamList, deriveProductionSet(regenerated, xxl.document, "size/xxl").set, {
      garmentName: "Graded production top XXL",
    });
    expect(pkg.json.length).toBeGreaterThan(1000);
    expect(pkg.dxf.entityCount).toBeGreaterThan(50);
    expect(pkg.dxf.dxf).toContain("CUT");
    expect(compareIR(pkg.ir, importIRFromJSON(pkg.json))).toEqual([]);
    const pkgAgain = exportProductionPackage(xxl.document, seamList, deriveProductionSet(regenerated, xxl.document, "size/xxl").set, {
      garmentName: "Graded production top XXL",
    });
    expect(pkgAgain.dxf.dxf).toBe(pkg.dxf.dxf);
    for (const spec of SIZES) {
      const graded = regenerated.graded.find((candidate) => candidate.sizeId === spec.id)!;
      const derived = deriveProductionSet(regenerated, graded.document, spec.id);
      const sizePkg = exportProductionPackage(graded.document, seamList, derived.set, {
        garmentName: `Graded production top ${spec.label}`,
      });
      expect(compareIR(sizePkg.ir, importIRFromJSON(sizePkg.json))).toEqual([]);
      expect(sizePkg.dxf.entityCount).toBeGreaterThan(50);
    }

    const workflowMs = Date.now() - workflowStarted;
    expect(generationMs).toBeLessThan(2000);
    expect(workflowMs).toBeLessThan(30000);
  });

  it("persists the complete commercial document and reproduces derived output", () => {
    const { doc, seamList } = buildCommercialDocument();
    const generated = regenerateAll(doc).document;
    const serialized = serializeGradingDocument(generated);
    const parsed = parseGradingDocument(serialized);

    expect(validateGradingDocument(parsed)).toHaveLength(0);
    expect(parsed.master.document.id).toBe(generated.master.document.id);
    expect(parsed.sizeSet.sizes).toHaveLength(6);
    expect(parsed.sizeSet.baseSizeId).toBe("size/m");
    expect(parsed.sizeSet.measurementDefinitions).toHaveLength(2);
    for (const size of parsed.sizeSet.sizes) {
      expect(size.measurements.map((entry) => entry.measurementId).sort()).toEqual(["center-back-length", "chest"]);
    }
    expect(parsed.gradingPoints).toHaveLength(8);
    expect(parsed.ruleTable.rules).toHaveLength(8);
    expect(parsed.markingAnchors ?? []).toHaveLength(2);
    expect(parsed.graded).toHaveLength(6);

    const fresh = deriveSize(generated, "size/xxl");
    const reloaded = deriveSize(parsed, "size/xxl");
    expect(canonicalJson(reloaded.graded)).toBe(canonicalJson(fresh.graded));
    expect(reloaded.report.diagnostics).toEqual(fresh.report.diagnostics);
    const derivedReloaded = deriveProductionSet(parsed, reloaded.graded.document, "size/xxl");
    const readiness = productionReadiness(reloaded.graded.document, seamList, derivedReloaded.set);
    expect(readiness.state).toBe("READY_FOR_EXPORT");
  });
});
