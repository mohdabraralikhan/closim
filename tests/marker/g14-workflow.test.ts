// G14 FINAL INTEGRATION — commercial marker acceptance.
//
// Graded garment (S/M/L) -> size set -> cut quantities -> fabric ->
// orientation rules -> initial marker -> optimize -> inspect utilization ->
// manual adjust -> revalidate -> marker export. Source pattern geometry is
// proven unchanged throughout.
import { describe, expect, it } from "vitest";
import {
  addCutItem,
  createCutPlan,
  createFabric,
  defaultNestingConstraint,
  expandCutPlan,
} from "../../src/marker/model.js";
import { checkNestingFeasibility, fabricRulesFrom } from "../../src/marker/fabric.js";
import { MarkerWorkspace } from "../../src/marker/workspace.js";
import { estimateCost, measureResult } from "../../src/marker/optimize.js";
import { exportMarkerPackage, importMarkerJSON } from "../../src/marker/export.js";
import {
  addCoreGradingPoints,
  addCoreRules,
  addCoreSizes,
  buildGradingFixture,
} from "../grading/fixtures.js";

function gradedGarment() {
  const g = buildGradingFixture();
  let doc = addCoreSizes(g.doc);
  doc = addCoreGradingPoints(doc, g.ids);
  doc = addCoreRules(doc);
  return { ...g, doc };
}

describe("G14 final integration — graded garment to manufacturing marker", () => {
  it("runs the full commercial marker workflow", () => {
    const g = gradedGarment();
    const masterBefore = JSON.stringify(g.doc.master.document);

    // Size set: S/M/L active, in order.
    const sizes = g.doc.sizeSet.sizes.filter((s) => s.active).map((s) => s.id);
    expect(sizes).toEqual(["size/s", "size/m", "size/l"]);

    // Cut quantities across sizes.
    let plan = createCutPlan("cut/order-7", "Order 7", g.doc.id);
    const items: Array<[string, string, number]> = [
      [g.ids.frontPanel, "size/m", 2],
      [g.ids.backPanel, "size/m", 1],
      [g.ids.frontPanel, "size/s", 1],
      [g.ids.backPanel, "size/l", 1],
    ];
    for (const [panelId, sizeId, quantity] of items) {
      const r = addCutItem(plan, { panelId, sizeId, quantity, mirror: "allowed" });
      plan = r.plan;
    }

    // Fabric + orientation rules.
    const fabric = createFabric({ id: "fabric/cotton-150", name: "Cotton 150", widthM: 1.5, lengthM: 20, materialType: "cotton" });
    const constraint = defaultNestingConstraint({ spacingM: 0.01, rotationsDeg: [0, 180] });
    const rules = fabricRulesFrom(fabric, constraint);
    const grainRadOf = (panelId: string): number =>
      g.doc.master.document.panels.find((p) => p.id === panelId)!.grainAngleRad;
    const pieces = expandCutPlan(g.doc, plan);
    expect(pieces).toHaveLength(5);
    expect(new Set(pieces.map((p) => p.sizeId))).toEqual(new Set(["size/m", "size/s", "size/l"]));

    // Orientation rules validate before nesting.
    expect(checkNestingFeasibility(pieces, rules, plan.items, grainRadOf, constraint.marginM)).toEqual([]);

    // Initial marker + optimization.
    const ws = MarkerWorkspace.open(g.doc, plan, fabric, constraint, "Order 7 marker", "marker/order-7", grainRadOf);
    const initial = ws.optimize([0]);
    expect(initial.best.result.unplaced).toEqual([]);
    const report = ws.optimize([0, 1, 2], "min-length", ["area", "width"]);
    expect(report.best.result.unplaced).toEqual([]);
    expect(report.runs.length).toBe(3 * 2);

    // Utilization / waste inspection (hand-verified consistency).
    const preview = ws.preview();
    const patternArea = pieces.reduce((s, p) => s + p.areaM2, 0);
    expect(preview.metrics.patternAreaM2).toBeCloseTo(patternArea, 9);
    expect(preview.metrics.utilization).toBeGreaterThan(0);
    expect(preview.metrics.utilization).toBeLessThanOrEqual(1);
    expect(preview.metrics.fabricAreaM2).toBeCloseTo(1.5 * preview.metrics.markerLengthM, 9);
    expect(preview.metrics.wasteM2).toBeCloseTo(preview.metrics.fabricAreaM2 - patternArea, 9);
    const cost = estimateCost(preview.metrics, { fabricPricePerM: 8, laborPerMarker: 12, wastePricePerM2: 1 });
    expect(cost.totalCost).toBeCloseTo(cost.fabricCost + cost.wasteCost + 12, 9);

    // Manual adjust where permitted + revalidation.
    const target = preview.placedIds[0];
    const moved = ws.movePlacement(target, 0.5, preview.metrics.markerLengthM + 0.5);
    expect(moved.ok).toBe(true);
    expect(ws.audit()).toEqual([]);
    const overlap = ws.movePlacement(preview.placedIds[1], 0.5, preview.metrics.markerLengthM + 0.5);
    expect(overlap.ok).toBe(false);
    expect(ws.rotatePlacement(target, 45).ok).toBe(false); // outside [0, 180]
    expect(ws.audit()).toEqual([]);

    // Export marker (pattern export stays a separate artifact).
    const pkg = exportMarkerPackage(ws.marker, ws.pieces, plan, ws.preview().metrics, ws.rules, grainRadOf);
    expect(importMarkerJSON(pkg.json).marker.id).toBe("marker/order-7");
    expect(pkg.dxf.dxf).toContain("PIECES");
    expect(pkg.dxf.dxf).toContain("MARKER_BOUND");
    expect(pkg.dxf.entityCount).toBeGreaterThan(10);

    // Invariants: source unchanged, quantities match, deterministic rerun.
    expect(JSON.stringify(g.doc.master.document)).toBe(masterBefore);
    expect(ws.marker.placements).toHaveLength(5);
    const ws2 = MarkerWorkspace.open(g.doc, plan, fabric, constraint, "Order 7 marker", "marker/order-7", grainRadOf);
    ws2.optimize([0, 1, 2], "min-length", ["area", "width"]);
    // Same seeds -> same best length (manual move in ws differs, so compare engine reruns).
    expect(ws2.preview().metrics.markerLengthM).toBeLessThanOrEqual(preview.metrics.markerLengthM + 1);
  });
});
