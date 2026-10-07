import { describe, expect, it } from "vitest";
import { generateBOMRequirements, validateBOM, type BOM } from "../../src/production/bom.js";
import { calculateCutPlan, createProductionRun } from "../../src/production/run.js";
import { buildGradingFixture, addCoreSizes } from "../grading/fixtures.js";
import type { Marker } from "../../src/marker/model.js";

function markerFor(sizeId: string, panels: string[]): Marker {
  const pieces = panels.map((panelId) => ({
    instanceId: `${sizeId}/${panelId}`,
    cutItemId: `cut/${panelId}`,
    panelId,
    sizeId,
    mirrored: false,
    polygon: [[0, 0], [0.2, 0], [0.2, 0.3], [0, 0.3]] as [number, number][],
    areaM2: 0.06,
    grainRad: Math.PI / 2,
  }));
  return {
    schemaVersion: 1,
    id: `marker/${sizeId}`,
    name: "Adversarial marker",
    fabric: { id: "fabric/main", name: "Shell", widthM: 1.5, usableWidthM: 1.4, lengthM: 10, materialType: "shell" },
    cutPlanId: "cutplan/adversarial",
    constraint: { spacingM: 0.01, marginM: 0.01, rotationsDeg: [0], mirrorDefault: "as-authored", grainToleranceDeg: 0, directionalFabric: false },
    pieces,
    placements: pieces.map((piece, index) => ({
      instanceId: piece.instanceId, x: 0.05, y: 0.01 + index * 0.35, rotationDeg: 0, mirrored: false, manual: false,
    })),
    revision: 1,
  };
}

describe("G18F manufacturing integrity attacks", () => {
  it("accepts zero-sized demand without creating cuts and handles huge demand without expansion", () => {
    const fixture = buildGradingFixture();
    const grading = addCoreSizes(fixture.doc);
    const panelMaterialIds = {
      [fixture.ids.frontPanel]: "fabric/main",
      [fixture.ids.backPanel]: "fabric/main",
    };
    const zero = createProductionRun({
      id: "run/zero", garmentId: "g", styleNumber: "s", revision: "A", gradingId: grading.id,
      status: "planned", sizeQuantities: { "size/s": 0 }, panelMaterialIds,
    });
    const emptyPlan = calculateCutPlan(zero, grading, []);
    expect(emptyPlan.complete).toBe(true);
    expect(emptyPlan.produced["size/s"]).toBe(0);

    const huge = createProductionRun({
      id: "run/huge", garmentId: "g", styleNumber: "s", revision: "A", gradingId: grading.id,
      status: "planned", sizeQuantities: { "size/s": Number.MAX_SAFE_INTEGER }, panelMaterialIds,
    });
    const hugePlan = calculateCutPlan(huge, grading, []);
    expect(hugePlan.complete).toBe(false);
    expect(hugePlan.remaining["size/s"]).toBe(Number.MAX_SAFE_INTEGER);
    expect(hugePlan.cutPieces["size/s"][fixture.ids.frontPanel]).toBe(0);
  });

  it("rejects unsafe quantities and detects unplaced, over-cut, or fabric-incompatible markers", () => {
    const fixture = buildGradingFixture();
    const grading = addCoreSizes(fixture.doc);
    const panelMaterialIds = {
      [fixture.ids.frontPanel]: "fabric/main",
      [fixture.ids.backPanel]: "fabric/main",
    };
    const run = createProductionRun({
      id: "run/attack", garmentId: "g", styleNumber: "s", revision: "A", gradingId: grading.id,
      status: "planned", sizeQuantities: { "size/s": 1 }, panelMaterialIds,
      batches: [{ id: "batch/attack", markerAssignments: [{ markerId: "marker/size/s", fabricId: "fabric/main", repeats: 1 }] }],
    });
    const unplaced = markerFor("size/s", [fixture.ids.frontPanel, fixture.ids.backPanel]);
    unplaced.placements = [];
    const bad = calculateCutPlan(run, grading, [unplaced]);
    expect(bad.diagnostics.map((item) => item.code)).toContain("unplaced-marker-piece");
    expect(bad.complete).toBe(false);

    const incompatible = markerFor("size/s", [fixture.ids.frontPanel, fixture.ids.backPanel]);
    incompatible.fabric.id = "fabric/other";
    const mismatchedRun = {
      ...run,
      batches: [{ id: "batch/attack", markerAssignments: [{ markerId: incompatible.id, fabricId: "fabric/other", repeats: 1 }] }],
    };
    const mismatch = calculateCutPlan(mismatchedRun, grading, [incompatible]);
    expect(mismatch.diagnostics.map((item) => item.code)).toContain("incompatible-material");

    const overcut = markerFor("size/s", [fixture.ids.frontPanel, fixture.ids.backPanel]);
    const overcutRun = {
      ...run,
      batches: [{ id: "batch/attack", markerAssignments: [{ markerId: overcut.id, fabricId: "fabric/main", repeats: 2 }] }],
    };
    expect(calculateCutPlan(overcutRun, grading, [overcut]).diagnostics.map((item) => item.code))
      .toContain("excess-quantity");
    expect(() => createProductionRun({
      id: "unsafe", garmentId: "g", styleNumber: "s", revision: "A", gradingId: grading.id,
      status: "planned", sizeQuantities: { "size/s": Number.MAX_SAFE_INTEGER + 1 }, panelMaterialIds,
    })).toThrow(/safe integer/);
  });

  it("validates ambiguous BOM assignments, missing dependencies, and overflow", () => {
    const bom: BOM = {
      schemaVersion: 1,
      id: "bom/attack",
      garmentId: "g",
      revision: "A",
      materials: [{ id: "fabric/main", name: "Shell", category: "fabric", unit: "m" }],
      items: [
        { id: "item/1", materialId: "fabric/main", basis: "fixed", quantity: 1, unit: "m", wasteAllowance: 0 },
        { id: "item/2", materialId: "fabric/main", basis: "fixed", quantity: 2, unit: "m", wasteAllowance: 0 },
      ],
      markerReferences: [{ markerId: "marker/missing", fabricId: "fabric/main" }],
    };
    expect(validateBOM(bom, { markers: [] }).map((issue) => issue.code)).toContain("duplicate-item");
    expect(validateBOM(bom, { markers: [] }).map((issue) => issue.code)).toContain("missing-marker");
    expect(() => generateBOMRequirements({
      ...bom,
      items: [{ ...bom.items[0], basis: "per-garment", quantity: Number.MAX_VALUE }],
    }, { garmentQuantity: Number.MAX_SAFE_INTEGER })).toThrow(/overflows/);
    expect(() => generateBOMRequirements(bom, {
      garmentQuantity: 1,
      sizeQuantities: { S: -1 },
    })).toThrow(/non-negative/);
  });
});
