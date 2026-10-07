// G14A tests: fabric, cut plan, expansion, marker validation, persistence.
import { describe, expect, it } from "vitest";
import {
  addCutItem,
  createCutPlan,
  createFabric,
  createMarker,
  cutPlanInstances,
  defaultNestingConstraint,
  deserializeCutPlan,
  deserializeMarker,
  expandCutPlan,
  serializeCutPlan,
  serializeMarker,
  validateMarker,
} from "../../src/marker/model.js";
import {
  addCoreGradingPoints,
  addCoreRules,
  addCoreSizes,
  buildGradingFixture,
} from "../grading/fixtures.js";
import { deriveSize } from "../../src/grading/derive.js";

function gradedFixture() {
  let g = buildGradingFixture();
  let doc = addCoreSizes(g.doc);
  doc = addCoreGradingPoints(doc, g.ids);
  doc = addCoreRules(doc);
  return { ...g, doc };
}

describe("G14A fabric and cut plan", () => {
  it("creates fabrics with usable-width rules", () => {
    const fabric = createFabric({ id: "fabric/cotton", name: "Cotton", widthM: 1.5, lengthM: 10, materialType: "cotton" });
    expect(fabric.usableWidthM).toBe(1.5);
    const narrowed = createFabric({ id: "f", name: "n", widthM: 1.5, usableWidthM: 1.4, lengthM: 10, materialType: "c" });
    expect(narrowed.usableWidthM).toBe(1.4);
    expect(() => createFabric({ id: "f", name: "n", widthM: 0, lengthM: 1, materialType: "c" })).toThrowError(/positive/);
    expect(() => createFabric({ id: "f", name: "n", widthM: 1, usableWidthM: 2, lengthM: 1, materialType: "c" })).toThrowError(/usable width/);
  });

  it("builds cut plans without touching the pattern", () => {
    const { doc } = gradedFixture();
    let plan = createCutPlan("cut/main", "Main", doc.id);
    const a = addCutItem(plan, { panelId: "p1", sizeId: "size/m", quantity: 2, mirror: "allowed" });
    plan = a.plan;
    const b = addCutItem(plan, { panelId: "p2", sizeId: "size/s", quantity: 3, mirror: "forbidden" });
    plan = b.plan;
    expect(cutPlanInstances(plan)).toBe(5);
    expect(() => addCutItem(plan, { id: a.id, panelId: "p", sizeId: "s", quantity: 1, mirror: "allowed" })).toThrowError(/duplicated/);
    expect(() => addCutItem(plan, { panelId: "p", sizeId: "s", quantity: 0, mirror: "allowed" })).toThrowError(/positive integer/);
    expect(serializeCutPlan(deserializeCutPlan(serializeCutPlan(plan)))).toBe(serializeCutPlan(plan));
  });

  it("expands cut items to stable piece instances", () => {
    const g = gradedFixture();
    let plan = createCutPlan("cut/main", "Main", g.doc.id);
    const a = addCutItem(plan, { panelId: g.ids.frontPanel, sizeId: "size/m", quantity: 2, mirror: "allowed" });
    plan = a.plan;
    const b = addCutItem(plan, { panelId: g.ids.backPanel, sizeId: "size/s", quantity: 1, mirror: "required" });
    plan = b.plan;
    const pieces = expandCutPlan(plan ? { ...g.doc } : g.doc, plan);
    // 2 as-authored + 1 as-authored + 1 mirrored.
    expect(pieces.map((p) => p.instanceId)).toEqual([
      `${a.id}#1`, `${a.id}#2`, `${b.id}#1`, `${b.id}#1/m`,
    ]);
    expect(pieces[3].mirrored).toBe(true);
    expect(pieces.every((p) => p.polygon.length >= 3 && p.areaM2 > 0)).toBe(true);
    // Sizes really differ: S back vs M front areas differ by grading deltas.
    expect(pieces[0].areaM2).not.toBeCloseTo(pieces[2].areaM2, 6);
    // Wrong grading reference is rejected.
    expect(() => expandCutPlan(g.doc, { ...plan, gradingId: "other" })).toThrowError(/references grading/);
    // Missing panel is rejected.
    const bad = addCutItem(plan, { panelId: "ghost", sizeId: "size/m", quantity: 1, mirror: "allowed" });
    expect(() => expandCutPlan(g.doc, bad.plan)).toThrowError(/missing panel/);
  });

  it("validates markers structurally", () => {
    const g = gradedFixture();
    let plan = createCutPlan("cut/main", "Main", g.doc.id);
    const a = addCutItem(plan, { panelId: g.ids.frontPanel, sizeId: "size/m", quantity: 1, mirror: "allowed" });
    plan = a.plan;
    const fabric = createFabric({ id: "f", name: "F", widthM: 1.5, lengthM: 5, materialType: "c" });
    const pieces = expandCutPlan(g.doc, plan);
    const marker = createMarker("marker/1", "M1", fabric, plan.id, defaultNestingConstraint(), pieces);
    expect(validateMarker(marker)).toEqual([]);
    expect(marker.revision).toBe(1);
    // Duplicate placements and unknown pieces are caught.
    const dup = {
      ...marker,
      placements: [
        { instanceId: pieces[0].instanceId, x: 0, y: 0, rotationDeg: 0, mirrored: false, manual: false },
        { instanceId: pieces[0].instanceId, x: 1, y: 1, rotationDeg: 0, mirrored: false, manual: false },
      ],
    };
    expect(validateMarker(dup).map((d) => d.code)).toContain("duplicate-id");
    const ghost = {
      ...marker,
      placements: [{ instanceId: "ghost", x: 0, y: 0, rotationDeg: 0, mirrored: false, manual: false }],
    };
    expect(validateMarker(ghost).map((d) => d.code)).toContain("missing-reference");
    expect(validateMarker({ ...marker, pieces: [] }).map((d) => d.code)).toContain("missing-quantity");
    // Round-trip is byte-identical.
    expect(serializeMarker(deserializeMarker(serializeMarker(marker)))).toBe(serializeMarker(marker));
    expect(() => deserializeMarker("nope")).toThrowError(/JSON/);
  });

  it("derives graded geometry per size without mutating the master", () => {
    const g = gradedFixture();
    const before = JSON.stringify(g.doc.master.document);
    const { graded } = deriveSize(g.doc, "size/m");
    expect(graded.document.panels).toHaveLength(2);
    expect(JSON.stringify(g.doc.master.document)).toBe(before);
  });
});
