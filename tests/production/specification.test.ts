import { describe, expect, it } from "vitest";
import { movePoint } from "../../src/pattern/cad.js";
import { createGarmentProject } from "../../src/garment/project.js";
import {
  addSize,
  createGradingDocument,
  createMasterPattern,
  createRuleTable,
  createSize,
  createSizeSet,
} from "../../src/grading/index.js";
import {
  createProductionSpecification,
  deserializeTechPack,
  deserializeProductionSpecification,
  generateTechPack,
  serializeProductionSpecification,
  serializeTechPack,
} from "../../src/production/specification.js";
import { addGarment, createProject, deserializeProject, serializeProject } from "../../src/project/project.js";
import { rectFixture } from "../cad/fixtures.js";
import { createBOMFromSpecification } from "../../src/production/bom.js";

describe("G18A production specification and tech pack", () => {
  function fixture() {
    const pattern = rectFixture().document;
    const panel = pattern.panels[0];
    const points = pattern.points.filter((point) => point.panelId === panel.id);
    const garment = createGarmentProject("garment/shirt", "Shirt", pattern);
    const grading = createGradingDocument({
      id: "grading/shirt",
      name: "Shirt grading",
      master: createMasterPattern("master/shirt", "Shirt master", pattern),
      sizeSet: createSizeSet("sizes/shirt", "Shirt sizes"),
      ruleTable: createRuleTable("rules/shirt", "Shirt rules"),
    });
    const sizedGrading = addSize(
      addSize(grading, createSize({ id: "size/s", label: "S" })),
      createSize({ id: "size/m", label: "M" }),
    );
    const specification = createProductionSpecification({
      id: "spec/shirt",
      garmentId: garment.id,
      garmentName: "Classic shirt",
      styleNumber: "CS-001",
      revision: "B",
      sizeRange: ["size/s", "size/m"],
      materials: [{ id: "fabric/shell", name: "Cotton poplin", category: "shell" }],
      colorways: [{
        id: "colorway/white",
        name: "White",
        materialColors: { "fabric/shell": "#ffffff" },
      }],
      panels: [{ panelId: panel.id, materialId: "fabric/shell", cutQuantity: 2 }],
      construction: [{
        id: "construction/side-seam",
        kind: "seam",
        title: "Side seam",
        data: { stitch: "lockstitch", seamAllowanceM: 0.01 },
      }],
      measurements: [{
        id: "measurement/body-width",
        name: "Body width",
        source: { kind: "panel-width", panelId: panel.id },
        unit: "m",
        targetsBySize: {
          base: { targetM: 0.4, toleranceM: 0.002 },
          "size/s": { targetM: 0.4, toleranceM: 0.002 },
          "size/m": { targetM: 0.4, toleranceM: 0.002 },
        },
      }, {
        id: "measurement/edge",
        name: "Bottom edge",
        source: { kind: "segment-length", panelId: panel.id, segmentId: panel.boundaryLoops[0].segmentIds[0] },
        unit: "m",
        targetsBySize: {},
      }],
      productionNotes: ["Cut on grain"],
      finishingNotes: ["Press seams open"],
    });
    return { garment, grading: sizedGrading, specification, panel, points };
  }

  it("generates deterministic size-specific technical documents and records geometry measurements", () => {
    const { garment, grading, specification } = fixture();
    const first = generateTechPack(garment, specification, { grading });
    const again = generateTechPack(garment, specification, { grading });
    expect(first).toEqual(again);
    const saved = serializeTechPack(first);
    expect(serializeTechPack(deserializeTechPack(saved))).toBe(saved);
    expect(first.sizes.map((size) => size.label)).toEqual(["S", "M"]);
    expect(first.drawings.map((drawing) => drawing.sizeId)).toEqual(["size/s", "size/m"]);
    expect(first.drawings[0].svg).toContain("CS-001 B");
    expect(first.measurements.filter((item) => item.measurementId === "measurement/body-width")
      .map((item) => [item.measuredM, item.status])).toEqual([[0.4, "pass"], [0.4, "pass"]]);
    expect(first.measurements.filter((item) => item.measurementId === "measurement/edge")
      .every((item) => item.status === "missing-target")).toBe(true);
    expect(first.materials[0].category).toBe("shell");
    expect(first.panels[0].materialId).toBe("fabric/shell");
  });

  it("reports missing geometry, detects geometry changes, and regenerates from the source", () => {
    const { garment, specification, panel, points } = fixture();
    const incomplete = {
      ...specification,
      measurements: [{
        ...specification.measurements[0],
        source: { kind: "point-distance" as const, panelId: panel.id, pointAId: "gone", pointBId: points[0].id },
      }],
    };
    const missing = generateTechPack(garment, incomplete);
    expect(missing.validation.status).toBe("errors");
    expect(missing.measurements[0].status).toBe("missing-source");

    let nextPattern = movePoint(garment.pattern, panel.id, points[1].id, [points[1].x + 0.05, points[1].y]);
    nextPattern = movePoint(nextPattern, panel.id, points[2].id, [points[2].x + 0.05, points[2].y]);
    const editedGarment = {
      ...garment,
      metadata: { ...garment.metadata, revision: 2 },
      pattern: nextPattern,
    };
    const originalPack = generateTechPack(garment, specification);
    const regenerated = generateTechPack(editedGarment, specification);
    expect(regenerated.sourceFingerprint).not.toBe(originalPack.sourceFingerprint);
    expect(regenerated.garmentRevision).toBe(2);
    expect(regenerated.measurements[0].status).toBe("out-of-tolerance");
    expect(regenerated.measurements[0].measuredM).toBeCloseTo(0.45);
  });

  it("round-trips specifications and saves them with the garment project", () => {
    const { garment, specification } = fixture();
    const serialized = serializeProductionSpecification(specification);
    expect(serializeProductionSpecification(deserializeProductionSpecification(serialized))).toBe(serialized);
    const project = addGarment(createProject("project/1", "Shirt project", "2026-10-06"), {
      id: garment.id,
      name: garment.metadata.name,
      garment,
      productionSpecification: specification,
    });
    const saved = serializeProject(project);
    expect(serializeProject(deserializeProject(saved))).toBe(saved);
  });

  it("generates panel-material BOM items without imposing a category or unit catalogue", () => {
    const { specification } = fixture();
    const bom = createBOMFromSpecification(specification, {
      id: "bom/shirt",
      revision: specification.revision,
      materialUnits: { "fabric/shell": "linear-yard" },
    });
    expect(bom.materials[0].category).toBe("shell");
    expect(bom.materials[0].unit).toBe("linear-yard");
    expect(bom.items).toMatchObject([{
      materialId: "fabric/shell",
      panelId: specification.panels[0].panelId,
      basis: "per-panel",
      quantity: 1,
    }]);
  });
});
