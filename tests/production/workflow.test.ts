import { describe, expect, it } from "vitest";
import {
  deserializeBOM,
  generateBOMRequirements,
  serializeBOM,
  validateBOM,
  type BOM,
} from "../../src/production/bom.js";
import {
  calculateCutPlan,
  createProductionRun,
  deserializeProductionRun,
  markerConsumptionForRun,
  serializeProductionRun,
} from "../../src/production/run.js";
import {
  createRevision,
  createRevisionLedger,
  deserializeRevisionLedger,
  recordRevisionArtifact,
  rollbackRevision,
  serializeRevisionLedger,
  setRevisionStatus,
} from "../../src/production/revision.js";
import { propagateArtifactStaleness } from "../../src/production/reporting.js";
import { buildGradingFixture, addCoreSizes } from "../grading/fixtures.js";
import { createProductionSpecification, generateTechPack } from "../../src/production/specification.js";
import { createGarmentProject } from "../../src/garment/project.js";
import { buildManufacturingReports } from "../../src/production/reporting.js";
import { generateProductionPackage } from "../../src/production/package.js";
import { recordProductionPackage } from "../../src/production/package.js";
import type { Marker } from "../../src/marker/model.js";

describe("G18B BOM and materials", () => {
  function bomFixture(): BOM {
    return {
      schemaVersion: 1,
      id: "bom/shirt",
      garmentId: "garment/shirt",
      revision: "A",
      materials: [{ id: "fabric/shell", name: "Poplin", category: "shell", unit: "m" }],
      items: [{
        id: "bom/shell",
        materialId: "fabric/shell",
        basis: "per-garment",
        quantity: 1.2,
        unit: "m",
        wasteAllowance: 0.1,
        cost: { unitCost: 4, currency: "USD" },
      }],
      markerReferences: [],
    };
  }

  it("calculates transparent theoretical requirements, waste allowance, and costs", () => {
    const requirements = generateBOMRequirements(bomFixture(), {
      garmentQuantity: 10,
      markerMetrics: {},
    });
    expect(requirements[0].theoreticalQuantity).toBe(12);
    expect(requirements[0].requiredQuantity).toBeCloseTo(13.2);
    expect(requirements[0].estimatedCost).toBeCloseTo(52.8);
    expect(requirements[0].currency).toBe("USD");
    expect(requirements[0].estimatedConsumption).toBeUndefined();
    expect(generateBOMRequirements(bomFixture(), {
      garmentQuantity: 2,
      markerMetrics: { marker1: { lengthM: 8, widthM: 1.5, actualConsumptionM: 9 } },
    })[0].actualConsumption).toBeUndefined();
  });

  it("uses marker estimates distinctly from theoretical and actual consumption, validates, and round-trips", () => {
    const bom = bomFixture();
    bom.markerReferences = [
      { markerId: "marker/shell", fabricId: "fabric/shell" },
      { markerId: "marker/shell-2", fabricId: "fabric/shell" },
    ];
    const requirements = generateBOMRequirements(bom, {
      garmentQuantity: 2,
      markerMetrics: {
        "marker/shell": { lengthM: 2.5, widthM: 1.4, actualConsumptionM: 5.2 },
        "marker/shell-2": { lengthM: 1.5, widthM: 1.4, repeats: 2, actualConsumptionM: 3.1 },
      },
    });
    expect(requirements[0]).toMatchObject({
      theoreticalQuantity: 2.4,
      estimatedConsumption: 8,
      actualConsumption: 8.3,
    });
    expect(validateBOM(bom, { markers: [{
      id: "marker/shell",
      fabric: { id: "fabric/shell" },
    } as Marker, {
      id: "marker/shell-2",
      fabric: { id: "fabric/shell" },
    } as Marker] })).toEqual([]);
    const bad = { ...bom, items: [...bom.items, { ...bom.items[0], id: "duplicate", unit: "kg" }] };
    expect(validateBOM(bad).map((issue) => issue.code)).toContain("inconsistent-assignment");
    const saved = serializeBOM(bom);
    expect(serializeBOM(deserializeBOM(saved))).toBe(saved);
  });
});

describe("G18C production runs and cut plans", () => {
  function markerFixture(sizeId: string, panelIds: string[], repeats = 1): Marker {
    const pieces = panelIds.map((panelId, index) => ({
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
      name: `Marker ${sizeId}`,
      fabric: { id: "fabric/main", name: "Shell", widthM: 1.5, usableWidthM: 1.4, lengthM: 10, materialType: "shell" },
      cutPlanId: `cutplan/${sizeId}`,
      constraint: {
        spacingM: 0.01, marginM: 0.01, rotationsDeg: [0], mirrorDefault: "as-authored",
        grainToleranceDeg: 0, directionalFabric: false,
      },
      pieces,
      placements: pieces.map((piece, index) => ({
        instanceId: piece.instanceId, x: 0.05, y: 0.01 + index * 0.35, rotationDeg: 0, mirrored: false, manual: false,
      })),
      revision: repeats,
    };
  }

  it("plans a multi-size run from multiple markers and reports remaining/incomplete counts", () => {
    const fixture = buildGradingFixture();
    const grading = addCoreSizes(fixture.doc);
    const run = createProductionRun({
      id: "run/1",
      garmentId: "garment/1",
      styleNumber: "STYLE-1",
      revision: "B",
      gradingId: grading.id,
      status: "planned",
      sizeQuantities: { "size/s": 3, "size/m": 1 },
      panelMaterialIds: {
        [fixture.ids.frontPanel]: "fabric/main",
        [fixture.ids.backPanel]: "fabric/main",
      },
      batches: [{
        id: "batch/1",
        markerAssignments: [
          { markerId: "marker/size-s", fabricId: "fabric/main", repeats: 2 },
          { markerId: "marker/size-m", fabricId: "fabric/main", repeats: 1 },
        ],
      }],
    });

    const markers = [
      markerFixture("size/s", [fixture.ids.frontPanel, fixture.ids.backPanel]),
      markerFixture("size/m", [fixture.ids.frontPanel, fixture.ids.backPanel]),
    ];
    markers[0].id = "marker/size-s";
    markers[1].id = "marker/size-m";
    const result = calculateCutPlan(run, grading, markers);
    expect(result.produced).toEqual({ "size/s": 2, "size/m": 1 });
    expect(result.remaining).toEqual({ "size/s": 1, "size/m": 0 });
    expect(result.requiredPieces["size/s"][fixture.ids.frontPanel]).toBe(3);
    expect(result.cutPieces["size/s"][fixture.ids.backPanel]).toBe(2);
    expect(result.complete).toBe(false);
    expect(result.diagnostics.map((diagnostic) => diagnostic.code)).toContain("incomplete-quantity");
    expect(result.batches[0].fabricRequirements[0].lengthM).toBeGreaterThan(0);
    expect(markerConsumptionForRun(run, markers)["marker/size-s"]).toMatchObject({
      repeats: 2,
      estimatedConsumptionM: expect.any(Number),
      estimatedConsumptionAreaM2: expect.any(Number),
    });
  });

  it("combines markers from multiple fabrics to fulfill one size and reports inventory shortfall", () => {
    const fixture = buildGradingFixture();
    const grading = addCoreSizes(fixture.doc);
    const front = markerFixture("size/s", [fixture.ids.frontPanel]);
    front.id = "marker/front";
    front.fabric.id = "fabric/front";
    const back = markerFixture("size/s", [fixture.ids.backPanel]);
    back.id = "marker/back";
    back.fabric.id = "fabric/back";
    const run = createProductionRun({
      id: "run/multifabric",
      garmentId: "garment/1",
      styleNumber: "STYLE-1",
      revision: "A",
      gradingId: grading.id,
      status: "planned",
      sizeQuantities: { "size/s": 1 },
      panelMaterialIds: {
        [fixture.ids.frontPanel]: "fabric/front",
        [fixture.ids.backPanel]: "fabric/back",
      },
      fabricAvailableM: { "fabric/front": 0.1, "fabric/back": 2 },
      batches: [{
        id: "batch/multi",
        markerAssignments: [
          { markerId: front.id, fabricId: "fabric/front", repeats: 1 },
          { markerId: back.id, fabricId: "fabric/back", repeats: 1 },
        ],
      }],
    });
    const result = calculateCutPlan(run, grading, [front, back]);
    expect(result.produced["size/s"]).toBe(1);
    expect(result.remaining["size/s"]).toBe(0);
    expect(result.complete).toBe(false);
    expect(result.batches[0].fabricRequirements.map((item) => item.fabricId).sort())
      .toEqual(["fabric/back", "fabric/front"]);
    expect(result.fabricInventory.find((item) => item.fabricId === "fabric/front")?.shortfallM)
      .toBeGreaterThan(0);
  });

  it("checks fabric inventory across all batches instead of independently per batch", () => {
    const fixture = buildGradingFixture();
    const grading = addCoreSizes(fixture.doc);
    const marker = markerFixture("size/s", [fixture.ids.frontPanel, fixture.ids.backPanel]);
    marker.id = "marker/split";
    const run = createProductionRun({
      id: "run/split",
      garmentId: "garment/1",
      styleNumber: "STYLE-1",
      revision: "A",
      gradingId: grading.id,
      status: "planned",
      sizeQuantities: { "size/s": 2 },
      panelMaterialIds: {
        [fixture.ids.frontPanel]: "fabric/main",
        [fixture.ids.backPanel]: "fabric/main",
      },
      fabricAvailableM: { "fabric/main": 1 },
      batches: [
        { id: "batch/1", markerAssignments: [{ markerId: marker.id, fabricId: "fabric/main", repeats: 1 }] },
        { id: "batch/2", markerAssignments: [{ markerId: marker.id, fabricId: "fabric/main", repeats: 1 }] },
      ],
    });

    const result = calculateCutPlan(run, grading, [marker]);
    expect(result.produced["size/s"]).toBe(2);
    expect(result.fabricInventory[0]).toMatchObject({
      fabricId: "fabric/main",
      availableLengthM: 1,
      remainingLengthM: 0,
    });
    expect(result.fabricInventory[0].requiredLengthM).toBeCloseTo(1.32);
    expect(result.fabricInventory[0].shortfallM).toBeCloseTo(0.32);
    expect(result.complete).toBe(false);
  });

  it("rejects invalid quantities, missing sizes, incompatible fabrics, and saves the run", () => {
    const fixture = buildGradingFixture();
    const grading = addCoreSizes(fixture.doc);
    expect(() => createProductionRun({
      id: "bad", garmentId: "g", styleNumber: "s", revision: "A", gradingId: grading.id,
      status: "draft", sizeQuantities: { S: -1 }, panelMaterialIds: {},
    })).toThrow(/non-negative/);
    const run = createProductionRun({
      id: "run", garmentId: "g", styleNumber: "s", revision: "A", gradingId: grading.id,
      status: "draft", sizeQuantities: { missing: 1 }, panelMaterialIds: {},
    });
    const result = calculateCutPlan(run, grading, []);
    expect(result.diagnostics.map((item) => item.code)).toContain("missing-size");
    const saved = serializeProductionRun(run);
    expect(serializeProductionRun(deserializeProductionRun(saved))).toBe(saved);
  });
});

describe("G18D revisions and traceability", () => {
  it("branches revisions, records exact artifacts, rolls back, and persists history", () => {
    let ledger = createRevisionLedger("garment/1");
    ledger = createRevision({
      ledger, id: "rev/1", author: "designer", timestamp: "2026-01-01",
      changeSummary: "Initial", sourceProjectVersion: "project/v1", sourceFingerprint: "source/a",
    });
    ledger = recordRevisionArtifact(ledger, "rev/1", {
      id: "pattern/a", kind: "pattern", fingerprint: "pattern/hash",
    }, { author: "designer", timestamp: "2026-01-01" });
    ledger = setRevisionStatus(ledger, "rev/1", "Released", {
      author: "designer", timestamp: "2026-01-02",
    });
    ledger = createRevision({
      ledger, id: "rev/2", parentRevisionId: "rev/1", author: "designer",
      timestamp: "2026-01-03", changeSummary: "Revise fit",
      sourceProjectVersion: "project/v1", sourceFingerprint: "source/b",
    });
    const branched = rollbackRevision(ledger, "rev/1", {
      id: "rev/3", author: "designer", timestamp: "2026-01-04", sourceProjectVersion: "project/v1",
    });
    expect(branched.revisions[2].parentRevisionId).toBe("rev/1");
    expect(branched.revisions[2].sourceFingerprint).toBe("source/a");
    expect(branched.revisions[2].artifacts[0].revisionId).toBe("rev/3");
    expect(branched.history.map((event) => event.kind)).toContain("rollback");
    const saved = serializeRevisionLedger(branched);
    expect(serializeRevisionLedger(deserializeRevisionLedger(saved))).toBe(saved);
  });
});

describe("G18E readiness propagation and final package", () => {
  it("propagates staleness through production dependencies", () => {
    const artifacts = [
      { id: "grade", kind: "grading" as const, dependencies: ["pattern"], sourceFingerprint: "a", currentSourceFingerprint: "a", state: "current" as const },
      { id: "marker", kind: "marker" as const, dependencies: ["grade"], sourceFingerprint: "a", currentSourceFingerprint: "a", state: "current" as const },
      { id: "bom", kind: "bom" as const, dependencies: ["marker", "material"], sourceFingerprint: "a", currentSourceFingerprint: "a", state: "current" as const },
      { id: "pack", kind: "package" as const, dependencies: ["marker", "bom"], sourceFingerprint: "a", currentSourceFingerprint: "a", state: "current" as const },
    ];
    expect(propagateArtifactStaleness(artifacts, ["pattern"]).map((item) => item.state))
      .toEqual(["stale", "stale", "stale", "stale"]);
    expect(propagateArtifactStaleness(artifacts, ["material"]).map((item) => item.state))
      .toEqual(["current", "current", "stale", "stale"]);
    expect(propagateArtifactStaleness(artifacts.map((item) => ({ ...item, state: "stale" as const })), [])
      .map((item) => item.state)).toEqual(["current", "current", "current", "current"]);
  });

  it("generates an organized traceable package only from the released current revision", () => {
    const patternFixture = buildGradingFixture();
    const garment = createGarmentProject("garment/1", "Shirt", patternFixture.document, { seams: [patternFixture.seam] });
    const specification = createProductionSpecification({
      id: "spec/1", garmentId: garment.id, garmentName: "Shirt", styleNumber: "SH-1", revision: "A",
      sizeRange: [], materials: [{ id: "fabric/main", name: "Shell", category: "shell" }],
      colorways: [], panels: [], construction: [], measurements: [],
      productionNotes: [], finishingNotes: [],
    });
    const techPack = generateTechPack(garment, specification);
    const bom: BOM = {
      schemaVersion: 1, id: "bom/1", garmentId: garment.id, revision: "A",
      materials: [{ id: "fabric/main", name: "Shell", category: "shell", unit: "m" }],
      items: [{ id: "item/1", materialId: "fabric/main", basis: "per-garment", quantity: 1, unit: "m", wasteAllowance: 0 }],
      markerReferences: [],
    };
    const traceIds = ["pattern", bom.id, "run/1", `tech-pack:${techPack.sourceFingerprint}`, "cut-plan:run/1"];
    const traceArtifacts = traceIds.map((id) => ({
      id,
      kind: id.startsWith("tech-pack:") ? "tech-pack" as const : id.startsWith("cut-plan:") ? "cut-plan" as const : "pattern" as const,
      dependencies: [],
      sourceFingerprint: `fingerprint/${id}`,
      currentSourceFingerprint: `fingerprint/${id}`,
      sourceRevisionId: "rev/1",
      state: "current" as const,
    }));
    const revision = {
      id: "rev/1", number: 1, parentRevisionId: null, author: "designer", timestamp: "2026-01-01",
      changeSummary: "Release", approvalState: "configured", sourceProjectVersion: "project/v1",
      status: "Released" as const, sourceFingerprint: techPack.sourceFingerprint,
      artifacts: traceArtifacts.map((artifact) => ({
        id: artifact.id,
        kind: artifact.kind,
        fingerprint: artifact.sourceFingerprint,
        revisionId: "rev/1",
      })),
    };
    const run = createProductionRun({
      id: "run/1", garmentId: garment.id, styleNumber: "SH-1", revision: "A",
      gradingId: "grading/1", status: "complete", sizeQuantities: {},
      panelMaterialIds: {},
    });
    const cutPlan = {
      runId: run.id, requested: {}, produced: {}, remaining: {}, requiredPieces: {}, cutPieces: {},
      batches: [], fabricInventory: [], diagnostics: [], complete: true,
    };
    const report = buildManufacturingReports({
      garmentId: garment.id, garmentName: "Shirt", garmentRevision: 1, pattern: garment.pattern,
      specification, techPack, bom, run, cutPlan, artifacts: [],
    });
    expect(report.dashboard.validation.warnings).toContain("production export is missing");
    const input = {
      garment, specification, revision, markers: [], bom, run, cutPlan, techPack,
      dashboard: report.dashboard, reports: report.reports, artifacts: traceArtifacts,
    };
    const pkg = generateProductionPackage(input);
    expect(generateProductionPackage(input).fingerprint).toBe(pkg.fingerprint);
    expect(generateProductionPackage({
      ...input,
      artifacts: [...traceArtifacts, {
        id: "old-package",
        kind: "production-package" as const,
        dependencies: [],
        sourceFingerprint: "old",
        currentSourceFingerprint: "new",
        state: "stale" as const,
      }],
    }).fingerprint).toBe(pkg.fingerprint);
    expect(pkg.files.map((file) => file.path)).toEqual(expect.arrayContaining([
      expect.stringContaining("/patterns/"),
      expect.stringContaining("/technical/"),
      expect.stringContaining("/materials/"),
      expect.stringContaining("/reports/"),
      expect.stringContaining("/previews/"),
      expect.stringContaining("/metadata/"),
    ]));
    expect(pkg.archive[0]).toBe(0x50);
    expect(pkg.archive[1]).toBe(0x4b);
    const artifactManifest = JSON.parse(pkg.files.find((file) => file.path.endsWith("/metadata/artifacts.json"))?.bytes as string) as Array<{ path: string; revisionId: string; garmentId: string; generation: { sourceFingerprint: string } }>;
    expect(artifactManifest.some((artifact) =>
      artifact.path.endsWith("/patterns/base.json") &&
      artifact.revisionId === revision.id &&
      artifact.garmentId === garment.id &&
      artifact.generation.sourceFingerprint === revision.sourceFingerprint)).toBe(true);
    let ledger = createRevisionLedger(garment.id);
    ledger = createRevision({
      ledger,
      id: revision.id,
      author: revision.author,
      timestamp: revision.timestamp,
      changeSummary: revision.changeSummary,
      sourceProjectVersion: revision.sourceProjectVersion,
      sourceFingerprint: revision.sourceFingerprint,
      status: "Released",
    });
    ledger = recordProductionPackage(ledger, pkg, { author: "designer", timestamp: "2026-01-02" });
    expect(ledger.revisions[0].artifacts[0]).toMatchObject({
      id: pkg.id,
      kind: "production-package",
      fingerprint: pkg.fingerprint,
      revisionId: revision.id,
    });
    expect(() => generateProductionPackage({
      ...input, artifacts: traceArtifacts.map((artifact) => artifact.id === "pattern" ? { ...artifact, state: "stale" as const } : artifact),
    })).toThrow(/stale/);
  });
});
