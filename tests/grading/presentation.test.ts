// G12D multi-size presentation tests: layer construction across 2/5/10+
// sizes, per-size visibility, grading-point markers and rule vectors, labels,
// pairwise comparison, size-scoped picking, the master-redirect editing
// contract, and master modification + regeneration through the view.
import { describe, expect, it } from "vitest";
import { movePoint } from "../../src/pattern/cad.js";
import {
  addGradingPoint,
  addMeasurementDefinition,
  addRule,
  addSize,
  assignMeasurement,
  compareSizes,
  createGradingDocument,
  createGradingPoint,
  createMasterPattern,
  createMeasurementDefinition,
  createRule,
  createRuleTable,
  createSize,
  createSizeSet,
  createSizeView,
  defaultSizeViewOptions,
  documentForSize,
  editIntent,
  GradingError,
  pickAtPosition,
  regenerateAll,
  type GradingDocument,
  type SizeViewOptions,
} from "../../src/grading/index.js";
import { buildGradingFixture } from "./fixtures.js";

function hemDocument(labels: string[]): { doc: GradingDocument; ids: ReturnType<typeof buildGradingFixture>["ids"] } {
  const fixture = buildGradingFixture();
  let doc = createGradingDocument({
    id: "grading/g12d",
    name: "G12D presentation",
    master: createMasterPattern("master/g12d", "G12D master", fixture.document),
    sizeSet: createSizeSet("sizeset/g12d", "G12D sizes"),
    ruleTable: createRuleTable("ruletable/g12d", "G12D rules"),
    seams: [],
  });
  for (const label of labels) {
    doc = addSize(doc, createSize({ id: `size/g${label}`, label }));
  }
  doc = addGradingPoint(doc, createGradingPoint("gp/hem-side", {
    kind: "corner", panelId: fixture.ids.frontPanel,
    segmentIdA: fixture.ids.segments.ab, segmentIdB: fixture.ids.segments.bc,
  }));
  const deltas: Record<string, [number, number]> = {};
  for (const label of labels) deltas[`size/g${label}`] = [0.005, 0];
  doc = addRule(doc, createRule("rule/hem", "gp/hem-side", "transition", deltas));
  return { doc, ids: fixture.ids };
}

function step(index: number): number {
  return 0.005 * (index + 1);
}

function maxX(outline: Array<[number, number]>): number {
  return Math.max(...outline.map((v) => v[0]));
}

function hasVertex(outline: Array<[number, number]>, expected: [number, number]): boolean {
  return outline.some(([x, y]) => Math.abs(x - expected[0]) < 1e-9 && Math.abs(y - expected[1]) < 1e-9);
}

describe("size view layers", () => {
  it("builds master-highlighted layers for two sizes", () => {
    const { doc } = hemDocument(["1", "2"]);
    const view = createSizeView(doc, defaultSizeViewOptions());
    expect(view.mode).toBe("overlay");
    expect(view.layers.map((layer) => layer.sizeId)).toEqual(["size/g1", "size/g2"]);
    expect(view.layers.map((layer) => layer.nestedOrder)).toEqual([0, 1]);
    expect(view.masterLayer?.sizeId).toBe("master");
    expect(view.masterLayer?.nestedOrder).toBe(-1);
    expect(view.layers.every((layer) => !layer.invalid && layer.visible)).toBe(true);
    expect(view.layers.every((layer) => layer.panels.length === 2)).toBe(true);
    expect(view.masterLayer?.panels.length).toBe(2);
    // Labels sit on the top edge of each size's combined bounding box
    // (both panels together: x in [0, 0.8], y top 0.6).
    expect(view.labels.map((label) => label.text)).toEqual(["1", "2"]);
    expect(view.labels[0].position[0]).toBeCloseTo(0.4, 6);
    expect(view.labels[0].position[1]).toBeCloseTo(0.6, 6);
  });

  it("renders rule vectors and grading-point markers when enabled", () => {
    const { doc } = hemDocument(["1", "2"]);
    const options: SizeViewOptions = { ...defaultSizeViewOptions(), showRuleVectors: true, showGradingPoints: true };
    const view = createSizeView(doc, options);
    const first = view.layers[0];
    expect(first.ruleVectors).toHaveLength(1);
    expect(first.ruleVectors[0].ruleId).toBe("rule/hem");
    expect(first.ruleVectors[0].from[0]).toBeCloseTo(0.4, 10);
    expect(first.ruleVectors[0].to[0]).toBeCloseTo(0.4 + step(0), 10);
    expect(first.gradingPoints).toHaveLength(1);
    expect(first.gradingPoints[0].gradingPointId).toBe("gp/hem-side");
    expect(first.gradingPoints[0].position[0]).toBeCloseTo(0.4 + step(0), 10);

    const plain = createSizeView(doc, defaultSizeViewOptions());
    expect(plain.layers[0].ruleVectors).toHaveLength(0);
    expect(plain.layers[0].gradingPoints).toHaveLength(0);
  });

  it("supports five and ten-plus sizes with deterministic nesting order", () => {
    for (const count of [5, 10, 12]) {
      const labels = Array.from({ length: count }, (_, i) => String(i + 1));
      const { doc } = hemDocument(labels);
      const view = createSizeView(doc, defaultSizeViewOptions());
      expect(view.layers).toHaveLength(count);
      expect(view.layers.map((layer) => layer.nestedOrder)).toEqual(labels.map((_, i) => i));
      expect(view.layers.every((layer) => !layer.invalid)).toBe(true);
      // Concentric growth: each size's hem steps out by the transition delta.
      expect(maxX(view.layers[0].panels[0].outline)).toBeCloseTo(0.4 + step(0), 10);
      expect(maxX(view.layers[count - 1].panels[0].outline)).toBeCloseTo(0.4 + step(count - 1), 10);
    }
  });

  it("shows only the active size in active mode", () => {
    const { doc } = hemDocument(["1", "2", "3"]);
    const view = createSizeView(doc, { ...defaultSizeViewOptions(), mode: "active", activeSizeId: "size/g2" });
    expect(view.layers.map((layer) => layer.sizeId)).toEqual(["size/g2"]);
    expect(view.layers[0].nestedOrder).toBe(1);
  });

  it("keeps hidden sizes in render order but flags them invisible", () => {
    const { doc } = hemDocument(["1", "2", "3"]);
    const view = createSizeView(doc, { ...defaultSizeViewOptions(), visibleSizeIds: ["size/g1", "size/g3"] });
    expect(view.layers.map((layer) => layer.sizeId)).toEqual(["size/g1", "size/g2", "size/g3"]);
    expect(view.layers.map((layer) => layer.visible)).toEqual([true, false, true]);
    expect(view.labels.map((label) => label.sizeId)).toEqual(["size/g1", "size/g3"]);
  });

  it("marks a size invalid instead of failing the whole view", () => {
    const { doc } = hemDocument(["1", "2"]);
    const broken = addSize(doc, createSize({ id: "size/g3", label: "3" }));
    const view = createSizeView(broken, defaultSizeViewOptions());
    const invalid = view.layers.find((layer) => layer.sizeId === "size/g3")!;
    expect(invalid.invalid).toBe(true);
    expect(invalid.diagnostics[0].code).toBe("invalid-document");
    expect(invalid.diagnostics[0].message).toMatch(/missing-rule/);
    expect(invalid.panels).toHaveLength(0);
    expect(view.layers.find((layer) => layer.sizeId === "size/g1")!.invalid).toBe(false);
  });

  it("refuses to present an empty size set", () => {
    const fixture = buildGradingFixture();
    const doc = createGradingDocument({
      id: "grading/empty", name: "Empty",
      master: createMasterPattern("master/empty", "Empty master", fixture.document),
      sizeSet: createSizeSet("sizeset/empty", "No sizes"),
      ruleTable: createRuleTable("ruletable/empty", "No rules"),
      seams: [],
    });
    expect(() => createSizeView(doc, defaultSizeViewOptions())).toThrowError(GradingError);
  });
});

describe("comparison", () => {
  it("compares two sizes across points, seams and measurements", () => {
    const { doc } = hemDocument(["1", "2"]);
    let measured = addMeasurementDefinition(doc, createMeasurementDefinition({ id: "measurement/chest", name: "Chest", unit: "cm" }));
    measured = assignMeasurement(measured, "size/g1", "measurement/chest", 90, { unit: "cm" });
    measured = assignMeasurement(measured, "size/g2", "measurement/chest", 96, { unit: "cm" });

    const comparison = compareSizes(measured, "size/g1", "size/g2");
    expect(comparison.pointDisplacements).toHaveLength(1);
    expect(comparison.pointDisplacements[0].gradingPointId).toBe("gp/hem-side");
    expect(comparison.pointDisplacements[0].distanceM).toBeCloseTo(0.005, 10);
    expect(comparison.maxPointDistanceM).toBeCloseTo(0.005, 10);
    expect(comparison.measurementDeltas).toHaveLength(1);
    expect(comparison.measurementDeltas[0].measurementId).toBe("measurement/chest");
    expect(comparison.measurementDeltas[0].valueAM).toBeCloseTo(0.9, 12);
    expect(comparison.measurementDeltas[0].valueBM).toBeCloseTo(0.96, 12);
    expect(comparison.measurementDeltas[0].deltaM).toBeCloseTo(0.06, 12);
    expect(comparison.seamLengthDeltas).toHaveLength(0);

    const viaView = createSizeView(measured, {
      ...defaultSizeViewOptions(),
      activeSizeId: "size/g1",
      comparisonSizeId: "size/g2",
    });
    expect(viaView.comparison?.maxPointDistanceM).toBeCloseTo(0.005, 10);
  });
});

describe("picking and selection", () => {
  it("picks points and segments within one named size only", () => {
    const { doc, ids } = hemDocument(["1", "2"]);
    const onMaster = pickAtPosition(doc, "master", [0.4, 0], 0.01);
    expect(onMaster?.sizeId).toBe("master");
    expect(onMaster?.kind).toBe("point");
    expect(onMaster?.id).toBe(ids.points.B);

    // On size/g2 the hem point has graded to (0.41, 0); picking there finds
    // B, while the unmoved back-panel point E still owns (0.4, 0).
    const onSize = pickAtPosition(doc, "size/g2", [0.41, 0], 0.005);
    expect(onSize?.sizeId).toBe("size/g2");
    expect(onSize?.id).toBe(ids.points.B);
    expect(pickAtPosition(doc, "size/g2", [0.4, 0], 0.005)?.id).toBe(ids.points.E);

    const midEdge = pickAtPosition(doc, "master", [0.2, 0.6], 0.05);
    expect(midEdge?.kind).toBe("segment");
    expect(midEdge?.id).toBe(ids.segments.cd);

    expect(pickAtPosition(doc, "master", [5, 5], 0.01)).toBeNull();
    expect(() => pickAtPosition(doc, "size/none", [0, 0], 0.01)).toThrowError(GradingError);
  });

  it("redirects derived-size edits to the master", () => {
    expect(editIntent({ sizeId: "size/g1", kind: "point", id: "p" })).toEqual({
      editable: false,
      redirectSizeId: "master",
      reason: expect.stringMatching(/regenerate/),
    });
    expect(editIntent({ sizeId: "master", kind: "point", id: "p" })).toEqual({ editable: true });
  });
});

describe("master modification and regeneration", () => {
  it("reflects master edits after regeneration through the view", () => {
    const { doc, ids } = hemDocument(["1", "2"]);
    const before = createSizeView(regenerateAll(doc).document, defaultSizeViewOptions());
    expect(hasVertex(before.layers[0].panels[0].outline, [0.4 + step(0), 0])).toBe(true);

    const regenerated = regenerateAll({
      ...doc,
      master: { ...doc.master, document: movePoint(doc.master.document, ids.frontPanel, ids.points.B, [0.45, 0]) },
    }).document;
    const after = createSizeView(regenerated, defaultSizeViewOptions());
    expect(hasVertex(after.layers[0].panels[0].outline, [0.45 + step(0), 0])).toBe(true);
    expect(hasVertex(after.layers[0].panels[0].outline, [0.4 + step(0), 0])).toBe(false);
  });

  it("serves fresh cached geometry and re-derives stale sizes", () => {
    const { doc, ids } = hemDocument(["1"]);
    const regenerated = regenerateAll(doc).document;
    const cached = regenerated.graded[0].document;
    expect(documentForSize(regenerated, "size/g1")).toBe(cached);

    const staleMaster = {
      ...regenerated,
      master: {
        ...regenerated.master,
        document: movePoint(regenerated.master.document, ids.frontPanel, ids.points.D, [0, 0.61]),
      },
    };
    // Master fingerprint changed, so the cached entry is ignored and the
    // size re-derives against the edited master.
    const rederived = documentForSize(staleMaster, "size/g1");
    expect(rederived).not.toBe(cached);
    expect(rederivedPoints(rederived, ids.points.D)).toEqual([[0, 0.61]]);
  });
});

function rederivedPoints(doc: ReturnType<typeof documentForSize>, pointId: string): Array<[number, number]> {
  const point = doc.points.find((candidate) => candidate.id === pointId)!;
  return [[point.x, point.y]];
}
