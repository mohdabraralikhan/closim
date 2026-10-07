// G12C measurement & size-system tests: measurement catalogue schema, unit
// conversion at the boundary, size-set operations (duplicate/rename on top of
// insert/remove/reorder), validation detection (missing measurements, invalid
// units, impossible values, inconsistent ordering) and persistence.
import { describe, expect, it } from "vitest";
import {
  addMeasurementDefinition,
  addSize,
  addGradingPoint,
  assignMeasurement,
  createGradingDocument,
  createMasterPattern,
  createMeasurementDefinition,
  createGradingPoint,
  createRuleTable,
  createSize,
  createSizeSet,
  duplicateSize,
  deriveSize,
  fingerprintGradingDocument,
  insertSize,
  moveSize,
  parseGradingDocument,
  removeMeasurement,
  removeMeasurementDefinition,
  removeSize,
  renameSize,
  serializeGradingDocument,
  validateGradingDocument,
  convertLength,
  fromMetres,
  toMetres,
  GradingError,
  type GradingDocument,
} from "../../src/grading/index.js";
import { buildGradingFixture } from "./fixtures.js";

function emptyDocument(): GradingDocument {
  const fixture = buildGradingFixture();
  return createGradingDocument({
    id: "grading/g12c",
    name: "G12C sizing",
    master: createMasterPattern("master/g12c", "G12C master", fixture.document),
    sizeSet: createSizeSet("sizeset/g12c", "G12C sizes"),
    ruleTable: createRuleTable("ruletable/g12c", "G12C rules"),
    seams: [],
  });
}

/** Sizes with the given labels; the first becomes the base size. */
function withSizes(doc: GradingDocument, labels: string[]): GradingDocument {
  let next = doc;
  for (const label of labels) next = addSize(next, createSize({ id: `size/${label.toLowerCase()}`, label }));
  return next;
}

function chestCatalogue(doc: GradingDocument, ordering: "none" | "increasing" | "decreasing" = "none"): GradingDocument {
  return addMeasurementDefinition(doc, createMeasurementDefinition({
    id: "measurement/chest",
    name: "Chest circumference",
    description: "Around the fullest part of the chest",
    unit: "cm",
    type: "body",
    toleranceM: 0.005,
    ordering,
  }));
}

function dress(doc: GradingDocument, valuesCm: number[]): GradingDocument {
  let next = doc;
  if (valuesCm.length !== doc.sizeSet.sizes.length) throw new Error("dress() needs one value per size");
  doc.sizeSet.sizes.forEach((size, index) => {
    next = assignMeasurement(next, size.id, "measurement/chest", valuesCm[index], { unit: "cm", source: "size-chart" });
  });
  return next;
}

describe("units", () => {
  it("converts between all supported units through canonical metres", () => {
    expect(toMetres(1, "m")).toBe(1);
    expect(toMetres(100, "cm")).toBeCloseTo(1, 12);
    expect(toMetres(1000, "mm")).toBeCloseTo(1, 12);
    expect(toMetres(1, "in")).toBeCloseTo(0.0254, 15);
    expect(fromMetres(1, "in")).toBeCloseTo(39.37007874015748, 10);
    expect(convertLength(1, "in", "cm")).toBeCloseTo(2.54, 12);
    expect(convertLength(25.4, "mm", "in")).toBeCloseTo(1, 12);
  });

  it("rejects unknown units and non-finite values", () => {
    expect(() => toMetres(1, "ft" as never)).toThrowError(GradingError);
    expect(() => fromMetres(Number.NaN, "m")).toThrowError(GradingError);
    expect(() => convertLength(1, "m", "yard" as never)).toThrowError(GradingError);
  });
});

describe("size systems", () => {
  it("supports alphabetical, numeric and custom size labels", () => {
    const alphabetical = withSizes(emptyDocument(), ["XS", "S", "M", "L", "XL"]);
    expect(alphabetical.sizeSet.sizes.map((size) => size.label)).toEqual(["XS", "S", "M", "L", "XL"]);
    expect(alphabetical.sizeSet.baseSizeId).toBe("size/xs");

    const numeric = withSizes(emptyDocument(), ["38", "40", "42", "44"]);
    expect(numeric.sizeSet.sizes.map((size) => size.label)).toEqual(["38", "40", "42", "44"]);

    const custom = withSizes(emptyDocument(), ["petite", "regular", "tall"]);
    expect(custom.sizeSet.sizes[2].label).toBe("tall");
  });

  it("records measurements with unit, source and tolerance in canonical metres", () => {
    let doc = chestCatalogue(withSizes(emptyDocument(), ["S", "M"]));
    doc = assignMeasurement(doc, "size/s", "measurement/chest", 92, { unit: "cm", source: "size-chart", toleranceM: 0.004 });
    doc = assignMeasurement(doc, "size/m", "measurement/chest", 0.98, { source: "body-scan" });
    const s = doc.sizeSet.sizes[0].measurements[0];
    const m = doc.sizeSet.sizes[1].measurements[0];
    expect(s.valueM).toBeCloseTo(0.92, 12);
    expect(s.unit).toBe("cm");
    expect(s.source).toBe("size-chart");
    expect(s.toleranceM).toBe(0.004);
    expect(m.valueM).toBeCloseTo(0.98, 12);
    expect(m.unit).toBe("m");
  });

  it("replaces an existing measurement on re-assignment", () => {
    let doc = chestCatalogue(withSizes(emptyDocument(), ["S"]));
    doc = assignMeasurement(doc, "size/s", "measurement/chest", 90, { unit: "cm" });
    doc = assignMeasurement(doc, "size/s", "measurement/chest", 94, { unit: "cm" });
    expect(doc.sizeSet.sizes[0].measurements).toHaveLength(1);
    expect(doc.sizeSet.sizes[0].measurements[0].valueM).toBeCloseTo(0.94, 12);
  });

  it("removes single measurement values", () => {
    let doc = chestCatalogue(withSizes(emptyDocument(), ["S"]));
    doc = assignMeasurement(doc, "size/s", "measurement/chest", 90, { unit: "cm" });
    doc = removeMeasurement(doc, "size/s", "measurement/chest");
    expect(doc.sizeSet.sizes[0].measurements).toHaveLength(0);
    expect(() => removeMeasurement(doc, "size/s", "measurement/chest")).toThrowError(GradingError);
  });
});

describe("size-set operations", () => {
  it("duplicates a size with values, a unique label and stable position", () => {
    let doc = chestCatalogue(withSizes(emptyDocument(), ["S", "M", "L"]));
    doc = dress(doc, [90, 96, 102]);
    doc = duplicateSize(doc, "size/m", "size/m-tall", { label: "M tall" });
    const copy = doc.sizeSet.sizes.find((size) => size.id === "size/m-tall")!;
    expect(copy.label).toBe("M tall");
    expect(copy.measurements[0].valueM).toBeCloseTo(0.96, 12);
    expect(doc.sizeSet.sizes.map((size) => size.id)).toEqual(["size/s", "size/m", "size/m-tall", "size/l"]);
    expect(doc.sizeSet.baseSizeId).toBe("size/s");
  });

  it("defaults the duplicate label and rejects label collisions", () => {
    let doc = withSizes(emptyDocument(), ["S", "M"]);
    doc = duplicateSize(doc, "size/m", "size/m2");
    expect(doc.sizeSet.sizes.find((size) => size.id === "size/m2")?.label).toBe("M copy");
    expect(() => duplicateSize(doc, "size/m", "size/m3")).toThrowError(/already in use/);
    expect(() => duplicateSize(doc, "size/none", "size/x")).toThrowError(GradingError);
    expect(() => duplicateSize(doc, "size/m", "size/s")).toThrowError(/already exists/);
  });

  it("renames a size and rejects empty or colliding labels", () => {
    let doc = withSizes(emptyDocument(), ["S", "M"]);
    doc = renameSize(doc, "size/m", "MEDIUM", "Medium");
    const renamed = doc.sizeSet.sizes.find((size) => size.id === "size/m")!;
    expect(renamed.label).toBe("MEDIUM");
    expect(renamed.displayName).toBe("Medium");
    expect(() => renameSize(doc, "size/m", "S")).toThrowError(/already in use/);
    expect(() => renameSize(doc, "size/m", "  ")).toThrowError(GradingError);
    expect(() => renameSize(doc, "size/none", "X")).toThrowError(GradingError);
  });

  it("reorders, inserts and removes sizes without breaking identity", () => {
    let doc = withSizes(emptyDocument(), ["S", "M", "L"]);
    doc = moveSize(doc, "size/l", 0);
    expect(doc.sizeSet.sizes.map((size) => size.id)).toEqual(["size/l", "size/s", "size/m"]);
    doc = insertSize(doc, createSize({ id: "size/xs", label: "XS" }), 0);
    doc = removeSize(doc, "size/m");
    expect(doc.sizeSet.sizes.map((size) => size.id)).toEqual(["size/xs", "size/l", "size/s"]);
    expect(validateGradingDocument(doc).filter((d) => d.code === "duplicate-id")).toHaveLength(0);
  });
});

describe("size-system validation", () => {
  it("detects missing measurements against the catalogue", () => {
    let doc = chestCatalogue(withSizes(emptyDocument(), ["S", "M"]));
    doc = dress(doc, [90, 96]);
    expect(validateGradingDocument(doc).filter((d) => d.code === "missing-measurement")).toHaveLength(0);
    const incomplete = removeMeasurement(doc, "size/m", "measurement/chest");
    const missing = validateGradingDocument(incomplete).find((d) => d.code === "missing-measurement");
    expect(missing).toBeDefined();
    expect(missing?.sizeId).toBe("size/m");
    expect(missing?.measurementId).toBe("measurement/chest");
  });

  it("detects unknown measurement references", () => {
    const doc = withSizes(emptyDocument(), ["S"]);
    const broken = JSON.parse(JSON.stringify(doc)) as GradingDocument;
    broken.sizeSet.sizes[0].measurements.push({ measurementId: "measurement/ghost", valueM: 0.9, unit: "m" });
    const diagnostic = validateGradingDocument(broken).find((d) => d.code === "unknown-entity");
    expect(diagnostic?.measurementId).toBe("measurement/ghost");
  });

  it("detects invalid units", () => {
    const doc = withSizes(emptyDocument(), ["S"]);
    const broken = JSON.parse(JSON.stringify(doc)) as GradingDocument;
    broken.sizeSet.measurementDefinitions.push({
      id: "measurement/hip", name: "Hip", unit: "ft" as never, type: "body", ordering: "none",
    });
    const diagnostic = validateGradingDocument(broken).find((d) => d.code === "invalid-unit");
    expect(diagnostic?.measurementId).toBe("measurement/hip");
  });

  it("flags zero-value measurements as impossible", () => {
    let doc = chestCatalogue(withSizes(emptyDocument(), ["S"]));
    doc = assignMeasurement(doc, "size/s", "measurement/chest", 0);
    const diagnostic = validateGradingDocument(doc).find((d) => d.code === "invalid-measurement");
    expect(diagnostic).toBeDefined();
    expect(diagnostic?.message).toMatch(/impossible/i);
  });

  it("detects inconsistent size ordering against the declared progression", () => {
    let doc = chestCatalogue(withSizes(emptyDocument(), ["XS", "S", "M"]), "increasing");
    doc = dress(doc, [88, 92, 96]);
    expect(validateGradingDocument(doc).filter((d) => d.code === "inconsistent-ordering")).toHaveLength(0);

    const decreasing = dress(chestCatalogue(withSizes(emptyDocument(), ["XS", "S", "M"]), "increasing"), [88, 92, 90]);
    const diagnostic = validateGradingDocument(decreasing).find((d) => d.code === "inconsistent-ordering");
    expect(diagnostic).toBeDefined();
    expect(diagnostic?.measurementId).toBe("measurement/chest");
  });

  it("removing a definition cascades per-size values", () => {
    let doc = chestCatalogue(withSizes(emptyDocument(), ["S", "M"]));
    doc = dress(doc, [90, 96]);
    doc = removeMeasurementDefinition(doc, "measurement/chest");
    expect(doc.sizeSet.measurementDefinitions).toHaveLength(0);
    expect(doc.sizeSet.sizes.every((size) => size.measurements.length === 0)).toBe(true);
    expect(() => removeMeasurementDefinition(doc, "measurement/chest")).toThrowError(GradingError);
  });

  it("never implies grading deltas from measurements alone", () => {
    const fixture = buildGradingFixture();
    let doc = createGradingDocument({
      id: "grading/g12c-implies",
      name: "G12C no-auto-grading",
      master: createMasterPattern("master/g12c", "G12C master", fixture.document),
      sizeSet: createSizeSet("sizeset/g12c", "G12C sizes"),
      ruleTable: createRuleTable("ruletable/g12c", "G12C rules"),
      seams: [],
    });
    doc = withSizes(doc, ["S", "M"]);
    doc = chestCatalogue(doc);
    doc = dress(doc, [90, 96]);
    doc = addGradingPoint(doc, createGradingPoint("gp/hem", {
      kind: "corner", panelId: fixture.ids.frontPanel,
      segmentIdA: fixture.ids.segments.ab, segmentIdB: fixture.ids.segments.bc,
    }));
    // A fully measured size system with no rules still cannot move geometry.
    expect(() => deriveSize(doc, "size/s")).toThrowError(/missing-rule/);
  });
});

describe("persistence", () => {
  it("round-trips the full size system through serialization", () => {
    let doc = chestCatalogue(withSizes(emptyDocument(), ["XS", "S", "M", "L", "XL"]));
    doc = dress(doc, [84, 90, 96, 102, 108]);
    doc = duplicateSize(doc, "size/m", "size/m-tall", { label: "M tall" });
    doc = assignMeasurement(doc, "size/m-tall", "measurement/chest", 97, { unit: "cm", source: "custom" });
    const serialized = serializeGradingDocument(doc);
    const parsed = parseGradingDocument(serialized);
    expect(parsed.sizeSet).toEqual(doc.sizeSet);
  });

  it("refuses to serialize a size system with missing measurements", () => {
    let doc = chestCatalogue(withSizes(emptyDocument(), ["S", "M"]));
    doc = dress(doc, [90, 96]);
    const incomplete = removeMeasurement(doc, "size/m", "measurement/chest");
    expect(() => serializeGradingDocument(incomplete)).toThrowError(/missing-measurement/);
  });

  it("measurement edits change the derivation fingerprint deterministically", () => {
    let doc = chestCatalogue(withSizes(emptyDocument(), ["S"]));
    doc = assignMeasurement(doc, "size/s", "measurement/chest", 90, { unit: "cm" });
    const before = fingerprintGradingDocument(doc);
    doc = assignMeasurement(doc, "size/s", "measurement/chest", 91, { unit: "cm" });
    expect(fingerprintGradingDocument(doc)).not.toBe(before);
  });
});
