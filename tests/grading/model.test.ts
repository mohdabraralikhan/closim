import { describe, expect, it } from "vitest";
import { createBoundaryLoop, createPanel, createPatternDocument } from "../../src/pattern/cad.js";
import {
  addGradingPoint,
  addRule,
  addSize,
  assignMeasurement,
  createGradingDocument,
  createGradingPoint,
  createMasterPattern,
  createRule,
  createRuleTable,
  createSize,
  createSizeSet,
  insertSize,
  moveSize,
  removeGradingPoint,
  removeSize,
  setBaseSize,
  setRuleDelta,
  setSizeActive,
  GradingError,
} from "../../src/grading/index.js";
import { addCoreSizes, buildGradingFixture, buildTwoPanelDocument } from "./fixtures.js";

function emptyDocument() {
  const fixture = buildTwoPanelDocument();
  const master = createMasterPattern("master/m", "Master", fixture.document);
  return createGradingDocument({
    id: "grading/m",
    name: "Grading",
    master,
    sizeSet: createSizeSet("sizeset/m", "Sizes"),
    ruleTable: createRuleTable("ruletable/m", "Rules"),
  });
}

describe("G12A model: size set operations", () => {
  it("makes the first added size the base and rejects duplicate size IDs", () => {
    let doc = emptyDocument();
    doc = addSize(doc, createSize({ id: "size/m", label: "M" }));
    expect(doc.sizeSet.baseSizeId).toBe("size/m");
    doc = addSize(doc, createSize({ id: "size/l", label: "L" }));
    expect(() => addSize(doc, createSize({ id: "size/m", label: "M duplicate" }))).toThrowError(GradingError);
  });

  it("keeps ordering under insertion and reordering without changing identity", () => {
    let doc = addCoreSizes(emptyDocument());
    doc = insertSize(doc, createSize({ id: "size/xs", label: "XS" }), 0);
    expect(doc.sizeSet.sizes.map((s) => s.id)).toEqual(["size/xs", "size/s", "size/m", "size/l"]);
    doc = moveSize(doc, "size/xs", 2);
    expect(doc.sizeSet.sizes.map((s) => s.id)).toEqual(["size/s", "size/m", "size/xs", "size/l"]);
    expect(doc.sizeSet.baseSizeId).toBe("size/s");
  });

  it("requires a replacement when removing the last remaining base size, otherwise promotes", () => {
    let doc = addCoreSizes(emptyDocument());
    doc = removeSize(doc, "size/s");
    expect(doc.sizeSet.baseSizeId).toBe("size/m");
    doc = removeSize(doc, "size/m");
    expect(doc.sizeSet.baseSizeId).toBe("size/l");
    expect(() => removeSize(doc, "size/l")).toThrowError(/replacement base size/);
  });

  it("rejects unknown sizes and non-finite measurements", () => {
    let doc = addCoreSizes(emptyDocument());
    expect(() => setSizeActive(doc, "size/none", false)).toThrowError(GradingError);
    expect(() => setBaseSize(doc, "size/none")).toThrowError(GradingError);
    expect(() => assignMeasurement(doc, "size/m", "chest", -1)).toThrowError(GradingError);
    doc = assignMeasurement(doc, "size/m", "chest", 0.96);
    const entry = doc.sizeSet.sizes[1].measurements.find((m) => m.measurementId === "chest")!;
    expect(entry.valueM).toBe(0.96);
    expect(entry.unit).toBe("m");
  });

  it("toggles active state without deleting the size", () => {
    let doc = addCoreSizes(emptyDocument());
    doc = setSizeActive(doc, "size/l", false);
    expect(doc.sizeSet.sizes.find((s) => s.id === "size/l")?.active).toBe(false);
    expect(doc.sizeSet.sizes).toHaveLength(3);
  });
});

describe("G12A model: grading points and rules", () => {
  it("rejects duplicate grading point IDs and anchors referencing missing entities", () => {
    const fixture = buildGradingFixture();
    let doc = addCoreSizes(fixture.doc);
    doc = addGradingPoint(doc, createGradingPoint("gp/a", { kind: "point", panelId: fixture.ids.frontPanel, pointId: fixture.ids.points.A }));
    expect(() => addGradingPoint(doc, createGradingPoint("gp/a", { kind: "point", panelId: fixture.ids.frontPanel, pointId: fixture.ids.points.B })))
      .toThrowError(/already exists/);
    expect(() => addGradingPoint(doc, createGradingPoint("gp/b", { kind: "point", panelId: fixture.ids.frontPanel, pointId: "doc/g12fixture/point/999" })))
      .toThrowError(/does not exist/);
    expect(() => addGradingPoint(doc, createGradingPoint("gp/c", { kind: "point", panelId: "doc/g12fixture/panel/999", pointId: fixture.ids.points.A })))
      .toThrowError(/does not exist/);
  });

  it("rejects two grading points that displace the same pattern point", () => {
    const fixture = buildGradingFixture();
    let doc = addCoreSizes(fixture.doc);
    doc = addGradingPoint(doc, createGradingPoint("gp/first", { kind: "point", panelId: fixture.ids.frontPanel, pointId: fixture.ids.points.A }));
    // A corner of ab+da resolves to the same vertex A.
    expect(() => addGradingPoint(doc, createGradingPoint("gp/second", { kind: "corner", panelId: fixture.ids.frontPanel, segmentIdA: fixture.ids.segments.ab, segmentIdB: fixture.ids.segments.da })))
      .toThrowError(/both displace pattern point/);
  });

  it("rejects rules with unknown grading points, duplicates per point, and invalid deltas", () => {
    const fixture = buildGradingFixture();
    let doc = addCoreSizes(fixture.doc);
    doc = addGradingPoint(doc, createGradingPoint("gp/a", { kind: "point", panelId: fixture.ids.frontPanel, pointId: fixture.ids.points.A }));
    doc = addRule(doc, createRule("rule/a", "gp/a", "per-size", { "size/s": [0.01, 0] }));
    expect(() => addRule(doc, createRule("rule/b", "gp/none", "per-size", { "size/s": [0, 0] }))).toThrowError(/unknown grading point/);
    expect(() => addRule(doc, createRule("rule/c", "gp/a", "transition", { "size/s": [0, 0] }))).toThrowError(/already has a rule/);
    expect(() => createRule("rule/d", "gp/a", "per-size", { "size/s": [Number.NaN, 0] })).toThrowError(GradingError);
    expect(() => setRuleDelta(doc, "rule/a", "size/none", [0, 0])).toThrowError(GradingError);
    doc = setRuleDelta(doc, "rule/a", "size/m", [0.02, 0.01]);
    expect(doc.ruleTable.rules[0].deltas["size/m"]).toEqual([0.02, 0.01]);
  });

  it("removing a grading point drops its rule", () => {
    const fixture = buildGradingFixture();
    let doc = addCoreSizes(fixture.doc);
    doc = addGradingPoint(doc, createGradingPoint("gp/a", { kind: "point", panelId: fixture.ids.frontPanel, pointId: fixture.ids.points.A }));
    doc = addRule(doc, createRule("rule/a", "gp/a", "per-size", { "size/s": [0.01, 0] }));
    doc = removeGradingPoint(doc, "gp/a");
    expect(doc.gradingPoints).toHaveLength(0);
    expect(doc.ruleTable.rules).toHaveLength(0);
  });
});

describe("G12A model: master pattern integrity", () => {
  it("rejects master patterns built from invalid pattern documents", () => {
    let broken = createPatternDocument("doc/broken", "Broken");
    const panel = createPanel(broken, "Broken panel");
    broken = panel.document;
    // outer loop with no segments -> open boundary
    const loop = createBoundaryLoop(broken, panel.panelId, "outer");
    broken = loop.document;
    expect(() => createMasterPattern("master/bad", "Bad", broken)).toThrowError(/invalid/);
  });

  it("clone-creating never aliases the input document", () => {
    const fixture = buildTwoPanelDocument();
    const master = createMasterPattern("master/m", "Master", fixture.document);
    master.document.points[0].x = 999;
    expect(fixture.document.points[0].x).toBe(0);
  });
});
