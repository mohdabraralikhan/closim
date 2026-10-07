// G11E tests: IR, JSON round-trip, DXF writer, export gate, adversarial.
import { describe, expect, it } from "vitest";
import {
  addAllowance,
  addCutLine,
  addDrillMark,
  addGrainline,
  addInternalLine,
  addLabelRegion,
  addNotch,
  createProductionSet,
  setPanelMeta,
} from "../../src/cad/production.js";
import { centeredGrainline } from "../../src/cad/markings.js";
import {
  buildExportIR,
  compareIR,
  exportIRToDXF,
  exportIRToJSON,
  exportProductionPackage,
  importIRFromJSON,
} from "../../src/cad/export.js";
import { createPatternDocument, createPanel } from "../../src/pattern/cad.js";
import { addRectPanel } from "../../src/garment/tshirt.js";
import { circleFixture } from "./fixtures.js";
import type { Seam } from "../../src/garment/sewing.js";

function readyGarment() {
  let document = createPatternDocument("g11e", "ready");
  const a = addRectPanel(document, "A", [0, 0], 0.4, 0.3);
  document = a.document;
  const b = addRectPanel(document, "B", [0, 0], 0.4, 0.3);
  document = b.document;
  const seam: Seam = {
    id: "seam/ab",
    sideA: { panelId: a.refs.panelId, loopId: a.refs.loopId, segmentIds: [a.refs.segmentIds[1]], reversed: false },
    sideB: { panelId: b.refs.panelId, loopId: b.refs.loopId, segmentIds: [b.refs.segmentIds[3]], reversed: false },
    stitchCount: 5,
  };
  let set = createProductionSet();
  for (const r of [a.refs, b.refs]) {
    set = addAllowance(set, r.panelId, r.loopId, 0.01).set;
    const g = centeredGrainline(document, r.panelId);
    set = addGrainline(set, r.panelId, g.from, g.to).set;
    set = setPanelMeta(set, { panelId: r.panelId, cutQuantity: 2 });
    set = addCutLine(set, r.panelId, r.loopId, "allowance").set;
  }
  set = addNotch(set, a.refs.panelId, a.refs.loopId, a.refs.segmentIds[0], 0.5, "single", 0.005).set;
  set = addDrillMark(set, b.refs.panelId, [0.2, 0.15], "circle").set;
  set = addInternalLine(set, a.refs.panelId, [[0.05, 0.05], [0.35, 0.25]], "dart", "dart").set;
  set = addLabelRegion(set, a.refs.panelId, [0.05, 0.05], [0.35, 0.12], { panel: "A" }).set;
  return { document, a: a.refs, b: b.refs, seam, set };
}

describe("G11E intermediate representation", () => {
  it("builds a deterministic IR with all layers", () => {
    const { document, seam, set } = readyGarment();
    const ir1 = buildExportIR(document, [seam], set, { garmentName: "Top" });
    const ir2 = buildExportIR(document, [seam], set, { garmentName: "Top" });
    expect(exportIRToJSON(ir1)).toBe(exportIRToJSON(ir2));
    expect(ir1.panels).toHaveLength(2);
    expect(ir1.panels[0].sewing).toHaveLength(4);
    expect(ir1.panels[0].allowance).not.toBeNull();
    expect(ir1.panels[0].notches).toHaveLength(1);
    expect(ir1.panels[0].labelRegions).toHaveLength(1);
    expect(ir1.panels[0].cutQuantity).toBe(2);
    expect(ir1.seams[0].lengthAM).toBeCloseTo(0.3, 9);
    expect(ir1.readiness.state).toBe("UNCHECKED");
  });

  it("round-trips JSON losslessly", () => {
    const { document, seam, set } = readyGarment();
    const ir = buildExportIR(document, [seam], set);
    const back = importIRFromJSON(exportIRToJSON(ir));
    expect(compareIR(ir, back)).toEqual([]);
    expect(() => importIRFromJSON("nope")).toThrowError(/parse/);
    expect(() => importIRFromJSON('{"format":"x","version":1,"panels":[]}')).toThrowError(/shape/);
  });
});

describe("G11E DXF writer", () => {
  it("emits R12 sections, layers, and millimetre geometry", () => {
    const { document, seam, set } = readyGarment();
    const ir = buildExportIR(document, [seam], set);
    const out1 = exportIRToDXF(ir);
    const out2 = exportIRToDXF(ir);
    expect(out1.dxf).toBe(out2.dxf);
    expect(out1.dxf).toContain("SECTION");
    expect(out1.dxf).toContain("ENTITIES");
    expect(out1.dxf).toContain("EOF");
    expect(out1.dxf).toContain("$INSUNITS");
    for (const layer of ["CUT", "SEW", "ALLOWANCE", "NOTCH", "GRAIN", "DRILL", "INTERNAL", "LABEL", "DIM"]) {
      expect(out1.dxf).toContain(`\n2\n${layer}\n`);
    }
    expect(out1.dxf).toContain("LINE");
    expect(out1.dxf).toContain("CIRCLE");
    expect(out1.dxf).toContain("TEXT");
    expect(out1.entityCount).toBeGreaterThan(20);
    // Sewing rect 400x300 mm lands on the CUT layer in millimetres.
    expect(out1.dxf).toContain("400.0000");
    expect(out1.warnings.length).toBeGreaterThan(0); // subset + flattening notes
    expect(out1.warnings.some((w) => w.startsWith("DXF subset"))).toBe(true);
  });

  it("warns on faceted arcs", () => {
    const c = circleFixture(0.1);
    let set = createProductionSet();
    set = addAllowance(set, c.panelId, c.loopId, 0.01).set;
    set = setPanelMeta(set, { panelId: c.panelId, cutQuantity: 1 });
    const g = centeredGrainline(c.document, c.panelId);
    set = addGrainline(set, c.panelId, g.from, g.to).set;
    const ir = buildExportIR(c.document, [], set);
    const out = exportIRToDXF(ir);
    expect(out.warnings.some((w) => w.includes("faceted"))).toBe(true);
  });
});

describe("G11E export gate", () => {
  it("exports a READY garment to JSON + DXF", () => {
    const { document, seam, set } = readyGarment();
    const started = Date.now();
    const pkg = exportProductionPackage(document, [seam], set, { garmentName: "Top" });
    const elapsedMs = Date.now() - started;
    expect(pkg.readiness.state).toBe("READY_FOR_EXPORT");
    expect(pkg.ir.readiness.state).toBe("READY_FOR_EXPORT");
    expect(compareIR(pkg.ir, importIRFromJSON(pkg.json))).toEqual([]);
    expect(pkg.dxf.dxf.length).toBeGreaterThan(1000);
    expect(elapsedMs).toBeLessThan(5000);
  });

  it("refuses INVALID and WARNINGS states without writing files", () => {
    const { document, a, seam, set } = readyGarment();
    // WARNINGS: drop one grainline.
    const noGrain = { ...set, grainlines: set.grainlines.slice(1) };
    expect(() => exportProductionPackage(document, [seam], noGrain)).toThrowError(/WARNINGS/);
    // INVALID: drop cut metadata.
    const noMeta = { ...set, panelMeta: [] };
    expect(() => exportProductionPackage(document, [seam], noMeta)).toThrowError(/INVALID/);
    // INVALID: seam mismatch — sideA bottom (0.4) vs sideB left (0.3).
    const mismatched: Seam = {
      ...seam,
      sideA: { panelId: a.panelId, loopId: a.loopId, segmentIds: [a.segmentIds[0]], reversed: false },
    };
    expect(() => exportProductionPackage(document, [mismatched], set)).toThrowError(/INVALID/);
  });
});

describe("G11E adversarial export", () => {
  it("rejects invalid native documents before any adapter runs", () => {
    let bad = createPatternDocument("bad");
    const p = createPanel(bad, "ghost");
    bad = p.document;
    expect(() => buildExportIR(bad, [], createProductionSet())).toThrowError(/invalid/);
    expect(() => exportProductionPackage(bad, [], createProductionSet())).toThrowError(/INVALID/);
  });

  it("handles huge and tiny coordinates in millimetres", () => {
    let document = createPatternDocument("scale");
    const big = addRectPanel(document, "big", [0, 0], 1e4, 1e4);
    document = big.document;
    let set = createProductionSet();
    set = addAllowance(set, big.refs.panelId, big.refs.loopId, 1).set;
    set = setPanelMeta(set, { panelId: big.refs.panelId, cutQuantity: 1 });
    const g = centeredGrainline(document, big.refs.panelId);
    set = addGrainline(set, big.refs.panelId, g.from, g.to).set;
    const ir = buildExportIR(document, [], set);
    const out = exportIRToDXF(ir);
    expect(out.dxf).toContain("10000000.0000"); // 1e4 m in mm
  });

  it("never drops markings silently: IR preserves every entity", () => {
    const { document, seam, set } = readyGarment();
    const ir = buildExportIR(document, [seam], set);
    expect(ir.panels[0].notches.length).toBe(set.notches.filter((n) => n.panelId === ir.panels[0].panelId).length);
    expect(ir.panels[0].grainlines.length).toBe(1);
    expect(ir.panels[0].drills.length + ir.panels[1].drills.length).toBe(1);
  });
});
