// G11 FINAL INTEGRATION — commercial acceptance test.
//
// A 4-panel top is engineered for production end to end: sewing boundaries,
// allowances, notches, grainlines, folds, drills, metadata, cut quantities,
// measurements, validation, tech-sheet preview, and export to JSON + DXF.
// The native document stays authoritative; exports are derived artifacts.
import { describe, expect, it } from "vitest";
import { createPatternDocument, movePoint } from "../../src/pattern/cad.js";
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
  deserializeProductionSet,
  serializeProductionSet,
  setPanelMeta,
} from "../../src/cad/production.js";
import { buildLabelFields, centeredGrainline } from "../../src/cad/markings.js";
import { productionReadiness } from "../../src/cad/readiness.js";
import { renderTechSheet } from "../../src/cad/techsheet.js";
import {
  buildExportIR,
  compareIR,
  exportProductionPackage,
  importIRFromJSON,
} from "../../src/cad/export.js";

function panels() {
  let document = createPatternDocument("g11-top", "Production top");
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
  const seg = (r: { panelId: string; loopId: string; segmentIds: string[] }, k: number): string => r.segmentIds[k];
  const side = (r: { panelId: string; loopId: string; segmentIds: string[] }, k: number, reversed = false) => ({
    panelId: r.panelId, loopId: r.loopId, segmentIds: [seg(r, k)], reversed,
  });
  return [
    { id: "seam/shoulder", sideA: side(p.front, 2), sideB: side(p.back, 2, true), stitchCount: 7 },
    { id: "seam/left", sideA: side(p.front, 3), sideB: side(p.back, 3, true), stitchCount: 7 },
    { id: "seam/right", sideA: side(p.front, 1), sideB: side(p.back, 1, true), stitchCount: 7 },
    { id: "seam/sleeve-l", sideA: side(p.front, 3), sideB: side(p.sleeveL, 1), stitchCount: 5 },
    { id: "seam/sleeve-r", sideA: side(p.front, 1), sideB: side(p.sleeveR, 3, true), stitchCount: 5 },
  ];
}

function engineer(document: ReturnType<typeof panels>["document"], p: ReturnType<typeof panels>) {
  let set = createProductionSet();
  const refs = [p.front, p.back, p.sleeveL, p.sleeveR];
  for (const r of refs) {
    set = addAllowance(set, r.panelId, r.loopId, 0.01).set;
    const g = centeredGrainline(document, r.panelId);
    set = addGrainline(set, r.panelId, g.from, g.to).set;
    set = addNotch(set, r.panelId, r.loopId, r.segmentIds[0], 0.5, "single", 0.005).set;
    set = addCutLine(set, r.panelId, r.loopId, "sewing").set;
  }
  // Front extras: fold, drill, dart, annotation, label, metadata.
  set = addFoldLine(set, p.front.panelId, [0.1, 0.1], [0.1, 0.5], "valley", "pleat").set;
  set = addDrillMark(set, p.front.panelId, [0.23, 0.31], "circle").set;
  set = addInternalLine(set, p.front.panelId, [[0.1, 0.1], [0.2, 0.2]], "dart", "front dart").set;
  set = addAnnotation(set, p.front.panelId, [0.23, 0.4], "match pocket").set;
  const fields = buildLabelFields({
    garmentName: "Production top", panelName: "front", panelNumber: 1,
    size: "M", cutQuantity: 2, material: "cotton",
  });
  set = addLabelRegion(set, p.front.panelId, [0.05, 0.05], [0.41, 0.14], fields).set;
  set = setPanelMeta(set, { panelId: p.front.panelId, cutQuantity: 2, section: "body" });
  set = setPanelMeta(set, { panelId: p.back.panelId, cutQuantity: 2, section: "body" });
  set = setPanelMeta(set, { panelId: p.sleeveL.panelId, cutQuantity: 1, section: "sleeve", mirrorPair: "sleeve-right" });
  set = setPanelMeta(set, { panelId: p.sleeveR.panelId, cutQuantity: 1, section: "sleeve", mirrorPair: "sleeve-left" });
  // Front cuts on the allowance line; the rest cut on sewing.
  set = addCutLine(set, p.front.panelId, p.front.loopId, "allowance").set;
  return set;
}

describe("G11 final integration — production-ready top", () => {
  it("engineers, validates, previews, and exports a real pattern file", () => {
    // 1-2. open garment + edit pattern (hem -1 cm on front AND back alike,
    // keeping paired side seams equal — the gate below would flag otherwise).
    const p = panels();
    let document = p.document;
    for (const pid of [p.front.panelId, p.back.panelId, p.sleeveL.panelId, p.sleeveR.panelId]) {
      const hemPts = document.points.filter((pt) => pt.panelId === pid && pt.y === 0);
      for (const pt of hemPts) document = movePoint(document, pid, pt.id, [pt.x, pt.y - 0.01]);
    }

    // 3-10. production engineering.
    const set = engineer(document, { ...p, document });

    // 11. measurements.
    const readinessSeams = seams({ ...p, document });
    const report = productionReadiness(document, readinessSeams, set);

    // 12-13. validate: READY with zero critical errors to resolve.
    expect(report.state).toBe("READY_FOR_EXPORT");
    expect(report.errorCount).toBe(0);
    expect(report.totalCutAreaM2).toBeGreaterThan(0);
    expect(report.panels).toHaveLength(4);
    expect(report.seams).toHaveLength(5);
    for (const row of report.seams) expect(row.withinTolerance).toBe(true);

    // 14. tech-sheet preview.
    const sheet = renderTechSheet(document, set, { title: "Production top" });
    expect(sheet.panelIds).toHaveLength(4);
    expect(sheet.svg).toContain("Production top");

    // 15. export to JSON + DXF with timing/size capture.
    const started = Date.now();
    const pkg = exportProductionPackage(document, readinessSeams, set, { garmentName: "Production top" });
    const elapsedMs = Date.now() - started;
    expect(compareIR(pkg.ir, importIRFromJSON(pkg.json))).toEqual([]);
    expect(pkg.dxf.entityCount).toBeGreaterThan(50);
    expect(pkg.dxf.dxf).toContain("CUT");
    expect(pkg.dxf.warnings.length).toBeGreaterThan(0);
    expect(elapsedMs).toBeLessThan(5000);
    expect(pkg.json.length).toBeGreaterThan(1000);

    // Native source of truth: reload the production set, re-export identical IR.
    const setAgain = deserializeProductionSet(serializeProductionSet(set));
    expect(serializeProductionSet(setAgain)).toBe(serializeProductionSet(set));
    const irAgain = buildExportIR(document, readinessSeams, setAgain, {
      garmentName: "Production top", readiness: report,
    });
    expect(compareIR(pkg.ir, irAgain)).toEqual([]);

    // Edit loop: widen one allowance -> derived artifacts change deterministically.
    const widened = { ...set, allowances: set.allowances.map((a) => ({ ...a, defaultM: 0.015 })) };
    const pkg2 = exportProductionPackage(document, readinessSeams, widened, { garmentName: "Production top" });
    expect(pkg2.dxf.dxf).not.toBe(pkg.dxf.dxf);
    expect(exportProductionPackage(document, readinessSeams, widened, { garmentName: "Production top" }).dxf.dxf)
      .toBe(pkg2.dxf.dxf);
  });
});
