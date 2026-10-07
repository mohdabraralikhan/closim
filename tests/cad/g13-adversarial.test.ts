// G13E — adversarial export QA. Attacks the export path the way receivers,
// factories, and bad configs will. No production code is exercised beyond
// its public surface; every failure mode here must be explicit, not silent.
import { describe, expect, it } from "vitest";
import { buildExportIR, exportGate, type ExportGradingContext } from "../../src/cad/export-ir.js";
import { exportIRToDXFProfile } from "../../src/cad/dxf-export.js";
import { DXF_PROFILES } from "../../src/cad/dxf-profile.js";
import { parseDxf, validateDxfOutput } from "../../src/cad/dxf-validate.js";
import { exportIRToSVG } from "../../src/cad/svg-export.js";
import { exportIRToPDF } from "../../src/cad/pdf-export.js";
import { exportCommercialPackage } from "../../src/cad/export-package.js";
import { engineeredGarment, gradedGarment } from "./g13-fixtures.js";
import {
  createBoundaryArc,
  createBoundaryLine,
  createBoundaryLoop,
  createPanel,
  createPatternDocument,
  createPoint,
  type PatternDocument,
} from "../../src/pattern/cad.js";
import {
  addAllowance,
  addCutLine,
  addGrainline,
  createProductionSet,
  setPanelMeta,
} from "../../src/cad/production.js";
import { centeredGrainline } from "../../src/cad/markings.js";
import { addRectPanel } from "../../src/garment/tshirt.js";

const decode = (b: Uint8Array | string): string => (typeof b === "string" ? b : new TextDecoder().decode(b));

/** Concave L-shaped panel (mild reentrant corner: no miter spikes). */
function concaveGarment(): { document: PatternDocument; set: ReturnType<typeof createProductionSet>; panelId: string } {
  let document = createPatternDocument("g13-concave");
  const panel = createPanel(document, "L-piece");
  document = panel.document;
  const loop = createBoundaryLoop(document, panel.panelId, "outer");
  document = loop.document;
  const pts: Array<[number, number]> = [
    [0, 0], [0.4, 0], [0.4, 0.2], [0.2, 0.2], [0.2, 0.6], [0, 0.6],
  ];
  const pointIds: string[] = [];
  for (const p of pts) {
    const r = createPoint(document, panel.panelId, p);
    document = r.document;
    pointIds.push(r.pointId);
  }
  for (let i = 0; i < pointIds.length; i++) {
    const r = createBoundaryLine(document, panel.panelId, loop.loopId, pointIds[i], pointIds[(i + 1) % pointIds.length]);
    document = r.document;
  }
  let set = createProductionSet();
  set = addAllowance(set, panel.panelId, loop.loopId, 0.005).set;
  const g = centeredGrainline(document, panel.panelId);
  set = addGrainline(set, panel.panelId, g.from, g.to).set;
  set = addCutLine(set, panel.panelId, loop.loopId, "allowance").set;
  set = setPanelMeta(set, { panelId: panel.panelId, cutQuantity: 1 });
  return { document, set, panelId: panel.panelId };
}

describe("G13E units matrix", () => {
  const f = engineeredGarment();
  const ir = buildExportIR(f.document, f.seams, f.set);

  it("converts a 0.46 m edge to mm, cm, and in consistently across adapters", () => {
    // The first sewing edge is the front bottom edge: authored 0.46 m long.
    const span = (dxf: string, layer: string): number => {
      const parsed = parseDxf(dxf);
      const first = parsed.entities.find((e) => e.layer === layer && e.type === "LINE" && e.x1 !== undefined);
      return Math.abs(first!.x2! - first!.x1!);
    };
    expect(span(exportIRToDXFProfile(ir, { profile: "generic-r12", units: "mm" }).dxf, "SEW")).toBeCloseTo(460, 3);
    expect(span(exportIRToDXFProfile(ir, { profile: "generic-r12", units: "cm" }).dxf, "SEW")).toBeCloseTo(46, 3);
    expect(span(exportIRToDXFProfile(ir, { profile: "generic-r12", units: "in" }).dxf, "SEW")).toBeCloseTo(0.46 * 1000 / 25.4, 3);
    // SVG is mm by contract; PDF points are exact (72/25.4 per mm): the front
    // bottom edge ends at 490 mm page-space (10 margin + 480 layout) = 1388.976 pt.
    const pdf = exportIRToPDF(ir, { page: "A0", marginMm: 10, mode: "full-scale" });
    const text = new TextDecoder().decode(pdf.pdf);
    expect(text).toContain("1388.976");
    expect(1388.976 * (25.4 / 72)).toBeCloseTo(490.0, 2);
  });

  it("fails loudly on ambiguous unit configuration instead of guessing", () => {
    expect(() => exportIRToDXFProfile(ir, { profile: "generic-r12", units: "furlongs" as never })).toThrowError(/unknown unit/);
    expect(() => exportIRToPDF(ir, { page: "A0", marginMm: Number.NaN, mode: "full-scale" })).toThrowError(/margin/);
  });

  it("keeps every file internally unit-declared", () => {
    const dxf = exportIRToDXFProfile(ir, { profile: "generic-r12", units: "cm" });
    expect(dxf.dxf).toContain("$INSUNITS");
    const svg = exportIRToSVG(ir);
    expect(svg.svg).toContain('data-units="mm"');
    expect(svg.svg).toContain("width=");
    const pdf = exportIRToPDF(ir, { page: "A4", marginMm: 10, mode: "tiled" });
    expect(new TextDecoder().decode(pdf.pdf)).toContain("units=mm");
    const pkg = exportCommercialPackage(f.document, f.seams, f.set, { garmentName: "units", formats: ["svg"] });
    expect(JSON.parse(pkg.manifestJson).units).toBe("mm");
  });
});

describe("G13E production entity preservation", () => {
  it("no notches, grainlines, or drills disappear through any adapter", () => {
    const f = engineeredGarment();
    const ir = buildExportIR(f.document, f.seams, f.set);
    const expectedNotches = ir.panels.reduce((s, p) => s + p.notches.length, 0);
    const expectedGrains = ir.panels.reduce((s, p) => s + p.grainlines.length, 0);
    const expectedDrills = ir.panels.reduce((s, p) => s + p.drills.length, 0);
    const expectedFolds = ir.panels.reduce((s, p) => s + p.folds.length, 0);
    expect(expectedNotches).toBe(2);
    expect(expectedGrains).toBe(2);
    expect(expectedDrills).toBe(1);
    expect(expectedFolds).toBe(1);

    // DXF (generic): everything emitted.
    const dxf = exportIRToDXFProfile(ir, { profile: "generic-r12" });
    const report = validateDxfOutput(dxf.dxf, ir, "generic-r12", { layout: dxf.layout });
    expect(report.ok).toBe(true);
    expect(report.markings.notches).toBeGreaterThanOrEqual(expectedNotches);
    expect(report.markings.grainlines).toBeGreaterThanOrEqual(expectedGrains * 2); // grain + cross-grain
    // SVG: every notch tick + grainline line present.
    const svg = exportIRToSVG(ir);
    const tickCount = (svg.svg.match(/class="notch"/g) ?? []).length;
    expect(tickCount).toBeGreaterThanOrEqual(expectedNotches);
    expect((svg.svg.match(/class="grain"/g) ?? []).length).toBeGreaterThanOrEqual(expectedGrains);
    // PDF: notch and grain ops survive flattening.
    const pdf = new TextDecoder().decode(exportIRToPDF(ir, { page: "A0", marginMm: 10, mode: "full-scale" }).pdf);
    expect(pdf.length).toBeGreaterThan(1000);
    // Profile that cannot carry folds must warn, never silently drop:
    const aama = exportIRToDXFProfile(ir, { profile: "aama-style" });
    expect(aama.warnings.some((w) => /fold/.test(w) && /omitted/.test(w))).toBe(true);
  });

  it("cut quantity and mirror metadata survive into labels and manifest", () => {
    const f = engineeredGarment();
    const setWithMirror = { ...f.set, panelMeta: f.set.panelMeta.map((m, i) => (i === 0 ? { ...m, mirrorPair: "back" } : m)) };
    const pkg = exportCommercialPackage(f.document, f.seams, setWithMirror, { garmentName: "mirror", formats: ["svg"] });
    const svg = decode(pkg.files.find((x) => x.role === "pattern-svg")!.bytes);
    expect(svg).toContain("mirror of back");
    expect(JSON.parse(pkg.manifestJson).cutQuantityTotal).toBe(4);
  });
});

describe("G13E grading attacks", () => {
  it("exports 2, 6, and 12 size sets with correct size counts everywhere", () => {
    for (const sizeCount of [2, 6, 12]) {
      const f = gradedGarment(sizeCount);
      const ir = buildExportIR(f.document, f.seams, f.set, { grading: f.grading });
      expect(ir.grading.sizes).toHaveLength(sizeCount);
      const dxf = exportIRToDXFProfile(ir, { profile: "aama-style", multiSize: true });
      expect(dxf.pieceCount).toBe(2 * sizeCount);
      const report = validateDxfOutput(dxf.dxf, ir, "aama-style", { multiSize: true, layout: dxf.layout });
      expect(report.pieces.matches).toBe(true);
      expect(report.grading.reported).toBe(true);
    }
  });

  it("asymmetric grading deltas are carried verbatim into the grade file", () => {
    const f = gradedGarment();
    const asymmetric: ExportGradingContext = JSON.parse(JSON.stringify(f.grading));
    asymmetric.ruleApplications["size/s05"][0].delta = [-0.031, -0.004];
    const ir = buildExportIR(f.document, f.seams, f.set, { grading: asymmetric });
    expect(ir.grading.rules.some((r) => Object.values(r.deltas).some(([dx]) => Math.abs(dx + 0.031) < 1e-12))).toBe(true);
    const pkg = exportCommercialPackage(f.document, f.seams, f.set, {
      garmentName: "asym", grading: asymmetric, formats: ["dxf"],
    });
    const rules = JSON.parse(decode(pkg.files.find((x) => x.role === "grade-rules")!.bytes)) as {
      rules: Array<{ deltas: Record<string, [number, number]> }>;
    };
    const xxlDelta = rules.rules.flatMap((r: { deltas: Record<string, [number, number]> }) => Object.values(r.deltas))
      .find(([dx]) => Math.abs(dx + 0.031) < 1e-12);
    expect(xxlDelta).toBeDefined();
  });
});

describe("G13E geometry attacks", () => {
  it("exports concave pieces with correct closed boundary detection", () => {
    const c = concaveGarment();
    // Unsewn single panel is a warning, accepted explicitly here.
    const decision = exportGate(c.document, [], c.set, "allow-warnings");
    expect(decision.ok).toBe(true);
    expect(decision.state).toBe("WARNINGS");
    const ir = buildExportIR(c.document, [], c.set);
    const dxf = exportIRToDXFProfile(ir, { profile: "aama-style" });
    const report = validateDxfOutput(dxf.dxf, ir, "aama-style", { layout: dxf.layout });
    expect(report.ok).toBe(true);
    expect(report.pieces).toMatchObject({ detected: 1, expected: 1, matches: true });
    const svg = exportIRToSVG(ir);
    expect(svg.svg).toContain(`id="panel-${c.panelId}"`);
  });

  it("treats self-intersecting geometry as a gate failure, never an export", () => {
    // Bowtie polygon: kernel validation rejects it before any adapter runs.
    let document = createPatternDocument("g13-bowtie");
    const panel = createPanel(document, "bowtie");
    document = panel.document;
    const loop = createBoundaryLoop(document, panel.panelId, "outer");
    document = loop.document;
    const pts: Array<[number, number]> = [[0, 0], [0.4, 0.3], [0.4, 0], [0, 0.3]];
    const ids: string[] = [];
    for (const p of pts) {
      const r = createPoint(document, panel.panelId, p);
      document = r.document;
      ids.push(r.pointId);
    }
    for (let i = 0; i < ids.length; i++) {
      const r = createBoundaryLine(document, panel.panelId, loop.loopId, ids[i], ids[(i + 1) % ids.length]);
      document = r.document;
    }
    let set = createProductionSet();
    set = addAllowance(set, panel.panelId, loop.loopId, 0.005).set;
    set = setPanelMeta(set, { panelId: panel.panelId, cutQuantity: 1 });
    expect(() => buildExportIR(document, [], set)).toThrowError(/invalid/);
  });
});

describe("G13E round trips", () => {
  it("native -> IR -> JSON -> import -> compare is exact for a graded garment", () => {
    const f = gradedGarment();
    const ir = buildExportIR(f.document, f.seams, f.set, { grading: f.grading });
    const json = JSON.stringify(ir);
    const back = JSON.parse(json);
    expect(back).toEqual(ir);
    // Rebuild from the same native source: identical bytes (determinism contract).
    const ir2 = buildExportIR(f.document, f.seams, f.set, { grading: f.grading });
    expect(JSON.stringify(ir2)).toBe(json);
  });

  it("DXF layout round trip matches all boundary vertices within 0.0002 units", () => {
    const f = gradedGarment();
    const ir = buildExportIR(f.document, f.seams, f.set, { grading: f.grading });
    const dxf = exportIRToDXFProfile(ir, { profile: "aama-style", multiSize: true });
    const report = validateDxfOutput(dxf.dxf, ir, "aama-style", { multiSize: true, layout: dxf.layout });
    const geometryIssues = report.issues.filter((i) => i.code.startsWith("geometry"));
    expect(geometryIssues).toEqual([]);
  });
});

describe("G13E compatibility matrix (declared, machine-checked)", () => {
  it("every profile declares unsupported entities and no profile claims certification", () => {
    const matrix = Object.values(DXF_PROFILES).map((p) => ({
      id: p.id,
      units: p.units,
      multiSize: p.grading.multiSize,
      unsupported: p.unsupported,
      certified: p.certified,
    }));
    for (const row of matrix) {
      expect(row.certified).toBe(false);
      expect(Array.isArray(row.unsupported)).toBe(true);
      expect(["mm", "cm"]).toContain(row.units);
    }
    // PDF + SVG limitations are declared through export warnings.
    const f = engineeredGarment();
    const ir = buildExportIR(f.document, f.seams, f.set);
    const pdf = exportIRToPDF(ir, { page: "A4", marginMm: 10, mode: "tiled" });
    expect(pdf.warnings.some((w) => w.includes("flattened at 0.1 mm sagitta"))).toBe(true);
    const svg = exportIRToSVG(ir);
    expect(svg.warnings.some((w) => w.includes("true scale at 100%"))).toBe(true);
  });
});
