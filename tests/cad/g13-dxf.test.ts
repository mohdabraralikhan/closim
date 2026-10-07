// G13B tests: profile-driven DXF export, post-write validation, round trip.
import { describe, expect, it } from "vitest";
import { buildExportIR, exportGate } from "../../src/cad/export-ir.js";
import { exportIRToDXFProfile } from "../../src/cad/dxf-export.js";
import { DXF_PROFILES } from "../../src/cad/dxf-profile.js";
import { parseDxf, validateDxfOutput } from "../../src/cad/dxf-validate.js";
import { engineeredGarment, gradedGarment } from "./g13-fixtures.js";
import { createPatternDocument } from "../../src/pattern/cad.js";
import { addRectPanel } from "../../src/garment/tshirt.js";
import { addAllowance, addGrainline, createProductionSet, setPanelMeta } from "../../src/cad/production.js";
import { centeredGrainline } from "../../src/cad/markings.js";

describe("G13B profiles", () => {
  it("declares non-certified profiles with explicit layer conventions", () => {
    for (const id of ["generic-r12", "aama-style", "astm-oriented"] as const) {
      const p = DXF_PROFILES[id];
      expect(p.certified).toBe(false);
      expect(p.layers.boundary).toBeDefined();
      expect(p.complianceNote.length).toBeGreaterThan(40);
      expect(Object.keys(p.layers)).toContain("notch");
      expect(Object.keys(p.layers)).toContain("grain");
      expect(Object.keys(p.layers)).toContain("drill");
    }
    expect(DXF_PROFILES["aama-style"].layers.boundary).toBe("1");
    expect(DXF_PROFILES["aama-style"].layers.sewing).toBe("14");
    expect(DXF_PROFILES["astm-oriented"].units).toBe("cm");
  });
});

describe("G13B DXF writer", () => {
  it("emits deterministic single-size DXF with exact arcs, correct units, and warnings", () => {
    const f = gradedGarment();
    const decision = exportGate(f.document, f.seams, f.set, "strict");
    expect(decision.ok).toBe(true);
    const ir = buildExportIR(f.document, f.seams, f.set, { grading: f.grading });
    const a = exportIRToDXFProfile(ir, { profile: "aama-style" });
    const b = exportIRToDXFProfile(ir, { profile: "aama-style" });
    expect(a.dxf).toBe(b.dxf);
    expect(a.units).toBe("mm");
    expect(a.pieceCount).toBe(2);
    expect(a.dxf).toContain("$INSUNITS");
    expect(a.dxf).toContain("AC1009");
    // Numeric AAMA-style layers present as LAYER table entries.
    for (const layer of ["1", "8", "10", "11", "13", "14", "16"]) {
      expect(a.dxf).toContain(`\n2\n${layer}\n`);
    }
    // Single-size grading notice is present (grade data never silently dropped).
    expect(a.warnings.some((w) => w.includes("single-size DXF"))).toBe(true);
    expect(a.warnings.some((w) => /NOT certified/i.test(w))).toBe(true);
    expect(a.entityCount).toBeGreaterThan(20);
  });

  it("emits multi-size DXF copies per size with size labels", () => {
    const f = gradedGarment();
    const ir = buildExportIR(f.document, f.seams, f.set, { grading: f.grading });
    const out = exportIRToDXFProfile(ir, { profile: "aama-style", multiSize: true });
    expect(out.pieceCount).toBe(12); // 2 panels x 6 sizes
    // Band offsets: each size's pieces sit at a distinct y range (never overlapping).
    const bySize = new Map<string, Set<number>>();
    for (const l of out.layout) {
      if (!bySize.has(l.sizeLabel ?? "")) bySize.set(l.sizeLabel ?? "", new Set());
      bySize.get(l.sizeLabel ?? "")!.add(l.oy);
    }
    expect(bySize.size).toBe(6);
    for (const oys of bySize.values()) expect(oys.size).toBe(1);
    expect(out.warnings.some((w) => w.includes("multi-size DXF"))).toBe(true);
    const parsed = parseDxf(out.dxf);
    for (const size of ["XS", "S", "M", "L", "XL", "XXL"]) {
      expect(parsed.textValues.some((t) => t.includes(`(${size})`))).toBe(true);
    }
  });

  it("round-trips through the parser and passes structural validation", () => {
    const f = gradedGarment();
    const ir = buildExportIR(f.document, f.seams, f.set, { grading: f.grading });
    const out = exportIRToDXFProfile(ir, { profile: "aama-style", multiSize: true });
    const report = validateDxfOutput(out.dxf, ir, "aama-style", { multiSize: true, layout: out.layout });
    expect(report.ok).toBe(true);
    expect(report.issues).toEqual([]);
    expect(report.units.consistent).toBe(true);
    expect(report.pieces).toEqual({ detected: 12, expected: 12, matches: true });
    expect(report.markings.notches).toBeGreaterThanOrEqual(4); // 2 panels x 1 notch x 6 sizes
    expect(report.markings.grainlines).toBeGreaterThanOrEqual(12);
    expect(report.markings.drills).toBeGreaterThanOrEqual(6);
    expect(report.grading).toEqual({ present: true, reported: true });
  });

  it("generic profile uses named layers and supports dimensions", () => {
    const f = engineeredGarment();
    const ir = buildExportIR(f.document, f.seams, f.set);
    const out = exportIRToDXFProfile(ir, { profile: "generic-r12" });
    for (const layer of ["CUT", "SEW", "ALLOWANCE", "NOTCH", "GRAIN", "FOLD", "DRILL", "INTERNAL", "LABEL", "DIM"]) {
      expect(out.dxf).toContain(`\n2\n${layer}\n`);
    }
    expect(out.warnings.some((w) => w.includes("generic DXF R12"))).toBe(true);
  });

  it("astm-oriented profile exports in centimetres with declared insunits", () => {
    const f = engineeredGarment();
    const ir = buildExportIR(f.document, f.seams, f.set);
    const out = exportIRToDXFProfile(ir, { profile: "astm-oriented" });
    expect(out.units).toBe("cm");
    const parsed = parseDxf(out.dxf);
    expect(parsed.headerVars["$INSUNITS"]).toBe("5");
    // Layout extent: 2 pieces (0.48 m each incl. allowance) + 0.1 m gap = 1.06 m = 106 cm.
    const boundaryXs = parsed.entities
      .filter((e) => e.layer === "1" && e.type === "LINE" && e.x1 !== undefined)
      .flatMap((e) => [e.x1!, e.x2!]);
    expect(Math.max(...boundaryXs)).toBeCloseTo(106.0, 4);
    // A mm export would reach 1060 — confirm centimetre scaling is real.
    expect(Math.max(...boundaryXs)).toBeLessThan(200);
  });

  it("reports unsupported entities as warnings instead of dropping them silently", () => {
    const f = engineeredGarment();
    const ir = buildExportIR(f.document, f.seams, f.set);
    // aama-style does not support folds/construction.
    const out = exportIRToDXFProfile(ir, { profile: "aama-style" });
    expect(out.warnings.some((w) => /fold/.test(w) && /omitted/.test(w))).toBe(true);
    const report = validateDxfOutput(out.dxf, ir, "aama-style");
    expect(report.unsupportedReported).toContain("fold lines omitted by profile");
  });
});

describe("G13B validation failures", () => {
  it("flags piece-count mismatch and wrong units", () => {
    const f = engineeredGarment();
    const ir = buildExportIR(f.document, f.seams, f.set);
    const out = exportIRToDXFProfile(ir, { profile: "aama-style" });
    // Corrupt: remove one complete LINE entity block from the boundary layer.
    const start = out.dxf.indexOf("\n0\nLINE\n");
    const nextEntity = out.dxf.indexOf("\n0\n", start + 5);
    const broken = out.dxf.slice(0, start + 1) + out.dxf.slice(nextEntity + 1);
    expect(broken.length).toBeLessThan(out.dxf.length);
    const report = validateDxfOutput(broken, ir, "aama-style");
    expect(report.ok).toBe(false);
    expect(report.issues.some((i) => i.code === "piece-count")).toBe(true);
    // Wrong units header ($INSUNITS 4=mm -> 1=inches).
    const badUnits = out.dxf.replace("70\n4\n0\nENDSEC", "70\n1\n0\nENDSEC");
    expect(badUnits).not.toBe(out.dxf);
    const report2 = validateDxfOutput(badUnits, ir, "aama-style");
    expect(report2.units.consistent).toBe(false);
  });

  it("rejects multiSize without grading data", () => {
    const f = engineeredGarment();
    const ir = buildExportIR(f.document, f.seams, f.set);
    expect(() => exportIRToDXFProfile(ir, { profile: "aama-style", multiSize: true })).toThrowError(/no grading sizes/);
  });
});

describe("G13B adversarial geometry", () => {
  it("handles tiny pieces without precision collapse", () => {
    let document = createPatternDocument("g13-tiny");
    const panel = addRectPanel(document, "tiny", [0, 0], 0.02, 0.015);
    document = panel.document;
    let set = createProductionSet();
    set = addAllowance(set, panel.refs.panelId, panel.refs.loopId, 0.002).set;
    const g = centeredGrainline(document, panel.refs.panelId);
    set = addGrainline(set, panel.refs.panelId, g.from, g.to).set;
    set = setPanelMeta(set, { panelId: panel.refs.panelId, cutQuantity: 1 });
    const ir = buildExportIR(document, [], set);
    const out = exportIRToDXFProfile(ir, { profile: "aama-style" });
    expect(out.dxf).toContain("22.0000"); // 0.022 m cut ring extent -> 22 mm
    const report = validateDxfOutput(out.dxf, ir, "aama-style");
    expect(report.pieces.matches).toBe(true);
  });

  it("handles very large coordinates deterministically", () => {
    let document = createPatternDocument("g13-huge");
    const panel = addRectPanel(document, "huge", [0, 0], 1000, 500);
    document = panel.document;
    let set = createProductionSet();
    set = addAllowance(set, panel.refs.panelId, panel.refs.loopId, 0.5).set;
    const g = centeredGrainline(document, panel.refs.panelId);
    set = addGrainline(set, panel.refs.panelId, g.from, g.to).set;
    set = setPanelMeta(set, { panelId: panel.refs.panelId, cutQuantity: 1 });
    const ir = buildExportIR(document, [], set);
    const out = exportIRToDXFProfile(ir, { profile: "aama-style" });
    expect(out.dxf).toContain("1001000.0000"); // 1001 m cut-ring extent in mm
    expect(out.dxf).toBe(exportIRToDXFProfile(ir, { profile: "aama-style" }).dxf);
  });

  it("escapes non-ASCII labels deterministically", () => {
    const f = engineeredGarment();
    const ir = buildExportIR(f.document, f.seams, f.set, { garmentName: "Ünïcode – top ✓" });
    const out = exportIRToDXFProfile(ir, { profile: "aama-style" });
    // Non-ASCII collapsed to '?' placeholders, never raw multi-byte text.
    const parsed = parseDxf(out.dxf);
    for (const t of parsed.textValues) expect(t).toMatch(/^[\x20-\x7E]*$/);
  });
});
