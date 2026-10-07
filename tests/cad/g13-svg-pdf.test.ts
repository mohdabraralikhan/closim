// G13C tests: true-scale SVG + PDF export (full-scale and tiled).
import { describe, expect, it } from "vitest";
import { buildExportIR, convertM, exportGate } from "../../src/cad/export-ir.js";
import { exportIRToSVG } from "../../src/cad/svg-export.js";
import { PAGE_PRESETS, exportIRToPDF } from "../../src/cad/pdf-export.js";
import { engineeredGarment, gradedGarment } from "./g13-fixtures.js";
import { circleFixture } from "./fixtures.js";
import { addAllowance, addGrainline, createProductionSet, setPanelMeta } from "../../src/cad/production.js";
import { centeredGrainline } from "../../src/cad/markings.js";

describe("G13C SVG", () => {
  it("declares real-world dimensions through width/height + metadata (1 unit = 1 mm)", () => {
    const f = gradedGarment();
    const decision = exportGate(f.document, f.seams, f.set, "strict");
    expect(decision.ok).toBe(true);
    const ir = buildExportIR(f.document, f.seams, f.set, { grading: f.grading });
    const out = exportIRToSVG(ir, { title: "G13 top" });
    // Layout: 2 pieces of 0.48 m (allowance included) + 0.1 m gap + margins.
    expect(out.widthUnits).toBeCloseTo(20 + 480 + 100 + 480 + 20, 3); // margins 20 mm each side
    expect(out.svg).toContain(`width="${out.widthUnits}mm"`);
    expect(out.svg).toContain(`height="${out.heightUnits}mm"`);
    expect(out.svg).toContain('data-units="mm"');
    expect(out.svg).toContain('data-true-scale="1"');
    expect(out.svg).toContain("true scale");
  });

  it("emits real SVG arc commands for curved boundaries (no flattening)", () => {
    // Circular panel from two semicircular arcs: the SVG path must carry A commands.
    const c = circleFixture(0.1);
    let set = createProductionSet();
    set = addAllowance(set, c.panelId, c.loopId, 0.01).set;
    const g = centeredGrainline(c.document, c.panelId);
    set = addGrainline(set, c.panelId, g.from, g.to).set;
    set = setPanelMeta(set, { panelId: c.panelId, cutQuantity: 1 });
    const ir = buildExportIR(c.document, [], set);
    // Cut = sewing (no cut line authored): the sewing ring is 2 exact arcs.
    expect(ir.panels[0].sewingEdges).toHaveLength(2);
    expect(ir.panels[0].sewingEdges.every((e) => e.kind === "arc")).toBe(true);
    const out = exportIRToSVG(ir);
    expect(out.svg).toMatch(/<path class="cut" d="M [\d.-]+ [\d.-]+ A 100 100 0 [01] [01] [\d.-]+ [\d.-]+ A 100 100 0 [01] [01] [\d.-]+ [\d.-]+ Z"\/>/);
  });

  it("includes all production markings as classed elements", () => {
    const f = engineeredGarment();
    const ir = buildExportIR(f.document, f.seams, f.set);
    const out = exportIRToSVG(ir);
    for (const cls of ["cut", "notch", "grain", "fold", "drill", "internal"]) {
      expect(out.svg).toContain(`class="${cls}`);
    }
    expect(out.svg).toContain("×2"); // cut quantity in piece label
  });

  it("is deterministic", () => {
    const f = engineeredGarment();
    const ir = buildExportIR(f.document, f.seams, f.set);
    expect(exportIRToSVG(ir).svg).toBe(exportIRToSVG(ir).svg);
  });
});

describe("G13C PDF", () => {
  const f = engineeredGarment();
  const ir = buildExportIR(f.document, f.seams, f.set);

  it("produces a structurally valid deterministic PDF", () => {
    const out = exportIRToPDF(ir, { page: "A4", marginMm: 10, mode: "full-scale" });
    const text = new TextDecoder().decode(out.pdf);
    expect(text.startsWith("%PDF-1.4")).toBe(true);
    expect(text).toContain("%%EOF");
    expect(text).toContain("/Type /Catalog");
    expect(text).toContain("/Count 1");
    // Info dictionary declares units + scale for machine checks.
    expect(text).toContain("units=mm");
    expect(text).toContain("scale=100%");
    const out2 = exportIRToPDF(ir, { page: "A4", marginMm: 10, mode: "full-scale" });
    expect(Buffer.from(out.pdf).equals(Buffer.from(out2.pdf))).toBe(true);
  });

  it("sizes full-scale pages to content + margins (never fit-to-page)", () => {
    const out = exportIRToPDF(ir, { page: "A0", marginMm: 10, mode: "full-scale" });
    // Layout = 2 x 480 mm + 100 mm gap; page = layout + 2 x 10 mm margins + header.
    expect(out.pageSizesMm[0].widthMm).toBeCloseTo(1080, 3);
    expect(out.pageCount).toBe(1);
    // MediaBox in points matches the declared mm size exactly (72/25.4 per mm).
    const text = new TextDecoder().decode(out.pdf);
    const media = text.match(/\/MediaBox \[0 0 ([\d.]+) ([\d.]+)\]/)!;
    expect(Number(media[1])).toBeCloseTo(1080 * (72 / 25.4), 2);
    expect(text).toContain("SCALE 100%");
  });

  it("tiles deterministically across A4 with registration, calibration, and page numbers", () => {
    const out = exportIRToPDF(ir, { page: "A4", marginMm: 10, mode: "tiled", overlapMm: 10, calibrationMm: 50 });
    const expectedCols = Math.ceil((1080 - 10) / (190 - 10));
    const expectedRows = Math.ceil((567 - 10) / (271 - 10));
    expect(out.pageCount).toBe(expectedCols * expectedRows);
    const text = new TextDecoder().decode(out.pdf);
    expect(text).toContain("calibration: exactly 50 mm at 100%");
    expect(text).toContain("tile 1/" + expectedCols + " col");
    expect(text).toContain("page 1");
    if (out.pageCount > 1) expect(text).toContain("page 2");
    expect(text).toContain("next ->");
    const out2 = exportIRToPDF(ir, { page: "A4", marginMm: 10, mode: "tiled", overlapMm: 10, calibrationMm: 50 });
    expect(Buffer.from(out.pdf).equals(Buffer.from(out2.pdf))).toBe(true);
  });

  it("draws a calibration square with exact millimetre dimensions in the content stream", () => {
    const out = exportIRToPDF(ir, { page: "A4", marginMm: 10, mode: "tiled", calibrationMm: 50 });
    const text = new TextDecoder().decode(out.pdf);
    const expectedPt = 50 * (72 / 25.4);
    // Extract every 4-point closed stroked path (rect): calibration squares use
    // stroke 0.4 mm = 1.134 pt; label boxes use 0.2 mm = 0.567 pt.
    const rects: Array<{ w: number; h: number }> = [];
    const re = /([\d.-]+) ([\d.-]+) m\n([\d.-]+) ([\d.-]+) l\n([\d.-]+) ([\d.-]+) l\n([\d.-]+) ([\d.-]+) l\nh\n([\d.]+) w S/g;
    for (const m of text.matchAll(re)) {
      const strokePt = Number(m[9]);
      if (Math.abs(strokePt - 0.4 * (72 / 25.4)) > 0.02) continue;
      const x0 = Number(m[1]), y0 = Number(m[2]), x1 = Number(m[3]), y1 = Number(m[4]);
      const x2 = Number(m[5]), y2 = Number(m[6]), x3 = Number(m[7]), y3 = Number(m[8]);
      rects.push({ w: Math.abs(x1 - x0), h: Math.abs(y2 - y1) });
      void x2; void x3; void y3;
    }
    const calibration = rects.filter((r) => Math.abs(r.w - expectedPt) < 0.05 && Math.abs(r.h - expectedPt) < 0.05);
    expect(calibration.length).toBeGreaterThanOrEqual(out.pageCount);
    for (const r of calibration.slice(0, out.pageCount)) {
      expect(r.w).toBeCloseTo(expectedPt, 1);
      expect(r.h).toBeCloseTo(expectedPt, 1);
    }
  });

  it("respects the configurable page-size table (custom + preset)", () => {
    expect(PAGE_PRESETS.A4).toEqual({ id: "A4", widthMm: 210, heightMm: 297 });
    const custom = { id: "custom-roll", widthMm: 1300, heightMm: 3000 };
    const out = exportIRToPDF(ir, { page: custom, marginMm: 5, mode: "full-scale" });
    const text = new TextDecoder().decode(out.pdf);
    expect(text).toContain("/MediaBox [0 0");
    expect(out.pageSizesMm[0].id).toBe("custom-roll");
    expect(() => exportIRToPDF(ir, { page: "bogus", marginMm: 10, mode: "full-scale" })).toThrowError(/unknown page preset/);
  });

  it("carries multiple sizes when the IR is graded (per-page scale text intact)", () => {
    const g = gradedGarment();
    const gir = buildExportIR(g.document, g.seams, g.set, { grading: g.grading });
    const out = exportIRToPDF(gir, { page: "A3", marginMm: 8, mode: "tiled" });
    expect(out.pageCount).toBeGreaterThan(1);
    const text = new TextDecoder().decode(out.pdf);
    expect((text.match(/SCALE 100%/g) ?? []).length).toBe(out.pageCount);
  });
});

describe("G13C unit conversion consistency", () => {
  it("PDF point coordinates match SVG mm coordinates exactly", () => {
    // 0.46 m: SVG mm = 460; PDF pt = 460 * 72/25.4 = 1303.937...
    expect(convertM(0.46, "mm")).toBeCloseTo(460, 6);
    expect(convertM(0.46, "mm") * (72 / 25.4)).toBeCloseTo(1303.9370078740158, 6);
  });
});
