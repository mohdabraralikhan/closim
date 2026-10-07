// G13 FINAL — commercial acceptance scenario.
//
// Start from a production-engineered graded garment (XS–XXL), then walk the
// required workflow end to end:
//
//   open garment -> validate -> size set -> profile -> preview -> export
//   -> validate generated files -> build customer package
//
// Everything asserts on generated artifacts (not internal state), mirrors the
// acceptance criteria, and records timing/package metrics for the G13 report.
import { describe, expect, it } from "vitest";
import { buildExportIR, exportGate, exportIRToJSON, convertM } from "../../src/cad/export-ir.js";
import { exportIRToDXFProfile } from "../../src/cad/dxf-export.js";
import { validateDxfOutput } from "../../src/cad/dxf-validate.js";
import { exportIRToSVG } from "../../src/cad/svg-export.js";
import { exportIRToPDF } from "../../src/cad/pdf-export.js";
import { exportCommercialPackage, sha256Of } from "../../src/cad/export-package.js";
import { exportGradeRules } from "../../src/cad/grade-rules-export.js";
import { gradedGarment } from "./g13-fixtures.js";

const decode = (b: Uint8Array | string): string => (typeof b === "string" ? b : new TextDecoder().decode(b));

describe("G13 FINAL — production export gate", () => {
  it("runs the full commercial workflow for a graded XS–XXL garment", () => {
    // 1. Open garment (deterministic fixture: engineered 2-panel top, 6 sizes).
    const f = gradedGarment(6);
    expect(f.document.panels).toHaveLength(2);
    expect(f.grading.sizes).toHaveLength(6);

    // 2. Validate — strict gate must be READY before anything else.
    const gateStarted = Date.now();
    const decision = exportGate(f.document, f.seams, f.set, "strict");
    const gateMs = Date.now() - gateStarted;
    expect(decision.ok).toBe(true);
    expect(decision.state).toBe("READY_FOR_EXPORT");

    // 3-4. Size set + export profile.
    const ir = buildExportIR(f.document, f.seams, f.set, {
      garmentName: "Graded tee",
      styleId: "style/g13",
      revision: 1,
      grading: f.grading,
      readiness: decision.readiness,
    });
    expect(ir.version).toBe(2);
    expect(ir.units).toBe("m");
    expect(ir.grading.sizes.map((s) => s.label)).toEqual(["XS", "S", "M", "L", "XL", "XXL"]);

    // 5. Preview (true-scale SVG).
    const preview = exportIRToSVG(ir, { title: "Graded tee — preview" });
    expect(preview.svg).toContain('data-true-scale="1"');
    expect(preview.svg).toContain("Graded tee — preview");

    // 6. Export: DXF (single + multi), PDF full-scale + tiled.
    const started = Date.now();
    const dxfSingle = exportIRToDXFProfile(ir, { profile: "aama-style" });
    const dxfMulti = exportIRToDXFProfile(ir, { profile: "aama-style", multiSize: true });
    const pdfFull = exportIRToPDF(ir, { page: "A0", marginMm: 10, mode: "full-scale" });
    const pdfTiled = exportIRToPDF(ir, { page: "A4", marginMm: 10, mode: "tiled", overlapMm: 10 });
    const exportMs = Date.now() - started;
    expect(dxfSingle.units).toBe("mm");
    expect(dxfMulti.pieceCount).toBe(12);
    expect(pdfFull.pageCount).toBe(1);
    expect(pdfTiled.pageCount).toBeGreaterThan(1);

    // 7. Validate generated files (parse what we wrote).
    const reportSingle = validateDxfOutput(dxfSingle.dxf, ir, "aama-style", { layout: dxfSingle.layout });
    const reportMulti = validateDxfOutput(dxfMulti.dxf, ir, "aama-style", { multiSize: true, layout: dxfMulti.layout });
    expect(reportSingle.ok).toBe(true);
    expect(reportMulti.ok).toBe(true);
    expect(reportMulti.pieces).toMatchObject({ detected: 12, expected: 12 });
    const pdfText = new TextDecoder().decode(pdfTiled.pdf);
    expect(pdfText).toContain("%%EOF");
    expect(pdfText).toContain("calibration: exactly 50 mm at 100%");

    // Separate grade-rule artifacts (grade data is never discarded).
    const grades = exportGradeRules(f.grading, { garmentName: "Graded tee", styleId: "style/g13" });
    expect(grades.sizeCount).toBe(6);
    expect(grades.ruleCount).toBe(2);
    const gradesBack = JSON.parse(grades.json);
    expect(gradesBack.rules).toHaveLength(2);

    // 8. Build the customer package (one action).
    const pkgStarted = Date.now();
    const pkg = exportCommercialPackage(f.document, f.seams, f.set, {
      garmentName: "Graded tee",
      revision: 1,
      grading: f.grading,
      dxfMultiSize: true,
      generatedAt: "2026-10-06T00:00:00.000Z",
    });
    const pkgMs = Date.now() - pkgStarted;
    const paths = pkg.files.map((file) => file.path);
    expect(paths).toContain("graded-tee/patterns/graded-tee.sizes.rev1.dxf");
    expect(paths).toContain("graded-tee/patterns/graded-tee.sizes.rev1.a4-tiled.pdf");
    expect(paths).toContain("graded-tee/patterns/graded-tee.sizes.rev1.a0-fullscale.pdf");
    expect(paths).toContain("graded-tee/patterns/graded-tee.sizes.rev1.svg");
    expect(paths).toContain("graded-tee/grading/grade-rules.json");
    expect(paths).toContain("graded-tee/grading/size-chart.csv");
    expect(paths).toContain("graded-tee/metadata/manifest.json");
    expect(paths).toContain("graded-tee/validation/validation-manifest.json");
    expect(paths).toContain("graded-tee/preview/preview.svg");
    expect(paths).toContain("graded-tee/README.md");

    // Manifest is self-describing.
    const manifest = JSON.parse(pkg.manifestJson);
    expect(manifest.productName).toBe("Graded tee");
    expect(manifest.sizeSet.sizes).toEqual(["XS", "S", "M", "L", "XL", "XXL"]);
    expect(manifest.validationStatus).toBe("READY_FOR_EXPORT");
    expect(manifest.units).toBe("mm");
    expect(manifest.pieceCount).toBe(2);
    // Checksums verify every file.
    for (const file of pkg.files) expect(file.sha256).toBe(sha256Of(file.bytes));
    // Reproducibility.
    const pkg2 = exportCommercialPackage(f.document, f.seams, f.set, {
      garmentName: "Graded tee",
      revision: 1,
      grading: f.grading,
      dxfMultiSize: true,
      generatedAt: "2026-10-06T00:00:00.000Z",
    });
    expect(pkg2.files.map((x) => x.path)).toEqual(pkg.files.map((x) => x.path));
    for (let i = 0; i < pkg.files.length; i++) {
      expect(pkg2.files[i].sha256).toBe(pkg.files[i].sha256);
    }

    // Scale safety: XS and XXL differ by exactly 5 size steps x 4 mm per shoulder.
    const xs = ir.grading.sizes[0];
    const xxl = ir.grading.sizes[5];
    expect(xxl.measurements["meas/chest"] - xs.measurements["meas/chest"]).toBeCloseTo(0.25, 9);
    expect(convertM(0.001, "mm")).toBeCloseTo(1, 9);

    // IR round trip.
    expect(JSON.parse(exportIRToJSON(ir))).toEqual(ir);

    // Metrics for the G13 report.
    const packageBytes = pkg.files.reduce((s, file) => s + (typeof file.bytes === "string" ? file.bytes.length : file.bytes.byteLength), 0);
    console.log(`[G13-FINAL] gate=${gateMs}ms export=${exportMs}ms package=${pkgMs}ms files=${pkg.files.length} packageBytes=${packageBytes} dxfEntities=${dxfMulti.entityCount} pdfPages=${pdfTiled.pageCount}`);
    expect(gateMs).toBeLessThan(5000);
    expect(exportMs).toBeLessThan(30000);
    expect(pkgMs).toBeLessThan(30000);
  });
});
