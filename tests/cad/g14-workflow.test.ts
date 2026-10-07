// G14 FINAL — marker-making acceptance scenario.
//
// Walks the required workflow end to end on the graded production garment:
//
//   garment -> gate -> IR -> marker request (quantities + mirrors) -> nest
//   -> efficiency report -> marker SVG -> whole-lay package (ZIP)
//
// Everything asserts on generated artifacts and logs the G14 metrics line
// for the report.
import { describe, expect, it } from "vitest";
import { exportGate } from "../../src/cad/export-ir.js";
import { buildMarker } from "../../src/cad/marker.js";
import { exportMarkerSVG } from "../../src/cad/marker-svg.js";
import { exportCommercialPackage } from "../../src/cad/export-package.js";
import { gradedGarment } from "./g13-fixtures.js";

describe("G14 FINAL — marker making acceptance", () => {
  it("runs the full nesting workflow for a graded XS-XXL garment", () => {
    // 1. Garment + gate (same production-engineered fixture as G13).
    const f = gradedGarment(6);
    const gateStarted = Date.now();
    const decision = exportGate(f.document, f.seams, f.set, "strict");
    const gateMs = Date.now() - gateStarted;
    expect(decision.ok).toBe(true);
    expect(decision.state).toBe("READY_FOR_EXPORT");

    // 2. Marker request: every panel in every size, mirrors resolved,
    //    cut quantities honoured (2 of each from panelMeta).
    const pieces: Array<{ panelId: string; sizeLabel: string; quantity: number }> = [];
    for (const size of f.grading.sizes) {
      for (const panel of f.document.panels) {
        pieces.push({ panelId: panel.id, sizeLabel: size.label, quantity: 2 });
    }
    }
    const started = Date.now();
    const marker = buildMarker(
      f.document,
      f.seams,
      f.set,
      pieces,
      // 50 mm placement grid: deterministic coarse nesting, fast enough for
      // the acceptance run (fine grids trade runtime for tightness).
      { widthM: 1.6, colStepM: 0.05, rowStepM: 0.05, garmentName: "Graded tee", styleId: "style/g13", revision: 1 },
    );
    const markerMs = Date.now() - started;

    // 3. Every requested copy is placed; nothing declined.
    expect(marker.placements).toHaveLength(2 * 6 * 2);
    expect(marker.declined).toHaveLength(0);
    expect(marker.sizes).toEqual(["XS", "S", "M", "L", "XL", "XXL"]);

    // 4. Efficiency is reported on the used frame, in true-scale m^2.
    expect(marker.efficiency.frameAreaM2).toBeCloseTo(marker.lengthM * marker.widthM, 9);
    expect(marker.efficiency.pieceAreaM2).toBeGreaterThan(0);
    expect(marker.efficiency.efficiency).toBeGreaterThan(0.3);
    expect(marker.efficiency.efficiency).toBeLessThan(1);
    // 24 cut pieces of 0.48x0.64 => area sanity.
    expect(marker.efficiency.pieceAreaM2).toBeCloseTo(24 * 0.48 * 0.64, 9);

    // 5. Determinism: same garment + request => identical placement layout.
    const f2 = gradedGarment(6);
    const marker2 = buildMarker(
      f2.document,
      f2.seams,
      f2.set,
      pieces,
      { widthM: 1.6, colStepM: 0.05, rowStepM: 0.05, garmentName: "Graded tee", styleId: "style/g13", revision: 1 },
    );
    expect(JSON.stringify(marker2.placements)).toBe(JSON.stringify(marker.placements));
    expect(marker2.lengthM).toBe(marker.lengthM);

    // 6. Marker SVG (true scale) + whole-lay package bundle.
    const svg = exportMarkerSVG(marker, { title: "Graded tee — marker" });
    expect(svg.svg).toContain('data-true-scale="1"');
    expect(svg.svg).toContain("efficiency");
    const pkg = exportCommercialPackage(f.document, f.seams, f.set, {
      garmentName: "Graded tee",
      grading: f.grading,
      formats: ["dxf", "svg", "zip"],
    });
    const zip = pkg.files.find((file) => file.path.endsWith(".zip"));
    expect(zip).toBeDefined();
    const packageBytes = (zip!.bytes as Uint8Array).length;

    console.log(
      `[G14-FINAL] gate=${gateMs}ms marker=${markerMs}ms pieces=${marker.placements.length} ` +
      `widthM=${marker.widthM} lengthM=${marker.lengthM.toFixed(3)} ` +
      `efficiency=${(marker.efficiency.efficiency * 100).toFixed(1)}% ` +
      `svg=${svg.svg.length}B packageFiles=${pkg.files.length} packageZipBytes=${packageBytes}`,
    );

    void gateMs; // logged above
  });
});
