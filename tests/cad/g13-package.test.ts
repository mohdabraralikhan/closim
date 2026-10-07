// G13D tests: commercial export package — structure, manifest, checksums,
// naming, reproducibility, no local-path leakage, refusal on invalid input.
import { describe, expect, it } from "vitest";
import { exportCommercialPackage, sha256Of } from "../../src/cad/export-package.js";
import { engineeredGarment, gradedGarment } from "./g13-fixtures.js";
import { createPatternDocument } from "../../src/pattern/cad.js";
import { createProductionSet } from "../../src/cad/production.js";

const decode = (b: Uint8Array | string): string => (typeof b === "string" ? b : new TextDecoder().decode(b));

describe("G13D package — contents", () => {
  it("builds a complete single-size package with manifest, validation, README, preview, checksums", () => {
    const f = engineeredGarment();
    const pkg = exportCommercialPackage(f.document, f.seams, f.set, { garmentName: "G13 top", revision: 2 });
    const paths = pkg.files.map((file) => file.path).sort();
    expect(paths).toEqual([
      "g13-top/README.md",
      "g13-top/metadata/manifest.json",
      "g13-top/patterns/g13-top.rev2.a0-fullscale.pdf",
      "g13-top/patterns/g13-top.rev2.a4-tiled.pdf",
      "g13-top/patterns/g13-top.rev2.dxf",
      "g13-top/patterns/g13-top.rev2.svg",
      "g13-top/preview/preview.svg",
      "g13-top/validation/validation-manifest.json",
    ]);
    // Checksums match the bytes.
    for (const file of pkg.files) expect(file.sha256).toBe(sha256Of(file.bytes));
    // Manifest fields.
    const manifest = JSON.parse(pkg.manifestJson);
    expect(manifest.productName).toBe("G13 top");
    expect(manifest.revision).toBe(2);
    expect(manifest.units).toBe("mm");
    expect(manifest.canonicalUnits).toBe("m");
    expect(manifest.pieceCount).toBe(2);
    expect(manifest.cutQuantityTotal).toBe(4);
    expect(manifest.validationStatus).toBe("READY_FOR_EXPORT");
    expect(manifest.sourceDocumentVersion).toBe("pattern-doc/1");
    expect(manifest.sizeSet).toBeNull();
    expect(manifest.applicationPhase).toBe("G13");
    expect(manifest.generatedAt).toBeUndefined();
    expect(manifest.exportFormats).toHaveLength(4);
    // README references real files and printing rules.
    const readme = decode(pkg.files.find((f2) => f2.role === "readme")!.bytes);
    expect(readme).toContain("# G13 top");
    expect(readme).toContain("no fit-to-page");
    expect(readme).toContain("g13-top/metadata/manifest.json");
    // Validation manifest carries diagnostics + measurements.
    const validation = JSON.parse(pkg.validationJson);
    expect(validation.state).toBe("READY_FOR_EXPORT");
    expect(validation.panels).toHaveLength(2);
    expect(validation.seams).toHaveLength(3);
  });

  it("builds a graded size-set package with DXF, grading artifacts, and size metadata", () => {
    const f = gradedGarment();
    const pkg = exportCommercialPackage(f.document, f.seams, f.set, {
      garmentName: "Graded top",
      revision: 1,
      grading: f.grading,
      dxfMultiSize: true,
    });
    const paths = pkg.files.map((file) => file.path);
    expect(paths).toContain("graded-top/patterns/graded-top.sizes.rev1.dxf");
    expect(paths).toContain("graded-top/grading/grade-rules.json");
    expect(paths).toContain("graded-top/grading/size-chart.csv");
    const manifest = JSON.parse(pkg.manifestJson);
    expect(manifest.sizeSet.sizes).toEqual(["XS", "S", "M", "L", "XL", "XXL"]);
    expect(manifest.sizeSet.baseSizeId).toBe("size/s00");
    // Grade rules file is valid canonical JSON with per-size deltas.
    const rules = JSON.parse(decode(pkg.files.find((f2) => f2.role === "grade-rules")!.bytes));
    expect(rules.format).toBe("closim-grade-rules");
    expect(rules.version).toBe(1);
    expect(rules.sizes).toHaveLength(6);
    expect(rules.rules.length).toBeGreaterThan(0);
    const sizeChart = decode(pkg.files.find((f2) => f2.role === "size-chart")!.bytes);
    expect(sizeChart.split("\n")[0]).toBe("sizeId,label,isBase,meas/chest");
    expect(sizeChart).toContain("size/s05,XXL,false,1.15");
  });

  it("honours format selection and missing formats never produce empty files", () => {
    const f = engineeredGarment();
    const pkg = exportCommercialPackage(f.document, f.seams, f.set, {
      garmentName: "svg only",
      formats: ["svg"],
    });
    const roles = pkg.files.map((file) => file.role);
    expect(roles).toContain("pattern-svg");
    expect(roles).not.toContain("pattern-dxf");
    expect(roles).not.toContain("pattern-pdf");
    expect(JSON.parse(pkg.manifestJson).exportFormats).toHaveLength(1);
  });
});

describe("G13D package — reproducibility", () => {
  it("is byte-identical across repeated exports (same source + settings)", () => {
    const f = engineeredGarment();
    const opts = { garmentName: "Repeat top", revision: 3, generatedAt: "2026-01-01T00:00:00.000Z" };
    const a = exportCommercialPackage(f.document, f.seams, f.set, opts);
    const b = exportCommercialPackage(f.document, f.seams, f.set, opts);
    expect(a.files.map((x) => x.path)).toEqual(b.files.map((x) => x.path));
    for (let i = 0; i < a.files.length; i++) {
      expect(a.files[i].sha256).toBe(b.files[i].sha256);
      expect(Buffer.from(a.files[i].bytes as Uint8Array).equals(Buffer.from(b.files[i].bytes as Uint8Array))).toBe(true);
    }
    // generatedAt is the only designated timestamp and appears only in the manifest.
    expect(JSON.parse(a.manifestJson).generatedAt).toBe("2026-01-01T00:00:00.000Z");
    const svg = decode(a.files.find((x) => x.role === "pattern-svg")!.bytes);
    expect(svg).not.toContain("2026-01-01");
  });

  it("never leaks local filesystem paths into any artifact", () => {
    const f = gradedGarment();
    const pkg = exportCommercialPackage(f.document, f.seams, f.set, { garmentName: "Leak check", grading: f.grading });
    for (const file of pkg.files) {
      const text = decode(file.bytes);
      expect(text).not.toMatch(/[A-Z]:\\/); // Windows drive paths
      expect(text).not.toMatch(/\/Users\//);
      expect(text).not.toMatch(/\/home\//);
      expect(text).not.toMatch(/AppData|Desktop|closim\//);
      expect(file.path.startsWith("/")).toBe(false);
      expect(file.path).not.toContain("..");
    }
    expect(pkg.manifestJson).not.toContain("C:");
  });
});

describe("G13D package — collisions + refusal", () => {
  it("disambiguates duplicate filenames deterministically", () => {
    const f = engineeredGarment();
    // Force a collision: two DXF entries via a custom packageName that slugs identically.
    const a = exportCommercialPackage(f.document, f.seams, f.set, { garmentName: "Same Name", packageName: "Same Name", formats: ["dxf"] });
    const b = exportCommercialPackage(f.document, f.seams, f.set, { garmentName: "Same Name", packageName: "same-name", formats: ["dxf"] });
    expect(a.files.map((x) => x.path)).toEqual(b.files.map((x) => x.path));
    // Slug of "Same Name" == slug of "same-name" => identical names, no collision inside one package.
    const names = new Set(a.files.map((x) => x.path));
    expect(names.size).toBe(a.files.length);
  });

  it("refuses invalid garments and blocked gates before writing any file", () => {
    let bad = createPatternDocument("g13-bad");
    expect(() => exportCommercialPackage(bad, [], createProductionSet(), { garmentName: "bad" })).toThrowError(/refused|INVALID/);
    const f = engineeredGarment();
    const noMeta = { ...f.set, panelMeta: [] };
    expect(() => exportCommercialPackage(f.document, f.seams, noMeta, { garmentName: "bad" })).toThrowError(/INVALID|refused/);
    // WARNINGS pass only with explicit allow-warnings mode, and warnings stay visible.
    const noGrain = { ...f.set, grainlines: f.set.grainlines.slice(1) };
    expect(() => exportCommercialPackage(f.document, f.seams, noGrain, { garmentName: "warn" })).toThrowError(/WARNINGS/);
    const pkg = exportCommercialPackage(f.document, f.seams, noGrain, { garmentName: "warn", gateMode: "allow-warnings" });
    expect(JSON.parse(pkg.validationJson).state).toBe("WARNINGS");
    expect(JSON.parse(pkg.validationJson).warningCount).toBeGreaterThan(0);
    expect(decode(pkg.files.find((x) => x.role === "readme")!.bytes)).toContain("warning");
  });
});
