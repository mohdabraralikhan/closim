// G14-era follow-up tests — ZIP bundling of the export package and the
// foreign-file apparel DXF importer (interop).
import { describe, expect, it } from "vitest";
import { execSync } from "node:child_process";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { exportCommercialPackage } from "../../src/cad/export-package.js";
import { crc32, createZipArchive } from "../../src/cad/zip.js";
import { importApparelDxf } from "../../src/cad/dxf-import.js";
import { exportIRToDXFProfile } from "../../src/cad/dxf-export.js";
import { buildExportIR2 } from "../../src/cad/index.js";
import { engineeredGarment } from "./g13-fixtures.js";

describe("G14 package: ZIP bundle", () => {
  it("emits a single distributable archive when 'zip' is in formats", () => {
    const f = engineeredGarment();
    const pkg = exportCommercialPackage(f.document, f.seams, f.set, {
      garmentName: "G14 zip top",
      formats: ["dxf", "svg", "zip"],
    });
    const zipFile = pkg.files.find((file) => file.path.endsWith(".zip"));
    expect(zipFile).toBeDefined();
    expect(zipFile!.role).toBe("package-zip");
    // Zip must contain every other package file, path-exact.
    const inner = pkg.files.filter((file) => !file.path.endsWith(".zip"));
    expect(inner.length).toBeGreaterThanOrEqual(5);
    const zipBytes = zipFile!.bytes as Uint8Array;
    expect(zipBytes[0]).toBe(0x50);
    expect(zipBytes[1]).toBe(0x4b);
    const innerNames = inner.map((file) => file.path);
    const namesText = new TextDecoder().decode(zipBytes);
    for (const name of innerNames) expect(namesText).toContain(name);
  });

  it("is byte-identical across runs (deterministic) and CRCs verify", () => {
    const a = engineeredGarment();
    const b = engineeredGarment();
    const opts = {
      garmentName: "G14 determinism",
      formats: ["dxf", "svg", "zip"] as const,
    };
    const p1 = exportCommercialPackage(a.document, a.seams, a.set, opts);
    const p2 = exportCommercialPackage(b.document, b.seams, b.set, opts);
    const z1 = p1.files.find((file) => file.path.endsWith(".zip"))!.bytes as Uint8Array;
    const z2 = p2.files.find((file) => file.path.endsWith(".zip"))!.bytes as Uint8Array;
    expect(Buffer.from(z1).equals(Buffer.from(z2))).toBe(true);
    // CRC32 self-check on a known vector + every stored entry is verified by
    // the external unzip tool in the next test; here we check the algorithm.
    expect(crc32(new TextEncoder().encode("123456789"))).toBe(0xcbf43926);
  });

  it("produces an archive a standard unzip tool accepts and verifies", () => {
    const f = engineeredGarment();
    const pkg = exportCommercialPackage(f.document, f.seams, f.set, {
      garmentName: "G14 unzip check",
      formats: ["dxf", "svg", "zip"],
    });
    const zipFile = pkg.files.find((file) => file.path.endsWith(".zip"))!;
    const dir = mkdtempSync(join(tmpdir(), "closim-zip-"));
    const zipPath = join(dir, "pkg.zip");
    try {
      writeFileSync(zipPath, zipFile.bytes as Uint8Array);
      // Python's zipfile module is an independent third-party ZIP reader:
      // testzip() streams every entry and verifies every CRC-32.
      const pyPath = join(dir, "checkzip.py");
      writeFileSync(
        pyPath,
        ["import zipfile,sys",
          "z = zipfile.ZipFile(sys.argv[1])",
          "bad = z.testzip()",
          "print('entries=', len(z.namelist()), 'bad=', bad)",
        ].join("\n") + "\n",
      );
      const out = execSync(`python "${pyPath}" "${zipPath}"`, { encoding: "utf8" });
      expect(out).toContain("bad= None");
      expect(Number(/entries=\s*(\d+)/.exec(out)![1])).toBeGreaterThanOrEqual(5);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("still works without 'zip' and never checksums the archive into itself", () => {
    const f = engineeredGarment();
    const pkg = exportCommercialPackage(f.document, f.seams, f.set, {
      garmentName: "G14 no zip",
      formats: ["dxf"] as const,
    });
    expect(pkg.files.some((file) => file.path.endsWith(".zip"))).toBe(false);
    const withZip = exportCommercialPackage(f.document, f.seams, f.set, {
      garmentName: "G14 no zip",
      formats: ["dxf", "zip"] as const,
    });
    const nonZip = withZip.files.filter((file) => !file.path.endsWith(".zip"));
    const z = withZip.files.find((file) => file.path.endsWith(".zip"))!;
    const zBytes = Buffer.from(z.bytes as Uint8Array);
    for (const file of nonZip) {
      // The zip contains the PRE-zip payload bytes of each file; its own
      // checksum was computed after the payload set was fixed, so the zip's
      // sha256 in the file list describes the archive itself (not cyclic —
      // the archive does not include itself).
      expect(zBytes.includes(Buffer.from(file.sha256, "hex"))).toBe(false);
    }
  });
});

describe("G14 interop: foreign-file DXF importer", () => {
  it("reads our own aama-style export back into pieces (units, layer conventions)", () => {
    const f = engineeredGarment();
    const ir = buildExportIR2(f.document, f.seams, f.set, { garmentName: "G14 interop" });
    const dxf = exportIRToDXFProfile(ir, { profile: "aama-style", units: "mm" });
    const report = importApparelDxf(dxf.dxf, { profile: "aama-style" });
    expect(report.ok).toBe(true);
    expect(report.units).toBe("mm");
    expect(report.pieces).toHaveLength(2);
    const front = report.pieces.find((p) => p.ringM.length > 0)!;
    // Cut boundary is the sewing ring + 10 mm allowance => ~480 x 640 mm.
    const xs = front.ringM.map((p) => p[0]);
    const ys = front.ringM.map((p) => p[1]);
    const w = Math.max(...xs) - Math.min(...xs);
    const h = Math.max(...ys) - Math.min(...ys);
    expect(w).toBeCloseTo(0.48, 3);
    expect(h).toBeCloseTo(0.64, 3);
  });

  it("rejects a file with no $INSUNITS unless units are declared (no guessing)", () => {
    const dxfNoUnits = [
      "0", "SECTION", "2", "HEADER", "0", "ENDSEC",
      "0", "SECTION", "2", "TABLES",
      "0", "LAYER", "2", "1", "70", "0", "62", "7", "6", "CONTINUOUS",
      "0", "ENDSEC",
      "0", "SECTION", "2", "ENTITIES",
      "0", "LINE", "8", "1", "10", "0", "20", "0", "11", "100", "21", "0",
      "0", "LINE", "8", "1", "10", "100", "20", "0", "11", "100", "21", "50",
      "0", "LINE", "8", "1", "10", "100", "20", "50", "11", "0", "21", "50",
      "0", "LINE", "8", "1", "10", "0", "20", "50", "11", "0", "21", "0",
      "0", "ENDSEC", "0", "EOF",
    ].join("\n") + "\n";
    const failed = importApparelDxf(dxfNoUnits);
    expect(failed.ok).toBe(false);
    expect(failed.mismatches.some((m) => m.code === "insunits-missing")).toBe(true);
    const declared = importApparelDxf(dxfNoUnits, { units: "mm" });
    expect(declared.ok).toBe(true);
    expect(declared.mismatches.some((m) => m.code === "insunits-missing")).toBe(true);
    expect(declared.units).toBe("mm");
  });

  it("reports mismatches honestly: unknown $INSUNITS and foreign layers", () => {
    const dxfWeird = [
      "0", "SECTION", "2", "HEADER",
      "9", "$INSUNITS", "70", "21",
      "0", "ENDSEC",
      "0", "SECTION", "2", "TABLES",
      "0", "LAYER", "2", "CUT_DOUBT", "70", "0", "62", "7", "6", "CONTINUOUS",
      "0", "ENDSEC",
      "0", "SECTION", "2", "ENTITIES",
      "0", "LINE", "8", "CUT_DOUBT", "10", "0", "20", "0", "11", "100", "21", "0",
      "0", "LINE", "8", "CUT_DOUBT", "10", "100", "20", "0", "11", "100", "21", "50",
      "0", "LINE", "8", "CUT_DOUBT", "10", "100", "20", "50", "11", "0", "21", "50",
      "0", "LINE", "8", "CUT_DOUBT", "10", "0", "20", "50", "11", "0", "21", "0",
      "0", "ENDSEC", "0", "EOF",
    ].join("\n") + "\n";
    const report = importApparelDxf(dxfWeird);
    expect(report.ok).toBe(false);
    expect(report.mismatches.some((m) => m.code === "insunits-unknown")).toBe(true);
    expect(report.mismatches.some((m) => m.code === "boundary-empty")).toBe(true);
  });

  it("converts astm-oriented cm files to metres correctly", () => {
    const f = engineeredGarment();
    const ir = buildExportIR2(f.document, f.seams, f.set, { garmentName: "G14 cm interop" });
    const dxf = exportIRToDXFProfile(ir, { profile: "astm-oriented", units: "cm" });
    const report = importApparelDxf(dxf.dxf, { profile: "astm-oriented" });
    expect(report.ok).toBe(true);
    expect(report.units).toBe("cm");
    const front = report.pieces[0];
    const xs = front.ringM.map((p) => p[0]);
    const w = Math.max(...xs) - Math.min(...xs);
    expect(w).toBeCloseTo(0.48, 3); // metres regardless of file units
  });
});
