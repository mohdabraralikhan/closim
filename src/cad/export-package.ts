// G13D — commercial export-package builder.
//
// One export action produces a coherent, self-describing digital garment
// package: patterns (DXF/SVG/PDF), grading artifacts, machine-readable
// metadata + validation manifests, a preview, checksums, and a README.
//
// Reproducibility: same source + same settings produce byte-identical files
// except explicitly designated fields (`generatedAt` lives only in the
// manifest and is parameterized, not read from the clock). Filenames are
// deterministic; duplicate logical names get disambiguating suffixes
// deterministically (never random). No local filesystem paths are embedded.
//
// The package builder composes the G13A gate + IR and the G13B/C adapters;
// it is itself derived data and never feeds back into the native document.

import { createHash } from "node:crypto";
import { PatternCadError, type PatternDocument } from "../pattern/cad.js";
import type { Seam } from "../garment/sewing.js";
import type { ProductionSet } from "./production.js";
import { canonicalJson, buildExportIR, exportGate, type ExportGateDecision, type ExportGateMode, type ExportGradingContext, type ExportIR, type ExportUnits, requireExportUnits } from "./export-ir.js";
import { exportIRToDXFProfile } from "./dxf-export.js";
import type { DxfProfileId } from "./dxf-profile.js";
import { validateDxfOutput } from "./dxf-validate.js";
import { exportIRToSVG } from "./svg-export.js";
import { exportIRToPDF, PAGE_PRESETS, type PageSize } from "./pdf-export.js";
import { exportGradeRules } from "./grade-rules-export.js";
import { createZipArchive } from "./zip.js";

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

export interface ExportPackageOptions {
  garmentName: string;
  /** Deterministic package root folder name (sanitized). */
  packageName?: string;
  revision?: number;
  units?: ExportUnits;
  gateMode?: ExportGateMode;
  /** Grading context (sizes, rules) — omit for single-size master export. */
  grading?: ExportGradingContext;
  /** Pattern formats to emit. Default: all. "zip" bundles everything into one archive. */
  formats?: Array<"dxf" | "svg" | "pdf-fullscale" | "pdf-tiled" | "zip"> | readonly ("dxf" | "svg" | "pdf-fullscale" | "pdf-tiled" | "zip")[];
  dxfProfile?: DxfProfileId;
  dxfMultiSize?: boolean;
  pdfPage?: string | PageSize;
  pdfMarginMm?: number;
  pdfOverlapMm?: number;
  pdfCalibrationMm?: number;
  /** Explicit manifest timestamp (ISO string). Omit for reproducibility. */
  generatedAt?: string;
  /** Extra free-text fields copied into the manifest (no paths allowed). */
  notes?: string;
}

export interface ExportPackageFile {
  /** Package-relative path with forward slashes. */
  path: string;
  bytes: Uint8Array | string;
  sha256: string;
  /** Logical role for consumers. */
  role: "pattern-dxf" | "pattern-svg" | "pattern-pdf" | "grade-rules" | "size-chart" | "metadata" | "readme" | "preview" | "validation" | "package-zip";
}

export interface ExportPackage {
  files: ExportPackageFile[];
  manifestJson: string;
  validationJson: string;
  readmeMd: string;
  decision: ExportGateDecision;
  ir: ExportIR;
  warnings: string[];
}

// ---------------------------------------------------------------------------
// Deterministic naming
// ---------------------------------------------------------------------------

function slug(text: string): string {
  const s = text.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
  return s.length > 0 ? s : "garment";
}

/** Unit-aware deterministic name: style[.panel][.size].revN[.formatLabel].ext. */
function fileNameFor(parts: { style: string; panel?: string; size?: string; formatLabel?: string; revision: number; ext: string; used: Set<string> }): string {
  const segs = [slug(parts.style)];
  if (parts.panel) segs.push(slug(parts.panel));
  if (parts.size) segs.push(slug(parts.size));
  segs.push(`rev${parts.revision}`);
  if (parts.formatLabel) segs.push(slug(parts.formatLabel));
  const base = segs.join(".");
  let candidate = `${base}.${parts.ext}`;
  let n = 2;
  while (parts.used.has(candidate)) {
    candidate = `${base}-${n}.${parts.ext}`;
    n++;
  }
  parts.used.add(candidate);
  return candidate;
}

// ---------------------------------------------------------------------------
// Checksums
// ---------------------------------------------------------------------------

export function sha256Of(data: Uint8Array | string): string {
  const h = createHash("sha256");
  h.update(typeof data === "string" ? Buffer.from(data, "utf8") : Buffer.from(data));
  return h.digest("hex");
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

/**
 * Validate → gate → IR → adapters → package. Throws unless the gate passes
 * (or mode is allow-warnings and the state is exactly WARNINGS). The returned
 * package is in-memory; writing it to disk is the caller's job.
 */
export function exportCommercialPackage(
  doc: PatternDocument,
  seams: readonly Seam[],
  set: ProductionSet,
  opts: ExportPackageOptions,
): ExportPackage {
  const units = requireExportUnits(opts.units ?? "mm", "package export");
  const formats = opts.formats ?? ["dxf", "svg", "pdf-fullscale", "pdf-tiled"];
  if (!doc || !Array.isArray(doc.panels) || doc.panels.length === 0) {
    throw new PatternCadError("invalid-document", "export refused: garment has no panels");
  }
  const decision = exportGate(doc, seams, set, opts.gateMode ?? "strict");
  if (decision.blocked) {
    const first = decision.readiness.diagnostics[0];
    throw new PatternCadError(
      "invalid-document",
      `export refused (state=${decision.state}): ${first ? `${first.severity} ${first.code}: ${first.message}` : "validation failed"}`,
      first?.entityId,
    );
  }
  const grading = opts.grading ?? null;
  const revision = opts.revision ?? 1;
  const ir = buildExportIR(doc, seams, set, {
    garmentName: opts.garmentName,
    revision,
    grading: grading ?? undefined,
    readiness: decision.readiness,
  });

  const warnings: string[] = [...decision.readiness.diagnostics.map((d) => `${d.severity}: ${d.code}: ${messageOf(d.message)}`)];
  const used = new Set<string>();
  const files: ExportPackageFile[] = [];
  const addFile = (path: string, bytes: Uint8Array | string, role: ExportPackageFile["role"]): ExportPackageFile => {
    const file: ExportPackageFile = { path, bytes, sha256: sha256Of(bytes), role };
    files.push(file);
    return file;
  };
  const root = slug(opts.packageName ?? opts.garmentName);
  const styleSeg = slug(opts.garmentName);

  // Patterns.
  const sizeTag = grading && grading.sizes.length > 1 ? "sizes" : undefined;
  if (formats.includes("dxf")) {
    const dxf = exportIRToDXFProfile(ir, {
      profile: opts.dxfProfile ?? "aama-style",
      units,
      multiSize: opts.dxfMultiSize ?? (grading !== null),
      gapM: 0.1,
    });
    warnings.push(...dxf.warnings);
    addFile(`${root}/patterns/${fileNameFor({ style: styleSeg, size: sizeTag, revision, ext: "dxf", used })}`, dxf.dxf, "pattern-dxf");
  }
  if (formats.includes("svg")) {
    const svg = exportIRToSVG(ir, { units });
    warnings.push(...svg.warnings);
    addFile(`${root}/patterns/${fileNameFor({ style: styleSeg, size: sizeTag, revision, ext: "svg", used })}`, svg.svg, "pattern-svg");
  }
  if (formats.includes("pdf-fullscale")) {
    const pdf = exportIRToPDF(ir, {
      page: opts.pdfPage ?? "A0",
      marginMm: opts.pdfMarginMm ?? 10,
      mode: "full-scale",
      units,
    });
    warnings.push(...pdf.warnings);
    addFile(`${root}/patterns/${fileNameFor({ style: styleSeg, size: sizeTag, formatLabel: "a0-fullscale", revision, ext: "pdf", used })}`, pdf.pdf, "pattern-pdf");
  }
  if (formats.includes("pdf-tiled")) {
    const pdf = exportIRToPDF(ir, {
      page: opts.pdfPage ?? "A4",
      marginMm: opts.pdfMarginMm ?? 10,
      mode: "tiled",
      overlapMm: opts.pdfOverlapMm ?? 10,
      calibrationMm: opts.pdfCalibrationMm ?? 50,
      units,
    });
    warnings.push(...pdf.warnings);
    addFile(`${root}/patterns/${fileNameFor({ style: styleSeg, size: sizeTag, formatLabel: "a4-tiled", revision, ext: "pdf", used })}`, pdf.pdf, "pattern-pdf");
  }

  // Grading artifacts (separate files; DXF carries size text only).
  if (grading) {
    const gr = exportGradeRules(grading, { garmentName: opts.garmentName, styleId: ir.style.styleId });
    warnings.push(...gr.warnings);
    addFile(`${root}/grading/grade-rules.json`, gr.json, "grade-rules");
    addFile(`${root}/grading/size-chart.csv`, gr.csv, "size-chart");
  }

  // Preview (lightweight SVG tech view).
  const preview = exportIRToSVG(ir, { units, title: `${opts.garmentName} — preview` });
  addFile(`${root}/preview/preview.svg`, preview.svg, "preview");

  // Validation manifest.
  const validationManifest = {
    format: "closim-export-validation",
    version: 1,
    state: decision.state,
    blocked: decision.blocked,
    errorCount: decision.readiness.errorCount,
    warningCount: decision.readiness.warningCount,
    panels: decision.readiness.panels,
    seams: decision.readiness.seams,
    totalCutAreaM2: decision.readiness.totalCutAreaM2,
    diagnostics: decision.readiness.diagnostics,
  };
  addFile(`${root}/validation/validation-manifest.json`, canonicalJson(validationManifest), "validation");

  // Metadata manifest (machine-readable; no filesystem paths).
  const manifest = {
    format: "closim-export-manifest",
    version: 1,
    productName: opts.garmentName,
    revision,
    ...(opts.generatedAt !== undefined ? { generatedAt: opts.generatedAt } : {}),
    application: "closim",
    applicationPhase: "G13",
    units,
    canonicalUnits: "m",
    sizeSet: grading
      ? { id: grading.sizeSetId ?? null, name: grading.sizeSetName ?? null, baseSizeId: grading.baseSizeId, sizes: grading.sizes.map((s) => s.label) }
      : null,
    pieceCount: ir.panels.length,
    cutQuantityTotal: ir.panels.reduce((s, p) => s + p.cutQuantity, 0),
    exportFormats: files.filter((f) => f.role === "pattern-dxf" || f.role === "pattern-svg" || f.role === "pattern-pdf").map((f) => f.path),
    validationStatus: decision.state,
    sourceDocumentVersion: `pattern-doc/${doc.schemaVersion}`,
    gateMode: opts.gateMode ?? "strict",
    notes: opts.notes ?? null,
  };
  const manifestJson = canonicalJson(manifest);
  addFile(`${root}/metadata/manifest.json`, manifestJson, "metadata");

  // README (human-facing; last so it can enumerate real file paths).
  const readmeLines: string[] = [];
  readmeLines.push(`# ${opts.garmentName}`);
  readmeLines.push("");
  readmeLines.push(`Revision ${revision}. Exported by closim G13.`);
  readmeLines.push("");
  readmeLines.push(`- Garment: ${opts.garmentName} (${ir.panels.length} pieces, total cut quantity ${manifest.cutQuantityTotal})`);
  readmeLines.push(`- Sizes: ${grading ? grading.sizes.map((s) => s.label).join(", ") : "single (master geometry)"}`);
  readmeLines.push(`- Units: files are in ${units}; the native document uses metres. True scale at 100%.`);
  readmeLines.push(`- Validation: ${decision.state}${decision.readiness.warningCount > 0 ? ` (${decision.readiness.warningCount} warning(s) — see validation/validation-manifest.json)` : ""}`);
  readmeLines.push("");
  readmeLines.push("## Files");
  for (const f of files) {
    readmeLines.push(`- \`${f.path}\` — ${roleLabel(f.role)} (sha256 ${f.sha256.slice(0, 12)}…)`);
  }
  readmeLines.push("");
  readmeLines.push("## Printing");
  readmeLines.push("- DXF: manufacturing interchange for pattern CAD (not a certified AAMA/ASTM file — see warnings).");
  readmeLines.push("- PDF tiled: print at 100% (no fit-to-page), measure the calibration square on every page before assembly.");
  readmeLines.push("- PDF full-scale: large-format plot at 100% on the declared page size.");
  readmeLines.push("");
  readmeLines.push("## Intended use");
  readmeLines.push("Digital-download garment pattern package. The pattern files are derived from the closim native document; re-export after any edit.");
  const readmeMd = readmeLines.join("\n") + "\n";
  addFile(`${root}/README.md`, readmeMd, "readme");

  // Single distributable archive (optional). Deterministic: fixed timestamps,
  // STORE method, entries in the order above — same inputs, same zip bytes.
  // The zip is emitted AFTER checksums are computed, so it never checksums
  // itself and stays reproducible.
  if (formats.includes("zip")) {
    const archive = createZipArchive(
      files.map((f) => ({ path: f.path, bytes: f.bytes })),
      { comment: `closim export package ${manifest.productName} rev${revision}` },
    );
    addFile(`${root}.zip`, archive, "package-zip");
    warnings.push("package includes a ZIP bundle (uncompressed STORE entries for reproducibility)");
  }

  return { files, manifestJson, validationJson: canonicalJson(validationManifest), readmeMd, decision, ir, warnings };
}

function messageOf(message: string): string {
  return message.length > 140 ? `${message.slice(0, 137)}…` : message;
}

function roleLabel(role: ExportPackageFile["role"]): string {
  switch (role) {
    case "pattern-dxf": return "DXF pattern (pattern CAD interchange)";
    case "pattern-svg": return "SVG pattern (true-scale vector)";
    case "pattern-pdf": return "PDF pattern (print at 100%)";
    case "grade-rules": return "grade rules (machine-readable, canonical JSON)";
    case "size-chart": return "size chart (CSV measurements)";
    case "metadata": return "metadata manifest";
    case "validation": return "validation manifest";
    case "preview": return "lightweight preview";
    case "readme": return "package README";
    case "package-zip": return "single distributable ZIP archive of this package";
  }
}
