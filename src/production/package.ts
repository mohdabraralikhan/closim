import { serializePatternDocument } from "../pattern/cad.js";
import type { GarmentProject } from "../garment/project.js";
import type { GradingDocument } from "../grading/types.js";
import { deriveSize } from "../grading/derive.js";
import type { Marker } from "../marker/model.js";
import { serializeMarker } from "../marker/model.js";
import type { BOM } from "./bom.js";
import { serializeBOM, validateBOM } from "./bom.js";
import type { ManufacturingDashboard, ManufacturingReports, DependencyArtifact } from "./reporting.js";
import type { ProductionRevision } from "./revision.js";
import type { ProductionRun, CutPlanResult } from "./run.js";
import { calculateCutPlan } from "./run.js";
import type { ProductionSpecification, TechPack } from "./specification-model.js";
import { serializeTechPack } from "./specification.js";
import { createZipArchive, type ZipEntry } from "../cad/zip.js";
import type { ExportPackage } from "../cad/export-package.js";
import { recordRevisionArtifact, type RevisionLedger } from "./revision.js";

export interface ProductionPackage {
  id: string;
  garmentId: string;
  revisionId: string;
  sourceFingerprint: string;
  fingerprint: string;
  files: ZipEntry[];
  archive: Uint8Array;
}

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value !== null && typeof value === "object") {
    const obj = value as Record<string, unknown>;
    return `{${Object.keys(obj).sort().map((key) => `${JSON.stringify(key)}:${canonical(obj[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function slug(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "garment";
}

function fingerprintBytes(bytes: Uint8Array): string {
  let hash = 2166136261;
  for (const byte of bytes) {
    hash ^= byte;
    hash = Math.imul(hash, 16777619);
  }
  return `package/${(hash >>> 0).toString(16).padStart(8, "0")}`;
}

export function generateProductionPackage(input: {
  garment: GarmentProject;
  specification: ProductionSpecification;
  revision: ProductionRevision;
  grading?: GradingDocument;
  markers: Marker[];
  bom: BOM;
  run: ProductionRun;
  cutPlan: CutPlanResult;
  techPack: TechPack;
  dashboard: ManufacturingDashboard;
  reports: ManufacturingReports;
  artifacts: DependencyArtifact[];
  panelPiecesPerGarment?: Record<string, number>;
  commercialExport?: ExportPackage;
}): ProductionPackage {
  const { garment, specification, revision, techPack } = input;
  if (revision.status !== "Released") throw new Error("production package requires a Released revision");
  if (garment.id !== specification.garmentId || input.run.garmentId !== garment.id) {
    throw new Error("production package inputs refer to different garments");
  }

  if (input.run.styleNumber !== specification.styleNumber || input.run.revision !== specification.revision) {
    throw new Error("production run style or revision does not match the production specification");
  }
  if (techPack.garmentId !== garment.id || techPack.styleNumber !== specification.styleNumber ||
    techPack.styleRevision !== specification.revision || techPack.garmentRevision !== garment.metadata.revision) {
    throw new Error("tech pack garment or revision does not match production package inputs");
  }
  if (input.bom.garmentId !== garment.id || input.bom.revision !== specification.revision) {
    throw new Error("BOM garment or revision does not match production specification");
  }
  if (input.grading && input.run.gradingId !== input.grading.id) {
    throw new Error("production run grading reference does not match package grading");
  }
  if (revision.sourceFingerprint !== techPack.sourceFingerprint) throw new Error("tech pack source does not match released revision");
  const stale = input.artifacts.filter((artifact) =>
    artifact.kind !== "package" && artifact.kind !== "production-package" && artifact.state !== "current");
  if (stale.length) throw new Error(`production package refused: artifact '${stale[0].id}' is ${stale[0].state}`);
  if (input.run.status === "draft" || input.run.status === "cancelled") {
    throw new Error(`production package refused for a ${input.run.status} run`);
  }
  if (input.run.revision !== specification.revision) throw new Error("production run revision does not match specification revision");
  if (!input.cutPlan.complete) throw new Error("production package refused: cut plan is incomplete");
  if (input.cutPlan.runId !== input.run.id) throw new Error("cut plan belongs to a different production run");
  if (techPack.validation.status === "errors") throw new Error("production package refused: tech pack has validation errors");
  const bomIssues = validateBOM(input.bom, {
    markers: input.markers,
    knownPanels: garment.pattern.panels.map((panel) => panel.id),
  });
  if (bomIssues.length) throw new Error(`production package refused: ${bomIssues[0].message}`);
  for (const panel of specification.panels) {
    const runMaterialId = input.run.panelMaterialIds[panel.panelId];
    if (panel.materialId && runMaterialId !== panel.materialId) {
      throw new Error(`panel '${panel.panelId}' BOM material does not match the production run`);
    }
  }
  if (input.dashboard.validation.status === "errors" || input.reports.validation.status === "errors") {
    throw new Error("production package refused: manufacturing reports contain validation errors");
  }
  if (!input.grading && Object.values(input.run.sizeQuantities).some((quantity) => quantity > 0)) {
    throw new Error("production package needs grading data for a non-empty size run");
  }
  if (input.grading) {
    const calculated = calculateCutPlan(input.run, input.grading, input.markers, {
      panelPiecesPerGarment: input.panelPiecesPerGarment,
    });
    if (canonical(calculated) !== canonical(input.cutPlan)) {
      throw new Error("production package cut plan differs from the regenerated authoritative plan");
    }
  }

  const requiredArtifacts = [
    ...input.markers.map((marker) => marker.id),
    input.bom.id,
    input.run.id,
    `tech-pack:${techPack.sourceFingerprint}`,
    `cut-plan:${input.run.id}`,
    ...(input.grading ? [`grading:${input.grading.id}`] : []),
    "pattern",
  ];
  for (const id of requiredArtifacts) {
    const artifact = input.artifacts.find((entry) => entry.id === id);
    if (!artifact) throw new Error(`production package refused: missing trace artifact '${id}'`);
    if (artifact.state !== "current") throw new Error(`production package refused: artifact '${id}' is ${artifact.state}`);
    if (artifact.sourceFingerprint !== artifact.currentSourceFingerprint) {
      throw new Error(`production package refused: artifact '${id}' source fingerprint is stale`);
    }
    if (artifact.sourceRevisionId !== revision.id) {
      throw new Error(`production package refused: artifact '${id}' belongs to another revision`);
    }
    const recorded = revision.artifacts.find((entry) => entry.id === id);
    if (!recorded || recorded.revisionId !== revision.id || recorded.fingerprint !== artifact.sourceFingerprint) {
      throw new Error(`production package refused: artifact '${id}' is not recorded on revision '${revision.id}'`);
    }
  }

  const root = `${slug(specification.styleNumber)}-rev${revision.number}`;
  const files: ZipEntry[] = [];
  const add = (path: string, bytes: string | Uint8Array): void => { files.push({ path: `${root}/${path}`, bytes }); };
  const requestedSizeIds = new Set(Object.keys(input.run.sizeQuantities));
  for (const sizeId of requestedSizeIds) {
    if (!techPack.sizes.some((size) => size.id === sizeId)) {
      throw new Error(`production package tech pack is missing run size '${sizeId}'`);
    }
  }
  const sizeDocs = input.grading
    ? input.grading.sizeSet.sizes.filter((size) => size.active && requestedSizeIds.has(size.id)).map((size) => ({
      id: size.id,
      document: deriveSize(input.grading!, size.id).graded.document,
    }))
    : [{ id: "base", document: garment.pattern }];
  for (const entry of sizeDocs) {
    add(`patterns/${slug(entry.id)}.json`, serializePatternDocument(entry.document));
  }
  if (input.grading) {
    add("grading/grade-document.json", canonical(input.grading));
    add("grading/size-chart.json", canonical({
      sizes: input.grading.sizeSet.sizes.filter((size) => size.active && requestedSizeIds.has(size.id)).map((size) => ({
        id: size.id, label: size.label, measurements: size.measurements,
      })),
    }));
  }
  for (const marker of input.markers) add(`markers/${slug(marker.id)}.json`, serializeMarker(marker));
  for (const file of input.commercialExport?.files ?? []) {
    if (file.role === "package-zip") continue;
    const pathParts = file.path.split("/");
    const directory = pathParts.find((part) =>
      ["patterns", "grading", "preview", "validation", "metadata"].includes(part));
    const basename = pathParts[pathParts.length - 1];
    const targetDirectory = directory === "preview" ? "previews" : directory === "validation" ? "reports" : directory ?? "reports";
    add(`${targetDirectory}/g13-${basename}`, file.bytes);
  }
  add("technical/tech-pack.json", serializeTechPack(techPack));
  for (const drawing of techPack.drawings) add(`previews/${slug(drawing.sizeId)}.svg`, drawing.svg);
  add("materials/bom.json", serializeBOM(input.bom));
  add("materials/requirements.json", canonical(input.dashboard.materials));
  add("reports/pattern.json", canonical(input.reports.pattern));
  add("reports/markers.json", canonical(input.reports.markers));
  add("reports/materials.json", canonical(input.reports.materials));
  add("reports/production.json", canonical({ run: input.run, cutPlan: input.cutPlan }));
  add("reports/validation.json", canonical(input.reports.validation));
  add("reports/dashboard.json", canonical(input.dashboard));
  const artifactMetadata = files.map((file) => {
    const relative = file.path.slice(root.length + 1);
    const sizeDoc = sizeDocs.find((entry) => relative.includes(slug(entry.id)));
    return {
      path: file.path,
      garmentId: garment.id,
      revisionId: revision.id,
      revisionNumber: revision.number,
      ...(sizeDoc ? { sizeId: sizeDoc.id } : {}),
      format: relative.split(".").pop() ?? "unknown",
      generation: { application: "CLOTHSIM", configVersion: 1, sourceFingerprint: revision.sourceFingerprint },
    };
  });
  add("metadata/artifacts.json", canonical(artifactMetadata));
  add("metadata/manifest.json", canonical({
    format: "closim-production-package",
    version: 1,
    garmentId: garment.id,
    garmentName: specification.garmentName,
    styleNumber: specification.styleNumber,
    revisionId: revision.id,
    revisionNumber: revision.number,
    revisionStatus: revision.status,
    sourceFingerprint: revision.sourceFingerprint,
    productionRunId: input.run.id,
    productionRunRevision: input.run.revision,
    sizes: sizeDocs.map((size) => size.id),
    formats: ["json", "svg", "zip"],
    generation: { application: "CLOTHSIM", formatVersion: 1 },
    includedCommercialExport: input.commercialExport !== undefined,
  }));
  const paths = new Set<string>();
  for (const file of files) {
    if (paths.has(file.path)) throw new Error(`duplicate production package path '${file.path}'`);
    paths.add(file.path);
  }
  const archive = createZipArchive(files);
  const fingerprint = fingerprintBytes(archive);
  return {
    id: `production-package/${revision.id}/${input.run.id}`,
    garmentId: garment.id,
    revisionId: revision.id,
    sourceFingerprint: revision.sourceFingerprint,
    fingerprint,
    files,
    archive,
  };
}

export function recordProductionPackage(
  ledger: RevisionLedger,
  pkg: ProductionPackage,
  options: { timestamp: string; author: string },
): RevisionLedger {
  const revision = ledger.revisions.find((entry) => entry.id === pkg.revisionId);
  if (!revision || revision.sourceFingerprint !== pkg.sourceFingerprint) {
    throw new Error(`package '${pkg.id}' does not match the revision ledger source`);
  }
  return recordRevisionArtifact(ledger, revision.id, {
    id: pkg.id,
    kind: "production-package",
    fingerprint: pkg.fingerprint,
  }, options);
}
