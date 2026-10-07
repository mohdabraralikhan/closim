import type { PatternDocument } from "../pattern/cad.js";
import type { GarmentProject } from "../garment/project.js";
import { validateGarmentProject } from "../garment/project.js";
import type { GradingDocument } from "../grading/types.js";
import { deriveSize } from "../grading/derive.js";
import { deriveProductionSet } from "../grading/engine.js";
import { createProductionSet, type ProductionSet } from "../cad/production.js";
import { renderTechSheet } from "../cad/techsheet.js";
import { productionReadiness } from "../cad/readiness.js";
import { resolveSegment, sampleLoopLocal } from "../cad/queries.js";
import type { Seam } from "../garment/sewing.js";
import { serializePatternDocument } from "../pattern/cad.js";
import {
  validateProductionSpecification,
  validateTechPack,
  fingerprintProductionSource,
  type MeasurementSource,
  type MeasurementStatus,
  type MeasurementTarget,
  type ProductionSpecification,
  type TechPack,
  type TechPackMeasurement,
} from "./specification-model.js";

export * from "./specification-model.js";

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value !== null && typeof value === "object") {
    const record = value as Record<string, unknown>;
    return `{${Object.keys(record).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`).join(",")}}`;
  }
  if (typeof value === "number" && Object.is(value, -0)) return "0";
  return JSON.stringify(value);
}

function measure(source: MeasurementSource, document: PatternDocument, seams: readonly Seam[]): number | undefined {
  switch (source.kind) {
    case "point-distance": {
      const a = document.points.find((point) => point.id === source.pointAId && point.panelId === source.panelId);
      const b = document.points.find((point) => point.id === source.pointBId && point.panelId === source.panelId);
      if (!a || !b) return undefined;
      return Math.hypot(a.x - b.x, a.y - b.y);
    }
    case "segment-length": {
      const segment = document.segments.find((item) => item.id === source.segmentId && item.panelId === source.panelId);
      if (!segment || !document.points.some((point) => point.id === segment.startPointId && point.panelId === source.panelId) ||
        !document.points.some((point) => point.id === segment.endPointId && point.panelId === source.panelId) ||
        (segment.kind === "arc" && !document.points.some((point) => point.id === segment.centerPointId && point.panelId === source.panelId))) {
        return undefined;
      }
      return resolveSegment(document, source.segmentId, source.panelId).length;
    }
    case "panel-width":
    case "panel-height": {
      const panel = document.panels.find((candidate) => candidate.id === source.panelId);
      const outer = panel?.boundaryLoops.find((loop) => loop.role === "outer");
      if (!panel || !outer) return undefined;
      const points = sampleLoopLocal(document, panel.id, outer.id);
      if (!points.length) return undefined;
      const values = points.map((point) => point[source.kind === "panel-width" ? 0 : 1]);
      return Math.max(...values) - Math.min(...values);
    }
    case "seam-length": {
      const seam = seams.find((candidate) => candidate.id === source.seamId);
      if (!seam) return undefined;
      const side = source.side === "a" ? seam.sideA : seam.sideB;
      if (side.segmentIds.some((id) => !document.segments.some((segment) => segment.id === id && segment.panelId === side.panelId))) {
        return undefined;
      }
      return side.segmentIds.reduce((sum, id) => sum + resolveSegment(document, id, side.panelId).length, 0);
    }
  }
}

function measurementStatus(
  value: MeasurementTarget | undefined,
  measuredM: number | undefined,
): MeasurementStatus {
  if (measuredM === undefined) return "missing-source";
  if (value?.targetM === undefined) return "missing-target";
  const tolerance = value.toleranceM ?? 0;
  return Math.abs(measuredM - value.targetM) <= tolerance ? "pass" : "out-of-tolerance";
}

export function generateTechPack(
  garment: GarmentProject,
  specification: ProductionSpecification,
  options: { grading?: GradingDocument; production?: ProductionSet } = {},
): TechPack {
  const specErrors = validateProductionSpecification(specification);
  if (specErrors.length) throw new Error(`cannot generate tech pack: ${specErrors[0]}`);
  if (specification.garmentId !== garment.id) throw new Error("production specification references a different garment");
  const garmentValidation = validateGarmentProject(garment);
  const errors = garmentValidation.valid ? [] : garmentValidation.diagnostics.map((item) => item.message);
  const warnings: string[] = [];
  const grading = options.grading;
  if (grading && serializePatternDocument(grading.master.document) !== serializePatternDocument(garment.pattern)) {
    errors.push("grading master pattern differs from the authoritative garment pattern");
  }
  const requestedSizeIds = specification.sizeRange.length
    ? specification.sizeRange
    : grading?.sizeSet.sizes.filter((size) => size.active).map((size) => size.id) ?? [];
  const sizeEntries = grading
    ? requestedSizeIds.flatMap((id) => {
      const size = grading.sizeSet.sizes.find((candidate) => candidate.id === id && candidate.active);
      if (!size) {
        errors.push(`production size '${id}' is missing or inactive in grading`);
        return [];
      }
      return [{ id: size.id, label: size.label }];
    })
    : [{ id: "base", label: "Base" }];
  if (!grading && specification.sizeRange.length > 0) {
    warnings.push("grading is unavailable; technical drawings and measurements use the base pattern only");
  }
  if (specification.measurements.length === 0) warnings.push("no technical measurements are defined");
  const sourceDocs: Array<{ sizeId: string; document: PatternDocument; production: ProductionSet }> = [];
  for (const { id } of sizeEntries) {
    if (id === "base" || !grading) {
      sourceDocs.push({
        sizeId: id,
        document: garment.pattern,
        production: options.production ?? createProductionSet(),
      });
      continue;
    }
    try {
      const { graded } = deriveSize(grading, id);
      let production = createProductionSet();
      const productionSource = options.production ?? grading.production;
      if (productionSource) {
        const derived = deriveProductionSet({ ...grading, production: productionSource }, graded.document, id);
        production = derived.set;
        errors.push(...derived.issues.map((issue) => `size '${id}' production: ${issue.message}`));
      }
      sourceDocs.push({ sizeId: id, document: graded.document, production });
    } catch (error) {
      errors.push(`size '${id}' could not be derived: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  const drawings = sourceDocs.map(({ sizeId, document, production }) => {
    const sheet = renderTechSheet(document, production, {
      title: `${specification.styleNumber} ${specification.revision} — ${sizeId}`,
    });
    return { sizeId, svg: sheet.svg, widthPx: sheet.widthPx, heightPx: sheet.heightPx };
  });
  const measurements: TechPackMeasurement[] = [];
  for (const { sizeId, document } of sourceDocs) {
    for (const definition of specification.measurements) {
      const target = definition.targetsBySize[sizeId];
      const measuredM = measure(definition.source, document, garment.seams);
      measurements.push({
        measurementId: definition.id,
        name: definition.name,
        sizeId,
        ...(target?.targetM !== undefined ? { targetM: target.targetM } : {}),
        ...(target?.toleranceM !== undefined ? { toleranceM: target.toleranceM } : {}),
        ...(measuredM !== undefined ? { measuredM } : {}),
        status: measurementStatus(target, measuredM),
      });
    }
  }
  for (const size of sizeEntries.filter((entry) => !sourceDocs.some((source) => source.sizeId === entry.id))) {
    for (const definition of specification.measurements) {
      const target = definition.targetsBySize[size.id];
      measurements.push({
        measurementId: definition.id,
        name: definition.name,
        sizeId: size.id,
        ...(target?.targetM !== undefined ? { targetM: target.targetM } : {}),
        ...(target?.toleranceM !== undefined ? { toleranceM: target.toleranceM } : {}),
        status: "missing-source",
      });
    }
  }
  if (measurements.some((item) => item.status === "missing-target")) warnings.push("one or more measurements have no target value");
  if (measurements.some((item) => item.status === "missing-source")) errors.push("one or more measurement geometry references cannot be resolved");
  const panels = garment.pattern.panels.map((panel) => {
    const entry = specification.panels.find((candidate) => candidate.panelId === panel.id);
    return { name: panel.name, panelId: panel.id, ...(entry ? clone(entry) : {}) };
  });
  for (const panel of specification.panels) {
    if (!garment.pattern.panels.some((candidate) => candidate.id === panel.panelId)) {
      errors.push(`panel specification references missing panel '${panel.panelId}'`);
    }
  }
  if (options.production && !grading) {
    const readiness = productionReadiness(garment.pattern, garment.seams, options.production);
    errors.push(...readiness.diagnostics.filter((item) => item.severity === "error").map((item) => item.message));
    warnings.push(...readiness.diagnostics.filter((item) => item.severity === "warning").map((item) => item.message));
  }
  if (grading && (options.production || grading.production)) {
    for (const source of sourceDocs) {
      const readiness = productionReadiness(source.document, grading.seams, source.production);
      errors.push(...readiness.diagnostics.filter((item) => item.severity === "error")
        .map((item) => `size '${source.sizeId}': ${item.message}`));
      warnings.push(...readiness.diagnostics.filter((item) => item.severity === "warning")
        .map((item) => `size '${source.sizeId}': ${item.message}`));
    }
  }
  const sourceFingerprint = fingerprintProductionSource({
    garment,
    specification,
    grading,
    production: options.production,
  });
  return {
    schemaVersion: 1,
    garmentId: garment.id,
    styleNumber: specification.styleNumber,
    styleRevision: specification.revision,
    garmentRevision: garment.metadata.revision,
    sourceFingerprint,
    overview: {
      garmentName: specification.garmentName,
      styleNumber: specification.styleNumber,
      revision: specification.revision,
    },
    drawings,
    sizes: sizeEntries,
    measurements,
    materials: clone(specification.materials),
    colorways: clone(specification.colorways),
    panels,
    construction: clone(specification.construction),
    productionNotes: clone(specification.productionNotes),
    finishingNotes: clone(specification.finishingNotes),
    validation: {
      status: errors.length ? "errors" : warnings.length ? "warnings" : "ready",
      errors,
      warnings,
    },
  };
}

export function serializeTechPack(techPack: TechPack): string {
  const errors = validateTechPack(techPack);
  if (errors.length) throw new Error(`cannot serialize invalid tech pack: ${errors[0]}`);
  return canonicalJson(techPack);
}

export function deserializeTechPack(serialized: string): TechPack {
  let parsed: unknown;
  try {
    parsed = JSON.parse(serialized);
  } catch {
    throw new Error("serialized tech pack is not valid JSON");
  }
  const errors = validateTechPack(parsed as TechPack);
  if (errors.length) throw new Error(`invalid tech pack: ${errors[0]}`);
  return clone(parsed as TechPack);
}
