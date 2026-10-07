export const PRODUCTION_SPECIFICATION_VERSION = 1;

export type MeasurementSource =
  | { kind: "point-distance"; panelId: string; pointAId: string; pointBId: string }
  | { kind: "segment-length"; panelId: string; segmentId: string }
  | { kind: "panel-width" | "panel-height"; panelId: string }
  | { kind: "seam-length"; seamId: string; side: "a" | "b" };

export interface MeasurementTarget {
  targetM?: number;
  toleranceM?: number;
}

export interface TechnicalMeasurement {
  id: string;
  name: string;
  source: MeasurementSource;
  unit: "m";
  targetsBySize: Record<string, MeasurementTarget>;
}

export type ConstructionKind = "seam" | "stitch" | "closure" | "pocket" | "fold" | "trim" | "finishing";

export interface ConstructionDetail {
  id: string;
  kind: ConstructionKind;
  title: string;
  data: Record<string, string | number | boolean>;
}

export interface ProductionMaterial {
  id: string;
  name: string;
  category: string;
  composition?: string;
  supplierReference?: string;
}

export interface ProductionColorway {
  id: string;
  name: string;
  code?: string;
  materialColors: Record<string, string>;
}

export interface PanelSpecification {
  panelId: string;
  materialId?: string;
  cutQuantity?: number;
  notes?: string;
}

export interface ProductionSpecification {
  schemaVersion: typeof PRODUCTION_SPECIFICATION_VERSION;
  id: string;
  garmentId: string;
  garmentName: string;
  styleNumber: string;
  revision: string;
  sizeRange: string[];
  materials: ProductionMaterial[];
  colorways: ProductionColorway[];
  panels: PanelSpecification[];
  construction: ConstructionDetail[];
  measurements: TechnicalMeasurement[];
  productionNotes: string[];
  finishingNotes: string[];
}

export function fingerprintProductionSource(input: {
  garment: unknown;
  specification: ProductionSpecification;
  grading?: unknown;
  production?: unknown;
}): string {
  const text = canonicalJson({
    garment: input.garment,
    specification: input.specification,
    grading: input.grading ?? null,
    production: input.production ?? null,
  });
  let hash = 2166136261;
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return `tech/${(hash >>> 0).toString(16).padStart(8, "0")}`;
}

export type MeasurementStatus = "pass" | "out-of-tolerance" | "missing-target" | "missing-source";

export interface TechPackMeasurement {
  measurementId: string;
  name: string;
  sizeId: string;
  targetM?: number;
  toleranceM?: number;
  measuredM?: number;
  status: MeasurementStatus;
}

export interface TechPack {
  schemaVersion: 1;
  garmentId: string;
  styleNumber: string;
  styleRevision: string;
  garmentRevision: number;
  sourceFingerprint: string;
  overview: { garmentName: string; styleNumber: string; revision: string };
  drawings: Array<{ sizeId: string; svg: string; widthPx: number; heightPx: number }>;
  sizes: Array<{ id: string; label: string }>;
  measurements: TechPackMeasurement[];
  materials: ProductionMaterial[];
  colorways: ProductionColorway[];
  panels: Array<PanelSpecification & { name: string }>;
  construction: ConstructionDetail[];
  productionNotes: string[];
  finishingNotes: string[];
  validation: { status: "ready" | "warnings" | "errors"; errors: string[]; warnings: string[] };
}

export function validateTechPack(techPack: TechPack): string[] {
  if (!techPack || techPack.schemaVersion !== 1 || !techPack.garmentId || !techPack.styleNumber ||
    !techPack.styleRevision || !techPack.sourceFingerprint || !techPack.overview ||
    !Array.isArray(techPack.drawings) || !Array.isArray(techPack.sizes) ||
    !Array.isArray(techPack.measurements) || !Array.isArray(techPack.panels) ||
    !Array.isArray(techPack.materials) || !Array.isArray(techPack.construction) ||
    !Array.isArray(techPack.productionNotes) || !Array.isArray(techPack.finishingNotes) ||
    !techPack.validation || !Array.isArray(techPack.validation.errors) || !Array.isArray(techPack.validation.warnings)) {
    return ["tech pack shape or schema is invalid"];
  }
  const errors: string[] = [];
  if (!Number.isSafeInteger(techPack.garmentRevision) || techPack.garmentRevision < 1) {
    errors.push("garment revision must be a positive safe integer");
  }
  const sizes = new Set(techPack.sizes.map((size) => size.id));
  if (sizes.size !== techPack.sizes.length || techPack.sizes.some((size) => !size.id || !size.label)) {
    errors.push("tech pack sizes must have unique non-empty IDs and labels");
  }
  const drawingSizes = new Set(techPack.drawings.map((drawing) => drawing.sizeId));
  if (drawingSizes.size !== techPack.drawings.length ||
    techPack.drawings.some((drawing) => !sizes.has(drawing.sizeId) || !drawing.svg ||
      !Number.isSafeInteger(drawing.widthPx) || drawing.widthPx <= 0 ||
      !Number.isSafeInteger(drawing.heightPx) || drawing.heightPx <= 0)) {
    errors.push("tech pack drawings must be valid and map uniquely to listed sizes");
  }
  const measurementStatuses = new Set(["pass", "out-of-tolerance", "missing-target", "missing-source"]);
  for (const measurement of techPack.measurements) {
    if (!measurement.measurementId || !measurement.name || !sizes.has(measurement.sizeId) ||
      !measurementStatuses.has(measurement.status)) {
      errors.push(`tech pack measurement '${measurement.measurementId}' has invalid references or status`);
    }
    for (const value of [measurement.targetM, measurement.toleranceM, measurement.measuredM]) {
      if (value !== undefined && !Number.isFinite(value)) errors.push(`measurement '${measurement.measurementId}' has a non-finite value`);
    }
    if (measurement.toleranceM !== undefined && measurement.toleranceM < 0) {
      errors.push(`measurement '${measurement.measurementId}' has a negative tolerance`);
    }
    const expectedMeasurementStatus = measurement.measuredM === undefined
      ? "missing-source"
      : measurement.targetM === undefined
        ? "missing-target"
        : Math.abs(measurement.measuredM - measurement.targetM) <= (measurement.toleranceM ?? 0)
          ? "pass"
          : "out-of-tolerance";
    if (measurement.status !== expectedMeasurementStatus) {
      errors.push(`measurement '${measurement.measurementId}' status disagrees with its values`);
    }
  }
  const expectedStatus = techPack.validation.errors.length ? "errors"
    : techPack.validation.warnings.length ? "warnings" : "ready";
  if (techPack.validation.status !== expectedStatus) errors.push("tech pack validation status disagrees with its diagnostics");
  return errors;
}

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

export function createProductionSpecification(input: Omit<ProductionSpecification, "schemaVersion">): ProductionSpecification {
  const specification: ProductionSpecification = {
    schemaVersion: PRODUCTION_SPECIFICATION_VERSION,
    ...clone(input),
  };
  const errors = validateProductionSpecification(specification);
  if (errors.length) throw new Error(`invalid production specification: ${errors[0]}`);
  return specification;
}

export function validateProductionSpecification(specification: ProductionSpecification): string[] {
  const errors: string[] = [];
  if (!specification || typeof specification !== "object" ||
    specification.schemaVersion !== PRODUCTION_SPECIFICATION_VERSION) {
    return ["shape or schema version is invalid"];
  }
  for (const [key, value] of [
    ["id", specification.id],
    ["garmentId", specification.garmentId],
    ["garmentName", specification.garmentName],
    ["styleNumber", specification.styleNumber],
    ["revision", specification.revision],
  ]) {
    if (typeof value !== "string" || !value.trim()) errors.push(`${key} must be non-empty`);
  }
  if (!Array.isArray(specification.sizeRange) ||
    specification.sizeRange.some((id) => typeof id !== "string" || !id.trim()) ||
    new Set(specification.sizeRange).size !== specification.sizeRange.length) {
    errors.push("sizeRange must contain unique non-empty size IDs");
  }
  const uniqueIds = <T extends { id: string }>(items: T[], label: string): void => {
    if (!Array.isArray(items)) {
      errors.push(`${label} must be an array`);
      return;
    }
    const ids = items.map((item) => item?.id);
    if (ids.some((id) => typeof id !== "string" || !id.trim()) || new Set(ids).size !== ids.length) {
      errors.push(`${label} must have unique non-empty IDs`);
    }
  };
  uniqueIds(specification.materials, "materials");
  uniqueIds(specification.colorways, "colorways");
  uniqueIds(specification.construction, "construction");
  uniqueIds(specification.measurements, "measurements");
  if (!Array.isArray(specification.panels)) errors.push("panels must be an array");
  else if (specification.panels.some((panel) => !panel?.panelId?.trim()) ||
    new Set(specification.panels.map((panel) => panel.panelId)).size !== specification.panels.length) {
    errors.push("panel specifications must have unique non-empty panel IDs");
  }
  for (const material of specification.materials ?? []) {
    if (!material.name?.trim() || !material.category?.trim()) errors.push(`material '${material.id}' needs a name and category`);
  }
  for (const colorway of specification.colorways ?? []) {
    if (!colorway.name?.trim()) errors.push(`colorway '${colorway.id}' needs a name`);
  }
  const materialIds = new Set((specification.materials ?? []).map((item) => item.id));
  for (const colorway of specification.colorways ?? []) {
    for (const materialId of Object.keys(colorway.materialColors ?? {})) {
      if (!materialIds.has(materialId)) errors.push(`colorway '${colorway.id}' references unknown material '${materialId}'`);
    }
  }
  for (const panel of specification.panels ?? []) {
    if (panel.cutQuantity !== undefined && (!Number.isInteger(panel.cutQuantity) || panel.cutQuantity < 1)) {
      errors.push(`panel '${panel.panelId}' cutQuantity must be a positive integer`);
    }
    if (panel.materialId && !materialIds.has(panel.materialId)) {
      errors.push(`panel '${panel.panelId}' references unknown material '${panel.materialId}'`);
    }
  }
  for (const measurement of specification.measurements ?? []) {
    if (!measurement.name?.trim()) errors.push(`measurement '${measurement.id}' needs a name`);
    if (measurement.unit !== "m") errors.push(`measurement '${measurement.id}' must use canonical metres`);
    const source = measurement.source;
    if (!source || typeof source !== "object" ||
      (source.kind === "point-distance" && (!source.panelId || !source.pointAId || !source.pointBId)) ||
      (source.kind === "segment-length" && (!source.panelId || !source.segmentId)) ||
      ((source.kind === "panel-width" || source.kind === "panel-height") && !source.panelId) ||
      (source.kind === "seam-length" && (!source.seamId || (source.side !== "a" && source.side !== "b"))) ||
      !["point-distance", "segment-length", "panel-width", "panel-height", "seam-length"].includes(source.kind)) {
      errors.push(`measurement '${measurement.id}' has an invalid geometry source`);
    }
    for (const [sizeId, target] of Object.entries(measurement.targetsBySize ?? {})) {
      for (const key of ["targetM", "toleranceM"] as const) {
        const value = target[key];
        if (value !== undefined && (!Number.isFinite(value) || (key === "toleranceM" && value < 0))) {
          errors.push(`measurement '${measurement.id}' has invalid ${key} for '${sizeId}'`);
        }
      }
    }
  }
  for (const detail of specification.construction ?? []) {
    if (!detail.title?.trim() || !detail.kind || !detail.data || typeof detail.data !== "object") {
      errors.push(`construction detail '${detail.id}' must have a kind, title, and structured data`);
    }
    for (const value of Object.values(detail.data ?? {})) {
      if (typeof value === "number" && !Number.isFinite(value)) errors.push(`construction detail '${detail.id}' has a non-finite value`);
      if (typeof value !== "string" && typeof value !== "number" && typeof value !== "boolean") {
        errors.push(`construction detail '${detail.id}' contains unsupported data`);
      }
    }
  }
  if (!Array.isArray(specification.productionNotes) || !Array.isArray(specification.finishingNotes)) {
    errors.push("productionNotes and finishingNotes must be arrays");
  }
  return errors;
}

export function serializeProductionSpecification(specification: ProductionSpecification): string {
  const errors = validateProductionSpecification(specification);
  if (errors.length) throw new Error(`cannot serialize invalid production specification: ${errors[0]}`);
  return canonicalJson(specification);
}

export function deserializeProductionSpecification(serialized: string): ProductionSpecification {
  let parsed: unknown;
  try {
    parsed = JSON.parse(serialized);
  } catch {
    throw new Error("serialized production specification is not valid JSON");
  }
  const errors = validateProductionSpecification(parsed as ProductionSpecification);
  if (errors.length) throw new Error(`invalid production specification: ${errors[0]}`);
  return clone(parsed as ProductionSpecification);
}
