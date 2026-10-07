import type { Marker } from "../marker/model.js";

export const BOM_SCHEMA_VERSION = 1;

export type QuantityBasis = "per-garment" | "per-size" | "per-panel" | "fixed" | "length" | "area" | "count";

export interface BOMMaterial {
  id: string;
  name: string;
  category: string;
  unit: string;
  supplierReference?: string;
}

export interface MaterialCost {
  unitCost: number;
  currency: string;
}

export interface BOMItem {
  id: string;
  materialId?: string;
  componentId?: string;
  panelId?: string;
  sizeId?: string;
  basis: QuantityBasis;
  quantity: number;
  unit: string;
  wasteAllowance: number;
  cost?: MaterialCost;
  notes?: string;
}

export interface BOM {
  schemaVersion: typeof BOM_SCHEMA_VERSION;
  id: string;
  garmentId: string;
  revision: string;
  materials: BOMMaterial[];
  items: BOMItem[];
  markerReferences: Array<{ markerId: string; fabricId: string }>;
}

export interface BOMIssue {
  code: "missing-material" | "invalid-quantity" | "invalid-unit" | "duplicate-item" |
    "inconsistent-assignment" | "missing-component" | "missing-marker";
  message: string;
  itemId?: string;
}

export interface BOMRequirement {
  itemId: string;
  materialId?: string;
  basis: QuantityBasis;
  baseQuantity: number;
  wasteAllowance: number;
  requiredQuantity: number;
  theoreticalQuantity: number;
  estimatedConsumption?: number;
  estimatedConsumptionUnit?: string;
  actualConsumption?: number;
  actualConsumptionUnit?: string;
  unit: string;
  currency?: string;
  estimatedCost?: number;
}

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value !== null && typeof value === "object") {
    const obj = value as Record<string, unknown>;
    return `{${Object.keys(obj).sort().map((key) => `${JSON.stringify(key)}:${canonical(obj[key])}`).join(",")}}`;
  }
  return JSON.stringify(typeof value === "number" && Object.is(value, -0) ? 0 : value);
}

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

export function validateBOM(
  bom: BOM,
  options: { knownComponents?: readonly string[]; knownPanels?: readonly string[]; markers?: readonly Marker[] } = {},
): BOMIssue[] {
  const issues: BOMIssue[] = [];
  if (!bom || bom.schemaVersion !== BOM_SCHEMA_VERSION || !bom.id || !bom.garmentId || !bom.revision) {
    return [{ code: "invalid-quantity", message: "BOM shape, id, garment, revision, or schema is invalid" }];
  }
  const materials = new Map<string, BOMMaterial>();
  for (const material of bom.materials ?? []) {
    if (!material.id || materials.has(material.id)) {
      issues.push({ code: "inconsistent-assignment", message: `material id '${material.id}' is empty or duplicated` });
    }
    if (!material.name?.trim() || !material.category?.trim() || !material.unit?.trim()) {
      issues.push({ code: "invalid-unit", message: `material '${material.id}' needs a name, category, and unit` });
    }
    materials.set(material.id, material);
  }
  const itemIds = new Set<string>();
  const itemKeys = new Set<string>();
  for (const item of bom.items ?? []) {
    if (!item.id || itemIds.has(item.id)) issues.push({ code: "duplicate-item", message: `duplicate or empty BOM item '${item.id}'`, itemId: item.id });
    itemIds.add(item.id);
    const itemKey = `${item.materialId ?? ""}|${item.componentId ?? ""}|${item.panelId ?? ""}|${item.sizeId ?? ""}|${item.basis}|${item.unit}`;
    if (itemKeys.has(itemKey)) {
      issues.push({ code: "duplicate-item", message: `item '${item.id}' duplicates a material/component basis assignment`, itemId: item.id });
    }
    itemKeys.add(itemKey);
    if (!["per-garment", "per-size", "per-panel", "fixed", "length", "area", "count"].includes(item.basis)) {
      issues.push({ code: "invalid-quantity", message: `item '${item.id}' has unknown quantity basis`, itemId: item.id });
    }
    if (!Number.isFinite(item.quantity) || item.quantity <= 0 ||
      !Number.isFinite(item.wasteAllowance) || item.wasteAllowance < 0) {
      issues.push({ code: "invalid-quantity", message: `item '${item.id}' quantity must be positive and waste finite/non-negative`, itemId: item.id });
    }
    if (!item.unit?.trim()) issues.push({ code: "invalid-unit", message: `item '${item.id}' needs a unit`, itemId: item.id });
    if (item.materialId && !materials.has(item.materialId)) {
      issues.push({ code: "missing-material", message: `item '${item.id}' references missing material '${item.materialId}'`, itemId: item.id });
    }
    if (item.materialId && materials.has(item.materialId) && item.unit !== materials.get(item.materialId)!.unit) {
      issues.push({ code: "inconsistent-assignment", message: `item '${item.id}' unit '${item.unit}' differs from material unit '${materials.get(item.materialId)!.unit}'`, itemId: item.id });
    }
    if (!item.materialId && !item.componentId) {
      issues.push({ code: "missing-material", message: `item '${item.id}' needs a material or component assignment`, itemId: item.id });
    }
    if (item.componentId && options.knownComponents && !options.knownComponents.includes(item.componentId)) {
      issues.push({ code: "missing-component", message: `item '${item.id}' references missing component '${item.componentId}'`, itemId: item.id });
    }
    if (item.panelId && options.knownPanels && !options.knownPanels.includes(item.panelId)) {
      issues.push({ code: "inconsistent-assignment", message: `item '${item.id}' references unknown panel '${item.panelId}'`, itemId: item.id });
    }
    if (item.cost && (!Number.isFinite(item.cost.unitCost) || item.cost.unitCost < 0 ||
      typeof item.cost.currency !== "string" || !item.cost.currency.trim())) {
      issues.push({ code: "invalid-quantity", message: `item '${item.id}' has invalid price or currency`, itemId: item.id });
    }
  }
  const knownMarkers = new Set((options.markers ?? []).map((marker) => marker.id));
  const markerReferenceKeys = new Set<string>();
  for (const ref of bom.markerReferences ?? []) {
    if (!ref.markerId || !ref.fabricId || (options.markers && !knownMarkers.has(ref.markerId))) {
      issues.push({ code: "missing-marker", message: `marker assignment '${ref.markerId}' is missing or invalid` });
    }
    const key = `${ref.markerId}|${ref.fabricId}`;
    if (markerReferenceKeys.has(key)) {
      issues.push({ code: "inconsistent-assignment", message: `marker '${ref.markerId}' is assigned to fabric '${ref.fabricId}' more than once` });
    }
    markerReferenceKeys.add(key);
    if (!materials.has(ref.fabricId)) {
      issues.push({ code: "missing-material", message: `marker '${ref.markerId}' references missing fabric material '${ref.fabricId}'` });
    }
    const marker = options.markers?.find((entry) => entry.id === ref.markerId);
    if (marker && marker.fabric.id !== ref.fabricId) {
      issues.push({ code: "inconsistent-assignment", message: `marker '${ref.markerId}' fabric does not match '${ref.fabricId}'` });
    }
  }
  return issues;
}

export function generateBOMRequirements(
  bom: BOM,
  input: {
    garmentQuantity: number;
    sizeQuantities?: Record<string, number>;
    panelQuantities?: Record<string, number>;
    markerMetrics?: Record<string, {
      lengthM: number;
      widthM: number;
      repeats?: number;
      estimatedConsumptionM?: number;
      estimatedConsumptionAreaM2?: number;
      actualConsumptionM?: number;
    }>;
  },
): BOMRequirement[] {
  if (!Number.isInteger(input.garmentQuantity) || input.garmentQuantity < 0) {
    throw new Error("garment quantity must be a non-negative integer");
  }
  for (const [label, quantities] of [["size", input.sizeQuantities], ["panel", input.panelQuantities]] as const) {
    for (const [id, quantity] of Object.entries(quantities ?? {})) {
      if (!Number.isSafeInteger(quantity) || quantity < 0) {
        throw new Error(`${label} '${id}' quantity must be a non-negative safe integer`);
      }
    }
  }
  const issues = validateBOM(bom);
  if (issues.length) throw new Error(`cannot calculate invalid BOM requirements: ${issues[0].message}`);
  for (const [markerId, metrics] of Object.entries(input.markerMetrics ?? {})) {
    for (const [name, value] of Object.entries(metrics)) {
      if (value !== undefined && (!Number.isFinite(value) || value < 0)) {
        throw new Error(`marker '${markerId}' ${name} must be finite and non-negative`);
      }
    }
  }
  return bom.items.map((item) => {
    let multiplier: number;
    switch (item.basis) {
      case "per-garment": case "count": multiplier = input.garmentQuantity; break;
      case "per-size":
        multiplier = item.sizeId
          ? input.sizeQuantities?.[item.sizeId] ?? 0
          : Object.values(input.sizeQuantities ?? {}).reduce((sum, quantity) => sum + quantity, 0);
        break;
      case "per-panel":
        multiplier = item.panelId
          ? input.panelQuantities?.[item.panelId] ?? 0
          : Object.values(input.panelQuantities ?? {}).reduce((sum, quantity) => sum + quantity, 0);
        break;
      case "fixed": multiplier = 1; break;
      case "length": case "area": multiplier = input.garmentQuantity; break;
    }

    const theoreticalQuantity = item.quantity * multiplier;
    const requiredQuantity = theoreticalQuantity * (1 + item.wasteAllowance);
    if (!Number.isFinite(theoreticalQuantity) || !Number.isFinite(requiredQuantity)) {
      throw new Error(`BOM item '${item.id}' requirement overflows finite numeric range`);
    }
    const markerReferences = bom.markerReferences.filter((ref) => ref.fabricId === item.materialId);
    const markerMetrics = markerReferences.map((ref) => input.markerMetrics?.[ref.markerId]);
    const hasAllMarkerMetrics = markerMetrics.length > 0 && markerMetrics.every((metrics) => metrics !== undefined);
    const estimatedConsumption = hasAllMarkerMetrics
      ? markerMetrics.reduce((sum, metrics) => sum + (item.basis === "area"
        ? metrics!.estimatedConsumptionAreaM2 ?? metrics!.lengthM * metrics!.widthM * (metrics!.repeats ?? multiplier)
        : metrics!.estimatedConsumptionM ?? metrics!.lengthM * (metrics!.repeats ?? multiplier)), 0)
      : undefined;
    const hasAllActualConsumption = hasAllMarkerMetrics &&
      markerMetrics.every((metrics) => metrics!.actualConsumptionM !== undefined);
    const actualConsumption = hasAllActualConsumption
      ? markerMetrics.reduce((sum, metrics) => sum + metrics!.actualConsumptionM!, 0)
      : undefined;
    return {
      itemId: item.id,
      ...(item.materialId ? { materialId: item.materialId } : {}),
      basis: item.basis,
      baseQuantity: item.quantity,
      wasteAllowance: item.wasteAllowance,
      requiredQuantity,
      theoreticalQuantity,
      ...(estimatedConsumption !== undefined ? {
        estimatedConsumption,
        estimatedConsumptionUnit: item.basis === "area" ? "m2" : "m",
      } : {}),
      ...(actualConsumption !== undefined ? { actualConsumption, actualConsumptionUnit: "m" } : {}),
      unit: item.unit,
      ...(item.cost ? {
        currency: item.cost.currency,
        estimatedCost: requiredQuantity * item.cost.unitCost,
      } : {}),
    };
  });
}

export function createBOMFromSpecification(
  specification: import("./specification-model.js").ProductionSpecification,
  options: {
    id: string;
    revision: string;
    materialUnits: Record<string, string>;
    items?: BOMItem[];
    markerReferences?: BOM["markerReferences"];
  },
): BOM {
  const materials: BOMMaterial[] = specification.materials.map((material) => {
    const unit = options.materialUnits[material.id];
    if (!unit?.trim()) throw new Error(`BOM unit is required for material '${material.id}'`);
    return {
      id: material.id,
      name: material.name,
      category: material.category,
      unit,
      ...(material.supplierReference ? { supplierReference: material.supplierReference } : {}),
    };
  });
  const panelItems: BOMItem[] = specification.panels.flatMap((panel) => {
    if (!panel.materialId) return [];
    const material = materials.find((entry) => entry.id === panel.materialId);
    if (!material) throw new Error(`panel '${panel.panelId}' references missing material '${panel.materialId}'`);
    return [{
      id: `bom/panel/${panel.panelId}`,
      materialId: material.id,
      panelId: panel.panelId,
      basis: "per-panel",
      quantity: 1,
      unit: material.unit,
      wasteAllowance: 0,
      ...(panel.notes ? { notes: panel.notes } : {}),
    }];
  });
  const bom: BOM = {
    schemaVersion: BOM_SCHEMA_VERSION,
    id: options.id,
    garmentId: specification.garmentId,
    revision: options.revision,
    materials,
    items: [...panelItems, ...(options.items ?? [])],
    markerReferences: clone(options.markerReferences ?? []),
  };
  const issues = validateBOM(bom, { knownPanels: specification.panels.map((panel) => panel.panelId) });
  if (issues.length) throw new Error(`cannot generate BOM: ${issues[0].message}`);
  return bom;
}

export function serializeBOM(bom: BOM): string {
  const issues = validateBOM(bom);
  if (issues.length) throw new Error(`cannot serialize invalid BOM: ${issues[0].message}`);
  return canonical(bom);
}

export function deserializeBOM(serialized: string): BOM {
  let parsed: unknown;
  try {
    parsed = JSON.parse(serialized);
  } catch {
    throw new Error("serialized BOM is not valid JSON");
  }
  const bom = parsed as BOM;
  const issues = validateBOM(bom);
  if (issues.length) throw new Error(`invalid BOM: ${issues[0].message}`);
  return clone(bom);
}
