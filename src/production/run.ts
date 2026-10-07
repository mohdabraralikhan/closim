import type { GradingDocument } from "../grading/types.js";
import { deriveSize } from "../grading/derive.js";
import { validateMarker, type Marker } from "../marker/model.js";
import { placedPolygon } from "../marker/nest.js";
import { auditPlacements } from "../marker/nest.js";
import { fabricRulesFrom } from "../marker/fabric.js";

export const PRODUCTION_RUN_VERSION = 1;

export type ProductionRunStatus = "draft" | "planned" | "cutting" | "complete" | "cancelled";

export interface MarkerAssignment {
  markerId: string;
  fabricId: string;
  repeats: number;
}

export interface CutBatch {
  id: string;
  markerAssignments: MarkerAssignment[];
}

export interface ProductionRun {
  schemaVersion: typeof PRODUCTION_RUN_VERSION;
  id: string;
  garmentId: string;
  styleNumber: string;
  revision: string;
  gradingId: string;
  status: ProductionRunStatus;
  sizeQuantities: Record<string, number>;
  panelMaterialIds: Record<string, string>;
  /** Known usable inventory length by material/fabric id (metres). */
  fabricAvailableM?: Record<string, number>;
  batches: CutBatch[];
}

export interface CutPlanResult {
  runId: string;
  requested: Record<string, number>;
  produced: Record<string, number>;
  remaining: Record<string, number>;
  requiredPieces: Record<string, Record<string, number>>;
  cutPieces: Record<string, Record<string, number>>;
  batches: Array<{
    batchId: string;
    assignments: MarkerAssignment[];
    fabricRequirements: Array<{
      fabricId: string;
      lengthM: number;
      estimatedWasteM2: number;
    }>;
  }>;
  fabricInventory: Array<{
    fabricId: string;
    requiredLengthM: number;
    availableLengthM?: number;
    remainingLengthM?: number;
    shortfallM?: number;
  }>;
  diagnostics: Array<{ severity: "error" | "warning"; code: string; message: string }>;
  complete: boolean;
}

export function markerConsumptionForRun(
  run: ProductionRun,
  markers: readonly Marker[],
  actualConsumptionMByMarker: Record<string, number> = {},
): Record<string, {
  lengthM: number;
  widthM: number;
  repeats: number;
  estimatedConsumptionM: number;
  estimatedConsumptionAreaM2: number;
  actualConsumptionM?: number;
}> {
  const markerById = new Map(markers.map((marker) => [marker.id, marker]));
  const totals = new Map<string, { lengthM: number; widthM: number; repeats: number }>();
  for (const batch of run.batches) {
    for (const assignment of batch.markerAssignments) {
      const marker = markerById.get(assignment.markerId);
      if (!marker) throw new Error(`marker '${assignment.markerId}' does not exist for run '${run.id}'`);
      const issues = validateMarker(marker);
      if (issues.length) throw new Error(`marker '${marker.id}' is invalid: ${issues[0].message}`);
      const rules = fabricRulesFrom(marker.fabric, marker.constraint);
      const audit = auditPlacements(marker, marker.pieces, rules, () => 0);
      if (audit.length || marker.pieces.some((piece) =>
        !marker.placements.some((placement) => placement.instanceId === piece.instanceId))) {
        throw new Error(`marker '${marker.id}' placements are incomplete or invalid`);
      }
      let lengthM = 0;
      for (const piece of marker.pieces) {
        const placement = marker.placements.find((entry) => entry.instanceId === piece.instanceId)!;
        lengthM = Math.max(lengthM, ...placedPolygon(piece, placement).map((point) => point[1]));
      }
      const current = totals.get(marker.id) ?? {
        lengthM,
        widthM: marker.fabric.usableWidthM,
        repeats: 0,
      };
      if (current.lengthM !== lengthM || current.widthM !== marker.fabric.usableWidthM) {
        throw new Error(`marker '${marker.id}' has conflicting configurations within the production run`);
      }
      current.repeats += assignment.repeats;
      totals.set(marker.id, current);
    }
  }
  return Object.fromEntries([...totals].map(([markerId, value]) => {
    const actualConsumptionM = actualConsumptionMByMarker[markerId];
    if (actualConsumptionM !== undefined && (!Number.isFinite(actualConsumptionM) || actualConsumptionM < 0)) {
      throw new Error(`actual consumption for marker '${markerId}' must be finite and non-negative`);
    }
    return [markerId, {
      ...value,
      estimatedConsumptionM: value.lengthM * value.repeats,
      estimatedConsumptionAreaM2: value.lengthM * value.widthM * value.repeats,
      ...(actualConsumptionM !== undefined ? { actualConsumptionM } : {}),
    }];
  }));
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

export function createProductionRun(input: Omit<ProductionRun, "schemaVersion" | "batches"> & { batches?: CutBatch[] }): ProductionRun {
  const run: ProductionRun = {
    schemaVersion: PRODUCTION_RUN_VERSION,
    ...clone(input),
    batches: clone(input.batches ?? []),
  };
  const issues = validateProductionRun(run);
  if (issues.length) throw new Error(`invalid production run: ${issues[0]}`);
  return run;
}

export function validateProductionRun(run: ProductionRun): string[] {
  if (!run || run.schemaVersion !== PRODUCTION_RUN_VERSION || !run.id || !run.garmentId ||
    !run.styleNumber || !run.revision || !run.gradingId) {
    return ["run shape or required identity is invalid"];
  }
  const errors: string[] = [];
  if (!run.sizeQuantities || typeof run.sizeQuantities !== "object" || Array.isArray(run.sizeQuantities)) {
    errors.push("size quantities must be an object");
  }
  if (!run.panelMaterialIds || typeof run.panelMaterialIds !== "object" || Array.isArray(run.panelMaterialIds)) {
    errors.push("panel material assignments must be an object");
  }
  if (!Array.isArray(run.batches)) errors.push("cut batches must be an array");
  const status: ProductionRunStatus[] = ["draft", "planned", "cutting", "complete", "cancelled"];
  if (!status.includes(run.status)) errors.push(`unknown status '${String(run.status)}'`);
  const sizes = new Set<string>();
  for (const [sizeId, quantity] of Object.entries(run.sizeQuantities ?? {})) {
    if (!sizeId || sizes.has(sizeId)) errors.push(`size id '${sizeId}' is invalid or duplicated`);
    sizes.add(sizeId);
    if (!Number.isSafeInteger(quantity) || quantity < 0) errors.push(`size '${sizeId}' quantity must be a non-negative safe integer`);
  }
  for (const [fabricId, length] of Object.entries(run.fabricAvailableM ?? {})) {
    if (!fabricId || !Number.isFinite(length) || length < 0) errors.push(`fabric '${fabricId}' available quantity must be finite and non-negative`);
  }
  for (const [panelId, fabricId] of Object.entries(run.panelMaterialIds ?? {})) {
    if (!panelId || !fabricId?.trim()) errors.push(`panel '${panelId}' has an invalid material/fabric assignment`);
  }
  if (!Array.isArray(run.batches)) return errors;
  const seenBatches = new Set<string>();
  for (const batch of run.batches ?? []) {
    if (!batch.id || seenBatches.has(batch.id)) errors.push(`batch id '${batch.id}' is invalid or duplicated`);
    seenBatches.add(batch.id);
    for (const assignment of batch.markerAssignments ?? []) {
      if (!assignment.markerId || !assignment.fabricId || !Number.isSafeInteger(assignment.repeats) || assignment.repeats < 1) {
        errors.push(`batch '${batch.id}' has an invalid marker assignment`);
      }
    }
  }
  return errors;
}

export function calculateCutPlan(
  run: ProductionRun,
  grading: GradingDocument,
  markers: readonly Marker[],
  options: { panelPiecesPerGarment?: Record<string, number> } = {},
): CutPlanResult {
  const errors = validateProductionRun(run);
  if (errors.length) throw new Error(`cannot calculate cut plan: ${errors[0]}`);
  if (run.gradingId !== grading.id) throw new Error("production run references a different grading document");
  const diagnostics: CutPlanResult["diagnostics"] = [];
  const activeSizes = new Map(grading.sizeSet.sizes.filter((size) => size.active).map((size) => [size.id, size]));
  for (const sizeId of Object.keys(run.sizeQuantities)) {
    if (!activeSizes.has(sizeId)) diagnostics.push({ severity: "error", code: "missing-size", message: `size '${sizeId}' is missing or inactive` });
  }

  const panelIds = new Set<string>();
  for (const sizeId of Object.keys(run.sizeQuantities)) {
    if (!activeSizes.has(sizeId)) continue;
    try {
      const { graded } = deriveSize(grading, sizeId);
      for (const panel of graded.document.panels) panelIds.add(panel.id);
    } catch (error) {
      diagnostics.push({ severity: "error", code: "grading-failed", message: `size '${sizeId}' grading failed: ${error instanceof Error ? error.message : String(error)}` });
    }
  }
  const panelCounts = new Map<string, number>();
  for (const panelId of [...panelIds].sort()) {
    const count = options.panelPiecesPerGarment?.[panelId] ?? 1;
    if (!Number.isSafeInteger(count) || count < 1) {
      diagnostics.push({ severity: "error", code: "invalid-piece-count", message: `panel '${panelId}' piece count must be a positive safe integer` });
    }
    panelCounts.set(panelId, count);
    if (!run.panelMaterialIds[panelId]) {
      diagnostics.push({ severity: "error", code: "invalid-fabric", message: `panel '${panelId}' has no material/fabric assignment` });
    }
  }
  for (const panelId of Object.keys(options.panelPiecesPerGarment ?? {})) {
    if (!panelIds.has(panelId)) {
      diagnostics.push({ severity: "error", code: "missing-pattern-piece", message: `piece-count assignment references unknown panel '${panelId}'` });
    }
  }
  for (const panelId of Object.keys(run.panelMaterialIds)) {
    if (!panelIds.has(panelId)) {
      diagnostics.push({ severity: "error", code: "missing-pattern-piece", message: `material assignment references unknown panel '${panelId}'` });
    }
  }
  const produced: Record<string, number> = Object.fromEntries(Object.keys(run.sizeQuantities).map((id) => [id, 0]));
  const requiredPieces: Record<string, Record<string, number>> = {};
  const cutPieces: Record<string, Record<string, number>> = {};
  for (const [sizeId, quantity] of Object.entries(run.sizeQuantities)) {
    requiredPieces[sizeId] = {};
    cutPieces[sizeId] = {};
    for (const [panelId, perGarment] of panelCounts) {
      const required = quantity * perGarment;
      if (!Number.isSafeInteger(required)) {
        diagnostics.push({ severity: "error", code: "quantity-overflow", message: `size '${sizeId}' required piece count exceeds safe integer range` });
      }
      requiredPieces[sizeId][panelId] = Number.isSafeInteger(required) ? required : 0;
      cutPieces[sizeId][panelId] = 0;
    }
  }
  const markerById = new Map(markers.map((marker) => [marker.id, marker]));
  if (markerById.size !== markers.length) {
    diagnostics.push({ severity: "error", code: "duplicate-marker", message: "marker input contains duplicate marker ids" });
  }
  const batchResults: CutPlanResult["batches"] = [];
  for (const batch of run.batches) {
    const fabricUse = new Map<string, { lengthM: number; estimatedWasteM2: number }>();
    for (const assignment of batch.markerAssignments) {
      const marker = markerById.get(assignment.markerId);
      if (!marker) {
        diagnostics.push({ severity: "error", code: "missing-marker", message: `batch '${batch.id}' marker '${assignment.markerId}' does not exist` });
        continue;
      }
      const markerIssues = validateMarker(marker);
      let markerValid = markerIssues.length === 0;
      for (const issue of markerIssues) {
        diagnostics.push({ severity: "error", code: "invalid-marker", message: `marker '${marker.id}': ${issue.message}` });
      }
      try {
        const rules = fabricRulesFrom(marker.fabric, marker.constraint);
        for (const issue of auditPlacements(marker, marker.pieces, rules, () => 0)) {
          diagnostics.push({ severity: "error", code: "invalid-marker", message: `marker '${marker.id}': ${issue.message}` });
          markerValid = false;
        }
      } catch (error) {
        markerValid = false;
        diagnostics.push({
          severity: "error",
          code: "invalid-marker",
          message: `marker '${marker.id}' fabric rules are invalid: ${error instanceof Error ? error.message : String(error)}`,
        });
      }
      if (marker.fabric.id !== assignment.fabricId) {
        diagnostics.push({ severity: "error", code: "invalid-fabric", message: `marker '${marker.id}' uses fabric '${marker.fabric.id}', not '${assignment.fabricId}'` });
        continue;
      }
      const totalsBySize = new Map<string, Map<string, number>>();
      for (const piece of marker.pieces) {
        if (!panelIds.has(piece.panelId)) {
          diagnostics.push({ severity: "error", code: "missing-pattern-piece", message: `marker '${marker.id}' has unknown pattern panel '${piece.panelId}'` });
          continue;
        }
        const panels = totalsBySize.get(piece.sizeId) ?? new Map<string, number>();
        panels.set(piece.panelId, (panels.get(piece.panelId) ?? 0) + 1);
        totalsBySize.set(piece.sizeId, panels);
      }
      const placementById = new Map(marker.placements.map((placement) => [placement.instanceId, placement]));
      const placedIds = new Set<string>();
      let markerLengthM = 0;
      let patternAreaM2 = 0;
      for (const piece of marker.pieces) {
        const placement = placementById.get(piece.instanceId);
        if (!placement) {
          diagnostics.push({ severity: "error", code: "unplaced-marker-piece", message: `marker '${marker.id}' piece '${piece.instanceId}' is not placed` });
          markerValid = false;
          continue;
        }
        if (placedIds.has(piece.instanceId)) {
          diagnostics.push({ severity: "error", code: "duplicate-marker-placement", message: `marker '${marker.id}' duplicates placement '${piece.instanceId}'` });
          markerValid = false;
          continue;
        }
        placedIds.add(piece.instanceId);
        markerLengthM = Math.max(markerLengthM, ...placedPolygon(piece, placement).map((point) => point[1]));
        patternAreaM2 += piece.areaM2;
      }
      if (!markerValid) continue;
      const markerFabricArea = marker.fabric.usableWidthM * markerLengthM;
      const markerWasteM2 = Math.max(0, markerFabricArea - patternAreaM2);
      const consumedLength = markerLengthM * assignment.repeats;
      for (const [sizeId, sizePanels] of totalsBySize) {
        if (produced[sizeId] === undefined) {
          diagnostics.push({ severity: "error", code: "missing-size", message: `marker '${marker.id}' contains unplanned size '${sizeId}'` });
          continue;
        }
        for (const [panelId, count] of sizePanels) {
          const cutQuantity = count * assignment.repeats;
          if (!Number.isSafeInteger(cutQuantity) ||
            !Number.isSafeInteger((cutPieces[sizeId][panelId] ?? 0) + cutQuantity)) {
            diagnostics.push({ severity: "error", code: "quantity-overflow", message: `marker '${marker.id}' piece quantity exceeds safe integer range` });
            continue;
          }
          cutPieces[sizeId][panelId] = (cutPieces[sizeId][panelId] ?? 0) + cutQuantity;
          const materialId = run.panelMaterialIds[panelId];
          if (materialId !== assignment.fabricId) {
            diagnostics.push({ severity: "error", code: "incompatible-material", message: `panel '${panelId}' requires '${materialId}', marker uses '${assignment.fabricId}'` });
          }
        }
      }
      const prior = fabricUse.get(assignment.fabricId) ?? { lengthM: 0, estimatedWasteM2: 0 };
      prior.lengthM += consumedLength;
      prior.estimatedWasteM2 += markerWasteM2 * assignment.repeats;
      fabricUse.set(assignment.fabricId, prior);
    }
    batchResults.push({
      batchId: batch.id,
      assignments: clone(batch.markerAssignments),
      fabricRequirements: [...fabricUse].sort(([a], [b]) => a.localeCompare(b))
        .map(([fabricId, value]) => ({ fabricId, ...value })),
    });
  }
  const runFabricUse = new Map<string, number>();
  for (const batch of batchResults) {
    for (const requirement of batch.fabricRequirements) {
      runFabricUse.set(requirement.fabricId,
        (runFabricUse.get(requirement.fabricId) ?? 0) + requirement.lengthM);
    }
  }
  const fabricInventory: CutPlanResult["fabricInventory"] = [...runFabricUse]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([fabricId, requiredLengthM]) => {
      const availableLengthM = run.fabricAvailableM?.[fabricId];
      return {
        fabricId,
        requiredLengthM,
        ...(availableLengthM !== undefined ? {
          availableLengthM,
          remainingLengthM: Math.max(0, availableLengthM - requiredLengthM),
          shortfallM: Math.max(0, requiredLengthM - availableLengthM),
        } : {}),
      };
    });
  for (const [sizeId, panels] of Object.entries(cutPieces)) {
    const completeGarments = [...panelCounts].map(([panelId, perGarment]) =>
      Math.floor((panels[panelId] ?? 0) / perGarment));
    produced[sizeId] = Math.min(run.sizeQuantities[sizeId] ?? 0,
      completeGarments.length ? Math.min(...completeGarments) : 0);
    for (const [panelId, needed] of Object.entries(requiredPieces[sizeId])) {
      if ((panels[panelId] ?? 0) > needed) {
        diagnostics.push({ severity: "error", code: "excess-quantity", message: `size '${sizeId}' panel '${panelId}' assigned ${(panels[panelId] ?? 0) - needed} excess pieces` });
      }
      if ((panels[panelId] ?? 0) < needed) {
        diagnostics.push({ severity: "warning", code: "incomplete-quantity", message: `size '${sizeId}' panel '${panelId}' needs ${needed} pieces, assigned ${panels[panelId] ?? 0}` });
      }
      if ((panels[panelId] ?? 0) % (panelCounts.get(panelId) ?? 1) !== 0) {
        diagnostics.push({ severity: "warning", code: "partial-garment", message: `size '${sizeId}' panel '${panelId}' quantity leaves incomplete garments` });
      }
    }
  }
  const remaining = Object.fromEntries(Object.entries(run.sizeQuantities).map(([sizeId, quantity]) =>
    [sizeId, Math.max(0, quantity - (produced[sizeId] ?? 0))]));
  const complete = Object.values(remaining).every((quantity) => quantity === 0) &&
    diagnostics.every((diagnostic) => diagnostic.severity !== "error") &&
    fabricInventory.every((fabric) => (fabric.shortfallM ?? 0) === 0);
  return {
    runId: run.id,
    requested: clone(run.sizeQuantities),
    produced,
    remaining,
    requiredPieces,
    cutPieces,
    batches: batchResults,
    fabricInventory,
    diagnostics,
    complete,
  };
}

export function serializeProductionRun(run: ProductionRun): string {
  const errors = validateProductionRun(run);
  if (errors.length) throw new Error(`cannot serialize invalid production run: ${errors[0]}`);
  return canonical(run);
}

export function deserializeProductionRun(serialized: string): ProductionRun {
  let parsed: unknown;
  try {
    parsed = JSON.parse(serialized);
  } catch {
    throw new Error("serialized production run is not valid JSON");
  }
  const run = parsed as ProductionRun;
  const errors = validateProductionRun(run);
  if (errors.length) throw new Error(`invalid production run: ${errors[0]}`);
  return clone(run);
}
