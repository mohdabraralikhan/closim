import type { PatternDocument } from "../pattern/cad.js";
import type { GradingDocument } from "../grading/types.js";
import { deriveSize } from "../grading/derive.js";
import type { Marker } from "../marker/model.js";
import type { BOM, BOMRequirement, BOMIssue } from "./bom.js";
import type { ProductionRun, CutPlanResult } from "./run.js";
import type { ProductionSpecification, TechPack } from "./specification-model.js";
import type { ProductionSet } from "../cad/production.js";
import { placedPolygon } from "../marker/nest.js";

export type ArtifactState = "current" | "stale" | "missing" | "invalid";
export type ProductionArtifactKind =
  | "pattern" | "grading" | "engineering" | "marker" | "bom" | "run" | "tech-pack" | "package" | "production-package";

export interface DependencyArtifact {
  id: string;
  kind: ProductionArtifactKind;
  dependencies: string[];
  sourceFingerprint: string;
  currentSourceFingerprint: string;
  state: ArtifactState;
  sourceRevisionId?: string;
}

export interface ManufacturingDashboard {
  garmentId: string;
  garmentName: string;
  styleNumber: string;
  revision: string;
  sizes: Array<{ id: string; label: string; quantity: number }>;
  plannedQuantity: number;
  materials: Array<{ id: string; name: string; category: string; quantity?: number; unit?: string; cost?: number; currency?: string }>;
  fabricConsumption: Array<{ markerId: string; fabricId: string; lengthM: number; utilization?: number; wasteM2?: number }>;
  bomSummary: { itemCount: number; missingMaterials: number; estimatedCost?: number; currency?: string };
  validation: { status: "ready" | "warnings" | "errors"; errors: string[]; warnings: string[] };
  exportStatus: ArtifactState;
  artifacts: DependencyArtifact[];
}

export interface ManufacturingReports {
  pattern: {
    pieceCount: number;
    sizes: Array<{ id: string; label: string; pieceCount: number }>;
    measurements: TechPack["measurements"];
    annotations: Array<{ id: string; panelId: string; note: string }>;
  };
  markers: Array<{ id: string; fabricId: string; widthM: number; lengthM?: number; utilization?: number; wasteM2?: number; pieceCount: number }>;
  materials: { requirements: BOMRequirement[]; issues: BOMIssue[] };
  production: CutPlanResult | null;
  validation: ManufacturingDashboard["validation"];
}

export interface ProductionFingerprint {
  captured: string;
  current: string;
}

export function createProductionDependencyGraph(input: {
  pattern: ProductionFingerprint;
  grading?: ProductionFingerprint;
  engineering?: ProductionFingerprint;
  materials?: ProductionFingerprint;
  specification?: ProductionFingerprint;
  bom?: ProductionFingerprint;
  markers?: Array<{ id: string; fingerprint: ProductionFingerprint; fabricId: string }>;
  run?: ProductionFingerprint;
  techPack?: ProductionFingerprint;
  package?: ProductionFingerprint;
}): DependencyArtifact[] {
  const result: DependencyArtifact[] = [];
  const add = (
    id: string,
    kind: ProductionArtifactKind,
    dependencies: string[],
    value?: ProductionFingerprint,
  ): void => {
    result.push({
      id,
      kind,
      dependencies,
      sourceFingerprint: value?.captured ?? "",
      currentSourceFingerprint: value?.current ?? "",
      state: value ? (value.captured === value.current ? "current" : "stale") : "missing",
    });
  };
  add("pattern", "pattern", [], input.pattern);
  if (input.grading) add("grading", "grading", ["pattern"], input.grading);
  const gradeDependency = input.grading ? ["grading"] : [];
  add("engineering", "engineering", ["pattern", ...gradeDependency], input.engineering);
  add("materials", "engineering", [], input.materials);
  add("specification", "engineering", ["pattern", "materials"], input.specification);
  for (const marker of input.markers ?? []) {
    add(marker.id, "marker", [...gradeDependency, "engineering", `material:${marker.fabricId}`], marker.fingerprint);
    add(`material:${marker.fabricId}`, "engineering", [], input.materials);
  }
  add("bom", "bom", ["materials", "specification", ...(input.markers ?? []).map((marker) => marker.id)], input.bom);
  add("run", "run", [...gradeDependency, "bom", ...(input.markers ?? []).map((marker) => marker.id)], input.run);
  add("tech-pack", "tech-pack", ["pattern", ...gradeDependency, "engineering", "specification", "bom"], input.techPack);
  add("package", "production-package", [
    "pattern", ...gradeDependency, "engineering", "materials", "specification", "bom", "run", "tech-pack",
    ...(input.markers ?? []).map((marker) => marker.id),
  ], input.package);
  return propagateArtifactStaleness(result, []);
}

export function propagateArtifactStaleness(
  artifacts: readonly DependencyArtifact[],
  changedSourceIds: readonly string[],
): DependencyArtifact[] {
  const next = artifacts.map((artifact) => ({ ...artifact, dependencies: [...artifact.dependencies] }));
  const stale = new Set(changedSourceIds);
  for (const artifact of next) {
    if (artifact.state === "missing" || artifact.state === "invalid" ||
      artifact.sourceFingerprint !== artifact.currentSourceFingerprint) stale.add(artifact.id);
  }
  let updated = true;
  while (updated) {
    updated = false;
    for (const artifact of next) {
      if (artifact.dependencies.some((dependency) => stale.has(dependency)) && !stale.has(artifact.id)) {
        stale.add(artifact.id);
        updated = true;
      }
    }
  }
  return next.map((artifact) => ({
    ...artifact,
    state: artifact.state === "missing" || artifact.state === "invalid"
      ? artifact.state
      : stale.has(artifact.id) ? "stale" : "current",
  }));
}

export function buildManufacturingReports(input: {
  garmentId: string;
  garmentName: string;
  garmentRevision: number;
  pattern: PatternDocument;
  grading?: GradingDocument;
  specification: ProductionSpecification;
  techPack: TechPack;
  bom?: BOM;
  bomIssues?: BOMIssue[];
  bomRequirements?: BOMRequirement[];
  markers?: Marker[];
  markerMetrics?: Record<string, { lengthM: number; utilization: number; wasteM2: number }>;
  run?: ProductionRun;
  cutPlan?: CutPlanResult;
  artifacts?: DependencyArtifact[];
  production?: ProductionSet;
}): { dashboard: ManufacturingDashboard; reports: ManufacturingReports } {
  const errors = [...input.techPack.validation.errors];
  const warnings = [...input.techPack.validation.warnings];
  if (!input.bom) warnings.push("BOM has not been generated");
  if (!input.run) warnings.push("production run has not been created");
  for (const issue of input.bomIssues ?? []) errors.push(issue.message);
  for (const diagnostic of input.cutPlan?.diagnostics ?? []) {
    (diagnostic.severity === "error" ? errors : warnings).push(diagnostic.message);
  }
  for (const artifact of input.artifacts ?? []) {
    if (artifact.state !== "current") {
      warnings.push(`artifact '${artifact.id}' is ${artifact.state}`);
    }
  }
  const sizes = input.grading
    ? input.grading.sizeSet.sizes.filter((size) => size.active).map((size) => ({
      id: size.id,
      label: size.label,
      quantity: input.run?.sizeQuantities[size.id] ?? 0,
    }))
    : [{ id: "base", label: "Base", quantity: input.run?.sizeQuantities.base ?? 0 }];
  const plannedQuantity = sizes.reduce((sum, size) => sum + size.quantity, 0);
  const markerRows = (input.markers ?? []).map((marker) => {
    const placementById = new Map(marker.placements.map((placement) => [placement.instanceId, placement]));
    let lengthM = 0;
    let areaM2 = 0;
    for (const piece of marker.pieces) {
      const placement = placementById.get(piece.instanceId);
      if (!placement) continue;
      lengthM = Math.max(lengthM, ...placedPolygon(piece, placement).map((point) => point[1]));
      areaM2 += piece.areaM2;
    }
    const fabricArea = marker.fabric.usableWidthM * lengthM;
    return {
      id: marker.id,
      fabricId: marker.fabric.id,
      widthM: marker.fabric.usableWidthM,
      lengthM,
      utilization: input.markerMetrics?.[marker.id]?.utilization ?? (fabricArea > 0 ? areaM2 / fabricArea : 0),
      wasteM2: input.markerMetrics?.[marker.id]?.wasteM2 ?? Math.max(0, fabricArea - areaM2),
      pieceCount: marker.pieces.length,
    };
  });
  const requirements = input.bomRequirements ?? [];
  const costsByCurrency = new Map<string, number>();
  for (const requirement of requirements) {
    if (requirement.estimatedCost !== undefined && requirement.currency) {
      costsByCurrency.set(requirement.currency, (costsByCurrency.get(requirement.currency) ?? 0) + requirement.estimatedCost);
    }
  }
  let estimatedCost: number | undefined;
  let currency: string | undefined;
  if (costsByCurrency.size === 1) {
    [currency, estimatedCost] = [...costsByCurrency.entries()][0];
  } else if (costsByCurrency.size > 1) {
    warnings.push("BOM estimates use multiple currencies and are not aggregated");
  }
  if (input.cutPlan && !input.cutPlan.complete) warnings.push("cut plan does not fulfill all planned production quantities");
  const artifacts = input.artifacts ?? [];
  const exportArtifact = artifacts.find((artifact) => artifact.kind === "package" || artifact.kind === "production-package");
  const exportStatus = exportArtifact?.state ?? "missing";
  if (exportStatus !== "current") warnings.push(`production export is ${exportStatus}`);
  const validationStatus = errors.length ? "errors" : warnings.length ? "warnings" : "ready";
  const dashboard: ManufacturingDashboard = {
    garmentId: input.garmentId,
    garmentName: input.garmentName,
    styleNumber: input.specification.styleNumber,
    revision: input.specification.revision,
    sizes,
    plannedQuantity,
    materials: input.specification.materials.map((material) => {
      const itemReqs = requirements.filter((requirement) => requirement.materialId === material.id);
      return {
        id: material.id,
        name: material.name,
        category: material.category,
        ...(itemReqs.length ? {
          quantity: itemReqs.reduce((sum, requirement) => sum + requirement.requiredQuantity, 0),
          unit: itemReqs[0].unit,
        } : {}),
        ...(() => {
          const ownRequirements = itemReqs.filter((requirement) => requirement.estimatedCost !== undefined && requirement.currency);
          const currencies = new Set(ownRequirements.map((requirement) => requirement.currency));
          return currencies.size === 1
            ? {
              cost: ownRequirements.reduce((sum, requirement) => sum + requirement.estimatedCost!, 0),
              currency: ownRequirements[0].currency,
            }
            : {};
        })(),
      };
    }),
    fabricConsumption: markerRows.map((row) => ({
      markerId: row.id,
      fabricId: row.fabricId,
      lengthM: row.lengthM ?? 0,
      ...(row.utilization !== undefined ? { utilization: row.utilization } : {}),
      ...(row.wasteM2 !== undefined ? { wasteM2: row.wasteM2 } : {}),
    })),
    bomSummary: {
      itemCount: input.bom?.items.length ?? 0,
      missingMaterials: input.bomIssues?.filter((issue) => issue.code === "missing-material").length ?? 0,
      ...(estimatedCost !== undefined ? { estimatedCost, currency } : {}),
    },
    validation: { status: validationStatus, errors, warnings },
    exportStatus,
    artifacts: artifacts.map((artifact) => ({ ...artifact })),
  };
  const sizeReports = sizes.map((size) => {
    let pieceCount = input.pattern.panels.length;
    if (input.grading && size.id !== "base") {
      try {
        pieceCount = deriveSize(input.grading, size.id).graded.document.panels.length;
      } catch {
        pieceCount = 0;
      }
    }
    return { id: size.id, label: size.label, pieceCount };
  });
  const reports: ManufacturingReports = {
    pattern: {
      pieceCount: input.pattern.panels.length,
      sizes: sizeReports,
      measurements: input.techPack.measurements,
      annotations: (input.production?.annotations ?? []).map((annotation) => ({
        id: annotation.id,
        panelId: annotation.panelId,
        note: annotation.note,
      })),
    },
    markers: markerRows,
    materials: { requirements, issues: input.bomIssues ?? [] },
    production: input.cutPlan ?? null,
    validation: dashboard.validation,
  };
  if (input.run && !input.cutPlan) {
    warnings.push("production run has no calculated cut plan");
    dashboard.validation.warnings = warnings;
    dashboard.validation.status = dashboard.validation.errors.length ? "errors" : "warnings";
    reports.validation.warnings = warnings;
    reports.validation.status = dashboard.validation.status;
  }
  return { dashboard, reports };
}
