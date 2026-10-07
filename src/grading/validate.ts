// Whole-document grading validation: shape checks, duplicate IDs, reference
// integrity and anchor resolvability. Returns all diagnostics; derivation
// runs these checks and throws on the first blocking error.
import { validateProductionSet } from "../cad/production.js";
import type { EntityId } from "../pattern/cad.js";
import { resolveAnchor } from "./anchors.js";
import { isLengthUnit } from "./units.js";
import type { GradingDiagnostic, GradingDocument, GradeRule, Vec2 } from "./types.js";

export function validateGradingDocument(document: GradingDocument): GradingDiagnostic[] {
  const diagnostics: GradingDiagnostic[] = [];
  if (!document || typeof document !== "object") {
    return [{ code: "invalid-document", message: "grading document is not an object" }];
  }
  if (document.schemaVersion !== 1) {
    return [{ code: "invalid-document", message: `unsupported grading schema version ${String(document.schemaVersion)}` }];
  }
  requireId(diagnostics, document.id, "grading document");
  requireId(diagnostics, document.master?.id, "master pattern");
  requireId(diagnostics, document.sizeSet?.id, "size set");
  requireId(diagnostics, document.ruleTable?.id, "rule table");
  if (!document.master || !document.sizeSet || !document.ruleTable || !Array.isArray(document.gradingPoints) ||
      !Array.isArray(document.seams) || !Array.isArray(document.graded)) {
    diagnostics.push({ code: "invalid-document", message: "grading document is missing required collections" });
    return diagnostics;
  }

  // Master pattern must be a usable pattern document.
  if (!document.master.document || document.master.document.schemaVersion !== 1) {
    diagnostics.push({ code: "invalid-document", message: "master pattern document is missing or has an unsupported schema" });
    return diagnostics;
  }

  // Measurement catalogue: unique IDs, known units, sane tolerances/ordering.
  const measurementDefinitionIds = new Set<EntityId>();
  for (const definition of document.sizeSet.measurementDefinitions ?? []) {
    if (!requireId(diagnostics, definition.id, "measurement definition", (id) => ({ measurementId: id }))) continue;
    if (measurementDefinitionIds.has(definition.id)) {
      diagnostics.push({ code: "duplicate-id", message: `measurement definition '${definition.id}' is duplicated`, measurementId: definition.id });
    }
    measurementDefinitionIds.add(definition.id);
    if (typeof definition.name !== "string" || definition.name.trim() === "") {
      diagnostics.push({ code: "invalid-argument", message: `measurement definition '${definition.id}' needs a non-empty name`, measurementId: definition.id });
    }
    if (!isLengthUnit(definition.unit)) {
      diagnostics.push({ code: "invalid-unit", message: `measurement definition '${definition.id}' has an unknown unit '${String(definition.unit)}'`, measurementId: definition.id });
    }
    if (definition.toleranceM !== undefined && (typeof definition.toleranceM !== "number" || !Number.isFinite(definition.toleranceM) || definition.toleranceM <= 0)) {
      diagnostics.push({ code: "invalid-measurement", message: `measurement definition '${definition.id}' tolerance must be a positive finite number`, measurementId: definition.id });
    }
    if (definition.ordering !== undefined && definition.ordering !== "none" && definition.ordering !== "increasing" && definition.ordering !== "decreasing") {
      diagnostics.push({ code: "invalid-argument", message: `measurement definition '${definition.id}' ordering must be 'none', 'increasing' or 'decreasing'`, measurementId: definition.id });
    }
  }

  // Size set: unique IDs, non-empty labels, structured measurements, base size.
  const sizeIds = new Set<EntityId>();
  const labels = new Map<string, EntityId>();
  for (const size of document.sizeSet.sizes ?? []) {
    if (!requireId(diagnostics, size.id, "size")) continue;
    if (sizeIds.has(size.id)) {
      diagnostics.push({ code: "duplicate-id", message: `size '${size.id}' is duplicated`, sizeId: size.id });
    }
    sizeIds.add(size.id);
    if (typeof size.label !== "string" || size.label.trim() === "") {
      diagnostics.push({ code: "invalid-argument", message: `size '${size.id}' needs a non-empty label`, sizeId: size.id });
    } else {
      const owner = labels.get(size.label);
      if (owner !== undefined) {
        diagnostics.push({ code: "duplicate-id", message: `size label '${size.label}' is shared by '${owner}' and '${size.id}'`, sizeId: size.id });
      } else labels.set(size.label, size.id);
    }
    const seenMeasurements = new Set<EntityId>();
    for (const entry of size.measurements ?? []) {
      if (typeof entry.measurementId !== "string" || entry.measurementId.trim() === "") {
        diagnostics.push({ code: "invalid-argument", message: `size '${size.id}' has an entry with an empty measurement ID`, sizeId: size.id });
        continue;
      }
      if (seenMeasurements.has(entry.measurementId)) {
        diagnostics.push({ code: "duplicate-id", message: `size '${size.id}' records measurement '${entry.measurementId}' twice`, sizeId: size.id, measurementId: entry.measurementId });
      }
      seenMeasurements.add(entry.measurementId);
      if (!measurementDefinitionIds.has(entry.measurementId)) {
        diagnostics.push({ code: "unknown-entity", message: `size '${size.id}' references unknown measurement '${entry.measurementId}'`, sizeId: size.id, measurementId: entry.measurementId });
      }
      if (typeof entry.valueM !== "number" || !Number.isFinite(entry.valueM) || entry.valueM < 0) {
        diagnostics.push({ code: "invalid-measurement", message: `size '${size.id}' measurement '${entry.measurementId}' must be a finite non-negative value in metres`, sizeId: size.id, measurementId: entry.measurementId });
      } else if (entry.valueM === 0) {
        diagnostics.push({ code: "invalid-measurement", message: `size '${size.id}' measurement '${entry.measurementId}' is zero, which is impossible for a body measurement`, sizeId: size.id, measurementId: entry.measurementId });
      }
      if (!isLengthUnit(entry.unit)) {
        diagnostics.push({ code: "invalid-unit", message: `size '${size.id}' measurement '${entry.measurementId}' has an unknown unit '${String(entry.unit)}'`, sizeId: size.id, measurementId: entry.measurementId });
      }
      if (entry.toleranceM !== undefined && (typeof entry.toleranceM !== "number" || !Number.isFinite(entry.toleranceM) || entry.toleranceM <= 0)) {
        diagnostics.push({ code: "invalid-measurement", message: `size '${size.id}' measurement '${entry.measurementId}' tolerance must be a positive finite number`, sizeId: size.id, measurementId: entry.measurementId });
      }
    }
    // The catalogue is authoritative: every size must carry every defined measurement.
    for (const definitionId of measurementDefinitionIds) {
      if (!seenMeasurements.has(definitionId)) {
        diagnostics.push({ code: "missing-measurement", message: `size '${size.id}' is missing measurement '${definitionId}'`, sizeId: size.id, measurementId: definitionId });
      }
    }
    if (typeof size.active !== "boolean") {
      diagnostics.push({ code: "invalid-argument", message: `size '${size.id}' active flag must be boolean`, sizeId: size.id });
    }
  }

  // Expected progression of catalogued measurements across the ordered sizes.
  const activeSizes = (document.sizeSet.sizes ?? []).filter((size) => size.active === true);
  for (const definition of document.sizeSet.measurementDefinitions ?? []) {
    if (!measurementDefinitionIds.has(definition.id)) continue;
    if (definition.ordering === undefined || definition.ordering === "none") continue;
    for (let i = 1; i < activeSizes.length; i++) {
      const earlier = (activeSizes[i - 1].measurements ?? []).find((entry) => entry.measurementId === definition.id);
      const later = (activeSizes[i].measurements ?? []).find((entry) => entry.measurementId === definition.id);
      if (!earlier || !later) continue;
      const EPS_M = 1e-9;
      const ascending = later.valueM >= earlier.valueM - EPS_M;
      const descending = later.valueM <= earlier.valueM + EPS_M;
      if (definition.ordering === "increasing" && !ascending) {
        diagnostics.push({
          code: "inconsistent-ordering",
          message: `measurement '${definition.id}' decreases from size '${activeSizes[i - 1].id}' to '${activeSizes[i].id}' but is marked increasing`,
          sizeId: activeSizes[i].id,
          measurementId: definition.id,
        });
      }
      if (definition.ordering === "decreasing" && !descending) {
        diagnostics.push({
          code: "inconsistent-ordering",
          message: `measurement '${definition.id}' increases from size '${activeSizes[i - 1].id}' to '${activeSizes[i].id}' but is marked decreasing`,
          sizeId: activeSizes[i].id,
          measurementId: definition.id,
        });
      }
    }
  }
  const base = document.sizeSet.baseSizeId;
  if (typeof base !== "string" || base.trim() === "") {
    diagnostics.push({ code: "unknown-base-size", message: "size set has no base size selected" });
  } else if (!sizeIds.has(base)) {
    diagnostics.push({ code: "unknown-base-size", message: `base size '${base}' does not exist in the size set`, sizeId: base });
  }
  if ((document.sizeSet.sizes?.length ?? 0) === 0) {
    diagnostics.push({ code: "unknown-size", message: "size set contains no sizes" });
  }

  // Grading points: unique IDs and resolvable anchors.
  const gradingPointIds = new Set<EntityId>();
  const resolvedTargets = new Map<EntityId, EntityId>();
  for (const gradingPoint of document.gradingPoints) {
    if (!requireId(diagnostics, gradingPoint.id, "grading point")) continue;
    if (gradingPointIds.has(gradingPoint.id)) {
      diagnostics.push({ code: "duplicate-id", message: `grading point '${gradingPoint.id}' is duplicated`, gradingPointId: gradingPoint.id });
      continue;
    }
    gradingPointIds.add(gradingPoint.id);
    if (!gradingPoint.anchor || typeof gradingPoint.anchor.kind !== "string") {
      diagnostics.push({ code: "invalid-grading-point", message: `grading point '${gradingPoint.id}' has no valid anchor`, gradingPointId: gradingPoint.id });
      continue;
    }
    try {
      const resolution = resolveAnchor(document.master.document, gradingPoint.anchor, document.seams);
      for (const pointId of resolution.pointIds) {
        const owner = resolvedTargets.get(pointId);
        if (owner !== undefined) {
          diagnostics.push({
            code: "conflicting-anchor",
            message: `grading points '${owner}' and '${gradingPoint.id}' both displace pattern point '${pointId}'`,
            gradingPointId: gradingPoint.id,
          });
        } else resolvedTargets.set(pointId, gradingPoint.id);
      }
    } catch (error) {
      diagnostics.push(anchorError(gradingPoint.id, error));
    }
  }

  // Rules: unique IDs, one per grading point, known targets, finite deltas.
  const ruleIds = new Set<EntityId>();
  const ruleOwners = new Set<EntityId>();
  for (const rule of document.ruleTable.rules ?? []) {
    if (!requireId(diagnostics, rule.id, "rule", (id) => ({ ruleId: id }))) continue;
    if (ruleIds.has(rule.id)) {
      diagnostics.push({ code: "duplicate-id", message: `rule '${rule.id}' is duplicated`, ruleId: rule.id });
      continue;
    }
    ruleIds.add(rule.id);
    if (!gradingPointIds.has(rule.gradingPointId)) {
      diagnostics.push({ code: "unknown-entity", message: `rule '${rule.id}' references unknown grading point '${rule.gradingPointId}'`, ruleId: rule.id });
    } else if (ruleOwners.has(rule.gradingPointId)) {
      diagnostics.push({ code: "duplicate-rule", message: `grading point '${rule.gradingPointId}' has more than one rule`, ruleId: rule.id });
    }
    ruleOwners.add(rule.gradingPointId);
    if (rule.mode !== "per-size" && rule.mode !== "transition") {
      diagnostics.push({ code: "invalid-rule", message: `rule '${rule.id}' has unknown mode '${String(rule.mode)}'`, ruleId: rule.id });
    }
    for (const [sizeId, delta] of Object.entries(rule.deltas ?? {})) {
      if (!sizeIds.has(sizeId)) {
        diagnostics.push({ code: "unknown-size", message: `rule '${rule.id}' has a delta for unknown size '${sizeId}'`, ruleId: rule.id, sizeId });
      }
      if (!isDelta(delta)) {
        diagnostics.push({ code: "invalid-rule", message: `rule '${rule.id}' delta for size '${sizeId}' must be a finite [dx, dy] pair`, ruleId: rule.id, sizeId });
      }
    }
  }

  // Panel adjustments: unique IDs, unique (panel, size), known references.
  const panelIds = new Set<EntityId>(document.master.document.panels.map((panel) => panel.id));
  const panelAdjustments = document.ruleTable.panelAdjustments ?? [];
  const adjustmentIds = new Set<EntityId>();
  const adjustmentPairs = new Set<string>();
  for (const adjustment of panelAdjustments) {
    if (!requireId(diagnostics, adjustment.id, "panel adjustment", (id) => ({ ruleId: id }))) continue;
    if (adjustmentIds.has(adjustment.id)) {
      diagnostics.push({ code: "duplicate-id", message: `panel adjustment '${adjustment.id}' is duplicated`, ruleId: adjustment.id });
    }
    adjustmentIds.add(adjustment.id);
    if (!panelIds.has(adjustment.panelId)) {
      diagnostics.push({ code: "unknown-entity", message: `panel adjustment '${adjustment.id}' references unknown panel '${adjustment.panelId}'`, ruleId: adjustment.id, panelId: adjustment.panelId });
    }
    if (!sizeIds.has(adjustment.sizeId)) {
      diagnostics.push({ code: "unknown-size", message: `panel adjustment '${adjustment.id}' references unknown size '${adjustment.sizeId}'`, ruleId: adjustment.id, sizeId: adjustment.sizeId });
    }
    if (!isDelta(adjustment.delta)) {
      diagnostics.push({ code: "invalid-rule", message: `panel adjustment '${adjustment.id}' delta must be a finite [dx, dy] pair`, ruleId: adjustment.id });
    }
    const pairKey = `${adjustment.panelId}\u0000${adjustment.sizeId}`;
    if (adjustmentPairs.has(pairKey)) {
      diagnostics.push({ code: "conflicting-rule", message: `panel '${adjustment.panelId}' has more than one adjustment for size '${adjustment.sizeId}'`, ruleId: adjustment.id, panelId: adjustment.panelId, sizeId: adjustment.sizeId });
    } else adjustmentPairs.add(pairKey);
  }

  // Mirror bindings: unique IDs, known points, no rules on targets, no chains.
  const mirrors = document.ruleTable.mirrors ?? [];
  const mirrorTargetIds = new Set<EntityId>();
  const mirrorIds = new Set<EntityId>();
  for (const mirror of mirrors) {
    if (!requireId(diagnostics, mirror.id, "mirror binding", (id) => ({ ruleId: id }))) continue;
    if (mirrorIds.has(mirror.id)) {
      diagnostics.push({ code: "duplicate-id", message: `mirror binding '${mirror.id}' is duplicated`, ruleId: mirror.id });
    }
    mirrorIds.add(mirror.id);
    if (mirror.axis !== "x" && mirror.axis !== "y") {
      diagnostics.push({ code: "invalid-rule", message: `mirror binding '${mirror.id}' axis must be 'x' or 'y'`, ruleId: mirror.id });
    }
    if (!gradingPointIds.has(mirror.gradingPointId)) {
      diagnostics.push({ code: "unknown-entity", message: `mirror binding '${mirror.id}' references unknown grading point '${mirror.gradingPointId}'`, ruleId: mirror.id });
    } else {
      mirrorTargetIds.add(mirror.gradingPointId);
      if (ruleOwners.has(mirror.gradingPointId)) {
        diagnostics.push({ code: "conflicting-rule", message: `grading point '${mirror.gradingPointId}' carries a rule and is also mirror-bound`, ruleId: mirror.id, gradingPointId: mirror.gradingPointId });
      }
    }
    if (!gradingPointIds.has(mirror.sourceGradingPointId)) {
      diagnostics.push({ code: "unknown-entity", message: `mirror binding '${mirror.id}' references unknown source grading point '${mirror.sourceGradingPointId}'`, ruleId: mirror.id });
    } else if (!ruleOwners.has(mirror.sourceGradingPointId)) {
      diagnostics.push({ code: "invalid-rule", message: `mirror binding '${mirror.id}' source grading point '${mirror.sourceGradingPointId}' has no rule to mirror`, ruleId: mirror.id });
    }
    if (mirror.gradingPointId === mirror.sourceGradingPointId) {
      diagnostics.push({ code: "conflicting-rule", message: `mirror binding '${mirror.id}' cannot mirror itself`, ruleId: mirror.id });
    }
  }
  for (const mirror of mirrors) {
    if (mirrorTargetIds.has(mirror.sourceGradingPointId)) {
      diagnostics.push({ code: "conflicting-rule", message: `mirror chains are ambiguous; grading point '${mirror.sourceGradingPointId}' is itself mirrored`, ruleId: mirror.id });
    }
  }

  // Production sidecar + marking anchors.
  if (document.production !== undefined) {
    for (const issue of validateProductionSet(document.master.document, document.production)) {
      diagnostics.push({
        code: issue.code === "missing-reference" ? "unknown-entity" : issue.code === "duplicate-id" ? "duplicate-id" : "invalid-document",
        message: `production set: ${issue.message}`,
      });
    }
  } else if ((document.markingAnchors ?? []).length > 0) {
    diagnostics.push({ code: "invalid-document", message: "marking anchors exist but the document has no production set" });
  }
  const markingAnchorIds = new Set<EntityId>();
  for (const anchor of document.markingAnchors ?? []) {
    if (!requireId(diagnostics, anchor.id, "marking anchor", (id) => ({ ruleId: id }))) continue;
    if (markingAnchorIds.has(anchor.id)) {
      diagnostics.push({ code: "duplicate-id", message: `marking anchor '${anchor.id}' is duplicated`, ruleId: anchor.id });
    }
    markingAnchorIds.add(anchor.id);
    if (!document.production) continue;
    if (!gradingPointIds.has(anchor.gradingPointId)) {
      diagnostics.push({ code: "unknown-entity", message: `marking anchor '${anchor.id}' references unknown grading point '${anchor.gradingPointId}'`, ruleId: anchor.id });
    }
    const collections: Record<string, Array<{ id: EntityId; panelId: EntityId }>> = {
      grainline: document.production.grainlines,
      fold: document.production.folds,
      drill: document.production.drills,
      annotation: document.production.annotations,
      labelRegion: document.production.labelRegions,
    };
    const entity = collections[anchor.kind]?.find((candidate) => candidate.id === anchor.entityId);
    if (!entity) {
      diagnostics.push({ code: "unknown-entity", message: `marking anchor '${anchor.id}' references unknown ${anchor.kind} '${anchor.entityId}'`, ruleId: anchor.id });
    } else if (entity.panelId !== anchor.panelId) {
      diagnostics.push({ code: "unknown-entity", message: `marking anchor '${anchor.id}' panel '${anchor.panelId}' does not match the entity panel '${entity.panelId}'`, ruleId: anchor.id });
    }
  }

  // Derived cache: unique size IDs, self-consistent identity.
  const gradedSizeIds = new Set<EntityId>();
  for (const graded of document.graded) {
    if (!requireId(diagnostics, graded.id, "graded pattern")) continue;
    if (gradedSizeIds.has(graded.sizeId)) {
      diagnostics.push({ code: "invalid-document", message: `derived cache holds multiple patterns for size '${graded.sizeId}'`, sizeId: graded.sizeId });
    }
    gradedSizeIds.add(graded.sizeId);
    if (graded.masterId !== document.master.id) {
      diagnostics.push({ code: "invalid-document", message: `derived pattern '${graded.id}' references foreign master '${graded.masterId}'` });
    }
    if (!graded.document || graded.document.id !== document.master.document.id) {
      diagnostics.push({ code: "invalid-document", message: `derived pattern '${graded.id}' does not carry the master document ID` });
    }
  }

  return diagnostics;
}

function isDelta(value: unknown): value is Vec2 {
  return Array.isArray(value) && value.length === 2 &&
    typeof value[0] === "number" && Number.isFinite(value[0]) &&
    typeof value[1] === "number" && Number.isFinite(value[1]);
}

function requireId(
  diagnostics: GradingDiagnostic[],
  id: unknown,
  label: string,
  extra?: (id: EntityId) => Partial<GradingDiagnostic>,
): id is EntityId {
  if (typeof id !== "string" || id.trim() === "") {
    diagnostics.push({ code: "invalid-argument", message: `${label} needs a non-empty stable ID`, ...extra?.(id as EntityId) });
    return false;
  }
  return true;
}

function anchorError(gradingPointId: EntityId, error: unknown): GradingDiagnostic {
  if (error instanceof Error && "code" in error) {
    const code = (error as { code: string }).code as GradingDiagnostic["code"];
    if (code === "unknown-entity" || code === "invalid-grading-point" || code === "invalid-argument") {
      return { code, message: `grading point '${gradingPointId}': ${error.message}`, gradingPointId };
    }
  }
  return { code: "invalid-grading-point", message: `grading point '${gradingPointId}': ${error instanceof Error ? error.message : String(error)}`, gradingPointId };
}
