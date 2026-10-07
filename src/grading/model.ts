// Pure construction and manipulation operations for grading documents.
// Every operation clones its input, validates deterministically and throws
// GradingError on bad references — no silent mutations, no array-position
// references.
import { validatePatternDocument, type EntityId } from "../pattern/cad.js";
import type { ProductionSet } from "../cad/production.js";
import { resolveAnchor } from "./anchors.js";
import { isLengthUnit, toMetres } from "./units.js";
import type {
  GradeRule,
  GradeRuleTable,
  GradedPattern,
  GradingAnchor,
  GradingDocument,
  GradingPoint,
  MarkingAnchor,
  MarkingKind,
  MasterPattern,
  MeasurementDefinition,
  MirrorAxis,
  MirrorBinding,
  PanelAdjustment,
  SizeDefinition,
  SizeMeasurement,
  SizeSet,
  Vec2,
} from "./types.js";
import { GradingError } from "./types.js";

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function requireNonEmptyId(id: EntityId, label: string): EntityId {
  if (typeof id !== "string" || id.trim() === "") {
    throw new GradingError("invalid-argument", `${label} needs a non-empty stable ID`);
  }
  return id;
}

// ---------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------

function validateSizeMeasurement(entry: SizeMeasurement, sizeId: EntityId): SizeMeasurement {
  requireNonEmptyId(entry.measurementId, "measurement ID");
  if (typeof entry.valueM !== "number" || !Number.isFinite(entry.valueM) || entry.valueM < 0) {
    throw new GradingError("invalid-measurement", `measurement '${entry.measurementId}' on size '${sizeId}' must be a finite non-negative value in metres`, entry.measurementId);
  }
  if (!isLengthUnit(entry.unit)) {
    throw new GradingError("invalid-unit", `measurement '${entry.measurementId}' on size '${sizeId}' has an unknown unit '${String(entry.unit)}'`, entry.measurementId);
  }
  if (entry.toleranceM !== undefined && (typeof entry.toleranceM !== "number" || !Number.isFinite(entry.toleranceM) || entry.toleranceM <= 0)) {
    throw new GradingError("invalid-measurement", `measurement '${entry.measurementId}' on size '${sizeId}' tolerance must be a positive finite number`, entry.measurementId);
  }
  const clean: SizeMeasurement = {
    measurementId: entry.measurementId,
    valueM: entry.valueM,
    unit: entry.unit,
  };
  if (entry.source !== undefined) clean.source = entry.source;
  if (entry.toleranceM !== undefined) clean.toleranceM = entry.toleranceM;
  return clean;
}

export function createSize(partial: {
  id: EntityId;
  label: string;
  displayName?: string;
  measurements?: SizeMeasurement[];
  active?: boolean;
}): SizeDefinition {
  requireNonEmptyId(partial.id, "size");
  if (typeof partial.label !== "string" || partial.label.trim() === "") {
    throw new GradingError("invalid-argument", `size '${partial.id}' needs a non-empty label`, partial.id);
  }
  const measurements = (partial.measurements ?? []).map((entry) => validateSizeMeasurement(entry, partial.id));
  return {
    id: partial.id,
    label: partial.label,
    displayName: partial.displayName ?? partial.label,
    measurements,
    active: partial.active ?? true,
  };
}

export function createSizeSet(id: EntityId, name: string): SizeSet {
  requireNonEmptyId(id, "size set");
  return { id, name, baseSizeId: "", sizes: [], measurementDefinitions: [] };
}

export function createMeasurementDefinition(partial: {
  id: EntityId;
  name: string;
  description?: string;
  unit?: MeasurementDefinition["unit"];
  type?: string;
  toleranceM?: number;
  ordering?: MeasurementDefinition["ordering"];
}): MeasurementDefinition {
  requireNonEmptyId(partial.id, "measurement definition");
  if (typeof partial.name !== "string" || partial.name.trim() === "") {
    throw new GradingError("invalid-argument", `measurement definition '${partial.id}' needs a non-empty name`, partial.id);
  }
  const unit = partial.unit ?? "m";
  if (!isLengthUnit(unit)) {
    throw new GradingError("invalid-unit", `measurement definition '${partial.id}' has an unknown unit '${String(unit)}'`, partial.id);
  }
  const ordering = partial.ordering ?? "none";
  if (ordering !== "none" && ordering !== "increasing" && ordering !== "decreasing") {
    throw new GradingError("invalid-argument", `measurement definition '${partial.id}' ordering must be 'none', 'increasing' or 'decreasing'`, partial.id);
  }
  if (partial.toleranceM !== undefined && (typeof partial.toleranceM !== "number" || !Number.isFinite(partial.toleranceM) || partial.toleranceM <= 0)) {
    throw new GradingError("invalid-measurement", `measurement definition '${partial.id}' tolerance must be a positive finite number`, partial.id);
  }
  const definition: MeasurementDefinition = {
    id: partial.id,
    name: partial.name,
    unit,
    type: partial.type ?? "body",
    ordering,
  };
  if (partial.description !== undefined) definition.description = partial.description;
  if (partial.toleranceM !== undefined) definition.toleranceM = partial.toleranceM;
  return definition;
}

export function addMeasurementDefinition(document: GradingDocument, definition: MeasurementDefinition): GradingDocument {
  const next = clone(document);
  if ((next.sizeSet.measurementDefinitions ?? []).some((candidate) => candidate.id === definition.id)) {
    throw new GradingError("duplicate-id", `measurement definition '${definition.id}' already exists`, definition.id);
  }
  next.sizeSet.measurementDefinitions.push(clone(definition));
  return next;
}

export function removeMeasurementDefinition(document: GradingDocument, measurementId: EntityId): GradingDocument {
  const next = clone(document);
  const index = (next.sizeSet.measurementDefinitions ?? []).findIndex((candidate) => candidate.id === measurementId);
  if (index < 0) throw new GradingError("unknown-entity", `measurement definition '${measurementId}' does not exist`, measurementId);
  next.sizeSet.measurementDefinitions.splice(index, 1);
  // Keep the document internally consistent: no size may keep a value for a
  // measurement that no longer exists.
  for (const size of next.sizeSet.sizes) {
    size.measurements = size.measurements.filter((entry) => entry.measurementId !== measurementId);
  }
  return next;
}

export function createMasterPattern(id: EntityId, name: string, document: MasterPattern["document"]): MasterPattern {
  requireNonEmptyId(id, "master pattern");
  const validation = validatePatternDocument(document);
  if (!validation.valid) {
    throw new GradingError("invalid-document", `master pattern document is invalid: ${validation.diagnostics.map((d) => d.code).join(", ")}`);
  }
  return { id, name, document: clone(document) };
}

export function createRuleTable(id: EntityId, name: string): GradeRuleTable {
  requireNonEmptyId(id, "rule table");
  return { id, name, rules: [], panelAdjustments: [], mirrors: [] };
}

export function createGradingPoint(id: EntityId, anchor: GradingAnchor, note?: string): GradingPoint {
  requireNonEmptyId(id, "grading point");
  if (!anchor || typeof anchor.kind !== "string") {
    throw new GradingError("invalid-grading-point", `grading point '${id}' needs a valid anchor`, id);
  }
  return note === undefined ? { id, anchor } : { id, anchor, note };
}

export function createRule(id: EntityId, gradingPointId: EntityId, mode: GradeRule["mode"], deltas: Record<EntityId, Vec2>): GradeRule {
  requireNonEmptyId(id, "rule");
  requireNonEmptyId(gradingPointId, "rule grading point");
  if (mode !== "per-size" && mode !== "transition") {
    throw new GradingError("invalid-rule", `rule '${id}' mode must be 'per-size' or 'transition'`, id);
  }
  const clean: Record<EntityId, Vec2> = {};
  for (const [sizeId, delta] of Object.entries(deltas)) {
    if (!Array.isArray(delta) || delta.length !== 2 || !delta.every((v) => Number.isFinite(v))) {
      throw new GradingError("invalid-rule", `rule '${id}' delta for size '${sizeId}' must be a finite [dx, dy] pair`, id);
    }
    clean[sizeId] = [delta[0], delta[1]];
  }
  return { id, gradingPointId, mode, deltas: clean };
}

export function createGradingDocument(partial: {
  id: EntityId;
  name: string;
  master: MasterPattern;
  sizeSet: SizeSet;
  ruleTable: GradeRuleTable;
  gradingPoints?: GradingPoint[];
  seams?: GradingDocument["seams"];
}): GradingDocument {
  requireNonEmptyId(partial.id, "grading document");
  const document: GradingDocument = {
    schemaVersion: 1,
    id: partial.id,
    name: partial.name,
    master: clone(partial.master),
    sizeSet: clone(partial.sizeSet),
    ruleTable: clone(partial.ruleTable),
    gradingPoints: clone(partial.gradingPoints ?? []),
    seams: clone(partial.seams ?? []),
    graded: [],
  };
  // Structural anchor validation against the master (catches typos early);
  // cross-grading-point conflicts are checked in validate/derive.
  const seen = new Set<EntityId>();
  for (const gradingPoint of document.gradingPoints) {
    if (seen.has(gradingPoint.id)) {
      throw new GradingError("duplicate-id", `grading point '${gradingPoint.id}' is duplicated`, gradingPoint.id);
    }
    seen.add(gradingPoint.id);
    resolveAnchor(document.master.document, gradingPoint.anchor, document.seams);
  }
  return document;
}

// ---------------------------------------------------------------------------
// Size-set operations
// ---------------------------------------------------------------------------

export function addSize(document: GradingDocument, size: SizeDefinition): GradingDocument {
  const next = clone(document);
  if (next.sizeSet.sizes.some((candidate) => candidate.id === size.id)) {
    throw new GradingError("duplicate-id", `size '${size.id}' already exists`, size.id);
  }
  next.sizeSet.sizes.push(clone(size));
  if (next.sizeSet.baseSizeId === "") next.sizeSet.baseSizeId = size.id;
  return next;
}

export function insertSize(document: GradingDocument, size: SizeDefinition, atIndex?: number): GradingDocument {
  const next = clone(document);
  if (next.sizeSet.sizes.some((candidate) => candidate.id === size.id)) {
    throw new GradingError("duplicate-id", `size '${size.id}' already exists`, size.id);
  }
  const index = atIndex ?? next.sizeSet.sizes.length;
  if (!Number.isInteger(index) || index < 0 || index > next.sizeSet.sizes.length) {
    throw new GradingError("invalid-argument", `insertion index ${String(atIndex)} is out of range`);
  }
  next.sizeSet.sizes.splice(index, 0, clone(size));
  if (next.sizeSet.baseSizeId === "") next.sizeSet.baseSizeId = size.id;
  return next;
}

export function removeSize(document: GradingDocument, sizeId: EntityId, replacementBaseSizeId?: EntityId): GradingDocument {
  const next = clone(document);
  const index = next.sizeSet.sizes.findIndex((candidate) => candidate.id === sizeId);
  if (index < 0) throw new GradingError("unknown-size", `size '${sizeId}' does not exist`, sizeId);
  if (next.sizeSet.baseSizeId === sizeId) {
    const replacement = replacementBaseSizeId ?? next.sizeSet.sizes.find((candidate) => candidate.id !== sizeId)?.id;
    if (!replacement) throw new GradingError("unknown-base-size", `removing base size '${sizeId}' requires a replacement base size`, sizeId);
    if (!next.sizeSet.sizes.some((candidate) => candidate.id === replacement)) {
      throw new GradingError("unknown-base-size", `replacement base size '${replacement}' does not exist`, replacement);
    }
    next.sizeSet.baseSizeId = replacement;
  }
  next.sizeSet.sizes.splice(index, 1);
  // Keep the document internally consistent: no rule deltas, panel
  // adjustments or derived entries may reference a size that no longer exists.
  for (const rule of next.ruleTable.rules) delete rule.deltas[sizeId];
  next.ruleTable.panelAdjustments = (next.ruleTable.panelAdjustments ?? []).filter((a) => a.sizeId !== sizeId);
  next.graded = next.graded.filter((graded) => graded.sizeId !== sizeId);
  return next;
}

export function moveSize(document: GradingDocument, sizeId: EntityId, toIndex: number): GradingDocument {
  const next = clone(document);
  const from = next.sizeSet.sizes.findIndex((candidate) => candidate.id === sizeId);
  if (from < 0) throw new GradingError("unknown-size", `size '${sizeId}' does not exist`, sizeId);
  if (!Number.isInteger(toIndex) || toIndex < 0 || toIndex >= next.sizeSet.sizes.length) {
    throw new GradingError("invalid-argument", `target index ${String(toIndex)} is out of range`);
  }
  const [size] = next.sizeSet.sizes.splice(from, 1);
  next.sizeSet.sizes.splice(toIndex, 0, size);
  return next;
}

export function setBaseSize(document: GradingDocument, sizeId: EntityId): GradingDocument {
  const next = clone(document);
  if (!next.sizeSet.sizes.some((candidate) => candidate.id === sizeId)) {
    throw new GradingError("unknown-size", `size '${sizeId}' does not exist`, sizeId);
  }
  next.sizeSet.baseSizeId = sizeId;
  return next;
}

export function setSizeActive(document: GradingDocument, sizeId: EntityId, active: boolean): GradingDocument {
  const next = clone(document);
  const size = next.sizeSet.sizes.find((candidate) => candidate.id === sizeId);
  if (!size) throw new GradingError("unknown-size", `size '${sizeId}' does not exist`, sizeId);
  size.active = active;
  return next;
}

export function assignMeasurement(
  document: GradingDocument,
  sizeId: EntityId,
  measurementId: EntityId,
  value: number,
  options: { unit?: SizeMeasurement["unit"]; source?: string; toleranceM?: number } = {},
): GradingDocument {
  requireNonEmptyId(measurementId, "measurement ID");
  const unit = options.unit ?? "m";
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new GradingError("invalid-measurement", `measurement '${measurementId}' must be a finite non-negative value`, measurementId);
  }
  const valueM = toMetres(value, unit);
  const next = clone(document);
  const size = next.sizeSet.sizes.find((candidate) => candidate.id === sizeId);
  if (!size) throw new GradingError("unknown-size", `size '${sizeId}' does not exist`, sizeId);
  const entry: SizeMeasurement = { measurementId, valueM, unit };
  if (options.source !== undefined) entry.source = options.source;
  if (options.toleranceM !== undefined) {
    if (!Number.isFinite(options.toleranceM) || options.toleranceM <= 0) {
      throw new GradingError("invalid-measurement", `measurement '${measurementId}' tolerance must be a positive finite number`, measurementId);
    }
    entry.toleranceM = options.toleranceM;
  }
  const existing = size.measurements.findIndex((candidate) => candidate.measurementId === measurementId);
  if (existing >= 0) size.measurements[existing] = entry;
  else size.measurements.push(entry);
  return next;
}

export function removeMeasurement(document: GradingDocument, sizeId: EntityId, measurementId: EntityId): GradingDocument {
  const next = clone(document);
  const size = next.sizeSet.sizes.find((candidate) => candidate.id === sizeId);
  if (!size) throw new GradingError("unknown-size", `size '${sizeId}' does not exist`, sizeId);
  const index = size.measurements.findIndex((candidate) => candidate.measurementId === measurementId);
  if (index < 0) {
    throw new GradingError("unknown-entity", `size '${sizeId}' has no measurement '${measurementId}'`, measurementId);
  }
  size.measurements.splice(index, 1);
  return next;
}

/** Duplicate a size (values included) under a new stable ID and unique label. */
export function duplicateSize(
  document: GradingDocument,
  sizeId: EntityId,
  newSizeId: EntityId,
  options: { label?: string; atIndex?: number } = {},
): GradingDocument {
  requireNonEmptyId(newSizeId, "duplicated size");
  const next = clone(document);
  const source = next.sizeSet.sizes.find((candidate) => candidate.id === sizeId);
  if (!source) throw new GradingError("unknown-size", `size '${sizeId}' does not exist`, sizeId);
  if (next.sizeSet.sizes.some((candidate) => candidate.id === newSizeId)) {
    throw new GradingError("duplicate-id", `size '${newSizeId}' already exists`, newSizeId);
  }
  const label = options.label ?? `${source.label} copy`;
  if (typeof label !== "string" || label.trim() === "") {
    throw new GradingError("invalid-argument", `duplicated size '${newSizeId}' needs a non-empty label`, newSizeId);
  }
  if (next.sizeSet.sizes.some((candidate) => candidate.label === label)) {
    throw new GradingError("duplicate-id", `size label '${label}' is already in use; pass an explicit label`, newSizeId);
  }
  const copy: SizeDefinition = { ...clone(source), id: newSizeId, label, displayName: label };
  const index = options.atIndex ?? next.sizeSet.sizes.indexOf(source) + 1;
  if (!Number.isInteger(index) || index < 0 || index > next.sizeSet.sizes.length) {
    throw new GradingError("invalid-argument", `insertion index ${String(options.atIndex)} is out of range`);
  }
  next.sizeSet.sizes.splice(index, 0, copy);
  return next;
}

export function renameSize(document: GradingDocument, sizeId: EntityId, label: string, displayName?: string): GradingDocument {
  if (typeof label !== "string" || label.trim() === "") {
    throw new GradingError("invalid-argument", `size '${sizeId}' needs a non-empty label`, sizeId);
  }
  const next = clone(document);
  const size = next.sizeSet.sizes.find((candidate) => candidate.id === sizeId);
  if (!size) throw new GradingError("unknown-size", `size '${sizeId}' does not exist`, sizeId);
  if (next.sizeSet.sizes.some((candidate) => candidate.id !== sizeId && candidate.label === label)) {
    throw new GradingError("duplicate-id", `size label '${label}' is already in use`, sizeId);
  }
  size.label = label;
  size.displayName = displayName ?? label;
  return next;
}

// ---------------------------------------------------------------------------
// Grading point + rule operations
// ---------------------------------------------------------------------------

export function addGradingPoint(document: GradingDocument, gradingPoint: GradingPoint): GradingDocument {
  const next = clone(document);
  if (next.gradingPoints.some((candidate) => candidate.id === gradingPoint.id)) {
    throw new GradingError("duplicate-id", `grading point '${gradingPoint.id}' already exists`, gradingPoint.id);
  }
  // Resolve eagerly: anchors must reference existing master entities now, so
  // later deletions surface as explicit staleness/validation errors instead
  // of silent misses.
  const resolution = resolveAnchor(next.master.document, gradingPoint.anchor, next.seams);
  for (const pointId of resolution.pointIds) {
    const conflict = next.gradingPoints.find((candidate) => {
      try {
        return resolveAnchor(next.master.document, candidate.anchor, next.seams).pointIds.includes(pointId);
      } catch {
        return false;
      }
    });
    if (conflict) {
      throw new GradingError("conflicting-anchor", `grading points '${conflict.id}' and '${gradingPoint.id}' both displace pattern point '${pointId}'`, gradingPoint.id);
    }
  }
  next.gradingPoints.push(clone(gradingPoint));
  return next;
}

export function removeGradingPoint(document: GradingDocument, gradingPointId: EntityId): GradingDocument {
  const next = clone(document);
  const index = next.gradingPoints.findIndex((candidate) => candidate.id === gradingPointId);
  if (index < 0) throw new GradingError("unknown-entity", `grading point '${gradingPointId}' does not exist`, gradingPointId);
  next.gradingPoints.splice(index, 1);
  next.ruleTable.rules = next.ruleTable.rules.filter((rule) => rule.gradingPointId !== gradingPointId);
  next.ruleTable.mirrors = (next.ruleTable.mirrors ?? []).filter((mirror) =>
    mirror.gradingPointId !== gradingPointId && mirror.sourceGradingPointId !== gradingPointId);
  next.markingAnchors = (next.markingAnchors ?? []).filter((anchor) => anchor.gradingPointId !== gradingPointId);
  return next;
}

export function addRule(document: GradingDocument, rule: GradeRule): GradingDocument {
  const next = clone(document);
  if (next.ruleTable.rules.some((candidate) => candidate.id === rule.id)) {
    throw new GradingError("duplicate-id", `rule '${rule.id}' already exists`, rule.id);
  }
  if (!next.gradingPoints.some((candidate) => candidate.id === rule.gradingPointId)) {
    throw new GradingError("unknown-entity", `rule '${rule.id}' references unknown grading point '${rule.gradingPointId}'`, rule.id);
  }
  if (next.ruleTable.rules.some((candidate) => candidate.gradingPointId === rule.gradingPointId)) {
    throw new GradingError("duplicate-rule", `grading point '${rule.gradingPointId}' already has a rule`, rule.id);
  }
  const mirror = (next.ruleTable.mirrors ?? []).find((candidate) => candidate.gradingPointId === rule.gradingPointId);
  if (mirror) {
    throw new GradingError("conflicting-rule", `grading point '${rule.gradingPointId}' is mirror-bound by '${mirror.id}' and cannot carry its own rule`, rule.id);
  }
  next.ruleTable.rules.push(clone(rule));
  return next;
}

export function removeRule(document: GradingDocument, ruleId: EntityId): GradingDocument {
  const next = clone(document);
  const index = next.ruleTable.rules.findIndex((candidate) => candidate.id === ruleId);
  if (index < 0) throw new GradingError("unknown-entity", `rule '${ruleId}' does not exist`, ruleId);
  next.ruleTable.rules.splice(index, 1);
  return next;
}

export function setRuleDelta(document: GradingDocument, ruleId: EntityId, sizeId: EntityId, delta: Vec2): GradingDocument {
  if (!Array.isArray(delta) || delta.length !== 2 || !delta.every((v) => Number.isFinite(v))) {
    throw new GradingError("invalid-rule", `delta for size '${sizeId}' must be a finite [dx, dy] pair`, ruleId);
  }
  const next = clone(document);
  const rule = next.ruleTable.rules.find((candidate) => candidate.id === ruleId);
  if (!rule) throw new GradingError("unknown-entity", `rule '${ruleId}' does not exist`, ruleId);
  if (!next.sizeSet.sizes.some((candidate) => candidate.id === sizeId)) {
    throw new GradingError("unknown-size", `size '${sizeId}' does not exist`, sizeId);
  }
  rule.deltas[sizeId] = [delta[0], delta[1]];
  return next;
}

// ---------------------------------------------------------------------------
// Panel adjustments, mirror bindings, production markings
// ---------------------------------------------------------------------------

export function createPanelAdjustment(id: EntityId, panelId: EntityId, sizeId: EntityId, delta: Vec2): PanelAdjustment {
  requireNonEmptyId(id, "panel adjustment");
  requireNonEmptyId(panelId, "panel adjustment panel");
  requireNonEmptyId(sizeId, "panel adjustment size");
  if (!Array.isArray(delta) || delta.length !== 2 || !delta.every((v) => Number.isFinite(v))) {
    throw new GradingError("invalid-rule", `panel adjustment '${id}' delta must be a finite [dx, dy] pair`, id);
  }
  return { id, panelId, sizeId, delta: [delta[0], delta[1]] };
}

export function addPanelAdjustment(document: GradingDocument, adjustment: PanelAdjustment): GradingDocument {
  const next = clone(document);
  const adjustments = next.ruleTable.panelAdjustments ?? (next.ruleTable.panelAdjustments = []);
  if (adjustments.some((candidate) => candidate.id === adjustment.id)) {
    throw new GradingError("duplicate-id", `panel adjustment '${adjustment.id}' already exists`, adjustment.id);
  }
  if (!next.master.document.panels.some((panel) => panel.id === adjustment.panelId)) {
    throw new GradingError("unknown-entity", `panel adjustment '${adjustment.id}' references unknown panel '${adjustment.panelId}'`, adjustment.id);
  }
  if (!next.sizeSet.sizes.some((candidate) => candidate.id === adjustment.sizeId)) {
    throw new GradingError("unknown-size", `panel adjustment '${adjustment.id}' references unknown size '${adjustment.sizeId}'`, adjustment.id);
  }
  if (adjustments.some((candidate) => candidate.panelId === adjustment.panelId && candidate.sizeId === adjustment.sizeId)) {
    throw new GradingError("conflicting-rule", `panel '${adjustment.panelId}' already has an adjustment for size '${adjustment.sizeId}'`, adjustment.id);
  }
  adjustments.push(clone(adjustment));
  return next;
}

export function removePanelAdjustment(document: GradingDocument, adjustmentId: EntityId): GradingDocument {
  const next = clone(document);
  const adjustments = next.ruleTable.panelAdjustments ?? [];
  const index = adjustments.findIndex((candidate) => candidate.id === adjustmentId);
  if (index < 0) throw new GradingError("unknown-entity", `panel adjustment '${adjustmentId}' does not exist`, adjustmentId);
  adjustments.splice(index, 1);
  return next;
}

export function createMirrorBinding(id: EntityId, gradingPointId: EntityId, sourceGradingPointId: EntityId, axis: MirrorAxis): MirrorBinding {
  requireNonEmptyId(id, "mirror binding");
  requireNonEmptyId(gradingPointId, "mirror binding grading point");
  requireNonEmptyId(sourceGradingPointId, "mirror binding source");
  if (axis !== "x" && axis !== "y") {
    throw new GradingError("invalid-rule", `mirror binding '${id}' axis must be 'x' or 'y'`, id);
  }
  return { id, gradingPointId, sourceGradingPointId, axis };
}

export function addMirrorBinding(document: GradingDocument, binding: MirrorBinding): GradingDocument {
  const next = clone(document);
  const mirrors = next.ruleTable.mirrors ?? (next.ruleTable.mirrors = []);
  if (mirrors.some((candidate) => candidate.id === binding.id)) {
    throw new GradingError("duplicate-id", `mirror binding '${binding.id}' already exists`, binding.id);
  }
  for (const pointId of [binding.gradingPointId, binding.sourceGradingPointId]) {
    if (!next.gradingPoints.some((candidate) => candidate.id === pointId)) {
      throw new GradingError("unknown-entity", `mirror binding '${binding.id}' references unknown grading point '${pointId}'`, binding.id);
    }
  }
  if (binding.gradingPointId === binding.sourceGradingPointId) {
    throw new GradingError("conflicting-rule", `mirror binding '${binding.id}' cannot mirror itself`, binding.id);
  }
  if (next.ruleTable.rules.some((rule) => rule.gradingPointId === binding.gradingPointId)) {
    throw new GradingError("conflicting-rule", `grading point '${binding.gradingPointId}' already carries a rule and cannot be mirror-bound`, binding.id);
  }
  if (mirrors.some((candidate) => candidate.gradingPointId === binding.gradingPointId)) {
    throw new GradingError("conflicting-rule", `grading point '${binding.gradingPointId}' is already mirror-bound`, binding.id);
  }
  if (mirrors.some((candidate) => candidate.gradingPointId === binding.sourceGradingPointId)) {
    throw new GradingError("conflicting-rule", `mirror chains are ambiguous; grading point '${binding.sourceGradingPointId}' is itself mirrored`, binding.id);
  }
  mirrors.push(clone(binding));
  return next;
}

export function removeMirrorBinding(document: GradingDocument, bindingId: EntityId): GradingDocument {
  const next = clone(document);
  const mirrors = next.ruleTable.mirrors ?? [];
  const index = mirrors.findIndex((candidate) => candidate.id === bindingId);
  if (index < 0) throw new GradingError("unknown-entity", `mirror binding '${bindingId}' does not exist`, bindingId);
  mirrors.splice(index, 1);
  return next;
}

export function assignProduction(document: GradingDocument, production: ProductionSet): GradingDocument {
  const next = clone(document);
  next.production = clone(production);
  return next;
}

const MARKING_KIND_FIELDS: Record<MarkingKind, keyof ProductionSet> = {
  grainline: "grainlines",
  fold: "folds",
  drill: "drills",
  annotation: "annotations",
  labelRegion: "labelRegions",
};

export function createMarkingAnchor(
  id: EntityId,
  panelId: EntityId,
  kind: MarkingKind,
  entityId: EntityId,
  gradingPointId: EntityId,
): MarkingAnchor {
  requireNonEmptyId(id, "marking anchor");
  requireNonEmptyId(panelId, "marking anchor panel");
  requireNonEmptyId(entityId, "marking anchor entity");
  requireNonEmptyId(gradingPointId, "marking anchor grading point");
  if (!(kind in MARKING_KIND_FIELDS)) {
    throw new GradingError("invalid-argument", `marking anchor '${id}' has unknown kind '${String(kind)}'`, id);
  }
  return { id, panelId, kind, entityId, gradingPointId };
}

export function addMarkingAnchor(document: GradingDocument, anchor: MarkingAnchor): GradingDocument {
  const next = clone(document);
  if (!next.production) {
    throw new GradingError("invalid-argument", `cannot add marking anchor '${anchor.id}': the document has no production set`, anchor.id);
  }
  const anchors = next.markingAnchors ?? (next.markingAnchors = []);
  if (anchors.some((candidate) => candidate.id === anchor.id)) {
    throw new GradingError("duplicate-id", `marking anchor '${anchor.id}' already exists`, anchor.id);
  }
  if (!next.gradingPoints.some((candidate) => candidate.id === anchor.gradingPointId)) {
    throw new GradingError("unknown-entity", `marking anchor '${anchor.id}' references unknown grading point '${anchor.gradingPointId}'`, anchor.id);
  }
  const collection = next.production[MARKING_KIND_FIELDS[anchor.kind]] as Array<{ id: EntityId; panelId: EntityId }> | undefined;
  const entity = collection?.find((candidate) => candidate.id === anchor.entityId);
  if (!entity) {
    throw new GradingError("unknown-entity", `marking anchor '${anchor.id}' references unknown ${anchor.kind} '${anchor.entityId}'`, anchor.id);
  }
  if (entity.panelId !== anchor.panelId) {
    throw new GradingError("unknown-entity", `marking anchor '${anchor.id}' panel '${anchor.panelId}' does not match the entity panel '${entity.panelId}'`, anchor.id);
  }
  anchors.push(clone(anchor));
  return next;
}

export function removeMarkingAnchor(document: GradingDocument, anchorId: EntityId): GradingDocument {
  const next = clone(document);
  const anchors = next.markingAnchors ?? [];
  const index = anchors.findIndex((candidate) => candidate.id === anchorId);
  if (index < 0) throw new GradingError("unknown-entity", `marking anchor '${anchorId}' does not exist`, anchorId);
  anchors.splice(index, 1);
  return next;
}

// ---------------------------------------------------------------------------
// Derived cache maintenance
// ---------------------------------------------------------------------------

export function replaceGradedCache(document: GradingDocument, graded: GradedPattern[]): GradingDocument {
  const next = clone(document);
  next.graded = clone(graded);
  return next;
}
