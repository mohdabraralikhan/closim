// G12A grading data model. The master pattern is the source of truth; sizes,
// rules and graded documents are derived, persistent, stable-ID structures.
// Deltas are expressed in the referenced panel's local pattern coordinates
// (the same space PatternPoint.x/y live in), matching CAD movePoint semantics.
import type { EntityId, PatternDocument } from "../pattern/cad.js";
import type { Seam } from "../garment/sewing.js";
import type { ProductionSet } from "../cad/production.js";
import type { LengthUnit } from "./units.js";

export type Vec2 = [number, number];

/**
 * A reusable measurement in the size system catalogue ("chest", "waist", ...).
 * Definitions are shared across sizes; per-size values live on SizeDefinition.
 * Defining a measurement never implies a pattern delta — grading rules remain
 * authoritative for geometry.
 */
export interface MeasurementDefinition {
  id: EntityId;
  name: string;
  description?: string;
  /** Unit values are entered/displayed in; storage is always canonical metres. */
  unit: LengthUnit;
  /** Source/type tag ("body", "block", "imported", ...). */
  type: string;
  /** Optional tolerance in canonical metres. */
  toleranceM?: number;
  /** Expected value progression across the ordered active sizes ("none" = unchecked). */
  ordering: "none" | "increasing" | "decreasing";
}

/** One measurement value recorded on one size. */
export interface SizeMeasurement {
  /** References a MeasurementDefinition in the size set. */
  measurementId: EntityId;
  /** Canonical metres. */
  valueM: number;
  /** Unit the value was entered in (display fidelity; valueM is canonical). */
  unit: LengthUnit;
  /** Optional provenance ("body-scan", "size-chart", ...). */
  source?: string;
  /** Optional tolerance in canonical metres; overrides the definition's. */
  toleranceM?: number;
}

export interface SizeDefinition {
  id: EntityId;
  /** Short textual/numeric label ("M", "42", any market convention). */
  label: string;
  displayName: string;
  /** Body-measurement values by stable measurement ID, canonical units (metres). */
  measurements: SizeMeasurement[];
  active: boolean;
}

export interface SizeSet {
  id: EntityId;
  name: string;
  /** Set to "" until the first size is added; must resolve once sizes exist. */
  baseSizeId: EntityId;
  /** Array order is the size ordering; identity is the size ID, never the index. */
  sizes: SizeDefinition[];
  /** Reusable measurement catalogue; definitions never imply grading deltas. */
  measurementDefinitions: MeasurementDefinition[];
}

export interface MasterPattern {
  id: EntityId;
  name: string;
  document: PatternDocument;
}

export type GradingAnchor =
  | { kind: "point"; panelId: EntityId; pointId: EntityId }
  | { kind: "corner"; panelId: EntityId; segmentIdA: EntityId; segmentIdB: EntityId }
  | { kind: "edge-relative"; panelId: EntityId; segmentId: EntityId; t: number }
  | { kind: "seam-point"; seamId: EntityId; side: "a" | "b"; segmentId: EntityId; endpoint: "start" | "end" };

export interface GradingPoint {
  id: EntityId;
  anchor: GradingAnchor;
  note?: string;
}

export type RuleMode = "per-size" | "transition";

export interface GradeRule {
  id: EntityId;
  gradingPointId: EntityId;
  /**
   * "per-size": deltas[sizeId] is the master→size delta.
   * "transition": deltas[sizeId] is the delta from the previous active size
   * (or from the master for the first active size) to sizeId, accumulated in
   * size-set order.
   */
  mode: RuleMode;
  deltas: Record<EntityId, Vec2>;
}

export interface GradeRuleTable {
  id: EntityId;
  name: string;
  rules: GradeRule[];
  /**
   * Panel-wide extra deltas composed ON TOP of point rules. Precedence is
   * explicit: effective delta = point rule delta (or mirrored source rule
   * delta) + panel adjustments for (anchor panel, size). At most one
   * adjustment per (panel, size); duplicates are rejected as ambiguous.
   */
  panelAdjustments: PanelAdjustment[];
  /**
   * Mirrored grading points inherit their source point's rule with one
   * component negated (mirrored panels). A mirrored point must not carry its
   * own rule; a mirror source must not itself be mirrored (no chains).
   */
  mirrors: MirrorBinding[];
}

export interface PanelAdjustment {
  id: EntityId;
  panelId: EntityId;
  sizeId: EntityId;
  /** Panel-local extra [dx, dy] added to every grading point anchored on this panel for this size. */
  delta: Vec2;
}

export type MirrorAxis = "x" | "y";

export interface MirrorBinding {
  id: EntityId;
  /** Grading point receiving mirrored deltas; must not carry its own rule. */
  gradingPointId: EntityId;
  /** Grading point whose rule is mirrored; must have a rule and must not itself be mirrored. */
  sourceGradingPointId: EntityId;
  /** "x": mirror across the panel-local X axis (negates dy); "y": negates dx. */
  axis: MirrorAxis;
}

export interface GradedPattern {
  id: EntityId;
  masterId: EntityId;
  sizeId: EntityId;
  /** Derived document; keeps the master's documentId so entity IDs stay stable across sizes. */
  document: PatternDocument;
  /** Fingerprint of (master, size set, rules, grading points, seams) at derivation time. */
  sourceFingerprint: string;
}

export interface GradingDocument {
  schemaVersion: 1;
  id: EntityId;
  name: string;
  master: MasterPattern;
  sizeSet: SizeSet;
  ruleTable: GradeRuleTable;
  gradingPoints: GradingPoint[];
  /** Seam context required to resolve seam-point anchors. */
  seams: Seam[];
  /**
   * Optional G11 production sidecar authored against the MASTER geometry.
   * Per-size production sets are derived on demand (deriveProductionSet) and
   * never cached. T/ID-referenced entities (allowances, notches, cut lines,
   * panel meta) copy verbatim; position-based markings copy verbatim unless a
   * marking anchor binds them to a grading point.
   */
  production?: ProductionSet;
  /**
   * Explicit bindings displacing position-based production markings
   * (grainlines, folds, drills, annotations, label regions) by a grading
   * point's effective delta. Unbound markings are copied verbatim and
   * validated against the derived panel with explicit issues.
   */
  markingAnchors?: MarkingAnchor[];
  /** Derived-size cache; at most one entry per size ID. */
  graded: GradedPattern[];
}

export type MarkingKind = "grainline" | "fold" | "drill" | "annotation" | "labelRegion";

export interface MarkingAnchor {
  id: EntityId;
  panelId: EntityId;
  kind: MarkingKind;
  /** Stable production-entity ID inside GradingDocument.production. */
  entityId: EntityId;
  /** Grading point whose effective delta displaces the marking's positions. */
  gradingPointId: EntityId;
}

export type GradingErrorCode =
  | "invalid-argument"
  | "invalid-document"
  | "duplicate-id"
  | "duplicate-rule"
  | "conflicting-anchor"
  | "conflicting-rule"
  | "unknown-size"
  | "unknown-base-size"
  | "unknown-entity"
  | "invalid-grading-point"
  | "invalid-rule"
  | "missing-rule"
  | "stale-derivation"
  | "seam-mismatch"
  | "inconsistent-direction"
  | "outside-panel"
  | "invalid-unit"
  | "invalid-measurement"
  | "missing-measurement"
  | "inconsistent-ordering";

export class GradingError extends Error {
  readonly code: GradingErrorCode;
  readonly entityId?: EntityId;

  constructor(code: GradingErrorCode, message: string, entityId?: EntityId) {
    super(`grading (${code}): ${message}`);
    this.name = "GradingError";
    this.code = code;
    this.entityId = entityId;
  }
}

export interface GradingDiagnostic {
  code: GradingErrorCode;
  message: string;
  gradingPointId?: EntityId;
  ruleId?: EntityId;
  sizeId?: EntityId;
  panelId?: EntityId;
  measurementId?: EntityId;
}

export interface AnchorResolution {
  /** Pattern points displaced by grade rules (vertex-anchored points). */
  pointIds: EntityId[];
  /** Evaluated anchor position in panel-local pattern coordinates. */
  position: Vec2;
  /** "vertex": rule deltas move pattern points. "evaluated": position is derived from the (graded) segment; deltas apply to the reported position only. */
  displacement: "vertex" | "evaluated";
}

export interface AnchorEvaluation {
  gradingPointId: EntityId;
  anchor: GradingAnchor;
  masterPosition: Vec2;
  gradedPosition: Vec2;
  pointIds: EntityId[];
  displacement: "vertex" | "evaluated";
}

export interface RuleApplication {
  gradingPointId: EntityId;
  ruleId: EntityId;
  mode: RuleMode;
  /** Total effective delta: point rule (or mirrored source rule) + panel adjustments. */
  delta: Vec2;
  pointId: EntityId;
  masterPosition: Vec2;
  gradedPosition: Vec2;
  /** Set when the delta came from a mirrored source rule. */
  mirrorOfId?: EntityId;
  /** Panel adjustments composed into the delta, in stored order. */
  panelAdjustmentIds?: EntityId[];
}

export interface SeamCheckRow {
  seamId: EntityId;
  lengthAM: number;
  lengthBM: number;
  diffM: number;
  withinTolerance: boolean;
}

export interface ProductionIssue {
  code: "outside-panel" | "unknown-marking-anchor";
  message: string;
  /** Production entity the issue applies to (when localizable). */
  entityId?: EntityId;
  markingAnchorId?: EntityId;
}

export interface DerivedProduction {
  set: ProductionSet;
  issues: ProductionIssue[];
}

export interface GradingDerivationReport {
  sizeId: EntityId;
  applied: RuleApplication[];
  /** Anchors that contribute positions but no vertex displacement (intermediate edge-relative points). */
  evaluated: AnchorEvaluation[];
  diagnostics: GradingDiagnostic[];
  /** Post-derivation paired seam lengths (quality gate; never blocks derivation). */
  seams: SeamCheckRow[];
}
