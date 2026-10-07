// G14A — marker & nesting data model.
//
// A marker is a manufacturing document derived from (never mixed into) the
// authoritative pattern:
//
//   Graded Pattern -> CutPlan -> Marker -> NestingResult
//
// Quantities live in the cut plan, never in the base pattern. Placements
// reference stable panel/size ids. All geometry here is in fabric metres:
// X across the usable width, Y along the marker length.

import {
  PatternCadError,
  type EntityId,
  type PatternDocument,
} from "../pattern/cad.js";
import type { Vec2 } from "../cad/geom.js";
import { sampleLoopLocal } from "../cad/queries.js";
import { deriveSize } from "../grading/derive.js";
import type { GradingDocument } from "../grading/types.js";

export const MARKER_SCHEMA_VERSION = 1;

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

// ---------------------------------------------------------------------------
// Fabric
// ---------------------------------------------------------------------------

export interface Fabric {
  id: string;
  name: string;
  /** Full physical width in metres (> 0). */
  widthM: number;
  /** Usable width in metres (> 0, <= width). */
  usableWidthM: number;
  /** Roll length in metres (> 0). Informational; markers may be shorter. */
  lengthM: number;
  materialType: string;
  notes?: string;
}

export interface FabricRoll extends Fabric {
  rollId: string;
}

export function createFabric(partial: {
  id: string;
  name: string;
  widthM: number;
  usableWidthM?: number;
  lengthM: number;
  materialType: string;
  notes?: string;
}): Fabric {
  const { id, name, widthM, lengthM, materialType } = partial;
  if (!id || !name || !materialType) {
    throw new PatternCadError("invalid-document", "fabric needs id, name, and material type");
  }
  if (!(widthM > 0) || !(lengthM > 0)) {
    throw new PatternCadError("invalid-transform", "fabric width and length must be positive", id);
  }
  const usableWidthM = partial.usableWidthM ?? widthM;
  if (!(usableWidthM > 0) || usableWidthM > widthM) {
    throw new PatternCadError("invalid-transform", "usable width must be positive and <= physical width", id);
  }
  return {
    id, name, widthM, usableWidthM, lengthM, materialType,
    ...(partial.notes ? { notes: partial.notes } : {}),
  };
}

// ---------------------------------------------------------------------------
// Cut plan
// ---------------------------------------------------------------------------

export type MirrorPolicy = "required" | "allowed" | "forbidden";

export interface CutItem {
  id: string;
  panelId: EntityId;
  sizeId: EntityId;
  quantity: number;
  mirror: MirrorPolicy;
  materialId?: string;
  /** Per-item rotation override in degrees (resolved against fabric rules). */
  rotationsDeg?: number[];
  note?: string;
}

export interface CutPlan {
  id: string;
  name: string;
  gradingId: string;
  items: CutItem[];
}

export function createCutPlan(id: string, name: string, gradingId: string): CutPlan {
  if (!id || !name || !gradingId) {
    throw new PatternCadError("invalid-document", "cut plan needs id, name, and grading reference");
  }
  return { id, name, gradingId, items: [] };
}

export function addCutItem(plan: CutPlan, item: Omit<CutItem, "id"> & { id?: string }): { plan: CutPlan; id: string } {
  const next = clone(plan);
  const id = item.id ?? `${plan.id}/item/${String(next.items.length + 1).padStart(4, "0")}`;
  if (next.items.some((i) => i.id === id)) {
    throw new PatternCadError("duplicate-id", `cut item '${id}' is duplicated`, id);
  }
  if (!item.panelId || !item.sizeId) {
    throw new PatternCadError("missing-reference", "cut item needs panel and size references", id);
  }
  if (!Number.isInteger(item.quantity) || item.quantity < 1) {
    throw new PatternCadError("invalid-transform", "cut quantity must be a positive integer", id);
  }
  if (item.mirror !== "required" && item.mirror !== "allowed" && item.mirror !== "forbidden") {
    throw new PatternCadError("invalid-transform", `unknown mirror policy '${String(item.mirror)}'`, id);
  }
  if (item.rotationsDeg !== undefined) {
    if (!Array.isArray(item.rotationsDeg) || item.rotationsDeg.length === 0 ||
      !item.rotationsDeg.every((r) => Number.isFinite(r))) {
      throw new PatternCadError("invalid-transform", "rotation override must be a non-empty finite list", id);
    }
  }
  next.items.push({ ...clone(item), id } as CutItem);
  return { plan: next, id };
}

/** Total instances (sum of quantities). */
export function cutPlanInstances(plan: CutPlan): number {
  return plan.items.reduce((s, i) => s + i.quantity, 0);
}

// ---------------------------------------------------------------------------
// Marker pieces (resolved geometry, still unplaced)
// ---------------------------------------------------------------------------

export interface MarkerPiece {
  /** Stable instance id: `${cutItemId}#${index}` (+ `/m` when mirrored). */
  instanceId: string;
  cutItemId: string;
  panelId: EntityId;
  sizeId: EntityId;
  mirrored: boolean;
  /** Outer boundary samples in piece-local metres (grading-derived). */
  polygon: Vec2[];
  areaM2: number;
  /** Grain direction in piece-local radians (from panel metadata). */
  grainRad: number;
}

/**
 * Expand a cut plan against a grading document into concrete piece
 * instances. Mirror-required items produce mirrored copies (tracked in the
 * instance id); mirror-allowed items produce one as-authored instance and
 * the engine may also try the mirrored variant.
 */
export function expandCutPlan(grading: GradingDocument, plan: CutPlan): MarkerPiece[] {
  if (plan.gradingId !== grading.id) {
    throw new PatternCadError(
      "missing-reference",
      `cut plan references grading '${plan.gradingId}', not '${grading.id}'`,
      plan.id,
    );
  }
  const pieces: MarkerPiece[] = [];
  for (const item of plan.items) {
    const { graded } = deriveSize(grading, item.sizeId);
    const panel = graded.document.panels.find((p) => p.id === item.panelId);
    if (!panel) {
      throw new PatternCadError(
        "missing-reference",
        `cut item '${item.id}' references missing panel '${item.panelId}'`,
        item.id,
      );
    }
    const outer = panel.boundaryLoops.find((l) => l.role === "outer");
    if (!outer) {
      throw new PatternCadError("open-boundary", `panel '${item.panelId}' has no outer loop`, item.panelId);
    }
    const polygon = sampleLoopLocal(graded.document, panel.id, outer.id);
    if (polygon.length < 3) {
      throw new PatternCadError("degenerate-panel", `panel '${item.panelId}' samples to nothing`, item.panelId);
    }
    const base: Omit<MarkerPiece, "instanceId" | "mirrored" | "polygon"> = {
      cutItemId: item.id,
      panelId: item.panelId,
      sizeId: item.sizeId,
      areaM2: Math.abs(signedAreaOf(polygon)),
      grainRad: panel.grainAngleRad,
    };
    for (let k = 0; k < item.quantity; k++) {
      pieces.push({ ...clone(base), instanceId: `${item.id}#${k + 1}`, mirrored: false, polygon: clone(polygon) });
      if (item.mirror === "required") {
        pieces.push({
          ...clone(base), instanceId: `${item.id}#${k + 1}/m`, mirrored: true,
          polygon: mirrorPolygon(polygon),
        });
      }
    }
  }
  return pieces;
}

function signedAreaOf(polygon: Vec2[]): number {
  let s = 0;
  for (let i = 0; i < polygon.length; i++) {
    const p = polygon[i], q = polygon[(i + 1) % polygon.length];
    s += p[0] * q[1] - q[0] * p[1];
  }
  return 0.5 * s;
}

function mirrorPolygon(polygon: Vec2[]): Vec2[] {
  return polygon.map((p) => [-p[0], p[1]] as Vec2);
}

// ---------------------------------------------------------------------------
// Constraints / placements / marker
// ---------------------------------------------------------------------------

export interface NestingConstraint {
  /** Piece spacing (cut gap) in metres (>= 0). */
  spacingM: number;
  /** Bottom + side margin inside the usable width (m, >= 0). */
  marginM: number;
  /** Default allowed rotations in degrees (discrete). */
  rotationsDeg: number[];
  /** Default mirror policy when the cut item allows choice. */
  mirrorDefault: "mirrored" | "as-authored";
  /** Grain rule: pieces must keep grain within this angle of fabric +Y (deg, 0 = off). */
  grainToleranceDeg: number;
  /** Fabric is directional (nap/print): only 0°/180°-class rotations may apply. */
  directionalFabric: boolean;
}

export function defaultNestingConstraint(overrides: Partial<NestingConstraint> = {}): NestingConstraint {
  const base: NestingConstraint = {
    spacingM: 0.01,
    marginM: 0.01,
    rotationsDeg: [0, 180],
    mirrorDefault: "as-authored",
    grainToleranceDeg: 0,
    directionalFabric: false,
  };
  const merged = { ...base, ...overrides };
  validateNestingConstraint(merged);
  return merged;
}

export function validateNestingConstraint(constraint: NestingConstraint): void {
  if (!(constraint.spacingM >= 0) || !Number.isFinite(constraint.spacingM)) {
    throw new PatternCadError("invalid-transform", "spacing must be finite and >= 0");
  }
  if (!(constraint.marginM >= 0) || !Number.isFinite(constraint.marginM)) {
    throw new PatternCadError("invalid-transform", "margin must be finite and >= 0");
  }
  if (!Array.isArray(constraint.rotationsDeg) || constraint.rotationsDeg.length === 0 ||
    !constraint.rotationsDeg.every((r) => Number.isFinite(r))) {
    throw new PatternCadError("invalid-transform", "rotations must be a non-empty finite list");
  }
  if (constraint.mirrorDefault !== "mirrored" && constraint.mirrorDefault !== "as-authored") {
    throw new PatternCadError("invalid-transform", "unknown mirror default");
  }
  if (!(constraint.grainToleranceDeg >= 0) || !Number.isFinite(constraint.grainToleranceDeg)) {
    throw new PatternCadError("invalid-transform", "grain tolerance must be finite and >= 0");
  }
}

export interface Placement {
  instanceId: string;
  /** Translation in fabric metres. */
  x: number;
  y: number;
  /** Applied rotation in degrees (one of the allowed set). */
  rotationDeg: number;
  mirrored: boolean;
  /** True when placed by hand rather than the engine. */
  manual: boolean;
}

export interface Marker {
  schemaVersion: typeof MARKER_SCHEMA_VERSION;
  id: string;
  name: string;
  fabric: Fabric;
  cutPlanId: string;
  constraint: NestingConstraint;
  /** Piece instances in deterministic order (cut-plan order). */
  pieces: MarkerPiece[];
  /** Placements keyed by instance id (absent = unplaced). */
  placements: Placement[];
  revision: number;
}

export function createMarker(
  id: string,
  name: string,
  fabric: Fabric,
  cutPlanId: string,
  constraint: NestingConstraint,
  pieces: MarkerPiece[],
): Marker {
  if (!id || !name || !cutPlanId) {
    throw new PatternCadError("invalid-document", "marker needs id, name, and cut-plan reference");
  }
  const seen = new Set<string>();
  for (const piece of pieces) {
    if (seen.has(piece.instanceId)) {
      throw new PatternCadError("duplicate-id", `duplicate piece instance '${piece.instanceId}'`, piece.instanceId);
    }
    seen.add(piece.instanceId);
  }
  return {
    schemaVersion: MARKER_SCHEMA_VERSION,
    id, name, fabric: clone(fabric), cutPlanId,
    constraint: clone(constraint), pieces: clone(pieces), placements: [], revision: 1,
  };
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

export type MarkerDiagnosticCode =
  | "invalid-document"
  | "missing-reference"
  | "duplicate-id"
  | "overlap"
  | "outside-fabric"
  | "spacing-violation"
  | "invalid-orientation"
  | "invalid-geometry"
  | "missing-quantity";

export interface MarkerDiagnostic {
  code: MarkerDiagnosticCode;
  message: string;
  entityId?: string;
}

export function validateMarker(marker: Marker): MarkerDiagnostic[] {
  const diagnostics: MarkerDiagnostic[] = [];
  const fail = (code: MarkerDiagnosticCode, message: string, entityId?: string): void => {
    diagnostics.push({ code, message, ...(entityId ? { entityId } : {}) });
  };
  if (!marker || typeof marker !== "object" || marker.schemaVersion !== MARKER_SCHEMA_VERSION) {
    return [{ code: "invalid-document", message: "marker shape or schema version is invalid" }];
  }
  try {
    createFabric(marker.fabric);
  } catch (error) {
    fail("invalid-geometry", error instanceof Error ? error.message : String(error), marker.fabric?.id);
  }
  try {
    validateNestingConstraint(marker.constraint);
  } catch (error) {
    fail("invalid-geometry", error instanceof Error ? error.message : String(error));
  }
  if (!Array.isArray(marker.pieces) || marker.pieces.length === 0) {
    fail("missing-quantity", "marker has no piece instances");
    return diagnostics;
  }
  const pieceIds = new Set(marker.pieces.map((p) => p.instanceId));
  if (pieceIds.size !== marker.pieces.length) {
    fail("duplicate-id", "duplicate piece instance ids");
  }
  for (const piece of marker.pieces) {
    if (!Array.isArray(piece.polygon) || piece.polygon.length < 3) {
      fail("invalid-geometry", `piece '${piece.instanceId}' has no usable polygon`, piece.instanceId);
    }
    if (!(piece.areaM2 > 0) || !Number.isFinite(piece.areaM2)) {
      fail("invalid-geometry", `piece '${piece.instanceId}' has invalid area`, piece.instanceId);
    }
  }
  const seenPlacement = new Set<string>();
  for (const placement of marker.placements ?? []) {
    if (!pieceIds.has(placement.instanceId)) {
      fail("missing-reference", `placement references unknown piece '${placement.instanceId}'`, placement.instanceId);
      continue;
    }
    if (seenPlacement.has(placement.instanceId)) {
      fail("duplicate-id", `duplicate placement for '${placement.instanceId}'`, placement.instanceId);
    }
    seenPlacement.add(placement.instanceId);
    if (!Number.isFinite(placement.x) || !Number.isFinite(placement.y) || !Number.isFinite(placement.rotationDeg)) {
      fail("invalid-geometry", `placement for '${placement.instanceId}' is non-finite`, placement.instanceId);
    }
  }
  return diagnostics;
}

// ---------------------------------------------------------------------------
// Persistence (canonical JSON)
// ---------------------------------------------------------------------------

function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (value !== null && typeof value === "object") {
    const record = value as Record<string, unknown>;
    return `{${Object.keys(record).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(record[key])}`).join(",")}}`;
  }
  if (typeof value === "number" && Object.is(value, -0)) return "0";
  return JSON.stringify(value);
}

export function serializeMarker(marker: Marker): string {
  return canonicalJson(marker);
}

export function deserializeMarker(serialized: string): Marker {
  let parsed: unknown;
  try {
    parsed = JSON.parse(serialized);
  } catch {
    throw new PatternCadError("invalid-document", "serialized marker is not valid JSON");
  }
  const marker = parsed as Marker;
  const diagnostics = validateMarker(marker);
  if (diagnostics.length > 0) {
    throw new PatternCadError(
      "invalid-document",
      `invalid marker: ${diagnostics.map((d) => `${d.code}: ${d.message}`).join("; ")}`,
    );
  }
  return clone(marker);
}

export function serializeCutPlan(plan: CutPlan): string {
  return canonicalJson(plan);
}

export function deserializeCutPlan(serialized: string): CutPlan {
  let parsed: unknown;
  try {
    parsed = JSON.parse(serialized);
  } catch {
    throw new PatternCadError("invalid-document", "serialized cut plan is not valid JSON");
  }
  const plan = parsed as CutPlan;
  if (!plan || typeof plan !== "object" || !plan.id || !Array.isArray(plan.items)) {
    throw new PatternCadError("invalid-document", "cut plan shape is invalid");
  }
  return clone(plan);
}
