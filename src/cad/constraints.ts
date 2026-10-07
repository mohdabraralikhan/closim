// G9C — pattern constraints, measurements, and parametric editing.
//
// The kernel (src/pattern/cad.ts) owns the document model and a single
// coincident-constraint record. This module owns the *dimensional* layer:
// explicit constraint objects with stable ids, a minimum projection solver
// (distance, equal/fixed length, horizontal/vertical, parallel,
// perpendicular, angle), live measurements, and unit conversion.
//
// Design decisions:
//   - Constraints live in a sidecar ConstraintSet referencing stable kernel
//     ids — the kernel document is never edited, so G8A validation,
//     serialization, and the G8 pipeline are unaffected.
//   - Canonical internal unit is the metre (kernel frame). UI conversion
//     (formatLength/parseLengthToM) is pure and never touches storage.
//   - The solver only applies rigid-or-symmetric point moves and reports
//     residuals; contradictory constraints come back as `unsatisfied`, never
//     as silently invalid geometry.

import {
  PatternCadError,
  type EntityId,
  type PatternDocument,
  type PatternPoint,
  type PatternSegment,
} from "../pattern/cad.js";
import { resolveStitchPairs, type Seam } from "../garment/sewing.js";
import type { Vec2 } from "./geom.js";
import { getPanel, getPoint, getSegment, measurePanel } from "./queries.js";

export type LengthUnit = "mm" | "cm" | "m" | "in";

const MM_PER_UNIT: Record<LengthUnit, number> = { mm: 1, cm: 10, m: 1000, in: 25.4 };

/** UI display only: metres -> requested unit. */
export function formatLength(metres: number, unit: LengthUnit): number {
  if (!Number.isFinite(metres)) throw new PatternCadError("invalid-transform", "length must be finite");
  return (metres * 1000) / MM_PER_UNIT[unit];
}

/** UI input only: requested unit -> canonical metres. */
export function parseLengthToM(value: number, unit: LengthUnit): number {
  if (!Number.isFinite(value)) throw new PatternCadError("invalid-transform", "length must be finite");
  if (!(MM_PER_UNIT[unit] > 0)) throw new PatternCadError("invalid-transform", `unknown unit '${unit}'`);
  return (value * MM_PER_UNIT[unit]) / 1000;
}

// ---------------------------------------------------------------------------
// Constraint model
// ---------------------------------------------------------------------------

interface ConstraintBase {
  id: string;
  panelId: EntityId;
  enabled: boolean;
}

export interface DistanceConstraint extends ConstraintBase {
  kind: "distance";
  pointAId: EntityId;
  pointBId: EntityId;
  targetM: number;
}

export interface FixedLengthConstraint extends ConstraintBase {
  kind: "fixed-length";
  segmentId: EntityId;
  targetM: number;
}

export interface EqualLengthConstraint extends ConstraintBase {
  kind: "equal-length";
  segmentAId: EntityId;
  segmentBId: EntityId;
}

export interface HorizontalConstraint extends ConstraintBase {
  kind: "horizontal";
  pointAId: EntityId;
  pointBId: EntityId;
}

export interface VerticalConstraint extends ConstraintBase {
  kind: "vertical";
  pointAId: EntityId;
  pointBId: EntityId;
}

export interface ParallelConstraint extends ConstraintBase {
  kind: "parallel";
  segmentAId: EntityId;
  segmentBId: EntityId;
}

export interface PerpendicularConstraint extends ConstraintBase {
  kind: "perpendicular";
  segmentAId: EntityId;
  segmentBId: EntityId;
}

export interface AngleConstraint extends ConstraintBase {
  kind: "angle";
  segmentAId: EntityId;
  segmentBId: EntityId;
  targetRad: number;
}

export type Constraint =
  | DistanceConstraint
  | FixedLengthConstraint
  | EqualLengthConstraint
  | HorizontalConstraint
  | VerticalConstraint
  | ParallelConstraint
  | PerpendicularConstraint
  | AngleConstraint;

export interface ConstraintSet {
  version: 1;
  nextIndex: number;
  constraints: Constraint[];
}

export function createConstraintSet(): ConstraintSet {
  return { version: 1, nextIndex: 1, constraints: [] };
}

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function allocateId(set: ConstraintSet): string {
  return `constraint/${String(set.nextIndex++).padStart(8, "0")}`;
}

function pushConstraint(set: ConstraintSet, constraint: Constraint): { set: ConstraintSet; id: string } {
  const next: ConstraintSet = { ...clone(set), constraints: [...clone(set.constraints), clone(constraint)] };
  return { set: next, id: constraint.id };
}

function base(set: ConstraintSet, panelId: EntityId): { id: string; panelId: EntityId; enabled: boolean } {
  return { id: allocateId(set), panelId, enabled: true };
}

export function addDistanceConstraint(
  set: ConstraintSet, panelId: EntityId, pointAId: EntityId, pointBId: EntityId, targetM: number,
): { set: ConstraintSet; id: string } {
  return pushConstraint(set, { ...base(set, panelId), kind: "distance", pointAId, pointBId, targetM });
}

export function addFixedLengthConstraint(
  set: ConstraintSet, panelId: EntityId, segmentId: EntityId, targetM: number,
): { set: ConstraintSet; id: string } {
  return pushConstraint(set, { ...base(set, panelId), kind: "fixed-length", segmentId, targetM });
}

export function addEqualLengthConstraint(
  set: ConstraintSet, panelId: EntityId, segmentAId: EntityId, segmentBId: EntityId,
): { set: ConstraintSet; id: string } {
  return pushConstraint(set, { ...base(set, panelId), kind: "equal-length", segmentAId, segmentBId });
}

export function addHorizontalConstraint(
  set: ConstraintSet, panelId: EntityId, pointAId: EntityId, pointBId: EntityId,
): { set: ConstraintSet; id: string } {
  return pushConstraint(set, { ...base(set, panelId), kind: "horizontal", pointAId, pointBId });
}

export function addVerticalConstraint(
  set: ConstraintSet, panelId: EntityId, pointAId: EntityId, pointBId: EntityId,
): { set: ConstraintSet; id: string } {
  return pushConstraint(set, { ...base(set, panelId), kind: "vertical", pointAId, pointBId });
}

export function addParallelConstraint(
  set: ConstraintSet, panelId: EntityId, segmentAId: EntityId, segmentBId: EntityId,
): { set: ConstraintSet; id: string } {
  return pushConstraint(set, { ...base(set, panelId), kind: "parallel", segmentAId, segmentBId });
}

export function addPerpendicularConstraint(
  set: ConstraintSet, panelId: EntityId, segmentAId: EntityId, segmentBId: EntityId,
): { set: ConstraintSet; id: string } {
  return pushConstraint(set, { ...base(set, panelId), kind: "perpendicular", segmentAId, segmentBId });
}

export function addAngleConstraint(
  set: ConstraintSet, panelId: EntityId, segmentAId: EntityId, segmentBId: EntityId, targetRad: number,
): { set: ConstraintSet; id: string } {
  return pushConstraint(set, { ...base(set, panelId), kind: "angle", segmentAId, segmentBId, targetRad });
}

export function setConstraintEnabled(set: ConstraintSet, id: string, enabled: boolean): ConstraintSet {
  const next = clone(set);
  const found = next.constraints.find((c) => c.id === id);
  if (!found) throw new PatternCadError("missing-reference", `constraint '${id}' does not exist`, id);
  found.enabled = enabled;
  return next;
}

export function removeConstraint(set: ConstraintSet, id: string): ConstraintSet {
  if (!set.constraints.some((c) => c.id === id)) {
    throw new PatternCadError("missing-reference", `constraint '${id}' does not exist`, id);
  }
  return { ...clone(set), constraints: clone(set.constraints.filter((c) => c.id !== id)) };
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

export type ConstraintDiagnosticCode =
  | "missing-reference"
  | "invalid-target"
  | "unknown-kind"
  | "duplicate-id";

export interface ConstraintDiagnostic {
  code: ConstraintDiagnosticCode;
  message: string;
  constraintId?: string;
}

function pointInPanel(doc: PatternDocument, panelId: EntityId, pointId: EntityId): PatternPoint {
  const p = doc.points.find((q) => q.id === pointId);
  if (!p || p.panelId !== panelId) {
    throw new PatternCadError("missing-reference", `point '${pointId}' is not in panel '${panelId}'`, pointId);
  }
  return p;
}

function segmentInPanel(doc: PatternDocument, panelId: EntityId, segmentId: EntityId): PatternSegment {
  const s = doc.segments.find((q) => q.id === segmentId);
  if (!s || s.panelId !== panelId) {
    throw new PatternCadError("missing-reference", `segment '${segmentId}' is not in panel '${panelId}'`, segmentId);
  }
  return s;
}

export function validateConstraintSet(doc: PatternDocument, set: ConstraintSet): ConstraintDiagnostic[] {
  const diagnostics: ConstraintDiagnostic[] = [];
  if (!set || typeof set !== "object" || set.version !== 1 || !Array.isArray(set.constraints)) {
    return [{ code: "unknown-kind", message: "constraint set shape or version is invalid" }];
  }
  const seen = new Set<string>();
  const kinds = new Set(["distance", "fixed-length", "equal-length", "horizontal", "vertical", "parallel", "perpendicular", "angle"]);
  for (const c of set.constraints) {
    const fail = (code: ConstraintDiagnosticCode, message: string): void => {
      diagnostics.push({ code, message, constraintId: (c as Constraint)?.id });
    };
    if (typeof c?.id !== "string" || c.id.length === 0 || seen.has(c.id)) {
      fail("duplicate-id", `constraint id '${String(c?.id)}' is missing or duplicated`);
      continue;
    }
    seen.add(c.id);
    if (!kinds.has(c.kind)) {
      fail("unknown-kind", `constraint '${c.id}' has unknown kind '${String(c.kind)}'`);
      continue;
    }
    if (!doc.panels.some((p) => p.id === c.panelId)) {
      fail("missing-reference", `constraint '${c.id}' references missing panel '${c.panelId}'`);
      continue;
    }
    try {
      switch (c.kind) {
        case "distance":
          pointInPanel(doc, c.panelId, c.pointAId);
          pointInPanel(doc, c.panelId, c.pointBId);
          if (!Number.isFinite(c.targetM) || !(c.targetM > 1e-12)) fail("invalid-target", `distance target must be positive`);
          break;
        case "fixed-length":
          segmentInPanel(doc, c.panelId, c.segmentId);
          if (!Number.isFinite(c.targetM) || !(c.targetM > 1e-12)) fail("invalid-target", `length target must be positive`);
          break;
        case "equal-length":
        case "parallel":
        case "perpendicular":
          segmentInPanel(doc, c.panelId, c.segmentAId);
          segmentInPanel(doc, c.panelId, c.segmentBId);
          break;
        case "horizontal":
        case "vertical":
          pointInPanel(doc, c.panelId, c.pointAId);
          pointInPanel(doc, c.panelId, c.pointBId);
          break;
        case "angle":
          segmentInPanel(doc, c.panelId, c.segmentAId);
          segmentInPanel(doc, c.panelId, c.segmentBId);
          if (!Number.isFinite(c.targetRad)) fail("invalid-target", `angle target must be finite`);
          break;
      }
    } catch (error) {
      fail("missing-reference", error instanceof Error ? error.message : String(error));
    }
  }
  return diagnostics;
}

// ---------------------------------------------------------------------------
// Residuals (0 = satisfied)
// ---------------------------------------------------------------------------

function pos(p: PatternPoint): Vec2 {
  return [p.x, p.y];
}

function chord(doc: PatternDocument, segment: PatternSegment): { a: Vec2; b: Vec2 } {
  const a = getPoint(doc, segment.startPointId, segment.panelId);
  const b = getPoint(doc, segment.endPointId, segment.panelId);
  return { a: pos(a), b: pos(b) };
}

function chordAngle(doc: PatternDocument, segment: PatternSegment): number {
  const { a, b } = chord(doc, segment);
  return Math.atan2(b[1] - a[1], b[0] - a[0]);
}

function angleDiff(a: number, b: number): number {
  let d = (a - b) % (2 * Math.PI);
  if (d > Math.PI) d -= 2 * Math.PI;
  if (d < -Math.PI) d += 2 * Math.PI;
  return d;
}

/** Residual in natural units (metres for lengths, radians for angles). */
export function constraintResidual(doc: PatternDocument, constraint: Constraint): number {
  const panelId = constraint.panelId;
  switch (constraint.kind) {
    case "distance": {
      const a = pointInPanel(doc, panelId, constraint.pointAId);
      const b = pointInPanel(doc, panelId, constraint.pointBId);
      return Math.abs(Math.hypot(b.x - a.x, b.y - a.y) - constraint.targetM);
    }
    case "fixed-length": {
      const s = segmentInPanel(doc, panelId, constraint.segmentId);
      const { a, b } = chord(doc, s);
      return Math.abs(Math.hypot(b[0] - a[0], b[1] - a[1]) - constraint.targetM);
    }
    case "equal-length": {
      const sa = segmentInPanel(doc, panelId, constraint.segmentAId);
      const sb = segmentInPanel(doc, panelId, constraint.segmentBId);
      const ca = chord(doc, sa), cb = chord(doc, sb);
      return Math.abs(
        Math.hypot(ca.b[0] - ca.a[0], ca.b[1] - ca.a[1]) - Math.hypot(cb.b[0] - cb.a[0], cb.b[1] - cb.a[1]),
      );
    }
    case "horizontal": {
      const a = pointInPanel(doc, panelId, constraint.pointAId);
      const b = pointInPanel(doc, panelId, constraint.pointBId);
      return Math.abs(b.y - a.y);
    }
    case "vertical": {
      const a = pointInPanel(doc, panelId, constraint.pointAId);
      const b = pointInPanel(doc, panelId, constraint.pointBId);
      return Math.abs(b.x - a.x);
    }
    case "parallel": {
      const sa = segmentInPanel(doc, panelId, constraint.segmentAId);
      const sb = segmentInPanel(doc, panelId, constraint.segmentBId);
      const d = angleDiff(chordAngle(doc, sa), chordAngle(doc, sb));
      return Math.min(Math.abs(d), Math.abs(Math.PI - Math.abs(d)));
    }
    case "perpendicular": {
      const sa = segmentInPanel(doc, panelId, constraint.segmentAId);
      const sb = segmentInPanel(doc, panelId, constraint.segmentBId);
      const d = Math.abs(angleDiff(chordAngle(doc, sa), chordAngle(doc, sb)));
      return Math.abs(d - Math.PI / 2);
    }
    case "angle": {
      const sa = segmentInPanel(doc, panelId, constraint.segmentAId);
      const sb = segmentInPanel(doc, panelId, constraint.segmentBId);
      return Math.abs(angleDiff(chordAngle(doc, sb) - chordAngle(doc, sa), constraint.targetRad));
    }
  }
}

// ---------------------------------------------------------------------------
// Minimum projection solver
// ---------------------------------------------------------------------------

export interface SolveOptions {
  maxIterations?: number;
  tolerance?: number;
}

export interface SolveResult {
  document: PatternDocument;
  iterations: number;
  residuals: Record<string, number>;
  satisfied: boolean;
  unsatisfied: string[];
}

function mutablePoint(doc: PatternDocument, panelId: EntityId, pointId: EntityId): PatternPoint {
  const p = doc.points.find((q) => q.id === pointId && q.panelId === panelId);
  if (!p) throw new PatternCadError("missing-reference", `point '${pointId}' is not in panel '${panelId}'`, pointId);
  return p;
}

/** Rigidly rotate every point listed around a pivot (used for arc-safe re-aiming). */
function rotatePoints(doc: PatternDocument, panelId: EntityId, pointIds: EntityId[], pivot: Vec2, angle: number): void {
  const c = Math.cos(angle), s = Math.sin(angle);
  for (const id of pointIds) {
    const p = mutablePoint(doc, panelId, id);
    const dx = p.x - pivot[0], dy = p.y - pivot[1];
    p.x = pivot[0] + c * dx - s * dy;
    p.y = pivot[1] + s * dx + c * dy;
  }
}

function segmentPointIds(doc: PatternDocument, panelId: EntityId, segmentId: EntityId): EntityId[] {
  const s = segmentInPanel(doc, panelId, segmentId);
  const ids = [s.startPointId, s.endPointId];
  if (s.kind === "arc") ids.push(s.centerPointId);
  return ids;
}

function projectOnce(doc: PatternDocument, constraint: Constraint): void {
  const panelId = constraint.panelId;
  switch (constraint.kind) {
    case "distance": {
      const a = mutablePoint(doc, panelId, constraint.pointAId);
      const b = mutablePoint(doc, panelId, constraint.pointBId);
      const dx = b.x - a.x, dy = b.y - a.y;
      const d = Math.hypot(dx, dy);
      if (!(d > 1e-12)) return; // coincident: no defined direction; stays unsatisfied
      const corr = (d - constraint.targetM) / 2 / d;
      a.x += dx * corr;
      a.y += dy * corr;
      b.x -= dx * corr;
      b.y -= dy * corr;
      return;
    }
    case "fixed-length":
    case "equal-length":
      return; // handled as a coupled pair below (equal) or rescale (fixed)
    case "horizontal": {
      const a = mutablePoint(doc, panelId, constraint.pointAId);
      const b = mutablePoint(doc, panelId, constraint.pointBId);
      const mid = (a.y + b.y) / 2;
      a.y = mid;
      b.y = mid;
      return;
    }
    case "vertical": {
      const a = mutablePoint(doc, panelId, constraint.pointAId);
      const b = mutablePoint(doc, panelId, constraint.pointBId);
      const mid = (a.x + b.x) / 2;
      a.x = mid;
      b.x = mid;
      return;
    }
    case "parallel":
    case "perpendicular":
    case "angle": {
      const sa = segmentInPanel(doc, panelId, constraint.segmentAId);
      const target =
        constraint.kind === "parallel"
          ? chordAngle(doc, sa)
          : constraint.kind === "perpendicular"
            ? chordAngle(doc, sa) + Math.PI / 2
            : chordAngle(doc, sa) + constraint.targetRad;
      const sb = segmentInPanel(doc, panelId, constraint.segmentBId);
      const ids = segmentPointIds(doc, panelId, constraint.segmentBId);
      const { a, b } = chord(doc, sb);
      const pivot: Vec2 = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
      rotatePoints(doc, panelId, ids, pivot, angleDiff(target, chordAngle(doc, sb)));
      return;
    }
  }
}

function rescaleSegment(doc: PatternDocument, panelId: EntityId, segmentId: EntityId, targetM: number): void {
  const s = segmentInPanel(doc, panelId, segmentId);
  const { a, b } = chord(doc, s);
  const d = Math.hypot(b[0] - a[0], b[1] - a[1]);
  if (!(d > 1e-12) || !(targetM > 1e-12)) return;
  const mid: Vec2 = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  const f = targetM / d;
  for (const id of segmentPointIds(doc, panelId, segmentId)) {
    const p = mutablePoint(doc, panelId, id);
    // Scale about the chord midpoint; the arc center rides along rigidly
    // only when it coincides with the midpoint, otherwise it scales too —
    // validation after solving reports any radius inconsistency explicitly.
    p.x = mid[0] + (p.x - mid[0]) * f;
    p.y = mid[1] + (p.y - mid[1]) * f;
  }
}

/**
 * Sequential projection over enabled constraints in stored order.
 * Deterministic: fixed order, fixed iteration cap, no randomness.
 * Returns per-constraint residuals; anything above tolerance is listed in
 * `unsatisfied` (contradiction/degeneracy signal — never silent).
 */
export function solveConstraints(
  doc: PatternDocument,
  set: ConstraintSet,
  opts: SolveOptions = {},
): SolveResult {
  const diagnostics = validateConstraintSet(doc, set);
  if (diagnostics.length > 0) {
    const first = diagnostics[0];
    throw new PatternCadError(
      first.code === "duplicate-id" ? "duplicate-id" : "missing-reference",
      `cannot solve: ${first.code}: ${first.message}`,
      first.constraintId,
    );
  }
  const maxIterations = opts.maxIterations ?? 100;
  const tolerance = opts.tolerance ?? 1e-9;
  const active = set.constraints.filter((c) => c.enabled);
  const next: PatternDocument = JSON.parse(JSON.stringify(doc)) as PatternDocument;
  let iterations = 0;
  let residuals: Record<string, number> = {};
  for (let k = 0; k < maxIterations; k++) {
    for (const c of active) {
      if (c.kind === "fixed-length") {
        rescaleSegment(next, c.panelId, c.segmentId, c.targetM);
      } else if (c.kind === "equal-length") {
        const sa = segmentInPanel(next, c.panelId, c.segmentAId);
        const sb = segmentInPanel(next, c.panelId, c.segmentBId);
        const la = Math.hypot(chord(next, sa).b[0] - chord(next, sa).a[0], chord(next, sa).b[1] - chord(next, sa).a[1]);
        const lb = Math.hypot(chord(next, sb).b[0] - chord(next, sb).a[0], chord(next, sb).b[1] - chord(next, sb).a[1]);
        const mean = (la + lb) / 2;
        if (mean > 1e-12) {
          rescaleSegment(next, c.panelId, c.segmentAId, mean);
          rescaleSegment(next, c.panelId, c.segmentBId, mean);
        }
      } else {
        projectOnce(next, c);
      }
    }
    iterations = k + 1;
    residuals = {};
    let worst = 0;
    for (const c of active) {
      const r = constraintResidual(next, c);
      residuals[c.id] = r;
      if (r > worst) worst = r;
    }
    if (worst <= tolerance) break;
  }
  if (active.length === 0) residuals = {};
  const unsatisfied = Object.entries(residuals)
    .filter(([, r]) => r > tolerance)
    .map(([id]) => id);
  return { document: next, iterations, residuals, satisfied: unsatisfied.length === 0, unsatisfied };
}

// ---------------------------------------------------------------------------
// Measurements (read-only, canonical metres)
// ---------------------------------------------------------------------------

export interface MeasurementSet {
  distances: Record<string, number>;
}

export function measurePointDistance(doc: PatternDocument, panelId: EntityId, pointAId: EntityId, pointBId: EntityId): number {
  const a = pointInPanel(doc, panelId, pointAId);
  const b = pointInPanel(doc, panelId, pointBId);
  return Math.hypot(b.x - a.x, b.y - a.y);
}

export function measureEdgeLength(doc: PatternDocument, panelId: EntityId, segmentId: EntityId): number {
  const s = segmentInPanel(doc, panelId, segmentId);
  const { a, b } = chord(doc, s);
  if (s.kind === "line") return Math.hypot(b[0] - a[0], b[1] - a[1]);
  const center = getPoint(doc, s.centerPointId, panelId);
  const radius = Math.hypot(a[0] - center.x, a[1] - center.y);
  return Math.abs(s.sweepRad) * radius;
}

export function measureSegmentAngle(doc: PatternDocument, panelId: EntityId, segmentAId: EntityId, segmentBId: EntityId): number {
  const sa = segmentInPanel(doc, panelId, segmentAId);
  const sb = segmentInPanel(doc, panelId, segmentBId);
  return Math.abs(angleDiff(chordAngle(doc, sb), chordAngle(doc, sa)));
}

export function measurePanelArea(doc: PatternDocument, panelId: EntityId): number {
  getPanel(doc, panelId);
  return measurePanel(doc, panelId).area;
}

export function measurePanelPerimeter(doc: PatternDocument, panelId: EntityId): number {
  getPanel(doc, panelId);
  return measurePanel(doc, panelId).perimeter;
}

/** Seam length in metres: arclength along side A's resolved stitch polyline. */
export function measureSeamLength(doc: PatternDocument, seam: Seam): number {
  const pairs = resolveStitchPairs(doc, seam);
  let total = 0;
  for (let i = 1; i < pairs.length; i++) {
    const a = pairs[i - 1].pointA, b = pairs[i].pointA;
    total += Math.hypot(b[0] - a[0], b[1] - a[1]);
  }
  return total;
}

// ---------------------------------------------------------------------------
// Persistence (sidecar + envelope)
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

export function serializeConstraintSet(set: ConstraintSet): string {
  return canonicalJson(set);
}

export function deserializeConstraintSet(serialized: string): ConstraintSet {
  let parsed: unknown;
  try {
    parsed = JSON.parse(serialized);
  } catch {
    throw new PatternCadError("invalid-document", "serialized constraint set is not valid JSON");
  }
  const set = parsed as ConstraintSet;
  if (!set || typeof set !== "object" || set.version !== 1 || !Array.isArray(set.constraints)) {
    throw new PatternCadError("invalid-document", "constraint set shape or version is invalid");
  }
  return clone(set);
}

/** Envelope binding a kernel document serialization to its constraint sidecar. */
export function serializeConstrainedDocument(documentJson: string, set: ConstraintSet): string {
  JSON.parse(documentJson); // must be valid JSON; kernel owns its schema
  return canonicalJson({ constraints: set, pattern: JSON.parse(documentJson) });
}

export function deserializeConstrainedDocument(serialized: string): { documentJson: string; set: ConstraintSet } {
  let parsed: unknown;
  try {
    parsed = JSON.parse(serialized);
  } catch {
    throw new PatternCadError("invalid-document", "serialized constrained document is not valid JSON");
  }
  const env = parsed as { pattern?: unknown; constraints?: unknown };
  if (!env || typeof env !== "object" || !env.pattern || !env.constraints) {
    throw new PatternCadError("invalid-document", "constrained document envelope is invalid");
  }
  return { documentJson: JSON.stringify(env.pattern), set: deserializeConstraintSet(JSON.stringify(env.constraints)) };
}
