import {
  measurePanelQuality,
  triangulatePatternPanel,
  validatePanel,
  type BoundaryPrimitive,
  type PatternPanel as GeometryPanel,
  type TriangulateOptions,
  type TriangulatedPanel,
  type Vec2,
} from "./pattern-geometry.js";

export type EntityId = string;
export type BoundaryRole = "outer" | "hole";
export type Winding = "ccw" | "cw";
export type GeometryRole = "boundary" | "construction";

export interface Transform2D {
  /** Pattern-workspace translation in metres. */
  translation: Vec2;
  rotationRad: number;
  /** Positive axis scales only; reflection is not a pattern edit operation. */
  scale: Vec2;
}

export interface PatternPoint {
  id: EntityId;
  panelId: EntityId;
  x: number;
  y: number;
  role: GeometryRole;
}

interface SegmentBase {
  id: EntityId;
  panelId: EntityId;
  role: GeometryRole;
  startPointId: EntityId;
  endPointId: EntityId;
}

export interface LineSegment extends SegmentBase {
  kind: "line";
}

export interface CircularArcSegment extends SegmentBase {
  kind: "arc";
  centerPointId: EntityId;
  /** Signed sweep in radians. Positive is counter-clockwise. */
  sweepRad: number;
}

export type PatternSegment = LineSegment | CircularArcSegment;

export interface BoundaryLoop {
  id: EntityId;
  panelId: EntityId;
  role: BoundaryRole;
  /** Authored orientation; reversal updates this field explicitly. */
  orientation: Winding;
  /** Ordered, closed list of boundary segment IDs. */
  segmentIds: EntityId[];
}

export type DimensionKind = "distance" | "segment-length";

export interface PatternDimension {
  id: EntityId;
  panelId: EntityId;
  kind: DimensionKind;
  pointAId?: EntityId;
  pointBId?: EntityId;
  segmentId?: EntityId;
  label?: string;
}

export interface CoincidentConstraint {
  id: EntityId;
  panelId: EntityId;
  kind: "coincident";
  pointAId: EntityId;
  pointBId: EntityId;
}

export type PatternConstraint = CoincidentConstraint;

export interface PatternPanel {
  id: EntityId;
  name: string;
  materialId: string;
  grainAngleRad: number;
  transform: Transform2D;
  boundaryLoops: BoundaryLoop[];
  constructionSegmentIds: EntityId[];
  dimensions: PatternDimension[];
  constraints: PatternConstraint[];
}

export interface PatternDocument {
  schemaVersion: 1;
  id: EntityId;
  name: string;
  /** Monotonic document-local ID allocation; never derived from time/randomness. */
  nextEntityIndex: number;
  panels: PatternPanel[];
  points: PatternPoint[];
  segments: PatternSegment[];
}

export type PatternDiagnosticCode =
  | "invalid-document"
  | "invalid-id"
  | "duplicate-id"
  | "missing-reference"
  | "invalid-transform"
  | "open-boundary"
  | "zero-length-edge"
  | "duplicate-consecutive-points"
  | "invalid-arc"
  | "self-intersection"
  | "invalid-winding"
  | "degenerate-panel"
  | "invalid-dimension"
  | "invalid-constraint"
  | "unsupported-operation";

export interface PatternDiagnostic {
  code: PatternDiagnosticCode;
  message: string;
  entityId?: EntityId;
  panelId?: EntityId;
}

export interface PatternValidationResult {
  valid: boolean;
  diagnostics: PatternDiagnostic[];
}

export class PatternCadError extends Error {
  readonly code: PatternDiagnosticCode;
  readonly entityId?: EntityId;

  constructor(code: PatternDiagnosticCode, message: string, entityId?: EntityId) {
    super(`pattern CAD (${code}): ${message}`);
    this.name = "PatternCadError";
    this.code = code;
    this.entityId = entityId;
  }
}

export class PatternCadValidationError extends Error {
  readonly diagnostics: PatternDiagnostic[];

  constructor(diagnostics: PatternDiagnostic[]) {
    super(`invalid pattern document:\n${diagnostics.map((d) => `- ${d.code}: ${d.message}`).join("\n")}`);
    this.name = "PatternCadValidationError";
    this.diagnostics = diagnostics;
  }
}

const DEFAULT_EPSILON_M = 1e-9;
const DEFAULT_MIN_AREA_M2 = 1e-12;
const TAU = 2 * Math.PI;

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function finite(value: number): boolean {
  return Number.isFinite(value);
}

function distance(a: readonly number[], b: readonly number[]): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

function pointPosition(point: PatternPoint): Vec2 {
  return [point.x, point.y];
}

function findPanel(document: PatternDocument, panelId: EntityId): PatternPanel {
  const panel = document.panels.find((candidate) => candidate.id === panelId);
  if (!panel) throw new PatternCadError("missing-reference", `panel '${panelId}' does not exist`, panelId);
  return panel;
}

function findPoint(document: PatternDocument, pointId: EntityId, panelId?: EntityId): PatternPoint {
  const point = document.points.find((candidate) => candidate.id === pointId);
  if (!point || (panelId !== undefined && point.panelId !== panelId)) {
    throw new PatternCadError("missing-reference", `point '${pointId}' does not exist in the requested panel`, pointId);
  }
  return point;
}

function findSegment(document: PatternDocument, segmentId: EntityId, panelId?: EntityId): PatternSegment {
  const segment = document.segments.find((candidate) => candidate.id === segmentId);
  if (!segment || (panelId !== undefined && segment.panelId !== panelId)) {
    throw new PatternCadError("missing-reference", `segment '${segmentId}' does not exist in the requested panel`, segmentId);
  }
  return segment;
}

function allocateId(document: PatternDocument, kind: string): EntityId {
  const occupied = new Set<string>([
    document.id,
    ...document.panels.map((item) => item.id),
    ...document.points.map((item) => item.id),
    ...document.segments.map((item) => item.id),
    ...document.panels.flatMap((panel) => [
      ...panel.boundaryLoops.map((loop) => loop.id),
      ...panel.dimensions.map((dimension) => dimension.id),
      ...panel.constraints.map((constraint) => constraint.id),
    ]),
  ]);
  for (;;) {
    const suffix = String(document.nextEntityIndex++).padStart(8, "0");
    const id = `${document.id}/${kind}/${suffix}`;
    if (!occupied.has(id)) return id;
  }
}

function ensurePanel(document: PatternDocument, panelId: EntityId): void {
  findPanel(document, panelId);
}

export function createPatternDocument(id: EntityId, name = "Untitled pattern"): PatternDocument {
  if (typeof id !== "string" || id.trim().length === 0) {
    throw new PatternCadError("invalid-id", "document ID must be a non-empty stable string");
  }
  return {
    schemaVersion: 1,
    id,
    name,
    nextEntityIndex: 1,
    panels: [],
    points: [],
    segments: [],
  };
}

export function createPanel(document: PatternDocument, name: string, materialId = "default-material"):
  { document: PatternDocument; panelId: EntityId } {
  const next = clone(document);
  const panelId = allocateId(next, "panel");
  next.panels.push({
    id: panelId,
    name,
    materialId,
    grainAngleRad: 0,
    transform: { translation: [0, 0], rotationRad: 0, scale: [1, 1] },
    boundaryLoops: [],
    constructionSegmentIds: [],
    dimensions: [],
    constraints: [],
  });
  return { document: next, panelId };
}

export function createBoundaryLoop(
  document: PatternDocument,
  panelId: EntityId,
  role: BoundaryRole,
  orientation: Winding = role === "outer" ? "ccw" : "cw",
): { document: PatternDocument; loopId: EntityId } {
  const next = clone(document);
  const panel = findPanel(next, panelId);
  const loopId = allocateId(next, "loop");
  panel.boundaryLoops.push({ id: loopId, panelId, role, orientation, segmentIds: [] });
  return { document: next, loopId };
}

export function createPoint(
  document: PatternDocument,
  panelId: EntityId,
  position: Vec2,
  role: GeometryRole = "boundary",
): { document: PatternDocument; pointId: EntityId } {
  ensurePanel(document, panelId);
  if (position.length !== 2 || !position.every(finite)) {
    throw new PatternCadError("invalid-transform", "point coordinates must be finite 2D values");
  }
  const next = clone(document);
  const pointId = allocateId(next, "point");
  next.points.push({ id: pointId, panelId, x: position[0], y: position[1], role });
  return { document: next, pointId };
}

export function createBoundaryLine(
  document: PatternDocument,
  panelId: EntityId,
  loopId: EntityId,
  startPointId: EntityId,
  endPointId: EntityId,
): { document: PatternDocument; segmentId: EntityId } {
  const next = clone(document);
  const panel = findPanel(next, panelId);
  const loop = panel.boundaryLoops.find((candidate) => candidate.id === loopId);
  if (!loop) throw new PatternCadError("missing-reference", `boundary loop '${loopId}' does not exist`, loopId);
  findPoint(next, startPointId, panelId);
  findPoint(next, endPointId, panelId);
  const segmentId = allocateId(next, "segment");
  next.segments.push({ id: segmentId, panelId, role: "boundary", kind: "line", startPointId, endPointId });
  loop.segmentIds.push(segmentId);
  return { document: next, segmentId };
}

export function createBoundaryArc(
  document: PatternDocument,
  panelId: EntityId,
  loopId: EntityId,
  startPointId: EntityId,
  endPointId: EntityId,
  centerPointId: EntityId,
  sweepRad: number,
): { document: PatternDocument; segmentId: EntityId } {
  const next = clone(document);
  const panel = findPanel(next, panelId);
  const loop = panel.boundaryLoops.find((candidate) => candidate.id === loopId);
  if (!loop) throw new PatternCadError("missing-reference", `boundary loop '${loopId}' does not exist`, loopId);
  findPoint(next, startPointId, panelId);
  findPoint(next, endPointId, panelId);
  findPoint(next, centerPointId, panelId);
  const segmentId = allocateId(next, "segment");
  next.segments.push({ id: segmentId, panelId, role: "boundary", kind: "arc", startPointId, endPointId, centerPointId, sweepRad });
  loop.segmentIds.push(segmentId);
  return { document: next, segmentId };
}

export function createConstructionLine(
  document: PatternDocument,
  panelId: EntityId,
  startPointId: EntityId,
  endPointId: EntityId,
): { document: PatternDocument; segmentId: EntityId } {
  const next = clone(document);
  const panel = findPanel(next, panelId);
  findPoint(next, startPointId, panelId);
  findPoint(next, endPointId, panelId);
  const segmentId = allocateId(next, "construction-segment");
  next.segments.push({ id: segmentId, panelId, role: "construction", kind: "line", startPointId, endPointId });
  panel.constructionSegmentIds.push(segmentId);
  return { document: next, segmentId };
}

export function createDistanceDimension(
  document: PatternDocument,
  panelId: EntityId,
  pointAId: EntityId,
  pointBId: EntityId,
  label?: string,
): { document: PatternDocument; dimensionId: EntityId } {
  const next = clone(document);
  const panel = findPanel(next, panelId);
  findPoint(next, pointAId, panelId);
  findPoint(next, pointBId, panelId);
  const dimensionId = allocateId(next, "dimension");
  panel.dimensions.push({ id: dimensionId, panelId, kind: "distance", pointAId, pointBId, label });
  return { document: next, dimensionId };
}

export function createCoincidentConstraint(
  document: PatternDocument,
  panelId: EntityId,
  pointAId: EntityId,
  pointBId: EntityId,
): { document: PatternDocument; constraintId: EntityId } {
  const next = clone(document);
  const panel = findPanel(next, panelId);
  findPoint(next, pointAId, panelId);
  findPoint(next, pointBId, panelId);
  const constraintId = allocateId(next, "constraint");
  panel.constraints.push({ id: constraintId, panelId, kind: "coincident", pointAId, pointBId });
  return { document: next, constraintId };
}

export function localToGlobal(panel: PatternPanel, local: Vec2): Vec2 {
  const [sx, sy] = panel.transform.scale;
  const c = Math.cos(panel.transform.rotationRad), s = Math.sin(panel.transform.rotationRad);
  return [
    panel.transform.translation[0] + c * sx * local[0] - s * sy * local[1],
    panel.transform.translation[1] + s * sx * local[0] + c * sy * local[1],
  ];
}

export function globalToLocal(panel: PatternPanel, global: Vec2): Vec2 {
  const [sx, sy] = panel.transform.scale;
  if (!(sx > 0) || !(sy > 0)) throw new PatternCadError("invalid-transform", "panel scale must be positive", panel.id);
  const x = global[0] - panel.transform.translation[0];
  const y = global[1] - panel.transform.translation[1];
  const c = Math.cos(panel.transform.rotationRad), s = Math.sin(panel.transform.rotationRad);
  return [(c * x + s * y) / sx, (-s * x + c * y) / sy];
}

export function movePoint(
  document: PatternDocument,
  panelId: EntityId,
  pointId: EntityId,
  position: Vec2,
  space: "local" | "global" = "local",
): PatternDocument {
  const next = clone(document);
  const panel = findPanel(next, panelId);
  const point = findPoint(next, pointId, panelId);
  if (!position.every(finite)) throw new PatternCadError("invalid-transform", "point coordinates must be finite", pointId);
  const local = space === "global" ? globalToLocal(panel, position) : position;
  point.x = local[0];
  point.y = local[1];
  return next;
}

export function movePanel(document: PatternDocument, panelId: EntityId, delta: Vec2): PatternDocument {
  if (!delta.every(finite)) throw new PatternCadError("invalid-transform", "panel translation must be finite", panelId);
  const next = clone(document);
  const panel = findPanel(next, panelId);
  panel.transform.translation = [panel.transform.translation[0] + delta[0], panel.transform.translation[1] + delta[1]];
  return next;
}

export function rotatePanel(document: PatternDocument, panelId: EntityId, angleRad: number, pivotGlobal: Vec2): PatternDocument {
  if (!finite(angleRad) || !pivotGlobal.every(finite)) {
    throw new PatternCadError("invalid-transform", "rotation and pivot must be finite", panelId);
  }
  const next = clone(document);
  const panel = findPanel(next, panelId);
  const pivotLocal = globalToLocal(panel, pivotGlobal);
  panel.transform.rotationRad += angleRad;
  const [sx, sy] = panel.transform.scale;
  const c = Math.cos(panel.transform.rotationRad), s = Math.sin(panel.transform.rotationRad);
  panel.transform.translation = [
    pivotGlobal[0] - (c * sx * pivotLocal[0] - s * sy * pivotLocal[1]),
    pivotGlobal[1] - (s * sx * pivotLocal[0] + c * sy * pivotLocal[1]),
  ];
  return next;
}

export interface ScaleOptions {
  /** Explicitly state why a garment dimension change is valid. */
  intent: "pattern-size-adjustment" | "layout-only";
  pivotGlobal?: Vec2;
}

export function scalePanel(
  document: PatternDocument,
  panelId: EntityId,
  scaleX: number,
  scaleY: number,
  options: ScaleOptions,
): PatternDocument {
  if (!options || !options.intent || !finite(scaleX) || !finite(scaleY) || !(scaleX > 0) || !(scaleY > 0)) {
    throw new PatternCadError("invalid-transform", "scaling requires an explicit intent and positive finite factors", panelId);
  }
  if (options.pivotGlobal && !options.pivotGlobal.every(finite)) {
    throw new PatternCadError("invalid-transform", "scale pivot must be finite", panelId);
  }
  const next = clone(document);
  const panel = findPanel(next, panelId);
  const pivot = options.pivotGlobal ?? panel.transform.translation;
  const localPivot = globalToLocal(panel, pivot);
  if (options.intent === "pattern-size-adjustment") {
    for (const point of next.points) {
      if (point.panelId !== panelId) continue;
      point.x = localPivot[0] + (point.x - localPivot[0]) * scaleX;
      point.y = localPivot[1] + (point.y - localPivot[1]) * scaleY;
    }
  } else if (options.intent === "layout-only") {
    panel.transform.scale = [panel.transform.scale[0] * scaleX, panel.transform.scale[1] * scaleY];
    const [sx, sy] = panel.transform.scale;
    const c = Math.cos(panel.transform.rotationRad), s = Math.sin(panel.transform.rotationRad);
    panel.transform.translation = [
      pivot[0] - (c * sx * localPivot[0] - s * sy * localPivot[1]),
      pivot[1] - (s * sx * localPivot[0] + c * sy * localPivot[1]),
    ];
  } else {
    throw new PatternCadError("invalid-transform", "scale intent must be pattern-size-adjustment or layout-only", panelId);
  }
  return next;
}

export function transformPanelAroundPivot(
  document: PatternDocument,
  panelId: EntityId,
  pivotGlobal: Vec2,
  translationDelta: Vec2,
  rotationDeltaRad: number,
): PatternDocument {
  const rotated = rotatePanel(document, panelId, rotationDeltaRad, pivotGlobal);
  return movePanel(rotated, panelId, translationDelta);
}

export function measureDistance(document: PatternDocument, panelId: EntityId, pointAId: EntityId, pointBId: EntityId): number {
  findPanel(document, panelId);
  return distance(pointPosition(findPoint(document, pointAId, panelId)), pointPosition(findPoint(document, pointBId, panelId)));
}

export function measureDimension(document: PatternDocument, panelId: EntityId, dimensionId: EntityId): number {
  const panel = findPanel(document, panelId);
  const dimension = panel.dimensions.find((item) => item.id === dimensionId);
  if (!dimension) throw new PatternCadError("missing-reference", `dimension '${dimensionId}' does not exist`, dimensionId);
  if (dimension.kind === "distance" && dimension.pointAId && dimension.pointBId) {
    return measureDistance(document, panelId, dimension.pointAId, dimension.pointBId);
  }
  if (dimension.kind === "segment-length" && dimension.segmentId) {
    return segmentLength(document, findSegment(document, dimension.segmentId, panelId));
  }
  throw new PatternCadError("invalid-dimension", `dimension '${dimensionId}' has incomplete references`, dimensionId);
}

function segmentLength(document: PatternDocument, segment: PatternSegment): number {
  const start = pointPosition(findPoint(document, segment.startPointId, segment.panelId));
  const end = pointPosition(findPoint(document, segment.endPointId, segment.panelId));
  if (segment.kind === "line") {
    return distance(start, end);
  }
  const center = pointPosition(findPoint(document, segment.centerPointId, segment.panelId));
  const radius = distance(start, center);
  const samples = sampleArc(start, center, segment.sweepRad, Math.max(radius * 1e-5, 1e-8));
  let total = 0;
  for (let i = 1; i < samples.length; i++) {
    total += distance(samples[i - 1], samples[i]);
  }
  return total;
}

function segmentEndpoints(document: PatternDocument, segment: PatternSegment): [PatternPoint, PatternPoint] {
  return [findPoint(document, segment.startPointId, segment.panelId), findPoint(document, segment.endPointId, segment.panelId)];
}

function loopSignedArea(document: PatternDocument, loop: BoundaryLoop, arcTolerance = 1e-5): number {
  const polygon: Vec2[] = [];
  for (const segmentId of loop.segmentIds) {
    const segment = findSegment(document, segmentId, loop.panelId);
    const [start, end] = segmentEndpoints(document, segment);
    const points = segment.kind === "line"
      ? [pointPosition(start), pointPosition(end)]
      : sampleArc(pointPosition(start), pointPosition(findPoint(document, segment.centerPointId, loop.panelId)), segment.sweepRad, arcTolerance);
    if (polygon.length === 0) polygon.push(...points);
    else polygon.push(...points.slice(1));
  }
  if (polygon.length > 1 && distance(polygon[0], polygon[polygon.length - 1]) <= DEFAULT_EPSILON_M) polygon.pop();
  let twiceArea = 0;
  for (let i = 0; i < polygon.length; i++) {
    const a = polygon[i], b = polygon[(i + 1) % polygon.length];
    twiceArea += a[0] * b[1] - b[0] * a[1];
  }
  return twiceArea * 0.5;
}

function sampleArc(start: Vec2, center: Vec2, sweepRad: number, sagittaTolerance: number): Vec2[] {
  const radius = distance(start, center);
  if (!(radius > 0) || !finite(sweepRad) || sweepRad === 0 || Math.abs(sweepRad) >= TAU) {
    throw new PatternCadError("invalid-arc", "arc radius and sweep must be finite, nonzero, and less than 2π");
  }
  const tolerance = Math.max(Number.MIN_VALUE, Math.min(sagittaTolerance, radius));
  const step = 2 * Math.acos(Math.max(-1, Math.min(1, 1 - tolerance / radius)));
  const count = Math.max(1, Math.ceil(Math.abs(sweepRad) / step));
  const startAngle = Math.atan2(start[1] - center[1], start[0] - center[0]);
  return Array.from({ length: count + 1 }, (_, i) => {
    const angle = startAngle + sweepRad * i / count;
    return [center[0] + radius * Math.cos(angle), center[1] + radius * Math.sin(angle)] as Vec2;
  });
}

function geometryPrimitives(document: PatternDocument, loop: BoundaryLoop, sagittaTolerance: number): BoundaryPrimitive[] {
  return loop.segmentIds.map((segmentId) => {
    const segment = findSegment(document, segmentId, loop.panelId);
    const [start, end] = segmentEndpoints(document, segment);
    if (segment.kind === "line") return { kind: "polyline", points: [pointPosition(start), pointPosition(end)] };
    const center = pointPosition(findPoint(document, segment.centerPointId, loop.panelId));
    return { kind: "polyline", points: sampleArc(pointPosition(start), center, segment.sweepRad, sagittaTolerance) };
  });
}

function toGeometryPanel(document: PatternDocument, panel: PatternPanel, sagittaTolerance: number): GeometryPanel {
  const outer = panel.boundaryLoops.find((loop) => loop.role === "outer");
  if (!outer) throw new PatternCadError("open-boundary", `panel '${panel.id}' has no outer loop`, panel.id);
  const holes = panel.boundaryLoops.filter((loop) => loop.role === "hole");
  return {
    id: panel.id,
    outline: geometryPrimitives(document, outer, sagittaTolerance),
    holes: holes.map((loop) => geometryPrimitives(document, loop, sagittaTolerance)),
    grainAngleRad: panel.grainAngleRad,
    materialId: panel.materialId,
  };
}

function pushDiagnostic(
  diagnostics: PatternDiagnostic[],
  code: PatternDiagnosticCode,
  message: string,
  entityId?: EntityId,
  panelId?: EntityId,
): void {
  diagnostics.push({ code, message, ...(entityId ? { entityId } : {}), ...(panelId ? { panelId } : {}) });
}

export function validatePatternDocument(
  document: PatternDocument,
  options: { epsilonM?: number; minimumAreaM2?: number; sagittaToleranceM?: number } = {},
): PatternValidationResult {
  const diagnostics: PatternDiagnostic[] = [];
  if (!document || typeof document !== "object" || document.schemaVersion !== 1 || !Array.isArray(document.panels) ||
      !Array.isArray(document.points) || !Array.isArray(document.segments)) {
    return { valid: false, diagnostics: [{ code: "invalid-document", message: "document shape or schema version is invalid" }] };
  }
  if (typeof document.id !== "string" || document.id.trim() === "") {
    pushDiagnostic(diagnostics, "invalid-id", "document ID must be a non-empty string");
  }
  if (!Number.isSafeInteger(document.nextEntityIndex) || document.nextEntityIndex < 1) {
    pushDiagnostic(diagnostics, "invalid-document", "nextEntityIndex must be a positive safe integer");
  }
  const epsilon = options.epsilonM ?? DEFAULT_EPSILON_M;
  const minArea = options.minimumAreaM2 ?? DEFAULT_MIN_AREA_M2;
  const sagittaTolerance = options.sagittaToleranceM ?? 1e-5;
  const entityIds = new Set<string>();
  const claimId = (id: string, entityId?: string, panelId?: string): void => {
    if (typeof id !== "string" || id.trim() === "") {
      pushDiagnostic(diagnostics, "invalid-id", "entity ID must be a non-empty string", entityId, panelId);
    } else if (entityIds.has(id)) {
      pushDiagnostic(diagnostics, "duplicate-id", `entity ID '${id}' is duplicated`, id, panelId);
    } else entityIds.add(id);
  };
  claimId(document.id);
  for (const panel of document.panels) claimId(panel.id, panel.id);
  for (const point of document.points) claimId(point.id, point.id, point.panelId);
  for (const segment of document.segments) claimId(segment.id, segment.id, segment.panelId);

  const panelMap = new Map(document.panels.map((panel) => [panel.id, panel]));
  const pointMap = new Map(document.points.map((point) => [point.id, point]));
  const segmentMap = new Map(document.segments.map((segment) => [segment.id, segment]));
  const referencedBoundarySegments = new Map<string, string>();
  for (const panel of document.panels) {
    for (const loop of panel.boundaryLoops ?? []) {
      claimId(loop.id, loop.id, panel.id);
      if (loop.panelId !== panel.id) pushDiagnostic(diagnostics, "missing-reference", `loop '${loop.id}' has the wrong panel owner`, loop.id, panel.id);
      if (loop.segmentIds.length < 2) pushDiagnostic(diagnostics, "open-boundary", `loop '${loop.id}' needs at least two ordered segments`, loop.id, panel.id);
      for (let i = 0; i < loop.segmentIds.length; i++) {
        const segmentId = loop.segmentIds[i];
        const segment = segmentMap.get(segmentId);
        if (!segment || segment.panelId !== panel.id || segment.role !== "boundary") {
          pushDiagnostic(diagnostics, "missing-reference", `loop '${loop.id}' references invalid boundary segment '${segmentId}'`, segmentId, panel.id);
          continue;
        }
        if (referencedBoundarySegments.has(segmentId)) {
          pushDiagnostic(diagnostics, "duplicate-id", `boundary segment '${segmentId}' belongs to more than one loop`, segmentId, panel.id);
        } else referencedBoundarySegments.set(segmentId, loop.id);
        const nextSegment = segmentMap.get(loop.segmentIds[(i + 1) % loop.segmentIds.length]);
        if (nextSegment && segment.endPointId !== nextSegment.startPointId) {
          const a = pointMap.get(segment.endPointId), b = pointMap.get(nextSegment.startPointId);
          const samePosition = a && b && distance(pointPosition(a), pointPosition(b)) <= epsilon;
          pushDiagnostic(
            diagnostics,
            samePosition ? "duplicate-consecutive-points" : "open-boundary",
            samePosition ? `adjacent segments in loop '${loop.id}' use duplicate endpoint entities` : `loop '${loop.id}' is not closed between '${segment.id}' and '${nextSegment.id}'`,
            segment.id,
            panel.id,
          );
        }
      }
      if (loop.segmentIds.length >= 2 && loop.segmentIds.every((id) => segmentMap.has(id))) {
        try {
          const area = loopSignedArea(document, loop, sagittaTolerance);
          if (Math.abs(area) <= minArea) {
            pushDiagnostic(diagnostics, "degenerate-panel", `loop '${loop.id}' has area ${area} m²`, loop.id, panel.id);
          } else {
            const actual: Winding = area > 0 ? "ccw" : "cw";
            if (actual !== loop.orientation) pushDiagnostic(diagnostics, "invalid-winding", `loop '${loop.id}' records ${loop.orientation} but its geometry is ${actual}`, loop.id, panel.id);
          }
        } catch (error) {
          const code = error instanceof PatternCadError ? error.code : "invalid-arc";
          pushDiagnostic(diagnostics, code, error instanceof Error ? error.message : String(error), loop.id, panel.id);
        }
      }
    }
    const outerLoops = (panel.boundaryLoops ?? []).filter((loop) => loop.role === "outer");
    if (outerLoops.length !== 1) pushDiagnostic(diagnostics, "open-boundary", `panel '${panel.id}' must have exactly one outer boundary`, panel.id, panel.id);
    const transform = panel.transform;
    if (!transform || !transform.translation?.every(finite) || !transform.scale?.every((v) => finite(v) && v > 0) || !finite(transform.rotationRad)) {
      pushDiagnostic(diagnostics, "invalid-transform", `panel '${panel.id}' has an invalid transform`, panel.id, panel.id);
    }
    if (!finite(panel.grainAngleRad)) pushDiagnostic(diagnostics, "invalid-transform", `panel '${panel.id}' grain angle must be finite`, panel.id, panel.id);
    for (const segmentId of panel.constructionSegmentIds ?? []) {
      const segment = segmentMap.get(segmentId);
      if (!segment || segment.panelId !== panel.id || segment.role !== "construction") {
        pushDiagnostic(diagnostics, "missing-reference", `panel '${panel.id}' references invalid construction segment '${segmentId}'`, segmentId, panel.id);
      }
    }
    for (const dimension of panel.dimensions ?? []) {
      claimId(dimension.id, dimension.id, panel.id);
      if (dimension.panelId !== panel.id) pushDiagnostic(diagnostics, "missing-reference", `dimension '${dimension.id}' owner does not match`, dimension.id, panel.id);
      if (dimension.kind === "distance") {
        if (!dimension.pointAId || !dimension.pointBId || pointMap.get(dimension.pointAId)?.panelId !== panel.id || pointMap.get(dimension.pointBId)?.panelId !== panel.id) {
          pushDiagnostic(diagnostics, "invalid-dimension", `dimension '${dimension.id}' references missing points`, dimension.id, panel.id);
        }
      } else if (dimension.kind === "segment-length") {
        if (!dimension.segmentId || segmentMap.get(dimension.segmentId)?.panelId !== panel.id) {
          pushDiagnostic(diagnostics, "invalid-dimension", `dimension '${dimension.id}' references a missing segment`, dimension.id, panel.id);
        }
      } else pushDiagnostic(diagnostics, "invalid-dimension", `dimension '${dimension.id}' has an unknown kind`, dimension.id, panel.id);
    }
    for (const constraint of panel.constraints ?? []) {
      claimId(constraint.id, constraint.id, panel.id);
      if (constraint.panelId !== panel.id || pointMap.get(constraint.pointAId)?.panelId !== panel.id || pointMap.get(constraint.pointBId)?.panelId !== panel.id) {
        pushDiagnostic(diagnostics, "invalid-constraint", `constraint '${constraint.id}' references missing points or panel`, constraint.id, panel.id);
      }
    }
    if (outerLoops.length === 1 && outerLoops[0].segmentIds.length >= 2) {
      try {
        const geometry = toGeometryPanel(document, panel, sagittaTolerance);
        validatePanel(geometry, { eps: epsilon, minArea });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        const code: PatternDiagnosticCode = /self-intersect|touch/i.test(message)
          ? "self-intersection"
          : /zero-area|area/i.test(message)
            ? "degenerate-panel"
            : /invalid-arc/i.test(message)
              ? "invalid-arc"
              : "open-boundary";
        pushDiagnostic(diagnostics, code, message, panel.id, panel.id);
      }
    }
  }
  for (const point of document.points) {
    if (!panelMap.has(point.panelId)) pushDiagnostic(diagnostics, "missing-reference", `point '${point.id}' references missing panel '${point.panelId}'`, point.id);
    if (!finite(point.x) || !finite(point.y)) pushDiagnostic(diagnostics, "invalid-transform", `point '${point.id}' is non-finite`, point.id, point.panelId);
  }
  for (const segment of document.segments) {
    if (!panelMap.has(segment.panelId)) pushDiagnostic(diagnostics, "missing-reference", `segment '${segment.id}' references missing panel '${segment.panelId}'`, segment.id);
    const start = pointMap.get(segment.startPointId), end = pointMap.get(segment.endPointId);
    if (!start || !end || start.panelId !== segment.panelId || end.panelId !== segment.panelId) {
      pushDiagnostic(diagnostics, "missing-reference", `segment '${segment.id}' references missing or foreign endpoints`, segment.id, segment.panelId);
      continue;
    }
    if (distance(pointPosition(start), pointPosition(end)) <= epsilon) {
      pushDiagnostic(diagnostics, "zero-length-edge", `segment '${segment.id}' has zero length`, segment.id, segment.panelId);
    }
    if (segment.kind === "arc") {
      const center = pointMap.get(segment.centerPointId);
      if (!center || center.panelId !== segment.panelId) {
        pushDiagnostic(diagnostics, "missing-reference", `arc '${segment.id}' references a missing center`, segment.id, segment.panelId);
      } else {
        const r0 = distance(pointPosition(start), pointPosition(center));
        const r1 = distance(pointPosition(end), pointPosition(center));
        const startAngle = Math.atan2(start.y - center.y, start.x - center.x);
        const expectedEnd: Vec2 = [center.x + r0 * Math.cos(startAngle + segment.sweepRad), center.y + r0 * Math.sin(startAngle + segment.sweepRad)];
        const arcValid = finite(segment.sweepRad) && Math.abs(segment.sweepRad) > epsilon && Math.abs(segment.sweepRad) < TAU &&
          r0 > epsilon && Math.abs(r0 - r1) <= Math.max(epsilon, r0 * 1e-8) &&
          distance(expectedEnd, pointPosition(end)) <= Math.max(epsilon, r0 * 1e-8);
        if (!arcValid) pushDiagnostic(diagnostics, "invalid-arc", `arc '${segment.id}' has inconsistent radii or sweep`, segment.id, segment.panelId);
      }
    }
    if (segment.role === "boundary" && !referencedBoundarySegments.has(segment.id)) {
      pushDiagnostic(diagnostics, "missing-reference", `boundary segment '${segment.id}' is not assigned to a loop`, segment.id, segment.panelId);
    }
  }
  return { valid: diagnostics.length === 0, diagnostics };
}

export function triangulateCadPanel(
  document: PatternDocument,
  panelId: EntityId,
  options: TriangulateOptions = {},
): TriangulatedPanel {
  const result = validatePatternDocument(document, {
    epsilonM: options.eps,
    minimumAreaM2: options.minArea,
    sagittaToleranceM: options.sagittaTol,
  });
  if (!result.valid) throw new PatternCadValidationError(result.diagnostics);
  const panel = findPanel(document, panelId);
  const geometry = toGeometryPanel(document, panel, options.sagittaTol ?? 1e-5);
  return triangulatePatternPanel(geometry, options);
}

export function qualityOfCadPanel(
  document: PatternDocument,
  panelId: EntityId,
  triangulation = triangulateCadPanel(document, panelId),
): ReturnType<typeof measurePanelQuality> {
  const panel = findPanel(document, panelId);
  return measurePanelQuality(toGeometryPanel(document, panel, 1e-5), triangulation);
}

export function insertBoundaryPoint(
  document: PatternDocument,
  panelId: EntityId,
  loopId: EntityId,
  segmentId: EntityId,
  position: Vec2,
  toleranceM = 1e-8,
): { document: PatternDocument; pointId: EntityId; segmentIds: [EntityId, EntityId] } {
  const currentPanel = findPanel(document, panelId);
  const loop = currentPanel.boundaryLoops.find((candidate) => candidate.id === loopId);
  const segment = findSegment(document, segmentId, panelId);
  if (!loop || !loop.segmentIds.includes(segmentId)) throw new PatternCadError("missing-reference", `segment '${segmentId}' is not in loop '${loopId}'`, segmentId);
  if (segment.kind === "arc") {
    const start = pointPosition(findPoint(document, segment.startPointId, panelId));
    const center = pointPosition(findPoint(document, segment.centerPointId, panelId));
    const radius = distance(start, center);
    if (!(radius > 0)) throw new PatternCadError("zero-length-edge", `segment '${segmentId}' has zero radius`, segmentId);
    const radial = distance(position, center);
    // Tolerance scales with radius so large panels accept proportional deviation.
    const radialTolerance = Math.max(toleranceM, radius * 1e-9);
    if (Math.abs(radial - radius) > radialTolerance) {
      throw new PatternCadError("unsupported-operation", "inserted point must lie on the boundary arc", segmentId);
    }
    const startAngle = Math.atan2(start[1] - center[1], start[0] - center[0]);
    const pointAngle = Math.atan2(position[1] - center[1], position[0] - center[0]);
    const sweep = segment.sweepRad;
    // Normalize the angular offset into the sweep direction.
    let delta = pointAngle - startAngle;
    if (sweep > 0) {
      while (delta < 0) delta += TAU;
      while (delta >= TAU) delta -= TAU;
    } else {
      while (delta > 0) delta -= TAU;
      while (delta <= -TAU) delta += TAU;
    }
    const t = delta / sweep;
    // Endpoint angular tolerance scales with the sweep so tiny arcs stay strict.
    const angularTolerance = toleranceM / radius;
    if (!(t > angularTolerance / Math.abs(sweep) && t < 1 - angularTolerance / Math.abs(sweep))) {
      throw new PatternCadError("unsupported-operation", "inserted point must lie strictly inside the boundary arc", segmentId);
    }
    // Snap to the exact circle so the authored arc stays radius-consistent.
    const snapped: Vec2 = [center[0] + radius * Math.cos(startAngle + delta), center[1] + radius * Math.sin(startAngle + delta)];
    return splitArcAtT(document, panelId, loopId, segmentId, t, snapped);
  }
  const start = pointPosition(findPoint(document, segment.startPointId, panelId));
  const end = pointPosition(findPoint(document, segment.endPointId, panelId));
  const dx = end[0] - start[0], dy = end[1] - start[1];
  const lengthSquared = dx * dx + dy * dy;
  if (!(lengthSquared > 0)) throw new PatternCadError("zero-length-edge", `segment '${segmentId}' has zero length`, segmentId);
  const t = ((position[0] - start[0]) * dx + (position[1] - start[1]) * dy) / lengthSquared;
  const projection: Vec2 = [start[0] + t * dx, start[1] + t * dy];
  if (!(t > toleranceM && t < 1 - toleranceM) || distance(position, projection) > toleranceM) {
    throw new PatternCadError("unsupported-operation", "inserted point must lie strictly inside the boundary segment", segmentId);
  }
  let next = clone(document);
  const pointResult = createPoint(next, panelId, position, "boundary");
  next = pointResult.document;
  const firstId = allocateId(next, "segment");
  const secondId = allocateId(next, "segment");
  const old = findSegment(next, segmentId, panelId) as LineSegment;
  const first: LineSegment = { ...old, id: firstId, startPointId: old.startPointId, endPointId: pointResult.pointId };
  const second: LineSegment = { ...old, id: secondId, startPointId: pointResult.pointId, endPointId: old.endPointId };
  next.segments = next.segments.filter((candidate) => candidate.id !== segmentId);
  next.segments.push(first, second);
  const updatedPanel = findPanel(next, panelId);
  const updatedLoop = updatedPanel.boundaryLoops.find((candidate) => candidate.id === loopId)!;
  const index = updatedLoop.segmentIds.indexOf(segmentId);
  updatedLoop.segmentIds.splice(index, 1, firstId, secondId);
  return { document: next, pointId: pointResult.pointId, segmentIds: [firstId, secondId] };
}

function splitArcAtT(
  document: PatternDocument,
  panelId: EntityId,
  loopId: EntityId,
  segmentId: EntityId,
  t: number,
  position: Vec2,
): { document: PatternDocument; pointId: EntityId; segmentIds: [EntityId, EntityId] } {
  const segment = findSegment(document, segmentId, panelId);
  if (segment.kind !== "arc") throw new PatternCadError("unsupported-operation", "arc split requires an arc segment", segmentId);
  let next = clone(document);
  const pointResult = createPoint(next, panelId, position, "boundary");
  next = pointResult.document;
  const firstId = allocateId(next, "segment");
  const secondId = allocateId(next, "segment");
  const old = findSegment(next, segmentId, panelId) as CircularArcSegment;
  const first: CircularArcSegment = { ...old, id: firstId, startPointId: old.startPointId, endPointId: pointResult.pointId, sweepRad: old.sweepRad * t };
  const second: CircularArcSegment = { ...old, id: secondId, startPointId: pointResult.pointId, endPointId: old.endPointId, sweepRad: old.sweepRad * (1 - t) };
  next.segments = next.segments.filter((candidate) => candidate.id !== segmentId);
  next.segments.push(first, second);
  const updatedPanel = findPanel(next, panelId);
  const updatedLoop = updatedPanel.boundaryLoops.find((candidate) => candidate.id === loopId)!;
  const index = updatedLoop.segmentIds.indexOf(segmentId);
  updatedLoop.segmentIds.splice(index, 1, firstId, secondId);
  return { document: next, pointId: pointResult.pointId, segmentIds: [firstId, secondId] };
}

export function splitBoundarySegment(
  document: PatternDocument,
  panelId: EntityId,
  loopId: EntityId,
  segmentId: EntityId,
  t: number,
): { document: PatternDocument; pointId: EntityId; segmentIds: [EntityId, EntityId] } {
  if (!finite(t) || !(t > 0 && t < 1)) throw new PatternCadError("unsupported-operation", "split parameter must be strictly inside (0,1)", segmentId);
  const segment = findSegment(document, segmentId, panelId);
  if (segment.kind === "arc") {
    const start = pointPosition(findPoint(document, segment.startPointId, panelId));
    const center = pointPosition(findPoint(document, segment.centerPointId, panelId));
    const radius = distance(start, center);
    if (!(radius > 0)) throw new PatternCadError("zero-length-edge", `segment '${segmentId}' has zero radius`, segmentId);
    const startAngle = Math.atan2(start[1] - center[1], start[0] - center[0]);
    const position: Vec2 = [
      center[0] + radius * Math.cos(startAngle + segment.sweepRad * t),
      center[1] + radius * Math.sin(startAngle + segment.sweepRad * t),
    ];
    const loop = findPanel(document, panelId).boundaryLoops.find((candidate) => candidate.id === loopId);
    if (!loop || !loop.segmentIds.includes(segmentId)) {
      throw new PatternCadError("missing-reference", `segment '${segmentId}' is not in loop '${loopId}'`, segmentId);
    }
    return splitArcAtT(document, panelId, loopId, segmentId, t, position);
  }
  if (segment.kind !== "line") throw new PatternCadError("unsupported-operation", "segment split currently supports lines only", segmentId);
  const a = pointPosition(findPoint(document, segment.startPointId, panelId));
  const b = pointPosition(findPoint(document, segment.endPointId, panelId));
  return insertBoundaryPoint(document, panelId, loopId, segmentId, [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
}

export function reverseBoundaryOrientation(document: PatternDocument, panelId: EntityId, loopId: EntityId): PatternDocument {
  const next = clone(document);
  const panel = findPanel(next, panelId);
  const loop = panel.boundaryLoops.find((candidate) => candidate.id === loopId);
  if (!loop) throw new PatternCadError("missing-reference", `boundary loop '${loopId}' does not exist`, loopId);
  loop.segmentIds.reverse();
  loop.orientation = loop.orientation === "ccw" ? "cw" : "ccw";
  for (const segmentId of loop.segmentIds) {
    const segment = findSegment(next, segmentId, panelId);
    [segment.startPointId, segment.endPointId] = [segment.endPointId, segment.startPointId];
    if (segment.kind === "arc") segment.sweepRad = -segment.sweepRad;
  }
  return next;
}

export function duplicatePanel(
  document: PatternDocument,
  panelId: EntityId,
  translation: Vec2 = [0, 0],
): { document: PatternDocument; panelId: EntityId } {
  if (!translation.every(finite)) throw new PatternCadError("invalid-transform", "duplicate translation must be finite", panelId);
  const next = clone(document);
  const source = findPanel(next, panelId);
  const pointIds = next.points.filter((point) => point.panelId === panelId);
  const segmentIds = next.segments.filter((segment) => segment.panelId === panelId);
  const idMap = new Map<string, string>();
  const newPanelId = allocateId(next, "panel");
  idMap.set(panelId, newPanelId);
  for (const point of pointIds) idMap.set(point.id, allocateId(next, "point"));
  for (const segment of segmentIds) idMap.set(segment.id, allocateId(next, segment.role === "boundary" ? "segment" : "construction-segment"));
  for (const loop of source.boundaryLoops) idMap.set(loop.id, allocateId(next, "loop"));
  for (const dimension of source.dimensions) idMap.set(dimension.id, allocateId(next, "dimension"));
  for (const constraint of source.constraints) idMap.set(constraint.id, allocateId(next, "constraint"));

  const panelCopy = clone(source);
  panelCopy.id = newPanelId;
  panelCopy.name = `${source.name} copy`;
  panelCopy.transform.translation = [source.transform.translation[0] + translation[0], source.transform.translation[1] + translation[1]];
  panelCopy.boundaryLoops = source.boundaryLoops.map((loop) => ({
    ...clone(loop), id: idMap.get(loop.id)!, panelId: newPanelId, segmentIds: loop.segmentIds.map((id) => idMap.get(id)!),
  }));
  panelCopy.constructionSegmentIds = source.constructionSegmentIds.map((id) => idMap.get(id)!);
  panelCopy.dimensions = source.dimensions.map((dimension) => ({
    ...clone(dimension), id: idMap.get(dimension.id)!, panelId: newPanelId,
    ...(dimension.pointAId ? { pointAId: idMap.get(dimension.pointAId)! } : {}),
    ...(dimension.pointBId ? { pointBId: idMap.get(dimension.pointBId)! } : {}),
    ...(dimension.segmentId ? { segmentId: idMap.get(dimension.segmentId)! } : {}),
  }));
  panelCopy.constraints = source.constraints.map((constraint) => ({
    ...clone(constraint), id: idMap.get(constraint.id)!, panelId: newPanelId,
    pointAId: idMap.get(constraint.pointAId)!, pointBId: idMap.get(constraint.pointBId)!,
  }));
  for (const point of pointIds) next.points.push({ ...clone(point), id: idMap.get(point.id)!, panelId: newPanelId });
  for (const segment of segmentIds) next.segments.push({
    ...clone(segment), id: idMap.get(segment.id)!, panelId: newPanelId,
    startPointId: idMap.get(segment.startPointId)!, endPointId: idMap.get(segment.endPointId)!,
    ...(segment.kind === "arc" ? { centerPointId: idMap.get(segment.centerPointId)! } : {}),
  });
  next.panels.push(panelCopy);
  return { document: next, panelId: newPanelId };
}

export function deletePanel(document: PatternDocument, panelId: EntityId): PatternDocument {
  const next = clone(document);
  findPanel(next, panelId);
  next.panels = next.panels.filter((panel) => panel.id !== panelId);
  next.points = next.points.filter((point) => point.panelId !== panelId);
  next.segments = next.segments.filter((segment) => segment.panelId !== panelId);
  return next;
}

export function canonicalPatternDocument(document: PatternDocument): PatternDocument {
  const result = validatePatternDocument(document);
  if (!result.valid) throw new PatternCadValidationError(result.diagnostics);
  return clone(document);
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

export function serializePatternDocument(document: PatternDocument): string {
  return canonicalJson(canonicalPatternDocument(document));
}

export function deserializePatternDocument(serialized: string): PatternDocument {
  let parsed: unknown;
  try {
    parsed = JSON.parse(serialized);
  } catch {
    throw new PatternCadError("invalid-document", "serialized document is not valid JSON");
  }
  const result = validatePatternDocument(parsed as PatternDocument);
  if (!result.valid) throw new PatternCadValidationError(result.diagnostics);
  return clone(parsed as PatternDocument);
}
