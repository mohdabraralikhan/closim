// G16 component framework: reusable garment components with explicit
// dependencies, fingerprint-tracked regeneration, and a derivation registry
// that other tracks extend WITHOUT editing this file:
//
//   registerComponentDerivation("collar-stand", deriveCollarStand)
//
// Built-in derivations cover the integration garment (collar/cuff bands,
// patch pockets, buttons, buttonholes). They are intentionally simple
// rectangular/marking constructions; richer parametric versions register
// under new type names and supersede them.
//
// Components derive panels + G8B seams + production marking descriptors.
// Identity survives regeneration when sources are unchanged; source edits
// invalidate explicitly (never silently).

import {
  PatternCadError,
  deletePanel,
  validatePatternDocument,
  type EntityId,
  type PatternDocument,
} from "../pattern/cad.js";
import { dist, type Vec2 } from "../cad/geom.js";
import { getLoop, getPanel, getPoint, getSegment } from "../cad/queries.js";
import { buildPanelFromRing } from "../cad/draft.js";
import type { Seam } from "../garment/sewing.js";
import type { DerivedFold, DerivedNotch } from "./features.js";
import type { DrillMarkKind, InternalLineKind } from "../cad/production.js";

export const COMPONENT_SET_VERSION = 1;

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function finiteVec(v: unknown): v is Vec2 {
  return Array.isArray(v) && v.length === 2 && v.every((n) => Number.isFinite(n));
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

export interface ComponentFeature {
  kind: "component";
  id: string;
  /** Host panel (placement surface / source owner). */
  panelId: EntityId;
  componentType: string;
  params: Record<string, unknown>;
  dependencies: EntityId[];
  derived: { panels: EntityId[]; seams: string[]; markings: string[] };
  depFingerprint: string;
  status: "current" | "stale" | "invalid";
  statusReason?: string;
}

export interface ComponentSet {
  version: typeof COMPONENT_SET_VERSION;
  nextIndex: number;
  components: ComponentFeature[];
}

export function createComponentSet(): ComponentSet {
  return { version: COMPONENT_SET_VERSION, nextIndex: 1, components: [] };
}

export interface DerivedDrill {
  panelId: EntityId;
  pos: Vec2;
  mark: DrillMarkKind;
  radiusM: number;
}

export interface DerivedInternal {
  panelId: EntityId;
  points: Vec2[];
  kind: InternalLineKind;
  label?: string;
}

export interface DerivedAnnotation {
  panelId: EntityId;
  pos: Vec2;
  note: string;
}

export interface DerivedGrainline {
  panelId: EntityId;
  from: Vec2;
  to: Vec2;
}

export interface ComponentDerivation {
  document: PatternDocument;
  panels: EntityId[];
  seams: Seam[];
  folds: DerivedFold[];
  notches: DerivedNotch[];
  drills: DerivedDrill[];
  internals: DerivedInternal[];
  annotations: DerivedAnnotation[];
  grainlines: DerivedGrainline[];
}

export type ComponentDerivationFn = (
  doc: PatternDocument,
  feature: ComponentFeature,
) => ComponentDerivation;

const registry = new Map<string, ComponentDerivationFn>();

/** Register a derivation for a component type (B/C/D extension point). */
export function registerComponentDerivation(type: string, fn: ComponentDerivationFn): void {
  if (registry.has(type)) {
    throw new PatternCadError("duplicate-id", `component derivation for '${type}' is already registered`);
  }
  registry.set(type, fn);
}

export function registeredComponentTypes(): string[] {
  return [...registry.keys()].sort();
}

function requireParams(feature: ComponentFeature, names: string[]): void {
  for (const name of names) {
    if (feature.params[name] === undefined) {
      throw new PatternCadError("missing-reference", `component '${feature.id}' misses param '${name}'`, feature.id);
    }
  }
}

function finiteParam(feature: ComponentFeature, name: string, positive = false): number {
  const value = feature.params[name];
  if (typeof value !== "number" || !Number.isFinite(value) || (positive && !(value > 0))) {
    throw new PatternCadError("invalid-transform", `component '${feature.id}' param '${name}' invalid`, feature.id);
  }
  return value;
}

function edgeEndpoints(doc: PatternDocument, panelId: EntityId, segmentId: EntityId): [Vec2, Vec2] {
  const segment = getSegment(doc, segmentId, panelId);
  const a = getPoint(doc, segment.startPointId, panelId);
  const b = getPoint(doc, segment.endPointId, panelId);
  return [[a.x, a.y], [b.x, b.y]];
}

function stitchCountFor(lengthM: number): number {
  return Math.max(2, Math.round(lengthM / 0.01) + 1);
}

/** Rectangular band (collar/cuff) from a source edge: length = edge, height = param. */
function deriveBand(doc: PatternDocument, feature: ComponentFeature, label: string): ComponentDerivation {
  requireParams(feature, ["edgeLoopId", "edgeSegmentId", "heightM", "name"]);
  const edgeLoopId = feature.params.edgeLoopId as string;
  const edgeSegmentId = feature.params.edgeSegmentId as string;
  const heightM = finiteParam(feature, "heightM", true);
  const name = String(feature.params.name);
  const loop = getLoop(doc, feature.panelId, edgeLoopId);
  if (!loop.segmentIds.includes(edgeSegmentId)) {
    throw new PatternCadError("missing-reference", `edge '${edgeSegmentId}' is not in loop '${edgeLoopId}'`, feature.id);
  }
  const [a, b] = edgeEndpoints(doc, feature.panelId, edgeSegmentId);
  const lengthM = dist(a, b);
  if (!(lengthM > 1e-9)) throw new PatternCadError("zero-length-edge", "band source edge has zero length", feature.id);
  const empty: ComponentDerivation = {
    document: doc, panels: [], seams: [], folds: [], notches: [],
    drills: [], internals: [], annotations: [], grainlines: [],
  };
  const ring: Vec2[] = [[0, 0], [lengthM, 0], [lengthM, heightM], [0, heightM]];
  const built = buildPanelFromRing(empty.document, name, "cotton", 0, ring);
  const panelId = built.panelId;
  const bandLoop = getPanel(built.document, panelId).boundaryLoops[0];
  const bottomId = bandLoop.segmentIds[0];
  const seamId = `seam/${feature.id}`;
  return {
    document: built.document,
    panels: [panelId],
    seams: [{
      id: seamId,
      sideA: { panelId, loopId: bandLoop.id, segmentIds: [bottomId], reversed: false },
      sideB: { panelId: feature.panelId, loopId: edgeLoopId, segmentIds: [edgeSegmentId], reversed: true },
      stitchCount: stitchCountFor(lengthM),
      metadata: { kind: `${label}-attachment` },
    }],
    folds: [],
    notches: [{
      panelId, loopId: bandLoop.id, segmentId: bandLoop.segmentIds[2],
      t: 0.5, kind: "single", depthM: 0.004,
    }],
    drills: [],
    internals: [],
    annotations: [],
    grainlines: [{ panelId, from: [lengthM / 2, heightM * 0.2], to: [lengthM / 2, heightM * 0.8] }],
  };
}

function derivePatchPocket(doc: PatternDocument, feature: ComponentFeature): ComponentDerivation {
  requireParams(feature, ["center", "widthM", "heightM", "name"]);
  const center = feature.params.center;
  if (!finiteVec(center)) throw new PatternCadError("invalid-transform", `component '${feature.id}' center invalid`, feature.id);
  const widthM = finiteParam(feature, "widthM", true);
  const heightM = finiteParam(feature, "heightM", true);
  const name = String(feature.params.name);
  const [cx, cy] = center;
  const x0 = cx - widthM / 2, x1 = cx + widthM / 2;
  const y0 = cy - heightM / 2, y1 = cy + heightM / 2;
  const built = buildPanelFromRing(doc, name, "cotton", 0, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]);
  return {
    document: built.document,
    panels: [built.panelId],
    seams: [],
    folds: [],
    notches: [],
    drills: [],
    internals: [{
      panelId: feature.panelId,
      points: [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]],
      kind: "pocket",
      label: `pocket placement ${feature.id}`,
    }],
    annotations: [],
    grainlines: [],
  };
}

function deriveButton(doc: PatternDocument, feature: ComponentFeature): ComponentDerivation {
  requireParams(feature, ["pos", "diameterM"]);
  const pos = feature.params.pos;
  if (!finiteVec(pos)) throw new PatternCadError("invalid-transform", `component '${feature.id}' pos invalid`, feature.id);
  const diameterM = finiteParam(feature, "diameterM", true);
  getPanel(doc, feature.panelId);
  return {
    document: doc,
    panels: [],
    seams: [],
    folds: [],
    notches: [],
    drills: [{ panelId: feature.panelId, pos: [pos[0], pos[1]], mark: "point", radiusM: 0.002 }],
    internals: [],
    annotations: [{ panelId: feature.panelId, pos: [pos[0], pos[1]], note: `button ${diameterM * 1000}mm ${feature.id}` }],
    grainlines: [],
  };
}

function deriveButtonhole(doc: PatternDocument, feature: ComponentFeature): ComponentDerivation {
  requireParams(feature, ["pos", "lengthM", "angleRad"]);
  const pos = feature.params.pos;
  if (!finiteVec(pos)) throw new PatternCadError("invalid-transform", `component '${feature.id}' pos invalid`, feature.id);
  const lengthM = finiteParam(feature, "lengthM", true);
  const angleRad = finiteParam(feature, "angleRad");
  getPanel(doc, feature.panelId);
  const dx = (Math.cos(angleRad) * lengthM) / 2, dy = (Math.sin(angleRad) * lengthM) / 2;
  return {
    document: doc,
    panels: [],
    seams: [],
    folds: [],
    notches: [],
    drills: [],
    internals: [{
      panelId: feature.panelId,
      points: [[pos[0] - dx, pos[1] - dy], [pos[0] + dx, pos[1] + dy]],
      kind: "buttonhole",
      label: `buttonhole ${feature.id}`,
    }],
    annotations: [],
    grainlines: [],
  };
}

registerComponentDerivation("collar-band", (doc, feature) => deriveBand(doc, feature, "collar"));
registerComponentDerivation("cuff-band", (doc, feature) => deriveBand(doc, feature, "cuff"));
registerComponentDerivation("patch-pocket", derivePatchPocket);
registerComponentDerivation("button", deriveButton);
registerComponentDerivation("buttonhole", deriveButtonhole);

// ---------------------------------------------------------------------------
// Component set ops
// ---------------------------------------------------------------------------

export function addComponent(
  set: ComponentSet,
  panelId: EntityId,
  componentType: string,
  params: Record<string, unknown>,
  dependencies: EntityId[],
  note?: string,
): { set: ComponentSet; id: string } {
  if (!registry.has(componentType)) {
    throw new PatternCadError("missing-reference", `no derivation registered for component type '${componentType}'`);
  }
  if (typeof params !== "object" || params === null) {
    throw new PatternCadError("invalid-transform", "component params must be an object");
  }
  const next = clone(set);
  const id = `construction/component/${String(next.nextIndex++).padStart(8, "0")}`;
  next.components.push({
    kind: "component", id, panelId, componentType,
    params: clone(params), dependencies: [...dependencies],
    derived: { panels: [], seams: [], markings: [] },
    depFingerprint: "",
    status: "stale",
    ...(note ? { statusReason: note } : {}),
  });
  return { set: next, id };
}

export function removeComponent(set: ComponentSet, id: string): ComponentSet {
  if (!set.components.some((c) => c.id === id)) {
    throw new PatternCadError("missing-reference", `component '${id}' does not exist`, id);
  }
  const next = clone(set);
  next.components = next.components.filter((c) => c.id !== id);
  return next;
}

export function updateComponentParams(set: ComponentSet, id: string, params: Record<string, unknown>): ComponentSet {
  const next = clone(set);
  const feature = next.components.find((c) => c.id === id);
  if (!feature) throw new PatternCadError("missing-reference", `component '${id}' does not exist`, id);
  feature.params = { ...feature.params, ...clone(params) };
  feature.status = "stale";
  feature.statusReason = "params updated";
  return next;
}

// ---------------------------------------------------------------------------
// Fingerprints + validation
// ---------------------------------------------------------------------------

function collectDependencyPoints(doc: PatternDocument, dependencies: EntityId[]): Vec2[] {
  const points: Vec2[] = [];
  const seen = new Set<string>();
  const pushPoint = (panelId: string, pointId: string): void => {
    const key = `${panelId}/${pointId}`;
    if (seen.has(key)) return;
    seen.add(key);
    try {
      const p = getPoint(doc, pointId, panelId);
      points.push([p.x, p.y]);
    } catch {
      // Missing refs are reported by validation, not here.
    }
  };
  for (const id of dependencies) {
    const point = doc.points.find((p) => p.id === id);
    if (point) {
      pushPoint(point.panelId, point.id);
      continue;
    }
    const segment = doc.segments.find((s) => s.id === id);
    if (segment) {
      pushPoint(segment.panelId, segment.startPointId);
      pushPoint(segment.panelId, segment.endPointId);
      continue;
    }
    const panel = doc.panels.find((p) => p.id === id);
    if (panel) {
      for (const pt of doc.points) {
        if (pt.panelId === id) pushPoint(id, pt.id);
      }
      continue;
    }
    for (const p of doc.panels) {
      const loop = p.boundaryLoops.find((l) => l.id === id);
      if (loop) {
        for (const sid of loop.segmentIds) {
          const s = doc.segments.find((q) => q.id === sid);
          if (s) {
            pushPoint(s.panelId, s.startPointId);
            pushPoint(s.panelId, s.endPointId);
          }
        }
      }
    }
  }
  return points.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
}

export function fingerprintDependencies(doc: PatternDocument, dependencies: EntityId[]): string {
  const pts = collectDependencyPoints(doc, dependencies);
  const s = pts.map((p) => `${p[0].toFixed(9)},${p[1].toFixed(9)}`).join(";");
  let hash = 2166136261;
  for (let i = 0; i < s.length; i++) {
    hash ^= s.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return `fp/${(hash >>> 0).toString(16)}/${pts.length}`;
}

export interface ComponentStatus {
  id: string;
  componentType: string;
  ok: boolean;
  status: ComponentFeature["status"];
  reason?: string;
}

export function validateComponents(doc: PatternDocument, set: ComponentSet): ComponentStatus[] {
  return set.components.map((feature) => {
    if (!doc.panels.some((p) => p.id === feature.panelId)) {
      return { id: feature.id, componentType: feature.componentType, ok: false, status: "invalid", reason: `host panel '${feature.panelId}' is gone` };
    }
    // Every dependency must still resolve to something.
    for (const id of feature.dependencies) {
      const known =
        doc.points.some((p) => p.id === id) ||
        doc.segments.some((s) => s.id === id) ||
        doc.panels.some((p) => p.id === id) ||
        doc.panels.some((p) => p.boundaryLoops.some((l) => l.id === id));
      if (!known) {
        return { id: feature.id, componentType: feature.componentType, ok: false, status: "invalid", reason: `dependency '${id}' is gone` };
      }
    }
    if (!registry.has(feature.componentType)) {
      return { id: feature.id, componentType: feature.componentType, ok: false, status: "invalid", reason: `no derivation for '${feature.componentType}'` };
    }
    const current = fingerprintDependencies(doc, feature.dependencies);
    if (feature.status === "current" && feature.depFingerprint === current &&
      feature.derived.panels.every((pid) => doc.panels.some((p) => p.id === pid))) {
      return { id: feature.id, componentType: feature.componentType, ok: true, status: "current" };
    }
    return {
      id: feature.id, componentType: feature.componentType, ok: true, status: "stale",
      reason: feature.depFingerprint !== current ? "sources changed" : "never derived",
    };
  });
}

// ---------------------------------------------------------------------------
// Derivation across the set
// ---------------------------------------------------------------------------

export interface ComponentDerivationReport {
  document: PatternDocument;
  seams: Seam[];
  folds: DerivedFold[];
  notches: DerivedNotch[];
  drills: DerivedDrill[];
  internals: DerivedInternal[];
  annotations: DerivedAnnotation[];
  grainlines: DerivedGrainline[];
  applied: string[];
  failed: Array<{ id: string; reason: string }>;
  invalidatedSeamIds: string[];
  set: ComponentSet;
}

export interface SeamRef {
  id: string;
  sideA: { panelId: string };
  sideB: { panelId: string };
}

/**
 * Regenerate every stale component in stored order. Unchanged components are
 * skipped (identity preserved); changed ones delete their old derived panels
 * and re-derive. Caller-owned seams touching deleted panels are reported.
 */
export function deriveComponents(
  doc: PatternDocument,
  set: ComponentSet,
  seams: ReadonlyArray<SeamRef> = [],
): ComponentDerivationReport {
  let nextSet = clone(set);
  let next = clone(doc);
  const out: ComponentDerivationReport = {
    document: next, seams: [], folds: [], notches: [], drills: [],
    internals: [], annotations: [], grainlines: [],
    applied: [], failed: [], invalidatedSeamIds: [], set: nextSet,
  };
  for (const feature of nextSet.components) {
    const status = validateComponents(next, { ...nextSet, components: [feature] })[0];
    if (!status.ok) {
      out.failed.push({ id: feature.id, reason: status.reason ?? "invalid" });
      feature.status = "invalid";
      feature.statusReason = status.reason;
      continue;
    }
    if (status.status === "current") {
      out.applied.push(feature.id);
      continue;
    }
    // Remove previously derived panels (fresh derivation follows). Caller
    // seams touching replaced panel ids are reported whether or not the old
    // panels are present in this input (stateless re-derivation is normal).
    const replaced = [...feature.derived.panels];
    for (const pid of replaced) {
      if (next.panels.some((p) => p.id === pid)) {
        next = deletePanel(next, pid);
      }
    }
    for (const seam of seams) {
      if ((replaced.some((pid) => seam.sideA.panelId === pid || seam.sideB.panelId === pid)) &&
        !out.invalidatedSeamIds.includes(seam.id)) {
        out.invalidatedSeamIds.push(seam.id);
      }
    }
    try {
      const fn = registry.get(feature.componentType)!;
      const derived = fn(next, feature);
      next = derived.document;
      feature.derived = {
        panels: [...derived.panels],
        seams: derived.seams.map((s) => s.id),
        markings: [],
      };
      feature.depFingerprint = fingerprintDependencies(next, feature.dependencies);
      feature.status = "current";
      delete feature.statusReason;
      out.seams.push(...derived.seams);
      out.folds.push(...derived.folds);
      out.notches.push(...derived.notches);
      out.drills.push(...derived.drills);
      out.internals.push(...derived.internals);
      out.annotations.push(...derived.annotations);
      out.grainlines.push(...derived.grainlines);
      out.applied.push(feature.id);
    } catch (error) {
      feature.status = "invalid";
      feature.statusReason = error instanceof Error ? error.message : String(error);
      out.failed.push({ id: feature.id, reason: feature.statusReason });
    }
  }
  const validation = validatePatternDocument(next);
  if (!validation.valid) {
    const first = validation.diagnostics[0];
    throw new PatternCadError(first.code, `component derivation produced invalid geometry: ${first.message}`, first.entityId);
  }
  out.document = next;
  out.set = nextSet;
  return out;
}

// ---------------------------------------------------------------------------
// Persistence
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

export function serializeComponentSet(set: ComponentSet): string {
  return canonicalJson(set);
}

export function deserializeComponentSet(serialized: string): ComponentSet {
  let parsed: unknown;
  try {
    parsed = JSON.parse(serialized);
  } catch {
    throw new PatternCadError("invalid-document", "serialized component set is not valid JSON");
  }
  const set = parsed as ComponentSet;
  if (!set || typeof set !== "object" || set.version !== COMPONENT_SET_VERSION || !Array.isArray(set.components)) {
    throw new PatternCadError("invalid-document", "component set shape or version is invalid");
  }
  return clone(set);
}
