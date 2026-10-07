// G10C 2D->3D synchronization core.
// The pattern document is authoritative; the 3D garment is derived. This module
// classifies what a pattern change means for the derived state so the app can
// avoid rebuild work, decide whether interaction state survives, and fall back
// to full reconstruction safely.

import type { PatternDocument, PatternPanel } from "../pattern/cad.js";
import type { AssembledGarment, PanelPlacement } from "../garment/assembly.js";
import { rebuildGarment, type GarmentProject, type RebuildResult } from "../garment/project.js";

export type RebuildLevel =
  | "none"
  | "geometry-refresh"
  | "panel-remesh"
  | "topology-rebuild"
  | "full-assembly";

export interface RebuildClassification {
  level: RebuildLevel;
  /** Panels whose content changed (empty for level "none"). */
  panelIds: string[];
}

const LEVEL_ORDER: RebuildLevel[] = ["none", "geometry-refresh", "panel-remesh", "topology-rebuild", "full-assembly"];

function maxLevel(a: RebuildLevel, b: RebuildLevel): RebuildLevel {
  return LEVEL_ORDER.indexOf(a) >= LEVEL_ORDER.indexOf(b) ? a : b;
}

function transformEquals(a: PatternPanel, b: PatternPanel): boolean {
  const ta = a.transform;
  const tb = b.transform;
  return ta.translation[0] === tb.translation[0]
    && ta.translation[1] === tb.translation[1]
    && ta.rotationRad === tb.rotationRad
    && ta.scale[0] === tb.scale[0]
    && ta.scale[1] === tb.scale[1];
}

function boundaryEquals(a: PatternPanel, b: PatternPanel): boolean {
  if (a.boundaryLoops.length !== b.boundaryLoops.length) return false;
  for (let i = 0; i < a.boundaryLoops.length; i++) {
    const la = a.boundaryLoops[i];
    const lb = b.boundaryLoops[i];
    if (la.id !== lb.id || la.role !== lb.role || la.orientation !== lb.orientation) return false;
    if (la.segmentIds.length !== lb.segmentIds.length) return false;
    for (let s = 0; s < la.segmentIds.length; s++) {
      if (la.segmentIds[s] !== lb.segmentIds[s]) return false;
    }
  }
  return true;
}

function geometryEquals(before: PatternDocument, after: PatternDocument, panelA: PatternPanel, panelB: PatternPanel): boolean {
  const pointsB = new Map(
    after.points.filter((p) => p.panelId === panelB.id).map((p) => [p.id, p]),
  );
  const pointsA = before.points.filter((p) => p.panelId === panelA.id);
  if (pointsA.length !== pointsB.size) return false;
  for (const pa of pointsA) {
    const pb = pointsB.get(pa.id);
    if (!pb) return false;
    if (pa.x !== pb.x || pa.y !== pb.y || pa.role !== pb.role) return false;
  }
  const segmentsB = new Map(
    after.segments.filter((s) => s.panelId === panelB.id).map((s) => [s.id, s]),
  );
  const segmentsA = before.segments.filter((s) => s.panelId === panelA.id);
  if (segmentsA.length !== segmentsB.size) return false;
  for (const sa of segmentsA) {
    const sb = segmentsB.get(sa.id);
    if (!sb) return false;
    if (sa.kind !== sb.kind) return false;
    if (sa.startPointId !== sb.startPointId || sa.endPointId !== sb.endPointId) return false;
    if (sa.kind === "arc" && sb.kind === "arc") {
      if (sa.centerPointId !== sb.centerPointId || sa.sweepRad !== sb.sweepRad) return false;
    }
  }
  return true;
}

function panelChanged(before: PatternDocument, after: PatternDocument, panelA: PatternPanel, panelB: PatternPanel): "transform" | "boundary" | null {
  if (!transformEquals(panelA, panelB)) return "transform";
  if (!boundaryEquals(panelA, panelB)) return "boundary";
  if (!geometryEquals(before, after, panelA, panelB)) return "boundary";
  return null;
}

/**
 * Classify a pattern edit. Level semantics:
 *   none             -> derived state untouched
 *   geometry-refresh -> panel transform changed; triangulation counts are
 *                       stable, mesh indices keep their meaning
 *   panel-remesh     -> boundary geometry changed; mesh indices are invalid
 *   topology-rebuild -> panels added/removed; seam/panel references may break
 *   full-assembly    -> structural change that requires full reconstruction
 */
export function classifyPatternChange(before: PatternDocument, after: PatternDocument): RebuildClassification {
  const beforePanels = new Map(before.panels.map((p) => [p.id, p]));
  const afterPanels = new Map(after.panels.map((p) => [p.id, p]));
  const changed: string[] = [];
  let level: RebuildLevel = "none";
  for (const [id, panelBefore] of beforePanels) {
    const panelAfter = afterPanels.get(id);
    if (!panelAfter) {
      changed.push(id);
      level = maxLevel(level, "topology-rebuild");
      continue;
    }
    const change = panelChanged(before, after, panelBefore, panelAfter);
    if (change === "transform") {
      changed.push(id);
      level = maxLevel(level, "geometry-refresh");
    } else if (change === "boundary") {
      changed.push(id);
      level = maxLevel(level, "panel-remesh");
    }
  }
  for (const id of afterPanels.keys()) {
    if (!beforePanels.has(id)) {
      changed.push(id);
      level = maxLevel(level, "topology-rebuild");
    }
  }
  if (before.id !== after.id) level = "full-assembly";
  return { level, panelIds: changed };
}

export interface PanelVertexRange {
  panelId: string;
  vertexStart: number;
  vertexCount: number;
  triangleStart: number;
  triangleCount: number;
}

/** 3D panel ID -> mesh ranges (the 2D panel ID equals the 3D panel ID). */
export function panelVertexRanges(assembled: AssembledGarment): Map<string, PanelVertexRange> {
  const map = new Map<string, PanelVertexRange>();
  for (const r of assembled.panelRanges) {
    map.set(r.panelId, {
      panelId: r.panelId,
      vertexStart: r.vertexStart,
      vertexCount: r.vertexCount,
      triangleStart: r.triangleStart,
      triangleCount: r.triangleCount,
    });
  }
  return map;
}

export function vertexPanelId(assembled: AssembledGarment, vertex: number): string | null {
  for (const r of assembled.panelRanges) {
    if (vertex >= r.vertexStart && vertex < r.vertexStart + r.vertexCount) return r.panelId;
  }
  return null;
}

export interface SeamMapping {
  seamId: string;
  weldIndices: number[];
}

/** Seam ID -> weld pair indices in the assembled mesh (stable seam IDs). */
export function seamWeldMap(assembled: AssembledGarment): Map<string, SeamMapping> {
  const map = new Map<string, SeamMapping>();
  assembled.weldPairs.forEach((weld, index) => {
    let entry = map.get(weld.seamId);
    if (!entry) {
      entry = { seamId: weld.seamId, weldIndices: [] };
      map.set(weld.seamId, entry);
    }
    entry.weldIndices.push(index);
  });
  return map;
}

/**
 * Whether interaction state (pins, vertex regions) can carry across a rebuild.
 * Mesh indices are only meaningful when the rebuild did not remesh the panels
 * actually present in both documents; the safe default is to drop them.
 */
export function interactionStateSurvives(classification: RebuildClassification): boolean {
  return classification.level === "none";
}

export interface RebuildPlan {
  project: GarmentProject;
  classification: RebuildClassification;
  result: RebuildResult;
}

/**
 * The G10C update workflow for a validated pattern edit:
 * classify -> rebuild -> (caller: workspace.setGarment + sim attach).
 * Throws when the edited pattern is invalid; the caller keeps the old state.
 */
export function rebuildWithPlan(
  project: GarmentProject,
  editedPattern: PatternDocument,
  assembleOpts?: Parameters<typeof rebuildGarment>[1],
): RebuildPlan {
  const classification = classifyPatternChange(project.pattern, editedPattern);
  const nextProject: GarmentProject = {
    ...JSON.parse(JSON.stringify(project)) as GarmentProject,
    pattern: editedPattern,
  };
  nextProject.metadata.revision = project.metadata.revision + 1;
  const result = rebuildGarment(nextProject, assembleOpts);
  return { project: nextProject, classification, result };
}

/** Apply a placement array to a plan's project (assembly-level transform only). */
export function placementsOf(project: GarmentProject): PanelPlacement[] {
  return project.placements;
}
