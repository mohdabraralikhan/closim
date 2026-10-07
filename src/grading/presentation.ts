// G12D — multi-size presentation core (headless, renderer-ready).
//
// Pure data model a renderer binds to: per-size outline layers, grading-point
// markers, rule vectors, size labels, pairwise comparison, picking and the
// master-redirect editing contract. No DOM, no rendering library, no cloth
// simulation — generating or displaying any number of sizes never leaves
// pure pattern geometry.
//
// Nested view: grading is anchored to fixed master positions, so sizes are
// concentric by construction; layers carry a deterministic nested order and
// the master layer can be highlighted on top. No marker-nesting algorithm is
// implied or required.
import { localToGlobal, type EntityId, type PatternDocument, type PatternPanel } from "../pattern/cad.js";
import { evaluateSegmentPoint, findPanel } from "./anchors.js";
import { activeOrderedSizes, deriveSize, fingerprintGradingDocument } from "./derive.js";
import { seamSideLengths } from "./engine.js";
import type { GradingAnchor, GradingDiagnostic, GradingDocument, Vec2 } from "./types.js";
import { GradingError } from "./types.js";

export type SizeViewMode = "active" | "overlay" | "nested";

export type SizeSelectionTarget = "master" | EntityId;

export interface SizeSelection {
  /** "master" or a size ID — selection always names both entity and size. */
  sizeId: SizeSelectionTarget;
  kind: "point" | "segment" | "panel";
  id: EntityId;
}

export interface SizeViewOptions {
  mode: SizeViewMode;
  /** Active size for "active" mode; defaults to the base size. */
  activeSizeId?: EntityId;
  /** Per-size visibility; hidden sizes keep their layer and render order. */
  visibleSizeIds: EntityId[] | "all";
  showMaster: boolean;
  showGradingPoints: boolean;
  showRuleVectors: boolean;
  showLabels: boolean;
  /** Second size for pairwise comparison (compared against the active size). */
  comparisonSizeId?: EntityId;
}

export interface SizeViewPanelOutline {
  panelId: EntityId;
  loopId: EntityId;
  /** Closed outline in global coordinates; arcs sampled into chords. */
  outline: Vec2[];
}

export interface SizeViewLayer {
  sizeId: SizeSelectionTarget;
  label: string;
  visible: boolean;
  /** Render order for nested views: 0 is the base size, master is -1. */
  nestedOrder: number;
  panels: SizeViewPanelOutline[];
  gradingPoints: Array<{ gradingPointId: EntityId; position: Vec2; displacement: "vertex" | "evaluated" }>;
  /** Master→graded arrows per applied rule (global coordinates). */
  ruleVectors: Array<{ gradingPointId: EntityId; ruleId: EntityId; from: Vec2; to: Vec2 }>;
  /** Quality + validity diagnostics for this size (never truncated silently). */
  diagnostics: GradingDiagnostic[];
  invalid: boolean;
}

export interface SizeViewLabel {
  sizeId: SizeSelectionTarget;
  text: string;
  position: Vec2;
}

export interface SizeComparison {
  sizeIdA: EntityId;
  sizeIdB: EntityId;
  pointDisplacements: Array<{ gradingPointId: EntityId; from: Vec2; to: Vec2; distanceM: number }>;
  seamLengthDeltas: Array<{ seamId: EntityId; lengthAM: number; lengthBM: number; diffM: number }>;
  measurementDeltas: Array<{ measurementId: EntityId; valueAM: number; valueBM: number; deltaM: number }>;
  maxPointDistanceM: number;
}

export interface SizeView {
  mode: SizeViewMode;
  layers: SizeViewLayer[];
  masterLayer: SizeViewLayer | null;
  labels: SizeViewLabel[];
  comparison: SizeComparison | null;
}

const ARC_SAMPLES = 24;

function labelFor(document: GradingDocument, sizeId: SizeSelectionTarget): string {
  if (sizeId === "master") return document.master.name;
  return document.sizeSet.sizes.find((size) => size.id === sizeId)?.label ?? sizeId;
}

function outerLoopId(panel: PatternPanel): EntityId {
  const loop = panel.boundaryLoops.find((candidate) => candidate.role === "outer") ?? panel.boundaryLoops[0];
  if (!loop) throw new GradingError("unknown-entity", `panel '${panel.id}' has no boundary loop`, panel.id);
  return loop.id;
}

/** Closed panel outline in global coordinates; every arc is sampled into chords. */
export function panelOutline(document: PatternDocument, panelId: EntityId, loopId?: EntityId): Vec2[] {
  const panel: PatternPanel = findPanel(document, panelId);
  const loop = loopId
    ? panel.boundaryLoops.find((candidate) => candidate.id === loopId)
    : (panel.boundaryLoops.find((candidate) => candidate.role === "outer") ?? panel.boundaryLoops[0]);
  if (!loop) throw new GradingError("unknown-entity", `panel '${panelId}' has no boundary loop`, panelId);
  const outline: Vec2[] = [];
  for (const segmentId of loop.segmentIds) {
    const segment = document.segments.find((candidate) => candidate.id === segmentId);
    if (!segment || segment.panelId !== panelId) {
      throw new GradingError("unknown-entity", `boundary segment '${segmentId}' does not belong to panel '${panelId}'`, segmentId);
    }
    const samples = segment.kind === "arc" ? ARC_SAMPLES : 1;
    for (let i = 0; i < samples; i++) {
      const local = evaluateSegmentPoint(document, segmentId, i / samples);
      outline.push(localToGlobal(panel, local));
    }
  }
  return outline;
}

function panelOfPoint(doc: PatternDocument, pointId: EntityId): PatternPanel {
  const point = doc.points.find((candidate) => candidate.id === pointId);
  if (!point) throw new GradingError("unknown-entity", `point '${pointId}' does not exist`, pointId);
  return findPanel(doc, point.panelId);
}

function panelOfAnchor(doc: PatternDocument, anchor: GradingAnchor): PatternPanel {
  if (anchor.kind === "seam-point") {
    const segment = doc.segments.find((candidate) => candidate.id === anchor.segmentId);
    if (!segment) throw new GradingError("unknown-entity", `segment '${anchor.segmentId}' does not exist`, anchor.segmentId);
    const pointId = anchor.endpoint === "start" ? segment.startPointId : segment.endPointId;
    return panelOfPoint(doc, pointId);
  }
  return findPanel(doc, anchor.panelId);
}

function toGlobal(panel: PatternPanel | null, local: Vec2): Vec2 {
  return panel ? localToGlobal(panel, local) : local;
}

function masterOutlinePanels(document: GradingDocument): SizeViewPanelOutline[] {
  return document.master.document.panels.map((panel) => ({
    panelId: panel.id,
    loopId: outerLoopId(panel),
    outline: panelOutline(document.master.document, panel.id),
  }));
}

interface DerivedLayerData {
  panels: SizeViewPanelOutline[];
  gradingPoints: SizeViewLayer["gradingPoints"];
  ruleVectors: SizeViewLayer["ruleVectors"];
  diagnostics: GradingDiagnostic[];
  invalid: boolean;
}

function deriveLayerData(document: GradingDocument, sizeId: EntityId): DerivedLayerData {
  try {
    const { graded, report } = deriveSize(document, sizeId);
    const derived = graded.document;
    const panels = derived.panels.map((panel) => ({
      panelId: panel.id,
      loopId: outerLoopId(panel),
      outline: panelOutline(derived, panel.id),
    }));
    const gradingPoints: DerivedLayerData["gradingPoints"] = [];
    for (const row of report.applied) {
      gradingPoints.push({
        gradingPointId: row.gradingPointId,
        position: toGlobal(panelOfPoint(derived, row.pointId), row.gradedPosition),
        displacement: "vertex",
      });
    }
    for (const row of report.evaluated) {
      gradingPoints.push({
        gradingPointId: row.gradingPointId,
        position: toGlobal(panelOfAnchor(document.master.document, row.anchor), row.gradedPosition),
        displacement: "evaluated",
      });
    }
    const ruleVectors = report.applied.map((row) => {
      const panel = panelOfPoint(document.master.document, row.pointId);
      return {
        gradingPointId: row.gradingPointId,
        ruleId: row.ruleId,
        from: toGlobal(panel, row.masterPosition),
        to: toGlobal(panel, row.gradedPosition),
      };
    });
    return { panels, gradingPoints, ruleVectors, diagnostics: report.diagnostics, invalid: false };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return {
      panels: [],
      gradingPoints: [],
      ruleVectors: [],
      diagnostics: [{ code: "invalid-document", message: `size '${sizeId}' cannot be derived: ${message}` }],
      invalid: true,
    };
  }
}

export function defaultSizeViewOptions(): SizeViewOptions {
  return {
    mode: "overlay",
    visibleSizeIds: "all",
    showMaster: true,
    showGradingPoints: false,
    showRuleVectors: false,
    showLabels: true,
  };
}

/** Resolve the pattern document a selection target names ("master" or size). */
export function documentForSize(document: GradingDocument, sizeId: SizeSelectionTarget): PatternDocument {
  if (sizeId === "master") return document.master.document;
  const size = document.sizeSet.sizes.find((candidate) => candidate.id === sizeId);
  if (!size) throw new GradingError("unknown-size", `size '${sizeId}' does not exist in the size set`, sizeId);
  const cached = document.graded.find((candidate) => candidate.sizeId === sizeId);
  if (cached && cached.sourceFingerprint === fingerprintGradingDocument(document)) return cached.document;
  return deriveSize(document, sizeId).graded.document;
}

/** Build the complete renderer-ready multi-size view. */
export function createSizeView(document: GradingDocument, options: SizeViewOptions): SizeView {
  const ordered = activeOrderedSizes(document.sizeSet);
  if (ordered.length === 0) {
    throw new GradingError("unknown-size", "size set contains no active sizes to present");
  }
  const activeSizeId = options.activeSizeId ?? document.sizeSet.baseSizeId;
  if (!activeSizeId || !ordered.some((size) => size.id === activeSizeId)) {
    throw new GradingError("unknown-size", `active size '${String(activeSizeId)}' is not an active size`, activeSizeId ?? undefined);
  }
  const isVisible = (sizeId: EntityId): boolean =>
    options.visibleSizeIds === "all" || options.visibleSizeIds.includes(sizeId);
  const drawIds = options.mode === "active" ? [activeSizeId] : ordered.map((size) => size.id);

  const layers: SizeViewLayer[] = [];
  ordered.forEach((size, index) => {
    if (!drawIds.includes(size.id)) return;
    const data = deriveLayerData(document, size.id);
    layers.push({
      sizeId: size.id,
      label: size.label,
      visible: isVisible(size.id),
      nestedOrder: index,
      panels: data.panels,
      gradingPoints: options.showGradingPoints ? data.gradingPoints : [],
      ruleVectors: options.showRuleVectors ? data.ruleVectors : [],
      diagnostics: data.diagnostics,
      invalid: data.invalid,
    });
  });

  const masterLayer: SizeViewLayer | null = options.showMaster
    ? {
        sizeId: "master",
        label: labelFor(document, "master"),
        visible: true,
        nestedOrder: -1,
        panels: masterOutlinePanels(document),
        gradingPoints: [],
        ruleVectors: [],
        diagnostics: [],
        invalid: false,
      }
    : null;

  const labels: SizeViewLabel[] = options.showLabels
    ? layers.filter((layer) => layer.visible && layer.panels.length > 0).map((layer) => ({
        sizeId: layer.sizeId,
        text: layer.label,
        position: bboxTopCenter(layer.panels.map((panel) => panel.outline)),
      }))
    : [];

  const comparisonSizeId = options.comparisonSizeId;
  const comparison = comparisonSizeId && comparisonSizeId !== activeSizeId
    ? compareSizes(document, activeSizeId, comparisonSizeId)
    : null;

  return { mode: options.mode, layers, masterLayer, labels, comparison };
}

function bboxTopCenter(outlines: Vec2[][]): Vec2 {
  let minX = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const outline of outlines) {
    for (const [x, y] of outline) {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }
  }
  if (!Number.isFinite(minX)) return [0, 0];
  return [(minX + maxX) / 2, maxY];
}

// ---------------------------------------------------------------------------
// Comparison, picking, editing contract
// ---------------------------------------------------------------------------

function gradedPointPositions(document: GradingDocument, sizeId: EntityId): Map<EntityId, { position: Vec2; panel: PatternPanel | null }> {
  const { report } = deriveSize(document, sizeId);
  const positions = new Map<EntityId, { position: Vec2; panel: PatternPanel | null }>();
  for (const row of report.applied) {
    positions.set(row.gradingPointId, {
      position: row.gradedPosition,
      panel: panelOfPoint(document.master.document, row.pointId),
    });
  }
  for (const row of report.evaluated) {
    positions.set(row.gradingPointId, {
      position: row.gradedPosition,
      panel: panelOfAnchor(document.master.document, row.anchor),
    });
  }
  return positions;
}

/** Pairwise size comparison: point displacements, seam length deltas, measurement deltas. */
export function compareSizes(document: GradingDocument, sizeIdA: EntityId, sizeIdB: EntityId): SizeComparison {
  const positionsA = gradedPointPositions(document, sizeIdA);
  const positionsB = gradedPointPositions(document, sizeIdB);
  const pointDisplacements: SizeComparison["pointDisplacements"] = [];
  let maxPointDistanceM = 0;
  for (const [gradingPointId, a] of positionsA) {
    const b = positionsB.get(gradingPointId);
    if (!b) continue;
    const distanceM = Math.hypot(b.position[0] - a.position[0], b.position[1] - a.position[1]);
    maxPointDistanceM = Math.max(maxPointDistanceM, distanceM);
    pointDisplacements.push({ gradingPointId, from: a.position, to: b.position, distanceM });
  }
  const seamLengthDeltas: SizeComparison["seamLengthDeltas"] = [];
  for (const seam of document.seams) {
    try {
      const docA = deriveSize(document, sizeIdA).graded.document;
      const docB = deriveSize(document, sizeIdB).graded.document;
      const a = seamSideLengths(docA, seam);
      const b = seamSideLengths(docB, seam);
      const lengthAM = (a.lengthAM + a.lengthBM) / 2;
      const lengthBM = (b.lengthAM + b.lengthBM) / 2;
      seamLengthDeltas.push({ seamId: seam.id, lengthAM, lengthBM, diffM: Math.abs(lengthAM - lengthBM) });
    } catch {
      // Unmeasurable seam on either size: skipped, derivation diagnostics carry the reason.
    }
  }
  const measurementDeltas: SizeComparison["measurementDeltas"] = [];
  const sizeA = document.sizeSet.sizes.find((size) => size.id === sizeIdA);
  const sizeB = document.sizeSet.sizes.find((size) => size.id === sizeIdB);
  if (sizeA && sizeB) {
    for (const definition of document.sizeSet.measurementDefinitions ?? []) {
      const valueA = sizeA.measurements.find((entry) => entry.measurementId === definition.id);
      const valueB = sizeB.measurements.find((entry) => entry.measurementId === definition.id);
      if (!valueA || !valueB) continue;
      measurementDeltas.push({
        measurementId: definition.id,
        valueAM: valueA.valueM,
        valueBM: valueB.valueM,
        deltaM: valueB.valueM - valueA.valueM,
      });
    }
  }
  return { sizeIdA, sizeIdB, pointDisplacements, seamLengthDeltas, measurementDeltas, maxPointDistanceM };
}

/** Pick the nearest point, boundary segment or panel of one specific size. */
export function pickAtPosition(
  document: GradingDocument,
  sizeId: SizeSelectionTarget,
  position: Vec2,
  toleranceM: number,
): SizeSelection | null {
  const doc = documentForSize(document, sizeId);
  const candidates: Array<{ kind: SizeSelection["kind"]; id: EntityId; distance: number }> = [];
  const consider = (kind: SizeSelection["kind"], id: EntityId, distance: number): void => {
    if (distance <= toleranceM) candidates.push({ kind, id, distance });
  };
  for (const panel of doc.panels) {
    for (const point of doc.points) {
      if (point.panelId !== panel.id) continue;
      const global = localToGlobal(panel, [point.x, point.y]);
      consider("point", point.id, Math.hypot(global[0] - position[0], global[1] - position[1]));
    }
    for (const loop of panel.boundaryLoops) {
      for (const segmentId of loop.segmentIds) {
        const segment = doc.segments.find((candidate) => candidate.id === segmentId);
        if (!segment) continue;
        const samples = segment.kind === "arc" ? ARC_SAMPLES + 1 : 33;
        for (let i = 0; i < samples; i++) {
          const global = localToGlobal(panel, evaluateSegmentPoint(doc, segmentId, i / (samples - 1)));
          consider("segment", segmentId, Math.hypot(global[0] - position[0], global[1] - position[1]));
        }
      }
    }
  }
  if (candidates.length === 0) return null;
  const best = candidates.reduce((acc, candidate) => (candidate.distance < acc.distance ? candidate : acc));
  return { sizeId, kind: best.kind, id: best.id };
}

/**
 * Editing contract: derived sizes are not independently editable. Selecting
 * on a derived size redirects editing to the master; after a master edit the
 * caller regenerates all sizes.
 */
export function editIntent(selection: SizeSelection): { editable: boolean; redirectSizeId?: "master"; reason?: string } {
  if (selection.sizeId === "master") return { editable: true };
  return {
    editable: false,
    redirectSizeId: "master",
    reason: `selection names size '${selection.sizeId}'; derived sizes are not independently editable — edit the master and regenerate`,
  };
}
