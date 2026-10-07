// G12B — grade rule engine. Turns grading definitions into deterministic
// target-size deltas and derived production sets.
//
// Explicit precedence, no ambiguous conflicts:
//   effective delta (grading point, size) =
//     point rule delta   — per-size (master→size) or accumulated transition
//                        — OR a mirrored source rule delta with the axis
//                          component negated (mirror bindings and own rules
//                          are mutually exclusive; chains are rejected)
//   + Σ panel adjustments for (anchor panel, size)   — at most one per pair
//
// Quality signals (seam mismatch, inconsistent size direction, markings
// outside the derived panel) are reported as diagnostics/issues and never
// silently ignored; only invalid geometry blocks derivation.
import { pointInPanel } from "../cad/queries.js";
import { resolveStitchPairs, type Seam } from "../garment/sewing.js";
import type { EntityId, PatternDocument } from "../pattern/cad.js";
import type {
  DerivedProduction,
  GradeRule,
  GradingDiagnostic,
  GradingDocument,
  GradingPoint,
  MirrorBinding,
  PanelAdjustment,
  ProductionIssue,
  SeamCheckRow,
  SizeDefinition,
  Vec2,
} from "./types.js";
import { GradingError } from "./types.js";

export const DEFAULT_SEAM_TOLERANCE_M = 0.003;

export function activeOrderedSizes(sizeSet: GradingDocument["sizeSet"]): SizeDefinition[] {
  return sizeSet.sizes.filter((size) => size.active);
}

/** Resolve the raw rule delta that applies to one grading point for one target size. */
export function ruleDeltaForSize(rule: GradeRule, sizes: readonly SizeDefinition[], targetSizeId: EntityId): Vec2 {
  const readDelta = (sizeId: EntityId): Vec2 => {
    const delta = rule.deltas[sizeId];
    if (!Array.isArray(delta) || delta.length !== 2 || !delta.every((v) => Number.isFinite(v))) {
      throw new GradingError("missing-rule", `rule '${rule.id}' has no finite delta for size '${sizeId}'`, rule.gradingPointId);
    }
    return [delta[0], delta[1]];
  };
  if (rule.mode === "per-size") {
    return readDelta(targetSizeId);
  }
  // transition: accumulate from the master through every active size up to and including the target.
  const index = sizes.findIndex((size) => size.id === targetSizeId);
  if (index < 0) {
    throw new GradingError("unknown-size", `size '${targetSizeId}' is not part of the active size order`, rule.gradingPointId);
  }
  const total: Vec2 = [0, 0];
  for (let i = 0; i <= index; i++) {
    const delta = readDelta(sizes[i].id);
    total[0] += delta[0];
    total[1] += delta[1];
  }
  return total;
}

/** Per-derivation resolution context built once and shared by all passes. */
export interface DeltaContext {
  rulesByGradingPoint: Map<EntityId, GradeRule>;
  mirrorsByGradingPoint: Map<EntityId, MirrorBinding>;
  panelAdjustments: PanelAdjustment[];
}

export function buildDeltaContext(document: GradingDocument): DeltaContext {
  const rulesByGradingPoint = new Map<EntityId, GradeRule>();
  for (const rule of document.ruleTable.rules) {
    rulesByGradingPoint.set(rule.gradingPointId, rule);
  }
  const mirrorsByGradingPoint = new Map<EntityId, MirrorBinding>();
  for (const mirror of document.ruleTable.mirrors ?? []) {
    mirrorsByGradingPoint.set(mirror.gradingPointId, mirror);
  }
  return {
    rulesByGradingPoint,
    mirrorsByGradingPoint,
    panelAdjustments: document.ruleTable.panelAdjustments ?? [],
  };
}

/** The panel a grading point's anchor lives on (seam-point anchors resolve via their segment). */
export function gradingPointPanel(document: GradingDocument, gradingPoint: GradingPoint): EntityId {
  const anchor = gradingPoint.anchor;
  if (anchor.kind === "seam-point") {
    const segment = document.master.document.segments.find((candidate) => candidate.id === anchor.segmentId);
    if (!segment) {
      throw new GradingError("unknown-entity", `grading point '${gradingPoint.id}' references unknown segment '${anchor.segmentId}'`, gradingPoint.id);
    }
    return segment.panelId;
  }
  return anchor.panelId;
}

/**
 * Compose the effective delta for one grading point and target size.
 * Requires `rule` to be present unless the point is mirror-bound; vertex
 * callers enforce missing-rule semantics before calling.
 */
export function composeDelta(
  document: GradingDocument,
  gradingPoint: GradingPoint,
  rule: GradeRule | undefined,
  activeSizes: readonly SizeDefinition[],
  targetSizeId: EntityId,
  context: DeltaContext,
): { delta: Vec2; mirrorOfId?: EntityId; panelAdjustmentIds?: EntityId[] } {
  const mirror = context.mirrorsByGradingPoint.get(gradingPoint.id);
  let delta: Vec2;
  let mirrorOfId: EntityId | undefined;
  if (mirror) {
    const sourceRule = context.rulesByGradingPoint.get(mirror.sourceGradingPointId);
    if (!sourceRule) {
      throw new GradingError("invalid-rule", `mirror '${mirror.id}' source grading point '${mirror.sourceGradingPointId}' has no rule`, gradingPoint.id);
    }
    const source = ruleDeltaForSize(sourceRule, activeSizes, targetSizeId);
    // Normalize -0 away: negating a zero component must not leak -0 into
    // documents, reports or serialized output.
    delta = mirror.axis === "x"
      ? [source[0], source[1] === 0 ? 0 : -source[1]]
      : [source[0] === 0 ? 0 : -source[0], source[1]];
    mirrorOfId = mirror.sourceGradingPointId;
  } else if (rule) {
    delta = ruleDeltaForSize(rule, activeSizes, targetSizeId);
  } else {
    delta = [0, 0];
  }
  const panelId = gradingPointPanel(document, gradingPoint);
  const adjustmentIds: EntityId[] = [];
  for (const adjustment of context.panelAdjustments) {
    if (adjustment.panelId === panelId && adjustment.sizeId === targetSizeId) {
      delta = [delta[0] + adjustment.delta[0], delta[1] + adjustment.delta[1]];
      adjustmentIds.push(adjustment.id);
    }
  }
  return {
    delta,
    ...(mirrorOfId !== undefined ? { mirrorOfId } : {}),
    ...(adjustmentIds.length > 0 ? { panelAdjustmentIds: adjustmentIds } : {}),
  };
}

/** Standalone effective-delta lookup (used for marking displacement). */
export function effectiveDeltaForPoint(document: GradingDocument, gradingPointId: EntityId, sizeId: EntityId): Vec2 {
  const gradingPoint = document.gradingPoints.find((candidate) => candidate.id === gradingPointId);
  if (!gradingPoint) {
    throw new GradingError("unknown-entity", `grading point '${gradingPointId}' does not exist`, gradingPointId);
  }
  const context = buildDeltaContext(document);
  return composeDelta(
    document,
    gradingPoint,
    context.rulesByGradingPoint.get(gradingPointId),
    activeOrderedSizes(document.sizeSet),
    sizeId,
    context,
  ).delta;
}

/**
 * Transition-rule direction audit: flags components whose sign flips between
 * consecutive active sizes (an inconsistent grade direction). Warnings only —
 * they never block derivation.
 */
export function checkRuleConsistency(document: GradingDocument): GradingDiagnostic[] {
  const diagnostics: GradingDiagnostic[] = [];
  const sizes = activeOrderedSizes(document.sizeSet);
  for (const rule of document.ruleTable.rules) {
    if (rule.mode !== "transition" || sizes.length < 3) continue;
    for (let i = 2; i < sizes.length; i++) {
      const earlier = rule.deltas[sizes[i - 1].id];
      const later = rule.deltas[sizes[i].id];
      if (!Array.isArray(earlier) || !Array.isArray(later)) continue;
      for (let component = 0; component < 2; component++) {
        if (earlier[component] !== 0 && later[component] !== 0 && Math.sign(earlier[component]) !== Math.sign(later[component])) {
          diagnostics.push({
            code: "inconsistent-direction",
            message: `rule '${rule.id}' flips direction on ${component === 0 ? "dx" : "dy"} between '${sizes[i - 1].id}' and '${sizes[i].id}'`,
            ruleId: rule.id,
            sizeId: sizes[i].id,
          });
        }
      }
    }
  }
  return diagnostics;
}

/** Paired seam side lengths (arclength along each resolved stitch polyline). */
export function seamSideLengths(doc: PatternDocument, seam: Seam): { lengthAM: number; lengthBM: number } {
  const pairs = resolveStitchPairs(doc, seam);
  let lengthAM = 0;
  let lengthBM = 0;
  for (let i = 1; i < pairs.length; i++) {
    lengthAM += Math.hypot(pairs[i].pointA[0] - pairs[i - 1].pointA[0], pairs[i].pointA[1] - pairs[i - 1].pointA[1]);
    lengthBM += Math.hypot(pairs[i].pointB[0] - pairs[i - 1].pointB[0], pairs[i].pointB[1] - pairs[i - 1].pointB[1]);
  }
  return { lengthAM, lengthBM };
}

export function checkSeams(
  doc: PatternDocument,
  seams: readonly Seam[],
  toleranceM = DEFAULT_SEAM_TOLERANCE_M,
): { rows: SeamCheckRow[]; diagnostics: GradingDiagnostic[] } {
  const rows: SeamCheckRow[] = [];
  const diagnostics: GradingDiagnostic[] = [];
  for (const seam of seams) {
    try {
      const { lengthAM, lengthBM } = seamSideLengths(doc, seam);
      const diffM = Math.abs(lengthAM - lengthBM);
      const withinTolerance = diffM <= toleranceM;
      rows.push({ seamId: seam.id, lengthAM, lengthBM, diffM, withinTolerance });
      if (!withinTolerance) {
        diagnostics.push({
          code: "seam-mismatch",
          message: `seam '${seam.id}' sides differ by ${(diffM * 1000).toFixed(1)} mm after grading (tolerance ${(toleranceM * 1000).toFixed(1)} mm)`,
        });
      }
    } catch {
      diagnostics.push({
        code: "seam-mismatch",
        message: `seam '${seam.id}' cannot be measured on the derived geometry`,
      });
    }
  }
  return { rows, diagnostics };
}

// ---------------------------------------------------------------------------
// Production derivation
// ---------------------------------------------------------------------------

type ProductionSidecar = NonNullable<GradingDocument["production"]>;

interface MarkingPlacement {
  kind: string;
  entityId: EntityId;
  panelId: EntityId;
  positions: Vec2[];
  markingAnchorId?: EntityId;
}

/**
 * Derive the per-size production set. T/ID-referenced entities (allowances,
 * notches, cut lines, panel meta) copy verbatim — they key on stable IDs and
 * arclength fractions. Position-based markings copy verbatim unless a marking
 * anchor binds them to a grading point; every marked position is then checked
 * against the derived panel (issues, never silent drops or moves).
 */
export function deriveProductionSet(document: GradingDocument, gradedDocument: PatternDocument, sizeId: EntityId): DerivedProduction {
  const set = document.production;
  if (!set) {
    throw new GradingError("invalid-argument", "grading document has no production set");
  }
  const next = JSON.parse(JSON.stringify(set)) as ProductionSidecar;
  const issues: ProductionIssue[] = [];
  const displaced: MarkingPlacement[] = [];

  const shift = (p: Vec2, d: Vec2): Vec2 => [p[0] + d[0], p[1] + d[1]];

  for (const anchor of document.markingAnchors ?? []) {
    let delta: Vec2;
    try {
      delta = effectiveDeltaForPoint(document, anchor.gradingPointId, sizeId);
    } catch (error) {
      issues.push({
        code: "unknown-marking-anchor",
        message: `marking anchor '${anchor.id}' grading point cannot be resolved: ${error instanceof Error ? error.message : String(error)}`,
        entityId: anchor.entityId,
        markingAnchorId: anchor.id,
      });
      continue;
    }
    switch (anchor.kind) {
      case "grainline": {
        const entity = next.grainlines.find((g) => g.id === anchor.entityId);
        if (!entity) { issues.push(orphanAnchor(anchor)); break; }
        entity.from = shift(entity.from, delta);
        entity.to = shift(entity.to, delta);
        displaced.push({ kind: "grainline", entityId: entity.id, panelId: entity.panelId, positions: [entity.from, entity.to], markingAnchorId: anchor.id });
        break;
      }
      case "fold": {
        const entity = next.folds.find((f) => f.id === anchor.entityId);
        if (!entity) { issues.push(orphanAnchor(anchor)); break; }
        entity.a = shift(entity.a, delta);
        entity.b = shift(entity.b, delta);
        displaced.push({ kind: "fold", entityId: entity.id, panelId: entity.panelId, positions: [entity.a, entity.b], markingAnchorId: anchor.id });
        break;
      }
      case "drill": {
        const entity = next.drills.find((d) => d.id === anchor.entityId);
        if (!entity) { issues.push(orphanAnchor(anchor)); break; }
        entity.pos = shift(entity.pos, delta);
        displaced.push({ kind: "drill", entityId: entity.id, panelId: entity.panelId, positions: [entity.pos], markingAnchorId: anchor.id });
        break;
      }
      case "annotation": {
        const entity = next.annotations.find((a) => a.id === anchor.entityId);
        if (!entity) { issues.push(orphanAnchor(anchor)); break; }
        entity.pos = shift(entity.pos, delta);
        displaced.push({ kind: "annotation", entityId: entity.id, panelId: entity.panelId, positions: [entity.pos], markingAnchorId: anchor.id });
        break;
      }
      case "labelRegion": {
        const entity = next.labelRegions.find((l) => l.id === anchor.entityId);
        if (!entity) { issues.push(orphanAnchor(anchor)); break; }
        entity.min = shift(entity.min, delta);
        entity.max = shift(entity.max, delta);
        displaced.push({ kind: "labelRegion", entityId: entity.id, panelId: entity.panelId, positions: [entity.min, entity.max], markingAnchorId: anchor.id });
        break;
      }
    }
  }

  const anchored = new Set((document.markingAnchors ?? []).map((a) => `${a.kind}/${a.entityId}`));
  const placements: MarkingPlacement[] = [
    ...displaced,
    ...verbatimPlacements(next, anchored),
  ];
  for (const placement of placements) {
    for (const position of placement.positions) {
      if (!pointInPanel(gradedDocument, placement.panelId, position)) {
        issues.push({
          code: "outside-panel",
          message: `${placement.kind} '${placement.entityId}' falls outside panel '${placement.panelId}' for size '${sizeId}'`,
          entityId: placement.entityId,
          ...(placement.markingAnchorId !== undefined ? { markingAnchorId: placement.markingAnchorId } : {}),
        });
      }
    }
  }

  return { set: next, issues };
}

function orphanAnchor(anchor: { id: EntityId; entityId: EntityId; kind: string }): ProductionIssue {
  return {
    code: "unknown-marking-anchor",
    message: `marking anchor '${anchor.id}' references unknown ${anchor.kind} '${anchor.entityId}'`,
    entityId: anchor.entityId,
    markingAnchorId: anchor.id,
  };
}

function verbatimPlacements(set: ProductionSidecar, anchored: Set<string>): MarkingPlacement[] {
  const placements: MarkingPlacement[] = [];
  const skip = (kind: string, entityId: EntityId): boolean => anchored.has(`${kind}/${entityId}`);
  for (const grainline of set.grainlines) {
    if (!skip("grainline", grainline.id)) placements.push({ kind: "grainline", entityId: grainline.id, panelId: grainline.panelId, positions: [grainline.from, grainline.to] });
  }
  for (const fold of set.folds) {
    if (!skip("fold", fold.id)) placements.push({ kind: "fold", entityId: fold.id, panelId: fold.panelId, positions: [fold.a, fold.b] });
  }
  for (const drill of set.drills) {
    if (!skip("drill", drill.id)) placements.push({ kind: "drill", entityId: drill.id, panelId: drill.panelId, positions: [drill.pos] });
  }
  for (const annotation of set.annotations) {
    if (!skip("annotation", annotation.id)) placements.push({ kind: "annotation", entityId: annotation.id, panelId: annotation.panelId, positions: [annotation.pos] });
  }
  for (const labelRegion of set.labelRegions) {
    if (!skip("labelRegion", labelRegion.id)) placements.push({ kind: "labelRegion", entityId: labelRegion.id, panelId: labelRegion.panelId, positions: [labelRegion.min, labelRegion.max] });
  }
  return placements;
}
