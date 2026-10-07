// Grade-rule evaluation and deterministic derived-geometry generation.
// master geometry + size definition + grading rules -> GradedPattern.
// The master document is never mutated: derivation clones through CAD ops.
import {
  movePoint,
  validatePatternDocument,
  type EntityId,
} from "../pattern/cad.js";
import { resolveAnchor } from "./anchors.js";
import {
  buildDeltaContext,
  checkRuleConsistency,
  checkSeams,
  composeDelta,
  type DeltaContext,
} from "./engine.js";
import { canonicalJson } from "./serialize.js";
import { validateGradingDocument } from "./validate.js";
import type {
  AnchorResolution,
  GradedPattern,
  GradingDerivationReport,
  GradingDiagnostic,
  GradingDocument,
  RuleApplication,
  Vec2,
} from "./types.js";
import { GradingError } from "./types.js";

export { activeOrderedSizes, ruleDeltaForSize } from "./engine.js";
import { activeOrderedSizes, ruleDeltaForSize } from "./engine.js";

export interface DeriveOptions {
  /** When true (default) a grading point without a rule aborts derivation; otherwise it is reported as a diagnostic and left ungraded. */
  strict?: boolean;
  /** Max admissible |lengthA - lengthB| per seam on derived geometry (quality diagnostic, never blocks). */
  seamMismatchM?: number;
}

// ---------------------------------------------------------------------------
// Fingerprinting (deterministic invalidation)
// ---------------------------------------------------------------------------

const FNV_OFFSET = 0xcbf29ce484222325n;
const FNV_PRIME = 0x100000001b3n;
const FNV_MASK = 0xffffffffffffffffn;

function fnv1a64(input: string): string {
  let hash = FNV_OFFSET;
  for (let i = 0; i < input.length; i++) {
    hash ^= BigInt(input.charCodeAt(i));
    hash = (hash * FNV_PRIME) & FNV_MASK;
  }
  return hash.toString(16).padStart(16, "0");
}

/** Everything derived geometry depends on: master geometry, sizes, rules, grading points, seams, production data. */
export function fingerprintGradingDocument(document: GradingDocument): string {
  const inputs = {
    masterId: document.master.id,
    masterDocument: document.master.document,
    sizeSet: document.sizeSet,
    ruleTable: document.ruleTable,
    gradingPoints: document.gradingPoints,
    seams: document.seams,
    production: document.production ?? null,
    markingAnchors: document.markingAnchors ?? null,
  };
  return fnv1a64(canonicalJson(inputs));
}

export function isStale(document: GradingDocument, graded: GradedPattern): boolean {
  return graded.masterId !== document.master.id ||
    graded.document.id !== document.master.document.id ||
    graded.sourceFingerprint !== fingerprintGradingDocument(document);
}

// ---------------------------------------------------------------------------
// Derivation
// ---------------------------------------------------------------------------

export function deriveSize(document: GradingDocument, sizeId: EntityId, options: DeriveOptions = {}):
  { graded: GradedPattern; report: GradingDerivationReport } {
  const strict = options.strict ?? true;
  const blocking = validateGradingDocument(document);
  if (blocking.length > 0) {
    throw new GradingError("invalid-document", `grading document is invalid: ${blocking[0].message}`);
  }
  const size = document.sizeSet.sizes.find((candidate) => candidate.id === sizeId);
  if (!size) throw new GradingError("unknown-size", `size '${sizeId}' does not exist in the size set`, sizeId);
  if (!size.active) throw new GradingError("unknown-size", `size '${sizeId}' is inactive and cannot be graded`, sizeId);

  const activeSizes = activeOrderedSizes(document.sizeSet);
  const context: DeltaContext = buildDeltaContext(document);

  const applied: RuleApplication[] = [];
  const evaluated: GradingDerivationReport["evaluated"] = [];
  const diagnostics: GradingDiagnostic[] = [];
  const moves: Array<{ panelId: EntityId; pointId: EntityId; position: Vec2 }> = [];
  const masterResolutions = new Map<EntityId, AnchorResolution>();

  // Pass 1: resolve anchors against the master and evaluate rules.
  for (const gradingPoint of document.gradingPoints) {
    const resolution = resolveAnchor(document.master.document, gradingPoint.anchor, document.seams);
    masterResolutions.set(gradingPoint.id, resolution);
    const rule = context.rulesByGradingPoint.get(gradingPoint.id);
    if (resolution.displacement === "vertex") {
      const pointId = resolution.pointIds[0];
      const masterPoint = document.master.document.points.find((candidate) => candidate.id === pointId)!;
      if (!rule && !context.mirrorsByGradingPoint.has(gradingPoint.id)) {
        if (strict) {
          throw new GradingError("missing-rule", `grading point '${gradingPoint.id}' has no rule for the document`, gradingPoint.id);
        }
        diagnostics.push({ code: "missing-rule", message: `grading point '${gradingPoint.id}' has no rule; left at master position`, gradingPointId: gradingPoint.id });
        continue;
      }
      const { delta } = composeDelta(document, gradingPoint, rule, activeSizes, sizeId, context);
      moves.push({ panelId: masterPoint.panelId, pointId, position: [masterPoint.x + delta[0], masterPoint.y + delta[1]] });
    }
  }

  // Pass 2: build the derived document via validated CAD operations.
  let derived = document.master.document;
  for (const move of moves) {
    derived = movePoint(derived, move.panelId, move.pointId, move.position, "local");
  }

  // Pass 3: report applications against the derived geometry. Evaluated-only
  // anchors follow the graded curve of their endpoints, plus their own delta.
  for (const gradingPoint of document.gradingPoints) {
    const rule = context.rulesByGradingPoint.get(gradingPoint.id);
    const derivedResolution = resolveAnchor(derived, gradingPoint.anchor, document.seams);
    if (derivedResolution.displacement === "vertex") {
      if (!rule && !context.mirrorsByGradingPoint.has(gradingPoint.id)) continue;
      const pointId = derivedResolution.pointIds[0];
      const gradedPoint = derived.points.find((candidate) => candidate.id === pointId)!;
      const composed = composeDelta(document, gradingPoint, rule, activeSizes, sizeId, context);
      applied.push({
        gradingPointId: gradingPoint.id,
        ruleId: composed.mirrorOfId !== undefined ? context.rulesByGradingPoint.get(composed.mirrorOfId)!.id : rule!.id,
        mode: (composed.mirrorOfId !== undefined ? context.rulesByGradingPoint.get(composed.mirrorOfId)! : rule!).mode,
        delta: composed.delta,
        pointId,
        masterPosition: masterResolutions.get(gradingPoint.id)!.position,
        gradedPosition: [gradedPoint.x, gradedPoint.y],
        ...(composed.mirrorOfId !== undefined ? { mirrorOfId: composed.mirrorOfId } : {}),
        ...(composed.panelAdjustmentIds !== undefined ? { panelAdjustmentIds: composed.panelAdjustmentIds } : {}),
      });
    } else {
      const composed = composeDelta(document, gradingPoint, rule, activeSizes, sizeId, context);
      evaluated.push({
        gradingPointId: gradingPoint.id,
        anchor: gradingPoint.anchor,
        masterPosition: masterResolutions.get(gradingPoint.id)!.position,
        gradedPosition: [derivedResolution.position[0] + composed.delta[0], derivedResolution.position[1] + composed.delta[1]],
        pointIds: [],
        displacement: "evaluated",
      });
    }
  }

  diagnostics.push(...checkRuleConsistency(document));
  const seamCheck = checkSeams(derived, document.seams, options.seamMismatchM);
  diagnostics.push(...seamCheck.diagnostics);

  const geometry = validatePatternDocument(derived);
  if (!geometry.valid) {
    throw new GradingError(
      "invalid-document",
      `derived geometry for size '${sizeId}' is invalid: ${geometry.diagnostics.map((d) => `${d.code}: ${d.message}`).join("; ")}`,
      sizeId,
    );
  }

  const graded: GradedPattern = {
    id: `${document.master.id}/graded/${sizeId}`,
    masterId: document.master.id,
    sizeId,
    document: derived,
    sourceFingerprint: fingerprintGradingDocument(document),
  };
  return { graded, report: { sizeId, applied, evaluated, diagnostics, seams: seamCheck.rows } };
}

export function findGraded(document: GradingDocument, sizeId: EntityId): GradedPattern | undefined {
  return document.graded.find((candidate) => candidate.sizeId === sizeId);
}

/** Insert or replace the derived pattern for a size, keeping cache order stable. */
export function upsertGraded(document: GradingDocument, graded: GradedPattern): GradingDocument {
  const next = JSON.parse(JSON.stringify(document)) as GradingDocument;
  const index = next.graded.findIndex((candidate) => candidate.sizeId === graded.sizeId);
  if (index >= 0) next.graded[index] = graded;
  else next.graded.push(graded);
  return next;
}

/** Re-derive every active size in size-set order, replacing the derived cache. */
export function regenerateAll(document: GradingDocument, options: DeriveOptions = {}):
  { document: GradingDocument; reports: GradingDerivationReport[] } {
  let next = document;
  const reports: GradingDerivationReport[] = [];
  for (const size of activeOrderedSizes(document.sizeSet)) {
    const result = deriveSize(document, size.id, options);
    next = upsertGraded(next, result.graded);
    reports.push(result.report);
  }
  return { document: next, reports };
}
