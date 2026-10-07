// G11C — production measurement and validation gate.
//
// Answers "is this pattern ready for export?" with a machine-readable report.
// Layers, from hardest to softest failure:
//
//   INVALID          — any error: broken kernel geometry, dangling refs,
//                      self-intersecting allowance, seam mismatch, missing
//                      cut quantity, markings outside panels.
//   WARNINGS         — no errors but warnings: unsewn panels, missing
//                      grainlines, miter spikes, duplicate notches.
//   READY_FOR_EXPORT — clean.
//
// A warning is never silently treated as valid: the state distinguishes them
// and every diagnostic carries severity, location, and a suggested action.
// Geometry is never modified to hide a discrepancy.

import {
  validatePatternDocument,
  type EntityId,
  type PatternDocument,
} from "../pattern/cad.js";
import type { Vec2 } from "./geom.js";
import { measurePanel } from "./queries.js";
import { measureSeamLength } from "./constraints.js";
import { resolveStitchPairs, validateSeams, type Seam } from "../garment/sewing.js";
import {
  allowanceBoundary,
  validateProductionSet,
  type ProductionSet,
} from "./production.js";
import { validateMarkings } from "./markings.js";

export type ReadinessState = "INVALID" | "WARNINGS" | "READY_FOR_EXPORT";
export type Severity = "error" | "warning" | "info";

export interface ReadinessDiagnostic {
  severity: Severity;
  code: string;
  message: string;
  entityId?: string;
  panelId?: string;
  location?: Vec2;
  suggestedAction: string;
}

export interface PanelMeasurementRow {
  panelId: string;
  areaM2: number;
  perimeterM: number;
  cutQuantity: number;
  cutAreaM2: number;
}

export interface SeamMeasurementRow {
  seamId: string;
  lengthAM: number;
  lengthBM: number;
  diffM: number;
  withinTolerance: boolean;
}

export interface ReadinessReport {
  state: ReadinessState;
  diagnostics: ReadinessDiagnostic[];
  errorCount: number;
  warningCount: number;
  panels: PanelMeasurementRow[];
  totalCutAreaM2: number;
  seams: SeamMeasurementRow[];
}

export interface ReadinessTolerances {
  /** Max admissible |lengthA - lengthB| per seam (m). Default 3 mm. */
  seamMismatchM?: number;
  /** Spikes above this multiple of local allowance are warnings (G11A flags at 10x). */
  spikeFactor?: number;
}

const DEFAULT_SEAM_TOL_M = 0.003;

function push(
  list: ReadinessDiagnostic[],
  severity: Severity,
  code: string,
  message: string,
  suggestedAction: string,
  extra: { entityId?: string; panelId?: string; location?: Vec2 } = {},
): void {
  list.push({ severity, code, message, suggestedAction, ...extra });
}

/** Authoritative side lengths for one seam (arclength along each resolved stitch polyline). */
export function seamSideLengths(doc: PatternDocument, seam: Seam): { lengthAM: number; lengthBM: number } {
  return { lengthAM: seamSideLength(doc, seam, "A"), lengthBM: seamSideLength(doc, seam, "B") };
}

function seamSideLength(doc: PatternDocument, seam: Seam, side: "A" | "B"): number {
  const pairs = resolveStitchPairs(doc, seam);
  let total = 0;
  for (let i = 1; i < pairs.length; i++) {
    const a = side === "A" ? pairs[i - 1].pointA : pairs[i - 1].pointB;
    const b = side === "A" ? pairs[i].pointA : pairs[i].pointB;
    total += Math.hypot(b[0] - a[0], b[1] - a[1]);
  }
  return total;
}

export function productionReadiness(
  doc: PatternDocument,
  seams: readonly Seam[],
  set: ProductionSet,
  tolerances: ReadinessTolerances = {},
): ReadinessReport {
  const seamTolM = tolerances.seamMismatchM ?? DEFAULT_SEAM_TOL_M;
  const diagnostics: ReadinessDiagnostic[] = [];

  // 1. Kernel geometry (design layer).
  const kernel = validatePatternDocument(doc);
  for (const d of kernel.diagnostics) {
    push(diagnostics, "error", `pattern:${d.code}`, d.message,
      "Fix the pattern geometry in the 2D editor before production.",
      { entityId: d.entityId, panelId: d.panelId });
  }

  // 2. Production references.
  for (const d of validateProductionSet(doc, set)) {
    const severity: Severity = d.code === "outside-panel" ? "error" : "error";
    push(diagnostics, severity, `production:${d.code}`, d.message,
      d.code === "missing-reference"
        ? "Re-link or delete the orphaned production entity."
        : d.code === "outside-panel"
          ? "Move the marking inside its panel or fix the panel boundary."
          : "Correct the production value (allowance, quantity, or text).",
      { entityId: d.entityId });
  }

  // 3. Marking semantics.
  for (const d of validateMarkings(doc, set)) {
    const severity: Severity =
      d.code === "orphaned-marking" || d.code === "outside-panel" ? "error" : "warning";
    push(diagnostics, severity, `marking:${d.code}`, d.message,
      d.code === "orphaned-marking"
        ? "Delete the orphaned marking or restore its boundary edge."
        : d.code === "duplicate-marking"
          ? "Remove or reposition the duplicated notch."
          : d.code === "degenerate-placement"
            ? "Move the notch away from the corner vertex."
            : "Move the marking inside its panel.",
      { entityId: d.entityId });
  }

  // 4. Allowance derivation issues.
  for (const allowance of set.allowances) {
    let boundary;
    try {
      boundary = allowanceBoundary(doc, set, allowance.panelId, allowance.loopId);
    } catch {
      continue; // missing refs already reported above
    }
    for (const issue of boundary.issues) {
      if (issue.code === "self-intersection" || issue.code === "degenerate-ring") {
        push(diagnostics, "error", `allowance:${issue.code}`, issue.message,
          "Reduce the allowance, split the allowance by edge, or rework the corner.",
          { entityId: allowance.id, panelId: allowance.panelId });
      } else if (issue.code === "spike") {
        push(diagnostics, "warning", `allowance:${issue.code}`, issue.message,
          "Accept the miter, reduce the allowance at the corner, or clip the spike.",
          { entityId: allowance.id, panelId: allowance.panelId });
      } else {
        push(diagnostics, "info", `allowance:${issue.code}`, issue.message,
          "No action required; recorded for traceability.",
          { entityId: allowance.id, panelId: allowance.panelId });
      }
    }
  }

  // 5. Seams: references + paired-length comparison (never auto-fixed).
  const seamCheck = validateSeams(doc, seams);
  for (const d of seamCheck.diagnostics) {
    push(diagnostics, "error", `seam:${d.code}`, d.message,
      "Fix the seam definition (panels, loops, segments, stitch count).",
      { entityId: d.seamId });
  }
  const seamRows: SeamMeasurementRow[] = [];
  if (seamCheck.valid) {
    for (const seam of seams) {
      let lengthAM = 0, lengthBM = 0;
      try {
        lengthAM = measureSeamLength(doc, seam);
        lengthBM = seamSideLength(doc, seam, "B");
      } catch {
        push(diagnostics, "error", "seam:unmeasurable", `seam '${seam.id}' cannot be measured`,
          "Fix the seam references first.", { entityId: seam.id });
        continue;
      }
      const diffM = Math.abs(lengthAM - lengthBM);
      const withinTolerance = diffM <= seamTolM;
      seamRows.push({ seamId: seam.id, lengthAM, lengthBM, diffM, withinTolerance });
      if (!withinTolerance) {
        push(diagnostics, "error", "seam:mismatch",
          `seam '${seam.id}' sides differ by ${(diffM * 1000).toFixed(1)} mm (A=${(lengthAM * 1000).toFixed(1)} mm, B=${(lengthBM * 1000).toFixed(1)} mm, tol=${(seamTolM * 1000).toFixed(1)} mm)`,
          "Ease the longer side, re-cut, or request an intentional-mismatch waiver in a later phase (not silent).",
          { entityId: seam.id });
      }
      if (seam.stitchCount < 2) {
        push(diagnostics, "error", "seam:stitch-count", `seam '${seam.id}' needs at least 2 stitches`,
          "Raise the stitch count.", { entityId: seam.id });
      }
    }
  }

  // 6. Panels: measurements, cut quantities, grainlines, sewn-ness.
  const panels: PanelMeasurementRow[] = [];
  let totalCutAreaM2 = 0;
  const sewnPanels = new Set<string>();
  for (const seam of seams) {
    sewnPanels.add(seam.sideA.panelId);
    sewnPanels.add(seam.sideB.panelId);
  }
  for (const panel of doc.panels) {
    let areaM2 = 0, perimeterM = 0;
    try {
      const measured = measurePanel(doc, panel.id);
      areaM2 = measured.area;
      perimeterM = measured.perimeter;
    } catch {
      push(diagnostics, "error", "panel:unmeasurable", `panel '${panel.id}' cannot be measured`,
        "Fix the panel boundary first.", { panelId: panel.id });
    }
    const meta = set.panelMeta.find((m) => m.panelId === panel.id);
    if (!meta) {
      push(diagnostics, "error", "panel:missing-meta", `panel '${panel.id}' has no cut quantity`,
        "Enter a cut quantity (and section/notes) in the panel metadata.", { panelId: panel.id });
    }
    const cutQuantity = meta?.cutQuantity ?? 0;
    panels.push({ panelId: panel.id, areaM2, perimeterM, cutQuantity, cutAreaM2: areaM2 * cutQuantity });
    totalCutAreaM2 += areaM2 * cutQuantity;
    if (!set.grainlines.some((g) => g.panelId === panel.id)) {
      push(diagnostics, "warning", "panel:missing-grainline", `panel '${panel.id}' has no grainline`,
        "Add a grainline before export.", { panelId: panel.id });
    }
    if (!sewnPanels.has(panel.id)) {
      push(diagnostics, "warning", "panel:unsewn", `panel '${panel.id}' is not joined by any seam`,
        "Sew the panel, or confirm it is intentionally loose (lining, facing).", { panelId: panel.id });
    }
  }

  // Deterministic order: errors, then warnings, then info; ties by code + entity.
  const rank: Record<Severity, number> = { error: 0, warning: 1, info: 2 };
  diagnostics.sort((a, b) =>
    rank[a.severity] - rank[b.severity] ||
    (a.code < b.code ? -1 : a.code > b.code ? 1 : 0) ||
    ((a.entityId ?? "") < (b.entityId ?? "") ? -1 : 1),
  );
  const errorCount = diagnostics.filter((d) => d.severity === "error").length;
  const warningCount = diagnostics.filter((d) => d.severity === "warning").length;
  const state: ReadinessState =
    errorCount > 0 ? "INVALID" : warningCount > 0 ? "WARNINGS" : "READY_FOR_EXPORT";
  return { state, diagnostics, errorCount, warningCount, panels, totalCutAreaM2, seams: seamRows };
}

export type { EntityId };
