// G11B — notches, markings, and construction metadata.
//
// G11A (`production.ts`) owns entity storage and allowance derivation. This
// module owns marking *interpretation*: notch tick geometry for presentation
// and export, orphan/invalidation tracking across boundary edits, grainline
// construction helpers, label-field builders, panel numbering, and marking
// validation (orphans, duplicates, escapes, degenerate placement).
//
// Representation and validation only — no fashion-specific automation.

import {
  PatternCadError,
  type EntityId,
  type PatternDocument,
} from "../pattern/cad.js";
import { distToSegment } from "../pattern/pattern-geometry.js";
import type { Vec2 } from "./geom.js";
import {
  getPanel,
  getSegment,
  pointInPanel,
  sampleLoopLocal,
} from "./queries.js";
import {
  notchFrame,
  type Grainline,
  type Notch,
  type ProductionSet,
} from "./production.js";

const EPS_M = 1e-9;

export type MarkingStatus = "ok" | "orphaned";

export interface NotchTick {
  /** Tick segment in panel-local metres (edge point -> outward tip). */
  from: Vec2;
  to: Vec2;
}

/** Tick geometry for a notch: single = 1 tick, double = 2, custom = 1 wide tick. */
export function notchTicks(doc: PatternDocument, notch: Notch): { ticks: NotchTick[]; frame: ReturnType<typeof notchFrame> } {
  getSegment(doc, notch.segmentId, notch.panelId); // orphan guard: throws when gone
  const frame = notchFrame(doc, notch);
  const tip = (base: Vec2): Vec2 => [
    base[0] + frame.outward[0] * notch.depthM,
    base[1] + frame.outward[1] * notch.depthM,
  ];
  if (notch.kind === "double") {
    const gap = notch.depthM / 2;
    const b1: Vec2 = [frame.pos[0] - frame.tangent[0] * gap, frame.pos[1] - frame.tangent[1] * gap];
    const b2: Vec2 = [frame.pos[0] + frame.tangent[0] * gap, frame.pos[1] + frame.tangent[1] * gap];
    return { ticks: [{ from: b1, to: tip(b1) }, { from: b2, to: tip(b2) }], frame };
  }
  return { ticks: [{ from: frame.pos, to: tip(frame.pos) }], frame };
}

/** Per-notch reference status: "orphaned" when the edge/loop/panel is gone. */
export function notchStatus(doc: PatternDocument, notch: Notch): MarkingStatus {
  const panel = doc.panels.find((p) => p.id === notch.panelId);
  const loop = panel?.boundaryLoops.find((l) => l.id === notch.loopId);
  if (!loop) return "orphaned";
  if (!loop.segmentIds.includes(notch.segmentId)) return "orphaned";
  return "ok";
}

/** All notch ids whose boundary edge no longer exists (edit invalidation). */
export function orphanedNotchIds(doc: PatternDocument, set: ProductionSet): string[] {
  return set.notches.filter((n) => notchStatus(doc, n) === "orphaned").map((n) => n.id);
}

/** Panel ids referenced by production data but missing from the document. */
export function orphanedPanelIds(doc: PatternDocument, set: ProductionSet): string[] {
  const alive = new Set(doc.panels.map((p) => p.id));
  const refs = new Set<string>();
  for (const e of [...set.allowances, ...set.grainlines, ...set.folds, ...set.drills,
    ...set.internals, ...set.cutLines, ...set.annotations, ...set.labelRegions]) {
    refs.add(e.panelId);
  }
  for (const n of set.notches) refs.add(n.panelId);
  for (const m of set.panelMeta) refs.add(m.panelId);
  return [...refs].filter((id) => !alive.has(id)).sort();
}

// ---------------------------------------------------------------------------
// Grainline construction
// ---------------------------------------------------------------------------

/** Centroid of a panel's outer loop samples (deterministic). */
export function panelCentroid(doc: PatternDocument, panelId: EntityId): Vec2 {
  const panel = getPanel(doc, panelId);
  const outer = panel.boundaryLoops.find((l) => l.role === "outer");
  if (!outer) throw new PatternCadError("open-boundary", "panel has no outer loop", panelId);
  const pts = sampleLoopLocal(doc, panelId, outer.id);
  if (pts.length === 0) throw new PatternCadError("degenerate-panel", "outer loop samples to nothing", panelId);
  let sx = 0, sy = 0;
  for (const p of pts) {
    sx += p[0];
    sy += p[1];
  }
  return [sx / pts.length, sy / pts.length];
}

/**
 * Centered grainline through the panel centroid at an absolute angle.
 * Default: straight grain (+y), length = half the outer-loop bbox height
 * (clamped to (0, bbox diagonal]).
 */
export function centeredGrainline(
  doc: PatternDocument,
  panelId: EntityId,
  angleRad = Math.PI / 2,
  lengthM?: number,
): { from: Vec2; to: Vec2 } {
  if (!Number.isFinite(angleRad)) {
    throw new PatternCadError("invalid-transform", "grainline angle must be finite", panelId);
  }
  const center = panelCentroid(doc, panelId);
  const panel = getPanel(doc, panelId);
  const outer = panel.boundaryLoops.find((l) => l.role === "outer")!;
  const pts = sampleLoopLocal(doc, panelId, outer.id);
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const p of pts) {
    if (p[0] < minX) minX = p[0];
    if (p[0] > maxX) maxX = p[0];
    if (p[1] < minY) minY = p[1];
    if (p[1] > maxY) maxY = p[1];
  }
  const diag = Math.hypot(maxX - minX, maxY - minY);
  const total = lengthM ?? (maxY - minY) / 2;
  if (!Number.isFinite(total) || !(total > 0) || total > diag) {
    throw new PatternCadError("invalid-transform", "grainline length must be positive and fit the panel", panelId);
  }
  const dir: Vec2 = [Math.cos(angleRad), Math.sin(angleRad)];
  return {
    from: [center[0] - (dir[0] * total) / 2, center[1] - (dir[1] * total) / 2],
    to: [center[0] + (dir[0] * total) / 2, center[1] + (dir[1] * total) / 2],
  };
}

/** Cross-grain mark: same center and length, rotated 90°. */
export function crossGrainline(grainline: Grainline): { from: Vec2; to: Vec2 } {
  const cx = (grainline.from[0] + grainline.to[0]) / 2;
  const cy = (grainline.from[1] + grainline.to[1]) / 2;
  const dx = grainline.to[0] - grainline.from[0];
  const dy = grainline.to[1] - grainline.from[1];
  const half: Vec2 = [-dy / 2, dx / 2];
  return { from: [cx - half[0], cy - half[1]], to: [cx + half[0], cy + half[1]] };
}

// ---------------------------------------------------------------------------
// Metadata: numbering + label fields
// ---------------------------------------------------------------------------

/** Deterministic 1-based panel numbers in document order. */
export function assignPanelNumbers(doc: PatternDocument): Record<string, number> {
  const out: Record<string, number> = {};
  doc.panels.forEach((p, i) => {
    out[p.id] = i + 1;
  });
  return out;
}

export interface LabelFieldInput {
  garmentName: string;
  panelName: string;
  panelNumber: number;
  size?: string;
  cutQuantity: number;
  material: string;
  mirror?: string;
  notes?: string;
}

/** Canonical label fields for a LabelRegion (fixed key order). */
export function buildLabelFields(input: LabelFieldInput): Record<string, string> {
  if (!input.garmentName || !input.panelName || !input.material) {
    throw new PatternCadError("invalid-transform", "label needs garment, panel, and material names");
  }
  if (!Number.isInteger(input.panelNumber) || input.panelNumber < 1) {
    throw new PatternCadError("invalid-transform", "label needs a positive panel number");
  }
  if (!Number.isInteger(input.cutQuantity) || input.cutQuantity < 1) {
    throw new PatternCadError("invalid-transform", "label needs a positive cut quantity");
  }
  const fields: Record<string, string> = {
    garment: input.garmentName,
    panel: input.panelName,
    number: String(input.panelNumber),
    size: input.size ?? "",
    cut: `cut ${input.cutQuantity}`,
    material: input.material,
  };
  if (input.mirror) fields["mirror"] = input.mirror;
  if (input.notes) fields["notes"] = input.notes;
  return fields;
}

// ---------------------------------------------------------------------------
// Marking validation (extends production.ts reference checks)
// ---------------------------------------------------------------------------

export type MarkingDiagnosticCode =
  | "orphaned-marking"
  | "duplicate-marking"
  | "outside-panel"
  | "degenerate-placement";

export interface MarkingDiagnostic {
  code: MarkingDiagnosticCode;
  message: string;
  entityId?: string;
}

function loopSamples(doc: PatternDocument, panelId: EntityId, loopId: EntityId): Vec2[] {
  try {
    return sampleLoopLocal(doc, panelId, loopId);
  } catch {
    return [];
  }
}

/** True when pos is inside the panel or within eps of its outer boundary. */
function insideOrOn(doc: PatternDocument, panelId: EntityId, pos: Vec2, eps = 1e-9): boolean {
  try {
    if (pointInPanel(doc, panelId, pos)) return true;
  } catch {
    return false;
  }
  const panel = doc.panels.find((p) => p.id === panelId);
  const outer = panel?.boundaryLoops.find((l) => l.role === "outer");
  if (!outer) return false;
  const pts = loopSamples(doc, panelId, outer.id);
  for (let i = 0; i < pts.length; i++) {
    if (distToSegment(pos, pts[i], pts[(i + 1) % pts.length]) <= eps) return true;
  }
  return false;
}

export function validateMarkings(doc: PatternDocument, set: ProductionSet): MarkingDiagnostic[] {
  const diagnostics: MarkingDiagnostic[] = [];
  const fail = (code: MarkingDiagnosticCode, message: string, entityId?: string): void => {
    diagnostics.push({ code, message, ...(entityId ? { entityId } : {}) });
  };
  // Orphaned notches (edge/loop/panel gone after a boundary edit).
  for (const id of orphanedNotchIds(doc, set)) {
    fail("orphaned-marking", `notch '${id}' lost its boundary edge`, id);
  }
  for (const id of orphanedPanelIds(doc, set)) {
    fail("orphaned-marking", `production data references deleted panel '${id}'`, id);
  }
  // Duplicate notch positions on the same edge.
  const byEdge = new Map<string, Array<{ id: string; t: number }>>();
  for (const n of set.notches) {
    if (notchStatus(doc, n) !== "ok") continue;
    const key = `${n.panelId}/${n.segmentId}`;
    if (!byEdge.has(key)) byEdge.set(key, []);
    byEdge.get(key)!.push({ id: n.id, t: n.t });
  }
  for (const list of byEdge.values()) {
    for (let i = 0; i < list.length; i++) {
      for (let j = i + 1; j < list.length; j++) {
        if (Math.abs(list[i].t - list[j].t) <= 1e-9) {
          fail("duplicate-marking", `notches '${list[i].id}' and '${list[j].id}' coincide`, list[j].id);
        }
      }
    }
  }
  // Notches at edge endpoints are degenerate placement (ambiguous side).
  for (const n of set.notches) {
    if (notchStatus(doc, n) !== "ok") continue;
    if (n.t <= 1e-9 || n.t >= 1 - 1e-9) {
      fail("degenerate-placement", `notch '${n.id}' sits exactly on a vertex`, n.id);
    }
  }
  // Internal lines must not escape the panel (sampled, concave-safe).
  for (const line of set.internals) {
    if (!doc.panels.some((p) => p.id === line.panelId)) continue;
    let escaped = false;
    for (let i = 0; i < line.points.length - 1 && !escaped; i++) {
      for (let k = 0; k <= 8; k++) {
        const p: Vec2 = [
          line.points[i][0] + ((line.points[i + 1][0] - line.points[i][0]) * k) / 8,
          line.points[i][1] + ((line.points[i + 1][1] - line.points[i][1]) * k) / 8,
        ];
        if (!insideOrOn(doc, line.panelId, p)) {
          escaped = true;
          break;
        }
      }
    }
    if (escaped) fail("outside-panel", `internal line '${line.id}' escapes its panel`, line.id);
  }
  // Fold endpoints may rest on the boundary but not outside it.
  for (const fold of set.folds) {
    if (!doc.panels.some((p) => p.id === fold.panelId)) continue;
    if (!insideOrOn(doc, fold.panelId, fold.a) || !insideOrOn(doc, fold.panelId, fold.b)) {
      fail("outside-panel", `fold '${fold.id}' leaves its panel`, fold.id);
    }
  }
  // Grainlines must keep both anchors inside (construction margin, not edge-to-edge).
  for (const grain of set.grainlines) {
    if (!doc.panels.some((p) => p.id === grain.panelId)) continue;
    let inside = true;
    try {
      inside = pointInPanel(doc, grain.panelId, grain.from) && pointInPanel(doc, grain.panelId, grain.to);
    } catch {
      inside = false;
    }
    if (!inside) fail("outside-panel", `grainline '${grain.id}' leaves its panel`, grain.id);
  }
  return diagnostics;
}
