// G14 — marker making: automated piece nesting and efficiency reporting.
//
// A marker is the cut plan for one fabric lay: all requested pieces
// (quantities honoured, mirror partners resolved) placed inside a frame of a
// given usable width at a limited set of orientations, without overlaps.
//
// Design contract (determinism first):
//   - Identical inputs produce identical markers. No randomness, no
//     wall-clock, no iteration-order dependence. Placement is a
//     bottom-left (BL) first-fit heuristic over a canonical piece order:
//     larger max-dimension first, ties by panelId then sizeLabel then id.
//   - Placements are axis-aligned (0/90/180/270 deg) and step-quantized
//     (`colStepM` / `rowStepM`): feasibility is searched on a grid, so a
//     tighter inter-piece gap than one step is not discovered. This is the
//     documented v1 trade-off for determinism and simplicity.
//   - Grain: the fabric warp runs along marker +X. A piece with an authored
//     grainline may only rotate so that its grain stays parallel to the warp
//     (0 mod 180 deg) within `grainToleranceRad`. Pieces without a grainline
//     may rotate freely (configurable). Grain that cannot be honoured is
//     placed unrotated and flagged in `warnings` — never silently off-grain.
//   - Collisions: polygons (piece footprints + explicit clearance) may not
//     overlap each other or leave the frame. Tests combine point-in-polygon,
//     edge crossing, and a separating-axis scan (conservative for
//     non-convex footprints).
//   - Efficiency = placed piece boundary area / used frame area, in
//     true-scale m^2 from the export IR (single geometry source).
//
// The marker is derived data (like the export IR): it is never fed back
// into the native document, and the gate rules apply before any geometry
// is touched.

import type { Vec2 } from "./geom.js";
import { signedArea } from "./geom.js";
import { PatternCadError, type PatternDocument } from "../pattern/cad.js";
import type { Seam } from "../garment/sewing.js";
import type { ProductionSet } from "./production.js";
import {
  exportGate,
  buildExportIR,
  type ExportGateMode,
  type ExportIR,
  type ExportPanelIR,
} from "./export-ir.js";
import { exportMarkerSVG } from "./marker-svg.js";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/** Orientation of a placed piece. Angles are CCW degrees. */
export type MarkerRotation = 0 | 90 | 180 | 270;

export interface MarkerPieceInput {
  panelId: string;
  /** Per-size instance label, when building a graded marker. */
  sizeLabel?: string;
  /** Copies to place (cut quantity). Default 1. */
  quantity?: number;
  /** Mirror the piece before placing (resolves a mirrored pair partner). */
  mirrored?: boolean;
}

export interface MarkerOptions {
  /** Usable marker width in metres (fabric width minus margins). */
  widthM: number;
  /** Frame length in metres; omit for an open-ended marker whose length is computed. */
  lengthM?: number;
  /** Minimum clearance between pieces and to the frame, metres. Default 0.005. */
  clearanceM?: number;
  /** Grain tolerance vs the warp axis (0 mod 180 deg), radians. Default 1 deg. */
  grainToleranceRad?: number;
  /** Allow free 90-degree rotation for pieces without a grainline. Default true. */
  rotateUnconstrained?: boolean;
  /** Horizontal placement candidate step, metres. Default 0.01. */
  colStepM?: number;
  /** Vertical placement candidate step, metres. Default 0.01. */
  rowStepM?: number;
}

export interface MarkerPlacement {
  pieceId: string;
  panelId: string;
  sizeLabel: string | null;
  mirrored: boolean;
  rotation: MarkerRotation;
  /** Lower-left of the placed footprint bbox, metres (frame origin bottom-left). */
  x: number;
  y: number;
}

export interface MarkerEfficiency {
  /** Sum of placed piece boundary areas, m^2 (declined pieces excluded). */
  pieceAreaM2: number;
  /** Used frame area (used length x width), m^2. */
  frameAreaM2: number;
  /** pieceAreaM2 / frameAreaM2, in [0, 1]. */
  efficiency: number;
  /** frameAreaM2 - pieceAreaM2, m^2. */
  wasteM2: number;
}

export interface Marker {
  format: "closim-marker";
  version: 1;
  widthM: number;
  /** Frame length actually used (computed for open-ended markers). */
  lengthM: number;
  /** Fabric orientation contract: warp along +X of the marker frame. */
  warpDirection: "x";
  clearanceM: number;
  placements: MarkerPlacement[];
  /** Placed footprints in frame coordinates (metres, CCW, aligned with placements). */
  footprints: Array<{ pieceId: string; ring: Vec2[] }>;
  efficiency: MarkerEfficiency;
  /** Pieces that could not be placed, with reasons (never silent). */
  declined: Array<{ pieceId: string; reason: string }>;
  warnings: string[];
  style: { garmentName: string; styleId: string; revision: number };
  /** Sizes present in this marker (empty for single-size). */
  sizes: string[];
}

export interface BuildMarkerOptions extends MarkerOptions {
  garmentName?: string;
  styleId?: string;
  revision?: number;
  gateMode?: ExportGateMode;
  /** Skip the production gate (pre-validated documents only). Default false. */
  skipGate?: boolean;
  /**
   * Keep going when pieces cannot be placed: report them in `declined`
   * instead of throwing. Default false (all-or-nothing).
   */
  relaxed?: boolean;
}

// ---------------------------------------------------------------------------
// Footprint extraction from the export IR
// ---------------------------------------------------------------------------

interface PieceCandidate {
  pieceId: string;
  panelId: string;
  sizeLabel: string | null;
  mirrored: boolean;
  /** Boundary polygon in metres, origin at the piece bbox lower-left, CCW. */
  ring: Vec2[];
  /** Authored grainline direction in radians (canonical orientation), or null. */
  grainRad: number | null;
  areaM2: number;
  w: number;
  h: number;
}

/** Boundary polygon of one panel (cut line preferred over sewing line), CCW. */
function panelRing(panel: ExportPanelIR): Vec2[] {
  const ring = panel.cutEdges ?? panel.sewingEdges;
  const pts: Vec2[] = ring.map((e) => e.a);
  if (signedArea(pts) < 0) pts.reverse();
  return pts;
}

function candidatesFor(
  ir: ExportIR,
  inputs: MarkerPieceInput[],
): { candidates: PieceCandidate[]; declined: Array<{ pieceId: string; reason: string }>; warnings: string[] } {
  const byId = new Map(ir.panels.map((p) => [p.panelId, p]));
  const candidates: PieceCandidate[] = [];
  const declined: Array<{ pieceId: string; reason: string }> = [];
  const warnings: string[] = [];
  const usedIds = new Set<string>();

  const emitOne = (input: MarkerPieceInput, copy: number, quantity: number): void => {
    const panel = byId.get(input.panelId);
    if (!panel) {
      declined.push({ pieceId: input.panelId, reason: `panel '${input.panelId}' not present in the export IR` });
      return;
    }
    const ring = panelRing(panel);
    if (ring.length < 3) {
      declined.push({ pieceId: input.panelId, reason: "boundary ring has fewer than 3 vertices" });
      return;
    }
    // Mirror across the piece's own vertical centre line.
    let finalRing = ring;
    if (input.mirrored) {
      let minX = Infinity, maxX = -Infinity;
      for (const p of ring) { if (p[0] < minX) minX = p[0]; if (p[0] > maxX) maxX = p[0]; }
      const mid = (minX + maxX) / 2;
      finalRing = ring.map((p) => [2 * mid - p[0], p[1]] as Vec2).reverse();
      // Mirroring flips orientation back to CW; re-canonicalize below.
    }
    if (signedArea(finalRing) < 0) finalRing = [...finalRing].reverse();
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const p of finalRing) {
      if (p[0] < minX) minX = p[0];
      if (p[1] < minY) minY = p[1];
      if (p[0] > maxX) maxX = p[0];
      if (p[1] > maxY) maxY = p[1];
    }
    const grain = panel.grainlines[0];
    const grainRad = grain
      ? Math.atan2(grain.to[1] - grain.from[1], grain.to[0] - grain.from[0])
      : null;
    const sizePart = input.sizeLabel !== undefined ? `.${input.sizeLabel.toLowerCase()}` : "";
    const copyPart = quantity > 1 ? `#${copy + 1}` : "";
    const mirrorPart = input.mirrored ? "-m" : "";
    let pieceId = `${input.panelId}${sizePart}${copyPart}${mirrorPart}`;
    for (let n = 2; usedIds.has(pieceId); n++) pieceId = `${input.panelId}${sizePart}${copyPart}${mirrorPart}-${n}`;
    usedIds.add(pieceId);
    candidates.push({
      pieceId,
      panelId: input.panelId,
      sizeLabel: input.sizeLabel ?? null,
      mirrored: input.mirrored === true,
      ring: finalRing.map((p) => [p[0] - minX, p[1] - minY] as Vec2),
      grainRad,
      areaM2: Math.abs(signedArea(finalRing)),
      w: maxX - minX,
      h: maxY - minY,
    });
  };

  for (const input of inputs) {
    const q = Math.max(1, Math.floor(input.quantity ?? 1));
    for (let c = 0; c < q; c++) emitOne(input, c, q);
  }
  if (candidates.length === 0 && declined.length === 0) {
    warnings.push("marker requested no pieces");
  }
  return { candidates, declined, warnings };
}

// ---------------------------------------------------------------------------
// Deterministic nesting (bottom-left first fit)
// ---------------------------------------------------------------------------

function rotatePoint(p: Vec2, r: MarkerRotation): Vec2 {
  switch (r) {
    case 0: return [p[0], p[1]];
    case 90: return [-p[1], p[0]];
    case 180: return [-p[0], -p[1]];
    case 270: return [p[1], -p[0]];
  }
}

/** Footprint polygon of a candidate at rotation r, origin at its bbox lower-left. */
function footprintFor(c: PieceCandidate, r: MarkerRotation): Vec2[] {
  const pts = c.ring.map((p) => rotatePoint(p, r));
  let minX = Infinity, minY = Infinity;
  for (const p of pts) { if (p[0] < minX) minX = p[0]; if (p[1] < minY) minY = p[1]; }
  return pts.map((p) => [p[0] - minX, p[1] - minY] as Vec2);
}

function bboxOf(ring: readonly Vec2[]): { w: number; h: number } {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of ring) {
    if (p[0] < minX) minX = p[0];
    if (p[1] < minY) minY = p[1];
    if (p[0] > maxX) maxX = p[0];
    if (p[1] > maxY) maxY = p[1];
  }
  return { w: maxX - minX, h: maxY - minY };
}

function pointInPolygon(p: Vec2, poly: readonly Vec2[]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const a = poly[i], b = poly[j];
    if ((a[1] > p[1]) !== (b[1] > p[1])) {
      const x = a[0] + ((p[1] - a[1]) / (b[1] - a[1])) * (b[0] - a[0]);
      if (x > p[0]) inside = !inside;
    }
  }
  return inside;
}

/** Proper/touching segment intersection (closed segments, tiny epsilon). */
function segmentsCross(a1: Vec2, a2: Vec2, b1: Vec2, b2: Vec2): boolean {
  const d1 = [a2[0] - a1[0], a2[1] - a1[1]];
  const d2 = [b2[0] - b1[0], b2[1] - b1[1]];
  const den = d1[0] * d2[1] - d1[1] * d2[0];
  const scale = Math.hypot(d1[0], d1[1]) * Math.hypot(d2[0], d2[1]);
  if (scale === 0) return false;
  const eps = 1e-9;
  if (Math.abs(den) <= eps * scale) {
    // Parallel: contact only via endpoint proximity (covered by containment
    // checks in practice; treat as non-crossing here).
    return false;
  }
  const t = ((b1[0] - a1[0]) * d2[1] - (b1[1] - a1[1]) * d2[0]) / den;
  const u = ((b1[0] - a1[0]) * d1[1] - (b1[1] - a1[1]) * d1[0]) / den;
  return t >= -eps && t <= 1 + eps && u >= -eps && u <= 1 + eps;
}

/** Edges of the two polygons cross? (catches overlaps without contained vertices) */
function edgesCross(A: readonly Vec2[], B: readonly Vec2[]): boolean {
  for (let i = 0; i < A.length; i++) {
    const a1 = A[i], a2 = A[(i + 1) % A.length];
    for (let j = 0; j < B.length; j++) {
      const b1 = B[j], b2 = B[(j + 1) % B.length];
      if (segmentsCross(a1, a2, b1, b2)) return true;
    }
  }
  return false;
}

/** Separating-axis overlap with clearance inflation (conservative for non-convex). */
function satOverlap(A: readonly Vec2[], B: readonly Vec2[], clearance: number): boolean {
  const axes: Vec2[] = [];
  const pushAxes = (poly: readonly Vec2[]): void => {
    for (let i = 0; i < poly.length; i++) {
      const p = poly[i], q = poly[(i + 1) % poly.length];
      const ex = q[0] - p[0], ey = q[1] - p[1];
      const l = Math.hypot(ex, ey);
      if (l === 0) continue;
      axes.push([-ey / l, ex / l]);
    }
  };
  pushAxes(A);
  pushAxes(B);
  for (const ax of axes) {
    let aMin = Infinity, aMax = -Infinity, bMin = Infinity, bMax = -Infinity;
    for (const p of A) { const d = p[0] * ax[0] + p[1] * ax[1]; if (d < aMin) aMin = d; if (d > aMax) aMax = d; }
    for (const p of B) { const d = p[0] * ax[0] + p[1] * ax[1]; if (d < bMin) bMin = d; if (d > bMax) bMax = d; }
    if (aMax + clearance <= bMin || bMax + clearance <= aMin) return false;
  }
  return true;
}

/** Full collision predicate: containment, edge crossing, or SAT overlap. */
function piecesCollide(aAt: Vec2, a: readonly Vec2[], bAt: Vec2, b: readonly Vec2[], clearance: number): boolean {
  const A = a.map((p) => [p[0] + aAt[0], p[1] + aAt[1]] as Vec2);
  const B = b.map((p) => [p[0] + bAt[0], p[1] + bAt[1]] as Vec2);
  if (A.some((p) => pointInPolygon(p, B))) return true;
  if (B.some((p) => pointInPolygon(p, A))) return true;
  if (edgesCross(A, B)) return true;
  return satOverlap(A, B, clearance);
}

/** Canonical order: larger max-dimension first; ties by panelId, sizeLabel, id. */
function canonicalOrder(c: PieceCandidate, d: PieceCandidate): number {
  const ac = Math.max(c.w, c.h);
  const bc = Math.max(d.w, d.h);
  if (ac !== bc) return bc - ac;
  if (c.panelId !== d.panelId) return c.panelId < d.panelId ? -1 : 1;
  const cs = c.sizeLabel ?? "";
  const ds = d.sizeLabel ?? "";
  if (cs !== ds) return cs < ds ? -1 : 1;
  return c.pieceId < d.pieceId ? -1 : c.pieceId > d.pieceId ? 1 : 0;
}

interface RotationChoice { rotations: MarkerRotation[]; warning?: string }

function rotationsFor(c: PieceCandidate, opts: MarkerOptions): RotationChoice {
  const tol = opts.grainToleranceRad ?? (1 * Math.PI) / 180;
  const free = opts.rotateUnconstrained ?? true;
  const ALL: MarkerRotation[] = [0, 90, 180, 270];
  if (c.grainRad === null) {
    return free
      ? { rotations: ALL }
      : { rotations: [0], warning: `${c.pieceId}: no grainline; free rotation disabled by configuration` };
  }
  // Warp is marker +X. The piece grain after rotation must be parallel to
  // the warp (0 mod 180 deg) within tolerance.
  const ok: MarkerRotation[] = [];
  for (const r of ALL) {
    const dir = c.grainRad + (r * Math.PI) / 180;
    let norm = dir % Math.PI;
    if (norm < 0) norm += Math.PI;
    const delta = Math.min(norm, Math.PI - norm);
    if (delta <= tol) ok.push(r);
  }
  if (ok.length === 0) {
    return {
      rotations: [0],
      warning: `${c.pieceId}: grainline at ${((c.grainRad * 180) / Math.PI).toFixed(1)}deg cannot be aligned with the warp by 90-degree rotation (tolerance ${((tol * 180) / Math.PI).toFixed(2)}deg); placed unrotated and flagged`,
    };
  }
  return { rotations: ok };
}

// ---------------------------------------------------------------------------
// Marker construction
// ---------------------------------------------------------------------------

/**
 * Gate -> IR -> footprints -> deterministic BL nesting -> efficiency.
 * Throws when the gate blocks, the frame is impossible for a piece, or (in
 * strict mode) any requested piece ends up declined. Use `relaxed: true` to
 * accept a partial marker with `declined` populated.
 */
export function buildMarker(
  doc: PatternDocument,
  seams: readonly Seam[],
  set: ProductionSet,
  pieces: MarkerPieceInput[],
  opts: BuildMarkerOptions = { widthM: NaN }, // callers must supply a real width; NaN triggers the runtime validation below
): Marker {
  if (!Array.isArray(pieces) || pieces.length === 0) {
    throw new PatternCadError("invalid-transform", "marker needs at least one piece input");
  }
  const width = opts.widthM;
  if (!(width > 0) || !Number.isFinite(width)) {
    throw new PatternCadError("invalid-transform", "marker width must be a positive finite length");
  }
  const clearance = opts.clearanceM ?? 0.005;
  const colStep = opts.colStepM ?? 0.01;
  const rowStep = opts.rowStepM ?? 0.01;
  if (clearance < 0 || colStep <= 0 || rowStep <= 0) {
    throw new PatternCadError("invalid-transform", "marker clearance must be >= 0 and steps must be positive");
  }

  let ir: ExportIR;
  if (opts.skipGate) {
    ir = buildExportIR(doc, seams, set, { garmentName: opts.garmentName, revision: opts.revision ?? 1 });
  } else {
    const decision = exportGate(doc, seams, set, opts.gateMode ?? "strict");
    if (decision.blocked) {
      const first = decision.readiness.diagnostics[0];
      throw new PatternCadError(
        "invalid-document",
        `marker refused (state=${decision.state}): ${first ? `${first.severity} ${first.code}: ${first.message}` : "validation failed"}`,
        first?.entityId,
      );
    }
    ir = buildExportIR(doc, seams, set, {
      garmentName: opts.garmentName,
      styleId: opts.styleId,
      revision: opts.revision ?? 1,
      readiness: decision.readiness,
    });
  }

  const { candidates, declined, warnings } = candidatesFor(ir, pieces);
  const ordered = [...candidates].sort(canonicalOrder);

  const placed: MarkerPlacement[] = [];
  const placedFootprints: Vec2[][] = [];
  const openEnded = opts.lengthM === undefined;
  const hardLength = openEnded ? Infinity : opts.lengthM!;
  let usedHeight = 0;
  let usedWidth = 0;

  for (const c of ordered) {
    const choice = rotationsFor(c, opts);
    if (choice.warning) warnings.push(choice.warning);
    let best: { x: number; y: number; r: MarkerRotation; ring: Vec2[]; w: number; h: number } | null = null;
    // Rotations whose footprints coincide (e.g. 180 deg of a symmetric
    // piece) have identical placement outcomes — try each distinct
    // footprint once, keeping the smallest rotation angle.
    const seen = new Set<string>();
    for (const r of choice.rotations) {
      const fp0 = footprintFor(c, r);
      const key = fp0.map((p) => `${p[0].toFixed(9)},${p[1].toFixed(9)}`).join(";");
      if (seen.has(key)) continue;
      seen.add(key);
      const fp = fp0;
      const bb = bboxOf(fp);
      if (bb.w + 2 * clearance > width) continue; // this orientation cannot fit the width
      // Bottom-left first fit: scan rows bottom-up, columns left-to-right.
      scan: for (let y = 0; ; y += rowStep) {
        if (y + bb.h + 2 * clearance > hardLength) break;
        for (let x = 0; ; x += colStep) {
          if (x + bb.w + 2 * clearance > width) break;
          const at: Vec2 = [x + clearance, y + clearance];
          let collide = false;
          for (let k = 0; k < placedFootprints.length; k++) {
            if (piecesCollide(at, fp, [placed[k].x, placed[k].y], placedFootprints[k], clearance)) {
              collide = true;
              break;
            }
          }
          if (!collide) {
            if (best === null || y < best.y || (y === best.y && x < best.x)) {
              best = { x: at[0], y: at[1], r, ring: fp, w: bb.w, h: bb.h };
            }
            break scan;
          }
        }
      }
    }
    if (best === null) {
      declined.push({
        pieceId: c.pieceId,
        reason: openEnded
          ? "no allowed orientation fits the marker width"
          : "no free position within the frame at the allowed orientations",
      });
      continue;
    }
    placed.push({
      pieceId: c.pieceId,
      panelId: c.panelId,
      sizeLabel: c.sizeLabel,
      mirrored: c.mirrored,
      rotation: best.r,
      x: best.x,
      y: best.y,
    });
    placedFootprints.push(best.ring);
    usedHeight = Math.max(usedHeight, best.y + best.h);
    usedWidth = Math.max(usedWidth, best.x + best.w);
  }

  if (declined.length > 0 && !opts.relaxed) {
    const first = declined[0];
    throw new PatternCadError(
      "invalid-transform",
      `marker incomplete: piece '${first.pieceId}' could not be placed — ${first.reason}`,
    );
  }

  const placedAreaM2 = placed.reduce((s, p) => {
    const c = ordered.find((q) => q.pieceId === p.pieceId);
    return s + (c ? c.areaM2 : 0);
  }, 0);
  const usedLength = openEnded ? usedHeight + clearance : opts.lengthM!;
  const frameAreaM2 = usedLength * width;
  const efficiency: MarkerEfficiency = {
    pieceAreaM2: placedAreaM2,
    frameAreaM2,
    efficiency: frameAreaM2 > 0 ? placedAreaM2 / frameAreaM2 : 0,
    wasteM2: frameAreaM2 - placedAreaM2,
  };

  const placedSet = new Set(placed.map((p) => p.pieceId));
  const footprints = ordered
    .filter((c) => placedSet.has(c.pieceId))
    .map((c) => {
      const p = placed.find((q) => q.pieceId === c.pieceId)!;
      const fp = footprintFor(c, p.rotation);
      return { pieceId: c.pieceId, ring: fp.map((q) => [q[0] + p.x, q[1] + p.y] as Vec2) };
    });

  const sizes = [...new Set(candidates.map((c) => c.sizeLabel).filter((s): s is string => s !== null))];
  return {
    format: "closim-marker",
    version: 1,
    widthM: width,
    lengthM: usedLength,
    warpDirection: "x",
    clearanceM: clearance,
    placements: placed,
    footprints,
    efficiency,
    declined: [...declined],
    warnings,
    style: {
      garmentName: opts.garmentName ?? ir.style.garmentName,
      styleId: opts.styleId ?? ir.style.styleId,
      revision: opts.revision ?? 1,
    },
    sizes,
  };
}

export { exportMarkerSVG };
