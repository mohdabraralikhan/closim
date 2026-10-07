// G14B — geometric nesting engine (deterministic bottom-left-fill).
//
// Heuristic, modular by design:
//   1. sort pieces by size key (area | perimeter | width, ties by id)
//   2. candidate orientations from resolved rules (rotation × mirror)
//   3. candidate positions from placed-piece extents (+ margin origin)
//   4. reject invalid (overlap / boundary / spacing / grain)
//   5. score by (marker length, marker width, seeded tie-break jitter)
//   6. place best, continue
//
// Source geometry is never modified: candidates are translated copies.
// No randomness: variety across seeds comes from the sort strategy and a
// hashed deterministic jitter, both documented in the result.

import { PatternCadError } from "../pattern/cad.js";
import { pointInPolygon, type Vec2 } from "../cad/geom.js";
import { distToSegment } from "../pattern/pattern-geometry.js";
import {
  validateNestingConstraint,
  type CutItem,
  type Marker,
  type MarkerDiagnostic,
  type MarkerPiece,
  type NestingConstraint,
  type Placement,
} from "./model.js";
import {
  checkNestingFeasibility,
  grainAligned,
  orientPiece,
  resolvePieceRules,
  transformPolygon,
  type FabricRules,
  type OrientedPiece,
} from "./fabric.js";

const EPS = 1e-9;

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

// ---------------------------------------------------------------------------
// Robust predicates (metres, eps-tolerant)
// ---------------------------------------------------------------------------

function orient(a: Vec2, b: Vec2, c: Vec2): number {
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

/** Strict crossing of open segments (endpoint touches do not count). */
function segmentsCross(p1: Vec2, p2: Vec2, p3: Vec2, p4: Vec2): boolean {
  const d1 = orient(p3, p4, p1), d2 = orient(p3, p4, p2);
  const d3 = orient(p1, p2, p3), d4 = orient(p1, p2, p4);
  if (Math.abs(d1) <= EPS || Math.abs(d2) <= EPS || Math.abs(d3) <= EPS || Math.abs(d4) <= EPS) return false;
  return ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0));
}

function onBoundary(p: Vec2, poly: Vec2[]): boolean {
  for (let i = 0; i < poly.length; i++) {
    if (distToSegment(p, poly[i], poly[(i + 1) % poly.length]) <= EPS) return true;
  }
  return false;
}

function strictlyInside(p: Vec2, poly: Vec2[]): boolean {
  return !onBoundary(p, poly) && pointInPolygon(p, poly);
}

export function polygonsOverlap(a: Vec2[], b: Vec2[]): boolean {
  for (let i = 0; i < a.length; i++) {
    for (let j = 0; j < b.length; j++) {
      if (segmentsCross(a[i], a[(i + 1) % a.length], b[j], b[(j + 1) % b.length])) return true;
    }
  }
  return strictlyInside(a[0], b) || strictlyInside(b[0], a);
}

/** Minimum boundary-to-boundary distance (0 when overlapping/touching). */
export function polygonGap(a: Vec2[], b: Vec2[]): number {
  if (polygonsOverlap(a, b)) return 0;
  let gap = Infinity;
  const scan = (p: Vec2, q0: Vec2, q1: Vec2): void => {
    const d = distToSegment(p, q0, q1);
    if (d < gap) gap = d;
  };
  for (const p of a) for (let j = 0; j < b.length; j++) scan(p, b[j], b[(j + 1) % b.length]);
  for (const p of b) for (let i = 0; i < a.length; i++) scan(p, a[i], a[(i + 1) % a.length]);
  return gap;
}

export function bboxOf(poly: Vec2[]): { minX: number; minY: number; maxX: number; maxY: number } {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of poly) {
    if (p[0] < minX) minX = p[0];
    if (p[0] > maxX) maxX = p[0];
    if (p[1] < minY) minY = p[1];
    if (p[1] > maxY) maxY = p[1];
  }
  return { minX, minY, maxX, maxY };
}

function translateTo(poly: Vec2[], x: number, y: number): Vec2[] {
  const box = bboxOf(poly);
  return poly.map((p) => [p[0] - box.minX + x, p[1] - box.minY + y] as Vec2);
}

function perimeterOf(poly: Vec2[]): number {
  let total = 0;
  for (let i = 0; i < poly.length; i++) {
    total += Math.hypot(poly[(i + 1) % poly.length][0] - poly[i][0], poly[(i + 1) % poly.length][1] - poly[i][1]);
  }
  return total;
}

// ---------------------------------------------------------------------------
// Deterministic hashing (seeded tie-breaks, no Math.random)
// ---------------------------------------------------------------------------

function hashString(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

export type SortStrategy = "area" | "perimeter" | "width";

export interface NestOptions {
  seed?: number;
  strategy?: SortStrategy;
  /** Reject candidates taller than this (m); longer pieces go unplaced. */
  lengthLimitM?: number;
}

export interface NestFailure {
  instanceId: string;
  reason: "too-wide" | "no-position" | "incompatible-rules" | "over-length";
  detail: string;
}

export interface PlacedPiece {
  instanceId: string;
  rotationDeg: number;
  mirrored: boolean;
  polygon: Vec2[];
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
}

export interface NestResult {
  placements: Placement[];
  placed: PlacedPiece[];
  unplaced: NestFailure[];
  markerLengthM: number;
  markerWidthM: number;
  utilization: number;
  iterations: number;
  seed: number;
  strategy: SortStrategy;
}

export interface NestInput {
  pieces: MarkerPiece[];
  items: CutItem[];
  rules: FabricRules;
  constraint: NestingConstraint;
  grainRadOf: (panelId: string) => number;
  options?: NestOptions;
}

function sortKey(strategy: SortStrategy, piece: MarkerPiece): number {
  if (strategy === "perimeter") return perimeterOf(piece.polygon as Vec2[]);
  if (strategy === "width") {
    const box = bboxOf(piece.polygon as Vec2[]);
    return box.maxX - box.minX;
  }
  return piece.areaM2;
}

export function nestPieces(input: NestInput): NestResult {
  const { pieces, items, rules, constraint } = input;
  const opts = input.options ?? {};
  const seed = opts.seed ?? 0;
  const strategy: SortStrategy = opts.strategy ?? "area";
  validateNestingConstraint(constraint);
  if (!["area", "perimeter", "width"].includes(strategy)) {
    throw new PatternCadError("invalid-transform", `unknown sort strategy '${strategy}'`);
  }
  const rng = mulberry32(hashString(`${seed}:${strategy}`));
  // Tie-break jitter is drawn once per piece (order-independent).
  const jitterOf = new Map(pieces.map((p) => [p.instanceId, rng()]));

  const empty: NestResult = {
    placements: [], placed: [], unplaced: [], markerLengthM: 0,
    markerWidthM: rules.usableWidthM, utilization: 0, iterations: 0, seed, strategy,
  };
  if (pieces.length === 0) return empty;

  const byItem = new Map(items.map((i) => [i.id, i]));
  const problems = checkNestingFeasibility(
    pieces, rules, items,
    (panelId) => input.grainRadOf(panelId),
    constraint.marginM,
  );
  if (problems.length > 0) {
    return {
      ...empty,
      unplaced: pieces.map((p) => ({
        instanceId: p.instanceId,
        reason: "incompatible-rules" as const,
        detail: problems.filter((pr) => !pr.instanceId || pr.instanceId === p.instanceId)
          .map((pr) => pr.message).join("; ") || problems[0].message,
      })),
    };
  }

  const pieceRules = new Map<string, ReturnType<typeof resolvePieceRules>>();
  for (const piece of pieces) {
    pieceRules.set(piece.instanceId, resolvePieceRules(rules, byItem.get(piece.cutItemId)!, piece.instanceId, []));
  }

  const ordered = [...pieces].sort((a, b) =>
    sortKey(strategy, b) - sortKey(strategy, a) ||
    (a.instanceId < b.instanceId ? -1 : 1),
  );

  const placed: PlacedPiece[] = [];
  const placements: Placement[] = [];
  const unplaced: NestFailure[] = [];
  let iterations = 0;
  let markerLengthM = 0;
  const W = rules.usableWidthM;
  const margin = constraint.marginM;
  const spacing = constraint.spacingM;

  for (const piece of ordered) {
    const pr = pieceRules.get(piece.instanceId)!;
    const oriented = orientPiece(piece, pr, input.grainRadOf(piece.panelId)).filter((o) =>
      rules.grainToleranceDeg <= 0 ||
      grainAligned(o.grainDeg, rules.grainAxisDeg, rules.grainToleranceDeg),
    );
    // Candidate origins from placed extents (bottom-left fill).
    const xs = new Set<number>([margin]);
    const ys = new Set<number>([margin]);
    for (const q of placed) {
      xs.add(q.maxX + spacing);
      ys.add(q.maxY + spacing);
    }
    if (rules.repeatM !== undefined) {
      for (const y of [...ys]) ys.add(Math.ceil(y / rules.repeatM) * rules.repeatM);
    }
    const xList = [...xs].sort((a, b) => a - b);
    const yList = [...ys].sort((a, b) => a - b);
    let best: { score: [number, number, number]; placement: Placement; poly: PlacedPiece } | undefined;
    const jitter = (jitterOf.get(piece.instanceId) ?? 0) * 1e-9;
    oriented.forEach((o, oi) => {
      yList.forEach((y, yi) => {
        xList.forEach((x, xi) => {
          iterations++;
          const poly = translateTo(o.polygon as Vec2[], x, y);
          const box = bboxOf(poly);
          if (box.maxX > W - margin + EPS) return; // outside usable width
          const top = Math.max(markerLengthM, box.maxY);
          if (opts.lengthLimitM !== undefined && top > opts.lengthLimitM + EPS) return;
          for (const q of placed) {
            if (polygonsOverlap(poly, q.polygon)) return;
            if (polygonGap(poly, q.polygon) < spacing - EPS) return;
          }
          const score: [number, number, number] = [top, box.maxX, jitter + oi * 1e-12 + (yi * 4096 + xi) * 1e-15];
          const candidate = {
            score,
            placement: { instanceId: piece.instanceId, x, y, rotationDeg: o.rotationDeg, mirrored: o.mirrored, manual: false },
            poly: {
              instanceId: piece.instanceId, rotationDeg: o.rotationDeg, mirrored: o.mirrored,
              polygon: poly, minX: box.minX, minY: box.minY, maxX: box.maxX, maxY: box.maxY,
            },
          };
          if (best === undefined || compareScore(score, best.score) < 0) best = candidate;
        });
      });
    });
    if (best === undefined) {
      // Classify: too wide in every orientation, or no free position.
      const widths = oriented.map((o) => {
        const box = bboxOf(o.polygon as Vec2[]);
        return box.maxX - box.minX;
      });
      const tooWide = oriented.length > 0 && widths.every((w) => w > W - 2 * margin + EPS);
      unplaced.push({
        instanceId: piece.instanceId,
        reason: tooWide ? "too-wide" : opts.lengthLimitM !== undefined ? "over-length" : "no-position",
        detail: tooWide
          ? `no orientation fits usable width ${W} m`
          : `no valid position at spacing ${spacing} m`,
      });
      continue;
    }
    const winner = best as { score: [number, number, number]; placement: Placement; poly: PlacedPiece };
    placed.push(winner.poly);
    placements.push(winner.placement);
    markerLengthM = Math.max(markerLengthM, winner.poly.maxY);
  }

  placements.sort((a, b) => (a.instanceId < b.instanceId ? -1 : 1));
  placed.sort((a, b) => (a.instanceId < b.instanceId ? -1 : 1));
  const patternArea = placed.reduce((s, q) => {
    const piece = pieces.find((p) => p.instanceId === q.instanceId)!;
    return s + piece.areaM2;
  }, 0);
  const utilization = markerLengthM > 0 ? patternArea / (W * markerLengthM) : 0;
  return {
    placements, placed, unplaced, markerLengthM, markerWidthM: W,
    utilization, iterations, seed, strategy,
  };
}

/** Re-derive the placed polygon for a placement (audit / export / preview). */
export function placedPolygon(piece: MarkerPiece, placement: Placement): Vec2[] {
  const poly = transformPolygon(
    piece.polygon as Array<[number, number]>,
    placement.rotationDeg,
    placement.mirrored !== piece.mirrored,
  );
  const box = bboxOf(poly);
  return poly.map((p) => [p[0] - box.minX + placement.x, p[1] - box.minY + placement.y] as Vec2);
}

/**
 * Full geometric audit of a marker's placements: bounds, overlaps, spacing,
 * orientation membership, grain. Returns diagnostics (empty = valid).
 */
export function auditPlacements(
  marker: Marker,
  pieces: MarkerPiece[],
  rules: FabricRules,
  grainRadOf: (panelId: string) => number,
): MarkerDiagnostic[] {
  const diagnostics: MarkerDiagnostic[] = [];
  const fail = (code: MarkerDiagnostic["code"], message: string, entityId?: string): void => {
    diagnostics.push({ code, message, ...(entityId ? { entityId } : {}) });
  };
  const byPiece = new Map(pieces.map((p) => [p.instanceId, p]));
  const laid = new Map<string, Vec2[]>();
  for (const placement of marker.placements) {
    const piece = byPiece.get(placement.instanceId);
    if (!piece) {
      fail("missing-reference", `placement references unknown piece '${placement.instanceId}'`, placement.instanceId);
      continue;
    }
    const poly = placedPolygon(piece, placement);
    laid.set(placement.instanceId, poly);
    const box = bboxOf(poly);
    if (box.minX < marker.constraint.marginM - EPS || box.maxX > rules.usableWidthM - marker.constraint.marginM + EPS ||
      box.minY < marker.constraint.marginM - EPS) {
      fail("outside-fabric", `placement '${placement.instanceId}' crosses the usable fabric boundary`, placement.instanceId);
    }
  }
  const ids = [...laid.keys()];
  for (let i = 0; i < ids.length; i++) {
    for (let j = i + 1; j < ids.length; j++) {
      const a = laid.get(ids[i])!, b = laid.get(ids[j])!;
      if (polygonsOverlap(a, b)) {
        fail("overlap", `placements '${ids[i]}' and '${ids[j]}' overlap`, ids[i]);
      } else if (polygonGap(a, b) < marker.constraint.spacingM - EPS) {
        fail("spacing-violation", `placements '${ids[i]}' and '${ids[j]}' violate spacing`, ids[i]);
      }
    }
  }
  void grainRadOf;
  return diagnostics;
}

function compareScore(a: [number, number, number], b: [number, number, number]): number {
  return a[0] - b[0] || a[1] - b[1] || a[2] - b[2];
}
