// G7D foundational 2D pattern geometry — independent of 3D simulation.
//
// Supports polygon boundaries, polyline segments, circular arcs, holes,
// winding normalization, point-in-panel, arclength boundary parameterization,
// and deterministic constrained triangulation (hole bridging + ear clipping).
//
// Determinism contract: every operation iterates inputs in stored order with
// index tie-breaks; triangulating the same panel twice (or a winding-flipped
// copy) yields bitwise-identical output. No randomness, no hash maps with
// nondeterministic iteration, no solver contact.
//
// Pattern rest coordinates (x, 0, y) are the FEM rest-space metric:
// panelToRestMesh() emits exactly what preprocess(positions, uv, indices)
// consumes (uv = pattern xy), so rest Dm/areas derive from pattern space.

export type Vec2 = [number, number];

export class PatternError extends Error {
  readonly code: string;
  constructor(code: string, message: string) {
    super(`pattern geometry (${code}): ${message}`);
    this.name = "PatternError";
    this.code = code;
  }
}

// ---------------------------------------------------------------------------
// Authoring primitives
// ---------------------------------------------------------------------------

export type BoundaryPrimitive =
  | { kind: "polyline"; points: Vec2[] }
  | { kind: "arc"; center: Vec2; radius: number; a0: number; a1: number };

export interface PatternPanel {
  id: string;
  /** Outer boundary as an ordered primitive loop (any winding; normalized). */
  outline: BoundaryPrimitive[];
  /** Holes as ordered primitive loops (any winding; normalized). */
  holes: BoundaryPrimitive[][];
  /** Grain direction in pattern space, radians from +x. Metadata only. */
  grainAngleRad: number;
  /** Opaque material reference (resolved by the material layer, G7A). */
  materialId: string;
}

export interface ValidatedPanel {
  id: string;
  outer: Vec2[];
  holes: Vec2[][];
  outerReversed: boolean;
  holesReversed: boolean[];
  grainAngleRad: number;
  materialId: string;
}

export interface TriangulatedPanel {
  panelId: string;
  /** Interleaved xy, length 2n. Rest-space 3D is (x, 0, y). */
  vertices: Float64Array;
  triangles: Uint32Array;
  /** True boundary edges only (interior diagonals excluded). */
  boundaryEdges: BoundaryEdgeTag[];
  grainAngleRad: number;
  materialId: string;
}

export interface BoundaryEdgeTag {
  a: number;
  b: number;
  /** 'outer' or hole index. */
  loop: "outer" | number;
  /** Index of the source edge within the sanitized loop. */
  edgeIndex: number;
  /** Arclength parameters of a/b along the loop, in [0,1). */
  t0: number;
  t1: number;
}

export interface PanelQuality {
  panelArea: number;
  meshArea: number;
  areaRelErr: number;
  /** Hausdorff distance true-boundary -> triangulated boundary (meters). */
  boundaryDeviation: number;
  minTriArea: number;
  /** min over tris of 4*sqrt(3)*area/(e1^2+e2^2+e3^2); 1 = equilateral. */
  minShapeQuality: number;
}

// ---------------------------------------------------------------------------
// Small vector helpers (module-private)
// ---------------------------------------------------------------------------

const sub = (a: Vec2, b: Vec2): Vec2 => [a[0] - b[0], a[1] - b[1]];
const cross = (a: Vec2, b: Vec2): number => a[0] * b[1] - a[1] * b[0];
const dot = (a: Vec2, b: Vec2): number => a[0] * b[0] + a[1] * b[1];
const len2 = (a: Vec2): number => a[0] * a[0] + a[1] * a[1];

export function signedArea(loop: Vec2[]): number {
  let s = 0;
  for (let i = 0; i < loop.length; i++) {
    const p = loop[i], q = loop[(i + 1) % loop.length];
    s += p[0] * q[1] - q[0] * p[1];
  }
  return 0.5 * s;
}

/** Distance from p to segment ab. */
export function distToSegment(p: Vec2, a: Vec2, b: Vec2): number {
  const ab = sub(b, a);
  const d2 = len2(ab);
  if (d2 === 0) return Math.hypot(p[0] - a[0], p[1] - a[1]);
  let t = dot(sub(p, a), ab) / d2;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(p[0] - (a[0] + t * ab[0]), p[1] - (a[1] + t * ab[1]));
}

function orient(a: Vec2, b: Vec2, c: Vec2): number {
  return cross(sub(b, a), sub(c, a));
}

/** Proper intersection of open segments (shared endpoints excluded). */
function segmentsCross(p1: Vec2, p2: Vec2, p3: Vec2, p4: Vec2, eps: number): boolean {
  const d1 = orient(p3, p4, p1);
  const d2 = orient(p3, p4, p2);
  const d3 = orient(p1, p2, p3);
  const d4 = orient(p1, p2, p4);
  if (((d1 > eps && d2 < -eps) || (d1 < -eps && d2 > eps)) &&
      ((d3 > eps && d4 < -eps) || (d3 < -eps && d4 > eps))) return true;
  return false;
}

/** True when point and polygon vertex coincide or point lies on an edge. */
function onBoundary(p: Vec2, loop: Vec2[], eps: number): boolean {
  for (let i = 0; i < loop.length; i++) {
    if (distToSegment(p, loop[i], loop[(i + 1) % loop.length]) <= eps) return true;
  }
  return false;
}

/** Strictly-inside test (ray cast +x; boundary excluded by caller check). */
function strictlyInside(p: Vec2, loop: Vec2[]): boolean {
  let inside = false;
  for (let i = 0, j = loop.length - 1; i < loop.length; j = i++) {
    const a = loop[i], b = loop[j];
    if ((a[1] > p[1]) !== (b[1] > p[1])) {
      const x = a[0] + ((p[1] - a[1]) / (b[1] - a[1])) * (b[0] - a[0]);
      if (x > p[0]) inside = !inside;
    }
  }
  return inside;
}

// ---------------------------------------------------------------------------
// Arc approximation (deterministic chord subdivision by sagitta tolerance)
// ---------------------------------------------------------------------------

/** Sample an arc CCW from a0 to a1 (wraps past 2π; full circle when span>=2π). */
export function approximateArc(
  center: Vec2, radius: number, a0: number, a1: number, sagittaTol: number,
): Vec2[] {
  if (!(radius > 0)) throw new PatternError("invalid-arc", `radius must be > 0, got ${radius}`);
  if (!(sagittaTol > 0)) throw new PatternError("invalid-arc", `sagittaTol must be > 0, got ${sagittaTol}`);
  const TAU = 2 * Math.PI;
  let span = a1 - a0;
  while (span < 0) span += TAU;
  if (span === 0 || span >= TAU) span = TAU; // a0 == a1 (or more): full circle
  // Sagitta s = r(1 - cos(span/2n)) <= tol  =>  n >= span / (2*acos(1 - tol/r)).
  const c = Math.min(1, Math.max(-1, 1 - Math.min(sagittaTol, radius) / radius));
  const step = 2 * Math.acos(c);
  const n = Math.max(1, Math.ceil(span / step));
  const pts: Vec2[] = [];
  for (let i = 0; i <= n; i++) {
    const a = a0 + (span * i) / n;
    pts.push([center[0] + radius * Math.cos(a), center[1] + radius * Math.sin(a)]);
  }
  return pts;
}

// ---------------------------------------------------------------------------
// Panel validation + normalization
// ---------------------------------------------------------------------------

export interface ValidateOptions {
  /** Tolerance for duplicate/collinear cleanup (meters). Default 1e-9. */
  eps?: number;
  /** Minimum admissible loop area (m^2). Default 1e-12. */
  minArea?: number;
}

/** Drop consecutive duplicates and consecutive collinear points (in order). */
export function sanitizeLoop(points: Vec2[], eps: number): Vec2[] {
  const dedup: Vec2[] = [];
  for (const p of points) {
    const q = dedup[dedup.length - 1];
    if (q === undefined || Math.hypot(p[0] - q[0], p[1] - q[1]) > eps) dedup.push([p[0], p[1]]);
  }
  if (dedup.length > 0) {
    const f = dedup[0], l = dedup[dedup.length - 1];
    if (dedup.length > 1 && Math.hypot(f[0] - l[0], f[1] - l[1]) <= eps) dedup.pop();
  }
  // Strip collinear middles (geometry-preserving: same segments).
  let out = dedup;
  if (out.length >= 3) {
    for (;;) {
      let removed = false;
      const kept: Vec2[] = [];
      for (let i = 0; i < out.length; i++) {
        const a = out[(i + out.length - 1) % out.length];
        const b = out[i];
        const c = out[(i + 1) % out.length];
        const ab = sub(b, a), bc = sub(c, b);
        const denom = Math.sqrt(len2(ab) * len2(bc));
        const sin = denom === 0 ? 0 : Math.abs(cross(ab, bc)) / denom;
        if (out.length - kept.length > 3 && sin * Math.min(Math.sqrt(len2(ab)), Math.sqrt(len2(bc))) <= eps) {
          removed = true; // b adds nothing: drop it
        } else kept.push(b);
      }
      out = kept;
      if (!removed) break;
    }
  }
  return out;
}

function primitivesToLoop(prims: BoundaryPrimitive[], sagittaTol: number): Vec2[] {
  const pts: Vec2[] = [];
  for (const pr of prims) {
    if (pr.kind === "polyline") {
      for (const p of pr.points) pts.push([p[0], p[1]]);
    } else {
      const arc = approximateArc(pr.center, pr.radius, pr.a0, pr.a1, sagittaTol);
      for (const p of arc) pts.push(p);
    }
  }
  return pts;
}

/** True when loops share any proper crossing or touching edge pair. */
function loopsTouchOrCross(a: Vec2[], b: Vec2[], same: boolean, eps: number): boolean {
  for (let i = 0; i < a.length; i++) {
    const a0 = a[i], a1 = a[(i + 1) % a.length];
    for (let j = 0; j < b.length; j++) {
      if (same && (j === i || (j + 1) % b.length === i || j === (i + 1) % a.length)) continue;
      const b0 = b[j], b1 = b[(j + 1) % b.length];
      if (segmentsCross(a0, a1, b0, b1, eps)) return true;
      // Touching (non-crossing contact) is also invalid between boundaries.
      if (distToSegment(b0, a0, a1) <= eps || distToSegment(b1, a0, a1) <= eps ||
          distToSegment(a0, b0, b1) <= eps || distToSegment(a1, b0, b1) <= eps) {
        // Shared endpoints are only legal for consecutive edges of one loop.
        if (same) {
          const adjacent = (j === (i + 1) % a.length) || (i === (j + 1) % b.length);
          if (adjacent) continue;
        }
        return true;
      }
    }
  }
  return false;
}

export function validatePanel(
  panel: PatternPanel, opts: ValidateOptions & { sagittaTol?: number } = {},
): ValidatedPanel {
  const eps = opts.eps ?? 1e-9;
  const minArea = opts.minArea ?? 1e-12;
  const sagittaTol = opts.sagittaTol ?? 1e-4;
  if (!panel || typeof panel.id !== "string" || panel.id.length === 0) {
    throw new PatternError("invalid-panel", "panel needs a non-empty string id");
  }
  if (!Array.isArray(panel.outline) || panel.outline.length === 0) {
    throw new PatternError("empty-loop", "outer outline is empty");
  }
  let outer = sanitizeLoop(primitivesToLoop(panel.outline, sagittaTol), eps);
  if (outer.length < 3) throw new PatternError("empty-loop", "outer outline has fewer than 3 distinct points");
  const outerArea = signedArea(outer);
  if (Math.abs(outerArea) <= minArea) {
    throw new PatternError("zero-area-panel", `outer area ${outerArea} <= ${minArea}`);
  }
  let outerReversed = false;
  if (outerArea < 0) {
    outer = outer.reverse();
    outerReversed = true;
  }
  if (loopsTouchOrCross(outer, outer, true, eps)) {
    throw new PatternError("self-intersecting-boundary", "outer boundary self-intersects or self-touches");
  }
  const holes: Vec2[][] = [];
  const holesReversed: boolean[] = [];
  const holeList = panel.holes ?? [];
  for (let h = 0; h < holeList.length; h++) {
    let hole = sanitizeLoop(primitivesToLoop(holeList[h], sagittaTol), eps);
    if (hole.length < 3) throw new PatternError("invalid-hole", `hole ${h} has fewer than 3 distinct points`);
    const ha = signedArea(hole);
    if (Math.abs(ha) <= minArea) throw new PatternError("invalid-hole", `hole ${h} area ${ha} <= ${minArea}`);
    let rev = false;
    if (ha > 0) {
      hole = hole.reverse();
      rev = true;
    }
    if (loopsTouchOrCross(hole, hole, true, eps)) {
      throw new PatternError("invalid-hole", `hole ${h} self-intersects or self-touches`);
    }
    // Strict containment: a vertex strictly inside + no touch/cross.
    if (!strictlyInside(hole[0], outer) || onBoundary(hole[0], outer, eps)) {
      throw new PatternError("hole-outside", `hole ${h} is not strictly inside the outer boundary`);
    }
    for (const q of hole) {
      if (!strictlyInside(q, outer) && !onBoundary(q, outer, eps)) {
        throw new PatternError("hole-outside", `hole ${h} escapes the outer boundary`);
      }
    }
    if (loopsTouchOrCross(outer, hole, false, eps)) {
      throw new PatternError("hole-touching", `hole ${h} touches or crosses the outer boundary`);
    }
    holes.push(hole);
    holesReversed.push(rev);
  }
  for (let i = 0; i < holes.length; i++) {
    for (let j = i + 1; j < holes.length; j++) {
      if (loopsTouchOrCross(holes[i], holes[j], false, eps)) {
        throw new PatternError("holes-overlap", `holes ${i} and ${j} touch or cross`);
      }
      if (strictlyInside(holes[j][0], holes[i]) || strictlyInside(holes[i][0], holes[j])) {
        throw new PatternError("holes-overlap", `holes ${i} and ${j} nest (nested holes unsupported)`);
      }
    }
  }
  return {
    id: panel.id, outer, holes, outerReversed, holesReversed,
    grainAngleRad: panel.grainAngleRad, materialId: panel.materialId,
  };
}

/** Closed-panel membership: inside outer, outside all holes; boundary counts in. */
export function pointInPanel(p: Vec2, panel: ValidatedPanel, eps = 1e-9): boolean {
  if (onBoundary(p, panel.outer, eps)) return true;
  if (!strictlyInside(p, panel.outer)) return false;
  for (const h of panel.holes) {
    if (onBoundary(p, h, eps)) return true;
    if (strictlyInside(p, h)) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Boundary parameterization (arclength)
// ---------------------------------------------------------------------------

export interface LoopParameterization {
  /** Cumulative arclength at each vertex (length n+1, last = total). */
  breaks: number[];
  total: number;
}

export function parameterizeLoop(loop: Vec2[]): LoopParameterization {
  const breaks: number[] = [0];
  for (let i = 0; i < loop.length; i++) {
    const a = loop[i], b = loop[(i + 1) % loop.length];
    breaks.push(breaks[i] + Math.hypot(b[0] - a[0], b[1] - a[1]));
  }
  return { breaks, total: breaks[loop.length] };
}

/** Arclength interpolation: t in [0,1) maps around the loop from vertex 0. */
export function boundaryPoint(loop: Vec2[], param: LoopParameterization, t: number): Vec2 {
  if (param.total === 0) throw new PatternError("empty-loop", "cannot parameterize a zero-length loop");
  let u = ((t % 1) + 1) % 1;
  let s = u * param.total;
  // Deterministic linear scan (loop sizes are pattern-scale, not hot-loop).
  let i = 0;
  while (i < loop.length - 1 && s > param.breaks[i + 1]) i++;
  const segLen = param.breaks[i + 1] - param.breaks[i];
  const f = segLen === 0 ? 0 : (s - param.breaks[i]) / segLen;
  const a = loop[i], b = loop[(i + 1) % loop.length];
  return [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])];
}

// ---------------------------------------------------------------------------
// Deterministic constrained triangulation (hole bridging + ear clipping)
// ---------------------------------------------------------------------------

function bridgeHoles(outer: Vec2[], holes: Vec2[][], eps: number): Vec2[] {
  // Splice each hole into the current simple polygon via a bridge segment
  // that touches no other edge. Candidates are tried in index order
  // (hole vertex, then polygon vertex), so the result is deterministic.
  // The merged polygon keeps positive (CCW) area: outer CCW minus holes.
  let poly = outer.map((p) => [p[0], p[1]] as Vec2);
  for (const hole of holes) {
    let done = false;
    for (let hi = 0; hi < hole.length && !done; hi++) {
      for (let pi = 0; pi < poly.length && !done; pi++) {
        if (!bridgeProbe(poly, hole, hi, pi, eps)) continue;
        const merged: Vec2[] = [];
        for (let i = 0; i <= pi; i++) merged.push(poly[i]);
        for (let k = 0; k <= hole.length; k++) merged.push(hole[(hi + k) % hole.length]);
        for (let i = pi; i < poly.length; i++) merged.push(poly[i]);
        poly = merged;
        done = true;
      }
    }
    if (!done) throw new PatternError("unbridgeable", "no non-touching bridge segment found for a hole");
  }
  return poly;
}

/**
 * Shared bridge probe (used identically by bridging and tag replay, so the
 * two can never diverge). Edges incident to either bridge endpoint are
 * skipped; every other edge must be strictly clear of both endpoints and
 * uncrossed by the segment.
 */
function bridgeProbe(poly: Vec2[], hole: Vec2[], hi: number, pi: number, eps: number): boolean {
  const a = hole[hi], b = poly[pi];
  if (Math.hypot(a[0] - b[0], a[1] - b[1]) <= eps) return false;
  const clearExcept = (loop: Vec2[], skipA: number, skipB: number): boolean => {
    for (let i = 0; i < loop.length; i++) {
      if (i === skipA || i === skipB) continue;
      const c = loop[i], d = loop[(i + 1) % loop.length];
      // Edge incident to the opposite endpoint is skipped by the same rule
      // (each loop passes its own endpoint index as both skipA/skipB only
      // for its own vertex; the other endpoint is never a vertex of this
      // loop for valid panels, and touching it rejects below).
      if (segmentsCross(a, b, c, d, eps)) return false;
      if (distToSegment(a, c, d) <= eps) return false;
      if (distToSegment(b, c, d) <= eps) return false;
    }
    return true;
  };
  // In its own loop, skip the two edges incident to the endpoint vertex.
  const prev = (v: number, n: number): number => (v + n - 1) % n;
  if (!clearExcept(hole, prev(hi, hole.length), hi)) return false;
  if (!clearExcept(poly, prev(pi, poly.length), pi)) return false;
  return true;
}

function pointInTriangleStrict(p: Vec2, a: Vec2, b: Vec2, c: Vec2, eps: number): boolean {
  // Barycentric with an eps margin: points on edges do NOT count as inside.
  const v0 = sub(c, a), v1 = sub(b, a), v2 = sub(p, a);
  const d00 = dot(v0, v0), d01 = dot(v0, v1), d11 = dot(v1, v1);
  const d20 = dot(v2, v0), d21 = dot(v2, v1);
  const den = d00 * d11 - d01 * d01;
  if (Math.abs(den) <= eps) return false;
  const v = (d11 * d20 - d01 * d21) / den;
  const w = (d00 * d21 - d01 * d20) / den;
  const u = 1 - v - w;
  return u > eps && v > eps && w > eps;
}

function triArea2(a: Vec2, b: Vec2, c: Vec2): number {
  return Math.abs(cross(sub(b, a), sub(c, a))) * 0.5;
}

function earClip(poly: Vec2[], minTriArea: number): Array<[number, number, number]> {
  // poly must be simple + CCW. Returns vertex-index triples into poly.
  // Convexity uses a scale-free sine threshold; areas use minTriArea.
  const n = poly.length;
  if (n < 3) throw new PatternError("degenerate-output", "polygon has fewer than 3 vertices");
  const areaOk = (a: Vec2, b: Vec2, c: Vec2): boolean => triArea2(a, b, c) > minTriArea;
  if (n === 3) {
    if (!areaOk(poly[0], poly[1], poly[2])) {
      throw new PatternError("degenerate-output", "sole triangle has zero area");
    }
    return [[0, 1, 2]];
  }
  const alive: number[] = poly.map((_, i) => i);
  const tris: Array<[number, number, number]> = [];
  const isConvex = (pos: number): boolean => {
    const prev = alive[(pos + alive.length - 1) % alive.length];
    const cur = alive[pos];
    const next = alive[(pos + 1) % alive.length];
    const e1 = sub(poly[cur], poly[prev]);
    const e2 = sub(poly[next], poly[cur]);
    const denom = Math.sqrt(len2(e1) * len2(e2));
    if (denom === 0) return false;
    return cross(e1, e2) / denom > 1e-12; // scale-free left-turn test
  };
  const isEar = (pos: number): boolean => {
    if (!isConvex(pos)) return false;
    const prev = alive[(pos + alive.length - 1) % alive.length];
    const cur = alive[pos];
    const next = alive[(pos + 1) % alive.length];
    for (let k = 0; k < alive.length; k++) {
      const vi = alive[k];
      if (vi === prev || vi === cur || vi === next) continue;
      if (pointInTriangleStrict(poly[vi], poly[prev], poly[cur], poly[next], 1e-12)) return false;
    }
    return true;
  };
  let guard = 0;
  let pos = 0;
  while (alive.length > 3) {
    if (guard++ > alive.length * alive.length * 4) {
      throw new PatternError("degenerate-output", "ear clipping stalled (no ear found)");
    }
    pos = pos % alive.length;
    if (isEar(pos)) {
      const prev = alive[(pos + alive.length - 1) % alive.length];
      const cur = alive[pos];
      const next = alive[(pos + 1) % alive.length];
      tris.push([prev, cur, next]);
      alive.splice(pos, 1);
    } else {
      pos++;
    }
  }
  const [a, b, c] = alive;
  if (!areaOk(poly[a], poly[b], poly[c])) {
    throw new PatternError("degenerate-output", "final triangle has zero area");
  }
  tris.push([a, b, c]);
  return tris;
}

export interface TriangulateOptions extends ValidateOptions {
  sagittaTol?: number;
  /** Minimum admissible triangle area (m^2). Default 1e-14. */
  minTriArea?: number;
}

export function triangulatePatternPanel(
  panel: PatternPanel, opts: TriangulateOptions = {},
): TriangulatedPanel {
  const v = validatePanel(panel, opts);
  try {
    return triangulateValidated(v, opts);
  } catch (error) {
    if (!(error instanceof PatternError) || error.code !== "degenerate-output") throw error;
    // Recovery (G16): a 180° boundary vertex (e.g. from splitting a straight
    // edge) can strand the order-dependent ear clipper with a collinear final
    // triple. Drop near-collinear loop vertices and retry exactly once.
    // Inputs that triangulated before never reach this path, so their
    // outputs are byte-identical.
    const stripped = stripCollinearLoops(v, opts.eps ?? 1e-9);
    if (!stripped) throw error;
    const retry: PatternPanel = {
      id: panel.id,
      outline: [{ kind: "polyline", points: stripped.outer }],
      holes: stripped.holes.map((h) => [{ kind: "polyline" as const, points: h }]),
      grainAngleRad: panel.grainAngleRad,
      materialId: panel.materialId,
    };
    return triangulateValidated(validatePanel(retry, opts), opts);
  }
}

/**
 * Drop consecutive near-collinear vertices (sine·minEdge <= eps) from
 * validated loops. Returns null when nothing can be dropped (caller then
 * rethrows the original error). Geometry-preserving up to eps.
 */
function stripCollinearLoops(v: ValidatedPanel, eps: number): { outer: Vec2[]; holes: Vec2[][] } | null {
  const strip = (loop: Vec2[]): Vec2[] | null => {
    if (loop.length < 4) return null;
    const kept: Vec2[] = [];
    for (let i = 0; i < loop.length; i++) {
      const a = loop[(i + loop.length - 1) % loop.length];
      const b = loop[i];
      const c = loop[(i + 1) % loop.length];
      const abx = b[0] - a[0], aby = b[1] - a[1];
      const bcx = c[0] - b[0], bcy = c[1] - b[1];
      const denom = Math.hypot(abx, aby) * Math.hypot(bcx, bcy);
      const sin = denom === 0 ? 0 : Math.abs(abx * bcy - aby * bcx) / denom;
      if (sin * Math.min(Math.hypot(abx, aby), Math.hypot(bcx, bcy)) <= eps) continue;
      kept.push(b);
    }
    if (kept.length < 3 || kept.length === loop.length) return null;
    return kept;
  };
  const outer = strip(v.outer);
  if (!outer) return null;
  const holes: Vec2[][] = [];
  for (const hole of v.holes) {
    const stripped = strip(hole);
    holes.push(stripped ?? hole.map((p) => [p[0], p[1]] as Vec2));
  }
  return { outer, holes };
}

function triangulateValidated(
  v: ValidatedPanel, opts: TriangulateOptions = {},
): TriangulatedPanel {
  const eps = opts.eps ?? 1e-9;
  const minTriArea = opts.minTriArea ?? 1e-14;
  const poly = bridgeHoles(v.outer, v.holes, eps);
  const tris = earClip(poly, minTriArea);
  const n = poly.length;
  const vertices = new Float64Array(n * 2);
  for (let i = 0; i < n; i++) {
    vertices[i * 2] = poly[i][0];
    vertices[i * 2 + 1] = poly[i][1];
  }
  const triangles = new Uint32Array(tris.length * 3);
  tris.forEach((t, k) => {
    triangles[k * 3] = t[0];
    triangles[k * 3 + 1] = t[1];
    triangles[k * 3 + 2] = t[2];
  });
  // Boundary edges: edges used by exactly one triangle, mapped back to the
  // source loops (outer + holes in order) with arclength parameters.
  const loops: Vec2[][] = [v.outer, ...v.holes];
  const params = loops.map(parameterizeLoop);
  const loopOfVertex = new Array<number>(n).fill(-1);
  const indexInLoop = new Array<number>(n).fill(-1);
  {
    // Reconstruct the merged-vertex -> (loop, index) map the same way
    // bridgeHoles splices: replay bridging while tracking tags.
    type Tagged = { p: Vec2; loop: number; idx: number };
    let tagged: Tagged[] = outerTagged(v.outer);
    for (let h = 0; h < v.holes.length; h++) {
      tagged = spliceTagged(tagged, v.holes[h], h + 1, eps);
    }
    if (tagged.length !== n) throw new PatternError("degenerate-output", "tag replay diverged from bridging");
    for (let i = 0; i < n; i++) {
      loopOfVertex[i] = tagged[i].loop;
      indexInLoop[i] = tagged[i].idx;
    }
  }
  const useCount = new Map<string, number>();
  const edgeKey = (a: number, b: number): string => (a < b ? `${a}_${b}` : `${b}_${a}`);
  for (let t = 0; t < tris.length; t++) {
    const [a, b, c] = tris[t];
    for (const k of [edgeKey(a, b), edgeKey(b, c), edgeKey(c, a)]) {
      useCount.set(k, (useCount.get(k) ?? 0) + 1);
    }
  }
  const boundaryEdges: BoundaryEdgeTag[] = [];
  for (let t = 0; t < tris.length; t++) {
    const [a, b, c] = tris[t];
    const edges: Array<[number, number]> = [[a, b], [b, c], [c, a]];
    for (const [u, w] of edges) {
      if (useCount.get(edgeKey(u, w)) !== 1) continue;
      const lu = loopOfVertex[u], lw = loopOfVertex[w];
      if (lu < 0 || lw < 0) {
        throw new PatternError("degenerate-output", "untagged boundary edge (bridging desync)");
      }
      // Bridge slits join different loops: traversed twice geometrically but
      // as distinct index pairs, each used once — they are interior, not
      // boundary. A same-loop once-used edge that is not loop-consecutive is
      // a genuine desync and throws below.
      if (lu !== lw) continue;
      const loop = loops[lu];
      const iu = indexInLoop[u], iw = indexInLoop[w];
      // Consecutive along the loop (in either direction; bridges double back).
      const fwd = (iu + 1) % loop.length === iw;
      const bwd = (iw + 1) % loop.length === iu;
      if (!fwd && !bwd) {
        throw new PatternError("degenerate-output", "non-loop edge used once (bridging desync)");
      }
      const edgeIndex = fwd ? iu : iw;
      const prm = params[lu];
      const t0 = prm.breaks[edgeIndex] / prm.total;
      const t1 = prm.breaks[edgeIndex + 1] / prm.total;
      boundaryEdges.push({
        a: u, b: w,
        loop: lu === 0 ? "outer" : lu - 1,
        edgeIndex, t0, t1,
      });
    }
  }
  // Deterministic order: sort by (loop, edgeIndex, a, b).
  boundaryEdges.sort((p, q) => {
    const pl = p.loop === "outer" ? -1 : (p.loop as number);
    const ql = q.loop === "outer" ? -1 : (q.loop as number);
    return pl - ql || p.edgeIndex - q.edgeIndex || p.a - q.a || p.b - q.b;
  });
  // Degenerate-output gate: every triangle must carry real area.
  for (let t = 0; t < tris.length; t++) {
    const [a, b, c] = tris[t];
    const ar = triArea2(poly[a], poly[b], poly[c]);
    if (!(ar > minTriArea)) {
      throw new PatternError("degenerate-output", `triangle ${t} area ${ar} <= ${minTriArea}`);
    }
  }
  return {
    panelId: v.id, vertices, triangles, boundaryEdges,
    grainAngleRad: v.grainAngleRad, materialId: v.materialId,
  };
}

function outerTagged(outer: Vec2[]): Array<{ p: Vec2; loop: number; idx: number }> {
  return outer.map((p, i) => ({ p, loop: 0, idx: i }));
}

/** Tagged replay of bridgeHoles splicing (same candidate order/probe). */
function spliceTagged(
  tagged: Array<{ p: Vec2; loop: number; idx: number }>,
  hole: Vec2[], holeNo: number, eps: number,
): Array<{ p: Vec2; loop: number; idx: number }> {
  const plain = tagged.map((t) => t.p);
  for (let hi = 0; hi < hole.length; hi++) {
    for (let pi = 0; pi < plain.length; pi++) {
      if (!bridgeProbe(plain, hole, hi, pi, eps)) continue;
      const merged: Array<{ p: Vec2; loop: number; idx: number }> = [];
      for (let i = 0; i <= pi; i++) merged.push(tagged[i]);
      for (let k = 0; k <= hole.length; k++) {
        const src = hole[(hi + k) % hole.length];
        merged.push({ p: src, loop: holeNo, idx: (hi + k) % hole.length });
      }
      for (let i = pi; i < tagged.length; i++) merged.push(tagged[i]);
      return merged;
    }
  }
  throw new PatternError("unbridgeable", "tag replay found no bridge (diverged from bridging)");
}

// ---------------------------------------------------------------------------
// Quality metrics
// ---------------------------------------------------------------------------

function denseTrueBoundary(panel: PatternPanel, perEdgeSamples: number): Vec2[] {
  // Dense samples of the TRUE boundary (arcs sampled finely; polylines as-is
  // plus midpoints) for Hausdorff measurement against triangulated edges.
  const pts: Vec2[] = [];
  const pushLoop = (prims: BoundaryPrimitive[]): void => {
    for (const pr of prims) {
      if (pr.kind === "polyline") {
        for (let i = 0; i < pr.points.length; i++) {
          const a = pr.points[i], b = pr.points[(i + 1) % pr.points.length];
          pts.push([a[0], a[1]]);
          for (let k = 1; k < perEdgeSamples; k++) {
            const f = k / perEdgeSamples;
            pts.push([a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])]);
          }
        }
      } else {
        const approx = approximateArc(pr.center, pr.radius, pr.a0, pr.a1, 1e-6);
        for (let i = 0; i < approx.length - 1; i++) {
          const a = approx[i], b = approx[i + 1];
          pts.push([a[0], a[1]]);
          for (let k = 1; k < perEdgeSamples; k++) {
            const f = k / perEdgeSamples;
            pts.push([a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])]);
          }
        }
      }
    }
  };
  pushLoop(panel.outline);
  for (const h of panel.holes ?? []) pushLoop(h);
  return pts;
}

export function measurePanelQuality(
  panel: PatternPanel, tri: TriangulatedPanel, opts: ValidateOptions = {},
): PanelQuality {
  const v = validatePanel(panel, opts);
  const panelArea = Math.abs(signedArea(v.outer)) - v.holes.reduce((s, h) => s + Math.abs(signedArea(h)), 0);
  const pos = (i: number): Vec2 => [tri.vertices[i * 2], tri.vertices[i * 2 + 1]];
  let meshArea = 0;
  let minTriArea = Infinity;
  let minShape = Infinity;
  for (let t = 0; t < tri.triangles.length; t += 3) {
    const a = pos(tri.triangles[t]), b = pos(tri.triangles[t + 1]), c = pos(tri.triangles[t + 2]);
    const ar = triArea2(a, b, c);
    meshArea += ar;
    minTriArea = Math.min(minTriArea, ar);
    const e1 = len2(sub(b, a)), e2 = len2(sub(c, b)), e3 = len2(sub(a, c));
    const q = (4 * Math.sqrt(3) * ar) / Math.max(e1 + e2 + e3, 1e-300);
    minShape = Math.min(minShape, q);
  }
  // Hausdorff: dense true-boundary samples -> triangulated boundary segments.
  const segs: Array<[Vec2, Vec2]> = tri.boundaryEdges.map((e) => [pos(e.a), pos(e.b)]);
  let dev = 0;
  for (const p of denseTrueBoundary(panel, 8)) {
    let best = Infinity;
    for (const [a, b] of segs) best = Math.min(best, distToSegment(p, a, b));
    dev = Math.max(dev, best);
  }
  return {
    panelArea, meshArea,
    areaRelErr: Math.abs(meshArea - panelArea) / Math.max(panelArea, 1e-300),
    boundaryDeviation: dev,
    minTriArea, minShapeQuality: minShape,
  };
}

// ---------------------------------------------------------------------------
// Rest-space bridge (pattern coordinates ARE the FEM rest metric)
// ---------------------------------------------------------------------------

export interface RestMesh {
  /** Flat xyz, y = 0. Direct preprocess() input. */
  positions: Float32Array;
  /** Pattern xy per vertex. Direct preprocess() uv input. */
  uv: Float32Array;
  indices: Uint32Array;
}

/** Lay the triangulated panel flat: rest (x, 0, y), uv = pattern xy. */
export function panelToRestMesh(tri: TriangulatedPanel): RestMesh {
  const n = tri.vertices.length / 2;
  const positions = new Float32Array(n * 3);
  const uv = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    positions[i * 3] = tri.vertices[i * 2];
    positions[i * 3 + 1] = 0;
    positions[i * 3 + 2] = tri.vertices[i * 2 + 1];
    uv[i * 2] = tri.vertices[i * 2];
    uv[i * 2 + 1] = tri.vertices[i * 2 + 1];
  }
  return { positions, uv, indices: Uint32Array.from(tri.triangles) };
}

// ---------------------------------------------------------------------------
// Serialization (plain data; validation on the way back in)
// ---------------------------------------------------------------------------

export function panelToJSON(panel: PatternPanel): string {
  return JSON.stringify(panel);
}

export function panelFromJSON(json: string): PatternPanel {
  let o: unknown;
  try {
    o = JSON.parse(json);
  } catch {
    throw new PatternError("invalid-json", "panel JSON does not parse");
  }
  const p = o as PatternPanel;
  if (!p || typeof p !== "object" || !Array.isArray((p as PatternPanel).outline)) {
    throw new PatternError("invalid-json", "panel JSON has no outline array");
  }
  return p;
}

export function triangulatedToJSON(tri: TriangulatedPanel): string {
  return JSON.stringify({
    panelId: tri.panelId,
    vertices: Array.from(tri.vertices),
    triangles: Array.from(tri.triangles),
    boundaryEdges: tri.boundaryEdges,
    grainAngleRad: tri.grainAngleRad,
    materialId: tri.materialId,
  });
}

export function triangulatedFromJSON(json: string): TriangulatedPanel {
  let o: unknown;
  try {
    o = JSON.parse(json);
  } catch {
    throw new PatternError("invalid-json", "triangulation JSON does not parse");
  }
  const p = o as {
    panelId: unknown; vertices: unknown; triangles: unknown;
    boundaryEdges: unknown; grainAngleRad: unknown; materialId: unknown;
  };
  if (!p || typeof p !== "object" || typeof p.panelId !== "string" ||
    !Array.isArray(p.vertices) || !Array.isArray(p.triangles) || !Array.isArray(p.boundaryEdges)) {
    throw new PatternError("invalid-json", "triangulation JSON has wrong shape");
  }
  return {
    panelId: p.panelId,
    vertices: Float64Array.from(p.vertices as number[]),
    triangles: Uint32Array.from(p.triangles as number[]),
    boundaryEdges: p.boundaryEdges as BoundaryEdgeTag[],
    grainAngleRad: Number(p.grainAngleRad),
    materialId: String(p.materialId),
  };
}
