// G9A — robust 2D geometry predicates for pattern CAD.
//
// Pure functions, no document state. Scale-aware orientation tests (sine
// comparisons instead of raw cross products) so predicates behave the same on
// millimeter-scale and meter-scale patterns. Nothing in this file snaps:
// callers decide when coordinates are close enough (see snap.ts).
//
// These helpers operate on RAW coordinates. The PatternDocument stores panel
// points in panel-local space and validates in panel-local space, so G9
// editing operations call these with local coordinates; UI-facing queries
// (hit-test, snap) transform into workspace space first and then call them.
// Arcs use the G8A model: center + radius + start angle + signed sweep,
// which is exactly what PatternSegment (kind "arc": centerPointId + sweepRad)
// resolves to once endpoints are known.
//
// Determinism: every function is order-stable over its inputs; no randomness,
// no hash iteration, no implicit tolerance defaults except where documented.

import type { Vec2 } from "../pattern/pattern-geometry.js";

export type { Vec2 };

export interface ArcGeometry {
  center: Vec2;
  radius: number;
  /** Angle (radians) of the start endpoint relative to the center. */
  a0: number;
  /** Signed sweep: positive = CCW from start to end, strictly within (-2pi, 2pi). */
  sweep: number;
}

// ---------------------------------------------------------------------------
// Vector helpers
// ---------------------------------------------------------------------------

/** Signed cross product of (b-a) x (c-a) (2x signed area of the triangle). */
export function orient(a: Vec2, b: Vec2, c: Vec2): number {
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

export function sub(a: Vec2, b: Vec2): Vec2 {
  return [a[0] - b[0], a[1] - b[1]];
}

export function add(a: Vec2, b: Vec2): Vec2 {
  return [a[0] + b[0], a[1] + b[1]];
}

export function dot(a: Vec2, b: Vec2): number {
  return a[0] * b[0] + a[1] * b[1];
}

export function cross(a: Vec2, b: Vec2): number {
  return a[0] * b[1] - a[1] * b[0];
}

export function len(a: Vec2): number {
  return Math.hypot(a[0], a[1]);
}

export function dist(a: Vec2, b: Vec2): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

export function normalize(a: Vec2): Vec2 {
  const l = len(a);
  if (l === 0) return [0, 0];
  return [a[0] / l, a[1] / l];
}

/** Left normal of ab (rotate +90 degrees). Used by offset construction. */
export function leftNormal(a: Vec2, b: Vec2): Vec2 {
  return [-(b[1] - a[1]), b[0] - a[0]];
}

export function lerp(a: Vec2, b: Vec2, t: number): Vec2 {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
}

export function finiteVec(a: Vec2): boolean {
  return Number.isFinite(a[0]) && Number.isFinite(a[1]);
}

// ---------------------------------------------------------------------------
// Scale-aware predicates
// ---------------------------------------------------------------------------

/**
 * True when |sin(angle between ab and ac)| <= eps — i.e. a, b, c are
 * collinear to within the angular tolerance. Dimensionless, so it behaves
 * identically at any coordinate scale. Zero-length legs count as collinear.
 */
export function collinear(a: Vec2, b: Vec2, c: Vec2, eps: number): boolean {
  const ab = sub(b, a);
  const ac = sub(c, a);
  const denom = len(ab) * len(ac);
  if (denom === 0) return true;
  return Math.abs(cross(ab, ac)) <= eps * denom;
}

/** True when the turn a->b->c is counter-clockwise beyond the sine tolerance. */
export function leftTurn(a: Vec2, b: Vec2, c: Vec2, eps: number): boolean {
  const ab = sub(b, a);
  const bc = sub(c, b);
  const denom = len(ab) * len(bc);
  if (denom === 0) return false;
  return cross(ab, bc) > eps * denom;
}

// ---------------------------------------------------------------------------
// Segment queries
// ---------------------------------------------------------------------------

export interface SegmentProjection {
  /** Clamped parameter along ab, in [0,1]. */
  t: number;
  pos: Vec2;
  distance: number;
}

/** Nearest point on the CLOSED segment ab. */
export function projectOnSegment(p: Vec2, a: Vec2, b: Vec2): SegmentProjection {
  const ab = sub(b, a);
  const d2 = dot(ab, ab);
  if (d2 === 0) return { t: 0, pos: [a[0], a[1]], distance: dist(p, a) };
  let t = dot(sub(p, a), ab) / d2;
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  const pos = lerp(a, b, t);
  return { t, pos, distance: dist(p, pos) };
}

/** Raw (unclamped) parameter of p along the infinite line a->b. */
export function paramAlong(p: Vec2, a: Vec2, b: Vec2): number {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const d2 = dx * dx + dy * dy;
  if (d2 === 0) return 0;
  return ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / d2;
}

/**
 * Infinite-line intersection (line p1->p2 with line p3->p4).
 * Returns null when the lines are parallel within the scale-aware sine
 * tolerance `eps`, or when either line has zero direction.
 */
export function lineIntersection(
  p1: Vec2, p2: Vec2, p3: Vec2, p4: Vec2, eps: number,
): Vec2 | null {
  const d1 = sub(p2, p1);
  const d2 = sub(p4, p3);
  const den = cross(d1, d2);
  const denScale = len(d1) * len(d2);
  if (denScale === 0 || Math.abs(den) <= eps * denScale) return null;
  const t = cross(sub(p3, p1), d2) / den;
  return lerp(p1, p2, t);
}

export type SegmentIntersectionKind = "proper" | "touch-interior" | "touch-endpoint";

export interface SegmentIntersection {
  kind: SegmentIntersectionKind;
  pos: Vec2;
  /** Parameter along the first segment, clamped to [0,1]. */
  t1: number;
  /** Parameter along the second segment, clamped to [0,1]. */
  t2: number;
}

/**
 * Closed-segment intersection with classification. Deterministic single
 * result. "proper" = the segments cross at a parameter strictly interior to
 * both; "touch-interior" = an endpoint of one lies on the interior of the
 * other (or parallel contact within eps); "touch-endpoint" = contact only at
 * (near-)endpoints of both. Shared endpoints of adjacent boundary edges are
 * the caller's business — this function reports them as "touch-endpoint".
 */
export function segmentIntersection(
  p1: Vec2, p2: Vec2, p3: Vec2, p4: Vec2, eps: number,
): SegmentIntersection | null {
  const d1 = sub(p2, p1);
  const d2 = sub(p4, p3);
  const den = cross(d1, d2);
  const l1 = len(d1);
  const l2 = len(d2);
  const denScale = l1 * l2;
  if (denScale === 0) return null;

  let t: number;
  let u: number;
  if (Math.abs(den) > eps * denScale) {
    t = cross(sub(p3, p1), d2) / den;
    u = cross(sub(p3, p1), d1) / den;
  } else {
    // Parallel within tolerance: contact only when an endpoint of one
    // segment lies on the other (within eps). Deterministic probe order.
    const probes: Array<{ t: number; u: number }> = [
      { t: 0, u: projectOnSegment(p1, p3, p4).t },
      { t: 1, u: projectOnSegment(p2, p3, p4).t },
      { t: projectOnSegment(p3, p1, p2).t, u: 0 },
      { t: projectOnSegment(p4, p1, p2).t, u: 1 },
    ];
    for (const pr of probes) {
      const q1 = lerp(p1, p2, pr.t);
      const q2 = lerp(p3, p4, pr.u);
      if (dist(q1, q2) <= eps) {
        const ends1 = pr.t <= eps || pr.t >= 1 - eps;
        const ends2 = pr.u <= eps || pr.u >= 1 - eps;
        return {
          kind: ends1 && ends2 ? "touch-endpoint" : "touch-interior",
          pos: q1,
          t1: pr.t,
          t2: pr.u,
        };
      }
    }
    return null;
  }

  const on1 = t >= -eps && t <= 1 + eps;
  const on2 = u >= -eps && u <= 1 + eps;
  if (!on1 || !on2) return null;

  const tc = t < 0 ? 0 : t > 1 ? 1 : t;
  const uc = u < 0 ? 0 : u > 1 ? 1 : u;
  const pos = lerp(p1, p2, tc);
  const ends1 = tc <= eps || tc >= 1 - eps;
  const ends2 = uc <= eps || uc >= 1 - eps;
  const kind: SegmentIntersectionKind =
    ends1 && ends2 ? "touch-endpoint" : ends1 || ends2 ? "touch-interior" : "proper";
  return { kind, pos, t1: tc, t2: uc };
}

// ---------------------------------------------------------------------------
// Arc math (center + radius + start angle + signed sweep)
// ---------------------------------------------------------------------------

/** End angle of an arc (a0 + sweep, unnormalized). */
export function arcEndAngle(arc: ArcGeometry): number {
  return arc.a0 + arc.sweep;
}

export function arcLength(arc: ArcGeometry): number {
  return arc.radius * Math.abs(arc.sweep);
}

/** Point at fraction f in [0,1] along the sweep (f clamped). */
export function pointOnArc(arc: ArcGeometry, f: number): Vec2 {
  const fc = f < 0 ? 0 : f > 1 ? 1 : f;
  const ang = arc.a0 + arc.sweep * fc;
  return [
    arc.center[0] + arc.radius * Math.cos(ang),
    arc.center[1] + arc.radius * Math.sin(ang),
  ];
}

/** Angle at fraction f along the sweep (f NOT clamped — for extrapolation). */
export function angleAt(arc: ArcGeometry, f: number): number {
  return arc.a0 + arc.sweep * f;
}

/** Is `ang` within the arc's sweep? `angEps` is radians (scale-free). */
export function angleOnArc(arc: ArcGeometry, ang: number, angEps: number): boolean {
  const rel = normalizeAngle(ang - arc.a0);
  if (arc.sweep > 0) return rel >= -angEps && rel <= arc.sweep + angEps;
  return rel <= angEps && rel >= arc.sweep - angEps;
}

/** Normalize an angle delta into (-pi, pi]. */
export function normalizeAngle(d: number): number {
  let x = d;
  const TAU = 2 * Math.PI;
  while (x > Math.PI) x -= TAU;
  while (x <= -Math.PI) x += TAU;
  return x;
}

/**
 * Deterministic arc sampling with sagitta-bounded subdivision.
 * Returns count+1 points from the start angle to the end angle inclusive.
 * The subdivision count depends only on (radius, sweep, sagittaTol), so the
 * same arc always samples to the same polyline.
 */
export function sampleArc(
  center: Vec2, radius: number, a0: number, sweep: number, sagittaTol: number,
): Vec2[] {
  if (!(radius > 0)) throw new Error(`sampleArc: radius must be > 0, got ${radius}`);
  if (!Number.isFinite(sweep) || sweep === 0 || Math.abs(sweep) >= 2 * Math.PI) {
    throw new Error(`sampleArc: sweep must be finite, nonzero, and |sweep| < 2pi, got ${sweep}`);
  }
  const tol = Math.max(Number.MIN_VALUE, Math.min(sagittaTol, radius));
  const step = 2 * Math.acos(Math.max(-1, Math.min(1, 1 - tol / radius)));
  const n = Math.max(1, Math.ceil(Math.abs(sweep) / step));
  const pts: Vec2[] = [];
  for (let i = 0; i <= n; i++) {
    const ang = a0 + (sweep * i) / n;
    pts.push([center[0] + radius * Math.cos(ang), center[1] + radius * Math.sin(ang)]);
  }
  return pts;
}

/**
 * Intersections of the CLOSED arc with the segment p1-p2.
 * Deterministic order: by segment parameter t ascending, then by coordinates.
 * `angEps` is an angular tolerance in radians (use linearEps / radius).
 */
export function arcSegmentIntersections(
  arc: ArcGeometry, p1: Vec2, p2: Vec2, angEps: number,
): Array<{ pos: Vec2; t: number }> {
  const out: Array<{ pos: Vec2; t: number }> = [];
  const d = sub(p2, p1);
  const A = dot(d, d);
  if (A === 0) return out;
  const f = sub(p1, arc.center);
  const B = 2 * dot(f, d);
  const C = dot(f, f) - arc.radius * arc.radius;
  const disc = B * B - 4 * A * C;
  const lenEps = angEps * Math.max(1, arc.radius);
  const push = (pos: Vec2, t: number): void => {
    if (Math.abs(dist(pos, arc.center) - arc.radius) > lenEps) return;
    const ang = Math.atan2(pos[1] - arc.center[1], pos[0] - arc.center[0]);
    if (!angleOnArc(arc, ang, angEps)) return;
    if (!out.some((o) => dist(o.pos, pos) <= lenEps)) out.push({ pos, t });
  };
  if (disc >= 0) {
    const sq = Math.sqrt(disc);
    const roots = sq === 0 ? [-B / (2 * A)] : [(-B - sq) / (2 * A), (-B + sq) / (2 * A)];
    for (const t of roots) {
      if (t < 0 || t > 1) continue;
      push(lerp(p1, p2, t), t);
    }
  }
  // Endpoint touches (root outside [0,1] but endpoint on the circle).
  push(p1, 0);
  push(p2, 1);
  out.sort((p, q) => p.t - q.t || p.pos[0] - q.pos[0] || p.pos[1] - q.pos[1]);
  return out;
}

/**
 * Intersections of two CLOSED arcs.
 * Deterministic order: by angle on the first arc ascending.
 * Same-circle arcs return [] (infinitely many hits — callers must reject
 * coincident circles before relying on a unique intersection).
 */
export function arcArcIntersections(a1: ArcGeometry, a2: ArcGeometry, angEps: number): Vec2[] {
  const dd = dist(a1.center, a2.center);
  const r1 = a1.radius;
  const r2 = a2.radius;
  const lenEps = angEps * Math.max(1, r1, r2);
  // Concentric arcs (dd ~ 0) share every point when radii match — callers
  // cannot use a unique intersection there. Equal radii at DISTINCT centers
  // are perfectly fine (two unit circles intersect twice).
  if (dd <= lenEps) return [];
  if (dd > r1 + r2 + lenEps) return [];
  if (dd < Math.abs(r1 - r2) - lenEps) return [];
  const x = (dd * dd + r1 * r1 - r2 * r2) / (2 * dd);
  const h = Math.sqrt(Math.max(0, r1 * r1 - x * x));
  const base: Vec2 = [
    a1.center[0] + (x / dd) * (a2.center[0] - a1.center[0]),
    a1.center[1] + (x / dd) * (a2.center[1] - a1.center[1]),
  ];
  const nx = -(a2.center[1] - a1.center[1]) / dd;
  const ny = (a2.center[0] - a1.center[0]) / dd;
  const raw: Vec2[] = h === 0
    ? [base]
    : [
        [base[0] + nx * h, base[1] + ny * h],
        [base[0] - nx * h, base[1] - ny * h],
      ];
  const ang1 = (p: Vec2): number => Math.atan2(p[1] - a1.center[1], p[0] - a1.center[0]);
  const ang2 = (p: Vec2): number => Math.atan2(p[1] - a2.center[1], p[0] - a2.center[0]);
  const out = raw.filter((p) => {
    if (Math.abs(dist(p, a1.center) - r1) > lenEps) return false;
    if (Math.abs(dist(p, a2.center) - r2) > lenEps) return false;
    return angleOnArc(a1, ang1(p), angEps) && angleOnArc(a2, ang2(p), angEps);
  });
  out.sort((p, q) => normalizeAngle(ang1(p) - ang1(q)) || p[0] - q[0] || p[1] - q[1]);
  const dedup: Vec2[] = [];
  for (const p of out) {
    if (!dedup.some((d0) => dist(d0, p) <= lenEps)) dedup.push(p);
  }
  return dedup;
}

/** Nearest point on the CLOSED arc to p (projection, or nearest endpoint). */
export function nearestOnArc(arc: ArcGeometry, p: Vec2): Vec2 {
  const ang = Math.atan2(p[1] - arc.center[1], p[0] - arc.center[0]);
  if (angleOnArc(arc, ang, 0)) {
    return [
      arc.center[0] + arc.radius * Math.cos(ang),
      arc.center[1] + arc.radius * Math.sin(ang),
    ];
  }
  const endA = pointOnArc(arc, 0);
  const endB = pointOnArc(arc, 1);
  return dist(p, endA) <= dist(p, endB) ? endA : endB;
}

// ---------------------------------------------------------------------------
// Polygon queries (chord polyline; arcs are sampled by the caller)
// ---------------------------------------------------------------------------

/** Signed area (shoelace). Positive = CCW. */
export function signedArea(loop: Vec2[]): number {
  let s = 0;
  for (let i = 0; i < loop.length; i++) {
    const p = loop[i];
    const q = loop[(i + 1) % loop.length];
    s += p[0] * q[1] - q[0] * p[1];
  }
  return 0.5 * s;
}

export function perimeter(loop: Vec2[], closed = true): number {
  let s = 0;
  const n = loop.length;
  const last = closed ? n : n - 1;
  for (let i = 0; i < last; i++) {
    s += dist(loop[i], loop[(i + 1) % n]);
  }
  return s;
}

export function bbox(points: Vec2[]): { min: Vec2; max: Vec2 } | null {
  if (points.length === 0) return null;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of points) {
    if (p[0] < minX) minX = p[0];
    if (p[1] < minY) minY = p[1];
    if (p[0] > maxX) maxX = p[0];
    if (p[1] > maxY) maxY = p[1];
  }
  return { min: [minX, minY], max: [maxX, maxY] };
}

/** Strict even-odd containment (boundary excluded — test that separately). */
export function pointInPolygon(p: Vec2, loop: Vec2[]): boolean {
  let inside = false;
  for (let i = 0, j = loop.length - 1; i < loop.length; j = i++) {
    const a = loop[i];
    const b = loop[j];
    if ((a[1] > p[1]) !== (b[1] > p[1])) {
      const x = a[0] + ((p[1] - a[1]) / (b[1] - a[1])) * (b[0] - a[0]);
      if (x > p[0]) inside = !inside;
    }
  }
  return inside;
}

/**
 * Self-intersections of a closed chord loop: every non-adjacent edge pair
 * that properly crosses or touches within eps. Adjacent pairs (sharing an
 * endpoint, including the wrap pair) are skipped. Deterministic: pairs
 * visited in (i, j) index order.
 */
export function selfIntersections(
  loop: Vec2[], eps: number,
): Array<{ i: number; j: number; pos: Vec2 }> {
  const n = loop.length;
  const hits: Array<{ i: number; j: number; pos: Vec2 }> = [];
  for (let i = 0; i < n; i++) {
    const a0 = loop[i];
    const a1 = loop[(i + 1) % n];
    for (let j = i + 2; j < n; j++) {
      if (i === 0 && j === n - 1) continue; // wrap-adjacent pair
      const b0 = loop[j];
      const b1 = loop[(j + 1) % n];
      const hit = segmentIntersection(a0, a1, b0, b1, eps);
      if (hit) hits.push({ i, j, pos: hit.pos });
    }
  }
  return hits;
}

// ---------------------------------------------------------------------------
// Transforms
// ---------------------------------------------------------------------------

export function rotateAround(p: Vec2, pivot: Vec2, angle: number): Vec2 {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const dx = p[0] - pivot[0];
  const dy = p[1] - pivot[1];
  return [pivot[0] + dx * c - dy * s, pivot[1] + dx * s + dy * c];
}

/** Reflect p across the infinite line through lineA-lineB. */
export function reflectAcrossLine(p: Vec2, lineA: Vec2, lineB: Vec2): Vec2 {
  const d = sub(lineB, lineA);
  const d2 = dot(d, d);
  if (d2 === 0) throw new Error("reflect: degenerate mirror line");
  const t = dot(sub(p, lineA), d) / d2;
  const proj: Vec2 = [lineA[0] + d[0] * t, lineA[1] + d[1] * t];
  return [2 * proj[0] - p[0], 2 * proj[1] - p[1]];
}
