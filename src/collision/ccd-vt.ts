// Vertex-triangle CCD over segment x(t) = x0 + t*(x1-x0), t in [0,1].
// Coplanarity volume V(t) = dot(p-a, (b-a) x (c-a)) is cubic; roots are
// candidate contact times. A root is a TOI when the projected point is inside
// the triangle (barycentric tolerance) and the pair is within thickness.

import { solveCubic01 } from "./cubic.js";
import { closestPointVertexTriangle } from "./closest-point.js";

function cross(o: number[], u: number[], v: number[]): void {
  o[0] = u[1] * v[2] - u[2] * v[1];
  o[1] = u[2] * v[0] - u[0] * v[2];
  o[2] = u[0] * v[1] - u[1] * v[0];
}

function dot(u: number[], v: number[]): number {
  return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

/** Earliest TOI in [0,1], or Infinity when safe. thickness = contact thickness. */
export function vtCCD(
  x0: ArrayLike<number>, x1: ArrayLike<number>,
  p: number, a: number, b: number, c: number,
  thickness: number,
): number {
  // Resting contact at t=0 counts as TOI 0.
  const d0 = closestPointVertexTriangle(
    x0[p * 3], x0[p * 3 + 1], x0[p * 3 + 2],
    x0[a * 3], x0[a * 3 + 1], x0[a * 3 + 2],
    x0[b * 3], x0[b * 3 + 1], x0[b * 3 + 2],
    x0[c * 3], x0[c * 3 + 1], x0[c * 3 + 2],
  ).dist;
  if (d0 <= thickness) return 0;
  // Relative-motion early-out: uniform translation cancels in r(t) = p(t)-q(t),
  // so only DEVIATION from the stencil mean motion can close the distance
  // (|Δr| <= 2*sqrt(3)*dev < 4*dev). Rigidly translating pairs always skip.
  let mx = 0, my = 0, mz = 0;
  for (const v of [p, a, b, c]) {
    mx += x1[v * 3] - x0[v * 3];
    my += x1[v * 3 + 1] - x0[v * 3 + 1];
    mz += x1[v * 3 + 2] - x0[v * 3 + 2];
  }
  mx /= 4; my /= 4; mz /= 4;
  let dev = 0;
  for (const v of [p, a, b, c]) {
    const dx = x1[v * 3] - x0[v * 3] - mx;
    const dy = x1[v * 3 + 1] - x0[v * 3 + 1] - my;
    const dz = x1[v * 3 + 2] - x0[v * 3 + 2] - mz;
    const m = Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz));
    if (m > dev) dev = m;
  }
  if (d0 - 4 * dev > thickness) return Infinity;

  // Edge vectors as linear polynomials: e(t) = e0 + t*de.
  const p0 = [x0[p * 3], x0[p * 3 + 1], x0[p * 3 + 2]];
  const dp = [x1[p * 3] - x0[p * 3], x1[p * 3 + 1] - x0[p * 3 + 1], x1[p * 3 + 2] - x0[p * 3 + 2]];
  const a0 = [x0[a * 3], x0[a * 3 + 1], x0[a * 3 + 2]];
  const da = [x1[a * 3] - x0[a * 3], x1[a * 3 + 1] - x0[a * 3 + 1], x1[a * 3 + 2] - x0[a * 3 + 2]];
  const b0 = [x0[b * 3], x0[b * 3 + 1], x0[b * 3 + 2]];
  const db = [x1[b * 3] - x0[b * 3], x1[b * 3 + 1] - x0[b * 3 + 1], x1[b * 3 + 2] - x0[b * 3 + 2]];
  const c0 = [x0[c * 3], x0[c * 3 + 1], x0[c * 3 + 2]];
  const dc = [x1[c * 3] - x0[c * 3], x1[c * 3 + 1] - x0[c * 3 + 1], x1[c * 3 + 2] - x0[c * 3 + 2]];

  // u(t) = p-a, v(t) = b-a, w(t) = c-a ; V = u . (v x w).
  // Expand with poly arithmetic per component (degree <= 1 each).
  const u0 = [p0[0] - a0[0], p0[1] - a0[1], p0[2] - a0[2]];
  const du = [dp[0] - da[0], dp[1] - da[1], dp[2] - da[2]];
  const v0 = [b0[0] - a0[0], b0[1] - a0[1], b0[2] - a0[2]];
  const dv = [db[0] - da[0], db[1] - da[1], db[2] - da[2]];
  const w0 = [c0[0] - a0[0], c0[1] - a0[1], c0[2] - a0[2]];
  const dw = [dc[0] - da[0], dc[1] - da[1], dc[2] - da[2]];

  // n(t) = v x w : n0 + t n1 + t^2 n2
  const n0 = [0, 0, 0], n1 = [0, 0, 0], n2 = [0, 0, 0];
  const t1 = [0, 0, 0], t2 = [0, 0, 0];
  cross(n0, v0, w0);
  cross(t1, v0, dw); cross(t2, dv, w0);
  for (let i = 0; i < 3; i++) n1[i] = t1[i] + t2[i];
  cross(n2, dv, dw);

  // V(t) = (u0 + t du) . (n0 + t n1 + t^2 n2)
  const c0c = dot(u0, n0);
  const c1c = dot(u0, n1) + dot(du, n0);
  const c2c = dot(u0, n2) + dot(du, n1);
  const c3c = dot(du, n2);
  const roots = solveCubic01(c3c, c2c, c1c, c0c);

  for (const t of roots) {
    // positions at t
    const pp = [p0[0] + dp[0] * t, p0[1] + dp[1] * t, p0[2] + dp[2] * t];
    const aa = [a0[0] + da[0] * t, a0[1] + da[1] * t, a0[2] + da[2] * t];
    const bb = [b0[0] + db[0] * t, b0[1] + db[1] * t, b0[2] + db[2] * t];
    const cc = [c0[0] + dc[0] * t, c0[1] + dc[1] * t, c0[2] + dc[2] * t];
    const cp = closestPointVertexTriangle(pp[0], pp[1], pp[2], aa[0], aa[1], aa[2], bb[0], bb[1], bb[2], cc[0], cc[1], cc[2]);
    // Conservative: coplanar within contact thickness counts as impact
    // (covers face, edge and vertex hits via the closest-point distance).
    if (cp.dist <= thickness) return t;
  }

  // Conservative fallback: dense-sampled minimum distance. If the swept
  // minimum is within thickness but no root qualified (grazing), report it
  // rather than silently claiming safe.
  let minD = Infinity, minT = 0;
  const N = 16;
  for (let k = 0; k <= N; k++) {
    const t = k / N;
    const pp = [p0[0] + dp[0] * t, p0[1] + dp[1] * t, p0[2] + dp[2] * t];
    const aa = [a0[0] + da[0] * t, a0[1] + da[1] * t, a0[2] + da[2] * t];
    const bb = [b0[0] + db[0] * t, b0[1] + db[1] * t, b0[2] + db[2] * t];
    const cc = [c0[0] + dc[0] * t, c0[1] + dc[1] * t, c0[2] + dc[2] * t];
    const d = closestPointVertexTriangle(pp[0], pp[1], pp[2], aa[0], aa[1], aa[2], bb[0], bb[1], bb[2], cc[0], cc[1], cc[2]).dist;
    if (d < minD) { minD = d; minT = t; }
  }
  if (minD <= thickness) return minT;
  return Infinity;
}
