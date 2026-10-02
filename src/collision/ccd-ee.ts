// Edge-edge CCD over x(t) = x0 + t*(x1-x0).
// Coplanarity scalar S(t) = dot(c-a, (b-a) x (d-c)) is cubic. Roots are
// candidates; a root is a TOI when segment parameters are in [0,1] and the
// distance is within thickness. Conservative dense-sample fallback included.

import { solveCubic01 } from "./cubic.js";
import { closestPointEdgeEdge } from "./closest-point.js";

function cross(o: number[], u: number[], v: number[]): void {
  o[0] = u[1] * v[2] - u[2] * v[1];
  o[1] = u[2] * v[0] - u[0] * v[2];
  o[2] = u[0] * v[1] - u[1] * v[0];
}

function dot(u: number[], v: number[]): number {
  return u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
}

function at(p0: number[], dp: number[], t: number): number[] {
  return [p0[0] + dp[0] * t, p0[1] + dp[1] * t, p0[2] + dp[2] * t];
}

/** Earliest TOI in [0,1], or Infinity when safe. */
export function eeCCD(
  x0: ArrayLike<number>, x1: ArrayLike<number>,
  a: number, b: number, c: number, d: number,
  thickness: number,
): number {
  const P = (v: number) => [x0[v * 3], x0[v * 3 + 1], x0[v * 3 + 2]];
  const D = (v: number) => [x1[v * 3] - x0[v * 3], x1[v * 3 + 1] - x0[v * 3 + 1], x1[v * 3 + 2] - x0[v * 3 + 2]];
  const a0 = P(a), da = D(a), b0 = P(b), db = D(b), c0 = P(c), dc = D(c), d0 = P(d), dd = D(d);

  const q = (t: number) =>
    closestPointEdgeEdge(
      ...at(a0, da, t) as [number, number, number],
      ...at(b0, db, t) as [number, number, number],
      ...at(c0, dc, t) as [number, number, number],
      ...at(d0, dd, t) as [number, number, number],
    );
  const q0 = q(0).dist;
  if (q0 <= thickness) return 0;
  // Relative-motion early-out (see ccd-vt.ts; translation cancels in r(t)).
  let mx = 0, my = 0, mz = 0;
  for (const v of [a, b, c, d]) {
    mx += x1[v * 3] - x0[v * 3];
    my += x1[v * 3 + 1] - x0[v * 3 + 1];
    mz += x1[v * 3 + 2] - x0[v * 3 + 2];
  }
  mx /= 4; my /= 4; mz /= 4;
  let dev = 0;
  for (const v of [a, b, c, d]) {
    const dx = x1[v * 3] - x0[v * 3] - mx;
    const dy = x1[v * 3 + 1] - x0[v * 3 + 1] - my;
    const dz = x1[v * 3 + 2] - x0[v * 3 + 2] - mz;
    const m = Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz));
    if (m > dev) dev = m;
  }
  if (q0 - 4 * dev > thickness) return Infinity;

  // e1(t) = b-a, e2(t) = d-c, r(t) = c-a ; S = r . (e1 x e2)
  const e10 = [b0[0] - a0[0], b0[1] - a0[1], b0[2] - a0[2]];
  const de1 = [db[0] - da[0], db[1] - da[1], db[2] - da[2]];
  const e20 = [d0[0] - c0[0], d0[1] - c0[1], d0[2] - c0[2]];
  const de2 = [dd[0] - dc[0], dd[1] - dc[1], dd[2] - dc[2]];
  const r0 = [c0[0] - a0[0], c0[1] - a0[1], c0[2] - a0[2]];
  const dr = [dc[0] - da[0], dc[1] - da[1], dc[2] - da[2]];

  const n0 = [0, 0, 0], n1 = [0, 0, 0], n2 = [0, 0, 0];
  const t1 = [0, 0, 0], t2 = [0, 0, 0];
  cross(n0, e10, e20);
  cross(t1, e10, de2); cross(t2, de1, e20);
  for (let i = 0; i < 3; i++) n1[i] = t1[i] + t2[i];
  cross(n2, de1, de2);

  const c0c = dot(r0, n0);
  const c1c = dot(r0, n1) + dot(dr, n0);
  const c2c = dot(r0, n2) + dot(dr, n1);
  const c3c = dot(dr, n2);
  const roots = solveCubic01(c3c, c2c, c1c, c0c);

  for (const t of roots) {
    const cp = closestPointEdgeEdge(
      ...at(a0, da, t) as [number, number, number],
      ...at(b0, db, t) as [number, number, number],
      ...at(c0, dc, t) as [number, number, number],
      ...at(d0, dd, t) as [number, number, number],
    );
    // Require genuinely close + interior (parallel overlaps still count if close).
    if (cp.dist <= thickness && cp.s >= -1e-6 && cp.s <= 1 + 1e-6 && cp.t >= -1e-6 && cp.t <= 1 + 1e-6) return t;
  }

  let minD = Infinity, minT = 0;
  const N = 16;
  for (let k = 0; k <= N; k++) {
    const t = k / N;
    const dd2 = q(t).dist;
    if (dd2 < minD) { minD = dd2; minT = t; }
  }
  if (minD <= thickness) return minT;
  return Infinity;
}
