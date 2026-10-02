// GpuContactSystem — G2 GPU CCD + active-contact compaction with an exact CPU
// mirror (FP32 via Math.fround, same formulas as the WGSL kernels).
//
// Pipeline (GPU-resident on device; mirrored here for headless CI):
//
//   G1 candidate triangle pairs (canonical, a < b)
//     -> closest-vt/ee.wgsl:   per-pair thread loops 6 VT + 9 EE primitives
//     -> ccd-vt/ee.wgsl:       resting check, deviation early-out, cubic
//                              coplanarity + root validation, 16-sample fallback
//     -> contact-compact.wgsl: distance classification vs dHat, atomic append,
//                              contactCount / contactOverflow / scannedCount
//     -> existing barrier-gradient.wgsl consumes the 64 B frozen records
//        (w / n+kind / id / prm) — G2 writes them layout-compatible.
//     -> existing barrier-hvp.wgsl consumes the same records (frozen HVP).
//
// Preserved exactly from Phase 1: VT/EE definitions, adjacency exclusion
// (upstream in G1 + edge-endpoint rule in expansion), dHat/dMin, cubic +
// conservative-advancement + root-selection CCD policy, barrier formulation
// with hard core, frozen closest-point projection, lagged Coulomb friction,
// Infinity-means-safe (never failure), explicit overflow (never truncation).
//
// Documented deviations (robustness, not physics):
//  D1. Degenerate (zero-area) triangles / zero-length edges: CPU Ericson can
//      yield 0/0 -> NaN. GPU returns a finite vertex-vertex fallback instead.
//      The CPU mesh builder rejects det < 1e-12, so real scenes never diverge.
//  D2. WGSL has no cbrt(): cubic uses sign(x)*pow(|x|,1/3) with a zero guard.
//      TOI tolerance absorbs the ulp-level difference (measured, not assumed).
//  D3. Static cloth-vs-mesh contact rides EXTENDED position buffers
//      (cloth ++ static, static motion zero, ids remapped at upload). The
//      device never branches on negative ids — same math as CPU extArrays.
//  D4. Floor records carry floorY in prm.z (barrier frozenDist subtracts it).
//      VT/EE prm layout is untouched.
//
// Readback invariant: build() inputs are already-resident buffers on device.
// Only counters + compact status are ever mapped (positionReadbacks stays 0).
// CCD is recomputed for EVERY Armijo trial: each alpha changes the segment,
// so each trial calls build() with a new x1 (CPU control, GPU compute).

import { expandTriPair } from "../../collision/self-collision.js";
import type { CandidatePairs } from "../../collision/broadphase.js";

const f = (v: number): number => Math.fround(v);

// ---------------------------------------------------------------------------
// G2 tolerances (FP32 vs f64 CPU reference; TOI calibrated by ccd-parity tests)
// ---------------------------------------------------------------------------

/** Closest-point distance: absolute tolerance in meters. */
export const G2_DIST_ABS_TOL = 1e-6;
/** Closest-point distance: relative tolerance. */
export const G2_DIST_REL_TOL = 1e-4;
/** TOI absolute tolerance (cubic roots in FP32; calibrated, see tests). */
export const G2_TOI_ABS_TOL = 1e-3;
/** Barrier energy relative tolerance (FP32 accumulation). */
export const G2_BARRIER_REL_TOL = 1e-3;
/** Friction force absolute tolerance (N, scene-scale aware in tests). */
export const G2_FRICTION_ABS_TOL = 1e-6;

// ---------------------------------------------------------------------------
// Contact kinds (match barrier-gradient.wgsl kind encoding)
// ---------------------------------------------------------------------------

export const CK_VT = 0;
export const CK_EE = 1;
export const CK_FLOOR = 2;

export type CcdStatus = "resting" | "impact" | "safe" | "failure";

// Exact copies of the (module-private) CPU key formats in contact-assembly.ts.
export function gpuVtKey(p: number, a: number, b: number, c: number): string {
  const t = [a, b, c].sort((x, y) => x - y).join("_");
  return `vt:${p}:${t}`;
}

export function gpuEeKey(a: number, b: number, c: number, d: number): string {
  const e1 = a < b ? `${a}_${b}` : `${b}_${a}`;
  const e2 = c < d ? `${c}_${d}` : `${d}_${c}`;
  return e1 < e2 ? `ee:${e1}:${e2}` : `ee:${e2}:${e1}`;
}

// ---------------------------------------------------------------------------
// FP32 closest point — Ericson 5.1.5 (VT) / 5.1.9 (EE), every op frounded
// ---------------------------------------------------------------------------

export interface ClosestResult {
  s: number; t: number;
  dist: number;
  rx: number; ry: number; rz: number;
  degenerate: boolean;
}

function hypot3(x: number, y: number, z: number): number {
  return Math.hypot(x, y, z);
}

/** FP32 port of closestPointVertexTriangle. Never NaN (see D1). */
export function closestVtFP32(
  px: number, py: number, pz: number,
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number,
  cx: number, cy: number, cz: number,
): ClosestResult {
  px = f(px); py = f(py); pz = f(pz);
  ax = f(ax); ay = f(ay); az = f(az);
  bx = f(bx); by = f(by); bz = f(bz);
  cx = f(cx); cy = f(cy); cz = f(cz);
  const abx = f(bx - ax), aby = f(by - ay), abz = f(bz - az);
  const acx = f(cx - ax), acy = f(cy - ay), acz = f(cz - az);
  // D1: degenerate triangle -> closest of the three vertex-vertex distances.
  const n2x = f(f(aby * acz) - f(abz * acy));
  const n2y = f(f(abz * acx) - f(abx * acz));
  const n2z = f(f(abx * acy) - f(aby * acx));
  const area2 = f(f(f(n2x * n2x) + f(n2y * n2y)) + f(n2z * n2z));
  const edgeScale = f(f(f(abx * abx) + f(aby * aby)) + f(abz * abz) + f(f(f(acx * acx) + f(acy * acy)) + f(acz * acz)));
  if (!(area2 > f(1e-30 * edgeScale))) {
    let best = { s: 0, t: 0, rx: f(px - ax), ry: f(py - ay), rz: f(pz - az) };
    let bd = hypot3(best.rx, best.ry, best.rz);
    const cand = [
      { s: 1, t: 0, rx: f(px - bx), ry: f(py - by), rz: f(pz - bz) },
      { s: 0, t: 1, rx: f(px - cx), ry: f(py - cy), rz: f(pz - cz) },
    ];
    for (const c of cand) {
      const d = hypot3(c.rx, c.ry, c.rz);
      if (d < bd) { bd = d; best = c; }
    }
    return { s: best.s, t: best.t, dist: bd, rx: best.rx, ry: best.ry, rz: best.rz, degenerate: true };
  }
  const apx = f(px - ax), apy = f(py - ay), apz = f(pz - az);
  const d1 = f(f(f(abx * apx) + f(aby * apy)) + f(abz * apz));
  const d2 = f(f(f(acx * apx) + f(acy * apy)) + f(acz * apz));
  if (d1 <= 0 && d2 <= 0) return packVt(0, 0, f(px - ax), f(py - ay), f(pz - az));
  const bpx = f(px - bx), bpy = f(py - by), bpz = f(pz - bz);
  const d3 = f(f(f(abx * bpx) + f(aby * bpy)) + f(abz * bpz));
  const d4 = f(f(f(acx * bpx) + f(acy * bpy)) + f(acz * bpz));
  if (d3 >= 0 && d4 <= d3) return packVt(1, 0, f(px - bx), f(py - by), f(pz - bz));
  const vc = f(f(d1 * d4) - f(d3 * d2));
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const den = f(d1 - d3); // = |ab|^2 > 0 here (non-degenerate)
    const v = den > 0 ? f(d1 / den) : 0;
    return packVt(v, 0, f(px - f(f(ax + f(v * abx)))), f(py - f(f(ay + f(v * aby)))), f(pz - f(f(az + f(v * abz)))));
  }
  const cpx = f(px - cx), cpy = f(py - cy), cpz = f(pz - cz);
  const d5 = f(f(f(abx * cpx) + f(aby * cpy)) + f(abz * cpz));
  const d6 = f(f(f(acx * cpx) + f(acy * cpy)) + f(acz * cpz));
  if (d6 >= 0 && d5 <= d6) return packVt(0, 1, f(px - cx), f(py - cy), f(pz - cz));
  const vb = f(f(d5 * d2) - f(d1 * d6));
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const den = f(d2 - d6); // = |ac|^2 > 0 here
    const w = den > 0 ? f(d2 / den) : 0;
    return packVt(0, w, f(px - f(f(ax + f(w * acx)))), f(py - f(f(ay + f(w * acy)))), f(pz - f(f(az + f(w * acz)))));
  }
  const va = f(f(d3 * d6) - f(d5 * d4));
  if (va <= 0 && f(d4 - d3) >= 0 && f(d5 - d6) >= 0) {
    const den = f(f(d4 - d3) + f(d5 - d6));
    const w = den > 0 ? f(f(d4 - d3) / den) : 0;
    const qx = f(f(bx + f(w * f(cx - bx)))), qy = f(f(by + f(w * f(cy - by)))), qz = f(f(bz + f(w * f(cz - bz))));
    return packVt(f(1 - w), w, f(px - qx), f(py - qy), f(pz - qz));
  }
  const denom = f(f(va + vb) + vc);
  if (!(denom > 0)) {
    // Paranoia guard (unreachable after the D1 check): vertex fallback.
    return packVt(0, 0, f(px - ax), f(py - ay), f(pz - az), true);
  }
  const v = f(f(vb / denom)), w = f(f(vc / denom));
  const qx = f(f(ax + f(abx * v)) + f(acx * w));
  const qy = f(f(ay + f(aby * v)) + f(acy * w));
  const qz = f(f(az + f(abz * v)) + f(acz * w));
  return packVt(v, w, f(px - qx), f(py - qy), f(pz - qz));
}

function packVt(s: number, t: number, rx: number, ry: number, rz: number, deg = false): ClosestResult {
  return { s, t, dist: hypot3(rx, ry, rz), rx, ry, rz, degenerate: deg };
}

function clampFP32(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/** FP32 port of closestPointEdgeEdge. Finite for all inputs (CPU guards ported). */
export function closestEeFP32(
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number,
  cx: number, cy: number, cz: number,
  dx: number, dy: number, dz: number,
): ClosestResult {
  ax = f(ax); ay = f(ay); az = f(az);
  bx = f(bx); by = f(by); bz = f(bz);
  cx = f(cx); cy = f(cy); cz = f(cz);
  dx = f(dx); dy = f(dy); dz = f(dz);
  const d1x = f(bx - ax), d1y = f(by - ay), d1z = f(bz - az);
  const d2x = f(dx - cx), d2y = f(dy - cy), d2z = f(dz - cz);
  const rx = f(ax - cx), ry = f(ay - cy), rz = f(az - cz);
  const a = f(f(f(d1x * d1x) + f(d1y * d1y)) + f(d1z * d1z));
  const e = f(f(f(d2x * d2x) + f(d2y * d2y)) + f(d2z * d2z));
  const ff = f(f(f(d2x * rx) + f(d2y * ry)) + f(d2z * rz));
  let s: number, t: number;
  let deg = false;
  if (a <= 1e-30 && e <= 1e-30) {
    s = 0; t = 0; deg = true;
  } else if (a <= 1e-30) {
    s = 0; t = clampFP32(f(ff / e), 0, 1); deg = true;
  } else {
    const c = f(f(f(d1x * rx) + f(d1y * ry)) + f(d1z * rz));
    if (e <= 1e-30) {
      t = 0; s = clampFP32(f(-c / a), 0, 1); deg = true;
    } else {
      const b = f(f(f(d1x * d2x) + f(d1y * d2y)) + f(d1z * d2z));
      const denom = f(f(a * e) - f(b * b));
      s = denom > 1e-30 ? clampFP32(f(f(f(b * ff) - f(c * e)) / denom), 0, 1) : 0;
      t = f(f(f(b * s) + ff) / e);
      if (t < 0) { t = 0; s = clampFP32(f(-c / a), 0, 1); }
      else if (t > 1) { t = 1; s = clampFP32(f(f(b - c) / a), 0, 1); }
    }
  }
  const axp = f(f(ax + f(d1x * s))), ayp = f(f(ay + f(d1y * s))), azp = f(f(az + f(d1z * s)));
  const cxp = f(f(cx + f(d2x * t))), cyp = f(f(cy + f(d2y * t))), czp = f(f(cz + f(d2z * t)));
  const rx2 = f(axp - cxp), ry2 = f(ayp - cyp), rz2 = f(azp - czp);
  return { s, t, dist: hypot3(rx2, ry2, rz2), rx: rx2, ry: ry2, rz: rz2, degenerate: deg };
}

// ---------------------------------------------------------------------------
// FP32 cubic — port of solveCubic01 (Cardano + quadratic fallback + polish)
// ---------------------------------------------------------------------------

/** Sorted roots in [0,1] of a3 t^3 + a2 t^2 + a1 t + a0 = 0 (FP32 mirror). */
export function solveCubic01FP32(a3: number, a2: number, a1: number, a0: number): number[] {
  const eps = 1e-12;
  if (Math.abs(a3) < eps) return solveQuadratic01FP32(a2, a1, a0);
  const A = f(a2 / a3), B = f(a1 / a3), C = f(a0 / a3);
  const sqA = f(A * A);
  const p = f(f(f(3 * B) - sqA) / 3);
  const q = f(f(f(f(f(2 * sqA * A) - f(f(9 * A * B))) + f(27 * C))) / 27);
  const roots: number[] = [];
  const disc = f(f(f(q * q) / 4) + f(f(f(p * p * p)) / 27));
  const shift = f(A / 3);
  if (disc > eps) {
    const sd = Math.sqrt(disc);
    // D2: WGSL has no cbrt(); both sides use sign(x)*|x|^(1/3) semantics.
    const u = fcbrt(f(f(-q / 2) + sd));
    const v = fcbrt(f(f(-q / 2) - sd));
    roots.push(f(f(u + v) - shift));
  } else if (Math.abs(disc) <= eps) {
    const u = fcbrt(f(-q / 2));
    roots.push(f(f(2 * u) - shift), f(f(-u) - shift));
  } else {
    const r = Math.sqrt(-(p * p * p) / 27);
    const arg = Math.min(1, Math.max(-1, -q / (2 * r)));
    const phi = Math.acos(arg);
    const s = f(2 * fcbrt(r));
    roots.push(
      f(f(s * Math.cos(phi / 3)) - shift),
      f(f(s * Math.cos((phi + 2 * Math.PI) / 3)) - shift),
      f(f(s * Math.cos((phi + 4 * Math.PI) / 3)) - shift),
    );
  }
  const out: number[] = [];
  for (let r of roots) {
    if (!isFinite(r)) continue;
    for (let k = 0; k < 4; k++) {
      const fv = f(f(f(f(f(f(a3 * r) + a2) * r) + a1) * r) + a0);
      const df = f(f(f(f(3 * a3 * r) + f(2 * a2)) * r) + a1);
      if (Math.abs(df) < 1e-18) break;
      r = f(r - f(fv / df));
    }
    if (r >= -1e-9 && r <= 1 + 1e-9) out.push(Math.min(1, Math.max(0, r)));
  }
  out.sort((x, y) => x - y);
  return out.filter((v, i) => i === 0 || v - out[i - 1] > 1e-9);
}

/** WGSL-compatible cube root: sign(x)*|x|^(1/3), exact 0 at 0 (see D2). */
export function fcbrt(x: number): number {
  if (x === 0) return 0;
  return f(Math.sign(x) * Math.pow(Math.abs(x), 1 / 3));
}

function solveQuadratic01FP32(a2: number, a1: number, a0: number): number[] {
  const eps = 1e-12;
  if (Math.abs(a2) < eps) {
    if (Math.abs(a1) < eps) return [];
    const r = -a0 / a1;
    return r >= 0 && r <= 1 ? [r] : [];
  }
  const disc = a1 * a1 - 4 * a2 * a0;
  if (disc < 0) return [];
  const sd = Math.sqrt(disc);
  const out: number[] = [];
  for (const r of [(-a1 - sd) / (2 * a2), (-a1 + sd) / (2 * a2)]) {
    if (r >= 0 && r <= 1) out.push(r);
  }
  return out.sort((x, y) => x - y);
}

// ---------------------------------------------------------------------------
// FP32 CCD — vtCCD / eeCCD policy ported exactly (early-outs, roots, fallback)
// ---------------------------------------------------------------------------

export interface CcdResult {
  toi: number; // in [0,1], or Infinity when safe
  status: CcdStatus;
}

type Vec3 = [number, number, number];

function sub3(a: Vec3, b: Vec3): Vec3 {
  return [f(a[0] - b[0]), f(a[1] - b[1]), f(a[2] - b[2])];
}

function cross3(u: Vec3, v: Vec3): Vec3 {
  return [
    f(f(u[1] * v[2]) - f(u[2] * v[1])),
    f(f(u[2] * v[0]) - f(u[0] * v[2])),
    f(f(u[0] * v[1]) - f(u[1] * v[0])),
  ];
}

function dot3(u: Vec3, v: Vec3): number {
  return f(f(f(u[0] * v[0]) + f(u[1] * v[1])) + f(u[2] * v[2]));
}

function at3(p0: Vec3, dp: Vec3, t: number): Vec3 {
  return [f(p0[0] + f(dp[0] * t)), f(p0[1] + f(dp[1] * t)), f(p0[2] + f(dp[2] * t))];
}

/** Deviation early-out shared by VT/EE: uniform translation cancels in r(t). */
function deviationEarlyOut(
  P0: (v: number) => Vec3, D: (v: number) => Vec3,
  ids: number[], d0: number, thickness: number,
): boolean {
  let mx = 0, my = 0, mz = 0;
  for (const v of ids) {
    const d = D(v);
    mx = f(mx + d[0]); my = f(my + d[1]); mz = f(mz + d[2]);
  }
  mx = f(mx / ids.length); my = f(my / ids.length); mz = f(mz / ids.length);
  void P0;
  let dev = 0;
  for (const v of ids) {
    const d = D(v);
    const m = Math.max(Math.abs(f(d[0] - mx)), Math.abs(f(d[1] - my)), Math.abs(f(d[2] - mz)));
    if (m > dev) dev = m;
  }
  return f(d0 - f(4 * dev)) > thickness;
}

/** FP32 port of vtCCD. Positions are read f32-rounded (device-held values). */
export function vtCCDFP32(
  X0: ArrayLike<number>, X1: ArrayLike<number>,
  p: number, a: number, b: number, c: number,
  thickness: number,
): CcdResult {
  const P = (X: ArrayLike<number>, v: number): Vec3 =>
    [f(X[v * 3]), f(X[v * 3 + 1]), f(X[v * 3 + 2])];
  const p0 = P(X0, p), a0 = P(X0, a), b0 = P(X0, b), c0 = P(X0, c);
  const P1 = (v: number): Vec3 => P(X1, v);
  const dp = sub3(P1(p), p0), da = sub3(P1(a), a0),
    db = sub3(P1(b), b0), dc = sub3(P1(c), c0);
  const cp0 = closestVtFP32(
    p0[0], p0[1], p0[2], a0[0], a0[1], a0[2], b0[0], b0[1], b0[2], c0[0], c0[1], c0[2]);
  const d0 = cp0.dist;
  if (!isFinite(d0)) return { toi: 0, status: "failure" };
  if (d0 <= thickness) return { toi: 0, status: "resting" };
  const D = (v: number): Vec3 =>
    v === p ? dp : v === a ? da : v === b ? db : dc;
  if (deviationEarlyOut(() => p0, D, [p, a, b, c], d0, thickness)) {
    return { toi: Infinity, status: "safe" };
  }
  const u0 = sub3(p0, a0), du = sub3(dp, da);
  const v0 = sub3(b0, a0), dv = sub3(db, da);
  const w0 = sub3(c0, a0), dw = sub3(dc, da);
  const n0 = cross3(v0, w0);
  const t1 = cross3(v0, dw), t2 = cross3(dv, w0);
  const n1: Vec3 = [f(t1[0] + t2[0]), f(t1[1] + t2[1]), f(t1[2] + t2[2])];
  const n2 = cross3(dv, dw);
  const c0c = dot3(u0, n0);
  const c1c = f(dot3(u0, n1) + dot3(du, n0));
  const c2c = f(dot3(u0, n2) + dot3(du, n1));
  const c3c = dot3(du, n2);
  if (![c0c, c1c, c2c, c3c].every(isFinite)) return { toi: 0, status: "failure" };
  const roots = solveCubic01FP32(c3c, c2c, c1c, c0c);
  for (const t of roots) {
    const pp = at3(p0, dp, t), aa = at3(a0, da, t),
      bb = at3(b0, db, t), cc = at3(c0, dc, t);
    const cp = closestVtFP32(
      pp[0], pp[1], pp[2], aa[0], aa[1], aa[2], bb[0], bb[1], bb[2], cc[0], cc[1], cc[2]);
    if (!isFinite(cp.dist)) return { toi: 0, status: "failure" };
    if (cp.dist <= thickness) return { toi: t, status: "impact" };
  }
  let minD = Infinity, minT = 0;
  const N = 16;
  for (let k = 0; k <= N; k++) {
    const t = k / N;
    const pp = at3(p0, dp, t), aa = at3(a0, da, t),
      bb = at3(b0, db, t), cc = at3(c0, dc, t);
    const d = closestVtFP32(
      pp[0], pp[1], pp[2], aa[0], aa[1], aa[2], bb[0], bb[1], bb[2], cc[0], cc[1], cc[2]).dist;
    if (!isFinite(d)) return { toi: 0, status: "failure" };
    if (d < minD) { minD = d; minT = t; }
  }
  if (minD <= thickness) return { toi: minT, status: "impact" };
  return { toi: Infinity, status: "safe" };
}

/** FP32 port of eeCCD. */
export function eeCCDFP32(
  X0: ArrayLike<number>, X1: ArrayLike<number>,
  a: number, b: number, c: number, d: number,
  thickness: number,
): CcdResult {
  const P = (X: ArrayLike<number>, v: number): Vec3 =>
    [f(X[v * 3]), f(X[v * 3 + 1]), f(X[v * 3 + 2])];
  const a0 = P(X0, a), b0 = P(X0, b), c0 = P(X0, c), d0 = P(X0, d);
  const P1 = (v: number): Vec3 => P(X1, v);
  const da = sub3(P1(a), a0), db = sub3(P1(b), b0),
    dc = sub3(P1(c), c0), dd = sub3(P1(d), d0);
  const q = (t: number): ClosestResult => {
    const A = at3(a0, da, t), B = at3(b0, db, t),
      C = at3(c0, dc, t), Dd = at3(d0, dd, t);
    return closestEeFP32(
      A[0], A[1], A[2], B[0], B[1], B[2], C[0], C[1], C[2], Dd[0], Dd[1], Dd[2]);
  };
  const q0 = q(0).dist;
  if (!isFinite(q0)) return { toi: 0, status: "failure" };
  if (q0 <= thickness) return { toi: 0, status: "resting" };
  const D = (v: number): Vec3 =>
    v === a ? da : v === b ? db : v === c ? dc : dd;
  if (deviationEarlyOut(() => a0, D, [a, b, c, d], q0, thickness)) {
    return { toi: Infinity, status: "safe" };
  }
  const e10 = sub3(b0, a0), de1 = sub3(db, da);
  const e20 = sub3(d0, c0), de2 = sub3(dd, dc);
  const r0 = sub3(c0, a0), dr = sub3(dc, da);
  const n0 = cross3(e10, e20);
  const t1 = cross3(e10, de2), t2 = cross3(de1, e20);
  const n1: Vec3 = [f(t1[0] + t2[0]), f(t1[1] + t2[1]), f(t1[2] + t2[2])];
  const n2 = cross3(de1, de2);
  const c0c = dot3(r0, n0);
  const c1c = f(dot3(r0, n1) + dot3(dr, n0));
  const c2c = f(dot3(r0, n2) + dot3(dr, n1));
  const c3c = dot3(dr, n2);
  if (![c0c, c1c, c2c, c3c].every(isFinite)) return { toi: 0, status: "failure" };
  const roots = solveCubic01FP32(c3c, c2c, c1c, c0c);
  for (const t of roots) {
    const cp = q(t);
    if (!isFinite(cp.dist)) return { toi: 0, status: "failure" };
    if (cp.dist <= thickness &&
      cp.s >= -1e-6 && cp.s <= 1 + 1e-6 && cp.t >= -1e-6 && cp.t <= 1 + 1e-6) {
      return { toi: t, status: "impact" };
    }
  }
  let minD = Infinity, minT = 0;
  const N = 16;
  for (let k = 0; k <= N; k++) {
    const dd2 = q(k / N).dist;
    if (!isFinite(dd2)) return { toi: 0, status: "failure" };
    if (dd2 < minD) { minD = dd2; minT = k / N; }
  }
  if (minD <= thickness) return { toi: minT, status: "impact" };
  return { toi: Infinity, status: "safe" };
}

// ---------------------------------------------------------------------------
// Contact records (64 B frozen layout, barrier-compatible) + compaction
// ---------------------------------------------------------------------------

export interface GpuContactRecord {
  kind: number; // 0 VT, 1 EE, 2 floor
  /** Vertex ids in EXTENDED space (static remapped >= n); floor: [p,-1,-1,-1]. */
  ids: [number, number, number, number];
  /** Frozen weights: VT (w0,w1,w2,0); EE (wa,wb,wc,wd); floor (0,0,0,0). */
  w: [number, number, number, number];
  n: [number, number, number];
  dist: number; // frozen distance at x1
  toi: number;  // CCD TOI over x0->x1 (Infinity = no crossing)
  status: CcdStatus;
  key: string; // CPU-identical key (static ids un-remapped for comparison)
}

export interface GpuContactDiagnostics {
  contactCount: number;
  contactOverflow: 0 | 1;
  scannedCount: number;
  vtCount: number;
  eeCount: number;
  floorCount: number;
  failures: number;
  minDist: number;
  minTOI: number;
}

export interface GpuContactSet {
  contacts: GpuContactRecord[]; // canonical order (sorted by key), deterministic
  diagnostics: GpuContactDiagnostics;
}

/** Ordering-independent contact-set comparison (the G2 parity gate). */
export function compareContactSets(
  cpuKeys: string[],
  gpu: GpuContactSet,
): { match: boolean; missing: string[]; extra: string[] } {
  const cpuSet = new Set(cpuKeys);
  const gpuSet = new Set(gpu.contacts.map((c) => c.key));
  const missing = cpuKeys.filter((k) => !gpuSet.has(k));
  const extra = gpu.contacts.map((c) => c.key).filter((k) => !cpuSet.has(k));
  return { match: missing.length === 0 && extra.length === 0, missing, extra };
}

/**
 * Multiset comparison: sorted key arrays must agree ELEMENT-WISE, including
 * duplicate multiplicity. The CPU active set repeats a key when one primitive
 * is reachable via several triangle pairs (barrier sums every entry), so the
 * multiset — not the set — is the exactly-preserving gate for barrier parity.
 */
export function compareContactMultisets(
  cpuKeys: string[],
  gpuKeys: string[],
): { match: boolean; missing: string[]; extra: string[] } {
  const a = [...cpuKeys].sort();
  const b = [...gpuKeys].sort();
  const missing: string[] = [];
  const extra: string[] = [];
  let i = 0, j = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) { i++; j++; }
    else if (a[i] < b[j]) { missing.push(a[i]); i++; }
    else { extra.push(b[j]); j++; }
  }
  while (i < a.length) { missing.push(a[i]); i++; }
  while (j < b.length) { extra.push(b[j]); j++; }
  return { match: missing.length === 0 && extra.length === 0, missing, extra };
}

/**
 * FP32 classification boundary band (meters): contacts with |dist - dHat| or
 * dist itself inside the band are excluded from exact-classification gates.
 * FP32 input rounding (~1e-9 at scene scale) legitimately flips classification
 * exactly at the threshold; neither side is wrong. Band membership is
 * reported, never silently dropped from diagnostics.
 */
export const G2_BOUNDARY_BAND = 1e-6;

export interface GpuContactConfig {
  indices: Uint32Array;
  triCount: number;
  dHatM: number;
  dMinM: number;
  kappaJ: number;
  frictionMu: number;
  frictionEpsM: number;
  contactCapacity?: number;
  floorY?: number | null;
  staticPos?: Float32Array | null;
  staticIdx?: Uint32Array | null;
}

/**
 * G2 contact builder: G1 pairs -> primitives -> closest/CCD -> compaction.
 * Static cloth-vs-mesh contact rides extended position buffers (D3); floor is
 * kind 2. CCD is recomputed on every build() call — one call per Armijo trial.
 */
export class GpuContactSystem {
  readonly indices: Uint32Array;
  readonly triCount: number;
  readonly dHat: number;
  readonly dMin: number;
  readonly kappa: number;
  readonly mu: number;
  readonly fricEps: number;
  readonly contactCapacity: number;
  floorY: number | null;
  staticPos: Float32Array | null;
  staticIdx: Uint32Array | null;
  /** Lagged friction state (key -> normal + multiplier), refreshed by commit(). */
  lagged = new Map<string, { nx: number; ny: number; nz: number; lambdaN: number }>();
  xStep: Float64Array = new Float64Array(0);
  /** Must stay 0: inputs are already-resident buffers on device. */
  positionReadbacks = 0;
  counterReads = 0;
  lastSet: GpuContactSet | null = null;

  constructor(config: GpuContactConfig) {
    this.indices = config.indices;
    this.triCount = config.triCount;
    this.dHat = config.dHatM;
    this.dMin = config.dMinM;
    this.kappa = config.kappaJ;
    this.mu = config.frictionMu;
    this.fricEps = config.frictionEpsM;
    this.contactCapacity = config.contactCapacity ?? 4096;
    this.floorY = config.floorY ?? null;
    this.staticPos = config.staticPos ?? null;
    this.staticIdx = config.staticIdx ?? null;
  }

  beginStep(x0: Float64Array): void {
    this.xStep = Float64Array.from(x0);
    this.lagged.clear();
    this.lastSet = null;
  }

  n(): number {
    return this.xStep.length / 3;
  }

  /** Extended buffers: cloth ++ static (static motion zero). */
  extArrays(x1: ArrayLike<number>): { X0: Float64Array; X1: Float64Array; n: number } {
    const n = this.n();
    if (!this.staticPos) {
      return { X0: Float64Array.from(this.xStep), X1: Float64Array.from(x1), n };
    }
    const ns = this.staticPos.length / 3;
    const X0 = new Float64Array((n + ns) * 3);
    const X1 = new Float64Array((n + ns) * 3);
    X0.set(this.xStep, 0);
    X0.set(this.staticPos, n * 3);
    X1.set(Float64Array.from(x1), 0);
    X1.set(this.staticPos, n * 3);
    return { X0, X1, n };
  }

  remap(v: number, n: number): number {
    return v >= 0 ? v : n + (-v - 2);
  }

  unremapKeyId(v: number, n: number): number {
    return v >= n ? -((v - n) + 2) : v;
  }

  /**
   * Full G2 build for one Newton segment xStep -> x1 with G1 candidate pairs.
   * Static pairs (cloth tri vs static tri) come from `staticPairs`.
   */
  async build(
    x1: ArrayLike<number>,
    pairs: CandidatePairs,
    staticPairs?: Array<[number, number]>,
  ): Promise<GpuContactSet> {
    void this.positionReadbacks;
    const { X0, X1, n } = this.extArrays(x1);
    const emitted: GpuContactRecord[] = [];
    let scanned = 0; // every evaluated primitive (matches contactScanned)
    let requested = 0; // active appends requested (drives overflow)
    let failures = 0;
    let minD = Infinity;
    let minTOI = Infinity;
    const consider = (rec: GpuContactRecord | null): void => {
      scanned++;
      if (!rec) return;
      requested++;
      if (rec.dist < minD) minD = rec.dist;
      if (rec.toi < minTOI) minTOI = rec.toi;
      if (rec.status === "failure") failures++;
      if (emitted.length < this.contactCapacity) emitted.push(rec);
    };
    // ---- cloth-cloth primitives (one pair-thread loops 6 VT + 9 EE) ----
    for (const { a: tA, b: tB } of pairs.pairs) {
      const A = [this.indices[tA * 3], this.indices[tA * 3 + 1], this.indices[tA * 3 + 2]];
      const B = [this.indices[tB * 3], this.indices[tB * 3 + 1], this.indices[tB * 3 + 2]];
      for (const p of A) {
        consider(this.classifyVt(X0, X1, n, p, B[0], B[1], B[2]));
      }
      for (const p of B) {
        consider(this.classifyVt(X0, X1, n, p, A[0], A[1], A[2]));
      }
      const TRI_EDGES = [[0, 1], [1, 2], [2, 0]] as const;
      for (const [ea0, ea1] of TRI_EDGES) {
        for (const [eb0, eb1] of TRI_EDGES) {
          const a = A[ea0], b = A[ea1], c = B[eb0], d = B[eb1];
          if (a === c || a === d || b === c || b === d) continue;
          consider(this.classifyEe(X0, X1, n, a, b, c, d));
        }
      }
    }
    // ---- cloth-static primitives (negative ids, zero static motion) ----
    if (this.staticPos && this.staticIdx && staticPairs) {
      for (const [ct, st] of staticPairs) {
        const A = [this.indices[ct * 3], this.indices[ct * 3 + 1], this.indices[ct * 3 + 2]];
        const sv: [number, number, number] = [
          -(this.staticIdx[st * 3] + 2),
          -(this.staticIdx[st * 3 + 1] + 2),
          -(this.staticIdx[st * 3 + 2] + 2),
        ];
        for (const p of A) {
          consider(this.classifyVt(X0, X1, n, p, sv[0], sv[1], sv[2]));
        }
        const be: Array<[number, number]> = [[sv[0], sv[1]], [sv[1], sv[2]], [sv[2], sv[0]]];
        const ae: Array<[number, number]> = [[A[0], A[1]], [A[1], A[2]], [A[2], A[0]]];
        for (const [a, b] of ae) {
          for (const [c, d] of be) {
            consider(this.classifyEe(X0, X1, n, a, b, c, d));
          }
        }
      }
    }
    // ---- floor primitives ----
    let floorCount = 0;
    if (this.floorY !== null) {
      const fy = this.floorY;
      const x1a = X1;
      for (let v = 0; v < n; v++) {
        const d = f(x1a[v * 3 + 1] - fy);
        if (d < minD) minD = d;
        if (d >= this.dHat) continue;
        const rec: GpuContactRecord = {
          kind: CK_FLOOR, ids: [v, -1, -1, -1], w: [0, 0, 0, 0],
          n: [0, 1, 0], dist: d, toi: d <= this.dMin ? 0 : Infinity,
          status: d <= this.dMin ? "resting" : "safe",
          key: `floor:${v}`,
        };
        if (rec.toi < minTOI) minTOI = rec.toi;
        scanned++;
        if (emitted.length < this.contactCapacity) { emitted.push(rec); floorCount++; }
      }
    }
    this.counterReads++;
    const overflow = requested > this.contactCapacity ? 1 : 0;
    // Canonical deterministic order. Duplicates are KEPT (multiset parity):
    // the CPU active set legitimately repeats a key when one primitive is
    // reachable via several triangle pairs, and the barrier sums every entry
    // (duplicate energy is the Phase 1 behavior). Pair-threads on device
    // cannot dedupe across work-items either, so emission-with-duplicates is
    // the exactly-preserving port. Array.sort is stable (ES2019+).
    emitted.sort((u, v) => (u.key < v.key ? -1 : u.key > v.key ? 1 : 0));
    const contacts = emitted;
    const set: GpuContactSet = {
      contacts,
      diagnostics: {
        contactCount: contacts.length,
        contactOverflow: overflow as 0 | 1,
        scannedCount: scanned,
        vtCount: contacts.filter((c) => c.kind === CK_VT).length,
        eeCount: contacts.filter((c) => c.kind === CK_EE).length,
        floorCount: contacts.filter((c) => c.kind === CK_FLOOR).length,
        failures,
        minDist: minD,
        minTOI: minTOI,
      },
    };
    void floorCount;
    this.lastSet = set;
    return set;
  }

  private posAt(X: ArrayLike<number>, v: number): Vec3 {
    return [f(X[v * 3]), f(X[v * 3 + 1]), f(X[v * 3 + 2])];
  }

  /** VT classify at x1 + CCD over the segment. Null when inactive. */
  classifyVt(
    X0: ArrayLike<number>, X1: ArrayLike<number>, n: number,
    p: number, a: number, b: number, c: number,
  ): GpuContactRecord | null {
    const rp = this.remap(p, n), ra = this.remap(a, n),
      rb = this.remap(b, n), rc = this.remap(c, n);
    const P = this.posAt(X1, rp);
    const A = this.posAt(X1, ra), B = this.posAt(X1, rb), C = this.posAt(X1, rc);
    const cp = closestVtFP32(
      P[0], P[1], P[2], A[0], A[1], A[2], B[0], B[1], B[2], C[0], C[1], C[2]);
    if (!isFinite(cp.dist)) {
      return {
        kind: CK_VT, ids: [rp, ra, rb, rc], w: [0, 0, 0, 0], n: [0, 1, 0],
        dist: Infinity, toi: 0, status: "failure", key: this.vtKeyOf(n, p, a, b, c),
      };
    }
    if (cp.dist >= this.dHat || cp.dist < 1e-12) return null;
    const nx = f(cp.rx / cp.dist), ny = f(cp.ry / cp.dist), nz = f(cp.rz / cp.dist);
    const w1 = f(cp.s), w2 = f(cp.t), w0 = f(f(1 - w1) - w2);
    const ccd = vtCCDFP32(X0, X1, rp, ra, rb, rc, this.dMin);
    return {
      kind: CK_VT, ids: [rp, ra, rb, rc], w: [w0, w1, w2, 0], n: [nx, ny, nz],
      dist: cp.dist, toi: ccd.toi, status: ccd.status,
      key: this.vtKeyOf(n, p, a, b, c),
    };
  }

  /** EE classify at x1 + CCD over the segment. Null when inactive. */
  classifyEe(
    X0: ArrayLike<number>, X1: ArrayLike<number>, n: number,
    a: number, b: number, c: number, d: number,
  ): GpuContactRecord | null {
    const ra = this.remap(a, n), rb = this.remap(b, n),
      rc = this.remap(c, n), rd = this.remap(d, n);
    const A = this.posAt(X1, ra), B = this.posAt(X1, rb),
      C = this.posAt(X1, rc), D = this.posAt(X1, rd);
    const cp = closestEeFP32(
      A[0], A[1], A[2], B[0], B[1], B[2], C[0], C[1], C[2], D[0], D[1], D[2]);
    if (!isFinite(cp.dist)) {
      return {
        kind: CK_EE, ids: [ra, rb, rc, rd], w: [0, 0, 0, 0], n: [0, 1, 0],
        dist: Infinity, toi: 0, status: "failure", key: this.eeKeyOf(n, a, b, c, d),
      };
    }
    if (cp.dist >= this.dHat || cp.dist < 1e-12) return null;
    const nx = f(cp.rx / cp.dist), ny = f(cp.ry / cp.dist), nz = f(cp.rz / cp.dist);
    const w0 = f(1 - cp.s), w1 = f(cp.s), w2 = f(f(1 - cp.t) * -1), w3 = f(-cp.t);
    const ccd = eeCCDFP32(X0, X1, ra, rb, rc, rd, this.dMin);
    return {
      kind: CK_EE, ids: [ra, rb, rc, rd], w: [w0, w1, w2, w3], n: [nx, ny, nz],
      dist: cp.dist, toi: ccd.toi, status: ccd.status,
      key: this.eeKeyOf(n, a, b, c, d),
    };
  }

  private vtKeyOf(n: number, p: number, a: number, b: number, c: number): string {
    return gpuVtKey(
      this.unremapKeyId(this.remap(p, n), n),
      this.unremapKeyId(this.remap(a, n), n),
      this.unremapKeyId(this.remap(b, n), n),
      this.unremapKeyId(this.remap(c, n), n),
    );
  }

  private eeKeyOf(n: number, a: number, b: number, c: number, d: number): string {
    return gpuEeKey(
      this.unremapKeyId(this.remap(a, n), n),
      this.unremapKeyId(this.remap(b, n), n),
      this.unremapKeyId(this.remap(c, n), n),
      this.unremapKeyId(this.remap(d, n), n),
    );
  }

  // ---- frozen barrier energy + gradient (FP32 mirror of barrier-gradient.wgsl)
  // ---- plus lagged Coulomb friction (FP32 mirror of friction.wgsl) ----

  private frozenDist(X: ArrayLike<number>, r: GpuContactRecord): number {
    if (r.kind === CK_FLOOR) return f(X[r.ids[0] * 3 + 1] - this.floorY!);
    const n = r.n;
    if (r.kind === CK_VT) {
      const [p, a, b, c] = r.ids;
      const [w0, w1, w2] = r.w;
      const qx = f(f(f(w0 * X[a * 3]) + f(w1 * X[b * 3])) + f(w2 * X[c * 3]));
      const qy = f(f(f(w0 * X[a * 3 + 1]) + f(w1 * X[b * 3 + 1])) + f(w2 * X[c * 3 + 1]));
      const qz = f(f(f(w0 * X[a * 3 + 2]) + f(w1 * X[b * 3 + 2])) + f(w2 * X[c * 3 + 2]));
      return f(f(f(n[0] * f(X[p * 3] - qx)) + f(n[1] * f(X[p * 3 + 1] - qy))) + f(n[2] * f(X[p * 3 + 2] - qz)));
    }
    const [a, b, c, d] = r.ids;
    const [wa, wb, wc, wd] = r.w;
    const rx = f(f(f(f(wa * X[a * 3]) + f(wb * X[b * 3])) + f(wc * X[c * 3])) + f(wd * X[d * 3]));
    const ry = f(f(f(f(wa * X[a * 3 + 1]) + f(wb * X[b * 3 + 1])) + f(wc * X[c * 3 + 1])) + f(wd * X[d * 3 + 1]));
    const rz = f(f(f(f(wa * X[a * 3 + 2]) + f(wb * X[b * 3 + 2])) + f(wc * X[c * 3 + 2])) + f(wd * X[d * 3 + 2]));
    return f(f(f(n[0] * rx) + f(n[1] * ry)) + f(n[2] * rz));
  }

  private static barrierValueFP32(d: number, dHat: number): number {
    if (d >= dHat) return 0;
    const dc = Math.max(d, 1e-12);
    const t = f(dc - dHat);
    return f(f(-(t * t)) * Math.log(dc / dHat));
  }

  private static barrierGradScalarFP32(d: number, dHat: number, kappa: number): number {
    if (d >= dHat) return 0;
    const dc = Math.max(d, 1e-12);
    const t = f(dc - dHat);
    const db = f(f(f(-2 * t) * Math.log(dc / dHat)) - f(f(t * t) / dc));
    return f(kappa * db);
  }

  /** Barrier energy + gradient over the compact set (w = 1 per contact). */
  barrierEnergyGrad(
    x: ArrayLike<number>, set: GpuContactSet,
  ): { energy: number; grad: Float32Array } {
    const n = this.n();
    const grad = new Float32Array(n * 3);
    let energy = 0;
    for (const r of set.contacts) {
      const d = Math.max(this.frozenDist(x, r), 1e-12);
      if (d >= this.dHat) continue;
      energy += f(this.kappa * GpuContactSystem.barrierValueFP32(d, this.dHat));
      const g = GpuContactSystem.barrierGradScalarFP32(d, this.dHat, this.kappa);
      this.addDistGrad(grad, n, r, g);
    }
    return { energy, grad };
  }

  private addDistGrad(grad: Float32Array, n: number, r: GpuContactRecord, g: number): void {
    const [nx, ny, nz] = r.n;
    if (r.kind === CK_FLOOR) {
      grad[r.ids[0] * 3 + 1] += g;
      return;
    }
    if (r.kind === CK_VT) {
      const [p, a, b, c] = r.ids;
      const ws = [r.w[0], r.w[1], r.w[2]];
      const vs = [a, b, c];
      grad[p * 3] = f(grad[p * 3] + f(g * nx));
      grad[p * 3 + 1] = f(grad[p * 3 + 1] + f(g * ny));
      grad[p * 3 + 2] = f(grad[p * 3 + 2] + f(g * nz));
      for (let k = 0; k < 3; k++) {
        const v = vs[k];
        if (v < 0 || v >= n) continue; // static verts carry no gradient
        grad[v * 3] = f(grad[v * 3] + f(-g * ws[k] * nx));
        grad[v * 3 + 1] = f(grad[v * 3 + 1] + f(-g * ws[k] * ny));
        grad[v * 3 + 2] = f(grad[v * 3 + 2] + f(-g * ws[k] * nz));
      }
      return;
    }
    const ws = [r.w[0], r.w[1], r.w[2], r.w[3]];
    for (let k = 0; k < 4; k++) {
      const v = r.ids[k];
      if (v < 0 || v >= n) continue;
      grad[v * 3] = f(grad[v * 3] + f(f(g * ws[k]) * nx));
      grad[v * 3 + 1] = f(grad[v * 3 + 1] + f(f(g * ws[k]) * ny));
      grad[v * 3 + 2] = f(grad[v * 3 + 2] + f(f(g * ws[k]) * nz));
    }
  }

  /** Relative slip of the frozen stencil from step start (FP32). */
  private relativeSlip(x: ArrayLike<number>, r: GpuContactRecord): Vec3 {
    const n = this.n();
    const D = (v: number): Vec3 => {
      if (v < 0 || v >= n) return [0, 0, 0];
      return [f(x[v * 3] - this.xStep[v * 3]), f(x[v * 3 + 1] - this.xStep[v * 3 + 1]), f(x[v * 3 + 2] - this.xStep[v * 3 + 2])];
    };
    if (r.kind === CK_FLOOR) return D(r.ids[0]);
    if (r.kind === CK_VT) {
      const [p, a, b, c] = r.ids;
      const dp = D(p), da = D(a), db = D(b), dc = D(c);
      const [w0, w1, w2] = r.w;
      return [
        f(dp[0] - f(f(f(w0 * da[0]) + f(w1 * db[0])) + f(w2 * dc[0]))),
        f(dp[1] - f(f(f(w0 * da[1]) + f(w1 * db[1])) + f(w2 * dc[1]))),
        f(dp[2] - f(f(f(w0 * da[2]) + f(w1 * db[2])) + f(w2 * dc[2]))),
      ];
    }
    const ds = [D(r.ids[0]), D(r.ids[1]), D(r.ids[2]), D(r.ids[3])];
    const [wa, wb, wc, wd] = r.w;
    const ws = [wa, wb, wc, wd];
    const out: Vec3 = [0, 0, 0];
    for (let k = 0; k < 3; k++) {
      out[k] = f(f(f(f(ws[0] * ds[0][k]) + f(ws[1] * ds[1][k])) + f(ws[2] * ds[2][k])) + f(ws[3] * ds[3][k]));
    }
    return out;
  }

  /** Lagged Coulomb friction forces over the compact set (residual only). */
  frictionForces(
    x: ArrayLike<number>, set: GpuContactSet,
  ): { grad: Float32Array; work: number } {
    const n = this.n();
    const grad = new Float32Array(n * 3);
    let work = 0;
    for (const r of set.contacts) {
      const lag = this.lagged.get(r.key);
      const nx = lag?.nx ?? r.n[0], ny = lag?.ny ?? r.n[1], nz = lag?.nz ?? r.n[2];
      const lambdaN = lag?.lambdaN ?? Math.max(0, -GpuContactSystem.barrierGradScalarFP32(
        Math.max(r.dist, 1e-12), this.dHat, this.kappa));
      const u = this.relativeSlip(x, r);
      const un = f(f(f(u[0] * nx) + f(u[1] * ny)) + f(u[2] * nz));
      const tx = f(u[0] - f(un * nx)), ty = f(u[1] - f(un * ny)), tz = f(u[2] - f(un * nz));
      const mag = Math.sqrt(f(f(f(tx * tx) + f(ty * ty)) + f(f(tz * tz) + f(this.fricEps * this.fricEps))));
      const s = f(f(this.mu * Math.max(0, lambdaN)) / mag);
      const fx = f(-s * tx), fy = f(-s * ty), fz = f(-s * tz);
      work = f(work + f(f(f(fx * u[0]) + f(fy * u[1])) + f(fz * u[2])));
      this.addFrictionForce(grad, n, r, fx, fy, fz);
    }
    return { grad, work };
  }

  private addFrictionForce(
    grad: Float32Array, n: number, r: GpuContactRecord,
    fx: number, fy: number, fz: number,
  ): void {
    // NOTE: gradient accumulates the NEGATIVE physical force (residual side),
    // exactly like CPU addFrictionForce(grad, c, -fx, -fy, -fz).
    const gx = -fx, gy = -fy, gz = -fz;
    if (r.kind === CK_FLOOR) {
      const v = r.ids[0];
      grad[v * 3] += gx; grad[v * 3 + 1] += gy; grad[v * 3 + 2] += gz;
      return;
    }
    if (r.kind === CK_VT) {
      const [p, a, b, c] = r.ids;
      grad[p * 3] += gx; grad[p * 3 + 1] += gy; grad[p * 3 + 2] += gz;
      const ws = [r.w[0], r.w[1], r.w[2]];
      for (const [k, v] of [a, b, c].map((vv, kk) => [kk, vv] as [number, number])) {
        if (v < 0 || v >= n) continue;
        grad[v * 3] += -gx * ws[k];
        grad[v * 3 + 1] += -gy * ws[k];
        grad[v * 3 + 2] += -gz * ws[k];
      }
      return;
    }
    // EE: +f on edge ab (s-weighted via w0,w1), -f on edge cd (via w2,w3).
    // CPU pairs: [[a,w0],[b,w1],[c,-w2],[d,-w3]] with grad += f*w.
    const entries: Array<[number, number]> = [
      [r.ids[0], r.w[0]], [r.ids[1], r.w[1]],
      [r.ids[2], -r.w[2]], [r.ids[3], -r.w[3]],
    ];
    for (const [v, w] of entries) {
      if (v < 0 || v >= n) continue;
      grad[v * 3] += gx * w; grad[v * 3 + 1] += gy * w; grad[v * 3 + 2] += gz * w;
    }
  }

  /** Refresh lagged normals/multipliers from an accepted iterate (CPU commit). */
  commit(x: ArrayLike<number>, set: GpuContactSet): void {
    const { X1 } = this.extArrays(x);
    for (const r of set.contacts) {
      const d = Math.max(this.frozenDist(X1, r), 1e-12);
      if (d >= this.dHat) continue;
      const lambdaN = Math.max(0, -GpuContactSystem.barrierGradScalarFP32(d, this.dHat, this.kappa));
      this.lagged.set(r.key, { nx: r.n[0], ny: r.n[1], nz: r.n[2], lambdaN });
    }
  }

  private static b2Coeff(d: number, dHat: number, kappa: number): number {
    const L = Math.log(d / dHat);
    const u = f(d - dHat);
    const b2 = -(2 * L + (4 * u) / d - (u * u) / (d * d));
    return kappa * Math.max(b2, 0);
  }

  private distJvp(X: ArrayLike<number>, r: GpuContactRecord, v: ArrayLike<number>): number {
    const n = this.n();
    void X;
    const dotN = (vv: number): number => {
      if (vv < 0 || vv >= n) return 0;
      return f(f(f(r.n[0] * v[vv * 3]) + f(r.n[1] * v[vv * 3 + 1])) + f(r.n[2] * v[vv * 3 + 2]));
    };
    if (r.kind === CK_FLOOR) {
      const p = r.ids[0];
      return f(v[p * 3 + 1]);
    }
    if (r.kind === CK_VT) {
      let j = dotN(r.ids[0]);
      const ws = [r.w[0], r.w[1], r.w[2]];
      for (let k = 0; k < 3; k++) j = f(j - f(ws[k] * dotN([r.ids[1], r.ids[2], r.ids[3]][k])));
      return j;
    }
    let j = 0;
    const ws = [r.w[0], r.w[1], r.w[2], r.w[3]];
    for (let k = 0; k < 4; k++) j = f(j + f(ws[k] * dotN(r.ids[k])));
    return j;
  }

  /**
   * Frozen barrier HVP over the compact set (FP32 port of CPU applyHvp):
   * out += kappa*max(b''(d),0) * (J v) * J^T. Friction contributes nothing.
   */
  applyHvp(
    x: ArrayLike<number>, v: ArrayLike<number>,
    set: GpuContactSet, out: Float32Array,
  ): void {
    const n = this.n();
    const { X1 } = this.extArrays(x);
    out.fill(0);
    const dd = new Float32Array(set.contacts.length);
    for (let i = 0; i < set.contacts.length; i++) {
      const r = set.contacts[i];
      const d = Math.max(this.frozenDist(X1, r), 1e-12);
      if (d >= this.dHat) { dd[i] = 0; continue; }
      dd[i] = f(GpuContactSystem.b2Coeff(d, this.dHat, this.kappa) * this.distJvp(X1, r, v));
    }
    for (let i = 0; i < set.contacts.length; i++) {
      if (dd[i] === 0) continue;
      this.addDistGrad(out, n, set.contacts[i], dd[i]);
    }
  }

  /**
   * Frozen-barrier Jacobi diagonal add-on (FP32 port of CPU addDiagEstimate).
   * `diag` accumulates in place over the cloth DOFs (extended/static skipped).
   */
  addDiagEstimate(
    x: ArrayLike<number>, set: GpuContactSet, diag: Float32Array,
  ): void {
    const n = this.n();
    const { X1 } = this.extArrays(x);
    for (const r of set.contacts) {
      const d = Math.max(this.frozenDist(X1, r), 1e-12);
      if (d >= this.dHat) continue;
      const L = Math.log(d / this.dHat);
      const u = f(d - this.dHat);
      const b2 = Math.max(-(2 * L + (4 * u) / d - (u * u) / (d * d)), 0);
      const k = f(this.kappa * b2);
      if (k === 0) continue;
      if (r.kind === CK_FLOOR) {
        diag[r.ids[0] * 3 + 1] += k;
        continue;
      }
      const n2 = [f(r.n[0] * r.n[0]), f(r.n[1] * r.n[1]), f(r.n[2] * r.n[2])];
      if (r.kind === CK_VT) {
        for (let kk = 0; kk < 3; kk++) diag[r.ids[0] * 3 + kk] += f(k * n2[kk]);
        const ws = [r.w[0], r.w[1], r.w[2]];
        const vs = [r.ids[1], r.ids[2], r.ids[3]];
        for (let j = 0; j < 3; j++) {
          if (vs[j] < 0 || vs[j] >= n) continue;
          for (let kk = 0; kk < 3; kk++) {
            diag[vs[j] * 3 + kk] += f(f(k * f(ws[j] * ws[j])) * n2[kk]);
          }
        }
        continue;
      }
      const ws = [r.w[0], r.w[1], r.w[2], r.w[3]];
      for (let j = 0; j < 4; j++) {
        const vv = r.ids[j];
        if (vv < 0 || vv >= n) continue;
        for (let kk = 0; kk < 3; kk++) {
          diag[vv * 3 + kk] += f(f(k * f(ws[j] * ws[j])) * n2[kk]);
        }
      }
    }
  }

  /**
   * Trial validity over the segment xStep -> xT (CPU checkTrial semantics):
   * exact distance <= dMin or a fresh crossing (0 < toi < 1-1e-9) or any
   * numerical failure rejects the trial. Infinity is safe, never failure.
   */
  checkTrialValidity(
    xT: ArrayLike<number>, pairs: CandidatePairs,
    staticPairs?: Array<[number, number]>,
  ): { valid: boolean; minDist: number; minTOI: number } {
    const { X0, X1, n } = this.extArrays(xT);
    let minDist = Infinity;
    let minTOI = Infinity;
    // Exact distances at the trial over all primitives in the pair set.
    const dists: number[] = [];
    const prims = this.enumeratePrimitives(pairs, staticPairs);
    for (const q of prims) {
      const d = this.exactPrimitiveDist(X1, n, q);
      if (!isFinite(d)) return { valid: false, minDist, minTOI };
      dists.push(d);
      if (d < minDist) minDist = d;
    }
    if (this.floorY !== null) {
      for (let v = 0; v < n; v++) {
        const d = f(X1[v * 3 + 1] - this.floorY);
        if (d < minDist) minDist = d;
      }
    }
    if (minDist <= this.dMin) return { valid: false, minDist, minTOI };
    for (const q of prims) {
      const toi = this.primitiveTOI(X0, X1, n, q);
      if (!isFinite(toi) && toi !== Infinity) return { valid: false, minDist, minTOI };
      if (Number.isNaN(toi)) return { valid: false, minDist, minTOI };
      if (toi < minTOI) minTOI = toi;
      if (toi > 0 && toi < 1 - 1e-9) return { valid: false, minDist, minTOI };
    }
    if (this.floorY !== null) {
      for (let v = 0; v < n; v++) {
        const y0 = X0[v * 3 + 1] - this.floorY;
        const yT = X1[v * 3 + 1] - this.floorY;
        if (y0 - Math.abs(yT - y0) > this.dMin) continue;
        if (y0 > this.dMin && yT <= this.dMin && y0 > yT) {
          const toi = (y0 - this.dMin) / (y0 - yT);
          if (toi < minTOI) minTOI = toi;
          if (toi < 1 - 1e-9) return { valid: false, minDist, minTOI };
        }
      }
    }
    return { valid: true, minDist, minTOI };
  }

  enumeratePrimitives(
    pairs: CandidatePairs, staticPairs?: Array<[number, number]>,
  ): Array<{ kind: 0; p: number; a: number; b: number; c: number } | { kind: 1; a: number; b: number; c: number; d: number }> {
    const out: Array<{ kind: 0; p: number; a: number; b: number; c: number } | { kind: 1; a: number; b: number; c: number; d: number }> = [];
    for (const { a: tA, b: tB } of pairs.pairs) {
      const A = [this.indices[tA * 3], this.indices[tA * 3 + 1], this.indices[tA * 3 + 2]];
      const B = [this.indices[tB * 3], this.indices[tB * 3 + 1], this.indices[tB * 3 + 2]];
      for (const p of A) out.push({ kind: 0, p, a: B[0], b: B[1], c: B[2] });
      for (const p of B) out.push({ kind: 0, p, a: A[0], b: A[1], c: A[2] });
      const TRI_EDGES = [[0, 1], [1, 2], [2, 0]] as const;
      for (const [ea0, ea1] of TRI_EDGES) {
        for (const [eb0, eb1] of TRI_EDGES) {
          const a = A[ea0], b = A[ea1], c = B[eb0], d = B[eb1];
          if (a === c || a === d || b === c || b === d) continue;
          out.push({ kind: 1, a, b, c, d });
        }
      }
    }
    if (this.staticPos && this.staticIdx && staticPairs) {
      for (const [ct, st] of staticPairs) {
        const A = [this.indices[ct * 3], this.indices[ct * 3 + 1], this.indices[ct * 3 + 2]];
        const sv: [number, number, number] = [
          -(this.staticIdx[st * 3] + 2),
          -(this.staticIdx[st * 3 + 1] + 2),
          -(this.staticIdx[st * 3 + 2] + 2),
        ];
        for (const p of A) out.push({ kind: 0, p, a: sv[0], b: sv[1], c: sv[2] });
        const be: Array<[number, number]> = [[sv[0], sv[1]], [sv[1], sv[2]], [sv[2], sv[0]]];
        const ae: Array<[number, number]> = [[A[0], A[1]], [A[1], A[2]], [A[2], A[0]]];
        for (const [a, b] of ae) {
          for (const [c, d] of be) out.push({ kind: 1, a, b, c, d });
        }
      }
    }
    return out;
  }

  exactPrimitiveDist(
    X: ArrayLike<number>, n: number,
    q: { kind: 0; p: number; a: number; b: number; c: number } | { kind: 1; a: number; b: number; c: number; d: number },
  ): number {
    const at = (v: number): Vec3 => {
      const r = this.remap(v, n);
      return this.posAt(X, r);
    };
    if (q.kind === 0) {
      const P = at(q.p), A = at(q.a), B = at(q.b), C = at(q.c);
      return closestVtFP32(
        P[0], P[1], P[2], A[0], A[1], A[2], B[0], B[1], B[2], C[0], C[1], C[2]).dist;
    }
    const A = at(q.a), B = at(q.b), C = at(q.c), D = at(q.d);
    return closestEeFP32(
      A[0], A[1], A[2], B[0], B[1], B[2], C[0], C[1], C[2], D[0], D[1], D[2]).dist;
  }

  primitiveTOI(
    X0: ArrayLike<number>, X1: ArrayLike<number>, n: number,
    q: { kind: 0; p: number; a: number; b: number; c: number } | { kind: 1; a: number; b: number; c: number; d: number },
  ): number {
    if (q.kind === 0) {
      return vtCCDFP32(
        X0, X1, this.remap(q.p, n), this.remap(q.a, n),
        this.remap(q.b, n), this.remap(q.c, n), this.dMin).toi;
    }
    return eeCCDFP32(
      X0, X1, this.remap(q.a, n), this.remap(q.b, n),
      this.remap(q.c, n), this.remap(q.d, n), this.dMin).toi;
  }

  /**
   * Encode one record into the 64 B frozen layout consumed by
   * barrier-gradient.wgsl (w / n+kind / id / prm). Static ids are already
   * remapped to extended space (D3); floor carries floorY in prm.z (D4).
   */
  encodeRecord64(r: GpuContactRecord): Float32Array {
    const out = new Float32Array(16);
    out[0] = r.w[0]; out[1] = r.w[1]; out[2] = r.w[2]; out[3] = r.w[3];
    out[4] = r.n[0]; out[5] = r.n[1]; out[6] = r.n[2]; out[7] = r.kind;
    const u = new Uint32Array(out.buffer);
    u[8] = r.ids[0] >>> 0; u[9] = (r.ids[1] < 0 ? 0xffffffff : r.ids[1]) >>> 0;
    u[10] = (r.ids[2] < 0 ? 0xffffffff : r.ids[2]) >>> 0;
    u[11] = (r.ids[3] < 0 ? 0xffffffff : r.ids[3]) >>> 0;
    out[12] = this.dHat; out[13] = this.kappa;
    out[14] = r.kind === CK_FLOOR ? (this.floorY ?? 0) : this.mu;
    out[15] = this.fricEps;
    return out;
  }
}
