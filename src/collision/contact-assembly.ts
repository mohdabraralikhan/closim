// ContactSystem: the single boundary between solver and collision algorithms.
// Frozen-projection reference model: per Newton iteration the active set and
// closest-point parameters (s,t,n) are LAGGED (spec section 6); energy,
// gradient and HVP are then mutually consistent analytic functions of x with
// a linear distance d(x) = n . r(x). The barrier b(d) is convex on (0,dHat),
// so frozen contact Hessians are SPD and PCG-friendly. Friction is lagged
// Coulomb (spec section 7): normal info from the last commit, displacement
// from step start; friction enters the residual only (no Hessian).

import { TriBvh, buildAdjacencyExclusions, triPairKey } from "./bvh.js";
import { overlaps, type Aabb } from "./aabb.js";
import { closestPointVertexTriangle, closestPointEdgeEdge } from "./closest-point.js";
import { barrierValue, barrierGradScalar, barrierNormalForce } from "./barrier.js";
import { coulombForce } from "./friction.js";
import { expandTriPair } from "./self-collision.js";
import { vtCCD } from "./ccd-vt.js";
import { eeCCD } from "./ccd-ee.js";
import {
  type ContactParams, type ContactDiagnostics, emptyDiagnostics,
} from "./types.js";

interface Frozen {
  kind: number; // 0 = VT, 1 = EE, 2 = floor
  // vertex ids (floor: p = vertex, rest -1)
  p: number; a: number; b: number; c: number; d: number;
  // frozen weights: VT q = w0*a + w1*b + w2*c ; EE r = wa*a + wb*b + wc*c + wd*d
  w0: number; w1: number; w2: number; w3: number;
  nx: number; ny: number; nz: number;
  key: string;
}

interface Lagged { nx: number; ny: number; nz: number; lambdaN: number }

interface Cand { kind: number; p: number; a: number; b: number; c: number; d: number }

const K_VT = 0, K_EE = 1;

/** Read-only view of a frozen active contact for velocity filtering. */
export interface ActiveContactView {
  kind: number; // 0 VT, 1 EE, 2 floor
  p: number; a: number; b: number; c: number; d: number;
  w0: number; w1: number; w2: number;
  nx: number; ny: number; nz: number;
}

export class ContactSystem {
  params: ContactParams;
  clothIdx: Uint32Array;
  triCount: number;
  exclusions: Set<number>;
  floorY: number | null = null;
  staticPos: Float32Array | null = null;
  staticIdx: Uint32Array | null = null;

  xStep: Float64Array = new Float64Array(0);
  lagged = new Map<string, Lagged>();
  active: Frozen[] = [];
  candidates: Cand[] = [];
  diag: ContactDiagnostics = emptyDiagnostics();

  private bvh = new TriBvh();

  constructor(params: ContactParams, clothIndices: Uint32Array) {
    this.params = { ...params };
    this.clothIdx = clothIndices;
    this.triCount = clothIndices.length / 3;
    this.exclusions = buildAdjacencyExclusions(clothIndices, this.triCount);
  }

  setFloor(y: number | null): void { this.floorY = y; }
  setStaticMesh(pos: Float32Array, idx: Uint32Array): void {
    this.staticPos = pos; this.staticIdx = idx;
  }

  beginStep(x0: Float64Array): void {
    this.xStep = Float64Array.from(x0);
    this.lagged.clear();
    this.active = [];
    this.candidates = [];
    this.diag = emptyDiagnostics();
  }

  addRejection(): void { this.diag.rejectedLineSearchSteps++; }

  // ---- candidate generation (BVH over swept xStep -> x) ----

  private buildCandidates(x: Float64Array, pad: number): Cand[] {
    this.bvh.build(this.xStep, x, this.clothIdx, this.triCount, pad);
    const pairs = this.bvh.selfPairs(this.exclusions);
    const out: Cand[] = [];
    for (const [tA, tB] of pairs) {
      const { vt, ee } = expandTriPair(tA, tB, this.clothIdx);
      for (const q of vt) out.push({ kind: K_VT, p: q.p, a: q.a, b: q.b, c: q.c, d: -1 });
      for (const q of ee) out.push({ kind: K_EE, p: -1, a: q.a, b: q.b, c: q.c, d: q.d });
    }
    // cloth vs static triangles
    if (this.staticPos && this.staticIdx) {
      const sTri = this.staticIdx.length / 3;
      const sBoxes: Aabb[] = new Array(sTri);
      for (let t = 0; t < sTri; t++) {
        const i0 = this.staticIdx[t * 3], i1 = this.staticIdx[t * 3 + 1], i2 = this.staticIdx[t * 3 + 2];
        sBoxes[t] = staticTriAabb(this.staticPos, i0, i1, i2, pad);
      }
      const clothTris = this.queryBvhAll(sBoxes);
      for (const [ct, st] of clothTris) {
        const A = [this.clothIdx[ct * 3], this.clothIdx[ct * 3 + 1], this.clothIdx[ct * 3 + 2]];
        const B = [this.staticIdx[st * 3], this.staticIdx[st * 3 + 1], this.staticIdx[st * 3 + 2]];
        for (const p of A) out.push({ kind: K_VT, p, a: -(B[0] + 2), b: -(B[1] + 2), c: -(B[2] + 2), d: -2 });
        // static edges vs cloth edges (cloth-cloth edge enumeration reused on A only vs B edges)
        const be: Array<[number, number]> = [[B[0], B[1]], [B[1], B[2]], [B[2], B[0]]];
        const ae: Array<[number, number]> = [[A[0], A[1]], [A[1], A[2]], [A[2], A[0]]];
        for (const [a, b] of ae) for (const [c, d] of be) {
          out.push({ kind: K_EE, p: -1, a, b, c: -(c + 2), d: -(d + 2) });
        }
      }
    }
    // deterministic order
    out.sort((u, v) => u.kind - v.kind || u.p - v.p || u.a - v.a || u.b - v.b || u.c - v.c || u.d - v.d);
    this.diag.candidatePairs = out.length;
    return out;
  }

  /** Cloth-tri ids overlapping each static box: returns [clothTri, staticTri]. */
  private queryBvhAll(sBoxes: Aabb[]): Array<[number, number]> {
    const out: Array<[number, number]> = [];
    const nodes = this.bvh.nodes;
    if (this.bvh.root < 0) return out;
    for (let s = 0; s < sBoxes.length; s++) {
      const stack: number[] = [this.bvh.root];
      while (stack.length > 0) {
        const n = stack.pop()!;
        const nd = nodes[n];
        if (!overlaps(nd.aabb, sBoxes[s])) continue;
        if (nd.tri >= 0) { out.push([nd.tri, s]); continue; }
        stack.push(nd.left, nd.right);
      }
    }
    out.sort((u, v) => u[0] - v[0] || u[1] - v[1]);
    return out;
  }

  // ---- closest-point helpers (negative ids = static verts, offset -2) ----

  private clothX(x: Float64Array, v: number): [number, number, number] {
    return [x[v * 3], x[v * 3 + 1], x[v * 3 + 2]];
  }

  private posOf(x: Float64Array, v: number): [number, number, number] {
    if (v >= 0) return this.clothX(x, v);
    const s = this.staticPos!;
    const i = -v - 2;
    return [s[i * 3], s[i * 3 + 1], s[i * 3 + 2]];
  }

  private stepPosOf(v: number): [number, number, number] {
    if (v >= 0) return [this.xStep[v * 3], this.xStep[v * 3 + 1], this.xStep[v * 3 + 2]];
    const s = this.staticPos!;
    const i = -v - 2;
    return [s[i * 3], s[i * 3 + 1], s[i * 3 + 2]];
  }

  // ---- active set ----

  updateActiveSet(x: Float64Array): void {
    const dHat = this.params.dHatM;
    this.candidates = this.buildCandidates(x, dHat);
    const active: Frozen[] = [];
    let minD = Infinity;
    for (const q of this.candidates) {
      if (q.kind === K_VT) {
        const [px, py, pz] = this.posOf(x, q.p);
        const [ax, ay, az] = this.posOf(x, q.a);
        const [bx, by, bz] = this.posOf(x, q.b);
        const [cx, cy, cz] = this.posOf(x, q.c);
        const cp = closestPointVertexTriangle(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz);
        if (cp.dist < minD) minD = cp.dist;
        if (cp.dist >= dHat || cp.dist < 1e-12) continue;
        const nx = cp.rx / cp.dist, ny = cp.ry / cp.dist, nz = cp.rz / cp.dist;
        const w1 = cp.s, w2 = cp.t, w0 = 1 - cp.s - cp.t;
        active.push({
          kind: K_VT, p: q.p, a: q.a, b: q.b, c: q.c, d: -1,
          w0, w1, w2, w3: 0, nx, ny, nz, key: vtKey(q.p, q.a, q.b, q.c),
        });
      } else {
        const [ax, ay, az] = this.posOf(x, q.a);
        const [bx, by, bz] = this.posOf(x, q.b);
        const [cx, cy, cz] = this.posOf(x, q.c);
        const [dx, dy, dz] = this.posOf(x, q.d);
        const cp = closestPointEdgeEdge(ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz);
        if (cp.dist < minD) minD = cp.dist;
        if (cp.dist >= dHat || cp.dist < 1e-12) continue;
        const nx = cp.rx / cp.dist, ny = cp.ry / cp.dist, nz = cp.rz / cp.dist;
        active.push({
          kind: K_EE, p: -1, a: q.a, b: q.b, c: q.c, d: q.d,
          w0: 1 - cp.s, w1: cp.s, w2: -(1 - cp.t), w3: -cp.t,
          nx, ny, nz, key: eeKey(q.a, q.b, q.c, q.d),
        });
      }
    }
    // floor actives (penetrating vertices INCLUDED: d may be negative;
    // frozenDist/energy clamp it downstream so the barrier pushes them out)
    if (this.floorY !== null) {
      const n = x.length / 3;
      for (let v = 0; v < n; v++) {
        const d = x[v * 3 + 1] - this.floorY;
        if (d < minD) minD = d;
        if (d >= dHat) continue;
        active.push({
          kind: 2, p: v, a: -1, b: -1, c: -1, d: -1,
          w0: 0, w1: 0, w2: 0, w3: 0, nx: 0, ny: 1, nz: 0, key: `floor:${v}`,
        });
      }
    }
    this.active = active;
    this.diag.activePairs = active.length;
    this.diag.vtPairs = active.filter((c) => c.kind === K_VT).length;
    this.diag.eePairs = active.filter((c) => c.kind === K_EE).length;
    this.diag.floorPairs = active.filter((c) => c.kind === 2).length;
    if (minD < this.diag.minDistance) this.diag.minDistance = minD;
  }

  private frozenDist(x: Float64Array, c: Frozen): number {
    if (c.kind === 2) return x[c.p * 3 + 1] - this.floorY!;
    if (c.kind === K_VT) {
      let ddx: number, ddy: number, ddz: number;
      if (c.a >= 0) {
        const qx = c.w0 * x[c.a * 3] + c.w1 * x[c.b * 3] + c.w2 * x[c.c * 3];
        const qy = c.w0 * x[c.a * 3 + 1] + c.w1 * x[c.b * 3 + 1] + c.w2 * x[c.c * 3 + 1];
        const qz = c.w0 * x[c.a * 3 + 2] + c.w1 * x[c.b * 3 + 2] + c.w2 * x[c.c * 3 + 2];
        ddx = x[c.p * 3] - qx; ddy = x[c.p * 3 + 1] - qy; ddz = x[c.p * 3 + 2] - qz;
      } else {
        const s = this.staticPos!;
        const ia = -c.a - 2, ib = -c.b - 2, ic = -c.c - 2;
        const sx = c.w0 * s[ia * 3] + c.w1 * s[ib * 3] + c.w2 * s[ic * 3];
        const sy = c.w0 * s[ia * 3 + 1] + c.w1 * s[ib * 3 + 1] + c.w2 * s[ic * 3 + 1];
        const sz = c.w0 * s[ia * 3 + 2] + c.w1 * s[ib * 3 + 2] + c.w2 * s[ic * 3 + 2];
        ddx = x[c.p * 3] - sx; ddy = x[c.p * 3 + 1] - sy; ddz = x[c.p * 3 + 2] - sz;
      }
      return c.nx * ddx + c.ny * ddy + c.nz * ddz;
    }
    // EE: r = wa*a + wb*b + wc*c + wd*d with wc,wd <= 0
    const axp = c.a >= 0 ? x[c.a * 3] : this.staticPos![(-c.a - 2) * 3];
    const ayp = c.a >= 0 ? x[c.a * 3 + 1] : this.staticPos![(-c.a - 2) * 3 + 1];
    const azp = c.a >= 0 ? x[c.a * 3 + 2] : this.staticPos![(-c.a - 2) * 3 + 2];
    const bxp = c.b >= 0 ? x[c.b * 3] : this.staticPos![(-c.b - 2) * 3];
    const byp = c.b >= 0 ? x[c.b * 3 + 1] : this.staticPos![(-c.b - 2) * 3 + 1];
    const bzp = c.b >= 0 ? x[c.b * 3 + 2] : this.staticPos![(-c.b - 2) * 3 + 2];
    const cxp = c.c >= 0 ? x[c.c * 3] : this.staticPos![(-c.c - 2) * 3];
    const cyp = c.c >= 0 ? x[c.c * 3 + 1] : this.staticPos![(-c.c - 2) * 3 + 1];
    const czp = c.c >= 0 ? x[c.c * 3 + 2] : this.staticPos![(-c.c - 2) * 3 + 2];
    const dxp = c.d >= 0 ? x[c.d * 3] : this.staticPos![(-c.d - 2) * 3];
    const dyp = c.d >= 0 ? x[c.d * 3 + 1] : this.staticPos![(-c.d - 2) * 3 + 1];
    const dzp = c.d >= 0 ? x[c.d * 3 + 2] : this.staticPos![(-c.d - 2) * 3 + 2];
    const rx = c.w0 * axp + c.w1 * bxp + c.w2 * cxp + c.w3 * dxp;
    const ry = c.w0 * ayp + c.w1 * byp + c.w2 * cyp + c.w3 * dyp;
    const rz = c.w0 * azp + c.w1 * bzp + c.w2 * czp + c.w3 * dzp;
    return c.nx * rx + c.ny * ry + c.nz * rz;
  }

  // ---- energy + gradient (barrier + lagged friction) ----

  energyGrad(x: Float64Array): { energy: number; grad: Float64Array } {
    const { dHatM: dHat, kappaJ: kappa, frictionMu: mu, frictionEpsM: eps } = this.params;
    const grad = new Float64Array(x.length);
    let energy = 0;
    let fwork = 0;
    for (const c of this.active) {
      const d = Math.max(this.frozenDist(x, c), 1e-12);
      if (d >= dHat) continue;
      energy += kappa * barrierValue(d, dHat);
      const g = barrierGradScalar(d, dHat, kappa, 1); // dE/dd (negative = repulsive)
      this.addDistGrad(grad, c, g);
      // lagged friction
      const lag = this.lagged.get(c.key);
      const nx = lag?.nx ?? c.nx, ny = lag?.ny ?? c.ny, nz = lag?.nz ?? c.nz;
      const lambdaN = lag?.lambdaN ?? barrierNormalForce(d, dHat, kappa, 1);
      const u = this.relativeSlip(x, c);
      const [fx, fy, fz] = coulombForce(u[0], u[1], u[2], nx, ny, nz, lambdaN, mu, eps);
      this.addFrictionForce(grad, c, -fx, -fy, -fz);
      fwork += fx * u[0] + fy * u[1] + fz * u[2];
    }
    this.diag.barrierEnergy = energy;
    this.diag.frictionWork += fwork;
    return { energy, grad };
  }

  /** dE/dx += g * dd/dx for frozen distance. g = dE/dd. */
  private addDistGrad(grad: Float64Array, c: Frozen, g: number): void {
    if (c.kind === 2) {
      grad[c.p * 3 + 1] += g; // dd/dy = 1
      return;
    }
    if (c.kind === K_VT) {
      grad[c.p * 3] += g * c.nx;
      grad[c.p * 3 + 1] += g * c.ny;
      grad[c.p * 3 + 2] += g * c.nz;
      const ws = [c.w0, c.w1, c.w2];
      const vs = [c.a, c.b, c.c];
      for (let k = 0; k < 3; k++) {
        if (vs[k] < 0) continue; // static: no gradient
        grad[vs[k] * 3] += -g * ws[k] * c.nx;
        grad[vs[k] * 3 + 1] += -g * ws[k] * c.ny;
        grad[vs[k] * 3 + 2] += -g * ws[k] * c.nz;
      }
      return;
    }
    const ws = [c.w0, c.w1, c.w2, c.w3];
    const vs = [c.a, c.b, c.c, c.d];
    for (let k = 0; k < 4; k++) {
      if (vs[k] < 0) continue;
      grad[vs[k] * 3] += g * ws[k] * c.nx;
      grad[vs[k] * 3 + 1] += g * ws[k] * c.ny;
      grad[vs[k] * 3 + 2] += g * ws[k] * c.nz;
    }
  }

  /** residual += -f physical friction force distributed over the stencil. */
  private addFrictionForce(grad: Float64Array, c: Frozen, fx: number, fy: number, fz: number): void {
    if (c.kind === 2) {
      grad[c.p * 3] += fx; grad[c.p * 3 + 1] += fy; grad[c.p * 3 + 2] += fz;
      return;
    }
    if (c.kind === K_VT) {
      grad[c.p * 3] += fx; grad[c.p * 3 + 1] += fy; grad[c.p * 3 + 2] += fz;
      const ws = [c.w0, c.w1, c.w2];
      const vs = [c.a, c.b, c.c];
      for (let k = 0; k < 3; k++) {
        if (vs[k] < 0) continue;
        grad[vs[k] * 3] += -fx * ws[k];
        grad[vs[k] * 3 + 1] += -fy * ws[k];
        grad[vs[k] * 3 + 2] += -fz * ws[k];
      }
      return;
    }
    // EE: +f on edge ab (s-weighted), -f on edge cd (t-weighted)
    const pairs: Array<[number, number]> = [[c.a, c.w0], [c.b, c.w1], [c.c, -c.w2], [c.d, -c.w3]];
    for (const [v, w] of pairs) {
      if (v < 0) continue;
      grad[v * 3] += fx * w; grad[v * 3 + 1] += fy * w; grad[v * 3 + 2] += fz * w;
    }
  }

  /** Relative slip of the contact stencil from step start (current s,t weights). */
  private relativeSlip(x: Float64Array, c: Frozen): [number, number, number] {
    const D = (v: number): [number, number, number] => {
      if (v < 0) return [0, 0, 0];
      return [x[v * 3] - this.xStep[v * 3], x[v * 3 + 1] - this.xStep[v * 3 + 1], x[v * 3 + 2] - this.xStep[v * 3 + 2]];
    };
    if (c.kind === 2) {
      const d = D(c.p);
      return d;
    }
    if (c.kind === K_VT) {
      const dp = D(c.p), da = D(c.a), db = D(c.b), dc = D(c.d);
      return [
        dp[0] - (c.w0 * da[0] + c.w1 * db[0] + c.w2 * dc[0]),
        dp[1] - (c.w0 * da[1] + c.w1 * db[1] + c.w2 * dc[1]),
        dp[2] - (c.w0 * da[2] + c.w1 * db[2] + c.w2 * dc[2]),
      ];
    }
    const da = D(c.a), db = D(c.b), dc = D(c.c), dd = D(c.d);
    return [
      c.w0 * da[0] + c.w1 * db[0] + c.w2 * dc[0] + c.w3 * dd[0],
      c.w0 * da[1] + c.w1 * db[1] + c.w2 * dc[1] + c.w3 * dd[1],
      c.w0 * da[2] + c.w1 * db[2] + c.w2 * dc[2] + c.w3 * dd[2],
    ];
  }

  // ---- analytic HVP (barrier only; friction lagged out) ----

  applyHvp(x: Float64Array, v: Float64Array, out: Float64Array): void {
    const { dHatM: dHat, kappaJ: kappa } = this.params;
    out.fill(0);
    const dd = new Float64Array(this.active.length);
    for (let i = 0; i < this.active.length; i++) {
      const c = this.active[i];
      const d = Math.max(this.frozenDist(x, c), 1e-12);
      if (d >= dHat) { dd[i] = 0; continue; }
      const L = Math.log(d / dHat);
      const u = d - dHat;
      // b''(d) = -(2L + 4u/d - u^2/d^2)
      const b2 = -(2 * L + (4 * u) / d - (u * u) / (d * d));
      const coeff = kappa * Math.max(b2, 0);
      // dd/dx . v
      const jv = this.distJvp(c, v);
      dd[i] = coeff * jv;
    }
    for (let i = 0; i < this.active.length; i++) {
      if (dd[i] === 0) continue;
      this.addDistGrad(out, this.active[i], dd[i]);
    }
  }

  private distJvp(c: Frozen, v: Float64Array): number {
    if (c.kind === 2) return v[c.p * 3 + 1];
    if (c.kind === K_VT) {
      let j = c.nx * v[c.p * 3] + c.ny * v[c.p * 3 + 1] + c.nz * v[c.p * 3 + 2];
      const ws = [c.w0, c.w1, c.w2];
      const vs = [c.a, c.b, c.c];
      for (let k = 0; k < 3; k++) {
        if (vs[k] < 0) continue;
        j -= ws[k] * (c.nx * v[vs[k] * 3] + c.ny * v[vs[k] * 3 + 1] + c.nz * v[vs[k] * 3 + 2]);
      }
      return j;
    }
    const ws = [c.w0, c.w1, c.w2, c.w3];
    const vs = [c.a, c.b, c.c, c.d];
    let j = 0;
    for (let k = 0; k < 4; k++) {
      if (vs[k] < 0) continue;
      j += ws[k] * (c.nx * v[vs[k] * 3] + c.ny * v[vs[k] * 3 + 1] + c.nz * v[vs[k] * 3 + 2]);
    }
    return j;
  }

  // ---- merit energy under the FROZEN active set (for Armijo trials) ----

  /** Barrier energy at xTrial using the frozen (s,t,n) model — no rebuild. */
  meritEnergy(x: Float64Array): number {
    const { dHatM: dHat, kappaJ: kappa } = this.params;
    let e = 0;
    for (const c of this.active) {
      const d = Math.max(this.frozenDist(x, c), 1e-12);
      if (d >= dHat) continue;
      e += kappa * barrierValue(d, dHat);
    }
    return e;
  }

  /** Read-only snapshot of the current frozen active set. */
  activeList(): ActiveContactView[] {
    return this.active;
  }

  /** Add frozen-barrier diagonal curvature into the Jacobi preconditioner. */
  addDiagEstimate(
    diag: Float64Array, x: Float64Array,
    masses: Float32Array, invH2: number, beta: number,
  ): void {
    void masses; void invH2; void beta;
    const { dHatM: dHat, kappaJ: kappa } = this.params;
    for (const c of this.active) {
      const d = Math.max(this.frozenDist(x, c), 1e-12);
      if (d >= dHat) continue;
      const L = Math.log(d / dHat);
      const u = d - dHat;
      const b2 = Math.max(-(2 * L + (4 * u) / d - (u * u) / (d * d)), 0);
      const k = kappa * b2;
      if (c.kind === 2) {
        diag[c.p * 3 + 1] += k;
        continue;
      }
      const n2 = [c.nx * c.nx, c.ny * c.ny, c.nz * c.nz];
      if (c.kind === K_VT) {
        for (let kk = 0; kk < 3; kk++) diag[c.p * 3 + kk] += k * n2[kk];
        const ws = [c.w0, c.w1, c.w2];
        const vs = [c.a, c.b, c.c];
        for (let j = 0; j < 3; j++) {
          if (vs[j] < 0) continue;
          for (let kk = 0; kk < 3; kk++) diag[vs[j] * 3 + kk] += k * ws[j] * ws[j] * n2[kk];
        }
        continue;
      }
      const ws = [c.w0, c.w1, c.w2, c.w3];
      const vs = [c.a, c.b, c.c, c.d];
      for (let j = 0; j < 4; j++) {
        if (vs[j] < 0) continue;
        for (let kk = 0; kk < 3; kk++) diag[vs[j] * 3 + kk] += k * ws[j] * ws[j] * n2[kk];
      }
    }
  }

  // ---- commit (accepted iterate): refresh lagged normals / multipliers ----

  commit(x: Float64Array): void {
    const { dHatM: dHat, kappaJ: kappa } = this.params;
    for (const c of this.active) {
      const d = Math.max(this.frozenDist(x, c), 1e-12);
      if (d >= dHat) continue;
      this.lagged.set(c.key, {
        nx: c.nx, ny: c.ny, nz: c.nz,
        lambdaN: barrierNormalForce(d, dHat, kappa, 1),
      });
    }
  }

  // ---- line-search validation against the Newton segment ----

  checkTrial(xT: Float64Array): { valid: boolean; minDist: number; minTOI: number } {
    const dMin = this.params.dMinM;
    // Tight pad: a pair can only violate the hard core (reach <= dMin) or
    // cross it if its swept trajectories come within dMin, so dMin-pad
    // candidates are sufficient for validity (barrier zone uses dHat pad).
    const cands = this.buildCandidates(xT, dMin);
    let minDist = Infinity;
    let minTOI = Infinity;
    // exact distances at trial
    for (const q of cands) {
      const d = this.exactDist(xT, q);
      if (d < minDist) minDist = d;
      if (!isFinite(d)) { this.diag.ccdFailures++; return { valid: false, minDist, minTOI }; }
    }
    if (this.floorY !== null) {
      const n = xT.length / 3;
      for (let v = 0; v < n; v++) {
        const d = xT[v * 3 + 1] - this.floorY;
        if (d < minDist) minDist = d;
      }
    }
    if (minDist <= dMin) return { valid: false, minDist, minTOI };
    for (const q of cands) {
      // CCD kernels early-out internally on relative motion (rigid
      // translation cancels) and return 0 for resting pairs, so no outer
      // distance query is needed: 0 < toi < 1 is exactly a fresh crossing.
      // Static verts (negative ids) ride in extended arrays with zero motion.
      let toi = Infinity;
      if (q.kind === K_VT) {
        if (q.p < 0 && q.a < 0) continue; // static-static: impossible
        if (q.p < 0 || q.a < 0) {
          toi = this.staticVtCCD(xT, q.p, q.a, q.b, q.c, dMin);
        } else {
          toi = vtCCD(this.xStep, xT, q.p, q.a, q.b, q.c, dMin);
        }
      } else {
        if (q.a < 0 && q.b < 0 && q.c < 0 && q.d < 0) continue;
        if (q.a < 0 || q.b < 0 || q.c < 0 || q.d < 0) {
          toi = this.staticEeCCD(xT, q.a, q.b, q.c, q.d, dMin);
        } else {
          toi = eeCCD(this.xStep, xT, q.a, q.b, q.c, q.d, dMin);
        }
      }
      if (Number.isNaN(toi)) { this.diag.ccdFailures++; return { valid: false, minDist, minTOI }; }
      if (toi < minTOI) minTOI = toi;
      if (toi > 0 && toi < 1 - 1e-9) return { valid: false, minDist, minTOI };
    }
    // floor CCD (single-sided: y(t) drops by at most the vertex motion)
    if (this.floorY !== null) {
      const n = xT.length / 3;
      for (let v = 0; v < n; v++) {
        const y0 = this.xStep[v * 3 + 1] - this.floorY;
        const yT = xT[v * 3 + 1] - this.floorY;
        if (y0 - Math.abs(yT - y0) > dMin) continue;
        if (y0 > dMin && yT <= dMin && y0 > yT) {
          const toi = (y0 - dMin) / (y0 - yT);
          if (toi < minTOI) minTOI = toi;
          if (toi < 1 - 1e-9) return { valid: false, minDist, minTOI };
        }
      }
    }
    if (minDist < this.diag.minDistance) this.diag.minDistance = minDist;
    if (minTOI < this.diag.minTOI) this.diag.minTOI = minTOI;
    return { valid: true, minDist, minTOI };
  }

  /** CCD with static verts: remap negative ids into extended arrays (zero motion). */
  private extArrays(xT: Float64Array): { X0: Float64Array; X1: Float64Array; n: number } {
    const n = this.xStep.length / 3;
    const ns = this.staticPos!.length / 3;
    const X0 = new Float64Array((n + ns) * 3);
    const X1 = new Float64Array((n + ns) * 3);
    X0.set(this.xStep, 0);
    X0.set(this.staticPos!, n * 3);
    X1.set(xT, 0);
    X1.set(this.staticPos!, n * 3);
    return { X0, X1, n };
  }

  private remap(v: number, n: number): number {
    return v >= 0 ? v : n + (-v - 2);
  }

  private staticVtCCD(xT: Float64Array, p: number, a: number, b: number, c: number, dMin: number): number {
    const { X0, X1, n } = this.extArrays(xT);
    return vtCCD(X0, X1, this.remap(p, n), this.remap(a, n), this.remap(b, n), this.remap(c, n), dMin);
  }

  private staticEeCCD(xT: Float64Array, a: number, b: number, c: number, d: number, dMin: number): number {
    const { X0, X1, n } = this.extArrays(xT);
    return eeCCD(X0, X1, this.remap(a, n), this.remap(b, n), this.remap(c, n), this.remap(d, n), dMin);
  }

  private exactDist(x: Float64Array, q: Cand): number {    if (q.kind === K_VT) {
      const [px, py, pz] = this.posOf(x, q.p);
      const [ax, ay, az] = this.posOf(x, q.a);
      const [bx, by, bz] = this.posOf(x, q.b);
      const [cx, cy, cz] = this.posOf(x, q.c);
      return closestPointVertexTriangle(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz).dist;
    }
    const [ax, ay, az] = this.posOf(x, q.a);
    const [bx, by, bz] = this.posOf(x, q.b);
    const [cx, cy, cz] = this.posOf(x, q.c);
    const [dx, dy, dz] = this.posOf(x, q.d);
    return closestPointEdgeEdge(ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz).dist;
  }
}

function staticTriAabb(pos: Float32Array, i0: number, i1: number, i2: number, pad: number): Aabb {
  const xs = [pos[i0 * 3], pos[i1 * 3], pos[i2 * 3]];
  const ys = [pos[i0 * 3 + 1], pos[i1 * 3 + 1], pos[i2 * 3 + 1]];
  const zs = [pos[i0 * 3 + 2], pos[i1 * 3 + 2], pos[i2 * 3 + 2]];
  return {
    minX: Math.min(...xs) - pad, minY: Math.min(...ys) - pad, minZ: Math.min(...zs) - pad,
    maxX: Math.max(...xs) + pad, maxY: Math.max(...ys) + pad, maxZ: Math.max(...zs) + pad,
  };
}

function vtKey(p: number, a: number, b: number, c: number): string {
  const t = [a, b, c].sort((x, y) => x - y).join("_");
  return `vt:${p}:${t}`;
}

function eeKey(a: number, b: number, c: number, d: number): string {
  const e1 = a < b ? `${a}_${b}` : `${b}_${a}`;
  const e2 = c < d ? `${c}_${d}` : `${d}_${c}`;
  return e1 < e2 ? `ee:${e1}:${e2}` : `ee:${e2}:${e1}`;
}

export { triPairKey };
