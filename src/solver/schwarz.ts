// G5C one-level connectivity-aware additive Schwarz (CPU reference).
//
// Deterministic non-overlapping vertex patches (8-32 verts/domain) built from
// mesh connectivity + Morton locality. Per-domain local Hessian assembled from
// the analytic membrane 9x9 blocks (membrane-blocks.ts) for intra-domain corner
// pairs + the Jacobi diagonal (M/h^2 + beta + contact curvature), factored
// once per Newton iteration with dense Cholesky (Jacobi-diagonal fallback per
// domain when indefinite). Apply is additive: z = sum_d R_d^T A_d^{-1} R_d r.
//
// The exact PCG operator (HVP) is UNCHANGED — this only approximates M^{-1}.
// Bending stays out (inexact Newton, same as HVP). Pins: caller filters r/z
// (same convention as block-jacobi.ts / the GPU kernels).

import { triangleHessian9 } from "../physics/membrane-blocks.js";
import type { ClothMeshData } from "../mesh/mesh.js";
import type { ClothMaterial } from "../physics/types.js";

export interface SchwarzDomains {
  /** vertex -> domain id (length n) */
  domainOf: Uint32Array;
  /** local index of vertex within its domain (length n) */
  localOf: Uint32Array;
  /** domain -> member vertices (creation order, deterministic) */
  members: number[][];
}

function morton3(x: number, y: number, z: number): number {
  // 10 bits per axis interleaved into 30 bits (rest-space locality only).
  const q = (v: number): number => Math.max(0, Math.min(1023, Math.floor(v * 1024)));
  const xi = q(x), yi = q(y), zi = q(z);
  let m = 0;
  for (let b = 9; b >= 0; b--) {
    m = (m << 3) | (((xi >> b) & 1) << 2) | (((yi >> b) & 1) << 1) | ((zi >> b) & 1);
  }
  return m >>> 0;
}

/**
 * Build deterministic domains: Morton-ordered seeds, BFS growth along mesh
 * connectivity to targetSize, then merge of sub-minimum remainders into the
 * smallest adjacent domain. Result: every domain in [minSize, maxSize] unless
 * the mesh itself is smaller than minSize (single domain then).
 */
export function buildSchwarzDomains(
  mesh: ClothMeshData,
  rest: ArrayLike<number>,
  targetSize = 16,
  minSize = 8,
  maxSize = 32,
): SchwarzDomains {
  const n = mesh.count;
  const idx = mesh.indices;
  // Vertex adjacency (sorted arrays for determinism).
  const adj: Set<number>[] = Array.from({ length: n }, () => new Set<number>());
  for (let t = 0; t < mesh.triCount; t++) {
    const a = idx[t * 3], b = idx[t * 3 + 1], c = idx[t * 3 + 2];
    adj[a].add(b); adj[a].add(c);
    adj[b].add(a); adj[b].add(c);
    adj[c].add(a); adj[c].add(b);
  }
  const adjArr = adj.map((s) => [...s].sort((x, y) => x - y));
  // Morton order over rest-space bbox.
  let mnx = Infinity, mny = Infinity, mnz = Infinity;
  let mxx = -Infinity, mxy = -Infinity, mxz = -Infinity;
  for (let v = 0; v < n; v++) {
    mnx = Math.min(mnx, rest[v * 3]); mny = Math.min(mny, rest[v * 3 + 1]); mnz = Math.min(mnz, rest[v * 3 + 2]);
    mxx = Math.max(mxx, rest[v * 3]); mxy = Math.max(mxy, rest[v * 3 + 1]); mxz = Math.max(mxz, rest[v * 3 + 2]);
  }
  const sx = Math.max(mxx - mnx, 1e-12), sy = Math.max(mxy - mny, 1e-12), sz = Math.max(mxz - mnz, 1e-12);
  const order = Array.from({ length: n }, (_, v) => v);
  const code = new Uint32Array(n);
  for (let v = 0; v < n; v++) {
    code[v] = morton3(
      (rest[v * 3] - mnx) / sx, (rest[v * 3 + 1] - mny) / sy, (rest[v * 3 + 2] - mnz) / sz,
    );
  }
  order.sort((a, b) => (code[a] - code[b]) || (a - b));

  const domainOf = new Uint32Array(n).fill(0xffffffff);
  const members: number[][] = [];
  // BFS growth from each unassigned Morton seed.
  for (const seed of order) {
    if (domainOf[seed] !== 0xffffffff) continue;
    const id = members.length;
    const mem: number[] = [seed];
    domainOf[seed] = id;
    const queue: number[] = [seed];
    while (mem.length < targetSize && queue.length > 0) {
      // Deterministic: drain in FIFO order, neighbors ascending.
      const cur = queue.shift()!;
      for (const w of adjArr[cur]) {
        if (mem.length >= targetSize) break;
        if (domainOf[w] !== 0xffffffff) continue;
        domainOf[w] = id;
        mem.push(w);
        queue.push(w);
      }
    }
    members.push(mem);
  }
  // Merge sub-minimum domains into the smallest adjacent domain.
  const domainAdj = (): Map<number, Set<number>> => {
    const m = new Map<number, Set<number>>();
    for (let d = 0; d < members.length; d++) m.set(d, new Set());
    for (let t = 0; t < mesh.triCount; t++) {
      const ds = [domainOf[idx[t * 3]], domainOf[idx[t * 3 + 1]], domainOf[idx[t * 3 + 2]]];
      for (const a of ds) for (const b of ds) if (a !== b) m.get(a)!.add(b);
    }
    return m;
  };
  for (;;) {
    let small = -1;
    for (let d = 0; d < members.length; d++) {
      if (members[d].length > 0 && members[d].length < minSize && n >= minSize) {
        if (small < 0 || members[d].length < members[small].length) small = d;
      }
    }
    if (small < 0) break;
    const adjM = domainAdj();
    let best = -1;
    for (const c of [...adjM.get(small)!].sort((a, b) => a - b)) {
      if (members[c].length === 0) continue;
      if (best < 0 || members[c].length < members[best].length) best = c;
    }
    if (best < 0) break; // isolated (single-domain mesh) — keep as is.
    for (const v of members[small]) { domainOf[v] = best; members[best].push(v); }
    members[small] = [];
  }
  // Compact ids (preserve creation order), sort members for determinism.
  const live = members.map((m, d) => ({ m, d })).filter((e) => e.m.length > 0);
  const remap = new Map<number, number>();
  live.forEach((e, k) => remap.set(e.d, k));
  const out: number[][] = live.map((e) => [...e.m].sort((a, b) => a - b));
  const finalOf = new Uint32Array(n);
  for (let v = 0; v < n; v++) finalOf[v] = remap.get(domainOf[v])!;
  const localOf = new Uint32Array(n);
  out.forEach((mem) => mem.forEach((v, l) => { localOf[v] = l; }));
  void maxSize; // growth caps at targetSize <= maxSize by construction.
  return { domainOf: finalOf, localOf, members: out };
}

export interface SchwarzFactors {
  domains: SchwarzDomains;
  /** per-domain Cholesky L (row-major, 3d x 3d) or Jacobi-diag inverse when flag=0 */
  data: Float64Array[];
  /** 1 = Cholesky ok, 0 = Jacobi-diagonal fallback */
  flag: Uint8Array;
}

/**
 * Assemble + factor local matrices. `diag` is the full Jacobi diagonal
 * (length n*3), same convention as buildBlockFactors.
 */
export function buildSchwarzFactors(
  x: ArrayLike<number>,
  mesh: ClothMeshData,
  mat: ClothMaterial,
  diag: ArrayLike<number>,
  domains: SchwarzDomains,
): SchwarzFactors {
  const idx = mesh.indices;
  const D = domains.members.length;
  const data: Float64Array[] = [];
  const flag = new Uint8Array(D);
  // Vertex -> incident triangles (one linear pass; avoids O(V*T) scans).
  const v2t: number[][] = Array.from({ length: mesh.count }, () => []);
  for (let t = 0; t < mesh.triCount; t++) {
    v2t[idx[t * 3]].push(t);
    v2t[idx[t * 3 + 1]].push(t);
    v2t[idx[t * 3 + 2]].push(t);
  }
  for (let d = 0; d < D; d++) {
    const mem = domains.members[d];
    const dv = mem.length;
    const dw = dv * 3;
    const A = new Float64Array(dw * dw);
    const loc = new Map<number, number>();
    mem.forEach((v, l) => loc.set(v, l));
    // Membrane coupling for intra-domain corner pairs, distributed from each
    // triangle's analytic 9x9 (computed once per triangle occurrence below by
    // corner-pair accumulation over incident triangles).
    for (const v of mem) {
      for (const t of v2t[v]) {
        const cvs = [idx[t * 3], idx[t * 3 + 1], idx[t * 3 + 2]];
        const ci = cvs.indexOf(v);
        // Only accumulate rows of v here (each row assembled once).
        const x0 = [x[cvs[0] * 3], x[cvs[0] * 3 + 1], x[cvs[0] * 3 + 2]];
        const x1 = [x[cvs[1] * 3], x[cvs[1] * 3 + 1], x[cvs[1] * 3 + 2]];
        const x2 = [x[cvs[2] * 3], x[cvs[2] * 3 + 1], x[cvs[2] * 3 + 2]];
        const inv4 = [mesh.invDm[t * 4], mesh.invDm[t * 4 + 1], mesh.invDm[t * 4 + 2], mesh.invDm[t * 4 + 3]];
        const { h } = triangleHessian9(x0, x1, x2, inv4, mesh.areas[t], mat);
        for (let cj = 0; cj < 3; cj++) {
          const w = cvs[cj];
          const lw = loc.get(w);
          if (lw === undefined) continue; // cross-domain coupling: dropped (one-level)
          const lv = loc.get(v)!;
          for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
            A[(lv * 3 + a) * dw + (lw * 3 + b)] += h[(ci * 3 + a) * 9 + (cj * 3 + b)];
          }
        }
      }
    }
    // Jacobi-diagonal embed on local diagonal.
    for (let l = 0; l < dv; l++) {
      for (let a = 0; a < 3; a++) {
        A[(l * 3 + a) * dw + (l * 3 + a)] += diag[mem[l] * 3 + a];
      }
    }
    // Symmetrize (analytic blocks symmetric to 1e-14; fp accumulation order).
    for (let i = 0; i < dw; i++) {
      for (let j = i + 1; j < dw; j++) {
        const s = 0.5 * (A[i * dw + j] + A[j * dw + i]);
        A[i * dw + j] = s; A[j * dw + i] = s;
      }
    }
    // Dense Cholesky (row-major L in place, lower triangle).
    let scale = 0;
    for (let i = 0; i < dw; i++) scale = Math.max(scale, Math.abs(A[i * dw + i]));
    const tiny = 1e-20 * (1 + scale);
    let ok = true;
    const L = Float64Array.from(A);
    for (let k = 0; k < dw && ok; k++) {
      let sum = L[k * dw + k];
      for (let s = 0; s < k; s++) sum -= L[k * dw + s] * L[k * dw + s];
      if (!(sum > tiny)) { ok = false; break; }
      L[k * dw + k] = Math.sqrt(sum);
      for (let i = k + 1; i < dw; i++) {
        let s2 = L[i * dw + k];
        for (let s = 0; s < k; s++) s2 -= L[i * dw + s] * L[k * dw + s];
        L[i * dw + k] = s2 / L[k * dw + k];
      }
    }
    if (ok) {
      // Keep lower triangle only (upper still holds the original A entries).
      for (let i = 0; i < dw; i++) for (let j = i + 1; j < dw; j++) L[i * dw + j] = 0;
      data.push(L);
      flag[d] = 1;
    } else {
      // Symmetric Jacobi-diagonal fallback (keeps the preconditioner SPD).
      const J = new Float64Array(dw * dw);
      for (let l = 0; l < dv; l++) {
        for (let a = 0; a < 3; a++) {
          const dd = diag[mem[l] * 3 + a];
          J[(l * 3 + a) * dw + (l * 3 + a)] = dd > 1e-12 ? 1 / dd : 0;
        }
      }
      data.push(J);
      flag[d] = 0;
    }
  }
  return { domains, data, flag };
}

/** Solve one local system (Cholesky or pre-inverted Jacobi fallback). */
export function solveLocal(factors: SchwarzFactors, d: number, rhs: ArrayLike<number>, out: Float64Array): void {
  const dv = factors.domains.members[d].length;
  const dw = dv * 3;
  const M = factors.data[d];
  if (factors.flag[d] === 1) {
    const y = new Float64Array(dw);
    for (let i = 0; i < dw; i++) {
      let s = rhs[i];
      for (let j = 0; j < i; j++) s -= M[i * dw + j] * y[j];
      y[i] = s / M[i * dw + i];
    }
    for (let i = dw - 1; i >= 0; i--) {
      let s = y[i];
      for (let j = i + 1; j < dw; j++) s -= M[j * dw + i] * out[j];
      out[i] = s / M[i * dw + i];
    }
  } else {
    for (let i = 0; i < dw; i++) {
      let s = 0;
      for (let j = 0; j < dw; j++) s += M[i * dw + j] * rhs[j];
      out[i] = s;
    }
  }
}

/** Additive apply: z = sum_d R_d^T A_d^{-1} R_d r (out must be zeroed or pass zeroed). */
export function applySchwarz(
  factors: SchwarzFactors,
  r: ArrayLike<number>,
  out: Float64Array,
  vertexCount: number,
): Float64Array {
  out.fill(0);
  const D = factors.domains.members.length;
  const rhs = new Float64Array(96 * 3);
  const sol = new Float64Array(96 * 3);
  for (let d = 0; d < D; d++) {
    const mem = factors.domains.members[d];
    const dv = mem.length;
    const dw = dv * 3;
    for (let l = 0; l < dv; l++) {
      rhs[l * 3] = r[mem[l] * 3];
      rhs[l * 3 + 1] = r[mem[l] * 3 + 1];
      rhs[l * 3 + 2] = r[mem[l] * 3 + 2];
    }
    solveLocal(factors, d, rhs.subarray(0, dw), sol);
    for (let l = 0; l < dv; l++) {
      out[mem[l] * 3] += sol[l * 3];
      out[mem[l] * 3 + 1] += sol[l * 3 + 1];
      out[mem[l] * 3 + 2] += sol[l * 3 + 2];
    }
  }
  void vertexCount;
  return out;
}
