// G5.5 assembled coarse system in 3x3 block-CSR (CPU reference).
//
// Aggregates = G5C Schwarz domains; coarse space = constant vector per
// aggregate (3 dofs each), same normalized R/P as mas.ts. Ac = R A P with
// A = membrane Hessian (+ Jacobi-diag embed), i.e. the same operator as the
// dense buildCoarseExact — but stored sparsely: aggregates couple iff they
// share a triangle, so the block graph is mesh-local and tiny.
//
// Layout: rowOffsets (D+1, block counts), colIndices (nnz block columns,
// sorted per row), values (nnz*9 row-major 3x3, ordered by block position).
// Pattern is STATIC (mesh topology); values refresh per Newton state.
// Membrane + diag-spread only in Ac (G5 convention); barrier/contact terms
// stay in the exact fine HVP. Pins are caller-filtered at prolongation.

import { triangleHessian9 } from "../physics/membrane-blocks.js";
import type { ClothMeshData } from "../mesh/mesh.js";
import type { ClothMaterial } from "../physics/types.js";
import type { SchwarzDomains } from "./schwarz.js";

export interface CoarsePattern {
  /** block-row offsets, length D+1 */
  rowOffsets: Uint32Array;
  /** block column per nonzero, length nnz (sorted within each row) */
  colIndices: Uint32Array;
  nnz: number;
  domains: number;
}

/** Block-CSR sparsity: one block per aggregate pair sharing a triangle. */
export function buildCoarsePattern(
  mesh: ClothMeshData,
  domains: SchwarzDomains,
): CoarsePattern {
  const D = domains.members.length;
  const rows: Set<number>[] = Array.from({ length: D }, () => new Set<number>());
  const idx = mesh.indices;
  for (let t = 0; t < mesh.triCount; t++) {
    const ds = [domains.domainOf[idx[t * 3]], domains.domainOf[idx[t * 3 + 1]], domains.domainOf[idx[t * 3 + 2]]];
    for (const a of ds) for (const b of ds) rows[a].add(b);
  }
  const rowOffsets = new Uint32Array(D + 1);
  const cols: number[] = [];
  for (let d = 0; d < D; d++) {
    const sorted = [...rows[d]].sort((x, y) => x - y);
    rowOffsets[d + 1] = rowOffsets[d] + sorted.length;
    cols.push(...sorted);
  }
  return { rowOffsets, colIndices: Uint32Array.from(cols), nnz: cols.length, domains: D };
}

/** Block position of (d1,d2) in the pattern, or -1. */
export function blockPos(pattern: CoarsePattern, d1: number, d2: number): number {
  const lo = pattern.rowOffsets[d1];
  const hi = pattern.rowOffsets[d1 + 1];
  for (let p = lo; p < hi; p++) {
    if (pattern.colIndices[p] === d2) return p;
    if (pattern.colIndices[p] > d2) break;
  }
  return -1;
}

export interface CoarseValues {
  pattern: CoarsePattern;
  /** nnz*9 row-major 3x3 blocks, symmetrized */
  blocks: Float64Array;
}

/** Raw (unsymmetrized) assembly — matches what the GPU kernel stores. */
export function assembleCoarseRaw(
  x: ArrayLike<number>,
  mesh: ClothMeshData,
  mat: ClothMaterial,
  diag: ArrayLike<number>,
  domains: SchwarzDomains,
  pattern: CoarsePattern,
): Float64Array {
  const blocks = new Float64Array(pattern.nnz * 9);
  const idx = mesh.indices;
  const sizes = domains.members.map((m) => m.length);
  for (let t = 0; t < mesh.triCount; t++) {
    const cvs = [idx[t * 3], idx[t * 3 + 1], idx[t * 3 + 2]];
    const x0 = [x[cvs[0] * 3], x[cvs[0] * 3 + 1], x[cvs[0] * 3 + 2]];
    const x1 = [x[cvs[1] * 3], x[cvs[1] * 3 + 1], x[cvs[1] * 3 + 2]];
    const x2 = [x[cvs[2] * 3], x[cvs[2] * 3 + 1], x[cvs[2] * 3 + 2]];
    const inv4 = [mesh.invDm[t * 4], mesh.invDm[t * 4 + 1], mesh.invDm[t * 4 + 2], mesh.invDm[t * 4 + 3]];
    const { h } = triangleHessian9(x0, x1, x2, inv4, mesh.areas[t], mat);
    for (let ci = 0; ci < 3; ci++) {
      for (let cj = 0; cj < 3; cj++) {
        const d1 = domains.domainOf[cvs[ci]];
        const d2 = domains.domainOf[cvs[cj]];
        const p = blockPos(pattern, d1, d2);
        if (p < 0) throw new Error(`coarse-csr: missing block (${d1},${d2}) — pattern/mesh mismatch`);
        const s = 1 / Math.sqrt(sizes[d1] * sizes[d2]);
        for (let a = 0; a < 3; a++) {
          for (let b = 0; b < 3; b++) {
            blocks[p * 9 + a * 3 + b] += s * h[(ci * 3 + a) * 9 + (cj * 3 + b)];
          }
        }
      }
    }
  }
  // R diag(H) P spread: diagonal entries of diagonal blocks only.
  domains.members.forEach((mem, d) => {
    const p = blockPos(pattern, d, d);
    const s = 1 / mem.length;
    for (const v of mem) {
      for (let a = 0; a < 3; a++) blocks[p * 9 + a * 3 + a] += s * diag[v * 3 + a];
    }
  });
  return blocks;
}

/**
 * Assemble Ac values: membrane corner-pair Hessian / sqrt(|d1||d2|) plus the
 * Jacobi-diagonal spread on diagonal blocks. Block-symmetrized
 * (B(d1,d2) = 0.5*(B + B(d2,d1)^T)) so the operator is exactly symmetric.
 */
export function assembleCoarseValues(
  x: ArrayLike<number>,
  mesh: ClothMeshData,
  mat: ClothMaterial,
  diag: ArrayLike<number>,
  domains: SchwarzDomains,
  pattern: CoarsePattern,
): CoarseValues {
  const blocks = assembleCoarseRaw(x, mesh, mat, diag, domains, pattern);
  // Exact block symmetrization: B(d1,d2) = 0.5*(B(d1,d2) + B(d2,d1)^T).
  const out = Float64Array.from(blocks);
  for (let d1 = 0; d1 < pattern.domains; d1++) {
    for (let p = pattern.rowOffsets[d1]; p < pattern.rowOffsets[d1 + 1]; p++) {
      const d2 = pattern.colIndices[p];
      const q = blockPos(pattern, d2, d1);
      for (let a = 0; a < 3; a++) {
        for (let b = 0; b < 3; b++) {
          out[p * 9 + a * 3 + b] = 0.5 * (blocks[p * 9 + a * 3 + b] + blocks[q * 9 + b * 3 + a]);
        }
      }
    }
  }
  return { pattern, blocks: out };
}

/** y = Ac x (block-CSR SpMV). */
export function coarseSpmv(
  values: CoarseValues,
  x: ArrayLike<number>,
  out: Float64Array,
): Float64Array {
  out.fill(0);
  const { pattern, blocks } = values;
  for (let d1 = 0; d1 < pattern.domains; d1++) {
    for (let p = pattern.rowOffsets[d1]; p < pattern.rowOffsets[d1 + 1]; p++) {
      const d2 = pattern.colIndices[p];
      for (let a = 0; a < 3; a++) {
        let s = 0;
        for (let b = 0; b < 3; b++) s += blocks[p * 9 + a * 3 + b] * x[d2 * 3 + b];
        out[d1 * 3 + a] += s;
      }
    }
  }
  return out;
}

export interface Sym33Inverse {
  inv: Float64Array; // 9 entries row-major
  ok: boolean; // false = Jacobi-diagonal fallback baked in
}

/**
 * Symmetric 3x3 inverse with the G5 Sylvester latch (same thresholds as
 * block-jacobi.ts / bj_build_factor, so CPU and GPU agree bit-near).
 * Fallback: diagonal 1/fb (fb = per-component pivot, e.g. true Ac diagonal).
 */
export function invertSym33(
  B00: number, B01: number, B02: number,
  B11: number, B12: number, B22: number,
  fb0: number, fb1: number, fb2: number,
): Sym33Inverse {
  const tr = Math.abs(B00) + Math.abs(B11) + Math.abs(B22);
  const tiny = 1e-20 * (1 + tr);
  let ok = true;
  if (!(B00 > tiny)) ok = false;
  if (ok && !(B00 * B11 - B01 * B01 > tiny * (1 + Math.abs(B00) + Math.abs(B11)))) ok = false;
  const det =
    B00 * (B11 * B22 - B12 * B12) -
    B01 * (B01 * B22 - B12 * B02) +
    B02 * (B01 * B12 - B11 * B02);
  if (ok && !(det > tiny * (1 + tr) * (1 + tr))) ok = false;
  const inv = new Float64Array(9);
  if (ok) {
    const s = 1 / det;
    inv[0] = (B11 * B22 - B12 * B12) * s;
    inv[1] = (B02 * B12 - B01 * B22) * s;
    inv[2] = (B01 * B12 - B11 * B02) * s;
    inv[3] = inv[1];
    inv[4] = (B00 * B22 - B02 * B02) * s;
    inv[5] = (B01 * B02 - B00 * B12) * s;
    inv[6] = inv[2];
    inv[7] = inv[5];
    inv[8] = (B00 * B11 - B01 * B01) * s;
    return { inv, ok: true };
  }
  inv[0] = fb0 > 1e-12 ? 1 / fb0 : 0;
  inv[4] = fb1 > 1e-12 ? 1 / fb1 : 0;
  inv[8] = fb2 > 1e-12 ? 1 / fb2 : 0;
  return { inv, ok: false };
}

/** Single block-Jacobi coarse apply (C0): z = Dblock^-1 r. No iteration. */
export function applyCoarseBlockJacobi(
  values: CoarseValues,
  coarseDiag: ArrayLike<number>,
  r: ArrayLike<number>,
  out: Float64Array,
): Float64Array {
  const D = values.pattern.domains;
  out.fill(0);
  for (let d = 0; d < D; d++) {
    const dp = blockPos(values.pattern, d, d);
    const B = values.blocks.subarray(dp * 9, dp * 9 + 9);
    // On-the-fly symmetrization mirrors the GPU read path exactly.
    const { inv } = invertSym33(
      B[0], 0.5 * (B[1] + B[3]), 0.5 * (B[2] + B[6]),
      B[4], 0.5 * (B[5] + B[7]), B[8],
      coarseDiag[d * 3], coarseDiag[d * 3 + 1], coarseDiag[d * 3 + 2],
    );
    for (let a = 0; a < 3; a++) {
      out[d * 3 + a] =
        inv[a * 3] * r[d * 3] + inv[a * 3 + 1] * r[d * 3 + 1] + inv[a * 3 + 2] * r[d * 3 + 2];
    }
  }
  return out;
}

export interface CoarsePcgResult {
  x: Float64Array;
  iters: number;
  residual: number;
}

/**
 * Fixed-K coarse PCG mirror of the planned GPU inner loop: block-Jacobi
 * (per-row diagonal-block inverse via invertSym33, fallback = true-Ac
 * diagonal), pAp<=tol breakdown latch (keep current iterate — fewer-iters
 * semantics, no descent fallback inside the coarse solve).
 */
export function coarsePcg(
  values: CoarseValues,
  coarseDiag: ArrayLike<number>,
  b: Float64Array,
  maxIters: number,
  tol = 1e-3,
): CoarsePcgResult {
  const n = b.length;
  const x = new Float64Array(n);
  const r = Float64Array.from(b);
  const z = new Float64Array(n);
  const p = new Float64Array(n);
  const Ap = new Float64Array(n);
  const D = values.pattern.domains;
  const applyPre = (rr: Float64Array, zz: Float64Array): void => {
    for (let d = 0; d < D; d++) {
      const dp = blockPos(values.pattern, d, d);
      const B = values.blocks.subarray(dp * 9, dp * 9 + 9);
      // On-the-fly symmetrization mirrors the GPU c_update_z read path.
      const { inv } = invertSym33(
        B[0], 0.5 * (B[1] + B[3]), 0.5 * (B[2] + B[6]),
        B[4], 0.5 * (B[5] + B[7]), B[8],
        coarseDiag[d * 3], coarseDiag[d * 3 + 1], coarseDiag[d * 3 + 2],
      );
      for (let a = 0; a < 3; a++) {
        zz[d * 3 + a] = inv[a * 3] * rr[d * 3] + inv[a * 3 + 1] * rr[d * 3 + 1] + inv[a * 3 + 2] * rr[d * 3 + 2];
      }
    }
  };
  applyPre(r, z);
  p.set(z);
  let rz = 0;
  for (let i = 0; i < n; i++) rz += r[i] * z[i];
  const bNorm = Math.sqrt(b.reduce((s, v) => s + v * v, 0));
  const absTol = Math.max(tol * Math.max(bNorm, 1e-12), 1e-12);
  let residual = Math.sqrt(r.reduce((s, v) => s + v * v, 0));
  if (residual < absTol) return { x, iters: 0, residual };
  const pApTol = 1e-30;
  for (let k = 0; k < maxIters; k++) {
    coarseSpmv(values, p, Ap);
    let pAp = 0;
    for (let i = 0; i < n; i++) pAp += p[i] * Ap[i];
    if (!(pAp > pApTol)) return { x, iters: k, residual };
    const alpha = rz / pAp;
    for (let i = 0; i < n; i++) {
      x[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
    }
    residual = Math.sqrt(r.reduce((s, v) => s + v * v, 0));
    if (residual < absTol) return { x, iters: k + 1, residual };
    applyPre(r, z);
    let rzNew = 0;
    for (let i = 0; i < n; i++) rzNew += r[i] * z[i];
    const beta = rzNew / rz;
    for (let i = 0; i < n; i++) p[i] = z[i] + beta * p[i];
    rz = rzNew;
  }
  return { x, iters: maxIters, residual };
}

/** Bilinear symmetry probe: u^T(Ac v) - v^T(Ac u) (exact symmetry => ~1e-12). */
export function bilinearAsymmetry(
  values: CoarseValues,
  u: ArrayLike<number>,
  v: ArrayLike<number>,
): { uv: number; vu: number; diff: number } {
  const n = values.pattern.domains * 3;
  const Au = new Float64Array(n);
  const Av = new Float64Array(n);
  coarseSpmv(values, Float64Array.from(u), Au);
  coarseSpmv(values, Float64Array.from(v), Av);
  let uv = 0, vu = 0;
  for (let i = 0; i < n; i++) {
    uv += u[i] * Av[i];
    vu += v[i] * Au[i];
  }
  return { uv, vu, diff: uv - vu };
}
