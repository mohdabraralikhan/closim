// G5D two-level multilevel additive Schwarz (CPU reference).
//
// Aggregates = G5C Schwarz domains. Coarse space: constant vector per
// aggregate (3 dofs each), normalized restriction:
//   R[(d*3+a)][v*3+a] = 1/sqrt(|D_d|)  for v in domain d   (R R^T = I_c)
// Two correction flavors:
//   (1) exact coarse (headroom reference): dense Ac = R A P + Cholesky;
//   (2) single-sweep damped Jacobi (GPU mirror): z_c = omega * Dc^-1 r_c
//       with Dc = diag(R diag(H) P) — NO extra HVP, one restrict/prolongate
//       pair per apply, all inside the same encoder on device.
// Two-level apply: z = Schwarz1(r) + P C(R r). Both flavors are symmetric
// (Dc diagonal; exact via SPD Cholesky), hence PCG-safe. The exact PCG
// operator (HVP) is unchanged; pins are caller-filtered as usual.

import { triangleHessian9 } from "../physics/membrane-blocks.js";
import {
  buildSchwarzFactors, applySchwarz, type SchwarzDomains, type SchwarzFactors,
} from "./schwarz.js";
import type { ClothMeshData } from "../mesh/mesh.js";
import type { ClothMaterial } from "../physics/types.js";

/** Coarse restriction of a fine vector: rc[d*3+a] = sum_{v in d} r[v*3+a]/sqrt(|d|). */
export function restrictCoarse(
  r: ArrayLike<number>,
  domains: SchwarzDomains,
  out?: Float64Array,
): Float64Array {
  const D = domains.members.length;
  const rc = out ?? new Float64Array(D * 3);
  rc.fill(0);
  domains.members.forEach((mem, d) => {
    const s = 1 / Math.sqrt(mem.length);
    for (const v of mem) {
      rc[d * 3] += r[v * 3] * s;
      rc[d * 3 + 1] += r[v * 3 + 1] * s;
      rc[d * 3 + 2] += r[v * 3 + 2] * s;
    }
  });
  return rc;
}

/** Coarse prolongation-add: z[v*3+a] += c[d*3+a]/sqrt(|d|). */
export function prolongateAdd(
  c: ArrayLike<number>,
  domains: SchwarzDomains,
  z: Float64Array,
): void {
  domains.members.forEach((mem, d) => {
    const s = 1 / Math.sqrt(mem.length);
    for (const v of mem) {
      z[v * 3] += c[d * 3] * s;
      z[v * 3 + 1] += c[d * 3 + 1] * s;
      z[v * 3 + 2] += c[d * 3 + 2] * s;
    }
  });
}

/**
 * True coarse diagonal Dc = diag(R A P): intra-aggregate membrane trace plus
 * the Jacobi-diagonal spread. The membrane part is essential: without it Dc
 * underestimates the coarse curvature and the damped sweep overshoots (mas2
 * can test worse than schwarz1). The GPU mirror sums the assembled local
 * matrices (schwarzMat already holds membrane + embedded diag, symmetrized —
 * the component-a entry sum divided by |d| is exactly this).
 */
export function buildCoarseDiag(
  x: ArrayLike<number>,
  mesh: ClothMeshData,
  mat: ClothMaterial,
  diag: ArrayLike<number>,
  domains: SchwarzDomains,
): Float64Array {
  const D = domains.members.length;
  const dc = new Float64Array(D * 3);
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
        if (d1 !== d2) continue; // off-diagonal coarse blocks have zero diagonal
        const s = 1 / sizes[d1];
        for (let a = 0; a < 3; a++) dc[d1 * 3 + a] += s * h[(ci * 3 + a) * 9 + (cj * 3 + a)];
      }
    }
  }
  domains.members.forEach((mem, d) => {
    const s = 1 / mem.length;
    for (const v of mem) {
      for (let a = 0; a < 3; a++) dc[d * 3 + a] += s * diag[v * 3 + a];
    }
  });
  return dc;
}

export interface MasTwoLevel {
  fine: SchwarzFactors;
  coarseDiag: Float64Array;
  omega: number;
}

/** Production-mirror two-level factors (single-sweep damped-Jacobi coarse). */
export function buildMasTwoLevel(
  x: ArrayLike<number>,
  mesh: ClothMeshData,
  mat: ClothMaterial,
  diag: ArrayLike<number>,
  domains: SchwarzDomains,
  omega = 0.5,
): MasTwoLevel {
  return {
    fine: buildSchwarzFactors(x, mesh, mat, diag, domains),
    coarseDiag: buildCoarseDiag(x, mesh, mat, diag, domains),
    omega,
  };
}

/** z = Schwarz1(r) + omega * P Dc^-1 R r. `tmp` scratch (D*3) reused across calls. */
export function applyMasTwoLevel(
  mas: MasTwoLevel,
  r: ArrayLike<number>,
  out: Float64Array,
  vertexCount: number,
  tmp?: Float64Array,
): Float64Array {
  applySchwarz(mas.fine, r, out, vertexCount);
  const D = mas.fine.domains.members.length;
  const rc = tmp ?? new Float64Array(D * 3);
  restrictCoarse(r, mas.fine.domains, rc);
  for (let i = 0; i < D * 3; i++) {
    const dd = mas.coarseDiag[i];
    rc[i] = dd > 1e-12 ? (mas.omega * rc[i]) / dd : 0;
  }
  prolongateAdd(rc, mas.fine.domains, out);
  return out;
}

// ---- exact-coarse headroom reference (dense Ac, small meshes only) ----

export interface CoarseExact {
  domains: SchwarzDomains;
  /** (3D)^2 row-major dense Ac */
  ac: Float64Array;
  /** Cholesky L of Ac (lower) or Jacobi fallback diag-inverse */
  chol: Float64Array;
  flag: 0 | 1;
}

/** Dense Ac = R A P with A = membrane Hessian (+ diag embed). */
export function buildCoarseExact(
  x: ArrayLike<number>,
  mesh: ClothMeshData,
  mat: ClothMaterial,
  diag: ArrayLike<number>,
  domains: SchwarzDomains,
): CoarseExact {
  const D = domains.members.length;
  const dc = D * 3;
  const ac = new Float64Array(dc * dc);
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
        const s = 1 / Math.sqrt(sizes[d1] * sizes[d2]);
        for (let a = 0; a < 3; a++) {
          for (let b = 0; b < 3; b++) {
            ac[(d1 * 3 + a) * dc + (d2 * 3 + b)] += s * h[(ci * 3 + a) * 9 + (cj * 3 + b)];
          }
        }
      }
    }
  }
  // R diag(H) P: diagonal spread.
  domains.members.forEach((mem, d) => {
    const s = 1 / mem.length;
    for (const v of mem) {
      for (let a = 0; a < 3; a++) ac[(d * 3 + a) * dc + (d * 3 + a)] += s * diag[v * 3 + a];
    }
  });
  for (let i = 0; i < dc; i++) {
    for (let j = i + 1; j < dc; j++) {
      const s = 0.5 * (ac[i * dc + j] + ac[j * dc + i]);
      ac[i * dc + j] = s; ac[j * dc + i] = s;
    }
  }
  // Dense Cholesky with symmetric Jacobi fallback.
  const L = Float64Array.from(ac);
  let scale = 0;
  for (let i = 0; i < dc; i++) scale = Math.max(scale, Math.abs(L[i * dc + i]));
  const tiny = 1e-20 * (1 + scale);
  let ok = true;
  for (let k = 0; k < dc && ok; k++) {
    let sum = L[k * dc + k];
    for (let s = 0; s < k; s++) sum -= L[k * dc + s] * L[k * dc + s];
    if (!(sum > tiny)) { ok = false; break; }
    L[k * dc + k] = Math.sqrt(sum);
    for (let i = k + 1; i < dc; i++) {
      let s2 = L[i * dc + k];
      for (let s = 0; s < k; s++) s2 -= L[i * dc + s] * L[k * dc + s];
      L[i * dc + k] = s2 / L[k * dc + k];
    }
  }
  if (ok) {
    for (let i = 0; i < dc; i++) for (let j = i + 1; j < dc; j++) L[i * dc + j] = 0;
    return { domains, ac, chol: L, flag: 1 };
  }
  const J = new Float64Array(dc * dc);
  for (let i = 0; i < dc; i++) J[i * dc + i] = ac[i * dc + i] > 1e-12 ? 1 / ac[i * dc + i] : 0;
  return { domains, ac, chol: J, flag: 0 };
}

/** Solve the coarse system (Cholesky or fallback). */
export function solveCoarseExact(coarse: CoarseExact, rhs: ArrayLike<number>, out: Float64Array): void {
  const dc = coarse.domains.members.length * 3;
  if (coarse.flag === 1) {
    const y = new Float64Array(dc);
    for (let i = 0; i < dc; i++) {
      let s = rhs[i];
      for (let j = 0; j < i; j++) s -= coarse.chol[i * dc + j] * y[j];
      y[i] = s / coarse.chol[i * dc + i];
    }
    for (let i = dc - 1; i >= 0; i--) {
      let s = y[i];
      for (let j = i + 1; j < dc; j++) s -= coarse.chol[j * dc + i] * out[j];
      out[i] = s / coarse.chol[i * dc + i];
    }
  } else {
    for (let i = 0; i < dc; i++) {
      let s = 0;
      for (let j = 0; j < dc; j++) s += coarse.chol[i * dc + j] * rhs[j];
      out[i] = s;
    }
  }
}
