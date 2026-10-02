// G5B block-Jacobi preconditioner (CPU reference + GPU-factor mirror).
//
// Per-vertex 3x3 blocks Hvv of the membrane Hessian (membrane-blocks.ts),
// embedded with the Jacobi diagonal (M/h^2 + beta + contact curvature):
//   B_v = Hvv_membrane(v) + diagEmbed(diag[v*3..v*3+2])
// factored once per Newton iteration (not per PCG iteration).
//
// Factorization mirrors shaders/block-jacobi.wgsl `bj_build_factor`:
// Sylvester PD check (relative to diagonal scale); success -> adjugate/det,
// fail -> diagonal Jacobi fallback pre-baked (no branch in the apply path).
// Bending stays out (inexact Newton, same as the HVP operator).
// Pins are handled by the caller's filter (apply preserves zeros structurally
// only when r is already filtered; pcg.ts filters z after apply like the GPU).

import { evalMembraneBlocks } from "../physics/membrane-blocks.js";
import type { ClothMeshData } from "../mesh/mesh.js";
import type { ClothMaterial } from "../physics/types.js";

export interface BlockFactors {
  /** n*9 row-major 3x3 inverse (or Jacobi-fallback diagonal) per vertex */
  inv: Float64Array;
  /** n flags: 1 = Cholesky/adjugate ok, 0 = Jacobi fallback */
  flag: Uint8Array;
}

/**
 * Build per-vertex block inverses. `diag` is the full Jacobi diagonal
 * (length n*3: M/h^2 + beta + contact), same buffer the GPU factor reads.
 */
export function buildBlockFactors(
  x: ArrayLike<number>,
  mesh: ClothMeshData,
  mat: ClothMaterial,
  diag: ArrayLike<number>,
): BlockFactors {
  const n = mesh.count;
  const blocks = evalMembraneBlocks(x, mesh, mat);
  const inv = new Float64Array(n * 9);
  const flag = new Uint8Array(n);
  for (let v = 0; v < n; v++) {
    const h00 = blocks[v * 9];     const h01 = blocks[v * 9 + 1]; const h02 = blocks[v * 9 + 2];
    const h10 = blocks[v * 9 + 3]; const h11 = blocks[v * 9 + 4]; const h12 = blocks[v * 9 + 5];
    const h20 = blocks[v * 9 + 6]; const h21 = blocks[v * 9 + 7]; const h22 = blocks[v * 9 + 8];
    const d0 = diag[v * 3]; const d1 = diag[v * 3 + 1]; const d2 = diag[v * 3 + 2];
    const B00 = h00 + d0; const B11 = h11 + d1; const B22 = h22 + d2;
    // Off-diagonals: symmetrize (analytic blocks are symmetric to 1e-14;
    // the shader accumulates only the upper triangle and mirrors implicitly
    // via its entry formula — same values up to fp order).
    const B01 = 0.5 * (h01 + h10);
    const B02 = 0.5 * (h02 + h20);
    const B12 = 0.5 * (h12 + h21);
    // Sylvester PD check (mirror of bj_build_factor, float64 here).
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
    const base = v * 9;
    if (ok) {
      const s = 1 / det;
      inv[base]     = (B11 * B22 - B12 * B12) * s;
      inv[base + 1] = (B02 * B12 - B01 * B22) * s;
      inv[base + 2] = (B01 * B12 - B11 * B02) * s;
      inv[base + 3] = (B02 * B12 - B01 * B22) * s;
      inv[base + 4] = (B00 * B22 - B02 * B02) * s;
      inv[base + 5] = (B01 * B02 - B00 * B12) * s;
      inv[base + 6] = (B01 * B12 - B11 * B02) * s;
      inv[base + 7] = (B01 * B02 - B00 * B12) * s;
      inv[base + 8] = (B00 * B11 - B01 * B01) * s;
      flag[v] = 1;
    } else {
      const i0 = d0 > 1e-12 ? 1 / d0 : 0;
      const i1 = d1 > 1e-12 ? 1 / d1 : 0;
      const i2 = d2 > 1e-12 ? 1 / d2 : 0;
      inv[base] = i0; inv[base + 1] = 0;  inv[base + 2] = 0;
      inv[base + 3] = 0;  inv[base + 4] = i1; inv[base + 5] = 0;
      inv[base + 6] = 0;  inv[base + 7] = 0;  inv[base + 8] = i2;
      flag[v] = 0;
    }
  }
  return { inv, flag };
}

/** z = M^{-1} r with per-vertex 3x3 inverses (pins: caller filters z after). */
export function applyBlockFactors(
  factors: BlockFactors,
  r: ArrayLike<number>,
  out: Float64Array,
  vertexCount: number,
): Float64Array {
  for (let v = 0; v < vertexCount; v++) {
    const r0 = r[v * 3]; const r1 = r[v * 3 + 1]; const r2 = r[v * 3 + 2];
    const b = v * 9;
    out[v * 3]     = factors.inv[b] * r0 + factors.inv[b + 1] * r1 + factors.inv[b + 2] * r2;
    out[v * 3 + 1] = factors.inv[b + 3] * r0 + factors.inv[b + 4] * r1 + factors.inv[b + 5] * r2;
    out[v * 3 + 2] = factors.inv[b + 6] * r0 + factors.inv[b + 7] * r1 + factors.inv[b + 8] * r2;
  }
  return out;
}
