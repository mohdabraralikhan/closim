// G5B analytic per-triangle membrane Hessian (9x9, row-major, dofs ordered
// x0,y0,z0,x1,y1,z1,x2,y2,z2). Closed form from the HVP chain (membrane-hvp.ts):
// F is LINEAR in x (F_ij = sum_A M[A][i][j] x_A with invDm/corner coeffs),
// S = D:E with CONSTANT moduli D, so one more product rule gives H exactly:
//   g_A = s * sum_ij M[A][i][j] P_ij
//   H_AB = s * sum_ij M[A][i][j] dP_ij/dx_B
//   dP_ij/dx_B = sum_k (M[B][i][k] S_kj + F_ik sum_lm D_kjlm dE_lm/dx_B)
//   dE_lm/dx_B = 1/2 sum_p (M[B][p][l] F_pm + F_pl M[B][p][m])
// Voigt moduli on (00,11,01): D_0000=C00k, D_1111=C11k, D_0011=D_1100=C01k,
// D_0101=D_0110=D_1001=D_1010=G (each G: S_01 = 2*G*E_01, the shear convention).
// Verified against the FD oracle (tests) — including the convention.

import type { ClothMeshData } from "../mesh/mesh.js";
import type { ClothMaterial } from "./types.js";

export interface TriHessian {
  /** 81 entries row-major */
  h: Float64Array;
}

export function triangleHessian9(
  x0: ArrayLike<number>, x1: ArrayLike<number>, x2: ArrayLike<number>,
  inv: ArrayLike<number>, // [a,b,c,d]
  area: number,
  mat: ClothMaterial,
): TriHessian {
  const a = inv[0], b = inv[1], c = inv[2], d = inv[3];
  const X = [x0, x1, x2];
  // M[A][i][j] = delta_ai * K[v][j], K = corner F-coeffs
  const K = [[-(a + c), -(b + d)], [a, b], [c, d]];
  const F = [[0, 0], [0, 0], [0, 0]];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 2; j++) {
      for (let v = 0; v < 3; v++) F[i][j] += K[v][j] * X[v][i];
    }
  }
  const C = [[0, 0], [0, 0]];
  for (let l = 0; l < 2; l++) {
    for (let m = 0; m < 2; m++) {
      for (let i = 0; i < 3; i++) C[l][m] += F[i][l] * F[i][m];
    }
  }
  const E00 = 0.5 * (C[0][0] - 1), E11 = 0.5 * (C[1][1] - 1), E01 = 0.5 * C[0][1];
  const C00k = mat.stretchWarp, C11k = mat.stretchWeft, C01k = mat.stretchCoupling, G = mat.shear;
  const S00 = C00k * E00 + C01k * E11;
  const S11 = C11k * E11 + C01k * E00;
  const S01 = 2 * G * E01;
  const S = [[S00, S01], [S01, S11]];
  // D moduli as full 2x2x2x2 (only the Voigt slots nonzero)
  const D = [[[[0, 0], [0, 0]], [[0, 0], [0, 0]]], [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]];
  D[0][0][0][0] = C00k;
  D[1][1][1][1] = C11k;
  D[0][0][1][1] = C01k; D[1][1][0][0] = C01k;
  D[0][1][0][1] = G; D[0][1][1][0] = G; D[1][0][0][1] = G; D[1][0][1][0] = G;
  const s = mat.thickness * area;
  // M as [9][3][2]
  const M: number[][][] = [];
  for (let v = 0; v < 3; v++) {
    for (let ra = 0; ra < 3; ra++) {
      const row: number[][] = [[0, 0], [0, 0], [0, 0]];
      for (let j = 0; j < 2; j++) row[ra][j] = K[v][j];
      M.push(row);
    }
  }
  const h = new Float64Array(81);
  for (let A = 0; A < 9; A++) {
    for (let B = 0; B < 9; B++) {
      let sum = 0;
      for (let i = 0; i < 3; i++) {
        for (let j = 0; j < 2; j++) {
          const mA = M[A][i][j];
          if (mA === 0) continue;
          // dP_ij/dx_B
          let dP = 0;
          for (let k = 0; k < 2; k++) {
            dP += M[B][i][k] * S[k][j];
            let dE = 0;
            for (let l = 0; l < 2; l++) {
              for (let m2 = 0; m2 < 2; m2++) {
                const dd = D[k][j][l][m2];
                if (dd === 0) continue;
                let dElm = 0;
                for (let p = 0; p < 3; p++) {
                  dElm += M[B][p][l] * F[p][m2] + F[p][l] * M[B][p][m2];
                }
                dE += dd * 0.5 * dElm;
              }
            }
            dP += F[i][k] * dE;
          }
          sum += mA * dP;
        }
      }
      h[A * 9 + B] = s * sum;
    }
  }
  return { h };
}

/** Membrane diagonal 3x3 blocks assembled over the mesh (row-major n*9).
 *  Inertia/contact additions happen at solve level, not here. */
export function evalMembraneBlocks(
  x: ArrayLike<number>,
  mesh: ClothMeshData,
  mat: ClothMaterial,
  out?: Float64Array,
): Float64Array {
  const n = mesh.count;
  const blocks = out ?? new Float64Array(n * 9);
  if (out) blocks.fill(0);
  const m = mesh.triCount;
  const idx = mesh.indices;
  for (let t = 0; t < m; t++) {
    const i0 = idx[t * 3], i1 = idx[t * 3 + 1], i2 = idx[t * 3 + 2];
    const x0 = [x[i0 * 3], x[i0 * 3 + 1], x[i0 * 3 + 2]];
    const x1 = [x[i1 * 3], x[i1 * 3 + 1], x[i1 * 3 + 2]];
    const x2 = [x[i2 * 3], x[i2 * 3 + 1], x[i2 * 3 + 2]];
    const inv = [mesh.invDm[t * 4], mesh.invDm[t * 4 + 1], mesh.invDm[t * 4 + 2], mesh.invDm[t * 4 + 3]];
    const { h } = triangleHessian9(x0, x1, x2, inv, mesh.areas[t], mat);
    const vs = [i0, i1, i2];
    for (let vc = 0; vc < 3; vc++) {
      for (let a = 0; a < 3; a++) {
        for (let b = 0; b < 3; b++) {
          blocks[(vs[vc] * 9) + a * 3 + b] += h[(vc * 3 + a) * 9 + (vc * 3 + b)];
        }
      }
    }
  }
  return blocks;
}
