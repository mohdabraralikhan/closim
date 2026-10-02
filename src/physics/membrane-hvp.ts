// Analytic membrane HVP (G4A): y = H_membrane(x) * v WITHOUT finite differences.
// Line-by-line directional derivative of triangleEnergyGradient (membrane.ts):
//   v -> dDs -> dF -> dE -> dS -> dP -> dgrad
// The tangent moduli differentiate the implemented stress lines exactly:
//   dS00 = C00*dE00 + C01*dE11, dS11 = C11*dE11 + C01*dE00, dS01 = 2*G*dE01
// preserving the S01 = 2*G*E01 shear convention (NOT 4*G*E01).
// Membrane-only (bending excluded from Hessian, kept in gradient) — same
// inexact-Newton approximation as the FD oracle membraneHvp in fem.ts.

import type { ClothMeshData } from "../mesh/mesh.js";
import type { ClothMaterial } from "./types.js";

export interface TriHvp {
  /** length-9 directional gradient derivative in vertex order (v0,v1,v2 blocks) */
  out: Float64Array;
}

export function triangleHvp(
  x0: ArrayLike<number>, x1: ArrayLike<number>, x2: ArrayLike<number>,
  v0: ArrayLike<number>, v1: ArrayLike<number>, v2: ArrayLike<number>,
  inv: ArrayLike<number>, // [a,b,c,d]
  area: number,
  mat: ClothMaterial,
): TriHvp {
  const a = inv[0], b = inv[1], c = inv[2], d = inv[3];
  // Ds columns: e1 = x1-x0, e2 = x2-x0; directional: w1 = v1-v0, w2 = v2-v0
  const e1x = x1[0] - x0[0], e1y = x1[1] - x0[1], e1z = x1[2] - x0[2];
  const e2x = x2[0] - x0[0], e2y = x2[1] - x0[1], e2z = x2[2] - x0[2];
  const w1x = v1[0] - v0[0], w1y = v1[1] - v0[1], w1z = v1[2] - v0[2];
  const w2x = v2[0] - v0[0], w2y = v2[1] - v0[1], w2z = v2[2] - v0[2];
  // F columns (same as gradient)
  const F00 = a * e1x + c * e2x, F10 = a * e1y + c * e2y, F20 = a * e1z + c * e2z;
  const F01 = b * e1x + d * e2x, F11 = b * e1y + d * e2y, F21 = b * e1z + d * e2z;
  // dF columns (same stencil on the direction)
  const dF00 = a * w1x + c * w2x, dF10 = a * w1y + c * w2y, dF20 = a * w1z + c * w2z;
  const dF01 = b * w1x + d * w2x, dF11 = b * w1y + d * w2y, dF21 = b * w1z + d * w2z;
  const C00k = mat.stretchWarp, C11k = mat.stretchWeft, C01k = mat.stretchCoupling, G = mat.shear;
  const E00 = 0.5 * (F00 * F00 + F10 * F10 + F20 * F20 - 1);
  const E11 = 0.5 * (F01 * F01 + F11 * F11 + F21 * F21 - 1);
  const E01 = 0.5 * (F00 * F01 + F10 * F11 + F20 * F21);
  // dC = D_v(F^T F): dC00/dC11 pick up a factor 2, dC01 is the product rule
  const dC00 = 2 * (F00 * dF00 + F10 * dF10 + F20 * dF20);
  const dC11 = 2 * (F01 * dF01 + F11 * dF11 + F21 * dF21);
  const dC01 = dF00 * F01 + F00 * dF01 + dF10 * F11 + F10 * dF11 + dF20 * F21 + F20 * dF21;
  const dE00 = 0.5 * dC00, dE11 = 0.5 * dC11, dE01 = 0.5 * dC01;
  // Stress (same as gradient) + tangent moduli (exact derivative of S lines)
  const S00 = C00k * E00 + C01k * E11;
  const S11 = C11k * E11 + C01k * E00;
  const S01 = 2 * G * E01;
  const dS00 = C00k * dE00 + C01k * dE11;
  const dS11 = C11k * dE11 + C01k * dE00;
  const dS01 = 2 * G * dE01;
  // P = F * S (same) + dP (product rule on both factors)
  const dP00 = dF00 * S00 + F00 * dS00 + dF01 * S01 + F01 * dS01;
  const dP10 = dF10 * S00 + F10 * dS00 + dF11 * S01 + F11 * dS01;
  const dP20 = dF20 * S00 + F20 * dS00 + dF21 * S01 + F21 * dS01;
  const dP01 = dF00 * S01 + F00 * dS01 + dF01 * S11 + F01 * dS11;
  const dP11 = dF10 * S01 + F10 * dS01 + dF11 * S11 + F11 * dS11;
  const dP21 = dF20 * S01 + F20 * dS01 + dF21 * S11 + F21 * dS11;
  const s = mat.thickness * area;

  // dg1/dg2 differentiate the g1/g2 lines; dg0 = -(dg1+dg2) (translation sum)
  const out = new Float64Array(9);
  const P = [dP00, dP10, dP20, dP01, dP11, dP21];
  for (let i = 0; i < 3; i++) {
    const Pi0 = P[i], Pi1 = P[3 + i];
    const g1 = s * (Pi0 * a + Pi1 * b);
    const g2 = s * (Pi0 * c + Pi1 * d);
    const g0 = -(g1 + g2);
    out[0 * 3 + i] = g0;
    out[1 * 3 + i] = g1;
    out[2 * 3 + i] = g2;
  }
  return { out };
}

/** Membrane-only HVP over the mesh (membrane part of the Hessian-vector product).
 *  Same direct assembly as evalMembrane (G4B replaces both with a CSR gather);
 *  bending excluded — matches the FD oracle membraneHvp in fem.ts. */
export function evalMembraneHvp(
  x: ArrayLike<number>,
  p: ArrayLike<number>,
  mesh: ClothMeshData,
  mat: ClothMaterial,
  outHvp?: Float64Array,
): Float64Array {
  const n = mesh.count;
  const out = outHvp ?? new Float64Array(n * 3);
  if (outHvp) out.fill(0);
  const m = mesh.triCount;
  const idx = mesh.indices;
  for (let t = 0; t < m; t++) {
    const i0 = idx[t * 3], i1 = idx[t * 3 + 1], i2 = idx[t * 3 + 2];
    const x0 = [x[i0 * 3], x[i0 * 3 + 1], x[i0 * 3 + 2]];
    const x1 = [x[i1 * 3], x[i1 * 3 + 1], x[i1 * 3 + 2]];
    const x2 = [x[i2 * 3], x[i2 * 3 + 1], x[i2 * 3 + 2]];
    const v0 = [p[i0 * 3], p[i0 * 3 + 1], p[i0 * 3 + 2]];
    const v1 = [p[i1 * 3], p[i1 * 3 + 1], p[i1 * 3 + 2]];
    const v2 = [p[i2 * 3], p[i2 * 3 + 1], p[i2 * 3 + 2]];
    const inv = [mesh.invDm[t * 4], mesh.invDm[t * 4 + 1], mesh.invDm[t * 4 + 2], mesh.invDm[t * 4 + 3]];
    const r = triangleHvp(x0, x1, x2, v0, v1, v2, inv, mesh.areas[t], mat);
    const vs = [i0, i1, i2];
    for (let v = 0; v < 3; v++) {
      for (let i = 0; i < 3; i++) out[vs[v] * 3 + i] += r.out[v * 3 + i];
    }
  }
  return out;
}
