// Anisotropic membrane (orthotropic StVK) per triangle.
// F = Ds * invDm (3x2). C = F^T F. E = 0.5*(C-I).
// psi = 0.5*C00*E00^2 + 0.5*C11*E11^2 + C01*E00*E11 + 2*G*E01^2
// W = thickness * area * psi. Analytic gradient w.r.t. x0,x1,x2 (9 dofs).

import type { ClothMaterial } from "./types.js";

export interface TriGradient {
  energy: number;
  /** length-9 gradient in vertex order (x0,y0,z0,x1,...) */
  grad: Float64Array;
  E00: number;
  E11: number;
  E01: number;
}

export function triangleEnergyGradient(
  x0: ArrayLike<number>, x1: ArrayLike<number>, x2: ArrayLike<number>,
  inv: ArrayLike<number>, // [a,b,c,d]
  area: number,
  mat: ClothMaterial,
): TriGradient {
  const a = inv[0], b = inv[1], c = inv[2], d = inv[3];
  // Ds columns: e1 = x1-x0, e2 = x2-x0
  const e1x = x1[0] - x0[0], e1y = x1[1] - x0[1], e1z = x1[2] - x0[2];
  const e2x = x2[0] - x0[0], e2y = x2[1] - x0[1], e2z = x2[2] - x0[2];
  // F columns
  const F00 = a * e1x + c * e2x, F10 = a * e1y + c * e2y, F20 = a * e1z + c * e2z;
  const F01 = b * e1x + d * e2x, F11 = b * e1y + d * e2y, F21 = b * e1z + d * e2z;
  const C00 = F00 * F00 + F10 * F10 + F20 * F20;
  const C11 = F01 * F01 + F11 * F11 + F21 * F21;
  const C01 = F00 * F01 + F10 * F11 + F20 * F21;
  const E00 = 0.5 * (C00 - 1);
  const E11 = 0.5 * (C11 - 1);
  const E01 = 0.5 * C01;

  const C00k = mat.stretchWarp, C11k = mat.stretchWeft, C01k = mat.stretchCoupling, G = mat.shear;
  const psi = 0.5 * C00k * E00 * E00 + 0.5 * C11k * E11 * E11 + C01k * E00 * E11 + 2 * G * E01 * E01;
  const energy = mat.thickness * area * psi;

  // d psi / dE
  // S = 2nd Piola-Kirchhoff stress: S = d psi / dE as symmetric tensor.
  // Note: S01 = 2*G*E01 (not 4*G*E01): tr(E^2) = E00^2+E11^2+2*E01^2
  // gives S = 2*G*E off-diagonal. The naive scalar derivative 4*G*E01
  // double-counts because P = F*S already couples both F columns.
  const dPsi_dE00 = C00k * E00 + C01k * E11;
  const dPsi_dE11 = C11k * E11 + C01k * E00;
  const dPsi_dE01 = 2 * G * E01;
  // dE/dC = 0.5 on diagonal; chain to F:
  // S = second Piola-Kirchhoff stress (2x2): S = d psi / dE
  // P = F * S (3x2 first Piola)
  // dW/dF_ij = thickness*area * P_ij
  const S00 = dPsi_dE00, S11 = dPsi_dE11, S01 = dPsi_dE01;
  const P00 = F00 * S00 + F01 * S01;
  const P10 = F10 * S00 + F11 * S01;
  const P20 = F20 * S00 + F21 * S01;
  const P01 = F00 * S01 + F01 * S11;
  const P11 = F10 * S01 + F11 * S11;
  const P21 = F20 * S01 + F21 * S11;
  const s = mat.thickness * area;

  // dF/d x: F_i0 = a*(x1i-x0i)+c*(x2i-x0i); F_i1 = b*(x1i-x0i)+d*(x2i-x0i)
  // grad[x1i] = s*(P_i0*a + P_i1*b), grad[x2i] = s*(P_i0*c + P_i1*d), grad[x0i] = -(sum)
  const grad = new Float64Array(9);
  for (let i = 0; i < 3; i++) {
    const Pi0 = i === 0 ? P00 : i === 1 ? P10 : P20;
    const Pi1 = i === 0 ? P01 : i === 1 ? P11 : P21;
    const g1 = s * (Pi0 * a + Pi1 * b);
    const g2 = s * (Pi0 * c + Pi1 * d);
    const g0 = -(g1 + g2);
    grad[0 * 3 + i] = g0;
    grad[1 * 3 + i] = g1;
    grad[2 * 3 + i] = g2;
  }
  // NOTE: grad layout above is per-vertex blocks of 3? We wrote [v][i]; flatten as
  // [x0,y0,z0, x1,y1,z1, x2,y2,z2] i.e. grad[v*3+i]. Fix indexing:
  const out = new Float64Array(9);
  for (let v = 0; v < 3; v++) for (let i = 0; i < 3; i++) out[v * 3 + i] = grad[v * 3 + i];
  return { energy, grad: out, E00, E11, E01 };
}
