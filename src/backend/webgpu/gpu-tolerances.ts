// CPU/GPU comparison tolerances + FP32 numerical mirror (Phase 2 §4, §26).
//
// GPU baseline is FP32/vec4f; the CPU reference is Float64. Bit identity is
// NOT expected — accuracy is measured against the CPU reference:
//
//   position abs <= 1e-4 m, position rel <= 1e-3, energy rel <= 2e-3
//
// `fp32MembraneGradient` re-evaluates the CPU membrane formula with every
// intermediate rounded via Math.fround, predicting the GPU's FP32 error
// without requiring a GPU device — so CI without WebGPU still exercises the
// tolerance logic and the shear-convention port (S01 = 2*G*E01).

import type { ClothMeshData } from "../../mesh/mesh.js";
import type { ClothMaterial } from "../../physics/types.js";

export const GPU_POS_ABS_TOL = 1e-4;
export const GPU_POS_REL_TOL = 1e-3;
export const GPU_ENERGY_REL_TOL = 2e-3;
export const GPU_GRAD_DOT_TOL = 2e-3;

export interface CpuGpuComparison {
  maxAbsPosErr: number;
  maxRelPosErr: number;
  relEnergyErr: number;
  posPass: boolean;
  energyPass: boolean;
}

/** Compare CPU Float64 positions vs GPU (or FP32-mirror) positions. */
export function comparePositions(
  cpu: ArrayLike<number>,
  gpu: ArrayLike<number>,
  absTol = GPU_POS_ABS_TOL,
  relTol = GPU_POS_REL_TOL,
): { maxAbs: number; maxRel: number; pass: boolean } {
  let maxAbs = 0;
  let maxRel = 0;
  const n = Math.min(cpu.length, gpu.length);
  for (let i = 0; i < n; i++) {
    const a = cpu[i], b = gpu[i];
    const abs = Math.abs(a - b);
    if (abs > maxAbs) maxAbs = abs;
    const denom = Math.max(Math.abs(a), 1e-9);
    const rel = abs / denom;
    if (rel > maxRel) maxRel = rel;
  }
  return { maxAbs, maxRel, pass: maxAbs <= absTol || maxRel <= relTol };
}

export function compareEnergy(cpuE: number, gpuE: number, relTol = GPU_ENERGY_REL_TOL): { rel: number; pass: boolean } {
  const rel = Math.abs(cpuE - gpuE) / Math.max(Math.abs(cpuE), 1e-12);
  return { rel, pass: rel <= relTol };
}

/** Directional-derivative check: d/dh E(x + h p)|0 == g . p (relative tol). */
export function directionalCheck(
  energy: (x: Float64Array) => number,
  grad: Float64Array,
  x: Float64Array,
  p: Float64Array,
  h = 1e-7,
): { analytic: number; fd: number; relErr: number; pass: boolean } {
  let analytic = 0;
  for (let i = 0; i < grad.length; i++) analytic += grad[i] * p[i];
  const xp = new Float64Array(x.length);
  const xm = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) { xp[i] = x[i] + h * p[i]; xm[i] = x[i] - h * p[i]; }
  const fd = (energy(xp) - energy(xm)) / (2 * h);
  const relErr = Math.abs(fd - analytic) / Math.max(Math.abs(analytic), 1e-12);
  return { analytic, fd, relErr, pass: relErr <= GPU_GRAD_DOT_TOL };
}

const f = (v: number): number => Math.fround(v);

/**
 * FP32 mirror of triangleEnergyGradient (membrane.ts), every op rounded.
 * Structural port of the WGSL membrane-gradient.wgsl kernel — validates the
 * ported formula (including the S01 = 2*G*E01 convention) on CPU-only CI.
 */
export function fp32TriangleEnergyGradient(
  x0: ArrayLike<number>, x1: ArrayLike<number>, x2: ArrayLike<number>,
  inv: ArrayLike<number>, area: number, mat: ClothMaterial,
): { energy: number; grad: Float32Array } {
  const a = f(inv[0]), b = f(inv[1]), c = f(inv[2]), d = f(inv[3]);
  const e1x = f(f(x1[0]) - f(x0[0])), e1y = f(f(x1[1]) - f(x0[1])), e1z = f(f(x1[2]) - f(x0[2]));
  const e2x = f(f(x2[0]) - f(x0[0])), e2y = f(f(x2[1]) - f(x0[1])), e2z = f(f(x2[2]) - f(x0[2]));
  const F00 = f(f(a * e1x) + f(c * e2x)), F10 = f(f(a * e1y) + f(c * e2y)), F20 = f(f(a * e1z) + f(c * e2z));
  const F01 = f(f(b * e1x) + f(d * e2x)), F11 = f(f(b * e1y) + f(d * e2y)), F21 = f(f(b * e1z) + f(d * e2z));
  const C00 = f(f(f(F00 * F00) + f(F10 * F10)) + f(F20 * F20));
  const C11 = f(f(f(F01 * F01) + f(F11 * F11)) + f(F21 * F21));
  const C01 = f(f(f(F00 * F01) + f(F10 * F11)) + f(F20 * F21));
  const E00 = f(0.5 * f(C00 - 1)), E11 = f(0.5 * f(C11 - 1)), E01 = f(0.5 * C01);
  const C00k = mat.stretchWarp, C11k = mat.stretchWeft, C01k = mat.stretchCoupling, G = mat.shear;
  const psi = f(f(f(0.5 * C00k * E00 * E00) + f(0.5 * C11k * E11 * E11)) + f(f(C01k * E00 * E11) + f(2 * G * E01 * E01)));
  const s = f(mat.thickness * area);
  const energy = f(s * psi);
  // CRITICAL convention: S01 = 2*G*E01 (matches CPU + WGSL).
  const S00 = f(f(C00k * E00) + f(C01k * E11));
  const S11 = f(f(C11k * E11) + f(C01k * E00));
  const S01 = f(2 * G * E01);
  const P00 = f(f(F00 * S00) + f(F01 * S01)), P10 = f(f(F10 * S00) + f(F11 * S01)), P20 = f(f(F20 * S00) + f(F21 * S01));
  const P01 = f(f(F00 * S01) + f(F01 * S11)), P11 = f(f(F10 * S01) + f(F11 * S11)), P21 = f(f(F20 * S01) + f(F21 * S11));
  const grad = new Float32Array(9);
  const P0 = [P00, P10, P20], P1 = [P01, P11, P21];
  for (let i = 0; i < 3; i++) {
    const g1 = f(s * f(f(P0[i] * a) + f(P1[i] * b)));
    const g2 = f(s * f(f(P0[i] * c) + f(P1[i] * d)));
    const g0 = f(-(g1 + g2));
    grad[i] = g0; grad[3 + i] = g1; grad[6 + i] = g2;
  }
  return { energy, grad };
}

/** Full-mesh FP32 membrane gradient mirror (membrane-only, bending excluded). */
export function fp32MembraneMesh(
  x: ArrayLike<number>, mesh: ClothMeshData, mat: ClothMaterial,
): { energy: number; grad: Float32Array } {
  const grad = new Float32Array(mesh.count * 3);
  let energy = 0;
  for (let t = 0; t < mesh.triCount; t++) {
    const i0 = mesh.indices[t * 3], i1 = mesh.indices[t * 3 + 1], i2 = mesh.indices[t * 3 + 2];
    const r = fp32TriangleEnergyGradient(
      [x[i0 * 3], x[i0 * 3 + 1], x[i0 * 3 + 2]],
      [x[i1 * 3], x[i1 * 3 + 1], x[i1 * 3 + 2]],
      [x[i2 * 3], x[i2 * 3 + 1], x[i2 * 3 + 2]],
      [mesh.invDm[t * 4], mesh.invDm[t * 4 + 1], mesh.invDm[t * 4 + 2], mesh.invDm[t * 4 + 3]],
      mesh.areas[t], mat,
    );
    energy += r.energy;
    const vs = [i0, i1, i2];
    for (let v = 0; v < 3; v++) for (let k = 0; k < 3; k++) grad[vs[v] * 3 + k] += r.grad[v * 3 + k];
  }
  return { energy, grad };
}
