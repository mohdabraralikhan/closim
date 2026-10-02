// Assembly: total internal energy + gradient over full cloth state.
// HVP via central differences of the analytic gradient (matrix-free, GPU-ready pattern).

import type { ClothMeshData } from "../mesh/mesh.js";
import type { ClothMaterial } from "./types.js";
import { triangleEnergyGradient } from "./membrane.js";
import { addHingeGradient, hingeEnergy } from "./bending.js";

export interface FemEval {
  energy: number;
  grad: Float64Array; // length 3n
  maxStrain: number;
}

export function evalInternal(
  x: ArrayLike<number>,
  mesh: ClothMeshData,
  mat: ClothMaterial,
  outGrad?: Float64Array,
): FemEval {
  const n = mesh.count;
  const grad = outGrad ?? new Float64Array(n * 3);
  if (outGrad) grad.fill(0);
  let energy = 0;
  let maxStrain = 0;

  const m = mesh.triCount;
  const idx = mesh.indices;
  for (let t = 0; t < m; t++) {
    const i0 = idx[t * 3], i1 = idx[t * 3 + 1], i2 = idx[t * 3 + 2];
    const x0 = [x[i0 * 3], x[i0 * 3 + 1], x[i0 * 3 + 2]];
    const x1 = [x[i1 * 3], x[i1 * 3 + 1], x[i1 * 3 + 2]];
    const x2 = [x[i2 * 3], x[i2 * 3 + 1], x[i2 * 3 + 2]];
    const inv = [mesh.invDm[t * 4], mesh.invDm[t * 4 + 1], mesh.invDm[t * 4 + 2], mesh.invDm[t * 4 + 3]];
    const r = triangleEnergyGradient(x0, x1, x2, inv, mesh.areas[t], mat);
    energy += r.energy;
    maxStrain = Math.max(maxStrain, Math.abs(r.E00), Math.abs(r.E11), Math.abs(r.E01));
    const vs = [i0, i1, i2];
    for (let v = 0; v < 3; v++) {
      for (let i = 0; i < 3; i++) grad[vs[v] * 3 + i] += r.grad[v * 3 + i];
    }
  }

  const kb = 0.5 * (mat.bendWarp + mat.bendWeft);
  if (kb > 0) {
    for (const h of mesh.hinges) {
      energy += addHingeGradient(
        x as Float32Array, grad,
        h.v0, h.v1, h.v2, h.v3, h.restAngle, h.edgeLen, h.areaSum, kb,
      );
    }
  }
  return { energy, grad, maxStrain };
}

export function internalEnergyOnly(x: ArrayLike<number>, mesh: ClothMeshData, mat: ClothMaterial): number {
  let e = 0;
  const m = mesh.triCount;
  const idx = mesh.indices;
  for (let t = 0; t < m; t++) {
    const i0 = idx[t * 3], i1 = idx[t * 3 + 1], i2 = idx[t * 3 + 2];
    const x0 = [x[i0 * 3], x[i0 * 3 + 1], x[i0 * 3 + 2]];
    const x1 = [x[i1 * 3], x[i1 * 3 + 1], x[i1 * 3 + 2]];
    const x2 = [x[i2 * 3], x[i2 * 3 + 1], x[i2 * 3 + 2]];
    const inv = [mesh.invDm[t * 4], mesh.invDm[t * 4 + 1], mesh.invDm[t * 4 + 2], mesh.invDm[t * 4 + 3]];
    e += triangleEnergyGradient(x0, x1, x2, inv, mesh.areas[t], mat).energy;
  }
  const kb = 0.5 * (mat.bendWarp + mat.bendWeft);
  if (kb > 0) {
    for (const h of mesh.hinges) {
      e += hingeEnergy(x, h.v0, h.v1, h.v2, h.v3, h.restAngle, h.edgeLen, h.areaSum, kb);
    }
  }
  return e;
}

/** Membrane-only gradient (analytic, cheap). Bending excluded — used for fast HVP. */
export function evalMembrane(
  x: ArrayLike<number>,
  mesh: ClothMeshData,
  mat: ClothMaterial,
  outGrad?: Float64Array,
): { energy: number; grad: Float64Array } {
  const n = mesh.count;
  const grad = outGrad ?? new Float64Array(n * 3);
  if (outGrad) grad.fill(0);
  let energy = 0;
  const m = mesh.triCount;
  const idx = mesh.indices;
  for (let t = 0; t < m; t++) {
    const i0 = idx[t * 3], i1 = idx[t * 3 + 1], i2 = idx[t * 3 + 2];
    const x0 = [x[i0 * 3], x[i0 * 3 + 1], x[i0 * 3 + 2]];
    const x1 = [x[i1 * 3], x[i1 * 3 + 1], x[i1 * 3 + 2]];
    const x2 = [x[i2 * 3], x[i2 * 3 + 1], x[i2 * 3 + 2]];
    const inv = [mesh.invDm[t * 4], mesh.invDm[t * 4 + 1], mesh.invDm[t * 4 + 2], mesh.invDm[t * 4 + 3]];
    const r = triangleEnergyGradient(x0, x1, x2, inv, mesh.areas[t], mat);
    energy += r.energy;
    const vs = [i0, i1, i2];
    for (let v = 0; v < 3; v++) {
      for (let i = 0; i < 3; i++) grad[vs[v] * 3 + i] += r.grad[v * 3 + i];
    }
  }
  return { energy, grad };
}

/** Membrane-only HVP (cheap): bending excluded from Hessian, kept in gradient.
 *  Valid inexact-Newton approximation — bending stiffness << membrane. */
export function membraneHvp(
  x: Float64Array,
  p: Float64Array,
  mesh: ClothMeshData,
  mat: ClothMaterial,
  eps = 1e-8,
): Float64Array {
  let pNorm = 0;
  for (let i = 0; i < p.length; i++) pNorm += p[i] * p[i];
  pNorm = Math.sqrt(pNorm);
  const h = eps * (1 + pNorm);
  const xp = new Float64Array(x.length);
  const xm = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) {
    xp[i] = x[i] + h * p[i];
    xm[i] = x[i] - h * p[i];
  }
  const gp = evalMembrane(xp, mesh, mat).grad;
  const gm = evalMembrane(xm, mesh, mat).grad;
  const out = new Float64Array(x.length);
  for (let i = 0; i < out.length; i++) out[i] = (gp[i] - gm[i]) / (2 * h);
  return out;
}
