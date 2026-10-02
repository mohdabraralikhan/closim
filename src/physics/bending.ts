// Hinge bending: W = 0.5 * k * (theta - theta0)^2 * edgeLen^2 / areaSum
// Reference gradient via central finite differences (fine for V0 small meshes;
// analytic hinge Hessian is a Phase-2 optimization).

import { dihedral } from "../mesh/mesh.js";

export function hingeEnergy(
  x: ArrayLike<number>,
  v0: number, v1: number, v2: number, v3: number,
  restAngle: number, edgeLen: number, areaSum: number, k: number,
): number {
  const th = dihedral(x, v0, v1, v2, v3);
  const d = th - restAngle;
  const w = edgeLen * edgeLen / Math.max(areaSum, 1e-12);
  return 0.5 * k * d * d * w;
}

/** Add bending energy gradient into `grad` (length 3n). eps in meters. */
export function addHingeGradient(
  x: Float32Array | Float64Array,
  grad: Float64Array,
  v0: number, v1: number, v2: number, v3: number,
  restAngle: number, edgeLen: number, areaSum: number, k: number,
  eps = 1e-7,
): number {
  const verts = [v0, v1, v2, v3];
  const e = hingeEnergy(x, v0, v1, v2, v3, restAngle, edgeLen, areaSum, k);
  const tmp = Float64Array.from(x as Float64Array);
  for (const v of verts) {
    for (let i = 0; i < 3; i++) {
      const idx = v * 3 + i;
      const old = tmp[idx];
      tmp[idx] = old + eps;
      const ep = hingeEnergy(tmp, v0, v1, v2, v3, restAngle, edgeLen, areaSum, k);
      tmp[idx] = old - eps;
      const em = hingeEnergy(tmp, v0, v1, v2, v3, restAngle, edgeLen, areaSum, k);
      tmp[idx] = old;
      grad[idx] += (ep - em) / (2 * eps);
    }
  }
  return e;
}
