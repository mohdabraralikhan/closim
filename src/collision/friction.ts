// Lagged (semi-implicit) Coulomb friction.
// lambdaN is lagged from the last accepted iterate; u is the relative
// tangential displacement over the current Newton trial.
// f_t = -mu * lambdaN * u_t / sqrt(|u_t|^2 + eps^2),  u_t = u - (u.n)n.

export function tangentialPart(
  ux: number, uy: number, uz: number,
  nx: number, ny: number, nz: number,
): [number, number, number] {
  const vn = ux * nx + uy * ny + uz * nz;
  return [ux - vn * nx, uy - vn * ny, uz - vn * nz];
}

export function coulombForce(
  ux: number, uy: number, uz: number,
  nx: number, ny: number, nz: number,
  lambdaN: number, mu: number, eps: number,
): [number, number, number] {
  const [tx, ty, tz] = tangentialPart(ux, uy, uz, nx, ny, nz);
  const mag = Math.sqrt(tx * tx + ty * ty + tz * tz + eps * eps);
  const s = (mu * Math.max(0, lambdaN)) / mag;
  return [-s * tx, -s * ty, -s * tz];
}

/** Friction invariants check: tangentiality + cone bound. */
export function checkInvariants(
  fx: number, fy: number, fz: number,
  nx: number, ny: number, nz: number,
  lambdaN: number, mu: number, tol = 1e-9,
): { tangential: boolean; inCone: boolean } {
  const n = Math.abs(fx * nx + fy * ny + fz * nz);
  const mag = Math.hypot(fx, fy, fz);
  return {
    tangential: n <= tol * (1 + mag),
    inCone: mag <= mu * Math.max(0, lambdaN) + 1e-12,
  };
}
