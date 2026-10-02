// Barrier contact + friction stubs (Phase 1).
// Target barrier: B(d) = -k (d - dHat)^2 log(d / dHat), d < dHat; lagged Coulomb friction.

export interface ContactParams {
  dHat: number; // contact activation distance (m)
  stiffness: number; // barrier weight
  frictionMu: number;
}

export const DEFAULT_CONTACT: ContactParams = { dHat: 1e-3, stiffness: 1e4, frictionMu: 0.3 };

export function barrierEnergy(dist: number, params: ContactParams): number {
  if (dist >= params.dHat) return 0;
  const d = Math.max(dist, 1e-9);
  const t = d - params.dHat;
  return -params.stiffness * t * t * Math.log(d / params.dHat);
}
