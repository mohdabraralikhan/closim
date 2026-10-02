// Phase 1 contact data model. SoA internally; this struct is the exchange type.

export type ContactKind = "vt" | "ee";

export interface ContactPair {
  kind: ContactKind;
  // VT: a,b,c = triangle vertices, p in d = external vertex (d unused otherwise).
  // EE: a,b and c,d are the two edges.
  a: number;
  b: number;
  c: number;
  d: number;
  // VT: barycentric coords of closest point (weights for b and c).
  // EE: segment parameters s (on ab) and t (on cd).
  s: number;
  t: number;
  /** Per-contact weight (associated area / length scale). Multiplies barrier energy. */
  w: number;
  /** Lagged normal (unit) from the last accepted iterate. */
  nx: number;
  ny: number;
  nz: number;
  /** Lagged normal-force magnitude (N) from the last accepted iterate. */
  lambdaN: number;
}

export interface ContactParams {
  /** Activation distance dHat in meters. Barrier active for 0 < d < dHat. */
  dHatM: number;
  /** Barrier energy scale in Joules per unit weight: E = kappaJ * w * b(d).
   *  Documented SI units — NOT a game-style stiffness. */
  kappaJ: number;
  /** Minimum admissible distance in meters; trials at/below dMin are rejected. */
  dMinM: number;
  /** Coulomb coefficient. */
  frictionMu: number;
  /** Friction regularization length in meters. */
  frictionEpsM: number;
}

export const DEFAULT_CONTACT_PARAMS: ContactParams = {
  dHatM: 0.002,
  kappaJ: 50,
  dMinM: 1e-4,
  frictionMu: 0.3,
  frictionEpsM: 1e-4,
};

export interface ContactDiagnostics {
  candidatePairs: number;
  activePairs: number;
  vtPairs: number;
  eePairs: number;
  floorPairs: number;
  minDistance: number;
  minTOI: number;
  barrierEnergy: number;
  frictionWork: number;
  rejectedLineSearchSteps: number;
  ccdFailures: number;
}

export function emptyDiagnostics(): ContactDiagnostics {
  return {
    candidatePairs: 0,
    activePairs: 0,
    vtPairs: 0,
    eePairs: 0,
    floorPairs: 0,
    minDistance: Infinity,
    minTOI: Infinity,
    barrierEnergy: 0,
    frictionWork: 0,
    rejectedLineSearchSteps: 0,
    ccdFailures: 0,
  };
}
