// Smooth barrier: b(d) = -(d-dHat)^2 * log(d/dHat), 0 < d < dHat; 0 beyond.
// Energy per contact: E = kappaJ * w * b(d). Force magnitude: -dE/dd along +n.

export function barrierValue(d: number, dHat: number): number {
  if (d >= dHat) return 0;
  const dc = Math.max(d, 1e-12);
  const t = dc - dHat;
  return -t * t * Math.log(dc / dHat);
}

/** d/db scalar: returns dE/dd (includes kappaJ*w). Positive pushes d larger. */
export function barrierGradScalar(d: number, dHat: number, kappaJ: number, w: number): number {
  if (d >= dHat) return 0;
  const dc = Math.max(d, 1e-12);
  const t = dc - dHat;
  // d/dd [-(d-H)^2 log(d/H)] = -2(d-H) log(d/H) - (d-H)^2 / d
  const db = -2 * t * Math.log(dc / dHat) - (t * t) / dc;
  return kappaJ * w * db;
}

/** Normal-force magnitude estimate (N): lambdaN = -dE/dd for d < dHat (repulsive). */
export function barrierNormalForce(d: number, dHat: number, kappaJ: number, w: number): number {
  return Math.max(0, -barrierGradScalar(d, dHat, kappaJ, w));
}
