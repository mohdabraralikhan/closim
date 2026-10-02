// Cubic root solver for CCD coplanarity equations. Returns sorted roots in [0,1].
// Coefficients for a3 t^3 + a2 t^2 + a1 t + a0 = 0.

export function solveCubic01(a3: number, a2: number, a1: number, a0: number): number[] {
  const eps = 1e-12;
  if (Math.abs(a3) < eps) return solveQuadratic01(a2, a1, a0);
  // normalize
  const A = a2 / a3, B = a1 / a3, C = a0 / a3;
  // depressed cubic t^3 + pt + q via t = y - A/3
  const sqA = A * A;
  const p = (3 * B - sqA) / 3;
  const q = (2 * sqA * A - 9 * A * B + 27 * C) / 27;
  const roots: number[] = [];
  const disc = (q * q) / 4 + (p * p * p) / 27;
  const shift = A / 3;
  if (disc > eps) {
    const sd = Math.sqrt(disc);
    const u = Math.cbrt(-q / 2 + sd);
    const v = Math.cbrt(-q / 2 - sd);
    roots.push(u + v - shift);
  } else if (Math.abs(disc) <= eps) {
    const u = Math.cbrt(-q / 2);
    roots.push(2 * u - shift, -u - shift);
  } else {
    const r = Math.sqrt(-(p * p * p) / 27);
    const phi = Math.acos(Math.min(1, Math.max(-1, -q / (2 * r))));
    const s = 2 * Math.cbrt(r);
    roots.push(
      s * Math.cos(phi / 3) - shift,
      s * Math.cos((phi + 2 * Math.PI) / 3) - shift,
      s * Math.cos((phi + 4 * Math.PI) / 3) - shift,
    );
  }
  // Newton polish + filter
  const out: number[] = [];
  for (let r of roots) {
    if (!isFinite(r)) continue;
    for (let k = 0; k < 4; k++) {
      const f = ((a3 * r + a2) * r + a1) * r + a0;
      const df = (3 * a3 * r + 2 * a2) * r + a1;
      if (Math.abs(df) < 1e-18) break;
      r -= f / df;
    }
    if (r >= -1e-9 && r <= 1 + 1e-9) out.push(Math.min(1, Math.max(0, r)));
  }
  out.sort((x, y) => x - y);
  // dedupe
  return out.filter((v, i) => i === 0 || v - out[i - 1] > 1e-9);
}

function solveQuadratic01(a2: number, a1: number, a0: number): number[] {
  const eps = 1e-12;
  if (Math.abs(a2) < eps) {
    if (Math.abs(a1) < eps) return [];
    const r = -a0 / a1;
    return r >= 0 && r <= 1 ? [r] : [];
  }
  const disc = a1 * a1 - 4 * a2 * a0;
  if (disc < 0) return [];
  const sd = Math.sqrt(disc);
  const out: number[] = [];
  for (const r of [(-a1 - sd) / (2 * a2), (-a1 + sd) / (2 * a2)]) {
    if (r >= 0 && r <= 1) out.push(r);
  }
  return out.sort((x, y) => x - y);
}
