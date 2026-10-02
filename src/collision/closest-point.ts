// Closest-point queries with lagged (s,t) convention.
// VT: q(s,t) = (1-s-t)*a + s*b + t*c. EE: A(s) = a + s(b-a), C(t) = c + t(d-c).

export interface VtClosest {
  s: number;
  t: number;
  dist: number;
  // closest - p vector (from vertex to triangle point is -r; we return r = p - q)
  rx: number;
  ry: number;
  rz: number;
}

export function closestPointVertexTriangle(
  px: number, py: number, pz: number,
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number,
  cx: number, cy: number, cz: number,
): VtClosest {
  // Ericson 5.1.5, tracked to return barycentric (s for b, t for c).
  const abx = bx - ax, aby = by - ay, abz = bz - az;
  const acx = cx - ax, acy = cy - ay, acz = cz - az;
  const apx = px - ax, apy = py - ay, apz = pz - az;
  const d1 = abx * apx + aby * apy + abz * apz;
  const d2 = acx * apx + acy * apy + acz * apz;
  if (d1 <= 0 && d2 <= 0) return pack(0, 0, px - ax, py - ay, pz - az);
  const bpx = px - bx, bpy = py - by, bpz = pz - bz;
  const d3 = abx * bpx + aby * bpy + abz * bpz;
  const d4 = acx * bpx + acy * bpy + acz * bpz;
  if (d3 >= 0 && d4 <= d3) return pack(1, 0, px - bx, py - by, pz - bz);
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const v = d1 / (d1 - d3);
    return pack(v, 0, px - (ax + v * abx), py - (ay + v * aby), pz - (az + v * abz));
  }
  const cpx = px - cx, cpy = py - cy, cpz = pz - cz;
  const d5 = abx * cpx + aby * cpy + abz * cpz;
  const d6 = acx * cpx + acy * cpy + acz * cpz;
  if (d6 >= 0 && d5 <= d6) return pack(0, 1, px - cx, py - cy, pz - cz);
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w = d2 / (d2 - d6);
    return pack(0, w, px - (ax + w * acx), py - (ay + w * acy), pz - (az + w * acz));
  }
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const w = (d4 - d3) / (d4 - d3 + (d5 - d6));
    const qx = bx + w * (cx - bx), qy = by + w * (cy - by), qz = bz + w * (cz - bz);
    return pack(1 - w, w, px - qx, py - qy, pz - qz);
  }
  const denom = 1 / (va + vb + vc);
  const v = vb * denom;
  const w = vc * denom;
  const qx = ax + abx * v + acx * w;
  const qy = ay + aby * v + acy * w;
  const qz = az + abz * v + acz * w;
  return pack(v, w, px - qx, py - qy, pz - qz);

  function pack(s: number, t: number, rx: number, ry: number, rz: number): VtClosest {
    return { s, t, dist: Math.hypot(rx, ry, rz), rx, ry, rz };
  }
}

export interface EeClosest {
  s: number;
  t: number;
  dist: number;
  rx: number;
  ry: number;
  rz: number;
}

export function closestPointEdgeEdge(
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number,
  cx: number, cy: number, cz: number,
  dx: number, dy: number, dz: number,
): EeClosest {
  // Ericson 5.1.9 segment-segment closest.
  const d1x = bx - ax, d1y = by - ay, d1z = bz - az;
  const d2x = dx - cx, d2y = dy - cy, d2z = dz - cz;
  const rx = ax - cx, ry = ay - cy, rz = az - cz;
  const a = d1x * d1x + d1y * d1y + d1z * d1z;
  const e = d2x * d2x + d2y * d2y + d2z * d2z;
  const f = d2x * rx + d2y * ry + d2z * rz;
  let s: number, t: number;
  if (a <= 1e-30 && e <= 1e-30) {
    s = 0; t = 0;
  } else if (a <= 1e-30) {
    s = 0; t = clamp(f / e, 0, 1);
  } else {
    const c = d1x * rx + d1y * ry + d1z * rz;
    if (e <= 1e-30) {
      t = 0; s = clamp(-c / a, 0, 1);
    } else {
      const b = d1x * d2x + d1y * d2y + d1z * d2z;
      const denom = a * e - b * b;
      s = denom > 1e-30 ? clamp((b * f - c * e) / denom, 0, 1) : 0;
      t = (b * s + f) / e;
      if (t < 0) { t = 0; s = clamp(-c / a, 0, 1); }
      else if (t > 1) { t = 1; s = clamp((b - c) / a, 0, 1); }
    }
  }
  const axp = ax + d1x * s, ayp = ay + d1y * s, azp = az + d1z * s;
  const cxp = cx + d2x * t, cyp = cy + d2y * t, czp = cz + d2z * t;
  const rx2 = axp - cxp, ry2 = ayp - cyp, rz2 = azp - czp;
  return { s, t, dist: Math.hypot(rx2, ry2, rz2), rx: rx2, ry: ry2, rz: rz2 };
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}
