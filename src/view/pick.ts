// G10A picking: CPU ray casting against the current render positions.
// Pure math, no rendering library, so picks are deterministic and testable.
// All distances are world meters; ray parameter t is distance along the ray.

import type { AssembledGarment } from "../garment/assembly.js";
import type { AvatarSpec } from "../garment/avatar.js";
import { type Mat4 } from "./camera.js";
import { type Vec3, WorkspaceError } from "./types.js";

export interface Ray {
  origin: Vec3;
  dir: Vec3;
}

export function invert4(m: Mat4): Mat4 {
  const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
  const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
  const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
  const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];
  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;
  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (Math.abs(det) < 1e-20) throw new WorkspaceError("invalid-state", "singular view-projection matrix");
  det = 1 / det;
  const out = new Float64Array(16);
  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
  return out;
}

function transformPoint(m: Mat4, x: number, y: number, z: number, w: number): Vec3 {
  const tx = m[0] * x + m[4] * y + m[8] * z + m[12] * w;
  const ty = m[1] * x + m[5] * y + m[9] * z + m[13] * w;
  const tz = m[2] * x + m[6] * y + m[10] * z + m[14] * w;
  const tw = m[3] * x + m[7] * y + m[11] * z + m[15] * w;
  if (Math.abs(tw) < 1e-20) throw new WorkspaceError("invalid-state", "degenerate unproject");
  return [tx / tw, ty / tw, tz / tw];
}

/** ndc in [-1,1]; ndcY = +1 is viewport top. */
export function screenRay(viewProj: Mat4, ndcX: number, ndcY: number): Ray {
  const inv = invert4(viewProj);
  const near = transformPoint(inv, ndcX, ndcY, -1, 1);
  const far = transformPoint(inv, ndcX, ndcY, 1, 1);
  const dx = far[0] - near[0];
  const dy = far[1] - near[1];
  const dz = far[2] - near[2];
  const len = Math.hypot(dx, dy, dz);
  if (len < 1e-15) throw new WorkspaceError("invalid-state", "degenerate pick ray");
  return { origin: near, dir: [dx / len, dy / len, dz / len] };
}

/** Distance along the ray to a double-sided triangle, or null. */
export function rayTriangleDistance(origin: Vec3, dir: Vec3, ax: number, ay: number, az: number, bx: number, by: number, bz: number, cx: number, cy: number, cz: number): number | null {
  const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
  const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
  const px = dir[1] * e2z - dir[2] * e2y;
  const py = dir[2] * e2x - dir[0] * e2z;
  const pz = dir[0] * e2y - dir[1] * e2x;
  const det = e1x * px + e1y * py + e1z * pz;
  if (Math.abs(det) < 1e-14) return null;
  const invDet = 1 / det;
  const tx = origin[0] - ax, ty = origin[1] - ay, tz = origin[2] - az;
  const u = (tx * px + ty * py + tz * pz) * invDet;
  if (u < -1e-9 || u > 1 + 1e-9) return null;
  const qx = ty * e1z - tz * e1y;
  const qy = tz * e1x - tx * e1z;
  const qz = tx * e1y - ty * e1x;
  const v = (dir[0] * qx + dir[1] * qy + dir[2] * qz) * invDet;
  if (v < -1e-9 || u + v > 1 + 1e-9) return null;
  const t = (e2x * qx + e2y * qy + e2z * qz) * invDet;
  return t > 1e-9 ? t : null;
}

/** Ray vs AABB slab test. */
export function rayIntersectsBox(ray: Ray, min: Vec3, max: Vec3): boolean {
  let tmin = 0;
  let tmax = Infinity;
  for (let i = 0; i < 3; i++) {
    const d = ray.dir[i];
    const o = ray.origin[i];
    if (Math.abs(d) < 1e-15) {
      if (o < min[i] || o > max[i]) return false;
      continue;
    }
    let t1 = (min[i] - o) / d;
    let t2 = (max[i] - o) / d;
    if (t1 > t2) { const tmp = t1; t1 = t2; t2 = tmp; }
    tmin = Math.max(tmin, t1);
    tmax = Math.min(tmax, t2);
    if (tmin > tmax) return false;
  }
  return true;
}

export interface PanelHit {
  panelId: string;
  triangle: number;
  vertex: number;
  distance: number;
  point: Vec3;
}

function panelForTriangle(assembled: AssembledGarment, triangle: number): string {
  for (const r of assembled.panelRanges) {
    if (triangle >= r.triangleStart && triangle < r.triangleStart + r.triangleCount) return r.panelId;
  }
  return "";
}

export function pickPanel(assembled: AssembledGarment, positions: Float32Array, ray: Ray): PanelHit | null {
  const indices = assembled.indices;
  const triCount = indices.length / 3;
  let best: PanelHit | null = null;
  for (let t = 0; t < triCount; t++) {
    const ia = indices[t * 3];
    const ib = indices[t * 3 + 1];
    const ic = indices[t * 3 + 2];
    const t0 = rayTriangleDistance(
      ray.origin, ray.dir,
      positions[ia * 3], positions[ia * 3 + 1], positions[ia * 3 + 2],
      positions[ib * 3], positions[ib * 3 + 1], positions[ib * 3 + 2],
      positions[ic * 3], positions[ic * 3 + 1], positions[ic * 3 + 2],
    );
    if (t0 === null) continue;
    if (best === null || t0 < best.distance) {
      const vertex = [ia, ib, ic].reduce((a, b) => (dist2ToRay(ray, positions, a) <= dist2ToRay(ray, positions, b) ? a : b));
      best = {
        panelId: panelForTriangle(assembled, t),
        triangle: t,
        vertex,
        distance: t0,
        point: [ray.origin[0] + t0 * ray.dir[0], ray.origin[1] + t0 * ray.dir[1], ray.origin[2] + t0 * ray.dir[2]],
      };
    }
  }
  return best;
}

function dist2ToRay(ray: Ray, positions: Float32Array, vertex: number): number {
  const px = positions[vertex * 3] - ray.origin[0];
  const py = positions[vertex * 3 + 1] - ray.origin[1];
  const pz = positions[vertex * 3 + 2] - ray.origin[2];
  const along = px * ray.dir[0] + py * ray.dir[1] + pz * ray.dir[2];
  if (along < 0) return Infinity;
  const cx = px - along * ray.dir[0];
  const cy = py - along * ray.dir[1];
  const cz = pz - along * ray.dir[2];
  return cx * cx + cy * cy + cz * cz;
}

/** Nearest mesh vertex within worldRadiusM of the ray (ray parameter > 0). */
export function pickVertex(assembled: AssembledGarment, positions: Float32Array, ray: Ray, worldRadiusM = 0.01): { vertex: number; distanceM: number; alongRay: number } | null {
  const n = positions.length / 3;
  const r2 = worldRadiusM * worldRadiusM;
  let bestVertex = -1;
  let bestDist2 = r2;
  let bestAlong = 0;
  for (let i = 0; i < n; i++) {
    const along = alongRayOf(ray, positions, i);
    if (along < 0) continue;
    const d2 = dist2ToRay(ray, positions, i);
    if (d2 > r2) continue;
    const better = bestVertex < 0
      || d2 < bestDist2 - 1e-18
      || (d2 <= bestDist2 + 1e-18 && along < bestAlong);
    if (better) {
      bestVertex = i;
      bestDist2 = d2;
      bestAlong = along;
    }
  }
  if (bestVertex < 0) return null;
  return { vertex: bestVertex, distanceM: Math.sqrt(bestDist2), alongRay: bestAlong };
}

function alongRayOf(ray: Ray, positions: Float32Array, vertex: number): number {
  return (positions[vertex * 3] - ray.origin[0]) * ray.dir[0]
    + (positions[vertex * 3 + 1] - ray.origin[1]) * ray.dir[1]
    + (positions[vertex * 3 + 2] - ray.origin[2]) * ray.dir[2];
}

export interface SeamHit {
  seamId: string;
  weldIndex: number;
  distance: number;
  point: Vec3;
}

/** Pick a seam by proximity of the ray to a weld-pair segment. */
export function pickSeam(assembled: AssembledGarment, positions: Float32Array, ray: Ray, maxRayDistanceM = 0.012): SeamHit | null {
  let best: SeamHit | null = null;
  for (let w = 0; w < assembled.weldPairs.length; w++) {
    const pair = assembled.weldPairs[w];
    const a = pair.vertexA * 3;
    const b = pair.vertexB * 3;
    const hit = raySegmentDistance(ray,
      positions[a], positions[a + 1], positions[a + 2],
      positions[b], positions[b + 1], positions[b + 2]);
    if (hit === null || hit.perpDist > maxRayDistanceM) continue;
    if (best === null || hit.perpDist < best.distance) {
      best = {
        seamId: pair.seamId,
        weldIndex: w,
        distance: hit.perpDist,
        point: [
          ray.origin[0] + hit.tRay * ray.dir[0],
          ray.origin[1] + hit.tRay * ray.dir[1],
          ray.origin[2] + hit.tRay * ray.dir[2],
        ],
      };
    }
  }
  return best;
}

function raySegmentDistance(ray: Ray, ax: number, ay: number, az: number, bx: number, by: number, bz: number): { perpDist: number; tRay: number } | null {
  // Segment-segment closest points (Ericson), with the ray as a long segment.
  const L = 1e6;
  const p0x = ray.origin[0], p0y = ray.origin[1], p0z = ray.origin[2];
  const d1x = ray.dir[0] * L, d1y = ray.dir[1] * L, d1z = ray.dir[2] * L;
  const d2x = bx - ax, d2y = by - ay, d2z = bz - az;
  const rx = p0x - ax, ry = p0y - ay, rz = p0z - az;
  const A = d1x * d1x + d1y * d1y + d1z * d1z;
  const E = d2x * d2x + d2y * d2y + d2z * d2z;
  const F = d2x * rx + d2y * ry + d2z * rz;
  const EPS = 1e-14;
  let s = 0;
  let t = 0;
  if (A <= EPS && E <= EPS) return null;
  if (A <= EPS) {
    t = clamp01(F / E);
  } else {
    const C = d1x * rx + d1y * ry + d1z * rz;
    if (E <= EPS) {
      s = clamp01(-C / A);
    } else {
      const B = d1x * d2x + d1y * d2y + d1z * d2z;
      const denom = A * E - B * B;
      s = denom !== 0 ? clamp01((B * F - C * E) / denom) : 0;
      t = (B * s + F) / E;
      if (t < 0) { t = 0; s = clamp01(-C / A); }
      else if (t > 1) { t = 1; s = clamp01((B - C) / A); }
    }
  }
  const p1x = p0x + d1x * s, p1y = p0y + d1y * s, p1z = p0z + d1z * s;
  const p2x = ax + d2x * t, p2y = ay + d2y * t, p2z = az + d2z * t;
  return {
    perpDist: Math.hypot(p1x - p2x, p1y - p2y, p1z - p2z),
    tRay: s * L,
  };
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

export interface AvatarHit {
  distance: number;
  triangle: number;
  point: Vec3;
}

export function pickAvatar(avatar: AvatarSpec, ray: Ray): AvatarHit | null {
  const pos = avatar.positions;
  const idx = avatar.indices;
  let best: AvatarHit | null = null;
  for (let t = 0; t < idx.length / 3; t++) {
    const ia = idx[t * 3];
    const ib = idx[t * 3 + 1];
    const ic = idx[t * 3 + 2];
    const dist = rayTriangleDistance(
      ray.origin, ray.dir,
      pos[ia * 3], pos[ia * 3 + 1], pos[ia * 3 + 2],
      pos[ib * 3], pos[ib * 3 + 1], pos[ib * 3 + 2],
      pos[ic * 3], pos[ic * 3 + 1], pos[ic * 3 + 2],
    );
    if (dist === null) continue;
    if (best === null || dist < best.distance) {
      best = {
        distance: dist,
        triangle: t,
        point: [ray.origin[0] + dist * ray.dir[0], ray.origin[1] + dist * ray.dir[1], ray.origin[2] + dist * ray.dir[2]],
      };
    }
  }
  return best;
}
