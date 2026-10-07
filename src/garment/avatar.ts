/**
 * G8C — deterministic analytic avatar for garment fitting.
 *
 * The visible avatar is deliberately NOT part of this model. This module
 * provides simplified closed collision shapes (capsule / box) authored in SI
 * metres, merged into a single triangle mesh that plugs into the existing
 * public collision boundary (`ContactSystem.setStaticMesh`), plus a
 * deterministic signed-distance query used for placement diagnostics.
 *
 * No solver, FEM, or collision mathematics are modified here.
 */

import { closestPointVertexTriangle } from "../collision/closest-point.js";

export type Vec3 = [number, number, number];

export interface AvatarSpec {
  id: string;
  bodyPart: string;
  /** Flat xyz, SI metres. Closed mesh (capsule/box tessellation). */
  positions: number[];
  /** Triangle triples into positions. */
  indices: number[];
  /** Surface thickness metadata in metres; not consumed by collision. */
  thicknessM: number;
}

export interface CapsuleAvatarOptions {
  id?: string;
  bodyPart?: string;
  radiusM?: number;
  /** Length of the cylindrical section, excluding hemispheres. */
  cylinderLengthM?: number;
  /** Center of the capsule in world metres. */
  center?: Vec3;
  segments?: number;
  hemisphereRings?: number;
  thicknessM?: number;
}

export interface BoxAvatarOptions {
  id?: string;
  bodyPart?: string;
  halfExtentsM?: Vec3;
  center?: Vec3;
  thicknessM?: number;
}

function assertFinite(value: number, name: string): void {
  if (!Number.isFinite(value)) throw new RangeError(`avatar: ${name} must be finite`);
}

function positive(value: number, name: string): number {
  assertFinite(value, name);
  if (!(value > 0)) throw new RangeError(`avatar: ${name} must be positive`);
  return value;
}

function nonnegative(value: number, name: string): number {
  assertFinite(value, name);
  if (value < 0) throw new RangeError(`avatar: ${name} must be nonnegative`);
  return value;
}

/** Vertical capsule (torso stand-in) centred at `center`, axis along +y. */
export function makeCapsuleAvatar(opts: CapsuleAvatarOptions = {}): AvatarSpec {
  const radiusM = positive(opts.radiusM ?? 0.16, "radiusM");
  const lengthM = nonnegative(opts.cylinderLengthM ?? 0.5, "cylinderLengthM");
  const center = opts.center ?? [0, 0.9, 0];
  center.forEach((v, i) => assertFinite(v, `center[${i}]`));
  const segments = opts.segments ?? 12;
  const hemi = opts.hemisphereRings ?? 3;
  if (!Number.isInteger(segments) || segments < 6) throw new RangeError("avatar: segments must be an integer >= 6");
  if (!Number.isInteger(hemi) || hemi < 2) throw new RangeError("avatar: hemisphereRings must be an integer >= 2");
  const half = lengthM * 0.5;
  // Deterministic ring stack: top pole ring -> equator -> bottom pole ring.
  const ringY: number[] = [];
  for (let j = 0; j <= hemi; j++) ringY.push(half + radiusM * Math.cos((Math.PI / 2) * (j / hemi)));
  for (let j = 1; j <= hemi; j++) ringY.push(-half - radiusM * Math.sin((Math.PI / 2) * (j / hemi)));
  const positions: number[] = [];
  for (const y of ringY) {
    const capCenter = y > half ? half : y < -half ? -half : y;
    const radial = Math.sqrt(Math.max(0, radiusM * radiusM - (y - capCenter) * (y - capCenter)));
    for (let i = 0; i < segments; i++) {
      const phi = (2 * Math.PI * i) / segments;
      positions.push(
        center[0] + radial * Math.cos(phi),
        center[1] + y,
        center[2] + radial * Math.sin(phi),
      );
    }
  }
  const indices: number[] = [];
  for (let ring = 0; ring < ringY.length - 1; ring++) {
    for (let i = 0; i < segments; i++) {
      const next = (i + 1) % segments;
      const a = ring * segments + i, b = ring * segments + next;
      const c = a + segments, d = b + segments;
      indices.push(a, b, c, b, d, c);
    }
  }
  return {
    id: opts.id ?? "avatar/torso",
    bodyPart: opts.bodyPart ?? "torso",
    positions,
    indices,
    thicknessM: opts.thicknessM ?? 0.005,
  };
}

/** Axis-aligned box avatar centred at `center`. */
export function makeBoxAvatar(opts: BoxAvatarOptions = {}): AvatarSpec {
  const half = opts.halfExtentsM ?? [0.18, 0.3, 0.12];
  const [hx, hy, hz] = half.map((v, i) => positive(v, `halfExtentsM[${i}]`));
  const center = opts.center ?? [0, 0.9, 0];
  center.forEach((v, i) => assertFinite(v, `center[${i}]`));
  const [cx, cy, cz] = center;
  return {
    id: opts.id ?? "avatar/torso-box",
    bodyPart: opts.bodyPart ?? "torso",
    positions: [
      cx - hx, cy - hy, cz - hz, cx + hx, cy - hy, cz - hz,
      cx + hx, cy + hy, cz - hz, cx - hx, cy + hy, cz - hz,
      cx - hx, cy - hy, cz + hz, cx + hx, cy - hy, cz + hz,
      cx + hx, cy + hy, cz + hz, cx - hx, cy + hy, cz + hz,
    ],
    indices: [
      0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7,
      0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5,
      2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7,
    ],
    thicknessM: opts.thicknessM ?? 0.005,
  };
}

export function validateAvatarSpec(avatar: AvatarSpec): void {
  if (!avatar || typeof avatar.id !== "string" || avatar.id.length === 0) {
    throw new RangeError("avatar: id must be a non-empty string");
  }
  if (!Array.isArray(avatar.positions) || avatar.positions.length < 9 || avatar.positions.length % 3 !== 0) {
    throw new RangeError("avatar: positions must hold at least one triangle of xyz triples");
  }
  if (!Array.isArray(avatar.indices) || avatar.indices.length < 3 || avatar.indices.length % 3 !== 0) {
    throw new RangeError("avatar: indices must hold at least one triangle triple");
  }
  for (let i = 0; i < avatar.positions.length; i++) assertFinite(avatar.positions[i], `positions[${i}]`);
  const n = avatar.positions.length / 3;
  for (let i = 0; i < avatar.indices.length; i++) {
    const v = avatar.indices[i];
    if (!Number.isInteger(v) || v < 0 || v >= n) throw new RangeError(`avatar: invalid triangle index ${v}`);
  }
  nonnegative(avatar.thicknessM, "thicknessM");
}

/** Merge avatar parts into one static-mesh pair for `setStaticMesh`. */
export function avatarToStaticMesh(avatars: AvatarSpec | AvatarSpec[]): {
  positions: Float32Array;
  indices: Uint32Array;
} {
  const list = Array.isArray(avatars) ? avatars : [avatars];
  if (list.length === 0) throw new RangeError("avatar: at least one avatar mesh is required");
  let nv = 0, ni = 0;
  for (const a of list) {
    validateAvatarSpec(a);
    nv += a.positions.length / 3;
    ni += a.indices.length;
  }
  const positions = new Float32Array(nv * 3);
  const indices = new Uint32Array(ni);
  let vo = 0, io = 0;
  for (const a of list) {
    positions.set(a.positions, vo * 3);
    for (let k = 0; k < a.indices.length; k++) indices[io + k] = a.indices[k] + vo;
    vo += a.positions.length / 3;
    io += a.indices.length;
  }
  return { positions, indices };
}

// ---------------------------------------------------------------------------
// Signed distance (diagnostics only — never fed back into the solver)
// ---------------------------------------------------------------------------

/**
 * Fixed tilt direction for inside/outside ray casts. Deliberately NOT
 * axis-aligned: axis-aligned rays graze shared tessellation diagonals
 * (e.g. the exact centre of a box face), double-counting the crossing and
 * flipping the sign. A fixed irrational-ish tilt keeps the query
 * deterministic while making edge-grazing measure-zero.
 */
const RAY_DIR: Vec3 = (() => {
  const x = 1, y = 0.317, z = 0.113;
  const n = Math.hypot(x, y, z);
  return [x / n, y / n, z / n];
})();

function rayCrossesTriangle(
  ox: number, oy: number, oz: number,
  ax: number, ay: number, az: number,
  bx: number, by: number, bz: number,
  cx: number, cy: number, cz: number,
): boolean {
  // General Möller–Trumbore for ray origin o along RAY_DIR.
  const e1x = bx - ax, e1y = by - ay, e1z = bz - az;
  const e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
  const [dx, dy, dz] = RAY_DIR;
  // p = dir x e2.
  const px = dy * e2z - dz * e2y;
  const py = dz * e2x - dx * e2z;
  const pz = dx * e2y - dy * e2x;
  const det = e1x * px + e1y * py + e1z * pz;
  if (Math.abs(det) < 1e-30) return false;
  const inv = 1 / det;
  const tx = ox - ax, ty = oy - ay, tz = oz - az;
  const u = (tx * px + ty * py + tz * pz) * inv;
  if (u < 0 || u > 1) return false;
  // q = tvec x e1.
  const qx = ty * e1z - tz * e1y;
  const qy = tz * e1x - tx * e1z;
  const qz = tx * e1y - ty * e1x;
  const v = (dx * qx + dy * qy + dz * qz) * inv;
  if (v < 0 || u + v > 1) return false;
  const t = (e2x * qx + e2y * qy + e2z * qz) * inv;
  return t > 1e-9;
}

/** True when p is strictly inside the closed avatar mesh (ray-cast +x). */
export function pointInsideAvatar(p: Vec3, avatar: AvatarSpec): boolean {
  let crossings = 0;
  for (let t = 0; t < avatar.indices.length; t += 3) {
    const a = avatar.indices[t] * 3, b = avatar.indices[t + 1] * 3, c = avatar.indices[t + 2] * 3;
    if (
      rayCrossesTriangle(
        p[0], p[1], p[2],
        avatar.positions[a], avatar.positions[a + 1], avatar.positions[a + 2],
        avatar.positions[b], avatar.positions[b + 1], avatar.positions[b + 2],
        avatar.positions[c], avatar.positions[c + 1], avatar.positions[c + 2],
      )
    ) crossings++;
  }
  return crossings % 2 === 1;
}

/**
 * Signed distance from p to the avatar surface. Negative = inside.
 * Unsigned part uses the exact closest-point kernels; the sign uses a
 * deterministic ray cast. O(tris); diagnostics-scale only.
 */
export function signedDistanceToAvatar(p: Vec3, avatar: AvatarSpec): number {
  let minD = Infinity;
  for (let t = 0; t < avatar.indices.length; t += 3) {
    const a = avatar.indices[t], b = avatar.indices[t + 1], c = avatar.indices[t + 2];
    const d = closestPointVertexTriangle(
      p[0], p[1], p[2],
      avatar.positions[a * 3], avatar.positions[a * 3 + 1], avatar.positions[a * 3 + 2],
      avatar.positions[b * 3], avatar.positions[b * 3 + 1], avatar.positions[b * 3 + 2],
      avatar.positions[c * 3], avatar.positions[c * 3 + 1], avatar.positions[c * 3 + 2],
    ).dist;
    if (d < minD) minD = d;
  }
  return pointInsideAvatar(p, avatar) ? -minD : minD;
}

/** Canonical JSON for deterministic avatar persistence (sorted keys). */
export function avatarToCanonical(avatar: AvatarSpec): string {
  validateAvatarSpec(avatar);
  return JSON.stringify({
    bodyPart: avatar.bodyPart,
    id: avatar.id,
    indices: [...avatar.indices],
    positions: [...avatar.positions],
    thicknessM: avatar.thicknessM,
  });
}

export function avatarFromJSON(json: string): AvatarSpec {
  let parsed: unknown;
  try {
    parsed = JSON.parse(json);
  } catch {
    throw new RangeError("avatar: serialized avatar is not valid JSON");
  }
  const a = parsed as AvatarSpec;
  validateAvatarSpec(a);
  return { id: a.id, bodyPart: a.bodyPart, positions: [...a.positions], indices: [...a.indices], thicknessM: a.thicknessM };
}
