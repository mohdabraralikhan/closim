/**
 * Avatar collision geometry and motion adapter.
 *
 * Geometry is authored in local SI metres. The visible avatar is deliberately
 * not part of this model: callers provide a separate, simplified collision
 * shape (or compound of shapes) for simulation.
 */

export type Vec3 = readonly [number, number, number];
export type Quat = readonly [number, number, number, number]; // x, y, z, w

export interface Transform {
  translation: Vec3;
  rotation: Quat;
}

export const IDENTITY_TRANSFORM: Transform = {
  translation: [0, 0, 0],
  rotation: [0, 0, 0, 1],
};

export interface MotionProvider {
  /** Return the body's rigid world transform at time in seconds. */
  sample(timeSeconds: number): Transform;
}

export interface TriangleMeshGeometry {
  kind: "triangle-mesh";
  positions: ArrayLike<number>; // xyz triples, local metres
  indices: ArrayLike<number>; // triangle triples
  triangleBodyParts?: readonly string[];
}

export interface SphereGeometry {
  kind: "sphere";
  radiusM: number;
  /** Low-cost simulation tessellation; defaults are intentionally coarse. */
  segments?: number;
  rings?: number;
}

export interface CapsuleGeometry {
  kind: "capsule";
  radiusM: number;
  /** Length of the cylindrical section, excluding hemispheres. */
  cylinderLengthM: number;
  segments?: number;
  hemisphereRings?: number;
}

export interface BoxGeometry {
  kind: "box";
  halfExtentsM: Vec3;
}

export interface CompoundChild {
  geometry: Exclude<ColliderGeometry, CompoundGeometry>;
  transform?: Transform;
  bodyPart?: string;
}

export interface CompoundGeometry {
  kind: "compound";
  children: readonly CompoundChild[];
}

export type ColliderGeometry = TriangleMeshGeometry | SphereGeometry | CapsuleGeometry | BoxGeometry | CompoundGeometry;

export interface AvatarCollider {
  id: string;
  /** Semantic label such as torso, left-upper-arm, or right-foot. */
  bodyPart: string;
  geometry: ColliderGeometry;
  /** Local rigid transform relative to the body's motion transform. */
  transform?: Transform;
  motion?: MotionProvider;
  /** Surface thickness metadata in metres; not consumed by collision physics. */
  thicknessM: number;
}

/** Static motion source useful for non-animated body parts. */
export class StaticMotionProvider implements MotionProvider {
  constructor(private readonly transform: Transform = IDENTITY_TRANSFORM) {
    validateTransform(transform);
  }
  sample(timeSeconds: number): Transform {
    assertFinite(timeSeconds, "timeSeconds");
    return cloneTransform(this.transform);
  }
}

export interface MotionInterval {
  t0: number;
  t1: number;
  start: Transform;
  end: Transform;
}

export function sampleMotion(provider: MotionProvider, t0: number, t1: number): MotionInterval {
  assertFinite(t0, "t0");
  assertFinite(t1, "t1");
  if (t1 < t0) throw new RangeError("motion interval must satisfy t1 >= t0");
  const start = provider.sample(t0), end = provider.sample(t1);
  validateTransform(start);
  validateTransform(end);
  return { t0, t1, start: cloneTransform(start), end: cloneTransform(end) };
}

/** Translation lerp plus shortest-arc quaternion slerp (nlerp near zero angle). */
export function interpolateTransform(a: Transform, b: Transform, alpha: number): Transform {
  validateTransform(a);
  validateTransform(b);
  assertFinite(alpha, "alpha");
  if (alpha < 0 || alpha > 1) throw new RangeError("interpolation alpha must be in [0, 1]");
  const qa = canonicalQuat(a.rotation), qb0 = canonicalQuat(b.rotation);
  let qb: [number, number, number, number] = [...qb0];
  let dot = qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3];
  if (dot < 0) {
    qb = [-qb[0], -qb[1], -qb[2], -qb[3]];
    dot = -dot;
  }
  dot = Math.max(-1, Math.min(1, dot));
  let q: [number, number, number, number];
  if (dot > 0.9995) {
    q = [
      qa[0] + alpha * (qb[0] - qa[0]), qa[1] + alpha * (qb[1] - qa[1]),
      qa[2] + alpha * (qb[2] - qa[2]), qa[3] + alpha * (qb[3] - qa[3]),
    ];
  } else {
    const theta = Math.acos(dot), sinTheta = Math.sin(theta);
    const wa = Math.sin((1 - alpha) * theta) / sinTheta;
    const wb = Math.sin(alpha * theta) / sinTheta;
    q = [qa[0] * wa + qb[0] * wb, qa[1] * wa + qb[1] * wb,
      qa[2] * wa + qb[2] * wb, qa[3] * wa + qb[3] * wb];
  }
  return {
    translation: [
      a.translation[0] + alpha * (b.translation[0] - a.translation[0]),
      a.translation[1] + alpha * (b.translation[1] - a.translation[1]),
      a.translation[2] + alpha * (b.translation[2] - a.translation[2]),
    ],
    rotation: normalizedQuat(q),
  };
}

export interface PreprocessedCollisionMesh {
  positions: Float32Array;
  indices: Uint32Array;
  triangleBodyParts: string[];
}

/** Validate, weld exact duplicate vertices, and discard duplicate/degenerate faces. */
export function preprocessCollisionMesh(mesh: TriangleMeshGeometry, defaultBodyPart: string): PreprocessedCollisionMesh {
  validateGeometry(mesh);
  const vertexMap = new Map<string, number>();
  const positions: number[] = [];
  const oldToNew = new Uint32Array(mesh.positions.length / 3);
  for (let i = 0; i < oldToNew.length; i++) {
    const x = canonicalZero(mesh.positions[3 * i]);
    const y = canonicalZero(mesh.positions[3 * i + 1]);
    const z = canonicalZero(mesh.positions[3 * i + 2]);
    if (![x, y, z].every((value) => Number.isFinite(Math.fround(value)))) {
      throw new RangeError(`vertex ${i} cannot be represented in collision Float32 buffers`);
    }
    const key = `${x},${y},${z}`;
    let id = vertexMap.get(key);
    if (id === undefined) {
      id = positions.length / 3;
      vertexMap.set(key, id);
      positions.push(x, y, z);
    }
    oldToNew[i] = id;
  }
  const indices: number[] = [];
  const labels: string[] = [];
  const seen = new Set<string>();
  for (let t = 0; t < mesh.indices.length / 3; t++) {
    const a = oldToNew[mesh.indices[3 * t]], b = oldToNew[mesh.indices[3 * t + 1]], c = oldToNew[mesh.indices[3 * t + 2]];
    if (a === b || b === c || c === a || triangleAreaSquared(positions, a, b, c) <= 1e-24) continue;
    const key = [a, b, c].sort((u, v) => u - v).join(":");
    if (seen.has(key)) continue;
    seen.add(key);
    indices.push(a, b, c);
    labels.push(mesh.triangleBodyParts?.[t] ?? defaultBodyPart);
  }
  if (indices.length === 0) throw new RangeError("collision mesh has no nondegenerate triangles");
  return { positions: Float32Array.from(positions), indices: Uint32Array.from(indices), triangleBodyParts: labels };
}

/** Adapter shape mirrors ContactSystem's existing setStaticMesh(pos, idx) boundary. */
export interface CollisionObject {
  positions: Float32Array;
  indices: Uint32Array;
  /** Preserved metadata only; existing ContactSystem has no per-object thickness input. */
  thicknessM: number;
  triangleBodyParts: string[];
}

export interface CollisionObjectMotion {
  interval: MotionInterval;
  start: CollisionObject;
  end: CollisionObject;
}

export function toCollisionObject(collider: AvatarCollider, timeSeconds: number): CollisionObject {
  validateCollider(collider);
  assertFinite(timeSeconds, "timeSeconds");
  const world = collider.motion?.sample(timeSeconds) ?? IDENTITY_TRANSFORM;
  validateTransform(world);
  const root = composeTransforms(world, collider.transform ?? IDENTITY_TRANSFORM);
  const local = geometryMesh(collider.geometry, collider.bodyPart);
  const transformed = transformMesh(local, root);
  const prepared = preprocessCollisionMesh(transformed, collider.bodyPart);
  return {
    positions: prepared.positions,
    indices: prepared.indices,
    thicknessM: collider.thicknessM,
    triangleBodyParts: prepared.triangleBodyParts,
  };
}

export function toCollisionObjectMotion(collider: AvatarCollider, t0: number, t1: number): CollisionObjectMotion {
  const interval = sampleMotion(collider.motion ?? new StaticMotionProvider(), t0, t1);
  return {
    interval,
    start: toCollisionObject({ ...collider, motion: new StaticMotionProvider(interval.start) }, t0),
    end: toCollisionObject({ ...collider, motion: new StaticMotionProvider(interval.end) }, t1),
  };
}

/** Attach the sampled mesh through the public ContactSystem API. */
export function attachCollisionObject(
  target: { setStaticMesh(positions: Float32Array, indices: Uint32Array): void },
  object: CollisionObject,
): void {
  target.setStaticMesh(object.positions, object.indices);
}

/** Return a floor height only when all mesh vertices form a horizontal plane. */
export function horizontalPlaneHeight(object: CollisionObject, toleranceM = 1e-6): number {
  if (object.positions.length < 9 || object.positions.length % 3 !== 0) throw new RangeError("invalid collision positions");
  assertFinite(toleranceM, "toleranceM");
  if (toleranceM < 0) throw new RangeError("toleranceM must be nonnegative");
  const y = object.positions[1];
  for (let i = 0; i < object.positions.length; i += 3) {
    if (Math.abs(object.positions[i + 1] - y) > toleranceM) throw new RangeError("collider is not a horizontal plane");
  }
  return y;
}

function geometryMesh(geometry: ColliderGeometry, bodyPart: string): TriangleMeshGeometry {
  switch (geometry.kind) {
    case "triangle-mesh": return geometry;
    case "sphere": return sphereMesh(geometry);
    case "capsule": return capsuleMesh(geometry);
    case "box": return boxMesh(geometry);
    case "compound": {
      const positions: number[] = [], indices: number[] = [], labels: string[] = [];
      for (const child of geometry.children) {
        const childPart = child.bodyPart ?? bodyPart;
        const childMesh = transformMesh(geometryMesh(child.geometry, childPart), child.transform ?? IDENTITY_TRANSFORM);
        const offset = positions.length / 3;
        positions.push(...Array.from(childMesh.positions));
        for (let i = 0; i < childMesh.indices.length; i++) indices.push(childMesh.indices[i] + offset);
        labels.push(...(childMesh.triangleBodyParts ?? new Array(childMesh.indices.length / 3).fill(childPart)));
      }
      if (indices.length === 0) throw new RangeError("compound collider must contain geometry");
      return { kind: "triangle-mesh", positions, indices, triangleBodyParts: labels };
    }
  }
}

function sphereMesh(g: SphereGeometry): TriangleMeshGeometry {
  const r = positive(g.radiusM, "sphere radiusM");
  const segments = resolution(g.segments ?? 12, "sphere segments", 6, 64);
  const rings = resolution(g.rings ?? 6, "sphere rings", 3, 32);
  const positions: number[] = [0, r, 0];
  for (let j = 1; j < rings; j++) {
    const theta = Math.PI * j / rings;
    for (let i = 0; i < segments; i++) {
      const phi = 2 * Math.PI * i / segments;
      positions.push(r * Math.sin(theta) * Math.cos(phi), r * Math.cos(theta), r * Math.sin(theta) * Math.sin(phi));
    }
  }
  const bottom = positions.length / 3;
  positions.push(0, -r, 0);
  const indices: number[] = [];
  for (let i = 0; i < segments; i++) {
    const next = (i + 1) % segments;
    indices.push(0, 1 + i, 1 + next);
    for (let ring = 0; ring < rings - 2; ring++) {
      const a = 1 + ring * segments + i, b = 1 + ring * segments + next;
      const c = a + segments, d = b + segments;
      indices.push(a, c, b, b, c, d);
    }
    const last = 1 + (rings - 2) * segments;
    indices.push(last + i, bottom, last + next);
  }
  return { kind: "triangle-mesh", positions, indices };
}

function capsuleMesh(g: CapsuleGeometry): TriangleMeshGeometry {
  const r = positive(g.radiusM, "capsule radiusM");
  const length = nonnegative(g.cylinderLengthM, "capsule cylinderLengthM");
  const segments = resolution(g.segments ?? 12, "capsule segments", 6, 64);
  const hemi = resolution(g.hemisphereRings ?? 3, "capsule hemisphereRings", 2, 16);
  const half = length * 0.5;
  const ringY: number[] = [];
  for (let j = 0; j <= hemi; j++) ringY.push(half + r * Math.cos((Math.PI / 2) * j / hemi));
  for (let j = 1; j <= hemi; j++) ringY.push(-half - r * Math.sin((Math.PI / 2) * j / hemi));
  const positions: number[] = [];
  for (const y of ringY) {
    const capCenter = y > half ? half : y < -half ? -half : y;
    const radial = Math.sqrt(Math.max(0, r * r - (y - capCenter) ** 2));
    for (let i = 0; i < segments; i++) {
      const phi = 2 * Math.PI * i / segments;
      positions.push(radial * Math.cos(phi), y, radial * Math.sin(phi));
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
  return { kind: "triangle-mesh", positions, indices };
}

function boxMesh(g: BoxGeometry): TriangleMeshGeometry {
  const [x, y, z] = g.halfExtentsM.map((v, i) => positive(v, `box halfExtentsM[${i}]`)) as [number, number, number];
  return {
    kind: "triangle-mesh",
    positions: [-x, -y, -z, x, -y, -z, x, y, -z, -x, y, -z, -x, -y, z, x, -y, z, x, y, z, -x, y, z],
    indices: [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
      1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7],
  };
}

function transformMesh(mesh: TriangleMeshGeometry, transform: Transform): TriangleMeshGeometry {
  validateTransform(transform);
  const q = normalizedQuat(transform.rotation);
  const positions = new Float64Array(mesh.positions.length);
  for (let i = 0; i < mesh.positions.length; i += 3) {
    const p = rotate(q, [mesh.positions[i], mesh.positions[i + 1], mesh.positions[i + 2]]);
    positions[i] = p[0] + transform.translation[0];
    positions[i + 1] = p[1] + transform.translation[1];
    positions[i + 2] = p[2] + transform.translation[2];
  }
  return { kind: "triangle-mesh", positions, indices: mesh.indices, triangleBodyParts: mesh.triangleBodyParts };
}

function composeTransforms(a: Transform, b: Transform): Transform {
  const qa = normalizedQuat(a.rotation), qb = normalizedQuat(b.rotation);
  const p = rotate(qa, b.translation);
  return {
    translation: [a.translation[0] + p[0], a.translation[1] + p[1], a.translation[2] + p[2]],
    rotation: [
      qa[3] * qb[0] + qa[0] * qb[3] + qa[1] * qb[2] - qa[2] * qb[1],
      qa[3] * qb[1] - qa[0] * qb[2] + qa[1] * qb[3] + qa[2] * qb[0],
      qa[3] * qb[2] + qa[0] * qb[1] - qa[1] * qb[0] + qa[2] * qb[3],
      qa[3] * qb[3] - qa[0] * qb[0] - qa[1] * qb[1] - qa[2] * qb[2],
    ],
  };
}

function rotate(q: Quat, p: Vec3): [number, number, number] {
  const [x, y, z, w] = q;
  const tx = 2 * (y * p[2] - z * p[1]);
  const ty = 2 * (z * p[0] - x * p[2]);
  const tz = 2 * (x * p[1] - y * p[0]);
  return [p[0] + w * tx + y * tz - z * ty,
    p[1] + w * ty + z * tx - x * tz,
    p[2] + w * tz + x * ty - y * tx];
}

function validateCollider(collider: AvatarCollider): void {
  if (!collider.id || !collider.bodyPart) throw new RangeError("collider id and bodyPart are required");
  nonnegative(collider.thicknessM, "thicknessM");
  if (collider.transform) validateTransform(collider.transform);
  validateGeometry(collider.geometry);
}

function validateGeometry(g: ColliderGeometry): void {
  switch (g.kind) {
    case "triangle-mesh": {
      if (g.positions.length < 9 || g.positions.length % 3 !== 0 || g.indices.length < 3 || g.indices.length % 3 !== 0) {
        throw new RangeError("triangle mesh requires xyz vertices and triangle indices");
      }
      for (let i = 0; i < g.positions.length; i++) assertFinite(g.positions[i], `positions[${i}]`);
      for (let i = 0; i < g.indices.length; i++) {
        const index = g.indices[i];
        if (!Number.isInteger(index) || index < 0 || index >= g.positions.length / 3) throw new RangeError(`invalid triangle index ${index}`);
      }
      if (g.triangleBodyParts && g.triangleBodyParts.length !== g.indices.length / 3) throw new RangeError("triangleBodyParts length mismatch");
      return;
    }
    case "sphere": positive(g.radiusM, "sphere radiusM"); resolution(g.segments ?? 12, "sphere segments", 6, 64); resolution(g.rings ?? 6, "sphere rings", 3, 32); return;
    case "capsule": positive(g.radiusM, "capsule radiusM"); nonnegative(g.cylinderLengthM, "capsule cylinderLengthM"); resolution(g.segments ?? 12, "capsule segments", 6, 64); resolution(g.hemisphereRings ?? 3, "capsule hemisphereRings", 2, 16); return;
    case "box": g.halfExtentsM.forEach((v, i) => positive(v, `box halfExtentsM[${i}]`)); return;
    case "compound":
      if (g.children.length === 0) throw new RangeError("compound collider must contain children");
      for (const child of g.children) {
        if (child.transform) validateTransform(child.transform);
        validateGeometry(child.geometry);
      }
      return;
  }
}

function validateTransform(t: Transform): void {
  if (t.translation.length !== 3 || t.rotation.length !== 4) throw new RangeError("invalid rigid transform dimensions");
  t.translation.forEach((v, i) => assertFinite(v, `translation[${i}]`));
  t.rotation.forEach((v, i) => assertFinite(v, `rotation[${i}]`));
  const norm = Math.hypot(...t.rotation);
  if (!(norm > 1e-12)) throw new RangeError("rotation quaternion must be nonzero");
}

function normalizedQuat(q: Quat): [number, number, number, number] {
  const n = Math.hypot(...q);
  if (!(n > 1e-12) || !Number.isFinite(n)) throw new RangeError("rotation quaternion must be finite and nonzero");
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

function canonicalQuat(q: Quat): [number, number, number, number] {
  const unit = normalizedQuat(q);
  for (const component of unit) {
    if (Math.abs(component) <= 1e-15) continue;
    return component < 0 ? [-unit[0], -unit[1], -unit[2], -unit[3]] : unit;
  }
  return unit;
}

function cloneTransform(t: Transform): Transform {
  return { translation: [t.translation[0], t.translation[1], t.translation[2]], rotation: [...t.rotation] };
}

function assertFinite(value: number, name: string): void {
  if (!Number.isFinite(value)) throw new RangeError(`${name} must be finite`);
}

function positive(value: number, name: string): number {
  assertFinite(value, name);
  if (!(value > 0)) throw new RangeError(`${name} must be positive`);
  return value;
}

function nonnegative(value: number, name: string): number {
  assertFinite(value, name);
  if (value < 0) throw new RangeError(`${name} must be nonnegative`);
  return value;
}

function resolution(value: number, name: string, min: number, max: number): number {
  if (!Number.isInteger(value) || value < min || value > max) throw new RangeError(`${name} must be an integer in [${min}, ${max}]`);
  return value;
}

function triangleAreaSquared(p: ArrayLike<number>, a: number, b: number, c: number): number {
  const ux = p[3 * b] - p[3 * a], uy = p[3 * b + 1] - p[3 * a + 1], uz = p[3 * b + 2] - p[3 * a + 2];
  const vx = p[3 * c] - p[3 * a], vy = p[3 * c + 1] - p[3 * a + 1], vz = p[3 * c + 2] - p[3 * a + 2];
  const x = uy * vz - uz * vy, y = uz * vx - ux * vz, z = ux * vy - uy * vx;
  return x * x + y * y + z * z;
}

function canonicalZero(v: number): number { return Object.is(v, -0) ? 0 : v; }
