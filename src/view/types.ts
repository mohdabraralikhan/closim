// G10 viewport core: shared types, error taxonomy, visibility flags.
// This module is pure TypeScript (no DOM, no rendering library) so the whole
// viewport state layer stays testable headless under vitest.

export type Vec3 = [number, number, number];

export type EntityKind = "garment" | "panel" | "seam" | "avatar" | "region";

export interface EntityRef {
  kind: EntityKind;
  /** Stable ID: "garment", avatar id, pattern panel id, seam id, or region label. */
  id: string;
}

export type WorkspaceErrorCode =
  | "no-garment"
  | "invalid-entity"
  | "stale-reference"
  | "invalid-pin"
  | "invalid-transform"
  | "sim-running"
  | "invalid-state";

export class WorkspaceError extends Error {
  readonly code: WorkspaceErrorCode;

  constructor(code: WorkspaceErrorCode, message: string) {
    super(`workspace (${code}): ${message}`);
    this.code = code;
  }
}

export interface VisibilityFlags {
  garment: boolean;
  avatar: boolean;
  grid: boolean;
  seams: boolean;
  panelBoundaries: boolean;
  pins: boolean;
  wireframe: boolean;
  normals: boolean;
  penetrationHeat: boolean;
}

export function defaultVisibility(): VisibilityFlags {
  return {
    garment: true,
    avatar: true,
    grid: true,
    seams: true,
    panelBoundaries: true,
    pins: true,
    wireframe: false,
    normals: false,
    penetrationHeat: false,
  };
}

export type VisibilityKey = keyof VisibilityFlags;

export function setVisibility(flags: VisibilityFlags, key: VisibilityKey, value: boolean): void {
  flags[key] = value;
}

export function toggleVisibility(flags: VisibilityFlags, key: VisibilityKey): void {
  flags[key] = !flags[key];
}

export interface Bounds {
  min: Vec3;
  max: Vec3;
}

export function computeBounds(positions: Float32Array | Float64Array, start = 0, count = positions.length / 3 - start): Bounds {
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  const end = start + count;
  for (let i = start; i < end; i++) {
    const x = positions[i * 3];
    const y = positions[i * 3 + 1];
    const z = positions[i * 3 + 2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
      throw new WorkspaceError("invalid-state", `non-finite position at vertex ${i}`);
    }
    if (x < min[0]) min[0] = x;
    if (y < min[1]) min[1] = y;
    if (z < min[2]) min[2] = z;
    if (x > max[0]) max[0] = x;
    if (y > max[1]) max[1] = y;
    if (z > max[2]) max[2] = z;
  }
  if (count <= 0) return { min: [0, 0, 0], max: [0, 0, 0] };
  return { min, max };
}

export function boundsCenter(b: Bounds): Vec3 {
  return [
    (b.min[0] + b.max[0]) / 2,
    (b.min[1] + b.max[1]) / 2,
    (b.min[2] + b.max[2]) / 2,
  ];
}
