/**
 * Real-body avatar support: Wavefront OBJ loading, cleanup, and
 * deterministic decimation for the collision proxy.
 *
 * Pipeline for open-source bodies (MakeHuman/MPFB CC0 assets, Anny
 * Apache-2.0 bakes): authored mesh (10k+ tris) renders as-is, while the
 * solver collides against a vertex-clustered proxy (1-3k tris) referenced
 * from `AvatarSpec.collision`. Everything here is deterministic:
 * first-found wins, index-order iteration, no randomness.
 */

import { validateAvatarSpec, type AvatarSpec } from "./avatar.js";

export interface ObjParseOptions {
  /** Multiply every coordinate (e.g. 0.01 for cm, 0.1 for dm). Default 1. */
  unitScale?: number;
  /** Drop faces referencing missing vertices instead of throwing. Default false. */
  tolerant?: boolean;
}

export interface ObjMesh {
  positions: number[];
  indices: number[];
}

/** Minimal Wavefront OBJ reader: `v` positions + `f` faces (any `a/b/c` form, fan-triangulated). */
export function parseAvatarOBJ(text: string, opts: ObjParseOptions = {}): ObjMesh {
  const scale = opts.unitScale ?? 1;
  if (!(scale > 0) || !Number.isFinite(scale)) throw new RangeError("avatar-mesh: unitScale must be positive");
  const verts: number[][] = [];
  const vertIndex = new Map<string, number>();
  const positions: number[] = [];
  const indices: number[] = [];
  const lines = text.split(/\r?\n/);
  for (let ln = 0; ln < lines.length; ln++) {
    const line = lines[ln].trim();
    if (line.length === 0 || line.startsWith("#")) continue;
    const parts = line.split(/\s+/);
    if (parts[0] === "v") {
      if (parts.length < 4) {
        if (opts.tolerant) continue;
        throw new RangeError(`avatar-mesh: malformed vertex on line ${ln + 1}`);
      }
      const x = Number(parts[1]) * scale;
      const y = Number(parts[2]) * scale;
      const z = Number(parts[3]) * scale;
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
        if (opts.tolerant) {
          verts.push([Number.NaN, Number.NaN, Number.NaN]);
          continue;
        }
        throw new RangeError(`avatar-mesh: non-finite vertex on line ${ln + 1}`);
      }
      verts.push([x, y, z]);
    } else if (parts[0] === "f") {
      if (parts.length < 4) {
        if (opts.tolerant) continue;
        throw new RangeError(`avatar-mesh: malformed face on line ${ln + 1}`);
      }
      // Fan triangulation over welded vertices: deterministic in stored
      // order, shared across faces so components/decimation see one mesh.
      const face: number[] = [];
      let bad = false;
      for (let k = 1; k < parts.length; k++) {
        const vi = Number(parts[k].split("/")[0]);
        if (!Number.isInteger(vi) || vi === 0 || Math.abs(vi) > verts.length) {
          bad = true;
          break;
        }
        face.push(vi > 0 ? vi - 1 : verts.length + vi);
      }
      if (bad) {
        if (opts.tolerant) continue;
        throw new RangeError(`avatar-mesh: bad face index on line ${ln + 1}`);
      }
      const welded: number[] = [];
      for (const v of face) {
        if (!Number.isFinite(verts[v][0])) {
          bad = true;
          break;
        }
        const key = `${verts[v][0]},${verts[v][1]},${verts[v][2]}`;
        let idx = vertIndex.get(key);
        if (idx === undefined) {
          idx = positions.length / 3;
          vertIndex.set(key, idx);
          positions.push(verts[v][0], verts[v][1], verts[v][2]);
        }
        welded.push(idx);
      }
      if (bad) {
        if (opts.tolerant) continue;
        throw new RangeError(`avatar-mesh: face references a non-finite vertex on line ${ln + 1}`);
      }
      for (let k = 1; k + 1 < welded.length; k++) indices.push(welded[0], welded[k], welded[k + 1]);
    }
  }
  if (indices.length === 0) throw new RangeError("avatar-mesh: OBJ contains no faces");
  return { positions, indices };
}

export interface MeshStats {
  vertexCount: number;
  triCount: number;
  min: [number, number, number];
  max: [number, number, number];
  degenerateTris: number;
}

/** Axis-aligned bounds + degenerate-triangle count (zero-area in 3D). */
export function meshStats(positions: number[], indices: number[]): MeshStats {
  const min: [number, number, number] = [Infinity, Infinity, Infinity];
  const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < positions.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      if (positions[i + k] < min[k]) min[k] = positions[i + k];
      if (positions[i + k] > max[k]) max[k] = positions[i + k];
    }
  }
  let degenerate = 0;
  for (let t = 0; t < indices.length; t += 3) {
    const a = indices[t] * 3, b = indices[t + 1] * 3, c = indices[t + 2] * 3;
    const ux = positions[b] - positions[a], uy = positions[b + 1] - positions[a + 1], uz = positions[b + 2] - positions[a + 2];
    const vx = positions[c] - positions[a], vy = positions[c + 1] - positions[a + 1], vz = positions[c + 2] - positions[a + 2];
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    if (nx * nx + ny * ny + nz * nz <= 0) degenerate++;
  }
  return { vertexCount: positions.length / 3, triCount: indices.length / 3, min, max, degenerateTris: degenerate };
}

/**
 * Keep only the largest vertex-connected component (drops eyeballs, teeth,
 * inner mouth, stray shells from character exports). Deterministic.
 */
export function keepLargestComponent(positions: number[], indices: number[]): ObjMesh {
  const n = positions.length / 3;
  const parent = Array.from({ length: n }, (_, i) => i);
  const find = (x: number): number => (parent[x] === x ? x : (parent[x] = find(parent[x])));
  for (let t = 0; t < indices.length; t += 3) {
    const a = find(indices[t]), b = find(indices[t + 1]), c = find(indices[t + 2]);
    const r = Math.min(a, b, c);
    parent[a] = r;
    parent[b] = r;
    parent[c] = r;
  }
  const sizes = new Map<number, number>();
  for (let i = 0; i < n; i++) sizes.set(find(i), (sizes.get(find(i)) ?? 0) + 1);
  let best = -1, bestSize = -1;
  for (const [root, size] of sizes) {
    if (size > bestSize) {
      bestSize = size;
      best = root;
    }
  }
  const remap = new Map<number, number>();
  const outPositions: number[] = [];
  const outIndices: number[] = [];
  for (let t = 0; t < indices.length; t += 3) {
    const tri = [indices[t], indices[t + 1], indices[t + 2]];
    if (find(tri[0]) !== best) continue;
    for (const v of tri) {
      if (!remap.has(v)) {
        remap.set(v, outPositions.length / 3);
        outPositions.push(positions[v * 3], positions[v * 3 + 1], positions[v * 3 + 2]);
      }
      outIndices.push(remap.get(v)!);
    }
  }
  return { positions: outPositions, indices: outIndices };
}

export interface DecimateOptions {
  /** Uniform grid cell in metres. Larger = coarser proxy. */
  cellM: number;
}

/**
 * Vertex-clustering decimation: every vertex snaps to its grid cell's first
 * occupant; triangles collapsing to an edge/point are dropped, duplicate
 * triples are dropped. Deterministic and topology-agnostic — ideal for a
 * collision proxy where render fidelity does not matter.
 */
export function decimateMesh(positions: number[], indices: number[], opts: DecimateOptions): ObjMesh {
  const cell = opts.cellM;
  if (!(cell > 0) || !Number.isFinite(cell)) throw new RangeError("avatar-mesh: decimation cellM must be positive");
  const rep = new Map<string, number>();
  const clusterOf = new Array<number>(positions.length / 3);
  const outPositions: number[] = [];
  for (let v = 0; v < positions.length / 3; v++) {
    const key = `${Math.floor(positions[v * 3] / cell)},${Math.floor(positions[v * 3 + 1] / cell)},${Math.floor(positions[v * 3 + 2] / cell)}`;
    let c = rep.get(key);
    if (c === undefined) {
      c = outPositions.length / 3;
      rep.set(key, c);
      outPositions.push(positions[v * 3], positions[v * 3 + 1], positions[v * 3 + 2]);
    }
    clusterOf[v] = c;
  }
  const seen = new Set<string>();
  const outIndices: number[] = [];
  for (let t = 0; t < indices.length; t += 3) {
    const a = clusterOf[indices[t]], b = clusterOf[indices[t + 1]], c = clusterOf[indices[t + 2]];
    if (a === b || b === c || a === c) continue;
    const key = `${a},${b},${c}`;
    if (seen.has(key)) continue;
    seen.add(key);
    outIndices.push(a, b, c);
  }
  if (outIndices.length === 0) throw new RangeError("avatar-mesh: decimation collapsed every triangle; use a smaller cellM");
  return { positions: outPositions, indices: outIndices };
}

export interface MeshAvatarOptions {
  id?: string;
  bodyPart?: string;
  thicknessM?: number;
  /** When set, attached as the solver/render... (see below) proxy. */
  proxyCellM?: number;
}

/**
 * Build an AvatarSpec from an OBJ-derived mesh. The full-resolution mesh is
 * kept for rendering; when `proxyCellM` is set, a decimated copy is attached
 * as `collision` and the solver/contact/diagnostics use it instead.
 */
export function makeMeshAvatar(
  mesh: ObjMesh,
  source: string,
  opts: MeshAvatarOptions = {},
): AvatarSpec {
  const stats = meshStats(mesh.positions, mesh.indices);
  if (stats.degenerateTris > 0) {
    throw new RangeError(`avatar-mesh: ${source} has ${stats.degenerateTris} degenerate triangles`);
  }
  const spec: AvatarSpec = {
    id: opts.id ?? `avatar/${source}`,
    bodyPart: opts.bodyPart ?? "full-body",
    positions: [...mesh.positions],
    indices: [...mesh.indices],
    thicknessM: opts.thicknessM ?? 0.005,
  };
  if (opts.proxyCellM !== undefined) {
    const proxy = decimateMesh(mesh.positions, mesh.indices, { cellM: opts.proxyCellM });
    spec.collision = {
      id: `${spec.id}/proxy`,
      bodyPart: spec.bodyPart,
      positions: proxy.positions,
      indices: proxy.indices,
      thicknessM: spec.thicknessM,
    };
  }
  validateAvatarSpec(spec);
  return spec;
}
