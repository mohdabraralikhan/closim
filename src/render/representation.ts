// G15A — render representation (derived presentation state).
//
// Data flow is one-way: simulation state -> RenderGarment -> adapter.
// Nothing here owns physics: positions are COPIES refreshed by explicit
// sync calls, topology changes are detected (not silently absorbed) via a
// topology key, and quality levels only tune presentation, never the solve.
//
// Render modes: shaded | wireframe | fabric | technical | collision.
// Quality: draft | preview | high (adapter-side pixel behavior + overlays).

import { PatternCadError } from "../pattern/cad.js";
import type { AssembledGarment } from "../garment/assembly.js";
import type { AvatarSpec } from "../garment/avatar.js";
import { signedDistanceToAvatar } from "../garment/avatar.js";
import type { PanelUV } from "./uvgen.js";
import { generatePanelUVs } from "./uvgen.js";

export type RenderMode = "shaded" | "wireframe" | "fabric" | "technical" | "collision";
export type QualityLevel = "draft" | "preview" | "high";

export interface QualitySettings {
  level: QualityLevel;
  /** Renderer pixel ratio multiplier (adapter-applied). */
  pixelRatio: number;
  antialias: boolean;
  shadows: boolean;
  /** Texture anisotropy level (adapter-applied). */
  textureAnisotropy: number;
  /** Draw panel-boundary overlay lines. */
  boundaryOverlay: boolean;
  /** Draw seam polylines. */
  seamOverlay: boolean;
}

export const QUALITY_SETTINGS: Record<QualityLevel, QualitySettings> = {
  draft: { level: "draft", pixelRatio: 1, antialias: false, shadows: false, textureAnisotropy: 1, boundaryOverlay: false, seamOverlay: false },
  preview: { level: "preview", pixelRatio: 1.5, antialias: true, shadows: false, textureAnisotropy: 4, boundaryOverlay: true, seamOverlay: true },
  high: { level: "high", pixelRatio: 2, antialias: true, shadows: true, textureAnisotropy: 8, boundaryOverlay: true, seamOverlay: true },
};

export function qualitySettings(level: QualityLevel): QualitySettings {
  const settings = QUALITY_SETTINGS[level];
  if (!settings) throw new PatternCadError("invalid-document", `unknown quality level '${level}'`);
  return { ...settings };
}

export interface RenderVisibility {
  garment: boolean;
  avatar: boolean;
  seams: boolean;
  boundaries: boolean;
  pins: boolean;
  wireframe: boolean;
  normals: boolean;
}

export function defaultVisibility(): RenderVisibility {
  return { garment: true, avatar: true, seams: true, boundaries: true, pins: true, wireframe: false, normals: false };
}

export type VisibilityKey = keyof RenderVisibility;

export function setRenderVisibility(visibility: RenderVisibility, key: VisibilityKey, value: boolean): RenderVisibility {
  if (!(key in visibility)) throw new PatternCadError("invalid-document", `unknown visibility key '${key}'`);
  return { ...visibility, [key]: value };
}

// ---------------------------------------------------------------------------
// Normals (area-weighted, degenerate-safe, deterministic)
// ---------------------------------------------------------------------------

export function computeNormals(positions: Float32Array, indices: Uint32Array): Float32Array {
  const n = positions.length / 3;
  const normals = new Float32Array(n * 3);
  for (let t = 0; t < indices.length; t += 3) {
    const a = indices[t * 3], b = indices[t * 3 + 1], c = indices[t * 3 + 2];
    const abx = positions[b * 3] - positions[a * 3];
    const aby = positions[b * 3 + 1] - positions[a * 3 + 1];
    const abz = positions[b * 3 + 2] - positions[a * 3 + 2];
    const acx = positions[c * 3] - positions[a * 3];
    const acy = positions[c * 3 + 1] - positions[a * 3 + 1];
    const acz = positions[c * 3 + 2] - positions[a * 3 + 2];
    // Unnormalized cross product = 2 × area-weighted face normal.
    const nx = aby * acz - abz * acy;
    const ny = abz * acx - abx * acz;
    const nz = abx * acy - aby * acx;
    normals[a * 3] += nx;
    normals[a * 3 + 1] += ny;
    normals[a * 3 + 2] += nz;
    normals[b * 3] += nx;
    normals[b * 3 + 1] += ny;
    normals[b * 3 + 2] += nz;
    normals[c * 3] += nx;
    normals[c * 3 + 1] += ny;
    normals[c * 3 + 2] += nz;
  }
  for (let i = 0; i < n; i++) {
    const x = normals[i * 3], y = normals[i * 3 + 1], z = normals[i * 3 + 2];
    const len = Math.hypot(x, y, z);
    if (len > 1e-30) {
      normals[i * 3] = x / len;
      normals[i * 3 + 1] = y / len;
      normals[i * 3 + 2] = z / len;
    } else {
      normals[i * 3] = 0;
      normals[i * 3 + 1] = 1;
      normals[i * 3 + 2] = 0;
    }
  }
  return normals;
}

// ---------------------------------------------------------------------------
// Render garment (position copies + derived normals/UVs, topology-keyed)
// ---------------------------------------------------------------------------

export interface RenderGarment {
  garmentId: string;
  positions: Float32Array;
  normals: Float32Array;
  indices: Uint32Array;
  panelUVs: PanelUV[];
  panelRanges: AssembledGarment["panelRanges"];
  materialOf: Record<string, string>;
  /** Simulation epoch consumed at the last sync (caller-owned counter). */
  epoch: number;
  topologyKey: string;
}

export function topologyKeyOf(assembled: AssembledGarment): string {
  const parts = [
    assembled.positions.length,
    Array.from(assembled.indices.slice(0, 64)).join(","),
    assembled.indices.length,
    assembled.panelRanges.map((r) => `${r.panelId}:${r.vertexStart}:${r.vertexCount}:${r.triangleStart}:${r.triangleCount}`).join(";"),
  ];
  let hash = 2166136261;
  const s = parts.join("|");
  for (let i = 0; i < s.length; i++) {
    hash ^= s.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return `topo/${(hash >>> 0).toString(16)}`;
}

export function buildRenderGarment(
  garmentId: string,
  assembled: AssembledGarment,
  positions: Float32Array,
  materialOf: Record<string, string>,
  epoch = 0,
): RenderGarment {
  if (positions.length !== assembled.positions.length) {
    throw new PatternCadError("invalid-document", "render positions do not match the garment vertex count");
  }
  const panelUVs = assembled.panelRanges.map((r) => generatePanelUVs(assembled, r.panelId));
  return {
    garmentId,
    positions: Float32Array.from(positions),
    normals: computeNormals(positions, assembled.indices),
    indices: Uint32Array.from(assembled.indices),
    panelUVs,
    panelRanges: JSON.parse(JSON.stringify(assembled.panelRanges)) as AssembledGarment["panelRanges"],
    materialOf: { ...materialOf },
    epoch,
    topologyKey: topologyKeyOf(assembled),
  };
}

export type SyncResult = "updated" | "topology-changed";

/**
 * Explicit sync from fresh simulation state. Same topology: positions and
 * normals refresh in place (no reallocation). Changed topology (or vertex
 * count): returns "topology-changed" and leaves state untouched — the
 * caller rebuilds, so stale buffers are impossible.
 */
export function syncRenderPositions(
  render: RenderGarment,
  assembled: AssembledGarment,
  positions: Float32Array,
  epoch: number,
): SyncResult {
  if (positions.length !== render.positions.length || topologyKeyOf(assembled) !== render.topologyKey) {
    return "topology-changed";
  }
  render.positions.set(positions);
  render.normals.set(computeNormals(positions, render.indices));
  render.epoch = epoch;
  return "updated";
}

// ---------------------------------------------------------------------------
// Avatar + overlays (derived)
// ---------------------------------------------------------------------------

export interface RenderAvatar {
  positions: Float32Array;
  indices: Uint32Array;
  normals: Float32Array;
}

export function buildRenderAvatar(avatar: AvatarSpec): RenderAvatar {
  const positions = Float32Array.from(avatar.positions);
  const indices = Uint32Array.from(avatar.indices);
  return { positions, indices, normals: computeNormals(positions, indices) };
}

/** Seam polylines in current render positions (weld-pair chains per seam). */
export function seamPolylines(render: RenderGarment, assembled: AssembledGarment): Map<string, Array<[number, number, number]>> {
  const out = new Map<string, Array<[number, number, number]>>();
  const bySeam = new Map<string, typeof assembled.weldPairs>();
  for (const weld of assembled.weldPairs) {
    if (!bySeam.has(weld.seamId)) bySeam.set(weld.seamId, []);
    bySeam.get(weld.seamId)!.push(weld);
  }
  for (const [seamId, welds] of bySeam) {
    const sorted = [...welds].sort((a, b) => a.stitchIndex - b.stitchIndex);
    out.set(
      seamId,
      sorted.map((w) => {
        const ax = render.positions[w.vertexA * 3], ay = render.positions[w.vertexA * 3 + 1], az = render.positions[w.vertexA * 3 + 2];
        const bx = render.positions[w.vertexB * 3], by = render.positions[w.vertexB * 3 + 1], bz = render.positions[w.vertexB * 3 + 2];
        return [(ax + bx) / 2, (ay + by) / 2, (az + bz) / 2];
      }),
    );
  }
  return out;
}

/** Panel boundary segments in current render positions (single-use edges). */
export function boundarySegments(render: RenderGarment): Array<[number, number, number, number, number, number]> {
  const counts = new Map<string, number>();
  const orientation = new Map<string, [number, number]>();
  const key = (a: number, b: number): string => (a < b ? `${a}_${b}` : `${b}_${a}`);
  for (let t = 0; t < render.indices.length; t += 3) {
    for (let e = 0; e < 3; e++) {
      const a = render.indices[t * 3 + e], b = render.indices[t * 3 + ((e + 1) % 3)];
      const k = key(a, b);
      counts.set(k, (counts.get(k) ?? 0) + 1);
      orientation.set(k, [a, b]);
    }
  }
  const out: Array<[number, number, number, number, number, number]> = [];
  for (const [k, count] of counts) {
    if (count !== 1) continue;
    const [a, b] = orientation.get(k)!;
    out.push([
      render.positions[a * 3], render.positions[a * 3 + 1], render.positions[a * 3 + 2],
      render.positions[b * 3], render.positions[b * 3 + 1], render.positions[b * 3 + 2],
    ]);
  }
  return out;
}

/**
 * Collision-view heat per vertex: 0 = clear, 1 = at/past the avatar surface.
 * Proximity band is dHat (matches the solver's activation distance).
 */
export function contactHeat(
  positions: Float32Array,
  avatar: AvatarSpec,
  dHatM: number,
): Float32Array {
  if (!(dHatM > 0)) throw new PatternCadError("invalid-transform", "heat band must be positive");
  const heat = new Float32Array(positions.length / 3);
  for (let i = 0; i < heat.length; i++) {
    const d = signedDistanceToAvatar(
      [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]],
      avatar,
    );
    heat[i] = d <= 0 ? 1 : Math.max(0, 1 - d / dHatM);
  }
  return heat;
}
