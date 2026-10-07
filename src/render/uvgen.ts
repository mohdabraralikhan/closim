// G15C — deterministic per-panel UV generation and shading helpers.
//
// UVs live in pattern (rest) space, never in simulation positions: deforming
// the cloth cannot move the print. Each panel maps its pattern bbox to
// [0,1]²; with grainAlign the grain direction runs along +V. Material texture
// scale/rotation apply at render time (adapter), not here.

import { PatternCadError } from "../pattern/cad.js";
import type { AssembledGarment } from "../garment/assembly.js";
import type { Vec2 } from "../cad/geom.js";

export interface PanelUV {
  panelId: string;
  /** n*2, vertex order matches the panel's assembled vertex range. */
  uv: Float32Array;
  /** Pattern-space origin of UV (0,0) and axes (for mapping guides). */
  origin: Vec2;
  uAxis: Vec2;
  vAxis: Vec2;
  /** Grain angle (rad, pattern space) the frame was aligned to, if any. */
  grainRad: number | null;
  /** Normalization extents (exact inverse mapping). */
  min: Vec2;
  size: Vec2;
  center: Vec2;
}

function panelRange(assembled: AssembledGarment, panelId: string) {
  const range = assembled.panelRanges.find((r) => r.panelId === panelId);
  if (!range) throw new PatternCadError("missing-reference", `panel '${panelId}' not in garment`, panelId);
  return range;
}

/**
 * Generate canonical panel UVs from pattern-space coordinates in a single
 * deterministic pass: bbox, optional grain-aligned rotation about the bbox
 * center (grain maps to +V), then renormalize to [0,1]².
 */
export function generatePanelUVs(
  assembled: AssembledGarment,
  panelId: string,
  opts: { grainAlign?: boolean; grainRad?: number } = {},
): PanelUV {
  const range = panelRange(assembled, panelId);
  const n = range.vertexCount;
  const local: Vec2[] = [];
  for (let i = 0; i < n; i++) {
    const g = range.vertexStart + i;
    local.push([assembled.uv[g * 2], assembled.uv[g * 2 + 1]]);
  }
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const p of local) {
    if (p[0] < minX) minX = p[0];
    if (p[0] > maxX) maxX = p[0];
    if (p[1] < minY) minY = p[1];
    if (p[1] > maxY) maxY = p[1];
  }
  const dx = maxX - minX, dy = maxY - minY;
  if (!(dx > 0) || !(dy > 0)) {
    throw new PatternCadError("degenerate-panel", `panel '${panelId}' has zero pattern extent`, panelId);
  }
  let angle = 0;
  let grainRad: number | null = null;
  if (opts.grainAlign) {
    grainRad = opts.grainRad ?? 0;
    angle = Math.PI / 2 - grainRad;
  }
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  const c = Math.cos(angle), s = Math.sin(angle);
  const rotated = local.map((p) => {
    const x = p[0] - cx, y = p[1] - cy;
    return [c * x - s * y, s * x + c * y] as Vec2;
  });
  let rMinX = Infinity, rMaxX = -Infinity, rMinY = Infinity, rMaxY = -Infinity;
  for (const p of rotated) {
    if (p[0] < rMinX) rMinX = p[0];
    if (p[0] > rMaxX) rMaxX = p[0];
    if (p[1] < rMinY) rMinY = p[1];
    if (p[1] > rMaxY) rMaxY = p[1];
  }
  const rdx = rMaxX - rMinX || 1, rdy = rMaxY - rMinY || 1;
  const uv = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    uv[i * 2] = (rotated[i][0] - rMinX) / rdx;
    uv[i * 2 + 1] = (rotated[i][1] - rMinY) / rdy;
  }
  return {
    panelId, uv,
    origin: [rMinX + cx, rMinY + cy],
    uAxis: [c, s],
    vAxis: [-s, c],
    grainRad,
    min: [rMinX, rMinY],
    size: [rdx, rdy],
    center: [cx, cy],
  };
}

/** Map a pattern-space point through a panel UV frame (exact inverse). */
export function mapPointToUV(frame: PanelUV, patternXY: Vec2): Vec2 {
  const angle = frame.grainRad === null ? 0 : Math.PI / 2 - frame.grainRad;
  const c = Math.cos(angle), s = Math.sin(angle);
  const dx = patternXY[0] - frame.center[0], dy = patternXY[1] - frame.center[1];
  const rx = c * dx - s * dy, ry = s * dx + c * dy;
  return [(rx - frame.min[0]) / frame.size[0], (ry - frame.min[1]) / frame.size[1]];
}

/** Closed UV loop of a panel boundary (overlays, topstitch guides). */
export function boundaryUVLoop(frame: PanelUV, boundaryPattern: Vec2[]): Vec2[] {
  return boundaryPattern.map((p) => mapPointToUV(frame, p));
}
