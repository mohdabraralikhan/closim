// Shared Phase-1 test helpers: patch builders + merge + min-distance audit.
import { buildGrid, preprocess, type ClothMeshData } from "../src/mesh/mesh.js";
import { closestPointVertexTriangle, closestPointEdgeEdge } from "../src/collision/closest-point.js";
import { buildEdges } from "../src/collision/self-collision.js";

export function offsetMesh(
  nx: number, ny: number, w: number, h: number,
  ox: number, oy: number, oz: number,
): { positions: Float32Array; uv: Float32Array; indices: Uint32Array } {
  const g = buildGrid(nx, ny, w, h);
  for (let i = 0; i < g.positions.length / 3; i++) {
    g.positions[i * 3] += ox;
    g.positions[i * 3 + 1] += oy;
    g.positions[i * 3 + 2] += oz;
  }
  return g;
}

export function mergeParts(
  parts: Array<{ positions: Float32Array; uv: Float32Array; indices: Uint32Array }>,
): { positions: Float32Array; uv: Float32Array; indices: Uint32Array } {
  let nv = 0, ni = 0;
  for (const p of parts) { nv += p.positions.length / 3; ni += p.indices.length; }
  const positions = new Float32Array(nv * 3);
  const uv = new Float32Array(nv * 2);
  const indices = new Uint32Array(ni);
  let vo = 0, io = 0;
  for (const p of parts) {
    const n = p.positions.length / 3;
    positions.set(p.positions, vo * 3);
    uv.set(p.uv, vo * 2);
    for (let k = 0; k < p.indices.length; k++) indices[io + k] = p.indices[k] + vo;
    vo += n; io += p.indices.length;
  }
  return { positions, uv, indices };
}

export function preprocessMerged(
  parts: Array<{ positions: Float32Array; uv: Float32Array; indices: Uint32Array }>,
  arealDensity: number,
): ClothMeshData {
  const m = mergeParts(parts);
  return preprocess(m.positions, m.uv, m.indices, arealDensity);
}

/** Brute-force minimum VT/EE distance over non-adjacent pairs (audit only). */
export function auditMinDistance(x: ArrayLike<number>, mesh: ClothMeshData, excludeAdjacent: boolean): number {
  const idx = mesh.indices;
  const m = mesh.triCount;
  const adj = new Set<string>();
  if (excludeAdjacent) {
    const v2t = new Map<number, number[]>();
    for (let t = 0; t < m; t++) {
      for (let k = 0; k < 3; k++) {
        const v = idx[t * 3 + k];
        if (!v2t.has(v)) v2t.set(v, []);
        v2t.get(v)!.push(t);
      }
    }
    for (const list of v2t.values()) {
      for (let i = 0; i < list.length; i++) {
        for (let j = i + 1; j < list.length; j++) {
          adj.add(`${Math.min(list[i], list[j])}_${Math.max(list[i], list[j])}`);
        }
      }
    }
  }
  const P = (v: number): [number, number, number] => [x[v * 3], x[v * 3 + 1], x[v * 3 + 2]];
  let minD = Infinity;
  for (let tA = 0; tA < m; tA++) {
    for (let tB = tA + 1; tB < m; tB++) {
      if (excludeAdjacent && adj.has(`${tA}_${tB}`)) continue;
      const A = [idx[tA * 3], idx[tA * 3 + 1], idx[tA * 3 + 2]];
      const B = [idx[tB * 3], idx[tB * 3 + 1], idx[tB * 3 + 2]];
      for (const p of A) {
        const [px, py, pz] = P(p);
        const [ax, ay, az] = P(B[0]); const [bx, by, bz] = P(B[1]); const [cx, cy, cz] = P(B[2]);
        const d = closestPointVertexTriangle(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz).dist;
        if (d < minD) minD = d;
      }
      for (const p of B) {
        const [px, py, pz] = P(p);
        const [ax, ay, az] = P(A[0]); const [bx, by, bz] = P(A[1]); const [cx, cy, cz] = P(A[2]);
        const d = closestPointVertexTriangle(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz).dist;
        if (d < minD) minD = d;
      }
    }
  }
  // EE over unique edges
  const edges = buildEdges(idx);
  for (let i = 0; i < edges.length; i++) {
    for (let j = i + 1; j < edges.length; j++) {
      const [a, b] = edges[i]; const [c, d] = edges[j];
      if (a === c || a === d || b === c || b === d) continue;
      const [ax, ay, az] = P(a); const [bx, by, bz] = P(b);
      const [cx, cy, cz] = P(c); const [dx, dy, dz] = P(d);
      const dd = closestPointEdgeEdge(ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz).dist;
      if (dd < minD) minD = dd;
    }
  }
  return minD;
}
