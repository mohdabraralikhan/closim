// Binary median-split BVH over cloth triangles, rebuilt per evaluate (deterministic).
// Leaves hold triangle ids; internal nodes hold AABBs. Deterministic: stable
// sort by centroid along longest axis, no randomness.

import { sweptTriAabb, overlaps, type Aabb } from "./aabb.js";

interface Node {
  aabb: Aabb;
  left: number; // child indices into nodes, -1 for leaf
  right: number;
  tri: number; // triangle id if leaf, else -1
}

export class TriBvh {
  nodes: Node[] = [];
  root = -1;

  /** Build over swept AABBs of all triangles between x0 and x1 (+pad). */
  build(
    x0: ArrayLike<number>, x1: ArrayLike<number>,
    indices: Uint32Array, triCount: number, pad: number,
  ): void {
    const boxes: Aabb[] = new Array(triCount);
    const cx = new Float64Array(triCount);
    const cy = new Float64Array(triCount);
    const cz = new Float64Array(triCount);
    for (let t = 0; t < triCount; t++) {
      const i0 = indices[t * 3], i1 = indices[t * 3 + 1], i2 = indices[t * 3 + 2];
      boxes[t] = sweptTriAabb(x0, x1, i0, i1, i2, pad);
      cx[t] = (x0[i0 * 3] + x0[i1 * 3] + x0[i2 * 3] + x1[i0 * 3] + x1[i1 * 3] + x1[i2 * 3]) / 6;
      cy[t] = (x0[i0 * 3 + 1] + x0[i1 * 3 + 1] + x0[i2 * 3 + 1] + x1[i0 * 3 + 1] + x1[i1 * 3 + 1] + x1[i2 * 3 + 1]) / 6;
      cz[t] = (x0[i0 * 3 + 2] + x0[i1 * 3 + 2] + x0[i2 * 3 + 2] + x1[i0 * 3 + 2] + x1[i1 * 3 + 2] + x1[i2 * 3 + 2]) / 6;
    }
    this.nodes = [];
    if (triCount === 0) {
      this.root = -1;
      return;
    }
    const order = Array.from({ length: triCount }, (_, i) => i);
    this.root = this.buildRecursive(order, boxes, cx, cy, cz);
  }

  private buildRecursive(order: number[], boxes: Aabb[], cx: Float64Array, cy: Float64Array, cz: Float64Array): number {
    if (order.length === 1) {
      const id = this.nodes.length;
      this.nodes.push({ aabb: boxes[order[0]], left: -1, right: -1, tri: order[0] });
      return id;
    }
    // longest axis of centroid spread
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity;
    for (const t of order) {
      if (cx[t] < minX) minX = cx[t];
      if (cx[t] > maxX) maxX = cx[t];
      if (cy[t] < minY) minY = cy[t];
      if (cy[t] > maxY) maxY = cy[t];
      if (cz[t] < minZ) minZ = cz[t];
      if (cz[t] > maxZ) maxZ = cz[t];
    }
    const sx = maxX - minX, sy = maxY - minY, sz = maxZ - minZ;
    const axis = sx >= sy && sx >= sz ? 0 : sy >= sz ? 1 : 2;
    const key = axis === 0 ? cx : axis === 1 ? cy : cz;
    order.sort((p, q) => key[p] - key[q] || p - q);
    const mid = Math.max(1, Math.floor(order.length / 2));
    const left = this.buildRecursive(order.slice(0, mid), boxes, cx, cy, cz);
    const right = this.buildRecursive(order.slice(mid), boxes, cx, cy, cz);
    const a = this.nodes[left].aabb, b = this.nodes[right].aabb;
    const id = this.nodes.length;
    this.nodes.push({
      aabb: {
        minX: Math.min(a.minX, b.minX), minY: Math.min(a.minY, b.minY), minZ: Math.min(a.minZ, b.minZ),
        maxX: Math.max(a.maxX, b.maxX), maxY: Math.max(a.maxY, b.maxY), maxZ: Math.max(a.maxZ, b.maxZ),
      },
      left, right, tri: -1,
    });
    return id;
  }

  /** All overlapping triangle pairs (tA < tB), deduplicated, sorted. */
  selfPairs(excluded: Set<number>): Array<[number, number]> {
    const out: Array<[number, number]> = [];
    if (this.root < 0) return out;
    const stack: Array<[number, number]> = [[this.root, this.root]];
    while (stack.length > 0) {
      const [na, nb] = stack.pop()!;
      const A = this.nodes[na], B = this.nodes[nb];
      if (!overlaps(A.aabb, B.aabb)) continue;
      const leafA = A.tri >= 0, leafB = B.tri >= 0;
      if (leafA && leafB) {
        // G1 audit fix: the traversal visits each node-pair in ONE orientation
        // ([X,Y] with X pre-order before Y — reverse encounters are never
        // generated), so `tA >= tB -> skip` silently DROPPED cross-subtree pairs
        // whose left-subtree triangle held the larger id (e.g. tris {1,2} met
        // only as encounter (2,1)). Normalize instead: each unordered leaf pair
        // is still encountered exactly once (pre-order invariant), so no
        // duplicates arise and no previously-found pair changes.
        const a = A.tri, b = B.tri;
        if (a === b) continue; // identical leaf
        const tA = Math.min(a, b), tB = Math.max(a, b);
        const key = tA * 1000003 + tB;
        if (excluded.has(key)) continue;
        out.push([tA, tB]);
        continue;
      }
      if (na === nb) {
        // same internal node: expand pairs of children
        stack.push([A.left, A.left]);
        stack.push([A.left, A.right]);
        stack.push([A.right, A.right]);
      } else if (leafA) {
        stack.push([na, B.left]);
        stack.push([na, B.right]);
      } else if (leafB) {
        stack.push([A.left, nb]);
        stack.push([A.right, nb]);
      } else {
        stack.push([A.left, B.left]);
        stack.push([A.left, B.right]);
        stack.push([A.right, B.left]);
        stack.push([A.right, B.right]);
      }
    }
    out.sort((p, q) => p[0] - q[0] || p[1] - q[1]);
    return out;
  }
}

/** Adjacent-triangle exclusion key (order independent, tri ids). */
export function triPairKey(tA: number, tB: number): number {
  return Math.min(tA, tB) * 1000003 + Math.max(tA, tB);
}

/** Build exclusion set: triangle pairs sharing an edge OR a vertex. */
export function buildAdjacencyExclusions(indices: Uint32Array, triCount: number): Set<number> {
  const vertToTris = new Map<number, number[]>();
  for (let t = 0; t < triCount; t++) {
    for (let k = 0; k < 3; k++) {
      const v = indices[t * 3 + k];
      if (!vertToTris.has(v)) vertToTris.set(v, []);
      vertToTris.get(v)!.push(t);
    }
  }
  const excluded = new Set<number>();
  for (const list of vertToTris.values()) {
    for (let i = 0; i < list.length; i++) {
      for (let j = i + 1; j < list.length; j++) {
        excluded.add(triPairKey(list[i], list[j]));
      }
    }
  }
  return excluded;
}
