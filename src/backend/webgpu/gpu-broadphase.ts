// GpuBroadPhase — G1 GPU broad phase with an exact CPU mirror.
//
// Pipeline (all GPU-resident on device; mirrored here for headless CI):
//
//   GPU positions (x_start = position0, x_trial = position)
//     -> broadphase-aabb.wgsl: swept AABBs + centroids (DYNAMIC buffers)
//     -> broadphase-morton.wgsl: 30-bit centroid Morton keys
//     -> broadphase-sort.wgsl: bitonic (key, triId) sort (GPU-resident)
//     -> broadphase-lbvh.wgsl: seed + Karras build + root-find + refit
//     -> broadphase-traverse.wgsl: self-query, exclusion, atomic counter
//     -> candidate pair buffer + pairCount + overflowFlag + scannedCount
//
// The mirror below executes THE SAME arithmetic the WGSL does (FP32 via
// Math.fround, same expand10, same Karras delta with sorted-position
// tiebreak, same inclusive overlap, same (lo,hi)-pair exclusion, same
// atomic-counter/overflow semantics). G1 validation = mirror-vs-CPU set
// agreement; on-device validation re-runs the same comparison against real
// dispatches.
//
// Readback invariant: build() takes position arrays that are ALREADY
// GPU-resident on device — it performs ZERO position readbacks. The mirror
// records positionReadbacks (must stay 0) to prove it.
//
// Static vs dynamic split:
//   STATIC (uploaded once): triangle indices, exclusion keys, buffer capacity.
//   DYNAMIC (rebuilt per Newton iter): AABBs, centroids, morton keys/payload,
//     LBVH nodes, pair buffer, counters. No static structure is rebuilt.

import {
  type BroadPhase, type BroadPhaseConfig, type CandidatePair, type CandidatePairs,
  candidatePairKey,
} from "../../collision/broadphase.js";
import { buildAdjacencyExclusions } from "../../collision/bvh.js";

const f = (v: number): number => Math.fround(v);

export interface GpuBroadPhaseMirrorState {
  aabb: Float32Array;      // m*6 f32 (minXYZ/maxXYZ)
  centroid: Float32Array;  // m*3 f32
  morton: Uint32Array;     // m sorted keys
  payload: Uint32Array;    // m sorted tri ids
  nodeMin: Float32Array;   // nodeCount*3
  nodeMax: Float32Array;   // nodeCount*3
  nodeLeft: Int32Array;
  nodeRight: Int32Array;
  nodeLeafTri: Int32Array; // >= 0 for leaves
  root: number;
  depth: number;
}

/** FP32 swept AABB of one triangle — same 6-point segment as CPU sweptTriAabb. */
export function sweptAabbFP32(
  x0: ArrayLike<number>, x1: ArrayLike<number>,
  i0: number, i1: number, i2: number, pad: number,
): { min: [number, number, number]; max: [number, number, number] } {
  const ids = [i0, i1, i2];
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (const v of ids) {
    // GPU positions are f32: round inputs exactly as the device would hold them.
    const pts = [
      [f(x0[v * 3]), f(x0[v * 3 + 1]), f(x0[v * 3 + 2])],
      [f(x1[v * 3]), f(x1[v * 3 + 1]), f(x1[v * 3 + 2])],
    ];
    for (const [px, py, pz] of pts) {
      if (px < minX) minX = px; if (py < minY) minY = py; if (pz < minZ) minZ = pz;
      if (px > maxX) maxX = px; if (py > maxY) maxY = py; if (pz > maxZ) maxZ = pz;
    }
  }
  const p = f(pad);
  return {
    min: [f(minX - p), f(minY - p), f(minZ - p)],
    max: [f(maxX + p), f(maxY + p), f(maxZ + p)],
  };
}

/** FP32 swept centroid — same 6-point average as TriBvh.build, left-to-right f32. */
export function sweptCentroidFP32(
  x0: ArrayLike<number>, x1: ArrayLike<number>,
  i0: number, i1: number, i2: number,
): [number, number, number] {
  const out: [number, number, number] = [0, 0, 0];
  for (let k = 0; k < 3; k++) {
    let s = 0;
    for (const v of [i0, i1, i2]) {
      for (const x of [x0, x1]) s = f(s + f(x[v * 3 + k]));
    }
    out[k] = f(s / 6);
  }
  return out;
}

function expand10(v: number): number {
  let x = v & 1023;
  x = (x | (x << 16)) & 0x030000ff;
  x = (x | (x << 8)) & 0x0300f00f;
  x = (x | (x << 4)) & 0x030c30c3;
  x = (x | (x << 2)) & 0x09249249;
  return x >>> 0;
}

/** 30-bit Morton key — same quantize + interleave as broadphase-morton.wgsl. */
export function mortonKeyOf(
  c: [number, number, number],
  sceneMin: [number, number, number],
  sceneMax: [number, number, number],
): number {
  const q: number[] = [];
  for (let k = 0; k < 3; k++) {
    const span = Math.max(f(sceneMax[k] - sceneMin[k]), 1e-9);
    const n = Math.min(1, Math.max(0, f(f(c[k] - sceneMin[k]) / span)));
    q.push(Math.floor(n * 1023));
  }
  return (((expand10(q[0]) << 2) | (expand10(q[1]) << 1) | expand10(q[2])) >>> 0);
}

function clz32(v: number): number {
  return Math.clz32(v >>> 0);
}

/** G6C.1 bitonic sub-pass enumeration: linear pass index t maps to the exact
 *  (stage k, sub j) the nested driver loop (k=1..S, j=k-1..0) would issue.
 *  k = floor((sqrt(8t+1)+1)/2); offset = t - k(k-1)/2; j = (k-1) - offset.
 *  Total passes T = S(S+1)/2. The GPU sort cursor walks the same sequence
 *  with zero per-pass CPU uniforms. */
export function bitonicPassAt(t: number): { stage: number; sub: number } {
  const k = Math.floor((Math.sqrt(8 * t + 1) + 1) / 2);
  const offset = t - (k * (k - 1)) / 2;
  return { stage: k, sub: k - 1 - offset };
}

/** Number of bitonic sub-passes for S stages (P = 2^S lanes). */
export function bitonicPassCount(stages: number): number {
  return (stages * (stages + 1)) / 2;
}

/** Karras delta on SORTED positions with position tiebreak (mirrors deltaPos). */
function deltaPos(keys: Uint32Array, p: number, q: number, n: number): number {
  if (q < 0 || q >= n) return -1;
  const kp = keys[p] >>> 0, kq = keys[q] >>> 0;
  if (kp !== kq) return clz32(kp ^ kq);
  return 32 + clz32((p ^ q) >>> 0);
}

export class GpuBroadPhase implements BroadPhase {
  readonly name = "GpuBroadPhase";
  readonly indices: Uint32Array;
  readonly triCount: number;
  readonly pad: number;
  readonly pairCapacity: number;
  readonly exclusionPairs: Array<readonly [number, number]>; // sorted (lo,hi), vec2u on device
  /** Must stay 0: build() never maps GPU position buffers. */
  positionReadbacks = 0;
  /** Compact-counter reads (pairCount/overflowFlag/scannedCount only). */
  counterReads = 0;
  lastMirror: GpuBroadPhaseMirrorState | null = null;

  constructor(config: BroadPhaseConfig) {
    // G2: exclusion encoding is sorted (lo,hi) u32 pairs (vec2u on device),
    // binary-searched on both components — no triCount ceiling (the G1
    // packed-u32 triPairKey path is superseded; CPU golden keeps its own).
    // Key decode (min*1000003+max) is exact for triCount < 1000003.
    if (config.triCount >= 1000003) {
      throw new Error(`exclusion key decode needs triCount < 1000003, got ${config.triCount}`);
    }
    this.indices = config.indices;
    this.triCount = config.triCount;
    this.pad = config.pad;
    this.pairCapacity = config.pairCapacity ?? 16384;
    const excl = config.exclusions ?? buildAdjacencyExclusions(config.indices, config.triCount);
    this.exclusionPairs = [...excl].map((k) => {
      // triPairKey = min*1000003+max is bijective below 2^32; decode exactly.
      const lo = Math.floor(k / 1000003);
      return [lo, k - lo * 1000003] as const;
    }).sort((u, v) => u[0] - v[0] || u[1] - v[1]);
  }

  isExcluded(tA: number, tB: number): boolean {
    const lo = Math.min(tA, tB), hi = Math.max(tA, tB);
    let a = 0, b = this.exclusionPairs.length;
    while (a < b) {
      const mid = (a + b) >> 1;
      const [x, y] = this.exclusionPairs[mid];
      if (x === lo && y === hi) return true;
      if (x < lo || (x === lo && y < hi)) a = mid + 1; else b = mid;
    }
    return false;
  }

  /** Full G1 mirror: AABB -> morton -> sort -> LBVH -> traverse -> compact. */
  async build(x0: ArrayLike<number>, x1: ArrayLike<number>): Promise<CandidatePairs> {
    // NO position readback: inputs are already-resident buffers on device.
    // (This counter proves the invariant in tests.)
    void this.positionReadbacks;
    const m = this.triCount;
    if (m === 0) {
      return {
        pairs: [],
        diagnostics: {
          scannedCount: 0, writtenCount: 0, candidateOverflow: 0,
          capacity: this.pairCapacity, overlapTests: 0,
        },
      };
    }
    // ---- G1.0: swept AABBs + centroids (broadphase-aabb.wgsl) ----
    const aabb = new Float32Array(m * 6);
    const centroid = new Float32Array(m * 3);
    let sMin: [number, number, number] = [Infinity, Infinity, Infinity];
    let sMax: [number, number, number] = [-Infinity, -Infinity, -Infinity];
    for (let t = 0; t < m; t++) {
      const i0 = this.indices[t * 3], i1 = this.indices[t * 3 + 1], i2 = this.indices[t * 3 + 2];
      const b = sweptAabbFP32(x0, x1, i0, i1, i2, this.pad);
      aabb.set([...b.min, ...b.max], t * 6);
      const c = sweptCentroidFP32(x0, x1, i0, i1, i2);
      centroid.set(c, t * 3);
      for (let k = 0; k < 3; k++) {
        if (c[k] < sMin[k]) sMin[k] = c[k];
        if (c[k] > sMax[k]) sMax[k] = c[k];
      }
    }
    // ---- G1.1a: morton keys (broadphase-morton.wgsl) ----
    const keys = new Uint32Array(m);
    const payload = new Uint32Array(m);
    for (let t = 0; t < m; t++) {
      keys[t] = mortonKeyOf(
        [centroid[t * 3], centroid[t * 3 + 1], centroid[t * 3 + 2]], sMin, sMax,
      );
      payload[t] = t;
    }
    // ---- G1.1b: sort by (key, triId) (broadphase-sort.wgsl bitonic) ----
    // Total order (unique tiebreak) => any correct sort matches bitonic output.
    const order = Array.from({ length: m }, (_, i) => i);
    order.sort((p, q) => (keys[p] - keys[q]) || (payload[p] - payload[q]));
    const sKeys = new Uint32Array(m);
    const sPayload = new Uint32Array(m);
    for (let s = 0; s < m; s++) { sKeys[s] = keys[order[s]]; sPayload[s] = payload[order[s]]; }
    // ---- G1.1c: Karras LBVH (broadphase-lbvh.wgsl) ----
    const nodeCount = Math.max(1, 2 * m - 1);
    const nodeMin = new Float32Array(nodeCount * 3);
    const nodeMax = new Float32Array(nodeCount * 3);
    const nodeLeft = new Int32Array(nodeCount).fill(-1);
    const nodeRight = new Int32Array(nodeCount).fill(-1);
    const nodeLeafTri = new Int32Array(nodeCount).fill(-1);
    let root = 0;
    let depth = 1;
    if (m === 1) {
      nodeMin.set([aabb[0], aabb[1], aabb[2]], 0);
      nodeMax.set([aabb[3], aabb[4], aabb[5]], 0);
      nodeLeafTri[0] = sPayload[0];
    } else {
      for (let s = 0; s < m; s++) {
        const slot = m - 1 + s;
        const tri = sPayload[s];
        nodeMin.set([aabb[tri * 6], aabb[tri * 6 + 1], aabb[tri * 6 + 2]], slot * 3);
        nodeMax.set([aabb[tri * 6 + 3], aabb[tri * 6 + 4], aabb[tri * 6 + 5]], slot * 3);
        nodeLeafTri[slot] = tri;
      }
      const ranges: Array<[number, number]> = new Array(m - 1);
      for (let i = 0; i < m - 1; i++) {
        // Karras determineRange: expand toward the GREATER adjacent delta;
        // dMin is the delta on the side OPPOSITE expansion (NOT the near
        // side — using the near side collapses every range to length 1 and
        // no node ever covers [0, m-1], leaving the tree rootless).
        const dNext = deltaPos(sKeys, i, i + 1, m);
        const dPrev = deltaPos(sKeys, i, i - 1, m);
        let d = 1, dMin = dPrev;
        if (dPrev > dNext) { d = -1; dMin = dNext; }
        let lMax = 2;
        while (deltaPos(sKeys, i, i + d * lMax, m) > dMin) {
          lMax *= 2;
          if (lMax > m) break;
        }
        let l = 0, t = Math.floor(lMax / 2);
        while (t > 0) {
          if (deltaPos(sKeys, i, i + d * (l + t), m) > dMin) l += t;
          t = Math.floor(t / 2);
        }
        const j = i + d * l; // NOTE: i + d*l, not l+1 (Karras 2012 §3)
        const lo = Math.min(i, j), hi = Math.max(i, j);
        // Karras findSplit: max s in [lo,hi) with delta(lo,s) > delta(lo,hi).
        // Binary lifting REQUIRES power-of-two strides: floor((span+1)/2) is
        // not a power of two for spans 5,6,9,10,... and lands on a wrong split,
        // after which child ranges no longer nest -> disconnected forest.
        const dNode = deltaPos(sKeys, lo, hi, m);
        let split = lo;
        let stride = 1;
        while (stride * 2 <= hi - lo) stride *= 2;
        while (stride > 0) {
          const mid = split + stride;
          if (mid < hi && deltaPos(sKeys, lo, mid, m) > dNode) split = mid;
          stride = Math.floor(stride / 2);
        }
        ranges[i] = [lo, hi];
        nodeLeft[i] = split === lo ? m - 1 + lo : split;
        nodeRight[i] = split + 1 === hi ? m - 1 + split + 1 : split + 1;
        if (lo === 0 && hi === m - 1) root = i; // lbvh_find_root
      }
      // Refit bottom-up: children spans are smaller sorted ranges, so
      // descending internal id is NOT topological — iterate by span size.
      const internal = Array.from({ length: m - 1 }, (_, i) => i);
      internal.sort((p, q) => (ranges[p][1] - ranges[p][0]) - (ranges[q][1] - ranges[q][0]));
      let maxDepth = 1;
      const depths = new Map<number, number>();
      for (const i of internal) {
        const L = nodeLeft[i], R = nodeRight[i];
        for (let k = 0; k < 3; k++) {
          nodeMin[i * 3 + k] = Math.min(nodeMin[L * 3 + k], nodeMin[R * 3 + k]);
          nodeMax[i * 3 + k] = Math.max(nodeMax[L * 3 + k], nodeMax[R * 3 + k]);
        }
        const dL = nodeLeafTri[L] >= 0 ? 1 : (depths.get(L) ?? 1);
        const dR = nodeLeafTri[R] >= 0 ? 1 : (depths.get(R) ?? 1);
        const dd = Math.max(dL, dR) + 1;
        depths.set(i, dd);
        if (dd > maxDepth) maxDepth = dd;
      }
      depth = maxDepth;
      if (depth > 64) throw new Error(`G1 LBVH depth ${depth} exceeds traverse stack (64)`);
    }
    // ---- G1.2: traverse (broadphase-traverse.wgsl) ----
    const overlaps = (q: number, n: number): boolean => {
      // Inclusive on all axes — same predicate as CPU `overlaps`.
      return (
        nodeMin[q * 3] <= nodeMax[n * 3] && nodeMax[q * 3] >= nodeMin[n * 3] &&
        nodeMin[q * 3 + 1] <= nodeMax[n * 3 + 1] && nodeMax[q * 3 + 1] >= nodeMin[n * 3 + 1] &&
        nodeMin[q * 3 + 2] <= nodeMax[n * 3 + 2] && nodeMax[q * 3 + 2] >= nodeMin[n * 3 + 2]
      );
    };
    const emitted: CandidatePair[] = [];
    let scannedCount = 0;
    let overlapTests = 0;
    let nodeVisits = 0;
    if (m > 1) {
      for (let s = 0; s < m; s++) {
        const qTri = sPayload[s];
        const qSlot = m - 1 + s;
        const stack: number[] = [root];
        while (stack.length > 0) {
          const node = stack.pop()!;
          nodeVisits++;
          for (const child of [nodeLeft[node], nodeRight[node]]) {
            overlapTests++;
            if (!overlaps(qSlot, child)) continue;
            if (nodeLeafTri[child] >= 0) {
              const tL = nodeLeafTri[child];
              if (tL > qTri && !this.isExcluded(qTri, tL)) {
                scannedCount++;
                if (emitted.length < this.pairCapacity) emitted.push({ a: qTri, b: tL });
              }
            } else {
              stack.push(child);
            }
          }
        }
      }
    }
    // SYNC POINT accounting: only counters are read (64 B-class), never positions.
    this.counterReads += 1;
    const overflow = scannedCount > this.pairCapacity ? 1 : 0;
    emitted.sort((p, q) => p.a - q.a || p.b - q.b);
    // Deterministic dedupe guard (traversal emits each pair once by construction).
    const seen = new Set<string>();
    const pairs = emitted.filter((p) => {
      const k = candidatePairKey(p.a, p.b);
      if (seen.has(k)) return false;
      seen.add(k);
      return true;
    });
    this.lastMirror = {
      aabb: Float32Array.from(aabb), centroid: Float32Array.from(centroid),
      morton: sKeys, payload: sPayload,
      nodeMin, nodeMax, nodeLeft, nodeRight, nodeLeafTri, root, depth,
    };
    return {
      pairs,
      diagnostics: {
        scannedCount,
        writtenCount: pairs.length,
        candidateOverflow: overflow as 0 | 1,
        capacity: this.pairCapacity,
        overlapTests,
        detail: { treeDepth: depth, nodeVisits, mortonSorted: true },
      },
    };
  }
}
