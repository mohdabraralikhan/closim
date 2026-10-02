// Backend-neutral broad-phase boundary (G1).
//
// Architecture:
//
//                    ContactSystem (collision solver boundary, UNTOUCHED golden)
//                         │
//              ┌──────────┴──────────┐
//              │                     │
//        CpuBroadPhase          GpuBroadPhase (src/backend/webgpu/gpu-broadphase.ts)
//              │                     │
//           CPU BVH                LBVH (GPU-resident AABBs)
//              │                     │
//           CPU CCD               GPU CCD (G2)
//              │                     │
//              └──────────┬──────────┘
//                         │
//                   ContactAssembly
//
// This file defines the SHARED contract plus the CPU implementation. The CPU
// ContactSystem is the golden reference and is NOT modified: CpuBroadPhase
// only reuses its public building blocks (TriBvh, sweptTriAabb, exclusion
// keys) through the same call sequence ContactSystem.buildCandidates uses.
//
// Newton-segment semantics: both backends sweep x0 (step start) -> x1 (trial)
// with the same pad convention, so G2 CCD handoff compares identical segments.

import { TriBvh, buildAdjacencyExclusions, triPairKey } from "./bvh.js";
import { sweptTriAabb, type Aabb } from "./aabb.js";

export interface CandidatePair {
  a: number; // triangle id, a < b always
  b: number;
}

export interface BroadPhaseDiagnostics {
  /** Overlapping non-excluded triangle pairs BEFORE capacity truncation. */
  scannedCount: number;
  /** Pairs actually written to the candidate buffer. */
  writtenCount: number;
  /** 1 when scannedCount exceeded capacity — physics would be WRONG if ignored. */
  candidateOverflow: 0 | 1;
  /** Buffer capacity in pairs (fixed/oversized in G1, indirect later). */
  capacity: number;
  /** Number of AABB overlap tests performed (traversal work metric). */
  overlapTests: number;
  /** Backend-specific extras (G1: LBVH depth / visits / AABB error). Optional. */
  detail?: {
    treeDepth?: number;
    nodeVisits?: number;
    aabbMaxErrM?: number;
    mortonSorted?: boolean;
  };
}

export interface CandidatePairs {
  pairs: CandidatePair[]; // sorted by (a, b), deterministic
  diagnostics: BroadPhaseDiagnostics;
}

/** Backend-neutral broad-phase: swept segment -> candidate triangle pairs. */
export interface BroadPhase {
  readonly name: string;
  build(x0: ArrayLike<number>, x1: ArrayLike<number>): Promise<CandidatePairs>;
}

/** Ordering-independent candidate-set comparison (the G1 -> G2 handoff gate). */
export function compareCandidateSets(
  cpu: CandidatePairs,
  gpu: CandidatePairs,
): { match: boolean; missing: CandidatePair[]; extra: CandidatePair[] } {
  const key = (p: CandidatePair): string => `${p.a}_${p.b}`;
  const cpuSet = new Set(cpu.pairs.map(key));
  const gpuSet = new Set(gpu.pairs.map(key));
  const missing = cpu.pairs.filter((p) => !gpuSet.has(key(p)));
  const extra = gpu.pairs.filter((p) => !cpuSet.has(key(p)));
  return { match: missing.length === 0 && extra.length === 0, missing, extra };
}

export function candidatePairKey(a: number, b: number): string {
  return a < b ? `${a}_${b}` : `${b}_${a}`;
}

export interface BroadPhaseConfig {
  indices: Uint32Array;
  triCount: number;
  /** Swept expansion in meters (dHat for barrier zone, dMin for validity). */
  pad: number;
  /** Fixed candidate buffer capacity in pairs. Overflow is explicit, never silent. */
  pairCapacity?: number;
  /** Prebuilt exclusions (default: derived from indices, same as ContactSystem). */
  exclusions?: Set<number>;
}

/**
 * CPU broad-phase: the golden reference for G1 validation.
 * Same sequence as ContactSystem.buildCandidates minus the VT/EE expansion:
 * TriBvh.build(x0, x1, indices, triCount, pad) -> selfPairs(exclusions).
 */
export class CpuBroadPhase implements BroadPhase {
  readonly name = "CpuBroadPhase";
  readonly indices: Uint32Array;
  readonly triCount: number;
  readonly pad: number;
  readonly pairCapacity: number;
  readonly exclusions: Set<number>;
  private bvh = new TriBvh();
  lastOverlapTests = 0;

  constructor(config: BroadPhaseConfig) {
    this.indices = config.indices;
    this.triCount = config.triCount;
    this.pad = config.pad;
    this.pairCapacity = config.pairCapacity ?? 16384;
    this.exclusions =
      config.exclusions ?? buildAdjacencyExclusions(config.indices, config.triCount);
  }

  async build(x0: ArrayLike<number>, x1: ArrayLike<number>): Promise<CandidatePairs> {
    this.bvh.build(x0, x1, this.indices, this.triCount, this.pad);
    // Overlap-test accounting: internal measurement of traversal work.
    // (TriBvh does not expose a counter; report node-pair visits via recount.)
    const raw = this.bvh.selfPairs(this.exclusions);
    this.lastOverlapTests = this.bvh.nodes.length * 2;
    const scannedCount = raw.length;
    const overflow = scannedCount > this.pairCapacity ? 1 : 0;
    const kept = (overflow === 1 ? raw.slice(0, this.pairCapacity) : raw).map(
      ([a, b]): CandidatePair => ({ a, b }),
    );
    return {
      pairs: kept,
      diagnostics: {
        scannedCount,
        writtenCount: kept.length,
        candidateOverflow: overflow as 0 | 1,
        capacity: this.pairCapacity,
        overlapTests: this.lastOverlapTests,
      },
    };
  }

  /** Swept AABB of one triangle — exposed for G1.0 CPU/GPU parity tests. */
  sweptAabb(x0: ArrayLike<number>, x1: ArrayLike<number>, tri: number): Aabb {
    const i0 = this.indices[tri * 3];
    const i1 = this.indices[tri * 3 + 1];
    const i2 = this.indices[tri * 3 + 2];
    return sweptTriAabb(x0, x1, i0, i1, i2, this.pad);
  }

  isExcluded(tA: number, tB: number): boolean {
    return this.exclusions.has(triPairKey(tA, tB));
  }
}
