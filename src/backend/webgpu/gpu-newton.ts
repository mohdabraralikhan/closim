// DeviceNewtonDriver — G3 full on-device Newton evaluation.
//
// One driver call = one Newton iterate evaluated END-TO-END on the device:
// G1 broadphase -> G2 CCD/compaction -> FEM/bending -> barrier/friction ->
// gradient -> RHS/Jacobi -> HVP -> GPU-PCG -> trial -> diagnostics, with the
// CPU seeing ONLY tiny scalars (4 B counters/dots), the 64 B status, and the
// 16 B exec marker. Positions/gradients/contacts/vectors NEVER cross.
//
// Per-Armijo-trial contact rebuild: every alpha changes the segment, so the
// driver re-runs G1+G2 + FEM-at-trial + diagnostics for each trial. The CPU
// Armijo decision (validity + sufficient decrease) reads only the status.
//
// PCG is FULLY GPU-side in ONE submit per solve: HVP (FD via axpy_vec4 +
// membrane/bending-free combine), scalar slots (alpha/beta/rzPrev) written by
// sdiv/scopy, breakdown latched by sflag_le, final residual + flag mapped
// once. Fixed-K iterations (no mid-solve convergence check); the driver
// reports the final residual and the CPU applies the descent fallback
// (neg_div, device-side) exactly like newton.ts.

import { GpuExecutor, STAGE_IDS } from "./gpu-executor.js";
import { GpuUniformSlot, UNIFORM_SLOT_STRIDE, nextPow2 } from "./gpu-buffers.js";
import { decodeSolverStatus, type SolverStatus } from "./gpu-buffers.js";
import type { BindEntry } from "./gpu-executor.js";

/** G6A production preconditioner modes. Default stays "jacobi"; c1-8 is an
 *  explicit candidate until the submit-cap rebaseline completes. */
export type PreconditionerMode = "jacobi" | "schwarz1" | "mas-c1-8";

/** Fallback record: every method change records requested/actual/reason. */
export interface PreconditionerFallback {
  requested: string;
  actual: string;
  reason: string;
}

/**
 * Map a production mode onto driver flags (pure; unit-tested). G6B batched
 * Armijo is orthogonal and composed separately in stepGpuDevice.
 */
export function applyPreconditionerMode(
  cfg: DeviceNewtonConfig, mode: PreconditionerMode,
): void {
  cfg.useBlockJacobi = false;
  cfg.useSchwarz = false;
  cfg.useMas = false;
  cfg.useCoarseC0 = false;
  cfg.useCoarsePcg = false;
  if (mode === "schwarz1") cfg.useSchwarz = true;
  else if (mode === "mas-c1-8") {
    cfg.useSchwarz = true; // fine level always on under coarse paths
    cfg.useCoarsePcg = true;
    cfg.coarseIters = 8;
  }
}

/**
 * Resolve effective flags with graceful degradation (pure; unit-tested).
 * Missing Schwarz topology (schwarzDoms <= 0) falls back toward Jacobi and
 * records each step down. Returns the fallback record or null when the
 * requested path is fully available.
 */
export function resolvePreconditioner(
  cfg: DeviceNewtonConfig, schwarzDoms: number,
): PreconditionerFallback | null {
  const requested = describeFlags(cfg);
  if (schwarzDoms > 0) return null;
  if (cfg.useCoarsePcg || cfg.useCoarseC0 || cfg.useMas || cfg.useSchwarz || cfg.useBlockJacobi) {
    cfg.useCoarsePcg = false;
    cfg.useCoarseC0 = false;
    cfg.useMas = false;
    cfg.useSchwarz = false;
    cfg.useBlockJacobi = false;
    return { requested, actual: "jacobi", reason: "missing-topology" };
  }
  return null;
}

function describeFlags(cfg: DeviceNewtonConfig): string {
  if (cfg.useCoarsePcg) return `mas-c1-${cfg.coarseIters}`;
  if (cfg.useCoarseC0) return "mas-c0";
  if (cfg.useMas) return "mas2";
  if (cfg.useSchwarz) return "schwarz1";
  if (cfg.useBlockJacobi) return "block";
  return "jacobi";
}

/** G6B compact batch status (one 64 B readback per batch).
 *  selectedIndex is -1 when the batch rejects everything. */
export interface ArmijoBatchStatus {
  accepted: boolean;
  selectedAlpha: number;
  selectedIndex: number;
  trialsEvaluated: number;
  armijoFails: number;
  ccdFails: number;
  barrierFails: number;
  overflowFails: number;
  finite: boolean;
  energy: number;
  gradNorm: number;
  minDist: number;
  minToi: number;
}

export function decodeArmijoStatus(buf: ArrayBuffer): ArmijoBatchStatus {
  const f = new Float32Array(buf);
  return {
    accepted: f[0] > 0.5,
    selectedAlpha: f[1],
    selectedIndex: f[2] > 1e9 ? -1 : Math.round(f[2]),
    trialsEvaluated: Math.round(f[3]),
    armijoFails: Math.round(f[4]),
    ccdFails: Math.round(f[5]),
    barrierFails: Math.round(f[6]),
    overflowFails: Math.round(f[7]),
    finite: f[8] > 0.5,
    energy: f[9],
    gradNorm: f[10],
    minDist: f[13],
    minToi: f[14],
  };
}

/**
 * G6C.3 adaptive batch-width policy (pure; unit-tested). Input is the last
 * accepted global trial index (trialHistory semantics: -1 per failed round,
 * null with no history yet); output is the next batch width in {2, 4, 8}.
 * Rationale from measured distributions: accept-first Newton iters dominate
 * smooth regimes (K=2 halves speculation waste), late accepts cluster on
 * stiff/contact rounds (K=8 preserves coverage), failed rounds keep maximum
 * coverage, and cold start honors the configured default.
 */
export function adaptiveBatchK(lastTrial: number | null, defaultK: number): number {
  const ladder = (k: number): number => (k <= 2 ? 2 : k <= 4 ? 4 : 8);
  if (lastTrial === null) return ladder(defaultK);
  if (lastTrial < 0) return 8;
  if (lastTrial <= 0) return 2;
  if (lastTrial <= 3) return 4;
  return 8;
}

/** G6C.2 compact Newton-round status (one 80 B read per round). */
export interface NewtonRoundStatus {
  iteration: number;
  converged: boolean;
  failure: boolean;
  directionValid: boolean;
  pcgBreakdown: boolean;
  armijoAccepted: boolean;
  batchIndex: number;
  selectedAlpha: number;
  selectedTrialIndex: number;
  gradNorm: number;
  stepNorm: number;
  merit: number;
  energy: number;
  minDistance: number;
  minToi: number;
  contactOverflow: boolean;
  ccdFailure: boolean;
  barrierFailure: boolean;
  residual: number;
  gtdx: number;
}

export function decodeNewtonStatus(buf: ArrayBuffer): NewtonRoundStatus {
  const f = new Float32Array(buf);
  return {
    iteration: Math.round(f[0]),
    converged: f[1] > 0.5,
    failure: f[2] > 0.5,
    directionValid: f[3] > 0.5,
    pcgBreakdown: f[4] > 0.5,
    armijoAccepted: f[5] > 0.5,
    batchIndex: Math.round(f[6]),
    selectedAlpha: f[7],
    selectedTrialIndex: f[8] > 1e9 ? -1 : Math.round(f[8]),
    gradNorm: f[9],
    stepNorm: f[10],
    merit: f[11],
    energy: f[12],
    minDistance: f[13] >= 1e29 ? Infinity : f[13],
    minToi: f[14] >= 2.0 - 1e-6 ? Infinity : f[14],
    contactOverflow: f[15] > 0.5,
    ccdFailure: f[16] > 0.5,
    barrierFailure: f[17] > 0.5,
    residual: f[18],
    gtdx: f[19],
  };
}

export interface DeviceNewtonConfig {
  maxNewton: number;
  pcgIters: number; // fixed K per solve (no mid-solve check)
  gradTol: number;
  armijoIters: number;
  trust: number; // max |dx| per Newton step (mirror of newton.ts TRUST)
  hFd: number; // FD step for membrane HVP (debug oracle path only)
  useAnalyticHvp: boolean; // G4A production path: analytic membrane HVP
  useGatherAssembly: boolean; // G4B production path: CSR gather, no O(n*m) scans
  useBlockJacobi: boolean; // G5B candidate: per-vertex 3x3 blocks (default: scalar Jacobi)
  useSchwarz: boolean; // G5C candidate: one-level additive Schwarz (takes precedence over block Jacobi)
  useMas: boolean; // G5D candidate: two-level MAS (Schwarz-1 + damped-Jacobi coarse; top precedence)
  masOmega: number; // G5D coarse damping (default 0.5)
  useCoarseC0: boolean; // G5.5 C0: Schwarz-1 + coarse block-Jacobi correction
  useCoarsePcg: boolean; // G5.5 C1: Schwarz-1 + inner coarse-PCG correction
  coarseIters: number; // G5.5 C1 inner iterations (4/8/16/32 ladder)
  useBatchedArmijo: boolean; // G6B: GPU batch trial evaluation/selection
  armijoBatchK: number; // G6B candidates per batch (2/4/8 ladder)
  useGpuNewtonControl: boolean; // G6C.2: GPU-side per-round decisions, one status read per round
  adaptiveK: boolean; // G6C.3: batch width from measured trial history (default: fixed armijoBatchK)
  useIndexedSort: boolean; // G6C.1: cursor-stepped sort (no per-pass uniforms)
  pApTol: number; // breakdown threshold (mirror of pcg.ts 1e-30)
  damping: number;
}

export const DEFAULT_NEWTON_CONFIG: DeviceNewtonConfig = {
  maxNewton: 10,
  pcgIters: 60,
  gradTol: 1e-5,
  armijoIters: 10,
  trust: 0.002,
  hFd: 1e-6,
  useAnalyticHvp: true, // G4A production path (FD kept as debug oracle)
  useGatherAssembly: true, // G4B production (scans kept as debug oracle)
  useBlockJacobi: false, // G5B: candidate only until measured better
  useSchwarz: false, // G5C: candidate only until measured better
  useMas: false, // G5D: candidate only until measured better
  masOmega: 0.5, // G5D coarse damping
  useCoarseC0: false, // G5.5 C0: candidate only until measured better
  useCoarsePcg: false, // G5.5 C1: candidate only until measured better
  coarseIters: 8, // G5.5 C1 default rung
  useBatchedArmijo: false, // G6B: candidate only until measured better
  armijoBatchK: 4, // G6B default batch size
  useGpuNewtonControl: false, // G6C.2: candidate only until measured better
  adaptiveK: false, // G6C.3: fixed batch widths until measured better
  useIndexedSort: true, // G6C.1: indexed path proven bit-exact (A/B) with fewer submits
  pApTol: 1e-30,
  damping: 1.0,
};

export interface DeviceCounts {
  n: number; // cloth verts
  nExt: number; // cloth + static verts
  m: number; // tris
  h: number; // hinges
  cap: number; // contact capacity
  pairs: number; // pair capacity
  schwarzDoms: number; // G5C Schwarz domain count (0 = unbuilt)
}

export interface EvalScalars {
  contactCount: number;
}

export interface NewtonStepStats {
  newtonItersUsed: number;
  pcgItersTotal: number;
  converged: boolean;
  acceptedFinal: boolean;
  status: SolverStatus;
  markerMask: number;
  markerCount: number;
  gradNorm0: number;
}

const B = (binding: number, buffer: string, offset?: number, size?: number): BindEntry => ({
  binding, buffer, offset, size,
});

export class DeviceNewtonDriver {
  ex: GpuExecutor;
  cfg: DeviceNewtonConfig;
  c: DeviceCounts;
  onStage: ((stage: number, ms: number) => void) | null = null;

  constructor(ex: GpuExecutor, counts: DeviceCounts, cfg: Partial<DeviceNewtonConfig> = {}) {
    this.ex = ex;
    this.c = counts;
    this.cfg = { ...DEFAULT_NEWTON_CONFIG, ...cfg };
  }

  get n3(): number {
    return this.c.n * 3;
  }

  w(count: number): number {
    return Math.max(1, Math.ceil(count / 64));
  }

  // ---- uniform bank helpers (control transfers, never readback) ----

  bankU(slot: number, v: number): void {
    this.ex.writeBankU32(slot, v);
  }

  bankF(slot: number, v: number): void {
    const a = new Float32Array(4);
    a[0] = v;
    this.ex.writeBuffer(this.ex.uniformBankName, a, slot * UNIFORM_SLOT_STRIDE);
  }

  bankVec4(slot: number, x: number, y: number, z: number, w: number): void {
    this.ex.writeBuffer(this.ex.uniformBankName, new Float32Array([x, y, z, w]), slot * UNIFORM_SLOT_STRIDE);
  }

  bank(slot: number): BindEntry {
    return { binding: -1, buffer: this.ex.uniformBankName, offset: slot * UNIFORM_SLOT_STRIDE, size: 16 };
  }

  private bg(binding: number, slot: number): BindEntry {
    const offset = slot * UNIFORM_SLOT_STRIDE;
    if (offset % 256 !== 0) {
      throw new Error(`DeviceNewtonDriver.bg: slot ${slot} violates 256 B uniform alignment`);
    }
    return { binding, buffer: this.ex.uniformBankName, offset, size: 16 };
  }

  /** Dispatch mark_stage(bit) in the current batch. */
  marksEnabled = true;
  /** G4C: stage bits accumulated since the last flush (mask semantics). */
  private pendingMarks = 0;

  /** Record a stage bit; encoded as one mask dispatch by flushMarks(). */
  mark(bit: number): void {
    if (!this.marksEnabled) return;
    this.pendingMarks |= (1 << bit);
  }

  /** Encode one mark_stage OR-ing all pending bits (no-op when none). */
  flushMarks(): void {
    if (!this.marksEnabled || this.pendingMarks === 0) return;
    this.bankU(GpuUniformSlot.StageBit, this.pendingMarks >>> 0);
    this.pendingMarks = 0;
    this.ex.runPass({
      shader: "marker", entry: "mark_stage",
      groups: [[
        B(0, "execMarker"),
        { binding: 1, buffer: this.ex.uniformBankName, offset: GpuUniformSlot.StageBit * UNIFORM_SLOT_STRIDE, size: 4 },
      ]],
      x: 1,
    });
  }

  // ---- SimParams refresh (64 B control upload) ----

  uploadSimParams(fields: {
    dt: number; gravity: [number, number, number];
    contactCount: number; newtonIteration: number; lineSearchAlpha: number;
    frictionMu: number;
  }): void {
    const buf = new ArrayBuffer(64);
    const f = new Float32Array(buf);
    const u = new Uint32Array(buf);
    f[0] = fields.dt;
    f[1] = 1 / (fields.dt * fields.dt);
    f[2] = fields.gravity[0]; f[3] = fields.gravity[1]; f[4] = fields.gravity[2];
    u[6] = this.c.n; u[7] = this.c.m; u[8] = this.c.h; u[9] = fields.contactCount;
    u[10] = fields.newtonIteration; u[11] = 0;
    f[12] = fields.lineSearchAlpha; f[13] = this.cfg.trust;
    f[14] = 0.002; f[15] = fields.frictionMu;
    this.ex.writeBuffer("simParams", new Uint8Array(buf));
  }

  // ---- G1 broadphase at a state buffer ----

  /** Broadphase with xTrial rebound to the position slot when atTrial=true. */
  broadphasePasses(pad: number, atTrial: boolean): void {
    this.aabbPass(pad, atTrial);
    this.mortonPass();
    this.sortPasses();
    this.lbvhPasses();
    this.traversePass();
  }

  aabbPass(pad: number, atTrial: boolean): void {
    const posBuf = atTrial ? "xTrial" : "position";
    const m = this.c.m;
    this.bankF(GpuUniformSlot.Pad, pad);
    // AABB
    this.ex.runPass({
      shader: "broadphase-aabb", entry: "main",
      groups: [[
        B(0, "simParams"),
        B(1, "position0"),
        B(2, posBuf),
        B(3, "triangles"),
        B(4, "triAabb"),
        B(5, "triCentroid"),
        this.bg(6, GpuUniformSlot.Pad),
      ]],
      x: this.w(m),
    });
    this.mark(STAGE_IDS.aabb);
  }

  mortonPass(): void {
    const m = this.c.m;
    // Morton (fixed canonical scene box [-2,2]^3 — G3.1: adaptive GPU bbox)
    this.bankVec4(GpuUniformSlot.SceneMin, -2, -2, -2, 0);
    this.bankVec4(GpuUniformSlot.SceneMax, 2, 2, 2, 0);
    this.ex.runPass({
      shader: "broadphase-morton", entry: "main",
      groups: [[
        B(0, "simParams"),
        B(1, "triCentroid"),
        B(2, "mortonKeys"),
        B(3, "mortonPayload"),
        this.bg(4, GpuUniformSlot.SceneMin),
        this.bg(5, GpuUniformSlot.SceneMax),
      ]],
      x: this.w(m),
    });
    this.mark(STAGE_IDS.morton);
  }

  sortPasses(): void {
    if (this.cfg.useIndexedSort) {
      this.sortPassesIndexed();
      return;
    }
    const m = this.c.m;
    // bitonic sort over P = nextPow2(m) lanes (tail pre-filled with INF
    // keys once at upload; the network sinks them to [m, P) every rebuild).
    // Standard Batcher stages: run length 2^k, distances 2^(k-1)..1.
    const P = nextPow2(Math.max(m, 1));
    const stages = Math.log2(P);
    for (let k = 1; k <= stages; k++) {
      for (let j = k - 1; j >= 0; j--) {
        this.bankU(GpuUniformSlot.SortN, P);
        this.bankU(GpuUniformSlot.SortStage, k);
        this.bankU(GpuUniformSlot.SortSub, j);
        this.ex.runPass({
          shader: "broadphase-sort", entry: "bitonic_sort_step",
          groups: [[
            B(0, "mortonKeys"),
            B(1, "mortonPayload"),
            this.bg(2, GpuUniformSlot.SortN),
            this.bg(3, GpuUniformSlot.SortStage),
            this.bg(4, GpuUniformSlot.SortSub),
          ]],
          x: this.w(P),
        });
      }
    }
    this.mark(STAGE_IDS.sort);
  }

  /** G6C.1 indexed sort: identical Batcher network, zero per-pass CPU
   *  uniforms. The cursor/triples walk the same (k,j) sequence the legacy
   *  loop would issue (see bitonicPassAt); all T sub-passes stay in flight
   *  with no added submits. sortParams uploaded once per scene. */
  sortPassesIndexed(): void {
    const m = this.c.m;
    const P = nextPow2(Math.max(m, 1));
    const stages = Math.log2(P);
    const T = (stages * (stages + 1)) / 2;
    // Cursor reset (one control write before the batch opens work).
    this.ex.writeBuffer("sortCursor", new Uint32Array([0, 0, 0, 0]));
    for (let t = 0; t < T; t++) {
      this.ex.runPass({
        shader: "broadphase-sort", entry: "sort_next",
        groups: [[B(6, "sortCursor")]],
        x: 1,
      });
      this.ex.runPass({
        shader: "broadphase-sort", entry: "sort_step_indexed",
        groups: [[
          B(0, "mortonKeys"),
          B(1, "mortonPayload"),
          B(5, "sortParams"),
          B(6, "sortCursor"),
        ]],
        x: this.w(P),
      });
    }
    this.mark(STAGE_IDS.sort);
  }

  lbvhPasses(): void {
    const m = this.c.m;
    // LBVH: seed + build + root + refit x D (bottom-up level induction)
    this.ex.runPass({
      shader: "broadphase-lbvh", entry: "lbvh_seed_leaves",
      groups: [[
        B(0, "simParams"),
        B(2, "mortonPayload"),
        B(3, "triAabb"),
        B(4, "lbvhMin"),
        B(5, "lbvhMax"),
        B(6, "lbvhChild"),
      ]],
      x: this.w(m),
    });
    this.ex.runPass({
      shader: "broadphase-lbvh", entry: "lbvh_build",
      groups: [[
        B(0, "simParams"),
        B(1, "mortonKeys"),
        B(6, "lbvhChild"),
        B(7, "lbvhRange"),
      ]],
      x: this.w(Math.max(m - 1, 1)),
    });
    this.ex.runPass({
      shader: "broadphase-lbvh", entry: "lbvh_find_root",
      groups: [[
        B(0, "simParams"),
        B(7, "lbvhRange"),
        B(8, "lbvhRoot"),
      ]],
      x: this.w(Math.max(m - 1, 1)),
    });
    const refitPasses = 2 * Math.ceil(Math.log2(Math.max(m, 2)));
    for (let r = 0; r < refitPasses; r++) {
      this.ex.runPass({
        shader: "broadphase-lbvh", entry: "lbvh_refit",
        groups: [[
          B(0, "simParams"),
          B(4, "lbvhMin"),
          B(5, "lbvhMax"),
          B(6, "lbvhChild"),
        ]],
        x: this.w(Math.max(m - 1, 1)),
      });
    }
    this.mark(STAGE_IDS.lbvh);
  }

  traversePass(): void {
    // traverse
    this.bankU(GpuUniformSlot.ExclusionCount, this.exclusionCount);
    this.bankU(GpuUniformSlot.PairCapacity, this.c.pairs);
    this.ex.runPass({
      shader: "broadphase-traverse", entry: "traverse",
      groups: [[
        B(0, "simParams"),
        B(1, "lbvhMin"),
        B(2, "lbvhMax"),
        B(3, "lbvhChild"),
        B(4, "mortonPayload"),
        B(5, "exclusionKeys"),
        this.bg(6, GpuUniformSlot.ExclusionCount),
        B(7, "candidatePairs"),
        B(8, "pairCount"),
        B(9, "overflowFlag"),
        B(10, "pairScanned"),
        this.bg(11, GpuUniformSlot.PairCapacity),
        B(12, "lbvhRoot"),
      ]],
      x: this.w(this.c.m),
    });
    this.mark(STAGE_IDS.traverse);
  }

  exclusionCount = 0;

  // ---- G2 CCD + compaction at a state buffer ----

  contactPasses(p: { dHat: number; kappa: number; mu: number; fricEps: number; floorY: number; floorOn: number; dMin: number; contactCapacity: number }, posBuf: string): void {
    this.bankF(GpuUniformSlot.DHat, p.dHat);
    this.bankF(GpuUniformSlot.Kappa, p.kappa);
    this.bankF(GpuUniformSlot.Mu, p.mu);
    this.bankF(GpuUniformSlot.FricEps, p.fricEps);
    this.bankF(GpuUniformSlot.FloorY, p.floorY);
    this.bankU(GpuUniformSlot.FloorOn, p.floorOn);
    this.bankF(GpuUniformSlot.DMin, p.dMin);
    this.bankU(GpuUniformSlot.ContactCapacity, p.contactCapacity);
    this.bankF(GpuUniformSlot.Thickness, p.dMin);
    // expand pairs -> primitive ids
    this.ex.runPass({
      shader: "contact-compact", entry: "expand_pairs",
      groups: [[
        B(1, "triangles"),
        B(2, "candidatePairs"),
        B(3, "pairCount"),
        B(4, "primIdsVT"),
        B(5, "primIdsEE"),
        B(6, "primCountVT"),
        B(7, "primCountEE"),
      ]],
      x: this.w(this.c.pairs),
    });
    this.mark(STAGE_IDS.expand);
    // NOTE: expand reads pairCount as storage (binding 3). The remaining
    // entries below bind only the buffers they use (auto-layout prunes rest).
    this.closestCcdPasses(posBuf);
  }

  private closestCcdPasses(posBuf: string): void {
    const primCap = Math.max(this.c.pairs * 15, 1);
    const wPrim = this.w(primCap);
    // VT closest + CCD (guarded by primCountVT storage count)
    this.ex.runPass({
      shader: "closest-vt", entry: "closest_vt",
      groups: [[
        B(0, "primIdsVT"), B(1, posBuf),
        B(2, "vtSTD"), B(5, "vtR"),
        B(6, "primCountVT"),
      ]],
      x: wPrim,
    });
    this.mark(STAGE_IDS.closestVT);
    this.ex.runPass({
      shader: "ccd-vt", entry: "ccd_vt",
      groups: [[
        B(1, "position0"), B(2, posBuf),
        B(3, "primIdsVT"), B(4, "primTOIVT"), B(5, "primFlagVT"),
        B(6, "primCountVT"), this.bg(7, GpuUniformSlot.Thickness),
      ]],
      x: wPrim,
    });
    this.mark(STAGE_IDS.ccdVT);
    // EE closest + CCD
    this.ex.runPass({
      shader: "closest-ee", entry: "closest_ee",
      groups: [[
        B(0, "primIdsEE"), B(1, posBuf),
        B(2, "eeSTD"), B(5, "eeR"),
        B(6, "primCountEE"),
      ]],
      x: wPrim,
    });
    this.mark(STAGE_IDS.closestEE);
    this.ex.runPass({
      shader: "ccd-ee", entry: "ccd_ee",
      groups: [[
        B(1, "position0"), B(2, posBuf),
        B(3, "primIdsEE"), B(4, "primTOIEE"), B(5, "primFlagEE"),
        B(6, "primCountEE"), this.bg(7, GpuUniformSlot.Thickness),
      ]],
      x: wPrim,
    });
    this.mark(STAGE_IDS.ccdEE);
    // compact VT + EE + floor (exact per-entry binding sets; auto-layout
    // prunes the rest, so over-binding here would be a validation error)
    const compactVT: BindEntry[][] = [[
      B(4, "primIdsVT"),
      B(8, "vtSTD"), B(11, "vtR"),
      B(16, "primTOIVT"), B(17, "primFlagVT"),
      B(20, "contactW"), B(21, "contactN"), B(22, "contactId"), B(23, "contactPrm"),
      B(24, "contactTOI"), B(25, "contactDist"),
      B(26, "contactCount"), B(27, "contactOverflow"), B(28, "contactScanned"), B(29, "contactFail"),
      this.bg(30, GpuUniformSlot.ContactCapacity),
      this.bg(31, GpuUniformSlot.DHat),
      this.bg(32, GpuUniformSlot.Kappa),
      this.bg(33, GpuUniformSlot.Mu),
      this.bg(34, GpuUniformSlot.FricEps),
      B(39, "primCountVT"),
    ]];
    const compactEE: BindEntry[][] = [[
      B(5, "primIdsEE"),
      B(12, "eeSTD"), B(15, "eeR"),
      B(18, "primTOIEE"), B(19, "primFlagEE"),
      B(20, "contactW"), B(21, "contactN"), B(22, "contactId"), B(23, "contactPrm"),
      B(24, "contactTOI"), B(25, "contactDist"),
      B(26, "contactCount"), B(27, "contactOverflow"), B(28, "contactScanned"), B(29, "contactFail"),
      this.bg(30, GpuUniformSlot.ContactCapacity),
      this.bg(31, GpuUniformSlot.DHat),
      this.bg(32, GpuUniformSlot.Kappa),
      this.bg(33, GpuUniformSlot.Mu),
      this.bg(34, GpuUniformSlot.FricEps),
      B(40, "primCountEE"),
    ]];
    const compactFloor: BindEntry[][] = [[
      B(0, "simParams"),
      B(20, "contactW"), B(21, "contactN"), B(22, "contactId"), B(23, "contactPrm"),
      B(24, "contactTOI"), B(25, "contactDist"),
      B(26, "contactCount"), B(27, "contactOverflow"), B(28, "contactScanned"),
      this.bg(30, GpuUniformSlot.ContactCapacity),
      this.bg(31, GpuUniformSlot.DHat),
      this.bg(32, GpuUniformSlot.Kappa),
      this.bg(34, GpuUniformSlot.FricEps),
      B(35, posBuf),
      this.bg(36, GpuUniformSlot.FloorY),
      this.bg(37, GpuUniformSlot.FloorOn),
      this.bg(38, GpuUniformSlot.DMin),
    ]];
    this.ex.runPass({ shader: "contact-compact", entry: "compact_pairs", groups: compactVT, x: wPrim });
    this.ex.runPass({ shader: "contact-compact", entry: "compact_pairs_ee", groups: compactEE, x: wPrim });
    this.mark(STAGE_IDS.compact);
    this.ex.runPass({ shader: "contact-compact", entry: "compact_floor", groups: compactFloor, x: this.w(this.c.n) });
    this.mark(STAGE_IDS.floor);
  }

  // ---- FEM + barrier/friction + gradient at a state buffer ----

  femPasses(mat: { c00: number; c11: number; c01: number; g: number; thickness: number }, atTrial: boolean): void {
    const posBuf = atTrial ? "xTrial" : "position";
    this.bankF(GpuUniformSlot.MatC00, mat.c00);
    this.bankF(GpuUniformSlot.MatC11, mat.c11);
    this.bankF(GpuUniformSlot.MatC01, mat.c01);
    this.bankF(GpuUniformSlot.MatG, mat.g);
    this.bankF(GpuUniformSlot.MatThickness, mat.thickness);
    this.bankU(GpuUniformSlot.OutSel, 0);
    this.ex.runPass({
      shader: "membrane-gradient", entry: "main",
      groups: [[
        B(0, "simParams"), B(1, posBuf), B(2, "triangles"), B(3, "dmInv"),
        B(4, "restArea"), B(6, "elementGradient"), B(7, "elementEnergy"),
        B(13, "elementGradientB"), this.bg(14, GpuUniformSlot.OutSel),
        this.bg(8, GpuUniformSlot.MatC00), this.bg(9, GpuUniformSlot.MatC11),
        this.bg(10, GpuUniformSlot.MatC01), this.bg(11, GpuUniformSlot.MatG),
        this.bg(12, GpuUniformSlot.MatThickness),
      ]],
      x: this.w(this.c.m),
    });
    this.mark(STAGE_IDS.membrane);
    this.ex.runPass({
      shader: "bending-gradient", entry: "main",
      groups: [[
        B(0, "simParams"), B(1, posBuf), B(2, "hinges"), B(3, "hingeMeta"),
        B(4, "hingeGradient"), B(5, "hingeEnergyOut"),
      ]],
      x: this.w(Math.max(this.c.h, 1)),
    });
    this.mark(STAGE_IDS.bending);
    // barrier + friction consume the compact set at the same state
    this.ex.runPass({
      shader: "barrier-gradient", entry: "main",
      groups: [[
        B(0, "simParams"), B(1, posBuf),
        B(2, "contactW"), B(3, "contactN"), B(4, "contactId"), B(5, "contactPrm"),
        B(6, "contactScratch"), B(7, "contactEnergy"), B(8, "contactDist"),
      ]],
      x: this.w(Math.max(this.c.cap, 1)),
    });
    this.mark(STAGE_IDS.barrier);
    // slip = pos - position0 (sub_v4 into slip)
    this.ex.writeBlas(this.c.nExt, 1);
    this.ex.runPass({
      shader: "blas", entry: "sub_v4",
      groups: [[this.blasBank(), B(4, posBuf), B(5, "position0"), B(6, "slip")]],
      x: this.w(this.c.nExt),
    });
    this.frictionPass();
    // assemble contact force + total gradient
    this.ex.runPass({
      shader: "contact-force", entry: "assemble_contact_force",
      groups: [[
        B(0, "simParams"),
        B(1, "contactW"), B(2, "contactN"), B(3, "contactId"),
        B(4, "contactScratch"), B(5, "frictionScratch"), B(6, "contactForce"),
      ]],
      x: this.w(this.c.n),
    });
    if (this.cfg.useGatherAssembly) {
      // G4B production: membrane gather (WRITE) + hinge gather (ADD) +
      // contact residual via blas add_into. No O(n*m/h) scans.
      this.ex.runPass({
        shader: "assemble-gather", entry: "gather_membrane_grad",
        groups: [[
          B(0, "simParams"), B(1, "vertexElementOffsets"), B(2, "vertexElementIds"),
          B(3, "vertexElementCorners"), B(4, "elementGradient"), B(5, "gradient"),
        ]],
        x: this.w(this.c.n),
      });
      this.ex.runPass({
        shader: "assemble-gather", entry: "gather_hinge_grad",
        groups: [[
          B(0, "simParams"), B(1, "vertexHingeOffsets"), B(2, "vertexHingeIds"),
          B(3, "vertexHingeCorners"), B(4, "hingeGradient"), B(5, "gradient"),
        ]],
        x: this.w(this.c.n),
      });
      this.ex.writeBlas(this.n3, 1);
      this.ex.runPass({
        shader: "blas", entry: "add_into",
        groups: [[this.blasBank(), B(1, "contactForce"), B(7, "gradient")]],
        x: this.w(this.n3),
      });
    } else {
      this.ex.runPass({
        shader: "assemble-gradient", entry: "main",
        groups: [[
          B(0, "simParams"), B(1, "triangles"), B(2, "elementGradient"),
          B(3, "hinges"), B(4, "hingeGradient"), B(5, "contactForce"),
          B(6, "gradient"), B(7, "gradientAlt"), this.bg(8, GpuUniformSlot.OutSel),
        ]],
        x: this.w(this.c.n),
      });
    }
    this.mark(STAGE_IDS.assemble);
  }

  private blasBank(): BindEntry {
    return { binding: 0, buffer: this.ex.uniformBankName, offset: 0, size: 16 };
  }

  private frictionPass(): void {
    this.ex.runPass({
      shader: "friction", entry: "main",
      groups: [[
        B(0, "simParams"),
        B(1, "contactW"), B(2, "contactN"), B(3, "contactId"),
        B(4, "laggedN"), B(5, "slip"), B(6, "frictionScratch"),
        B(7, "contactPrm"),
      ]],
      x: this.w(Math.max(this.c.cap, 1)),
    });
    this.mark(STAGE_IDS.friction);
  }

  // ---- RHS + Jacobi ----

  /** Total residual g = M(x-yHat)/h^2 + gradInternal at srcBuf (pins zeroed). */
  rhsAt(srcBuf: string): void {
    this.ex.runPass({
      shader: "newton-rhs", entry: "main",
      groups: [[
        B(0, "simParams"), B(1, srcBuf), B(2, "xReference"), B(3, "mass"),
        B(4, "gradient"), B(5, "pinMask"), B(6, "rhs"), B(7, "negRhs"),
      ]],
      x: this.w(this.n3),
    });
  }

  /** Jacobi diagonal (needs contactDiag from barrier-hvp contact_diag first). */
  jacobiOnly(beta: number): void {
    this.bankF(GpuUniformSlot.JacobiBeta, beta);
    this.ex.runPass({
      shader: "jacobi", entry: "main",
      groups: [[
        B(0, "simParams"), B(1, "mass"), B(2, "contactDiag"), B(3, "diag"),
        this.bg(4, GpuUniformSlot.JacobiBeta),
      ]],
      x: this.w(this.n3),
    });
  }

  rhsJacobi(beta: number): void {
    this.contactDiagPass();
    this.rhsAt("position");
    this.jacobiOnly(beta);
    this.mark(STAGE_IDS.newtonRHS);
    this.mark(STAGE_IDS.jacobi);
  }

  /** Barrier-curvature diagonal into contactDiag (writes all vertices). */
  contactDiagPass(): void {
    this.ex.runPass({
      shader: "barrier-hvp", entry: "contact_diag",
      groups: [[
        B(0, "simParams"),
        B(2, "contactW"), B(3, "contactN"), B(4, "contactId"), B(5, "contactPrm"),
        B(6, "contactDist"), B(10, "contactDiag"),
      ]],
      x: this.w(this.c.n),
    });
  }

  // ---- HVP y = M/h^2 v + HpMembrane(v) + HpBarrier(v), out = Ap ----

  /** G4A analytic membrane HVP: membrane_hvp + assemble_hvp -> hpMembrane.
   *  Reads position + pcgSearch (direction); no perturbed states, no FD. */
  analyticMembraneHvp(): void {
    this.ex.runPass({
      shader: "membrane-hvp", entry: "membrane_hvp",
      groups: [[
        B(0, "simParams"), B(1, "position"), B(2, "triangles"), B(3, "dmInv"),
        B(4, "restArea"), B(5, "pcgSearch"), B(6, "elementHVP"),
        this.bg(8, GpuUniformSlot.MatC00), this.bg(9, GpuUniformSlot.MatC11),
        this.bg(10, GpuUniformSlot.MatC01), this.bg(11, GpuUniformSlot.MatG),
        this.bg(12, GpuUniformSlot.MatThickness),
      ]],
      x: this.w(this.c.m),
    });
    if (this.cfg.useGatherAssembly) {
      // G4B: gather replaces the reference scan (same sums, CSR order).
      this.ex.runPass({
        shader: "assemble-gather", entry: "gather_membrane_hvp",
        groups: [[
          B(0, "simParams"), B(1, "vertexElementOffsets"), B(2, "vertexElementIds"),
          B(3, "vertexElementCorners"), B(4, "elementHVP"), B(5, "hpMembrane"),
        ]],
        x: this.w(this.c.n),
      });
    } else {
      this.ex.runPass({
        shader: "membrane-hvp", entry: "assemble_hvp",
        groups: [[
          B(0, "simParams"), B(2, "triangles"), B(6, "elementHVP"), B(7, "hpMembrane"),
        ]],
        x: this.w(this.c.n),
      });
    }
  }

  /** G3 FD membrane HVP oracle (kept as debug path): x+/-h*p membrane evals
   *  + fd_combine -> hpMembrane. Production uses analyticMembraneHvp. */
  fdMembraneHvp(): void {
    const h = this.cfg.hFd;
    const n3 = this.n3;
    // xPlus (xTrial scratch during PCG) = position + h*p
    // (v4 entries count vertices; dispatch width is exact regardless)
    this.ex.writeBlas(this.c.nExt, h);
    this.ex.runPass({
      shader: "blas", entry: "copy_v4",
      groups: [[this.blasBank(), B(4, "position"), B(6, "xTrial")]],
      x: this.w(this.c.nExt),
    });
    this.ex.runPass({
      shader: "blas", entry: "axpy_vec4",
      groups: [[this.blasBank(), B(1, "pcgSearch"), B(5, "xTrial")]],
      x: this.w(this.c.nExt),
    });
    // xMinus = position - h*p
    this.ex.writeBlas(this.c.nExt, -h);
    this.ex.runPass({
      shader: "blas", entry: "copy_v4",
      groups: [[this.blasBank(), B(4, "position"), B(6, "hvpXMinus")]],
      x: this.w(this.c.nExt),
    });
    this.ex.runPass({
      shader: "blas", entry: "axpy_vec4",
      groups: [[this.blasBank(), B(1, "pcgSearch"), B(5, "hvpXMinus")]],
      x: this.w(this.c.nExt),
    });
    // membrane at xPlus -> elemGradB, assemble (zero contact) -> hpPlus
    this.bankU(GpuUniformSlot.OutSel, 1);
    this.ex.runPass({
      shader: "membrane-gradient", entry: "main",
      groups: [[
        B(0, "simParams"), B(1, "xTrial"), B(2, "triangles"), B(3, "dmInv"),
        B(4, "restArea"), B(6, "elementGradient"), B(7, "elementEnergy"),
        B(13, "elementGradientB"), this.bg(14, GpuUniformSlot.OutSel),
        this.bg(8, GpuUniformSlot.MatC00), this.bg(9, GpuUniformSlot.MatC11),
        this.bg(10, GpuUniformSlot.MatC01), this.bg(11, GpuUniformSlot.MatG),
        this.bg(12, GpuUniformSlot.MatThickness),
      ]],
      x: this.w(this.c.m),
    });
    this.ex.runPass({
      shader: "assemble-gradient", entry: "main",
      groups: [[
        B(0, "simParams"), B(1, "triangles"), B(2, "elementGradientB"),
        B(3, "hinges"), B(4, "hingeGradient"), B(5, "contactForceZero"),
        B(6, "gradient"), B(7, "hpPlus"), this.bg(8, GpuUniformSlot.OutSel),
      ]],
      x: this.w(this.c.n),
    });
    // membrane at xMinus -> elemGradB, assemble -> hpMinus
    this.ex.runPass({
      shader: "membrane-gradient", entry: "main",
      groups: [[
        B(0, "simParams"), B(1, "hvpXMinus"), B(2, "triangles"), B(3, "dmInv"),
        B(4, "restArea"), B(6, "elementGradient"), B(7, "elementEnergy"),
        B(13, "elementGradientB"), this.bg(14, GpuUniformSlot.OutSel),
        this.bg(8, GpuUniformSlot.MatC00), this.bg(9, GpuUniformSlot.MatC11),
        this.bg(10, GpuUniformSlot.MatC01), this.bg(11, GpuUniformSlot.MatG),
        this.bg(12, GpuUniformSlot.MatThickness),
      ]],
      x: this.w(this.c.m),
    });
    this.ex.runPass({
      shader: "assemble-gradient", entry: "main",
      groups: [[
        B(0, "simParams"), B(1, "triangles"), B(2, "elementGradientB"),
        B(3, "hinges"), B(4, "hingeGradient"), B(5, "contactForceZero"),
        B(6, "gradient"), B(7, "hpMinus"), this.bg(8, GpuUniformSlot.OutSel),
      ]],
      x: this.w(this.c.n),
    });
    this.bankU(GpuUniformSlot.OutSel, 0);
    // fd_combine: hpMembrane = (hpPlus - hpMinus) / (2h)
    this.ex.writeBlas(n3, 1 / (2 * h));
    this.ex.runPass({
      shader: "blas", entry: "fd_combine",
      groups: [[this.blasBank(), B(1, "hpPlus"), B(2, "hpMinus"), B(3, "hpMembrane")]],
      x: this.w(n3),
    });
  }

  hvpPass(): void {
    if (this.cfg.useAnalyticHvp) this.analyticMembraneHvp();
    else this.fdMembraneHvp();
    const n3 = this.n3;
    // frozen barrier HVP: project + assemble (exact used sets)
    this.ex.runPass({
      shader: "barrier-hvp", entry: "barrier_hvp_project",
      groups: [[
        B(0, "simParams"), B(1, "pcgSearch"),
        B(2, "contactW"), B(3, "contactN"), B(4, "contactId"), B(5, "contactPrm"),
        B(6, "contactDist"), B(7, "jvOut"), B(8, "coeffOut"),
      ]],
      x: this.w(Math.max(this.c.cap, 1)),
    });
    this.ex.runPass({
      shader: "barrier-hvp", entry: "assemble_barrier_hvp",
      groups: [[
        B(0, "simParams"),
        B(2, "contactW"), B(3, "contactN"), B(4, "contactId"),
        B(7, "jvOut"), B(8, "coeffOut"), B(9, "hpBarrier"),
      ]],
      x: this.w(this.c.n),
    });
    // y = M/h^2 v + HpMembrane + HpBarrier -> pcgAp
    this.ex.runPass({
      shader: "hessian-vector", entry: "main",
      groups: [[
        B(0, "simParams"), B(1, "pcgSearch"), B(2, "mass"),
        B(3, "hpMembrane"), B(4, "hpBarrier"), B(5, "pcgAp"),
      ]],
      x: this.w(n3),
    });
    this.mark(STAGE_IDS.hvp);
  }

  // ---- reductions (scalar syncs; each is a labeled SYNC POINT) ----

  /** G5B block-Jacobi factors: membrane 3x3 + Jacobi diag, Cholesky-or-diag.
   *  Runs once per solve (state-dependent); diag must be built (jacobiPass). */
  blockFactorPasses(): void {
    this.ex.runPass({
      shader: "block-jacobi", entry: "bj_build_factor",
      groups: [[
        B(0, "simParams"), B(1, "position"), B(2, "triangles"), B(3, "dmInv"),
        B(4, "restArea"), B(5, "vertexElementOffsets"), B(6, "vertexElementIds"),
        B(7, "vertexElementCorners"), B(8, "diag"), B(9, "blockInv"), B(10, "blockFlag"),
        this.bg(11, GpuUniformSlot.MatC00), this.bg(12, GpuUniformSlot.MatC11),
        this.bg(13, GpuUniformSlot.MatC01), this.bg(14, GpuUniformSlot.MatG),
        this.bg(15, GpuUniformSlot.MatThickness),
      ]],
      x: this.w(this.c.n),
    });
    this.mark(STAGE_IDS.jacobi);
  }

  /** G5C Schwarz factors: assemble local matrices + dense Cholesky/inverse.
   *  Runs once per solve (state-dependent); diag must be built first. */
  schwarzBuildPasses(): void {
    const nDoms = this.c.schwarzDoms;
    if (nDoms <= 0) throw new Error("schwarzBuildPasses: no Schwarz topology (schwarzDoms = 0)");
    this.bankU(GpuUniformSlot.SchwarzDomains, nDoms);
    this.ex.runPass({
      shader: "schwarz", entry: "schwarz_assemble",
      groups: [[
        B(1, "position"), B(2, "triangles"), B(3, "dmInv"),
        B(4, "restArea"), B(5, "vertexElementOffsets"), B(6, "vertexElementIds"),
        B(7, "vertexElementCorners"), B(8, "diag"), B(9, "schwarzMat"),
        B(10, "schwarzDomain"), B(11, "schwarzLocal"), B(12, "schwarzVerts"),
        this.bg(13, GpuUniformSlot.MatC00), this.bg(14, GpuUniformSlot.MatC11),
        this.bg(15, GpuUniformSlot.MatC01), this.bg(16, GpuUniformSlot.MatG),
        this.bg(17, GpuUniformSlot.MatThickness),
        this.bg(18, GpuUniformSlot.SchwarzDomains),
      ]],
      x: this.w(nDoms * 24),
    });
    this.ex.runPass({
      shader: "schwarz", entry: "schwarz_factor",
      groups: [[
        B(20, "schwarzMat"), B(21, "schwarzInv"), B(22, "schwarzFlag"),
        B(23, "diag"), B(24, "schwarzVerts"),
        this.bg(25, GpuUniformSlot.SchwarzDomains),
      ]],
      x: this.w(nDoms),
    });
    this.mark(STAGE_IDS.jacobi);
  }

  /** G5D true coarse diagonal Dc = diag(R A P) from the assembled local
   *  matrices (once per solve, after schwarzBuildPasses). Includes
   *  intra-aggregate membrane curvature — restrict(diag) alone underestimates
   *  and the damped sweep overshoots. */
  masCoarseDiagPass(): void {
    const nDoms = this.c.schwarzDoms;
    this.bankU(GpuUniformSlot.MasCount, nDoms * 3);
    this.ex.runPass({
      shader: "mas", entry: "mas_coarse_diag",
      groups: [[
        B(40, "masCoarseDiag"), B(41, "schwarzMat"), B(42, "schwarzVerts"),
        this.bg(43, GpuUniformSlot.MasCount),
      ]],
      x: this.w(nDoms * 3),
    });
  }

  /** G5D coarse correction: z += P (omega Dc^-1 R r). Same encoder, no sync. */
  masCorrectPass(): void {
    const nDoms = this.c.schwarzDoms;
    this.bankU(GpuUniformSlot.MasCount, nDoms * 3);
    this.bankF(GpuUniformSlot.MasOmega, this.cfg.masOmega);
    this.ex.runPass({
      shader: "mas", entry: "mas_restrict",
      groups: [[
        B(1, "pcgResidual"), B(2, "masCoarseR"), B(3, "schwarzVerts"),
        this.bg(4, GpuUniformSlot.MasCount),
      ]],
      x: this.w(nDoms * 3),
    });
    this.ex.runPass({
      shader: "mas", entry: "mas_coarse_scale",
      groups: [[
        B(11, "masCoarseR"), B(12, "masCoarseDiag"), B(13, "masCoarseZ"),
        this.bg(10, GpuUniformSlot.MasOmega),
        this.bg(14, GpuUniformSlot.MasCount),
      ]],
      x: this.w(nDoms * 3),
    });
    this.ex.runPass({
      shader: "mas", entry: "mas_prolongate_add",
      groups: [[
        B(0, "simParams"),
        B(20, "masCoarseZ"), B(21, "pcgZ"), B(22, "schwarzVerts"),
        B(23, "pinMask"), B(24, "schwarzDomain"),
      ]],
      x: this.w(this.n3),
    });
    // prod = r*z must include the coarse part (recompute after prolongation).
    this.ex.writeBlas(this.n3, 1);
    this.ex.runPass({
      shader: "blas", entry: "mul",
      groups: [[this.blasBank(), B(1, "pcgResidual"), B(2, "pcgZ"), B(3, "pcgProd")]],
      x: this.w(this.n3),
    });
  }

  /** G5.5 coarse block-CSR values assembly (once per solve, after diag).
   *  Sets the persistent CoarseCount/CoarseGroups uniforms (amortized submits).
   *  The stored blocks are raw; the operator is symmetrized at READ time
   *  (spmv/z-step average B with B^T), so no transpose race and no temp. */
  coarseAssemblePasses(): void {
    const nDoms = this.c.schwarzDoms;
    if (nDoms <= 0) throw new Error("coarseAssemblePasses: no Schwarz topology (schwarzDoms = 0)");
    const coarseDofs = nDoms * 3;
    this.bankU(GpuUniformSlot.SchwarzDomains, nDoms);
    this.bankU(GpuUniformSlot.CoarseCount, coarseDofs);
    this.bankU(GpuUniformSlot.CoarseGroups, this.w(coarseDofs));
    this.ex.runPass({
      shader: "coarse", entry: "coarse_assemble_values",
      groups: [[
        B(1, "position"), B(2, "triangles"), B(3, "dmInv"),
        B(4, "restArea"), B(5, "vertexElementOffsets"), B(6, "vertexElementIds"),
        B(7, "vertexElementCorners"), B(8, "diag"),
        B(9, "schwarzDomain"), B(10, "schwarzVerts"),
        B(11, "coarseRowOffsets"), B(12, "coarseColIndices"), B(13, "coarseBlockValues"),
        this.bg(14, GpuUniformSlot.MatC00), this.bg(15, GpuUniformSlot.MatC11),
        this.bg(16, GpuUniformSlot.MatC01), this.bg(17, GpuUniformSlot.MatG),
        this.bg(18, GpuUniformSlot.MatThickness),
        this.bg(19, GpuUniformSlot.SchwarzDomains),
      ]],
      x: this.w(nDoms),
    });
    this.mark(STAGE_IDS.jacobi);
  }

  /** G5.5 coarse SpMV y = sym(Ac) x with rebindable vectors. */
  coarseSpmvPass(xBuf: string, yBuf: string): void {
    this.ex.runPass({
      shader: "coarse", entry: "coarse_spmv",
      groups: [[
        B(20, "coarseRowOffsets"), B(21, "coarseColIndices"), B(22, "coarseBlockValues"),
        B(23, xBuf), B(24, yBuf),
        this.bg(25, GpuUniformSlot.CoarseCount),
      ]],
      x: this.w(this.c.schwarzDoms * 3),
    });
  }

  /** G5.5 coarse reduce into reduceScratch using PERSISTENT count slots
   *  (no bankU here => no added submits inside the inner loop). */
  coarseReduceIntoScratch(srcBuf: string): void {
    const count = this.c.schwarzDoms * 3;
    const groups = this.w(count);
    this.ex.runPass({
      shader: "coarse", entry: "coarse_reduce_stage1",
      groups: [[
        B(60, srcBuf), B(61, "reduceScratch"),
        this.bg(62, GpuUniformSlot.CoarseCount),
      ]],
      x: groups,
    });
    this.ex.runPass({
      shader: "coarse", entry: "coarse_reduce_stage2",
      groups: [[
        B(61, "reduceScratch"),
        this.bg(63, GpuUniformSlot.CoarseGroups),
      ]],
      x: 1,
    });
  }

  /** G5.5: true when any assembled-coarse path is enabled. */
  coarseEnabled(): boolean {
    return this.cfg.useCoarseC0 || this.cfg.useCoarsePcg;
  }

  /** G5.5 cross-aggregate span count (explicit measurement pass, NOT hot-loop:
   *  called by tests/benchmark after a rebuild; counter zeroed with the rest). */
  spanPass(): void {
    this.ex.runPass({
      shader: "mas", entry: "mas_contact_span",
      groups: [[
        B(0, "simParams"),
        B(50, "contactN"), B(51, "contactId"), B(52, "schwarzDomain"),
        B(53, "masContactSpan"), B(54, "contactCount"),
      ]],
      x: this.w(this.c.cap),
    });
  }

  /** Read the span counter (4 B, alongside the contact-count sync category). */
  async readContactSpan(): Promise<number> {
    const raw = await this.ex.readSmall("masContactSpan", 4, "contact-span", "scalar");
    return new Uint32Array(raw)[0];
  }

  /** G5.5 C0 coarse correction: z += P Dblock^-1 R r (single block-Jacobi
   *  coarse apply, no iteration). Same encoder, no sync. */
  coarseCorrectC0Pass(): void {
    // restrict residual -> masCoarseR
    this.bankU(GpuUniformSlot.MasCount, this.c.schwarzDoms * 3);
    this.ex.runPass({
      shader: "mas", entry: "mas_restrict",
      groups: [[
        B(1, "pcgResidual"), B(2, "masCoarseR"), B(3, "schwarzVerts"),
        this.bg(4, GpuUniformSlot.MasCount),
      ]],
      x: this.w(this.c.schwarzDoms * 3),
    });
    // block-Jacobi apply masCoarseR -> masCoarseZ (prod scratch: coarseProd;
    // c_update_z reads zR(32)/zZ(33)/zProd(35)/zDiag(36)/zCount(37) plus the
    // pattern/value buffers (20-22) via its shared helper).
    this.ex.runPass({
      shader: "coarse", entry: "c_update_z",
      groups: [[
        B(20, "coarseRowOffsets"), B(21, "coarseColIndices"), B(22, "coarseBlockValues"),
        B(32, "masCoarseR"), B(33, "masCoarseZ"), B(35, "coarseProd"),
        B(36, "masCoarseDiag"),
        this.bg(37, GpuUniformSlot.CoarseCount),
      ]],
      x: this.w(this.c.schwarzDoms * 3),
    });
    this.ex.runPass({
      shader: "mas", entry: "mas_prolongate_add",
      groups: [[
        B(0, "simParams"),
        B(20, "masCoarseZ"), B(21, "pcgZ"), B(22, "schwarzVerts"),
        B(23, "pinMask"), B(24, "schwarzDomain"),
      ]],
      x: this.w(this.n3),
    });
    // prod = r*z must include the coarse part (recompute after prolongation).
    this.ex.writeBlas(this.n3, 1);
    this.ex.runPass({
      shader: "blas", entry: "mul",
      groups: [[this.blasBank(), B(1, "pcgResidual"), B(2, "pcgZ"), B(3, "pcgProd")]],
      x: this.w(this.n3),
    });
  }

  /** G5.5 C1 inner coarse solve: fixed-K PCG on Ac = R A P entirely
   *  GPU-side (no scalar readback, no uniform rewrites => no added submits).
   *  Solves Ac xc = R r into coarseX; caller prolongates + refreshes prod. */
  coarseSolvePasses(K: number): void {
    const cdofs = this.c.schwarzDoms * 3;
    const W = this.w(cdofs);
    // restrict residual -> masCoarseR (= inner b)
    this.bankU(GpuUniformSlot.MasCount, cdofs);
    this.ex.runPass({
      shader: "mas", entry: "mas_restrict",
      groups: [[
        B(1, "pcgResidual"), B(2, "masCoarseR"), B(3, "schwarzVerts"),
        this.bg(4, GpuUniformSlot.MasCount),
      ]],
      x: this.w(cdofs),
    });
    // c_init: x=0, r=b, z=Dblock^-1 r, p=z, prod=r*z
    this.ex.runPass({
      shader: "coarse", entry: "c_init",
      groups: [[
        B(20, "coarseRowOffsets"), B(21, "coarseColIndices"), B(22, "coarseBlockValues"),
        B(30, "masCoarseR"), B(31, "coarseX"), B(32, "coarseR"),
        B(33, "coarseZ"), B(34, "coarseP"), B(35, "coarseProd"),
        B(36, "masCoarseDiag"),
        this.bg(37, GpuUniformSlot.CoarseCount),
      ]],
      x: W,
    });
    this.coarseReduceIntoScratch("coarseProd");
    this.ex.runPass({
      shader: "blas", entry: "scopy",
      groups: [[B(1, "reduceScratch"), B(3, "coarseRzPrev")]],
      x: 1,
    });
    for (let k = 0; k < K; k++) {
      this.coarseSpmvPass("coarseP", "coarseAp");
      this.ex.runPass({
        shader: "coarse", entry: "c_mul",
        groups: [[
          B(54, "coarseP"), B(55, "coarseAp"), B(56, "coarseProd"),
          this.bg(57, GpuUniformSlot.CoarseCount),
        ]],
        x: W,
      });
      this.coarseReduceIntoScratch("coarseProd");
      this.ex.runPass({
        shader: "blas", entry: "sdiv",
        groups: [[B(1, "coarseRzPrev"), B(2, "reduceScratch"), B(3, "coarseAlpha")]],
        x: 1,
      });
      this.ex.runPass({
        shader: "coarse", entry: "c_break",
        groups: [[B(58, "reduceScratch"), B(59, "coarseBreak")]],
        x: 1,
      });
      this.ex.runPass({
        shader: "coarse", entry: "c_update_xr",
        groups: [[
          B(40, "coarseX"), B(41, "coarseR"), B(42, "coarseP"), B(43, "coarseAp"),
          B(44, "coarseAlpha"),
          this.bg(45, GpuUniformSlot.CoarseCount),
        ]],
        x: W,
      });
      this.ex.runPass({
        shader: "coarse", entry: "c_update_z",
        groups: [[
          B(20, "coarseRowOffsets"), B(21, "coarseColIndices"), B(22, "coarseBlockValues"),
          B(32, "coarseR"), B(33, "coarseZ"), B(35, "coarseProd"),
          B(36, "masCoarseDiag"),
          this.bg(37, GpuUniformSlot.CoarseCount),
        ]],
        x: W,
      });
      this.coarseReduceIntoScratch("coarseProd");
      this.ex.runPass({
        shader: "blas", entry: "sdiv",
        groups: [[B(1, "reduceScratch"), B(2, "coarseRzPrev"), B(3, "coarseBeta")]],
        x: 1,
      });
      this.ex.runPass({
        shader: "coarse", entry: "c_update_p2",
        groups: [[
          B(50, "coarseZ"), B(51, "coarseP"), B(52, "coarseBeta"),
          this.bg(53, GpuUniformSlot.CoarseCount),
        ]],
        x: W,
      });
      this.ex.runPass({
        shader: "blas", entry: "scopy",
        groups: [[B(1, "reduceScratch"), B(3, "coarseRzPrev")]],
        x: 1,
      });
    }
    // prolongate coarseX -> pcgZ (pin-filtered), then refresh fine prod.
    this.ex.runPass({
      shader: "mas", entry: "mas_prolongate_add",
      groups: [[
        B(0, "simParams"),
        B(20, "coarseX"), B(21, "pcgZ"), B(22, "schwarzVerts"),
        B(23, "pinMask"), B(24, "schwarzDomain"),
      ]],
      x: this.w(this.n3),
    });
    this.ex.writeBlas(this.n3, 1);
    this.ex.runPass({
      shader: "blas", entry: "mul",
      groups: [[this.blasBank(), B(1, "pcgResidual"), B(2, "pcgZ"), B(3, "pcgProd")]],
      x: this.w(this.n3),
    });
  }

  /** G5C Schwarz apply: z = Sinv r (pin-filtered), prod = r*z. */
  schwarzApplyPass(): void {
    this.ex.runPass({
      shader: "schwarz", entry: "schwarz_apply",
      groups: [[
        B(0, "simParams"),
        B(30, "pcgResidual"), B(31, "pcgZ"), B(32, "pinMask"), B(33, "pcgProd"),
        B(34, "schwarzInv"), B(35, "schwarzVerts"),
        B(36, "schwarzDomain"), B(37, "schwarzLocal"),
      ]],
      x: this.w(this.n3),
    });
  }

  /** Two-stage sum reduce of srcBuf into reduceScratch[0]; returns the scalar. */
  async reduceSum(srcBuf: string, count: number): Promise<number> {
    const groups = this.w(count);
    this.bankU(GpuUniformSlot.ReduceCount, count);
    this.bankU(GpuUniformSlot.ReduceGroups, groups);
    this.ex.beginBatch("reduce-sum");
    this.ex.runPass({
      shader: "pcg-reduce", entry: "reduce_stage1",
      groups: [[
        B(0, srcBuf), B(1, "reduceScratch"),
        this.bg(2, GpuUniformSlot.ReduceCount),
      ]],
      x: groups,
    });
    this.ex.runPass({
      shader: "pcg-reduce", entry: "reduce_stage2",
      groups: [[
        B(1, "reduceScratch"),
        this.bg(3, GpuUniformSlot.ReduceGroups),
      ]],
      x: 1,
    });
    await this.ex.submitBatch(false);
    const raw = await this.ex.readSmall("reduceScratch", 4, "reduce-sum", "scalar");
    return new Float32Array(raw)[0];
  }

  /** Max reduce (trust-region cap). */
  async reduceMax(srcBuf: string, count: number): Promise<number> {
    const groups = this.w(count);
    this.bankU(GpuUniformSlot.ReduceCount, count);
    this.bankU(GpuUniformSlot.ReduceGroups, groups);
    this.ex.beginBatch("reduce-max");
    this.ex.runPass({
      shader: "pcg-reduce", entry: "reduce_max_stage1",
      groups: [[
        B(0, srcBuf), B(1, "reduceScratch"),
        this.bg(2, GpuUniformSlot.ReduceCount),
      ]],
      x: groups,
    });
    this.ex.runPass({
      shader: "pcg-reduce", entry: "reduce_max_stage2",
      groups: [[
        B(1, "reduceScratch"),
        this.bg(3, GpuUniformSlot.ReduceGroups),
      ]],
      x: 1,
    });
    await this.ex.submitBatch(false);
    const raw = await this.ex.readSmall("reduceScratch", 4, "reduce-max", "scalar");
    return new Float32Array(raw)[0];
  }

  /** out[i] = x[i]*y[i] into pcgProd (for dot products). */
  mulInto(xBuf: string, yBuf: string): void {
    this.ex.writeBlas(this.n3, 1);
    this.ex.runPass({
      shader: "blas", entry: "mul",
      groups: [[this.blasBank(), B(1, xBuf), B(2, yBuf), B(3, "pcgProd")]],
      x: this.w(this.n3),
    });
  }

  // ---- full GPU-side PCG solve (ONE submit, K fixed iterations) ----

  /**
   * Runs K PCG iterations in a single encoder (no mid-solve readback).
   * Returns the final residual norm + breakdown latch (2 tiny syncs after).
   * b = negRhs, x0 = 0. Mirrors math/pcg.ts policy (Jacobi, pin filter,
   * pAp<=tol breakdown) with fixed-K instead of adaptive stopping.
   */
  /** Fallback record from the most recent solve (null = requested path ran). */
  lastFallback: PreconditionerFallback | null = null;
  /** Effective method name of the most recent solve (after resolution). */
  lastActualMethod = "jacobi";

  async pcgSolve(
    record?: { rz: number[] },
    opts: { readScalars?: boolean } = {},
  ): Promise<{ resNorm: number; breakdown: boolean }> {
    // G6C.2 no-read mode (readScalars false): residual/breakdown stay
    // GPU-side (pcg_report writes the Newton status lanes); returned scalars
    // are dummy and MUST NOT be consumed. Profiling (record) is incompatible
    // with no-read mode (it syncs per iteration).
    const readScalars = opts.readScalars ?? true;
    if (record && !readScalars) {
      throw new Error("pcgSolve: profiling record requires readScalars");
    }
    const K = this.cfg.pcgIters;
    const n3 = this.n3;
    const W = this.w(n3);
    const groups = W;
    this.zeroBreakFlag();
    this.ex.beginBatch("pcg-solve");
    // G6A graceful degradation first (may clear flags + record a fallback).
    this.lastFallback = resolvePreconditioner(this.cfg, this.c.schwarzDoms);
    // Preconditioner precedence (ONE consistent M^-1 per solve):
    // G5.5 coarse-PCG > G5.5 C0 > G5D MAS > G5C Schwarz > G5B block > Jacobi.
    this.lastActualMethod = describeFlags(this.cfg);
    const useCoarsePcg = this.cfg.useCoarsePcg;
    const useCoarseC0 = this.cfg.useCoarseC0 && !useCoarsePcg;
    const useMas = this.cfg.useMas && !useCoarseC0 && !useCoarsePcg;
    const useSchwarz = (this.cfg.useSchwarz || useMas || useCoarseC0 || useCoarsePcg);
    const useBlock = this.cfg.useBlockJacobi && !useSchwarz;
    if (useSchwarz) this.schwarzBuildPasses();
    else if (useBlock) this.blockFactorPasses();
    if (useMas || useCoarseC0 || useCoarsePcg) this.masCoarseDiagPass();
    if (useCoarseC0 || useCoarsePcg) this.coarseAssemblePasses();
    if (useCoarsePcg) this.ex.writeBuffer("coarseBreak", new Float32Array(4));
    // init: x=0, r=b, z=M^-1 r, p=z, prod=r*z (G5B: block variant when enabled).
    // NOTE: pcg_init_bj never reads `diag` (fallback pre-baked into blockInv),
    // so binding 2 is OMITTED — Dawn's auto-layout prunes it and over-binding
    // fails validation.
    if (useSchwarz) {
      // Seed with the scalar init (x=0, r=b; z/p/prod overwritten below).
      this.ex.runPass({
        shader: "pcg-update", entry: "pcg_init",
        groups: [[
          B(0, "simParams"), B(1, "negRhs"), B(2, "diag"), B(3, "searchDirection"),
          B(4, "pcgResidual"), B(5, "pcgZ"), B(6, "pcgSearch"), B(7, "pinMask"),
          B(8, "pcgProd"),
        ]],
        x: W,
      });
      this.schwarzApplyPass();
      if (useCoarsePcg) this.coarseSolvePasses(this.cfg.coarseIters);
      else if (useCoarseC0) this.coarseCorrectC0Pass();
      else if (useMas) this.masCorrectPass();
      // p = z (blas copy needs the bank params for count).
      this.ex.writeBlas(n3, 1);
      this.ex.runPass({
        shader: "blas", entry: "copy",
        groups: [[this.blasBank(), B(1, "pcgZ"), B(3, "pcgSearch")]],
        x: W,
      });
    } else if (useBlock) {
      this.ex.runPass({
        shader: "pcg-update", entry: "pcg_init_bj",
        groups: [[
          B(0, "simParams"), B(1, "negRhs"), B(3, "searchDirection"),
          B(4, "pcgResidual"), B(5, "pcgZ"), B(6, "pcgSearch"), B(7, "pinMask"),
          B(8, "pcgProd"), B(9, "blockInv"),
        ]],
        x: W,
      });
    } else {
      this.ex.runPass({
        shader: "pcg-update", entry: "pcg_init",
        groups: [[
          B(0, "simParams"), B(1, "negRhs"), B(2, "diag"), B(3, "searchDirection"),
          B(4, "pcgResidual"), B(5, "pcgZ"), B(6, "pcgSearch"), B(7, "pinMask"),
          B(8, "pcgProd"),
        ]],
        x: W,
      });
    }
    // rz0 = sum(prod)
    this.reduceIntoScratch("pcgProd", n3, groups);
    if (record) record.rz.push(await this.readReduceScratch());
    // rzPrev = rz0 (scopy uses no uniforms: exact two bindings)
    this.ex.runPass({
      shader: "blas", entry: "scopy",
      groups: [[B(1, "reduceScratch"), B(3, "rzPrevSlot")]],
      x: 1,
    });
    for (let k = 0; k < K; k++) {
      this.hvpPass();
      // pAp = p.Ap
      this.ex.writeBlas(n3, 1);
      this.ex.runPass({
        shader: "blas", entry: "mul",
        groups: [[this.blasBank(), B(1, "pcgSearch"), B(2, "pcgAp"), B(3, "pcgProd")]],
        x: W,
      });
      this.reduceIntoScratch("pcgProd", n3, groups);
      // alpha = rzPrev / pAp ; latch breakdown when pAp <= tol
      // (sdiv/sflag use no uniforms except sflag's threshold via bank)
      this.ex.runPass({
        shader: "blas", entry: "sdiv",
        groups: [[B(1, "rzPrevSlot"), B(2, "reduceScratch"), B(3, "alphaSlot")]],
        x: 1,
      });
      this.ex.writeBlas(1, this.cfg.pApTol);
      this.ex.runPass({
        shader: "blas", entry: "sflag_le",
        groups: [[this.blasBank(), B(1, "reduceScratch"), B(3, "breakFlag")]],
        x: 1,
      });
      // x += a p ; r -= a Ap (exact used set: params/x/r/p/pin + slots/Ap)
      this.ex.runPass({
        shader: "pcg-update", entry: "pcg_update_xr",
        groups: [
          [
            B(0, "simParams"), B(3, "searchDirection"),
            B(4, "pcgResidual"), B(6, "pcgSearch"), B(7, "pinMask"),
          ],
          [B(0, "alphaSlot"), B(1, "pcgAp")],
        ],
        x: W,
      });
      // z = M^-1 r ; prod = r*z (block / Schwarz / MAS / C0 / C1 variants).
      if (useSchwarz) {
        this.schwarzApplyPass();
        if (useCoarsePcg) this.coarseSolvePasses(this.cfg.coarseIters);
        else if (useCoarseC0) this.coarseCorrectC0Pass();
        else if (useMas) this.masCorrectPass();
      } else if (useBlock) {
        this.ex.runPass({
          shader: "pcg-update", entry: "pcg_update_z_bj",
          groups: [[
            B(0, "simParams"),
            B(4, "pcgResidual"), B(5, "pcgZ"), B(7, "pinMask"),
            B(8, "pcgProd"), B(9, "blockInv"),
          ]],
          x: W,
        });
      } else {
        this.ex.runPass({
          shader: "pcg-update", entry: "pcg_update_z",
          groups: [[
            B(0, "simParams"), B(2, "diag"),
            B(4, "pcgResidual"), B(5, "pcgZ"), B(7, "pinMask"),
            B(8, "pcgProd"),
          ]],
          x: W,
        });
      }
      this.reduceIntoScratch("pcgProd", n3, groups);
      // G5A convergence curve (measurement only): extra submit+sync per iter.
      if (record) record.rz.push(await this.readReduceScratch());
      // beta = rzNew / rzPrev ; p = z + beta p ; rzPrev = rzNew
      this.ex.runPass({
        shader: "blas", entry: "sdiv",
        groups: [[B(1, "reduceScratch"), B(2, "rzPrevSlot"), B(3, "betaSlot")]],
        x: 1,
      });
      this.ex.runPass({
        shader: "pcg-update", entry: "pcg_update_p2",
        groups: [
          [
            B(0, "simParams"), B(5, "pcgZ"), B(6, "pcgSearch"), B(7, "pinMask"),
          ],
          [B(2, "betaSlot")],
        ],
        x: W,
      });
      this.ex.runPass({
        shader: "blas", entry: "scopy",
        groups: [[B(1, "reduceScratch"), B(3, "rzPrevSlot")]],
        x: 1,
      });
    }
    // final residual norm^2 = r.r
    this.ex.writeBlas(n3, 1);
    this.ex.runPass({
      shader: "blas", entry: "mul",
      groups: [[this.blasBank(), B(1, "pcgResidual"), B(2, "pcgResidual"), B(3, "pcgProd")]],
      x: W,
    });
    this.reduceIntoScratch("pcgProd", n3, groups);
    this.mark(STAGE_IDS.pcg);
    if (!readScalars) {
      // G6C.2: report into the Newton status lanes; caller must NOT submit or
      // read before these land (same encoder discipline as everything else).
      this.ex.runPass({
        shader: "newton-control", entry: "pcg_report",
        groups: [[
          B(10, "reduceScratch"), B(11, "breakFlag"), B(12, "newtonStatus"),
        ]],
        x: 1,
      });
      return { resNorm: NaN, breakdown: false };
    }
    await this.ex.submitBatch(false);
    // SYNC POINTS (tiny): residual scalar + breakdown latch.
    const resRaw = await this.ex.readSmall("reduceScratch", 4, "pcg-residual", "scalar");
    const flagRaw = await this.ex.readSmall("breakFlag", 4, "pcg-breakdown", "scalar");
    return {
      resNorm: Math.sqrt(Math.max(0, new Float32Array(resRaw)[0])),
      breakdown: new Float32Array(flagRaw)[0] !== 0,
    };
  }

  /** G5A: read reduceScratch[0] now (extra submit+sync; measurement only). */
  private async readReduceScratch(): Promise<number> {
    await this.ex.submitBatch(false);
    const raw = await this.ex.readSmall("reduceScratch", 4, "reduce-profile", "scalar");
    return new Float32Array(raw)[0];
  }

  private reduceIntoScratch(srcBuf: string, count: number, groups: number): void {
    this.bankU(GpuUniformSlot.ReduceCount, count);
    this.bankU(GpuUniformSlot.ReduceGroups, groups);
    this.ex.runPass({
      shader: "pcg-reduce", entry: "reduce_stage1",
      groups: [[
        B(0, srcBuf), B(1, "reduceScratch"),
        this.bg(2, GpuUniformSlot.ReduceCount),
      ]],
      x: groups,
    });
    // stage2 reads partial + numGroups only (no src, no count)
    this.ex.runPass({
      shader: "pcg-reduce", entry: "reduce_stage2",
      groups: [[
        B(1, "reduceScratch"),
        this.bg(3, GpuUniformSlot.ReduceGroups),
      ]],
      x: 1,
    });
  }

  // ---- trial + diagnostics ----

  /** xTrial = position + alphaEff * dx (vec4, pins exact). Submits. */
  async applyTrial(alphaEff: number): Promise<void> {
    this.refreshSimParams(alphaEff);
    this.ex.beginBatch("apply-trial");
    this.ex.runPass({
      shader: "apply-step", entry: "apply_step_vec4",
      groups: [[
        B(0, "simParams"), B(2, "searchDirection"),
        B(4, "position"), B(5, "xTrial"), B(6, "pinMask"), B(7, "pinPos"),
      ]],
      x: this.w(this.c.n),
    });
    this.mark(STAGE_IDS.apply);
    await this.ex.submitBatch(false);
  }

  /**
   * Solver-provided SimParams refresh (rewrites lineSearchAlpha and any other
   * mutated fields, then uploads the 64 B image). Assigned by WebGpuSolver.
   */
  refreshSimParams: (alphaEff: number) => void = () => {
    throw new Error("DeviceNewtonDriver: solver must assign refreshSimParams");
  };

  /** Diagnostics passes only (no batch open/submit, no readback): for use
   *  inside larger batches (G6B Armijo) or standalone via diagnosticsAt. */
  diagnosticsPasses(srcBuf = "xTrial"): void {
    const n3 = this.n3;
    const statWidth = Math.max(n3, this.c.m, this.c.h, this.c.cap);
    const groups = this.w(statWidth);
    this.bankU(GpuUniformSlot.ReduceGroups, groups);
    this.ex.runPass({
      shader: "diagnostics", entry: "diag_stage1",
      groups: [[
        B(0, "simParams"), B(1, "rhs"), B(2, srcBuf), B(3, "xReference"),
        B(4, "mass"), B(5, "elementEnergy"), B(6, "hingeEnergyOut"),
        B(7, "contactEnergy"), B(8, "contactDist"), B(9, "contactTOI"), B(10, "contactFail"),
        B(11, "diagScratch"),
      ]],
      x: groups,
    });
    this.ex.runPass({
      shader: "diagnostics", entry: "diag_stage2",
      groups: [[
        B(11, "diagScratch"), B(12, "solverStatus"),
        this.bg(14, GpuUniformSlot.ReduceGroups),
      ]],
      x: 1,
    });
    this.mark(STAGE_IDS.diagnostics);
  }

  /** Diagnostics at an explicit state buffer + status/marker pack read (SYNC).
   *  gradNorm covers rhs (TOTAL residual incl. inertia, pins filtered) to
   *  match the CPU convergence criterion (not the internal-only gradient).
   *  The caller must have run rhsAt(srcBuf) first so rhs is state-consistent.
   */
  async diagnosticsAt(srcBuf = "xTrial"): Promise<{ status: SolverStatus; markerMask: number; markerCount: number }> {
    this.ex.beginBatch("diagnostics");
    this.diagnosticsPasses(srcBuf);
    this.flushMarks();
    await this.ex.submitBatch(false);
    // SYNC POINT: 64 B status + 16 B marker in one pack (two tiny maps).
    const sRaw = await this.ex.readSmall("solverStatus", 64, "newton-status", "status");
    const mRaw = await this.ex.readSmall("execMarker", 16, "exec-marker", "status");
    const status = decodeSolverStatus(sRaw);
    const mu = new Uint32Array(mRaw);
    return { status, markerMask: mu[2], markerCount: mu[3] };
  }

  /** G6B per-candidate trial evaluation inside the caller's batch (no submits,
   *  no syncs): clear + apply_k + G1/G2 + FEM + diagnostics + record. The
   *  caller owns the batch open/submit and the final select. `evalIndex`
   *  advances the marker epoch per candidate (debug continuity). */
  armijoTrialPasses(
    k: number,
    mat: { c00: number; c11: number; c01: number; g: number; thickness: number },
    contact: {
      dHat: number; kappa: number; mu: number; fricEps: number;
      floorY: number; floorOn: number; dMin: number; contactCapacity: number;
    },
    evalIndex: number,
  ): void {
    this.armijoClearPass();
    this.ex.runPass({
      shader: "armijo", entry: `apply_${k}` as "apply_0",
      groups: [[
        B(0, "simParams"), B(2, "searchDirection"),
        B(4, "position"), B(5, "xTrial"), B(6, "pinMask"), B(7, "pinPos"),
        B(8, "armijoAlphas"), B(9, "armijoCur"), B(10, "trustScaleStore"),
      ]],
      x: this.w(this.c.n),
    });
    this.mark(STAGE_IDS.apply);
    void evalIndex; // marker epochs advance per batch (not per trial) to avoid control-write flushes
    // G1+G2 at xTrial (same passes as evaluateNewtonState Batch A).
    this.broadphasePasses(0.002, true);
    this.contactPasses(contact, "xTrial");
    // FEM + RHS at xTrial (same as Batch B trial branch).
    this.femPasses(mat, true);
    this.contactDiagPass();
    this.rhsAt("xTrial");
    // Diagnostics into solverStatus (no submit/read) + verdict record.
    this.diagnosticsPasses("xTrial");
    this.ex.runPass({
      shader: "armijo", entry: "armijo_record",
      groups: [[
        B(8, "armijoAlphas"), B(9, "armijoCur"),
        B(40, "solverStatus"), B(41, "contactOverflow"), B(42, "armijoCandidates"),
        B(43, "armijoStatus"),
        B(45, "e0Store"), B(46, "gtdxStore"),
        this.bg(47, GpuUniformSlot.DMin),
        B(48, "trustScaleStore"),
      ]],
      x: 1,
    });
    this.mark(STAGE_IDS.diagnostics);
  }

  /** G6B sync-free trial rebuild (Batch A + Batch B encode, no submit, no
   *  reads): refreshes records/gradient/rhs at the current xTrial for commit.
   *  Cap-bounded (contactCount uniform stays at cap); sentinel-cleared tail
   *  slots are inert by the barrier/friction guards. */
  rebuildTrialPasses(
    mat: { c00: number; c11: number; c01: number; g: number; thickness: number },
    contact: {
      dHat: number; kappa: number; mu: number; fricEps: number;
      floorY: number; floorOn: number; dMin: number; contactCapacity: number;
    },
  ): void {
    this.broadphasePasses(0.002, true);
    this.contactPasses(contact, "xTrial");
    this.femPasses(mat, true);
    this.contactDiagPass();
    this.rhsAt("xTrial");
  }

  /** G6B first-valid-wins selection over the K candidate rows. */
  armijoSelectPass(): void {
    this.ex.runPass({
      shader: "armijo", entry: "armijo_select",
      groups: [[
        B(50, "armijoCandidates"), B(51, "armijoStatus"),
        this.bg(52, GpuUniformSlot.ArmijoK),
        this.bg(53, GpuUniformSlot.ArmijoPcgBd),
        this.bg(54, GpuUniformSlot.ArmijoNewtonConv),
      ]],
      x: 1,
    });
    this.mark(STAGE_IDS.diagnostics);
  }

  /**
   * G6B batched Armijo trials: K candidates evaluated + selected with ONE
   * status readback. Caller contract:
   * - no open batch (this opens + submits its own),
   * - solver sets simParams.contactCount = cap + newtonIteration BEFORE
   *   calling (cap-bounded candidate loops need it; restored by the next
   *   evaluateNewtonState which re-syncs the live count),
   * - E0/gtdx are the current accepted energy and descent dot (already
   *   synced on CPU, same values the sequential path would use).
   * Rejected trials never commit (commit happens only via acceptTrial,
   * exactly like the CPU solver). Returns the decoded batch status.
   */
  async armijoBatch(opts: {
    E0: number; gtdx: number; alphaBase: number; beta: number; K: number;
    trustScale: number;
    mat: { c00: number; c11: number; c01: number; g: number; thickness: number };
    contact: {
      dHat: number; kappa: number; mu: number; fricEps: number;
      floorY: number; floorOn: number; dMin: number; contactCapacity: number;
    };
    evalIndexBase: number;
    pcgBreakdown: boolean; newtonConverged: boolean;
    /** Seed E0/gtdx/trust stores from opts (G6B explicit batches). GPU Newton
     *  rounds pass false: stores are GPU-maintained across rounds. */
    seedStores?: boolean;
  }): Promise<ArmijoBatchStatus> {
    const K = opts.K;
    if (K < 1 || K > 8) throw new Error(`armijoBatch: K=${K} outside [1,8]`);
    // Amortized CPU writes (encoder null here => no-op flushes).
    // Alphas are RAW (trust folds on GPU); E0/gtdx/trust seed the mirrors the
    // record path reads (same f32 values the sequential path would use).
    const alphas = new Float32Array(8).fill(1.0);
    for (let j = 0; j < K; j++) {
      alphas[j] = opts.alphaBase * Math.pow(opts.beta, j);
    }
    this.ex.writeBuffer("armijoAlphas", alphas);
    const st0 = new Float32Array(16);
    st0[8] = 1.0; // finite lane starts true (record ANDs per candidate)
    this.ex.writeBuffer("armijoStatus", st0);
    if (opts.seedStores ?? true) {
      this.ex.writeBuffer("e0Store", new Float32Array([opts.E0, 0, 0, 0]));
      this.ex.writeBuffer("gtdxStore", new Float32Array([opts.gtdx, 0, 0, 0]));
      this.ex.writeBuffer("trustScaleStore", new Float32Array([opts.trustScale, 0, 0, 0]));
    }
    this.bankU(GpuUniformSlot.ArmijoK, K);
    this.bankF(GpuUniformSlot.ArmijoPcgBd, opts.pcgBreakdown ? 1 : 0);
    this.bankF(GpuUniformSlot.ArmijoNewtonConv, opts.newtonConverged ? 1 : 0);
    this.resetMarker(opts.evalIndexBase + 1);
    this.ex.beginBatch("armijo-batch");
    this.armijoBatchPasses(K, opts.mat, opts.contact, opts.evalIndexBase);
    await this.ex.submitBatch(false);
    // THE one status readback per batch (labeled SYNC POINT).
    const raw = await this.ex.readSmall("armijoStatus", 64, "armijo-status", "status");
    return decodeArmijoStatus(raw);
  }

  /** G6C.2 encode-only Armijo batch (no submit, no reads): candidate trials
   *  + select. The caller owns batching/submits and (for G6B) the status
   *  read; GPU-control rounds chain batches with predicated commit instead.
   *  armijoAlphas/status/K must be CPU-written beforehand (flush-only). */
  armijoBatchPasses(
    K: number,
    mat: { c00: number; c11: number; c01: number; g: number; thickness: number },
    contact: {
      dHat: number; kappa: number; mu: number; fricEps: number;
      floorY: number; floorOn: number; dMin: number; contactCapacity: number;
    },
    evalIndexBase: number,
  ): void {
    for (let j = 0; j < K; j++) {
      this.armijoTrialPasses(j, mat, contact, evalIndexBase + j);
    }
    this.armijoSelectPass();
  }



  /** G6C.2 one GPU-controlled Newton round: Xk refresh, convergence check,
   *  PCG (no scalar reads), GPU gtdx/trust/descent, B Armijo batches with
   *  predicated commit, round report. Exactly ONE status readback; the CPU
   *  breaks or continues from the decoded status. All decisions GPU-side. */
  async newtonRound(opts: {
    round: number; beta: number;
    mat: { c00: number; c11: number; c01: number; g: number; thickness: number };
    contact: {
      dHat: number; kappa: number; mu: number; fricEps: number;
      floorY: number; floorOn: number; dMin: number; contactCapacity: number;
    };
    batchKs: number[];
    evalIndexBase: number;
  }): Promise<NewtonRoundStatus> {
    // Per-round control zero + iteration lane (amortized control writes).
    this.ex.writeBuffer("newtonCtl", new Float32Array(16));
    const nst = new Float32Array(20);
    nst[0] = opts.round;
    this.ex.writeBuffer("newtonStatus", nst);
    this.ex.beginBatch("newton-round");
    // 1. Xk refresh at the accepted position (records current post-commit;
    //    round 0 reuses the E0 evaluation's records the same way).
    this.rhsJacobi(opts.beta);
    this.diagnosticsPasses("position");
    // 2. Convergence check on the fresh Xk status.
    this.ex.runPass({
      shader: "newton-control", entry: "newton_check",
      groups: [[
        B(80, "solverStatus"), B(81, "newtonCtl"), B(82, "newtonStatus"),
        this.bg(83, GpuUniformSlot.NewtonTol),
      ]],
      x: 1,
    });
    // 3. PCG solve (no scalar reads; residual/breakdown -> status lanes).
    await this.pcgSolve(undefined, { readScalars: false });
    // 4. gtdx dot, kept GPU-side (mul + reduce, then scopy into the mirror).
    this.ex.writeBlas(this.n3, 1);
    this.ex.runPass({
      shader: "blas", entry: "mul",
      groups: [[this.blasBank(), B(1, "rhs"), B(2, "searchDirection"), B(3, "pcgProd")]],
      x: this.w(this.n3),
    });
    this.reduceIntoScratch("pcgProd", this.n3, this.w(this.n3));
    this.ex.runPass({
      shader: "blas", entry: "scopy",
      groups: [[B(1, "reduceScratch"), B(3, "gtdxStore")]],
      x: 1,
    });
    // 5. Trust scale from the device-side max (reduce_max passes, no read).
    this.ex.writeBlas(this.n3, 1);
    this.ex.runPass({
      shader: "blas", entry: "absv",
      groups: [[this.blasBank(), B(1, "searchDirection"), B(3, "pcgProd")]],
      x: this.w(this.n3),
    });
    {
      const groups = this.w(this.n3);
      this.bankU(GpuUniformSlot.ReduceCount, this.n3);
      this.bankU(GpuUniformSlot.ReduceGroups, groups);
      this.ex.runPass({
        shader: "pcg-reduce", entry: "reduce_max_stage1",
        groups: [[
          B(0, "pcgProd"), B(1, "reduceScratch"),
          this.bg(2, GpuUniformSlot.ReduceCount),
        ]],
        x: groups,
      });
      this.ex.runPass({
        shader: "pcg-reduce", entry: "reduce_max_stage2",
        groups: [[
          B(1, "reduceScratch"),
          this.bg(3, GpuUniformSlot.ReduceGroups),
        ]],
        x: 1,
      });
    }
    this.ex.runPass({
      shader: "newton-control", entry: "trust_compute",
      groups: [[
        B(20, "reduceScratch"), B(21, "trustScaleStore"), B(22, "newtonStatus"),
        this.bg(23, GpuUniformSlot.NewtonTrust),
      ]],
      x: 1,
    });
    // 6. Descent fallback selection, fully GPU-side.
    this.ex.writeBlas(this.n3, 1);
    this.ex.runPass({
      shader: "blas", entry: "neg_div",
      groups: [[this.blasBank(), B(1, "rhs"), B(2, "diag"), B(3, "descentDir")]],
      x: this.w(this.n3),
    });
    this.ex.runPass({
      shader: "newton-control", entry: "descent_check",
      groups: [[
        B(30, "gtdxStore"), B(31, "breakFlag"), B(32, "newtonCtl"), B(33, "newtonStatus"),
      ]],
      x: 1,
    });
    this.ex.runPass({
      shader: "newton-control", entry: "select_fallback",
      groups: [[
        B(40, "descentDir"), B(41, "searchDirection"), B(42, "newtonCtl"),
        B(43, "simParams"),
      ]],
      x: this.w(this.n3),
    });
    // 7. B Armijo batches with predicated commit (per-batch CPU param writes
    //    flush only — no syncs; the loop always runs all slots (bounded)).
    let trialBase = 0;
    for (let b = 0; b < opts.batchKs.length; b++) {
      const K = opts.batchKs[b];
      const alphas = new Float32Array(8).fill(1.0);
      for (let j = 0; j < K; j++) alphas[j] = Math.pow(0.5, trialBase + j);
      this.ex.writeBuffer("armijoAlphas", alphas);
      const ast0 = new Float32Array(16);
      ast0[8] = 1.0;
      this.ex.writeBuffer("armijoStatus", ast0);
      this.bankU(GpuUniformSlot.ArmijoK, K);
      this.bankF(GpuUniformSlot.ArmijoPcgBd, 0);
      this.bankF(GpuUniformSlot.ArmijoNewtonConv, 0);
      this.ex.beginBatch("newton-round-batches");
      for (let j = 0; j < K; j++) {
        this.armijoTrialPasses(j, opts.mat, opts.contact, opts.evalIndexBase + trialBase + j);
      }
      this.armijoSelectPass();
      // commit arbitration + refresh + predicated commit
      this.ex.runPass({
        shader: "newton-control", entry: "commit_arm",
        groups: [[
          B(50, "armijoStatus"), B(52, "newtonCtl"),
          B(53, "e0Store"), B(54, "newtonStatus"),
          this.bg(55, GpuUniformSlot.ArmijoK),
        ]],
        x: 1,
      });
      // Commit refresh: re-materialize xTrial at the latched alpha, rebuild
      // records there, then predicated copy. (Without the re-apply, xTrial
      // would still hold the last candidate — not necessarily accepted.)
      this.ex.runPass({
        shader: "newton-control", entry: "commit_apply",
        groups: [[
          B(64, "position"), B(65, "searchDirection"), B(66, "xTrial"),
          B(67, "pinMask"), B(68, "pinPos"), B(69, "newtonCtl"),
        ]],
        x: this.w(this.c.n),
      });
      this.rebuildTrialPasses(opts.mat, opts.contact);
      this.ex.runPass({
        shader: "newton-control", entry: "commit_copy_if",
        groups: [[
          B(60, "xTrial"), B(61, "position"), B(62, "newtonCtl"),
          B(63, "simParams"),
        ]],
        x: this.w(this.c.nExt),
      });
      this.ex.runPass({
        shader: "newton-control", entry: "commit_lagged_if",
        groups: [[
          B(70, "contactN"), B(71, "contactDist"), B(72, "contactPrm"),
          B(73, "laggedN"), B(74, "newtonCtl"), B(75, "simParams"),
        ]],
        x: this.w(Math.max(this.c.cap, 1)),
      });
      trialBase += K;
    }
    // 8. Round report + the single status readback per round.
    this.ex.runPass({
      shader: "newton-control", entry: "round_report",
      groups: [[
        B(90, "newtonCtl"), B(91, "newtonStatus"), B(92, "gtdxStore"),
        B(93, "e0Store"), B(94, "armijoStatus"),
      ]],
      x: 1,
    });
    await this.ex.submitBatch(false);
    const raw = await this.ex.readSmall("newtonStatus", 80, "newton-status", "status");
    return decodeNewtonStatus(raw);
  }

  /** G6B contact-state clear for one trial (sentinels + counter zeros).
   *  Split in two passes: Dawn caps storage buffers at 16 per stage. */
  armijoClearPass(): void {
    this.ex.runPass({
      shader: "armijo", entry: "armijo_clear_records",
      groups: [[
        B(0, "simParams"),
        B(20, "contactW"), B(21, "contactN"), B(22, "contactId"),
        B(23, "contactPrm"), B(24, "contactDist"), B(25, "contactTOI"),
        B(26, "contactEnergy"), B(27, "contactScratch"), B(28, "frictionScratch"),
      ]],
      x: this.w(Math.max(this.c.cap, 1)),
    });
    this.ex.runPass({
      shader: "armijo", entry: "armijo_clear_counts",
      groups: [[
        B(29, "pairCount"), B(30, "overflowFlag"), B(31, "pairScanned"),
        B(32, "primCountVT"), B(33, "primCountEE"), B(34, "contactCount"),
        B(35, "contactOverflow"), B(36, "contactScanned"), B(37, "contactFail"),
      ]],
      x: 1,
    });
  }

  /** Accept: position = xTrial; refresh lagged friction state. Submits. */
  async acceptTrial(): Promise<void> {
    this.ex.beginBatch("accept-trial");
    this.ex.writeBlas(this.c.nExt, 1);
    this.ex.runPass({
      shader: "blas", entry: "copy_v4",
      groups: [[this.blasBank(), B(4, "xTrial"), B(6, "position")]],
      x: this.w(this.c.nExt),
    });
    this.ex.runPass({
      shader: "contact-force", entry: "commit_lagged",
      groups: [[
        B(0, "simParams"),
        B(2, "contactN"),
        B(7, "contactDist"), B(8, "contactPrm"), B(9, "laggedN"),
      ]],
      x: this.w(Math.max(this.c.cap, 1)),
    });
    await this.ex.submitBatch(false);
  }

  /**
   * Begin-step transfer (fully GPU-side, no readback): position0 = position.
   * position0 is the Newton-segment start (xStep) for CCD, slip, and the
   * velocity update. Upload only sets it once; WITHOUT this copy every step
   * after the first would reuse a stale xStep (wrong velocities/slip/TOIs).
   */
  async beginStepCopy(): Promise<void> {
    this.ex.beginBatch("begin-step");
    this.ex.writeBlas(this.c.nExt, 1);
    this.ex.runPass({
      shader: "blas", entry: "copy_v4",
      groups: [[this.blasBank(), B(4, "position"), B(6, "position0")]],
      x: this.w(this.c.nExt),
    });
    await this.ex.submitBatch(false);
  }

  /** Predictor: y_hat = x + h v + h^2 g into position + xReference, pins exact. */
  async predictorPass(): Promise<void> {
    this.ex.beginBatch("predictor");
    this.ex.runPass({
      shader: "predictor", entry: "main",
      groups: [[
        B(0, "simParams"), B(1, "position0"), B(2, "velocity"),
        B(3, "position"), B(4, "xReference"),
      ]],
      x: this.w(this.c.n),
    });
    this.mark(STAGE_IDS.predictor);
    // restore exact pins into position (rebound as xTrial4)
    this.ex.runPass({
      shader: "apply-step", entry: "enforce_pins",
      groups: [[
        B(0, "simParams"), B(5, "position"), B(6, "pinMask"), B(7, "pinPos"),
      ]],
      x: this.w(this.c.n),
    });
    await this.ex.submitBatch(false);
  }

  /**
   * Predictor bisection (device-side, mirrors CPU newton.ts): when y_hat
   * tunnels (minDistance <= dMin or a fresh crossing), bisect back along
   * x0 -> y_hat to the largest CCD-valid fraction and restart Newton there.
   * Each probe re-runs G1+G2 + min read at xProbe (diagnostics energies from
   * stale FEM buffers are IGNORED — only minDist/minTOI decide validity).
   * Runs ONLY when the E0 status already shows tunneling (common case: zero
   * extra cost). 8 probes max, pin-exact throughout.
   */
  async bisectPredictor(
    dMin: number,
    readStatus: () => Promise<{ minDistance: number; minToi: number }>,
    rebuildAtTrial: () => Promise<void>,
  ): Promise<{ bisected: boolean; frac: number }> {
    // x0 = position0, y_hat = position. Probe states live in xTrial.
    let lo = 0;
    let hi = 1;
    for (let b = 0; b < 8; b++) {
      const mid = 0.5 * (lo + hi);
      // xTrial = position0 + mid * (position - position0), pins enforced.
      this.ex.beginBatch("bisect-lerp");
      this.ex.writeBlas(this.c.nExt, 1);
      this.ex.runPass({
        shader: "blas", entry: "sub_v4",
        groups: [[this.blasBank(), B(4, "position"), B(5, "position0"), B(6, "hvpXMinus")]],
        x: this.w(this.c.nExt),
      });
      this.ex.runPass({
        shader: "blas", entry: "copy_v4",
        groups: [[this.blasBank(), B(4, "position0"), B(6, "xTrial")]],
        x: this.w(this.c.nExt),
      });
      this.ex.writeBlas(this.c.nExt, mid);
      this.ex.runPass({
        shader: "blas", entry: "axpy_v4",
        groups: [[this.blasBank(), B(4, "hvpXMinus"), B(5, "xTrial")]],
        x: this.w(this.c.nExt),
      });
      this.ex.runPass({
        shader: "apply-step", entry: "enforce_pins",
        groups: [[
          B(0, "simParams"), B(5, "xTrial"), B(6, "pinMask"), B(7, "pinPos"),
        ]],
        x: this.w(this.c.n),
      });
      await this.ex.submitBatch(false);
      await rebuildAtTrial();
      const s = await readStatus();
      const valid = s.minDistance > dMin && !(s.minToi > 0 && s.minToi < 1 - 1e-9);
      if (valid) lo = mid;
      else hi = mid;
      if (hi - lo < 1e-4) break;
    }
    if (lo <= 0) return { bisected: false, frac: 1 };
    // position = x0 + lo * (y_hat - x0), pins already exact in xTrial@lo.
    // Recompute the lo probe (cheap, exact) and copy it over.
    this.ex.beginBatch("bisect-apply");
    this.ex.writeBlas(this.c.nExt, 1);
    this.ex.runPass({
      shader: "blas", entry: "sub_v4",
      groups: [[this.blasBank(), B(4, "position"), B(5, "position0"), B(6, "hvpXMinus")]],
      x: this.w(this.c.nExt),
    });
    this.ex.runPass({
      shader: "blas", entry: "copy_v4",
      groups: [[this.blasBank(), B(4, "position0"), B(6, "xTrial")]],
      x: this.w(this.c.nExt),
    });
    this.ex.writeBlas(this.c.nExt, lo);
    this.ex.runPass({
      shader: "blas", entry: "axpy_v4",
      groups: [[this.blasBank(), B(4, "hvpXMinus"), B(5, "xTrial")]],
      x: this.w(this.c.nExt),
    });
    this.ex.runPass({
      shader: "apply-step", entry: "enforce_pins",
      groups: [[
        B(0, "simParams"), B(5, "xTrial"), B(6, "pinMask"), B(7, "pinPos"),
      ]],
      x: this.w(this.c.n),
    });
    this.ex.writeBlas(this.c.nExt, 1);
    this.ex.runPass({
      shader: "blas", entry: "copy_v4",
      groups: [[this.blasBank(), B(4, "xTrial"), B(6, "position")]],
      x: this.w(this.c.nExt),
    });
    await this.ex.submitBatch(false);
    return { bisected: true, frac: lo };
  }

  /** Zero all contact counters (4 B control writes; MUST precede every rebuild:
   *  traverse/expand/compact append via atomics and never reset themselves).
   *  Stale counts would shift append indices, fake overflow, and poison the
   *  primTotal storage guards on the NEXT evaluation. */
  zeroContactCounters(): void {
    const z = new Uint32Array([0, 0, 0, 0]);
    const names = ["pairCount", "overflowFlag", "pairScanned",
      "primCountVT", "primCountEE",
      "contactCount", "contactOverflow", "contactScanned", "contactFail"];
    // G5.5 span counter rides the same hygiene when the coarse path is on
    // (explicit measurement passes only; hot-loop submit counts unchanged
    // when all coarse flags are off).
    if (this.coarseEnabled()) names.push("masContactSpan");
    for (const name of names) {
      this.ex.writeBuffer(name, z);
    }
  }

  /** Zero the PCG breakdown latch before each solve (sticky otherwise). */
  zeroBreakFlag(): void {
    this.ex.writeBuffer("breakFlag", new Float32Array(4));
  }

  /** Zero the exec marker + stamp a new epoch (control transfer, per eval). */
  resetMarker(epoch: number): void {
    this.pendingMarks = 0;
    this.ex.writeBuffer("execMarker", new Uint32Array([0, epoch >>> 0, 0, 0]));
  }

  /** Read the compacted contact count (4 B SYNC POINT per rebuild). */
  async readContactCount(): Promise<number> {
    const raw = await this.ex.readSmall("contactCount", 4, "contact-count", "scalar");
    return new Uint32Array(raw)[0];
  }

  /** Read the exec marker (16 B, part of the status pack). */
  async readMarker(): Promise<{ magic: number; epoch: number; mask: number; count: number }> {
    this.flushMarks();
    await this.ex.submitBatch(false);
    const raw = await this.ex.readSmall("execMarker", 16, "exec-marker", "status");
    const u = new Uint32Array(raw);
    return { magic: u[0], epoch: u[1], mask: u[2], count: u[3] };
  }

  /** Velocity update at step end: v = ((x - x0) / h) * damp (device-side). */
  async finishVelocities(invH: number, damp: number): Promise<void> {
    this.ex.beginBatch("finish-velocities");
    this.ex.writeBlas(this.c.nExt, 1);
    this.ex.runPass({
      shader: "blas", entry: "sub_v4",
      groups: [[this.blasBank(), B(4, "position"), B(5, "position0"), B(6, "velocity")]],
      x: this.w(this.c.n),
    });
    this.ex.writeBlas(this.c.nExt, invH);
    this.ex.runPass({
      shader: "blas", entry: "scale_v4",
      groups: [[this.blasBank(), B(5, "velocity")]],
      x: this.w(this.c.n),
    });
    this.ex.writeBlas(this.c.nExt, damp);
    this.ex.runPass({
      shader: "blas", entry: "scale_v4",
      groups: [[this.blasBank(), B(5, "velocity")]],
      x: this.w(this.c.n),
    });
    await this.ex.submitBatch(false);
  }
}
