// WebGpuSolver — Phase 2 G0 numerical path (CPU control, GPU-resident state).
//
// Sequencing (§G0.1–G0.10): device -> buffers -> predictor -> membrane ->
// bending -> gradient reduction -> HVP -> PCG -> Newton -> CPU/GPU comparison.
// Barrier/contact migrate to GPU ONLY after strip-pull + FEM directional tests
// agree (G0 exit criterion). G1 (broadphase) and G2 (CCD) are stubbed with
// explicit CPU-fallback notes — no half-migrated contact path.
//
// Readback invariant (protected aggressively):
//   GPU -> GPU -> GPU -> ... with CPU seeing ONLY the 64 B SolverStatus per
//   Newton iteration. Positions/gradients/contacts/PCG vectors are NEVER read
//   per-iteration; they use explicit debug/snapshot APIs (copyBufferToBuffer
//   + MAP_READ staging, see gpu-readback.ts).
//
// The CPU CpuSolver remains the golden reference: every GPU result is
// validated against it within gpu-tolerances.ts thresholds, never bit-identical.

import type { ClothSolver } from "../solver.js";
import type { ClothMaterial } from "../../physics/types.js";
import type { ClothScene } from "../../physics/scene.js";
import { enforcePins } from "../../physics/scene.js";
import { CpuSolver } from "../cpu-solver.js";
import { initGpuDevice, type GpuDeviceHandle } from "./gpu-device.js";
import { detectGpuCapabilities, type GpuCapabilityReport } from "./gpu-capabilities.js";
import {
  gpuLayoutBytes, gpuTotalBytes, packVec4Positions, unpackVec4Positions,
  encodeSimParams, decodeSolverStatus,
  type SimParams, type SolverStatus,
} from "./gpu-buffers.js";
import { isWebGPUAvailable } from "./webgpu-dts.js";
import { readbackBuffer } from "./gpu-readback.js";
import { SHADER_ENTRY_POINTS, WORKGROUP_DEFAULT, dispatchWorkgroups } from "./gpu-pipelines.js";
import { GpuBroadPhase } from "./gpu-broadphase.js";
import type { BroadPhase, CandidatePairs } from "../../collision/broadphase.js";
import {
  GpuContactSystem, type GpuContactSet, type GpuContactDiagnostics,
} from "./gpu-contact.js";
import { GpuExecutor } from "./gpu-executor.js";
import { DeviceNewtonDriver, DEFAULT_NEWTON_CONFIG } from "./gpu-newton.js";
import { GpuUniformSlot, UNIFORM_SLOT_STRIDE, nextPow2 } from "./gpu-buffers.js";
import { buildAdjacencyExclusions } from "../../collision/bvh.js";
import { buildMembraneIncidence, buildHingeIncidence } from "../../mesh/incidence.js";
import { buildSchwarzDomains, type SchwarzDomains } from "../../solver/schwarz.js";
import { buildCoarsePattern, type CoarsePattern } from "../../solver/coarse-csr.js";
import { bitonicPassAt } from "./gpu-broadphase.js";
import { COARSE_MAX_DEGREE } from "./gpu-buffers.js";
import {
  applyPreconditionerMode, type PreconditionerMode, type PreconditionerFallback,
} from "./gpu-newton.js";

/** G6A per-step production report: separates fixed-state solver evidence
 *  (preconditioner, iterations, residual) from end-to-end trajectory data
 *  (energy, submits, syncs) so line-search path differences can't masquerade
 *  as preconditioner cost. */
export interface StepPreconditionerReport {
  requestedPreconditioner: string;
  actualPreconditioner: string;
  fallback: PreconditionerFallback | null;
  fallbackReason: string | null;
  submits: number;
  syncs: number;
  newtonIters: number;
  finePcgIters: number;
  coarsePcgIters: number;
  residual: number;
  energy: number;
  /** G6B Armijo instrumentation (zeros when the sequential path runs). */
  armijoTrials: number;
  armijoBatches: number;
  armijoSelects: number;
  armijoCommits: number;
  armijoReadbacks: number;
  submitsPerNewton: number[];
  statusReadbacksPerNewton: number[];
  /** G6C.2 control path actually exercised this step. */
  controlPath: "sequential" | "batched-armijo" | "gpu-newton";
}

export interface GpuDiagnostics extends SolverStatus {
  deviceReady: boolean;
  timestampQuery: boolean;
  encodeMs: number;
  submitMs: number;
  readbackMs: number;
  statusBytes: number;
}

export interface GpuBenchmarkSample {
  vertexCount: number;
  triangleCount: number;
  hingeCount: number;
  contactCount: number;
  newtonIters: number;
  pcgIters: number;
  gpuComputeMs: number | null; // null without timestamp-query
  cpuEncodeMs: number;
  readbackMs: number;
  msPerStep: number;
  stepsPerSec: number;
}

export const GPU_TOLERANCES = {
  positionAbsM: 1e-4,
  positionRel: 1e-3,
  energyRel: 2e-3,
} as const;

export class WebGpuSolver implements ClothSolver {
  private scene: ClothScene | null = null;
  private cpuFallback: CpuSolver | null = null;
  private handle: GpuDeviceHandle | null = null;
  private buffers: Map<string, any> = new Map();
  private layoutBytes: ReturnType<typeof gpuLayoutBytes> | null = null;
  private statusStaging: { data: ArrayBuffer; timingMs: number } | null = null;
  private lastStatus: SolverStatus | null = null;
  private encodeMs = 0;
  private submitMs = 0;
  private readbackMs = 0;
  private simParams: SimParams | null = null;
  /** Number of hot-loop status-only readbacks this step (1 per Newton iter). */
  hotLoopReadbacks = 0;
  /** Number of hot-loop FULL-state readbacks (must stay 0 in G0). */
  forbiddenReadbacks = 0;
  private broadphase: GpuBroadPhase | null = null;
  private contacts: GpuContactSystem | null = null;
  // ---- G3 device path (real WGSL execution; mirror stays for reference) ----
  executor: GpuExecutor | null = null;
  driver: DeviceNewtonDriver | null = null;
  deviceMode: "mirror" | "device" = "mirror";
  /** Mirror-step count (explicit CPU-fallback accounting, test 16). */
  fallbackUses = 0;
  /** Device-step count. */
  deviceSteps = 0;
  /** Force the mirror path even with a device (fallback accounting test). */
  forceMirror = false;
  /** Snapshot positions/velocities to CPU per step (sync API cost). */
  snapshotOnStep = true;
  private simImage = {
    dt: 1 / 60,
    gravity: [0, -9.81, 0] as [number, number, number],
    contactCount: 0,
    newtonIteration: 0,
    lineSearchAlpha: 1,
    frictionMu: 0.3,
  };
  private contactCountNow = 0;
  /** G5C Schwarz static topology (built in initDevice, uploaded once). */
  schwarzTopo: SchwarzDomains | null = null;
  /** G5.5 coarse block-CSR static pattern (built in initDevice, uploaded once). */
  coarsePattern: CoarsePattern | null = null;
  lastDeviceStatus: import("./gpu-buffers.js").SolverStatus | null = null;
  lastMarker = { mask: 0, count: 0 };
  lastPcg = { resNorm: Infinity, breakdown: false };
  lastDirDot = 0;
  /** G6A production preconditioner mode (default stays jacobi). */
  preconditionerMode: PreconditionerMode = "jacobi";
  private modeExplicit = false;
  /** G6C.3 adaptive-K trial history: global selected trial index per accepted
   *  GPU-control round (-1 for failed rounds), reset per step. */
  trialHistory: number[] = [];

  /** Select the production preconditioner mode. Takes effect on the next
   *  device step; direct driver.cfg flag mutation (tests) keeps working
   *  whenever the mode was never set explicitly. */
  setPreconditionerMode(mode: PreconditionerMode): void {
    this.preconditionerMode = mode;
    this.modeExplicit = true;
  }
  /** Fallback records accumulated since initDevice (requested/actual/reason). */
  fallbackLog: PreconditionerFallback[] = [];
  /** Per-step report from the most recent device step (see StepPreconditionerReport). */
  lastStepReport: StepPreconditionerReport | null = null;

  // ---- ClothSolver contract (sync; delegates to CPU until GPU stepping lands) ----
  initialize(scene: ClothScene): void {
    this.scene = scene;
    this.cpuFallback = new CpuSolver();
    this.cpuFallback.initialize(scene);
    const n = scene.mesh.count;
    this.layoutBytes = gpuLayoutBytes({
      vertexCount: n,
      triangleCount: scene.mesh.triCount,
      hingeCount: scene.mesh.hinges.length,
      contactCapacity: 4096,
    });
    this.simParams = {
      dt: 1 / 60, invDt2: 3600,
      gravityX: scene.gravity[0], gravityY: scene.gravity[1], gravityZ: scene.gravity[2],
      vertexCount: n, triangleCount: scene.mesh.triCount,
      hingeCount: scene.mesh.hinges.length, contactCount: 0,
      newtonIteration: 0, pcgIteration: 0,
      lineSearchAlpha: 1, trustRegion: 0.002,
      barrierActivation: 0.002, barrierEpsilon: 1e-12, frictionMu: 0.3,
    };
  }

  step(dt: number): void {
    if (!this.scene || !this.cpuFallback) throw new Error("WebGpuSolver not initialized");
    // G0: GPU kernels are encoded/dispatched via stepGpu() in browser runtimes.
    // On headless CI (no navigator.gpu) the golden CPU path advances state so
    // the engine stays usable and parity tests can run against the FP32 mirror.
    void dt;
    const t0 = performance.now();
    this.cpuFallback.step(dt);
    const t1 = performance.now();
    this.encodeMs = t1 - t0;
  }

  setMaterial(material: ClothMaterial): void {
    if (!this.scene) throw new Error("WebGpuSolver not initialized");
    this.scene.material = { ...material };
    this.cpuFallback?.setMaterial(material);
  }

  pinVertex(vertexId: number, position?: [number, number, number]): void {
    this.cpuFallback?.pinVertex(vertexId, position);
  }

  unpinVertex(vertexId: number): void {
    this.cpuFallback?.unpinVertex(vertexId);
  }

  getPositions(): Float64Array {
    if (!this.scene) throw new Error("WebGpuSolver not initialized");
    return this.cpuFallback?.getPositions() ?? this.scene.positions;
  }

  getVelocities(): Float64Array {
    if (!this.scene) throw new Error("WebGpuSolver not initialized");
    return this.cpuFallback?.getVelocities() ?? this.scene.velocities;
  }

  // ---- async GPU lifecycle (G0.1–G0.2) ----

  /** Compute-only device request. Safe to call on CI: resolves `ready:false`. */
  async initGpu(): Promise<GpuDeviceHandle> {
    this.handle = await initGpuDevice();
    return this.handle;
  }

  getDevice(): any {
    return this.handle?.device ?? null;
  }

  async capabilityReport(): Promise<GpuCapabilityReport> {
    return detectGpuCapabilities();
  }

  /** Allocate all persistent GPU storage buffers for the current scene (G0.2). */
  async allocateGpuBuffers(contactCapacity = 4096): Promise<{ bytes: number; count: number }> {
    if (!this.scene) throw new Error("WebGpuSolver not initialized");
    if (!this.handle?.ready) {
      const bytes = gpuTotalBytes({
        vertexCount: this.scene.mesh.count,
        triangleCount: this.scene.mesh.triCount,
        hingeCount: this.scene.mesh.hinges.length,
        contactCapacity,
      });
      return { bytes, count: 0 }; // dry-run sizing on headless CI
    }
    const device = this.handle.device;
    const n = this.scene.mesh.count;
    const m = this.scene.mesh.triCount;
    const h = this.scene.mesh.hinges.length;
    const layout = gpuLayoutBytes({ vertexCount: n, triangleCount: m, hingeCount: h, contactCapacity });
    this.layoutBytes = layout;
    const STORAGE = 0x0080 | 0x0008; // STORAGE | COPY_DST (numeric; avoids DOM enum dep)
    const mk = (label: string, size: number, extra = 0) => {
      const buf = device.createBuffer({ label: `cloth-g0/${label}`, size: Math.max(size, 16), usage: STORAGE | extra });
      this.buffers.set(label, buf);
      return buf;
    };
    const COPY_SRC = 0x0004;
    mk("position", layout.position, COPY_SRC);
    mk("position0", layout.position0);
    mk("velocity", layout.velocity, COPY_SRC);
    mk("mass", layout.mass);
    mk("inverseMass", layout.inverseMass);
    mk("pinMask", layout.pinMask);
    mk("triangles", layout.triangles);
    mk("hinges", layout.hinges);
    mk("hingeMeta", layout.hingeMeta);
    mk("dmInv", layout.dmInv);
    mk("restArea", layout.restArea);
    mk("elementGradient", layout.elementGradient, COPY_SRC);
    mk("hingeGradient", layout.hingeGradient);
    mk("gradient", layout.gradient, COPY_SRC);
    mk("rhs", layout.rhs);
    mk("searchDirection", layout.searchDirection);
    mk("diag", layout.diag);
    mk("pcgResidual", layout.pcgResidual);
    mk("pcgSearch", layout.pcgSearch);
    mk("pcgAp", layout.pcgAp);
    mk("pcgZ", layout.pcgZ);
    mk("xTrial", layout.xTrial, COPY_SRC);
    mk("xReference", layout.xReference);
    mk("contactData", Math.max(layout.contactData, 16));
    mk("simParams", layout.simParams, 0x0040); // + UNIFORM via numeric flag
    mk("solverStatus", layout.solverStatus, COPY_SRC);
    mk("reduceScratch", Math.max(layout.reduceScratch, 16), COPY_SRC);
    // G1 broad-phase: STATIC topology (triangles already above) vs DYNAMIC bounds.
    mk("triAabb", Math.max(layout.triAabb, 16), COPY_SRC);
    mk("triCentroid", Math.max(layout.triCentroid, 16));
    mk("mortonKeys", Math.max(layout.mortonKeys, 16));
    mk("mortonPayload", Math.max(layout.mortonPayload, 16));
    mk("lbvhNodes", Math.max(layout.lbvhNodes, 16));
    mk("candidatePairs", Math.max(layout.candidatePairs, 16), COPY_SRC);
    mk("pairCount", Math.max(layout.pairCount, 16), COPY_SRC);
    mk("overflowFlag", Math.max(layout.overflowFlag, 16), COPY_SRC);
    mk("exclusionKeys", Math.max(layout.exclusionKeys, 16));
    // G2 contact pipeline: primitive scratch + compact records + counters.
    mk("primIdsVT", Math.max(layout.primIdsVT, 16));
    mk("primIdsEE", Math.max(layout.primIdsEE, 16));
    mk("vtSTD", Math.max(layout.vtSTD, 16));
    mk("vtR", Math.max(layout.vtR, 16));
    mk("eeSTD", Math.max(layout.eeSTD, 16));
    mk("eeR", Math.max(layout.eeR, 16));
    mk("primTOIVT", Math.max(layout.primTOIVT, 16), COPY_SRC);
    mk("primFlagVT", Math.max(layout.primFlagVT, 16), COPY_SRC);
    mk("primTOIEE", Math.max(layout.primTOIEE, 16), COPY_SRC);
    mk("primFlagEE", Math.max(layout.primFlagEE, 16), COPY_SRC);
    mk("primCountVT", Math.max(layout.primCountVT, 16), COPY_SRC);
    mk("primCountEE", Math.max(layout.primCountEE, 16), COPY_SRC);
    mk("contactTOI", Math.max(layout.contactTOI, 16), COPY_SRC);
    mk("contactCount", Math.max(layout.contactCount, 16), COPY_SRC);
    mk("contactOverflow", Math.max(layout.contactOverflow, 16), COPY_SRC);
    mk("contactScanned", Math.max(layout.contactScanned, 16), COPY_SRC);
    mk("contactFail", Math.max(layout.contactFail, 16), COPY_SRC);
    return { bytes: gpuTotalBytes({ vertexCount: n, triangleCount: m, hingeCount: h, contactCapacity }), count: this.buffers.size };
  }

  /** Initial scene upload (allowed transfer #1). Packs xyz -> vec4f. */
  async uploadScene(): Promise<void> {
    if (!this.scene) throw new Error("WebGpuSolver not initialized");
    if (!this.handle?.ready) return; // headless: sizing only
    const device = this.handle.device;
    const n = this.scene.mesh.count;
    const queue = device.queue;
    const put = (label: string, data: ArrayBufferView) => {
      const buf = this.buffers.get(label);
      if (!buf) return;
      queue.writeBuffer(buf, 0, data as any);
    };
    put("position", packVec4Positions(this.scene.positions, n));
    put("position0", packVec4Positions(this.scene.positions, n));
    put("velocity", packVec4Positions(this.scene.velocities, n));
    put("mass", Float32Array.from(this.scene.mesh.masses));
    const inv = new Float32Array(n);
    for (let i = 0; i < n; i++) inv[i] = this.scene.mesh.masses[i] > 0 ? 1 / this.scene.mesh.masses[i] : 0;
    put("inverseMass", inv);
    const pin = new Uint32Array(n);
    for (const id of this.scene.pinned.keys()) pin[id] = 1;
    put("pinMask", pin);
    put("triangles", this.scene.mesh.indices);
    const hinges = new Uint32Array(this.scene.mesh.hinges.length * 4);
    const meta = new Float32Array(this.scene.mesh.hinges.length * 4);
    this.scene.mesh.hinges.forEach((hh, k) => {
      hinges[k * 4] = hh.v0; hinges[k * 4 + 1] = hh.v1; hinges[k * 4 + 2] = hh.v2; hinges[k * 4 + 3] = hh.v3;
      meta[k * 4] = hh.restAngle; meta[k * 4 + 1] = hh.edgeLen; meta[k * 4 + 2] = hh.areaSum;
      meta[k * 4 + 3] = 0.5 * (this.scene!.material.bendWarp + this.scene!.material.bendWeft);
    });
    put("hinges", hinges);
    put("hingeMeta", meta);
    const dm = new Float32Array(this.scene.mesh.triCount * 4);
    for (let t = 0; t < this.scene.mesh.triCount; t++) {
      dm[t * 4] = this.scene.mesh.invDm[t * 4]; dm[t * 4 + 1] = this.scene.mesh.invDm[t * 4 + 1];
      dm[t * 4 + 2] = this.scene.mesh.invDm[t * 4 + 2]; dm[t * 4 + 3] = this.scene.mesh.invDm[t * 4 + 3];
    }
    put("dmInv", dm);
    put("restArea", Float32Array.from(this.scene.mesh.areas));
    this.writeSimParams();
  }

  private writeSimParams(): void {
    if (!this.handle?.ready || !this.simParams) return;
    const buf = this.buffers.get("simParams");
    if (!buf) return;
    this.handle.device.queue.writeBuffer(buf, 0, encodeSimParams(this.simParams) as any);
  }

  // ---- G3 real-device path ----

  /**
   * Bring up the real WebGPU device (Dawn/Node binding or browser), allocate
   * the full resident buffer set, and upload the scene. Returns false (mirror
   * mode preserved) when no binding/adapter exists — tests skip in that case.
   */
  async initDevice(opts: { contactCapacity?: number; pairCapacity?: number } = {}): Promise<boolean> {
    if (!this.scene) throw new Error("WebGpuSolver not initialized");
    const ex = await GpuExecutor.create();
    if (!ex) {
      this.deviceMode = "mirror";
      return false;
    }
    this.executor = ex;
    const n = this.scene.mesh.count;
    const m = this.scene.mesh.triCount;
    const h = this.scene.mesh.hinges.length;
    const contact = this.scene.contact;
    const ns = contact?.staticPos ? contact.staticPos.length / 3 : 0;
    // G5C Schwarz topology (STATIC, deterministic): fixed-<=8 domains padded
    // to 8 (target 8 / min 1 / max 8 — merge disabled so nothing exceeds the
    // pad width). Built BEFORE layout so Schwarz buffers are exactly sized.
    const schwarzTopo = buildSchwarzDomains(this.scene.mesh, this.scene.mesh.restPositions, 8, 1, 8);
    for (const mem of schwarzTopo.members) {
      if (mem.length > 8) throw new Error(`G5C topology: domain of ${mem.length} verts exceeds pad width 8`);
    }
    this.schwarzTopo = schwarzTopo;
    // G5.5 coarse block-CSR pattern (STATIC mesh topology): built BEFORE
    // layout so value buffers are exactly sized. Row degree must fit the
    // GPU kernel's private accumulator cap (COARSE_MAX_DEGREE).
    const coarsePattern = buildCoarsePattern(this.scene.mesh, schwarzTopo);
    {
      const D = coarsePattern.domains;
      for (let d = 0; d < D; d++) {
        const deg = coarsePattern.rowOffsets[d + 1] - coarsePattern.rowOffsets[d];
        if (deg > COARSE_MAX_DEGREE) {
          throw new Error(`G5.5 pattern: row ${d} degree ${deg} exceeds kernel cap ${COARSE_MAX_DEGREE}`);
        }
      }
    }
    this.coarsePattern = coarsePattern;
    const layout = gpuLayoutBytes({
      vertexCount: n, triangleCount: m, hingeCount: h,
      contactCapacity: opts.contactCapacity ?? 4096,
      pairCapacity: opts.pairCapacity ?? 16384,
      staticCount: ns,
      schwarzDomainCount: schwarzTopo.members.length,
      coarseNnz: coarsePattern.nnz,
    });
    this.layoutBytes = layout;
    // Allocate every layout buffer in the executor (names match 1:1).
    const U = ex.usage;
    const STORAGE = U.STORAGE | U.COPY_DST | U.COPY_SRC;
    const UNIFORM = U.UNIFORM | U.COPY_DST;
    for (const [name, bytes] of Object.entries(layout) as Array<[string, number]>) {
      const isUniform = name === "simParams" || name === "uniformBank";
      ex.ensureBuffer(name, bytes, isUniform ? UNIFORM : STORAGE, `g3/${name}`);
    }
    this.uploadSceneDevice();
    const counts = {
      n, nExt: n + ns, m, h,
      cap: opts.contactCapacity ?? 4096,
      pairs: opts.pairCapacity ?? 16384,
      schwarzDoms: schwarzTopo.members.length,
    };
    const driver = new DeviceNewtonDriver(ex, counts);
    driver.exclusionCount = this.exclusionCountExact;
    driver.refreshSimParams = (alphaEff: number) => {
      this.simImage.lineSearchAlpha = alphaEff;
      this.pushSimParams();
    };
    this.driver = driver;
    this.deviceMode = "device";
    return true;
  }

  private exclusionCountDevice(): number {
    // Upper bound only (dispatch sizing); the traverse uniform ALWAYS carries
    // the exact uploaded key count (see exclusionCountExact).
    return this.scene ? this.scene.mesh.triCount * 6 : 0;
  }

  private exclusionCountExact = 0;

  /** Push the cached SimParams image (64 B control upload). */
  pushSimParams(): void {
    if (!this.executor || !this.driver) return;
    this.driver.uploadSimParams({
      dt: this.simImage.dt,
      gravity: this.simImage.gravity,
      contactCount: this.contactCountNow,
      newtonIteration: this.simImage.newtonIteration,
      lineSearchAlpha: this.simImage.lineSearchAlpha,
      frictionMu: this.simImage.frictionMu,
    });
  }

  /** Upload all scene state + zero scratch (allowed init transfer #1). */
  private uploadSceneDevice(): void {
    const ex = this.executor!;
    const scene = this.scene!;
    const n = scene.mesh.count;
    const contact = scene.contact;
    const ns = contact?.staticPos ? contact.staticPos.length / 3 : 0;
    const put = (label: string, data: ArrayBufferView): void => {
      ex.writeBuffer(label, data);
    };
    const packExt = (xyz: ArrayLike<number>): Float32Array => {
      const out = new Float32Array((n + ns) * 4);
      for (let i = 0; i < n; i++) {
        out[i * 4] = Math.fround(xyz[i * 3]);
        out[i * 4 + 1] = Math.fround(xyz[i * 3 + 1]);
        out[i * 4 + 2] = Math.fround(xyz[i * 3 + 2]);
      }
      if (contact?.staticPos) {
        for (let i = 0; i < ns; i++) {
          out[(n + i) * 4] = contact.staticPos[i * 3];
          out[(n + i) * 4 + 1] = contact.staticPos[i * 3 + 1];
          out[(n + i) * 4 + 2] = contact.staticPos[i * 3 + 2];
        }
      }
      return out;
    };
    put("position", packExt(scene.positions));
    put("position0", packExt(scene.positions));
    put("xTrial", packExt(scene.positions));
    put("hvpXMinus", packExt(scene.positions));
    put("slip", new Float32Array((n + ns) * 4));
    put("velocity", packVec4Positions(scene.velocities, n));
    put("mass", Float32Array.from(scene.mesh.masses));
    const inv = new Float32Array(n);
    for (let i = 0; i < n; i++) inv[i] = scene.mesh.masses[i] > 0 ? 1 / scene.mesh.masses[i] : 0;
    put("inverseMass", inv);
    const pin = new Uint32Array(n);
    const pinPos = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) {
      pinPos[i * 4] = scene.positions[i * 3];
      pinPos[i * 4 + 1] = scene.positions[i * 3 + 1];
      pinPos[i * 4 + 2] = scene.positions[i * 3 + 2];
    }
    for (const [id, p] of scene.pinned) {
      pin[id] = 1;
      pinPos[id * 4] = p[0]; pinPos[id * 4 + 1] = p[1]; pinPos[id * 4 + 2] = p[2];
    }
    put("pinMask", pin);
    put("pinPos", pinPos);
    put("triangles", scene.mesh.indices);
    const hinges = new Uint32Array(scene.mesh.hinges.length * 4);
    const meta = new Float32Array(scene.mesh.hinges.length * 4);
    scene.mesh.hinges.forEach((hh, k) => {
      hinges[k * 4] = hh.v0; hinges[k * 4 + 1] = hh.v1; hinges[k * 4 + 2] = hh.v2; hinges[k * 4 + 3] = hh.v3;
      meta[k * 4] = hh.restAngle; meta[k * 4 + 1] = hh.edgeLen; meta[k * 4 + 2] = hh.areaSum;
      meta[k * 4 + 3] = 0.5 * (scene.material.bendWarp + scene.material.bendWeft);
    });
    put("hinges", hinges);
    put("hingeMeta", meta);
    const dm = new Float32Array(scene.mesh.triCount * 4);
    for (let t = 0; t < scene.mesh.triCount; t++) {
      dm[t * 4] = scene.mesh.invDm[t * 4]; dm[t * 4 + 1] = scene.mesh.invDm[t * 4 + 1];
      dm[t * 4 + 2] = scene.mesh.invDm[t * 4 + 2]; dm[t * 4 + 3] = scene.mesh.invDm[t * 4 + 3];
    }
    put("dmInv", dm);
    put("restArea", Float32Array.from(scene.mesh.areas));
    // bitonic INF tail: morton lanes [m, P) start as key/payload 0xFFFFFFFF
    // (written once; the sort permutes but never clears them, and LBVH only
    // consumes the first m lanes, which the network keeps real-and-sorted)
    {
      const P2 = nextPow2(Math.max(scene.mesh.triCount, 1));
      const tail = Math.max(P2 - scene.mesh.triCount, 0);
      if (tail > 0) {
        const fill = new Uint32Array(tail).fill(0xffffffff);
        const kb = this.executor!.buffers.get("mortonKeys");
        const pb = this.executor!.buffers.get("mortonPayload");
        if (kb) this.executor!.device.queue.writeBuffer(kb, scene.mesh.triCount * 4, fill);
        if (pb) this.executor!.device.queue.writeBuffer(pb, scene.mesh.triCount * 4, fill);
      }
    }
    // G6C.1 indexed-sort static table: (P, stage, sub, 0) per sub-pass t
    // (depends only on mesh size; uploaded once, read by sort_step_indexed).
    {
      const P2 = nextPow2(Math.max(scene.mesh.triCount, 1));
      const stages = Math.log2(P2);
      const T = Math.max(1, (stages * (stages + 1)) / 2);
      const table = new Uint32Array(T * 4);
      for (let t = 0; t < T; t++) {
        const { stage, sub } = bitonicPassAt(t);
        table[t * 4] = P2;
        table[t * 4 + 1] = stage;
        table[t * 4 + 2] = sub;
        table[t * 4 + 3] = 0;
      }
      put("sortParams", table);
      put("sortCursor", new Uint32Array([0, 0, 0, 0]));
    }
    // exclusion keys as (lo,hi) vec2u pairs (G2 encoding, uploaded once)
    {
      const excl = buildAdjacencyExclusions(scene.mesh.indices, scene.mesh.triCount);
      const arr = new Uint32Array(Math.max(excl.size, 1) * 2);
      let k = 0;
      const sorted = [...excl].sort((a, b) => a - b);
      for (const key of sorted) {
        const lo = Math.floor(key / 1000003);
        arr[k * 2] = lo >>> 0;
        arr[k * 2 + 1] = (key - lo * 1000003) >>> 0;
        k++;
      }
      put("exclusionKeys", arr);
      this.bankU(GpuUniformSlot.ExclusionCount, excl.size);
      this.exclusionCountExact = excl.size;
      if (this.driver) this.driver.exclusionCount = excl.size;
    }
    // G4B incidence maps (STATIC topology, uploaded once)
    {
      const emap = buildMembraneIncidence(scene.mesh.indices, scene.mesh.triCount, n);
      put("vertexElementOffsets", emap.offsets);
      put("vertexElementIds", emap.ids);
      put("vertexElementCorners", emap.corners);
      const hmap = buildHingeIncidence(scene.mesh.hinges, scene.mesh.hinges.length, n);
      put("vertexHingeOffsets", hmap.offsets);
      put("vertexHingeIds", hmap.ids);
      put("vertexHingeCorners", hmap.corners);
    }
    // G5C Schwarz static topology (uploaded once; factors rebuilt per solve)
    {
      const topo = this.schwarzTopo;
      if (!topo) throw new Error("uploadSceneDevice: missing Schwarz topology (initDevice must build it)");
      const allocDoms = Math.floor((this.layoutBytes?.schwarzVerts ?? 32) / 32);
      if (topo.members.length > allocDoms) {
        throw new Error(`uploadSceneDevice: Schwarz domains ${topo.members.length} exceed allocation ${allocDoms}`);
      }
      put("schwarzDomain", topo.domainOf);
      put("schwarzLocal", topo.localOf);
      const D = topo.members.length;
      const verts = new Uint32Array(Math.max(D, 1) * 8).fill(0xffffffff);
      topo.members.forEach((mem, d) => mem.forEach((v, l) => { verts[d * 8 + l] = v; }));
      put("schwarzVerts", verts);
    }
    // G5.5 coarse block-CSR static pattern (uploaded once; values per solve)
    {
      const pat = this.coarsePattern;
      if (!pat) throw new Error("uploadSceneDevice: missing coarse pattern (initDevice must build it)");
      const allocNnz = Math.floor((this.layoutBytes?.coarseColIndices ?? 4) / 4);
      if (pat.nnz > allocNnz) {
        throw new Error(`uploadSceneDevice: coarse nnz ${pat.nnz} exceeds allocation ${allocNnz}`);
      }
      put("coarseRowOffsets", pat.rowOffsets);
      put("coarseColIndices", pat.colIndices);
      const rows = new Uint32Array(Math.max(pat.nnz, 1));
      for (let d = 0; d < pat.domains; d++) {
        for (let p = pat.rowOffsets[d]; p < pat.rowOffsets[d + 1]; p++) rows[p] = d;
      }
      put("coarseBlockRows", rows);
    }
    // zero scratch that must read 0 on first use
    const zeroNames = ["laggedN", "contactForceZero", "breakFlag", "execMarker", "contactCount",
      "contactOverflow", "contactScanned", "contactFail", "pairCount", "overflowFlag",
      "primCountVT", "primCountEE", "alphaSlot", "betaSlot", "rzPrevSlot",
      "schwarzMat", "schwarzInv", "schwarzFlag",
      "masCoarseR", "masCoarseZ", "masCoarseDiag",
      "coarseBlockValues", "coarseX", "coarseR", "coarseP", "coarseAp", "coarseZ", "coarseProd",
      "coarseAlpha", "coarseBeta", "coarseRzPrev", "coarseBreak", "masContactSpan"];
    for (const z of zeroNames) {
      const bytes = this.layoutBytes?.[z as keyof typeof this.layoutBytes] as number ?? 16;
      put(z, new Uint8Array(Math.max(16, bytes)));
    }
    // material + contact uniforms into the bank (control transfer)
    const mt = scene.material;
    this.bankF(GpuUniformSlot.MatC00, mt.stretchWarp);
    this.bankF(GpuUniformSlot.MatC11, mt.stretchWeft);
    this.bankF(GpuUniformSlot.MatC01, mt.stretchCoupling);
    this.bankF(GpuUniformSlot.MatG, mt.shear);
    this.bankF(GpuUniformSlot.MatThickness, mt.thickness);
    const cp = contact?.params;
    this.bankF(GpuUniformSlot.DHat, cp?.dHatM ?? 0.002);
    this.bankF(GpuUniformSlot.Kappa, cp?.kappaJ ?? 50);
    this.bankF(GpuUniformSlot.Mu, cp?.frictionMu ?? 0.3);
    this.bankF(GpuUniformSlot.FricEps, cp?.frictionEpsM ?? 1e-4);
    this.bankF(GpuUniformSlot.FloorY, contact?.floorY ?? 1e30);
    this.bankU(GpuUniformSlot.FloorOn, contact?.floorY !== null && contact?.floorY !== undefined ? 1 : 0);
    this.bankF(GpuUniformSlot.DMin, cp?.dMinM ?? 1e-4);
    this.bankF(GpuUniformSlot.Thickness, cp?.dMinM ?? 1e-4);
    this.simImage.gravity = [scene.gravity[0], scene.gravity[1], scene.gravity[2]];
    this.simImage.frictionMu = cp?.frictionMu ?? 0.3;
  }

  private bankF(slot: number, v: number): void {
    const a = new Float32Array(4);
    a[0] = v;
    this.executor!.writeBuffer("uniformBank", a, slot * UNIFORM_SLOT_STRIDE);
  }

  private bankU(slot: number, v: number): void {
    const a = new Uint32Array(4);
    a[0] = v >>> 0;
    this.executor!.writeBuffer("uniformBank", a, slot * UNIFORM_SLOT_STRIDE);
  }

  /**
   * G0 Newton control loop (CPU-side, §15). Encodes evaluation + PCG passes,
   * reads back ONLY the compact status buffer per iteration. The `dispatch`
   * callback is the browser command-encoder; on headless CI the loop runs
   * against the CPU golden path with status synthesized locally.
   */
  async stepGpu(dt: number, opts: { newtonIters?: number } = {}): Promise<GpuDiagnostics> {
    if (!this.scene) throw new Error("WebGpuSolver not initialized");
    // G3: real device path when available (and not forced to mirror).
    if (this.deviceMode === "device" && this.executor && this.driver && !this.forceMirror) {
      return this.stepGpuDevice(dt, opts);
    }
    this.fallbackUses++; // explicit CPU-fallback accounting (test 16)
    const newtonIters = opts.newtonIters ?? 10;
    const tEnc0 = performance.now();
    this.simParams!.dt = dt;
    this.simParams!.invDt2 = 1 / (dt * dt);
    // Command encoding would happen here on a real device:
    // predictor -> membrane -> bending -> assemble -> newton-rhs ->
    // jacobi -> [hvp -> pcg-update -> reduce] x pcgIters -> apply-step ->
    // diagnostics. Each stage is GPU->GPU; no JS readback between passes.
    void dispatchWorkgroups(this.scene.mesh.count, WORKGROUP_DEFAULT);
    void SHADER_ENTRY_POINTS;
    const tEnc1 = performance.now();
    this.encodeMs = tEnc1 - tEnc0;
    const tSub0 = performance.now();
    // Advance golden reference (headless parity) — on-device this is replaced
    // by queue.submit + a single status-buffer map per Newton iteration.
    this.cpuFallback!.step(dt);
    if (this.scene.contact) enforcePins(this.scene, this.scene.positions);
    const tSub1 = performance.now();
    this.submitMs = tSub1 - tSub0;
    // SYNCHRONIZATION POINT (the ONLY hot-loop readback): 64 B status.
    // On-device: copyBufferToBuffer(statusBuf -> staging) + mapAsync(READ).
    const tRb0 = performance.now();
    this.hotLoopReadbacks += newtonIters; // one compact read per Newton iter
    // forbiddenReadbacks stays 0: positions/gradients/contacts never mapped here.
    const status = this.synthesizeStatus();
    this.lastStatus = status;
    const tRb1 = performance.now();
    this.readbackMs = tRb1 - tRb0;
    return {
      ...status,
      deviceReady: this.handle?.ready ?? false,
      timestampQuery: this.handle?.timestampQueryEnabled ?? false,
      encodeMs: this.encodeMs, submitMs: this.submitMs, readbackMs: this.readbackMs,
      statusBytes: 64,
    };
  }

  private synthesizeStatus(): SolverStatus {
    // Headless stand-in for the diagnostics.wgsl 64 B payload: computed from
    // the golden CPU state so parity tests exercise the same thresholds.
    const s = this.cpuFallback!.lastStats;
    const finite = Number(Number.isFinite(s?.energy ?? 1) && Number.isFinite(s?.gradNorm ?? 1));
    return {
      energy: s?.energy ?? 0, barrierEnergy: s?.contact?.barrierEnergy ?? 0,
      gradNorm: s?.gradNorm ?? 0, directionDotGradient: 0,
      minDistance: s?.contact?.minDistance ?? Infinity, minToi: s?.contact?.minTOI ?? Infinity,
      finite, ccdSafe: 1, barrierSafe: 1,
      pcgBreakdown: 0, converged: s?.converged ? 1 : 0,
    };
  }

  // ---- G3 device Newton loop (CPU Armijo control, GPU everything else) ----

  private contactParamsNow(): {
    dHat: number; kappa: number; mu: number; fricEps: number;
    floorY: number; floorOn: number; dMin: number; contactCapacity: number;
  } {
    const cp = this.scene!.contact?.params;
    const floorY = this.scene!.contact?.floorY;
    return {
      dHat: cp?.dHatM ?? 0.002,
      kappa: cp?.kappaJ ?? 50,
      mu: cp?.frictionMu ?? 0.3,
      fricEps: cp?.frictionEpsM ?? 1e-4,
      floorY: floorY ?? 1e30,
      floorOn: floorY !== null && floorY !== undefined ? 1 : 0,
      dMin: cp?.dMinM ?? 1e-4,
      contactCapacity: this.driver!.c.cap,
    };
  }

  private materialNow(): { c00: number; c11: number; c01: number; g: number; thickness: number } {
    const mt = this.scene!.material;
    return { c00: mt.stretchWarp, c11: mt.stretchWeft, c01: mt.stretchCoupling, g: mt.shear, thickness: mt.thickness };
  }

  private jacobiBeta(): number {
    const mt = this.scene!.material;
    return Math.max(mt.stretchWarp, mt.stretchWeft, mt.shear) * mt.thickness * 0.1 + 1e-6;
  }

  /**
   * Configure + push step-level SimParams (dt, gravity, friction) before a
   * device evaluation sequence. Public for device tests.
   */
  configureStep(dt: number): void {
    if (!this.scene) throw new Error("WebGpuSolver not initialized");
    this.simImage.dt = dt;
    this.simImage.gravity = [this.scene.gravity[0], this.scene.gravity[1], this.scene.gravity[2]];
    this.simImage.contactCount = 0;
    this.simImage.newtonIteration = 0;
    this.simImage.lineSearchAlpha = 1;
    this.simImage.frictionMu = this.scene.contact?.params.frictionMu ?? 0.3;
    this.pushSimParams();
  }

  /**
   * One Newton evaluation at a state buffer: G1+G2 rebuild, FEM, gradient,
   * RHS/Jacobi, diagnostics. Submits 3 batches, syncs the contact count
   * (4 B) and the 64 B status + 16 B marker. No state readback.
   * Public for device parity tests (production path: stepGpuDevice).
   */
  async evaluateNewtonState(
    atTrial: boolean, evalIndex: number, dt: number,
  ): Promise<{ status: SolverStatus; markerMask: number; markerCount: number; contactCount: number }> {
    const ex = this.executor!;
    const driver = this.driver!;
    const posBuf = atTrial ? "xTrial" : "position";
    driver.resetMarker(evalIndex + 1);
    driver.zeroContactCounters();
    // Batch A: broadphase + CCD/compaction (all GPU->GPU).
    ex.beginBatch(atTrial ? "eval-trial-contact" : "eval-contact");
    driver.broadphasePasses(0.002, atTrial);
    driver.contactPasses(this.contactParamsNow(), posBuf);
    await ex.submitBatch(false);
    // SYNC POINT (tiny): compacted contact count -> SimParams refresh.
    const contactCount = await driver.readContactCount();
    this.contactCountNow = contactCount;
    this.simImage.newtonIteration = evalIndex;
    this.pushSimParams();
    // Batch B: FEM + gradient + RHS/Jacobi at the same state.
    // (Trial evals also refresh rhs so diagnostics norms the trial-state
    // total residual, exactly like the CPU convergence check.)
    ex.beginBatch(atTrial ? "eval-trial-fem" : "eval-fem");
    driver.femPasses(this.materialNow(), atTrial);
    if (atTrial) {
      driver.contactDiagPass();
      driver.rhsAt("xTrial");
    } else {
      driver.rhsJacobi(this.jacobiBeta());
    }
    await ex.submitBatch(false);
    // Batch C: diagnostics + status/marker pack.
    const { status, markerMask, markerCount } = await driver.diagnosticsAt(posBuf);
    // Accounting: one compact (64 B status + 16 B marker) sync per evaluation.
    this.hotLoopReadbacks++;
    // Freshest device-execution evidence (every eval, accepted or not).
    this.lastMarker = { mask: markerMask, count: markerCount };
    void posBuf;
    void dt;
    return { status, markerMask, markerCount, contactCount };
  }

  /** G6B per-step Armijo counters (reset at each stepGpuDevice entry). */
  private armijoTrials = 0;
  private armijoBatches = 0;
  private armijoSelects = 0;
  private armijoCommits = 0;
  private armijoReadbacks = 0;

  /**
   * G6B batched Armijo search: evaluates K alphas per GPU batch with one
   * status sync, committing at the selected alpha through the standard
   * single-trial path (identical records/gradient/commit semantics).
   * Returns the CPU-loop-compatible base alpha (effective / trustScale).
   */
  private async armijoBatchedSearch(opts: {
    E0: number; gtdx: number; trustScale: number; dMin: number;
    newtonIndex: number; pcgBreakdown: boolean; dt: number;
  }): Promise<{ alpha: number; accepted: boolean; status: import("./gpu-buffers.js").SolverStatus | null }> {
    const driver = this.driver!;
    const budget = driver.cfg.armijoIters;
    void opts.dMin; // validity uses device-side DMin (same bank value the CPU path reads)
    // G6C.3 adaptive widths: seed from the step's trial history (cold start
    // honors the configured default); each batch re-predicts from its outcome.
    let lastTrial: number | null = this.trialHistory.length > 0
      ? this.trialHistory[this.trialHistory.length - 1]
      : null;
    let alphaBase = 1;
    let trialsUsed = 0;
    while (trialsUsed < budget) {
      const Kwant = driver.cfg.adaptiveK
        ? adaptiveBatchK(lastTrial, driver.cfg.armijoBatchK)
        : Math.max(1, Math.min(8, driver.cfg.armijoBatchK));
      const K = Math.min(Kwant, budget - trialsUsed);
      // Cap-bounded candidate loops for this batch (live count re-synced by
      // the next evaluateNewtonState automatically).
      this.contactCountNow = driver.c.cap;
      this.simImage.newtonIteration = opts.newtonIndex * 100 + trialsUsed;
      this.pushSimParams();
      const st = await driver.armijoBatch({
        E0: opts.E0, gtdx: opts.gtdx, alphaBase, beta: 0.5, K,
        trustScale: opts.trustScale,
        mat: this.materialNow(), contact: this.contactParamsNow(),
        evalIndexBase: opts.newtonIndex * 100 + trialsUsed,
        pcgBreakdown: opts.pcgBreakdown, newtonConverged: false,
      });
      this.hotLoopReadbacks++; // THE one status sync per batch
      this.armijoReadbacks++;
      this.armijoBatches++;
      this.armijoSelects++;
      const adv = st.trialsEvaluated > 0 ? st.trialsEvaluated : K;
      trialsUsed += adv;
      this.armijoTrials += adv;
      lastTrial = -1;
      if (st.accepted) {
        // Commit WITHOUT re-evaluation syncs: xTrial is rebuilt at the
        // selected alpha and records refreshed sync-free; energy/gradNorm
        // for bookkeeping come from the batch status (same kernels that
        // produced the candidate verdict, so bit-identical to a re-read).
        // NOTE: selectedAlpha is already effective (trust folds on GPU).
        const ex = this.executor!;
        await driver.applyTrial(st.selectedAlpha);
        ex.beginBatch("commit-refresh");
        driver.rebuildTrialPasses(this.materialNow(), this.contactParamsNow());
        await ex.submitBatch(false);
        await driver.acceptTrial();
        this.armijoCommits++;
        const normToi = st.minToi >= 2.0 - 1e-6 ? Infinity : st.minToi;
        const normDist = st.minDist >= 1e29 ? Infinity : st.minDist;
        const status: import("./gpu-buffers.js").SolverStatus = {
          energy: st.energy,
          barrierEnergy: 0,
          gradNorm: st.gradNorm,
          directionDotGradient: opts.gtdx,
          minDistance: normDist,
          minToi: normToi,
          finite: st.finite ? 1 : 0,
          ccdSafe: 1,
          barrierSafe: 1,
          pcgBreakdown: 0,
          converged: 0,
        };
        lastTrial = trialsUsed - K + st.selectedIndex;
        return { alpha: st.selectedAlpha / opts.trustScale, accepted: true, status };
      }
      alphaBase *= Math.pow(0.5, adv);
    }
    return { alpha: alphaBase, accepted: false, status: null };
  }

  /**
   * G6B batch prologue (public for device tests): switch simParams to
   * cap-bounded candidate loops for an Armijo batch. The next
   * evaluateNewtonState re-syncs the live count automatically.
   */
  beginArmijoBatch(newtonIteration: number): void {
    if (!this.executor || !this.driver) return;
    this.contactCountNow = this.driver.c.cap;
    this.simImage.newtonIteration = newtonIteration;
    this.pushSimParams();
  }

  /** G6C.2 batch-width schedule for one Newton round (static unroll). */
  private batchKsFor(): number[] {
    const driver = this.driver!;
    const K0 = Math.max(1, Math.min(8, driver.cfg.armijoBatchK));
    const Ks: number[] = [];
    let rem = driver.cfg.armijoIters;
    while (rem > 0) {
      const k = Math.min(K0, rem);
      Ks.push(k);
      rem -= k;
    }
    return Ks;
  }

  /**
   * G6C.2 GPU-controlled Newton loop: unrolled rounds with every decision on
   * device (convergence, descent, Armijo batches, commit). Exactly ONE
   * compact status readback per round; the CPU breaks or continues from it.
   * Per-round control writes are amortized (no syncs); submits stay roughly
   * flat vs the sequential path (same passes + tiny control kernels).
   */
  private async stepNewtonGpuControlled(
    evStatus: import("./gpu-buffers.js").SolverStatus,
    newtonIters: number,
    submitsPerNewton: number[],
    statusReadbacksPerNewton: number[],
  ): Promise<{
    converged: boolean; acceptedFinal: boolean;
    newtonItersUsed: number; pcgSolves: number; pcgItersTotal: number; E0: number;
  }> {
    const ex = this.executor!;
    const driver = this.driver!;
    const cfg = DEFAULT_NEWTON_CONFIG;
    // Per-step persistent setup (amortized control writes, encoder null).
    this.contactCountNow = driver.c.cap;
    this.pushSimParams();
    this.bankF(GpuUniformSlot.NewtonTol, cfg.gradTol);
    this.bankF(GpuUniformSlot.NewtonTrust, cfg.trust);
    ex.writeBuffer("e0Store", new Float32Array([evStatus.energy, 0, 0, 0]));
    let E0 = evStatus.energy;
    let converged = false;
    let acceptedFinal = true;
    let newtonItersUsed = 0;
    let pcgSolves = 0;
    let pcgItersTotal = 0;
    // Fast path mirrors the sequential k=0 convergence check (no rounds).
    if (!Number.isFinite(evStatus.gradNorm)) throw new Error("device Newton: non-finite gradient");
    if (evStatus.gradNorm < cfg.gradTol) {
      converged = true;
    } else {
      const batchKs = this.batchKsFor();
      for (let k = 0; k < newtonIters; k++) {
        newtonItersUsed++;
        const submitsNewton0 = ex.ledger.submits;
        const syncsNewton0 = this.hotLoopReadbacks;
        ex.writeBuffer("newtonCtl", new Float32Array(16));
        const nst = new Float32Array(20);
        nst[0] = k;
        ex.writeBuffer("newtonStatus", nst);
        const st = await driver.newtonRound({
          round: k,
          beta: this.jacobiBeta(),
          mat: this.materialNow(),
          contact: this.contactParamsNow(),
          batchKs,
          evalIndexBase: k * 100,
        });
        pcgSolves++;
        pcgItersTotal += cfg.pcgIters;
        if (driver.lastFallback) this.fallbackLog.push(driver.lastFallback);
        if (!st.directionValid) {
          this.fallbackLog.push({
            requested: driver.lastActualMethod,
            actual: "jacobi-descent",
            reason: st.pcgBreakdown ? "pcg-breakdown" : "non-descent-direction",
          });
        }
        this.lastPcg = { resNorm: st.residual, breakdown: st.pcgBreakdown };
        this.lastDirDot = st.gtdx;
        this.trialHistory.push(st.armijoAccepted ? st.selectedTrialIndex : -1);
        E0 = st.energy;
        this.lastDeviceStatus = {
          energy: st.energy,
          barrierEnergy: 0,
          gradNorm: st.gradNorm,
          directionDotGradient: st.gtdx,
          minDistance: st.minDistance,
          minToi: st.minToi,
          // State finiteness (NOT Newton success: a reject-all round leaves
          // a finite state behind by predicated-commit construction).
          finite: Number.isFinite(st.energy) && Number.isFinite(st.gradNorm) ? 1 : 0,
          ccdSafe: st.ccdFailure ? 0 : 1,
          barrierSafe: st.barrierFailure ? 0 : 1,
          pcgBreakdown: st.pcgBreakdown ? 1 : 0,
          converged: st.converged ? 1 : 0,
        };
        submitsPerNewton.push(ex.ledger.submits - submitsNewton0);
        statusReadbacksPerNewton.push(this.hotLoopReadbacks - syncsNewton0);
        if (st.converged || st.failure) {
          converged = st.converged;
          if (st.failure) acceptedFinal = false;
          break;
        }
      }
    }
    return { converged, acceptedFinal, newtonItersUsed, pcgSolves, pcgItersTotal, E0 };
  }

  /** Scalar dot(a, b) with one tiny sync (SYNC POINT). */
  private async dotScalar(aBuf: string, bBuf: string): Promise<number> {
    const ex = this.executor!;
    const driver = this.driver!;
    ex.beginBatch("dot");
    driver.mulInto(aBuf, bBuf);
    await ex.submitBatch(false);
    return driver.reduceSum("pcgProd", driver.n3);
  }

  private async stepGpuDevice(dt: number, opts: { newtonIters?: number }): Promise<GpuDiagnostics> {
    const scene = this.scene!;
    const ex = this.executor!;
    const driver = this.driver!;
    const cfg = DEFAULT_NEWTON_CONFIG;
    const newtonIters = opts.newtonIters ?? cfg.maxNewton;
    const t0 = performance.now();
    // G6A: explicit production mode maps onto driver flags (direct flag
    // mutation keeps working when no mode was ever selected).
    if (this.modeExplicit) applyPreconditionerMode(driver.cfg, this.preconditionerMode);
    const requestedMode = this.modeExplicit ? this.preconditionerMode : "flags-direct";
    const submits0 = ex.ledger.submits;
    const syncs0 = this.hotLoopReadbacks;
    let newtonItersUsed = 0;
    let pcgSolves = 0;
    // G6B per-step Armijo instrumentation.
    this.armijoTrials = 0;
    this.armijoBatches = 0;
    this.armijoSelects = 0;
    this.armijoCommits = 0;
    this.armijoReadbacks = 0;
    this.trialHistory = [];
    const submitsPerNewton: number[] = [];
    const statusReadbacksPerNewton: number[] = [];
    this.simImage.dt = dt;
    this.simImage.contactCount = 0;
    this.simImage.lineSearchAlpha = 1;
    this.pushSimParams();
    // Deterministic status lanes: diagnostics rewrites [0..8]+[11..15] every
    // eval, but [3] dirDot / [9] breakdown / [10] converged are CPU-owned.
    // Zero once per step so a stale lane can never leak across steps.
    ex.writeBuffer("solverStatus", new Float32Array(16));
    // Begin-step transfer FIRST: position0 = accepted state (CCD/slip/velocity
    // segment start), then the predictor overwrites position with y_hat.
    await driver.beginStepCopy();
    // Predictor (pins exact via enforce_pins).
    await driver.predictorPass();
    // E0 evaluation at y_hat.
    let ev = await this.evaluateNewtonState(false, 0, dt);
    const dMin = this.contactParamsNow().dMin;
    // Tunneling predictor (fast impact): bisect back to a CCD-valid state
    // before Newton starts (mirrors CPU newton.ts predictor validation).
    // Costs nothing when y_hat is already valid (single comparison).
    {
      const s0 = ev.status;
      const tunneled = s0.finite === 1 &&
        (s0.minDistance <= dMin || (s0.minToi > 0 && s0.minToi < 1 - 1e-9));
      if (tunneled) {
        let probeStatus = s0;
        let idx = 0;
        const res = await driver.bisectPredictor(
          dMin,
          async () => probeStatus,
          async () => {
            probeStatus = (await this.evaluateNewtonState(true, 500 + idx++, dt)).status;
          },
        );
        void res;
        // Re-evaluate E0 at the corrected position (fresh records/gradient).
        ev = await this.evaluateNewtonState(false, 1, dt);
      }
    }
    let E0 = ev.status.energy;
    let gtdx0 = 0;
    void gtdx0;
    let converged = false;
    let acceptedFinal = true;
    let pcgItersTotal = 0;
    // G6C.2 GPU Newton control (explicit flag; default path below untouched).
    if (driver.cfg.useGpuNewtonControl) {
      const r = await this.stepNewtonGpuControlled(
        ev.status, newtonIters, submitsPerNewton, statusReadbacksPerNewton,
      );
      converged = r.converged;
      acceptedFinal = r.acceptedFinal;
      newtonItersUsed = r.newtonItersUsed;
      pcgSolves = r.pcgSolves;
      pcgItersTotal = r.pcgItersTotal;
      E0 = r.E0;
    } else {
    for (let k = 0; k < newtonIters; k++) {
      const st = k === 0 ? ev.status : (await this.evaluateXkAfterAccept()).status;
      void st;
      // Current gradient norm (from the latest diagnostics).
      const gnorm = k === 0 ? ev.status.gradNorm : this.lastDeviceStatus!.gradNorm;
      if (!Number.isFinite(gnorm)) throw new Error("device Newton: non-finite gradient");
      if (gnorm < cfg.gradTol) { converged = true; break; }
      newtonItersUsed++;
      const submitsNewton0 = ex.ledger.submits;
      const syncsNewton0 = this.hotLoopReadbacks;
      // PCG solve (fully GPU-side, one submit; 2 tiny syncs after).
      const pcg = await driver.pcgSolve();
      pcgSolves++;
      pcgItersTotal += cfg.pcgIters;
      this.lastPcg = pcg;
      // G6A: topology-degradation fallback surfaces here (one record/solve).
      if (driver.lastFallback) this.fallbackLog.push(driver.lastFallback);
      // Descent check: gtdx = g . dx (tiny sync). Fallback: Jacobi-scaled
      // steepest descent on device (mirror of newton.ts safeguard).
      let gtdx = await this.dotScalar("rhs", "searchDirection");
      if (!Number.isFinite(gtdx) || gtdx >= 0 || pcg.breakdown) {
        // G6A: method changed for this iterate — record requested/actual/reason.
        this.fallbackLog.push({
          requested: driver.lastActualMethod,
          actual: "jacobi-descent",
          reason: pcg.breakdown ? "pcg-breakdown" : "non-descent-direction",
        });
        ex.beginBatch("descent-fallback");
        ex.writeBlas(driver.n3, 1);
        ex.runPass({
          shader: "blas", entry: "neg_div",
          groups: [[
            { binding: 0, buffer: "uniformBank", offset: 0, size: 16 },
            { binding: 1, buffer: "rhs" },
            { binding: 2, buffer: "diag" },
            { binding: 3, buffer: "searchDirection" },
          ]],
          x: Math.max(1, Math.ceil(driver.n3 / 64)),
        });
        await ex.submitBatch(false);
        gtdx = await this.dotScalar("rhs", "searchDirection");
      }
      this.lastDirDot = gtdx;
      // Trust region: max|dx| (tiny sync), fold scale into trial alpha.
      ex.beginBatch("trust-abs");
      ex.writeBlas(driver.n3, 1);
      ex.runPass({
        shader: "blas", entry: "absv",
        groups: [[
          { binding: 0, buffer: "uniformBank", offset: 0, size: 16 },
          { binding: 1, buffer: "searchDirection" },
          { binding: 3, buffer: "pcgProd" },
        ]],
        x: Math.max(1, Math.ceil(driver.n3 / 64)),
      });
      await ex.submitBatch(false);
      const maxDx = await driver.reduceMax("pcgProd", driver.n3);
      const trustScale = maxDx > cfg.trust ? cfg.trust / maxDx : 1;
      // Armijo line search: fresh GPU contact rebuild per alpha (new segment).
      // G6B batched path evaluates K alphas per batch with one status sync;
      // the sequential path below is unchanged (default) for parity baseline.
      let alpha = 1;
      let accepted = false;
      if (driver.cfg.useBatchedArmijo) {
        const res = await this.armijoBatchedSearch({
          E0, gtdx, trustScale, dMin, newtonIndex: k,
          pcgBreakdown: pcg.breakdown, dt,
        });
        alpha = res.alpha;
        accepted = res.accepted;
        if (accepted && res.status) {
          E0 = res.status.energy;
          this.lastDeviceStatus = {
            ...res.status,
            energy: res.status.energy,
            directionDotGradient: gtdx,
          };
        }
      } else {
        for (let li = 0; li < cfg.armijoIters; li++) {
          await driver.applyTrial(alpha * trustScale);
          const tev = await this.evaluateNewtonState(true, k * 100 + li + 1, dt);
          const s = tev.status;
          const valid =
            s.finite === 1 && s.ccdSafe === 1 &&
            s.minDistance > dMin &&
            !(s.minToi > 0 && s.minToi < 1 - 1e-9);
          if (!valid || !Number.isFinite(s.energy)) {
            alpha *= 0.5;
            continue;
          }
          if (s.energy <= E0 + 1e-4 * alpha * trustScale * gtdx) {
            await driver.acceptTrial();
            E0 = s.energy;
            this.lastDeviceStatus = { ...s, directionDotGradient: gtdx };
            this.lastMarker = { mask: tev.markerMask, count: tev.markerCount };
            accepted = true;
            break;
          }
          alpha *= 0.5;
        }
      }
      submitsPerNewton.push(ex.ledger.submits - submitsNewton0);
      statusReadbacksPerNewton.push(this.hotLoopReadbacks - syncsNewton0);
      if (!accepted) { acceptedFinal = false; break; }
      if (alpha * trustScale < 1e-6) break;
    }
    } // end sequential-Newton else branch (GPU-control path above)
    // Velocities on device (+ restitution-0 filter), then per-step snapshot.
    const damp = 1 - Math.min(Math.max(scene.material.damping, 0), 0.1);
    await driver.finishVelocities(1 / dt, damp);
    ex.beginBatch("velocity-filter");
    ex.runPass({
      shader: "contact-force", entry: "velocity_filter",
      groups: [[
        { binding: 0, buffer: "simParams" },
        { binding: 1, buffer: "contactW" },
        { binding: 2, buffer: "contactN" },
        { binding: 3, buffer: "contactId" },
        { binding: 10, buffer: "velocity" },
      ]],
      x: Math.max(1, Math.ceil(this.driver!.c.cap / 64)),
    });
    await ex.submitBatch(false);
    const t1 = performance.now();
    this.encodeMs = t1 - t0;
    this.deviceSteps++;
    if (this.snapshotOnStep) await this.snapshotDeviceState();
    const st = this.lastDeviceStatus ?? ev.status;
    // CPU-owned status lanes (device leaves them zeroed by construction).
    const final: SolverStatus = {
      ...st,
      directionDotGradient: this.lastDirDot,
      pcgBreakdown: this.lastPcg.breakdown ? 1 : 0,
      converged: converged ? 1 : 0,
    };
    this.lastDeviceStatus = final;
    // G6A per-step production report (fixed-state vs end-to-end split).
    {
      const fb = this.fallbackLog.length > 0 ? this.fallbackLog[this.fallbackLog.length - 1] : null;
      const coarseActive = driver.cfg.useCoarsePcg || driver.cfg.useCoarseC0;
      this.lastStepReport = {
        requestedPreconditioner: requestedMode,
        // No solve ran (immediate convergence): mark explicitly so a stale
        // driver method name can never read as "ran Jacobi".
        actualPreconditioner: pcgSolves > 0 ? driver.lastActualMethod : `${requestedMode}:no-solve`,
        fallback: fb,
        fallbackReason: fb ? fb.reason : null,
        submits: ex.ledger.submits - submits0,
        syncs: this.hotLoopReadbacks - syncs0,
        newtonIters: newtonItersUsed,
        finePcgIters: pcgItersTotal,
        coarsePcgIters: coarseActive ? pcgSolves * driver.cfg.coarseIters : 0,
        residual: this.lastPcg.resNorm,
        energy: final.energy,
        armijoTrials: this.armijoTrials,
        armijoBatches: this.armijoBatches,
        armijoSelects: this.armijoSelects,
        armijoCommits: this.armijoCommits,
        armijoReadbacks: this.armijoReadbacks,
        submitsPerNewton,
        statusReadbacksPerNewton,
        controlPath: driver.cfg.useGpuNewtonControl
          ? "gpu-newton"
          : driver.cfg.useBatchedArmijo
            ? "batched-armijo"
            : "sequential",
      };
    }
    return {
      ...final,
      deviceReady: true,
      timestampQuery: ex.facts.timestampQuery,
      encodeMs: this.encodeMs, submitMs: 0, readbackMs: 0,
      statusBytes: 64,
    };
  }

  /**
   * Recompute RHS/Jacobi + diagnostics at the accepted position (next Newton
   * iterate reuses the accepted trial's records/gradient; only g/diag need a
   * refresh against the copied position buffer, whose content equals the
   * accepted xTrial bit-identically).
   */
  private async evaluateXkAfterAccept(): Promise<{ status: SolverStatus }> {
    const driver = this.driver!;
    const ex = this.executor!;
    ex.beginBatch("eval-xk");
    driver.rhsJacobi(this.jacobiBeta());
    await ex.submitBatch(false);
    // Gradient/records already valid from the accepted trial; refresh the
    // status image for the convergence check (position == accepted trial).
    const { status, markerMask, markerCount } = await driver.diagnosticsAt("position");
    void markerMask; void markerCount;
    const snap: SolverStatus = { ...status, directionDotGradient: this.lastDirDot };
    this.lastDeviceStatus = snap;
    return { status: snap };
  }

  /** Per-step snapshot: positions + velocities to CPU (labeled, NOT hot-loop). */
  private async snapshotDeviceState(): Promise<void> {
    const ex = this.executor!;
    const scene = this.scene!;
    const n = scene.mesh.count;
    const posRaw = await ex.readBufferDebug("position", "snapshot-positions", false);
    const pos = new Float32Array(posRaw);
    for (let i = 0; i < n; i++) {
      scene.positions[i * 3] = pos[i * 4];
      scene.positions[i * 3 + 1] = pos[i * 4 + 1];
      scene.positions[i * 3 + 2] = pos[i * 4 + 2];
    }
    const velRaw = await ex.readBufferDebug("velocity", "snapshot-velocities", false);
    const vel = new Float32Array(velRaw);
    for (let i = 0; i < n; i++) {
      scene.velocities[i * 3] = vel[i * 4];
      scene.velocities[i * 3 + 1] = vel[i * 4 + 1];
      scene.velocities[i * 3 + 2] = vel[i * 4 + 2];
    }
    enforcePins(scene, scene.positions);
  }

  /** Compact status readback (allowed hot-loop transfer). */
  async readbackDiagnostics(): Promise<GpuDiagnostics> {
    if (this.deviceMode === "device" && this.lastDeviceStatus) {
      return {
        ...this.lastDeviceStatus,
        deviceReady: true,
        timestampQuery: this.executor?.facts.timestampQuery ?? false,
        encodeMs: this.encodeMs, submitMs: this.submitMs, readbackMs: this.readbackMs,
        statusBytes: 64,
      };
    }
    if (this.handle?.ready) {
      const statusBuf = this.buffers.get("solverStatus");
      if (statusBuf) {
        // SYNC POINT (labeled): status-only copy + map; 64 B.
        const t0 = performance.now();
        const { data, timing } = await readbackBuffer(this.handle.device, statusBuf, 64, "solver-status");
        const t1 = performance.now();
        void t1;
        this.readbackMs += timing.mapMs;
        this.hotLoopReadbacks += 1;
        this.lastStatus = decodeSolverStatus(data);
      }
    }
    const st = this.lastStatus ?? this.synthesizeStatus();
    return {
      ...st, deviceReady: this.handle?.ready ?? false,
      timestampQuery: this.handle?.timestampQueryEnabled ?? false,
      encodeMs: this.encodeMs, submitMs: this.submitMs, readbackMs: this.readbackMs,
      statusBytes: 64,
    };
  }

  /** Explicit debug readback (NOT hot-loop): full positions via staging copy. */
  async readbackPositions(): Promise<Float64Array> {
    if (!this.scene) throw new Error("WebGpuSolver not initialized");
    if (this.deviceMode === "device") {
      // Device truth is snapshotted to scene arrays every step; no extra sync.
      return Float64Array.from(this.scene.positions);
    }
    if (!this.handle?.ready) return Float64Array.from(this.getPositions());
    // DEBUG SYNC POINT (labeled, off hot path): GPU -> COPY_SRC -> staging -> CPU.
    const posBuf = this.buffers.get("position");
    const n = this.scene.mesh.count;
    const { data } = await readbackBuffer(this.handle.device, posBuf, n * 16, "positions-debug");
    return unpackVec4Positions(new Float32Array(data), n);
  }

  /** G1 broad-phase: GPU-resident AABBs, zero position readback. */
  getBroadphase(pad = 0.002, pairCapacity = 16384): BroadPhase {
    if (!this.scene) throw new Error("WebGpuSolver not initialized");
    if (!this.broadphase) {
      this.broadphase = new GpuBroadPhase({
        indices: this.scene.mesh.indices,
        triCount: this.scene.mesh.triCount,
        pad,
        pairCapacity,
      });
    }
    return this.broadphase;
  }

  /**
   * G1 -> G2 handoff helper: build GPU candidate pairs for a Newton segment.
   * Positions stay GPU-resident on device; on headless CI the exact mirror
   * runs against the provided (already-available) arrays — never a mapping.
   */
  async buildBroadphasePairs(
    x0: Float64Array, x1: Float64Array,
    pad = 0.002, pairCapacity = 16384,
  ): Promise<CandidatePairs> {
    if (!this.scene) throw new Error("WebGpuSolver not initialized");
    const bp = new GpuBroadPhase({
      indices: this.scene.mesh.indices,
      triCount: this.scene.mesh.triCount,
      pad,
      pairCapacity,
    });
    return bp.build(x0, x1);
  }

  /** G1 active: LBVH broad-phase with CPU fallback until set-parity gates pass. */
  gpuBroadphaseStatus(): { stage: string; fallback: string } {
    return { stage: "G1-active", fallback: "CPU BVH (ContactSystem) remains golden until set-parity gates pass" };
  }

  /** G2 CCD + compaction: GPU-resident contacts, zero position readback. */
  getContactSystem(opts: {
    dHatM?: number; dMinM?: number; kappaJ?: number;
    frictionMu?: number; frictionEpsM?: number;
    contactCapacity?: number; floorY?: number | null;
  } = {}): GpuContactSystem {
    if (!this.scene) throw new Error("WebGpuSolver not initialized");
    if (!this.contacts) {
      const cs = this.scene.contact;
      this.contacts = new GpuContactSystem({
        indices: this.scene.mesh.indices,
        triCount: this.scene.mesh.triCount,
        dHatM: opts.dHatM ?? cs?.params.dHatM ?? 0.002,
        dMinM: opts.dMinM ?? cs?.params.dMinM ?? 1e-4,
        kappaJ: opts.kappaJ ?? cs?.params.kappaJ ?? 50,
        frictionMu: opts.frictionMu ?? cs?.params.frictionMu ?? 0.3,
        frictionEpsM: opts.frictionEpsM ?? cs?.params.frictionEpsM ?? 1e-4,
        contactCapacity: opts.contactCapacity ?? 4096,
        floorY: opts.floorY ?? cs?.floorY ?? null,
        staticPos: cs?.staticPos ?? null,
        staticIdx: cs?.staticIdx ?? null,
      });
    }
    return this.contacts;
  }

  /**
   * G2 contact build for ONE Newton segment x0 -> x1 with G1 pairs.
   * Called fresh for EVERY Armijo trial alpha (each alpha is a new segment).
   * Positions stay GPU-resident on device; the headless mirror runs against
   * the provided (already-available) arrays — never a mapping, so
   * forbiddenReadbacks is untouched.
   */
  async buildContacts(
    x0: Float64Array, x1: Float64Array,
    pairs: CandidatePairs,
    staticPairs?: Array<[number, number]>,
  ): Promise<GpuContactSet> {
    const gcs = this.getContactSystem();
    if (gcs.xStep.length !== x0.length) gcs.beginStep(x0);
    return gcs.build(x1, pairs, staticPairs);
  }

  /** Last G2 compact diagnostics (counters only — no state readback). */
  contactDiagnostics(): GpuContactDiagnostics | null {
    return this.contacts?.lastSet?.diagnostics ?? null;
  }

  gpuCcdStatus(): { stage: string; fallback: string } {
    return { stage: "G2-active", fallback: "CPU VT/EE CCD remains golden until TOI/set-parity gates pass" };
  }

  isWebGPUAvailable(): boolean {
    return isWebGPUAvailable();
  }
}
