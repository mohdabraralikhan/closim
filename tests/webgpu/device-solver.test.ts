// G3 device solver tests: one-step, strip-pull, gravity, floor, fold,
// friction parity (device vs CPU), plus readback/marker/status/PCG/fallback
// gates. Shared fixtures per scene (pipeline compile amortized). Skips
// without WebGPU. First heavy test per fixture pays Dawn compile (~seconds).
import { describe, it, expect, beforeAll } from "vitest";
import { buildGrid, preprocess } from "../../src/mesh/mesh.js";
import { createScene, pinColumn } from "../../src/physics/scene.js";
import { DEFAULT_MATERIAL } from "../../src/physics/types.js";
import { CpuSolver } from "../../src/backend/cpu-solver.js";
import { ContactSystem } from "../../src/collision/contact-assembly.js";
import { DEFAULT_CONTACT_PARAMS } from "../../src/collision/types.js";
import type { DeviceFixture } from "./device-setup.js";
import { sharedDevice } from "./device-setup.js";
import { offsetMesh, preprocessMerged } from "../helpers.js";

const P = { ...DEFAULT_CONTACT_PARAMS };

function stripBuild() {
  const g = buildGrid(6, 3, 0.2, 0.1);
  const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
  const scene = createScene(mesh, { ...DEFAULT_MATERIAL });
  pinColumn(scene, (x) => x < 1e-9);
  return scene;
}

function floorBuild() {
  const g = buildGrid(4, 4, 0.08, 0.08);
  for (let i = 0; i < g.positions.length / 3; i++) g.positions[i * 3 + 1] += 0.01;
  const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
  const scene = createScene(mesh, { ...DEFAULT_MATERIAL }, [0, -9.81, 0]);
  const contact = new ContactSystem({ ...P }, mesh.indices);
  contact.setFloor(0);
  scene.contact = contact;
  return scene;
}

function headOnBuild() {
  const A = offsetMesh(3, 3, 0.06, 0.06, -0.05, 0.02, 0);
  const B = offsetMesh(3, 3, 0.06, 0.06, 0.05, 0.02, 0);
  const mesh = preprocessMerged([A, B], 0.15);
  const scene = createScene(mesh, { ...DEFAULT_MATERIAL }, [0, 0, 0]);
  const nA = A.positions.length / 3;
  for (let i = 0; i < mesh.count; i++) {
    scene.velocities[i * 3] = i < nA ? 0.1 : -0.1;
  }
  const contact = new ContactSystem({ ...P, frictionMu: 0.6 }, mesh.indices);
  scene.contact = contact;
  return scene;
}

function freefallBuild() {
  // Unpinned patch under gravity, no contact: pure predictor + FEM parity.
  const g = buildGrid(3, 3, 0.06, 0.06);
  const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
  return createScene(mesh, { ...DEFAULT_MATERIAL }, [0, -9.81, 0]);
}

function foldBuild() {
  // Page-fold pose (right half mirrored over the left, layers 1.5 mm apart)
  // with self-contact: bending + self-contact relaxation on device vs CPU.
  // Rest shape stays flat (preprocess first), the FOLD is the start state.
  const g = offsetMesh(8, 2, 0.2, 0.03, 0, 0.02, 0);
  const mesh = preprocessMerged([g], 0.15);
  const scene = createScene(mesh, { ...DEFAULT_MATERIAL }, [0, -9.81, 0]);
  for (let i = 0; i < mesh.count; i++) {
    const x = scene.positions[i * 3];
    if (x > 0.1) {
      scene.positions[i * 3] = 0.1 - 2 * (x - 0.1);
      scene.positions[i * 3 + 1] += 0.0015;
    }
  }
  const contact = new ContactSystem({ ...P }, mesh.indices);
  scene.contact = contact;
  return scene;
}

let strip: DeviceFixture | null = null;
let floor: DeviceFixture | null = null;
let headon: DeviceFixture | null = null;
let freefall: DeviceFixture | null = null;
let fold: DeviceFixture | null = null;

beforeAll(async () => {
  strip = await sharedDevice("solver-strip", stripBuild, { contactCapacity: 64, pairCapacity: 512 });
  floor = await sharedDevice("solver-floor", floorBuild, { contactCapacity: 1024, pairCapacity: 2048 });
  headon = await sharedDevice("solver-headon", headOnBuild, { contactCapacity: 1024, pairCapacity: 2048 });
  freefall = await sharedDevice("solver-freefall", freefallBuild, { contactCapacity: 64, pairCapacity: 512 });
  fold = await sharedDevice("solver-fold", foldBuild, { contactCapacity: 1024, pairCapacity: 2048 });
  // lean validation budgets (parity, not production convergence)
  for (const f of [strip, floor, headon, freefall, fold]) {
    if (f?.solver.driver) f.solver.driver.cfg.pcgIters = 20;
  }
}, 240000);

function cpuRef(build: () => ReturnType<typeof stripBuild>): { solver: CpuSolver; scene: ReturnType<typeof stripBuild> } {
  const scene = build();
  const solver = new CpuSolver();
  solver.initialize(scene);
  return { solver, scene };
}

function maxAbsPosErr(a: Float64Array, b: Float64Array): number {
  let m = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

describe("G3 device solver parity", () => {
  it("7. one-step solver parity (device Newton vs CPU Newton)", async () => {
    if (!strip) return;
    const { solver } = strip;
    const ref = cpuRef(stripBuild);
    await solver.stepGpu(1 / 60, { newtonIters: 2 });
    ref.solver.step(1 / 60);
    const err = maxAbsPosErr(solver.getPositions(), ref.solver.getPositions());
    expect(Number.isFinite(err)).toBe(true);
    expect(err).toBeLessThan(2e-3);
  }, 180000);

  it("7b. one-step solver parity with analytic HVP (G4A production path)", async () => {
    if (!strip) return;
    const { solver } = strip;
    const scene = (solver as unknown as { scene: { positions: Float64Array; velocities: Float64Array; mesh: { count: number } } }).scene;
    const ref = cpuRef(stripBuild);
    // sync to rest so both start identically (prior tests advanced the fixture)
    const n = scene.mesh.count;
    const ex = solver.executor!;
    const pack = (X: Float64Array): Float32Array => {
      const out = new Float32Array(n * 4);
      for (let i = 0; i < n; i++) {
        out[i * 4] = X[i * 3]; out[i * 4 + 1] = X[i * 3 + 1]; out[i * 4 + 2] = X[i * 3 + 2];
      }
      return out;
    };
    ex.writeBuffer("position", pack(ref.scene.positions));
    ex.writeBuffer("position0", pack(ref.scene.positions));
    ex.writeBuffer("velocity", pack(ref.scene.velocities));
    const prev = solver.driver!.cfg.useAnalyticHvp;
    solver.driver!.cfg.useAnalyticHvp = true;
    try {
      await solver.stepGpu(1 / 60, { newtonIters: 2 });
    } finally {
      solver.driver!.cfg.useAnalyticHvp = prev;
    }
    ref.solver.step(1 / 60);
    const err = maxAbsPosErr(solver.getPositions(), ref.solver.getPositions());
    expect(Number.isFinite(err)).toBe(true);
    expect(err).toBeLessThan(2e-3);
  }, 180000);

  it("7c. one-step solver parity with gather assembly (G4B production path)", async () => {
    if (!strip) return;
    const { solver } = strip;
    const scene = (solver as unknown as { scene: { positions: Float64Array; velocities: Float64Array; mesh: { count: number } } }).scene;
    const ref = cpuRef(stripBuild);
    const n = scene.mesh.count;
    const ex = solver.executor!;
    const pack = (X: Float64Array): Float32Array => {
      const out = new Float32Array(n * 4);
      for (let i = 0; i < n; i++) {
        out[i * 4] = X[i * 3]; out[i * 4 + 1] = X[i * 3 + 1]; out[i * 4 + 2] = X[i * 3 + 2];
      }
      return out;
    };
    ex.writeBuffer("position", pack(ref.scene.positions));
    ex.writeBuffer("position0", pack(ref.scene.positions));
    ex.writeBuffer("velocity", pack(ref.scene.velocities));
    const prev = solver.driver!.cfg.useGatherAssembly;
    solver.driver!.cfg.useGatherAssembly = true;
    try {
      await solver.stepGpu(1 / 60, { newtonIters: 2 });
    } finally {
      solver.driver!.cfg.useGatherAssembly = prev;
    }
    ref.solver.step(1 / 60);
    const err = maxAbsPosErr(solver.getPositions(), ref.solver.getPositions());
    expect(Number.isFinite(err)).toBe(true);
    expect(err).toBeLessThan(2e-3);
  }, 180000);

  it("7d. submit-count regression (G4C batching lock)", async () => {
    if (!strip) return;
    const { solver, ex } = strip;
    // steady-state step: uniform coalescing + mark masks must hold submits
    // near the G4-measured ~205/step (pre-G4C baseline: 402). Cap has headroom
    // for path variation but fails on batching regressions (e.g. a rewrite
    // that re-introduces per-pass uniform flushes or per-stage mark writes).
    const s0 = ex.ledger.submits;
    const f0 = ex.ledger.forbiddenReadbacks;
    const c0 = ex.ledger.coalescedWrites;
    const r0 = solver.hotLoopReadbacks;
    await solver.stepGpu(1 / 60, { newtonIters: 2 });
    expect(ex.ledger.submits - s0).toBeLessThanOrEqual(240);
    expect(ex.ledger.forbiddenReadbacks).toBe(f0);
    expect(ex.ledger.coalescedWrites - c0).toBeGreaterThan(0);
    expect(solver.hotLoopReadbacks - r0).toBeLessThanOrEqual(6);
  }, 180000);

  it("8. strip-pull parity over 3 steps", async () => {
    if (!strip) return;
    const { solver } = strip;
    // reset device state to rest (prior tests advanced it)
    const scene = (solver as unknown as { scene: { positions: Float64Array; velocities: Float64Array; mesh: { count: number } } }).scene;
    const ref = cpuRef(stripBuild);
    // re-upload rest state so both start identically
    const ex = solver.executor!;
    const n = scene.mesh.count;
    const pack = (X: Float64Array): Float32Array => {
      const out = new Float32Array(n * 4);
      for (let i = 0; i < n; i++) {
        out[i * 4] = X[i * 3]; out[i * 4 + 1] = X[i * 3 + 1]; out[i * 4 + 2] = X[i * 3 + 2];
      }
      return out;
    };
    ex.writeBuffer("position", pack(ref.scene.positions));
    ex.writeBuffer("position0", pack(ref.scene.positions));
    ex.writeBuffer("velocity", pack(ref.scene.velocities));
    for (let s = 0; s < 3; s++) {
      await solver.stepGpu(1 / 60, { newtonIters: 2 });
      ref.solver.step(1 / 60);
    }
    const err = maxAbsPosErr(solver.getPositions(), ref.solver.getPositions());
    // Documented FP32-inexact-Newton regime (measured on GTX 1050):
    // single-step fresh error ~1e-4, steady tracking neighborhood ~1e-3
    // (soft bending modes + per-solve Armijo path choices, not a formulation
    // gap — kernel-level parity holds to 1e-4 and energies track).
    expect(err).toBeLessThan(3e-3);
    // sagging happens on device too (free end below start)
    const pos = solver.getPositions();
    let minY = Infinity;
    for (let i = 0; i < pos.length / 3; i++) minY = Math.min(minY, pos[i * 3 + 1]);
    expect(minY).toBeLessThan(-1e-4);
  }, 240000);

  it("9. gravity-only free fall matches the CPU (unpinned patch)", async () => {
    if (!freefall) return;
    const { solver } = freefall;
    const ref = cpuRef(freefallBuild);
    for (let s = 0; s < 2; s++) {
      await solver.stepGpu(1 / 60, { newtonIters: 2 });
      ref.solver.step(1 / 60);
    }
    const pos = solver.getPositions();
    const err = maxAbsPosErr(pos, ref.solver.getPositions());
    expect(err).toBeLessThan(1e-3);
    // ballistic sanity: 2 steps of free fall drop ~5.4 mm
    let minY = Infinity;
    for (let i = 0; i < pos.length / 3; i++) minY = Math.min(minY, pos[i * 3 + 1]);
    expect(minY).toBeLessThan(-3e-3);
  }, 180000);

  it("10. floor settling without penetration (device vs CPU)", async () => {
    if (!floor) return;
    const { solver } = floor;
    const ref = cpuRef(floorBuild);
    // Matched generous budgets: contact impact needs real Newton work on both
    // sides (device ni=2/pcg=20 under-solves a gravity-driven impact while
    // the CPU reference runs 10/60 by default — compare converged regimes).
    solver.driver!.cfg.pcgIters = 60;
    for (let s = 0; s < 3; s++) {
      await solver.stepGpu(1 / 60, { newtonIters: 10 });
      ref.solver.step(1 / 60);
    }
    solver.driver!.cfg.pcgIters = 20;
    const pos = solver.getPositions();
    let minY = Infinity;
    for (let i = 0; i < pos.length / 3; i++) minY = Math.min(minY, pos[i * 3 + 1]);
    expect(minY).toBeGreaterThan(-5e-4); // hard core respected on device
    const err = maxAbsPosErr(pos, ref.solver.getPositions());
    expect(err).toBeLessThan(5e-3);
  }, 240000);

  it("11. head-on patches without penetration (device vs CPU)", async () => {
    if (!headon) return;
    const { solver } = headon;
    const ref = cpuRef(headOnBuild);
    // sync device to the CPU reference start (shared fixture may be advanced)
    const scene = (solver as unknown as { scene: { positions: Float64Array; velocities: Float64Array; mesh: { count: number } } }).scene;
    const n = scene.mesh.count;
    const ex = solver.executor!;
    const pack = (X: Float64Array): Float32Array => {
      const out = new Float32Array(n * 4);
      for (let i = 0; i < n; i++) {
        out[i * 4] = X[i * 3]; out[i * 4 + 1] = X[i * 3 + 1]; out[i * 4 + 2] = X[i * 3 + 2];
      }
      return out;
    };
    ex.writeBuffer("position", pack(ref.scene.positions));
    ex.writeBuffer("position0", pack(ref.scene.positions));
    ex.writeBuffer("velocity", pack(ref.scene.velocities));
    for (let s = 0; s < 2; s++) {
      await solver.stepGpu(1 / 60, { newtonIters: 2 });
      ref.solver.step(1 / 60);
    }
    const err = maxAbsPosErr(solver.getPositions(), ref.solver.getPositions());
    expect(err).toBeLessThan(8e-3);
  }, 240000);

  it("11b. page-fold self-contact (device vs CPU, one step)", async () => {
    if (!fold) return;
    const { solver } = fold;
    const ref = cpuRef(foldBuild);
    // Matched generous budgets (device ni=10/pcg=60 = CPU defaults).
    // ONE step only: the fold is Lyapunov-chaotic — a 1e-7 perturbation of
    // the CPU reference itself grows to 5.7e-3 after 2 steps (5.7e4
    // amplification via stick-slip + bending snap-through), and device
    // atomic-scheduling jitter adds run-to-run rounding variation at the
    // same scale (observed 3.1e-3..5.4e-3 across identical one-step runs).
    // Multi-step tracking is not a parity criterion here; invariants are.
    solver.driver!.cfg.pcgIters = 60;
    await solver.stepGpu(1 / 60, { newtonIters: 10 });
    ref.solver.step(1 / 60);
    solver.driver!.cfg.pcgIters = 20;
    const err = maxAbsPosErr(solver.getPositions(), ref.solver.getPositions());
    expect(err).toBeLessThan(1e-2);
    // self-contact genuinely active on device (not a contact-free pass),
    // energies finite, layers separated (no penetration).
    const d = await solver.readbackDiagnostics();
    expect(d.finite).toBe(1);
    expect(Number.isFinite(d.energy)).toBe(true);
    expect(d.minDistance).toBeGreaterThan(0);
  }, 240000);

  it("12. friction parity with dissipation (device vs CPU)", async () => {
    if (!headon) return;
    const { solver } = headon;
    const ref = cpuRef(headOnBuild);
    // sync device to the CPU reference start (shared fixture may be advanced)
    const scene = (solver as unknown as { scene: { positions: Float64Array; velocities: Float64Array; mesh: { count: number } } }).scene;
    const n = scene.mesh.count;
    const ex = solver.executor!;
    const pack = (X: Float64Array): Float32Array => {
      const out = new Float32Array(n * 4);
      for (let i = 0; i < n; i++) {
        out[i * 4] = X[i * 3]; out[i * 4 + 1] = X[i * 3 + 1]; out[i * 4 + 2] = X[i * 3 + 2];
      }
      return out;
    };
    ex.writeBuffer("position", pack(ref.scene.positions));
    ex.writeBuffer("position0", pack(ref.scene.positions));
    ex.writeBuffer("velocity", pack(ref.scene.velocities));
    solver.configureStep(1 / 60);
    const E0 = (await solver.evaluateNewtonState(false, 900, 1 / 60)).status.energy;
    for (let s = 0; s < 3; s++) {
      await solver.stepGpu(1 / 60, { newtonIters: 2 });
      ref.solver.step(1 / 60);
    }
    const err = maxAbsPosErr(solver.getPositions(), ref.solver.getPositions());
    expect(err).toBeLessThan(8e-3);
    // friction + zero gravity: no energy creation across the steps
    const E1 = (await solver.readbackDiagnostics()).energy;
    expect(E1).toBeLessThanOrEqual(E0 * (1 + 5e-3));
  }, 240000);

  it("13. zero forbidden readbacks during device stepping", async () => {
    if (!strip) return;
    const { solver, ex } = strip;
    solver.snapshotOnStep = false;
    const dbg0 = ex.ledger.mappedDebugBytes;
    const forb0 = ex.ledger.forbiddenReadbacks;
    await solver.stepGpu(1 / 60, { newtonIters: 1 });
    solver.snapshotOnStep = true;
    expect(ex.ledger.forbiddenReadbacks).toBe(forb0);
    expect(ex.ledger.mappedDebugBytes).toBe(dbg0);
    expect(ex.ledger.mappedStatusBytes).toBeGreaterThan(0);
    expect(solver.hotLoopReadbacks).toBeGreaterThan(0);
  }, 180000);

  it("14. device status is valid (finite, magic, stage bits)", async () => {
    if (!strip) return;
    const { solver } = strip;
    const d = await solver.stepGpu(1 / 60, { newtonIters: 1 });
    expect(d.finite).toBe(1);
    expect(Number.isFinite(d.energy)).toBe(true);
    expect(Number.isFinite(d.gradNorm)).toBe(true);
    expect(solver.lastMarker.mask).not.toBe(0);
    expect(solver.lastMarker.count).toBeGreaterThan(0);
    // trial-eval stage bits must have executed (broadphase -> diagnostics).
    // (predictor belongs to the step-start eval, whose mask was superseded
    // by the fresher trial-eval mask — by design, masks are per-eval.)
    const { STAGE_IDS } = await import("../../src/backend/webgpu/gpu-executor.js");
    expect(solver.lastMarker.mask & (1 << STAGE_IDS.traverse)).toBeTruthy();
    expect(solver.lastMarker.mask & (1 << STAGE_IDS.membrane)).toBeTruthy();
    expect(solver.lastMarker.mask & (1 << STAGE_IDS.diagnostics)).toBeTruthy();
  }, 180000);

  it("15. GPU PCG converges on a real Newton system", async () => {
    if (!strip) return;
    const { solver, ex, driver } = strip;
    // set up a real evaluation, then solve and check residual improvement
    solver.configureStep(1 / 60);
    await solver.evaluateNewtonState(false, 1, 1 / 60);
    ex.beginBatch("pcg-test-norm");
    driver.mulInto("negRhs", "negRhs");
    await ex.submitBatch(false);
    const bNorm = Math.sqrt(Math.max(0, await driver.reduceSum("pcgProd", driver.n3)));
    const res = await driver.pcgSolve();
    expect(Number.isFinite(res.resNorm)).toBe(true);
    expect(Number.isFinite(bNorm)).toBe(true);
    if (!res.breakdown) {
      expect(res.resNorm).toBeLessThan(bNorm);
    }
    // breakdown path still leaves finite state (fallback takes over in steps)
    expect(typeof res.breakdown).toBe("boolean");
  }, 180000);

  it("16. explicit CPU-fallback accounting (forceMirror path)", async () => {
    if (!strip) return;
    const { solver, ex } = strip;
    const submits0 = ex.ledger.submits;
    const fb0 = solver.fallbackUses;
    solver.forceMirror = true;
    await solver.stepGpu(1 / 60, { newtonIters: 1 });
    solver.forceMirror = false;
    expect(solver.fallbackUses).toBe(fb0 + 1);
    // mirror path submits nothing to the device
    expect(ex.ledger.submits).toBe(submits0);
    expect(solver.deviceMode).toBe("device"); // device retained, mirror used once
  }, 120000);
});
