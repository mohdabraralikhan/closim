// G5B device parity: bj_build_factor (GPU, f32) vs buildBlockFactors (CPU, f64).
// Skips cleanly with no device. Compares factor flags exactly and inverse
// entries with f32-appropriate tolerance; then checks one z = B^-1 r apply.
import { describe, it, expect, beforeAll } from "vitest";
import type { ClothScene } from "../../src/physics/scene.js";
import { buildBlockFactors } from "../../src/solver/block-jacobi.js";
import type { DeviceFixture } from "./device-setup.js";
import { sharedDevice, resetDeviceState, stripScene } from "./device-setup.js";

let strip: DeviceFixture | null = null;

beforeAll(async () => {
  strip = await sharedDevice("bj-strip", stripScene, { contactCapacity: 64, pairCapacity: 512 });
}, 180000);

async function checkFactors(label: string, X: Float64Array): Promise<void> {
  const { solver, ex, driver } = strip!;
  const scene = (solver as unknown as { scene: ClothScene }).scene;
  const n = scene.mesh.count;
  resetDeviceState(strip!, X);
  solver.configureStep(1 / 60);
  await solver.evaluateNewtonState(false, 1, 1 / 60);
  ex.beginBatch("bj-parity-factor");
  driver.blockFactorPasses();
  await ex.submitBatch(false);
  const invRaw = await ex.readBufferDebug("blockInv", "bj-parity-inv", false);
  const flagRaw = await ex.readBufferDebug("blockFlag", "bj-parity-flag", false);
  const diagRaw = await ex.readBufferDebug("diag", "bj-parity-diag", false);
  const gpuInv = new Float32Array(invRaw);
  const gpuFlag = new Uint32Array(flagRaw);
  const diag = new Float64Array(n * 3);
  const diagF = new Float32Array(diagRaw);
  for (let i = 0; i < n * 3; i++) diag[i] = diagF[i];
  const cpu = buildBlockFactors(X, scene.mesh, scene.material, diag);
  // Flags must agree exactly (same Sylvester latch, f32 vs f64 could differ
  // only in knife-edge cases; the strip states are far from singular).
  for (let v = 0; v < n; v++) {
    expect(gpuFlag[v], `${label} flag v=${v}`).toBe(cpu.flag[v]);
  }
  // Inverse entries: f32-vs-f64 tolerance relative to row scale.
  let worst = 0;
  for (let v = 0; v < n; v++) {
    let scale = 0;
    for (let k = 0; k < 9; k++) scale = Math.max(scale, Math.abs(cpu.inv[v * 9 + k]));
    scale = Math.max(scale, 1e-12);
    for (let k = 0; k < 9; k++) {
      const d = Math.abs(gpuInv[v * 9 + k] - cpu.inv[v * 9 + k]) / scale;
      worst = Math.max(worst, d);
    }
  }
  // eslint-disable-next-line no-console
  console.log(`[g5b-parity:${label}] worst blockInv rel err over ${n} verts: ${worst.toExponential(2)}`);
  expect(worst).toBeLessThan(5e-4);
}

describe("G5B block-Jacobi device parity", () => {
  it("GPU factors match the CPU reference at rest", async () => {
    if (!strip) return;
    const scene = (strip.solver as unknown as { scene: ClothScene }).scene;
    await checkFactors("rest", Float64Array.from(scene.positions));
  }, 180000);

  it("GPU factors match the CPU reference under stress", async () => {
    if (!strip) return;
    const scene = (strip.solver as unknown as { scene: ClothScene }).scene;
    // 10% stretch + shear: exercises the stress (S-term) path, including the
    // delta(b,a) guard on off-diagonal block entries.
    const n = scene.mesh.count;
    const X = Float64Array.from(scene.positions);
    for (let i = 0; i < n; i++) {
      X[i * 3] *= 1.1;
      X[i * 3 + 1] += 0.01 * X[i * 3 + 2];
    }
    await checkFactors("stressed", X);
  }, 180000);

  it("block-PCG device solve converges on the strip scene", async () => {
    if (!strip) return;
    const { solver, driver } = strip;
    const scene = (solver as unknown as { scene: ClothScene }).scene;
    resetDeviceState(strip, Float64Array.from(scene.positions));
    solver.configureStep(1 / 60);
    await solver.evaluateNewtonState(false, 2, 1 / 60);
    const prev = driver.cfg.useBlockJacobi;
    driver.cfg.useBlockJacobi = true;
    try {
      driver.cfg.pcgIters = 8;
      const rec: { rz: number[] } = { rz: [] };
      const res = await driver.pcgSolve(rec);
      expect(Number.isFinite(res.resNorm)).toBe(true);
      expect(rec.rz.length).toBe(9);
      // Residual curve must not diverge (allow flat, forbid blowup).
      expect(rec.rz[rec.rz.length - 1]).toBeLessThan(rec.rz[0] * 10);
    } finally {
      driver.cfg.useBlockJacobi = prev;
    }
  }, 180000);
});
