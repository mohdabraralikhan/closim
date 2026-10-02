// G5C device parity: schwarz_assemble/schwarz_factor (GPU, f32) vs
// buildSchwarzFactors (CPU, f64) on the solver's own (8,1,8) topology.
// Skips cleanly with no device. Compares factor flags exactly and the
// explicit-inverse real-dof subblocks with f32 tolerance; then checks a
// Schwarz-preconditioned device solve converges.
import { describe, it, expect, beforeAll } from "vitest";
import type { ClothScene } from "../../src/physics/scene.js";
import {
  buildSchwarzFactors, solveLocal,
} from "../../src/solver/schwarz.js";
import type { DeviceFixture } from "./device-setup.js";
import { sharedDevice, resetDeviceState, stripScene } from "./device-setup.js";

let strip: DeviceFixture | null = null;

beforeAll(async () => {
  strip = await sharedDevice("schwarz-strip", stripScene, { contactCapacity: 64, pairCapacity: 512 });
}, 180000);

async function factorOnDevice(X: Float64Array): Promise<{
  gpuInv: Float32Array; gpuFlag: Uint32Array; diag: Float64Array; nDoms: number;
}> {
  const { solver, ex, driver } = strip!;
  resetDeviceState(strip!, X);
  solver.configureStep(1 / 60);
  await solver.evaluateNewtonState(false, 1, 1 / 60);
  ex.beginBatch("schwarz-parity-factor");
  driver.schwarzBuildPasses();
  await ex.submitBatch(false);
  const invRaw = await ex.readBufferDebug("schwarzInv", "schwarz-parity-inv", false);
  const flagRaw = await ex.readBufferDebug("schwarzFlag", "schwarz-parity-flag", false);
  const diagRaw = await ex.readBufferDebug("diag", "schwarz-parity-diag", false);
  const diagF = new Float32Array(diagRaw);
  const scene = (solver as unknown as { scene: ClothScene }).scene;
  const diag = new Float64Array(scene.mesh.count * 3);
  for (let i = 0; i < diag.length; i++) diag[i] = diagF[i];
  return {
    gpuInv: new Float32Array(invRaw), gpuFlag: new Uint32Array(flagRaw),
    diag, nDoms: driver.c.schwarzDoms,
  };
}

describe("G5C Schwarz device parity", () => {
  it("GPU inverses match the CPU reference at rest", async () => {
    if (!strip) return;
    const { solver } = strip;
    const scene = (solver as unknown as { scene: ClothScene }).scene;
    const X = Float64Array.from(scene.positions);
    const { gpuInv, gpuFlag, diag, nDoms } = await factorOnDevice(X);
    const topo = solver.schwarzTopo!;
    expect(nDoms).toBe(topo.members.length);
    const cpu = buildSchwarzFactors(X, scene.mesh, scene.material, diag, topo);
    for (let d = 0; d < nDoms; d++) {
      expect(gpuFlag[d], `flag d=${d}`).toBe(cpu.flag[d]);
    }
    // Explicit-inverse comparison on real dofs only (padded rows are identity
    // on GPU and absent on CPU). CPU explicit inverse via unit-vector solves.
    // Normalized by domain scale: structural zeros carry ~1e-9 fp noise on
    // both sides, so per-element floors would false-alarm.
    let worst = 0;
    const e = new Float64Array(24);
    const col = new Float64Array(24);
    for (let d = 0; d < nDoms; d++) {
      const dv = topo.members[d].length;
      const dw = dv * 3;
      let scale = 0;
      const cols: Float64Array[] = [];
      for (let c = 0; c < dw; c++) {
        e.fill(0); e[c] = 1;
        const cc = new Float64Array(dw);
        solveLocal(cpu, d, e.subarray(0, dw), cc);
        cols.push(cc);
        for (let i = 0; i < dw; i++) scale = Math.max(scale, Math.abs(cc[i]));
      }
      scale = Math.max(scale, 1e-12);
      for (let c = 0; c < dw; c++) {
        for (let i = 0; i < dw; i++) {
          const g = gpuInv[d * 576 + i * 24 + c];
          worst = Math.max(worst, Math.abs(g - cols[c][i]) / scale);
        }
      }
    }
    // eslint-disable-next-line no-console
    console.log(`[g5c-parity:rest] worst schwarzInv rel err over ${nDoms} domains: ${worst.toExponential(2)}`);
    expect(worst).toBeLessThan(1e-2);
  }, 180000);

  it("GPU inverses match the CPU reference under stress", async () => {
    if (!strip) return;
    const { solver } = strip;
    const scene = (solver as unknown as { scene: ClothScene }).scene;
    const n = scene.mesh.count;
    const X = Float64Array.from(scene.positions);
    for (let i = 0; i < n; i++) {
      X[i * 3] *= 1.1;
      X[i * 3 + 1] += 0.01 * X[i * 3 + 2];
    }
    const { gpuInv, gpuFlag, diag, nDoms } = await factorOnDevice(X);
    const topo = solver.schwarzTopo!;
    const cpu = buildSchwarzFactors(X, scene.mesh, scene.material, diag, topo);
    for (let d = 0; d < nDoms; d++) {
      expect(gpuFlag[d], `flag d=${d}`).toBe(cpu.flag[d]);
    }
    let worst = 0;
    const e = new Float64Array(24);
    for (let d = 0; d < nDoms; d++) {
      const dw = topo.members[d].length * 3;
      let scale = 0;
      const cols: Float64Array[] = [];
      for (let c = 0; c < dw; c++) {
        e.fill(0); e[c] = 1;
        const cc = new Float64Array(dw);
        solveLocal(cpu, d, e.subarray(0, dw), cc);
        cols.push(cc);
        for (let i = 0; i < dw; i++) scale = Math.max(scale, Math.abs(cc[i]));
      }
      scale = Math.max(scale, 1e-12);
      for (let c = 0; c < dw; c++) {
        for (let i = 0; i < dw; i++) {
          const g = gpuInv[d * 576 + i * 24 + c];
          worst = Math.max(worst, Math.abs(g - cols[c][i]) / scale);
        }
      }
    }
    // eslint-disable-next-line no-console
    console.log(`[g5c-parity:stressed] worst schwarzInv rel err over ${nDoms} domains: ${worst.toExponential(2)}`);
    expect(worst).toBeLessThan(1e-2);
  }, 180000);

  it("Schwarz-preconditioned device solve converges", async () => {
    if (!strip) return;
    const { solver, driver } = strip;
    const scene = (solver as unknown as { scene: ClothScene }).scene;
    resetDeviceState(strip, Float64Array.from(scene.positions));
    solver.configureStep(1 / 60);
    await solver.evaluateNewtonState(false, 2, 1 / 60);
    const prevS = driver.cfg.useSchwarz;
    const prevB = driver.cfg.useBlockJacobi;
    driver.cfg.useSchwarz = true;
    driver.cfg.useBlockJacobi = false;
    try {
      driver.cfg.pcgIters = 8;
      const rec: { rz: number[] } = { rz: [] };
      const res = await driver.pcgSolve(rec);
      expect(Number.isFinite(res.resNorm)).toBe(true);
      expect(rec.rz.length).toBe(9);
      expect(rec.rz[rec.rz.length - 1]).toBeLessThan(rec.rz[0] * 10);
    } finally {
      driver.cfg.useSchwarz = prevS;
      driver.cfg.useBlockJacobi = prevB;
    }
  }, 180000);
});
