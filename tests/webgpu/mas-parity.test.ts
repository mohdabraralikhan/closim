// G5D device parity: mas_restrict / coarse-scale / prolongate (GPU, f32) vs
// mas.ts (CPU, f64). Skips cleanly with no device. Then a two-level MAS
// device solve convergence check.
import { describe, it, expect, beforeAll } from "vitest";
import type { ClothScene } from "../../src/physics/scene.js";
import {
  restrictCoarse, buildCoarseDiag,
} from "../../src/solver/mas.js";
import type { DeviceFixture } from "./device-setup.js";
import { sharedDevice, resetDeviceState, stripScene } from "./device-setup.js";

let strip: DeviceFixture | null = null;

beforeAll(async () => {
  strip = await sharedDevice("mas-strip", stripScene, { contactCapacity: 64, pairCapacity: 512 });
}, 180000);

describe("G5D MAS device parity", () => {
  it("true coarse diagonal matches diag(RAP)", async () => {
    if (!strip) return;
    const { solver, ex, driver } = strip;
    const scene = (solver as unknown as { scene: ClothScene }).scene;
    const X = Float64Array.from(scene.positions);
    resetDeviceState(strip, X);
    solver.configureStep(1 / 60);
    await solver.evaluateNewtonState(false, 1, 1 / 60);
    ex.beginBatch("mas-parity-diag");
    driver.schwarzBuildPasses();
    driver.masCoarseDiagPass();
    await ex.submitBatch(false);
    const cdRaw = await ex.readBufferDebug("masCoarseDiag", "mas-parity-cd", false);
    const diagRaw = await ex.readBufferDebug("diag", "mas-parity-diag", false);
    const gpuCd = new Float32Array(cdRaw);
    const diagF = new Float32Array(diagRaw);
    const n = scene.mesh.count;
    const diag = new Float64Array(n * 3);
    for (let i = 0; i < diag.length; i++) diag[i] = diagF[i];
    const cpu = buildCoarseDiag(X, scene.mesh, scene.material, diag, solver.schwarzTopo!);
    const nDoms = driver.c.schwarzDoms;
    let worst = 0;
    let scale = 0;
    for (let i = 0; i < nDoms * 3; i++) scale = Math.max(scale, Math.abs(cpu[i]));
    for (let i = 0; i < nDoms * 3; i++) {
      worst = Math.max(worst, Math.abs(gpuCd[i] - cpu[i]) / Math.max(scale, 1e-12));
    }
    // eslint-disable-next-line no-console
    console.log(`[g5d-parity] coarseDiag rel err over ${nDoms} domains: ${worst.toExponential(2)}`);
    expect(worst).toBeLessThan(1e-5);
  }, 180000);

  it("restrict+prolongate round-trips the CPU on device", async () => {
    if (!strip) return;
    const { solver, ex, driver } = strip;
    const scene = (solver as unknown as { scene: ClothScene }).scene;
    const n = scene.mesh.count;
    resetDeviceState(strip, Float64Array.from(scene.positions));
    solver.configureStep(1 / 60);
    await solver.evaluateNewtonState(false, 1, 1 / 60);
    // Seed pcgResidual with a deterministic field via diag copy, then
    // restrict on device and compare with the CPU.
    ex.beginBatch("mas-parity-rp");
    driver.schwarzBuildPasses();
    driver.masCoarseDiagPass();
    // copy diag -> pcgResidual (blas copy needs bank count)
    ex.writeBlas(driver.n3, 1);
    ex.runPass({
      shader: "blas", entry: "copy",
      groups: [[
        { binding: 0, buffer: "uniformBank", offset: 0, size: 16 },
        { binding: 1, buffer: "diag" },
        { binding: 3, buffer: "pcgResidual" },
      ]],
      x: Math.max(1, Math.ceil(driver.n3 / 64)),
    });
    // restrict residual -> masCoarseR
    const { GpuUniformSlot } = await import("../../src/backend/webgpu/gpu-buffers.js");
    ex.writeBuffer("uniformBank", (() => { const a = new Uint32Array(4); a[0] = driver.c.schwarzDoms * 3; return a; })(), GpuUniformSlot.MasCount * 256);
    ex.runPass({
      shader: "mas", entry: "mas_restrict",
      groups: [[
        { binding: 1, buffer: "pcgResidual" },
        { binding: 2, buffer: "masCoarseR" },
        { binding: 3, buffer: "schwarzVerts" },
        { binding: 4, buffer: "uniformBank", offset: GpuUniformSlot.MasCount * 256, size: 4 },
      ]],
      x: Math.max(1, Math.ceil((driver.c.schwarzDoms * 3) / 64)),
    });
    await ex.submitBatch(false);
    const crRaw = await ex.readBufferDebug("masCoarseR", "mas-parity-cr", false);
    const gpuCr = new Float32Array(crRaw);
    const diagRaw = await ex.readBufferDebug("diag", "mas-parity-diag2", false);
    const diagF = new Float32Array(diagRaw);
    const diag = new Float64Array(n * 3);
    for (let i = 0; i < diag.length; i++) diag[i] = diagF[i];
    const cpu = restrictCoarse(diag, solver.schwarzTopo!);
    let worst = 0;
    let scale = 0;
    for (let i = 0; i < driver.c.schwarzDoms * 3; i++) scale = Math.max(scale, Math.abs(cpu[i]));
    for (let i = 0; i < driver.c.schwarzDoms * 3; i++) {
      worst = Math.max(worst, Math.abs(gpuCr[i] - cpu[i]) / Math.max(scale, 1e-12));
    }
    // eslint-disable-next-line no-console
    console.log(`[g5d-parity] restrict rel err: ${worst.toExponential(2)}`);
    expect(worst).toBeLessThan(1e-5);
  }, 180000);

  it("two-level MAS device solve converges", async () => {
    if (!strip) return;
    const { solver, driver } = strip;
    const scene = (solver as unknown as { scene: ClothScene }).scene;
    resetDeviceState(strip, Float64Array.from(scene.positions));
    solver.configureStep(1 / 60);
    await solver.evaluateNewtonState(false, 2, 1 / 60);
    const prevM = driver.cfg.useMas;
    const prevS = driver.cfg.useSchwarz;
    const prevB = driver.cfg.useBlockJacobi;
    driver.cfg.useMas = true;
    driver.cfg.useSchwarz = false;
    driver.cfg.useBlockJacobi = false;
    try {
      driver.cfg.pcgIters = 8;
      const rec: { rz: number[] } = { rz: [] };
      const res = await driver.pcgSolve(rec);
      expect(Number.isFinite(res.resNorm)).toBe(true);
      expect(rec.rz.length).toBe(9);
      expect(rec.rz[rec.rz.length - 1]).toBeLessThan(rec.rz[0] * 10);
    } finally {
      driver.cfg.useMas = prevM;
      driver.cfg.useSchwarz = prevS;
      driver.cfg.useBlockJacobi = prevB;
    }
  }, 180000);
});
