// G5.5 coarse block-CSR device parity: pattern, raw block values (rest +
// stressed), SpMV, and on-device bilinear symmetry u^T Ac v ~= v^T Ac u.
// Skips cleanly with no device.
import { describe, it, expect, beforeAll } from "vitest";
import type { ClothScene } from "../../src/physics/scene.js";
import {
  assembleCoarseRaw, coarseSpmv, bilinearAsymmetry,
} from "../../src/solver/coarse-csr.js";
import type { DeviceFixture } from "./device-setup.js";
import { sharedDevice, resetDeviceState, stripScene } from "./device-setup.js";

let strip: DeviceFixture | null = null;

beforeAll(async () => {
  strip = await sharedDevice("coarse-strip", stripScene, { contactCapacity: 64, pairCapacity: 512 });
}, 180000);

async function assembleOnDevice(X: Float64Array): Promise<{
  gpuBlocks: Float32Array; diag: Float64Array; coarseDofs: number;
}> {
  const { solver, ex, driver } = strip!;
  resetDeviceState(strip!, X);
  solver.configureStep(1 / 60);
  await solver.evaluateNewtonState(false, 1, 1 / 60);
  ex.beginBatch("coarse-parity-assemble");
  driver.coarseAssemblePasses();
  await ex.submitBatch(false);
  const pat = solver.coarsePattern!;
  const blkRaw = await ex.readBufferDebug("coarseBlockValues", "coarse-parity-vals", false);
  const diagRaw = await ex.readBufferDebug("diag", "coarse-parity-diag", false);
  const scene = (solver as unknown as { scene: ClothScene }).scene;
  const diag = new Float64Array(scene.mesh.count * 3);
  const diagF = new Float32Array(diagRaw);
  for (let i = 0; i < diag.length; i++) diag[i] = diagF[i];
  void pat;
  return { gpuBlocks: new Float32Array(blkRaw), diag, coarseDofs: driver.c.schwarzDoms * 3 };
}

function checkBlocks(
  label: string, gpuBlocks: Float32Array, cpuRaw: Float64Array, nnz: number,
): void {
  let worst = 0;
  let scale = 0;
  for (let i = 0; i < nnz * 9; i++) scale = Math.max(scale, Math.abs(cpuRaw[i]));
  scale = Math.max(scale, 1e-12);
  for (let i = 0; i < nnz * 9; i++) {
    worst = Math.max(worst, Math.abs(gpuBlocks[i] - cpuRaw[i]) / scale);
  }
  // eslint-disable-next-line no-console
  console.log(`[g5.5-parity:${label}] block-value rel err over ${nnz} blocks: ${worst.toExponential(2)}`);
  expect(worst).toBeLessThan(1e-3);
}

describe("G5.5 coarse block-CSR device parity", () => {
  it("pattern matches the CPU reference exactly", async () => {
    if (!strip) return;
    const { solver, ex } = strip;
    const pat = solver.coarsePattern!;
    const roRaw = await ex.readBufferDebug("coarseRowOffsets", "coarse-parity-rows", false);
    const ciRaw = await ex.readBufferDebug("coarseColIndices", "coarse-parity-cols", false);
    const brRaw = await ex.readBufferDebug("coarseBlockRows", "coarse-parity-brows", false);
    expect([...new Uint32Array(roRaw)]).toEqual([...pat.rowOffsets]);
    expect([...new Uint32Array(ciRaw)]).toEqual([...pat.colIndices]);
    const br = new Uint32Array(brRaw);
    for (let d = 0; d < pat.domains; d++) {
      for (let p = pat.rowOffsets[d]; p < pat.rowOffsets[d + 1]; p++) {
        expect(br[p]).toBe(d);
      }
    }
  }, 180000);

  it("block values match at rest", async () => {
    if (!strip) return;
    const { solver } = strip;
    const scene = (solver as unknown as { scene: ClothScene }).scene;
    const X = Float64Array.from(scene.positions);
    const { gpuBlocks, diag } = await assembleOnDevice(X);
    const pat = solver.coarsePattern!;
    const cpuRaw = assembleCoarseRaw(X, scene.mesh, scene.material, diag, solver.schwarzTopo!, pat);
    checkBlocks("rest", gpuBlocks, cpuRaw, pat.nnz);
  }, 180000);

  it("block values match under stress", async () => {
    if (!strip) return;
    const { solver } = strip;
    const scene = (solver as unknown as { scene: ClothScene }).scene;
    const n = scene.mesh.count;
    const X = Float64Array.from(scene.positions);
    for (let i = 0; i < n; i++) {
      X[i * 3] *= 1.1;
      X[i * 3 + 1] += 0.01 * X[i * 3 + 2];
    }
    const { gpuBlocks, diag } = await assembleOnDevice(X);
    const pat = solver.coarsePattern!;
    const cpuRaw = assembleCoarseRaw(X, scene.mesh, scene.material, diag, solver.schwarzTopo!, pat);
    checkBlocks("stressed", gpuBlocks, cpuRaw, pat.nnz);
  }, 180000);

  it("device SpMV matches the CPU operator", async () => {
    if (!strip) return;
    const { solver, ex, driver } = strip;
    const scene = (solver as unknown as { scene: ClothScene }).scene;
    const X = Float64Array.from(scene.positions);
    const { diag, coarseDofs } = await assembleOnDevice(X);
    const pat = solver.coarsePattern!;
    const cpuRaw = assembleCoarseRaw(X, scene.mesh, scene.material, diag, solver.schwarzTopo!, pat);
    // deterministic probe vector
    const v = new Float32Array(coarseDofs);
    for (let i = 0; i < coarseDofs; i++) v[i] = Math.cos(i * 1.31) * 2;
    ex.writeBuffer("coarseX", v);
    ex.beginBatch("coarse-parity-spmv");
    driver.coarseSpmvPass("coarseX", "coarseAp");
    await ex.submitBatch(false);
    const yRaw = await ex.readBufferDebug("coarseAp", "coarse-parity-y", false);
    const yGpu = new Float32Array(yRaw);
    // CPU reference: symmetrized operator applied (matches GPU read-time avg).
    const { assembleCoarseValues } = await import("../../src/solver/coarse-csr.js");
    const csr = assembleCoarseValues(X, scene.mesh, scene.material, diag, solver.schwarzTopo!, pat);
    const yRef = new Float64Array(coarseDofs);
    coarseSpmv(csr, v, yRef);
    void cpuRaw;
    let worst = 0;
    let scale = 0;
    for (let i = 0; i < coarseDofs; i++) scale = Math.max(scale, Math.abs(yRef[i]));
    scale = Math.max(scale, 1e-12);
    for (let i = 0; i < coarseDofs; i++) {
      worst = Math.max(worst, Math.abs(yGpu[i] - yRef[i]) / scale);
    }
    // eslint-disable-next-line no-console
    console.log(`[g5.5-parity:spmv] rel err over ${coarseDofs} dofs: ${worst.toExponential(2)}`);
    expect(worst).toBeLessThan(1e-4);
  }, 180000);

  it("on-device bilinear symmetry u^TAcv ~= v^TAcu", async () => {
    if (!strip) return;
    const { solver, ex, driver } = strip;
    const scene = (solver as unknown as { scene: ClothScene }).scene;
    const X = Float64Array.from(scene.positions);
    const { diag, coarseDofs } = await assembleOnDevice(X);
    // NOTE: device vectors are f32 — a previous revision used Float64Array
    // here, silently uploading 2x bytes into 48 B buffers (zeros on readback).
    const u = new Float32Array(coarseDofs);
    const v = new Float32Array(coarseDofs);
    let s = 0x5EED;
    const rnd = (): number => {
      s = (Math.imul(s, 1103515245) + 12345) >>> 0;
      return s / 4294967296;
    };
    for (let i = 0; i < coarseDofs; i++) {
      u[i] = rnd() * 2 - 1;
      v[i] = rnd() * 2 - 1;
    }
    ex.writeBuffer("coarseX", u);
    ex.writeBuffer("coarseP", v);
    ex.beginBatch("coarse-parity-bilin");
    driver.coarseSpmvPass("coarseX", "coarseAp"); // Au
    driver.coarseSpmvPass("coarseP", "coarseZ"); // Av
    await ex.submitBatch(false);
    // uv = u . Av via blas mul + device-side reduce, scalar readback only.
    ex.beginBatch("coarse-parity-dot");
    ex.writeBlas(coarseDofs, 1);
    ex.runPass({
      shader: "blas", entry: "mul",
      groups: [[
        { binding: 0, buffer: "uniformBank", offset: 0, size: 16 },
        { binding: 1, buffer: "coarseX" },
        { binding: 2, buffer: "coarseZ" },
        { binding: 3, buffer: "coarseProd" },
      ]],
      x: Math.max(1, Math.ceil(coarseDofs / 64)),
    });
    await ex.submitBatch(false);
    const uv = await driver.reduceSum("coarseProd", coarseDofs);
    ex.beginBatch("coarse-parity-dot2");
    ex.writeBlas(coarseDofs, 1);
    ex.runPass({
      shader: "blas", entry: "mul",
      groups: [[
        { binding: 0, buffer: "uniformBank", offset: 0, size: 16 },
        { binding: 1, buffer: "coarseP" },
        { binding: 2, buffer: "coarseAp" },
        { binding: 3, buffer: "coarseProd" },
      ]],
      x: Math.max(1, Math.ceil(coarseDofs / 64)),
    });
    await ex.submitBatch(false);
    const vu = await driver.reduceSum("coarseProd", coarseDofs);
    const scale = Math.max(Math.abs(uv), Math.abs(vu), 1e-300);
    // eslint-disable-next-line no-console
    console.log(`[g5.5-parity:bilin] uv=${uv.toExponential(4)} vu=${vu.toExponential(4)} rel=${(Math.abs(uv - vu) / scale).toExponential(2)}`);
    expect(Math.abs(uv - vu) / scale).toBeLessThan(1e-4);
    // cross-check against the CPU bilinear probe
    const pat = solver.coarsePattern!;
    const { assembleCoarseValues } = await import("../../src/solver/coarse-csr.js");
    const csr = assembleCoarseValues(X, scene.mesh, scene.material, diag, solver.schwarzTopo!, pat);
    const cpu = bilinearAsymmetry(csr, u, v);
    expect(Math.abs(cpu.diff) / Math.max(Math.abs(cpu.uv), Math.abs(cpu.vu), 1e-300)).toBeLessThan(1e-12);
  }, 180000);
});
