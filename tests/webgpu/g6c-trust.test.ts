// G6C trust scaling: the GPU path folds trust on device (trust_compute +
// apply-time multiply) while the CPU golden path (solver/newton.ts) scales dx
// in place. The two are MATHEMATICALLY equivalent but NOT bit-identical:
// CPU computes x + alpha*(s*dx) in f64, GPU computes x + ((alpha*s))*dx in
// f32 — multiplication order and precision both differ by construction.
// Exact CPU bit-equivalence is therefore NOT required for this path (the
// device is f32 throughout; Armijo decides by inequality with margin; the
// trajectory gates already budget 1e-4/1e-5 for this). What IS required:
// the GPU scale equals trust/maxDx within f32 rounding, and the folded apply
// matches the CPU formula within f32 rounding. Both directions tested here.
import { describe, it, expect, beforeAll } from "vitest";
import { GpuUniformSlot } from "../../src/backend/webgpu/gpu-buffers.js";
import type { DeviceFixture } from "./device-setup.js";
import { sharedDevice, stripScene } from "./device-setup.js";

let strip: DeviceFixture | null = null;

beforeAll(async () => {
  strip = await sharedDevice("g6c-trust-strip", stripScene, { contactCapacity: 64, pairCapacity: 512 });
}, 180000);

/** CPU golden formula (mirror of solver/newton.ts). */
function cpuTrustScale(maxDx: number, trust: number): number {
  return maxDx > trust ? trust / maxDx : 1;
}

describe("G6C trust scale: formula parity (pure)", () => {
  it("matches the CPU ternary incl. below/at/above trust and NaN", () => {
    const trust = 0.002;
    expect(cpuTrustScale(0.001, trust)).toBe(1);
    expect(cpuTrustScale(0.002, trust)).toBe(1); // strict > : at-trust unscaled
    expect(cpuTrustScale(0.008, trust)).toBeCloseTo(0.25, 15);
    // NaN-safe exactly like the WGSL `if (maxDx > trTrust)`: NaN > x is
    // false on both sides, so a NaN max leaves the scale at 1.0.
    expect(cpuTrustScale(NaN, trust)).toBe(1);
  });

  it("folded apply matches CPU scaled-dx within f32 rounding (not bitwise)", () => {
    const trust = 0.002;
    // Nasty mantissas: any dropped factor (or unfolded trust) fails hugely.
    const dx = [0.1 + 0.123456789, -0.3 - 0.987654321, 0.007777777777];
    let maxDx = 0;
    for (const v of dx) maxDx = Math.max(maxDx, Math.abs(v));
    const s = cpuTrustScale(maxDx, trust);
    expect(s).toBeLessThan(1);
    const rawAlpha = 0.125; // 0.5^3, exact in binary on both sides
    // CPU golden (f64): scale dx first, then alpha*dxScaled.
    const cpu = dx.map((v) => rawAlpha * (s * v));
    // GPU path (f32): fold scalars first, then aEff*dx.
    const f = Math.fround;
    const aEff = f(f(rawAlpha) * f(s));
    const gpu = dx.map((v) => f(f(aEff) * f(v)));
    let worstRel = 0;
    for (let i = 0; i < dx.length; i++) {
      worstRel = Math.max(worstRel, Math.abs(gpu[i] - cpu[i]) / Math.max(Math.abs(cpu[i]), 1e-30));
    }
    // eslint-disable-next-line no-console
    console.log(`[g6c-trust] folded-apply worst rel diff = ${worstRel.toExponential(3)} (scale=${s})`);
    expect(worstRel).toBeLessThan(1e-6);
    // Sanity: an unfolded (trust-dropped) apply would miss by ~300x here.
    const unfolded = dx.map((v) => f(f(rawAlpha) * f(v)));
    expect(Math.abs(unfolded[0] - cpu[0]) / Math.abs(cpu[0])).toBeGreaterThan(1);
  });
});

describe("G6C trust_compute kernel (device)", () => {
  it("device trust scale equals trust/maxDx with no CPU readback in the path", async () => {
    if (!strip) return;
    const { ex, driver } = strip;
    const trust = 0.002;
    const n3 = driver.n3;
    // Fixed dx with known max (index 1 dominates; below-trust lane included).
    const dx = new Float32Array(n3);
    for (let i = 0; i < n3; i++) dx[i] = 1e-4 * ((i * 37) % 11 - 5);
    dx[1] = -0.0078125;
    let maxCpu = 0;
    for (let i = 0; i < n3; i++) maxCpu = Math.max(maxCpu, Math.abs(dx[i]));
    expect(maxCpu).toBeCloseTo(0.0078125, 12);
    ex.writeBuffer("searchDirection", dx);
    // Same three passes the Newton round runs: absv -> reduce_max -> trust.
    ex.beginBatch("trust-probe-abs");
    ex.writeBlas(n3, 1);
    ex.runPass({
      shader: "blas", entry: "absv",
      groups: [[
        { binding: 0, buffer: "uniformBank", offset: 0, size: 16 },
        { binding: 1, buffer: "searchDirection" },
        { binding: 3, buffer: "pcgProd" },
      ]],
      x: Math.max(1, Math.ceil(n3 / 64)),
    });
    await ex.submitBatch(false);
    const maxDevice = await driver.reduceMax("pcgProd", n3);
    expect(maxDevice).toBeCloseTo(maxCpu, 6);
    // reduceScratch[0] still holds the max: run the kernel under test.
    driver.bankF(GpuUniformSlot.NewtonTrust, trust);
    ex.beginBatch("trust-probe-compute");
    ex.runPass({
      shader: "newton-control", entry: "trust_compute",
      groups: [[
        { binding: 20, buffer: "reduceScratch" },
        { binding: 21, buffer: "trustScaleStore" },
        { binding: 22, buffer: "newtonStatus" },
        { binding: 23, buffer: "uniformBank", offset: GpuUniformSlot.NewtonTrust * 256, size: 4 },
      ]],
      x: 1,
    });
    await ex.submitBatch(false);
    const scale = new Float32Array(await ex.readBufferDebug("trustScaleStore", "trust-scale", false))[0];
    const stepNorm = new Float32Array(await ex.readBufferDebug("newtonStatus", "trust-status", false))[10];
    const expected = trust / maxCpu;
    // eslint-disable-next-line no-console
    console.log(`[g6c-trust] device scale=${scale.toExponential(6)} expected=${expected.toExponential(6)} stepNorm=${stepNorm}`);
    expect(scale).toBeCloseTo(expected, 6);
    expect(stepNorm).toBeCloseTo(maxCpu, 6);
  }, 300000);
});
