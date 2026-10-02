// G5A instrumentation tests (fast, strip-scale): profiled-PCG equivalence
// and timestamp-capture mechanism self-check.
import { describe, it, expect, beforeAll } from "vitest";
import type { ClothScene } from "../../src/physics/scene.js";
import type { DeviceFixture } from "./device-setup.js";
import { sharedDevice, resetDeviceState, stripScene } from "./device-setup.js";

let strip: DeviceFixture | null = null;

beforeAll(async () => {
  strip = await sharedDevice("prof-strip", stripScene, { contactCapacity: 64, pairCapacity: 512 });
}, 180000);

describe("G5A PCG instrumentation", () => {
  it("profiled solve matches the unprofiled solve", async () => {
    if (!strip) return;
    const { solver, driver } = strip;
    const scene = (solver as unknown as { scene: ClothScene }).scene;
    resetDeviceState(strip, Float64Array.from(scene.positions));
    driver.cfg.pcgIters = 8;
    solver.configureStep(1 / 60);
    await solver.evaluateNewtonState(false, 1, 1 / 60);
    const rec: { rz: number[] } = { rz: [] };
    const prof = await driver.pcgSolve(rec);
    const plain = await driver.pcgSolve();
    expect(rec.rz.length).toBe(9); // rz0 + K iters
    for (const v of rec.rz) expect(Number.isFinite(v)).toBe(true);
    expect(rec.rz[0]).toBeGreaterThan(0);
    expect(prof.breakdown).toBe(plain.breakdown);
    expect(Math.abs(prof.resNorm - plain.resNorm) / Math.max(plain.resNorm, 1e-12))
      .toBeLessThan(1e-4);
  }, 180000);

  it("timestamp capture opens, labels passes, resolves monotonically", async () => {
    if (!strip) return;
    const { solver, ex, driver } = strip;
    if (!ex.facts.timestampQuery) return; // unsupported path: covered by fallback
    const scene = (solver as unknown as { scene: ClothScene }).scene;
    resetDeviceState(strip, Float64Array.from(scene.positions));
    expect(ex.beginTimestampCapture(64)).toBe(true);
    driver.predictorPass();
    await ex.submitBatch(true);
    const cap = await ex.endTimestampCapture();
    expect(cap).not.toBeNull();
    expect(cap!.labels.length).toBeGreaterThan(0);
    expect(cap!.labels[0]).toContain("predictor");
    const ticks = cap!.ticks as BigUint64Array;
    expect(ticks.length).toBe(cap!.labels.length * 2);
    for (let i = 1; i < ticks.length; i++) {
      expect(ticks[i] >= ticks[i - 1]).toBe(true);
    }
    // NOTE (backend finding, GTX 1050/Dawn): all queries within one submit
    // latch to the same value; only inter-submit deltas track wall (~1ns).
    // Pass-granularity GPU/overhead splits are NOT asserted here — the
    // benchmark uses wall-clock fallback as primary on this backend.
  }, 180000);
});
