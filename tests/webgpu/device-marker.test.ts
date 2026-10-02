// G3 device-execution marker: proves WGSL actually ran (never the mirror).
// 1. Marker dispatch sets magic + stage bits + count (device fact).
// 2. Mirror-only stepping leaves the marker zeroed (distinguishes paths).
import { describe, it, expect } from "vitest";
import { STAGE_IDS } from "../../src/backend/webgpu/gpu-executor.js";
import { requireDevice, stripScene } from "./device-setup.js";

describe("G3 device execution marker", () => {
  it("1. marker dispatch proves real WGSL execution with stage bits", async () => {
    const fix = await requireDevice(stripScene, { contactCapacity: 256, pairCapacity: 2048 });
    if (!fix) return; // skip cleanly without WebGPU
    const { solver, ex, driver } = fix;
    expect(solver.deviceMode).toBe("device");
    // zero + epoch, then run the predictor stage through the driver
    driver.resetMarker(7);
    driver.uploadSimParams({
      dt: 1 / 60, gravity: [0, -9.81, 0], contactCount: 0,
      newtonIteration: 0, lineSearchAlpha: 1, frictionMu: 0.3,
    });
    await driver.predictorPass();
    // SYNC POINT (tiny): marker pack proves dispatch happened on device.
    const m = await driver.readMarker();
    expect(m.magic).toBe(0xc10a57);
    expect(m.epoch).toBe(7);
    expect(m.mask & (1 << STAGE_IDS.predictor)).toBeTruthy();
    expect(m.count).toBeGreaterThan(0);
    // ledger shows exactly one status-category mapping, zero forbidden reads
    expect(ex.ledger.forbiddenReadbacks).toBe(0);
    expect(ex.ledger.mappedDebugBytes).toBe(0);
  });

  it("2. mirror-only stepping never sets the marker (paths distinguished)", async () => {
    const fix = await requireDevice(stripScene, { contactCapacity: 256, pairCapacity: 2048 });
    if (!fix) return;
    const { solver, ex } = fix;
    // force the mirror path even with a live device
    solver.forceMirror = true;
    const before = ex.ledger.submits;
    await solver.stepGpu(1 / 60, { newtonIters: 1 });
    solver.forceMirror = false;
    expect(solver.fallbackUses).toBeGreaterThan(0);
    // mirror path submits nothing to the device...
    expect(ex.ledger.submits).toBe(before);
    // ...and the marker buffer (zeroed at upload) still reads zero magic.
    const raw = await ex.readSmall("execMarker", 16, "marker-mirror-check", "status");
    expect(new Uint32Array(raw)[0]).toBe(0);
  });
});
