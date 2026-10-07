// G6C global selected-trial-index safety under non-uniform batch widths.
//
// The global index must be trialsBefore + localIndex (cumulative base), NOT
// batchIndex * batchWidth (only equal for uniform widths). Both sides carry
// the fix: the CPU helper globalTrialIndex and the device commit_arm's
// newtonCtl[7] cumulative base. This file tests the pure helper plus the
// commit_arm kernel directly (seeded batch statuses across reject/accept
// batches with INTERIOR width changes — the shape adaptive-K produces), and
// closes the loop with an end-to-end alpha<->index consistency check on a
// real newtonRound (selectedAlpha = 0.5^globalIdx * trustScale, so a wrong
// index disagrees with its own alpha).
import { describe, it, expect, beforeAll } from "vitest";
import { GpuUniformSlot } from "../../src/backend/webgpu/gpu-buffers.js";
import { globalTrialIndex } from "../../src/backend/webgpu/gpu-newton.js";
import type { DeviceFixture } from "./device-setup.js";
import { sharedDevice, stripScene } from "./device-setup.js";

let strip: DeviceFixture | null = null;

beforeAll(async () => {
  strip = await sharedDevice("g6c-trial-strip", stripScene, { contactCapacity: 64, pairCapacity: 512 });
}, 180000);

describe("G6C globalTrialIndex (pure)", () => {
  it("accumulates earlier widths, not batchIndex * width", () => {
    expect(globalTrialIndex(0, 0)).toBe(0);
    expect(globalTrialIndex(0, 3)).toBe(3);
    // Schedule [2,4,4], accept in batch 1 (bi=1) at local row 2.
    expect(globalTrialIndex(2, 2)).toBe(4);
    // The old bi*K identity reports 1*4+2 = 6 here — wrong by the
    // accumulated width mismatch (documented divergence, now fixed).
    expect(1 * 4 + 2).not.toBe(globalTrialIndex(2, 2));
    // Tail-nonuniform schedule [4,4,2], accept in the tail at local 1.
    expect(globalTrialIndex(8, 1)).toBe(9);
  });
});

describe("G6C commit_arm cumulative base (device)", () => {
  /** Run one commit_arm with a seeded batch verdict; returns newtonCtl. */
  async function runCommitArm(
    fix: DeviceFixture, K: number, verdict: {
      accepted: boolean; alpha: number; localIndex: number; energy: number;
      minDist: number; minToi: number; gradNorm?: number;
    },
  ): Promise<Float32Array> {
    const { ex, driver } = fix;
    const batch = new Float32Array(16);
    batch[0] = verdict.accepted ? 1 : 0;
    batch[1] = verdict.alpha;
    batch[2] = verdict.accepted ? verdict.localIndex : 4294967295.0;
    batch[3] = 1; // trialsEvaluated
    batch[8] = 1; // finite lane
    batch[9] = verdict.energy;
    batch[10] = verdict.gradNorm ?? 0;
    batch[13] = verdict.minDist;
    batch[14] = verdict.minToi;
    ex.writeBuffer("armijoStatus", batch);
    driver.bankU(GpuUniformSlot.ArmijoK, K);
    ex.beginBatch("commit-arm-probe");
    ex.runPass({
      shader: "newton-control", entry: "commit_arm",
      groups: [[
        { binding: 50, buffer: "armijoStatus" },
        { binding: 52, buffer: "newtonCtl" },
        { binding: 53, buffer: "e0Store" },
        { binding: 54, buffer: "newtonStatus" },
        { binding: 55, buffer: "uniformBank", offset: GpuUniformSlot.ArmijoK * 256, size: 4 },
      ]],
      x: 1,
    });
    await ex.submitBatch(false);
    return new Float32Array(await ex.readBufferDebug("newtonCtl", "commit-arm-ctl", false));
  }

  it("reject-then-accept across [2,4]: global index uses the cumulative base", async () => {
    if (!strip) return;
    const { ex } = strip;
    ex.writeBuffer("newtonCtl", new Float32Array(16));
    ex.writeBuffer("newtonStatus", new Float32Array(20));
    ex.writeBuffer("e0Store", new Float32Array([2.5, 0, 0, 0]));
    // Batch 0 (K=2) rejects: no latch, base advances by THIS width.
    let ctl = await runCommitArm(strip, 2, {
      accepted: false, alpha: 0, localIndex: 0, energy: 3.0, minDist: 1e30, minToi: 2.0,
    });
    expect(ctl[2]).toBe(0); // nbAccepted sticky off
    expect(ctl[5]).toBe(1); // nbBatch self-tracked
    expect(ctl[7]).toBe(2); // cumulative trial base = 2, not bi*K
    // Batch 1 (K=4) accepts at local row 2: global = 2 + 2 = 4.
    // The old bi*K+li identity would report 1*4+2 = 6.
    ctl = await runCommitArm(strip, 4, {
      accepted: true, alpha: 0.0625, localIndex: 2, energy: 2.0, minDist: 0.001, minToi: 2.0,
    });
    // eslint-disable-next-line no-console
    console.log(`[g6c-trial] latch idx=${ctl[8]} base=${ctl[7]} batch=${ctl[6]} merit=${new Float32Array(await ex.readBufferDebug("newtonStatus", "commit-arm-st", false))[11]}`);
    expect(ctl[2]).toBe(1);
    expect(ctl[4]).toBeCloseTo(0.0625, 7);
    expect(ctl[6]).toBe(1); // accepting batch index
    expect(ctl[8]).toBe(4); // cumulative base 2 + local 2 (NOT 6)
    expect(ctl[7]).toBe(6); // base advanced by this batch's width 4
    // Batch 2 (K=4) rejects after an accept: latch sticky, base still advances.
    ctl = await runCommitArm(strip, 4, {
      accepted: false, alpha: 0, localIndex: 0, energy: 3.0, minDist: 1e30, minToi: 2.0,
    });
    expect(ctl[2]).toBe(1);
    expect(ctl[8]).toBe(4); // latch untouched
    expect(ctl[7]).toBe(10); // 6 + 4
    expect(ctl[5]).toBe(3);
  }, 300000);

  it("holds round-start E0 and accepted diagnostics through later rejected batches", async () => {
    if (!strip) return;
    const { ex } = strip;
    ex.writeBuffer("newtonCtl", new Float32Array(16));
    ex.writeBuffer("newtonStatus", new Float32Array(20));
    ex.writeBuffer("e0Store", new Float32Array([5, 0, 0, 0]));
    await runCommitArm(strip, 2, {
      accepted: true, alpha: 0.5, localIndex: 0, energy: 3,
      minDist: 0.003, minToi: 2, gradNorm: 0.25,
    });
    // The next speculative batch still compares against the energy at round
    // start. The accepted energy is latched only after all batches complete.
    expect(new Float32Array(await ex.readBufferDebug("e0Store", "round-e0", false))[0]).toBe(5);
    await runCommitArm(strip, 4, {
      accepted: false, alpha: 0, localIndex: 0, energy: 4,
      minDist: 1e30, minToi: 0.5, gradNorm: 9,
    });
    ex.beginBatch("round-report-latch");
    ex.runPass({
      shader: "newton-control", entry: "round_report",
      groups: [[
        { binding: 90, buffer: "newtonCtl" },
        { binding: 91, buffer: "newtonStatus" },
        { binding: 92, buffer: "gtdxStore" },
        { binding: 93, buffer: "e0Store" },
      ]],
      x: 1,
    });
    await ex.submitBatch(false);
    const status = new Float32Array(await ex.readBufferDebug("newtonStatus", "latched-round-status", false));
    expect(status[2]).toBe(0);
    expect(status[9]).toBe(0.25);
    expect(status[12]).toBe(3);
    expect(status[13]).toBeCloseTo(0.003, 6);
    expect(status[14]).toBe(2);
    expect(new Float32Array(await ex.readBufferDebug("e0Store", "next-round-e0", false))[0]).toBe(3);
  }, 300000);

  it("reports convergence without a commit as done, not failure", async () => {
    if (!strip) return;
    const { ex } = strip;
    const ctl = new Float32Array(16);
    ctl[0] = 1; // done from newton_check
    ex.writeBuffer("newtonCtl", ctl);
    const status0 = new Float32Array(20);
    status0[1] = 1; // converged
    status0[9] = 1e-8;
    status0[13] = 0.004; // current-state min distance from newton_check
    status0[14] = 2;
    ex.writeBuffer("newtonStatus", status0);
    ex.writeBuffer("e0Store", new Float32Array([2, 0, 0, 0]));
    ex.writeBuffer("armijoStatus", new Float32Array(16));
    const staleBatch = new Float32Array(16);
    staleBatch[13] = 0;
    staleBatch[14] = 0.5;
    ex.writeBuffer("armijoStatus", staleBatch);
    ex.beginBatch("converged-round-report");
    ex.runPass({
      shader: "newton-control", entry: "round_report",
      groups: [[
        { binding: 90, buffer: "newtonCtl" },
        { binding: 91, buffer: "newtonStatus" },
        { binding: 92, buffer: "gtdxStore" },
        { binding: 93, buffer: "e0Store" },
      ]],
      x: 1,
    });
    await ex.submitBatch(false);
    const status = new Float32Array(await ex.readBufferDebug("newtonStatus", "converged-round-status", false));
    expect(status[1]).toBe(1);
    expect(status[2]).toBe(0);
    expect(status[5]).toBe(0);
    expect(status[7]).toBe(0);
    expect(status[8]).toBeGreaterThan(1e9);
    expect(status[13]).toBeCloseTo(0.004, 6);
    expect(status[14]).toBe(2);

    const retryCtl = new Float32Array(16);
    ex.writeBuffer("newtonCtl", retryCtl);
    const retryStatus = new Float32Array(20);
    retryStatus[13] = 0.004;
    retryStatus[14] = 2;
    ex.writeBuffer("newtonStatus", retryStatus);
    ex.beginBatch("rejected-round-report");
    ex.runPass({
      shader: "newton-control", entry: "round_report",
      groups: [[
        { binding: 90, buffer: "newtonCtl" },
        { binding: 91, buffer: "newtonStatus" },
        { binding: 92, buffer: "gtdxStore" },
        { binding: 93, buffer: "e0Store" },
      ]],
      x: 1,
    });
    await ex.submitBatch(false);
    const rejected = new Float32Array(await ex.readBufferDebug("newtonStatus", "rejected-round-status", false));
    expect(rejected[2]).toBe(1);
    expect(rejected[13]).toBeCloseTo(0.004, 6);
    expect(rejected[14]).toBe(2);
  }, 300000);

  it("end-to-end: selectedAlpha agrees with selectedTrialIndex on non-uniform schedules", async () => {
    if (!strip) return;
    const { solver, ex, driver } = strip;
    const scene = (solver as unknown as { scene: { positions: Float64Array } }).scene;
    const x0 = Float64Array.from(scene.positions);
    const schedules = [[2, 4, 4], [4, 3, 3], [8, 2]];
    for (const batchKs of schedules) {
      const total = batchKs.reduce((a, b) => a + b, 0);
      solver.configureStep(1 / 60);
      const ev = await solver.evaluateNewtonState(false, 1, 1 / 60);
      ex.writeBuffer("e0Store", new Float32Array([ev.status.energy, 0, 0, 0]));
      driver.bankF(GpuUniformSlot.NewtonTol, 1e-5);
      driver.bankF(GpuUniformSlot.NewtonTrust, 0.002);
      ex.writeBuffer("newtonCtl", new Float32Array(16));
      ex.writeBuffer("newtonStatus", new Float32Array(20));
      const rawMt = (solver as unknown as { scene: { material: Record<string, number> } }).scene.material;
      const beta = Math.max(rawMt.stretchWarp, rawMt.stretchWeft, rawMt.shear) * rawMt.thickness * 0.1 + 1e-6;
      const mt = (solver as unknown as {
        materialNow(): { c00: number; c11: number; c01: number; g: number; thickness: number };
        contactParamsNow(): {
          dHat: number; kappa: number; mu: number; fricEps: number;
          floorY: number; floorOn: number; dMin: number; contactCapacity: number;
        };
      });
      const st = await driver.newtonRound({
        round: 0, beta, mat: mt.materialNow(), contact: mt.contactParamsNow(),
        batchKs, evalIndexBase: 0,
      });
      const trustScale = new Float32Array(
        await ex.readBufferDebug("trustScaleStore", "trial-trust", false),
      )[0];
      // eslint-disable-next-line no-console
      console.log(`[g6c-trial] Ks=${JSON.stringify(batchKs)} accepted=${st.armijoAccepted} ` +
        `idx=${st.selectedTrialIndex} alpha=${st.selectedAlpha} trust=${trustScale}`);
      if (!st.armijoAccepted) continue;
      expect(st.selectedTrialIndex).toBeGreaterThanOrEqual(0);
      expect(st.selectedTrialIndex).toBeLessThan(total);
      // Alpha independently encodes the global index (raw alphas are exact
      // powers of 0.5 from the trial base); a miscomputed index disagrees.
      const rawAlpha = st.selectedAlpha / trustScale;
      const expectRaw = Math.pow(0.5, st.selectedTrialIndex);
      expect(Math.abs(rawAlpha - expectRaw) / expectRaw).toBeLessThan(1e-5);
      // Restore start state for the next schedule.
      const { resetDeviceState } = await import("./device-setup.js");
      resetDeviceState(strip, x0);
    }
  }, 600000);
});
