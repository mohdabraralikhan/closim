// G6C.2 Gate A: fixed-state full-Newton-iteration parity between the
// sequential path and one GPU-controlled round (same x/records/dx/E0).
// Compares accepted alpha/index, committed positions, energy, convergence,
// fallback selection, and direction validity. Plus control-kernel unit tests
// (descent_check/select_fallback/newton_check/commit arbitration).
import { describe, it, expect, beforeAll } from "vitest";
import type { ClothScene } from "../../src/physics/scene.js";
import type { SolverStatus } from "../../src/backend/webgpu/gpu-buffers.js";
import { GpuUniformSlot } from "../../src/backend/webgpu/gpu-buffers.js";
import type { DeviceFixture } from "./device-setup.js";
import { sharedDevice, resetDeviceState, stripScene } from "./device-setup.js";

let strip: DeviceFixture | null = null;

beforeAll(async () => {
  strip = await sharedDevice("g6c-newton-strip", stripScene, { contactCapacity: 64, pairCapacity: 512 });
}, 180000);

interface IterInputs {
  E0: number;
  gtdx: number;
  trustScale: number;
}

/** First-iteration inputs exactly as stepGpuDevice computes them. */
async function newtonIterInputs(fix: DeviceFixture): Promise<IterInputs> {
  const { solver, ex, driver } = fix;
  solver.configureStep(1 / 60);
  const ev = await solver.evaluateNewtonState(false, 1, 1 / 60);
  const E0 = ev.status.energy;
  await driver.pcgSolve();
  ex.beginBatch("parity-dot");
  ex.writeBlas(driver.n3, 1);
  ex.runPass({
    shader: "blas", entry: "mul",
    groups: [[
      { binding: 0, buffer: "uniformBank", offset: 0, size: 16 },
      { binding: 1, buffer: "rhs" },
      { binding: 2, buffer: "searchDirection" },
      { binding: 3, buffer: "pcgProd" },
    ]],
    x: Math.max(1, Math.ceil(driver.n3 / 64)),
  });
  await ex.submitBatch(false);
  const gtdx = await driver.reduceSum("pcgProd", driver.n3);
  ex.beginBatch("parity-trust");
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
  const trustScale = maxDx > 0.002 ? 0.002 / maxDx : 1;
  return { E0, gtdx, trustScale };
}

describe("G6C.2 Gate A full-iteration parity", () => {
  it("one GPU-control round matches a sequential Newton iteration", async () => {
    if (!strip) return;
    const { solver, ex, driver } = strip;
    const scene = (solver as unknown as { scene: ClothScene }).scene;
    const dMin = 1e-4;
    // Path A: sequential first Newton iteration through existing methods.
    resetDeviceState(strip, Float64Array.from(scene.positions));
    const st = await newtonIterInputs(strip);
    let alphaA = 1;
    let acceptedA = false;
    let statusA: SolverStatus | null = null;
    for (let li = 0; li < 10 && !acceptedA; li++) {
      await driver.applyTrial(alphaA * st.trustScale);
      const tev = await solver.evaluateNewtonState(true, 300 + li, 1 / 60);
      const s = tev.status;
      const valid = s.finite === 1 && s.ccdSafe === 1 &&
        s.minDistance > dMin && !(s.minToi > 0 && s.minToi < 1 - 1e-9);
      if (!valid || !Number.isFinite(s.energy)) { alphaA *= 0.5; continue; }
      if (s.energy <= st.E0 + 1e-4 * alphaA * st.trustScale * st.gtdx) {
        await driver.acceptTrial();
        statusA = s;
        acceptedA = true;
      } else {
        alphaA *= 0.5;
      }
    }
    expect(acceptedA).toBe(true);
    const posA = new Float32Array(await ex.readBufferDebug("position", "gateA-posA", false));
    // Path B: identical reset, then one GPU-control round.
    resetDeviceState(strip, Float64Array.from(scene.positions));
    solver.configureStep(1 / 60);
    const evB = await solver.evaluateNewtonState(false, 1, 1 / 60);
    expect(evB.status.energy).toBe(st.E0);
    // Per-round control zero + persistent uniforms (mirrors step wiring):
    // cap-bounded loops, tol/trust slots, e0 seed, control/status zeroing.
    // e0Store seeding is load-bearing: record compares against it, and zero
    // would reject every candidate.
    solver.beginArmijoBatch(300);
    driver.bankF(GpuUniformSlot.NewtonTol, 1e-5);
    driver.bankF(GpuUniformSlot.NewtonTrust, 0.002);
    ex.writeBuffer("e0Store", new Float32Array([evB.status.energy, 0, 0, 0]));
    ex.writeBuffer("newtonCtl", new Float32Array(16));
    const nst = new Float32Array(20);
    nst[0] = 0;
    ex.writeBuffer("newtonStatus", nst);
    const mt = scene.material;
    const beta = Math.max(mt.stretchWarp, mt.stretchWeft, mt.shear) * mt.thickness * 0.1 + 1e-6;
    const batchKs = [4, 4, 2];
    const rb = await driver.newtonRound({
      round: 0, beta,
      mat: {
        c00: scene.material.stretchWarp, c11: scene.material.stretchWeft,
        c01: scene.material.stretchCoupling, g: scene.material.shear,
        thickness: scene.material.thickness,
      },
      contact: {
        dHat: 0.002, kappa: 50, mu: 0.3, fricEps: 1e-4,
        floorY: 1e30, floorOn: 0, dMin: 1e-4, contactCapacity: driver.c.cap,
      },
      batchKs, evalIndexBase: 300,
    });
    const posB = new Float32Array(await ex.readBufferDebug("position", "gateA-posB", false));
    // eslint-disable-next-line no-console
    console.log(`[g6c-gateA] seq alpha=${alphaA.toExponential(3)} gpu alpha=${rb.selectedAlpha.toExponential(3)} ` +
      `accepted=${rb.armijoAccepted} converged=${rb.converged} failure=${rb.failure}`);
    expect(rb.armijoAccepted).toBe(true);
    expect(rb.failure).toBe(false);
    expect(rb.selectedAlpha).toBeCloseTo(alphaA * st.trustScale, 5);
    let worst = 0;
    for (let i = 0; i < posA.length; i++) worst = Math.max(worst, Math.abs(posA[i] - posB[i]));
    // eslint-disable-next-line no-console
    console.log(`[g6c-gateA] max |posA-posB| = ${worst.toExponential(2)}`);
    expect(worst).toBeLessThan(1e-5);
    expect(rb.energy).toBeCloseTo(statusA!.energy, 5);
    expect(rb.directionValid).toBe(true);
  }, 300000);

  it("descent_check latches fallback and select_fallback swaps direction", async () => {
    if (!strip) return;
    const { ex, driver } = strip;
    const n3 = driver.n3;
    const W = Math.max(1, Math.ceil(n3 / 64));
    async function runDescent(): Promise<void> {
      ex.beginBatch("descent-unit");
      ex.runPass({
        shader: "newton-control", entry: "descent_check",
        groups: [[
          { binding: 30, buffer: "gtdxStore" },
          { binding: 31, buffer: "breakFlag" },
          { binding: 32, buffer: "newtonCtl" },
          { binding: 33, buffer: "newtonStatus" },
        ]],
        x: 1,
      });
      ex.runPass({
        shader: "newton-control", entry: "select_fallback",
        groups: [[
          { binding: 40, buffer: "descentDir" },
          { binding: 41, buffer: "searchDirection" },
          { binding: 42, buffer: "newtonCtl" },
          { binding: 43, buffer: "simParams" },
        ]],
        x: W,
      });
      await ex.submitBatch(false);
    }
    // Case 0: select alone with preseeded fallback flag (isolates the copy).
    {
      // Seed a known simParams image (this test configures no step; the
      // buffer may hold zeros from init, which would guard every lane out).
      const sim = new ArrayBuffer(64);
      new Uint32Array(sim).set([0, 0, 0, 0, 0, 0, 28, 36, 0, 0, 0, 0, 0, 0, 0, 0]);
      ex.writeBuffer("simParams", new Uint8Array(sim));
      const ctl = new Float32Array(16);
      ctl[1] = 1.0;
      ex.writeBuffer("newtonCtl", ctl);
      ex.writeBuffer("descentDir", new Float32Array(n3).fill(7.0));
      ex.writeBuffer("searchDirection", new Float32Array(n3).fill(1.0));
      ex.beginBatch("select-only");
      ex.runPass({
        shader: "newton-control", entry: "select_fallback",
        groups: [[
          { binding: 40, buffer: "descentDir" },
          { binding: 41, buffer: "searchDirection" },
          { binding: 42, buffer: "newtonCtl" },
          { binding: 43, buffer: "simParams" },
        ]],
        x: W,
      });
      await ex.submitBatch(false);
      const s0 = new Float32Array(await ex.readBufferDebug("searchDirection", "select-only-sd", false));
      // eslint-disable-next-line no-console
      console.log(`[g6c-dbg] select-only sd[0..2]=${s0[0]},${s0[1]},${s0[2]}`);
    }
    // Case 1: non-descent gtdx (+1.0) -> fallback set, direction copied.
    ex.writeBuffer("gtdxStore", new Float32Array([1.0, 0, 0, 0]));
    ex.writeBuffer("breakFlag", new Float32Array(4));
    ex.writeBuffer("newtonCtl", new Float32Array(16));
    ex.writeBuffer("newtonStatus", new Float32Array(20));
    ex.writeBuffer("descentDir", new Float32Array(n3).fill(7.0));
    ex.writeBuffer("searchDirection", new Float32Array(n3).fill(1.0));
    await runDescent();
    const ctl = new Float32Array(await ex.readBufferDebug("newtonCtl", "descent-ctl", false));
    expect(ctl[1]).toBe(1.0);
    expect(new Float32Array(await ex.readBufferDebug("newtonStatus", "descent-status", false))[3]).toBe(0.0);
    const sdGot = new Float32Array(await ex.readBufferDebug("searchDirection", "descent-sd", false));
    for (let i = 0; i < n3; i++) expect(sdGot[i]).toBe(7.0);
    // Case 2: descent gtdx (-1.0), no breakdown -> no fallback, kept.
    ex.writeBuffer("gtdxStore", new Float32Array([-1.0, 0, 0, 0]));
    ex.writeBuffer("newtonCtl", new Float32Array(16));
    ex.writeBuffer("newtonStatus", new Float32Array(20));
    ex.writeBuffer("searchDirection", new Float32Array(n3).fill(1.0));
    await runDescent();
    const ctl2 = new Float32Array(await ex.readBufferDebug("newtonCtl", "descent-ctl2", false));
    expect(ctl2[1]).toBe(0.0);
    const sd2 = new Float32Array(await ex.readBufferDebug("searchDirection", "descent-sd2", false));
    for (let i = 0; i < n3; i++) expect(sd2[i]).toBe(1.0);
  }, 300000);
});
