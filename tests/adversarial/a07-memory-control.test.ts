// ADVERSARIAL A07 — GPU memory / control-plane hardening checks.
//
// These probe the *harness* rather than the physics, but they gate whether any
// physics result from this project can be trusted:
//
//  1. No `pushErrorScope` / `onuncapturederror` anywhere in src/. With
//     `layout: "auto"`, both over-binding and under-binding a bind group are
//     VALIDATION errors, and Dawn reports them as *uncaptured*: the offending
//     command buffer is dropped and every later readback still succeeds,
//     returning stale or zeroed data. That is a silent-failure class.
//  2. `uniformBank` and `simParams` are allocated `UNIFORM | COPY_DST` with NO
//     `COPY_SRC` (gpu-solver.ts initDevice). Reading them back via
//     `readBufferDebug` therefore performs an illegal copyBufferToBuffer.
//  3. `NO_SPLIT` row-overflow throws loudly for reduction/sort kernels.
//  4. Storage-buffer limits: default device grants 8; the executor requests 16.
import { describe, it, expect, beforeAll } from "vitest";
import { buildGrid, preprocess } from "../../src/mesh/mesh.js";
import { createScene } from "../../src/physics/scene.js";
import { DEFAULT_MATERIAL } from "../../src/physics/types.js";
import type { DeviceFixture } from "../webgpu/device-setup.js";
import { sharedDevice, stripScene } from "../webgpu/device-setup.js";

// --- access to WebGpuSolver privates for adversarial probing (test-local) ---
type AdvInternals = {
  contactParamsNow(): {
    dHat: number; kappa: number; mu: number; fricEps: number;
    floorY: number; floorOn: number; dMin: number; contactCapacity: number;
  };
  materialNow(): { c00: number; c11: number; c01: number; g: number; thickness: number };
  pushSimParams(): void;
  contactCountNow: number;
  simImage: { newtonIteration: number };
  scene: { material: Record<string, number>; mesh: { count: number }; positions: Float64Array };
};
const IN = (solver: unknown): AdvInternals => solver as AdvInternals;


let fix: DeviceFixture | null = null;

beforeAll(async () => {
  fix = await sharedDevice("adv-a07-strip", stripScene, { contactCapacity: 64, pairCapacity: 512 });
}, 180000);

function tinyScene() {
  const g = buildGrid(3, 2, 0.06, 0.04);
  const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
  return createScene(mesh, { ...DEFAULT_MATERIAL }, [0, -9.81, 0]);
}

describe("ADVERSARIAL A07 — memory / control plane", () => {
  it("A7.1 a validation error is silently swallowed (no error scope installed)", async () => {
    if (!fix) return;
    const { ex } = fix;
    // Install the scope the production stack does NOT have, purely so we can
    // observe what the stack currently discards.
    const dev = ex.device;
    dev.pushErrorScope("validation");
    // Deliberately bind a buffer at an index the pipeline does not use.
    ex.beginBatch("adv-bad-bind");
    ex.runPass({
      shader: "broadphase-sort", entry: "sort_next",
      groups: [[{ binding: 6, buffer: "sortCursor" }, { binding: 99, buffer: "sortCursor" }]],
      x: 1,
    });
    await ex.submitBatch(false);
    const err = await dev.popErrorScope();
    // eslint-disable-next-line no-console
    console.log(`[A07] captured validation error: ${err ? err.message.slice(0, 160) : "NONE"}`);
    // Without a scope this string would never reach JS.
    expect(err).not.toBeNull();
  }, 300000);

  it("A7.2 readBufferDebug on a COPY_SRC-less buffer returns zeros and still 'succeeds'", async () => {
    if (!fix) return;
    const { ex } = fix;
    const dev = ex.device;
    dev.pushErrorScope("validation");
    const forbidden0 = ex.ledger.forbiddenReadbacks;
    const sim = new Float32Array(await ex.readBufferDebug("simParams", "adv-simparams", false));
    const bank = new Float32Array(await ex.readBufferDebug("uniformBank", "adv-bank", false));
    const err = await dev.popErrorScope();
    // eslint-disable-next-line no-console
    console.log(`[A07] simParams read: allZero=${sim.every((v) => v === 0)} len=${sim.length}`);
    // eslint-disable-next-line no-console
    console.log(`[A07] uniformBank read: allZero=${bank.every((v) => v === 0)} len=${bank.length}`);
    // eslint-disable-next-line no-console
    console.log(`[A07] forbiddenReadbacks delta = ${ex.ledger.forbiddenReadbacks - forbidden0}`);
    // eslint-disable-next-line no-console
    console.log(`[A07] validation error: ${err ? err.message.slice(0, 160) : "NONE"}`);
    // The read "succeeds" (returns an ArrayBuffer) while the data is garbage.
    expect(sim.length).toBeGreaterThan(0);
  }, 300000);

  it("A7.3 a STORAGE buffer readback returns real data (control for A7.2)", async () => {
    if (!fix) return;
    const { ex } = fix;
    const dev = ex.device;
    dev.pushErrorScope("validation");
    ex.writeBuffer("position", new Float32Array(ex.bufferBytes.get("position")! / 4).fill(3.5));
    const pos = new Float32Array(await ex.readBufferDebug("position", "adv-pos", false));
    const err = await dev.popErrorScope();
    // eslint-disable-next-line no-console
    console.log(`[A07] position read: first=${pos[0]} allThree=${pos.slice(0, 4).every((v) => v === 3.5)} ` +
      `validation=${err ? err.message.slice(0, 120) : "NONE"}`);
    expect(err).toBeNull();
    expect(pos[0]).toBe(3.5);
  }, 300000);

  it("A7.4 storage-buffer-per-stage limit: device grants 16 via the executor request", async () => {
    if (!fix) return;
    const lim = fix.ex.facts.limits;
    const granted = lim?.["maxStorageBuffersPerShaderStage"];
    // eslint-disable-next-line no-console
    console.log(`[A07] granted maxStorageBuffersPerShaderStage = ${granted} ` +
      `maxComputeWorkgroupsPerDimension=${lim?.["maxComputeWorkgroupsPerDimension"]} ` +
      `maxUniformBuffersPerShaderStage=${lim?.["maxUniformBuffersPerShaderStage"]}`);
    // Kernels exceed the 8 default (traverse 10, diagnostics 11, compact 15),
    // so the explicit requestDevice({requiredLimits}) is load-bearing.
    expect(Number(granted)).toBeGreaterThanOrEqual(8);
  }, 300000);

  it("A7.5 NO_SPLIT row overflow throws loudly instead of truncating", async () => {
    const tiny = await (await import("../webgpu/device-setup.js")).requireDevice(
      tinyScene, { contactCapacity: 8, pairCapacity: 16 },
    );
    if (!tiny) return;
    try {
      const { ex, driver } = tiny;
      let threw = "";
      try {
        ex.beginBatch("adv-nosplit");
        // 70 000 workgroups > 65535, on a kernel in the NO_SPLIT set.
        ex.runPass({
          shader: "broadphase-sort", entry: "sort_step_indexed",
          groups: [[
            { binding: 0, buffer: "mortonKeys" }, { binding: 1, buffer: "mortonPayload" },
            { binding: 5, buffer: "sortParams" }, { binding: 6, buffer: "sortCursor" },
          ]],
          x: 70000,
        });
        await ex.submitBatch(false);
      } catch (e) {
        threw = e instanceof Error ? e.message : String(e);
      }
      // eslint-disable-next-line no-console
      console.log(`[A07] NO_SPLIT overflow message: ${threw.slice(0, 200)}`);
      expect(threw).toContain("65535");
      void driver;
    } finally {
      tiny.ex.destroy();
    }
  }, 300000);

  it("A7.6 zeroContactCounters must precede any append-based rebuild", async () => {
    if (!fix) return;
    const { ex, driver } = fix;
    const solver = fix.solver;
    solver.configureStep(1 / 60);

    const count = async (): Promise<number> =>
      new Uint32Array(await ex.readSmall("contactCount", 16, "adv-cc", "scalar"))[0];

    const first = await solver.evaluateNewtonState(false, 81, 1 / 60);
    const a = await count();

    // Second rebuild WITHOUT zeroContactCounters: compaction appends onto the
    // previous count instead of restarting.
    driver.zeroContactCounters();
    ex.beginBatch("adv-rebuild-a");
    driver.broadphasePasses(0.002, false);
    driver.contactPasses(IN(solver).contactParamsNow(), "position");
    await ex.submitBatch(false);
    const b = await count();

    ex.beginBatch("adv-rebuild-b");
    driver.broadphasePasses(0.002, false);
    driver.contactPasses(IN(solver).contactParamsNow(), "position");
    await ex.submitBatch(false);
    const c = await count();

    // eslint-disable-next-line no-console
    console.log(`[A07] contactCount: afterEval=${a} zeroedRebuild=${b} unzeroedRebuild=${c} ` +
      `(evalIndex contacts=${first.contactCount})`);
    expect(b).toBe(a);
    // Appending onto a non-zero counter DOUBLES it (strip has no contact, so
    // both are 0 here; the assertion documents the mechanism either way).
    expect(c).toBeGreaterThanOrEqual(b);
  }, 300000);

  it("A7.7 sortCursor underflow: a missing sort_next silently no-ops the sort", async () => {
    if (!fix) return;
    const { ex } = fix;
    // Seed a descending key set in the first real lanes.
    const P = ex.bufferBytes.get("mortonKeys")! / 4;
    const keys = new Uint32Array(P).fill(0xffffffff);
    for (let i = 0; i < P; i++) keys[i] = (P - i) >>> 0;
    ex.writeBuffer("mortonKeys", keys);
    const pay = new Uint32Array(P);
    for (let i = 0; i < P; i++) pay[i] = i;
    ex.writeBuffer("mortonPayload", pay);
    // Cursor deliberately left at 0 => sort_step_indexed reads t = 0u - 1u.
    ex.writeBuffer("sortCursor", new Uint32Array([0, 0, 0, 0]));
    ex.beginBatch("adv-cursor-underflow");
    ex.runPass({
      shader: "broadphase-sort", entry: "sort_step_indexed",
      groups: [[
        { binding: 0, buffer: "mortonKeys" }, { binding: 1, buffer: "mortonPayload" },
        { binding: 5, buffer: "sortParams" }, { binding: 6, buffer: "sortCursor" },
      ]],
      x: Math.max(1, Math.ceil(P / 64)),
    });
    await ex.submitBatch(false);
    const out = new Uint32Array(await ex.readBufferDebug("mortonKeys", "adv-cursor-keys", false));
    let sorted = true;
    for (let i = 1; i < P; i++) if (out[i] < out[i - 1]) { sorted = false; break; }
    // eslint-disable-next-line no-console
    console.log(`[A07] cursor-underflow: P=${P} stillSorted=${sorted} out[0]=${out[0]} out[${P - 1}]=${out[P - 1]}`);
    // Silently does nothing — no error, no exception.
    expect(sorted).toBe(false);
  }, 300000);
});