// G6C.3 adaptive batch widths: policy unit table, trial-distribution
// instrumentation across regimes, and adaptive-vs-fixed end-to-end.
import { describe, it, expect, beforeAll } from "vitest";
import { adaptiveBatchK } from "../../src/backend/webgpu/gpu-newton.js";
import type { ClothScene } from "../../src/physics/scene.js";
import type { DeviceFixture } from "./device-setup.js";
import { sharedDevice, resetDeviceState, requireDevice, stripScene } from "./device-setup.js";
import { buildGrid, preprocess } from "../../src/mesh/mesh.js";
import { createScene } from "../../src/physics/scene.js";
import { DEFAULT_MATERIAL } from "../../src/physics/types.js";
import { ContactSystem } from "../../src/collision/contact-assembly.js";
import { DEFAULT_CONTACT_PARAMS } from "../../src/collision/types.js";

describe("G6C.3 adaptive policy unit table", () => {
  it("maps histories onto the {2,4,8} ladder", () => {
    expect(adaptiveBatchK(null, 4)).toBe(4); // cold start honors default
    expect(adaptiveBatchK(null, 8)).toBe(8);
    expect(adaptiveBatchK(null, 2)).toBe(2);
    expect(adaptiveBatchK(-1, 4)).toBe(8); // failed round keeps coverage
    expect(adaptiveBatchK(0, 4)).toBe(2); // accept-first shrinks speculation
    expect(adaptiveBatchK(1, 4)).toBe(4);
    expect(adaptiveBatchK(3, 4)).toBe(4);
    expect(adaptiveBatchK(4, 4)).toBe(8);
    expect(adaptiveBatchK(9, 4)).toBe(8);
  });

  it("clamps arbitrary defaults onto the ladder", () => {
    expect(adaptiveBatchK(null, 1)).toBe(2);
    expect(adaptiveBatchK(null, 3)).toBe(4);
    expect(adaptiveBatchK(null, 7)).toBe(8);
    expect(adaptiveBatchK(null, 100)).toBe(8);
  });
});

let strip: DeviceFixture | null = null;

beforeAll(async () => {
  strip = await sharedDevice("g6c-adaptive-strip", stripScene, { contactCapacity: 64, pairCapacity: 512 });
}, 180000);

function stiffScene(memK: number): ReturnType<typeof createScene> {
  const g = buildGrid(6, 6, 0.12, 0.12);
  const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
  const b = DEFAULT_MATERIAL;
  return createScene(mesh, {
    ...b,
    stretchWarp: b.stretchWarp * memK, stretchWeft: b.stretchWeft * memK,
    shear: b.shear * memK,
  }, [0, -9.81, 0]);
}

function foldScene(): ReturnType<typeof createScene> {
  const w = 0.16;
  const g = buildGrid(8, 8, w, w);
  const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
  const scene = createScene(mesh, { ...DEFAULT_MATERIAL }, [0, -9.81, 0]);
  for (let i = 0; i < mesh.count; i++) {
    const x = scene.positions[i * 3];
    if (x > w / 2) {
      scene.positions[i * 3] = w / 2 - 2 * (x - w / 2);
      scene.positions[i * 3 + 1] += 0.0015;
    }
  }
  scene.contact = new ContactSystem({ ...DEFAULT_CONTACT_PARAMS }, mesh.indices);
  return scene;
}

describe("G6C.3 trial distributions + adaptive end-to-end", () => {
  it("logs accepted-trial distributions by regime (measurement first)", async () => {
    const cases: Array<{ name: string; build: () => ReturnType<typeof createScene> }> = [
      { name: "strip-1x", build: stripScene },
      { name: "strip-10x", build: () => stiffScene(10) },
      { name: "fold-1x", build: foldScene },
    ];
    for (const c of cases) {
      const fix = await requireDevice(c.build, { contactCapacity: 2048, pairCapacity: 8192 });
      if (!fix) {
        // eslint-disable-next-line no-console
        console.log(`[g6c-dist] no device for ${c.name} — skipped`);
        continue;
      }
      try {
        const { solver, driver } = fix;
        const scene = (solver as unknown as { scene: ClothScene }).scene;
        // Sequential baseline first: accepts here prove any batched rejects
        // are genuine stalls, not batch-path bugs.
        driver.cfg.useBatchedArmijo = false;
        const seqHist: number[] = [];
        for (let s = 0; s < 3; s++) {
          resetDeviceState(fix, Float64Array.from(scene.positions));
          await solver.stepGpu(1 / 60, { newtonIters: 2 });
          seqHist.push(...solver.trialHistory);
        }
        driver.cfg.useBatchedArmijo = true;
        driver.cfg.armijoBatchK = 8; // full coverage: observe natural accepts
        const hist: number[] = [];
        for (let s = 0; s < 3; s++) {
          resetDeviceState(fix, Float64Array.from(scene.positions));
          await solver.stepGpu(1 / 60, { newtonIters: 2 });
          hist.push(...solver.trialHistory);
        }
        driver.cfg.useBatchedArmijo = false;
        // eslint-disable-next-line no-console
        console.log(`[g6c-dist] ${c.name}: seq=${JSON.stringify(seqHist)} batched=${JSON.stringify(hist)}`);
        for (const t of hist) expect(Number.isInteger(t)).toBe(true);
      } finally {
        fix.ex.destroy();
      }
    }
  }, 600000);

  it("adaptive matches fixed-K4 end-to-end on strip (searching regime)", async () => {
    const fix = await requireDevice(stripScene, { contactCapacity: 64, pairCapacity: 512 });
    if (!fix) return;
    try {
      const { solver, ex, driver } = fix;
      const scene = (solver as unknown as { scene: ClothScene }).scene;
      const x0 = Float64Array.from(scene.positions);
      const n = scene.mesh.count;
      const res: Record<string, { submits: number; syncs: number; energy: number; trials: number[] }> = {};
      for (const adaptive of [false, true]) {
        resetDeviceState(fix, Float64Array.from(x0));
        ex.writeBuffer("velocity", new Float32Array(n * 4));
        driver.cfg.useBatchedArmijo = true;
        driver.cfg.armijoBatchK = 4;
        driver.cfg.adaptiveK = adaptive;
        const s0 = ex.ledger.submits;
        const r0 = solver.hotLoopReadbacks;
        const d = await solver.stepGpu(1 / 60, { newtonIters: 2 });
        res[adaptive ? "adaptive" : "fixed"] = {
          submits: ex.ledger.submits - s0,
          syncs: solver.hotLoopReadbacks - r0,
          energy: d.energy,
          trials: [...solver.trialHistory],
        };
      }
      driver.cfg.useBatchedArmijo = false;
      driver.cfg.adaptiveK = false;
      // eslint-disable-next-line no-console
      console.log(`[g6c-adaptive] fixed=${JSON.stringify(res.fixed)} adaptive=${JSON.stringify(res.adaptive)}`);
      expect(Number.isFinite(res.adaptive.energy)).toBe(true);
      expect(Number.isFinite(res.fixed.energy)).toBe(true);
      // Same selections (policy only changes coverage): identical energy.
      expect(res.adaptive.energy).toBe(res.fixed.energy);
      expect(res.adaptive.trials).toEqual(res.fixed.trials);
      // Adaptive shrinks speculation on accept-first histories.
      expect(res.adaptive.submits).toBeLessThanOrEqual(res.fixed.submits);
    } finally {
      fix.ex.destroy();
    }
  }, 600000);
});
