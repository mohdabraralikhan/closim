// G5.5 device coarse ladder: C0 + inner-PCG 4/8/16/32 on the strip scene at
// 1x and 10x membrane stiffness, fixed 20 outer PCG iterations, same Newton
// state per cell. Informational (loose asserts: finite, non-divergent);
// the numbers feed the G5.5 decision matrix.
import { describe, it, expect, beforeAll } from "vitest";
import type { ClothScene } from "../../src/physics/scene.js";
import type { DeviceFixture } from "./device-setup.js";
import { sharedDevice, resetDeviceState, stripScene } from "./device-setup.js";

let strip: DeviceFixture | null = null;

beforeAll(async () => {
  strip = await sharedDevice("coarse-ladder-strip", stripScene, { contactCapacity: 64, pairCapacity: 512 });
}, 180000);

function stiffen(scene: ClothScene, base: ClothScene["material"], memK: number): void {
  // Scene-level material: evaluateNewtonState re-uploads uniforms from
  // scene.material every eval (bank writes would be overwritten). Absolute
  // scaling off the captured base (never compounds across cells).
  scene.material = {
    ...base,
    stretchWarp: base.stretchWarp * memK,
    stretchWeft: base.stretchWeft * memK,
    stretchCoupling: base.stretchCoupling * memK,
    shear: base.shear * memK,
  };
}

describe("G5.5 device coarse ladder", () => {
  it("reports the C0/C1 ladder at 1x and 10x stiffness", async () => {
    if (!strip) return;
    const { solver, ex, driver } = strip;
    const scene = (solver as unknown as { scene: ClothScene }).scene;
    const cfgs: Array<{ name: string; apply: () => void }> = [
      { name: "schwarz1", apply: () => { driver.cfg.useSchwarz = true; } },
      { name: "c0", apply: () => { driver.cfg.useCoarseC0 = true; } },
      { name: "c1-4", apply: () => { driver.cfg.useCoarsePcg = true; driver.cfg.coarseIters = 4; } },
      { name: "c1-8", apply: () => { driver.cfg.useCoarsePcg = true; driver.cfg.coarseIters = 8; } },
      { name: "c1-16", apply: () => { driver.cfg.useCoarsePcg = true; driver.cfg.coarseIters = 16; } },
      { name: "c1-32", apply: () => { driver.cfg.useCoarsePcg = true; driver.cfg.coarseIters = 32; } },
    ];
    const prev = { ...driver.cfg };
    const prevMat = { ...scene.material };
    const baseMat = { ...scene.material };
    try {
      driver.cfg.pcgIters = 20;
      for (const memK of [1, 10]) {
        stiffen(scene, baseMat, memK);
        for (const c of cfgs) {
          Object.assign(driver.cfg, {
            useMas: false, useSchwarz: false, useBlockJacobi: false,
            useCoarseC0: false, useCoarsePcg: false,
          });
          c.apply();
          resetDeviceState(strip, Float64Array.from(scene.positions));
          solver.configureStep(1 / 60);
          await solver.evaluateNewtonState(false, 1, 1 / 60);
          const s0 = ex.ledger.submits;
          const rec: { rz: number[] } = { rz: [] };
          const res = await driver.pcgSolve(rec);
          const rel = rec.rz.length > 1 ? rec.rz[rec.rz.length - 1] / Math.max(rec.rz[0], 1e-300) : NaN;
          // eslint-disable-next-line no-console
          console.log(`[g5.5-ladder] memK=${memK} ${c.name}: resNorm=${res.resNorm.toExponential(3)} ` +
            `relRzEnd=${rel.toExponential(2)} breakdown=${res.breakdown} ` +
            `submits=${ex.ledger.submits - s0}`);
          expect(Number.isFinite(res.resNorm)).toBe(true);
        }
      }
    } finally {
      Object.assign(driver.cfg, prev);
      scene.material = prevMat;
    }
  }, 600000);
});
