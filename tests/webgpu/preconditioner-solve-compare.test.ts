// G5 device solve comparison: Jacobi / Block-Jacobi / Schwarz-1 / MAS-2 on
// the strip scene, fixed 20 PCG iterations, same Newton state. Reports final
// residual + convergence curve endpoint per method (informational; assertions
// are deliberately loose — finite, non-divergent, same-problem ordering is
// NOT asserted across drivers).
import { describe, it, expect, beforeAll } from "vitest";
import type { ClothScene } from "../../src/physics/scene.js";
import type { DeviceFixture } from "./device-setup.js";
import { sharedDevice, resetDeviceState, stripScene } from "./device-setup.js";

let strip: DeviceFixture | null = null;

beforeAll(async () => {
  strip = await sharedDevice("solve-cmp-strip", stripScene, { contactCapacity: 64, pairCapacity: 512 });
}, 180000);

describe("G5 device solve comparison", () => {
  it("reports residual per preconditioner on one fixed system", async () => {
    if (!strip) return;
    const { solver, ex, driver } = strip;
    const scene = (solver as unknown as { scene: ClothScene }).scene;
    const cfgs: Array<{ name: string; mas: boolean; schwarz: boolean; block: boolean }> = [
      { name: "jacobi", mas: false, schwarz: false, block: false },
      { name: "block", mas: false, schwarz: false, block: true },
      { name: "schwarz1", mas: false, schwarz: true, block: false },
      { name: "mas2", mas: true, schwarz: false, block: false },
    ];
    const prev = { ...driver.cfg };
    try {
      driver.cfg.pcgIters = 20;
      for (const c of cfgs) {
        resetDeviceState(strip, Float64Array.from(scene.positions));
        solver.configureStep(1 / 60);
        await solver.evaluateNewtonState(false, 1, 1 / 60);
        driver.cfg.useMas = c.mas;
        driver.cfg.useSchwarz = c.schwarz;
        driver.cfg.useBlockJacobi = c.block;
        const s0 = ex.ledger.submits;
        const rec: { rz: number[] } = { rz: [] };
        const res = await driver.pcgSolve(rec);
        const rel = rec.rz.length > 1 ? rec.rz[rec.rz.length - 1] / Math.max(rec.rz[0], 1e-300) : NaN;
        // eslint-disable-next-line no-console
        console.log(`[g5-device-solve] ${c.name}: resNorm=${res.resNorm.toExponential(3)} ` +
          `relRzEnd=${rel.toExponential(2)} breakdown=${res.breakdown} ` +
          `submits=${ex.ledger.submits - s0}`);
        expect(Number.isFinite(res.resNorm)).toBe(true);
        expect(res.resNorm).toBeLessThan(rec.rz[0] === 0 ? Infinity : Math.sqrt(Math.max(rec.rz[0], 0)) * 10 + 1e-6);
      }
    } finally {
      Object.assign(driver.cfg, prev);
    }
  }, 180000);
});
