// MIMO scale-contact stress: alternating contact/no-contact frames and
// candidate/trial buffer reuse.
//
// 1. Frames alternate between an engaged height (1.2 mm, all verts within
//    dHat) and a clear height (50 mm). Contact frames must reproduce the
//    canonical live multiset exactly; clear frames must be empty AND match a
//    never-contacted control (proves stale tail records are inert by the
//    count/cap guards rather than luck).
// 2. Rejected sequential trials never touch laggedN or position (bitwise),
//    across repeated trials on a contact-heavy scene.
// 3. laggedN written by an accept survives a clear frame untouched and the
//    next accept on re-engagement stays finite (stale-normal robustness).
import { describe, it, expect } from "vitest";
import type { ClothScene } from "../../../src/physics/scene.js";
import { requireDevice, resetDeviceState, type DeviceFixture } from "../../webgpu/device-setup.js";
import {
  BIG_CAP, GRID, IN, PAIR_CAP,
  buildScene, checkLiveRecords, maxDiff, multisetEqual, liveKeys,
  readContactLive, readCounters, readF32,
  type ScaleClass,
} from "./scale-helpers.js";

const DT = 1 / 60;

/** Pack positions with every Y replaced by `height` (frame teleport). */
function packFlatY(fix: DeviceFixture, x0: Float64Array, height: number): Float32Array {
  const scene = (fix.solver as unknown as { scene: ClothScene }).scene;
  const n = scene.mesh.count;
  const out = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    out[i * 4] = Math.fround(x0[i * 3]);
    out[i * 4 + 1] = height;
    out[i * 4 + 2] = Math.fround(x0[i * 3 + 2]);
  }
  return out;
}

async function evalFrame(fix: DeviceFixture, x0: Float64Array, height: number, idx: number): Promise<number> {
  const { solver, ex } = fix;
  const packed = packFlatY(fix, x0, height);
  ex.writeBuffer("position", packed);
  ex.writeBuffer("position0", packed);
  ex.writeBuffer("xTrial", packed);
  solver.configureStep(DT);
  const ev = await solver.evaluateNewtonState(false, idx, DT);
  return ev.contactCount;
}

describe("MIMO alternating contact/no-contact frames", () => {
  it("1k/floor: engaged frames reproduce the canonical set; clear frames match control", async () => {
    const cls: ScaleClass = "1k";
      const fix = await requireDevice(() => buildScene("floor", cls), {
        contactCapacity: BIG_CAP[cls], pairCapacity: PAIR_CAP[cls],
      });
      if (!fix) return;
      try {
        const { solver, ex, driver } = fix;
        const scene = (solver as unknown as { scene: ClothScene }).scene;
        const n = scene.mesh.count;
        const x0 = Float64Array.from(scene.positions);
        // Never-contacted control: fresh reset at clear height.
        resetDeviceState(fix, Float64Array.from(x0));
        const ctlCount = await evalFrame(fix, x0, 0.05, 900);
        expect(ctlCount).toBe(0);
        const ctlStatus = (await solver.evaluateNewtonState(false, 901, DT)).status;
        let refKeys: string[] | null = null;
        let frames = 0;
        for (let f = 0; f < 6; f++) {
          const engaged = f % 2 === 0;
          const count = await evalFrame(fix, x0, engaged ? 0.0012 : 0.05, 910 + f);
          if (engaged) {
            expect(count, `${cls} frame ${f}: engaged frame lost contact`).toBe(n);
            const c = await readCounters(fix);
            expect(c.contactOverflow).toBe(0);
            const live = await readContactLive(fix, count, driver.c.cap);
            const keys = liveKeys(live);
            if (refKeys === null) refKeys = keys;
            else {
              const cmp = multisetEqual(refKeys, keys);
              expect(cmp.equal, `${cls} frame ${f}: engaged multiset drifted (class B)`).toBe(true);
            }
          } else {
            expect(count, `${cls} frame ${f}: clear frame shows contacts`).toBe(0);
            // Stale-tail guard proof: clear-frame energies match the control
            // that never held contacts (barrier must skip retired slots).
            const st = (await solver.evaluateNewtonState(false, 920 + f, DT)).status;
            expect(st.energy, `${cls} frame ${f}: stale tail leaks into energy (class B)`).toBe(ctlStatus.energy);
            expect(st.gradNorm, `${cls} frame ${f}: stale tail leaks into gradNorm (class B)`).toBe(ctlStatus.gradNorm);
          }
          frames++;
        }
      expect(frames).toBe(6);
      // eslint-disable-next-line no-console
      console.log(`[scale-frames] ${cls}/floor: 3 engaged frames @ ${n} contacts stable, 3 clear frames match control E=${ctlStatus.energy.toExponential(4)}`);
      void ex;
      } finally {
        fix.ex.destroy();
      }
    }, 1800000);

  it("10k/floor: floor-kind records toggle with height; background stays valid", async () => {
    // At 1.6 mm cells the flat sheet self-contacts coplanarly (~134k VT/EE at
    // ANY height), so no frame is truly empty. The hygiene gate becomes the
    // floor-kind toggle: n floor records engaged, 0 clear, with every live
    // record valid and everything finite across 6 alternations.
    const cls: ScaleClass = "10k";
      const fix = await requireDevice(() => buildScene("floor", cls), {
        contactCapacity: 262144, pairCapacity: PAIR_CAP[cls],
      });
      if (!fix) return;
      try {
        const { solver, driver } = fix;
        const scene = (solver as unknown as { scene: ClothScene }).scene;
        const n = scene.mesh.count;
        const x0 = Float64Array.from(scene.positions);
        resetDeviceState(fix, Float64Array.from(x0));
        const dHat = IN(solver).contactParamsNow().dHat;
        for (let f = 0; f < 6; f++) {
          const engaged = f % 2 === 0;
          const count = await evalFrame(fix, x0, engaged ? 0.0012 : 0.05, 930 + f);
          const c = await readCounters(fix);
          const live = await readContactLive(fix, count, driver.c.cap);
          let floorKind = 0;
          for (let s = 0; s < live.live; s++) {
            if (Math.round(live.N[s * 4 + 3]) === 2) floorKind++;
          }
          const chk = checkLiveRecords(live, dHat);
          const st = (await solver.evaluateNewtonState(false, 940 + f, DT)).status;
          // eslint-disable-next-line no-console
          console.log(`[scale-frames] ${cls}/floor frame ${f} ${engaged ? "engaged" : "clear"}: ` +
            `count=${count} floorKind=${floorKind} pairOv=${c.pairOverflow} contactOv=${c.contactOverflow} ` +
            `E=${st.energy.toExponential(4)} finite=${st.finite}`);
          expect(floorKind, `${cls} frame ${f}: floor-kind toggle wrong (class B stale hygiene)`).toBe(engaged ? n : 0);
          expect(chk.errors, `${cls} frame ${f}: invalid live records (class B)`).toEqual([]);
          expect(Number.isFinite(st.energy)).toBe(true);
          expect(st.finite).toBe(1);
        }
      } finally {
        fix.ex.destroy();
      }
    }, 1800000);

  it("1k/dense: rejected sequential trials never touch laggedN or position", async () => {
    const fix = await requireDevice(() => buildScene("dense", "1k"), {
      contactCapacity: BIG_CAP["1k"], pairCapacity: PAIR_CAP["1k"],
    });
    if (!fix) return;
    try {
      const { solver, driver } = fix;
      driver.cfg.pcgIters = 25;
      const scene = (solver as unknown as { scene: ClothScene }).scene;
      const x0 = Float64Array.from(scene.positions);
      resetDeviceState(fix, Float64Array.from(x0));
      solver.configureStep(DT);
      await solver.evaluateNewtonState(false, 950, DT);
      await driver.pcgSolve();
      const posBefore = await readF32(fix, "position");
      const lagBefore = await readF32(fix, "laggedN");
      // Four sequential trial evaluations, deliberately never accepted.
      for (let k = 0; k < 4; k++) {
        await driver.applyTrial(Math.pow(0.5, k));
        await solver.evaluateNewtonState(true, 960 + k, DT);
      }
      const posAfter = await readF32(fix, "position");
      const lagAfter = await readF32(fix, "laggedN");
      // eslint-disable-next-line no-console
      console.log(`[scale-frames] 1k/dense trials: |dPos|=${maxDiff(posBefore, posAfter).toExponential(2)} ` +
        `|dLag|=${maxDiff(lagBefore, lagAfter).toExponential(2)}`);
      expect(maxDiff(posBefore, posAfter), "unevaluated trials moved position (class B)").toBe(0);
      expect(maxDiff(lagBefore, lagAfter), "unevaluated trials mutated laggedN (class B)").toBe(0);
    } finally {
      fix.ex.destroy();
    }
  }, 1800000);

  it("1k/floor: laggedN from an accept survives a clear frame; re-engagement stays finite", async () => {
    const fix = await requireDevice(() => buildScene("floor", "1k"), {
      contactCapacity: BIG_CAP["1k"], pairCapacity: PAIR_CAP["1k"],
    });
    if (!fix) return;
    try {
      const { solver, ex, driver } = fix;
      driver.cfg.pcgIters = 25;
      const scene = (solver as unknown as { scene: ClothScene }).scene;
      const x0 = Float64Array.from(scene.positions);
      resetDeviceState(fix, Float64Array.from(x0));
      solver.configureStep(DT);
      const ev = await solver.evaluateNewtonState(false, 970, DT);
      expect(ev.contactCount).toBeGreaterThan(0);
      await driver.pcgSolve();
      // Accept trial 0 through the standard commit path (minimal mirror of
      // the batched-search commit: apply + sync-free rebuild + accept).
      await driver.applyTrial(1.0);
      ex.beginBatch("scale-commit-refresh");
      driver.rebuildTrialPasses(IN(solver).materialNow(), IN(solver).contactParamsNow());
      await ex.submitBatch(false);
      await driver.acceptTrial();
      const lagAccepted = await readF32(fix, "laggedN");
      // Clear frame: teleport away, evaluate (no commit anywhere).
      const clearCount = await evalFrame(fix, x0, 0.05, 972);
      expect(clearCount).toBe(0);
      const lagAfterClear = await readF32(fix, "laggedN");
      expect(maxDiff(lagAccepted, lagAfterClear), "clear evaluation touched laggedN (class B)").toBe(0);
      // Re-engage + accept again: stale normals must not blow up.
      const reCount = await evalFrame(fix, x0, 0.0012, 973);
      expect(reCount).toBeGreaterThan(0);
      await driver.pcgSolve();
      await driver.applyTrial(1.0);
      ex.beginBatch("scale-commit-refresh2");
      driver.rebuildTrialPasses(IN(solver).materialNow(), IN(solver).contactParamsNow());
      await ex.submitBatch(false);
      await driver.acceptTrial();
      const st = await solver.evaluateNewtonState(false, 974, DT);
      // eslint-disable-next-line no-console
      console.log(`[scale-frames] 1k/floor laggedN: accepted contacts=${ev.contactCount} re-engaged=${reCount} ` +
        `E=${st.status.energy.toExponential(4)} finite=${st.status.finite}`);
      expect(Number.isFinite(st.status.energy)).toBe(true);
      expect(st.status.finite).toBe(1);
    } finally {
      fix.ex.destroy();
    }
  }, 1800000);
});
