// MIMO scale-contact stress: Armijo/Newton behavior at 1k (primary) with
// 10k/50k spots. Covers per-trial isolation (batch rows vs sequential
// trials), reject-all state preservation (laggedN + position bitwise),
// accept-then-reject round-boundary sync (S1 at scale), next-Newton-state
// consistency, pins-exact, and NaN/Inf scans.
//
// Class discipline: ordering differences are class A (never asserted);
// counter/validity/state divergences are class B.
import { describe, it, expect } from "vitest";
import { GpuUniformSlot } from "../../../src/backend/webgpu/gpu-buffers.js";
import type { ClothScene } from "../../../src/physics/scene.js";
import { requireDevice, resetDeviceState, type DeviceFixture } from "../../webgpu/device-setup.js";
import {
  BIG_CAP, GRID, IN, PAIR_CAP, REPS,
  buildScene, checkLiveRecords, findNonFinite, checkPinsExact, maxDiff,
  readContactLive, readCounters, readF32, readU32,
  type ScaleClass, type SceneName,
} from "./scale-helpers.js";

const DT = 1 / 60;

type NewtonPrep = {
  fix: DeviceFixture; E0: number; gtdx: number; trustScale: number; dMin: number;
};

/** E0 eval + PCG solve + device-side gtdx/trust (g6c-newton-parity pattern). */
async function prepareNewton(fix: DeviceFixture, evalIndex: number): Promise<NewtonPrep> {
  const { solver, ex, driver } = fix;
  solver.configureStep(DT);
  const ev = await solver.evaluateNewtonState(false, evalIndex, DT);
  const E0 = ev.status.energy;
  await driver.pcgSolve();
  ex.beginBatch("scale-dot");
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
  ex.beginBatch("scale-trust");
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
  return { fix, E0, gtdx, trustScale, dMin: IN(solver).contactParamsNow().dMin };
}

function seedNewtonControl(fix: DeviceFixture, E0: number): void {
  const { ex, driver } = fix;
  ex.writeBuffer("e0Store", new Float32Array([E0, 0, 0, 0]));
  driver.bankF(GpuUniformSlot.NewtonTol, 1e-5);
  driver.bankF(GpuUniformSlot.NewtonTrust, 0.002);
  ex.writeBuffer("newtonCtl", new Float32Array(16));
  ex.writeBuffer("newtonStatus", new Float32Array(20));
}

function newtonBeta(fix: DeviceFixture): number {
  const rawMt = (fix.solver as unknown as { scene: { material: Record<string, number> } }).scene.material;
  return Math.max(rawMt.stretchWarp, rawMt.stretchWeft, rawMt.shear) * rawMt.thickness * 0.1 + 1e-6;
}

describe("MIMO Armijo trial isolation at scale", () => {
  for (const [sceneName, cap] of [["fold", 131072], ["headon", 8192]] as Array<[SceneName, number]>) {
    it(`1k/${sceneName}: batch candidate rows match sequential trials`, async () => {
      const fix = await requireDevice(() => buildScene(sceneName, "1k"), {
        contactCapacity: cap, pairCapacity: PAIR_CAP["1k"],
      });
      if (!fix) return;
      try {
        const { solver, ex, driver } = fix;
        driver.cfg.pcgIters = 25;
        const scene = (solver as unknown as { scene: ClothScene }).scene;
        const x0 = Float64Array.from(scene.positions);
        resetDeviceState(fix, Float64Array.from(x0));
        const prep = await prepareNewton(fix, 200);
        solver.beginArmijoBatch(201);
        const K = 8;
        const st = await driver.armijoBatch({
          E0: prep.E0, gtdx: prep.gtdx, alphaBase: 1, beta: 0.5, K,
          trustScale: prep.trustScale,
          mat: IN(solver).materialNow(), contact: IN(solver).contactParamsNow(),
          evalIndexBase: 201, pcgBreakdown: false, newtonConverged: false,
        });
        // Comparability preconditions: under pair or contact overflow the
        // kept subsets race, so row-vs-trial equality is meaningless (class
        // A). Caps here are sized to fit; any overflow fails loudly here.
        const preIso = await readCounters(fix);
        expect(preIso.contactOverflow, "isolation: contact overflow breaks comparability").toBe(0);
        expect(preIso.pairOverflow, "isolation: pair overflow breaks comparability").toBe(0);
        const cand = new Float32Array(await ex.readBufferDebug("armijoCandidates", "scale-cand", false));
        const row = (k: number): number[] => Array.from(cand.slice(k * 8, k * 8 + 8));
        // Sequential re-evaluation of every candidate (no early break).
        // The validity gate mirrors the SEQUENTIAL path exactly (finite +
        // CCD + barrier); overflow is deliberately NOT gated here — the
        // batched record is GPU-stricter than sequential/CPU by documented
        // design (conservative), so the two verdicts differ by design under
        // overflow. Caps here are sized to fit, with preconditions below.
        const f = Math.fround;
        const seqOk: boolean[] = [];
        for (let k = 0; k < K; k++) {
          const rawAlpha = Math.pow(0.5, k);
          const alphaEff = f(f(rawAlpha) * f(prep.trustScale));
          await driver.applyTrial(alphaEff);
          const tev = await solver.evaluateNewtonState(true, 300 + k, DT);
          const s = tev.status;
          const valid = s.finite === 1 && s.ccdSafe === 1 &&
            s.minDistance > prep.dMin && !(s.minToi > 0 && s.minToi < 1 - 1e-9);
          const armijo = valid && Number.isFinite(s.energy) &&
            s.energy <= prep.E0 + 1e-4 * alphaEff * prep.gtdx;
          seqOk.push(valid && armijo);
          const r = row(k);
          expect(r[0], `1k/${sceneName} trial ${k}: batch alpha != sequential effective alpha (class B)`).toBe(alphaEff);
          expect(r[1], `1k/${sceneName} trial ${k}: batch energy != sequential energy (class B stale records?)`).toBe(s.energy);
          expect(r[3] > 0.5, `1k/${sceneName} trial ${k}: batch verdict != sequential verdict (class B)`).toBe(seqOk[k]);
          // eslint-disable-next-line no-console
          console.log(`[scale-newton] 1k/${sceneName} trial ${k}: batch ok=${r[0] > 0 && r[3] > 0.5} ` +
            `seq ok=${seqOk[k]} E=${s.energy.toExponential(4)} minDist=${s.minDistance.toExponential(3)}`);
        }
        const firstSeq = seqOk.findIndex(Boolean);
        if (st.accepted) {
          expect(st.selectedIndex, `1k/${sceneName}: batch selected ${st.selectedIndex} but first sequential ok is ${firstSeq} (class B)`).toBe(firstSeq);
        } else {
          expect(firstSeq).toBe(-1);
        }
        // eslint-disable-next-line no-console
        console.log(`[scale-newton] 1k/${sceneName}: batch accepted=${st.accepted} idx=${st.selectedIndex} firstSeqOk=${firstSeq}`);
      } finally {
        fix.ex.destroy();
      }
    }, 1800000);
  }
});

describe("MIMO reject-all preservation at scale", () => {
  it("1k/fold tiny cap: failed round leaves position+laggedN bitwise", async () => {
    const fix = await requireDevice(() => buildScene("fold", "1k"), {
      contactCapacity: 128, pairCapacity: PAIR_CAP["1k"],
    });
    if (!fix) return;
    try {
      const { solver, ex, driver } = fix;
      driver.cfg.pcgIters = 25;
      const scene = (solver as unknown as { scene: ClothScene }).scene;
      const x0 = Float64Array.from(scene.positions);
      resetDeviceState(fix, Float64Array.from(x0));
      const prep = await prepareNewton(fix, 400);
      seedNewtonControl(fix, prep.E0);
      const posBefore = await readF32(fix, "position");
      const lagBefore = await readF32(fix, "laggedN");
      const st = await driver.newtonRound({
        round: 0, beta: newtonBeta(fix),
        mat: IN(solver).materialNow(), contact: IN(solver).contactParamsNow(),
        batchKs: [4, 4, 2], evalIndexBase: 400,
      });
      // eslint-disable-next-line no-console
      console.log(`[scale-newton] 1k/fold tiny: accepted=${st.armijoAccepted} failure=${st.failure} energy=${st.energy}`);
      expect(st.armijoAccepted).toBe(false);
      expect(st.failure).toBe(true);
      expect(Number.isFinite(st.energy)).toBe(true);
      const posAfter = await readF32(fix, "position");
      const lagAfter = await readF32(fix, "laggedN");
      expect(maxDiff(posBefore, posAfter), "rejected round moved position (class B)").toBe(0);
      expect(maxDiff(lagBefore, lagAfter), "rejected round mutated laggedN (class B)").toBe(0);
      // The step still completes: velocities + snapshot stay finite.
      const d = await solver.stepGpu(DT, { newtonIters: 1 });
      expect(Number.isFinite(d.energy)).toBe(true);
      expect(d.finite).toBe(1);
    } finally {
      fix.ex.destroy();
    }
  }, 1800000);
});

describe("MIMO accept-then-reject sync + next-round consistency at scale", () => {
  it("1k/floor: accepted round buffers describe position; next round coherent", async () => {
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
      const prep = await prepareNewton(fix, 500);
      const laggedStart = await readF32(fix, "laggedN");
      seedNewtonControl(fix, prep.E0);
      const st0 = await driver.newtonRound({
        round: 0, beta: newtonBeta(fix),
        mat: IN(solver).materialNow(), contact: IN(solver).contactParamsNow(),
        batchKs: [4, 4, 2], evalIndexBase: 500,
      });
      expect(st0.armijoAccepted).toBe(true);
      const preC = await readCounters(fix);
      expect(preC.pairOverflow, "1k/floor: pair overflow breaks exact-S1 preconditions").toBe(0);
      expect(preC.contactOverflow, "1k/floor: contact overflow breaks exact-S1 preconditions").toBe(0);
      // S1 protocol: snapshot post-commit laggedN (production state to
      // restore afterwards), restore round-start laggedN (what the commit
      // rebuild baked friction with), rebuild from position.
      const laggedPost = await readF32(fix, "laggedN");
      const countLeft = (await (async (): Promise<number> => {
        const raw = await ex.readSmall("contactCount", 16, "scale-cc", "scalar");
        return new Uint32Array(raw)[0];
      })());
      const gradLeft = await readF32(fix, "gradient");
      ex.writeBuffer("laggedN", laggedStart);
      driver.zeroContactCounters();
      ex.beginBatch("scale-rederive");
      driver.broadphasePasses(0.002, false);
      driver.contactPasses(IN(solver).contactParamsNow(), "position");
      driver.femPasses(IN(solver).materialNow(), false);
      driver.contactDiagPass();
      driver.rhsAt("position");
      await ex.submitBatch(false);
      const countRef = (await (async (): Promise<number> => {
        const raw = await ex.readSmall("contactCount", 16, "scale-cc-ref", "scalar");
        return new Uint32Array(raw)[0];
      })());
      const gradRef = await readF32(fix, "gradient");
      const dGrad = maxDiff(gradLeft, gradRef);
      let gScale = 0;
      for (let i = 0; i < gradRef.length; i++) gScale = Math.max(gScale, Math.abs(gradRef[i]));
      // eslint-disable-next-line no-console
      console.log(`[scale-newton] 1k/floor S1: count ${countLeft} vs ${countRef} |grad|=${dGrad.toExponential(3)} scale=${gScale.toExponential(3)}`);
      expect(countLeft).toBe(countRef);
      expect(dGrad).toBeLessThan(1e-5);
      // Next-round coherence: restore the production post-commit laggedN,
      // take a fresh Xk evaluation, then run round 1. Its Xk refresh (lane 9)
      // must reproduce the fresh evaluation: both describe the accepted
      // position under the same laggedN, so round 1 decides on truth.
      ex.writeBuffer("laggedN", laggedPost);
      solver.configureStep(DT);
      const g0 = await solver.evaluateNewtonState(false, 501, DT);
      seedNewtonControl(fix, g0.status.energy);
      const st1 = await driver.newtonRound({
        round: 1, beta: newtonBeta(fix),
        mat: IN(solver).materialNow(), contact: IN(solver).contactParamsNow(),
        batchKs: [4, 4, 2], evalIndexBase: 600,
      });
      // eslint-disable-next-line no-console
      console.log(`[scale-newton] 1k/floor round1: accepted=${st1.armijoAccepted} converged=${st1.converged} ` +
        `failure=${st1.failure} energy=${st1.energy} gradNorm=${st1.gradNorm}`);
      expect(Number.isFinite(st1.energy)).toBe(true);
      expect(Number.isFinite(st1.gradNorm)).toBe(true);
      // Round 1's Xk refresh (lane 9) must match the fresh evaluation: both
      // describe the accepted position, so the next round decides on truth.
      expect(st1.gradNorm).toBe(g0.status.gradNorm);
    } finally {
      fix.ex.destroy();
    }
  }, 1800000);
});

describe("MIMO pins-exact + finiteness at scale", () => {
  it("10k/pinned-floor: full gpu step keeps pins bitwise + everything finite", async () => {
    const cls: ScaleClass = "10k";
    const fix = await requireDevice(() => buildScene("pinned-floor", cls), {
      contactCapacity: BIG_CAP[cls], pairCapacity: PAIR_CAP[cls],
    });
    if (!fix) return;
    try {
      const { solver, ex, driver } = fix;
      driver.cfg.useGpuNewtonControl = true;
      driver.cfg.pcgIters = 25;
      const scene = (solver as unknown as { scene: ClothScene }).scene;
      const n = scene.mesh.count;
      const x0 = Float64Array.from(scene.positions);
      resetDeviceState(fix, Float64Array.from(x0));
      ex.writeBuffer("velocity", new Float32Array(n * 4));
      const d = await solver.stepGpu(DT, { newtonIters: 2 });
      expect(Number.isFinite(d.energy)).toBe(true);
      expect(d.finite).toBe(1);
      const pos = await readF32(fix, "position");
      const mask = await readU32(fix, "pinMask");
      const pinPos = await readF32(fix, "pinPos");
      const pins = checkPinsExact(pos, mask, pinPos, n);
      const grad = await readF32(fix, "gradient");
      const rhs = await readF32(fix, "rhs");
      const lag = await readF32(fix, "laggedN");
      const badGrad = findNonFinite(grad, "gradient");
      const badRhs = findNonFinite(rhs, "rhs");
      const badLag = findNonFinite(lag, "laggedN");
      // eslint-disable-next-line no-console
      console.log(`[scale-newton] 10k/pinned-floor: pinned=${pins.pinned} worstPin=${pins.worst.toExponential(2)} ` +
        `E=${d.energy.toExponential(4)} badGrad=${badGrad.length} badRhs=${badRhs.length} badLag=${badLag.length}`);
      expect(pins.pinned).toBeGreaterThan(0);
      expect(pins.worst).toBe(0);
      expect(badGrad).toEqual([]);
      expect(badRhs).toEqual([]);
      expect(badLag).toEqual([]);
      driver.cfg.useGpuNewtonControl = false;
    } finally {
      fix.ex.destroy();
    }
  }, 1800000);

  it("1k/dense: post-accept buffers fully finite (friction included)", async () => {
    const fix = await requireDevice(() => buildScene("dense", "1k"), {
      contactCapacity: 262144, pairCapacity: PAIR_CAP["1k"],
    });
    if (!fix) return;
    try {
      const { solver, ex, driver } = fix;
      driver.cfg.pcgIters = 25;
      const scene = (solver as unknown as { scene: ClothScene }).scene;
      const x0 = Float64Array.from(scene.positions);
      resetDeviceState(fix, Float64Array.from(x0));
      const prep = await prepareNewton(fix, 700);
      seedNewtonControl(fix, prep.E0);
      const st = await driver.newtonRound({
        round: 0, beta: newtonBeta(fix),
        mat: IN(solver).materialNow(), contact: IN(solver).contactParamsNow(),
        batchKs: [4, 4, 2], evalIndexBase: 700,
      });
      // eslint-disable-next-line no-console
      console.log(`[scale-newton] 1k/dense round: accepted=${st.armijoAccepted} failure=${st.failure} E=${st.energy}`);
      const c = await readCounters(fix);
      const live = await readContactLive(fix, c.contactCount, driver.c.cap);
      const chk = checkLiveRecords(live, IN(solver).contactParamsNow().dHat);
      expect(chk.errors).toEqual([]);
      const grad = await readF32(fix, "gradient");
      const sd = await readF32(fix, "searchDirection");
      const ce = await readF32(fix, "contactEnergy");
      expect(findNonFinite(grad, "gradient")).toEqual([]);
      expect(findNonFinite(sd, "searchDirection")).toEqual([]);
      expect(findNonFinite(ce, "contactEnergy")).toEqual([]);
      // Overflow makes the round fail closed (trial invalid), never corrupt.
      if (c.contactOverflow === 1) expect(st.armijoAccepted).toBe(false);
    } finally {
      fix.ex.destroy();
    }
  }, 1800000);
});

describe("MIMO 10k/50k Newton spots", () => {
  it("10k/floor: accepted round commits exactly; next round stays finite", async () => {
    // 10k/floor pair volume (~818k measured) always overflows memory-safe
    // pair caps, so exact set equality is meaningless here (class A). The
    // truncation-immune gates: commit lands position==xTrial bitwise (the
    // crux of S1), counters stay exact, kept records valid, laggedN tail
    // untouched, everything finite, next round decides on coherent state.
    const fix = await requireDevice(() => buildScene("floor", "10k"), {
      contactCapacity: 262144, pairCapacity: PAIR_CAP["10k"],
    });
    if (!fix) return;
    try {
      const { solver, ex, driver } = fix;
      driver.cfg.pcgIters = 20;
      const scene = (solver as unknown as { scene: ClothScene }).scene;
      const x0 = Float64Array.from(scene.positions);
      resetDeviceState(fix, Float64Array.from(x0));
      const prep = await prepareNewton(fix, 800);
      const laggedPre = await readF32(fix, "laggedN");
      seedNewtonControl(fix, prep.E0);
      const st = await driver.newtonRound({
        round: 0, beta: newtonBeta(fix),
        mat: IN(solver).materialNow(), contact: IN(solver).contactParamsNow(),
        batchKs: [4, 4, 2], evalIndexBase: 800,
      });
      const c = await readCounters(fix);
      // eslint-disable-next-line no-console
      console.log(`[scale-newton] 10k/floor round: accepted=${st.armijoAccepted} alpha=${st.selectedAlpha} ` +
        `E=${st.energy} count=${c.contactCount} pairCount=${c.pairCount} ` +
        `pairOv=${c.pairOverflow} contactOv=${c.contactOverflow}`);
      expect(st.armijoAccepted).toBe(true);
      expect(Number.isFinite(st.energy)).toBe(true);
      // Counter exactness under saturation (race-free atomics).
      expect(c.pairScanned).toBe(c.pairCount);
      expect(c.contactCount <= c.contactScanned).toBe(true);
      expect(c.pairOverflow).toBe(c.pairCount > PAIR_CAP["10k"] ? 1 : 0);
      expect(c.contactOverflow).toBe(c.contactCount > 262144 ? 1 : 0);
      // Commit landed exactly: position IS the accepted trial, bitwise.
      const posAfter = await readF32(fix, "position");
      const trialAfter = await readF32(fix, "xTrial");
      expect(maxDiff(posAfter, trialAfter), "commit did not land xTrial exactly (class B)").toBe(0);
      // Kept records all valid (truncated or not, nothing corrupt is consumed).
      const live = await readContactLive(fix, c.contactCount, driver.c.cap);
      const chk = checkLiveRecords(live, IN(solver).contactParamsNow().dHat);
      expect(chk.errors).toEqual([]);
      expect(findNonFinite(await readF32(fix, "gradient"), "gradient")).toEqual([]);
      // laggedN discipline: slots beyond the live set untouched by the accept.
      const laggedPost = await readF32(fix, "laggedN");
      let tailDiff = 0;
      const lagCap = laggedPost.length / 4;
      for (let i = live.live; i < lagCap; i++) {
        for (let k = 0; k < 4; k++) {
          tailDiff = Math.max(tailDiff, Math.abs(laggedPost[i * 4 + k] - laggedPre[i * 4 + k]));
        }
      }
      expect(tailDiff, "accept touched laggedN beyond the live set (class B)").toBe(0);
      // Next round decides on coherent state and stays finite.
      seedNewtonControl(fix, st.energy);
      const st1 = await driver.newtonRound({
        round: 1, beta: newtonBeta(fix),
        mat: IN(solver).materialNow(), contact: IN(solver).contactParamsNow(),
        batchKs: [4, 4, 2], evalIndexBase: 810,
      });
      // eslint-disable-next-line no-console
      console.log(`[scale-newton] 10k/floor round1: accepted=${st1.armijoAccepted} converged=${st1.converged} ` +
        `failure=${st1.failure} E=${st1.energy}`);
      expect(Number.isFinite(st1.energy)).toBe(true);
      expect(Number.isFinite(st1.gradNorm)).toBe(true);
    } finally {
      fix.ex.destroy();
    }
  }, 1800000);

  it("50k/floor: evaluate + one round stays finite with coherent counts", async () => {
    const fix = await requireDevice(() => buildScene("floor", "50k"), {
      contactCapacity: BIG_CAP["50k"], pairCapacity: PAIR_CAP["50k"],
    });
    if (!fix) return;
    try {
      const { solver, ex, driver } = fix;
      driver.cfg.pcgIters = 20;
      const scene = (solver as unknown as { scene: ClothScene }).scene;
      const x0 = Float64Array.from(scene.positions);
      resetDeviceState(fix, Float64Array.from(x0));
      const prep = await prepareNewton(fix, 900);
      seedNewtonControl(fix, prep.E0);
      const st = await driver.newtonRound({
        round: 0, beta: newtonBeta(fix),
        mat: IN(solver).materialNow(), contact: IN(solver).contactParamsNow(),
        batchKs: [4, 4, 2], evalIndexBase: 900,
      });
      const c = await readCounters(fix);
      // eslint-disable-next-line no-console
      console.log(`[scale-newton] 50k/floor round: accepted=${st.armijoAccepted} E=${st.energy} ` +
        `count=${c.contactCount} overflow=${c.contactOverflow} scanned=${c.contactScanned}`);
      expect(Number.isFinite(st.energy)).toBe(true);
      expect(c.contactOverflow).toBe(c.contactCount > driver.c.cap ? 1 : 0);
      expect(c.contactCount <= c.contactScanned).toBe(true);
      expect(findNonFinite(await readF32(fix, "gradient"), "gradient")).toEqual([]);
    } finally {
      fix.ex.destroy();
    }
  }, 1800000);
});
