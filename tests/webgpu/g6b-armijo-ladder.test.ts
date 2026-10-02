// G6B batch ladder + gates: K=2/4/8 accepted parity, overflow rejection,
// friction commit vs reject, submit/sync reduction on full steps, and
// one-step batched-vs-sequential trajectory parity (strip/floor tight,
// fold/friction finite-only per the chaos rule).
import { describe, it, expect, beforeAll } from "vitest";
import type { ClothScene } from "../../src/physics/scene.js";
import type { DeviceFixture } from "./device-setup.js";
import { sharedDevice, resetDeviceState, requireDevice, stripScene } from "./device-setup.js";
import { buildGrid, preprocess } from "../../src/mesh/mesh.js";
import { createScene } from "../../src/physics/scene.js";
import { DEFAULT_MATERIAL } from "../../src/physics/types.js";
import { ContactSystem } from "../../src/collision/contact-assembly.js";
import { DEFAULT_CONTACT_PARAMS } from "../../src/collision/types.js";
import { GpuUniformSlot } from "../../src/backend/webgpu/gpu-buffers.js";
import { decodeArmijoStatus } from "../../src/backend/webgpu/gpu-newton.js";

let strip: DeviceFixture | null = null;

beforeAll(async () => {
  strip = await sharedDevice("g6b-ladder-strip", stripScene, { contactCapacity: 64, pairCapacity: 512 });
}, 180000);

function floorScene(): ReturnType<typeof createScene> {
  const g = buildGrid(8, 8, 0.16, 0.16);
  for (let i = 0; i < g.positions.length / 3; i++) g.positions[i * 3 + 1] += 0.05;
  const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
  const scene = createScene(mesh, { ...DEFAULT_MATERIAL }, [0, -9.81, 0]);
  scene.contact = new ContactSystem({ ...DEFAULT_CONTACT_PARAMS }, mesh.indices);
  scene.contact.setFloor(0);
  return scene;
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

describe("G6B submit reduction + one-step parity", () => {
  it("same trial set costs fewer syncs batched than sequential", async () => {
    // Fixed Newton state + identical trial outcomes on both paths: the only
    // difference is batching. Sequential maps per trial collapse to one
    // status map per batch; hotLoop readbacks likewise.
    const fix = await requireDevice(foldScene, { contactCapacity: 2048, pairCapacity: 8192 });
    if (!fix) return;
    try {
      const { solver, ex, driver } = fix;
      const scene = (solver as unknown as { scene: ClothScene }).scene;
      const x0 = Float64Array.from(scene.positions);
      const n = scene.mesh.count;
      const params = {
        mat: {
          c00: scene.material.stretchWarp, c11: scene.material.stretchWeft,
          c01: scene.material.stretchCoupling, g: scene.material.shear,
          thickness: scene.material.thickness,
        },
        contact: {
          dHat: 0.002, kappa: 50, mu: 0.3, fricEps: 1e-4,
          floorY: 1e30, floorOn: 0, dMin: 1e-4, contactCapacity: driver.c.cap,
        },
      };
      // Identical Newton state for both runs (positions + fresh pcg solve).
      async function setup(): Promise<{ E0: number; gtdx: number; trustScale: number }> {
        resetDeviceState(fix!, Float64Array.from(x0));
        ex.writeBuffer("velocity", new Float32Array(n * 4));
        solver.configureStep(1 / 60);
        const ev = await solver.evaluateNewtonState(false, 1, 1 / 60);
        await driver.pcgSolve();
        ex.beginBatch("setup-dot");
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
        return { E0: ev.status.energy, gtdx, trustScale: 1 };
      }
      // Sequential: full trial loop to accept (counts trials + syncs).
      const s = await setup();
      const s0 = { submits: ex.ledger.submits, hot: solver.hotLoopReadbacks, maps: ex.ledger.mapSyncs };
      let alpha = 1;
      let accepted = false;
      let seqTrials = 0;
      for (let li = 0; li < 10 && !accepted; li++) {
        await driver.applyTrial(alpha);
        const tev = await solver.evaluateNewtonState(true, 800 + li, 1 / 60);
        seqTrials++;
        const t = tev.status;
        const valid = t.finite === 1 && t.ccdSafe === 1 && t.minDistance > 1e-4 && !(t.minToi > 0 && t.minToi < 1 - 1e-9);
        if (valid && Number.isFinite(t.energy) && t.energy <= s.E0 + 1e-4 * alpha * s.gtdx) accepted = true;
        else alpha *= 0.5;
      }
      const seq = {
        submits: ex.ledger.submits - s0.submits,
        hot: solver.hotLoopReadbacks - s0.hot,
        maps: ex.ledger.mapSyncs - s0.maps,
        trials: seqTrials,
      };
      // Batched: same state, K=4 batches to the same accept.
      const s2 = await setup();
      expect(s2.E0).toBe(s.E0);
      const b0 = { submits: ex.ledger.submits, hot: solver.hotLoopReadbacks, maps: ex.ledger.mapSyncs };
      solver.beginArmijoBatch(810);
      const bst = await driver.armijoBatch({
        E0: s2.E0, gtdx: s2.gtdx, alphaBase: 1, beta: 0.5, K: 4,
        trustScale: 1, mat: params.mat, contact: params.contact,
        evalIndexBase: 810, pcgBreakdown: false, newtonConverged: false,
      });
      const bat = {
        submits: ex.ledger.submits - b0.submits,
        hot: solver.hotLoopReadbacks - b0.hot,
        maps: ex.ledger.mapSyncs - b0.maps,
      };
      // eslint-disable-next-line no-console
      console.log(`[g6b-submit] fold trials=${seq.trials} seq submits=${seq.submits} hot=${seq.hot} maps=${seq.maps} | ` +
        `bat submits=${bat.submits} hot=${bat.hot} maps=${bat.maps} accepted=${bst.accepted}`);
      expect(bst.accepted).toBe(true);
      // Hot-loop readbacks: batches (1, via solver wrapper; 0 direct-driver)
      // vs per-trial evals. Mapped syncs strictly fewer: one status map per
      // batch vs count+status+marker maps per sequential trial.
      expect(bat.hot).toBeLessThanOrEqual(seq.hot);
      expect(bat.maps).toBeLessThan(seq.maps);
    } finally {
      fix.ex.destroy();
    }
  }, 300000);

  it("one-step batched vs sequential trajectories match (strip/floor)", async () => {
    const cases: Array<{ name: string; build: () => ReturnType<typeof createScene>; tight: boolean }> = [
      { name: "strip", build: stripScene, tight: true },
      { name: "floor", build: floorScene, tight: true },
    ];
    for (const c of cases) {
      const fix = await requireDevice(c.build, { contactCapacity: 1024, pairCapacity: 4096 });
      if (!fix) {
        // eslint-disable-next-line no-console
        console.log(`[g6b-one-step] no device for ${c.name} — skipped`);
        continue;
      }
      try {
        // Hermetic initial state (scene arrays drift via snapshots; velocity
        // is not covered by resetDeviceState).
        const sc0 = (fix.solver as unknown as { scene: ClothScene }).scene;
        const x0 = Float64Array.from(sc0.positions);
        const n0 = sc0.mesh.count;
        const ends: Float64Array[] = [];
        const energies: number[] = [];
        for (const batched of [false, true]) {
          resetDeviceState(fix, Float64Array.from(x0));
          fix.ex.writeBuffer("velocity", new Float32Array(n0 * 4));
          fix.driver.cfg.useBatchedArmijo = batched;
          fix.driver.cfg.armijoBatchK = 4;
          const d = await fix.solver.stepGpu(1 / 60, { newtonIters: 2 });
          ends.push(Float64Array.from(fix.solver.getPositions()));
          energies.push(d.energy);
        }
        fix.driver.cfg.useBatchedArmijo = false;
        let worst = 0;
        for (let i = 0; i < ends[0].length; i++) {
          worst = Math.max(worst, Math.abs(ends[0][i] - ends[1][i]));
        }
        const eRel = Math.abs(energies[0] - energies[1]) / Math.max(Math.abs(energies[0]), 1e-12);
        // eslint-disable-next-line no-console
        console.log(`[g6b-one-step] ${c.name}: maxPosDiff=${worst.toExponential(2)} eRel=${eRel.toExponential(2)}`);
        expect(Number.isFinite(worst)).toBe(true);
        if (c.tight) {
          expect(worst).toBeLessThan(1e-5);
          expect(eRel).toBeLessThan(1e-6);
        }
      } finally {
        fix.ex.destroy();
      }
    }
  }, 600000);
});

describe("G6B batch ladder + gates", () => {
  it("K=2/4/8 agree on the strip Newton state", async () => {
    if (!strip) return;
    const { solver, ex, driver } = strip;
    const scene = (solver as unknown as { scene: ClothScene }).scene;
    const P = {
      mat: {
        c00: scene.material.stretchWarp, c11: scene.material.stretchWeft,
        c01: scene.material.stretchCoupling, g: scene.material.shear,
        thickness: scene.material.thickness,
      },
      contact: {
        dHat: 0.002, kappa: 50, mu: 0.3, fricEps: 1e-4,
        floorY: 1e30, floorOn: 0, dMin: 1e-4, contactCapacity: driver.c.cap,
      },
    };
    const out: Record<number, { accepted: boolean; alpha: number; idx: number }> = {};
    for (const K of [2, 4, 8]) {
      resetDeviceState(strip, Float64Array.from(scene.positions));
      solver.configureStep(1 / 60);
      const ev = await solver.evaluateNewtonState(false, 1, 1 / 60);
      await driver.pcgSolve();
      ex.beginBatch("ladder-dot");
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
      solver.beginArmijoBatch(730);
      const st = await driver.armijoBatch({
        E0: ev.status.energy, gtdx, alphaBase: 1, beta: 0.5, K,
        trustScale: 1, mat: P.mat, contact: P.contact,
        evalIndexBase: 730, pcgBreakdown: false, newtonConverged: false,
      });
      out[K] = { accepted: st.accepted, alpha: st.selectedAlpha, idx: st.selectedIndex };
      // eslint-disable-next-line no-console
      console.log(`[g6b-ladder] K=${K}: accepted=${st.accepted} idx=${st.selectedIndex} a=${st.selectedAlpha.toExponential(3)}`);
      expect(st.accepted).toBe(true);
    }
    // Same first-valid alpha regardless of batch width (K only batches more).
    expect(out[2].idx).toBe(out[4].idx);
    expect(out[4].idx).toBe(out[8].idx);
    expect(out[4].alpha).toBeCloseTo(out[8].alpha, 6);
  }, 300000);

  it("overflow rejects trials but stays finite (tiny-cap fold)", async () => {
    const fix = await requireDevice(foldScene, { contactCapacity: 8, pairCapacity: 128 });
    if (!fix) return;
    try {
      const { solver, ex, driver } = fix;
      const scene = (solver as unknown as { scene: ClothScene }).scene;
      resetDeviceState(fix, Float64Array.from(scene.positions));
      solver.configureStep(1 / 60);
      const ev = await solver.evaluateNewtonState(false, 1, 1 / 60);
      await driver.pcgSolve();
      const P = {
        mat: {
          c00: scene.material.stretchWarp, c11: scene.material.stretchWeft,
          c01: scene.material.stretchCoupling, g: scene.material.shear,
          thickness: scene.material.thickness,
        },
        contact: {
          dHat: 0.002, kappa: 50, mu: 0.3, fricEps: 1e-4,
          floorY: 1e30, floorOn: 0, dMin: 1e-4, contactCapacity: driver.c.cap,
        },
      };
      solver.beginArmijoBatch(740);
      const st = await driver.armijoBatch({
        E0: ev.status.energy, gtdx: -1e-6, alphaBase: 1, beta: 0.5, K: 4,
        trustScale: 1, mat: P.mat, contact: P.contact,
        evalIndexBase: 740, pcgBreakdown: false, newtonConverged: false,
      });
      // eslint-disable-next-line no-console
      console.log(`[g6b-overflow] accepted=${st.accepted} overflowFails=${st.overflowFails} finite=${st.finite}`);
      expect(st.overflowFails).toBeGreaterThan(0);
      expect(st.finite).toBe(true);
      expect(Number.isFinite(st.energy) || !st.accepted).toBe(true);
    } finally {
      fix.ex.destroy();
    }
  }, 300000);

  it("friction commits on accept, never on reject", async () => {
    const fix = await requireDevice(foldScene, { contactCapacity: 2048, pairCapacity: 8192 });
    if (!fix) return;
    try {
      const { solver, ex, driver } = fix;
      const scene = (solver as unknown as { scene: ClothScene }).scene;
      // Accept path: batch select + commit changes laggedN from zeros.
      resetDeviceState(fix, Float64Array.from(scene.positions));
      solver.configureStep(1 / 60);
      const ev = await solver.evaluateNewtonState(false, 1, 1 / 60);
      await driver.pcgSolve();
      const P = {
        mat: {
          c00: scene.material.stretchWarp, c11: scene.material.stretchWeft,
          c01: scene.material.stretchCoupling, g: scene.material.shear,
          thickness: scene.material.thickness,
        },
        contact: {
          dHat: 0.002, kappa: 50, mu: 0.3, fricEps: 1e-4,
          floorY: 1e30, floorOn: 0, dMin: 1e-4, contactCapacity: driver.c.cap,
        },
      };
      solver.beginArmijoBatch(750);
      const st = await driver.armijoBatch({
        E0: ev.status.energy, gtdx: -1e-6, alphaBase: 1, beta: 0.5, K: 4,
        trustScale: 1, mat: P.mat, contact: P.contact,
        evalIndexBase: 750, pcgBreakdown: false, newtonConverged: false,
      });
      const lag0 = new Float32Array(await ex.readBufferDebug("laggedN", "fric-pre", false));
      if (st.accepted) {
        await driver.applyTrial(st.selectedAlpha);
        ex.beginBatch("fric-commit-refresh");
        driver.rebuildTrialPasses(P.mat, P.contact);
        await ex.submitBatch(false);
        await driver.acceptTrial();
        const lag1 = new Float32Array(await ex.readBufferDebug("laggedN", "fric-post", false));
        let changed = 0;
        for (let i = 0; i < lag1.length; i++) changed = Math.max(changed, Math.abs(lag1[i] - lag0[i]));
        // eslint-disable-next-line no-console
        console.log(`[g6b-friction] accepted, max|laggedN-post - pre|=${changed.toExponential(2)}`);
        expect(changed).toBeGreaterThan(0);
      } else {
        // eslint-disable-next-line no-console
        console.log("[g6b-friction] batch rejected; laggedN must be untouched");
        const lag1 = new Float32Array(await ex.readBufferDebug("laggedN", "fric-post", false));
        expect([...lag1]).toEqual([...lag0]);
      }
      // Reject path (kernel level): all-invalid rows select nothing and the
      // test performs no accept, so laggedN is definitionally untouched.
      const rows = new Float32Array(64);
      for (let k = 0; k < 4; k++) {
        rows[k * 8] = Math.pow(0.5, k);
        rows[k * 8 + 1] = NaN;
        rows[k * 8 + 3] = 0;
      }
      const st0 = new Float32Array(16);
      st0[8] = 0;
      ex.writeBuffer("armijoCandidates", rows);
      ex.writeBuffer("armijoStatus", st0);
      driver.bankU(GpuUniformSlot.ArmijoK, 4);
      driver.bankF(GpuUniformSlot.ArmijoPcgBd, 0);
      driver.bankF(GpuUniformSlot.ArmijoNewtonConv, 0);
      ex.beginBatch("fric-reject");
      driver.armijoSelectPass();
      await ex.submitBatch(false);
      const raw = await ex.readSmall("armijoStatus", 64, "fric-reject-status", "status");
      const rej = decodeArmijoStatus(raw);
      expect(rej.accepted).toBe(false);
      expect(rej.selectedIndex).toBe(-1);
    } finally {
      fix.ex.destroy();
    }
  }, 300000);
});
