// G6B fixed-state CPU-vs-GPU Armijo parity (THE gate): identical x/records,
// Newton direction, E0, gtdx through the CPU-sequential trial path and the
// GPU batch path; exact comparison of selected alpha/index, merit, validity,
// per-candidate rows, and sync count (one status readback per batch).
import { describe, it, expect, beforeAll } from "vitest";
import type { ClothScene } from "../../src/physics/scene.js";
import type { SolverStatus } from "../../src/backend/webgpu/gpu-buffers.js";
import type { DeviceFixture } from "./device-setup.js";
import { sharedDevice, resetDeviceState, requireDevice, stripScene } from "./device-setup.js";
import { buildGrid, preprocess } from "../../src/mesh/mesh.js";
import { createScene } from "../../src/physics/scene.js";
import { DEFAULT_MATERIAL } from "../../src/physics/types.js";
import { ContactSystem } from "../../src/collision/contact-assembly.js";
import { DEFAULT_CONTACT_PARAMS } from "../../src/collision/types.js";
import { GpuUniformSlot } from "../../src/backend/webgpu/gpu-buffers.js";
import { decodeArmijoStatus, type ArmijoBatchStatus } from "../../src/backend/webgpu/gpu-newton.js";

let strip: DeviceFixture | null = null;

beforeAll(async () => {
  strip = await sharedDevice("g6b-parity-strip", stripScene, { contactCapacity: 64, pairCapacity: 512 });
}, 180000);

interface NewtonState {
  E0: number;
  gtdx: number;
  trustScale: number;
  dMin: number;
}

function sceneParams(scene: ClothScene, cap: number) {
  const cp = scene.contact?.params;
  const floorY = scene.contact?.floorY;
  return {
    mat: {
      c00: scene.material.stretchWarp, c11: scene.material.stretchWeft,
      c01: scene.material.stretchCoupling, g: scene.material.shear,
      thickness: scene.material.thickness,
    },
    contact: {
      dHat: cp?.dHatM ?? 0.002,
      kappa: cp?.kappaJ ?? 50,
      mu: cp?.frictionMu ?? 0.3,
      fricEps: cp?.frictionEpsM ?? 1e-4,
      floorY: floorY ?? 1e30,
      floorOn: floorY !== null && floorY !== undefined ? 1 : 0,
      dMin: cp?.dMinM ?? 1e-4,
      contactCapacity: cap,
    },
    dMin: cp?.dMinM ?? 1e-4,
  };
}

/** Replicate stepGpuDevice's Newton prologue through the trust scale. */
async function newtonPrologue(fix: DeviceFixture): Promise<NewtonState> {
  const { solver, ex, driver } = fix;
  solver.configureStep(1 / 60);
  const ev = await solver.evaluateNewtonState(false, 1, 1 / 60);
  const E0 = ev.status.energy;
  await driver.pcgSolve();
  // gtdx = rhs . searchDirection (same dot the solver uses)
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
  // trust scale via max|dx|
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
  const scene = (solver as unknown as { scene: ClothScene }).scene;
  return { E0, gtdx, trustScale, dMin: scene.contact?.params.dMinM ?? 1e-4 };
}

interface CpuTrial {
  li: number;
  alphaEff: number;
  accepted: boolean;
  energy: number;
  minDistance: number;
  minToi: number;
  finite: boolean;
  ccdSafe: boolean;
}

/** CPU-sequential Armijo mirror (exact copy of the stepGpuDevice li-loop). */
async function cpuSequential(
  fix: DeviceFixture, st: NewtonState, budget = 10,
): Promise<{ trials: CpuTrial[]; acceptedIndex: number; acceptedAlpha: number }> {
  const { solver, driver } = fix;
  const trials: CpuTrial[] = [];
  let alpha = 1;
  let acceptedIndex = -1;
  let acceptedAlpha = NaN;
  for (let li = 0; li < budget; li++) {
    await driver.applyTrial(alpha * st.trustScale);
    const tev = await solver.evaluateNewtonState(true, 700 + li, 1 / 60);
    const s: SolverStatus = tev.status;
    const valid =
      s.finite === 1 && s.ccdSafe === 1 &&
      s.minDistance > st.dMin &&
      !(s.minToi > 0 && s.minToi < 1 - 1e-9);
    const trial: CpuTrial = {
      li, alphaEff: alpha * st.trustScale, accepted: false,
      energy: s.energy, minDistance: s.minDistance, minToi: s.minToi,
      finite: s.finite === 1, ccdSafe: s.ccdSafe === 1,
    };
    if (!valid || !Number.isFinite(s.energy)) {
      trials.push(trial);
      alpha *= 0.5;
      continue;
    }
    if (s.energy <= st.E0 + 1e-4 * alpha * st.trustScale * st.gtdx) {
      trial.accepted = true;
      trials.push(trial);
      acceptedIndex = li;
      acceptedAlpha = alpha * st.trustScale;
      break;
    }
    trials.push(trial);
    alpha *= 0.5;
  }
  return { trials, acceptedIndex, acceptedAlpha };
}

/** Fold scene builder (dense self-contact from step 0). */
function buildFoldScene() {
  const w = 0.12;
  const g = buildGrid(6, 6, w, w);
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

describe("G6B fixed-state Armijo parity", () => {
  it("strip: batch selects the identical alpha with one sync", async () => {
    if (!strip) return;
    const { solver, ex, driver } = strip;
    const scene = (solver as unknown as { scene: ClothScene }).scene;
    resetDeviceState(strip, Float64Array.from(scene.positions));
    const st = await newtonPrologue(strip);
    const cpu = await cpuSequential(strip, st, 10);
    // GPU batches covering the same budget (K=4: 4+4+2)
    const P = sceneParams(scene, driver.c.cap);
    const batchAlphas: number[] = [];
    let acceptedIndex = -1;
    let acceptedAlpha = NaN;
    let alphaBase = 1;
    let trialsUsed = 0;
    let batches = 0;
    const maps0 = ex.ledger.mapSyncs;
    const candidateRows: number[][] = [];
    while (trialsUsed < 10) {
      const K = Math.min(4, 10 - trialsUsed);
      solver.beginArmijoBatch(700 + trialsUsed);
      const bst = await driver.armijoBatch({
        E0: st.E0, gtdx: st.gtdx, alphaBase, beta: 0.5, K,
        trustScale: st.trustScale, mat: P.mat, contact: P.contact,
        evalIndexBase: 700 + trialsUsed,
        pcgBreakdown: false, newtonConverged: false,
      });
      batches++;
      for (let j = 0; j < K; j++) batchAlphas.push(alphaBase * Math.pow(0.5, j) * st.trustScale);
      trialsUsed += bst.trialsEvaluated;
      if (bst.accepted) {
        acceptedIndex = trialsUsed - K + bst.selectedIndex;
        acceptedAlpha = bst.selectedAlpha;
        break;
      }
      alphaBase *= Math.pow(0.5, bst.trialsEvaluated || K);
    }
    const mapsUsed = ex.ledger.mapSyncs - maps0;
    // per-candidate rows for row-level comparison
    const candRaw = await ex.readBufferDebug("armijoCandidates", "parity-candidates", false);
    const cand = new Float32Array(candRaw);
    for (let j = 0; j < trialsUsed; j++) candidateRows.push([...cand.subarray(j * 8, j * 8 + 8)]);
    // eslint-disable-next-line no-console
    console.log(`[g6b-parity] strip: cpu li=${cpu.acceptedIndex} a=${cpu.acceptedAlpha.toExponential(4)} ` +
      `gpu li=${acceptedIndex} a=${acceptedAlpha.toExponential(4)} batches=${batches} maps=${mapsUsed}`);
    // THE gate: identical selection (rows carry effective alpha: raw alphas
    // buffer times the GPU-side trust fold, exactly like alpha*trustScale)...
    expect(acceptedIndex).toBe(cpu.acceptedIndex);
    if (cpu.acceptedIndex >= 0) {
      expect(acceptedAlpha).toBeCloseTo(cpu.acceptedAlpha, 6);
    }
    // ...identical per-trial alphas, merit and validity on the shared prefix
    // (the GPU batch always evaluates all K rows; the CPU stops at accept —
    // rows past the CPU horizon are computed but unused by selection).
    expect(batchAlphas.length).toBeGreaterThanOrEqual(cpu.trials.length);
    for (let j = 0; j < cpu.trials.length; j++) {
      const t = cpu.trials[j];
      const row = candidateRows[j];
      expect(row[0]).toBeCloseTo(t.alphaEff, 6);
      expect(row[1]).toBeCloseTo(t.energy, 5);
      expect(row[3] > 0.5).toBe(t.accepted);
    }
    // ...and exactly one status sync per batch.
    expect(mapsUsed).toBe(batches);
  }, 300000);

  it("strip: commit via batch path matches sequential commit bit-near", async () => {
    if (!strip) return;
    const { solver, ex, driver } = strip;
    const scene = (solver as unknown as { scene: ClothScene }).scene;
    const n = scene.mesh.count;
    // Path A (sequential, existing methods): accept at the CPU-selected alpha.
    resetDeviceState(strip, Float64Array.from(scene.positions));
    const st = await newtonPrologue(strip);
    const cpu = await cpuSequential(strip, st, 10);
    expect(cpu.acceptedIndex).toBeGreaterThanOrEqual(0);
    await driver.applyTrial(cpu.acceptedAlpha);
    await solver.evaluateNewtonState(true, 710, 1 / 60);
    await driver.acceptTrial();
    const posA = await ex.readBufferDebug("position", "commit-a-pos", false);
    const lagA = await ex.readBufferDebug("laggedN", "commit-a-lag", false);
    // Path B (batched): fresh identical state, batch select + sync-free commit.
    resetDeviceState(strip, Float64Array.from(scene.positions));
    const st2 = await newtonPrologue(strip);
    expect(st2.E0).toBe(st.E0);
    const P = sceneParams(scene, driver.c.cap);
    solver.beginArmijoBatch(720);
    const bst = await driver.armijoBatch({
      E0: st2.E0, gtdx: st2.gtdx, alphaBase: 1, beta: 0.5, K: 4,
      trustScale: st2.trustScale, mat: P.mat, contact: P.contact,
      evalIndexBase: 720, pcgBreakdown: false, newtonConverged: false,
    });
    expect(bst.accepted).toBe(true);
    expect(bst.selectedAlpha).toBeCloseTo(cpu.acceptedAlpha, 6);
    await driver.applyTrial(bst.selectedAlpha);
    ex.beginBatch("commit-refresh");
    driver.rebuildTrialPasses(P.mat, P.contact);
    await ex.submitBatch(false);
    await driver.acceptTrial();
    const posB = await ex.readBufferDebug("position", "commit-b-pos", false);
    const lagB = await ex.readBufferDebug("laggedN", "commit-b-lag", false);
    const pa = new Float32Array(posA);
    const pb = new Float32Array(posB);
    let worst = 0;
    for (let i = 0; i < pa.length; i++) {
      worst = Math.max(worst, Math.abs(pa[i] - pb[i]));
    }
    // eslint-disable-next-line no-console
    console.log(`[g6b-commit] max |posA-posB| = ${worst.toExponential(2)}`);
    expect(worst).toBeLessThan(1e-6);
    const la = new Float32Array(lagA);
    const lb = new Float32Array(lagB);
    expect(la.length).toBe(lb.length);
    for (let i = 0; i < la.length; i++) {
      expect(lb[i]).toBe(la[i]);
    }
  }, 300000);

  it("fold: multi-batch parity across reject-then-accept", async () => {
    const fix = await requireDevice(buildFoldScene, { contactCapacity: 4096, pairCapacity: 8192 });
    if (!fix) return;
    try {
      const { solver, ex, driver } = fix;
      const scene = (solver as unknown as { scene: ClothScene }).scene;
      resetDeviceState(fix, Float64Array.from(scene.positions));
      const st = await newtonPrologue(fix);
      const cpu = await cpuSequential(fix, st, 10);
      const P = sceneParams(scene, driver.c.cap);
      let alphaBase = 1;
      let trialsUsed = 0;
      let acceptedIndex = -1;
      let acceptedAlpha = NaN;
      let batches = 0;
      const gpuRows: number[][] = [];
      while (trialsUsed < 10) {
        const K = Math.min(4, 10 - trialsUsed);
        solver.beginArmijoBatch(700 + trialsUsed);
        const bst = await driver.armijoBatch({
          E0: st.E0, gtdx: st.gtdx, alphaBase, beta: 0.5, K,
          trustScale: st.trustScale, mat: P.mat, contact: P.contact,
          evalIndexBase: 700 + trialsUsed,
          pcgBreakdown: false, newtonConverged: false,
        });
        batches++;
        const candRaw = await ex.readBufferDebug("armijoCandidates", "parity-fold-cand", false);
        const cand = new Float32Array(candRaw);
        for (let j = 0; j < K; j++) gpuRows.push([...cand.subarray((trialsUsed + j) * 8, (trialsUsed + j) * 8 + 8)]);
        trialsUsed += bst.trialsEvaluated;
        if (bst.accepted) {
          acceptedIndex = trialsUsed - K + bst.selectedIndex;
          acceptedAlpha = bst.selectedAlpha;
          break;
        }
        alphaBase *= Math.pow(0.5, bst.trialsEvaluated || K);
      }
      // eslint-disable-next-line no-console
      console.log(`[g6b-parity] fold: cpu li=${cpu.acceptedIndex} gpu li=${acceptedIndex} batches=${batches}`);
      expect(acceptedIndex).toBe(cpu.acceptedIndex);
      if (cpu.acceptedIndex >= 0) {
        expect(acceptedAlpha).toBeCloseTo(cpu.acceptedAlpha, 5);
      }
      // Row-level parity on the shared prefix (tolerance looser under contact:
      // atomic compaction order can differ in the last ulp across runs).
      expect(gpuRows.length).toBeGreaterThanOrEqual(cpu.trials.length);
      for (let j = 0; j < cpu.trials.length; j++) {
        const t = cpu.trials[j];
        const row = gpuRows[j];
        expect(row[0]).toBeCloseTo(t.alphaEff, 6);
        const eScale = Math.max(Math.abs(t.energy), 1e-9);
        expect(Math.abs(row[1] - t.energy) / eScale).toBeLessThan(1e-4);
        expect(row[3] > 0.5).toBe(t.accepted);
        // validity lanes mirror the CPU verdicts
        expect(row[4]).toBeCloseTo(t.minDistance === Infinity ? 1e30 : t.minDistance, 4);
      }
    } finally {
      fix.ex.destroy();
    }
  }, 300000);

  it("select unit: first-valid-wins, reject-all, safe-Infinity", async () => {
    if (!strip) return;
    const { ex, driver } = strip;
    const rows = new Float32Array(64);
    const setRow = (k: number, alpha: number, energy: number, ok: boolean): void => {
      rows[k * 8] = alpha;
      rows[k * 8 + 1] = energy;
      rows[k * 8 + 2] = 0.01;
      rows[k * 8 + 3] = ok ? 1 : 0;
      rows[k * 8 + 4] = 0.01;
      rows[k * 8 + 5] = 2.0; // safe-Infinity TOI sentinel
      rows[k * 8 + 6] = 1;
      rows[k * 8 + 7] = 0;
    };
    async function runSelect(K: number): Promise<ArmijoBatchStatus> {
      const st0 = new Float32Array(16);
      st0[8] = 1.0;
      ex.writeBuffer("armijoCandidates", rows);
      ex.writeBuffer("armijoStatus", st0);
      driver.bankU(GpuUniformSlot.ArmijoK, K);
      driver.bankF(GpuUniformSlot.ArmijoPcgBd, 0);
      driver.bankF(GpuUniformSlot.ArmijoNewtonConv, 0);
      ex.beginBatch("select-unit");
      driver.armijoSelectPass();
      await ex.submitBatch(false);
      const raw = await ex.readSmall("armijoStatus", 64, "select-unit", "status");
      return decodeArmijoStatus(raw);
    }
    // invalid-first-valid-later -> index 1
    setRow(0, 1.0, 0.5, false);
    setRow(1, 0.5, 0.4, true);
    setRow(2, 0.25, 0.3, true);
    setRow(3, 0.125, 0.2, true);
    {
      const s = await runSelect(4);
      expect(s.accepted).toBe(true);
      expect(s.selectedIndex).toBe(1);
      expect(s.selectedAlpha).toBeCloseTo(0.5, 6);
      expect(s.energy).toBeCloseTo(0.4, 6);
    }
    // all invalid (NaN energy / failed rows) -> reject-all marker
    setRow(0, 1.0, NaN, false);
    setRow(1, 0.5, 0.4, false);
    setRow(2, 0.25, NaN, false);
    setRow(3, 0.125, 0.2, false);
    {
      const s = await runSelect(4);
      expect(s.accepted).toBe(false);
      expect(s.selectedIndex).toBe(-1);
      expect(s.finite).toBe(true); // status finite lane untouched by select
    }
    // all valid -> index 0 (largest alpha first)
    setRow(0, 1.0, 0.5, true);
    setRow(1, 0.5, 0.4, true);
    {
      const s = await runSelect(2);
      expect(s.accepted).toBe(true);
      expect(s.selectedIndex).toBe(0);
    }
  }, 300000);
});
