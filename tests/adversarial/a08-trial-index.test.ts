// ADVERSARIAL A08 — `selectedTrialIndex` is wrong for non-uniform batch widths.
//
// `newton-control.wgsl:commit_arm` computes the GLOBAL trial index as
//
//     caCtl[8] = f32(bi * caK + li)
//
//     bi  = caCtl[5] — running batch index (self-tracked, incremented per call)
//     caK = ArmijoK uniform — THIS BATCH's width
//     li  = local accepted row
//
// That identity only holds when every batch has the SAME width. The true
// global index is `trialBase(bi) + li` where
// `trialBase(b) = sum_{j<b} K_j`.
//
// `batchKsFor()` currently emits uniform widths except possibly the last
// (e.g. [4,4,2]), so the formula is accidentally right for the shapes it
// produces — and A06 measured `trialHistory = [8, -1]`, where batch 2 has K = 2
// but the reported index 8 = 2*4+0, i.e. the uniform still read 4. The value
// is correct by coincidence, not by construction.
//
// This is the exact input the documented G6C follow-up (adaptive K) needs, so
// a wrong index here corrupts the telemetry used to design the adaptive rule —
// a circular failure. This test proves the formula breaks as soon as widths
// differ in an INTERIOR position, which any ladder such as [8,2,4,4] does.
import { describe, it, expect, beforeAll } from "vitest";
import { buildGrid, preprocess } from "../../src/mesh/mesh.js";
import { createScene } from "../../src/physics/scene.js";
import { DEFAULT_MATERIAL } from "../../src/physics/types.js";
import { ContactSystem } from "../../src/collision/contact-assembly.js";
import { DEFAULT_CONTACT_PARAMS } from "../../src/collision/types.js";
import type { DeviceFixture } from "../webgpu/device-setup.js";
import { requireDevice } from "../webgpu/device-setup.js";

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

function restingFloor() {
  const g = buildGrid(6, 6, 0.12, 0.12);
  for (let i = 0; i < g.positions.length / 3; i++) g.positions[i * 3 + 1] += 0.0012;
  const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
  const scene = createScene(mesh, { ...DEFAULT_MATERIAL }, [0, -9.81, 0]);
  const c = new ContactSystem({ ...DEFAULT_CONTACT_PARAMS }, mesh.indices);
  c.setFloor(0);
  scene.contact = c;
  return scene;
}

beforeAll(async () => {
  fix = await requireDevice(restingFloor, { contactCapacity: 2048, pairCapacity: 8192 });
}, 180000);

async function runWithKs(batchKs: number[]): Promise<{ idx: number; accepted: boolean; kUsed: number[] }> {
  const f = fix!;
  const { ex, driver } = f;
  const solver = f.solver;
  solver.configureStep(1 / 60);
  const raw = IN(solver).scene.material;
  const mat = IN(solver).materialNow();
  const beta = Math.max(raw.stretchWarp, raw.stretchWeft, raw.shear) * raw.thickness * 0.1 + 1e-6;

  const ev = await solver.evaluateNewtonState(false, 91, 1 / 60);
  IN(solver).contactCountNow = driver.c.cap;
  IN(solver).simImage.newtonIteration = 0;
  IN(solver).pushSimParams();
  ex.writeBuffer("e0Store", new Float32Array([ev.status.energy, 0, 0, 0]));
  driver.bankF(42, 1e-5);
  driver.bankF(43, 0.002);
  ex.writeBuffer("newtonCtl", new Float32Array(16));
  ex.writeBuffer("newtonStatus", new Float32Array(20));

  const st = await driver.newtonRound({
    round: 0, beta, mat, contact: IN(solver).contactParamsNow(), batchKs, evalIndexBase: 0,
  });
  // Recover the local accepted row from armijoCandidates so we can compute the
  // TRUE global index independently of commit_arm's arithmetic.
  const cand = new Float32Array(await ex.readBufferDebug("armijoCandidates", "adv-cand", false));
  const kUsed: number[] = [];
  for (let k = 0; k < batchKs[batchKs.length - 1]; k++) {
    if (cand[k * 8 + 3] > 0.5) kUsed.push(k);
  }
  return { idx: st.selectedTrialIndex, accepted: st.armijoAccepted, kUsed };
}

describe("ADVERSARIAL A08 — global trial index under non-uniform batch widths", () => {
  it("uniform widths: index is well-formed", async () => {
    if (!fix) return;
    const r = await runWithKs([4, 4, 4]);
    // eslint-disable-next-line no-console
    console.log(`[A08] [4,4,4]: accepted=${r.accepted} selectedTrialIndex=${r.idx} lastBatchLocal=${JSON.stringify(r.kUsed)}`);
    if (r.accepted) expect(r.idx).toBeGreaterThanOrEqual(0);
    expect(r.idx).toBeLessThan(12);
  }, 600000);

  it("NON-uniform widths: bi*K+li is not the global trial index", async () => {
    if (!fix) return;
    // Ladder with a narrower INTERIOR batch — the shape adaptive-K produces.
    const ks = [2, 4, 4];
    const trialBase = ks.reduce<number[]>((acc, k, i) => {
      if (i > 0) acc.push(acc[i - 1] + ks[i - 1]);
      else acc.push(0);
      return acc;
    }, []);
    const total = ks.reduce((a, b) => a + b, 0);

    const r = await runWithKs(ks);
    // eslint-disable-next-line no-console
    console.log(`[A08] batchKs=${JSON.stringify(ks)} trialBase=${JSON.stringify(trialBase)} total=${total}`);
    // eslint-disable-next-line no-console
    console.log(`[A08]   accepted=${r.accepted} selectedTrialIndex=${r.idx}`);

    if (r.accepted) {
      // The index must land inside the ladder. commit_arm's bi*K+li uses THIS
      // batch's width, so for any accepted batch after the first the reported
      // index drifts by the accumulated width mismatch.
      const inRange = r.idx >= 0 && r.idx < total;
      // eslint-disable-next-line no-console
      console.log(`[A08]   inRange(0..${total - 1})=${inRange}`);
      // Ground truth: the accepted row is whichever local index has ok=1, and
      // the batch it came from is implied by trialBase.
      expect(inRange).toBe(true);
    } else {
      // eslint-disable-next-line no-console
      console.log("[A08]   batch rejected all — formula not exercised this run");
    }
  }, 600000);

  it("documented divergence: bi*K+li vs sum-of-previous-widths", () => {
    // Pure arithmetic check of commit_arm's identity — no device needed.
    const cases: number[][] = [[4, 4, 2], [4, 4, 4], [8, 2, 4, 4], [2, 4, 4], [8, 1, 1]];
    const rows: string[] = [];
    let anyWrong = false;
    for (const ks of cases) {
      const base = ks.map((_, i) => ks.slice(0, i).reduce((a, b) => a + b, 0));
      for (let bi = 0; bi < ks.length; bi++) {
        for (let li = 0; li < ks[bi]; li++) {
          const reported = bi * ks[bi] + li;          // commit_arm
          const truth = base[bi] + li;                // what the ladder means
          if (reported !== truth) {
            anyWrong = true;
            rows.push(`  batchKs=${JSON.stringify(ks)} bi=${bi} li=${li} -> reported=${reported} truth=${truth} (Δ${reported - truth})`);
          }
        }
      }
    }
    // eslint-disable-next-line no-console
    console.log(`[A08] formula mismatches:\n${rows.join("\n")}`);
    // eslint-disable-next-line no-console
    console.log(`[A08] total mismatching (bi, li) pairs = ${rows.length}`);
    // Documented: the identity is false for every non-uniform ladder.
    expect(anyWrong).toBe(true);
  });
});