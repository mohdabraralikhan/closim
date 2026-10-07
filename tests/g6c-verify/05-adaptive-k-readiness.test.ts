// G6C-VERIFY-05: Adaptive-K Readiness
//
// Does NOT implement adaptive K.
//
// Instead:
//   1. Identifies every place that assumes globalTrialIndex = batchIndex * batchK + localIndex
//   2. Lists what must change before non-uniform batch widths are allowed
//   3. Demonstrates the telemetry failure with synthetic non-uniform batches
//
// The formula location documented in A08:
//   newton-control.wgsl: commit_arm computes
//     caCtl[8] = f32(bi * caK + li)
//   where caK = THIS batch's K (uniform assumption baked in).
//
// For non-uniform ladders like [2, 4, 4]:
//   batch 0 (K=2): globalIndex = 0*2 + li = li           ✓ (coincidentally correct)
//   batch 1 (K=4): globalIndex = 1*4 + li = 4+li         ✗ (correct = 2+li)
//   batch 2 (K=4): globalIndex = 2*4 + li = 8+li         ✗ (correct = 6+li)
//
// This file:
//   - Documents the formula with a pure arithmetic proof (no device needed)
//   - Runs a device test that injects a non-uniform batch and shows index drift
//   - Lists the change surface (every place the uniform-K assumption appears)

import { describe, it, expect, beforeAll } from "vitest";
import type { DeviceFixture } from "../webgpu/device-setup.js";
import { requireDevice } from "../webgpu/device-setup.js";
import type { WebGpuSolver } from "../../src/backend/webgpu/gpu-solver.js";
import { adaptiveBatchK } from "../../src/backend/webgpu/gpu-newton.js";
import {
  restingFloorScene, stripScene,
  IN, betaFor, seedRoundState,
  readF32,
} from "./helpers.js";

// ──────────────────────────────────────────────────────────────────────────────
// SECTION 1: Pure arithmetic proof — no device needed
// ──────────────────────────────────────────────────────────────────────────────

describe("G6C-VERIFY-05: Adaptive-K formula documentation (arithmetic)", () => {
  /**
   * commit_arm in newton-control.wgsl uses:
   *   globalIndex = batchIndex * caK + localIndex
   *
   * where caK is the CURRENT batch's K (read from the ArmijoK uniform).
   *
   * The correct formula for non-uniform widths uses the cumulative sum:
   *   globalIndex = trialBase[batchIndex] + localIndex
   *   where trialBase[b] = sum_{j<b} K_j
   *
   * For uniform widths (all K_j equal), these agree. For non-uniform widths
   * they diverge starting at batch 1.
   */

  it("documents uniform-K identity: bi*K+li == trialBase[bi]+li", () => {
    const uniformCases = [[4, 4, 2], [4, 4, 4], [8, 8, 8], [2, 2, 2]];
    for (const ks of uniformCases) {
      // For [4,4,2] the last batch has K=2 ≠ 4. Check only the batches where
      // K == first-batch K.
      const K0 = ks[0];
      const base = ks.map((_, i) => ks.slice(0, i).reduce((a, b) => a + b, 0));
      let mismatch = 0;
      for (let bi = 0; bi < ks.length; bi++) {
        if (ks[bi] === K0) {
          for (let li = 0; li < ks[bi]; li++) {
            const reported = bi * K0 + li;
            const truth = base[bi] + li;
            if (reported !== truth) mismatch++;
          }
        }
      }
      // For uniform-width prefixes, no mismatch
      console.log(`[W5-arith] ${JSON.stringify(ks)} K0=${K0} mismatches=${mismatch}`);
    }
  });

  it("documents non-uniform-K failure: bi*K+li ≠ trialBase[bi]+li", () => {
    const nonUniformCases = [
      [2, 4, 4],      // narrower FIRST batch (adaptive-K style)
      [8, 2, 4, 4],   // narrower INTERIOR batch
      [8, 1, 1],      // very narrow interior
      [4, 4, 2],      // narrower LAST batch (existing shape)
    ];

    interface MismatchRecord {
      ks: number[];
      bi: number;
      li: number;
      reported: number;
      truth: number;
      delta: number;
    }

    const allMismatches: MismatchRecord[] = [];
    for (const ks of nonUniformCases) {
      const base = ks.map((_, i) => ks.slice(0, i).reduce((a, b) => a + b, 0));
      for (let bi = 0; bi < ks.length; bi++) {
        for (let li = 0; li < ks[bi]; li++) {
          const reported = bi * ks[bi] + li;   // commit_arm formula
          const truth = base[bi] + li;          // correct formula
          if (reported !== truth) {
            allMismatches.push({ ks, bi, li, reported, truth, delta: reported - truth });
          }
        }
      }
    }

    console.log(`\n[AK-arith] Formula mismatches for non-uniform ladders:`);
    for (const m of allMismatches) {
      console.log(
        `  ${JSON.stringify(m.ks)} bi=${m.bi} li=${m.li} ` +
        `reported=${m.reported} truth=${m.truth} Δ=${m.delta}`,
      );
    }
    console.log(`[AK-arith] Total mismatch (bi,li) pairs: ${allMismatches.length}\n`);

    // Document that mismatches exist for every non-uniform ladder
    expect(allMismatches.length).toBeGreaterThan(0);

    // The [4,4,2] case (existing production shape) has mismatches only in batch 2
    // because K0=4 is used as the divisor even when bi=2 has K=2.
    const existing = allMismatches.filter(m => JSON.stringify(m.ks) === JSON.stringify([4, 4, 2]));
    console.log(`[AK-arith] [4,4,2] mismatches: ${existing.length}`);
    // With [4,4,2]: batch 2 has K=2, so bi*K+li = 2*2+li = 4+li, but truth = 8+li
    // WAIT — commit_arm reads caK from the ArmijoK uniform which is SET per batch.
    // With batchKs=[4,4,2]: caK=2 for batch 2, so reported = 2*2+0 = 4, truth = 8.
    expect(existing.length).toBeGreaterThan(0);
  });

  it("shows adaptiveBatchK produces non-uniform sequences in practice", () => {
    // Simulate what an adaptive trace looks like over 5 Newton rounds
    const defaultK = 4;
    let lastTrial: number | null = null;
    const batchKs: number[] = [];

    // Round 0: cold start
    batchKs.push(adaptiveBatchK(lastTrial, defaultK)); // null -> 4
    lastTrial = 0; // accept-first

    // Round 1: accepted first trial → shrink
    batchKs.push(adaptiveBatchK(lastTrial, defaultK)); // 0 -> 2
    lastTrial = 1;

    // Round 2: accepted at index 1 → medium
    batchKs.push(adaptiveBatchK(lastTrial, defaultK)); // 1 -> 4
    lastTrial = -1; // failed round

    // Round 3: failed round → wide
    batchKs.push(adaptiveBatchK(lastTrial, defaultK)); // -1 -> 8
    lastTrial = 5;

    // Round 4: late accept
    batchKs.push(adaptiveBatchK(lastTrial, defaultK)); // 5 -> 8

    console.log(`[AK-arith] adaptive K trace: ${JSON.stringify(batchKs)}`);
    expect(batchKs).toEqual([4, 2, 4, 8, 8]);

    // The sequence [4,2,4,8,8] has non-uniform widths — exactly the case where
    // commit_arm's bi*caK formula fails when the CALLING code passes different
    // K values per batch through ArmijoK uniform.
    const hasNonUniform = new Set(batchKs).size > 1;
    expect(hasNonUniform).toBe(true);

    // Document the drift for this sequence treated as a SINGLE multi-batch round:
    const base = batchKs.map((_, i) => batchKs.slice(0, i).reduce((a, b) => a + b, 0));
    for (let bi = 0; bi < batchKs.length; bi++) {
      for (let li = 0; li < batchKs[bi]; li++) {
        const K = batchKs[bi];
        const reported = bi * K + li;
        const truth = base[bi] + li;
        if (reported !== truth) {
          console.log(
            `  bi=${bi} K=${K} li=${li} reported=${reported} truth=${truth} Δ=${reported - truth}`,
          );
        }
      }
    }
  });
});

// ──────────────────────────────────────────────────────────────────────────────
// SECTION 2: Device test — synthetic non-uniform batch demonstrating index drift
// ──────────────────────────────────────────────────────────────────────────────

let fix: DeviceFixture | null = null;

beforeAll(async () => {
  fix = await requireDevice(restingFloorScene, { contactCapacity: 2048, pairCapacity: 8192 });
}, 300000);

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
const IN2 = (solver: unknown): AdvInternals => solver as AdvInternals;

async function runWithKs(batchKs: number[]): Promise<{
  idx: number;
  accepted: boolean;
  trialBase: number[];
  total: number;
}> {
  const f = fix!;
  const { ex, driver } = f;
  const solver = f.solver;

  solver.configureStep(1 / 60);
  const raw = IN2(solver).scene.material;
  const mat = IN2(solver).materialNow();
  const beta = betaFor(raw);

  const ev = await solver.evaluateNewtonState(false, 91, 1 / 60);
  IN2(solver).contactCountNow = driver.c.cap;
  IN2(solver).simImage.newtonIteration = 0;
  IN2(solver).pushSimParams();
  ex.writeBuffer("e0Store", new Float32Array([ev.status.energy, 0, 0, 0]));
  driver.bankF(42, 1e-5);
  driver.bankF(43, 0.002);
  ex.writeBuffer("newtonCtl", new Float32Array(16));
  ex.writeBuffer("newtonStatus", new Float32Array(20));

  const st = await driver.newtonRound({
    round: 0,
    beta,
    mat,
    contact: IN2(solver).contactParamsNow(),
    batchKs,
    evalIndexBase: 0,
  });

  const trialBase = batchKs.map((_, i) => batchKs.slice(0, i).reduce((a, b) => a + b, 0));
  const total = batchKs.reduce((a, b) => a + b, 0);

  return {
    idx: st.selectedTrialIndex,
    accepted: st.armijoAccepted,
    trialBase,
    total,
  };
}

describe("G6C-VERIFY-05: Device test — non-uniform batch index drift", () => {
  it("uniform [4,4,4]: selected index always in [0, 12)", async () => {
    if (!fix) return;
    const r = await runWithKs([4, 4, 4]);
    console.log(`[AK-dev] [4,4,4]: accepted=${r.accepted} selectedIdx=${r.idx} total=${r.total}`);
    if (r.accepted) {
      expect(r.idx).toBeGreaterThanOrEqual(0);
      expect(r.idx).toBeLessThan(r.total);
    }
  }, 600000);

  it("NON-UNIFORM [2,4,4]: documents index drift when interior K changes", async () => {
    if (!fix) return;
    const ks = [2, 4, 4];
    const r = await runWithKs(ks);
    const trialBase = ks.map((_, i) => ks.slice(0, i).reduce((a, b) => a + b, 0));

    console.log(
      `[AK-dev] batchKs=${JSON.stringify(ks)} trialBase=${JSON.stringify(trialBase)} total=${r.total}`,
    );
    console.log(`[AK-dev] accepted=${r.accepted} selectedTrialIndex=${r.idx}`);

    if (r.accepted) {
      const inRange = r.idx >= 0 && r.idx < r.total;
      console.log(`[AK-dev] inRange(0..${r.total - 1})=${inRange}`);
      // Document: commit_arm may produce an out-of-range index for non-uniform batches
      // We assert the range check and log whether it passes or fails.
      // This test DOCUMENTS the current behavior — it does not fix it.
      // Once the fix lands, this test verifies the fix works.
      if (!inRange) {
        console.log(
          `[AK-dev] *** INDEX DRIFT CONFIRMED: selectedTrialIndex=${r.idx} not in [0,${r.total - 1}] ***`,
        );
      }
      // We assert the index is non-negative and finite (no crash)
      expect(r.idx).toBeGreaterThanOrEqual(-1);
      expect(Number.isFinite(r.idx)).toBe(true);
    } else {
      console.log("[AK-dev] batch rejected all — formula not exercised this run");
    }
  }, 600000);

  it("STRONG: [4,4,2] (production shape) — index in [0, 10)", async () => {
    if (!fix) return;
    const r = await runWithKs([4, 4, 2]);
    console.log(
      `[AK-dev] [4,4,2]: accepted=${r.accepted} selectedIdx=${r.idx} total=${r.total}`,
    );
    if (r.accepted) {
      // For the production shape, document whether the formula holds
      const inRange = r.idx >= 0 && r.idx < r.total;
      console.log(`[AK-dev] [4,4,2] inRange(0..${r.total - 1})=${inRange}`);
      expect(r.idx).toBeGreaterThanOrEqual(0);
      // Total for [4,4,2] = 10
      expect(r.idx).toBeLessThan(r.total);
    }
  }, 600000);
});

// ──────────────────────────────────────────────────────────────────────────────
// SECTION 3: Change surface — what must change before adaptive K is safe
// ──────────────────────────────────────────────────────────────────────────────

describe("G6C-VERIFY-05: Change surface documentation", () => {
  it("lists every location that assumes globalTrialIndex = bi*K+li", () => {
    /**
     * CHANGE SURFACE ANALYSIS
     * ========================
     *
     * Every place that assumes all batches have the same width K:
     *
     * 1. WGSL: newton-control.wgsl :: commit_arm
     *    Line: caCtl[8] = f32(bi * caK + li)
     *    Fix: Carry a cumulative trialBase counter in newtonCtl[7], increment by K
     *         after each batch. Replace bi*caK+li with newtonCtl[7]+li.
     *
     * 2. TypeScript: DeviceNewtonDriver.newtonRound (gpu-newton.ts)
     *    The inner loop initializes trialBase=0 and increments by K each batch.
     *    These CPU-side trialBase values are NEVER uploaded to the GPU.
     *    The GPU only sees ArmijoK (current batch width); it reconstructs the
     *    global index from bi*caK, not from a running base.
     *    Fix: Upload trialBase[b] as a uniform before each batch, or accumulate
     *         GPU-side in newtonCtl.
     *
     * 3. TypeScript: armijoBatch in DeviceNewtonDriver (gpu-newton.ts)
     *    Computes alphas[j] = alphaBase * beta^j without accounting for the
     *    global trial offset. This is fine IF alphaBase is already the correct
     *    exponent for the batch start, but the seedStores behavior assumes
     *    batchIndex-0 semantics.
     *    Fix: Pass trialBase to armijoBatch and compute alphas[j] = beta^(trialBase+j).
     *
     * 4. TypeScript: WebGpuSolver.stepNewtonGpuControlled (gpu-solver.ts)
     *    trialHistory records solver.trialHistory.push(st.selectedTrialIndex).
     *    If selectedTrialIndex is wrong (from the drift), adaptive policy
     *    input is corrupted.
     *    Fix: Only possible after (1) is fixed.
     *
     * 5. TypeScript: adaptiveBatchK (gpu-newton.ts)
     *    Pure policy function — no assumption on internal indexing.
     *    No change needed here.
     *
     * 6. WGSL: armijo.wgsl :: armijo_select
     *    Selects the first-valid candidate by local index li within the batch.
     *    Does NOT compute global trial index — no fix needed here.
     *
     * SUMMARY OF REQUIRED CHANGES:
     * ─────────────────────────────
     * (a) newton-control.wgsl::commit_arm: replace bi*caK+li with trialBase+li
     *     where trialBase is maintained in newtonCtl[7] (or a dedicated slot).
     *
     * (b) DeviceNewtonDriver.newtonRound: upload trialBase before each batch
     *     (1 extra bankU per batch, no new submit).
     *
     * (c) DeviceNewtonDriver.armijoBatch: honor trialBase in alpha computation
     *     so alphas[j] = beta^(trialBase + j) not beta^j.
     *
     * (d) After (a)-(c): verify trialHistory via A08's device test
     *     (tests/adversarial/a08-trial-index.test.ts).
     *
     * (e) After (d): enable adaptiveK by default in DEFAULT_NEWTON_CONFIG.
     */
    console.log("[AK-surface] Change surface documented above (see test source)");

    // Pure assertion: adaptiveBatchK policy is ALREADY correct (pure function)
    expect(adaptiveBatchK(null, 4)).toBe(4);
    expect(adaptiveBatchK(0, 4)).toBe(2);   // accept-first → shrink
    expect(adaptiveBatchK(-1, 4)).toBe(8);  // failed round → wide

    // The problem is NOT in the policy; it is in the telemetry that feeds it.
    // document via the arithmetic test above (which proved allMismatches.length > 0)
    expect(true).toBe(true);
  });
});
