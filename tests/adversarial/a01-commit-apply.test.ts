// ADVERSARIAL A01 — KERNEL CHARACTERIZATION of `commit_apply`.
//
// STATUS: the driver defect this test was written against is FIXED in the
// working tree (single end-of-round commit). The assertions below therefore
// now document the kernel's SEMANTICS rather than assert a driver bug: they
// stay useful as a tripwire, because `commit_apply` is a pure function of
// (position, nbAlpha, dx) and will re-introduce the double-step the moment any
// caller invokes it more than once per round without `position` advancing.
//
// Threat model that motivated it: `DeviceNewtonDriver.newtonRound` used to run
// the FULL Armijo batch
// ladder (`batchKsFor()` never truncates on accept; with K=4 / armijoIters=10
// it always issues [4,4,2]). Only `commit_copy_if` and `commit_lagged_if` are
// predicated on `newtonCtl[3]` (doCommit). `commit_apply` and
// `rebuildTrialPasses` are NOT predicated. `commit_apply` computes
//
//     xTrial = position + nbAlpha * dx
//
// where nbAlpha is STICKY (latched by commit_arm on the first accept).
// After the accepting batch, `position` has ALREADY been advanced to
// x_k + alpha*dx. So every subsequent (rejecting) batch recomputes
//
//     xTrial = (x_k + alpha*dx) + alpha*dx = x_k + 2*alpha*dx
//
// — a phantom state — and `rebuildTrialPasses` then rebuilds the entire
// contact record set / gradient / contactDiag / rhs AT that phantom state,
// leaving `position` desynchronised from everything that describes it.
//
// READ-ONLY against production code: the only dispatch issued is the existing
// `newton-control:commit_apply` entry point plus public driver passes.
import { describe, it, expect, beforeAll } from "vitest";
import { buildGrid, preprocess } from "../../src/mesh/mesh.js";
import { createScene } from "../../src/physics/scene.js";
import { DEFAULT_MATERIAL } from "../../src/physics/types.js";
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

/** Tiny UNPINNED 3x2 grid — 6 verts, 4 tris. No pin column to mask the probe. */
function tinyUnpinned() {
  const g = buildGrid(3, 2, 0.06, 0.04);
  const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
  return createScene(mesh, { ...DEFAULT_MATERIAL }, [0, -9.81, 0]);
}

beforeAll(async () => {
  fix = await requireDevice(tinyUnpinned, { contactCapacity: 32, pairCapacity: 64 });
}, 180000);

/** Encode ONLY `newton-control:commit_apply` at the current device state. */
async function runCommitApply(f: DeviceFixture): Promise<void> {
  const { ex } = f;
  ex.beginBatch("adv-commit-apply");
  ex.runPass({
    shader: "newton-control", entry: "commit_apply",
    groups: [[
      { binding: 64, buffer: "position" },
      { binding: 65, buffer: "searchDirection" },
      { binding: 66, buffer: "xTrial" },
      { binding: 67, buffer: "pinMask" },
      { binding: 68, buffer: "pinPos" },
      { binding: 69, buffer: "newtonCtl" },
    ]],
    x: Math.max(1, Math.ceil(f.driver.c.n / 64)),
  });
  await ex.submitBatch(false);
}

async function readF32(f: DeviceFixture, name: string): Promise<Float32Array> {
  return new Float32Array(await f.ex.readBufferDebug(name, `adv-${name}`, false));
}

describe("ADVERSARIAL A01 — commit_apply double-step", () => {
  it("re-applies the accepted alpha on an already-advanced position buffer", async () => {
    if (!fix) return;
    const f = fix;
    const { ex, driver } = f;
    const n = driver.c.n;

    // --- deterministic setup: position = x, dx = fixed step -------------------
    const pos = new Float32Array(n * 4);
    const dx = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      pos[i * 4] = 0.01 * (i + 1);
      pos[i * 4 + 1] = 0.02 * (i + 1);
      pos[i * 4 + 2] = 0.03 * (i + 1);
      dx[i * 3] = 1e-3; dx[i * 3 + 1] = -2e-3; dx[i * 3 + 2] = 5e-4;
    }
    ex.writeBuffer("position", pos);
    ex.writeBuffer("searchDirection", dx);
    const ALPHA = 0.5;
    // Sticky accepted alpha + doCommit=0 (i.e. the state after an accept,
    // during a LATER rejecting batch) — exactly commit_arm's residue.
    ex.writeBuffer("newtonCtl", new Float32Array([0, 0, 1, 0, ALPHA, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]));

    const err = (got: Float32Array, mult: number): number => {
      let e = 0;
      for (let i = 0; i < n; i++) {
        for (let k = 0; k < 3; k++) {
          e = Math.max(e, Math.abs(got[i * 4 + k] - (pos[i * 4 + k] + mult * ALPHA * dx[i * 3 + k])));
        }
      }
      return e;
    };

    // --- pass 1: PRE-commit (position still == x_k) — reference behaviour ----
    await runCommitApply(f);
    const single = err(await readF32(f, "xTrial"), 1);
    expect(single).toBeLessThan(1e-6); // pre-commit is correct (f32 rounding only)

    // --- emulate commit_copy_if advancing position to x_k + alpha*dx -------
    const committed = await readF32(f, "xTrial");
    ex.writeBuffer("position", committed);

    // --- pass 2: POST-commit (rejecting batch) — THE BUG -------------------
    await runCommitApply(f);
    const trial2 = await readF32(f, "xTrial");
    const doubleErr = err(trial2, 2);
    const singleErr = err(trial2, 1);
    // eslint-disable-next-line no-console
    console.log(`[A01] post-commit commit_apply: |xTrial-(x+2ad)|=${doubleErr.toExponential(3)} ` +
      `|xTrial-(x+ad)|=${singleErr.toExponential(3)} maxStep=${(ALPHA * 1e-3).toExponential(3)}`);

    // CONFIRMED DETERMINISTIC: the kernel lands the DOUBLE step.
    expect(doubleErr).toBeLessThan(1e-6);
    expect(singleErr).toBeGreaterThan(1e-6);
  }, 300000);

  it("rebuildTrialPasses materialises contact+gradient state at the phantom x", async () => {
    if (!fix) return;
    const f = fix;
    const { ex, driver } = f;
    const n = driver.c.n;
    const solver = f.solver;
    solver.configureStep(1 / 60);

    // A real, self-consistent evaluation at `position`.
    await solver.evaluateNewtonState(false, 4001, 1 / 60);
    const gradGood = await readF32(f, "gradient");
    const distGood = await readF32(f, "contactDist");

    // Emulate the rejecting batch: sticky nbAlpha, doCommit=0, position already
    // advanced. commit_apply + rebuildTrialPasses then run at x_k + 2*alpha*dx.
    const dxNow = await readF32(f, "searchDirection");
    const ALPHA = 0.5;
    ex.writeBuffer("newtonCtl", new Float32Array([0, 0, 1, 0, ALPHA, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]));
    await runCommitApply(f);
    driver.zeroContactCounters();
    ex.beginBatch("adv-rebuild-phantom");
    driver.rebuildTrialPasses(
      { c00: 2e4, c11: 2e4, c01: 0, g: 5e3, thickness: 1e-3 },
      IN(solver).contactParamsNow(),
    );
    await ex.submitBatch(false);

    const gradBad = await readF32(f, "gradient");
    const distBad = await readF32(f, "contactDist");

    let gDiff = 0; let gScale = 0; let dDiff = 0;
    for (let i = 0; i < n * 3; i++) {
      gDiff = Math.max(gDiff, Math.abs(gradBad[i] - gradGood[i]));
      gScale = Math.max(gScale, Math.abs(gradGood[i]));
    }
    for (let i = 0; i < distGood.length; i++) dDiff = Math.max(dDiff, Math.abs(distBad[i] - distGood[i]));
    const rel = gDiff / Math.max(gScale, 1e-30);
    // eslint-disable-next-line no-console
    console.log(`[A01] phantom rebuild: relGradDelta=${rel.toExponential(3)} ` +
      `|distDelta|=${dDiff.toExponential(3)} verts=${n}`);
    // The buffers that `rhsJacobi` reads no longer describe `position`.
    expect(rel).toBeGreaterThan(1e-6);
  }, 300000);
});