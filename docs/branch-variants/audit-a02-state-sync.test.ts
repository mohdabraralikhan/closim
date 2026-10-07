// ADVERSARIAL A02 — end-to-end proof of the round-boundary state desync.
//
// A01 proved `commit_apply` lands `x_k + 2*alpha*dx` once `position` has
// advanced. This file shows the CONSEQUENCE at the level a Newton round
// actually observes. The contract under test:
//
//   INVARIANT S: after `newtonRound` returns, every device buffer that the
//   next round's `rhsJacobi` reads (contactW/N/Id/Prm/Dist, `gradient`,
//   `contactDiag`, `rhs`) must describe `position`.
//
// `newtonRound`'s next-round refresh is ONLY `rhsJacobi` + `diagnostics` —
// it never rebuilds records. So if the round's final Armijo batch rejects
// after an earlier batch accepted, S is violated and round k+1's gradient,
// Jacobi diagonal, PCG rhs and convergence test all read a phantom state.
//
// The scene must have ENGAGED CONTACT: with empty active sets the records are
// identical in both states and the bug is invisible (this is exactly why the
// existing Gate-B "tight" scenes cannot see it).
//
// READ-ONLY: uses only public driver passes and an explicit re-derivation
// of the reference state for comparison.
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

/** 6x6 patch sitting 1.2 mm above the floor: inside dHat=2mm at t=0. */
function restingFloorScene() {
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
  fix = await requireDevice(restingFloorScene, { contactCapacity: 2048, pairCapacity: 8192 });
}, 180000);

async function readF32(f: DeviceFixture, name: string): Promise<Float32Array> {
  return new Float32Array(await f.ex.readBufferDebug(name, `adv-${name}`, false));
}

/** Reference: what the buffers SHOULD contain for the current `position`. */
async function rederiveAtPosition(f: DeviceFixture, solver: DeviceFixture["solver"]): Promise<void> {
  const { ex, driver } = f;
  driver.zeroContactCounters();
  ex.beginBatch("adv-rederive");
  driver.broadphasePasses(0.002, false);
  driver.contactPasses(IN(solver).contactParamsNow(), "position");
  driver.femPasses(IN(solver).materialNow(), false);
  driver.contactDiagPass();
  driver.rhsAt("position");
  await ex.submitBatch(false);
}

/** Max |a-b| over a device buffer, ignoring the vec4 w lane. */
function maxDiffVec4(a: Float32Array, b: Float32Array): number {
  let d = 0;
  for (let i = 0; i < a.length; i += 4) {
    for (let k = 0; k < 3; k++) d = Math.max(d, Math.abs(a[i + k] - b[i + k]));
  }
  return d;
}
function maxDiff(a: Float32Array, b: Float32Array): number {
  let d = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) d = Math.max(d, Math.abs(a[i] - b[i]));
  return d;
}

describe("ADVERSARIAL A02 — round-boundary state sync", () => {
  it("gpu-control round leaves contact+gradient buffers describing position", async () => {
    if (!fix) return;
    const f = fix;
    const { ex, driver } = f;
    const solver = f.solver;
    solver.configureStep(1 / 60);
    const raw = IN(solver).scene.material;
    const mat = IN(solver).materialNow();
    const beta = Math.max(raw.stretchWarp, raw.stretchWeft, raw.shear) * raw.thickness * 0.1 + 1e-6;

    // Prime: a real E0 evaluation establishes live contact records.
    const ev = await solver.evaluateNewtonState(false, 1, 1 / 60);
    // eslint-disable-next-line no-console
    console.log(`[A02] E0 contacts=${ev.contactCount} minDist=${ev.status.minDistance} ` +
      `energy=${ev.status.energy.toExponential(3)} gradNorm=${ev.status.gradNorm.toExponential(3)}`);
    expect(ev.contactCount).toBeGreaterThan(0); // contact MUST be engaged

    // Seed GPU-control control state exactly like stepNewtonGpuControlled.
    ex.writeBuffer("e0Store", new Float32Array([ev.status.energy, 0, 0, 0]));
    driver.bankF(42, 1e-5);   // NewtonTol
    driver.bankF(43, 0.002);  // NewtonTrust
    ex.writeBuffer("newtonCtl", new Float32Array(16));
    const nst = new Float32Array(20);
    nst[0] = 0;
    ex.writeBuffer("newtonStatus", nst);
    const laggedBefore = await readF32(f, "laggedN");

    const st = await driver.newtonRound({
      round: 0, beta,
      mat,
      contact: IN(solver).contactParamsNow(),
      batchKs: [4, 4, 2],
      evalIndexBase: 0,
    });
    // eslint-disable-next-line no-console
    console.log(`[A02] round0 accepted=${st.armijoAccepted} alpha=${st.selectedAlpha} ` +
      `trialIdx=${st.selectedTrialIndex} barrierFails=${st.barrierFailure} ` +
      `overflow=${st.contactOverflow} ccdFails=${st.ccdFailure}`);

    // Snapshot what the round left behind, then re-derive from `position`.
    const distAsLeft = await readF32(f, "contactDist");
    const wAsLeft = await readF32(f, "contactW");
    const gradAsLeft = await readF32(f, "gradient");
    const diagAsLeft = await readF32(f, "contactDiag");
    const elemAsLeft = await readF32(f, "elementGradient");
    const hingeAsLeft = await readF32(f, "hingeGradient");
    const forceAsLeft = await readF32(f, "contactForce");
    const frictionAsLeft = await readF32(f, "frictionScratch");
    const rhsAsLeft = await readF32(f, "rhs");
    const posAsLeft = await readF32(f, "position");
    const countAsLeft = new Uint32Array(
      await f.ex.readSmall("contactCount", 16, "adv-cc", "scalar"),
    )[0];
    const overflowAsLeft = new Uint32Array(
      await f.ex.readSmall("contactOverflow", 16, "adv-overflow", "scalar"),
    )[0];
    const scannedAsLeft = new Uint32Array(
      await f.ex.readSmall("contactScanned", 16, "adv-scanned", "scalar"),
    )[0];
    const failAsLeft = new Uint32Array(
      await f.ex.readSmall("contactFail", 16, "adv-fail", "scalar"),
    )[0];

    // The accepted trial is differentiated against the pre-commit lagged
    // friction normals; restore that constitutive state for a like-for-like
    // re-derivation (the round then commits its new lagged normals).
    ex.writeBuffer("laggedN", laggedBefore);
    await rederiveAtPosition(f, solver);

    const distRef = await readF32(f, "contactDist");
    const wRef = await readF32(f, "contactW");
    const gradRef = await readF32(f, "gradient");
    const diagRef = await readF32(f, "contactDiag");
    const elemRef = await readF32(f, "elementGradient");
    const hingeRef = await readF32(f, "hingeGradient");
    const forceRef = await readF32(f, "contactForce");
    const frictionRef = await readF32(f, "frictionScratch");
    const rhsRef = await readF32(f, "rhs");
    const posRef = await readF32(f, "position");

    const dDist = maxDiff(distAsLeft, distRef);
    const dW = maxDiffVec4(wAsLeft, wRef);
    const dGrad = maxDiff(gradAsLeft, gradRef);
    const dDiag = maxDiff(diagAsLeft, diagRef);
    const dPos = maxDiffVec4(posAsLeft, posRef);

    // eslint-disable-next-line no-console
    console.log(`[A02] SYNC: |position|=${dPos.toExponential(3)} |contactW|=${dW.toExponential(3)} ` +
      `|contactDist|=${dDist.toExponential(3)} |gradient|=${dGrad.toExponential(3)} ` +
      `|contactDiag|=${dDiag.toExponential(3)} countLeft=${countAsLeft}`);
    // eslint-disable-next-line no-console
    console.log(`[A02] scale: |gradRef|max=${Math.max(...Array.from(gradRef.slice(0, 36)).map(Math.abs)).toExponential(3)} ` +
      `|distRef|max=${Math.max(...Array.from(distRef).map(Math.abs)).toExponential(3)}`);
    console.log(`[A02] components elem=${maxDiff(elemAsLeft, elemRef)} hinge=${maxDiff(hingeAsLeft, hingeRef)} ` +
      `force=${maxDiff(forceAsLeft, forceRef)} friction=${maxDiff(frictionAsLeft, frictionRef)} rhs=${maxDiff(rhsAsLeft, rhsRef)}`);

    // `position` itself must be untouched by the re-derivation (sanity).
    expect(dPos).toBe(0);
    // One accepted xTrial must yield exactly one rebuilt contact set and a
    // self-consistent derivative state at the committed position.
    expect(st.armijoAccepted).toBe(true);
    expect(countAsLeft).toBe(ev.contactCount);
    expect(overflowAsLeft).toBe(0);
    expect(scannedAsLeft).toBeGreaterThan(0);
    expect(failAsLeft).toBe(0);
    expect(dW).toBe(0);
    expect(dDist).toBeLessThan(1e-6);
    expect(dGrad).toBeLessThan(1e-6);
    expect(dDiag).toBeLessThan(1e-6);

    // Next Newton round with an intentionally impossible distance threshold:
    // all Armijo batches reject, but the final rebuild still leaves one clean
    // contact set at the unchanged accepted position.
    IN(solver).contactCountNow = driver.c.cap;
    IN(solver).simImage.newtonIteration = 1;
    IN(solver).pushSimParams();
    ex.writeBuffer("newtonCtl", new Float32Array(16));
    ex.writeBuffer("newtonStatus", new Float32Array(20));
    const rejected = await driver.newtonRound({
      round: 1, beta, mat,
      contact: { ...IN(solver).contactParamsNow(), dMin: 0.01 },
      batchKs: [4, 4, 2], evalIndexBase: 100,
    });
    expect(rejected.armijoAccepted).toBe(false);
    expect(rejected.failure).toBe(true);
    const rejectCount = new Uint32Array(
      await ex.readSmall("contactCount", 16, "adv-reject-count", "scalar"),
    )[0];
    const rejectOverflow = new Uint32Array(
      await ex.readSmall("contactOverflow", 16, "adv-reject-overflow", "scalar"),
    )[0];
    const rejectScanned = new Uint32Array(
      await ex.readSmall("contactScanned", 16, "adv-reject-scanned", "scalar"),
    )[0];
    const rejectFail = new Uint32Array(
      await ex.readSmall("contactFail", 16, "adv-reject-fail", "scalar"),
    )[0];
    expect(rejectCount).toBe(ev.contactCount);
    expect(rejectOverflow).toBe(0);
    expect(rejectScanned).toBeGreaterThan(0);
    expect(rejectFail).toBe(0);
  }, 600000);

  it("CONTROL: the same invariant holds when every batch accepts (no trailing reject)", async () => {
    if (!fix) return;
    const f = fix;
    const { ex, driver } = f;
    const solver = f.solver;
    solver.configureStep(1 / 60);
    const raw = IN(solver).scene.material;
    const mat = IN(solver).materialNow();
    const beta = Math.max(raw.stretchWarp, raw.stretchWeft, raw.shear) * raw.thickness * 0.1 + 1e-6;

    const ev = await solver.evaluateNewtonState(false, 11, 1 / 60);
    expect(ev.contactCount).toBeGreaterThan(0);
    ex.writeBuffer("e0Store", new Float32Array([ev.status.energy, 0, 0, 0]));
    driver.bankF(42, 1e-5);
    driver.bankF(43, 0.002);
    ex.writeBuffer("newtonCtl", new Float32Array(16));
    ex.writeBuffer("newtonStatus", new Float32Array(20));
    const laggedBefore = await readF32(f, "laggedN");

    // Single batch of width 1: if it accepts, the round ENDS on an accepted
    // batch, so commit_apply ran against the pre-commit `position`.
    const st = await driver.newtonRound({
      round: 0, beta,
      mat,
      contact: IN(solver).contactParamsNow(),
      batchKs: [1],
      evalIndexBase: 0,
    });
    // eslint-disable-next-line no-console
    console.log(`[A02-control] batchKs=[1] accepted=${st.armijoAccepted} alpha=${st.selectedAlpha}`);

    const gradAsLeft = await readF32(f, "gradient");
    const diagAsLeft = await readF32(f, "contactDiag");
    ex.writeBuffer("laggedN", laggedBefore);
    await rederiveAtPosition(f, solver);
    const gradRef = await readF32(f, "gradient");
    const diagRef = await readF32(f, "contactDiag");
    const dGrad = maxDiff(gradAsLeft, gradRef);
    const dDiag = maxDiff(diagAsLeft, diagRef);
    // eslint-disable-next-line no-console
    console.log(`[A02-control] SYNC: |gradient|=${dGrad.toExponential(3)} |contactDiag|=${dDiag.toExponential(3)}`);

    if (st.armijoAccepted) {
      // Accepted on the ONLY (final) batch => no trailing reject => must be clean.
      // This passes today, which is exactly why the bug is invisible to the
      // existing reject-all gate (there nbAlpha stays 0).
      expect(dGrad).toBeLessThan(1e-6);
      expect(dDiag).toBeLessThan(1e-6);
    } else {
      // eslint-disable-next-line no-console
      console.log("[A02-control] batch rejected all — control inconclusive");
    }
  }, 600000);
});
