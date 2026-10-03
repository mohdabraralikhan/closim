// G6C S1 invariant: after `newtonRound` returns accepted, every device
// buffer that describes the accepted `position` must actually describe that
// same position (contact records, elementGradient, gradient, contactDiag,
// rhs). The test rebuilds G1/G2/FEM from `position` and compares
// contactCount / contactDist / gradient against the buffers the round left
// behind — a direct state comparison, not a trajectory comparison.
//
// Lagged-friction semantics (load-bearing for this test): the committed
// gradient is baked with the ROUND-START laggedN, exactly like the sequential
// golden path (trials linearize friction about the previous commit; the
// update lands in laggedN for FUTURE trials). The reference re-derivation
// therefore restores the round-start laggedN first; comparing against a
// re-derivation with post-commit laggedN measures formulation timing, not
// sync, and differs at friction scale.
import { describe, it, expect, beforeAll } from "vitest";
import { buildGrid, preprocess } from "../../src/mesh/mesh.js";
import { createScene } from "../../src/physics/scene.js";
import { DEFAULT_MATERIAL } from "../../src/physics/types.js";
import { ContactSystem } from "../../src/collision/contact-assembly.js";
import { DEFAULT_CONTACT_PARAMS } from "../../src/collision/types.js";
import type { DeviceFixture } from "./device-setup.js";
import { requireDevice } from "./device-setup.js";

let fix: DeviceFixture | null = null;

/** 6x6 patch 1.2 mm above the floor: contact engaged at t=0 (dHat = 2 mm). */
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
  return new Float32Array(await f.ex.readBufferDebug(name, `sync-${name}`, false));
}

// Access to WebGpuSolver step internals (same pattern as the Gate A tests).
type SyncInternals = {
  materialNow(): { c00: number; c11: number; c01: number; g: number; thickness: number };
  contactParamsNow(): {
    dHat: number; kappa: number; mu: number; fricEps: number;
    floorY: number; floorOn: number; dMin: number; contactCapacity: number;
  };
};
const IN = (solver: unknown): SyncInternals => solver as SyncInternals;

function maxDiff(a: Float32Array, b: Float32Array): number {
  let d = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) d = Math.max(d, Math.abs(a[i] - b[i]));
  return d;
}

/** Sorted-live comparison: contact slots are append-ordered and symmetric
 *  contacts can permute slots run-to-run (benign G2 race, invisible to all
 *  per-vertex quantities); the invariant is multiset equality of the live set. */
function liveSortedDist(all: Float32Array, count: number): Float32Array {
  return Float32Array.from(all.slice(0, count)).sort();
}

describe("G6C S1 round-boundary state sync", () => {
  it("accepted round leaves contact+gradient buffers describing position", async () => {
    if (!fix) return;
    const f = fix;
    const { ex, driver } = f;
    const solver = f.solver;
    solver.configureStep(1 / 60);
    const mat = IN(solver).materialNow();
    const rawMt = (solver as unknown as { scene: { material: Record<string, number> } }).scene.material;
    const beta = Math.max(rawMt.stretchWarp, rawMt.stretchWeft, rawMt.shear) * rawMt.thickness * 0.1 + 1e-6;

    const ev = await solver.evaluateNewtonState(false, 1, 1 / 60);
    expect(ev.contactCount).toBeGreaterThan(0);
    // Round-start laggedN: the commit rebuild linearizes friction about this.
    const laggedStart = await readF32(f, "laggedN");

    ex.writeBuffer("e0Store", new Float32Array([ev.status.energy, 0, 0, 0]));
    driver.bankF(42, 1e-5);
    driver.bankF(43, 0.002);
    ex.writeBuffer("newtonCtl", new Float32Array(16));
    ex.writeBuffer("newtonStatus", new Float32Array(20));

    const st = await driver.newtonRound({
      round: 0, beta, mat, contact: IN(solver).contactParamsNow(),
      batchKs: [4, 4, 2],
      evalIndexBase: 0,
    });
    expect(st.armijoAccepted).toBe(true);
    // eslint-disable-next-line no-console
    console.log(`[g6c-sync] accepted alpha=${st.selectedAlpha} trialIdx=${st.selectedTrialIndex}`);

    const countAsLeft = new Uint32Array(
      await ex.readSmall("contactCount", 16, "sync-cc", "scalar"),
    )[0];
    const distAsLeft = await readF32(f, "contactDist");
    const gradAsLeft = await readF32(f, "gradient");
    const diagAsLeft = await readF32(f, "contactDiag");
    const elemAsLeft = await readF32(f, "elementGradient");

    // Reference: G1/G2/FEM re-derived from `position` under the SAME laggedN
    // the commit rebuild used (round-start), then compared slot-for-slot.
    ex.writeBuffer("laggedN", laggedStart);
    driver.zeroContactCounters();
    ex.beginBatch("sync-rederive");
    driver.broadphasePasses(0.002, false);
    driver.contactPasses(IN(solver).contactParamsNow(), "position");
    driver.femPasses(IN(solver).materialNow(), false);
    driver.contactDiagPass();
    driver.rhsAt("position");
    await ex.submitBatch(false);

    const countRef = new Uint32Array(
      await ex.readSmall("contactCount", 16, "sync-cc-ref", "scalar"),
    )[0];
    const distRef = await readF32(f, "contactDist");
    const gradRef = await readF32(f, "gradient");
    const diagRef = await readF32(f, "contactDiag");
    const elemRef = await readF32(f, "elementGradient");

    const dDist = maxDiff(liveSortedDist(distAsLeft, countAsLeft), liveSortedDist(distRef, countRef));
    const dGrad = maxDiff(gradAsLeft, gradRef);
    const dDiag = maxDiff(diagAsLeft, diagRef);
    const dElem = maxDiff(elemAsLeft, elemRef);
    let gScale = 0;
    for (let i = 0; i < gradRef.length; i++) gScale = Math.max(gScale, Math.abs(gradRef[i]));
    // eslint-disable-next-line no-console
    console.log(`[g6c-sync] count ${countAsLeft} vs ${countRef} |contactDist|=${dDist.toExponential(3)} ` +
      `|gradient|=${dGrad.toExponential(3)} (scale ${gScale.toExponential(3)}) ` +
      `|contactDiag|=${dDiag.toExponential(3)} |elementGradient|=${dElem.toExponential(3)}`);

    // Same state + same laggedN through the same kernels: identical records.
    expect(countAsLeft).toBe(countRef);
    expect(dDist).toBeLessThan(1e-6);
    expect(dDiag).toBeLessThan(1e-6);
    expect(dElem).toBeLessThan(1e-6);
    // Gradient rides the same assembly; atomic-append order across two
    // rebuilds can permute slots, so the bar is tight but not bitwise (the
    // pre-fix desync measured 1.4e-2 here — 3+ orders above this bar).
    expect(dGrad).toBeLessThan(1e-5);
  }, 600000);
});
