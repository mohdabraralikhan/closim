// ADVERSARIAL A10 — Memory poisoning: stale records, stale laggedN, and un-cleared tail slots.
//
// Attack targets:
//   1. armijo_clear_records guard bug: it guards `if (i >= params.contactCount) return;`
//      instead of `contactCapacity`. If contactCount == 0, NOT A SINGLE SLOT is cleared!
//      Old records remain forever!
//   2. Stale laggedN persistence across contact / no-contact transitions:
//      When cloth leaves contact, laggedN retains stale normals and lambdaN.
//      Friction passes reading index i pair with unrelated contact stencils.
//   3. Counter doubling in rebuildTrialPasses:
//      Baseline G6 rebuildTrialPasses does not zero counters before broadphase/contact.
//      contactCount doubles, corrupting force and Hessian diagonal assemblies.
import { describe, it, expect, beforeAll } from "vitest";
import { buildGrid, preprocess } from "../../src/mesh/mesh.js";
import { createScene } from "../../src/physics/scene.js";
import { DEFAULT_MATERIAL } from "../../src/physics/types.js";
import { ContactSystem } from "../../src/collision/contact-assembly.js";
import { DEFAULT_CONTACT_PARAMS } from "../../src/collision/types.js";
import type { DeviceFixture } from "../webgpu/device-setup.js";
import { requireDevice } from "../webgpu/device-setup.js";

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

function restingFloorScene() {
  const g = buildGrid(4, 4, 0.1, 0.1);
  for (let i = 0; i < g.positions.length / 3; i++) g.positions[i * 3 + 1] += 0.002;
  const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
  const scene = createScene(mesh, { ...DEFAULT_MATERIAL }, [0, -9.81, 0]);
  const c = new ContactSystem({ ...DEFAULT_CONTACT_PARAMS }, mesh.indices);
  c.setFloor(0);
  scene.contact = c;
  return scene;
}

beforeAll(async () => {
  fix = await requireDevice(restingFloorScene, { contactCapacity: 256, pairCapacity: 1024 });
}, 180000);

async function readF32(f: DeviceFixture, name: string): Promise<Float32Array> {
  return new Float32Array(await f.ex.readBufferDebug(name, `adv-${name}`, false));
}

describe("ADVERSARIAL A10 — Memory poisoning & stale records", () => {
  it("A10.1 armijo_clear_records fails to clear when contactCount == 0 (guard bug)", async () => {
    if (!fix) return;
    const { ex, driver } = fix;
    const cap = driver.c.cap;

    // Poison contactDist with sentinel -999.0
    const poisoned = new Float32Array(cap).fill(-999.0);
    ex.writeBuffer("contactDist", poisoned);

    // Set simParams.contactCount = 0 (as is normal when no contacts have been detected yet!)
    IN(fix.solver).contactCountNow = 0;
    IN(fix.solver).pushSimParams();

    // Run armijoClearPass
    ex.beginBatch("adv-clear-empty");
    driver.armijoClearPass();
    await ex.submitBatch(false);

    const distAfter = await readF32(fix, "contactDist");

    // eslint-disable-next-line no-console
    console.log(`[A10] armijo_clear_records with contactCount=0: dist[0]=${distAfter[0]} dist[1]=${distAfter[1]}`);

    // If armijo_clear_records correctly cleared, dist[0] would be 1e30.
    // BUG: Because it checked `if (i >= params.contactCount) return;`, dist[0] is STILL -999.0!
    expect(distAfter[0]).toBe(-999.0);
  }, 180000);

  it("A10.2 armijo_clear_records only clears up to contactCount, leaving stale tail slots", async () => {
    if (!fix) return;
    const { ex, driver } = fix;
    const cap = driver.c.cap;

    // Poison all slots with -777.0
    ex.writeBuffer("contactDist", new Float32Array(cap).fill(-777.0));

    // Suppose previous step had 10 contacts
    IN(fix.solver).contactCountNow = 10;
    IN(fix.solver).pushSimParams();

    ex.beginBatch("adv-clear-partial");
    driver.armijoClearPass();
    await ex.submitBatch(false);

    const distAfter = await readF32(fix, "contactDist");

    // Slots 0..9 were cleared to 1e30
    expect(distAfter[0]).toBeGreaterThan(1e29);
    expect(distAfter[9]).toBeGreaterThan(1e29);

    // BUG: Slot 10..cap were NOT cleared and remain -777.0!
    // Any cap-bounded loop (like friction or barrier) that scans past contactCount sees stale data!
    expect(distAfter[10]).toBe(-777.0);
    expect(distAfter[cap - 1]).toBe(-777.0);
  }, 180000);

  it("A10.3 stale laggedN persists when transitioning from contact to no-contact", async () => {
    if (!fix) return;
    const { ex, driver } = fix;
    const cap = driver.c.cap;

    // Frame 1: Simulate active contact at index 0: normal (0, 1, 0), lambdaN = 50.0
    const lagInit = new Float32Array(cap * 4);
    lagInit[0] = 0.0; lagInit[1] = 1.0; lagInit[2] = 0.0; lagInit[3] = 50.0;
    ex.writeBuffer("laggedN", lagInit);

    // Frame 2: Cloth is pulled away from floor. No contacts exist.
    IN(fix.solver).contactCountNow = 0;
    IN(fix.solver).pushSimParams();

    // Newton round runs with no contacts.
    // Does commit_lagged_if or anything clear laggedN?
    // In commit_lagged_if: `if (i >= clParams.contactCount) return;`
    // Since contactCount is 0, it immediately returns!
    ex.writeBuffer("newtonCtl", new Float32Array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])); // doCommit=1
    ex.beginBatch("adv-lagged-commit-empty");
    ex.runPass({
      shader: "newton-control", entry: "commit_lagged_if",
      groups: [[
        { binding: 70, buffer: "contactN" }, { binding: 71, buffer: "contactDist" },
        { binding: 72, buffer: "contactPrm" }, { binding: 73, buffer: "laggedN" },
        { binding: 74, buffer: "newtonCtl" }, { binding: 75, buffer: "simParams" },
      ]],
      x: Math.max(1, Math.ceil(cap / 64)),
    });
    await ex.submitBatch(false);

    const lagAfter = await readF32(fix, "laggedN");

    // eslint-disable-next-line no-console
    console.log(`[A10] laggedN[0] after empty commit: n=(${lagAfter[0]}, ${lagAfter[1]}, ${lagAfter[2]}) lambdaN=${lagAfter[3]}`);

    // STALE STATE: laggedN[0] is still (0, 1, 0, 50.0)!
    expect(lagAfter[3]).toBe(50.0);
  }, 180000);

  it("A10.4 rebuildTrialPasses without clear doubles contactCount and appends duplicate contacts", async () => {
    if (!fix) return;
    const { ex, driver } = fix;
    const solver = fix.solver;
    solver.configureStep(1 / 60);

    const readCC = async (): Promise<number> =>
      new Uint32Array(await ex.readSmall("contactCount", 16, "adv-cc-read", "scalar"))[0];

    // Evaluate state at position
    const ev = await solver.evaluateNewtonState(false, 101, 1 / 60);
    const count0 = await readCC();
    expect(count0).toBe(ev.contactCount);

    // Call rebuildTrialPasses WITHOUT calling zeroContactCounters (exactly what baseline newtonRound does)
    ex.beginBatch("adv-rebuild-append");
    driver.rebuildTrialPasses(IN(solver).materialNow(), IN(solver).contactParamsNow());
    await ex.submitBatch(false);

    const count1 = await readCC();
    // eslint-disable-next-line no-console
    console.log(`[A10] contactCount before rebuild: ${count0}, after un-zeroed rebuild: ${count1}`);

    // In baseline 162a689, count1 == count0 * 2!
    expect(count1).toBe(count0 * 2);
  }, 180000);
});
