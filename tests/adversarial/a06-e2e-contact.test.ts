// ADVERSARIAL A06 — is the round-boundary desync REACHABLE end-to-end?
//
// A02 proved that after an accepting GPU-control round the device `gradient`
// no longer describes `position` (~22% of gradient scale) and that
// `contactCount` is left doubled (98 vs 49 live records). A02 drives
// `driver.newtonRound` directly, so the honest next question is whether
// `stepGpuDevice` ever observes the corruption.
//
// HARNESS NOTE (this cost two false positives before it was got right):
// `solver.getPositions()` refreshes `scene.positions`, so re-reading it per run
// starts the SECOND path from the FIRST path's output. The start state must be
// captured ONCE. `tests/webgpu/g6c-newton-e2e.test.ts` Gate B does this
// correctly; an earlier revision of this file did not and reported a spurious
// 8.2e-3 drift that does not exist.
import { describe, it, expect, beforeAll } from "vitest";
import { buildGrid, preprocess } from "../../src/mesh/mesh.js";
import { createScene } from "../../src/physics/scene.js";
import { DEFAULT_MATERIAL } from "../../src/physics/types.js";
import { ContactSystem } from "../../src/collision/contact-assembly.js";
import { DEFAULT_CONTACT_PARAMS } from "../../src/collision/types.js";
import type { ClothScene } from "../../src/physics/scene.js";
import type { DeviceFixture } from "../webgpu/device-setup.js";
import { requireDevice, resetDeviceState, stripScene } from "../webgpu/device-setup.js";

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

/** Resting on a floor with contact already engaged at t=0. */
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

function maxDrift(a: Float64Array, b: Float64Array): number {
  let d = 0;
  for (let i = 0; i < a.length; i++) d = Math.max(d, Math.abs(a[i] - b[i]));
  return d;
}

/** Run `steps` steps of one control path from a FIXED start state. */
async function runPath(
  f: DeviceFixture, gpu: boolean, x0: Float64Array, steps: number, newtonIters: number,
): Promise<{ traj: Float64Array[]; reports: string[] }> {
  const { driver, solver, ex } = f;
  const n = IN(solver).scene.mesh.count;
  resetDeviceState(f, Float64Array.from(x0));
  ex.writeBuffer("velocity", new Float32Array(n * 4));
  driver.cfg.useGpuNewtonControl = gpu;
  const traj: Float64Array[] = [];
  const reports: string[] = [];
  for (let s = 0; s < steps; s++) {
    await solver.stepGpu(1 / 60, { newtonIters });
    const rep = solver.lastStepReport!;
    reports.push(
      `gpu=${gpu ? 1 : 0} ni=${newtonIters} step=${s} newton=${rep.newtonIters} ` +
      `trials=${JSON.stringify(solver.trialHistory)} submits=${rep.submits} syncs=${rep.syncs} ` +
      `E=${rep.energy.toExponential(4)}`,
    );
    traj.push(Float64Array.from(solver.getPositions()));
  }
  return { traj, reports };
}

describe("ADVERSARIAL A06 — end-to-end reachability with engaged contact", () => {
  it("resting-floor: drift across Newton budgets, both paths from one start state", async () => {
    if (!fix) return;
    const f = fix;
    const scene = (f.solver as unknown as { scene: ClothScene }).scene;
    const x0 = Float64Array.from(scene.positions);   // captured ONCE
    const STEPS = 3;

    const summary: string[] = [];
    for (const ni of [2, 6, 10]) {
      const seq = await runPath(f, false, x0, STEPS, ni);
      const gpu = await runPath(f, true, x0, STEPS, ni);
      for (const r of gpu.reports) { /* eslint-disable-next-line no-console */ console.log(`[A06] ${r}`); }
      const drifts: number[] = [];
      for (let s = 0; s < STEPS; s++) drifts.push(maxDrift(seq.traj[s], gpu.traj[s]));
      for (const t of [...seq.traj, ...gpu.traj]) {
        for (const v of t) expect(Number.isFinite(v)).toBe(true);
      }
      summary.push(`ni=${ni} drift=[${drifts.map((d) => d.toExponential(3)).join(", ")}]`);
      // eslint-disable-next-line no-console
      console.log(`[A06] resting-floor @ newtonIters=${ni}: ` +
        drifts.map((d) => d.toExponential(3)).join(", "));
    }
    f.driver.cfg.useGpuNewtonControl = false;
    const motion = maxDrift(x0, (await runPath(f, false, x0, 1, 2)).traj[0]);
    // eslint-disable-next-line no-console
    console.log(`[A06] scene actually moves: ${motion.toExponential(3)}  |  ${summary.join("  |  ")}`);
    expect(motion).toBeGreaterThan(1e-9);
  }, 900000);

  it("control: contact-free strip reproduces Gate B exactly", async () => {
    const ctl = await requireDevice(stripScene, { contactCapacity: 2048, pairCapacity: 8192 });
    if (!ctl) return;
    try {
      const scene = (ctl.solver as unknown as { scene: ClothScene }).scene;
      const x0 = Float64Array.from(scene.positions);
      const a = await runPath(ctl, false, x0, 1, 2);
      const b = await runPath(ctl, true, x0, 1, 2);
      const d = maxDrift(a.traj[0], b.traj[0]);
      // eslint-disable-next-line no-console
      console.log(`[A06] contact-free strip drift = ${d.toExponential(3)} (Gate B reports 0.00e+0)`);
      ctl.driver.cfg.useGpuNewtonControl = false;
      expect(d).toBeLessThan(1e-4);
    } finally {
      ctl.ex.destroy();
    }
  }, 900000);
});