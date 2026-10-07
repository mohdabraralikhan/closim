// ADVERSARIAL A04 — stage-by-stage NaN bisection on a CONTACT-ENGAGED trial.
//
// A03 showed that on a resting-floor scene a GPU-control round leaves
// `gradient` and `rhs` entirely non-finite (147/147), and that every candidate
// is rejected with diagnostics reporting (gradNorm = 0, finite = 0). Per
// `diagnostics.wgsl` that pair is the exact signature of a NaN in `gradient`:
//
//     let g = gradient[i];
//     if (is_nan(g)) { fin = 0.0; } else { g2 = g * g; }
//     ...
//     statusOut[2] = sqrt(tG[0]);   // => 0
//
// So the bug is NOT in the line search — it is that assembling the gradient at
// the trial state produces non-finite values whenever CONTACT is engaged.
//
// This file bisects the existing driver passes (no production change) to find
// the exact pass that first introduces a non-finite lane.
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

function bad(a: Float32Array): { n: number; first: number } {
  let n = 0; let first = -1;
  for (let i = 0; i < a.length; i++) {
    if (!Number.isFinite(a[i])) { n++; if (first < 0) first = i; }
  }
  return { n, first };
}
function maxAbs(a: Float32Array): number {
  let m = 0;
  for (const v of a) if (Number.isFinite(v) && Math.abs(v) > m) m = Math.abs(v);
  return m;
}

const WATCH = [
  "gradient", "rhs", "diag", "contactDiag", "contactForce", "contactEnergy",
  "elementGradient", "elementEnergy", "hingeGradient", "xTrial", "dmInv", "restArea",
];

async function report(f: DeviceFixture, tag: string): Promise<Record<string, number>> {
  const out: Record<string, number> = {};
  const parts: string[] = [];
  for (const b of WATCH) {
    const a = await readF32(f, b);
    const s = bad(a);
    out[b] = s.n;
    parts.push(`${b}=${s.n}`);
  }
  // Material uniform slots 24..28 + OutSel(29) + Thickness(19) as f32.
  const bank = await readF32(f, "uniformBank");
  const slot = (i: number): string => bank[i * 4].toExponential(3);
  parts.push(`BANK[c00=${slot(24)},c11=${slot(25)},c01=${slot(26)},g=${slot(27)},t=${slot(28)},outSel=${slot(29)}]`);
  // eslint-disable-next-line no-console
  console.log(`[A04] ${tag.padEnd(30)} ${parts.join(" ")}`);
  return out;
}

describe("ADVERSARIAL A04 — where does the trial-state gradient go non-finite?", () => {
  it("bisects the trial evaluation pass by pass", async () => {
    if (!fix) return;
    const f = fix;
    const { ex, driver } = f;
    const solver = f.solver;
    solver.configureStep(1 / 60);
    const mat = IN(solver).materialNow();
    const contact = IN(solver).contactParamsNow();

    // 1. Clean reference state at `position`.
    const ev = await solver.evaluateNewtonState(false, 41, 1 / 60);
    // eslint-disable-next-line no-console
    console.log(`[A04] E0 contacts=${ev.contactCount} finite=${ev.status.finite} ` +
      `gradNorm=${ev.status.gradNorm.toExponential(3)} minD=${ev.status.minDistance}`);
    await report(f, "after E0 (position)");

    // 2. Prime the candidate machinery exactly as a GPU-control round does.
    ex.writeBuffer("e0Store", new Float32Array([ev.status.energy, 0, 0, 0]));
    ex.writeBuffer("gtdxStore", new Float32Array([-1e-2, 0, 0, 0]));
    ex.writeBuffer("trustScaleStore", new Float32Array([1.0, 0, 0, 0]));
    ex.writeBuffer("newtonCtl", new Float32Array(16));
    const n = driver.c.n;
    const dx = new Float32Array(n * 3);
    for (let i = 0; i < n * 3; i++) dx[i] = 1e-4 * Math.sin(i);
    ex.writeBuffer("searchDirection", dx);
    ex.writeBuffer("armijoAlphas", new Float32Array([1, 0.5, 0.25, 0.125, 1, 1, 1, 1]));

    const step = async (tag: string, fn: () => void) => {
      ex.beginBatch(`adv-${tag}`);
      fn();
      await ex.submitBatch(false);
      await report(f, tag);
    };

    await step("A: apply_0", () => {
      ex.runPass({
        shader: "armijo", entry: "apply_0",
        groups: [[
          { binding: 0, buffer: "simParams" }, { binding: 2, buffer: "searchDirection" },
          { binding: 4, buffer: "position" }, { binding: 5, buffer: "xTrial" },
          { binding: 6, buffer: "pinMask" }, { binding: 7, buffer: "pinPos" },
          { binding: 8, buffer: "armijoAlphas" }, { binding: 9, buffer: "armijoCur" },
          { binding: 10, buffer: "trustScaleStore" },
        ]],
        x: Math.max(1, Math.ceil(n / 64)),
      });
    });

    await step("B: + clear records", () => driver.armijoClearPass());
    await step("C: + broadphase(xTrial)", () => driver.broadphasePasses(0.002, true));
    await step("D: + contact(xTrial)", () => driver.contactPasses(contact, "xTrial"));

    const cnt = new Uint32Array(await ex.readSmall("contactCount", 16, "adv-cc", "scalar"))[0];
    const dist = await readF32(f, "contactDist");
    const prm = await readF32(f, "contactPrm");
    const ovf = new Uint32Array(await ex.readSmall("contactOverflow", 16, "adv-ovf", "scalar"))[0];
    let live = 0; let nanPrm = 0; let zeroDHat = 0; let minD = 1e30;
    for (let i = 0; i < cnt; i++) {
      if (dist[i] < 1e29) { live++; minD = Math.min(minD, dist[i]); }
      if (!Number.isFinite(prm[i * 4])) nanPrm++;
      if (prm[i * 4] <= 0) zeroDHat++;
    }
    // eslint-disable-next-line no-console
    console.log(`[A04] records: count=${cnt} live=${live} minLiveDist=${minD.toExponential(4)} ` +
      `overflow=${ovf} zeroDHatSlots=${zeroDHat} nonfinitePrm=${nanPrm}`);

    await step("E: + fem(xTrial)", () => driver.femPasses(mat, true));
    await step("F: + contactDiag", () => driver.contactDiagPass());
    await step("G: + rhsAt(xTrial)", () => driver.rhsAt("xTrial"));
    await step("H: + diagnostics", () => driver.diagnosticsPasses("xTrial"));

    const st = await readF32(f, "solverStatus");
    // eslint-disable-next-line no-console
    console.log(`[A04] solverStatus: E=${st[0].toExponential(4)} barrierE=${st[1].toExponential(3)} ` +
      `gradNorm=${st[2].toExponential(4)} minD=${st[4].toExponential(4)} minTOI=${st[5]} ` +
      `finite=${st[6]} ccdSafe=${st[7]} barrierSafe=${st[8]}`);
    // The round rejected every candidate precisely because of this pair.
    expect(Number.isFinite(st[2])).toBe(true);
    expect(st[6]).toBe(1);
  }, 600000);

  it("control: same bisection with contact DISABLED (floorOn = 0)", async () => {
    if (!fix) return;
    const f = fix;
    const { ex, driver } = f;
    const solver = f.solver;
    solver.configureStep(1 / 60);
    const mat = IN(solver).materialNow();
    const contact = { ...IN(solver).contactParamsNow(), floorOn: 0 };

    const ev = await solver.evaluateNewtonState(false, 51, 1 / 60);
    ex.writeBuffer("e0Store", new Float32Array([ev.status.energy, 0, 0, 0]));
    ex.writeBuffer("gtdxStore", new Float32Array([-1e-2, 0, 0, 0]));
    ex.writeBuffer("trustScaleStore", new Float32Array([1.0, 0, 0, 0]));
    const n = driver.c.n;
    const dx = new Float32Array(n * 3);
    for (let i = 0; i < n * 3; i++) dx[i] = 1e-4 * Math.sin(i);
    ex.writeBuffer("searchDirection", dx);
    ex.writeBuffer("armijoAlphas", new Float32Array([1, 0.5, 0.25, 0.125, 1, 1, 1, 1]));

    ex.beginBatch("adv-nocontact");
    ex.runPass({
      shader: "armijo", entry: "apply_0",
      groups: [[
        { binding: 0, buffer: "simParams" }, { binding: 2, buffer: "searchDirection" },
        { binding: 4, buffer: "position" }, { binding: 5, buffer: "xTrial" },
        { binding: 6, buffer: "pinMask" }, { binding: 7, buffer: "pinPos" },
        { binding: 8, buffer: "armijoAlphas" }, { binding: 9, buffer: "armijoCur" },
        { binding: 10, buffer: "trustScaleStore" },
      ]],
      x: Math.max(1, Math.ceil(n / 64)),
    });
    driver.armijoClearPass();
    driver.broadphasePasses(0.002, true);
    driver.contactPasses(contact, "xTrial");
    driver.femPasses(mat, true);
    driver.contactDiagPass();
    driver.rhsAt("xTrial");
    driver.diagnosticsPasses("xTrial");
    await ex.submitBatch(false);

    await report(f, "no-contact trial eval");
    const st = await readF32(f, "solverStatus");
    // eslint-disable-next-line no-console
    console.log(`[A04] no-contact solverStatus: E=${st[0].toExponential(4)} gradNorm=${st[2].toExponential(4)} ` +
      `minD=${st[4]} finite=${st[6]}`);
    expect(Number.isFinite(st[2])).toBe(true);
    expect(st[6]).toBe(1);
  }, 600000);
});