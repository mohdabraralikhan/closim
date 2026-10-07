// ADVERSARIAL A05 — isolate the membrane/bending pass at a trial state.
//
// A04 bisected `femPasses(atTrial=true)` on a resting-floor scene and showed
// `elementGradient` (648/648) and `elementEnergy` (72/72) go non-finite while
// `xTrial`, `dmInv` and `restArea` stay finite. That is the signature of a
// non-finite UNIFORM (every element affected, not a subset), since the
// membrane kernel is a pure function of (position, triangles, dmInv, restArea,
// matC00, matC11, matC01, matG, matThickness, outSel).
//
// This file replicates `membrane-gradient:main` one dispatch at a time with
// test-local bindings (no production change) so the failing pass AND the
// offending uniform are identified directly.
import { describe, it, beforeAll } from "vitest";
import { buildGrid, preprocess } from "../../src/mesh/mesh.js";
import { createScene } from "../../src/physics/scene.js";
import { DEFAULT_MATERIAL } from "../../src/physics/types.js";
import { ContactSystem } from "../../src/collision/contact-assembly.js";
import { DEFAULT_CONTACT_PARAMS } from "../../src/collision/types.js";
import { GpuUniformSlot, UNIFORM_SLOT_STRIDE } from "../../src/backend/webgpu/gpu-buffers.js";
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
function nBad(a: Float32Array): number {
  let n = 0;
  for (const v of a) if (!Number.isFinite(v)) n++;
  return n;
}

describe("ADVERSARIAL A05 — membrane pass isolation", () => {
  it("runs membrane-gradient alone at position and at xTrial", async () => {
    if (!fix) return;
    const f = fix;
    const { ex, driver } = f;
    const solver = f.solver;
    solver.configureStep(1 / 60);
    const raw = IN(solver).scene.material;
    const mat = IN(solver).materialNow();
    // eslint-disable-next-line no-console
    console.log(`[A05] materialNow() = ${JSON.stringify(mat)}`);
    // eslint-disable-next-line no-console
    console.log(`[A05] scene.material = ${JSON.stringify(raw)}`);

    const B = (binding: number, buffer: string) => ({ binding, buffer });
    const bg = (binding: number, slot: number) => ({
      binding, buffer: "uniformBank", offset: slot * UNIFORM_SLOT_STRIDE, size: 16,
    });

    // Prime a real state and a trial state.
    const ev = await solver.evaluateNewtonState(false, 61, 1 / 60);
    ex.writeBuffer("searchDirection", new Float32Array(driver.c.n * 3).fill(1e-5));
    ex.writeBuffer("armijoAlphas", new Float32Array([1, 0.5, 1, 1, 1, 1, 1, 1]));
    ex.writeBuffer("trustScaleStore", new Float32Array([1, 0, 0, 0]));

    const membrane = (posBuf: string) => {
      driver.bankF(GpuUniformSlot.MatC00, mat.c00);
      driver.bankF(GpuUniformSlot.MatC11, mat.c11);
      driver.bankF(GpuUniformSlot.MatC01, mat.c01);
      driver.bankF(GpuUniformSlot.MatG, mat.g);
      driver.bankF(GpuUniformSlot.MatThickness, mat.thickness);
      driver.bankU(GpuUniformSlot.OutSel, 0);
      ex.runPass({
        shader: "membrane-gradient", entry: "main",
        groups: [[
          B(0, "simParams"), B(1, posBuf), B(2, "triangles"), B(3, "dmInv"),
          B(4, "restArea"), B(6, "elementGradient"), B(7, "elementEnergy"),
          B(13, "elementGradientB"), bg(14, GpuUniformSlot.OutSel),
          bg(8, GpuUniformSlot.MatC00), bg(9, GpuUniformSlot.MatC11),
          bg(10, GpuUniformSlot.MatC01), bg(11, GpuUniformSlot.MatG),
          bg(12, GpuUniformSlot.MatThickness),
        ]],
        x: Math.max(1, Math.ceil(driver.c.m / 64)),
      });
    };

    const run = async (tag: string, posBuf: string) => {
      // Poison elementEnergy/elementGradient first so a skipped write is visible.
      ex.writeBuffer("elementEnergy", new Float32Array(driver.c.m).fill(NaN));
      ex.writeBuffer("elementGradient", new Float32Array(driver.c.m * 9).fill(NaN));
      ex.beginBatch(`adv-${tag}`);
      membrane(posBuf);
      await ex.submitBatch(false);
      const ee = nBad(await readF32(f, "elementEnergy"));
      const eg = nBad(await readF32(f, "elementGradient"));
      const eeVals = await readF32(f, "elementEnergy");
      // eslint-disable-next-line no-console
      console.log(`[A05] ${tag.padEnd(22)} elementEnergy=${ee}/${driver.c.m} elementGradient=${eg}/${driver.c.m * 9} ` +
        `e[0]=${eeVals[0]} e[1]=${eeVals[1]}`);
      return { ee, eg };
    };

    const atPos = await run("membrane @ position", "position");
    // Build xTrial = position + 1e-5 per DOF.
    ex.beginBatch("adv-make-trial");
    ex.runPass({
      shader: "armijo", entry: "apply_0",
      groups: [[
        B(0, "simParams"), B(2, "searchDirection"), B(4, "position"), B(5, "xTrial"),
        B(6, "pinMask"), B(7, "pinPos"), B(8, "armijoAlphas"), B(9, "armijoCur"),
        B(10, "trustScaleStore"),
      ]],
      x: Math.max(1, Math.ceil(driver.c.n / 64)),
    });
    await ex.submitBatch(false);
    const atTrial = await run("membrane @ xTrial", "xTrial");
    // eslint-disable-next-line no-console
    console.log(`[A05] E0 energy=${ev.status.energy.toExponential(4)} contacts=${ev.contactCount}`);
  }, 600000);

  it("checks whether forcing a real uniform write changes the outcome", async () => {
    if (!fix) return;
    const f = fix;
    const { ex, driver } = f;
    const solver = f.solver;
    solver.configureStep(1 / 60);
    const mat = IN(solver).materialNow();
    const B = (binding: number, buffer: string) => ({ binding, buffer });
    const bg = (binding: number, slot: number) => ({
      binding, buffer: "uniformBank", offset: slot * UNIFORM_SLOT_STRIDE, size: 16,
    });

    // Sentinel-then-real: forces TWO genuine writeBuffer calls, defeating the
    // G4C write-coalescing cache for these slots.
    const forced = async (posBuf: string, tag: string) => {
      ex.writeBuffer("elementEnergy", new Float32Array(driver.c.m).fill(NaN));
      ex.writeBuffer("elementGradient", new Float32Array(driver.c.m * 9).fill(NaN));
      ex.beginBatch(`adv-forced-${tag}`);
      for (const [slot, v] of [
        [GpuUniformSlot.MatC00, -1], [GpuUniformSlot.MatC11, -1], [GpuUniformSlot.MatC01, -1],
        [GpuUniformSlot.MatG, -1], [GpuUniformSlot.MatThickness, -1],
      ] as Array<[number, number]>) driver.bankF(slot, v);
      for (const [slot, v] of [
        [GpuUniformSlot.MatC00, mat.c00], [GpuUniformSlot.MatC11, mat.c11],
        [GpuUniformSlot.MatC01, mat.c01], [GpuUniformSlot.MatG, mat.g],
        [GpuUniformSlot.MatThickness, mat.thickness],
      ] as Array<[number, number]>) driver.bankF(slot, v);
      driver.bankU(GpuUniformSlot.OutSel, 0);
      ex.runPass({
        shader: "membrane-gradient", entry: "main",
        groups: [[
          B(0, "simParams"), B(1, posBuf), B(2, "triangles"), B(3, "dmInv"),
          B(4, "restArea"), B(6, "elementGradient"), B(7, "elementEnergy"),
          B(13, "elementGradientB"), bg(14, GpuUniformSlot.OutSel),
          bg(8, GpuUniformSlot.MatC00), bg(9, GpuUniformSlot.MatC11),
          bg(10, GpuUniformSlot.MatC01), bg(11, GpuUniformSlot.MatG),
          bg(12, GpuUniformSlot.MatThickness),
        ]],
        x: Math.max(1, Math.ceil(driver.c.m / 64)),
      });
      await ex.submitBatch(false);
      const ee = await readF32(f, "elementEnergy");
      const eg = await readF32(f, "elementGradient");
      // eslint-disable-next-line no-console
      console.log(`[A05] forced ${tag.padEnd(12)} elementEnergy bad=${nBad(ee)}/${ee.length} ` +
        `elementGradient bad=${nBad(eg)}/${eg.length} e[0]=${ee[0].toExponential(4)}`);
    };
    await forced("position", "position");
    await forced("xTrial", "xTrial");
  }, 600000);

  it("runs the FULL driver.femPasses twice and compares", async () => {
    if (!fix) return;
    const f = fix;
    const { ex, driver } = f;
    const solver = f.solver;
    solver.configureStep(1 / 60);
    const mat = IN(solver).materialNow();
    const contact = IN(solver).contactParamsNow();
    const sim = new Float32Array(await ex.readSmall("simParams", 64, "adv-sp", "scalar"));
    // eslint-disable-next-line no-console
    console.log(`[A05] simParams: n=${sim[6]} m=${sim[7]} h=${sim[8]} contactCount=${sim[9]} ` +
      `newtonIt=${sim[10]} dt=${sim[0]} invDt2=${sim[1]}`);

    const ev = await solver.evaluateNewtonState(false, 71, 1 / 60);
    ex.writeBuffer("searchDirection", new Float32Array(driver.c.n * 3).fill(1e-5));

    const scan = async (tag: string) => {
      const ee = await readF32(f, "elementEnergy");
      const eg = await readF32(f, "elementGradient");
      const gr = await readF32(f, "gradient");
      // eslint-disable-next-line no-console
      console.log(`[A05] ${tag.padEnd(30)} eE=${nBad(ee)}/${ee.length} eG=${nBad(eg)}/${eg.length} ` +
        `grad=${nBad(gr)}/${gr.length} eE[0]=${ee[0].toExponential(4)}`);
    };

    for (let rep = 0; rep < 3; rep++) {
      ex.writeBuffer("elementEnergy", new Float32Array(driver.c.m).fill(NaN));
      ex.writeBuffer("elementGradient", new Float32Array(driver.c.m * 9).fill(NaN));
      ex.beginBatch(`adv-fem-${rep}`);
      driver.contactPasses(contact, "xTrial");
      driver.femPasses(mat, true);
      await ex.submitBatch(false);
      await scan(`rep${rep}: contact+fem(xTrial)`);
    }
    // Same again WITHOUT the preceding contactPasses.
    for (let rep = 0; rep < 2; rep++) {
      ex.writeBuffer("elementEnergy", new Float32Array(driver.c.m).fill(NaN));
      ex.writeBuffer("elementGradient", new Float32Array(driver.c.m * 9).fill(NaN));
      ex.beginBatch(`adv-fem-nc-${rep}`);
      driver.femPasses(mat, true);
      await ex.submitBatch(false);
      await scan(`rep${rep}: fem(xTrial) only`);
    }
    void ev;
  }, 600000);
});