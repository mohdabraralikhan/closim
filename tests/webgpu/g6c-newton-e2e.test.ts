// G6C.2 end-to-end GPU Newton control: full steps via useGpuNewtonControl,
// readback budget vs the sequential path, reject-all round behavior, and
// Gate B scene coverage (strip/floor/plate/fold/friction, 1x/5x/10x).
import { describe, it, expect, beforeAll } from "vitest";
import type { ClothScene } from "../../src/physics/scene.js";
import type { DeviceFixture } from "./device-setup.js";
import { sharedDevice, resetDeviceState, requireDevice, stripScene } from "./device-setup.js";
import { buildGrid, preprocess } from "../../src/mesh/mesh.js";
import { offsetMesh as offsetMeshLocal, preprocessMerged } from "../helpers.js";
import { createScene } from "../../src/physics/scene.js";
import { DEFAULT_MATERIAL } from "../../src/physics/types.js";
import { ContactSystem } from "../../src/collision/contact-assembly.js";
import { DEFAULT_CONTACT_PARAMS } from "../../src/collision/types.js";

let strip: DeviceFixture | null = null;

beforeAll(async () => {
  strip = await sharedDevice("g6c-e2e-strip", stripScene, { contactCapacity: 64, pairCapacity: 512 });
}, 180000);

function floorScene(): ReturnType<typeof createScene> {
  const g = buildGrid(8, 8, 0.16, 0.16);
  for (let i = 0; i < g.positions.length / 3; i++) g.positions[i * 3 + 1] += 0.05;
  const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
  const scene = createScene(mesh, { ...DEFAULT_MATERIAL }, [0, -9.81, 0]);
  scene.contact = new ContactSystem({ ...DEFAULT_CONTACT_PARAMS }, mesh.indices);
  scene.contact.setFloor(0);
  return scene;
}

function foldScene(): ReturnType<typeof createScene> {
  const w = 0.16;
  const g = buildGrid(8, 8, w, w);
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

function stiffScene(memK: number): ReturnType<typeof createScene> {
  const g = buildGrid(6, 6, 0.12, 0.12);
  const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
  const b = DEFAULT_MATERIAL;
  const mat = {
    ...b,
    stretchWarp: b.stretchWarp * memK, stretchWeft: b.stretchWeft * memK,
    shear: b.shear * memK,
  };
  return createScene(mesh, mat, [0, -9.81, 0]);
}

function plateScene(): ReturnType<typeof createScene> {
  const g = offsetMeshLocal(4, 4, 0.08, 0.08, 0.01, 0.08, 0.01);
  const mesh = preprocessMerged([g], 0.15);
  const scene = createScene(mesh, { ...DEFAULT_MATERIAL }, [0, -9.81, 0]);
  const sPos = new Float32Array([
    -0.05, 0.03, -0.05,
    0.15, 0.03, -0.05,
    0.15, 0.03, 0.15,
    -0.05, 0.03, 0.15,
  ]);
  const sIdx = new Uint32Array([0, 1, 2, 0, 2, 3]);
  scene.contact = new ContactSystem(
    { ...DEFAULT_CONTACT_PARAMS, frictionMu: 0.3 },
    mesh.indices,
  );
  scene.contact.setStaticMesh(sPos, sIdx);
  return scene;
}

function frictionScene(): ReturnType<typeof createScene> {
  const gap = 0.004;
  const w = 0.08;
  const A = offsetMeshLocal(4, 4, w, w, -w / 2 - gap / 2, 0.02, -w / 2);
  const B = offsetMeshLocal(4, 4, w, w, w / 2 + gap / 2, 0.02, -w / 2);
  const mesh = preprocessMerged([A, B], 0.15);
  const scene = createScene(mesh, { ...DEFAULT_MATERIAL }, [0, 0, 0]);
  const nA = A.positions.length / 3;
  for (let i = 0; i < mesh.count; i++) {
    scene.velocities[i * 3] = i < nA ? 0.25 : -0.25;
  }
  scene.contact = new ContactSystem(
    { ...DEFAULT_CONTACT_PARAMS, frictionMu: 0.6 }, mesh.indices,
  );
  return scene;
}

describe("G6C.2 GPU Newton end-to-end", () => {
  it("gpu-control step finishes finite with one status read per round", async () => {
    if (!strip) return;
    const { solver, ex, driver } = strip;
    const scene = (solver as unknown as { scene: ClothScene }).scene;
    resetDeviceState(strip, Float64Array.from(scene.positions));
    driver.cfg.useGpuNewtonControl = true;
    try {
      const s0 = ex.ledger.submits;
      const m0 = ex.ledger.mapSyncs;
      const r0 = solver.hotLoopReadbacks;
      const d = await solver.stepGpu(1 / 60, { newtonIters: 2 });
      const rep = solver.lastStepReport!;
      // eslint-disable-next-line no-console
      console.log(`[g6c-e2e] control=${rep.controlPath} submits=${ex.ledger.submits - s0} ` +
        `hot=${solver.hotLoopReadbacks - r0} maps=${ex.ledger.mapSyncs - m0} ` +
        `newtonIters=${rep.newtonIters} trials=${JSON.stringify(solver.trialHistory)}`);
      expect(rep.controlPath).toBe("gpu-newton");
      expect(Number.isFinite(d.energy)).toBe(true);
      expect(d.finite).toBe(1);
      // Readback budget: E0 eval (3) + 1 compact status per round + snapshot
      // debug (2). Well under the sequential path's per-trial polling.
      expect(ex.ledger.mapSyncs - m0).toBeLessThanOrEqual(3 + rep.newtonIters * 1 + 2 + 2);
    } finally {
      driver.cfg.useGpuNewtonControl = false;
    }
  }, 300000);

  it("reject-all round fails gracefully with positions untouched", async () => {
    const fix = await requireDevice(foldScene, { contactCapacity: 8, pairCapacity: 128 });
    if (!fix) return;
    try {
      const { solver, ex, driver } = fix;
      const scene = (solver as unknown as { scene: ClothScene }).scene;
      resetDeviceState(fix, Float64Array.from(scene.positions));
      solver.configureStep(1 / 60);
      const ev = await solver.evaluateNewtonState(false, 1, 1 / 60);
      await driver.pcgSolve();
      solver.beginArmijoBatch(700);
      driver.bankF(12, 1e-4);
      ex.writeBuffer("e0Store", new Float32Array([ev.status.energy, 0, 0, 0]));
      ex.writeBuffer("newtonCtl", new Float32Array(16));
      const nst = new Float32Array(20);
      ex.writeBuffer("newtonStatus", nst);
      const mt = scene.material;
      const beta = Math.max(mt.stretchWarp, mt.stretchWeft, mt.shear) * mt.thickness * 0.1 + 1e-6;
      const before = new Float32Array(await ex.readBufferDebug("position", "reject-pre", false));
      const st = await driver.newtonRound({
        round: 0, beta,
        mat: {
          c00: mt.stretchWarp, c11: mt.stretchWeft, c01: mt.stretchCoupling,
          g: mt.shear, thickness: mt.thickness,
        },
        contact: {
          dHat: 0.002, kappa: 50, mu: 0.3, fricEps: 1e-4,
          floorY: 1e30, floorOn: 0, dMin: 1e-4, contactCapacity: driver.c.cap,
        },
        batchKs: [4, 4, 2],
        evalIndexBase: 700,
      });
      const after = new Float32Array(await ex.readBufferDebug("position", "reject-post", false));
      // eslint-disable-next-line no-console
      console.log(`[g6c-reject] failure=${st.failure} accepted=${st.armijoAccepted} finite-energy=${Number.isFinite(st.energy)}`);
      expect(st.failure).toBe(true);
      expect(st.armijoAccepted).toBe(false);
      expect(Number.isFinite(st.energy)).toBe(true);
      // Predicated commit held: position bitwise untouched.
      expect([...after]).toEqual([...before]);
    } finally {
      fix.ex.destroy();
    }
  }, 300000);

  it("readback budget: GPU control maps far fewer syncs per step", async () => {
    if (!strip) return;
    const { solver, ex, driver } = strip;
    const scene = (solver as unknown as { scene: ClothScene }).scene;
    const x0 = Float64Array.from(scene.positions);
    const n = scene.mesh.count;
    const res: Record<string, { maps: number; submits: number; newtonIters: number }> = {};
    for (const gpu of [false, true]) {
      resetDeviceState(strip, Float64Array.from(x0));
      ex.writeBuffer("velocity", new Float32Array(n * 4));
      driver.cfg.useGpuNewtonControl = gpu;
      const m0 = ex.ledger.mapSyncs;
      const s0 = ex.ledger.submits;
      await solver.stepGpu(1 / 60, { newtonIters: 2 });
      res[gpu ? "gpu" : "seq"] = {
        maps: ex.ledger.mapSyncs - m0,
        submits: ex.ledger.submits - s0,
        newtonIters: solver.lastStepReport!.newtonIters,
      };
    }
    driver.cfg.useGpuNewtonControl = false;
    // eslint-disable-next-line no-console
    console.log(`[g6c-readback] seq maps=${res.seq.maps} submits=${res.seq.submits} | ` +
      `gpu maps=${res.gpu.maps} submits=${res.gpu.submits}`);
    // The whole point of GPU Newton control: collapse per-trial/per-Newton
    // polling into one compact read per round (+E0/snapshot, unchanged).
    expect(res.gpu.maps).toBeLessThan(res.seq.maps);
    // Submits stay in the same band (same passes + tiny control kernels;
    // generous bound guards against encoder blowup, not a ranking).
    expect(res.gpu.submits).toBeLessThan(res.seq.submits * 3);
  }, 300000);

  it("Gate B scene coverage under GPU control", async () => {
    const cases: Array<{
      name: string; build: () => ReturnType<typeof createScene>;
      tight: boolean; contacts: boolean;
    }> = [
      { name: "strip", build: stripScene, tight: true, contacts: false },
      { name: "floor", build: floorScene, tight: true, contacts: false },
      { name: "plate", build: plateScene, tight: false, contacts: false }, // measured 0 after one step on both paths (5 cm gap, ~1.4 mm fall)
      { name: "fold", build: foldScene, tight: false, contacts: true },
      { name: "friction", build: frictionScene, tight: false, contacts: true },
      { name: "5x", build: () => stiffScene(5), tight: true, contacts: false },
      { name: "10x", build: () => stiffScene(10), tight: true, contacts: false },
    ];
    for (const c of cases) {
      const fix = await requireDevice(c.build, { contactCapacity: 2048, pairCapacity: 8192 });
      if (!fix) {
        // eslint-disable-next-line no-console
        console.log(`[g6c-gateB] no device for ${c.name} — skipped`);
        continue;
      }
      try {
        const { solver, driver } = fix;
        const scene = (solver as unknown as { scene: ClothScene }).scene;
        const x0 = Float64Array.from(scene.positions);
        const n = scene.mesh.count;
        // Scene velocities (frictionScene closes at ±0.25 m/s; every other
        // scene starts at rest, so this is a no-op for them). Zeroing here
        // instead would freeze the friction patches 4 mm apart and the
        // contacts:true check below would measure the harness, not the path.
        const v0 = Float64Array.from(scene.velocities);
        const seqPos: Float64Array[] = [];
        for (const gpu of [false, true]) {
          resetDeviceState(fix, Float64Array.from(x0));
          const vPack = new Float32Array(n * 4);
          for (let i = 0; i < n; i++) {
            vPack[i * 4] = Math.fround(v0[i * 3]);
            vPack[i * 4 + 1] = Math.fround(v0[i * 3 + 1]);
            vPack[i * 4 + 2] = Math.fround(v0[i * 3 + 2]);
          }
          fix.ex.writeBuffer("velocity", vPack);
          driver.cfg.useGpuNewtonControl = gpu;
          const d = await solver.stepGpu(1 / 60, { newtonIters: 2 });
          expect(Number.isFinite(d.energy)).toBe(true);
          expect(d.finite).toBe(1);
          seqPos.push(Float64Array.from(solver.getPositions()));
          // eslint-disable-next-line no-console
          console.log(`[g6c-gateB] ${c.name} gpu=${gpu ? 1 : 0} energy=${d.energy.toExponential(4)} ` +
            `control=${solver.lastStepReport!.controlPath}`);
        }
        driver.cfg.useGpuNewtonControl = false;
        if (c.tight) {
          let worst = 0;
          for (let i = 0; i < seqPos[0].length; i++) {
            worst = Math.max(worst, Math.abs(seqPos[0][i] - seqPos[1][i]));
          }
          // eslint-disable-next-line no-console
          console.log(`[g6c-gateB] ${c.name}: maxPosDiff=${worst.toExponential(2)}`);
          // Physical-trajectory tolerance ( NOT bit-exact: f32-vs-f64 Armijo
          // boundaries can flip an accepted alpha by one halving across two
          // Newton iters; single-iteration selection itself is exact per
          // Gate A at 1e-10). 1e-4 stays 30x inside the CPU parity bar, and
          // wrong-logic divergences measure >1e-3 (cf. G6B state-leak bug).
          expect(worst).toBeLessThan(1e-4);
        }
        if (c.contacts) {
          solver.configureStep(1 / 60);
          const ev = await solver.evaluateNewtonState(false, 950, 1 / 60);
          expect(ev.contactCount).toBeGreaterThan(0);
        }
      } finally {
        fix.ex.destroy();
      }
    }
  }, 900000);
});
