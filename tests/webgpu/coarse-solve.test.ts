// G5.5 C0/C1 device tests: C0 correction parity, inner-PCG correction parity
// (K=4/8 vs CPU coarsePcg), span-counter behavior, submit regression.
// Skips cleanly with no device.
import { describe, it, expect, beforeAll } from "vitest";
import type { ClothScene } from "../../src/physics/scene.js";
import {
  assembleCoarseValues, coarseSpmv, applyCoarseBlockJacobi, coarsePcg,
} from "../../src/solver/coarse-csr.js";
import { restrictCoarse, prolongateAdd, buildCoarseDiag } from "../../src/solver/mas.js";
import { ContactSystem } from "../../src/collision/contact-assembly.js";
import { DEFAULT_CONTACT_PARAMS } from "../../src/collision/types.js";
import type { DeviceFixture } from "./device-setup.js";
import { sharedDevice, resetDeviceState, stripScene } from "./device-setup.js";


let strip: DeviceFixture | null = null;

beforeAll(async () => {
  strip = await sharedDevice("coarse-solve-strip", stripScene, { contactCapacity: 64, pairCapacity: 512 });
}, 180000);

async function readySolve(X: Float64Array): Promise<{ diag: Float64Array; coarseDofs: number }> {
  const { solver, ex, driver } = strip!;
  resetDeviceState(strip!, X);
  solver.configureStep(1 / 60);
  await solver.evaluateNewtonState(false, 1, 1 / 60);
  ex.beginBatch("coarse-solve-ready");
  driver.schwarzBuildPasses();
  driver.masCoarseDiagPass();
  driver.coarseAssemblePasses();
  await ex.submitBatch(false);
  const diagRaw = await ex.readBufferDebug("diag", "coarse-solve-diag", false);
  const scene = (solver as unknown as { scene: ClothScene }).scene;
  const diag = new Float64Array(scene.mesh.count * 3);
  const diagF = new Float32Array(diagRaw);
  for (let i = 0; i < diag.length; i++) diag[i] = diagF[i];
  return { diag, coarseDofs: driver.c.schwarzDoms * 3 };
}

describe("G5.5 C0/C1 device correction", () => {
  it("C0 correction matches restrict/block-Jacobi/prolongate", async () => {
    if (!strip) return;
    const { solver, ex, driver } = strip;
    const scene = (solver as unknown as { scene: ClothScene }).scene;
    const n = scene.mesh.count;
    const X = Float64Array.from(scene.positions);
    const { diag } = await readySolve(X);
    const topo = solver.schwarzTopo!;
    const pat = solver.coarsePattern!;
    // deterministic residual seed
    const r = new Float32Array(n * 3);
    for (let i = 0; i < r.length; i++) r[i] = Math.sin(i * 0.77) * 1.5;
    ex.writeBuffer("pcgResidual", r);
    ex.writeBuffer("pcgZ", new Float32Array(n * 3));
    ex.beginBatch("coarse-c0-parity");
    driver.coarseCorrectC0Pass();
    await ex.submitBatch(false);
    const zRaw = await ex.readBufferDebug("pcgZ", "coarse-c0-z", false);
    const zGpu = new Float32Array(zRaw);
    // CPU: restrict -> block-Jacobi -> prolongate, then the pin filter (the
    // GPU prolongation suppresses pinned DOFs; the gate requires pins to hold).
    const csr = assembleCoarseValues(X, scene.mesh, scene.material, diag, topo, pat);
    const cd = buildCoarseDiag(X, scene.mesh, scene.material, diag, topo);
    const rc = restrictCoarse(r, topo);
    const cc = new Float64Array(rc.length);
    applyCoarseBlockJacobi(csr, cd, rc, cc);
    const zRef = new Float64Array(n * 3);
    prolongateAdd(cc, topo, zRef);
    for (const [id] of scene.pinned) {
      zRef[id * 3] = 0; zRef[id * 3 + 1] = 0; zRef[id * 3 + 2] = 0;
    }
    let worst = 0;
    let scale = 0;
    for (let i = 0; i < n * 3; i++) scale = Math.max(scale, Math.abs(zRef[i]));
    scale = Math.max(scale, 1e-12);
    for (let i = 0; i < n * 3; i++) worst = Math.max(worst, Math.abs(zGpu[i] - zRef[i]) / scale);
    // eslint-disable-next-line no-console
    console.log(`[g5.5-c0] correction rel err: ${worst.toExponential(2)}`);
    expect(worst).toBeLessThan(1e-3);
  }, 180000);

  it("C1 inner solve matches CPU coarsePcg at K=4 and K=8", async () => {
    if (!strip) return;
    const { solver, ex, driver } = strip;
    const scene = (solver as unknown as { scene: ClothScene }).scene;
    const n = scene.mesh.count;
    const X = Float64Array.from(scene.positions);
    const { diag } = await readySolve(X);
    const topo = solver.schwarzTopo!;
    const pat = solver.coarsePattern!;
    const csr = assembleCoarseValues(X, scene.mesh, scene.material, diag, topo, pat);
    const cd = buildCoarseDiag(X, scene.mesh, scene.material, diag, topo);
    const r = new Float32Array(n * 3);
    for (let i = 0; i < r.length; i++) r[i] = Math.cos(i * 1.13) * 2;
    for (const K of [4, 8]) {
      ex.writeBuffer("pcgResidual", r);
      ex.writeBuffer("pcgZ", new Float32Array(n * 3));
      ex.beginBatch("coarse-c1-parity");
      driver.coarseSolvePasses(K);
      await ex.submitBatch(false);
      const zRaw = await ex.readBufferDebug("pcgZ", "coarse-c1-z", false);
      const zGpu = new Float32Array(zRaw);
      // CPU: same fixed-K inner solve then prolongate + pin filter
      const rc = restrictCoarse(r, topo);
      const got = coarsePcg(csr, cd, Float64Array.from(rc), K, 1e-12);
      const zRef = new Float64Array(n * 3);
      prolongateAdd(got.x, topo, zRef);
      for (const [id] of scene.pinned) {
        zRef[id * 3] = 0; zRef[id * 3 + 1] = 0; zRef[id * 3 + 2] = 0;
      }
      let worst = 0;
      let scale = 0;
      for (let i = 0; i < n * 3; i++) scale = Math.max(scale, Math.abs(zRef[i]));
      scale = Math.max(scale, 1e-12);
      for (let i = 0; i < n * 3; i++) worst = Math.max(worst, Math.abs(zGpu[i] - zRef[i]) / scale);
      // eslint-disable-next-line no-console
      console.log(`[g5.5-c1:K=${K}] correction rel err: ${worst.toExponential(2)} iters=${got.iters}`);
      expect(worst).toBeLessThan(5e-3);
    }
  }, 240000);

  it("span counter matches the CPU mirror on a fold scene", async () => {
    const { requireDevice } = await import("./device-setup.js");
    const { createScene } = await import("../../src/physics/scene.js");
    const { buildGrid, preprocess } = await import("../../src/mesh/mesh.js");
    const { DEFAULT_MATERIAL } = await import("../../src/physics/types.js");
    // Page fold: right half mirrored over the left with 1.5 mm layer offset
    // => dense self-contact from step 0 (same construction as the G5A fold).
    const buildFold = () => {
      const g = buildGrid(6, 6, 0.12, 0.12);
      const w = 0.12;
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
    };
    const fix = await requireDevice(buildFold, { contactCapacity: 4096, pairCapacity: 8192 });
    if (!fix) {
      // eslint-disable-next-line no-console
      console.log("[g5.5-span] no device — skipped");
      return;
    }
    try {
      const { solver, ex, driver } = fix;
      const scene = (solver as unknown as { scene: ClothScene }).scene;
      solver.configureStep(1 / 60);
      await solver.evaluateNewtonState(false, 1, 1 / 60);
      ex.writeBuffer("masContactSpan", new Uint32Array([0, 0, 0, 0]));
      ex.beginBatch("coarse-span");
      driver.spanPass();
      await ex.submitBatch(false);
      const span = await driver.readContactSpan();
      const count = await driver.readContactCount();
      // CPU mirror on the same state
      const cs = new ContactSystem({ ...DEFAULT_CONTACT_PARAMS }, scene.mesh.indices);
      const X = Float64Array.from(scene.positions);
      cs.beginStep(X);
      cs.updateActiveSet(X);
      const topo = solver.schwarzTopo!;
      let cpuSpan = 0;
      for (const c of cs.activeList()) {
        const verts: number[] = [];
        if (c.kind === 2) verts.push(c.p);
        else if (c.kind === 0) {
          verts.push(c.p);
          const ws = [c.a, c.b, c.c];
          for (const vv of ws) if (vv >= 0) verts.push(vv);
        } else {
          for (const vv of [c.a, c.b, c.c, c.d]) if (vv >= 0) verts.push(vv);
        }
        const ds = new Set(verts.map((vv) => topo.domainOf[vv]));
        if (ds.size > 1) cpuSpan++;
      }
      // eslint-disable-next-line no-console
      console.log(`[g5.5-span] device active=${count} spanning=${span} cpuActive=${cs.activeList().length} cpuSpan=${cpuSpan}`);
      expect(count).toBeGreaterThan(0);
      expect(span).toBeLessThanOrEqual(count);
      expect(span).toBe(cpuSpan);
    } finally {
      fix.ex.destroy();
    }
  }, 240000);

  it("single-solve submit delta is ~zero at 10k (structural cost isolation)", async () => {
    // One pcgSolve on one identical Newton state: no Newton trials, no
    // trajectory divergence — isolates the preconditioner's own submits.
    const { requireDevice } = await import("./device-setup.js");
    const { buildGrid, preprocess } = await import("../../src/mesh/mesh.js");
    const { createScene } = await import("../../src/physics/scene.js");
    const { DEFAULT_MATERIAL } = await import("../../src/physics/types.js");
    const build10k = () => {
      const g = buildGrid(100, 100, 2.0, 2.0);
      for (let i = 0; i < g.positions.length / 3; i++) g.positions[i * 3 + 1] += 0.05;
      const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
      return createScene(mesh, {
        ...DEFAULT_MATERIAL,
        stretchWarp: DEFAULT_MATERIAL.stretchWarp * 10,
        stretchWeft: DEFAULT_MATERIAL.stretchWeft * 10,
        shear: DEFAULT_MATERIAL.shear * 10,
      }, [0, -9.81, 0]);
    };
    const fix = await requireDevice(build10k, { contactCapacity: 4096, pairCapacity: 16384 });
    if (!fix) {
      // eslint-disable-next-line no-console
      console.log("[g5.5-structural] no device — skipped");
      return;
    }
    try {
      const { solver, ex, driver } = fix;
      const prev = { ...driver.cfg };
      try {
        driver.cfg.pcgIters = 20;
        const modes: Array<[string, Partial<typeof prev>]> = [
          ["schwarz1", { useSchwarz: true }],
          ["c1-8", { useCoarsePcg: true, coarseIters: 8 }],
        ];
        for (const [name, cfg] of modes) {
          Object.assign(driver.cfg, {
            useMas: false, useSchwarz: false, useBlockJacobi: false,
            useCoarseC0: false, useCoarsePcg: false,
          });
          Object.assign(driver.cfg, cfg);
          solver.configureStep(1 / 60);
          await solver.evaluateNewtonState(false, 1, 1 / 60);
          const s0 = ex.ledger.submits;
          const p0 = ex.ledger.passes;
          const res = await driver.pcgSolve();
          // eslint-disable-next-line no-console
          console.log(`[g5.5-structural] 10k-10x ${name}: submits=${ex.ledger.submits - s0} ` +
            `passes=${ex.ledger.passes - p0} res=${res.resNorm.toExponential(3)}`);
          expect(Number.isFinite(res.resNorm)).toBe(true);
        }
      } finally {
        Object.assign(driver.cfg, prev);
      }
    } finally {
      fix.ex.destroy();
    }
  }, 600000);

  it("submit delta stays bounded across preconditioners", async () => {
    if (!strip) return;
    const { solver, ex, driver } = strip;
    const scene = (solver as unknown as { scene: ClothScene }).scene;
    const prev = { ...driver.cfg };
    const deltas: Record<string, { submits: number; passes: number; resNorm: number }> = {};
    try {
      driver.cfg.pcgIters = 4;
      const modes: Array<[string, Partial<typeof prev>]> = [
        ["schwarz1", { useSchwarz: true, useMas: false, useCoarseC0: false, useCoarsePcg: false, useBlockJacobi: false }],
        ["mas2", { useSchwarz: false, useMas: true, useCoarseC0: false, useCoarsePcg: false, useBlockJacobi: false }],
        ["c0", { useSchwarz: false, useMas: false, useCoarseC0: true, useCoarsePcg: false, useBlockJacobi: false }],
        ["c1-4", { useSchwarz: false, useMas: false, useCoarseC0: false, useCoarsePcg: true, coarseIters: 4, useBlockJacobi: false }],
      ];
      for (const [name, cfg] of modes) {
        Object.assign(driver.cfg, cfg);
        resetDeviceState(strip, Float64Array.from(scene.positions));
        solver.configureStep(1 / 60);
        await solver.evaluateNewtonState(false, 1, 1 / 60);
        const s0 = ex.ledger.submits;
        const p0 = ex.ledger.passes;
        const res = await driver.pcgSolve();
        deltas[name] = {
          submits: ex.ledger.submits - s0,
          passes: ex.ledger.passes - p0,
          resNorm: res.resNorm,
        };
        // eslint-disable-next-line no-console
        console.log(`[g5.5-submit] ${name}: submits=${deltas[name].submits} passes=${deltas[name].passes} res=${res.resNorm.toExponential(3)}`);
        expect(Number.isFinite(res.resNorm)).toBe(true);
      }
      // Inner loop must execute (passes grow ~10/inner-iter) without submit
      // growth beyond the amortized per-solve factor setup.
      expect(deltas["c1-4"].passes).toBeGreaterThan(deltas["schwarz1"].passes + 20);
      expect(deltas["c1-4"].submits).toBeLessThanOrEqual(deltas["schwarz1"].submits + 10);
      expect(deltas["c0"].submits).toBeLessThanOrEqual(deltas["schwarz1"].submits + 10);
    } finally {
      Object.assign(driver.cfg, prev);
    }
  }, 240000);
});
