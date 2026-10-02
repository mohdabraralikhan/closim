// G5.5 reduced decision matrix (real device, informational like G5A):
// schwarz1 vs c1-8 across the reduced cell set + full C1 ladder on stiff
// cells. Fixed production budgets (ni=2, pcg=20): residual curves, Newton
// convergence, submits, syncs, wall, span coverage, memory. No single score.
import { describe, it, expect } from "vitest";
import { buildGrid, preprocess } from "../../src/mesh/mesh.js";
import { createScene } from "../../src/physics/scene.js";
import { DEFAULT_MATERIAL, type ClothMaterial } from "../../src/physics/types.js";
import { ContactSystem } from "../../src/collision/contact-assembly.js";
import { DEFAULT_CONTACT_PARAMS } from "../../src/collision/types.js";
import type { DeviceFixture } from "./device-setup.js";
import { requireDevice, resetDeviceState } from "./device-setup.js";
import { offsetMesh, preprocessMerged } from "../helpers.js";

const P = { ...DEFAULT_CONTACT_PARAMS };

type ContactMode = "none" | "floor" | "self" | "fold" | "friction";

function scaledMaterial(memK: number, bendK: number): ClothMaterial {
  const b = DEFAULT_MATERIAL;
  return {
    ...b,
    stretchWarp: b.stretchWarp * memK,
    stretchWeft: b.stretchWeft * memK,
    stretchCoupling: b.stretchCoupling * memK,
    shear: b.shear * memK,
    bendWarp: b.bendWarp * bendK,
    bendWeft: b.bendWeft * bendK,
  };
}

function buildCellInner(n: number, contact: ContactMode) {
  const gap = 0.004;
  if (contact === "self" || contact === "friction") {
    const w = 0.02 * n;
    const A = offsetMesh(n, n, w, w, -w / 2 - gap / 2, 0.02, -w / 2);
    const B = offsetMesh(n, n, w, w, w / 2 + gap / 2, 0.02, -w / 2);
    const mesh = preprocessMerged([A, B], 0.15);
    const scene = createScene(mesh, { ...DEFAULT_MATERIAL }, [0, 0, 0]);
    const nA = A.positions.length / 3;
    for (let i = 0; i < mesh.count; i++) {
      scene.velocities[i * 3] = i < nA ? 0.25 : -0.25;
    }
    scene.contact = new ContactSystem(
      contact === "friction" ? { ...P, frictionMu: 0.6 } : { ...P }, mesh.indices,
    );
    return scene;
  }
  if (contact === "fold") {
    const w = 0.02 * n;
    const g = offsetMesh(n, n, w, w, 0, 0.02, -w / 2);
    const mesh = preprocessMerged([g], 0.15);
    const scene = createScene(mesh, { ...DEFAULT_MATERIAL }, [0, -9.81, 0]);
    for (let i = 0; i < mesh.count; i++) {
      const x = scene.positions[i * 3];
      if (x > w / 2) {
        scene.positions[i * 3] = w / 2 - 2 * (x - w / 2);
        scene.positions[i * 3 + 1] += 0.0015;
      }
    }
    scene.contact = new ContactSystem({ ...P }, mesh.indices);
    return scene;
  }
  const g = buildGrid(n, n, 0.02 * n, 0.02 * n);
  for (let i = 0; i < g.positions.length / 3; i++) g.positions[i * 3 + 1] += 0.05;
  const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
  const scene = createScene(mesh, { ...DEFAULT_MATERIAL }, [0, -9.81, 0]);
  if (contact === "floor") {
    scene.contact = new ContactSystem({ ...P }, mesh.indices);
    scene.contact.setFloor(0);
  }
  return scene;
}

function packV4(X: Float64Array, n: number): Float32Array {
  const out = new Float32Array(n * 4);
  for (let i = 0; i < n; i++) {
    out[i * 4] = Math.fround(X[i * 3]);
    out[i * 4 + 1] = Math.fround(X[i * 3 + 1]);
    out[i * 4 + 2] = Math.fround(X[i * 3 + 2]);
  }
  return out;
}

type MethodName = "schwarz1" | "c1-4" | "c1-8" | "c1-16" | "c1-32";

interface MatrixCell {
  size: string;
  contact: ContactMode;
  stiffness: string;
  method: MethodName;
  coarseIters: number;
  verts: number;
  converged: boolean;
  finite: boolean;
  gradNorm: number;
  resNorm: number;
  breakdown: boolean;
  submits: number;
  syncs: number;
  wallMsPerStep: number;
  spanCount: number;
  spanSpan: number;
  memMB: number;
}

function applyMethod(driver: DeviceFixture["driver"], method: MethodName): number {
  Object.assign(driver.cfg, {
    useMas: false, useSchwarz: false, useBlockJacobi: false,
    useCoarseC0: false, useCoarsePcg: false,
  });
  if (method === "schwarz1") {
    driver.cfg.useSchwarz = true;
    return 0;
  }
  driver.cfg.useCoarsePcg = true;
  const K = Number(method.slice(3));
  driver.cfg.coarseIters = K;
  return K;
}

async function runCell(
  fix: DeviceFixture,
  fresh: ReturnType<typeof buildCellInner>,
  mat: ClothMaterial,
  stiffness: string,
  size: string,
  contact: ContactMode,
  method: MethodName,
): Promise<MatrixCell> {
  const { solver, ex, driver } = fix;
  const scene = (solver as unknown as { scene: ReturnType<typeof buildCellInner> }).scene;
  const n = scene.mesh.count;
  resetDeviceState(fix, Float64Array.from(fresh.positions));
  ex.writeBuffer("velocity", packV4(Float64Array.from(fresh.velocities), n));
  scene.material = { ...mat };
  const hm = new Float32Array(scene.mesh.hinges.length * 4);
  const kb = 0.5 * (mat.bendWarp + mat.bendWeft);
  scene.mesh.hinges.forEach((h, k) => {
    hm[k * 4] = h.restAngle; hm[k * 4 + 1] = h.edgeLen;
    hm[k * 4 + 2] = h.areaSum; hm[k * 4 + 3] = kb;
  });
  ex.writeBuffer("hingeMeta", hm);
  driver.cfg.pcgIters = 20;
  const coarseIters = applyMethod(driver, method);
  let memMB = 0;
  for (const v of ex.bufferBytes.values()) memMB += v;
  memMB /= 1048576;
  const out: MatrixCell = {
    size, contact, stiffness, method, coarseIters,
    verts: n, converged: false, finite: false, gradNorm: NaN,
    resNorm: NaN, breakdown: false, submits: 0, syncs: 0,
    wallMsPerStep: NaN, spanCount: 0, spanSpan: 0, memMB,
  };
  try {
    const s0 = ex.ledger.submits;
    const r0 = solver.hotLoopReadbacks;
    const t0 = performance.now();
    const d1 = await solver.stepGpu(1 / 60, { newtonIters: 2 });
    await solver.stepGpu(1 / 60, { newtonIters: 2 });
    out.wallMsPerStep = (performance.now() - t0) / 2;
    out.submits = ex.ledger.submits - s0;
    out.syncs = solver.hotLoopReadbacks - r0;
    out.converged = d1.converged === 1;
    out.finite = Number.isFinite(d1.energy) && d1.finite === 1;
    out.gradNorm = d1.gradNorm;
    out.resNorm = solver.lastPcg.resNorm;
    out.breakdown = solver.lastPcg.breakdown;
    // span coverage on the live set (same sync category as contact count)
    ex.writeBuffer("masContactSpan", new Uint32Array([0, 0, 0, 0]));
    solver.configureStep(1 / 60);
    await solver.evaluateNewtonState(false, 950, 1 / 60);
    ex.beginBatch("matrix-span");
    driver.spanPass();
    await ex.submitBatch(false);
    out.spanSpan = await driver.readContactSpan();
    out.spanCount = await driver.readContactCount();
  } catch (err) {
    // eslint-disable-next-line no-console
    console.log(`[g5.5-cell-error] ${size}/${contact}/${stiffness}/${method}: ${String(err).slice(0, 160)}`);
  }
  // eslint-disable-next-line no-console
  console.log(`[g5.5-cell] ${JSON.stringify(out)}`);
  return out;
}

async function runCells(
  n: number, size: string,
  cells: Array<{ contact: ContactMode; stiffness: string; memK: number; methods: MethodName[] }>,
  assertFinite: boolean,
): Promise<void> {
  const triCount = 2 * n * n;
  const contacts = [...new Set(cells.map((c) => c.contact))];
  for (const contact of contacts) {
    const fix = await requireDevice(() => buildCellInner(n, contact), {
      contactCapacity: Math.max(4096, triCount * 2),
      pairCapacity: Math.min(1 << 20, triCount * 8),
    });
    if (!fix) {
      // eslint-disable-next-line no-console
      console.log(`[g5.5-skip] no device for ${size}/${contact}`);
      continue;
    }
    try {
      for (const cell of cells.filter((c) => c.contact === contact)) {
        for (const method of cell.methods) {
          const fresh = buildCellInner(n, contact);
          const res = await runCell(fix, fresh, scaledMaterial(cell.memK, 1), cell.stiffness, size, contact, method);
          if (assertFinite) expect(res.finite).toBe(true);
        }
      }
    } finally {
      fix.ex.destroy();
    }
  }
}

describe("G5.5 coarse decision matrix", () => {
  it("10k schwarz1 vs c1-8 at 1x/10x, none/floor/self", async () => {
    await runCells(100, "10k", [
      { contact: "none", stiffness: "1x", memK: 1, methods: ["schwarz1", "c1-8"] },
      { contact: "none", stiffness: "10x", memK: 10, methods: ["schwarz1", "c1-8"] },
      { contact: "floor", stiffness: "1x", memK: 1, methods: ["schwarz1", "c1-8"] },
      { contact: "floor", stiffness: "10x", memK: 10, methods: ["schwarz1", "c1-8"] },
      { contact: "self", stiffness: "1x", memK: 1, methods: ["schwarz1", "c1-8"] },
      { contact: "self", stiffness: "10x", memK: 10, methods: ["schwarz1", "c1-8"] },
    ], true);
  }, 1500000);

  it("10k stiff ladder 4/16/32 on 10x none/floor/self", async () => {
    await runCells(100, "10k", [
      { contact: "none", stiffness: "10x", memK: 10, methods: ["c1-4", "c1-16", "c1-32"] },
      { contact: "floor", stiffness: "10x", memK: 10, methods: ["c1-4", "c1-16", "c1-32"] },
      { contact: "self", stiffness: "10x", memK: 10, methods: ["c1-4", "c1-16", "c1-32"] },
    ], true);
  }, 1500000);

  it("50k spots: 10x none schwarz1/c1-8/c1-32", async () => {
    await runCells(224, "50k", [
      { contact: "none", stiffness: "10x", memK: 10, methods: ["schwarz1", "c1-8", "c1-32"] },
    ], false);
  }, 1500000);
});
