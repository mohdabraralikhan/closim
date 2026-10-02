// G5A PCG measurement matrix: stiffness x contact x size on the real device.
// Informational only (no perf assertions, no speedup claims). One fixture per
// (size, contact); stiffness varies per cell via scene.material + hingeMeta
// re-upload + full state reset. Fixed production budgets (ni=2, pcg=20):
// PCG iteration count is constant by design — the behavioral data are residual
// curves (profiled subset), Newton convergence, submits, syncs, wall.
//
// Timestamp queries: captured where supported, but this backend latches all
// queries within one submit to the same value (verified: inter-submit deltas
// track wall at ~1ns, intra-submit deltas are 0). The capture path is kept
// (portable, self-checked) with wall-clock fallback as primary here.
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
    // fast approach (0.5 m/s closing): 4 mm initial gap closes into the
    // barrier zone on step 1, deep contact on step 2 (2 steps must CONTACT).
    for (let i = 0; i < mesh.count; i++) {
      scene.velocities[i * 3] = i < nA ? 0.25 : -0.25;
    }
    scene.contact = new ContactSystem(
      contact === "friction" ? { ...P, frictionMu: 0.6 } : { ...P }, mesh.indices,
    );
    return scene;
  }
  if (contact === "fold") {
    // full n x n page fold (right half mirrored over the left, 1.5 mm layers):
    // dense self-contact from step 0 at every size class.
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

interface CellResult {
  size: string;
  contact: ContactMode;
  stiffness: string;
  verts: number;
  tris: number;
  converged: boolean;
  finite: boolean;
  e0: number;
  e1: number;
  gradNorm: number;
  resNorm: number;
  breakdown: boolean;
  submits: number;
  syncs: number;
  wallMsPerStep: number;
  memMB: number;
}

async function runCell(
  fix: DeviceFixture,
  fresh: ReturnType<typeof buildCellInner>,
  mat: ClothMaterial,
  stiffness: string,
  size: string,
  contact: ContactMode,
  profile: boolean,
): Promise<CellResult> {
  const { solver, ex, driver } = fix;
  const scene = (solver as unknown as { scene: ReturnType<typeof buildCellInner> }).scene;
  const n = scene.mesh.count;
  // fresh state: positions + velocities + material + hinge kb + lagged zero
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
  let memMB = 0;
  for (const v of ex.bufferBytes.values()) memMB += v;
  memMB /= 1048576;
  const out: CellResult = {
    size, contact, stiffness, verts: n, tris: scene.mesh.triCount,
    converged: false, finite: false, e0: NaN, e1: NaN, gradNorm: NaN,
    resNorm: NaN, breakdown: false, submits: 0, syncs: 0, wallMsPerStep: NaN, memMB,
  };
  try {
    const s0 = ex.ledger.submits;
    const r0 = solver.hotLoopReadbacks;
    const t0 = performance.now();
    const d0 = await solver.stepGpu(1 / 60, { newtonIters: 2 });
    const d1 = await solver.stepGpu(1 / 60, { newtonIters: 2 });
    out.wallMsPerStep = (performance.now() - t0) / 2;
    out.submits = ex.ledger.submits - s0;
    out.syncs = solver.hotLoopReadbacks - r0;
    out.converged = d1.converged === 1;
    out.finite = Number.isFinite(d1.energy) && d1.finite === 1;
    out.e0 = d0.energy;
    out.e1 = d1.energy;
    out.gradNorm = d1.gradNorm;
    out.resNorm = solver.lastPcg.resNorm;
    out.breakdown = solver.lastPcg.breakdown;
    if (profile && out.finite) {
      solver.configureStep(1 / 60);
      await solver.evaluateNewtonState(false, 900, 1 / 60);
      const rec: { rz: number[] } = { rz: [] };
      const pr = await driver.pcgSolve(rec);
      const rel = rec.rz.length > 1
        ? rec.rz[rec.rz.length - 1] / Math.max(rec.rz[0], 1e-300)
        : NaN;
      // eslint-disable-next-line no-console
      console.log(`[g5-curve] ${size}/${contact}/${stiffness}: n=${rec.rz.length} ` +
        `rz0=${rec.rz[0]?.toExponential(2)} relEnd=${rel.toExponential(2)} ` +
        `resNorm=${pr.resNorm.toExponential(2)} breakdown=${pr.breakdown}`);
    }
  } catch (err) {
    // Diverged cells are DATA (Jacobi failing is the signal), not failures.
    // eslint-disable-next-line no-console
    console.log(`[g5-cell-error] ${size}/${contact}/${stiffness}: ${String(err).slice(0, 160)}`);
  }
  // eslint-disable-next-line no-console
  console.log(`[g5-cell] ${JSON.stringify(out)}`);
  return out;
}

const STIFFNESS: Array<{ label: string; memK: number; bendK: number }> = [
  { label: "1x", memK: 1, bendK: 1 },
  { label: "2x", memK: 2, bendK: 1 },
  { label: "5x", memK: 5, bendK: 1 },
  { label: "10x", memK: 10, bendK: 1 },
  { label: "10xbend", memK: 1, bendK: 10 },
];

const CONTACTS_1K_10K: ContactMode[] = ["none", "floor", "self", "fold", "friction"];
const CONTACTS_50K: ContactMode[] = ["none", "floor"];

async function runMatrix(
  n: number, size: string, contacts: ContactMode[], profileEvery: number,
): Promise<void> {
  const triCount = 2 * n * n;
  for (const contact of contacts) {
    const fix = await requireDevice(() => buildCellInner(n, contact), {
      contactCapacity: Math.max(4096, triCount * 2),
      pairCapacity: Math.min(1 << 20, triCount * 8),
    });
    if (!fix) {
      // eslint-disable-next-line no-console
      console.log(`[g5-skip] no device for ${size}/${contact}`);
      continue;
    }
    try {
      let si = 0;
      for (const st of STIFFNESS) {
        const fresh = buildCellInner(n, contact);
        await runCell(fix, fresh, scaledMaterial(st.memK, st.bendK), st.label, size, contact, si % profileEvery === 0);
        si++;
      }
    } finally {
      fix.ex.destroy();
    }
  }
}

describe("G5A PCG measurement matrix", () => {
  it("profiles 1k across stiffness x contact", async () => {
    await runMatrix(32, "1k", CONTACTS_1K_10K, 2);
  }, 1500000);

  it("profiles 10k across stiffness x contact", async () => {
    await runMatrix(100, "10k", CONTACTS_1K_10K, 2);
  }, 1500000);

  it("profiles 50k selected cells", async () => {
    await runMatrix(224, "50k", CONTACTS_50K, 1);
  }, 1500000);
});
