// ADVERSARIAL A03 — diagnostic probe: where do NaNs and blanket rejections
// come from on a contact-engaged GPU-control round?
//
// A02 observed two anomalies that need isolating before they can be graded:
//   (i)  `gradient` compared NaN-vs-something after a gpu-control round;
//   (ii) a round with 10 geometrically-VALID candidates rejected all of them
//        (armijoAccepted=false while ccdFails/barrierFails/overflow all false).
// This file localises both. READ-ONLY: it only reads buffers and calls public
// driver entry points; it asserts nothing it has not first measured.
import { describe, it, beforeAll } from "vitest";
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

function nanScan(name: string, a: Float32Array): { nans: number; first: number; maxAbs: number } {
  let nans = 0; let first = -1; let maxAbs = 0;
  for (let i = 0; i < a.length; i++) {
    const v = a[i];
    if (Number.isNaN(v) || !Number.isFinite(v)) { nans++; if (first < 0) first = i; }
    else if (Math.abs(v) > maxAbs) maxAbs = Math.abs(v);
  }
  return { nans, first, maxAbs };
}

describe("ADVERSARIAL A03 — NaN localisation + blanket-rejection probe", () => {
  it("scans state buffers for non-finite values at a clean E0 evaluation", async () => {
    if (!fix) return;
    const f = fix;
    const solver = f.solver;
    solver.configureStep(1 / 60);
    const ev = await solver.evaluateNewtonState(false, 21, 1 / 60);
    // eslint-disable-next-line no-console
    console.log(`[A03] E0 contacts=${ev.contactCount} energy=${ev.status.energy.toExponential(4)} ` +
      `gradNorm=${ev.status.gradNorm.toExponential(4)} finite=${ev.status.finite} ` +
      `minDist=${ev.status.minDistance}`);
    for (const buf of ["gradient", "rhs", "diag", "contactDiag", "elementGradient", "position"]) {
      const s = nanScan(buf, await readF32(f, buf));
      // eslint-disable-next-line no-console
      console.log(`[A03] E0 scan ${buf}: nonfinite=${s.nans} firstIdx=${s.first} maxAbs=${s.maxAbs.toExponential(4)} len=${(await readF32(f, buf)).length}`);
    }
  }, 600000);

  it("instruments a gpu-control round: gtdx, trust, and per-candidate verdicts", async () => {
    if (!fix) return;
    const f = fix;
    const { ex, driver } = f;
    const solver = f.solver;
    solver.configureStep(1 / 60);
    const raw = IN(solver).scene.material;
    const mat = IN(solver).materialNow();
    const beta = Math.max(raw.stretchWarp, raw.stretchWeft, raw.shear) * raw.thickness * 0.1 + 1e-6;

    const ev = await solver.evaluateNewtonState(false, 31, 1 / 60);

    // Honour the documented newtonRound contract: contactCount = cap.
    IN(solver).contactCountNow = driver.c.cap;
    IN(solver).simImage.newtonIteration = 0;
    IN(solver).pushSimParams();

    ex.writeBuffer("e0Store", new Float32Array([ev.status.energy, 0, 0, 0]));
    driver.bankF(42, 1e-5);
    driver.bankF(43, 0.002);
    ex.writeBuffer("newtonCtl", new Float32Array(16));
    ex.writeBuffer("newtonStatus", new Float32Array(20));

    const st = await driver.newtonRound({
      round: 0, beta,
      mat,
      contact: IN(solver).contactParamsNow(),
      batchKs: [4, 4, 2],
      evalIndexBase: 0,
    });
    // eslint-disable-next-line no-console
    console.log(`[A03] round: accepted=${st.armijoAccepted} alpha=${st.selectedAlpha} ` +
      `trialIdx=${st.selectedTrialIndex} gtdx=${st.gtdx.toExponential(4)} ` +
      `stepNorm=${st.stepNorm.toExponential(4)} gradNorm=${st.gradNorm.toExponential(4)} ` +
      `residual=${st.residual.toExponential(4)} dirValid=${st.directionValid} ` +
      `merit=${st.merit.toExponential(4)} energy=${st.energy.toExponential(4)} ` +
      `barrierFails=${st.barrierFailure} ccdFails=${st.ccdFailure} overflow=${st.contactOverflow}`);

    // Candidate matrix: row k = [alpha, energy, gradNorm, ok, dist, toi, finite, overflow]
    const cand = await readF32(f, "armijoCandidates");
    const rows: string[] = [];
    for (let k = 0; k < 8; k++) {
      const r = Array.from(cand.slice(k * 8, k * 8 + 8));
      rows.push(`k=${k} a=${r[0].toExponential(2)} E=${r[1].toExponential(4)} ` +
        `|g|=${r[2].toExponential(3)} ok=${r[3]} d=${r[4].toExponential(3)} toi=${r[5].toExponential(3)} ` +
        `fin=${r[6]} ovf=${r[7]}`);
    }
    // eslint-disable-next-line no-console
    console.log(`[A03] candidates:\n  ${rows.join("\n  ")}`);

    const ast = await readF32(f, "armijoStatus");
    // eslint-disable-next-line no-console
    console.log(`[A03] armijoStatus: accepted=${ast[0]} selAlpha=${ast[1].toExponential(3)} ` +
      `selIdx=${ast[2]} trials=${ast[3]} armijoFails=${ast[4]} ccdFails=${ast[5]} ` +
      `barrierFails=${ast[6]} overflowFails=${ast[7]} finite=${ast[8]} ` +
      `E=${ast[9].toExponential(4)} |g|=${ast[10].toExponential(3)} minD=${ast[13]} minTOI=${ast[14]}`);
    // eslint-disable-next-line no-console
    console.log(`[A03] E0=${ev.status.energy.toExponential(6)} gtdxStore=${(await readF32(f, "gtdxStore"))[0].toExponential(4)} ` +
      `trustScale=${(await readF32(f, "trustScaleStore"))[0].toExponential(4)} ` +
      `rhsMaxAbs=${nanScan("rhs", await readF32(f, "rhs")).maxAbs.toExponential(4)}`);

    for (const buf of ["gradient", "rhs", "diag", "contactDiag"]) {
      const s = nanScan(buf, await readF32(f, buf));
      // eslint-disable-next-line no-console
      console.log(`[A03] post-round scan ${buf}: nonfinite=${s.nans} firstIdx=${s.first} maxAbs=${s.maxAbs.toExponential(4)}`);
    }
  }, 600000);
});