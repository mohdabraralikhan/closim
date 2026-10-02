import { WebGpuSolver } from "./src/backend/webgpu/gpu-solver.js";
import { stripScene, resetDeviceState } from "./tests/webgpu/device-setup.js";
import type { ClothScene } from "./src/physics/scene.js";

const scene = stripScene();
const solver = new WebGpuSolver();
solver.initialize(scene);
const ok = await solver.initDevice({ contactCapacity: 64, pairCapacity: 512 });
console.log("device:", ok);
if (!ok) process.exit(0);
const ex = solver.executor!;
const driver = solver.driver!;
const sc = (solver as unknown as { scene: ClothScene }).scene;
const fix = { solver, ex, driver } as never;
resetDeviceState(fix as never, Float64Array.from(sc.positions));
solver.configureStep(1 / 60);
await solver.evaluateNewtonState(false, 1, 1 / 60);
await driver.pcgSolve();
const mt = sc.material;
const mat = { c00: mt.stretchWarp, c11: mt.stretchWeft, c01: mt.stretchCoupling, g: mt.shear, thickness: mt.thickness };
const contact = { dHat: 0.002, kappa: 50, mu: 0.3, fricEps: 1e-4, floorY: 1e30, floorOn: 0, dMin: 1e-4, contactCapacity: 64 };
solver.beginArmijoBatch(701);
driver.bankF(12, 1e-4);
// manual K=2 batch with trial-8/9 alphas
const alphas = new Float32Array(8).fill(1.0);
alphas[0] = Math.pow(0.5, 8); alphas[1] = Math.pow(0.5, 9);
ex.writeBuffer("armijoAlphas", alphas);
const st0 = new Float32Array(16); st0[8] = 1.0;
ex.writeBuffer("armijoStatus", st0);
ex.writeBuffer("e0Store", new Float32Array([0.09, 0, 0, 0]));
ex.writeBuffer("gtdxStore", new Float32Array([-0.06, 0, 0, 0]));
ex.writeBuffer("trustScaleStore", new Float32Array([0.033, 0, 0, 0]));
async function runBatch(base: number, K: number, tag: string): Promise<void> {
  const alphas = new Float32Array(8).fill(1.0);
  for (let j = 0; j < K; j++) alphas[j] = Math.pow(0.5, base + j);
  ex.writeBuffer("armijoAlphas", alphas);
  const st0 = new Float32Array(16); st0[8] = 1.0;
  ex.writeBuffer("armijoStatus", st0);
  ex.beginBatch(`dbg-batch-${tag}`);
  for (let j = 0; j < K; j++) {
    driver.armijoTrialPasses(j, mat, contact, 701 + base + j);
  }
  driver.armijoSelectPass();
  // mirror newtonRound: commit arbitration + refresh + predicated commit
  ex.runPass({
    shader: "newton-control", entry: "commit_arm",
    groups: [[
      { binding: 50, buffer: "armijoStatus" },
      { binding: 51, buffer: "armijoCandidates" },
      { binding: 52, buffer: "newtonCtl" },
      { binding: 53, buffer: "e0Store" },
      { binding: 54, buffer: "newtonStatus" },
      { binding: 55, buffer: "uniformBank", offset: 39 * 256, size: 4 },
    ]],
    x: 1,
  });
  driver.rebuildTrialPasses(mat, contact);
  await ex.submitBatch(false);
  const sr = await ex.readBufferDebug("armijoStatus", `dbg-st-${tag}`, false);
  const sf = new Float32Array(sr);
  console.log(`${tag}: trials=${sf[3]} accepted=${sf[0]}`);
}
ex.writeBuffer("newtonCtl", new Float32Array(16));
ex.writeBuffer("newtonStatus", new Float32Array(20));
await runBatch(0, 4, "b0");
await runBatch(4, 4, "b1");
await runBatch(8, 2, "b2");
const cand = new Float32Array(await ex.readBufferDebug("armijoCandidates", "dbg-cand", false));
for (let j = 0; j < 4; j++) {
  console.log(`row${j} =`, [...cand.subarray(j * 8, j * 8 + 8)].map((x) => x.toExponential(1)).join(","));
}
ex.destroy();
