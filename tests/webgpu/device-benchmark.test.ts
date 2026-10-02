// G3 device benchmark: real Dawn dispatches on GTX 1050 at ~1k / ~10k / ~50k
// verts (patch settling onto a floor — full pipeline: predictor, broadphase,
// contact, membrane, bending, PCG, status-only readbacks). Informational only:
// no FPS assertions, no speedup claims. Reports host-side wall time, a full-E0
// evaluation probe (broadphase+contact+FEM+diagnostics, ends at the status
// sync), one PCG solve probe, submit/sync counts, and invariants.
//
// Cost drivers observed on GTX 1050 (documented tradeoffs, now quantified):
// FD-HVP PCG dominates (2 membrane+assemble evals per PCG iter), per-phase
// batching costs thousands of submits/step at 10k+, and the G0 reference
// assemble scan is O(n*m). Analytic membrane Hessian / prefix-sum gather /
// coarser batching are post-G3 optimizations, NOT G3 behavior changes.
import { describe, it } from "vitest";
import { buildGrid, preprocess } from "../../src/mesh/mesh.js";
import { createScene } from "../../src/physics/scene.js";
import { DEFAULT_MATERIAL } from "../../src/physics/types.js";
import { ContactSystem } from "../../src/collision/contact-assembly.js";
import { DEFAULT_CONTACT_PARAMS } from "../../src/collision/types.js";
import { sharedDevice } from "./device-setup.js";

const P = { ...DEFAULT_CONTACT_PARAMS };

async function benchSize(n: number, name: string): Promise<Record<string, number | string>> {
  const g = buildGrid(n, n, 0.02 * n, 0.02 * n);
  for (let i = 0; i < g.positions.length / 3; i++) g.positions[i * 3 + 1] += 0.05;
  const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
  const scene = createScene(mesh, { ...DEFAULT_MATERIAL }, [0, -9.81, 0]);
  const contact = new ContactSystem({ ...P }, mesh.indices);
  contact.setFloor(0);
  scene.contact = contact;
  const triCount = mesh.indices.length / 3;
  const fx = await sharedDevice(name, () => scene, {
    contactCapacity: Math.max(4096, triCount * 2),
    pairCapacity: Math.min(1 << 20, triCount * 8),
  });
  if (!fx) throw new Error("no WebGPU device for benchmark");
  const { solver, ex } = fx;
  solver.driver!.cfg.pcgIters = 20;
  // warmup (pays first-compile + cache warm) then 2 timed steps.
  // Stage split: the solver's encode/submit/readback accumulators are
  // per-call instantaneous (not cumulative), so stages are timed host-side
  // around calls that END at a device sync point — a full E0 evaluation
  // (broadphase+contact+FEM+diagnostics, ends in the status read) and one
  // PCG solve (ends in the residual syncs). Both are true GPU+host costs.
  await solver.stepGpu(1 / 60, { newtonIters: 2 });
  const sub0 = ex.ledger.submits;
  const rb0 = solver.hotLoopReadbacks;
  const t0 = performance.now();
  const STEPS = 2;
  for (let s = 0; s < STEPS; s++) {
    await solver.stepGpu(1 / 60, { newtonIters: 2 });
  }
  const wallMs = performance.now() - t0;
  const submitsPerStep = (ex.ledger.submits - sub0) / STEPS;
  const statusSyncsPerStep = (solver.hotLoopReadbacks - rb0) / STEPS;
  // single-shot stage probes at the final state (labeled as such)
  solver.configureStep(1 / 60);
  const tE0 = performance.now();
  const evalRes = await solver.evaluateNewtonState(false, 999, 1 / 60);
  const evalMs = performance.now() - tE0;
  const tP0 = performance.now();
  await solver.driver!.pcgSolve();
  const pcgMs = performance.now() - tP0;
  const d = await solver.readbackDiagnostics();
  const out = {
    verts: mesh.count,
    tris: triCount,
    steps: STEPS,
    newtonIters: 2,
    pcgIters: 20,
    wallMsPerStep: wallMs / STEPS,
    evalMsFullPipeline: evalMs,
    pcgMsSolve: pcgMs,
    evalContactCount: evalRes.contactCount,
    submitsPerStep: submitsPerStep,
    statusSyncsPerStep: statusSyncsPerStep,
    forbiddenReadbacks: ex.ledger.forbiddenReadbacks,
    energy: d.energy,
    minDistance: d.minDistance,
    finite: d.finite,
  };
  // minDistance = Infinity while still contact-free is the legitimate
  // no-contact sentinel (NOT a divergence signal), so it is exempt here.
  for (const [k, v] of Object.entries(out)) {
    if (k !== "minDistance" && typeof v === "number" && !Number.isFinite(v)) {
      throw new Error(`non-finite benchmark figure ${k} in ${name}`);
    }
  }
  if (d.finite !== 1) throw new Error(`benchmark diverged at ${name}`);
  return out;
}

describe("G3 device benchmark", () => {
  it("reports full-pipeline device timings at 1k / 10k / 50k verts", async () => {
    const small = await benchSize(32, "bench-1k");
    // eslint-disable-next-line no-console
    console.log("[g3-bench] 1k:", JSON.stringify(small));
    const medium = await benchSize(100, "bench-10k");
    // eslint-disable-next-line no-console
    console.log("[g3-bench] 10k:", JSON.stringify(medium));
    const large = await benchSize(224, "bench-50k");
    // eslint-disable-next-line no-console
    console.log("[g3-bench] 50k:", JSON.stringify(large));
  }, 1500000);
});
