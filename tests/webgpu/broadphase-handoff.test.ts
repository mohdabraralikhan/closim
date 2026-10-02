// G1 -> G2 handoff: CPU reference vs GPU pairs on identical scenes via the
// WebGpuSolver façade. Gate: same candidate set, GPU-resident AABBs, zero
// position readback. G2 CCD is active but CPU remains the golden reference.
import { describe, it, expect } from "vitest";
import { CpuBroadPhase, compareCandidateSets } from "../../src/collision/broadphase.js";
import { WebGpuSolver } from "../../src/backend/webgpu/gpu-solver.js";
import { buildGrid, preprocess } from "../../src/mesh/mesh.js";
import { createScene, pinColumn } from "../../src/physics/scene.js";
import { DEFAULT_MATERIAL } from "../../src/physics/types.js";
import { offsetMesh, preprocessMerged } from "../helpers.js";

describe("G1 solver handoff", () => {
  it("solver pairs match the CPU reference with zero position readback", async () => {
    const A = offsetMesh(3, 3, 0.06, 0.06, -0.05, 0.03, 0);
    const B = offsetMesh(3, 3, 0.06, 0.06, 0.05, 0.03, 0);
    const mesh = preprocessMerged([A, B], 0.15);
    const scene = createScene(mesh, { ...DEFAULT_MATERIAL });
    const solver = new WebGpuSolver();
    solver.initialize(scene);
    await solver.initGpu();
    const x0 = Float64Array.from(scene.positions);
    const x1 = Float64Array.from(x0);
    const nA = A.positions.length / 3;
    for (let i = 0; i < mesh.count; i++) {
      x1[i * 3] += i < nA ? 0.02 : -0.02; // swept into overlap
    }
    const pad = 0.002;
    const cpu = await new CpuBroadPhase({
      indices: mesh.indices, triCount: mesh.triCount, pad,
    }).build(x0, x1);
    const gpu = await solver.buildBroadphasePairs(x0, x1, pad);
    expect(cpu.pairs.length).toBeGreaterThan(0);
    const cmp = compareCandidateSets(cpu, gpu);
    expect(cmp.match).toBe(true);
    // Readback invariant: broad-phase never maps positions.
    expect(solver.forbiddenReadbacks).toBe(0);
    // G2 CCD active; CPU remains golden until TOI/set-parity gates pass.
    expect(solver.gpuCcdStatus().stage).toBe("G2-active");
  });

  it("strip scene exposes a usable GpuBroadPhase façade", async () => {
    const g = buildGrid(6, 3, 0.2, 0.1);
    const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
    const scene = createScene(mesh, { ...DEFAULT_MATERIAL });
    pinColumn(scene, (x) => x < 1e-9);
    const solver = new WebGpuSolver();
    solver.initialize(scene);
    const bp = solver.getBroadphase();
    expect(bp.name).toBe("GpuBroadPhase");
    const x0 = Float64Array.from(scene.positions);
    const res = await bp.build(x0, x0);
    expect(res.diagnostics.candidateOverflow).toBe(0);
  });
});
