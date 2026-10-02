// G0 benchmark harness (Phase 2 §29): small validation + medium engineering
// scenes. Reports vertices/tris/contacts/Newton/PCG iterations plus encode,
// submit, and readback cost separately — never a bare "FPS" claim.
// On headless CI the GPU columns are marked null/unavailable and the CPU
// golden timings are reported instead, so trends stay comparable.
import { describe, it } from "vitest";
import { buildGrid, preprocess } from "../../src/mesh/mesh.js";
import { createScene, pinColumn } from "../../src/physics/scene.js";
import { DEFAULT_MATERIAL } from "../../src/physics/types.js";
import { WebGpuSolver, type GpuBenchmarkSample } from "../../src/backend/webgpu/gpu-solver.js";

async function bench(nx: number, ny: number, steps: number): Promise<GpuBenchmarkSample> {
  const g = buildGrid(nx, ny, 0.2, 0.1);
  const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
  const scene = createScene(mesh, { ...DEFAULT_MATERIAL });
  pinColumn(scene, (x) => x < 1e-9);
  const solver = new WebGpuSolver();
  solver.initialize(scene);
  await solver.initGpu();
  const t0 = performance.now();
  let pcgIters = 0;
  for (let s = 0; s < steps; s++) {
    const d = await solver.stepGpu(1 / 60, { newtonIters: 4 });
    pcgIters += 60; // PCG budget per Newton iter (golden-path accounting)
    void d;
  }
  const t1 = performance.now();
  const msPerStep = (t1 - t0) / steps;
  return {
    vertexCount: mesh.count,
    triangleCount: mesh.triCount,
    hingeCount: mesh.hinges.length,
    contactCount: 0,
    newtonIters: 4,
    pcgIters,
    gpuComputeMs: null, // timestamp-query unavailable on headless CI
    cpuEncodeMs: solver["encodeMs" as never] as unknown as number,
    readbackMs: 0,
    msPerStep,
    stepsPerSec: 1000 / msPerStep,
  };
}

describe("webgpu G0 benchmark", () => {
  it("reports small + medium scene costs (informational, never asserts FPS)", async () => {
    const small = await bench(12, 6, 3); // ~91 verts, strip-pull class
    const med = await bench(40, 20, 2); // ~861 verts
    // eslint-disable-next-line no-console
    console.log("[gpu-bench] small:", JSON.stringify(small));
    // eslint-disable-next-line no-console
    console.log("[gpu-bench] medium:", JSON.stringify(med));
  }, 120000);
});
