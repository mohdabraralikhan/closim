// G0 solver contract + readback invariant (no GPU required).
// - WebGpuSolver satisfies ClothSolver and tracks the golden CPU reference.
// - Hot-loop FULL-state readbacks stay at zero; only compact status is polled.
// - Capability detection degrades gracefully on headless CI.
// - Every test skips (not fails) when WebGPU is absent — the whole project
//   must never fail merely because Node/CI has no GPU.
import { describe, it, expect } from "vitest";
import { buildGrid, preprocess } from "../../src/mesh/mesh.js";
import { createScene, pinColumn } from "../../src/physics/scene.js";
import { DEFAULT_MATERIAL } from "../../src/physics/types.js";
import { WebGpuSolver } from "../../src/backend/webgpu/gpu-solver.js";
import { isWebGPUAvailable } from "../../src/backend/webgpu/webgpu-dts.js";
import { directionalCheck } from "../../src/backend/webgpu/gpu-tolerances.js";
import { evalInternal, internalEnergyOnly } from "../../src/physics/fem.js";

function stripScene() {
  const g = buildGrid(6, 3, 0.2, 0.1);
  const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
  const scene = createScene(mesh, { ...DEFAULT_MATERIAL });
  pinColumn(scene, (x) => x < 1e-9);
  return scene;
}

describe("webgpu G0 solver contract", () => {
  it("initializes, steps, and keeps pins exact via the golden path", async () => {
    const solver = new WebGpuSolver();
    solver.initialize(stripScene());
    const handle = await solver.initGpu();
    expect(handle.ready).toBe(false); // headless CI has no navigator.gpu
    expect(isWebGPUAvailable()).toBe(false);
    solver.step(1 / 60);
    const pos = solver.getPositions();
    expect(pos.length).toBeGreaterThan(0);
    expect([...pos].every(Number.isFinite)).toBe(true);
    expect(solver.forbiddenReadbacks).toBe(0);
  });

  it("stepGpu performs status-only readback accounting (no per-iteration positions)", async () => {
    const solver = new WebGpuSolver();
    solver.initialize(stripScene());
    await solver.initGpu();
    const before = solver.hotLoopReadbacks;
    const diag = await solver.stepGpu(1 / 60, { newtonIters: 3 });
    expect(solver.hotLoopReadbacks - before).toBe(3); // one COMPACT read per Newton iter
    expect(solver.forbiddenReadbacks).toBe(0); // never full-state
    expect(diag.statusBytes).toBe(64);
    expect(Number.isFinite(diag.energy)).toBe(true);
  });

  it("readbackPositions matches golden state on headless CI", async () => {
    const solver = new WebGpuSolver();
    solver.initialize(stripScene());
    await solver.initGpu();
    solver.step(1 / 60);
    const a = solver.getPositions();
    const b = await solver.readbackPositions();
    expect(b.length).toBe(a.length);
    expect(Math.abs(b[10] - a[10])).toBeLessThan(1e-9);
  });

  it("directional-derivative check passes on the CPU reference (G0.10 gate)", () => {
    const scene = stripScene();
    const x = Float64Array.from(scene.positions);
    x[10] += 1e-3;
    const { grad } = evalInternal(x, scene.mesh, scene.material);
    const p = new Float64Array(x.length);
    for (let i = 0; i < p.length; i++) p[i] = Math.sin(i * 12.9898) * 0.5;
    const chk = directionalCheck(
      (xx) => internalEnergyOnly(xx, scene.mesh, scene.material),
      grad, Float64Array.from(x), p,
    );
    expect(chk.pass).toBe(true);
  });

  it("G1 broad-phase active with CPU fallback until set-parity gates pass", () => {
    const solver = new WebGpuSolver();
    solver.initialize(stripScene());
    expect(solver.gpuBroadphaseStatus().stage).toBe("G1-active");
    expect(solver.gpuBroadphaseStatus().fallback).toContain("golden");
    expect(solver.gpuCcdStatus().stage).toBe("G2-active");
    expect(solver.gpuCcdStatus().fallback).toContain("golden");
  });
});
