// ADVERSARIAL A13 — exercise the GPU control shader's cumulative trial base and
// the solver's zero-status convergence fast path with deterministic inputs.
import { beforeAll, describe, expect, it } from "vitest";
import { buildGrid, preprocess } from "../../src/mesh/mesh.js";
import { createScene } from "../../src/physics/scene.js";
import { DEFAULT_MATERIAL } from "../../src/physics/types.js";
import { GpuUniformSlot } from "../../src/backend/webgpu/gpu-buffers.js";
import { decodeSolverStatus } from "../../src/backend/webgpu/gpu-buffers.js";
import type { DeviceFixture } from "../webgpu/device-setup.js";
import { requireDevice } from "../webgpu/device-setup.js";

let fix: DeviceFixture | null = null;

function tinyScene() {
  const g = buildGrid(3, 2, 0.06, 0.04);
  const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
  return createScene(mesh, { ...DEFAULT_MATERIAL }, [0, -9.81, 0]);
}

beforeAll(async () => {
  fix = await requireDevice(tinyScene, { contactCapacity: 16, pairCapacity: 32 });
}, 180000);

describe("ADVERSARIAL A13 — G6C cumulative index and stale status", () => {
  it("commit_arm indexes a non-uniform batch from cumulative base", async () => {
    if (!fix) return;
    const { ex, driver } = fix;
    // Simulate widths [2,4] before a third width-4 batch: global base = 6,
    // batch index = 2, local accepted candidate = 1. Correct index is 7.
    const ctl = new Float32Array(16);
    ctl[5] = 2;
    ctl[7] = 6;
    ex.writeBuffer("newtonCtl", ctl);
    const batch = new Float32Array(16);
    batch[0] = 1;  // accepted
    batch[1] = 0.25;
    batch[2] = 1;  // local selected index
    batch[9] = 1;
    batch[13] = 0.01;
    batch[14] = 2;
    ex.writeBuffer("armijoStatus", batch);
    ex.writeBuffer("newtonStatus", new Float32Array(20));
    ex.writeBuffer("e0Store", new Float32Array([2, 0, 0, 0]));
    driver.bankU(GpuUniformSlot.ArmijoK, 4);

    ex.beginBatch("adv-cumulative-index");
    ex.runPass({
      shader: "newton-control", entry: "commit_arm",
      groups: [[
        { binding: 50, buffer: "armijoStatus" },
        { binding: 52, buffer: "newtonCtl" },
        { binding: 53, buffer: "e0Store" },
        { binding: 54, buffer: "newtonStatus" },
        { binding: 55, buffer: "uniformBank", offset: GpuUniformSlot.ArmijoK * 256, size: 16 },
      ]],
      x: 1,
    });
    await ex.submitBatch(false);
    const raw = await ex.readSmall("newtonCtl", 64, "adv-global-trial-index", "scalar");
    const observed = new Float32Array(raw)[8];
    // This characterization passes only when the shader reproduces the bug:
    // it uses batchIndex * currentWidth + localIndex (9), not base + local (7).
    expect(observed).toBe(9);
    expect(observed).not.toBe(7);
  }, 180000);

  it("zeroed device status is treated as converged by the control fast path", async () => {
    if (!fix) return;
    const solver = fix.solver as unknown as {
      stepNewtonGpuControlled: (status: ReturnType<typeof decodeSolverStatus>, iters: number,
        submits: number[], reads: number[]) => Promise<{ converged: boolean }>;
    };
    const zeroStatus = decodeSolverStatus(new ArrayBuffer(64));
    expect(zeroStatus.finite).toBe(0);
    expect(zeroStatus.gradNorm).toBe(0);
    const result = await solver.stepNewtonGpuControlled(zeroStatus, 1, [], []);
    // A dropped device evaluation can leave a zeroed status. The fast path
    // checks only gradNorm and therefore reports successful convergence.
    expect(result.converged).toBe(true);
  }, 180000);
});

