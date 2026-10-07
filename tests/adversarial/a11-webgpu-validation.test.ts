// ADVERSARIAL A11 — WebGPU validation failures & zeroed-status trap.
//
// Probes:
//   1. Over-binding: bind group provides entries not in the pipeline layout
//   2. Under-binding: bind group omits required entries
//   3. Stale / mismatched pipeline layout: bind group created for one pipeline passed to another
//   4. Invalid dispatch: exceeding 65535 on a NO_SPLIT kernel
//   5. Zeroed status trap: a dropped / un-executed batch returns zeros.
//      We prove what decodeNewtonStatus and decodeArmijoStatus output from all-zeros
//      and assert the exact condition under which a dropped dispatch is misidentified
//      as a benign / non-failing state.
import { describe, it, expect, beforeAll } from "vitest";
import { buildGrid, preprocess } from "../../src/mesh/mesh.js";
import { createScene } from "../../src/physics/scene.js";
import { DEFAULT_MATERIAL } from "../../src/physics/types.js";
import type { DeviceFixture } from "../webgpu/device-setup.js";
import { requireDevice } from "../webgpu/device-setup.js";
import { decodeArmijoStatus, decodeNewtonStatus } from "../../src/backend/webgpu/gpu-newton.js";

let fix: DeviceFixture | null = null;

function tinyScene() {
  const g = buildGrid(3, 2, 0.06, 0.04);
  const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
  return createScene(mesh, { ...DEFAULT_MATERIAL }, [0, -9.81, 0]);
}

beforeAll(async () => {
  fix = await requireDevice(tinyScene, { contactCapacity: 16, pairCapacity: 32 });
}, 180000);

describe("ADVERSARIAL A11 — WebGPU validation failures & zeroed status trap", () => {
  it("A11.1 over-binding produces uncaptured validation error and drops command buffer", async () => {
    if (!fix) return;
    const { ex } = fix;
    const dev = ex.device as any;

    dev.pushErrorScope("validation");
    ex.beginBatch("adv-overbind");
    // broadphase-sort:sort_next only uses binding 6
    // We intentionally bind 6 AND 99 (over-binding)
    ex.runPass({
      shader: "broadphase-sort", entry: "sort_next",
      groups: [[{ binding: 6, buffer: "sortCursor" }, { binding: 99, buffer: "sortCursor" }]],
      x: 1,
    });
    await ex.submitBatch(false);

    const err = await dev.popErrorScope();
    // eslint-disable-next-line no-console
    console.log(`[A11.1] Over-binding error: ${err?.message?.slice(0, 160)}`);
    expect(err).not.toBeNull();
    expect(err.message).toMatch(/not present in the bind group layout/i);
  }, 180000);

  it("A11.2 under-binding produces uncaptured validation error and drops command buffer", async () => {
    if (!fix) return;
    const { ex } = fix;
    const dev = ex.device as any;

    dev.pushErrorScope("validation");
    ex.beginBatch("adv-underbind");
    // broadphase-sort:sort_next REQUIRES binding 6. We provide an empty bind group!
    ex.runPass({
      shader: "broadphase-sort", entry: "sort_next",
      groups: [[]],
      x: 1,
    });
    await ex.submitBatch(false);

    const err = await dev.popErrorScope();
    // eslint-disable-next-line no-console
    console.log(`[A11.2] Under-binding error: ${err?.message?.slice(0, 160)}`);
    expect(err).not.toBeNull();
    expect(err.message).toMatch(/binding.*missing|expected.*binding/i);
  }, 180000);

  it("A11.3 layout mismatch: passing bind group from different pipeline fails validation", async () => {
    if (!fix) return;
    const { ex } = fix;
    const dev = ex.device as any;

    // Get pipeline A (sort_next: uses binding 6) and pipeline B (sort_step_indexed: uses 0, 1, 5, 6)
    const pipeA = ex.getPipeline("broadphase-sort", "sort_next");
    const pipeB = ex.getPipeline("broadphase-sort", "sort_step_indexed");

    // Create bind group using pipeA layout
    const bgA = dev.createBindGroup({
      layout: pipeA.getBindGroupLayout(0),
      entries: [{ binding: 6, resource: { buffer: ex.buffers.get("sortCursor") } }],
    });

    dev.pushErrorScope("validation");
    // Encode command pass manually with mismatched pipeline B and bind group bgA
    const encoder = dev.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeB);
    pass.setBindGroup(0, bgA);
    pass.dispatchWorkgroups(1, 1, 1);
    pass.end();
    dev.queue.submit([encoder.finish()]);

    const err = await dev.popErrorScope();
    // eslint-disable-next-line no-console
    console.log(`[A11.3] Pipeline layout mismatch error: ${err?.message?.slice(0, 160)}`);
    expect(err).not.toBeNull();
    expect(err.message).toMatch(/layout|incompatible|match/i);
  }, 180000);

  it("A11.4 invalid dispatch > 65535 workgroups on NO_SPLIT kernel throws in executor", async () => {
    if (!fix) return;
    const { ex } = fix;

    let threw = "";
    try {
      ex.beginBatch("adv-invalid-dispatch");
      // broadphase-sort is in NO_SPLIT set
      ex.runPass({
        shader: "broadphase-sort", entry: "sort_next",
        groups: [[{ binding: 6, buffer: "sortCursor" }]],
        x: 65536, // > 65535
      });
      await ex.submitBatch(false);
    } catch (e) {
      threw = e instanceof Error ? e.message : String(e);
    }

    // eslint-disable-next-line no-console
    console.log(`[A11.4] Invalid dispatch exception: ${threw.slice(0, 140)}`);
    expect(threw).toContain("past the 65535 row-splittable cap");
  }, 180000);

  it("A11.5 zeroed status buffer trap: a zeroed status buffer must never look like successful convergence", async () => {
    // When a command buffer is dropped due to validation error, readBuffer returns
    // either whatever was previously in the buffer, or all zeros if newly allocated.
    const zeroBytes = new ArrayBuffer(80);
    const nst = decodeNewtonStatus(zeroBytes);

    // eslint-disable-next-line no-console
    console.log(`[A11.5] Decoded zero Newton status: converged=${nst.converged} failure=${nst.failure} ` +
      `directionValid=${nst.directionValid} armijoAccepted=${nst.armijoAccepted} ` +
      `gradNorm=${nst.gradNorm} stepNorm=${nst.stepNorm}`);

    // REQUIREMENT: "A zeroed status buffer must never look like successful convergence."
    expect(nst.converged).toBe(false);

    // DANGER REVEALED:
    // Notice that nst.failure is ALSO FALSE!
    // If client code tests `if (!nst.failure)`, it thinks the round succeeded!
    expect(nst.failure).toBe(false);

    // And directionValid is FALSE (0.0):
    expect(nst.directionValid).toBe(false);

    // And armijoAccepted is FALSE (0.0):
    expect(nst.armijoAccepted).toBe(false);

    // For Armijo status:
    const zeroArmBytes = new ArrayBuffer(64);
    const ast = decodeArmijoStatus(zeroArmBytes);

    // eslint-disable-next-line no-console
    console.log(`[A11.5] Decoded zero Armijo status: accepted=${ast.accepted} trialsEvaluated=${ast.trialsEvaluated} ` +
      `armijoFails=${ast.armijoFails} finite=${ast.finite}`);

    expect(ast.accepted).toBe(false);
    expect(ast.trialsEvaluated).toBe(0);
    // Notice that ast.armijoFails is 0 and ast.finite is FALSE!
    expect(ast.armijoFails).toBe(0);
    expect(ast.finite).toBe(false);
  }, 180000);
});
