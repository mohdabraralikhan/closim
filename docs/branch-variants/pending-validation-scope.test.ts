// G6C validation error scopes: a poisoned WebGPU queue can leave status
// buffers stale/zero, which decodes as convergence. The executor therefore
// (a) records uncaptured errors via a device listener installed at create,
// and (b) wraps newtonRound/armijoBatch in a validation scope that is popped
// BEFORE the status read — any error throws instead of decoding.
//
// This file proves the mechanism in both directions: clean work leaves no
// errors, and an induced device validation error is observed (via the scope,
// the listener, or a synchronous throw — silence is the only failure).
import { describe, it, expect } from "vitest";
import { decodeSolverStatus, GpuUniformSlot } from "../../src/backend/webgpu/gpu-buffers.js";
import { loadWgslSource } from "../../src/backend/webgpu/gpu-executor.js";
import type { DeviceFixture } from "./device-setup.js";
import { requireDevice, stripScene } from "./device-setup.js";

describe("G6C validation error scopes", () => {
  it("empty scope pops clean and reports support honestly", async () => {
    const fix: DeviceFixture | null = await requireDevice(stripScene, { contactCapacity: 64, pairCapacity: 512 });
    if (!fix) return;
    try {
      const { ex } = fix;
      expect(typeof ex.validationScopesSupported).toBe("boolean");
      ex.pushValidationScope();
      await ex.popValidationScope("probe-empty");
      expect(ex.consumeUncapturedErrors()).toEqual([]);
    } finally {
      fix.ex.destroy();
    }
  }, 180000);

  it("clean newtonRound leaves no pending validation errors", async () => {
    const fix: DeviceFixture | null = await requireDevice(stripScene, { contactCapacity: 64, pairCapacity: 512 });
    if (!fix) return;
    try {
      const { solver, ex, driver } = fix;
      solver.configureStep(1 / 60);
      const ev = await solver.evaluateNewtonState(false, 1, 1 / 60);
      ex.writeBuffer("e0Store", new Float32Array([ev.status.energy, 0, 0, 0]));
      driver.bankF(GpuUniformSlot.NewtonTol, 1e-5);
      driver.bankF(GpuUniformSlot.NewtonTrust, 0.002);
      ex.writeBuffer("newtonCtl", new Float32Array(16));
      ex.writeBuffer("newtonStatus", new Float32Array(20));
      const rawMt = (solver as unknown as { scene: { material: Record<string, number> } }).scene.material;
      const beta = Math.max(rawMt.stretchWarp, rawMt.stretchWeft, rawMt.shear) * rawMt.thickness * 0.1 + 1e-6;
      const mt = (solver as unknown as {
        materialNow(): { c00: number; c11: number; c01: number; g: number; thickness: number };
        contactParamsNow(): {
          dHat: number; kappa: number; mu: number; fricEps: number;
          floorY: number; floorOn: number; dMin: number; contactCapacity: number;
        };
      });
      // newtonRound pops its own scope internally: a throw here would mean
      // the device flagged genuinely clean work (spurious scope noise).
      const st = await driver.newtonRound({
        round: 0, beta, mat: mt.materialNow(), contact: mt.contactParamsNow(),
        batchKs: [4, 4, 2], evalIndexBase: 0,
      });
      expect(st.armijoAccepted).toBe(true);
      expect(ex.consumeUncapturedErrors()).toEqual([]);
    } finally {
      fix.ex.destroy();
    }
  }, 300000);

  it("standalone Newton evaluation fails before returning a status after a submit error", async () => {
    const fix = await requireDevice(stripScene, { contactCapacity: 64, pairCapacity: 512 });
    if (!fix) return;
    try {
      const { solver, ex } = fix;
      solver.configureStep(1 / 60);
      const submit = ex.submitBatch.bind(ex);
      let injected = false;
      ex.submitBatch = async (wait = false): Promise<void> => {
        await submit(wait);
        if (!injected) {
          injected = true;
          const buffer = ex.buffers.get("breakFlag");
          (ex.device as any).queue.writeBuffer(buffer, 1 << 30, new Uint8Array(16));
        }
      };
      await expect(solver.evaluateNewtonState(false, 1, 1 / 60)).rejects.toThrow(
        /evaluate-newton-state|validation/i,
      );
      expect(injected).toBe(true);
    } finally {
      fix.ex.destroy();
    }
  }, 300000);

  it("an induced device validation error is observed, never silent", async () => {
    const fix: DeviceFixture | null = await requireDevice(stripScene, { contactCapacity: 64, pairCapacity: 512 });
    if (!fix) return;
    try {
      const { ex } = fix;
      if (!ex.validationScopesSupported && typeof (ex.device as any)?.pushErrorScope !== "function") {
        // eslint-disable-next-line no-console
        console.log("[g6c-validation] binding lacks error scopes — listener path only");
      }
      ex.pushValidationScope();
      // Out-of-bounds queue write: a device-timeline validation error per spec
      // (some bindings throw synchronously instead — also loud, also fine).
      let syncThrew = false;
      try {
        const buf = ex.buffers.get("breakFlag");
        (ex.device as any).queue.writeBuffer(buf, 1 << 30, new Uint8Array(16));
      } catch {
        syncThrew = true;
      }
      let scopeThrew = false;
      try {
        await ex.popValidationScope("probe-oob");
      } catch {
        scopeThrew = true;
      }
      // Give the uncaptured listener a chance to fire, then drain.
      try { await (ex.device as any).queue.onSubmittedWorkDone?.(); } catch { /* best effort */ }
      const drained = ex.consumeUncapturedErrors();
      // eslint-disable-next-line no-console
      console.log(`[g6c-validation] syncThrew=${syncThrew} scopeThrew=${scopeThrew} ` +
        `listenerCaught=${drained.length} supported=${ex.validationScopesSupported}`);
      expect(syncThrew || scopeThrew || drained.length > 0).toBe(true);
    } finally {
      fix.ex.destroy();
    }
  }, 180000);

  it("catches an intentionally invalid bind group in a scoped test operation", async () => {
    const fix: DeviceFixture | null = await requireDevice(stripScene, { contactCapacity: 64, pairCapacity: 512 });
    if (!fix) return;
    try {
      const { ex } = fix;
      const dev = ex.device as any;
      await expect(ex.withValidationScope("test-invalid-bind-group", async () => {
        const layout = dev.createBindGroupLayout({ entries: [] });
        dev.createBindGroup({
          layout,
          entries: [{ binding: 0, resource: { buffer: ex.buffers.get("breakFlag") } }],
        });
      })).rejects.toThrow(/validation|bind group|binding/i);
    } finally {
      fix.ex.destroy();
    }
  }, 180000);

  it("catches an invalid pipeline/layout without injecting it during initialization", async () => {
    const fix: DeviceFixture | null = await requireDevice(stripScene, { contactCapacity: 64, pairCapacity: 512 });
    if (!fix) return;
    try {
      const { ex } = fix;
      const dev = ex.device as any;
      await expect(ex.withValidationScope("test-invalid-pipeline-layout", async () => {
        const module = dev.createShaderModule({ label: "test/invalid-layout", code: loadWgslSource("marker") });
        const layout = dev.createPipelineLayout({ bindGroupLayouts: [] });
        dev.createComputePipeline({
          label: "test/invalid-layout-pipeline",
          layout,
          compute: { module, entryPoint: "mark_stage" },
        });
      })).rejects.toThrow(/validation|pipeline|layout|group/i);
    } finally {
      fix.ex.destroy();
    }
  }, 180000);

  it("routes a failed GPU step through CPU and cannot report convergence", async () => {
    const fix: DeviceFixture | null = await requireDevice(stripScene, { contactCapacity: 64, pairCapacity: 512 });
    if (!fix) return;
    try {
      const { solver, ex } = fix;
      const original = ex.withValidationScope.bind(ex);
      ex.withValidationScope = (label, operation) => original(label, async () => {
        if (label === "device-step") {
          const dev = ex.device as any;
          const layout = dev.createBindGroupLayout({ entries: [] });
          dev.createBindGroup({ layout, entries: [{ binding: 0, resource: { buffer: ex.buffers.get("breakFlag") } }] });
        }
        return operation();
      });
      const before = solver.fallbackUses;
      const status = await solver.stepGpu(1 / 60, { newtonIters: 1 });
      expect(status.failed).toBe(true);
      expect(status.converged).toBe(0);
      expect(status.finite).toBe(0);
      expect(status.requestedPath).toBe("GPU");
      expect(status.actualPath).toBe("CPU");
      expect(status.fallbackReason).toBe("validation-error");
      expect(solver.fallbackUses).toBe(before + 1);
      expect(solver.lastDevicePath).toEqual({
        requested: "GPU", actual: "CPU", fallbackReason: "validation-error",
      });
      const readback = await solver.readbackDiagnostics();
      expect(readback.failed).toBe(true);
      expect(readback.converged).toBe(0);
    } finally {
      fix.ex.destroy();
    }
  }, 180000);

  it("a zeroed or stale solver status cannot pass finite-and-converged", () => {
    const status = decodeSolverStatus(new ArrayBuffer(64));
    expect(status.finite && status.converged).toBe(0);
    expect(status.gradNorm).toBe(0);
    expect(status.converged).toBe(0);
  });

  it("assertNoUncapturedErrors throws and drains (pure listener path)", async () => {
    const fix: DeviceFixture | null = await requireDevice(stripScene, { contactCapacity: 64, pairCapacity: 512 });
    if (!fix) return;
    try {
      const { ex } = fix;
      ex.uncapturedErrors.push({ type: "test", message: "synthetic" });
      expect(() => ex.assertNoUncapturedErrors("probe-pure")).toThrow();
      // Drained by the throw: a second call is clean.
      ex.assertNoUncapturedErrors("probe-pure-clean");
      expect(ex.consumeUncapturedErrors()).toEqual([]);
    } finally {
      fix.ex.destroy();
    }
  }, 180000);
});
