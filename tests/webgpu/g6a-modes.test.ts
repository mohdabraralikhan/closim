// G6A production modes: unit logic (no device) + device selection, report,
// fallback accounting, and per-mode submit rebaseline (strip scene).
import { describe, it, expect, beforeAll } from "vitest";
import {
  applyPreconditionerMode, resolvePreconditioner, DEFAULT_NEWTON_CONFIG,
  type DeviceNewtonConfig,
} from "../../src/backend/webgpu/gpu-newton.js";
import type { ClothScene } from "../../src/physics/scene.js";
import type { DeviceFixture } from "./device-setup.js";
import { sharedDevice, resetDeviceState, stripScene } from "./device-setup.js";

function freshCfg(): DeviceNewtonConfig {
  return { ...DEFAULT_NEWTON_CONFIG };
}

describe("G6A mode mapping + resolution (unit)", () => {
  it("jacobi clears all candidate flags", () => {
    const cfg = freshCfg();
    cfg.useSchwarz = true;
    cfg.useCoarsePcg = true;
    applyPreconditionerMode(cfg, "jacobi");
    expect(cfg.useBlockJacobi).toBe(false);
    expect(cfg.useSchwarz).toBe(false);
    expect(cfg.useMas).toBe(false);
    expect(cfg.useCoarseC0).toBe(false);
    expect(cfg.useCoarsePcg).toBe(false);
  });

  it("schwarz1 sets only the Schwarz flag", () => {
    const cfg = freshCfg();
    applyPreconditionerMode(cfg, "schwarz1");
    expect(cfg.useSchwarz).toBe(true);
    expect(cfg.useCoarsePcg).toBe(false);
    expect(cfg.useMas).toBe(false);
    expect(cfg.useBlockJacobi).toBe(false);
  });

  it("mas-c1-8 sets Schwarz + coarse-PCG with K=8", () => {
    const cfg = freshCfg();
    applyPreconditionerMode(cfg, "mas-c1-8");
    expect(cfg.useSchwarz).toBe(true);
    expect(cfg.useCoarsePcg).toBe(true);
    expect(cfg.coarseIters).toBe(8);
    expect(cfg.useMas).toBe(false);
    expect(cfg.useCoarseC0).toBe(false);
  });

  it("healthy topology resolves with no fallback", () => {
    for (const mode of ["jacobi", "schwarz1", "mas-c1-8"] as const) {
      const cfg = freshCfg();
      applyPreconditionerMode(cfg, mode);
      expect(resolvePreconditioner(cfg, 4)).toBeNull();
      // flags untouched by resolution
      if (mode === "mas-c1-8") expect(cfg.useCoarsePcg).toBe(true);
      if (mode === "schwarz1") expect(cfg.useSchwarz).toBe(true);
    }
  });

  it("missing topology degrades every candidate to jacobi with a record", () => {
    const cfgs: DeviceNewtonConfig[] = [];
    for (const mode of ["schwarz1", "mas-c1-8"] as const) {
      const cfg = freshCfg();
      applyPreconditionerMode(cfg, mode);
      cfgs.push(cfg);
    }
    const block = freshCfg();
    block.useBlockJacobi = true;
    cfgs.push(block);
    for (const cfg of cfgs) {
      const fb = resolvePreconditioner(cfg, 0);
      expect(fb).not.toBeNull();
      expect(fb!.actual).toBe("jacobi");
      expect(fb!.reason).toBe("missing-topology");
      expect(cfg.useSchwarz).toBe(false);
      expect(cfg.useCoarsePcg).toBe(false);
      expect(cfg.useMas).toBe(false);
      expect(cfg.useBlockJacobi).toBe(false);
    }
  });

  it("jacobi with no topology needs no fallback", () => {
    const cfg = freshCfg();
    expect(resolvePreconditioner(cfg, 0)).toBeNull();
  });
});

let strip: DeviceFixture | null = null;

beforeAll(async () => {
  strip = await sharedDevice("g6a-strip", stripScene, { contactCapacity: 64, pairCapacity: 512 });
}, 180000);

describe("G6A device mode + report + rebaseline", () => {
  it("default mode stays jacobi with a populated report", async () => {
    if (!strip) return;
    const { solver, ex } = strip;
    const scene = (solver as unknown as { scene: ClothScene }).scene;
    resetDeviceState(strip, Float64Array.from(scene.positions));
    const s0 = ex.ledger.submits;
    const r0 = solver.hotLoopReadbacks;
    await solver.stepGpu(1 / 60, { newtonIters: 2 });
    const rep = solver.lastStepReport!;
    expect(rep).not.toBeNull();
    expect(rep.actualPreconditioner).toBe("jacobi");
    expect(rep.fallback).toBeNull();
    expect(rep.submits).toBe(ex.ledger.submits - s0);
    expect(rep.syncs).toBe(solver.hotLoopReadbacks - r0);
    expect(rep.newtonIters).toBeGreaterThan(0);
    expect(rep.finePcgIters).toBeGreaterThan(0);
    expect(rep.coarsePcgIters).toBe(0);
    expect(Number.isFinite(rep.residual)).toBe(true);
    expect(Number.isFinite(rep.energy)).toBe(true);
  }, 240000);

  it("mas-c1-8 mode reports coarse iters and still converges", async () => {
    if (!strip) return;
    const { solver, ex } = strip;
    const scene = (solver as unknown as { scene: ClothScene }).scene;
    solver.setPreconditionerMode("mas-c1-8");
    try {
      resetDeviceState(strip, Float64Array.from(scene.positions));
      await solver.stepGpu(1 / 60, { newtonIters: 2 });
      const rep = solver.lastStepReport!;
      expect(rep.requestedPreconditioner).toBe("mas-c1-8");
      expect(rep.actualPreconditioner).toBe("mas-c1-8");
      expect(rep.coarsePcgIters).toBeGreaterThan(0);
      expect(Number.isFinite(rep.residual)).toBe(true);
      void ex;
    } finally {
      // back to direct-flags behavior for other tests sharing this file only;
      // (shared fixtures are per-file, so this is hygiene, not coupling)
      (solver as unknown as { modeExplicit: boolean }).modeExplicit = false;
      solver.preconditionerMode = "jacobi";
    }
  }, 240000);

  it("missing topology falls back to jacobi with an accounting record", async () => {
    if (!strip) return;
    const { solver, driver } = strip;
    const scene = (solver as unknown as { scene: ClothScene }).scene;
    const realDoms = driver.c.schwarzDoms;
    expect(realDoms).toBeGreaterThan(0);
    driver.c.schwarzDoms = 0;
    solver.setPreconditionerMode("mas-c1-8");
    try {
      resetDeviceState(strip, Float64Array.from(scene.positions));
      solver.configureStep(1 / 60);
      await solver.evaluateNewtonState(false, 1, 1 / 60);
      const fbBefore = solver.fallbackLog.length;
      await driver.pcgSolve();
      expect(driver.lastFallback).not.toBeNull();
      expect(driver.lastFallback!.actual).toBe("jacobi");
      expect(driver.lastFallback!.reason).toBe("missing-topology");
      expect(solver.fallbackLog.length).toBe(fbBefore); // step-level log, not per-solve
      expect(driver.lastActualMethod).toBe("jacobi");
    } finally {
      driver.c.schwarzDoms = realDoms;
      (solver as unknown as { modeExplicit: boolean }).modeExplicit = false;
      solver.preconditionerMode = "jacobi";
      applyPreconditionerMode(driver.cfg, "jacobi");
    }
  }, 240000);

  it("per-mode submit rebaseline on strip (informational + generous caps)", async () => {
    if (!strip) return;
    const { solver, ex } = strip;
    const scene = (solver as unknown as { scene: ClothScene }).scene;
    // Hermetic reset: rest positions + zero velocities (scene arrays drift
    // via per-step snapshots, and resetDeviceState leaves velocity alone).
    const rest = stripScene();
    const packV4 = (X: Float64Array): Float32Array => {
      const n = scene.mesh.count;
      const out = new Float32Array(n * 4);
      for (let i = 0; i < n; i++) {
        out[i * 4] = Math.fround(X[i * 3]);
        out[i * 4 + 1] = Math.fround(X[i * 3 + 1]);
        out[i * 4 + 2] = Math.fround(X[i * 3 + 2]);
      }
      return out;
    };
    const table: Record<string, { submits: number; syncs: number }> = {};
    for (const mode of ["jacobi", "schwarz1", "mas-c1-8"] as const) {
      solver.setPreconditionerMode(mode);
      resetDeviceState(strip, Float64Array.from(rest.positions));
      ex.writeBuffer("velocity", packV4(Float64Array.from(rest.velocities)));
      // Warm steps first: early-transient steps do more Newton work by design.
      for (let w = 0; w < 6; w++) await solver.stepGpu(1 / 60, { newtonIters: 2 });
      const s0 = ex.ledger.submits;
      const r0 = solver.hotLoopReadbacks;
      await solver.stepGpu(1 / 60, { newtonIters: 2 });
      table[mode] = {
        submits: ex.ledger.submits - s0,
        syncs: solver.hotLoopReadbacks - r0,
      };
      // eslint-disable-next-line no-console
      console.log(`[g6a-rebaseline] strip ni=2 steady ${mode}: submits=${table[mode].submits} syncs=${table[mode].syncs}`);
    }
    (solver as unknown as { modeExplicit: boolean }).modeExplicit = false;
    solver.preconditionerMode = "jacobi";
    // Regression caps (transient anchors from identical hermetic resets; the
    // settled-state anchor remains 7d's 240). Mode deltas, not absolutes,
    // carry the rebaseline signal.
    expect(table["jacobi"].submits).toBeLessThanOrEqual(450);
    expect(table["schwarz1"].submits).toBeLessThanOrEqual(450);
    expect(table["mas-c1-8"].submits).toBeLessThanOrEqual(475);
    for (const mode of ["jacobi", "schwarz1", "mas-c1-8"] as const) {
      expect(table[mode].syncs).toBeLessThanOrEqual(12);
    }
  }, 240000);
});
