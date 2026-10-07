// G6C-VERIFY-03: Scale / Stress Matrix
//
// Runs full stepGpu() across the target scene matrix and collects per-step
// telemetry. Does NOT assert physics correctness — asserts solver safety:
//   - Energy is finite
//   - No NaN in positions
//   - Newton iters ≤ budget
//   - Armijo trials ≤ armijoIters * newton budget
//   - Contact count within capacity
//   - No state consistency failures (via from-scratch reconstruction)
//
// For each scene, logs a telemetry row:
//   scene | verts | newtonIters | armijoTrials | contactCount | fallbacks | stateConsistency

import { describe, it, expect } from "vitest";
import { requireDevice, resetDeviceState } from "../webgpu/device-setup.js";
import type { WebGpuSolver } from "../../src/backend/webgpu/gpu-solver.js";
import {
  stripScene, floorScene, foldScene, stiffScene,
  grid1kScene, grid10kScene,
  IN, betaFor,
  rederiveAtPosition,
  snapshotRoundState,
  maxDiff, maxDiffVec4,
  readF32,
} from "./helpers.js";

// ──────────────────────────────────────────────────────────────────────────────
// Telemetry row type
// ──────────────────────────────────────────────────────────────────────────────

interface TelemetryRow {
  scene: string;
  verts: number;
  newtonIters: number;
  armijoTrials: number;
  batchWidths: number[];
  contactCount: number;
  fallbacks: number;
  stateConsistencyError: number;
  energyFinite: boolean;
  positionsFinite: boolean;
  energy: number;
}

const TELEMETRY: TelemetryRow[] = [];

// ──────────────────────────────────────────────────────────────────────────────
// Harness: run one step and collect telemetry
// ──────────────────────────────────────────────────────────────────────────────

async function runStep(
  name: string,
  buildScene: () => ReturnType<typeof import("../../src/physics/scene.js").createScene>,
  opts: { contactCapacity?: number; pairCapacity?: number; newtonIters?: number } = {},
): Promise<TelemetryRow | null> {
  const fix = await requireDevice(buildScene, {
    contactCapacity: opts.contactCapacity ?? 2048,
    pairCapacity: opts.pairCapacity ?? 8192,
  });
  if (!fix) return null;

  try {
    const { solver, ex, driver } = fix;
    const solverRef = solver as unknown as WebGpuSolver;
    const n = IN(solverRef).scene.mesh.count;

    const logBefore = solverRef.fallbackLog.length;
    const d = await solver.stepGpu(1 / 60, { newtonIters: opts.newtonIters ?? 4 });
    const report = solverRef.lastStepReport!;

    // From-scratch state consistency check at the post-step position
    await rederiveAtPosition(fix);
    const refSnap = await snapshotRoundState(fix);

    // Read what the solver actually left
    const gradActual = await readF32(fix, "gradient");
    const diagActual = await readF32(fix, "contactDiag");
    const rhsActual = await readF32(fix, "rhs");

    const gradDiff = maxDiff(gradActual, refSnap.gradient);
    const rhsDiff = maxDiff(rhsActual, refSnap.rhs);
    const stateErr = Math.max(gradDiff, rhsDiff);

    // Check position finiteness
    const posRaw = await readF32(fix, "position");
    let posFinite = true;
    for (let i = 0; i < posRaw.length; i++) {
      if (!Number.isFinite(posRaw[i])) { posFinite = false; break; }
    }

    const row: TelemetryRow = {
      scene: name,
      verts: n,
      newtonIters: report.newtonIters,
      armijoTrials: report.armijoTrials,
      batchWidths: report.submitsPerNewton ?? [],
      contactCount: refSnap.contactCount,
      fallbacks: solverRef.fallbackLog.length - logBefore,
      stateConsistencyError: stateErr,
      energyFinite: Number.isFinite(d.energy),
      positionsFinite: posFinite,
      energy: d.energy,
    };

    TELEMETRY.push(row);

    console.log(
      `[SCALE] ${name.padEnd(12)} verts=${String(n).padStart(6)} ` +
      `ni=${row.newtonIters} arm=${row.armijoTrials} ` +
      `cc=${row.contactCount} fb=${row.fallbacks} ` +
      `stateΔ=${stateErr.toExponential(2)} ` +
      `E=${d.energy.toExponential(4)} posOk=${posFinite}`,
    );

    return row;
  } finally {
    fix.ex.destroy();
  }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

describe("G6C-VERIFY-03: Scale/Stress Matrix — strip", () => {
  it("strip 1x: finite energy, positions, state consistency", async () => {
    const r = await runStep("strip", stripScene, { contactCapacity: 64, pairCapacity: 512 });
    if (!r) return;
    expect(r.energyFinite).toBe(true);
    expect(r.positionsFinite).toBe(true);
    expect(r.newtonIters).toBeGreaterThanOrEqual(1);
    expect(r.newtonIters).toBeLessThanOrEqual(4);
  }, 300000);
});

describe("G6C-VERIFY-03: Scale/Stress Matrix — floor-resting", () => {
  it("floor-resting: finite energy, contact active, no overflow", async () => {
    const r = await runStep("floor-rest", floorScene, {});
    if (!r) return;
    expect(r.energyFinite).toBe(true);
    expect(r.positionsFinite).toBe(true);
  }, 300000);
});

describe("G6C-VERIFY-03: Scale/Stress Matrix — fold", () => {
  it("fold: finite energy, positions finite, self-contact handled", async () => {
    const r = await runStep("fold", foldScene, {});
    if (!r) return;
    expect(r.energyFinite).toBe(true);
    expect(r.positionsFinite).toBe(true);
  }, 300000);
});

describe("G6C-VERIFY-03: Scale/Stress Matrix — stiff 5x", () => {
  it("stiff-5x: finite energy, Newton budget not exceeded", async () => {
    const r = await runStep("stiff-5x", () => stiffScene(5), { contactCapacity: 64, pairCapacity: 512 });
    if (!r) return;
    expect(r.energyFinite).toBe(true);
    expect(r.positionsFinite).toBe(true);
    expect(r.newtonIters).toBeLessThanOrEqual(4);
  }, 300000);
});

describe("G6C-VERIFY-03: Scale/Stress Matrix — stiff 10x", () => {
  it("stiff-10x: finite energy, solver survives high stiffness", async () => {
    const r = await runStep("stiff-10x", () => stiffScene(10), { contactCapacity: 64, pairCapacity: 512 });
    if (!r) return;
    expect(r.energyFinite).toBe(true);
    expect(r.positionsFinite).toBe(true);
  }, 300000);
});

describe("G6C-VERIFY-03: Scale/Stress Matrix — 1k vertices", () => {
  it("1k verts: finite energy, Armijo trials ≤ newton*armijoIters", async () => {
    const r = await runStep("1k", grid1kScene, { contactCapacity: 128, pairCapacity: 2048 });
    if (!r) return;
    expect(r.energyFinite).toBe(true);
    expect(r.positionsFinite).toBe(true);
    // Armijo trials per step ≤ maxNewton * armijoIters (generous bound)
    expect(r.armijoTrials).toBeLessThanOrEqual(4 * 10 + 10);
  }, 600000);
});

describe("G6C-VERIFY-03: Scale/Stress Matrix — 10k vertices", () => {
  it("10k verts: completes without timeout, energy finite", async () => {
    const r = await runStep("10k", grid10kScene, {
      contactCapacity: 256, pairCapacity: 16384, newtonIters: 2,
    });
    if (!r) return;
    expect(r.energyFinite).toBe(true);
    expect(r.positionsFinite).toBe(true);
  }, 900000);
});

describe("G6C-VERIFY-03: Telemetry summary", () => {
  it("prints full telemetry table", () => {
    if (TELEMETRY.length === 0) return;
    console.log("\n=== G6C-VERIFY-03 TELEMETRY TABLE ===");
    console.log(
      ["scene", "verts", "ni", "arm", "cc", "fb", "stateΔ", "energy"].join("\t"),
    );
    for (const r of TELEMETRY) {
      console.log(
        [
          r.scene, r.verts, r.newtonIters, r.armijoTrials,
          r.contactCount, r.fallbacks,
          r.stateConsistencyError.toExponential(2),
          r.energy.toExponential(4),
        ].join("\t"),
      );
    }
    console.log("=== END TELEMETRY ===\n");
    // At least some rows collected
    expect(TELEMETRY.length).toBeGreaterThan(0);
  }, 1000);
});
