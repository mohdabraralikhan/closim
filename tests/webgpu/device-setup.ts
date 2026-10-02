// Shared device-test setup: builds a solver, brings up the real device, and
// returns null (→ test skips) when no binding/adapter exists. NEVER throws for
// missing hardware: every device test must skip cleanly on headless CI.
import { WebGpuSolver } from "../../src/backend/webgpu/gpu-solver.js";
import type { GpuExecutor } from "../../src/backend/webgpu/gpu-executor.js";
import type { DeviceNewtonDriver } from "../../src/backend/webgpu/gpu-newton.js";
import { buildGrid, preprocess } from "../../src/mesh/mesh.js";
import { createScene, pinColumn } from "../../src/physics/scene.js";
import { DEFAULT_MATERIAL } from "../../src/physics/types.js";
import { offsetMesh, preprocessMerged } from "../helpers.js";

export interface DeviceFixture {
  solver: WebGpuSolver;
  ex: GpuExecutor;
  driver: DeviceNewtonDriver;
}

const shared = new Map<string, DeviceFixture | null>();

/**
 * File-scoped shared device (one Dawn device per scene-shape per worker).
 * First use compiles ~40 pipelines (seconds); subsequent tests reuse them.
 * Call resetDeviceState before each test for isolation.
 */
export async function sharedDevice(
  name: string,
  build: () => ReturnType<typeof createScene>,
  opts?: { contactCapacity?: number; pairCapacity?: number },
): Promise<DeviceFixture | null> {
  if (shared.has(name)) return shared.get(name)!;
  const fix = await requireDevice(build, opts);
  shared.set(name, fix);
  return fix;
}

/** Restore device buffers to a segment state (control transfers, no readback). */
export function resetDeviceState(fix: DeviceFixture, x0: Float64Array, x1?: Float64Array): void {
  const { solver, ex } = fix;
  const scene = (solver as unknown as { scene: { mesh: { count: number } } }).scene;
  const n = scene.mesh.count;
  const pack = (X: Float64Array): Float32Array => {
    const out = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) {
      out[i * 4] = Math.fround(X[i * 3]);
      out[i * 4 + 1] = Math.fround(X[i * 3 + 1]);
      out[i * 4 + 2] = Math.fround(X[i * 3 + 2]);
    }
    return out;
  };
  ex.writeBuffer("position", pack(x1 ?? x0));
  ex.writeBuffer("position0", pack(x0));
  ex.writeBuffer("xTrial", pack(x1 ?? x0));
  const cap = solver.executor!.bufferBytes.get("laggedN") ?? 16;
  ex.writeBuffer("laggedN", new Uint8Array(Math.max(16, cap)));
  ex.writeBuffer("execMarker", new Uint32Array([0, 0, 0, 0]));
  ex.writeBuffer("contactCount", new Uint32Array([0, 0, 0, 0]));
  ex.writeBuffer("breakFlag", new Float32Array(4));
  solver.configureStep(1 / 60);
  // driver-level counter hygiene mirrors the solver path (evaluateNewtonState
  // zeroes before every rebuild; this covers direct driver use in tests)
  fix.driver.zeroContactCounters();
  fix.driver.zeroBreakFlag();
}

export async function requireDevice(
  build: () => ReturnType<typeof createScene>,
  opts?: { contactCapacity?: number; pairCapacity?: number },
): Promise<DeviceFixture | null> {
  const scene = build();
  const solver = new WebGpuSolver();
  solver.initialize(scene);
  let ok = false;
  try {
    ok = await solver.initDevice(opts);
  } catch {
    ok = false;
  }
  if (!ok || !solver.executor || !solver.driver) return null;
  return { solver, ex: solver.executor, driver: solver.driver };
}

export function stripScene(): ReturnType<typeof createScene> {
  const g = buildGrid(6, 3, 0.2, 0.1);
  const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
  const scene = createScene(mesh, { ...DEFAULT_MATERIAL });
  pinColumn(scene, (x) => x < 1e-9);
  return scene;
}

/** Two approaching patches (G2 scene shape, 1 mm final gap). */
export function headOnScene(): { scene: ReturnType<typeof createScene>; x0: Float64Array; nA: number } {
  const A = offsetMesh(2, 2, 0.05, 0.05, -0.03, 0.02, 0);
  const B = offsetMesh(2, 2, 0.05, 0.05, 0.03, 0.02, 0);
  const mesh = preprocessMerged([A, B], 0.15);
  const scene = createScene(mesh, { ...DEFAULT_MATERIAL }, [0, 0, 0]);
  const x0 = Float64Array.from(mesh.positions);
  return { scene, x0, nA: A.positions.length / 3 };
}

/** Normalize a device TOI (2.0 sentinel) to mirror/CPU Infinity. */
export function normTOI(t: number): number {
  return t >= 2.0 - 1e-6 ? Infinity : t;
}
