// G6C Verification Harness — shared helpers.
// All helpers read device state via public driver/executor APIs only.
// No production solver code is modified.

import { buildGrid, preprocess } from "../../src/mesh/mesh.js";
import { createScene, pinColumn } from "../../src/physics/scene.js";
import { DEFAULT_MATERIAL } from "../../src/physics/types.js";
import { ContactSystem } from "../../src/collision/contact-assembly.js";
import { DEFAULT_CONTACT_PARAMS } from "../../src/collision/types.js";
import type { DeviceFixture } from "../webgpu/device-setup.js";
import { requireDevice } from "../webgpu/device-setup.js";
import type { DeviceNewtonDriver } from "../../src/backend/webgpu/gpu-newton.js";
import type { GpuExecutor } from "../../src/backend/webgpu/gpu-executor.js";

// ---------------------------------------------------------------------------
// Scene builders
// ---------------------------------------------------------------------------

export function stripScene() {
  const g = buildGrid(6, 3, 0.2, 0.1);
  const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
  const scene = createScene(mesh, { ...DEFAULT_MATERIAL });
  pinColumn(scene, (x) => x < 1e-9);
  return scene;
}

export function floorScene() {
  const g = buildGrid(8, 8, 0.16, 0.16);
  for (let i = 0; i < g.positions.length / 3; i++) g.positions[i * 3 + 1] += 0.05;
  const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
  const scene = createScene(mesh, { ...DEFAULT_MATERIAL }, [0, -9.81, 0]);
  scene.contact = new ContactSystem({ ...DEFAULT_CONTACT_PARAMS }, mesh.indices);
  scene.contact.setFloor(0);
  return scene;
}

export function foldScene() {
  const w = 0.16;
  const g = buildGrid(8, 8, w, w);
  const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
  const scene = createScene(mesh, { ...DEFAULT_MATERIAL }, [0, -9.81, 0]);
  for (let i = 0; i < mesh.count; i++) {
    const x = scene.positions[i * 3];
    if (x > w / 2) {
      scene.positions[i * 3] = w / 2 - 2 * (x - w / 2);
      scene.positions[i * 3 + 1] += 0.0015;
    }
  }
  scene.contact = new ContactSystem({ ...DEFAULT_CONTACT_PARAMS }, mesh.indices);
  return scene;
}

export function stiffScene(memK: number) {
  const g = buildGrid(6, 6, 0.12, 0.12);
  const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
  const b = DEFAULT_MATERIAL;
  return createScene(mesh, {
    ...b,
    stretchWarp: b.stretchWarp * memK,
    stretchWeft: b.stretchWeft * memK,
    shear: b.shear * memK,
  }, [0, -9.81, 0]);
}

export function restingFloorScene() {
  const g = buildGrid(6, 6, 0.12, 0.12);
  for (let i = 0; i < g.positions.length / 3; i++) g.positions[i * 3 + 1] += 0.0012;
  const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
  const scene = createScene(mesh, { ...DEFAULT_MATERIAL }, [0, -9.81, 0]);
  const c = new ContactSystem({ ...DEFAULT_CONTACT_PARAMS }, mesh.indices);
  c.setFloor(0);
  scene.contact = c;
  return scene;
}

// 1k / 10k / 50k vertex grids (approximate — buildGrid gives count ≈ W*H)
export function grid1kScene() {
  const g = buildGrid(32, 32, 0.96, 0.96);
  const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
  return createScene(mesh, { ...DEFAULT_MATERIAL }, [0, -9.81, 0]);
}

export function grid10kScene() {
  const g = buildGrid(100, 100, 3, 3);
  const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
  return createScene(mesh, { ...DEFAULT_MATERIAL }, [0, -9.81, 0]);
}

// 50k is practical only on capable hardware — test skips gracefully.
export function grid50kScene() {
  const g = buildGrid(224, 224, 6.72, 6.72);
  const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
  return createScene(mesh, { ...DEFAULT_MATERIAL }, [0, -9.81, 0]);
}

// ---------------------------------------------------------------------------
// Solver private access (test-local cast only — no production changes)
// ---------------------------------------------------------------------------

export type AdvInternals = {
  contactParamsNow(): {
    dHat: number; kappa: number; mu: number; fricEps: number;
    floorY: number; floorOn: number; dMin: number; contactCapacity: number;
  };
  materialNow(): { c00: number; c11: number; c01: number; g: number; thickness: number };
  pushSimParams(): void;
  contactCountNow: number;
  simImage: { newtonIteration: number };
  scene: {
    material: Record<string, number>;
    mesh: { count: number; triCount: number };
    positions: Float64Array;
    contact: { setFloor: (y: number) => void } | null;
  };
};

export const IN = (solver: unknown): AdvInternals => solver as AdvInternals;

// ---------------------------------------------------------------------------
// Buffer read helpers
// ---------------------------------------------------------------------------

export async function readF32(f: DeviceFixture, name: string): Promise<Float32Array> {
  return new Float32Array(await f.ex.readBufferDebug(name, `verify-${name}`, false));
}

export async function readU32(f: DeviceFixture, name: string): Promise<Uint32Array> {
  return new Uint32Array(await f.ex.readBufferDebug(name, `verify-${name}`, false));
}

export async function readScalar32(f: DeviceFixture, name: string, bytes = 4): Promise<number> {
  const raw = await f.ex.readSmall(name, bytes, `verify-${name}`, "scalar");
  return new Float32Array(raw)[0];
}

export async function readScalarU32(f: DeviceFixture, name: string, bytes = 4): Promise<number> {
  const raw = await f.ex.readSmall(name, bytes, `verify-${name}`, "scalar");
  return new Uint32Array(raw)[0];
}

// ---------------------------------------------------------------------------
// State consistency snapshot
// ---------------------------------------------------------------------------

/** Snapshot all state-consistency-relevant buffers for a Newton round. */
export async function snapshotRoundState(f: DeviceFixture) {
  const [position, xTrial, gradient, contactDiag, rhs, contactDist, contactW] = await Promise.all([
    readF32(f, "position"),
    readF32(f, "xTrial"),
    readF32(f, "gradient"),
    readF32(f, "contactDiag"),
    readF32(f, "rhs"),
    readF32(f, "contactDist"),
    readF32(f, "contactW"),
  ]);
  const contactCount = await readScalarU32(f, "contactCount");
  return { position, xTrial, gradient, contactDiag, rhs, contactDist, contactW, contactCount };
}

/** Re-derive all contact + gradient + rhs buffers from scratch at `position`. */
export async function rederiveAtPosition(f: DeviceFixture): Promise<void> {
  const { ex, driver } = f;
  driver.zeroContactCounters();
  ex.beginBatch("verify-rederive");
  driver.broadphasePasses(0.002, false);
  driver.contactPasses(IN(f.solver).contactParamsNow(), "position");
  driver.femPasses(IN(f.solver).materialNow(), false);
  driver.contactDiagPass();
  driver.rhsAt("position");
  await ex.submitBatch(false);
}

/** Re-derive all contact + gradient + rhs buffers from scratch at `xTrial`. */
export async function rederiveAtTrial(f: DeviceFixture): Promise<void> {
  const { ex, driver } = f;
  driver.zeroContactCounters();
  ex.beginBatch("verify-rederive-trial");
  driver.broadphasePasses(0.002, true);
  driver.contactPasses(IN(f.solver).contactParamsNow(), "xTrial");
  driver.femPasses(IN(f.solver).materialNow(), true);
  driver.contactDiagPass();
  driver.rhsAt("xTrial");
  await ex.submitBatch(false);
}

// ---------------------------------------------------------------------------
// Diff helpers
// ---------------------------------------------------------------------------

export function maxDiff(a: Float32Array, b: Float32Array): number {
  let d = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) d = Math.max(d, Math.abs(a[i] - b[i]));
  return d;
}

export function maxDiffVec4(a: Float32Array, b: Float32Array): number {
  let d = 0;
  for (let i = 0; i < a.length; i += 4) {
    for (let k = 0; k < 3; k++) d = Math.max(d, Math.abs(a[i + k] - b[i + k]));
  }
  return d;
}

export function rmsVec4(a: Float32Array): number {
  let s = 0; let n = 0;
  for (let i = 0; i < a.length; i += 4) {
    for (let k = 0; k < 3; k++) { s += a[i + k] ** 2; n++; }
  }
  return n > 0 ? Math.sqrt(s / n) : 0;
}

export function maxAbs(a: Float32Array): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i]));
  return m;
}

export function allFinite(a: Float32Array): boolean {
  for (let i = 0; i < a.length; i++) if (!Number.isFinite(a[i])) return false;
  return true;
}

// ---------------------------------------------------------------------------
// Standard contact params (mirrors the solver's defaults)
// ---------------------------------------------------------------------------

export const CONTACT_PARAMS = {
  dHat: 0.002,
  kappa: 50,
  mu: 0.3,
  fricEps: 1e-4,
  floorY: 1e30,
  floorOn: 0,
  dMin: 1e-4,
};

// ---------------------------------------------------------------------------
// Standard Newton-round wiring (matches stepNewtonGpuControlled seeding)
// ---------------------------------------------------------------------------

export function seedRoundState(
  f: DeviceFixture,
  E0: number,
  round: number,
): void {
  const { ex, driver } = f;
  ex.writeBuffer("e0Store", new Float32Array([E0, 0, 0, 0]));
  driver.bankF(42, 1e-5);   // NewtonTol  (slot GpuUniformSlot.NewtonTol)
  driver.bankF(43, 0.002);  // NewtonTrust (slot GpuUniformSlot.NewtonTrust)
  ex.writeBuffer("newtonCtl", new Float32Array(16));
  const nst = new Float32Array(20);
  nst[0] = round;
  ex.writeBuffer("newtonStatus", nst);
}

// ---------------------------------------------------------------------------
// Beta calculation (mirror of newton.ts / stepGpuDevice)
// ---------------------------------------------------------------------------

export function betaFor(mat: Record<string, number>): number {
  return Math.max(mat.stretchWarp, mat.stretchWeft, mat.shear) * mat.thickness * 0.1 + 1e-6;
}
