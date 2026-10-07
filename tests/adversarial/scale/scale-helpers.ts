// MIMO scale-contact stress: shared scene builders, buffer readers, and
// invariant checkers for GPU contact handling at 1k / 10k / 50k verts.
//
// Design rules for every test in this directory:
// - NEVER require identical contact ORDER (atomic append order races).
// - Canonicalize contact identities (kind + id tuple) before comparison.
// - Distinguish class A (benign ordering differences) from class B
//   (physics/state corruption) explicitly in every assertion message.
import { buildGrid, preprocess } from "../../../src/mesh/mesh.js";
import { createScene, pinColumn } from "../../../src/physics/scene.js";
import { DEFAULT_MATERIAL } from "../../../src/physics/types.js";
import { ContactSystem } from "../../../src/collision/contact-assembly.js";
import { DEFAULT_CONTACT_PARAMS } from "../../../src/collision/types.js";
import { offsetMesh, preprocessMerged } from "../../helpers.js";
import type { DeviceFixture } from "../../webgpu/device-setup.js";

export type ScaleClass = "1k" | "10k" | "50k";

/** Grid resolution per class (single-patch scenes). verts = (g+1)^2. */
export const GRID: Record<ScaleClass, number> = { "1k": 32, "10k": 100, "50k": 224 };
/** Per-patch resolution for head-on scenes (verts = 2*(p+1)^2). */
export const PATCH: Record<ScaleClass, number> = { "1k": 22, "10k": 70, "50k": 158 };

/** Device caps per class. tiny forces overflow; big must clear it. */
export const TINY_CAP: Record<ScaleClass, number> = { "1k": 128, "10k": 512, "50k": 2048 };
export const BIG_CAP: Record<ScaleClass, number> = { "1k": 8192, "10k": 65536, "50k": 262144 };
export const PAIR_CAP: Record<ScaleClass, number> = { "1k": 65536, "10k": 196608, "50k": 131072 };

/** Repetitions per class (enough to expose ordering races, bounded by time). */
export const REPS: Record<ScaleClass, number> = { "1k": 8, "10k": 4, "50k": 2 };

export type SceneName = "fold" | "headon" | "floor" | "dense" | "pinned-floor";

/** 1. Fold: half the sheet reflected over, lifted 1.5 mm (self-contact). */
export function foldScene(g: number): ReturnType<typeof createScene> {
  const w = 0.16;
  const grid = buildGrid(g, g, w, w);
  const mesh = preprocess(grid.positions, grid.uv, grid.indices, 0.15);
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

/** 2. Head-on patches: 1 mm gap (engaged) + closing velocities. */
export function headOnScene(p: number): ReturnType<typeof createScene> {
  const w = 0.2;
  const gap = 0.001;
  const A = offsetMesh(p, p, w, w, -w / 2 - gap / 2, 0.02, -w / 2);
  const B = offsetMesh(p, p, w, w, w / 2 + gap / 2, 0.02, -w / 2);
  const mesh = preprocessMerged([A, B], 0.15);
  const scene = createScene(mesh, { ...DEFAULT_MATERIAL }, [0, 0, 0]);
  const nA = A.positions.length / 3;
  for (let i = 0; i < mesh.count; i++) {
    scene.velocities[i * 3] = i < nA ? 0.3 : -0.3;
  }
  scene.contact = new ContactSystem({ ...DEFAULT_CONTACT_PARAMS }, mesh.indices);
  return scene;
}

/** 3. Resting floor: sheet 1.2 mm above y=0 (dHat = 2 mm engages all verts). */
export function floorScene(g: number): ReturnType<typeof createScene> {
  const grid = buildGrid(g, g, 0.16, 0.16);
  for (let i = 0; i < grid.positions.length / 3; i++) grid.positions[i * 3 + 1] += 0.0012;
  const mesh = preprocess(grid.positions, grid.uv, grid.indices, 0.15);
  const scene = createScene(mesh, { ...DEFAULT_MATERIAL }, [0, -9.81, 0]);
  const c = new ContactSystem({ ...DEFAULT_CONTACT_PARAMS }, mesh.indices);
  c.setFloor(0);
  scene.contact = c;
  return scene;
}

/** 4. Dense self-contact: 3-layer stack 1 mm apart (all pairs within dHat). */
export function denseStackScene(g: number): ReturnType<typeof createScene> {
  const layers = [0, 0.001, 0.002].map((dy) =>
    offsetMesh(g, g, 0.16, 0.16, 0, dy, 0),
  );
  const mesh = preprocessMerged(layers, 0.15);
  const scene = createScene(mesh, { ...DEFAULT_MATERIAL }, [0, -9.81, 0]);
  scene.contact = new ContactSystem({ ...DEFAULT_CONTACT_PARAMS }, mesh.indices);
  return scene;
}

/** 5/6 use dense/fold scenes with tiny/oversized caps (no separate builder). */

/** 7. Alternating frames reuses floorScene with teleported heights (in test). */

/** Pinned floor: floor scene with the x=0 column pinned (pins-exact checks). */
export function pinnedFloorScene(g: number): ReturnType<typeof createScene> {
  const grid = buildGrid(g, g, 0.16, 0.16);
  for (let i = 0; i < grid.positions.length / 3; i++) grid.positions[i * 3 + 1] += 0.0012;
  const mesh = preprocess(grid.positions, grid.uv, grid.indices, 0.15);
  const scene = createScene(mesh, { ...DEFAULT_MATERIAL }, [0, -9.81, 0]);
  pinColumn(scene, (x) => x < 1e-9);
  const c = new ContactSystem({ ...DEFAULT_CONTACT_PARAMS }, mesh.indices);
  c.setFloor(0);
  scene.contact = c;
  return scene;
}

export function buildScene(name: SceneName, cls: ScaleClass): ReturnType<typeof createScene> {
  switch (name) {
    case "fold": return foldScene(GRID[cls]);
    case "headon": return headOnScene(PATCH[cls]);
    case "floor": return floorScene(GRID[cls]);
    case "dense": return denseStackScene(GRID[cls]);
    case "pinned-floor": return pinnedFloorScene(GRID[cls]);
  }
}

// --- access to WebGpuSolver privates (same pattern as other adversarial tests) ---
export type ScaleInternals = {
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
    mesh: { count: number };
    positions: Float64Array;
    velocities: Float64Array;
    pinned: Map<number, [number, number, number]>;
  };
};
export const IN = (solver: unknown): ScaleInternals => solver as ScaleInternals;

// --- buffer readers ---

export async function readCounter(fix: DeviceFixture, name: string): Promise<number> {
  const raw = await fix.ex.readSmall(name, 16, `scale-${name}`, "scalar");
  return new Uint32Array(raw)[0];
}

export interface CounterSet {
  pairCount: number; pairOverflow: number; pairScanned: number;
  primVT: number; primEE: number;
  contactCount: number; contactOverflow: number; contactScanned: number; contactFail: number;
}

export async function readCounters(fix: DeviceFixture): Promise<CounterSet> {
  const [pairCount, pairOverflow, pairScanned, primVT, primEE,
    contactCount, contactOverflow, contactScanned, contactFail] = await Promise.all([
    readCounter(fix, "pairCount"), readCounter(fix, "overflowFlag"),
    readCounter(fix, "pairScanned"), readCounter(fix, "primCountVT"),
    readCounter(fix, "primCountEE"), readCounter(fix, "contactCount"),
    readCounter(fix, "contactOverflow"), readCounter(fix, "contactScanned"),
    readCounter(fix, "contactFail"),
  ]);
  return {
    pairCount, pairOverflow, pairScanned, primVT, primEE,
    contactCount, contactOverflow, contactScanned, contactFail,
  };
}

export async function readF32(fix: DeviceFixture, name: string): Promise<Float32Array> {
  return new Float32Array(await fix.ex.readBufferDebug(name, `scale-${name}`, false));
}

export async function readU32(fix: DeviceFixture, name: string): Promise<Uint32Array> {
  return new Uint32Array(await fix.ex.readBufferDebug(name, `scale-${name}`, false));
}

export interface ContactLive {
  count: number; // attempted appends (may exceed capacity under overflow)
  cap: number;
  live: number; // min(count, cap): records actually stored
  W: Float32Array; N: Float32Array; Id: Uint32Array; Prm: Float32Array;
  Dist: Float32Array; TOI: Float32Array; Energy: Float32Array;
}

export async function readContactLive(fix: DeviceFixture, count: number, cap: number): Promise<ContactLive> {
  const [W, N, Id, Prm, Dist, TOI, Energy] = await Promise.all([
    readF32(fix, "contactW"), readF32(fix, "contactN"), readU32(fix, "contactId"),
    readF32(fix, "contactPrm"), readF32(fix, "contactDist"),
    readF32(fix, "contactTOI"), readF32(fix, "contactEnergy"),
  ]);
  return { count, cap, live: Math.min(count, cap), W, N, Id, Prm, Dist, TOI, Energy };
}

// --- canonicalization (order-free identity) ---

/** Canonical key for one contact record: kind + id tuple. */
export function contactKey(kind: number, i0: number, i1: number, i2: number, i3: number): string {
  return `${kind}:${i0},${i1},${i2},${i3}`;
}

/**
 * Map a device record to the CPU-mirror key format (gpuVtKey/gpuEeKey/floor).
 * Device ids are mesh-space here (no static meshes in the stress scenes).
 * Returns null for extended/static ids (not present in this suite).
 */
export function deviceKeyToMirrorKey(kind: number, i0: number, i1: number, i2: number, i3: number): string | null {
  const U32 = 0xffffffff;
  if (kind === 2) {
    if (i1 !== U32 || i2 !== U32 || i3 !== U32) return null;
    return `floor:${i0}`;
  }
  if (kind === 0) {
    const t = [i1, i2, i3].sort((x, y) => x - y).join("_");
    return `vt:${i0}:${t}`;
  }
  if (kind === 1) {
    const e1 = i0 < i1 ? `${i0}_${i1}` : `${i1}_${i0}`;
    const e2 = i2 < i3 ? `${i2}_${i3}` : `${i3}_${i2}`;
    return e1 < e2 ? `ee:${e1}:${e2}` : `ee:${e2}:${e1}`;
  }
  return null;
}

export function liveMirrorKeys(c: ContactLive): Array<string | null> {
  const out: Array<string | null> = [];
  for (let s = 0; s < c.live; s++) {
    const kind = Math.round(c.N[s * 4 + 3]);
    out.push(deviceKeyToMirrorKey(kind, c.Id[s * 4], c.Id[s * 4 + 1], c.Id[s * 4 + 2], c.Id[s * 4 + 3]));
  }
  return out;
}

export function liveKeys(c: ContactLive): string[] {
  const out: string[] = [];
  for (let s = 0; s < c.live; s++) {
    const kind = Math.round(c.N[s * 4 + 3]);
    out.push(contactKey(kind, c.Id[s * 4], c.Id[s * 4 + 1], c.Id[s * 4 + 2], c.Id[s * 4 + 3]));
  }
  return out.sort();
}

/** Exact duplicates inside one live set (same tuple twice). */
export function findDuplicates(sortedKeys: string[]): Array<{ key: string; times: number }> {
  const dups: Array<{ key: string; times: number }> = [];
  for (let i = 0; i < sortedKeys.length;) {
    let j = i + 1;
    while (j < sortedKeys.length && sortedKeys[j] === sortedKeys[i]) j++;
    if (j - i > 1) dups.push({ key: sortedKeys[i], times: j - i });
    i = j;
  }
  return dups;
}

export interface DupGroup {
  key: string;
  slots: number[];
  /** True when every record in the group is bitwise-identical (benign
   *  rediscovery of one primitive via several triangle pairs — the
   *  documented multiset semantics: barrier sums every entry). False means
   *  same identity with different payloads (torn/mixed writes: class B). */
  exact: boolean;
  firstDiff: string;
}

/** Group live slots by identity; verify payload equality within each group. */
export function checkDuplicateGroups(c: ContactLive): DupGroup[] {
  const byKey = new Map<string, number[]>();
  for (let s = 0; s < c.live; s++) {
    const kind = Math.round(c.N[s * 4 + 3]);
    const k = contactKey(kind, c.Id[s * 4], c.Id[s * 4 + 1], c.Id[s * 4 + 2], c.Id[s * 4 + 3]);
    const arr = byKey.get(k);
    if (arr) arr.push(s);
    else byKey.set(k, [s]);
  }
  const out: DupGroup[] = [];
  for (const [key, slots] of byKey) {
    if (slots.length < 2) continue;
    const r0 = slots[0];
    let exact = true;
    let firstDiff = "";
    for (let g = 1; g < slots.length && exact; g++) {
      const r = slots[g];
      for (let k = 0; k < 4; k++) {
        if (c.W[r * 4 + k] !== c.W[r0 * 4 + k]) { exact = false; firstDiff = `W[${k}]`; break; }
        if (c.N[r * 4 + k] !== c.N[r0 * 4 + k]) { exact = false; firstDiff = `N[${k}]`; break; }
        if (c.Prm[r * 4 + k] !== c.Prm[r0 * 4 + k]) { exact = false; firstDiff = `Prm[${k}]`; break; }
      }
      if (exact && c.Dist[r] !== c.Dist[r0]) { exact = false; firstDiff = "Dist"; }
      if (exact && c.TOI[r] !== c.TOI[r0]) { exact = false; firstDiff = "TOI"; }
    }
    out.push({ key, slots, exact, firstDiff });
  }
  return out;
}

export function multisetEqual(a: string[], b: string[]): { equal: boolean; onlyA: string[]; onlyB: string[] } {
  const ca = new Map<string, number>();
  const cb = new Map<string, number>();
  for (const k of a) ca.set(k, (ca.get(k) ?? 0) + 1);
  for (const k of b) cb.set(k, (cb.get(k) ?? 0) + 1);
  const onlyA: string[] = [];
  const onlyB: string[] = [];
  for (const [k, n] of ca) {
    const m = cb.get(k) ?? 0;
    for (let i = 0; i < n - m; i++) onlyA.push(k);
  }
  for (const [k, n] of cb) {
    const m = ca.get(k) ?? 0;
    for (let i = 0; i < n - m; i++) onlyB.push(k);
  }
  return { equal: onlyA.length === 0 && onlyB.length === 0, onlyA, onlyB };
}

// --- invariant checkers ---

export interface RecordCheck {
  checked: number;
  errors: string[];
}

/**
 * Validate every LIVE record (class B gate): kind domain, finiteness,
 * distance domain (barrier-inactive 1e30 sentinel allowed — it marks slots
 * the barrier itself retired), TOI domain. Stale slots beyond `live` are
 * NOT checked here (they are inert by the cap/count guards, verified
 * separately by the frames test).
 */
export function checkLiveRecords(c: ContactLive, dHat: number): RecordCheck {
  const errors: string[] = [];
  const push = (s: string): void => { if (errors.length < 8) errors.push(s); };
  for (let s = 0; s < c.live; s++) {
    const kind = Math.round(c.N[s * 4 + 3]);
    if (kind !== 0 && kind !== 1 && kind !== 2) { push(`slot ${s}: kind=${c.N[s * 4 + 3]}`); continue; }
    for (let k = 0; k < 4; k++) {
      if (!Number.isFinite(c.W[s * 4 + k])) push(`slot ${s}: W[${k}]=${c.W[s * 4 + k]}`);
      if (!Number.isFinite(c.N[s * 4 + k])) push(`slot ${s}: N[${k}]=${c.N[s * 4 + k]}`);
      if (!Number.isFinite(c.Prm[s * 4 + k])) push(`slot ${s}: Prm[${k}]=${c.Prm[s * 4 + k]}`);
    }
    const d = c.Dist[s];
    if (!Number.isFinite(d)) { push(`slot ${s}: dist=${d}`); continue; }
    if (d !== 1e30 && d >= dHat) push(`slot ${s}: dist ${d} >= dHat ${dHat}`);
    if (kind !== 2 && d !== 1e30 && d < 1e-12) push(`slot ${s}: kind ${kind} dist ${d} < 1e-12`);
    const toi = c.TOI[s];
    if (!Number.isFinite(toi)) push(`slot ${s}: toi=${toi}`);
    if (!Number.isFinite(c.Energy[s])) push(`slot ${s}: energy=${c.Energy[s]}`);
  }
  return { checked: c.live, errors };
}

/** Scan a float buffer for NaN/Inf (class B gate). */
export function findNonFinite(a: Float32Array, label: string, stride = 1): string[] {
  const bad: string[] = [];
  for (let i = 0; i < a.length; i += stride) {
    const v = a[i];
    if (!Number.isFinite(v)) {
      bad.push(`${label}[${i}]=${v}`);
      if (bad.length >= 8) break;
    }
  }
  return bad;
}

/** Pins exact: device position at pinned verts equals pinPos bitwise. */
export function checkPinsExact(
  position: Float32Array, pinMask: Uint32Array, pinPos: Float32Array, n: number,
): { pinned: number; worst: number } {
  let pinned = 0;
  let worst = 0;
  for (let v = 0; v < n; v++) {
    if (pinMask[v] === 0) continue;
    pinned++;
    for (let k = 0; k < 3; k++) {
      worst = Math.max(worst, Math.abs(position[v * 4 + k] - pinPos[v * 4 + k]));
    }
  }
  return { pinned, worst };
}

export function maxDiff(a: Float32Array, b: Float32Array): number {
  let d = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) d = Math.max(d, Math.abs(a[i] - b[i]));
  return d;
}
