// G1 broad-phase parity: CPU BVH vs GPU LBVH candidate sets (10 tests).
// The G1 -> G2 handoff gate is ordering-independent SET agreement on identical
// scenes. All scenes use clear overlap/separation margins (>> FP32 one-ulp)
// so the inclusive overlap predicate decides identically on both backends.
// No solver stepping: scenes are hand-posed, so the suite runs in milliseconds.
import { describe, it, expect } from "vitest";
import { buildGrid, preprocess } from "../../src/mesh/mesh.js";
import {
  CpuBroadPhase, compareCandidateSets,
} from "../../src/collision/broadphase.js";
import { GpuBroadPhase, sweptAabbFP32 } from "../../src/backend/webgpu/gpu-broadphase.js";
import { offsetMesh, preprocessMerged } from "../helpers.js";

const PAD = 0.002; // barrier-zone pad, same as DEFAULT_CONTACT_PARAMS.dHatM

function gridScene(nx: number, ny: number, w: number, h: number, ox = 0, oy = 0, oz = 0) {
  const g = offsetMesh(nx, ny, w, h, ox, oy, oz);
  const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
  const x0 = Float64Array.from(mesh.positions);
  return { mesh, x0 };
}

/** Posed trial state: x1 = x0 + per-vertex displacement fn. */
function pose(x0: Float64Array, fn: (x: number, y: number, z: number, id: number) => [number, number, number]): Float64Array {
  const x1 = Float64Array.from(x0);
  const n = x0.length / 3;
  for (let i = 0; i < n; i++) {
    const [dx, dy, dz] = fn(x0[i * 3], x0[i * 3 + 1], x0[i * 3 + 2], i);
    x1[i * 3] += dx; x1[i * 3 + 1] += dy; x1[i * 3 + 2] += dz;
  }
  return x1;
}

describe("G1 broad-phase parity", () => {
  it("1. swept AABB CPU/GPU parity (all tris, deformed segment)", async () => {
    const { mesh, x0 } = gridScene(4, 2, 0.2, 0.1);
    const x1 = pose(x0, (x) => [0.004 * Math.sin(x * 40), -0.003, 0.002]);
    const cpu = new CpuBroadPhase({ indices: mesh.indices, triCount: mesh.triCount, pad: PAD });
    let maxErr = 0;
    for (let t = 0; t < mesh.triCount; t++) {
      const a = cpu.sweptAabb(x0, x1, t);
      const i0 = mesh.indices[t * 3], i1 = mesh.indices[t * 3 + 1], i2 = mesh.indices[t * 3 + 2];
      const b = sweptAabbFP32(x0, x1, i0, i1, i2, PAD);
      const vals: Array<[number, number]> = [
        [a.minX, b.min[0]], [a.minY, b.min[1]], [a.minZ, b.min[2]],
        [a.maxX, b.max[0]], [a.maxY, b.max[1]], [a.maxZ, b.max[2]],
      ];
      for (const [u, v] of vals) maxErr = Math.max(maxErr, Math.abs(u - v));
    }
    expect(maxErr).toBeLessThan(1e-6); // FP32 one-ulp scale at 0.2 m extents
  });

  it("2. stationary AABB parity (x0 == x1: degenerate segment)", async () => {
    const { mesh, x0 } = gridScene(3, 3, 0.1, 0.1);
    const cpu = new CpuBroadPhase({ indices: mesh.indices, triCount: mesh.triCount, pad: PAD });
    const gpu = new GpuBroadPhase({ indices: mesh.indices, triCount: mesh.triCount, pad: PAD });
    const c = await cpu.build(x0, x0);
    const g = await gpu.build(x0, x0);
    const cmp = compareCandidateSets(c, g);
    expect(cmp.match).toBe(true);
    expect(g.diagnostics.candidateOverflow).toBe(0);
  });

  it("3. swept AABB parity (uniform translation: overlap preserved along motion)", async () => {
    // Two stacked patches 1 mm apart; both translate +x together 5 mm.
    // Swept boxes must still overlap (relative motion zero but pads overlap).
    const A = offsetMesh(2, 2, 0.05, 0.05, 0, 0, 0);
    const B = offsetMesh(2, 2, 0.05, 0.05, 0, 0.001, 0);
    const mesh = preprocessMerged([A, B], 0.15);
    const x0 = Float64Array.from(mesh.positions);
    const x1 = pose(x0, () => [0.005, 0, 0]);
    const cpu = new CpuBroadPhase({ indices: mesh.indices, triCount: mesh.triCount, pad: PAD });
    const gpu = new GpuBroadPhase({ indices: mesh.indices, triCount: mesh.triCount, pad: PAD });
    const c = await cpu.build(x0, x1);
    const g = await gpu.build(x0, x1);
    expect(c.pairs.length).toBeGreaterThan(0);
    const cmp = compareCandidateSets(c, g);
    expect(cmp.match).toBe(true);
  });

  it("4. adjacency exclusion parity (shared-vertex pairs never emitted)", async () => {
    const { mesh, x0 } = gridScene(3, 3, 0.1, 0.1);
    const x1 = pose(x0, () => [0, -0.001, 0]);
    const cpu = new CpuBroadPhase({ indices: mesh.indices, triCount: mesh.triCount, pad: PAD });
    const gpu = new GpuBroadPhase({ indices: mesh.indices, triCount: mesh.triCount, pad: PAD });
    // Spot-check the rule itself on both backends (edge-adjacent tris in the grid).
    expect(cpu.isExcluded(0, 1)).toBe(true);
    expect(gpu.isExcluded(0, 1)).toBe(true);
    const c = await cpu.build(x0, x1);
    const g = await gpu.build(x0, x1);
    // NOTE: a single connected patch still yields pairs — tris that touch at
    // the swept-box level but share no vertex are legitimately conservative
    // candidates (narrow phase filters them). The gate is SET parity, plus an
    // explicit no-shared-vertex audit on both outputs.
    const cmp = compareCandidateSets(c, g);
    expect(cmp.match).toBe(true);
    const sharedVert = (tA: number, tB: number): boolean => {
      const A = [mesh.indices[tA * 3], mesh.indices[tA * 3 + 1], mesh.indices[tA * 3 + 2]];
      const B = [mesh.indices[tB * 3], mesh.indices[tB * 3 + 1], mesh.indices[tB * 3 + 2]];
      return A.some((v) => B.includes(v));
    };
    for (const p of g.pairs) expect(sharedVert(p.a, p.b)).toBe(false);
  });

  it("5. non-adjacent pair detection (close patches found)", async () => {
    const A = offsetMesh(2, 2, 0.05, 0.05, -0.03, 0.02, 0);
    const B = offsetMesh(2, 2, 0.05, 0.05, 0.03, 0.02, 0);
    const mesh = preprocessMerged([A, B], 0.15);
    const x0 = Float64Array.from(mesh.positions);
    // Drive together until 1 mm apart (inside the 2 mm pad).
    const nA = A.positions.length / 3;
    const x1 = pose(x0, (x, y, z, id) => (id < nA ? [0.02, 0, 0] : [-0.02, 0, 0]));
    const cpu = new CpuBroadPhase({ indices: mesh.indices, triCount: mesh.triCount, pad: PAD });
    const gpu = new GpuBroadPhase({ indices: mesh.indices, triCount: mesh.triCount, pad: PAD });
    const c = await cpu.build(x0, x1);
    const g = await gpu.build(x0, x1);
    expect(c.pairs.length).toBeGreaterThan(0);
    const cmp = compareCandidateSets(c, g);
    expect(cmp.missing).toEqual([]);
    expect(cmp.extra).toEqual([]);
    expect(cmp.match).toBe(true);
  });

  it("6. no false cross-patch pairs in a separated scene", async () => {
    const A = offsetMesh(2, 2, 0.05, 0.05, -0.2, 0.1, 0);
    const B = offsetMesh(2, 2, 0.05, 0.05, 0.2, 0.1, 0);
    const mesh = preprocessMerged([A, B], 0.15);
    const x0 = Float64Array.from(mesh.positions);
    const x1 = pose(x0, () => [0, -0.001, 0]);
    const cpu = new CpuBroadPhase({ indices: mesh.indices, triCount: mesh.triCount, pad: PAD });
    const gpu = new GpuBroadPhase({ indices: mesh.indices, triCount: mesh.triCount, pad: PAD });
    const c = await cpu.build(x0, x1);
    const g = await gpu.build(x0, x1);
    // Within-patch box-touching pairs are legitimate (conservative); what must
    // be absent is any pair SPANNING the 0.35 m gap between the patches.
    const triPatch = mesh.triCount / 2;
    const crosses = (ps: Array<{ a: number; b: number }>): number =>
      ps.filter((p) => (p.a < triPatch) !== (p.b < triPatch)).length;
    expect(crosses(c.pairs)).toBe(0);
    expect(crosses(g.pairs)).toBe(0);
    const cmp = compareCandidateSets(c, g);
    expect(cmp.match).toBe(true);
    expect(g.diagnostics.candidateOverflow).toBe(0);
  });

  it("7. folded cloth candidate generation (page-fold pose)", async () => {
    // Right half folded back over the left half: layers 1.5 mm apart.
    const g = offsetMesh(8, 2, 0.2, 0.03, 0, 0.02, 0);
    const mesh = preprocessMerged([g], 0.15);
    const x0 = Float64Array.from(mesh.positions);
    const x1 = pose(x0, (x, y, z) => {
      if (x > 0.1) return [0.1 - 2 * (x - 0.1), 0.0015, 0]; // mirror right half over
      return [0, 0, 0];
    });
    const cpu = new CpuBroadPhase({ indices: mesh.indices, triCount: mesh.triCount, pad: PAD });
    const gpu = new GpuBroadPhase({ indices: mesh.indices, triCount: mesh.triCount, pad: PAD });
    const c = await cpu.build(x0, x1);
    const gres = await gpu.build(x0, x1);
    expect(c.pairs.length).toBeGreaterThan(0); // fold genuinely creates candidates
    const cmp = compareCandidateSets(c, gres);
    expect(cmp.match).toBe(true);
  });

  it("8. head-on patches candidate generation (pre-impact swept segment)", async () => {
    const A = offsetMesh(4, 4, 0.08, 0.08, -0.07, 0.02, 0);
    const B = offsetMesh(4, 4, 0.08, 0.08, 0.07, 0.02, 0);
    const mesh = preprocessMerged([A, B], 0.15);
    const x0 = Float64Array.from(mesh.positions);
    const nA = A.positions.length / 3;
    // One 60 Hz step of approach at 0.3 m/s: 5 mm each — swept into overlap.
    const x1 = pose(x0, (x, y, z, id) => (id < nA ? [0.005, 0, 0] : [-0.005, 0, 0]));
    const cpu = new CpuBroadPhase({ indices: mesh.indices, triCount: mesh.triCount, pad: PAD });
    const gpu = new GpuBroadPhase({ indices: mesh.indices, triCount: mesh.triCount, pad: PAD });
    const c = await cpu.build(x0, x1);
    const g = await gpu.build(x0, x1);
    expect(c.pairs.length).toBeGreaterThan(0);
    const cmp = compareCandidateSets(c, g);
    expect(cmp.match).toBe(true);
    expect(g.diagnostics.detail?.mortonSorted).toBe(true);
  });

  it("9. deterministic candidate count (same build twice)", async () => {
    const A = offsetMesh(3, 3, 0.06, 0.06, -0.04, 0.02, 0.01);
    const B = offsetMesh(3, 3, 0.06, 0.06, 0.04, 0.02, -0.01);
    const mesh = preprocessMerged([A, B], 0.15);
    const x0 = Float64Array.from(mesh.positions);
    const nA = A.positions.length / 3;
    const x1 = pose(x0, (x, y, z, id) => (id < nA ? [0.015, 0, 0.004] : [-0.015, 0, -0.004]));
    const cfg = { indices: mesh.indices, triCount: mesh.triCount, pad: PAD };
    const g1 = await new GpuBroadPhase(cfg).build(x0, x1);
    const g2 = await new GpuBroadPhase(cfg).build(x0, x1);
    expect(g1.pairs).toEqual(g2.pairs);
    expect(g1.diagnostics.scannedCount).toBe(g2.diagnostics.scannedCount);
  });

  it("10. overflow detection (tiny buffer flags instead of truncating silently)", async () => {
    const A = offsetMesh(4, 4, 0.08, 0.08, -0.07, 0.02, 0);
    const B = offsetMesh(4, 4, 0.08, 0.08, 0.07, 0.02, 0);
    const mesh = preprocessMerged([A, B], 0.15);
    const x0 = Float64Array.from(mesh.positions);
    const nA = A.positions.length / 3;
    const x1 = pose(x0, (x, y, z, id) => (id < nA ? [0.06, 0, 0] : [-0.06, 0, 0]));
    const full = await new GpuBroadPhase({
      indices: mesh.indices, triCount: mesh.triCount, pad: PAD,
    }).build(x0, x1);
    expect(full.diagnostics.scannedCount).toBeGreaterThan(2);
    const tiny = await new GpuBroadPhase({
      indices: mesh.indices, triCount: mesh.triCount, pad: PAD, pairCapacity: 2,
    }).build(x0, x1);
    expect(tiny.diagnostics.candidateOverflow).toBe(1);
    expect(tiny.diagnostics.scannedCount).toBe(full.diagnostics.scannedCount);
    expect(tiny.pairs.length).toBe(2); // truncated BUT flagged — caller must resize/fallback
  });
});
