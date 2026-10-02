// G2 compaction + scene parity: GPU contact sets vs CPU ContactSystem.
// Every scene compares ordering-independent contact keys, distances, TOIs,
// and classification on identical segments. Clear margins (>> FP32 ulp) so
// classification agrees exactly.
import { describe, it, expect } from "vitest";
import { ContactSystem } from "../../src/collision/contact-assembly.js";
import { DEFAULT_CONTACT_PARAMS } from "../../src/collision/types.js";
import { CpuBroadPhase } from "../../src/collision/broadphase.js";
import { GpuBroadPhase } from "../../src/backend/webgpu/gpu-broadphase.js";
import { preprocess } from "../../src/mesh/mesh.js";
import { sweptTriAabb, overlaps } from "../../src/collision/aabb.js";
import {
  GpuContactSystem, compareContactSets, compareContactMultisets,
  G2_DIST_ABS_TOL, G2_TOI_ABS_TOL, G2_BOUNDARY_BAND, CK_FLOOR,
} from "../../src/backend/webgpu/gpu-contact.js";
import { coulombForce } from "../../src/collision/friction.js";
import {
  closestPointVertexTriangle, closestPointEdgeEdge,
} from "../../src/collision/closest-point.js";
import { vtCCD } from "../../src/collision/ccd-vt.js";
import { eeCCD } from "../../src/collision/ccd-ee.js";
import { offsetMesh, preprocessMerged } from "../helpers.js";

const P = { ...DEFAULT_CONTACT_PARAMS };

function cpuActive(
  meshIndices: Uint32Array, x0: Float64Array, x1: Float64Array,
  opts: { floorY?: number | null; staticPos?: Float32Array | null; staticIdx?: Uint32Array | null; frictionMu?: number } = {},
): ContactSystem {
  const cs = new ContactSystem(
    { ...P, frictionMu: opts.frictionMu ?? P.frictionMu }, meshIndices,
  );
  if (opts.floorY !== undefined) cs.setFloor(opts.floorY);
  if (opts.staticPos && opts.staticIdx) cs.setStaticMesh(opts.staticPos, opts.staticIdx);
  cs.beginStep(x0);
  cs.updateActiveSet(x1);
  return cs;
}

async function gpuActive(
  meshIndices: Uint32Array, triCount: number,
  x0: Float64Array, x1: Float64Array,
  opts: { floorY?: number | null; staticPos?: Float32Array | null; staticIdx?: Uint32Array | null; contactCapacity?: number } = {},
): Promise<ReturnType<GpuContactSystem["build"]> extends Promise<infer T> ? T : never> {
  const gcs = new GpuContactSystem({
    indices: meshIndices, triCount,
    dHatM: P.dHatM, dMinM: P.dMinM, kappaJ: P.kappaJ,
    frictionMu: P.frictionMu, frictionEpsM: P.frictionEpsM,
    contactCapacity: opts.contactCapacity ?? 4096,
    floorY: opts.floorY ?? null,
    staticPos: opts.staticPos ?? null,
    staticIdx: opts.staticIdx ?? null,
  });
  gcs.beginStep(x0);
  const bp = await new CpuBroadPhase({ indices: meshIndices, triCount, pad: P.dHatM }).build(x0, x1);
  const staticPairs = bruteStaticPairs(meshIndices, triCount, x0, x1, opts.staticPos ?? null, opts.staticIdx ?? null, P.dHatM);
  return gcs.build(x1, bp, staticPairs);
}

/** Brute-force cloth-vs-static swept overlap (BVH-independent cross-check). */
function bruteStaticPairs(
  clothIdx: Uint32Array, triCount: number,
  x0: Float64Array, x1: Float64Array,
  staticPos: Float32Array | null, staticIdx: Uint32Array | null, pad: number,
): Array<[number, number]> {
  if (!staticPos || !staticIdx) return [];
  const out: Array<[number, number]> = [];
  const n = x0.length / 3;
  const at = (v: number): [number, number, number] =>
    v >= 0 ? [x0[v * 3], x0[v * 3 + 1], x0[v * 3 + 2]] : [0, 0, 0];
  void at; void n;
  for (let ct = 0; ct < triCount; ct++) {
    const box = sweptTriAabb(
      x0, x1, clothIdx[ct * 3], clothIdx[ct * 3 + 1], clothIdx[ct * 3 + 2], pad);
    const sTri = staticIdx.length / 3;
    for (let st = 0; st < sTri; st++) {
      const i0 = staticIdx[st * 3], i1 = staticIdx[st * 3 + 1], i2 = staticIdx[st * 3 + 2];
      const xs = [staticPos[i0 * 3], staticPos[i1 * 3], staticPos[i2 * 3]];
      const ys = [staticPos[i0 * 3 + 1], staticPos[i1 * 3 + 1], staticPos[i2 * 3 + 1]];
      const zs = [staticPos[i0 * 3 + 2], staticPos[i1 * 3 + 2], staticPos[i2 * 3 + 2]];
      const sb = {
        minX: Math.min(...xs) - pad, minY: Math.min(...ys) - pad, minZ: Math.min(...zs) - pad,
        maxX: Math.max(...xs) + pad, maxY: Math.max(...ys) + pad, maxZ: Math.max(...zs) + pad,
      };
      if (overlaps(box, sb)) out.push([ct, st]);
    }
  }
  return out;
}

function expectSetsMatch(
  cs: ContactSystem,
  g: { contacts: Array<{ key: string; kind: number; dist: number; toi: number; status: string }>; diagnostics: { contactOverflow: number; failures: number } },
  x0: Float64Array, x1: Float64Array,
  statics: { pos: Float32Array; idx: Uint32Array } | null = null,
  distTol = G2_DIST_ABS_TOL, toiTol = G2_TOI_ABS_TOL,
): void {
  // Boundary band (G2_BOUNDARY_BAND): FP32 legitimately flips classification
  // exactly at dHat; band members are excluded from the exact gates and
  // reported (never silently dropped from diagnostics).
  const inBand = (d: number): boolean =>
    Math.abs(d - P.dHatM) < G2_BOUNDARY_BAND || d < 1e-9;
  const cpuKeys = cs.active.map((c: { key: string }) => c.key);
  const gpuKeysAll = g.contacts.map((c) => c.key);
  const cmp = compareContactMultisets(
    cpuKeys.filter((k) => !bandKeysCPU(cs, k, x1, statics)),
    gpuKeysAll.filter((k) => {
      const r = g.contacts.find((c) => c.key === k);
      return r ? !inBand(r.dist) : true;
    }),
  );
  expect(cmp.missing).toEqual([]);
  expect(cmp.extra).toEqual([]);
  expect(cmp.match).toBe(true);
  expect(g.diagnostics.contactOverflow).toBe(0);
  expect(g.diagnostics.failures).toBe(0);
  // per-contact distance + TOI parity: recompute CPU-side at x1 / over x0->x1
  const posOf = (X: Float64Array, v: number): [number, number, number] => {
    if (v >= 0) return [X[v * 3], X[v * 3 + 1], X[v * 3 + 2]];
    const s = statics!.pos;
    const i = -v - 2;
    return [s[i * 3], s[i * 3 + 1], s[i * 3 + 2]];
  };
  const gByKey = new Map<string, Array<{ dist: number; toi: number }>>();
  for (const c of g.contacts) {
    if (!gByKey.has(c.key)) gByKey.set(c.key, []);
    gByKey.get(c.key)!.push(c);
  }
  type CpuC = {
    key: string; kind: number;
    p: number; a: number; b: number; c: number; d: number;
  };
  for (const c of cs.active as Array<CpuC>) {
    const at = (v: number): [number, number, number] => posOf(x1, v);
    let refD: number;
    let refT: number;
    if (c.kind === 2) continue; // floor dist checked explicitly per scene
    if (c.kind === 0) {
      const [px, py, pz] = at(c.p);
      const [ax, ay, az] = at(c.a);
      const [bx, by, bz] = at(c.b);
      const [cx, cy, cz] = at(c.c);
      refD = closestPointVertexTriangle(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz).dist;
      refT = vtCCD(x0, x1, c.p, c.a, c.b, c.c, P.dMinM);
    } else {
      const [ax, ay, az] = at(c.a);
      const [bx, by, bz] = at(c.b);
      const [cx, cy, cz] = at(c.c);
      const [dx, dy, dz] = at(c.d);
      refD = closestPointEdgeEdge(ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz).dist;
      refT = eeCCD(x0, x1, c.a, c.b, c.c, c.d, P.dMinM);
    }
    if (inBand(refD)) continue; // boundary: classification may legitimately flip
    const candidates = (gByKey.get(c.key) ?? []).filter((r) => !inBand(r.dist));
    expect(candidates.length).toBeGreaterThan(0);
    // distance agreement (nearest duplicate) + TOI agreement
    let bestD = Infinity;
    for (const r of candidates) bestD = Math.min(bestD, Math.abs(r.dist - refD));
    expect(bestD).toBeLessThan(distTol);
    if (refT === Infinity) {
      for (const r of candidates) expect(r.toi).toBe(Infinity);
    } else {
      let bestT = Infinity;
      for (const r of candidates) {
        if (r.toi === Infinity) continue;
        bestT = Math.min(bestT, Math.abs(r.toi - refT));
      }
      expect(bestT).toBeLessThan(toiTol);
    }
  }
}

/** CPU-side boundary-band membership for one active contact. */
function bandKeysCPU(
  cs: ContactSystem,
  key: string, x1: Float64Array,
  statics: { pos: Float32Array; idx: Uint32Array } | null,
): boolean {
  const c = (cs.active as Array<{
    key: string; kind: number;
    p: number; a: number; b: number; c: number; d: number;
  }>).find((e) => e.key === key);
  if (!c || c.kind === 2) return false;
  const posOf = (X: Float64Array, v: number): [number, number, number] => {
    if (v >= 0) return [X[v * 3], X[v * 3 + 1], X[v * 3 + 2]];
    const s = statics!.pos;
    const i = -v - 2;
    return [s[i * 3], s[i * 3 + 1], s[i * 3 + 2]];
  };
  let d: number;
  if (c.kind === 0) {
    const [px, py, pz] = posOf(x1, c.p);
    const [ax, ay, az] = posOf(x1, c.a);
    const [bx, by, bz] = posOf(x1, c.b);
    const [cx, cy, cz] = posOf(x1, c.c);
    d = closestPointVertexTriangle(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz).dist;
  } else {
    const [ax, ay, az] = posOf(x1, c.a);
    const [bx, by, bz] = posOf(x1, c.b);
    const [cx, cy, cz] = posOf(x1, c.c);
    const [dx, dy, dz] = posOf(x1, c.d);
    d = closestPointEdgeEdge(ax, ay, az, bx, by, bz, cx, cy, cz, dx, dy, dz).dist;
  }
  return Math.abs(d - P.dHatM) < G2_BOUNDARY_BAND || d < 1e-9;
}

describe("G2 contact compaction + scene parity", () => {
  it("19. empty contact set far apart (no overflow, no failures)", async () => {
    const A = offsetMesh(2, 2, 0.05, 0.05, -0.2, 0.1, 0);
    const B = offsetMesh(2, 2, 0.05, 0.05, 0.2, 0.1, 0);
    const mesh = preprocessMerged([A, B], 0.15);
    const x0 = Float64Array.from(mesh.positions);
    const cs = cpuActive(mesh.indices, x0, x0);
    const g = await gpuActive(mesh.indices, mesh.triCount, x0, x0);
    expect(cs.active.length).toBe(0);
    expect(g.contacts.length).toBe(0);
    expect(g.diagnostics.contactOverflow).toBe(0);
    expect(g.diagnostics.failures).toBe(0);
  });

  it("20. single floor contact compacts deterministically", async () => {
    const g0 = offsetMesh(1, 1, 0.05, 0.05, 0, 0.5, 0);
    const mesh = preprocessMerged([g0], 0.15);
    const x0 = Float64Array.from(mesh.positions);
    const x1 = Float64Array.from(x0);
    // drop one vertex to 1 mm above the floor; rest stay at 0.5 m
    x1[1] = 0.001;
    const cs = cpuActive(mesh.indices, x0, x1, { floorY: 0 });
    const g = await gpuActive(mesh.indices, mesh.triCount, x0, x1, { floorY: 0 });
    expect(cs.active.length).toBe(1);
    expect(g.contacts.length).toBe(1);
    expect(g.contacts[0].kind).toBe(CK_FLOOR);
    expect(g.contacts[0].key).toBe("floor:0");
    expect(Math.abs(g.contacts[0].dist - 0.001)).toBeLessThan(1e-6);
    const g2 = await gpuActive(mesh.indices, mesh.triCount, x0, x1, { floorY: 0 });
    expect(g2.contacts.map((c) => c.key)).toEqual(g.contacts.map((c) => c.key));
  });

  it("21. multiple contacts form a canonical sorted set", async () => {
    const A = offsetMesh(2, 2, 0.05, 0.05, -0.03, 0.02, 0);
    const B = offsetMesh(2, 2, 0.05, 0.05, 0.03, 0.02, 0);
    const mesh = preprocessMerged([A, B], 0.15);
    const x0 = Float64Array.from(mesh.positions);
    const x1 = Float64Array.from(x0);
    const nA = A.positions.length / 3;
    // shallow 1 mm approach: all contacts comfortably inside dHat, none near
    // the classification boundary and none near zero distance (deep
    // interpenetration would amplify FP32 ulps via huge barrier multipliers).
    for (let i = 0; i < mesh.count; i++) x1[i * 3] += i < nA ? 0.0045 : -0.0045;
    const cs = cpuActive(mesh.indices, x0, x1);
    const g = await gpuActive(mesh.indices, mesh.triCount, x0, x1);
    expect(cs.active.length).toBeGreaterThan(1);
    expectSetsMatch(cs, g, x0, x1);
    const keys = g.contacts.map((c) => c.key);
    expect([...keys].sort()).toEqual(keys); // canonical order
  });

  it("22. overflow is explicit (tiny capacity flags, keeps scanned)", async () => {
    const A = offsetMesh(2, 2, 0.05, 0.05, -0.03, 0.02, 0);
    const B = offsetMesh(2, 2, 0.05, 0.05, 0.03, 0.02, 0);
    const mesh = preprocessMerged([A, B], 0.15);
    const x0 = Float64Array.from(mesh.positions);
    const x1 = Float64Array.from(x0);
    const nA = A.positions.length / 3;
    for (let i = 0; i < mesh.count; i++) x1[i * 3] += i < nA ? 0.0045 : -0.0045;
    const full = await gpuActive(mesh.indices, mesh.triCount, x0, x1);
    expect(full.contacts.length).toBeGreaterThan(2);
    const tiny = await gpuActive(mesh.indices, mesh.triCount, x0, x1, { contactCapacity: 1 });
    expect(tiny.diagnostics.contactOverflow).toBe(1);
    expect(tiny.diagnostics.scannedCount).toBe(full.diagnostics.scannedCount);
    expect(tiny.contacts.length).toBe(1);
  });

  it("23. historical 4-triangle all-overlap: all 6 pairs incl. (1,2)", async () => {
    // 4 DISJOINT triangles (no shared verts -> zero exclusions), layered
    // 0.5 mm apart in y so every swept pair overlaps AND every primitive is
    // genuinely within the barrier zone.
    const nv = 12;
    const positions = new Float32Array(nv * 3);
    const uv = new Float32Array(nv * 2);
    const idx = new Uint32Array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    for (let t = 0; t < 4; t++) {
      const ox = t * 0.001, oy = t * 0.0005;
      const v = [[ox, oy, 0], [ox + 0.003, oy, 0], [ox, oy, 0.003]];
      for (let k = 0; k < 3; k++) {
        positions.set(v[k] as [number, number, number], (t * 3 + k) * 3);
        uv[(t * 3 + k) * 2] = v[k][0];
        uv[(t * 3 + k) * 2 + 1] = v[k][2];
      }
    }
    const mesh = preprocess(positions, uv, idx, 0.15);
    const x0 = Float64Array.from(mesh.positions);
    const x1 = Float64Array.from(x0);
    // G1 gate: all 6 unordered pairs, specifically (1,2)
    const cpu = await new CpuBroadPhase({ indices: idx, triCount: 4, pad: P.dHatM }).build(x0, x1);
    const gpu = await new GpuBroadPhase({ indices: idx, triCount: 4, pad: P.dHatM }).build(x0, x1);
    const key01 = (ps: Array<{ a: number; b: number }>): string[] =>
      ps.map((p) => `${p.a}_${p.b}`).sort();
    expect(key01(cpu.pairs)).toEqual(["0_1", "0_2", "0_3", "1_2", "1_3", "2_3"]);
    expect(key01(gpu.pairs)).toEqual(key01(cpu.pairs));
    // G2 gate: CPU active set contains contacts spanning tris 1 and 2
    // (tri 1 = verts 3,4,5; tri 2 = verts 6,7,8) — impossible under the old
    // pair-dropping BVH, which never even generated candidate (1,2).
    const cs = cpuActive(idx, x0, x1);
    const g = await gpuActive(idx, 4, x0, x1);
    const span12 = cs.active.filter((c: { p: number; a: number; b: number; c: number; d: number }) => {
      const ids = [c.p, c.a, c.b, c.c, c.d].filter((v) => v >= 0);
      const inT1 = ids.some((v) => v >= 3 && v <= 5);
      const inT2 = ids.some((v) => v >= 6 && v <= 8);
      return inT1 && inT2;
    });
    expect(span12.length).toBeGreaterThan(0);
    expectSetsMatch(cs, g, x0, x1);
  });

  it("24. floor scene parity (patch hovering in the barrier zone)", async () => {
    const g0 = offsetMesh(2, 2, 0.05, 0.05, 0, 0.001, 0);
    const mesh = preprocessMerged([g0], 0.15);
    const x0 = Float64Array.from(mesh.positions);
    const x1 = Float64Array.from(x0);
    for (let i = 0; i < mesh.count; i++) x1[i * 3 + 1] -= 0.0005; // settle to 0.5 mm
    const cs = cpuActive(mesh.indices, x0, x1, { floorY: 0 });
    const g = await gpuActive(mesh.indices, mesh.triCount, x0, x1, { floorY: 0 });
    expect(cs.active.length).toBeGreaterThan(0);
    expectSetsMatch(cs, g, x0, x1);
    // distance agreement on floor contacts
    for (const r of g.contacts) {
      expect(Math.abs(r.dist - 0.0005)).toBeLessThan(1e-6);
    }
    // penetrating trial is invalid on both (floor CCD branch)
    const xBad = Float64Array.from(x1);
    for (let i = 0; i < mesh.count; i++) xBad[i * 3 + 1] -= 0.002;
    cs.beginStep(x0);
    const cpuV = cs.checkTrial(xBad);
    const gcs = new GpuContactSystem({
      indices: mesh.indices, triCount: mesh.triCount,
      dHatM: P.dHatM, dMinM: P.dMinM, kappaJ: P.kappaJ,
      frictionMu: P.frictionMu, frictionEpsM: P.frictionEpsM, floorY: 0,
    });
    gcs.beginStep(x0);
    const bpBad = await new CpuBroadPhase({
      indices: mesh.indices, triCount: mesh.triCount, pad: P.dMinM,
    }).build(x0, xBad);
    const gpuV = gcs.checkTrialValidity(xBad, bpBad);
    expect(cpuV.valid).toBe(false);
    expect(gpuV.valid).toBe(false);
  });

  it("25. static plate parity (cloth patch vs static triangle)", async () => {
    const A = offsetMesh(2, 2, 0.06, 0.06, 0, 0.004, 0);
    const mesh = preprocessMerged([A], 0.15);
    const plate = offsetMesh(2, 2, 0.08, 0.08, -0.01, 0, -0.01);
    const x0 = Float64Array.from(mesh.positions);
    const x1 = Float64Array.from(x0);
    for (let i = 0; i < mesh.count; i++) x1[i * 3 + 1] -= 0.003; // approach to 1 mm
    const cs = cpuActive(mesh.indices, x0, x1, { staticPos: plate.positions, staticIdx: plate.indices });
    const g = await gpuActive(mesh.indices, mesh.triCount, x0, x1, {
      staticPos: plate.positions, staticIdx: plate.indices,
    });
    expect(cs.active.length).toBeGreaterThan(0);
    expectSetsMatch(cs, g, x0, x1, { pos: plate.positions, idx: plate.indices });
  });

  it("26. head-on patches: set + classification parity", async () => {
    const A = offsetMesh(4, 4, 0.08, 0.08, -0.07, 0.02, 0);
    const B = offsetMesh(4, 4, 0.08, 0.08, 0.07, 0.02, 0);
    const mesh = preprocessMerged([A, B], 0.15);
    const x0 = Float64Array.from(mesh.positions);
    const x1 = Float64Array.from(x0);
    const nA = A.positions.length / 3;
    // shallow 1 mm approach (60 mm initial gap closed to 1 mm)
    for (let i = 0; i < mesh.count; i++) x1[i * 3] += i < nA ? 0.0295 : -0.0295;
    const cs = cpuActive(mesh.indices, x0, x1);
    const g = await gpuActive(mesh.indices, mesh.triCount, x0, x1);
    expect(cs.active.length).toBeGreaterThan(0);
    expectSetsMatch(cs, g, x0, x1);
    // classification: every GPU record carries a non-failure CCD status
    for (const r of g.contacts) {
      expect(["resting", "impact", "safe"]).toContain(r.status);
    }
  });

  it("27. fold pose parity (page-fold her segment)", async () => {
    const g0 = offsetMesh(8, 2, 0.2, 0.03, 0, 0.02, 0);
    const mesh = preprocessMerged([g0], 0.15);
    const x0 = Float64Array.from(mesh.positions);
    const x1 = Float64Array.from(x0);
    for (let i = 0; i < mesh.count; i++) {
      const x = x0[i * 3];
      if (x > 0.1) {
        x1[i * 3] = 0.1 - 2 * (x - 0.1);
        x1[i * 3 + 1] += 0.0015;
      }
    }
    const cs = cpuActive(mesh.indices, x0, x1);
    const g = await gpuActive(mesh.indices, mesh.triCount, x0, x1);
    expect(cs.active.length).toBeGreaterThan(0);
    expectSetsMatch(cs, g, x0, x1);
  });

  it("28. friction invariants hold; mu = 0 kills friction", async () => {
    const A = offsetMesh(2, 2, 0.05, 0.05, -0.03, 0.02, 0.001);
    const B = offsetMesh(2, 2, 0.05, 0.05, 0.03, 0.02, -0.001);
    const mesh = preprocessMerged([A, B], 0.15);
    const x0 = Float64Array.from(mesh.positions);
    const x1 = Float64Array.from(x0);
    const nA = A.positions.length / 3;
    // shallow 1 mm approach + lateral slide (tangential slip exercises Coulomb;
    // deep interpenetration would inflate lambdaN and amplify FP32 ulps)
    for (let i = 0; i < mesh.count; i++) {
      x1[i * 3] += i < nA ? 0.0045 : -0.0045;
      x1[i * 3 + 2] += i < nA ? 0.002 : -0.002;
    }
    const mkSystem = (mu: number): GpuContactSystem => {
      const gcs = new GpuContactSystem({
        indices: mesh.indices, triCount: mesh.triCount,
        dHatM: P.dHatM, dMinM: P.dMinM, kappaJ: P.kappaJ,
        frictionMu: mu, frictionEpsM: P.frictionEpsM,
      });
      gcs.beginStep(x0);
      return gcs;
    };
    const gcs = mkSystem(P.frictionMu);
    const bp = await new CpuBroadPhase({ indices: mesh.indices, triCount: mesh.triCount, pad: P.dHatM }).build(x0, x1);
    const set = await gcs.build(x1, bp);
    expect(set.contacts.length).toBeGreaterThan(0);
    gcs.commit(x1, set);
    const { grad } = gcs.frictionForces(x1, set);
    for (const v of grad) expect(Number.isFinite(v)).toBe(true);
    // independent per-contact cone check with committed lagged state:
    // dot(ft, n) ~= 0 and |ft| <= mu * |lambdaN|
    for (const r of set.contacts.slice(0, 8)) {
      const lag = gcs.lagged.get(r.key)!;
      expect(lag).toBeDefined();
      const slip: [number, number, number] = [0.004, 0, 0.004];
      const [fx, fy, fz] = coulombForce(slip[0], slip[1], slip[2], lag.nx, lag.ny, lag.nz, lag.lambdaN, P.frictionMu, P.frictionEpsM);
      const tn = Math.abs(fx * lag.nx + fy * lag.ny + fz * lag.nz);
      const mag = Math.hypot(fx, fy, fz);
      expect(tn).toBeLessThan(1e-9 * (1 + mag) + 1e-9);
      expect(mag).toBeLessThanOrEqual(P.frictionMu * Math.max(0, lag.lambdaN) + 1e-9);
    }
    // mu = 0 -> identically zero friction residual
    const zero = mkSystem(0);
    const set0 = await zero.build(x1, bp);
    zero.commit(x1, set0);
    const f0 = zero.frictionForces(x1, set0);
    for (const v of f0.grad) expect(v).toBe(0);
  });

  it("29. Armijo-alpha recompute: fresh CCD per trial, validity agrees", async () => {
    const A = offsetMesh(3, 3, 0.06, 0.06, -0.05, 0.02, 0);
    const B = offsetMesh(3, 3, 0.06, 0.06, 0.05, 0.02, 0);
    const mesh = preprocessMerged([A, B], 0.15);
    const x0 = Float64Array.from(mesh.positions);
    const nA = A.positions.length / 3;
    // Newton direction: alpha = 1 closes the 40 mm gap to 1 mm (contacts);
    // smaller alphas stay clear (empty but validity must still agree).
    const dx = new Float64Array(x0.length);
    for (let i = 0; i < mesh.count; i++) dx[i * 3] = i < nA ? 0.0195 : -0.0195;
    const trial = (alpha: number): Float64Array => {
      const x = Float64Array.from(x0);
      for (let i = 0; i < x.length; i++) x[i] += alpha * dx[i];
      return x;
    };
    for (const alpha of [1, 0.5, 0.25]) {
      const xT = trial(alpha);
      const cs = cpuActive(mesh.indices, x0, x0); // beginStep binds xStep
      cs.beginStep(x0);
      const cpuV = cs.checkTrial(xT);
      // GPU: fresh pairs at dMin pad + fresh CCD for THIS alpha
      const bpMin = await new CpuBroadPhase({
        indices: mesh.indices, triCount: mesh.triCount, pad: P.dMinM,
      }).build(x0, xT);
      const gcs = new GpuContactSystem({
        indices: mesh.indices, triCount: mesh.triCount,
        dHatM: P.dHatM, dMinM: P.dMinM, kappaJ: P.kappaJ,
        frictionMu: P.frictionMu, frictionEpsM: P.frictionEpsM,
      });
      gcs.beginStep(x0);
      const gpuV = gcs.checkTrialValidity(xT, bpMin);
      expect(gpuV.valid).toBe(cpuV.valid);
      if (cpuV.valid) {
        // contact sets agree at every accepted trial, not just alpha = 1
        const bpHat = await new CpuBroadPhase({
          indices: mesh.indices, triCount: mesh.triCount, pad: P.dHatM,
        }).build(x0, xT);
        const set = await gcs.build(xT, bpHat);
        cs.updateActiveSet(xT);
        expectSetsMatch(cs, set, x0, xT);
      }
    }
  });

  it("30. barrier energy + gradient parity on an active set", async () => {
    const A = offsetMesh(2, 2, 0.05, 0.05, -0.03, 0.02, 0);
    const B = offsetMesh(2, 2, 0.05, 0.05, 0.03, 0.02, 0);
    const mesh = preprocessMerged([A, B], 0.15);
    const x0 = Float64Array.from(mesh.positions);
    const x1 = Float64Array.from(x0);
    const nA = A.positions.length / 3;
    for (let i = 0; i < mesh.count; i++) x1[i * 3] += i < nA ? 0.0045 : -0.0045;
    // mu = 0 on BOTH sides isolates the barrier (CPU energyGrad adds lagged
    // friction to the residual otherwise).
    const cs = cpuActive(mesh.indices, x0, x1, { frictionMu: 0 });
    cs.commit(x1);
    const ref = cs.energyGrad(x1);
    const gcs = new GpuContactSystem({
      indices: mesh.indices, triCount: mesh.triCount,
      dHatM: P.dHatM, dMinM: P.dMinM, kappaJ: P.kappaJ,
      frictionMu: 0, frictionEpsM: P.frictionEpsM, // mu=0 isolates barrier
    });
    gcs.beginStep(x0);
    const bp = await new CpuBroadPhase({ indices: mesh.indices, triCount: mesh.triCount, pad: P.dHatM }).build(x0, x1);
    const set = await gcs.build(x1, bp);
    gcs.commit(x1, set);
    const got = gcs.barrierEnergyGrad(x1, set);
    // barrier-only comparison (mu = 0 on both sides)
    const cpuKeys = cs.active.map((c: { key: string }) => c.key);
    expect(compareContactSets(cpuKeys, set).match).toBe(true);
    const relE = Math.abs(got.energy - ref.energy) / Math.max(Math.abs(ref.energy), 1e-12);
    expect(relE).toBeLessThan(1e-3);
    let maxG = 0;
    for (let i = 0; i < ref.grad.length; i++) {
      maxG = Math.max(maxG, Math.abs(got.grad[i] - ref.grad[i]));
    }
    const scale = Math.max(1e-9, ...Array.from(ref.grad).map(Math.abs));
    expect(maxG / scale).toBeLessThan(1e-4);
  });
});
