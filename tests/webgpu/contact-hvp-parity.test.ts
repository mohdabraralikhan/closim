// G2.5 HVP + Jacobi-diagonal mirror parity (headless; validates the FP32
// ports added for G3 device consumption against CPU applyHvp/addDiagEstimate).
import { describe, it, expect } from "vitest";
import { ContactSystem } from "../../src/collision/contact-assembly.js";
import { DEFAULT_CONTACT_PARAMS } from "../../src/collision/types.js";
import { CpuBroadPhase } from "../../src/collision/broadphase.js";
import {
  GpuContactSystem, compareContactMultisets,
} from "../../src/backend/webgpu/gpu-contact.js";
import { offsetMesh, preprocessMerged } from "../helpers.js";

const P = { ...DEFAULT_CONTACT_PARAMS };

describe("G2 contact HVP + diagonal parity", () => {
  it("barrier HVP matches CPU applyHvp (directional agreement)", async () => {
    const A = offsetMesh(2, 2, 0.05, 0.05, -0.03, 0.02, 0);
    const B = offsetMesh(2, 2, 0.05, 0.05, 0.03, 0.02, 0);
    const mesh = preprocessMerged([A, B], 0.15);
    const x0 = Float64Array.from(mesh.positions);
    const x1 = Float64Array.from(x0);
    const nA = A.positions.length / 3;
    for (let i = 0; i < mesh.count; i++) x1[i * 3] += i < nA ? 0.0045 : -0.0045;
    // CPU reference
    const cs = new ContactSystem({ ...P }, mesh.indices);
    cs.beginStep(x0);
    cs.updateActiveSet(x1);
    expect(cs.active.length).toBeGreaterThan(0);
    // mirror
    const gcs = new GpuContactSystem({
      indices: mesh.indices, triCount: mesh.triCount,
      dHatM: P.dHatM, dMinM: P.dMinM, kappaJ: P.kappaJ,
      frictionMu: P.frictionMu, frictionEpsM: P.frictionEpsM,
    });
    gcs.beginStep(x0);
    const bp = await new CpuBroadPhase({ indices: mesh.indices, triCount: mesh.triCount, pad: P.dHatM }).build(x0, x1);
    const set = await gcs.build(x1, bp);
    const cpuKeys = cs.active.map((c: { key: string }) => c.key);
    expect(compareContactMultisets(cpuKeys, set.contacts.map((r) => r.key)).match).toBe(true);
    // direction: deterministic sine field
    const v = new Float64Array(x1.length);
    for (let i = 0; i < v.length; i++) v[i] = Math.sin(i * 12.9898) * 0.01;
    const ref = new Float64Array(x1.length);
    cs.applyHvp(x1, v, ref);
    const out = new Float32Array(x1.length);
    gcs.applyHvp(x1, v, set, out);
    let maxAbs = 0;
    const scale = Math.max(1e-12, ...Array.from(ref).map(Math.abs));
    for (let i = 0; i < out.length; i++) {
      maxAbs = Math.max(maxAbs, Math.abs(out[i] - ref[i]));
    }
    expect(maxAbs / scale).toBeLessThan(1e-3);
  });

  it("Jacobi diagonal matches CPU addDiagEstimate", async () => {
    const A = offsetMesh(2, 2, 0.05, 0.05, -0.03, 0.02, 0);
    const B = offsetMesh(2, 2, 0.05, 0.05, 0.03, 0.02, 0);
    const mesh = preprocessMerged([A, B], 0.15);
    const x0 = Float64Array.from(mesh.positions);
    const x1 = Float64Array.from(x0);
    const nA = A.positions.length / 3;
    for (let i = 0; i < mesh.count; i++) x1[i * 3] += i < nA ? 0.0045 : -0.0045;
    const cs = new ContactSystem({ ...P }, mesh.indices);
    cs.beginStep(x0);
    cs.updateActiveSet(x1);
    const gcs = new GpuContactSystem({
      indices: mesh.indices, triCount: mesh.triCount,
      dHatM: P.dHatM, dMinM: P.dMinM, kappaJ: P.kappaJ,
      frictionMu: P.frictionMu, frictionEpsM: P.frictionEpsM,
    });
    gcs.beginStep(x0);
    const bp = await new CpuBroadPhase({ indices: mesh.indices, triCount: mesh.triCount, pad: P.dHatM }).build(x0, x1);
    const set = await gcs.build(x1, bp);
    const ref = new Float64Array(x1.length);
    const masses = new Float32Array(mesh.count).fill(1e-4);
    cs.addDiagEstimate(ref, x1, masses as unknown as Float32Array, 3600, 1);
    const got = new Float32Array(x1.length);
    gcs.addDiagEstimate(x1, set, got);
    let maxAbs = 0;
    const scale = Math.max(1e-12, ...Array.from(ref).map(Math.abs));
    for (let i = 0; i < got.length; i++) {
      maxAbs = Math.max(maxAbs, Math.abs(got[i] - ref[i]));
    }
    expect(maxAbs / scale).toBeLessThan(1e-3);
  });

  it("committed lagged state matches CPU lagged map", async () => {
    const A = offsetMesh(2, 2, 0.05, 0.05, -0.03, 0.02, 0);
    const B = offsetMesh(2, 2, 0.05, 0.05, 0.03, 0.02, 0);
    const mesh = preprocessMerged([A, B], 0.15);
    const x0 = Float64Array.from(mesh.positions);
    const x1 = Float64Array.from(x0);
    const nA = A.positions.length / 3;
    for (let i = 0; i < mesh.count; i++) x1[i * 3] += i < nA ? 0.0045 : -0.0045;
    const cs = new ContactSystem({ ...P }, mesh.indices);
    cs.beginStep(x0);
    cs.updateActiveSet(x1);
    cs.commit(x1);
    const gcs = new GpuContactSystem({
      indices: mesh.indices, triCount: mesh.triCount,
      dHatM: P.dHatM, dMinM: P.dMinM, kappaJ: P.kappaJ,
      frictionMu: P.frictionMu, frictionEpsM: P.frictionEpsM,
    });
    gcs.beginStep(x0);
    const bp = await new CpuBroadPhase({ indices: mesh.indices, triCount: mesh.triCount, pad: P.dHatM }).build(x0, x1);
    const set = await gcs.build(x1, bp);
    gcs.commit(x1, set);
    expect(gcs.lagged.size).toBe(cs.lagged.size);
    for (const [k, lag] of cs.lagged as Map<string, { nx: number; ny: number; nz: number; lambdaN: number }>) {
      const g = gcs.lagged.get(k)!;
      expect(g).toBeDefined();
      expect(Math.abs(g.nx - lag.nx)).toBeLessThan(1e-6);
      expect(Math.abs(g.lambdaN - lag.lambdaN) / Math.max(Math.abs(lag.lambdaN), 1e-12)).toBeLessThan(1e-4);
    }
  });
});
