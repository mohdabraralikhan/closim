// G0 buffer layout + WGSL consistency (no GPU required).
// Guards the alignment rules WGSL depends on: vec4f strides, 64 B uniform and
// status blocks, contact record sizing, and entry-point names.
import { describe, it, expect } from "vitest";
import {
  gpuLayoutBytes, gpuTotalBytes, packVec4Positions, unpackVec4Positions,
  encodeSimParams, decodeSolverStatus, SIM_PARAMS_BYTES, SOLVER_STATUS_BYTES,
  CANDIDATE_PAIR_BYTES, CONTACT_RECORD_BYTES,
} from "../../src/backend/webgpu/gpu-buffers.js";
import { SHADER_ENTRY_POINTS, WORKGROUP_DEFAULT, dispatchWorkgroups } from "../../src/backend/webgpu/gpu-pipelines.js";
import { validateG0Limits } from "../../src/backend/webgpu/gpu-capabilities.js";

describe("webgpu G0 buffer layout", () => {
  it("vec4f position/velocity strides are 16 B", () => {
    const l = gpuLayoutBytes({ vertexCount: 100, triangleCount: 150, hingeCount: 200, contactCapacity: 64 });
    expect(l.position).toBe(1600);
    expect(l.velocity).toBe(1600);
    expect(l.gradient).toBe(100 * 3 * 4);
    expect(l.elementGradient).toBe(150 * 9 * 4);
    expect(l.hingeGradient).toBe(200 * 12 * 4);
  });

  it("uniform + status blocks are 64 B", () => {
    expect(SIM_PARAMS_BYTES).toBe(64);
    expect(SOLVER_STATUS_BYTES).toBe(64);
    expect(encodeSimParams({
      dt: 1 / 60, invDt2: 3600, gravityX: 0, gravityY: -9.81, gravityZ: 0,
      vertexCount: 10, triangleCount: 8, hingeCount: 5, contactCount: 0,
      newtonIteration: 0, pcgIteration: 0, lineSearchAlpha: 1, trustRegion: 0.002,
      barrierActivation: 0.002, barrierEpsilon: 1e-12, frictionMu: 0.3,
    }).byteLength).toBe(64);
  });

  it("vec4 pack/unpack round-trips xyz", () => {
    const xyz = new Float64Array([1, 2, 3, -4, 5, -6]);
    const packed = packVec4Positions(xyz, 2);
    expect(packed.length).toBe(8);
    expect(packed[3]).toBe(0);
    expect(packed[7]).toBe(0);
    const back = unpackVec4Positions(packed, 2);
    expect(Array.from(back)).toEqual([1, 2, 3, -4, 5, -6]);
  });

  it("status decode maps the documented field order", () => {
    const buf = new ArrayBuffer(64);
    new Float32Array(buf).set([1, 2, 3, 4, 0.001, 0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    const st = decodeSolverStatus(buf);
    expect(st.energy).toBeCloseTo(1);
    expect(st.gradNorm).toBeCloseTo(3);
    expect(st.minDistance).toBeCloseTo(0.001);
  });

  it("total bytes scale linearly and stay modest for validation scenes", () => {
    // Validation sizing uses explicit small pair/prim capacities; production
    // defaults (16384 pairs) are honest about worst-case scratch instead.
    const small = gpuTotalBytes({
      vertexCount: 1000, triangleCount: 1800, hingeCount: 2500,
      contactCapacity: 4096, pairCapacity: 2048,
    });
    expect(small).toBeLessThan(8 * 1024 * 1024); // < 8 MiB for the 1k-vertex class
  });

  it("G2 primitive scratch is pairCapacity * 15 by default, overridable", () => {
    const dflt = gpuLayoutBytes({
      vertexCount: 100, triangleCount: 150, hingeCount: 200,
      contactCapacity: 64, pairCapacity: 128,
    });
    expect(dflt.primIdsVT).toBe(128 * 15 * 16);
    expect(dflt.vtSTD).toBe(128 * 15 * 16);
    expect(dflt.vtR).toBe(128 * 15 * 16);
    expect(dflt.contactTOI).toBe(64 * 4);
    expect(dflt.contactCount).toBe(16);
    const over = gpuLayoutBytes({
      vertexCount: 100, triangleCount: 150, hingeCount: 200,
      contactCapacity: 64, pairCapacity: 128, primCapacity: 512,
    });
    expect(over.primIdsVT).toBe(512 * 16);
  });

  it("G2 contact record matches the barrier 64 B frozen layout", () => {
    // CONTACT_RECORD_BYTES already asserted 64 B by construction; the G2
    // encode path is validated byte-for-byte in contact-compaction tests.
    expect(CANDIDATE_PAIR_BYTES).toBe(8);
    expect(CONTACT_RECORD_BYTES).toBe(64);
  });

  it("dispatch helper covers partial workgroups", () => {
    expect(dispatchWorkgroups(64, WORKGROUP_DEFAULT)).toBe(1);
    expect(dispatchWorkgroups(65, WORKGROUP_DEFAULT)).toBe(2);
  });

  it("every G0 shader declares at least one entry point", () => {
    for (const [name, eps] of Object.entries(SHADER_ENTRY_POINTS)) {
      expect(eps.length, name).toBeGreaterThan(0);
    }
  });

  it("G0 limit validator accepts a capable adapter and rejects a weak one", () => {
    const good = validateG0Limits({ maxStorageBufferBindingSize: 128 << 20, maxBufferSize: 256 << 20 });
    expect(good.ok).toBe(true);
    const bad = validateG0Limits({ maxStorageBufferBindingSize: 1024, maxBufferSize: 1024 });
    expect(bad.ok).toBe(false);
  });
});
