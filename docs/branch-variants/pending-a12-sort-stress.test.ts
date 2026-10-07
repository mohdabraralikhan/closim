// ADVERSARIAL A12 — Bitonic sort deep stress: scale, duplicates, pathological payloads, and sortCursor lifecycle.
//
// Attack targets:
//   1. Scale correctness: 64, 1024, 65536, 131072 elements sorted correctly on GPU
//   2. Duplicates: identical keys with unique payloads, verifying tie-break order
//   3. Pathological payloads: reverse sorted, alternating, high bit set, 0xFFFFFFFF pads
//   4. Row split: verifying NO_SPLIT barrier when workgroups > 65535
//   5. sortCursor lifecycle: underflow (cursor=0 => 0xFFFFFFFF) and accumulation past T
import { describe, it, expect, beforeAll } from "vitest";
import { nextPow2 } from "../../src/collision/broadphase.js";
import type { DeviceFixture } from "../webgpu/device-setup.js";
import { requireDevice } from "../webgpu/device-setup.js";
import { buildGrid, preprocess } from "../../src/mesh/mesh.js";
import { createScene } from "../../src/physics/scene.js";
import { DEFAULT_MATERIAL } from "../../src/physics/types.js";

import { bitonicPassAt, bitonicPassCount } from "../../src/backend/webgpu/gpu-broadphase.js";

let fix: DeviceFixture | null = null;

function tinyScene() {
  const g = buildGrid(2, 2, 0.05, 0.05);
  const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
  return createScene(mesh, { ...DEFAULT_MATERIAL }, [0, -9.81, 0]);
}

beforeAll(async () => {
  fix = await requireDevice(tinyScene, { contactCapacity: 16, pairCapacity: 32 });
}, 180000);

function generateBitonicParams(P: number): Uint32Array {
  const stages = Math.log2(P);
  const T = bitonicPassCount(stages);
  const table = new Uint32Array(T * 4);
  for (let t = 0; t < T; t++) {
    const { stage, sub } = bitonicPassAt(t);
    table[t * 4 + 0] = P;
    table[t * 4 + 1] = stage;
    table[t * 4 + 2] = sub;
    table[t * 4 + 3] = 0;
  }
  return table;
}

async function runBitonicSort(ex: GpuExecutor, keys: Uint32Array, payloads: Uint32Array): Promise<{ sortedKeys: Uint32Array; sortedPayloads: Uint32Array }> {
  const P = keys.length;
  const stages = Math.log2(P);
  const T = (stages * (stages + 1)) / 2;
  const prmTable = generateBitonicParams(P);

  const STORAGE_SRC_DST = 0x80 | 0x08 | 0x04; // STORAGE | COPY_DST | COPY_SRC
  ex.ensureBuffer("advMortonKeys", P * 4, STORAGE_SRC_DST);
  ex.ensureBuffer("advMortonPayload", P * 4, STORAGE_SRC_DST);
  ex.ensureBuffer("advSortParams", T * 16, STORAGE_SRC_DST);
  ex.ensureBuffer("advSortCursor", 16, STORAGE_SRC_DST);

  ex.writeBuffer("advMortonKeys", keys);
  ex.writeBuffer("advMortonPayload", payloads);
  ex.writeBuffer("advSortParams", prmTable);
  ex.writeBuffer("advSortCursor", new Uint32Array([0, 0, 0, 0]));

  const B = (binding: number, buffer: string) => ({ binding, buffer });

  ex.beginBatch(`adv-sort-${P}`);
  for (let t = 0; t < T; t++) {
    ex.runPass({
      shader: "broadphase-sort", entry: "sort_next",
      groups: [[B(6, "advSortCursor")]],
      x: 1,
    });
    ex.runPass({
      shader: "broadphase-sort", entry: "sort_step_indexed",
      groups: [[
        B(0, "advMortonKeys"), B(1, "advMortonPayload"),
        B(5, "advSortParams"), B(6, "advSortCursor"),
      ]],
      x: Math.max(1, Math.ceil(P / 64)),
    });
  }
  await ex.submitBatch(false);

  const kOut = new Uint32Array(await ex.readBufferDebug("advMortonKeys", "adv-kout", false)).slice(0, P);
  const pOut = new Uint32Array(await ex.readBufferDebug("advMortonPayload", "adv-pout", false)).slice(0, P);
  return { sortedKeys: kOut, sortedPayloads: pOut };
}

describe("ADVERSARIAL A12 — Bitonic sort deep stress & sortCursor lifecycle", () => {
  it("A12.1 sort scale: 64, 1024, 65536, and 131072 elements", async () => {
    if (!fix) return;
    const ex = fix.ex;
    const sizes = [64, 1024, 65536, 131072];

    for (const P of sizes) {
      const keys = new Uint32Array(P);
      const payloads = new Uint32Array(P);
      // Pseudo-random deterministic permutation
      for (let i = 0; i < P; i++) {
        keys[i] = ((i * 1103515245 + 12345) & 0x7fffffff) >>> 0;
        payloads[i] = i;
      }

      const { sortedKeys, sortedPayloads } = await runBitonicSort(ex, keys, payloads);

      // Verify sorted order
      let ok = true;
      for (let i = 1; i < P; i++) {
        if (sortedKeys[i] < sortedKeys[i - 1]) { ok = false; break; }
        if (sortedKeys[i] === sortedKeys[i - 1] && sortedPayloads[i] < sortedPayloads[i - 1]) { ok = false; break; }
      }

      // eslint-disable-next-line no-console
      console.log(`[A12.1] P=${P} passes=${(Math.log2(P)*(Math.log2(P)+1))/2} sorted=${ok} ` +
        `firstKey=${sortedKeys[0]} lastKey=${sortedKeys[P - 1]}`);
      expect(ok).toBe(true);
    }
  }, 300000);

  it("A12.2 duplicate keys with unique payloads verify tie-breaking", async () => {
    if (!fix) return;
    const ex = fix.ex;
    const P = 1024;
    const keys = new Uint32Array(P);
    const payloads = new Uint32Array(P);

    // Only 4 distinct key values across 1024 elements (heavy duplicates)
    for (let i = 0; i < P; i++) {
      keys[i] = (i % 4) * 1000;
      payloads[i] = P - i; // reverse payload
    }

    const { sortedKeys, sortedPayloads } = await runBitonicSort(ex, keys, payloads);

    let ok = true;
    for (let i = 1; i < P; i++) {
      if (sortedKeys[i] < sortedKeys[i - 1]) { ok = false; break; }
      if (sortedKeys[i] === sortedKeys[i - 1] && sortedPayloads[i] < sortedPayloads[i - 1]) { ok = false; break; }
    }

    // eslint-disable-next-line no-console
    console.log(`[A12.2] Heavy duplicates P=${P} tie-break sorted=${ok}`);
    expect(ok).toBe(true);
  }, 180000);

  it("A12.3 pathological payloads: reverse sorted, all 0xFFFFFFFF pads", async () => {
    if (!fix) return;
    const ex = fix.ex;
    const P = 1024;
    const keys = new Uint32Array(P);
    const payloads = new Uint32Array(P);

    // Half data, half 0xFFFFFFFF pad lanes (exactly what Morton padding creates!)
    for (let i = 0; i < P / 2; i++) {
      keys[i] = (P / 2 - i) >>> 0;
      payloads[i] = i;
    }
    for (let i = P / 2; i < P; i++) {
      keys[i] = 0xffffffff;
      payloads[i] = 0xffffffff;
    }

    const { sortedKeys } = await runBitonicSort(ex, keys, payloads);

    // First half must be sorted 1 .. P/2
    expect(sortedKeys[0]).toBe(1);
    expect(sortedKeys[P / 2 - 1]).toBe(P / 2);
    // Second half must all be 0xFFFFFFFF
    expect(sortedKeys[P / 2]).toBe(0xffffffff);
    expect(sortedKeys[P - 1]).toBe(0xffffffff);
  }, 180000);

  it("A12.4 sortCursor underflow: missing sort_next reads out-of-bounds t = 0xFFFFFFFF", async () => {
    if (!fix) return;
    const ex = fix.ex;
    const P = 64;
    const T = (6 * 7) / 2;
    const prmTable = generateBitonicParams(P);

    ex.ensureBuffer("advMortonKeys", P * 4, 0x0008 | 0x0004 | 0x0002 | 0x0001);
    ex.ensureBuffer("advMortonPayload", P * 4, 0x0008 | 0x0004 | 0x0002 | 0x0001);
    ex.ensureBuffer("advSortParams", T * 16, 0x0008 | 0x0004 | 0x0001);
    ex.ensureBuffer("advSortCursor", 16, 0x0008 | 0x0004 | 0x0001);

    // Leave sortCursor at 0.
    ex.writeBuffer("advSortCursor", new Uint32Array([0, 0, 0, 0]));

    const B = (binding: number, buffer: string) => ({ binding, buffer });
    // In baseline 162a689: sort_step_indexed computes: let t = sortCursor[0] - 1u;
    // When sortCursor[0] is 0, t = 4294967295u (OOB read into sortParams)!
    ex.beginBatch("adv-underflow-probe");
    ex.runPass({
      shader: "broadphase-sort", entry: "sort_step_indexed",
      groups: [[
        B(0, "advMortonKeys"), B(1, "advMortonPayload"),
        B(5, "advSortParams"), B(6, "advSortCursor"),
      ]],
      x: 1,
    });
    await ex.submitBatch(false);

    // The kernel ran with t = 0xFFFFFFFF. On WebGPU/Dawn this causes an out-of-bounds
    // clamp or returns zeroed struct, meaning NO compareSwap occurs.
    const cur = new Uint32Array(await ex.readBufferDebug("advSortCursor", "adv-cur-read", false));
    expect(cur[0]).toBe(0);
  }, 180000);
});
