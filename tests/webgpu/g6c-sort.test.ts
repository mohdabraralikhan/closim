// G6C.1 indexed sort: A/B bit-exactness vs the legacy uniform path,
// correctness against CPU-checked invariants, spec edge cases (duplicates,
// all-equal, reverse, max-keys), and submit evidence for the flush removal.
import { describe, it, expect, beforeAll } from "vitest";
import { nextPow2 } from "../../src/backend/webgpu/gpu-buffers.js";
import type { DeviceFixture } from "./device-setup.js";
import { sharedDevice, stripScene } from "./device-setup.js";

let strip: DeviceFixture | null = null;

beforeAll(async () => {
  strip = await sharedDevice("g6c-sort-strip", stripScene, { contactCapacity: 64, pairCapacity: 512 });
}, 180000);

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Seed keys/payload (tail INF), run one sort path, read both buffers back. */
async function runSortPath(
  fix: DeviceFixture, keys: Uint32Array, indexed: boolean,
): Promise<{ keys: Uint32Array; payload: Uint32Array; submits: number; passes: number }> {
  const { ex, driver } = fix;
  ex.writeBuffer("mortonKeys", keys);
  const P = keys.length;
  const payload = new Uint32Array(P);
  for (let i = 0; i < P; i++) payload[i] = i;
  ex.writeBuffer("mortonPayload", payload);
  const prev = driver.cfg.useIndexedSort;
  driver.cfg.useIndexedSort = indexed;
  const s0 = ex.ledger.submits;
  const p0 = ex.ledger.passes;
  try {
    ex.beginBatch(indexed ? "sort-indexed" : "sort-legacy");
    driver.sortPasses();
    await ex.submitBatch(false);
  } finally {
    driver.cfg.useIndexedSort = prev;
  }
  const kRaw = await ex.readBufferDebug("mortonKeys", "sort-keys", false);
  const pRaw = await ex.readBufferDebug("mortonPayload", "sort-payload", false);
  return {
    keys: new Uint32Array(kRaw),
    payload: new Uint32Array(pRaw),
    submits: ex.ledger.submits - s0,
    passes: ex.ledger.passes - p0,
  };
}

/** Sortedness + multiset + tiebreak invariants (CPU-checked). */
function checkSorted(
  keysIn: Uint32Array, payIn: Uint32Array,
  keysOut: Uint32Array, payOut: Uint32Array, m: number,
): void {
  // multiset preserved over the real lanes
  const norm = (k: Uint32Array, p: Uint32Array): string[] => {
    const arr: string[] = [];
    for (let i = 0; i < m; i++) arr.push(`${k[i]}:${p[i]}`);
    return arr.sort();
  };
  expect(norm(keysOut, payOut)).toEqual(norm(keysIn, payIn));
  // nondecreasing keys with payload tiebreak over ALL P lanes
  const P = keysOut.length;
  for (let i = 1; i < P; i++) {
    const ok = keysOut[i] > keysOut[i - 1] ||
      (keysOut[i] === keysOut[i - 1] && payOut[i] >= payOut[i - 1]);
    expect(ok, `lane ${i}: ${keysOut[i - 1]}:${payOut[i - 1]} -> ${keysOut[i]}:${payOut[i]}`).toBe(true);
  }
  // INF tail sinks past all real lanes (real keys < 0xFFFFFFFF here)
  for (let i = m; i < P; i++) expect(keysOut[i]).toBe(0xffffffff);
}

describe("G6C.1 indexed sort", () => {
  it("A/B bit-exact on randomized keys with duplicates", async () => {
    if (!strip) return;
    const { solver } = strip;
    const scene = (solver as unknown as { scene: { mesh: { triCount: number } } }).scene;
    const m = scene.mesh.triCount;
    const P = nextPow2(Math.max(m, 1));
    const rand = mulberry32(0x5017);
    const keys = new Uint32Array(P);
    for (let i = 0; i < m; i++) keys[i] = Math.floor(rand() * 8); // heavy duplicates
    for (let i = m; i < P; i++) keys[i] = 0xffffffff;
    const legacy = await runSortPath(strip, Uint32Array.from(keys), false);
    const indexed = await runSortPath(strip, Uint32Array.from(keys), true);
    expect([...indexed.keys]).toEqual([...legacy.keys]);
    expect([...indexed.payload]).toEqual([...legacy.payload]);
    // eslint-disable-next-line no-console
    console.log(`[g6c-sort] A/B exact over ${P} lanes; submits legacy=${legacy.submits} indexed=${indexed.submits} ` +
      `passes legacy=${legacy.passes} indexed=${indexed.passes}`);
    checkSorted(keys, Uint32Array.from({ length: P }, (_, i) => i), indexed.keys, indexed.payload, m);
    // Submit evidence: legacy flushes per sub-pass, indexed (almost) never.
    expect(indexed.submits).toBeLessThan(legacy.submits);
  }, 180000);

  it("edge cases: all-equal, reverse, max-keys", async () => {
    if (!strip) return;
    const { solver } = strip;
    const scene = (solver as unknown as { scene: { mesh: { triCount: number } } }).scene;
    const m = scene.mesh.triCount;
    const P = nextPow2(Math.max(m, 1));
    const cases: Array<{ name: string; fill: (i: number) => number }> = [
      { name: "all-equal", fill: () => 42 },
      { name: "reverse", fill: (i) => (P - 1 - i) >>> 0 },
      { name: "max-keys", fill: () => 0xffffffff },
    ];
    for (const c of cases) {
      const keys = new Uint32Array(P);
      for (let i = 0; i < m; i++) keys[i] = c.fill(i);
      for (let i = m; i < P; i++) keys[i] = 0xffffffff;
      const legacy = await runSortPath(strip, Uint32Array.from(keys), false);
      const indexed = await runSortPath(strip, Uint32Array.from(keys), true);
      // eslint-disable-next-line no-console
      console.log(`[g6c-sort] ${c.name}: A/B exact=${[...indexed.keys].every((v, i) => v === legacy.keys[i])}`);
      expect([...indexed.keys]).toEqual([...legacy.keys]);
      expect([...indexed.payload]).toEqual([...legacy.payload]);
      if (c.name !== "max-keys") {
        checkSorted(keys, Uint32Array.from({ length: P }, (_, i) => i), indexed.keys, indexed.payload, m);
      }
      // max-keys: real lanes indistinguishable from INF tail by design (G1
      // policy); the gate is indexed ≡ legacy, asserted above.
    }
  }, 180000);
});
