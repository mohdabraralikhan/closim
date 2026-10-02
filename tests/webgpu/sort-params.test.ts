// G6C.1 bitonic enumeration unit tests (no device): the closed-form index
// must reproduce the nested driver loop order for every stage count, and the
// pass count must match.
import { describe, it, expect } from "vitest";
import { bitonicPassAt, bitonicPassCount } from "../../src/backend/webgpu/gpu-broadphase.js";

function nestedOrder(stages: number): Array<{ stage: number; sub: number }> {
  const out: Array<{ stage: number; sub: number }> = [];
  for (let k = 1; k <= stages; k++) {
    for (let j = k - 1; j >= 0; j--) out.push({ stage: k, sub: j });
  }
  return out;
}

describe("G6C.1 bitonic pass enumeration", () => {
  it("matches the nested loop for S = 0..20", () => {
    for (let s = 0; s <= 20; s++) {
      const ref = nestedOrder(s);
      expect(bitonicPassCount(s)).toBe(ref.length);
      for (let t = 0; t < ref.length; t++) {
        expect(bitonicPassAt(t), `S=${s} t=${t}`).toEqual(ref[t]);
      }
    }
  });

  it("covers 50k-scale lane counts (S=16, T=136)", () => {
    expect(bitonicPassCount(16)).toBe(136);
    const ref = nestedOrder(16);
    for (let t = 0; t < ref.length; t++) {
      expect(bitonicPassAt(t)).toEqual(ref[t]);
    }
    // first/last spot checks
    expect(bitonicPassAt(0)).toEqual({ stage: 1, sub: 0 });
    expect(bitonicPassAt(135)).toEqual({ stage: 16, sub: 0 });
  });

  it("is monotone in the Batcher partial order", () => {
    // Within one stage k, sub descends; stages ascend. Spot-check S=4.
    const seq = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(bitonicPassAt);
    expect(seq.map((p) => p.stage)).toEqual([1, 2, 2, 3, 3, 3, 4, 4, 4, 4]);
    expect(seq.map((p) => p.sub)).toEqual([0, 1, 0, 2, 1, 0, 3, 2, 1, 0]);
  });
});
