// Self-collision pair expansion: triangle pair -> VT + EE descriptors.
// Skips edge pairs sharing an endpoint (zero-distance adjacency artifacts).

export interface VtDesc { p: number; a: number; b: number; c: number }
export interface EeDesc { a: number; b: number; c: number; d: number }

export function buildEdges(indices: Uint32Array): Array<[number, number]> {
  const set = new Map<string, [number, number]>();
  const m = indices.length / 3;
  for (let t = 0; t < m; t++) {
    const ids = [indices[t * 3], indices[t * 3 + 1], indices[t * 3 + 2]];
    for (let e = 0; e < 3; e++) {
      const u = ids[e], v = ids[(e + 1) % 3];
      const k = u < v ? `${u}_${v}` : `${v}_${u}`;
      if (!set.has(k)) set.set(k, [u, v]);
    }
  }
  return [...set.values()];
}

const TRI_EDGES = [[0, 1], [1, 2], [2, 0]] as const;

export function expandTriPair(
  tA: number, tB: number,
  indices: Uint32Array,
): { vt: VtDesc[]; ee: EeDesc[] } {
  const A = [indices[tA * 3], indices[tA * 3 + 1], indices[tA * 3 + 2]];
  const B = [indices[tB * 3], indices[tB * 3 + 1], indices[tB * 3 + 2]];
  const vt: VtDesc[] = [];
  for (const p of A) vt.push({ p, a: B[0], b: B[1], c: B[2] });
  for (const p of B) vt.push({ p, a: A[0], b: A[1], c: A[2] });
  const ee: EeDesc[] = [];
  for (const [ea0, ea1] of TRI_EDGES) {
    for (const [eb0, eb1] of TRI_EDGES) {
      const a = A[ea0], b = A[ea1], c = B[eb0], d = B[eb1];
      if (a === c || a === d || b === c || b === d) continue;
      ee.push({ a, b, c, d });
    }
  }
  return { vt, ee };
}
