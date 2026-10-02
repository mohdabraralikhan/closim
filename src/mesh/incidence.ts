// G4B topology incidence maps (STATIC — built once per mesh, uploaded once).
// CSR-like vertex -> incident-element lists for gather assembly:
//   offsets[v] .. offsets[v+1] index into ids/corners (offsets length n+1).
// Records are appended in ascending element order, so per-vertex lists are
// element-ordered and gather sums are BIT-IDENTICAL to direct scatter sums.
// Hinge corners follow assemble-gradient's first-match-wins else-if chain
// (valid meshes never repeat a vertex within one hinge).

import type { Hinge } from "../physics/types.js";

export interface IncidenceMap {
  /** length n+1 CSR row pointers (u32) */
  offsets: Uint32Array;
  /** element id per record (u32) */
  ids: Uint32Array;
  /** local corner per record: 0..2 membrane, 0..3 hinge (u32) */
  corners: Uint32Array;
}

function buildMap(
  vertexCount: number,
  recordCount: number,
  push: (emit: (v: number, id: number, c: number) => void) => void,
): IncidenceMap {
  const counts = new Uint32Array(vertexCount);
  push((v) => { counts[v]++; });
  const offsets = new Uint32Array(vertexCount + 1);
  for (let v = 0; v < vertexCount; v++) offsets[v + 1] = offsets[v] + counts[v];
  const ids = new Uint32Array(recordCount);
  const corners = new Uint32Array(recordCount);
  const cursor = Uint32Array.from(offsets.subarray(0, vertexCount));
  push((v, id, c) => {
    const k = cursor[v]++;
    ids[k] = id;
    corners[k] = c;
  });
  return { offsets, ids, corners };
}

/** Membrane incidence: one record per (triangle, corner). Total = 3 * triCount. */
export function buildMembraneIncidence(
  indices: ArrayLike<number>,
  triCount: number,
  vertexCount: number,
): IncidenceMap {
  return buildMap(vertexCount, 3 * triCount, (emit) => {
    for (let t = 0; t < triCount; t++) {
      for (let c = 0; c < 3; c++) {
        emit(indices[t * 3 + c], t, c);
      }
    }
  });
}

/** Hinge incidence: one record per (hinge, corner), first-match-wins per
 *  vertex (mirrors assemble-gradient's else-if chain). Total = 4 * hingeCount. */
export function buildHingeIncidence(
  hinges: ArrayLike<Hinge>,
  hingeCount: number,
  vertexCount: number,
): IncidenceMap {
  return buildMap(vertexCount, 4 * hingeCount, (emit) => {
    for (let hh = 0; hh < hingeCount; hh++) {
      const h = hinges[hh];
      const vs = [h.v0, h.v1, h.v2, h.v3];
      const seen = new Set<number>();
      for (let c = 0; c < 4; c++) {
        if (!seen.has(vs[c])) {
          seen.add(vs[c]);
          emit(vs[c], hh, c);
        }
      }
    }
  });
}

/** CPU gather oracle (test/mirror only): dst[v*3+i] += src[t*W+c*3+i].
 *  With element-ordered maps this reproduces direct scatter bit-exactly. */
export function gatherAdd(
  dst: Float64Array | Float32Array,
  src: ArrayLike<number>,
  elemWidth: number,
  map: IncidenceMap,
): void {
  for (let v = 0; v < map.offsets.length - 1; v++) {
    for (let k = map.offsets[v]; k < map.offsets[v + 1]; k++) {
      const base = map.ids[k] * elemWidth + map.corners[k] * 3;
      for (let i = 0; i < 3; i++) {
        (dst as Float64Array)[v * 3 + i] += src[base + i];
      }
    }
  }
}
