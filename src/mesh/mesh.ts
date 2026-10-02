import type { Hinge } from "../physics/types.js";

export interface ClothMeshData {
  count: number; // vertices
  triCount: number;
  positions: Float32Array; // n*3 (initial)
  restPositions: Float32Array;
  uv: Float32Array; // n*2
  indices: Uint32Array; // m*3
  invDm: Float32Array; // m*4 [a,b,c,d]
  areas: Float32Array; // m
  masses: Float32Array; // n lumped
  hinges: Hinge[];
}

/** Flat grid in XZ plane, y=0, x in [0,w], z in [0,h]. uv=(x,z). */
export function buildGrid(nx: number, ny: number, w: number, h: number): {
  positions: Float32Array;
  uv: Float32Array;
  indices: Uint32Array;
} {
  const n = (nx + 1) * (ny + 1);
  const positions = new Float32Array(n * 3);
  const uv = new Float32Array(n * 2);
  for (let j = 0; j <= ny; j++) {
    for (let i = 0; i <= nx; i++) {
      const id = j * (nx + 1) + i;
      const x = (i / nx) * w;
      const z = (j / ny) * h;
      positions[id * 3] = x;
      positions[id * 3 + 1] = 0;
      positions[id * 3 + 2] = z;
      uv[id * 2] = x;
      uv[id * 2 + 1] = z;
    }
  }
  const idx: number[] = [];
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const a = j * (nx + 1) + i;
      const b = a + 1;
      const c = a + (nx + 1);
      const d = c + 1;
      idx.push(a, c, b, b, c, d);
    }
  }
  return { positions, uv, indices: new Uint32Array(idx) };
}

export function preprocess(
  positions: Float32Array,
  uv: Float32Array,
  indices: Uint32Array,
  arealDensityKgM2: number,
): ClothMeshData {
  const n = positions.length / 3;
  const m = indices.length / 3;
  const invDm = new Float32Array(m * 4);
  const areas = new Float32Array(m);
  const masses = new Float32Array(n);

  for (let t = 0; t < m; t++) {
    const i0 = indices[t * 3];
    const i1 = indices[t * 3 + 1];
    const i2 = indices[t * 3 + 2];
    const u0 = uv[i0 * 2], v0 = uv[i0 * 2 + 1];
    const u1 = uv[i1 * 2], v1 = uv[i1 * 2 + 1];
    const u2 = uv[i2 * 2], v2 = uv[i2 * 2 + 1];
    const du1 = u1 - u0, dv1 = v1 - v0;
    const du2 = u2 - u0, dv2 = v2 - v0;
    const det = du1 * dv2 - du2 * dv1;
    if (Math.abs(det) < 1e-12) throw new Error(`degenerate triangle ${t}`);
    const area = 0.5 * Math.abs(det);
    areas[t] = area;
    // inv([[du1,du2],[dv1,dv2]])
    invDm[t * 4] = dv2 / det; // a
    invDm[t * 4 + 1] = -du2 / det; // b
    invDm[t * 4 + 2] = -dv1 / det; // c
    invDm[t * 4 + 3] = du1 / det; // d
    // Lumped mass: arealDensityKgM2 is AREA density (kg/m^2), so m = arealDensity * area / 3.
    // Thickness enters only the elastic energy (volume density), not inertia.
    const tm = arealDensityKgM2 * area / 3;
    masses[i0] += tm;
    masses[i1] += tm;
    masses[i2] += tm;
  }

  const hinges = buildHinges(indices, positions, areas);
  return {
    count: n, triCount: m,
    positions: Float32Array.from(positions),
    restPositions: Float32Array.from(positions),
    uv: Float32Array.from(uv),
    indices: Uint32Array.from(indices),
    invDm, areas, masses, hinges,
  };
}

function buildHinges(indices: Uint32Array, pos: Float32Array, areas: Float32Array): Hinge[] {
  const m = indices.length / 3;
  const edgeMap = new Map<string, { tri: number; a: number; b: number; opp: number }[]>();
  const key = (a: number, b: number) => (a < b ? `${a}_${b}` : `${b}_${a}`);
  for (let t = 0; t < m; t++) {
    const ids = [indices[t * 3], indices[t * 3 + 1], indices[t * 3 + 2]];
    for (let e = 0; e < 3; e++) {
      const a = ids[e], b = ids[(e + 1) % 3], opp = ids[(e + 2) % 3];
      const k = key(a, b);
      if (!edgeMap.has(k)) edgeMap.set(k, []);
      edgeMap.get(k)!.push({ tri: t, a, b, opp });
    }
  }
  const hinges: Hinge[] = [];
  for (const list of edgeMap.values()) {
    if (list.length !== 2) continue; // boundary has no hinge
    const [e0, e1] = list;
    // order: v0,v1 = shared edge, v2 = opp of tri0, v3 = opp of tri1
    const v0 = e0.a, v1 = e0.b, v2 = e0.opp, v3 = e1.opp;
    const restAngle = dihedral(pos, v0, v1, v2, v3);
    const ax = pos[v0 * 3] - pos[v1 * 3];
    const ay = pos[v0 * 3 + 1] - pos[v1 * 3 + 1];
    const az = pos[v0 * 3 + 2] - pos[v1 * 3 + 2];
    const edgeLen = Math.hypot(ax, ay, az);
    hinges.push({ v0, v1, v2, v3, restAngle, edgeLen, areaSum: areas[e0.tri] + areas[e1.tri] });
  }
  return hinges;
}

/** Dihedral angle between triangles (v0,v1,v2) and (v0,v1,v3), signed via edge direction. */
export function dihedral(x: ArrayLike<number>, v0: number, v1: number, v2: number, v3: number): number {
  const e0x = x[v1 * 3] - x[v0 * 3], e0y = x[v1 * 3 + 1] - x[v0 * 3 + 1], e0z = x[v1 * 3 + 2] - x[v0 * 3 + 2];
  const e1x = x[v2 * 3] - x[v0 * 3], e1y = x[v2 * 3 + 1] - x[v0 * 3 + 1], e1z = x[v2 * 3 + 2] - x[v0 * 3 + 2];
  const e2x = x[v3 * 3] - x[v0 * 3], e2y = x[v3 * 3 + 1] - x[v0 * 3 + 1], e2z = x[v3 * 3 + 2] - x[v0 * 3 + 2];
  // n1 = e0 x e1, n2 = e0 x e2 (note orientation may flip; use consistent sign)
  const n1 = cross(e0x, e0y, e0z, e1x, e1y, e1z);
  const n2 = cross(e0x, e0y, e0z, e2x, e2y, e2z);
  const l1 = Math.hypot(n1[0], n1[1], n1[2]) + 1e-30;
  const l2 = Math.hypot(n2[0], n2[1], n2[2]) + 1e-30;
  const cosT = Math.min(1, Math.max(-1, (n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2]) / (l1 * l2)));
  return Math.acos(cosT);
}

function cross(ax: number, ay: number, az: number, bx: number, by: number, bz: number): [number, number, number] {
  return [ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx];
}
