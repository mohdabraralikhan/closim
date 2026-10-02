// G4A CPU analytic membrane HVP vs the FD oracle (fem.ts membraneHvp),
// plus invariance/symmetry structural tests. Deterministic (mulberry32).
import { describe, it, expect } from "vitest";
import { triangleHvp, evalMembraneHvp } from "../src/physics/membrane-hvp.js";
import { triangleEnergyGradient } from "../src/physics/membrane.js";
import { membraneHvp, evalMembrane } from "../src/physics/fem.js";
import { buildGrid, preprocess } from "../src/mesh/mesh.js";
import type { ClothMaterial } from "../src/physics/types.js";

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randMat(rng: () => number): ClothMaterial {
  return {
    arealDensityKgM2: 0.15,
    thickness: 1e-4 + rng() * 1.9e-3,
    stretchWarp: 1e3 + rng() * 9.9e4,
    stretchWeft: 1e3 + rng() * 9.9e4,
    stretchCoupling: (rng() * 2 - 1) * 5000,
    shear: 1e3 + rng() * 1.9e4,
    bendWarp: 1e-5, bendWeft: 1e-5, damping: 0.001,
  };
}

function relErr(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let num = 0, den = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    num += d * d; den += b[i] * b[i];
  }
  return Math.sqrt(num / Math.max(den, 1e-300));
}

describe("G4A analytic membrane HVP", () => {
  it("matches the FD oracle on randomized elements/materials/directions", () => {
    const rng = mulberry32(0xC10A);
    let worst = 0;
    for (let k = 0; k < 200; k++) {
      const L1 = 0.01 + rng() * 0.09;
      const L2 = 0.01 + rng() * 0.09;
      const strain = k % 5 === 4 ? 1.0 : 0.3; // every 5th: large strain
      const dx = () => (rng() * 2 - 1) * strain * 0.05;
      // axis-aligned rest triangle (w.l.o.g.); deformation carries generality
      const x0 = [dx(), dx(), dx()];
      const x1 = [L1 + dx(), dx(), dx()];
      const x2 = [dx(), L2 + dx(), dx()];
      const v0 = [rng() * 2 - 1, rng() * 2 - 1, rng() * 2 - 1];
      const v1 = [rng() * 2 - 1, rng() * 2 - 1, rng() * 2 - 1];
      const v2 = [rng() * 2 - 1, rng() * 2 - 1, rng() * 2 - 1];
      const inv = [1 / L1, 0, 0, 1 / L2];
      const area = 0.5 * L1 * L2;
      const mat = randMat(rng);
      const an = triangleHvp(x0, x1, x2, v0, v1, v2, inv, area, mat).out;
      // FD oracle at element level: embed in a 1-triangle mesh evaluate
      const X = new Float64Array([...x0, ...x1, ...x2]);
      const P = new Float64Array([...v0, ...v1, ...v2]);
      const mesh = {
        count: 3, triCount: 1,
        indices: new Uint32Array([0, 1, 2]),
        invDm: new Float64Array(inv), areas: new Float64Array([area]),
        hinges: [],
      } as unknown as Parameters<typeof membraneHvp>[2];
      const fd = membraneHvp(X, P, mesh, mat);
      const e = relErr(an, fd);
      worst = Math.max(worst, e);
      expect(e).toBeLessThan(1e-7);
    }
    // eslint-disable-next-line no-console
    console.log(`[g4a] worst analytic-vs-FD rel err over 200 cases: ${worst.toExponential(2)}`);
  });

  it("matches at rest (identity F, zero strain)", () => {
    const mat = randMat(mulberry32(7));
    const x0 = [0, 0, 0], x1 = [0.05, 0, 0], x2 = [0, 0.04, 0];
    const v0 = [0.1, -0.2, 0.3], v1 = [-0.4, 0.5, -0.6], v2 = [0.7, 0.8, -0.9];
    const inv = [20, 0, 0, 25];
    const an = triangleHvp(x0, x1, x2, v0, v1, v2, inv, 0.001, mat).out;
    const X = new Float64Array([...x0, ...x1, ...x2]);
    const P = new Float64Array([...v0, ...v1, ...v2]);
    const mesh = {
      count: 3, triCount: 1,
      indices: new Uint32Array([0, 1, 2]),
      invDm: new Float64Array(inv), areas: new Float64Array([0.001]),
      hinges: [],
    } as unknown as Parameters<typeof membraneHvp>[2];
    expect(relErr(an, membraneHvp(X, P, mesh, mat))).toBeLessThan(1e-5);
  });

  it("rigid translation direction maps to exact zero", () => {
    const mat = randMat(mulberry32(11));
    const x0 = [0.01, -0.02, 0.03], x1 = [0.07, 0.01, -0.02], x2 = [-0.03, 0.05, 0.04];
    const t = [0.5, -0.3, 0.8];
    const an = triangleHvp(x0, x1, x2, t, t, t, [18, 1, -2, 22], 0.0012, mat).out;
    for (const v of an) expect(Math.abs(v)).toBe(0);
  });

  it("rigid rotation field rotates the gradient (geometric stiffness)", () => {
    const rng = mulberry32(13);
    const mat = randMat(rng);
    const x0 = [0.01, -0.02, 0.03], x1 = [0.07, 0.01, -0.02], x2 = [-0.03, 0.05, 0.04];
    const inv = [18, 1, -2, 22];
    const area = 0.0012;
    const w = [0.3, -0.5, 0.7]; // angular velocity
    const cross = (x: number[]): number[] => [
      w[1] * x[2] - w[2] * x[1],
      w[2] * x[0] - w[0] * x[2],
      w[0] * x[1] - w[1] * x[0],
    ];
    // Energy is rotation-invariant (dE = 0) but the GRADIENT rotates with the
    // body: H*v must equal the infinitesimal rotation of grad (initial-stress
    // / geometric part of the Hessian). FD gets this only approximately.
    const an = triangleHvp(x0, x1, x2, cross(x0), cross(x1), cross(x2), inv, area, mat).out;
    const g = triangleEnergyGradient(x0, x1, x2, inv, area, mat).grad;
    const expected = new Float64Array(9);
    for (let v = 0; v < 3; v++) {
      const gv = [g[v * 3], g[v * 3 + 1], g[v * 3 + 2]];
      const c = cross(gv);
      expected[v * 3] = c[0]; expected[v * 3 + 1] = c[1]; expected[v * 3 + 2] = c[2];
    }
    expect(relErr(an, expected)).toBeLessThan(1e-12);
  });

  it("is symmetric: u.Hv == v.Hu", () => {
    const rng = mulberry32(17);
    const mat = randMat(rng);
    const g = buildGrid(3, 3, 0.06, 0.06);
    const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
    const n = mesh.count;
    const x = Float64Array.from(mesh.positions);
    for (let i = 0; i < x.length; i++) x[i] += (rng() * 2 - 1) * 0.005;
    const u = new Float64Array(n * 3);
    const v = new Float64Array(n * 3);
    for (let i = 0; i < u.length; i++) { u[i] = rng() * 2 - 1; v[i] = rng() * 2 - 1; }
    const Hu = evalMembraneHvp(x, u, mesh, mat);
    const Hv = evalMembraneHvp(x, v, mesh, mat);
    let a = 0, b = 0;
    for (let i = 0; i < u.length; i++) { a += u[i] * Hv[i]; b += v[i] * Hu[i]; }
    expect(Math.abs(a - b) / Math.max(Math.abs(a), 1e-300)).toBeLessThan(1e-12);
  });

  it("mesh-level analytic matches FD oracle (assembly wiring)", () => {
    const rng = mulberry32(19);
    const mat = randMat(rng);
    const g = buildGrid(3, 3, 0.06, 0.06);
    const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
    const n = mesh.count;
    const x = Float64Array.from(mesh.positions);
    for (let i = 0; i < x.length; i++) x[i] += (rng() * 2 - 1) * 0.004;
    const p = new Float64Array(n * 3);
    for (let i = 0; i < p.length; i++) p[i] = rng() * 2 - 1;
    const an = evalMembraneHvp(x, p, mesh, mat);
    const fd = membraneHvp(x, p, mesh, mat);
    expect(relErr(an, fd)).toBeLessThan(1e-5);
  });

  it("zero direction gives zero output", () => {
    const mat = randMat(mulberry32(23));
    const g = buildGrid(2, 2, 0.05, 0.05);
    const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
    const n = mesh.count;
    const out = evalMembraneHvp(Float64Array.from(mesh.positions), new Float64Array(n * 3), mesh, mat);
    for (const v of out) expect(Math.abs(v)).toBe(0);
  });

  it("shear-only material under shear matches FD (S01 convention)", () => {
    const mat: ClothMaterial = {
      arealDensityKgM2: 0.15, thickness: 0.001,
      stretchWarp: 0, stretchWeft: 0, stretchCoupling: 0, shear: 8000,
      bendWarp: 0, bendWeft: 0, damping: 0,
    };
    // pure shear deformation of a unit right triangle
    const x0 = [0, 0, 0], x1 = [0.05, 0, 0], x2 = [0.02, 0.04, 0];
    const v0 = [0.1, 0.2, -0.1], v1 = [-0.3, 0.1, 0.2], v2 = [0.2, -0.4, 0.1];
    const inv = [20, 0, 0, 25];
    const an = triangleHvp(x0, x1, x2, v0, v1, v2, inv, 0.001, mat).out;
    const X = new Float64Array([...x0, ...x1, ...x2]);
    const P = new Float64Array([...v0, ...v1, ...v2]);
    const mesh = {
      count: 3, triCount: 1,
      indices: new Uint32Array([0, 1, 2]),
      invDm: new Float64Array(inv), areas: new Float64Array([0.001]),
      hinges: [],
    } as unknown as Parameters<typeof membraneHvp>[2];
    // shear-only stiffness is soft: absolute-scale comparison via FD
    expect(relErr(an, membraneHvp(X, P, mesh, mat))).toBeLessThan(1e-5);
    // and non-trivial (the shear convention actually contributes)
    let mag = 0;
    for (const v of an) mag += v * v;
    expect(mag).toBeGreaterThan(0);
  });
});
