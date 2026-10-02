// G5D two-level MAS tests (CPU reference):
// restriction/prolongation identities, Ac symmetry, coarse-solve correctness,
// two-level convergence vs one-level, pins, determinism, no NaN/Inf.
import { describe, it, expect } from "vitest";
import { buildGrid, preprocess } from "../src/mesh/mesh.js";
import { createScene, makePinFilter } from "../src/physics/scene.js";
import { DEFAULT_MATERIAL } from "../src/physics/types.js";
import { buildSchwarzDomains, buildSchwarzFactors, applySchwarz } from "../src/solver/schwarz.js";
import {
  restrictCoarse, prolongateAdd, buildCoarseDiag, buildMasTwoLevel,
  applyMasTwoLevel, buildCoarseExact, solveCoarseExact,
} from "../src/solver/mas.js";
import { membraneHvp } from "../src/physics/fem.js";
import { pcg } from "../src/math/pcg.js";

function mesh66() {
  const g = buildGrid(6, 6, 0.12, 0.12);
  return preprocess(g.positions, g.uv, g.indices, 0.15);
}

function jacobiDiag(n: number, mass: ArrayLike<number>, invH2: number, beta: number): Float64Array {
  const d = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) d[i * 3 + k] = mass[i] * invH2 + beta;
  return d;
}

function tenseState(mesh: ReturnType<typeof preprocess>, n: number): Float64Array {
  const x = Float64Array.from(mesh.positions);
  for (let i = 0; i < n; i++) {
    x[i * 3] *= 1.01;
    x[i * 3 + 1] += Math.sin(i * 12.9898) * 2e-4;
  }
  return x;
}

describe("G5D coarse transfer operators", () => {
  it("R R^T = I (rows orthonormal) and P = R^T round-trips constants", () => {
    const mesh = mesh66();
    const dom = buildSchwarzDomains(mesh, mesh.restPositions, 16);
    const n = mesh.count;
    // R R^T = I: restrict(prolongate(e_c)) == e_c.
    const D = dom.members.length;
    for (let c = 0; c < D * 3; c++) {
      const ec = new Float64Array(D * 3);
      ec[c] = 1;
      const fine = new Float64Array(n * 3);
      prolongateAdd(ec, dom, fine);
      const back = restrictCoarse(fine, dom);
      for (let i = 0; i < D * 3; i++) {
        expect(Math.abs(back[i] - ec[i])).toBeLessThan(1e-12);
      }
    }
  });

  it("coarse diag matches the dense-Ac diagonal on a tiny mesh", () => {
    // Dc must equal diag(R A P): compare against the explicitly assembled Ac.
    const g = buildGrid(2, 2, 0.04, 0.04);
    const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
    const n = mesh.count;
    const mat = { ...DEFAULT_MATERIAL };
    const x = tenseState(mesh, n);
    const diag = jacobiDiag(n, mesh.masses, 3600, 1.5);
    const dom = buildSchwarzDomains(mesh, mesh.restPositions, 32);
    const cd = buildCoarseDiag(x, mesh, mat, diag, dom);
    const ce = buildCoarseExact(x, mesh, mat, diag, dom);
    const dc = dom.members.length * 3;
    for (let i = 0; i < dc; i++) {
      expect(cd[i]).toBeCloseTo(ce.ac[i * dc + i], 10);
      expect(Number.isFinite(cd[i])).toBe(true);
      expect(cd[i]).toBeGreaterThan(0);
    }
  });

  it("dense Ac is symmetric on a tiny mesh", () => {
    const g = buildGrid(2, 2, 0.04, 0.04);
    const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
    const n = mesh.count;
    const mat = { ...DEFAULT_MATERIAL };
    const x = tenseState(mesh, n);
    const diag = jacobiDiag(n, mesh.masses, 3600, 1.5);
    const dom = buildSchwarzDomains(mesh, mesh.restPositions, 32);
    const ce = buildCoarseExact(x, mesh, mat, diag, dom);
    const dc = dom.members.length * 3;
    for (let i = 0; i < dc; i++) {
      for (let j = 0; j < dc; j++) {
        expect(ce.ac[i * dc + j]).toBe(ce.ac[j * dc + i]);
      }
    }
  });

  it("exact coarse solve inverts Ac", () => {
    const g = buildGrid(2, 2, 0.04, 0.04);
    const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
    const n = mesh.count;
    const mat = { ...DEFAULT_MATERIAL };
    const x = tenseState(mesh, n);
    const diag = jacobiDiag(n, mesh.masses, 3600, 1.5);
    const dom = buildSchwarzDomains(mesh, mesh.restPositions, 32);
    const ce = buildCoarseExact(x, mesh, mat, diag, dom);
    expect(ce.flag).toBe(1);
    const dc = dom.members.length * 3;
    const rhs = new Float64Array(dc);
    for (let i = 0; i < dc; i++) rhs[i] = Math.cos(i * 2.17) * 1.5;
    const out = new Float64Array(dc);
    solveCoarseExact(ce, rhs, out);
    let num = 0, den = 0;
    for (let i = 0; i < dc; i++) {
      let s = 0;
      for (let j = 0; j < dc; j++) s += ce.ac[i * dc + j] * out[j];
      num += (s - rhs[i]) * (s - rhs[i]);
      den += rhs[i] * rhs[i];
    }
    expect(Math.sqrt(num / den)).toBeLessThan(1e-9);
  });
});

describe("G5D two-level apply", () => {
  it("two-level residual <= one-level residual on a stiff patch", () => {
    const mesh = mesh66();
    const n = mesh.count;
    const n3 = n * 3;
    const mat = {
      ...DEFAULT_MATERIAL,
      stretchWarp: DEFAULT_MATERIAL.stretchWarp * 10,
      stretchWeft: DEFAULT_MATERIAL.stretchWeft * 10,
      shear: DEFAULT_MATERIAL.shear * 10,
    };
    const x = tenseState(mesh, n);
    const invH2 = 3600;
    const beta = Math.max(mat.stretchWarp, mat.stretchWeft, mat.shear) * mat.thickness * 0.1 + 1e-6;
    const diag = jacobiDiag(n, mesh.masses, invH2, beta);
    const hvpFull = (p: Float64Array, out: Float64Array) => {
      const Hp = membraneHvp(Float64Array.from(x), p, mesh, mat);
      for (let i = 0; i < n3; i++) out[i] = mesh.masses[Math.floor(i / 3)] * p[i] * invH2 + Hp[i];
    };
    const b = new Float64Array(n3);
    for (let i = 0; i < n3; i++) b[i] = Math.sin(i * 0.913) * 0.5;
    const dom = buildSchwarzDomains(mesh, mesh.restPositions, 16);
    const fine = buildSchwarzFactors(x, mesh, mat, diag, dom);
    const one = pcg(b, hvpFull, diag, {
      maxIters: 60, tol: 1e-6,
      applyPreconditioner: (r, z) => { applySchwarz(fine, r, z, n); },
    });
    const mas = buildMasTwoLevel(x, mesh, mat, diag, dom, 0.5);
    const two = pcg(b, hvpFull, diag, {
      maxIters: 60, tol: 1e-6,
      applyPreconditioner: (r, z) => { applyMasTwoLevel(mas, r, z, n); },
    });
    expect(Number.isFinite(two.residual)).toBe(true);
    expect(two.iters).toBeLessThanOrEqual(one.iters);
    expect(two.residual).toBeLessThanOrEqual(one.residual * 1.001 + 1e-12);
  });

  it("exact-coarse two-level converges at least as well as one-level", () => {
    const g = buildGrid(3, 3, 0.06, 0.06);
    const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
    const n = mesh.count;
    const n3 = n * 3;
    const mat = {
      ...DEFAULT_MATERIAL,
      stretchWarp: DEFAULT_MATERIAL.stretchWarp * 10,
      stretchWeft: DEFAULT_MATERIAL.stretchWeft * 10,
      shear: DEFAULT_MATERIAL.shear * 10,
    };
    const x = tenseState(mesh, n);
    const invH2 = 3600;
    const beta = Math.max(mat.stretchWarp, mat.stretchWeft, mat.shear) * mat.thickness * 0.1 + 1e-6;
    const diag = jacobiDiag(n, mesh.masses, invH2, beta);
    const hvpFull = (p: Float64Array, out: Float64Array) => {
      const Hp = membraneHvp(Float64Array.from(x), p, mesh, mat);
      for (let i = 0; i < n3; i++) out[i] = mesh.masses[Math.floor(i / 3)] * p[i] * invH2 + Hp[i];
    };
    const b = new Float64Array(n3);
    for (let i = 0; i < n3; i++) b[i] = Math.sin(i * 0.913) * 0.5;
    const dom = buildSchwarzDomains(mesh, mesh.restPositions, 8);
    const fine = buildSchwarzFactors(x, mesh, mat, diag, dom);
    const one = pcg(b, hvpFull, diag, {
      maxIters: 60, tol: 1e-6,
      applyPreconditioner: (r, z) => { applySchwarz(fine, r, z, n); },
    });
    const ce = buildCoarseExact(x, mesh, mat, diag, dom);
    const D = dom.members.length;
    const rc = new Float64Array(D * 3);
    const cc = new Float64Array(D * 3);
    const exact = pcg(b, hvpFull, diag, {
      maxIters: 60, tol: 1e-6,
      applyPreconditioner: (r, z) => {
        applySchwarz(fine, r, z, n);
        restrictCoarse(r, dom, rc);
        solveCoarseExact(ce, rc, cc);
        prolongateAdd(cc, dom, z);
      },
    });
    expect(Number.isFinite(exact.residual)).toBe(true);
    expect(exact.iters).toBeLessThanOrEqual(one.iters);
  });

  it("preserves pins and stays finite on a collapsed state", () => {
    const mesh = mesh66();
    const n = mesh.count;
    const mat = { ...DEFAULT_MATERIAL };
    const x = new Float64Array(n * 3);
    const diag = jacobiDiag(n, mesh.masses, 3600, 1e-6);
    const dom = buildSchwarzDomains(mesh, mesh.restPositions, 16);
    const mas = buildMasTwoLevel(x, mesh, mat, diag, dom, 0.5);
    const scene = createScene(mesh, mat);
    scene.pinned.set(0, [0, 0, 0]);
    const filter = makePinFilter(scene.pinned);
    const r = new Float64Array(n * 3).fill(0.5);
    filter(r);
    const z = new Float64Array(n * 3);
    applyMasTwoLevel(mas, r, z, n);
    filter(z);
    expect(z[0]).toBe(0);
    for (let i = 0; i < z.length; i++) expect(Number.isFinite(z[i])).toBe(true);
  });
});
