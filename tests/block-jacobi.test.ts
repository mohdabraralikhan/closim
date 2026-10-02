// G5B block-Jacobi preconditioner tests (CPU reference):
// inversion correctness, singular fallback, pin preservation,
// PCG convergence parity vs scalar Jacobi, no NaN/Inf.
import { describe, it, expect } from "vitest";
import { buildGrid, preprocess } from "../src/mesh/mesh.js";
import { createScene, makePinFilter } from "../src/physics/scene.js";
import { DEFAULT_MATERIAL } from "../src/physics/types.js";
import { evalMembraneBlocks } from "../src/physics/membrane-blocks.js";
import { buildBlockFactors, applyBlockFactors } from "../src/solver/block-jacobi.js";
import { membraneHvp } from "../src/physics/fem.js";
import { pcg } from "../src/math/pcg.js";

function smallMesh() {
  const g = buildGrid(3, 3, 0.06, 0.06);
  return preprocess(g.positions, g.uv, g.indices, 0.15);
}

function jacobiDiag(n: number, mass: ArrayLike<number>, invH2: number, beta: number): Float64Array {
  const d = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) d[i * 3 + k] = mass[i] * invH2 + beta;
  return d;
}

describe("G5B block-Jacobi factors", () => {
  it("inverts PD blocks: B * inv == I", () => {
    const mesh = smallMesh();
    const n = mesh.count;
    const mat = { ...DEFAULT_MATERIAL };
    const x = Float64Array.from(mesh.positions);
    for (let i = 0; i < x.length; i++) x[i] += Math.sin(i * 12.9898) * 0.002;
    const invH2 = 3600;
    const beta = Math.max(mat.stretchWarp, mat.stretchWeft, mat.shear) * mat.thickness * 0.1 + 1e-6;
    const diag = jacobiDiag(n, mesh.masses, invH2, beta);
    const blocks = evalMembraneBlocks(x, mesh, mat);
    const f = buildBlockFactors(x, mesh, mat, diag);
    let checked = 0;
    for (let v = 0; v < n; v++) {
      if (f.flag[v] !== 1) continue;
      // Rebuild B_v and check B*inv = I.
      const B = new Float64Array(9);
      for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
        B[a * 3 + b] = blocks[v * 9 + a * 3 + b] + (a === b ? diag[v * 3 + a] : 0);
      }
      // symmetrize like the builder
      B[1] = B[3] = 0.5 * (B[1] + B[3]);
      B[2] = B[6] = 0.5 * (B[2] + B[6]);
      B[5] = B[7] = 0.5 * (B[5] + B[7]);
      const I = f.inv.subarray(v * 9, v * 9 + 9);
      for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
        let s = 0;
        for (let k = 0; k < 3; k++) s += B[a * 3 + k] * I[k * 3 + b];
        const want = a === b ? 1 : 0;
        expect(Math.abs(s - want)).toBeLessThan(1e-9);
      }
      checked++;
    }
    expect(checked).toBeGreaterThan(0);
  });

  it("singular block falls back to Jacobi diagonal (flag 0, no NaN)", () => {
    const mesh = smallMesh();
    const n = mesh.count;
    const mat = { ...DEFAULT_MATERIAL };
    const x = Float64Array.from(mesh.positions);
    // Degenerate state: collapse everything to a point -> membrane Hessian
    // singular; mass term still keeps most blocks PD, so force the issue by
    // zeroing the Jacobi diag for vertex 4 (unphysical, tests the latch).
    const invH2 = 3600;
    const beta = 1e-6;
    const diag = jacobiDiag(n, mesh.masses, invH2, beta);
    diag[4 * 3] = 0; diag[4 * 3 + 1] = 0; diag[4 * 3 + 2] = 0;
    const collapsed = new Float64Array(x.length); // all zeros
    const f = buildBlockFactors(collapsed, mesh, mat, diag);
    for (let i = 0; i < f.inv.length; i++) {
      expect(Number.isFinite(f.inv[i])).toBe(true);
    }
    // Vertex 4 has zero diag + zero-area-ish Hessian -> must latch fallback.
    expect(f.flag[4]).toBe(0);
    const r = new Float64Array(n * 3).fill(1);
    const z = new Float64Array(n * 3);
    applyBlockFactors(f, r, z, n);
    for (let i = 0; i < z.length; i++) expect(Number.isFinite(z[i])).toBe(true);
    // Fallback rows are diagonal: z[4] must be 0 (1/0 guarded to 0).
    expect(z[4 * 3]).toBe(0);
  });

  it("preserves pins when the residual is filtered", () => {
    const mesh = smallMesh();
    const n = mesh.count;
    const mat = { ...DEFAULT_MATERIAL };
    const x = Float64Array.from(mesh.positions);
    const invH2 = 3600;
    const beta = Math.max(mat.stretchWarp, mat.stretchWeft, mat.shear) * mat.thickness * 0.1 + 1e-6;
    const diag = jacobiDiag(n, mesh.masses, invH2, beta);
    const f = buildBlockFactors(x, mesh, mat, diag);
    const scene = createScene(mesh, mat);
    scene.pinned.set(0, [x[0], x[1], x[2]]);
    scene.pinned.set(5, [x[15], x[16], x[17]]);
    const filter = makePinFilter(scene.pinned);
    const r = new Float64Array(n * 3);
    for (let i = 0; i < r.length; i++) r[i] = Math.sin(i * 3.7) * 2;
    filter(r);
    const z = new Float64Array(n * 3);
    applyBlockFactors(f, r, z, n);
    filter(z);
    expect(z[0]).toBe(0); expect(z[1]).toBe(0); expect(z[2]).toBe(0);
    expect(z[15]).toBe(0); expect(z[16]).toBe(0); expect(z[17]).toBe(0);
  });

  it("block-PCG converges in <= Jacobi-PCG iterations on a stiff patch", () => {
    const mesh = smallMesh();
    const n = mesh.count;
    const n3 = n * 3;
    // Stiff membrane + small shear deformation (Hessian PD-ish here).
    const mat = {
      ...DEFAULT_MATERIAL,
      stretchWarp: DEFAULT_MATERIAL.stretchWarp * 10,
      stretchWeft: DEFAULT_MATERIAL.stretchWeft * 10,
      shear: DEFAULT_MATERIAL.shear * 10,
    };
    const x = Float64Array.from(mesh.positions);
    for (let i = 0; i < n; i++) {
      x[i * 3] += 0.003 * Math.sin(i * 1.7);
      x[i * 3 + 1] += 0.001 * Math.cos(i * 2.3);
    }
    const invH2 = 3600;
    const beta = Math.max(mat.stretchWarp, mat.stretchWeft, mat.shear) * mat.thickness * 0.1 + 1e-6;
    const diag = new Float64Array(n3);
    for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) diag[i * 3 + k] = mesh.masses[i] * invH2 + beta;
    const hvpFull = (p: Float64Array, out: Float64Array) => {
      const Hp = membraneHvp(Float64Array.from(x), p, mesh, mat);
      for (let i = 0; i < n3; i++) out[i] = mesh.masses[Math.floor(i / 3)] * p[i] * invH2 + Hp[i];
    };
    const b = new Float64Array(n3);
    for (let i = 0; i < n3; i++) b[i] = Math.sin(i * 0.913) * 0.5;
    const j = pcg(b, hvpFull, diag, { maxIters: 60, tol: 1e-6 });
    const f = buildBlockFactors(x, mesh, mat, diag);
    const bj = pcg(b, hvpFull, diag, {
      maxIters: 60, tol: 1e-6,
      applyPreconditioner: (r, z) => { applyBlockFactors(f, r, z, n); },
    });
    expect(Number.isFinite(bj.residual)).toBe(true);
    expect(bj.residual).toBeLessThanOrEqual(j.residual * 1.001 + 1e-12);
    expect(bj.iters).toBeLessThanOrEqual(j.iters);
  });

  it("apply matches a direct per-vertex solve", () => {
    const mesh = smallMesh();
    const n = mesh.count;
    const mat = { ...DEFAULT_MATERIAL };
    const x = Float64Array.from(mesh.positions);
    const invH2 = 3600;
    const beta = 1e-6 + Math.max(mat.stretchWarp, mat.stretchWeft, mat.shear) * mat.thickness * 0.1;
    const diag = jacobiDiag(n, mesh.masses, invH2, beta);
    const f = buildBlockFactors(x, mesh, mat, diag);
    const r = new Float64Array(n * 3);
    for (let i = 0; i < r.length; i++) r[i] = Math.cos(i * 1.31) * 3;
    const z = new Float64Array(n * 3);
    applyBlockFactors(f, r, z, n);
    // Reference: row-major 3x3 matvec per vertex.
    for (let v = 0; v < n; v++) {
      for (let a = 0; a < 3; a++) {
        let s = 0;
        for (let b = 0; b < 3; b++) s += f.inv[v * 9 + a * 3 + b] * r[v * 3 + b];
        expect(z[v * 3 + a]).toBe(s);
      }
    }
  });
});
