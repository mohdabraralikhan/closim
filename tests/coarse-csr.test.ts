// G5.5 block-CSR coarse system tests (CPU reference):
// pattern validity, values vs dense Ac, symmetry (incl. randomized bilinear),
// SpMV vs dense, coarse-PCG vs Cholesky oracle, fallback, determinism.
import { describe, it, expect } from "vitest";
import { buildGrid, preprocess } from "../src/mesh/mesh.js";
import { DEFAULT_MATERIAL } from "../src/physics/types.js";
import { buildSchwarzDomains } from "../src/solver/schwarz.js";
import { buildCoarseExact, solveCoarseExact, buildCoarseDiag } from "../src/solver/mas.js";
import {
  buildCoarsePattern, blockPos, assembleCoarseValues, coarseSpmv,
  invertSym33, coarsePcg, bilinearAsymmetry,
} from "../src/solver/coarse-csr.js";

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function mesh66() {
  const g = buildGrid(6, 6, 0.12, 0.12);
  return preprocess(g.positions, g.uv, g.indices, 0.15);
}

function tenseState(mesh: ReturnType<typeof preprocess>, n: number): Float64Array {
  const x = Float64Array.from(mesh.positions);
  for (let i = 0; i < n; i++) {
    x[i * 3] *= 1.02;
    x[i * 3 + 1] += Math.sin(i * 12.9898) * 2e-4;
  }
  return x;
}

function jacobiDiag(n: number, mass: ArrayLike<number>, beta: number): Float64Array {
  const d = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) d[i * 3 + k] = mass[i] * 3600 + beta;
  return d;
}

describe("G5.5 coarse block-CSR pattern", () => {
  it("covers self blocks, is sorted, matches brute-force adjacency", () => {
    const mesh = mesh66();
    const dom = buildSchwarzDomains(mesh, mesh.restPositions, 16);
    const pat = buildCoarsePattern(mesh, dom);
    const D = dom.members.length;
    expect(pat.domains).toBe(D);
    expect(pat.rowOffsets.length).toBe(D + 1);
    expect(pat.rowOffsets[D]).toBe(pat.nnz);
    // brute force: independent re-derivation
    const brute: Set<number>[] = Array.from({ length: D }, () => new Set<number>());
    for (let t = 0; t < mesh.triCount; t++) {
      const ds = [dom.domainOf[mesh.indices[t * 3]], dom.domainOf[mesh.indices[t * 3 + 1]], dom.domainOf[mesh.indices[t * 3 + 2]]];
      for (const a of ds) for (const b of ds) brute[a].add(b);
    }
    for (let d = 0; d < D; d++) {
      const row: number[] = [];
      for (let p = pat.rowOffsets[d]; p < pat.rowOffsets[d + 1]; p++) row.push(pat.colIndices[p]);
      expect(row).toEqual([...brute[d]].sort((x, y) => x - y));
      expect(row).toContain(d); // self block always present
      expect(blockPos(pat, d, d)).toBeGreaterThanOrEqual(0);
    }
  });

  it("is deterministic across rebuilds", () => {
    const mesh = mesh66();
    const dom = buildSchwarzDomains(mesh, mesh.restPositions, 16);
    const a = buildCoarsePattern(mesh, dom);
    const b = buildCoarsePattern(mesh, dom);
    expect([...a.rowOffsets]).toEqual([...b.rowOffsets]);
    expect([...a.colIndices]).toEqual([...b.colIndices]);
  });
});

describe("G5.5 coarse values + symmetry", () => {
  it("blocks match the dense-Ac oracle block for block", () => {
    const g = buildGrid(3, 3, 0.06, 0.06);
    const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
    const n = mesh.count;
    const mat = { ...DEFAULT_MATERIAL };
    const x = tenseState(mesh, n);
    const beta = Math.max(mat.stretchWarp, mat.stretchWeft, mat.shear) * mat.thickness * 0.1 + 1e-6;
    const diag = jacobiDiag(n, mesh.masses, beta);
    const dom = buildSchwarzDomains(mesh, mesh.restPositions, 8);
    const pat = buildCoarsePattern(mesh, dom);
    const csr = assembleCoarseValues(x, mesh, mat, diag, dom, pat);
    const ce = buildCoarseExact(x, mesh, mat, diag, dom);
    const dc = dom.members.length * 3;
    for (let d1 = 0; d1 < dom.members.length; d1++) {
      for (let p = pat.rowOffsets[d1]; p < pat.rowOffsets[d1 + 1]; p++) {
        const d2 = pat.colIndices[p];
        for (let a = 0; a < 3; a++) {
          for (let b = 0; b < 3; b++) {
            expect(csr.blocks[p * 9 + a * 3 + b]).toBeCloseTo(ce.ac[(d1 * 3 + a) * dc + (d2 * 3 + b)], 9);
          }
        }
      }
    }
  });

  it("diagonal blocks carry the true-Ac diagonal on their diagonal", () => {
    const mesh = mesh66();
    const n = mesh.count;
    const mat = { ...DEFAULT_MATERIAL };
    const x = tenseState(mesh, n);
    const beta = Math.max(mat.stretchWarp, mat.stretchWeft, mat.shear) * mat.thickness * 0.1 + 1e-6;
    const diag = jacobiDiag(n, mesh.masses, beta);
    const dom = buildSchwarzDomains(mesh, mesh.restPositions, 16);
    const pat = buildCoarsePattern(mesh, dom);
    const csr = assembleCoarseValues(x, mesh, mat, diag, dom, pat);
    const cd = buildCoarseDiag(x, mesh, mat, diag, dom);
    for (let d = 0; d < dom.members.length; d++) {
      const p = blockPos(pat, d, d);
      for (let a = 0; a < 3; a++) {
        expect(csr.blocks[p * 9 + a * 3 + a]).toBeCloseTo(cd[d * 3 + a], 9);
      }
    }
  });

  it("bilinear symmetry holds for randomized u,v", () => {
    const rng = mulberry32(0xC0A55E);
    const mesh = mesh66();
    const n = mesh.count;
    const mat = { ...DEFAULT_MATERIAL };
    const x = tenseState(mesh, n);
    const beta = Math.max(mat.stretchWarp, mat.stretchWeft, mat.shear) * mat.thickness * 0.1 + 1e-6;
    const diag = jacobiDiag(n, mesh.masses, beta);
    const dom = buildSchwarzDomains(mesh, mesh.restPositions, 16);
    const pat = buildCoarsePattern(mesh, dom);
    const csr = assembleCoarseValues(x, mesh, mat, diag, dom, pat);
    const dc = dom.members.length * 3;
    for (let k = 0; k < 20; k++) {
      const u = new Float64Array(dc);
      const v = new Float64Array(dc);
      for (let i = 0; i < dc; i++) {
        u[i] = rng() * 2 - 1;
        v[i] = rng() * 2 - 1;
      }
      const { uv, vu, diff } = bilinearAsymmetry(csr, u, v);
      const scale = Math.max(Math.abs(uv), Math.abs(vu), 1e-300);
      expect(Math.abs(diff) / scale).toBeLessThan(1e-12);
    }
  });

  it("SpMV matches the dense oracle matvec", () => {
    const g = buildGrid(3, 3, 0.06, 0.06);
    const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
    const n = mesh.count;
    const mat = { ...DEFAULT_MATERIAL };
    const x = tenseState(mesh, n);
    const beta = Math.max(mat.stretchWarp, mat.stretchWeft, mat.shear) * mat.thickness * 0.1 + 1e-6;
    const diag = jacobiDiag(n, mesh.masses, beta);
    const dom = buildSchwarzDomains(mesh, mesh.restPositions, 8);
    const pat = buildCoarsePattern(mesh, dom);
    const csr = assembleCoarseValues(x, mesh, mat, diag, dom, pat);
    const ce = buildCoarseExact(x, mesh, mat, diag, dom);
    const dc = dom.members.length * 3;
    const v = new Float64Array(dc);
    for (let i = 0; i < dc; i++) v[i] = Math.cos(i * 1.31) * 2;
    const yS = new Float64Array(dc);
    coarseSpmv(csr, v, yS);
    for (let i = 0; i < dc; i++) {
      let s = 0;
      for (let j = 0; j < dc; j++) s += ce.ac[i * dc + j] * v[j];
      expect(yS[i]).toBeCloseTo(s, 9);
    }
  });
});

describe("G5.5 coarse solve", () => {
  it("invertSym33 inverts PD blocks and falls back on singular ones", () => {
    const { inv, ok } = invertSym33(4, 1, 0, 5, 2, 6, 1, 1, 1);
    expect(ok).toBe(true);
    // B * inv = I
    const B = [4, 1, 0, 1, 5, 2, 0, 2, 6];
    for (let a = 0; a < 3; a++) {
      for (let b = 0; b < 3; b++) {
        let s = 0;
        for (let k = 0; k < 3; k++) s += B[a * 3 + k] * inv[k * 3 + b];
        expect(Math.abs(s - (a === b ? 1 : 0))).toBeLessThan(1e-12);
      }
    }
    const sing = invertSym33(0, 0, 0, 0, 0, 0, 2, 0, 4);
    expect(sing.ok).toBe(false);
    expect(sing.inv[0]).toBe(0.5);
    expect(sing.inv[4]).toBe(0);
    expect(sing.inv[8]).toBe(0.25);
    for (const e of sing.inv) expect(Number.isFinite(e)).toBe(true);
  });

  it("coarse-PCG matches the Cholesky oracle on a tensile patch", () => {
    const g = buildGrid(3, 3, 0.06, 0.06);
    const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
    const n = mesh.count;
    const mat = { ...DEFAULT_MATERIAL };
    const x = tenseState(mesh, n);
    const beta = Math.max(mat.stretchWarp, mat.stretchWeft, mat.shear) * mat.thickness * 0.1 + 1e-6;
    const diag = jacobiDiag(n, mesh.masses, beta);
    const dom = buildSchwarzDomains(mesh, mesh.restPositions, 8);
    const pat = buildCoarsePattern(mesh, dom);
    const csr = assembleCoarseValues(x, mesh, mat, diag, dom, pat);
    const cd = buildCoarseDiag(x, mesh, mat, diag, dom);
    const ce = buildCoarseExact(x, mesh, mat, diag, dom);
    expect(ce.flag).toBe(1);
    const dc = dom.members.length * 3;
    const b = new Float64Array(dc);
    for (let i = 0; i < dc; i++) b[i] = Math.sin(i * 0.913) * 0.5;
    const ref = new Float64Array(dc);
    solveCoarseExact(ce, b, ref);
    for (const K of [4, 8, 16, 32]) {
      const got = coarsePcg(csr, cd, b, K);
      let num = 0, den = 0;
      for (let i = 0; i < dc; i++) {
        num += (got.x[i] - ref[i]) * (got.x[i] - ref[i]);
        den += ref[i] * ref[i];
      }
      // More iterations must not diverge from the oracle.
      expect(Math.sqrt(num / Math.max(den, 1e-300))).toBeLessThan(0.5);
      expect(Number.isFinite(got.residual)).toBe(true);
    }
    const full = coarsePcg(csr, cd, b, 60);
    expect(full.residual).toBeLessThan(1e-6 * Math.max(1, 0));
  });

  it("stays finite on a collapsed state", () => {
    const mesh = mesh66();
    const n = mesh.count;
    const mat = { ...DEFAULT_MATERIAL };
    const x = new Float64Array(n * 3);
    const diag = jacobiDiag(n, mesh.masses, 1e-6);
    const dom = buildSchwarzDomains(mesh, mesh.restPositions, 16);
    const pat = buildCoarsePattern(mesh, dom);
    const csr = assembleCoarseValues(x, mesh, mat, diag, dom, pat);
    for (const e of csr.blocks) expect(Number.isFinite(e)).toBe(true);
    const dc = dom.members.length * 3;
    const b = new Float64Array(dc).fill(0.25);
    const cd = buildCoarseDiag(x, mesh, mat, diag, dom);
    const got = coarsePcg(csr, cd, b, 8);
    for (const e of got.x) expect(Number.isFinite(e)).toBe(true);
    const y = new Float64Array(dc);
    coarseSpmv(csr, b, y);
    for (const e of y) expect(Number.isFinite(e)).toBe(true);
  });
});
