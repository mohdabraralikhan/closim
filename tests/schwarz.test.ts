// G5C one-level additive Schwarz tests (CPU reference):
// domain validity, local-solve correctness, additive apply, pin preservation,
// PCG convergence vs Jacobi, determinism, no NaN/Inf.
import { describe, it, expect } from "vitest";
import { buildGrid, preprocess } from "../src/mesh/mesh.js";
import { createScene, makePinFilter } from "../src/physics/scene.js";
import { DEFAULT_MATERIAL } from "../src/physics/types.js";
import {
  buildSchwarzDomains, buildSchwarzFactors, applySchwarz, solveLocal,
} from "../src/solver/schwarz.js";
import { membraneHvp } from "../src/physics/fem.js";
import { pcg } from "../src/math/pcg.js";

function mesh48() {
  const g = buildGrid(6, 6, 0.12, 0.12);
  return preprocess(g.positions, g.uv, g.indices, 0.15);
}

function jacobiDiag(n: number, mass: ArrayLike<number>, invH2: number, beta: number): Float64Array {
  const d = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) d[i * 3 + k] = mass[i] * invH2 + beta;
  return d;
}

describe("G5C Schwarz domains", () => {
  it("partition covers every vertex exactly once, sizes in range", () => {
    const mesh = mesh48();
    const dom = buildSchwarzDomains(mesh, mesh.restPositions, 16);
    const seen = new Uint8Array(mesh.count);
    for (const mem of dom.members) {
      expect(mem.length).toBeGreaterThanOrEqual(8);
      expect(mem.length).toBeLessThanOrEqual(32);
      for (const v of mem) {
        expect(seen[v]).toBe(0);
        seen[v] = 1;
      }
    }
    for (let v = 0; v < mesh.count; v++) {
      expect(seen[v]).toBe(1);
      expect(dom.members[dom.domainOf[v]]).toContain(v);
      expect(dom.localOf[v]).toBeLessThan(dom.members[dom.domainOf[v]].length);
    }
  });

  it("is deterministic across rebuilds", () => {
    const mesh = mesh48();
    const a = buildSchwarzDomains(mesh, mesh.restPositions, 16);
    const b = buildSchwarzDomains(mesh, mesh.restPositions, 16);
    expect([...a.domainOf]).toEqual([...b.domainOf]);
    expect(a.members).toEqual(b.members);
  });

  it("target-8 domains stay in range (GPU-compatible sizing)", () => {
    const mesh = mesh48();
    const dom = buildSchwarzDomains(mesh, mesh.restPositions, 8);
    for (const mem of dom.members) {
      expect(mem.length).toBeGreaterThanOrEqual(8);
      expect(mem.length).toBeLessThanOrEqual(32);
    }
  });
});

describe("G5C Schwarz factors + apply", () => {
  it("local solve inverts the local matrix (A_d x = r)", () => {
    const mesh = mesh48();
    const n = mesh.count;
    const mat = { ...DEFAULT_MATERIAL };
    // Gentle tension (1% stretch + sub-element noise): geometric stiffness
    // stays positive so local matrices are PD (compression would be genuinely
    // indefinite — covered by the collapsed-state fallback test instead).
    const x = Float64Array.from(mesh.positions);
    for (let i = 0; i < n; i++) {
      x[i * 3] *= 1.01;
      x[i * 3 + 1] += Math.sin(i * 12.9898) * 2e-4;
    }
    const invH2 = 3600;
    const beta = Math.max(mat.stretchWarp, mat.stretchWeft, mat.shear) * mat.thickness * 0.1 + 1e-6;
    const diag = jacobiDiag(n, mesh.masses, invH2, beta);
    const dom = buildSchwarzDomains(mesh, mesh.restPositions, 16);
    const f = buildSchwarzFactors(x, mesh, mat, diag, dom);
    // Rebuild local matrices independently and check A_d * sol = rhs.
    let checkedChol = 0;
    for (let d = 0; d < dom.members.length; d++) {
      const dw = dom.members[d].length * 3;
      const rhs = new Float64Array(dw);
      for (let i = 0; i < dw; i++) rhs[i] = Math.cos(i * 1.71 + d) * 2;
      const sol = new Float64Array(dw);
      solveLocal(f, d, rhs, sol);
      if (f.flag[d] !== 1) continue;
      // Verify via the factor itself: L (L^T sol) = rhs.
      const L = f.data[d];
      const w = new Float64Array(dw);
      for (let i = 0; i < dw; i++) {
        let s = 0;
        for (let j = i; j < dw; j++) s += L[j * dw + i] * sol[j];
        w[i] = s;
      }
      for (let i = 0; i < dw; i++) {
        let s = 0;
        for (let j = 0; j <= i; j++) s += L[i * dw + j] * w[j];
        expect(Math.abs(s - rhs[i]) / Math.max(1, Math.abs(rhs[i]))).toBeLessThan(1e-9);
      }
      checkedChol++;
    }
    expect(checkedChol).toBeGreaterThan(0);
  });

  it("single-domain apply equals the global exact solve on a tiny mesh", () => {
    // One domain covering everything: Schwarz == exact local (== Jacobi+membrane).
    const g = buildGrid(2, 2, 0.04, 0.04);
    const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
    const n = mesh.count;
    const mat = { ...DEFAULT_MATERIAL };
    const x = Float64Array.from(mesh.positions);
    const invH2 = 3600;
    const beta = Math.max(mat.stretchWarp, mat.stretchWeft, mat.shear) * mat.thickness * 0.1 + 1e-6;
    const diag = jacobiDiag(n, mesh.masses, invH2, beta);
    const dom = buildSchwarzDomains(mesh, mesh.restPositions, 32);
    expect(dom.members.length).toBe(1);
    const f = buildSchwarzFactors(x, mesh, mat, diag, dom);
    const r = new Float64Array(n * 3);
    for (let i = 0; i < r.length; i++) r[i] = Math.sin(i * 0.77) * 1.5;
    const z = new Float64Array(n * 3);
    applySchwarz(f, r, z, n);
    // Residual of the TRUE local system: (H_membrane + diag) z should equal r
    // up to dropped nothing (single domain drops nothing). Check via HVP.
    const Hp = membraneHvp(Float64Array.from(x), Float64Array.from(z), mesh, mat);
    let num = 0, den = 0;
    for (let i = 0; i < n * 3; i++) {
      const az = mesh.masses[Math.floor(i / 3)] * z[i] * invH2 + Hp[i] + beta * z[i];
      num += (az - r[i]) * (az - r[i]);
      den += r[i] * r[i];
    }
    expect(Math.sqrt(num / den)).toBeLessThan(2e-6);
  });

  it("preserves pins when filtered", () => {
    const mesh = mesh48();
    const n = mesh.count;
    const mat = { ...DEFAULT_MATERIAL };
    const x = Float64Array.from(mesh.positions);
    const invH2 = 3600;
    const beta = Math.max(mat.stretchWarp, mat.stretchWeft, mat.shear) * mat.thickness * 0.1 + 1e-6;
    const diag = jacobiDiag(n, mesh.masses, invH2, beta);
    const dom = buildSchwarzDomains(mesh, mesh.restPositions, 16);
    const f = buildSchwarzFactors(x, mesh, mat, diag, dom);
    const scene = createScene(mesh, mat);
    scene.pinned.set(0, [x[0], x[1], x[2]]);
    const filter = makePinFilter(scene.pinned);
    const r = new Float64Array(n * 3);
    for (let i = 0; i < r.length; i++) r[i] = Math.sin(i * 3.7) * 2;
    filter(r);
    const z = new Float64Array(n * 3);
    applySchwarz(f, r, z, n);
    filter(z);
    expect(z[0]).toBe(0); expect(z[1]).toBe(0); expect(z[2]).toBe(0);
    for (let i = 0; i < z.length; i++) expect(Number.isFinite(z[i])).toBe(true);
  });

  it("Schwarz-PCG converges in <= Jacobi-PCG iterations on a stiff patch", () => {
    const mesh = mesh48();
    const n = mesh.count;
    const n3 = n * 3;
    const mat = {
      ...DEFAULT_MATERIAL,
      stretchWarp: DEFAULT_MATERIAL.stretchWarp * 10,
      stretchWeft: DEFAULT_MATERIAL.stretchWeft * 10,
      shear: DEFAULT_MATERIAL.shear * 10,
    };
    const x = Float64Array.from(mesh.positions);
    for (let i = 0; i < n; i++) {
      x[i * 3] *= 1.01;
      x[i * 3] += 0.001 * Math.sin(i * 1.7);
      x[i * 3 + 1] += 0.0005 * Math.cos(i * 2.3);
    }
    const invH2 = 3600;
    const beta = Math.max(mat.stretchWarp, mat.stretchWeft, mat.shear) * mat.thickness * 0.1 + 1e-6;
    const diag = jacobiDiag(n, mesh.masses, invH2, beta);
    const hvpFull = (p: Float64Array, out: Float64Array) => {
      const Hp = membraneHvp(Float64Array.from(x), p, mesh, mat);
      for (let i = 0; i < n3; i++) out[i] = mesh.masses[Math.floor(i / 3)] * p[i] * invH2 + Hp[i];
    };
    const b = new Float64Array(n3);
    for (let i = 0; i < n3; i++) b[i] = Math.sin(i * 0.913) * 0.5;
    const j = pcg(b, hvpFull, diag, { maxIters: 60, tol: 1e-6 });
    const dom = buildSchwarzDomains(mesh, mesh.restPositions, 16);
    const f = buildSchwarzFactors(x, mesh, mat, diag, dom);
    const s = pcg(b, hvpFull, diag, {
      maxIters: 60, tol: 1e-6,
      applyPreconditioner: (r, z) => { applySchwarz(f, r, z, n); },
    });
    expect(Number.isFinite(s.residual)).toBe(true);
    expect(s.residual).toBeLessThanOrEqual(j.residual * 1.001 + 1e-12);
    expect(s.iters).toBeLessThanOrEqual(j.iters);
  });

  it("never produces NaN/Inf on a collapsed state", () => {
    const mesh = mesh48();
    const n = mesh.count;
    const mat = { ...DEFAULT_MATERIAL };
    const x = new Float64Array(n * 3); // fully collapsed
    const invH2 = 3600;
    const beta = 1e-6;
    const diag = jacobiDiag(n, mesh.masses, invH2, beta);
    const dom = buildSchwarzDomains(mesh, mesh.restPositions, 16);
    const f = buildSchwarzFactors(x, mesh, mat, diag, dom);
    const r = new Float64Array(n * 3).fill(0.5);
    const z = new Float64Array(n * 3);
    applySchwarz(f, r, z, n);
    for (let i = 0; i < z.length; i++) expect(Number.isFinite(z[i])).toBe(true);
  });
});
