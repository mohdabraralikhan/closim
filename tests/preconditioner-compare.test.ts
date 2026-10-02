// G5 measurement + regime comparison (CPU, fast):
// stiffness x contact cells, one representative Newton linear system each,
// solved with Jacobi / Block-Jacobi / Schwarz-1 / MAS-2 to fixed tolerance.
// Logs the comparison table (PCG iters, factor/apply/HVP time); asserts the
// hierarchy block/schwarz/mas <= jacobi on PD cells and finiteness everywhere.
// The exact HVP operator is identical for all methods (preconditioner-only).
import { describe, it, expect } from "vitest";
import { buildGrid, preprocess } from "../src/mesh/mesh.js";
import { DEFAULT_MATERIAL, type ClothMaterial } from "../src/physics/types.js";
import { ContactSystem } from "../src/collision/contact-assembly.js";
import { DEFAULT_CONTACT_PARAMS } from "../src/collision/types.js";
import { membraneHvp } from "../src/physics/fem.js";
import { pcg } from "../src/math/pcg.js";
import { buildBlockFactors, applyBlockFactors } from "../src/solver/block-jacobi.js";
import {
  buildSchwarzDomains, buildSchwarzFactors, applySchwarz,
} from "../src/solver/schwarz.js";
import {
  buildMasTwoLevel, applyMasTwoLevel, buildCoarseExact, solveCoarseExact,
  restrictCoarse, prolongateAdd, buildCoarseDiag,
} from "../src/solver/mas.js";
import {
  buildCoarsePattern, assembleCoarseValues, coarsePcg, applyCoarseBlockJacobi,
} from "../src/solver/coarse-csr.js";

const INV_H2 = 3600;

function scaled(mat: ClothMaterial, memK: number, bendK: number): ClothMaterial {
  return {
    ...mat,
    stretchWarp: mat.stretchWarp * memK,
    stretchWeft: mat.stretchWeft * memK,
    stretchCoupling: mat.stretchCoupling * memK,
    shear: mat.shear * memK,
    bendWarp: mat.bendWarp * bendK,
    bendWeft: mat.bendWeft * bendK,
  };
}

function jacobiDiag(n: number, mass: ArrayLike<number>, beta: number): Float64Array {
  const d = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) d[i * 3 + k] = mass[i] * INV_H2 + beta;
  return d;
}

function betaOf(mat: ClothMaterial): number {
  return Math.max(mat.stretchWarp, mat.stretchWeft, mat.shear) * mat.thickness * 0.1 + 1e-6;
}

interface Cell {
  stiffness: string;
  contact: string;
  method: string;
  iters: number;
  residual: number;
  factorMs: number;
  applyMsPerIter: number;
  hvpMsPerIter: number;
}

function buildCell(stiff: { label: string; memK: number; bendK: number }, contact: "none" | "floor"): {
  x: Float64Array; mesh: ReturnType<typeof preprocess>; mat: ClothMaterial;
  diag: Float64Array; contactSys: ContactSystem | null;
  hvpFull: (p: Float64Array, out: Float64Array) => void; b: Float64Array;
} {
  const g = buildGrid(6, 6, 0.12, 0.12);
  const mat = scaled(DEFAULT_MATERIAL, stiff.memK, stiff.bendK);
  const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
  const n = mesh.count;
  const n3 = n * 3;
  // Tensile state (PD-ish local systems): 2% x-stretch + sub-element noise.
  const x = Float64Array.from(mesh.positions);
  for (let i = 0; i < n; i++) {
    x[i * 3] *= 1.02;
    x[i * 3 + 1] += Math.sin(i * 12.9898) * 2e-4;
  }
  let contactSys: ContactSystem | null = null;
  const beta = betaOf(mat);
  const diag = jacobiDiag(n, mesh.masses, beta);
  if (contact === "floor") {
    // Rest the cloth inside the barrier zone (dHat = 2mm) so floor contacts
    // are active and stiffen the system exactly like a Newton iterate would.
    for (let i = 0; i < n; i++) x[i * 3 + 1] = 0.001;
    contactSys = new ContactSystem({ ...DEFAULT_CONTACT_PARAMS }, mesh.indices);
    contactSys.setFloor(0);
    contactSys.beginStep(Float64Array.from(x));
    contactSys.updateActiveSet(x);
    contactSys.addDiagEstimate(diag, x, mesh.masses, INV_H2, beta);
  }
  const cHp = new Float64Array(n3);
  const hvpFull = (p: Float64Array, out: Float64Array) => {
    const Hp = membraneHvp(Float64Array.from(x), p, mesh, mat);
    for (let i = 0; i < n3; i++) out[i] = mesh.masses[Math.floor(i / 3)] * p[i] * INV_H2 + Hp[i];
    if (contactSys) {
      contactSys.applyHvp(x, p, cHp);
      for (let i = 0; i < n3; i++) out[i] += cHp[i];
    }
  };
  const b = new Float64Array(n3);
  for (let i = 0; i < n3; i++) b[i] = Math.sin(i * 0.913) * 0.5;
  return { x, mesh, mat, diag, contactSys, hvpFull, b };
}

const STIFF = [
  { label: "1x", memK: 1, bendK: 1 },
  { label: "2x", memK: 2, bendK: 1 },
  { label: "5x", memK: 5, bendK: 1 },
  { label: "10x", memK: 10, bendK: 1 },
  { label: "10xbend", memK: 1, bendK: 10 },
];

describe("G5 preconditioner comparison", () => {
  it("reports Jacobi / BlockJ / Schwarz-1 / MAS-2 across stiffness x contact", () => {
    const rows: Cell[] = [];
    for (const st of STIFF) {
      for (const contact of ["none", "floor"] as const) {
        const { x, mesh, mat, diag, hvpFull, b } = buildCell(st, contact);
        const n = mesh.count;
        const n3 = n * 3;
        const dom = buildSchwarzDomains(mesh, mesh.restPositions, 16);
        // Time one HVP (operator cost, method-independent).
        const p0 = new Float64Array(n3).fill(0.1);
        const o0 = new Float64Array(n3);
        let t = performance.now();
        for (let k = 0; k < 5; k++) hvpFull(p0, o0);
        const hvpMs = (performance.now() - t) / 5;
        const methods: Array<{ name: string; build: () => { factorMs: number; apply: (r: Float64Array, z: Float64Array) => void } }> = [
          {
            name: "jacobi",
            build: () => ({ factorMs: 0, apply: (r, z) => { for (let i = 0; i < n3; i++) z[i] = diag[i] > 1e-12 ? r[i] / diag[i] : r[i]; } }),
          },
          {
            name: "block",
            build: () => {
              const t0 = performance.now();
              const f = buildBlockFactors(x, mesh, mat, diag);
              const factorMs = performance.now() - t0;
              return { factorMs, apply: (r: Float64Array, z: Float64Array) => { applyBlockFactors(f, r, z, n); } };
            },
          },
          {
            name: "schwarz1",
            build: () => {
              const t0 = performance.now();
              const f = buildSchwarzFactors(x, mesh, mat, diag, dom);
              const factorMs = performance.now() - t0;
              return { factorMs, apply: (r: Float64Array, z: Float64Array) => { applySchwarz(f, r, z, n); } };
            },
          },
          {
            name: "mas2",
            build: () => {
              const t0 = performance.now();
              const m = buildMasTwoLevel(x, mesh, mat, diag, dom, 0.5);
              const factorMs = performance.now() - t0;
              return { factorMs, apply: (r: Float64Array, z: Float64Array) => { applyMasTwoLevel(m, r, z, n); } };
            },
          },
          {
            name: "masX",
            build: () => {
              const t0 = performance.now();
              const fine = buildSchwarzFactors(x, mesh, mat, diag, dom);
              const ce = buildCoarseExact(x, mesh, mat, diag, dom);
              const factorMs = performance.now() - t0;
              const D = dom.members.length;
              const rc = new Float64Array(D * 3);
              const cc = new Float64Array(D * 3);
              return {
                factorMs,
                apply: (r: Float64Array, z: Float64Array) => {
                  applySchwarz(fine, r, z, n);
                  restrictCoarse(r, dom, rc);
                  solveCoarseExact(ce, rc, cc);
                  prolongateAdd(cc, dom, z);
                },
              };
            },
          },
          {
            name: "c0",
            build: () => {
              const t0 = performance.now();
              const fine = buildSchwarzFactors(x, mesh, mat, diag, dom);
              const pat = buildCoarsePattern(mesh, dom);
              const csr = assembleCoarseValues(x, mesh, mat, diag, dom, pat);
              const cd = buildCoarseDiag(x, mesh, mat, diag, dom);
              const factorMs = performance.now() - t0;
              const D = dom.members.length;
              const rc = new Float64Array(D * 3);
              const cc = new Float64Array(D * 3);
              return {
                factorMs,
                apply: (r: Float64Array, z: Float64Array) => {
                  applySchwarz(fine, r, z, n);
                  restrictCoarse(r, dom, rc);
                  applyCoarseBlockJacobi(csr, cd, rc, cc);
                  prolongateAdd(cc, dom, z);
                },
              };
            },
          },
          ...[4, 8, 32].map((K) => ({
            name: `c1-${K}`,
            build: () => {
              const t0 = performance.now();
              const fine = buildSchwarzFactors(x, mesh, mat, diag, dom);
              const pat = buildCoarsePattern(mesh, dom);
              const csr = assembleCoarseValues(x, mesh, mat, diag, dom, pat);
              const cd = buildCoarseDiag(x, mesh, mat, diag, dom);
              const factorMs = performance.now() - t0;
              const D = dom.members.length;
              const rc = new Float64Array(D * 3);
              return {
                factorMs,
                apply: (r: Float64Array, z: Float64Array) => {
                  applySchwarz(fine, r, z, n);
                  restrictCoarse(r, dom, rc);
                  const solved = coarsePcg(csr, cd, Float64Array.from(rc), K, 1e-12);
                  prolongateAdd(solved.x, dom, z);
                },
              };
            },
          })),
        ];
        for (const m of methods) {
          const { factorMs, apply } = m.build();
          // Time the apply path.
          const ra = new Float64Array(n3).fill(0.3);
          const za = new Float64Array(n3);
          let ta = performance.now();
          for (let k = 0; k < 20; k++) apply(ra, za);
          const applyMs = (performance.now() - ta) / 20;
          const res = pcg(b, hvpFull, diag, {
            maxIters: 200, tol: 1e-6,
            applyPreconditioner: m.name === "jacobi" ? undefined : apply,
          });
          rows.push({
            stiffness: st.label, contact, method: m.name,
            iters: res.iters, residual: res.residual,
            factorMs, applyMsPerIter: applyMs, hvpMsPerIter: hvpMs,
          });
        }
      }
    }
    // eslint-disable-next-line no-console
    console.log("[g5-compare] stiffness/contact/method: iters, residual, factorMs, applyMs/iter, hvpMs/iter");
    for (const r of rows) {
      // eslint-disable-next-line no-console
      console.log(`[g5-compare] ${r.stiffness}/${r.contact}/${r.method}: ` +
        `iters=${r.iters} res=${r.residual.toExponential(2)} ` +
        `factor=${r.factorMs.toFixed(2)}ms apply=${r.applyMsPerIter.toFixed(3)}ms hvp=${r.hvpMsPerIter.toFixed(3)}ms`);
    }
    // Hierarchy on PD cells: richer preconditioners never lose to Jacobi.
    // (10xbend is Hessian-identical to 1x — bending lives in the gradient only.)
    // masX (exact coarse) tracks schwarz1 within PCG noise (±2 iters): on
    // free-flight cells it wins outright (coarse space captures the soft
    // rigid-like modes); floor-barrier springs localize the system and the
    // coarse level adds little — both facts are the G5 regime evidence.
    for (const st of STIFF) {
      for (const contact of ["none", "floor"]) {
        const it = (m: string) => rows.find((r) => r.stiffness === st.label && r.contact === contact && r.method === m)!.iters;
        const re = (m: string) => rows.find((r) => r.stiffness === st.label && r.contact === contact && r.method === m)!.residual;
        for (const m of ["block", "schwarz1", "mas2", "masX", "c0", "c1-4", "c1-8", "c1-32"]) {
          expect(re(m), `${st.label}/${contact}/${m} finite`).toBeLessThan(1e-2);
          expect(it(m), `${st.label}/${contact}/${m} <= jacobi`).toBeLessThanOrEqual(it("jacobi"));
        }
        expect(it("masX"), `${st.label}/${contact}/masX ~= schwarz1`).toBeLessThanOrEqual(it("schwarz1") + 2);
        if (contact === "none") {
          expect(it("masX"), `${st.label}/none/masX <= schwarz1`).toBeLessThanOrEqual(it("schwarz1"));
        }
      }
    }
  }, 180000);
});
