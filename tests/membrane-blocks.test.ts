// G5B analytic 9x9 membrane Hessian blocks: symmetry, FD oracle parity,
// HVP consistency, rest-state nullspace. Deterministic (mulberry32).
import { describe, it, expect } from "vitest";
import { triangleHessian9, evalMembraneBlocks } from "../src/physics/membrane-blocks.js";
import { triangleHvp } from "../src/physics/membrane-hvp.js";
import { triangleEnergyGradient } from "../src/physics/membrane.js";
import { evalMembrane } from "../src/physics/fem.js";
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

function randTri(rng: () => number, strain: number) {
  const L1 = 0.01 + rng() * 0.09;
  const L2 = 0.01 + rng() * 0.09;
  const dx = () => (rng() * 2 - 1) * strain * 0.05;
  return {
    x: [[dx(), dx(), dx()], [L1 + dx(), dx(), dx()], [dx(), L2 + dx(), dx()]],
    inv: [1 / L1, 0, 0, 1 / L2],
    area: 0.5 * L1 * L2,
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

describe("G5B analytic membrane Hessian blocks", () => {
  it("is symmetric on randomized deformed elements", () => {
    const rng = mulberry32(0xB10C);
    for (let k = 0; k < 50; k++) {
      const { x, inv, area } = randTri(rng, 0.5);
      const { h } = triangleHessian9(x[0], x[1], x[2], inv, area, randMat(rng));
      let num = 0, den = 0;
      for (let a = 0; a < 9; a++) {
        for (let b = 0; b < 9; b++) {
          const d = h[a * 9 + b] - h[b * 9 + a];
          num += d * d; den += h[a * 9 + b] * h[a * 9 + b];
        }
      }
      expect(Math.sqrt(num / Math.max(den, 1e-300))).toBeLessThan(1e-14);
    }
  });

  it("matches the FD oracle column by column", () => {
    const rng = mulberry32(0xFD01);
    let worst = 0;
    for (let k = 0; k < 30; k++) {
      const { x, inv, area } = randTri(rng, k % 3 === 2 ? 1.0 : 0.4);
      const mat = randMat(rng);
      const { h } = triangleHessian9(x[0], x[1], x[2], inv, area, mat);
      // FD column B: central difference of the analytic gradient along e_B
      const t = 1e-7;
      const X = [...x[0], ...x[1], ...x[2]];
      for (let B = 0; B < 9; B++) {
        const Xp = [...X]; Xp[B] += t;
        const Xm = [...X]; Xm[B] -= t;
        const gp = triangleEnergyGradient(Xp.slice(0, 3), Xp.slice(3, 6), Xp.slice(6, 9), inv, area, mat).grad;
        const gm = triangleEnergyGradient(Xm.slice(0, 3), Xm.slice(3, 6), Xm.slice(6, 9), inv, area, mat).grad;
        for (let A = 0; A < 9; A++) {
          const fd = (gp[A] - gm[A]) / (2 * t);
          const an = h[A * 9 + B];
          const scale = Math.max(Math.abs(an), 1e-9);
          expect(Math.abs(fd - an) / scale).toBeLessThan(1e-4);
          worst = Math.max(worst, Math.abs(fd - an) / scale);
        }
      }
    }
    // eslint-disable-next-line no-console
    console.log(`[g5b] worst analytic-vs-FD entry rel err over 30 tris: ${worst.toExponential(2)}`);
  });

  it("contracts to the analytic HVP (H v == triangleHvp)", () => {
    const rng = mulberry32(0xC7AC);
    for (let k = 0; k < 50; k++) {
      const { x, inv, area } = randTri(rng, 0.5);
      const mat = randMat(rng);
      const v = new Float64Array(9);
      for (let i = 0; i < 9; i++) v[i] = rng() * 2 - 1;
      const { h } = triangleHessian9(x[0], x[1], x[2], inv, area, mat);
      const Hv = new Float64Array(9);
      for (let A = 0; A < 9; A++) {
        for (let B = 0; B < 9; B++) Hv[A] += h[A * 9 + B] * v[B];
      }
      const hv = triangleHvp(x[0], x[1], x[2], [v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], v[8]], inv, area, mat).out;
      expect(relErr(Hv, hv)).toBeLessThan(1e-12);
    }
  });

  it("has the 6-dim rigid nullspace at rest", () => {
    const mat = randMat(mulberry32(31));
    const x0 = [0, 0, 0], x1 = [0.05, 0, 0], x2 = [0, 0.04, 0];
    const { h } = triangleHessian9(x0, x1, x2, [20, 0, 0, 25], 0.001, mat);
    const modes: number[][] = [
      [1, 0, 0, 1, 0, 0, 1, 0, 0],
      [0, 1, 0, 0, 1, 0, 0, 1, 0],
      [0, 0, 1, 0, 0, 1, 0, 0, 1],
    ];
    // infinitesimal rotations about each axis through the origin
    const axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    for (const w of axes) {
      const m: number[] = [];
      for (const x of [x0, x1, x2]) {
        m.push(w[1] * x[2] - w[2] * x[1], w[2] * x[0] - w[0] * x[2], w[0] * x[1] - w[1] * x[0]);
      }
      modes.push(m);
    }
    let hmax = 0;
    for (const v of h) hmax = Math.max(hmax, Math.abs(v));
    for (const m of modes) {
      let num = 0;
      for (let A = 0; A < 9; A++) {
        let s = 0;
        for (let B = 0; B < 9; B++) s += h[A * 9 + B] * m[B];
        num += s * s;
      }
      expect(Math.sqrt(num) / Math.max(hmax, 1e-300)).toBeLessThan(1e-9);
    }
  });

  it("mesh blocks match FD-assembled diagonal blocks", () => {
    const rng = mulberry32(0xA53B);
    const g = buildGrid(3, 3, 0.06, 0.06);
    const mesh = preprocess(g.positions, g.uv, g.indices, 0.15);
    const n = mesh.count;
    const mat = randMat(rng);
    const x = Float64Array.from(mesh.positions);
    for (let i = 0; i < x.length; i++) x[i] += (rng() * 2 - 1) * 0.004;
    const blocks = evalMembraneBlocks(x, mesh, mat);
    // FD diagonal block per vertex: perturb x_v, read gradient change at v
    const t = 1e-7;
    for (let v = 0; v < n; v++) {
      for (let b = 0; b < 3; b++) {
        const xp = Float64Array.from(x); xp[v * 3 + b] += t;
        const xm = Float64Array.from(x); xm[v * 3 + b] -= t;
        const gp = evalMembrane(xp, mesh, mat).grad;
        const gm = evalMembrane(xm, mesh, mat).grad;
        for (let a = 0; a < 3; a++) {
          const fd = (gp[v * 3 + a] - gm[v * 3 + a]) / (2 * t);
          const an = blocks[v * 9 + a * 3 + b];
          expect(Math.abs(fd - an) / Math.max(Math.abs(an), 1e-6)).toBeLessThan(1e-4);
        }
      }
    }
  });
});
