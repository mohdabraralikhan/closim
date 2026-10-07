// G7A model parity: OrthotropicStVKMaterial must reproduce the legacy
// orthotropic-StVK numerics bitwise (it delegates to the exact same kernels).
// Covers the default material, randomized VALID materials (PSD-safe coupling),
// rest/stretched/sheared/jittered states, Float32 + Float64 inputs, analytic
// HVP vs the FD oracle, and out-buffer variants. Deterministic (mulberry32).
import { describe, it, expect } from "vitest";
import { buildGrid, preprocess } from "../../src/mesh/mesh.js";
import { DEFAULT_MATERIAL, type ClothMaterial } from "../../src/physics/types.js";
import { evalInternal, internalEnergyOnly, membraneHvp } from "../../src/physics/fem.js";
import { evalMembraneHvp } from "../../src/physics/membrane-hvp.js";
import {
  OrthotropicStVKMaterial,
  type PhysicalMaterialParams,
} from "../../src/physics/material-model.js";

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Randomized but always VALID physical params (coupling PSD-safe by rho). */
function randParams(rng: () => number): PhysicalMaterialParams {
  const c00 = 1e3 + rng() * 9.9e4;
  const c11 = 1e3 + rng() * 9.9e4;
  const rho = rng() * 2 - 1;
  return {
    inertia: { arealDensityKgM2: 0.05 + rng() * 0.3 },
    thicknessM: 1e-4 + rng() * 1.9e-3,
    membrane: {
      warpPa: c00,
      weftPa: c11,
      couplingPa: rho * 0.95 * Math.sqrt(c00 * c11),
      shearPa: rng() < 0.1 ? 0 : 1e3 + rng() * 1.9e4,
    },
    bending: { warpNm: rng() * 2e-5, weftNm: rng() * 2e-5 },
    dampingRatio: rng() * 0.1,
  };
}

function toLegacy(p: PhysicalMaterialParams): ClothMaterial {
  return {
    arealDensityKgM2: p.inertia.arealDensityKgM2,
    thickness: p.thicknessM,
    stretchWarp: p.membrane.warpPa,
    stretchWeft: p.membrane.weftPa,
    stretchCoupling: p.membrane.couplingPa,
    shear: p.membrane.shearPa,
    bendWarp: p.bending.warpNm,
    bendWeft: p.bending.weftNm,
    damping: p.dampingRatio,
  };
}

function testMesh(): ReturnType<typeof preprocess> {
  const g = buildGrid(3, 2, 0.06, 0.04);
  return preprocess(g.positions, g.uv, g.indices, 0.15);
}

/** Deterministic deformed states: rest, stretch, shear, jitter. */
function testStates(mesh: ReturnType<typeof preprocess>, rng: () => number): Float64Array[] {
  const base = Float64Array.from(mesh.positions);
  const n = mesh.count;
  const out: Float64Array[] = [Float64Array.from(base)];
  const stretched = Float64Array.from(base);
  for (let i = 0; i < n; i++) stretched[i * 3] *= 1.2;
  out.push(stretched);
  const sheared = Float64Array.from(base);
  for (let i = 0; i < n; i++) sheared[i * 3] += 0.3 * base[i * 3 + 1];
  out.push(sheared);
  const jittered = Float64Array.from(base);
  for (let i = 0; i < jittered.length; i++) jittered[i] += (rng() * 2 - 1) * 0.004;
  out.push(jittered);
  return out;
}

function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let d = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) d = Math.max(d, Math.abs(a[i] - b[i]));
  return d;
}

function relErr(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let num = 0, den = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    num += d * d; den += b[i] * b[i];
  }
  return Math.sqrt(num / Math.max(den, 1e-300));
}

describe("G7A orthotropic-StVK model parity", () => {
  it("default material: energy/gradient/HVP bitwise vs legacy on all states", async () => {
    const mesh = testMesh();
    const rng = mulberry32(0x6A11);
    const states = testStates(mesh, rng);
    const model = OrthotropicStVKMaterial.fromLegacy({ ...DEFAULT_MATERIAL });
    const dir = new Float64Array(mesh.count * 3);
    const rd = mulberry32(99);
    for (let i = 0; i < dir.length; i++) dir[i] = rd() * 2 - 1;
    for (let s = 0; s < states.length; s++) {
      const x = states[s];
      expect(model.energy(x, mesh), `state ${s}: energy`).toBe(internalEnergyOnly(x, mesh, DEFAULT_MATERIAL));
      const gNew = model.gradient(x, mesh);
      const gOld = evalInternal(x, mesh, DEFAULT_MATERIAL);
      expect(gNew.energy, `state ${s}: grad energy`).toBe(gOld.energy);
      expect(gNew.maxStrain, `state ${s}: maxStrain`).toBe(gOld.maxStrain);
      expect(maxAbsDiff(gNew.grad, gOld.grad), `state ${s}: gradient`).toBe(0);
      const hNew = model.hessianVector(x, dir, mesh);
      const hOld = evalMembraneHvp(x, dir, mesh, DEFAULT_MATERIAL);
      expect(maxAbsDiff(hNew, hOld), `state ${s}: analytic HVP`).toBe(0);
    }
  });

  it("randomized valid materials: bitwise parity on 40 material/state pairs", async () => {
    const mesh = testMesh();
    const rng = mulberry32(0x7A1C);
    let worstE = 0, worstG = 0, worstH = 0;
    for (let k = 0; k < 40; k++) {
      const params = randParams(rng);
      const legacy = toLegacy(params);
      const model = new OrthotropicStVKMaterial(params);
      const states = testStates(mesh, rng);
      const dir = new Float64Array(mesh.count * 3);
      for (let i = 0; i < dir.length; i++) dir[i] = rng() * 2 - 1;
      for (const x of states) {
        worstE = Math.max(worstE, Math.abs(model.energy(x, mesh) - internalEnergyOnly(x, mesh, legacy)));
        const gNew = model.gradient(x, mesh);
        const gOld = evalInternal(x, mesh, legacy);
        worstG = Math.max(worstG, maxAbsDiff(gNew.grad, gOld.grad), Math.abs(gNew.energy - gOld.energy));
        worstH = Math.max(worstH, maxAbsDiff(
          model.hessianVector(x, dir, mesh), evalMembraneHvp(x, dir, mesh, legacy)));
      }
    }
    // eslint-disable-next-line no-console
    console.log(`[g7a] worst bitwise gaps over 40 randomized materials: E=${worstE} G=${worstG} H=${worstH}`);
    expect(worstE).toBe(0);
    expect(worstG).toBe(0);
    expect(worstH).toBe(0);
  });

  it("model HVP matches the legacy FD oracle within 1e-7 (established bar)", async () => {
    const mesh = testMesh();
    const rng = mulberry32(0xFD07);
    let worst = 0;
    for (let k = 0; k < 20; k++) {
      const params = randParams(rng);
      const legacy = toLegacy(params);
      const model = new OrthotropicStVKMaterial(params);
      const x = testStates(mesh, rng)[3];
      const dir = new Float64Array(mesh.count * 3);
      for (let i = 0; i < dir.length; i++) dir[i] = rng() * 2 - 1;
      const e = relErr(
        model.hessianVector(x, dir, mesh),
        membraneHvp(Float64Array.from(x), Float64Array.from(dir), mesh, legacy),
      );
      worst = Math.max(worst, e);
      expect(e).toBeLessThan(1e-7);
    }
    // eslint-disable-next-line no-console
    console.log(`[g7a] worst model-vs-FD-oracle rel err over 20 cases: ${worst.toExponential(2)}`);
  });

  it("Float32 inputs + out-buffer variants stay bitwise", async () => {
    const mesh = testMesh();
    const model = OrthotropicStVKMaterial.fromLegacy({ ...DEFAULT_MATERIAL });
    const rng = mulberry32(5);
    const x64 = testStates(mesh, rng)[2];
    const x32 = Float32Array.from(x64);
    const dir = new Float64Array(mesh.count * 3).fill(0.25);
    // f32 input runs the identical code path on rounded inputs: compare
    // against legacy evaluated on the SAME rounded values.
    expect(model.energy(x32, mesh)).toBe(internalEnergyOnly(x32, mesh, DEFAULT_MATERIAL));
    expect(maxAbsDiff(
      model.gradient(x32, mesh).grad,
      evalInternal(x32, mesh, DEFAULT_MATERIAL).grad,
    )).toBe(0);
    // Out-buffer reuse zeroes before accumulate (matches legacy outGrad path).
    const out = new Float64Array(mesh.count * 3).fill(123.456);
    const r = model.gradient(x64, mesh, out);
    expect(r.grad).toBe(out);
    expect(maxAbsDiff(out, evalInternal(x64, mesh, DEFAULT_MATERIAL).grad)).toBe(0);
    const hout = new Float64Array(mesh.count * 3).fill(-7);
    const hr = model.hessianVector(x64, dir, mesh, hout);
    expect(hr).toBe(hout);
    expect(maxAbsDiff(hout, evalMembraneHvp(x64, dir, mesh, DEFAULT_MATERIAL))).toBe(0);
  });
});
