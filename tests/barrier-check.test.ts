// Phase 1 ladder: barrier gradient FD + HVP directional check (floor contact).
import { describe, it, expect } from "vitest";
import { ContactSystem } from "../src/collision/contact-assembly.js";
import { DEFAULT_CONTACT_PARAMS } from "../src/collision/types.js";
import { barrierValue } from "../src/collision/barrier.js";

describe("barrier energy", () => {
  it("is zero beyond dHat and blows up at zero", () => {
    expect(barrierValue(0.005, 0.002)).toBe(0);
    expect(barrierValue(0.002, 0.002)).toBe(0);
    expect(barrierValue(0.001, 0.002)).toBeGreaterThan(0);
    expect(barrierValue(1e-9, 0.002)).toBeGreaterThan(barrierValue(0.001, 0.002));
  });

  it("floor gradient matches finite differences", () => {
    const cs = new ContactSystem(DEFAULT_CONTACT_PARAMS, new Uint32Array(0));
    cs.setFloor(0);
    const x = new Float64Array([0.3, 0.001, 0.1, -0.2, 0.01, 0.4]); // v0 inside barrier, v1 outside
    cs.beginStep(Float64Array.from(x));
    cs.updateActiveSet(x);
    expect(cs.active.length).toBe(1);
    const { grad } = cs.energyGrad(x);
    const eps = 1e-7;
    let maxRel = 0;
    const e0 = cs.meritEnergy(x);
    for (let i = 0; i < x.length; i++) {
      const xp = Float64Array.from(x); xp[i] += eps;
      const xm = Float64Array.from(x); xm[i] -= eps;
      const num = (cs.meritEnergy(xp) - cs.meritEnergy(xm)) / (2 * eps);
      const ana = grad[i];
      const rel = Math.abs(num - ana) / (Math.abs(num) + 1e-12);
      maxRel = Math.max(maxRel, rel);
    }
    expect(maxRel).toBeLessThan(1e-6);
    expect(e0).toBeGreaterThan(0);
  });

  it("frozen HVP matches directional derivative of the barrier gradient", () => {
    const cs = new ContactSystem(DEFAULT_CONTACT_PARAMS, new Uint32Array(0));
    cs.setFloor(0);
    const x = new Float64Array([0.3, 0.0012, 0.1, 0.1, 0.0007, -0.3]);
    cs.beginStep(Float64Array.from(x));
    cs.updateActiveSet(x);
    expect(cs.active.length).toBe(2);
    const v = new Float64Array([0.1, -0.4, 0.2, -0.3, 0.5, 0.1]);
    const out = new Float64Array(x.length);
    cs.applyHvp(x, v, out);
    // FD of the BARRIER-ONLY gradient (meritEnergy), matching applyHvp's scope.
    // (energyGrad additionally carries lagged friction, deliberately Hessian-free.)
    const h = 1e-7;
    const fdGrad = (xx: Float64Array): Float64Array => {
      const gg = new Float64Array(xx.length);
      const e0 = cs.meritEnergy(xx);
      void e0;
      for (let i = 0; i < xx.length; i++) {
        const xp = Float64Array.from(xx); xp[i] += h;
        const xm = Float64Array.from(xx); xm[i] -= h;
        gg[i] = (cs.meritEnergy(xp) - cs.meritEnergy(xm)) / (2 * h);
      }
      return gg;
    };
    const xp = new Float64Array(x.length), xm = new Float64Array(x.length);
    for (let i = 0; i < x.length; i++) { xp[i] = x[i] + h * v[i]; xm[i] = x[i] - h * v[i]; }
    const gp = fdGrad(xp);
    const gm = fdGrad(xm);
    let maxRel = 0;
    for (let i = 0; i < x.length; i++) {
      const num = (gp[i] - gm[i]) / (2 * h);
      const rel = Math.abs(num - out[i]) / (Math.abs(num) + 1e-9);
      maxRel = Math.max(maxRel, rel);
    }
    expect(maxRel).toBeLessThan(1e-4);
  });
});
