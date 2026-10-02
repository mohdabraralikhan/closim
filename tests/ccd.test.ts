// Phase 1 ladder: VT/EE CCD tunneling + resting + miss cases.
import { describe, it, expect } from "vitest";
import { vtCCD } from "../src/collision/ccd-vt.js";
import { eeCCD } from "../src/collision/ccd-ee.js";

const DMIN = 1e-4;

function ring3(x0: number[], x1: number[]): { x0: Float64Array; x1: Float64Array } {
  return { x0: Float64Array.from(x0), x1: Float64Array.from(x1) };
}

describe("VT CCD", () => {
  it("catches a fast vertex tunneling through a triangle", () => {
    // triangle (0,1,2) in z=0 plane, vertex 3 flies +z -> -z through it
    const base = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0.2, 0.2, 0.01];
    const moved = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0.2, 0.2, -0.01];
    const { x0, x1 } = ring3(base, moved);
    const toi = vtCCD(x0, x1, 3, 0, 1, 2, DMIN);
    expect(toi).toBeGreaterThanOrEqual(0);
    expect(toi).toBeLessThan(1);
    expect(Math.abs(toi - 0.5)).toBeLessThan(0.05);
  });

  it("reports safe when the vertex stops before the triangle", () => {
    const base = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0.2, 0.2, 0.01];
    const moved = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0.2, 0.2, 0.005];
    const { x0, x1 } = ring3(base, moved);
    expect(vtCCD(x0, x1, 3, 0, 1, 2, DMIN)).toBe(Infinity);
  });

  it("reports TOI 0 for resting contact", () => {
    const base = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0.2, 0.2, 0.00005];
    const { x0, x1 } = ring3(base, base);
    expect(vtCCD(x0, x1, 3, 0, 1, 2, DMIN)).toBe(0);
  });

  it("ignores a vertex passing outside the triangle", () => {
    const base = [0, 0, 0, 1, 0, 0, 0, 1, 0, 5.0, 5.0, 0.01];
    const moved = [0, 0, 0, 1, 0, 0, 0, 1, 0, 5.0, 5.0, -0.01];
    const { x0, x1 } = ring3(base, moved);
    expect(vtCCD(x0, x1, 3, 0, 1, 2, DMIN)).toBe(Infinity);
  });
});

describe("EE CCD", () => {
  it("catches two crossing edges", () => {
    // edge ab along x at y=0,z=0 ; edge cd along z moving y +0.01 -> -0.01, crossing at origin
    const base = [-1, 0, 0, 1, 0, 0, 0, 0.01, -1, 0, 0.01, 1];
    const moved = [-1, 0, 0, 1, 0, 0, 0, -0.01, -1, 0, -0.01, 1];
    const { x0, x1 } = ring3(base, moved);
    const toi = eeCCD(x0, x1, 0, 1, 2, 3, DMIN);
    expect(toi).toBeGreaterThanOrEqual(0);
    expect(toi).toBeLessThan(1);
    expect(Math.abs(toi - 0.5)).toBeLessThan(0.1);
  });

  it("reports safe for parallel edges passing at distance", () => {
    const base = [-1, 0, 0, 1, 0, 0, -1, 0.5, -1, 1, 0.5, 1];
    const moved = [-1, 0, 0, 1, 0, 0, -1, 0.4, -1, 1, 0.4, 1];
    const { x0, x1 } = ring3(base, moved);
    expect(eeCCD(x0, x1, 0, 1, 2, 3, DMIN)).toBe(Infinity);
  });
});
