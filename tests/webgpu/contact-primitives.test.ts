// G2 primitive parity: FP32 closest-point mirror vs f64 CPU reference.
// Scenes use clear margins (>> FP32 ulp) so region classification agrees;
// degenerate cases assert finiteness, never NaN (documented D1 deviation).
import { describe, it, expect } from "vitest";
import {
  closestPointVertexTriangle, closestPointEdgeEdge,
} from "../../src/collision/closest-point.js";
import {
  closestVtFP32, closestEeFP32,
  G2_DIST_ABS_TOL, G2_DIST_REL_TOL,
} from "../../src/backend/webgpu/gpu-contact.js";

function vtParity(
  p: [number, number, number],
  a: [number, number, number], b: [number, number, number], c: [number, number, number],
  stTol = 1e-6,
): void {
  const ref = closestPointVertexTriangle(p[0], p[1], p[2], a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
  const g = closestVtFP32(p[0], p[1], p[2], a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]);
  expect(g.degenerate).toBe(false);
  expect(Math.abs(g.dist - ref.dist)).toBeLessThan(G2_DIST_ABS_TOL);
  expect(Math.abs(g.dist - ref.dist) / Math.max(ref.dist, 1e-9)).toBeLessThan(G2_DIST_REL_TOL);
  expect(Math.abs(g.s - ref.s)).toBeLessThan(stTol);
  expect(Math.abs(g.t - ref.t)).toBeLessThan(stTol);
}

function eeParity(
  a: [number, number, number], b: [number, number, number],
  c: [number, number, number], d: [number, number, number],
  stTol = 1e-6,
): void {
  const ref = closestPointEdgeEdge(
    a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], d[0], d[1], d[2]);
  const g = closestEeFP32(
    a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], d[0], d[1], d[2]);
  expect(g.degenerate).toBe(false);
  expect(Math.abs(g.dist - ref.dist)).toBeLessThan(G2_DIST_ABS_TOL);
  expect(Math.abs(g.s - ref.s)).toBeLessThan(stTol);
  expect(Math.abs(g.t - ref.t)).toBeLessThan(stTol);
}

describe("G2 closest-point primitives", () => {
  const A: [number, number, number] = [0, 0, 0];
  const B: [number, number, number] = [1, 0, 0];
  const C: [number, number, number] = [0, 1, 0];

  it("1. VT interior closest point", () => {
    const g = closestVtFP32(0.2, 0.2, 0.5, 0, 0, 0, 1, 0, 0, 0, 1, 0);
    expect(g.s).toBeCloseTo(0.2, 6);
    expect(g.t).toBeCloseTo(0.2, 6);
    expect(g.dist).toBeCloseTo(0.5, 6);
    vtParity([0.2, 0.2, 0.5], A, B, C);
  });

  it("2. VT vertex region (s = t = 0)", () => {
    const g = closestVtFP32(-0.5, -0.5, 0.3, 0, 0, 0, 1, 0, 0, 0, 1, 0);
    expect(g.s).toBe(0);
    expect(g.t).toBe(0);
    vtParity([-0.5, -0.5, 0.3], A, B, C);
  });

  it("3. VT edge region boundary", () => {
    const g = closestVtFP32(0.5, -0.2, 0.1, 0, 0, 0, 1, 0, 0, 0, 1, 0);
    expect(g.s).toBeCloseTo(0.5, 5);
    expect(g.t).toBe(0);
    vtParity([0.5, -0.2, 0.1], A, B, C);
  });

  it("4. VT degenerate triangle stays finite (D1 documented deviation)", () => {
    const g = closestVtFP32(0.3, 0.4, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1);
    expect(g.degenerate).toBe(true);
    expect(Number.isFinite(g.dist)).toBe(true);
    expect(Number.isFinite(g.s) && Number.isFinite(g.t)).toBe(true);
    const ref = Math.hypot(0.2, 0.3, 0.4);
    expect(Math.abs(g.dist - ref)).toBeLessThan(1e-6);
  });

  it("5. EE interior closest point", () => {
    // ab along x at y=0.1, cd along z at y=0: closest (0.3,0.1,0)-(0.3,0,0.2)-ish
    const g = closestEeFP32(0, 0.1, 0, 1, 0.1, 0, 0.3, 0, -0.2, 0.3, 0, 0.4);
    expect(g.dist).toBeCloseTo(0.1, 5);
    eeParity([0, 0.1, 0], [1, 0.1, 0], [0.3, 0, -0.2], [0.3, 0, 0.4]);
  });

  it("6. EE parallel offset is deterministic", () => {
    const g1 = closestEeFP32(0, 0, 0, 1, 0, 0, 0, 0.05, 0, 1, 0.05, 0);
    const g2 = closestEeFP32(0, 0, 0, 1, 0, 0, 0, 0.05, 0, 1, 0.05, 0);
    expect(g1.s).toBe(g2.s);
    expect(g1.t).toBe(g2.t);
    expect(g1.dist).toBeCloseTo(0.05, 6);
    eeParity([0, 0, 0], [1, 0, 0], [0, 0.05, 0], [1, 0.05, 0], 1e-4);
  });

  it("7. EE endpoint clamp parity", () => {
    // skew segments whose unconstrained minima fall outside [0,1]
    eeParity([0, 0, 0], [1, 0, 0], [2, 0.3, 0.1], [3, 0.5, -0.2]);
    eeParity([0, 0, 0], [0.5, 0, 0], [-1, 0.2, 0.1], [-0.5, 0.4, 0.3]);
  });

  it("8. EE degenerate edges stay finite", () => {
    const g = closestEeFP32(1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6);
    expect(g.degenerate).toBe(true);
    expect(Number.isFinite(g.dist)).toBe(true);
    expect(g.dist).toBeCloseTo(Math.hypot(3, 3, 3), 5);
    const h = closestEeFP32(1, 2, 3, 1, 2, 3, 0, 0, 0, 1, 0, 0);
    expect(Number.isFinite(h.dist)).toBe(true);
  });
});
