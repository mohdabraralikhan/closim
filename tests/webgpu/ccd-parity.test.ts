// G2 CCD parity: FP32 mirror vs f64 CPU vtCCD/eeCCD.
// Statuses must agree exactly on clear-margin scenes; TOI within G2_TOI_ABS_TOL.
// Infinity means safe and is asserted distinct from failure.
import { describe, it, expect } from "vitest";
import { vtCCD } from "../../src/collision/ccd-vt.js";
import { eeCCD } from "../../src/collision/ccd-ee.js";
import {
  vtCCDFP32, eeCCDFP32, G2_TOI_ABS_TOL,
} from "../../src/backend/webgpu/gpu-contact.js";

const DMIN = 1e-4;

function pack(pts: Array<[number, number, number]>): { x0: Float64Array; x1: Float64Array } {
  // pts: interleaved [p0.., p1..] halves: first half = t0, second half = t1
  const n = pts.length / 2;
  const x0 = new Float64Array(n * 3);
  const x1 = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) {
    x0.set(pts[i], i * 3);
    x1.set(pts[i + n], i * 3);
  }
  return { x0, x1 };
}

describe("G2 CCD parity", () => {
  const tri: Array<[number, number, number]> = [[0, 0, 0], [1, 0, 0], [0, 1, 0]];

  it("9. VT tunnel: falling vertex through triangle", () => {
    // p: 0.25,0.25,0.05 -> -0.05 ; tri static at z = 0
    const { x0, x1 } = pack([
      [0.25, 0.25, 0.05], [0, 0, 0], [1, 0, 0], [0, 1, 0],
      [0.25, 0.25, -0.05], [0, 0, 0], [1, 0, 0], [0, 1, 0],
    ]);
    void tri;
    const ref = vtCCD(x0, x1, 0, 1, 2, 3, DMIN);
    const g = vtCCDFP32(x0, x1, 0, 1, 2, 3, DMIN);
    expect(ref).toBeLessThan(1);
    expect(ref).toBeGreaterThan(0);
    expect(g.status).toBe("impact");
    expect(Math.abs(g.toi - ref)).toBeLessThan(G2_TOI_ABS_TOL);
    expect(Math.abs(ref - 0.5)).toBeLessThan(0.02);
  });

  it("10. VT miss: vertex moving away stays Infinity", () => {
    const { x0, x1 } = pack([
      [0.25, 0.25, 0.05], [0, 0, 0], [1, 0, 0], [0, 1, 0],
      [0.25, 0.25, 0.15], [0, 0, 0], [1, 0, 0], [0, 1, 0],
    ]);
    const ref = vtCCD(x0, x1, 0, 1, 2, 3, DMIN);
    const g = vtCCDFP32(x0, x1, 0, 1, 2, 3, DMIN);
    expect(ref).toBe(Infinity);
    expect(g.toi).toBe(Infinity);
    expect(g.status).toBe("safe");
  });

  it("11. VT resting contact reports TOI 0", () => {
    const { x0, x1 } = pack([
      [0.25, 0.25, 5e-5], [0, 0, 0], [1, 0, 0], [0, 1, 0],
      [0.25, 0.25, 5e-5], [0, 0, 0], [1, 0, 0], [0, 1, 0],
    ]);
    const ref = vtCCD(x0, x1, 0, 1, 2, 3, DMIN);
    const g = vtCCDFP32(x0, x1, 0, 1, 2, 3, DMIN);
    expect(ref).toBe(0);
    expect(g.toi).toBe(0);
    expect(g.status).toBe("resting");
  });

  it("12. VT static separated pair is safe on both", () => {
    const { x0, x1 } = pack([
      [0.25, 0.25, 0.01], [0, 0, 0], [1, 0, 0], [0, 1, 0],
      [0.25, 0.25, 0.01], [0, 0, 0], [1, 0, 0], [0, 1, 0],
    ]);
    const ref = vtCCD(x0, x1, 0, 1, 2, 3, DMIN);
    const g = vtCCDFP32(x0, x1, 0, 1, 2, 3, DMIN);
    expect(ref).toBe(Infinity);
    expect(g.status).toBe("safe");
  });

  it("13. EE crossing edges", () => {
    // ab: x-axis edge at z=0 sweeping y 0.05 -> -0.05; cd: static z-edge at
    // x=0,y=0. d0 = 0.05 (not resting); true crossing at t=0.5, s=t=0.5.
    const { x0, x1 } = pack([
      [-0.5, 0.05, 0], [0.5, 0.05, 0], [0, 0, -0.5], [0, 0, 0.5],
      [-0.5, -0.05, 0], [0.5, -0.05, 0], [0, 0, -0.5], [0, 0, 0.5],
    ]);
    const ref = eeCCD(x0, x1, 0, 1, 2, 3, DMIN);
    const g = eeCCDFP32(x0, x1, 0, 1, 2, 3, DMIN);
    expect(ref).toBeLessThan(1);
    expect(ref).toBeGreaterThan(0);
    expect(g.status).toBe("impact");
    expect(Math.abs(g.toi - ref)).toBeLessThan(G2_TOI_ABS_TOL);
    expect(Math.abs(ref - 0.5)).toBeLessThan(0.02);
  });

  it("14. EE miss stays Infinity", () => {
    const { x0, x1 } = pack([
      [-0.5, 0, 0.05], [0.5, 0, 0.05], [0, 2, -0.5], [0, 2, 0.5],
      [-0.5, 0, 0.06], [0.5, 0, 0.06], [0, 2, -0.5], [0, 2, 0.5],
    ]);
    const ref = eeCCD(x0, x1, 0, 1, 2, 3, DMIN);
    const g = eeCCDFP32(x0, x1, 0, 1, 2, 3, DMIN);
    expect(ref).toBe(Infinity);
    expect(g.toi).toBe(Infinity);
    expect(g.status).toBe("safe");
  });

  it("15. EE resting contact reports TOI 0", () => {
    const { x0, x1 } = pack([
      [0, 0, 5e-5], [1, 0, 5e-5], [0, 0, 0], [1, 0, 0],
      [0, 0, 5e-5], [1, 0, 5e-5], [0, 0, 0], [1, 0, 0],
    ]);
    const ref = eeCCD(x0, x1, 0, 1, 2, 3, DMIN);
    const g = eeCCDFP32(x0, x1, 0, 1, 2, 3, DMIN);
    expect(ref).toBe(0);
    expect(g.status).toBe("resting");
  });

  it("16. safe Infinity is never a failure", () => {
    // Rigid translation of the whole stencil: early-out path on both sides.
    const { x0, x1 } = pack([
      [0.1, 0.1, 0.02], [0, 0, 0], [1, 0, 0], [0, 1, 0],
      [0.6, 0.1, 0.02], [0.5, 0, 0], [1.5, 0, 0], [0.5, 1, 0],
    ]);
    const g = vtCCDFP32(x0, x1, 0, 1, 2, 3, DMIN);
    expect(g.status).toBe("safe");
    expect(g.toi).toBe(Infinity);
    expect(Number.isNaN(g.toi)).toBe(false);
  });

  it("17. VT TOI sweep parity across approach speeds", () => {
    let maxErr = 0;
    for (const h of [0.02, 0.05, 0.1, 0.2]) {
      for (const drop of [0.5 * h, 1.5 * h, 3 * h]) {
        const { x0, x1 } = pack([
          [0.3, 0.3, h], [0, 0, 0], [1, 0, 0], [0, 1, 0],
          [0.3, 0.3, h - drop], [0, 0, 0], [1, 0, 0], [0, 1, 0],
        ]);
        const ref = vtCCD(x0, x1, 0, 1, 2, 3, DMIN);
        const g = vtCCDFP32(x0, x1, 0, 1, 2, 3, DMIN);
        if (ref === Infinity) {
          expect(g.status).toBe("safe");
        } else {
          expect(g.status).toBe("impact");
          maxErr = Math.max(maxErr, Math.abs(g.toi - ref));
        }
      }
    }
    expect(maxErr).toBeLessThan(G2_TOI_ABS_TOL);
  });

  it("18. EE TOI sweep parity across approach heights", () => {
    // Same crossing geometry as test 13 with varying sweep amplitude, plus
    // one genuine miss (sweep never reaches the static edge).
    let maxErr = 0;
    for (const y0 of [0.02, 0.05, 0.1]) {
      const { x0, x1 } = pack([
        [-0.5, y0, 0], [0.5, y0, 0], [0, 0, -0.5], [0, 0, 0.5],
        [-0.5, -y0, 0], [0.5, -y0, 0], [0, 0, -0.5], [0, 0, 0.5],
      ]);
      const ref = eeCCD(x0, x1, 0, 1, 2, 3, DMIN);
      const g = eeCCDFP32(x0, x1, 0, 1, 2, 3, DMIN);
      expect(g.status).toBe("impact");
      maxErr = Math.max(maxErr, Math.abs(g.toi - ref));
    }
    {
      // miss: sweep y 0.05 -> 0.01 stays 100x above thickness throughout
      const { x0, x1 } = pack([
        [-0.5, 0.05, 0], [0.5, 0.05, 0], [0, 0, -0.5], [0, 0, 0.5],
        [-0.5, 0.01, 0], [0.5, 0.01, 0], [0, 0, -0.5], [0, 0, 0.5],
      ]);
      const ref = eeCCD(x0, x1, 0, 1, 2, 3, DMIN);
      const g = eeCCDFP32(x0, x1, 0, 1, 2, 3, DMIN);
      expect(ref).toBe(Infinity);
      expect(g.status).toBe("safe");
    }
    expect(maxErr).toBeLessThan(G2_TOI_ABS_TOL);
  });
});
