// G9A geometry predicates: scale-aware collinearity, segment queries,
// arc math + sampling, intersections, polygon queries, transforms.
// CPU-only, no document state, no device.
import { describe, it, expect } from "vitest";
import {
  angleOnArc,
  arcArcIntersections,
  arcLength,
  arcSegmentIntersections,
  bbox,
  collinear,
  dist,
  finiteVec,
  leftNormal,
  leftTurn,
  lineIntersection,
  lerp,
  nearestOnArc,
  normalizeAngle,
  orient,
  paramAlong,
  perimeter,
  pointInPolygon,
  pointOnArc,
  projectOnSegment,
  reflectAcrossLine,
  rotateAround,
  sampleArc,
  segmentIntersection,
  selfIntersections,
  signedArea,
  sub,
  type ArcGeometry,
} from "../../src/cad/geom.js";

const EPS = 1e-9;

function arc(center: [number, number], radius: number, a0: number, sweep: number): ArcGeometry {
  return { center, radius, a0, sweep };
}

describe("predicates", () => {
  it("collinear is scale-aware (dimensionless tolerance)", () => {
    expect(collinear([0, 0], [1, 0], [2, 0], EPS)).toBe(true);
    // Same angular test at micrometre coordinates: the deviation is 1e-18
    // over a 2e-6 arm (sine 5e-13) — inside the 1e-9 sine tolerance.
    expect(collinear([0, 0], [1e-6, 0], [2e-6, 1e-18], EPS)).toBe(true);
    expect(collinear([0, 0], [1e-6, 0], [2e-6, 1e-9], EPS)).toBe(false);
    expect(collinear([0, 0], [1, 0], [1, 1], EPS)).toBe(false);
    // Zero-length legs count as collinear (degenerate input, not an error).
    expect(collinear([0, 0], [0, 0], [1, 1], EPS)).toBe(true);
  });

  it("collinear behaves identically at mm and m scale", () => {
    const shape: Array<[number, number]> = [[0, 0], [100, 0], [200, 1e-6]];
    const scaled = shape.map(([x, y]) => [x * 1e-3, y * 1e-3] as [number, number]);
    expect(collinear(shape[0], shape[1], shape[2], 1e-7)).toBe(
      collinear(scaled[0], scaled[1], scaled[2], 1e-7),
    );
  });

  it("leftTurn detects counter-clockwise turns beyond tolerance", () => {
    expect(leftTurn([0, 0], [1, 0], [1, 1], EPS)).toBe(true);
    expect(leftTurn([0, 0], [1, 0], [1, -1], EPS)).toBe(false);
    expect(leftTurn([0, 0], [1, 0], [2, 0], EPS)).toBe(false); // straight
  });

  it("orient returns signed double-area", () => {
    expect(orient([0, 0], [1, 0], [0, 1])).toBeCloseTo(1, 12);
    expect(orient([0, 0], [1, 0], [0, -1])).toBeCloseTo(-1, 12);
    expect(orient([0, 0], [1, 0], [2, 0])).toBeCloseTo(0, 12);
  });

  it("finiteVec rejects NaN/Inf", () => {
    expect(finiteVec([1, 2])).toBe(true);
    expect(finiteVec([NaN, 0])).toBe(false);
    expect(finiteVec([0, Infinity])).toBe(false);
  });
});

describe("segment queries", () => {
  it("projectOnSegment clamps and reports distance", () => {
    const mid = projectOnSegment([0.5, 1], [0, 0], [1, 0]);
    expect(mid.t).toBeCloseTo(0.5, 12);
    expect(mid.pos).toEqual([0.5, 0]);
    expect(mid.distance).toBeCloseTo(1, 12);

    const before = projectOnSegment([-1, 0], [0, 0], [1, 0]);
    expect(before.t).toBe(0);
    expect(before.pos).toEqual([0, 0]);

    const after = projectOnSegment([2, 3], [0, 0], [1, 0]);
    expect(after.t).toBe(1);
    expect(after.pos).toEqual([1, 0]);

    const degenerate = projectOnSegment([1, 1], [0, 0], [0, 0]);
    expect(degenerate.t).toBe(0);
    expect(degenerate.distance).toBeCloseTo(Math.hypot(1, 1), 12);
  });

  it("paramAlong is unclamped", () => {
    expect(paramAlong([0.5, 9], [0, 0], [1, 0])).toBeCloseTo(0.5, 12);
    expect(paramAlong([-1, 9], [0, 0], [1, 0])).toBeCloseTo(-1, 12);
    expect(paramAlong([3, 9], [0, 0], [1, 0])).toBeCloseTo(3, 12);
    expect(paramAlong([1, 1], [0, 0], [0, 0])).toBe(0);
  });

  it("lineIntersection crosses or reports parallel", () => {
    const hit = lineIntersection([0, 0], [1, 1], [0, 1], [1, 0], EPS);
    expect(hit![0]).toBeCloseTo(0.5, 12);
    expect(hit![1]).toBeCloseTo(0.5, 12);
    expect(lineIntersection([0, 0], [1, 0], [0, 1], [1, 1], EPS)).toBeNull(); // parallel
    expect(lineIntersection([0, 0], [1, 0], [2, 0], [3, 0], EPS)).toBeNull(); // collinear
    expect(lineIntersection([0, 0], [0, 0], [0, 1], [1, 1], EPS)).toBeNull(); // zero direction
  });

  it("segmentIntersection classifies proper vs endpoint touches", () => {
    const proper = segmentIntersection([0, 0], [2, 2], [0, 2], [2, 0], EPS);
    expect(proper!.kind).toBe("proper");
    expect(proper!.pos[0]).toBeCloseTo(1, 12);

    const endpoint = segmentIntersection([0, 0], [1, 0], [1, 0], [1, 1], EPS);
    expect(endpoint!.kind).toBe("touch-endpoint");
    expect(endpoint!.pos).toEqual([1, 0]);

    expect(segmentIntersection([0, 0], [1, 0], [2, 0], [3, 0], EPS)).toBeNull(); // disjoint parallel
    expect(segmentIntersection([0, 0], [1, 0], [5, 5], [6, 6], EPS)).toBeNull(); // no crossing
  });

  it("parallel segments that touch are reported", () => {
    const hit = segmentIntersection([0, 0], [1, 0], [1, 0], [2, 0], EPS);
    expect(hit).not.toBeNull();
    expect(hit!.kind).toBe("touch-endpoint");
  });

  it("leftNormal is the +90 degrees normal with the segment's length", () => {
    const n = leftNormal([0, 0], [2, 0]);
    expect(n[0]).toBeCloseTo(0, 12); // -0. === 0 for arithmetic purposes
    expect(n[1]).toBe(2);
    const m = leftNormal([0, 0], [0, 3]);
    expect(m[0]).toBe(-3);
    expect(m[1]).toBeCloseTo(0, 12);
  });

  it("lerp and sub behave", () => {
    expect(lerp([0, 0], [2, 4], 0.5)).toEqual([1, 2]);
    expect(sub([3, 1], [1, 1])).toEqual([2, 0]);
    expect(dist([0, 0], [3, 4])).toBeCloseTo(5, 12);
  });
});

describe("arc math", () => {
  it("pointOnArc hits endpoints at f=0 and f=1", () => {
    const a = arc([0, 0], 2, 0, Math.PI / 2);
    const p0 = pointOnArc(a, 0);
    const p1 = pointOnArc(a, 1);
    expect(p0[0]).toBeCloseTo(2, 12);
    expect(p0[1]).toBeCloseTo(0, 12);
    expect(p1[0]).toBeCloseTo(0, 12);
    expect(p1[1]).toBeCloseTo(2, 12);
    expect(pointOnArc(a, 0.5)[0]).toBeCloseTo(Math.SQRT2, 10);
  });

  it("arcLength is radius * |sweep|", () => {
    expect(arcLength(arc([0, 0], 2, 0, Math.PI))).toBeCloseTo(2 * Math.PI, 12);
    expect(arcLength(arc([0, 0], 2, 0, -Math.PI / 2))).toBeCloseTo(Math.PI, 12);
  });

  it("sampleArc is deterministic, endpoints-exact, and sagitta-bounded", () => {
    const tol = 1e-4;
    const s1 = sampleArc([0, 0], 1, 0.3, 2.1, tol);
    const s2 = sampleArc([0, 0], 1, 0.3, 2.1, tol);
    expect(s1).toEqual(s2); // identical subdivision
    expect(s1[0][0]).toBeCloseTo(Math.cos(0.3), 12);
    expect(s1[s1.length - 1][0]).toBeCloseTo(Math.cos(0.3 + 2.1), 12);
    // Every sample lies on the circle.
    for (const p of s1) expect(Math.hypot(p[0], p[1])).toBeCloseTo(1, 9);
    // Sagitta bound: chord midpoint deviation <= tol (+ float slack).
    for (let i = 0; i + 1 < s1.length; i++) {
      const mid: [number, number] = [(s1[i][0] + s1[i + 1][0]) / 2, (s1[i][1] + s1[i + 1][1]) / 2];
      const dev = 1 - Math.hypot(mid[0], mid[1]);
      expect(dev).toBeLessThanOrEqual(tol + 1e-12);
    }
  });

  it("sampleArc rejects degenerate sweeps", () => {
    expect(() => sampleArc([0, 0], 1, 0, 0, 1e-4)).toThrow();
    expect(() => sampleArc([0, 0], 1, 0, 2 * Math.PI, 1e-4)).toThrow();
    expect(() => sampleArc([0, 0], 0, 0, 1, 1e-4)).toThrow();
  });

  it("arcSegmentIntersections finds line-circle crossings on the sweep", () => {
    const a = arc([0, 0], 1, 0, Math.PI); // upper half circle
    const hits = arcSegmentIntersections(a, [-2, 0], [2, 0], 1e-12);
    expect(hits.length).toBe(2);
    expect(hits.map((h) => h.pos[0])).toEqual([-1, 1]);
    // A chord strictly inside the upper sweep (y = 0.5 crosses at ±0.866):
    const chord = arcSegmentIntersections(a, [-2, 0.5], [2, 0.5], 1e-12);
    expect(chord.length).toBe(2);
    // Line below the circle: no hits.
    expect(arcSegmentIntersections(a, [-2, -2], [2, -2], 1e-12).length).toBe(0);
    // Sweep-limited: y = -0.5 touches the circle but not this sweep.
    const lowerChord = arcSegmentIntersections(a, [-2, -0.5], [2, -0.5], 1e-12);
    expect(lowerChord.length).toBe(0);
  });

  it("arcArcIntersections finds two crossings, rejects concentric/disjoint", () => {
    const upper = arc([0, 0], 1, 0, Math.PI);
    const lower = arc([0, 0], 1, Math.PI, Math.PI);
    expect(arcArcIntersections(upper, lower, 1e-12)).toEqual([]); // same circle

    const shifted = arc([1.2, 0], 1, 0, Math.PI); // upper half of shifted circle
    const hits = arcArcIntersections(upper, shifted, 1e-12);
    expect(hits.length).toBe(1); // the other crossing is outside one sweep

    const far = arc([5, 0], 1, 0, Math.PI);
    expect(arcArcIntersections(upper, far, 1e-12)).toEqual([]);
  });

  it("angleOnArc respects sweep direction", () => {
    const cw = arc([0, 0], 1, 0, -Math.PI / 2);
    expect(angleOnArc(cw, -Math.PI / 4, 1e-12)).toBe(true);
    expect(angleOnArc(cw, Math.PI / 4, 1e-12)).toBe(false);
    const ccw = arc([0, 0], 1, 0, Math.PI / 2);
    expect(angleOnArc(ccw, Math.PI / 4, 1e-12)).toBe(true);
    expect(angleOnArc(ccw, -Math.PI / 4, 1e-12)).toBe(false);
  });

  it("nearestOnArc projects inside the sweep and clamps outside", () => {
    const a = arc([0, 0], 1, 0, Math.PI / 2);
    const inside = nearestOnArc(a, [2, 2]);
    expect(inside[0]).toBeCloseTo(Math.SQRT1_2, 9);
    expect(inside[1]).toBeCloseTo(Math.SQRT1_2, 9);
    const beforeStart = nearestOnArc(a, [1, -1]);
    expect(beforeStart[0]).toBeCloseTo(1, 12);
    const afterEnd = nearestOnArc(a, [-1, 1]);
    expect(afterEnd[1]).toBeCloseTo(1, 12);
  });

  it("normalizeAngle folds into (-pi, pi]", () => {
    expect(normalizeAngle(0)).toBe(0);
    expect(normalizeAngle(Math.PI * 3)).toBeCloseTo(Math.PI, 12); // upper bound inclusive
    expect(normalizeAngle(-Math.PI * 3)).toBeCloseTo(Math.PI, 12);
    expect(normalizeAngle(Math.PI * 0.5)).toBeCloseTo(Math.PI * 0.5, 12);
    expect(normalizeAngle(2 * Math.PI)).toBeCloseTo(0, 12);
    expect(normalizeAngle(-0.5)).toBeCloseTo(-0.5, 12);
  });
});

describe("polygon queries", () => {
  const square: Array<[number, number]> = [[0, 0], [1, 0], [1, 1], [0, 1]];

  it("signedArea signs by winding", () => {
    expect(signedArea(square)).toBeCloseTo(1, 12);
    expect(signedArea([...square].reverse())).toBeCloseTo(-1, 12);
  });

  it("perimeter closes or stays open", () => {
    expect(perimeter(square)).toBeCloseTo(4, 12);
    expect(perimeter(square, false)).toBeCloseTo(3, 12);
  });

  it("bbox spans all points", () => {
    const b = bbox([[1, 2], [-1, 5], [3, 0]]);
    expect(b!.min).toEqual([-1, 0]);
    expect(b!.max).toEqual([3, 5]);
    expect(bbox([])).toBeNull();
  });

  it("pointInPolygon excludes the boundary", () => {
    expect(pointInPolygon([0.5, 0.5], square)).toBe(true);
    expect(pointInPolygon([1.5, 0.5], square)).toBe(false);
    expect(pointInPolygon([0.5, 1], square)).toBe(false); // on edge: caller's job
  });

  it("selfIntersections finds bowties and skips adjacent pairs", () => {
    expect(selfIntersections(square, EPS)).toEqual([]);
    const bowtie: Array<[number, number]> = [[0, 0], [1, 1], [1, 0], [0, 1]];
    const hits = selfIntersections(bowtie, EPS);
    expect(hits.length).toBe(1);
    expect(hits[0].pos[0]).toBeCloseTo(0.5, 9);
    // Repeated (spike) vertex: non-adjacent touching edges are reported.
    const spike: Array<[number, number]> = [[0, 0], [2, 0], [1, 0], [1, 2]];
    expect(selfIntersections(spike, EPS).length).toBeGreaterThan(0);
  });
});

describe("transforms", () => {
  it("rotateAround rotates about a pivot", () => {
    const p = rotateAround([1, 0], [0, 0], Math.PI / 2);
    expect(p[0]).toBeCloseTo(0, 12);
    expect(p[1]).toBeCloseTo(1, 12);
    const q = rotateAround([2, 1], [1, 1], Math.PI);
    expect(q[0]).toBeCloseTo(0, 12);
    expect(q[1]).toBeCloseTo(1, 12);
  });

  it("reflectAcrossLine fixes the line and mirrors across it", () => {
    const p: [number, number] = [1, 1];
    const r = reflectAcrossLine(p, [0, 0], [1, 0]);
    expect(r[0]).toBeCloseTo(1, 12);
    expect(r[1]).toBeCloseTo(-1, 12);
    // Points on the line are fixed.
    const on = reflectAcrossLine([2, 0], [0, 0], [1, 0]);
    expect(on[0]).toBeCloseTo(2, 12);
    expect(on[1]).toBeCloseTo(0, 12);
    // Double reflection = identity (to float precision).
    const back = reflectAcrossLine(r, [0, 0], [1, 0]);
    expect(back[0]).toBeCloseTo(p[0], 9);
    expect(back[1]).toBeCloseTo(p[1], 9);
    expect(() => reflectAcrossLine(p, [1, 1], [1, 1])).toThrow();
  });
});
