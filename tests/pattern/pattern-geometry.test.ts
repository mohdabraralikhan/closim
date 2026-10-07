// G7D pattern geometry: validation, winding, arcs, point-in-panel,
// parameterization, serialization, determinism. CPU-only, no device.
import { describe, it, expect } from "vitest";
import {
  PatternError,
  approximateArc,
  boundaryPoint,
  parameterizeLoop,
  pointInPanel,
  panelFromJSON,
  panelToJSON,
  triangulatePatternPanel,
  validatePanel,
  type PatternPanel,
} from "../../src/pattern/pattern-geometry.js";

function rectPanel(w = 0.1, h = 0.05): PatternPanel {
  return {
    id: "rect", outline: [{ kind: "polyline", points: [[0, 0], [w, 0], [w, h], [0, h]] }],
    holes: [], grainAngleRad: 0, materialId: "cotton-poplin",
  };
}

function reversedPanel(p: PatternPanel): PatternPanel {
  const rev = (pts: Array<[number, number]>): Array<[number, number]> =>
    pts.map((q) => [q[0], q[1]] as [number, number]).reverse();
  return {
    ...p,
    outline: p.outline.map((pr) => pr.kind === "polyline"
      ? { kind: "polyline", points: rev(pr.points) }
      : pr),
    holes: p.holes.map((h) => h.map((pr) => pr.kind === "polyline"
      ? { kind: "polyline", points: rev(pr.points) }
      : pr)),
  };
}

describe("G7D validation + winding", () => {
  it("accepts rectangle/triangle/L-shape with correct orientation flags", () => {
    const r = validatePanel(rectPanel());
    expect(r.outerReversed).toBe(false);
    expect(r.outer.length).toBe(4);
    const tri: PatternPanel = {
      id: "tri", outline: [{ kind: "polyline", points: [[0, 0], [0.1, 0], [0, 0.1]] }],
      holes: [], grainAngleRad: 0.3, materialId: "m",
    };
    expect(validatePanel(tri).outer.length).toBe(3);
    const lshape: PatternPanel = {
      id: "ell",
      outline: [{
        kind: "polyline",
        points: [[0, 0], [0.1, 0], [0.1, 0.04], [0.04, 0.04], [0.04, 0.1], [0, 0.1]],
      }],
      holes: [], grainAngleRad: 0, materialId: "m",
    };
    expect(validatePanel(lshape).outer.length).toBe(6);
  });

  it("normalizes reversed winding to identical geometry", () => {
    const fwd = validatePanel(rectPanel());
    const rev = validatePanel(reversedPanel(rectPanel()));
    expect(rev.outerReversed).toBe(true);
    expect(rev.outer).toEqual(fwd.outer);
    // Triangulation is winding-agnostic: bitwise-identical output.
    const a = triangulatePatternPanel(rectPanel());
    const b = triangulatePatternPanel(reversedPanel(rectPanel()));
    expect(JSON.stringify({ v: Array.from(b.vertices), t: Array.from(b.triangles) }))
      .toBe(JSON.stringify({ v: Array.from(a.vertices), t: Array.from(a.triangles) }));
  });

  it("rejects self-intersecting boundaries (bowtie)", () => {
    // Asymmetric bowtie: nonzero signed area, one proper edge crossing.
    const bowtie: PatternPanel = {
      id: "bow", outline: [{ kind: "polyline", points: [[0, 0], [0.1, 0.08], [0.1, 0], [0, 0.1]] }],
      holes: [], grainAngleRad: 0, materialId: "m",
    };
    let err: unknown = null;
    try {
      validatePanel(bowtie);
    } catch (e) {
      err = e;
    }
    expect(err instanceof PatternError).toBe(true);
    expect((err as PatternError).code).toBe("self-intersecting-boundary");
  });

  it("rejects zero-area panels (collinear / degenerate)", () => {
    const line: PatternPanel = {
      id: "line", outline: [{ kind: "polyline", points: [[0, 0], [0.1, 0], [0.2, 0]] }],
      holes: [], grainAngleRad: 0, materialId: "m",
    };
    expect(() => validatePanel(line)).toThrowError(/zero-area-panel/);
    const dup: PatternPanel = {
      id: "dup", outline: [{ kind: "polyline", points: [[0, 0], [0, 0], [0, 0]] }],
      holes: [], grainAngleRad: 0, materialId: "m",
    };
    expect(() => validatePanel(dup)).toThrowError(/empty-loop|zero-area-panel/);
  });

  it("rejects invalid holes (outside / touching / overlapping / nested)", () => {
    const base = (): PatternPanel => ({
      id: "h", outline: [{ kind: "polyline", points: [[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1]] }],
      holes: [], grainAngleRad: 0, materialId: "m",
    });
    const outside = base();
    outside.holes = [[{ kind: "polyline", points: [[0.2, 0.2], [0.3, 0.2], [0.3, 0.3], [0.2, 0.3]] }]];
    expect(() => validatePanel(outside)).toThrowError(/hole-outside/);
    const touching = base();
    touching.holes = [[{ kind: "polyline", points: [[0, 0.02], [0.04, 0.02], [0.04, 0.06], [0, 0.06]] }]];
    expect(() => validatePanel(touching)).toThrowError(/hole-touching|hole-outside/);
    const overlap = base();
    overlap.holes = [
      [{ kind: "polyline", points: [[0.02, 0.02], [0.06, 0.02], [0.06, 0.06], [0.02, 0.06]] }],
      [{ kind: "polyline", points: [[0.04, 0.04], [0.08, 0.04], [0.08, 0.08], [0.04, 0.08]] }],
    ];
    expect(() => validatePanel(overlap)).toThrowError(/holes-overlap/);
    const nested = base();
    nested.holes = [
      [{ kind: "polyline", points: [[0.01, 0.01], [0.09, 0.01], [0.09, 0.09], [0.01, 0.09]] }],
    ];
    // Inner hole nested inside the first hole.
    nested.holes.push([{ kind: "polyline", points: [[0.03, 0.03], [0.07, 0.03], [0.07, 0.07], [0.03, 0.07]] }]);
    expect(() => validatePanel(nested)).toThrowError(/holes-overlap/);
  });

  it("rejects degenerate output via minTriArea on a needle panel", () => {
    const needle: PatternPanel = {
      id: "needle", outline: [{ kind: "polyline", points: [[0, 0], [0.1, 0], [0.05, 1e-9]] }],
      holes: [], grainAngleRad: 0, materialId: "m",
    };
    // Area 5e-11 > minArea: valid panel, but every triangulation is slivers.
    expect(() => triangulatePatternPanel(needle, { minTriArea: 1e-9 })).toThrowError(/degenerate-output/);
  });
});

describe("G7D arcs", () => {
  it("approximates a quarter arc within sagitta tolerance", () => {
    const pts = approximateArc([0, 0], 0.05, 0, Math.PI / 2, 1e-4);
    expect(pts.length).toBeGreaterThan(4);
    // Endpoints exact.
    expect(pts[0][0]).toBeCloseTo(0.05, 12);
    expect(pts[0][1]).toBeCloseTo(0, 12);
    expect(pts[pts.length - 1][0]).toBeCloseTo(0, 12);
    expect(pts[pts.length - 1][1]).toBeCloseTo(0.05, 12);
    // Chordal sagitta bound: every chord midpoint within tol of the circle.
    for (let i = 0; i < pts.length - 1; i++) {
      const mx = (pts[i][0] + pts[i + 1][0]) / 2;
      const my = (pts[i][1] + pts[i + 1][1]) / 2;
      expect(Math.abs(Math.hypot(mx, my) - 0.05)).toBeLessThan(1e-4);
    }
  });

  it("full circles close (a0 == a1)", () => {
    const pts = approximateArc([0.05, 0.05], 0.01, 0.3, 0.3, 1e-4);
    const f = pts[0], l = pts[pts.length - 1];
    expect(Math.hypot(f[0] - l[0], f[1] - l[1])).toBeLessThan(1e-12);
    expect(pts.length).toBeGreaterThan(8);
  });

  it("rejects invalid arcs", () => {
    expect(() => approximateArc([0, 0], 0, 0, 1, 1e-4)).toThrowError(/invalid-arc/);
    expect(() => approximateArc([0, 0], 0.1, 0, 1, 0)).toThrowError(/invalid-arc/);
  });
});

describe("G7D point-in-panel + parameterization", () => {
  it("classifies inside/outside/boundary/hole correctly", () => {
    const panel: PatternPanel = {
      id: "h", outline: [{ kind: "polyline", points: [[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1]] }],
      holes: [[{ kind: "polyline", points: [[0.03, 0.03], [0.07, 0.03], [0.07, 0.07], [0.03, 0.07]] }]],
      grainAngleRad: 0, materialId: "m",
    };
    const v = validatePanel(panel);
    expect(pointInPanel([0.01, 0.01], v)).toBe(true);
    expect(pointInPanel([0.2, 0.2], v)).toBe(false);
    expect(pointInPanel([0.05, 0.05], v)).toBe(false); // inside hole
    expect(pointInPanel([0.05, 0], v)).toBe(true); // on outer boundary
    expect(pointInPanel([0.05, 0.03], v)).toBe(true); // on hole boundary
    expect(pointInPanel([0.03, 0.05], v)).toBe(true); // on hole boundary
  });

  it("arclength parameterization hits vertices at exact quarters", () => {
    const v = validatePanel(rectPanel(0.1, 0.1));
    const prm = parameterizeLoop(v.outer);
    expect(prm.total).toBeCloseTo(0.4, 12);
    expect(boundaryPoint(v.outer, prm, 0)).toEqual([0, 0]);
    const q = boundaryPoint(v.outer, prm, 0.25);
    expect(q[0]).toBeCloseTo(0.1, 12);
    expect(q[1]).toBeCloseTo(0, 12);
    const h = boundaryPoint(v.outer, prm, 0.5);
    expect(h[0]).toBeCloseTo(0.1, 12);
    expect(h[1]).toBeCloseTo(0.1, 12);
    // Wrap: t=1 maps to start.
    expect(boundaryPoint(v.outer, prm, 1)).toEqual([0, 0]);
  });
});

describe("G7D serialization + determinism", () => {
  it("panel round-trip triangulates bitwise-identically; input never mutated", () => {
    const panel: PatternPanel = {
      id: "h", outline: [{ kind: "polyline", points: [[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1]] }],
      holes: [[{ kind: "polyline", points: [[0.07, 0.03], [0.07, 0.07], [0.03, 0.07], [0.03, 0.03]] }]],
      grainAngleRad: 0.7, materialId: "denim-12oz",
    };
    const before = panelToJSON(panel);
    const a = triangulatePatternPanel(panel);
    const b = triangulatePatternPanel(panelFromJSON(panelToJSON(panel)));
    expect(panelToJSON(panel)).toBe(before); // input untouched
    expect(Array.from(b.vertices)).toEqual(Array.from(a.vertices));
    expect(Array.from(b.triangles)).toEqual(Array.from(a.triangles));
    expect(JSON.stringify(b.boundaryEdges)).toBe(JSON.stringify(a.boundaryEdges));
  });

  it("rejects malformed JSON", () => {
    expect(() => panelFromJSON("nope")).toThrowError(/invalid-json/);
    expect(() => panelFromJSON(JSON.stringify({ id: "x" }))).toThrowError(/invalid-json/);
  });
});
