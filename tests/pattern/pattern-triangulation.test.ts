import { describe, expect, it } from "vitest";
import { preprocess } from "../../src/mesh/mesh.js";
import {
  measurePanelQuality,
  panelToRestMesh,
  triangulatePatternPanel,
  triangulatedFromJSON,
  triangulatedToJSON,
  type PatternPanel,
  type TriangulatedPanel,
} from "../../src/pattern/pattern-geometry.js";

function panel(id: string, points: Array<[number, number]>, grainAngleRad = 0, materialId = "cotton-poplin"): PatternPanel {
  return {
    id,
    outline: [{ kind: "polyline", points }],
    holes: [],
    grainAngleRad,
    materialId,
  };
}

function rect(): PatternPanel {
  return panel("rect", [[0, 0], [0.1, 0], [0.1, 0.05], [0, 0.05]]);
}

function triangle(): PatternPanel {
  return panel("triangle", [[0, 0], [0.1, 0], [0, 0.1]]);
}

function lShape(): PatternPanel {
  return panel("l-shape", [[0, 0], [0.1, 0], [0.1, 0.04], [0.04, 0.04], [0.04, 0.1], [0, 0.1]]);
}

function panelWithHole(): PatternPanel {
  return {
    id: "hole",
    outline: [{ kind: "polyline", points: [[0, 0], [0.1, 0], [0.1, 0.1], [0, 0.1]] }],
    holes: [[{ kind: "polyline", points: [[0.03, 0.03], [0.07, 0.03], [0.07, 0.07], [0.03, 0.07]] }]],
    grainAngleRad: 0,
    materialId: "cotton-poplin",
  };
}

function positiveTriangleArea(tri: TriangulatedPanel, t: number): number {
  const ia = tri.triangles[3 * t] * 2;
  const ib = tri.triangles[3 * t + 1] * 2;
  const ic = tri.triangles[3 * t + 2] * 2;
  const ax = tri.vertices[ia], ay = tri.vertices[ia + 1];
  const bx = tri.vertices[ib], by = tri.vertices[ib + 1];
  const cx = tri.vertices[ic], cy = tri.vertices[ic + 1];
  return 0.5 * ((bx - ax) * (cy - ay) - (by - ay) * (cx - ax));
}

describe("G7D deterministic panel triangulation", () => {
  it("triangulates a rectangle and tags outer boundary arclength intervals", () => {
    const tri = triangulatePatternPanel(rect());
    expect(tri.vertices.length / 2).toBe(4);
    expect(tri.triangles.length / 3).toBe(2);
    expect(tri.boundaryEdges).toHaveLength(4);
    expect(tri.boundaryEdges.map((edge) => edge.loop)).toEqual(["outer", "outer", "outer", "outer"]);
    expect(tri.boundaryEdges.map((edge) => edge.edgeIndex)).toEqual([0, 1, 2, 3]);
    const intervals = tri.boundaryEdges.map(({ t0, t1 }) => [t0, t1]);
    const expected = [[0, 1 / 3], [1 / 3, 0.5], [0.5, 5 / 6], [5 / 6, 1]];
    for (let i = 0; i < expected.length; i++) {
      expect(intervals[i][0]).toBeCloseTo(expected[i][0], 12);
      expect(intervals[i][1]).toBeCloseTo(expected[i][1], 12);
    }
  });

  it("triangulates a triangle into one face with three boundary edges", () => {
    const tri = triangulatePatternPanel(triangle());
    expect(tri.vertices.length / 2).toBe(3);
    expect(tri.triangles.length / 3).toBe(1);
    expect(tri.boundaryEdges).toHaveLength(3);
  });

  it("triangulates an L-shape into four faces and tags its six outer edges", () => {
    const tri = triangulatePatternPanel(lShape());
    expect(tri.vertices.length / 2).toBe(6);
    expect(tri.triangles.length / 3).toBe(4);
    expect(tri.boundaryEdges).toHaveLength(6);
    expect(tri.boundaryEdges.every((edge) => edge.loop === "outer")).toBe(true);
  });

  it("triangulates a panel with a centered hole and preserves both boundary loops", () => {
    const tri = triangulatePatternPanel(panelWithHole());
    expect(tri.vertices.length / 2).toBe(10);
    expect(tri.triangles.length / 3).toBe(8);
    expect(tri.boundaryEdges).toHaveLength(8);
    expect(tri.boundaryEdges.filter((edge) => edge.loop === "outer")).toHaveLength(4);
    expect(tri.boundaryEdges.filter((edge) => edge.loop === 0)).toHaveLength(4);
  });

  it("satisfies Euler counts, valid indices, and positive face areas on all polygon fixtures", () => {
    for (const source of [rect(), triangle(), lShape(), panelWithHole()]) {
      const tri = triangulatePatternPanel(source);
      const vertexCount = tri.vertices.length / 2;
      expect(tri.triangles.length / 3).toBe(vertexCount - 2);
      expect(tri.triangles.every((index) => index < vertexCount)).toBe(true);
      for (let t = 0; t < tri.triangles.length / 3; t++) expect(positiveTriangleArea(tri, t)).toBeGreaterThan(0);
    }
  });

  it("reports rectangle area, boundary deviation, triangle area, and shape quality", () => {
    const source = rect();
    const tri = triangulatePatternPanel(source);
    const quality = measurePanelQuality(source, tri);
    expect(quality.panelArea).toBeCloseTo(0.005, 12);
    expect(quality.areaRelErr).toBeLessThan(1e-12);
    expect(quality.boundaryDeviation).toBeLessThan(1e-12);
    expect(quality.minTriArea).toBeCloseTo(0.0025, 12);
    expect(quality.minShapeQuality).toBeGreaterThanOrEqual(0.69);
    expect(quality.minShapeQuality).toBeLessThanOrEqual(0.70);
  });

  it("approximates a quarter-disc panel within area and boundary tolerances", () => {
    const source: PatternPanel = {
      id: "quarter-disc",
      outline: [
        { kind: "arc", center: [0, 0], radius: 0.05, a0: 0, a1: Math.PI / 2 },
        { kind: "polyline", points: [[0, 0]] },
      ],
      holes: [],
      grainAngleRad: 0,
      materialId: "cotton-poplin",
    };
    const tri = triangulatePatternPanel(source, { sagittaTol: 1e-4 });
    const quality = measurePanelQuality(source, tri);
    expect(quality.areaRelErr).toBeLessThan(5e-3);
    expect(quality.boundaryDeviation).toBeLessThan(2e-4);
    expect(quality.minTriArea).toBeGreaterThan(0);
  });

  it("bridges pattern rest coordinates into FEM preprocessing", () => {
    const tri = triangulatePatternPanel(rect());
    const rest = panelToRestMesh(tri);
    const mesh = preprocess(rest.positions, rest.uv, rest.indices, 0.15);
    const area = mesh.areas.reduce((sum, value) => sum + value, 0);
    expect(area).toBeCloseTo(0.005, 8);
    expect(mesh.hinges).toHaveLength(1);
    expect(mesh.masses.every((mass) => mass > 0)).toBe(true);
  });

  it("preserves material and grain metadata through triangulation JSON round-trip", () => {
    const source = panel("metadata", [[0, 0], [0.1, 0], [0.1, 0.05], [0, 0.05]], 0.7, "denim-12oz");
    const tri = triangulatePatternPanel(source);
    expect(tri.grainAngleRad).toBe(0.7);
    expect(tri.materialId).toBe("denim-12oz");
    const restored = triangulatedFromJSON(triangulatedToJSON(tri));
    expect(restored).toEqual(tri);
  });
});

describe("G7D collinear-vertex recovery (G16 regression)", () => {
  // Splitting a straight boundary edge inserts a 180° vertex. Depending on
  // its loop position it could strand the order-dependent ear clipper with a
  // collinear final triple ("degenerate-output"). The triangulator recovers
  // by dropping near-collinear vertices and retrying once; inputs that
  // triangulated before take the untouched fast path.
  it("triangulates straight-edge midpoints wherever they sit in the loop", () => {
    const variants = [
      panel("mid-top", [[0, 0], [0.46, 0], [0.46, 0.62], [0.23, 0.62], [0, 0.62]]),
      panel("mid-bottom", [[0, 0], [0.23, 0], [0.46, 0], [0.46, 0.62], [0, 0.62]]),
      panel("mid-left", [[0, 0], [0.46, 0], [0.46, 0.62], [0, 0.62], [0, 0.31]]),
      panel("mid-right", [[0, 0], [0.46, 0], [0.46, 0.31], [0.46, 0.62], [0, 0.62]]),
      panel("two-mids", [[0, 0], [0.46, 0], [0.46, 0.62], [0.23, 0.62], [0, 0.62], [0, 0.31]]),
    ];
    for (const source of variants) {
      const tri = triangulatePatternPanel(source);
      const quality = measurePanelQuality(source, tri);
      expect(quality.areaRelErr).toBeLessThan(1e-9);
      expect(quality.panelArea).toBeCloseTo(0.46 * 0.62, 9);
      for (let t = 0; t < tri.triangles.length / 3; t++) expect(positiveTriangleArea(tri, t)).toBeGreaterThan(0);
    }
  });

  it("recovers deterministically with identical repeated output", () => {
    const source = panel("mid-top", [[0, 0], [0.46, 0], [0.46, 0.62], [0.23, 0.62], [0, 0.62]]);
    const first = triangulatePatternPanel(source);
    const second = triangulatePatternPanel(source);
    expect(second).toEqual(first);
  });

  it("leaves non-collinear polygons on the untouched fast path", () => {
    // L-shape has no collinear vertices: output identical to pre-recovery code.
    const tri = triangulatePatternPanel(lShape());
    expect(tri.vertices.length / 2).toBe(6);
    expect(tri.triangles.length / 3).toBe(4);
  });
});
