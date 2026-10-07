import { describe, expect, it } from "vitest";
import {
  buildGarmentSimulationMesh,
  canonicalGarment,
  deserializeGarment,
  deterministicStitchId,
  GarmentValidationError,
  mapSeamParameter,
  serializeGarment,
  validateGarment,
} from "../src/garment/index.js";
import type {
  BoundaryLoop,
  Garment,
  PatternPanel,
  Seam,
} from "../src/garment/index.js";

function panel(id: string, xOffset: number, assignmentId = "cotton-assignment"): PatternPanel {
  const points: Array<[number, number]> = [
    [0, 0], [1, 0], [1, 0.5], [1, 1], [0.5, 1], [0, 1], [0, 0.5], [0.5, 0.5],
  ];
  const curve = (
    curveId: string,
    start: number,
    end: number,
    samples: Array<{ t: number; vertex: number }> = [{ t: 0, vertex: start }, { t: 1, vertex: end }],
    controlPoints?: Array<[number, number]>,
  ) => ({
    id: curveId,
    controlPoints: controlPoints ?? [points[start], points[end]],
    meshSamples: samples,
  });
  const outerBoundary: BoundaryLoop = {
    id: `${id}-outer`,
    curves: [
      curve(`${id}-bottom`, 0, 1),
      curve(`${id}-right`, 1, 3, [
        { t: 0, vertex: 1 }, { t: 0.5, vertex: 2 }, { t: 1, vertex: 3 },
      ]),
      curve(`${id}-top`, 3, 5, [
        { t: 0, vertex: 3 }, { t: 0.5, vertex: 4 }, { t: 1, vertex: 5 },
      ]),
      curve(`${id}-left`, 5, 0, [
        { t: 0, vertex: 5 }, { t: 0.5, vertex: 6 }, { t: 1, vertex: 0 },
      ]),
    ],
  };
  const indices: number[] = [];
  for (let i = 0; i < 7; i++) indices.push(i, (i + 1) % 7, 7);
  const positions = new Float32Array(points.length * 3);
  const patternCoordinates = new Float32Array(points.length * 2);
  for (let i = 0; i < points.length; i++) {
    positions[i * 3] = points[i][0] + xOffset;
    positions[i * 3 + 2] = points[i][1];
    patternCoordinates[i * 2] = points[i][0];
    patternCoordinates[i * 2 + 1] = points[i][1];
  }
  return {
    id,
    materialAssignmentId: assignmentId,
    outerBoundary,
    holes: [],
    mesh: {
      positions,
      patternCoordinates,
      indices: new Uint32Array(indices),
    },
  };
}

function baseGarment(): Garment {
  const left = panel("left-panel", 0);
  const right = panel("right-panel", 1, "lining-assignment");
  const seam: Seam = {
    id: "side-seam",
    sideA: { panelId: left.id, boundaryLoopId: left.outerBoundary.id, curveId: "left-panel-right" },
    sideB: { panelId: right.id, boundaryLoopId: right.outerBoundary.id, curveId: "right-panel-left" },
    correspondence: {
      mode: "one-to-one",
      orientation: "reversed",
      knots: [{ sideA: 0, sideB: 1 }, { sideA: 1, sideB: 0 }],
    },
    allowance: { sideA_m: 0.01, sideB_m: 0.01 },
    stitchTopology: {
      pairs: [
        { parameterA: 0, parameterB: 1 },
        { parameterA: 0.5, parameterB: 0.5 },
        { parameterA: 1, parameterB: 0 },
      ],
    },
  };
  return {
    id: "sample-garment",
    materialAssignments: [
      { id: "cotton-assignment", materialRef: "fabric:cotton-plain" },
      { id: "lining-assignment", materialRef: "fabric:cotton-lining" },
    ],
    panels: [left, right],
    seams: [seam],
  };
}

function panelWithHole(): PatternPanel {
  const points: Array<[number, number]> = [
    [0, 0], [2, 0], [2, 2], [0, 2],
    [0.75, 0.75], [1.25, 0.75], [1.25, 1.25], [0.75, 1.25],
  ];
  const loop = (id: string, vertexIds: number[]): BoundaryLoop => ({
    id,
    curves: vertexIds.map((start, i) => {
      const end = vertexIds[(i + 1) % vertexIds.length];
      return {
        id: `${id}-curve-${i}`,
        controlPoints: [points[start], points[end]],
        meshSamples: [{ t: 0, vertex: start }, { t: 1, vertex: end }],
      };
    }),
  });
  const indices = new Uint32Array([
    0, 1, 5, 0, 5, 4,
    1, 2, 6, 1, 6, 5,
    2, 3, 7, 2, 7, 6,
    3, 0, 4, 3, 4, 7,
  ]);
  const positions = new Float32Array(points.length * 3);
  const patternCoordinates = new Float32Array(points.length * 2);
  for (let i = 0; i < points.length; i++) {
    positions[i * 3] = points[i][0];
    positions[i * 3 + 2] = points[i][1];
    patternCoordinates[i * 2] = points[i][0];
    patternCoordinates[i * 2 + 1] = points[i][1];
  }
  return {
    id: "panel-with-hole",
    materialAssignmentId: "cotton-assignment",
    outerBoundary: loop("hole-panel-outer", [0, 1, 2, 3]),
    holes: [loop("hole-panel-opening", [4, 5, 6, 7])],
    mesh: { positions, patternCoordinates, indices },
  };
}

describe("G7C garment topology", () => {
  it("validates closed panel boundaries, holes, and curved boundary curves", () => {
    const garment = baseGarment();
    expect(() => validateGarment({
      id: "hole-garment",
      materialAssignments: garment.materialAssignments,
      panels: [panelWithHole()],
      seams: [],
    })).not.toThrow();
    const curved = garment.panels[0].outerBoundary.curves.find((item) => item.id === "left-panel-right")!;
    curved.controlPoints = [[1, 0], [1.1, 0.5], [1, 1]];
    garment.panels[0].mesh.patternCoordinates[2 * 2] = 1.05;
    expect(() => validateGarment(garment)).not.toThrow();
  });

  it("rejects an open or disconnected boundary loop", () => {
    const garment = baseGarment();
    garment.panels[0].outerBoundary.curves[1].controlPoints[0] = [1.1, 0];
    expect(() => validateGarment(garment)).toThrow(/not geometrically closed\/continuous/);
  });

  it("maps reversed edge orientation and emits explicit global stitch topology", () => {
    const result = buildGarmentSimulationMesh(baseGarment(), (materialRef) =>
      materialRef === "fabric:cotton-plain" ? 0.18 : 0.3);
    expect(mapSeamParameter(baseGarment().seams[0].correspondence, 0.25)).toBeCloseTo(0.75);
    expect(result.mesh.count).toBe(16);
    expect(result.panelRanges).toHaveLength(2);
    expect(result.panelRanges.map((range) => range.materialAssignmentId))
      .toEqual(["cotton-assignment", "lining-assignment"]);
    expect(result.triangleMaterialAssignments).toEqual([
      ...Array(7).fill("cotton-assignment"),
      ...Array(7).fill("lining-assignment"),
    ]);
    expect(result.stitches.map((stitch) => [stitch.vertexA, stitch.vertexB])).toEqual([
      [1, 8], [2, 14], [3, 13],
    ]);
    expect(result.stitches[0].allowance).toEqual({ sideA_m: 0.01, sideB_m: 0.01 });
    expect(result.triangleMaterialAssignments).toHaveLength(result.mesh.triCount);
    expect(result.stitches.map((stitch) => stitch.id)).toEqual([
      deterministicStitchId("side-seam", 0),
      deterministicStitchId("side-seam", 1),
      deterministicStitchId("side-seam", 2),
    ]);
    const shuffled = { ...baseGarment(), panels: [...baseGarment().panels].reverse() };
    const reordered = buildGarmentSimulationMesh(shuffled, () => 0.18);
    expect([...reordered.mesh.indices]).toEqual([...result.mesh.indices]);
    expect(reordered.stitches).toEqual(result.stitches);
  });

  it("supports segmented correspondence and many-to-one metadata", () => {
    const garment = baseGarment();
    const seam = garment.seams[0];
    seam.correspondence = {
      mode: "segmented",
      orientation: "same",
      knots: [{ sideA: 0, sideB: 0 }, { sideA: 0.5, sideB: 0.25 }, { sideA: 1, sideB: 1 }],
    };
    seam.stitchTopology.pairs = [
      { parameterA: 0, parameterB: 0 },
      { parameterA: 1, parameterB: 1 },
    ];
    expect(() => validateGarment(garment)).not.toThrow();
    expect(mapSeamParameter(seam.correspondence, 0.5)).toBe(0.25);

    seam.correspondence = {
      mode: "many-to-one",
      orientation: "same",
      knots: [{ sideA: 0, sideB: 0 }, { sideA: 0.5, sideB: 0 }, { sideA: 1, sideB: 1 }],
    };
    seam.sideB.curveId = "right-panel-right";
    seam.stitchTopology.pairs = [
      { parameterA: 0, parameterB: 0 },
      { parameterA: 0.5, parameterB: 0 },
      { parameterA: 1, parameterB: 1 },
    ];
    expect(() => validateGarment(garment)).not.toThrow();
  });

  it("rejects inconsistent stitch maps and duplicate seam boundaries", () => {
    const garment = baseGarment();
    garment.seams[0].stitchTopology.pairs[1].parameterB = 0;
    expect(() => validateGarment(garment)).toThrow(/disagrees with its parameter correspondence/);

    const duplicate = baseGarment();
    duplicate.seams.push({ ...duplicate.seams[0], id: "duplicate-seam" });
    expect(() => validateGarment(duplicate)).toThrow(/duplicate seam between/);
  });

  it("assigns deterministic stitch ids and round-trips canonical serialization", () => {
    const garment = baseGarment();
    const serializedA = serializeGarment(garment);
    const shuffled = { ...garment, panels: [...garment.panels].reverse() };
    expect(serializeGarment(shuffled)).toBe(serializedA);
    const loaded = deserializeGarment(serializedA);
    expect(loaded.panels[0].mesh.positions).toBeInstanceOf(Float32Array);
    expect(serializeGarment(loaded)).toBe(serializedA);
    expect(canonicalGarment(loaded).seams[0].stitchTopology.pairs[0].id)
      .toBe(deterministicStitchId("side-seam", 0));
    expect(() => deserializeGarment('{"id":"broken"}')).toThrow(GarmentValidationError);
  });
});
