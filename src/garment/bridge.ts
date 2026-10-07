import { preprocess } from "../mesh/mesh.js";
import type { Hinge } from "../physics/types.js";
import {
  canonicalGarment,
  deterministicStitchId,
  mapSeamParameter,
} from "./garment.js";
import type {
  Garment,
  GarmentSimulationMesh,
  PanelMesh,
  PatternBoundaryCurve,
  PatternPanel,
  Seam,
  SeamSide,
  SimulationStitchConstraint,
} from "./types.js";

export type ArealDensityResolver = (materialRef: string, assignmentId: string) => number;

interface ResolvedCurve {
  panel: PatternPanel;
  curve: PatternBoundaryCurve;
}

function resolveBoundaryCurve(garment: Garment, side: SeamSide): ResolvedCurve {
  const panel = garment.panels.find((item) => item.id === side.panelId)!;
  const loop = [panel.outerBoundary, ...panel.holes].find((item) => item.id === side.boundaryLoopId)!;
  return { panel, curve: loop.curves.find((item) => item.id === side.curveId)! };
}

function nearestSampleVertex(curve: PatternBoundaryCurve, parameter: number, seamId: string): number {
  const sample = curve.meshSamples.find((item) => Math.abs(item.t - parameter) <= 1e-8);
  if (!sample) throw new Error(`seam '${seamId}' parameter ${parameter} has no boundary mesh vertex`);
  return sample.vertex;
}

function panelMeshArrays(panel: PatternPanel): PanelMesh {
  return panel.mesh;
}

function offsetHinge(hinge: Hinge, offset: number): Hinge {
  return {
    ...hinge,
    v0: hinge.v0 + offset,
    v1: hinge.v1 + offset,
    v2: hinge.v2 + offset,
    v3: hinge.v3 + offset,
  };
}

function createStitches(
  garment: Garment,
  vertexOffsets: Map<string, number>,
): SimulationStitchConstraint[] {
  const stitches: SimulationStitchConstraint[] = [];
  for (const seam of garment.seams) {
    const sideA = resolveBoundaryCurve(garment, seam.sideA);
    const sideB = resolveBoundaryCurve(garment, seam.sideB);
    for (let i = 0; i < seam.stitchTopology.pairs.length; i++) {
      const pair = seam.stitchTopology.pairs[i];
      const id = pair.id ?? deterministicStitchId(seam.id, i);
      stitches.push({
        id,
        seamId: seam.id,
        vertexA: vertexOffsets.get(sideA.panel.id)! + nearestSampleVertex(sideA.curve, pair.parameterA, seam.id),
        vertexB: vertexOffsets.get(sideB.panel.id)! + nearestSampleVertex(sideB.curve, pair.parameterB, seam.id),
        parameterA: pair.parameterA,
        parameterB: pair.parameterB,
        allowance: { ...seam.allowance },
      });
    }
  }
  return stitches;
}

function validateStitchAgreement(seam: Seam): void {
  for (const [index, pair] of seam.stitchTopology.pairs.entries()) {
    const expectedB = mapSeamParameter(seam.correspondence, pair.parameterA);
    if (Math.abs(expectedB - pair.parameterB) > 1e-6) {
      throw new Error(`seam '${seam.id}' stitch ${index} disagrees with its correspondence`);
    }
  }
}

/**
 * Joins already-triangulated panels into the existing solver mesh format.
 * Stitches are emitted as topology records only; no vertices are welded and
 * no dynamic sewing constraints are applied.
 */
export function buildGarmentSimulationMesh(
  garment: Garment,
  resolveArealDensityKgM2: ArealDensityResolver,
): GarmentSimulationMesh {
  const canonical = canonicalGarment(garment);
  for (const seam of canonical.seams) validateStitchAgreement(seam);

  const totalVertices = canonical.panels.reduce((sum, panel) => sum + panel.mesh.positions.length / 3, 0);
  const totalTriangles = canonical.panels.reduce((sum, panel) => sum + panel.mesh.indices.length / 3, 0);
  const positions = new Float32Array(totalVertices * 3);
  const patternCoordinates = new Float32Array(totalVertices * 2);
  const indices = new Uint32Array(totalTriangles * 3);
  const invDm = new Float32Array(totalTriangles * 4);
  const areas = new Float32Array(totalTriangles);
  const masses = new Float32Array(totalVertices);
  const hinges: Hinge[] = [];
  const panelRanges: GarmentSimulationMesh["panelRanges"] = [];
  const triangleMaterialAssignments: string[] = [];
  const vertexOffsets = new Map<string, number>();
  let vertexOffset = 0;
  let triangleOffset = 0;

  const assignments = new Map(canonical.materialAssignments.map((assignment) => [assignment.id, assignment]));
  for (const panel of canonical.panels) {
    const local = panelMeshArrays(panel);
    const vertexCount = local.positions.length / 3;
    const triangleCount = local.indices.length / 3;
    const assignment = assignments.get(panel.materialAssignmentId)!;
    const density = resolveArealDensityKgM2(assignment.materialRef, assignment.id);
    if (!Number.isFinite(density) || density <= 0) {
      throw new Error(`material assignment '${assignment.id}' needs positive areal density in kg/m^2`);
    }
    const processed = preprocess(local.positions, local.patternCoordinates, local.indices, density);

    positions.set(local.positions, vertexOffset * 3);
    patternCoordinates.set(local.patternCoordinates, vertexOffset * 2);
    for (let i = 0; i < local.indices.length; i++) indices[triangleOffset * 3 + i] = local.indices[i] + vertexOffset;
    invDm.set(processed.invDm, triangleOffset * 4);
    areas.set(processed.areas, triangleOffset);
    masses.set(processed.masses, vertexOffset);
    hinges.push(...processed.hinges.map((hinge) => offsetHinge(hinge, vertexOffset)));
    triangleMaterialAssignments.push(...Array.from({ length: triangleCount }, () => assignment.id));
    panelRanges.push({
      panelId: panel.id,
      materialAssignmentId: assignment.id,
      vertexOffset,
      vertexCount,
      triangleOffset,
      triangleCount,
    });
    vertexOffsets.set(panel.id, vertexOffset);
    vertexOffset += vertexCount;
    triangleOffset += triangleCount;
  }

  return {
    mesh: {
      count: totalVertices,
      triCount: totalTriangles,
      positions: Float32Array.from(positions),
      restPositions: Float32Array.from(positions),
      uv: Float32Array.from(patternCoordinates),
      indices,
      invDm,
      areas,
      masses,
      hinges,
    },
    panelRanges,
    triangleMaterialAssignments,
    stitches: createStitches(canonical, vertexOffsets),
  };
}
