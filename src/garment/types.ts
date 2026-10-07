import type { ClothMeshData } from "../mesh/mesh.js";

export type Vec2 = readonly [number, number];

export interface BoundaryMeshSample {
  /** Parameter along this Bezier curve, in [0, 1]. */
  t: number;
  /** Vertex index in the owning panel mesh. */
  vertex: number;
}

export interface PatternBoundaryCurve {
  id: string;
  /** Two controls describe a line; three or four describe a quadratic/cubic Bezier. */
  controlPoints: Vec2[];
  /** Ordered samples emitted by the panel mesher, including t=0 and t=1. */
  meshSamples: BoundaryMeshSample[];
}

export interface BoundaryLoop {
  id: string;
  curves: PatternBoundaryCurve[];
}

export interface PanelMesh {
  /** Initial simulation-space positions in meters, xyz triplets. */
  positions: Float32Array;
  /** Pattern-space coordinates in meters, uv pairs. */
  patternCoordinates: Float32Array;
  indices: Uint32Array;
}

export interface PatternPanel {
  id: string;
  materialAssignmentId: string;
  outerBoundary: BoundaryLoop;
  holes: BoundaryLoop[];
  /** Triangulation is supplied by the pattern mesher; holes remain explicit metadata. */
  mesh: PanelMesh;
}

export interface MaterialAssignment {
  id: string;
  /** Stable reference to a calibrated material definition owned by the material layer. */
  materialRef: string;
}

export interface SeamSide {
  panelId: string;
  boundaryLoopId: string;
  curveId: string;
}

export interface SeamCorrespondenceKnot {
  /** Normalized parameter along side A's curve. */
  sideA: number;
  /** Normalized parameter along side B's curve. */
  sideB: number;
}

export type SeamCorrespondenceMode = "one-to-one" | "segmented" | "many-to-one";

export interface SeamCorrespondence {
  mode: SeamCorrespondenceMode;
  orientation: "same" | "reversed";
  /** Piecewise-linear parameter correspondence, covering each complete side. */
  knots: SeamCorrespondenceKnot[];
}

export interface SeamAllowance {
  /** Pattern seam allowances in meters; metadata only, not simulated thickness. */
  sideA_m: number;
  sideB_m: number;
}

export interface StitchPair {
  /** Stable authored id; if omitted, a deterministic seam-local id is generated. */
  id?: string;
  parameterA: number;
  parameterB: number;
}

export interface StitchTopology {
  /** Explicit, ordered vertex-pair constraints; these are not dynamically solved here. */
  pairs: StitchPair[];
}

export interface Seam {
  id: string;
  sideA: SeamSide;
  sideB: SeamSide;
  correspondence: SeamCorrespondence;
  allowance: SeamAllowance;
  stitchTopology: StitchTopology;
}

export interface Garment {
  id: string;
  materialAssignments: MaterialAssignment[];
  panels: PatternPanel[];
  seams: Seam[];
}

export interface PanelSimulationRange {
  panelId: string;
  materialAssignmentId: string;
  vertexOffset: number;
  vertexCount: number;
  triangleOffset: number;
  triangleCount: number;
}

export interface SimulationStitchConstraint {
  id: string;
  seamId: string;
  vertexA: number;
  vertexB: number;
  parameterA: number;
  parameterB: number;
  allowance: SeamAllowance;
}

export interface GarmentSimulationMesh {
  mesh: ClothMeshData;
  panelRanges: PanelSimulationRange[];
  /** Triangle-aligned material assignment ids in simulation-mesh order. */
  triangleMaterialAssignments: string[];
  /** Explicit topology only; no dynamic sewing behavior is executed. */
  stitches: SimulationStitchConstraint[];
}
