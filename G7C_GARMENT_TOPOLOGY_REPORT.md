# G7C Garment Topology + Seam Graph

**Status:** Implemented on `feature/g7c-garment-topology` in an isolated worktree. The G0-G6 solver, FEM/contact implementation, and numerical behavior are unchanged.

## Data model

`src/garment/types.ts` defines `Garment`, `PatternPanel`, closed outer/hole boundary loops, line/quadratic/cubic Bezier curves, panel meshes, material assignments, `SeamSide`, piecewise correspondence knots, seam allowances, and explicit stitch pairs. Pattern and allowance distances are in meters. Material assignments reference calibrated material IDs; artistic controls and calibrated physical parameters are not stored in the garment topology.

The validator checks globally unique entity IDs, boundary closure and curve continuity, boundary-mesh edge/sample agreement, finite panel mesh data, valid material references, seam-side references, monotonic full-edge mappings, correspondence orientation, allowance units/ranges, explicit stitch samples, and stitch-to-correspondence consistency. Duplicate unoriented seam pairs are rejected. One-to-one, segmented, reversed, and many-to-one mappings are represented with ordered normalized-parameter knots.

## Simulation bridge

`buildGarmentSimulationMesh` accepts pre-triangulated panel meshes and a caller-supplied resolver for areal density in kg/m². It canonicalizes panel/seam ordering, builds the existing `ClothMeshData` representation with panel/material ranges, and emits stable global stitch vertex pairs and seam-allowance metadata. Panel vertices remain separate: the bridge does not weld seams or execute dynamic sewing constraints. Hole loops remain first-class metadata; the supplied panel triangulation is responsible for preserving openings.

## Verification

`tests/garment.test.ts`: **6 passed**. Coverage includes closed/open boundaries, holes, curved curves, reversed orientation, segmented and many-to-one maps, seam/stitch consistency, duplicate seam rejection, material assignments, stable stitch IDs, order-independent bridge output, and typed-array serialization round trips.

The G7C source API passed an isolated strict TypeScript check. A full project build was not available from the fresh worktree because its dependencies are not installed there; the repository-wide check against the primary checkout also encounters the pre-existing `gpu-solver.ts` reference to undefined `adaptiveBatchK`. No dependency manifests were changed.

## Scope boundary

This is a data and conversion layer only. Pattern triangulation remains an input responsibility for the later pattern-geometry work; no solver internals, FEM/contact mathematics, renderer, editor, or dynamic sewn simulation were added. The model follows the panel/seam/topology-first direction identified by PatternGSL and ReWeaver without treating their research descriptions as a substitute for implementation validation.
