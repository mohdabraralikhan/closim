# G8A Pattern CAD Kernel

## Pre-implementation interface audit

Branch: `feature/g8a-pattern-cad`.

The initial repository state passed `npx.cmd tsc --noEmit`. The full Vitest suite passed **71 files / 345 tests** in 472.04 seconds. This records the regression baseline before G8A code changes.

### Interfaces to reuse

- **G7D pattern geometry** — `src/pattern/pattern-geometry.ts` currently defines the triangulation-facing `PatternPanel` (stable panel ID, ordered line/polyline and circular arc primitives, holes, grain angle, material ID), `validatePanel`, `triangulatePatternPanel`, `measurePanelQuality`, `panelToRestMesh`, and JSON conversions. The triangulator returns deterministic vertices, triangles, and boundary tags (`loop`, `edgeIndex`, and normalized arclength `t0/t1`). The rest bridge emits `(x, 0, y)` positions and pattern-space UVs.
- **Existing FEM mesh preprocessing** — `src/mesh/mesh.ts` exposes `preprocess(positions, uv, indices, arealDensityKgM2): ClothMeshData`. G8A must leave this data model and solver preprocessing unchanged.
- **G7C construction API** — available in the separate `feature/g7c-garment` worktree at `C:\Users\Abrar\Desktop\closim-g7c-garment\src\garment`. Its document uses stable IDs for panels, boundary loops, curves, material assignments, seams and seam sides. `SeamSide` resolves through panel ID + boundary-loop ID + curve ID; `SeamCorrespondence` holds deterministic parameter knots; `validateGarment`, `canonicalGarment`, `serializeGarment`, `deserializeGarment`, and `buildGarmentSimulationMesh` provide validation, persistence, and simulation-mesh conversion. The adapter returns panel ranges, triangle material assignments, and explicit stitch pairs.
- **G7A material abstraction** — `src/physics/material-model.ts` defines calibrated physical parameters and the `ClothMaterialModel` abstraction. The existing solver continues to consume the legacy `ClothMaterial` representation.
- **G7E avatar collider adapter** — present in `.worktrees/g7e-avatar-collider/src/collision/avatar-collider.ts`. It provides rigid transforms and motion sampling, collider geometry validation, `toCollisionObject`, and `attachCollisionObject`, which reaches the existing public static-mesh collision boundary.
- **Solver boundary** — `src/backend/solver.ts` defines `ClothSolver.initialize(scene)`, `step(dt)`, material/pinning methods, and state getters. `CpuSolver` and the WebGPU solver remain authoritative; G8A has no reason to edit them.

### G8A boundary

The G7D `PatternPanel` is triangulation input, not an editable CAD document: its point coordinates are embedded in primitive arrays, it does not expose point/segment IDs, and it does not distinguish construction lines from panel boundaries. G8A will add a CAD document/entity layer and convert its ordered boundary loops to the existing G7D panel type only when validating or triangulating. Construction entities will remain outside that conversion. Triangulation remains derived data, never the document source of truth.

### Working-tree preservation

The starting working tree contained unrelated G6C WebGPU edits, G7D/G7A source and tests, reports, and presentation artifacts. These were left intact. No solver, FEM, collision, or existing G7D geometry files were changed during preflight.

## G8A implementation result

The G8A source of truth is `src/pattern/cad.ts`. It models editable CAD entities separately from G7D triangulation data. The document contains stable panel, loop, point, and segment IDs; line and circular-arc boundary geometry; construction geometry; dimensions and constraints; panel transforms; and serialization. Triangulation is generated through an adapter to G7D and is never written back into the CAD document.

Kernel operations cover panel and entity creation, point/panel movement, pivot rotation and combined transforms, explicit-intent scaling, boundary insertion/splitting, winding reversal, panel duplication and deletion, measurements, JSON round-trips, and structured document validation. Arc definitions remain authored arcs; deterministic approximation is used only when passing geometry to the polygon triangulator. Validation reports malformed boundaries, bad references and IDs, degenerate geometry, self-intersections, invalid winding, and related geometry errors without silently repairing the document.

### Files added for G8A

- `src/pattern/cad.ts` — pattern CAD model, operations, conversion, validation, and serialization.
- `tests/pattern/pattern-cad.test.ts` — 15 focused operation, validation, pathological-geometry, and determinism tests.
- `G8A_PATTERN_CAD_REPORT.md` — interface audit and implementation report.

Other pre-existing working-tree files were preserved and are not included in this G8A file list.

### Verification

- `npx.cmd tsc --noEmit` — passed with no output.
- `npx.cmd vitest run tests/pattern/ --reporter=basic` — **3 files / 37 tests passed**.
- `npx.cmd vitest run --exclude '**/.worktrees/**' --reporter=dot` — **72 files / 360 tests passed**, duration 1,187.14 seconds. This run includes pre-existing G6C, material, and other uncommitted tests present in the workspace.

### Addendum — arc split/insert completion

- `insertBoundaryPoint` and `splitBoundarySegment` in `src/pattern/cad.ts` now
  support circular-arc segments: split at parameter `t` places the point at
  `startAngle + t·sweep` with child sweeps `t·sweep` / `(1−t)·sweep` sharing
  the authored center; insertion projects the requested position onto the arc,
  rejects off-arc/endpoint positions with structured `unsupported-operation`
  errors, and snaps to the exact circle so the authored arc stays
  radius-consistent for validation.
- `tests/pattern/pattern-cad.test.ts` now has **16 tests** (was 15), adding
  arc-split midpoint/sweep assertions, on-arc insertion, off-arc and
  endpoint rejection, arc-split determinism, and post-split triangulation.
- Verification: `npx tsc --noEmit` clean; `tests/pattern/` **3 files /
  38 tests passed** (22 G7D + 16 G8A); regression spot-check `tests/cad +
  tests/garment + tests/determinism.test.ts + tests/smoke.test.ts`
  **20 files / 174 tests passed**. No solver, FEM, collision, or G7D file
  modified.
- Known limitation (pre-existing, out of G8A scope): triangulating curved
  panels with the default fine sagitta (`1e-5`, ~350-gon on a unit
  semicircle) can stall the G7D ear-clipper — this also reproduces on the
  unsplit D-shape. Curved-panel triangulation tests pass an explicit coarse
  `sagittaTol` (e.g. `0.002`), matching the pre-existing arc test.

### Remaining G8 work

This report closes G8A only. G8B construction/seam graph, G8C fitting and solver handoff, G8D native garment project format, G8E vertical-slice/adversarial QA, and the end-to-end editable garment demonstration remain required before the product milestone. The roadmap's first sellable digital garment asset depends on completing those construction, fitting, edit/rebuild, and export workflows; G8A alone is not that milestone.
