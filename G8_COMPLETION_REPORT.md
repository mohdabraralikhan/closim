# G8 Completion Report — Construction, Fitting, Project Format, QA, Integration

## Scope

G8A (pattern CAD kernel, `src/pattern/cad.ts`) and G8B (sewing,
`src/garment/sewing.ts`) pre-existed. This round adds G8C, G8D, G8E, and the
final integration. No solver, FEM, collision, or GPU file was modified
(`git diff` touches only pre-existing uncommitted G6C WebGPU work).

## Interfaces reused (no new geometry/physics frameworks)

- G7D `triangulateCadPanel` / `validatePanel` (`src/pattern/pattern-geometry.ts`) — triangulation input only.
- G8A document/validation/ops (`src/pattern/cad.ts`) — source of truth for panels, IDs, transforms.
- G8B `validateSeams` / `resolveStitchPairs` (`src/garment/sewing.ts`) — stitch correspondence.
- `preprocess` (`src/mesh/mesh.ts`), `createScene` (`src/physics/scene.ts`),
  `ContactSystem.setStaticMesh` (`src/collision/contact-assembly.ts`),
  `ClothSolver` (`src/backend/solver.ts` + `CpuSolver`) — solver handoff.
- `closestPointVertexTriangle` (`src/collision/closest-point.ts`) — diagnostics only.

## Files added

- `src/garment/avatar.ts` — deterministic capsule/box avatar specs, static-mesh
  merge, signed-distance diagnostics (tilted-ray Möller–Trumbore).
- `src/garment/assembly.ts` — G8C: panel triangulation + yaw/translation
  placement, weld map, connectivity, strain/penetration/NaN diagnostics,
  `createFittingScene`, staged `runFitting`.
- `src/garment/project.ts` — G8D: versioned `GarmentProject` (schema 1),
  canonical-JSON serialize, validation, `rebuildGarment`, `applyPatternEdit`.
- `src/garment/tshirt.ts` — deterministic 4-panel / 5-seam T-shirt builder.
- `examples/g8-tshirt.ts` — runnable end-to-end demo (`npx tsx examples/g8-tshirt.ts`).
- `tests/garment/g8-assembly.test.ts` (13), `g8-project.test.ts` (7),
  `g8-vertical-slice.test.ts` (17), `g8-tshirt.test.ts` (1).

## Verification

- `npx tsc --noEmit` — clean.
- `tests/garment` — 5 files / 44 tests pass (incl. pre-existing `sewing.test.ts`).
- Regression: pattern+garment+cad+material+smoke+determinism+static-collider+floor-contact
  — 39 files / 285 pass; solver unit batch — 60 files / 300 pass;
  main-tree adversarial a01–a05 pass. (Two failures exist only in the stale
  `.worktrees/validation-hardening` mirror — pre-existing GPU buffer-usage bug,
  untouched by G8.)
- Demo output: 8 tris, 31 welds, 1 component, 0 penetrating verts,
  15/15 fitting steps ok, NaN-free, edit→rebuild→re-simulate loop works.

## Defects found and fixed during QA (all in new G8 code)

1. Triangle-index stride bug in assembly strain/connectivity loops
   (`indices[t*3]` with `t += 3`) — skipped every triangle after the first.
2. Axis-aligned inside/outside ray grazed shared tessellation diagonals and
   flipped the sign — replaced with a fixed tilted direction.
3. Test-expectation errors (Float32 rounding vs exact equality; duplicate-seam
   semantics; avatar radius vs panel layout) — corrected in tests, not production code.

## Assumptions / limits

- Sewing is resolved as placement + weld correspondence, NOT solver
  constraints: panels start co-located along seams; the solver sees one merged
  mesh. True sewn-constraint forces are post-G8 work.
- Avatar is an analytic stand-in (capsule/box), not a body scan; thickness is
  metadata until a collision API consumes it.
- GPU/CPU parity for fitting reuses the existing `ClothSolver` contract and
  was exercised on CPU; GPU fitting parity runs wherever the GPU suite runs.
