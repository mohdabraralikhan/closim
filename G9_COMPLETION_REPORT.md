# G9 — Professional 2D Pattern CAD (tracks B/C/D/E + integration)

G9A (geometry & editing engine, `src/cad/{geom,queries,history,selection,snap,ops}.ts`,
153 tests) pre-existed. This round adds G9B, G9C, G9D, G9E, and final integration.
No solver, FEM, collision, GPU, G8A, or G8B file was modified; the only edit to a
pre-existing file is three additive gesture passthroughs on `CadSession`
(`beginGesture`/`endGesture`/`inGesture` in `src/cad/history.ts`).

## Interfaces reused

- G8A `PatternDocument` + kernel ops/validation/serialization (`src/pattern/cad.ts`).
- G9A `geom` (predicates, transforms), `queries` (resolve/measure/hit/box),
  `ops` (delete/merge/trim/extend/mirror/split), `history.CadSession`,
  `selection`, `snap.snapPosition`.
- G8B `resolveStitchPairs` (seam-length measurement), G8 `assembleGarment`,
  `createGarmentProject`/`rebuildGarment`/`applyPatternEdit`, `runFitting` + `CpuSolver`.

## Files added

- `src/cad/draft.ts` — G9B: `draftLine`, `lineAtAngle`, `perpendicularAt`,
  `parallelThrough`, `midpointPoint`, `divideSegment`, `intersectionPoint`,
  `extendSegmentByDistance`, `trimSegmentToSegment`, `offsetLoop` (miter,
  new panel, validation-gated), `buildPanelFromRing`,
  `translate/rotate/mirrorPanelGeometry`, `arrangePanels`,
  `splitPanelByLine` / `joinPanelsAtSharedEdge` (fresh ids + explicit
  `invalidatedSeamIds`), `invalidatedSeamIdsForPanels`.
- `src/cad/constraints.ts` — G9C: metre-canonical model; `ConstraintSet`
  sidecar (stable ids, enabled flags) for distance / fixed-length /
  equal-length / horizontal / vertical / parallel / perpendicular / angle;
  sequential-projection `solveConstraints` with residuals + `unsatisfied`
  reporting; measurements (distance, edge, angle, area, perimeter, G8B seam
  length); `formatLength`/`parseLengthToM` (mm/cm/m/in, storage untouched);
  canonical sidecar + pattern/constraint envelope persistence.
- `src/cad/editor.ts` — G9D (headless): viewport math
  (`worldToScreen`/`screenToWorld`/`panByPixels`/`zoomAt`/`fitToView`),
  `EditorSession` pointer/keyboard state machine (11 tools, drag gestures via
  history coalescing, box select, shift multi-select, middle/space pan, wheel
  zoom, shortcuts, mm numeric input, Escape/right-click cancel priority,
  Delete, save/load with constraints, `getStatus`, `validateForAssembly`).
  No simulation runs on pointer movement by construction.
- `src/cad/index.ts` — barrel now also exports draft/constraints/editor.
- Tests: `tests/cad/draft.test.ts` (20), `constraints.test.ts` (18),
  `editor.test.ts` (17), `g9-adversarial.test.ts` (14, G9E),
  `g9-workflow.test.ts` (1, 19-step acceptance).

## Verification

- `tests/cad + tests/pattern + tests/garment + tests/view`: **29 files /
  378 tests pass** (G9 scope: 22 files / 305; G10 view track: 7 files / 73).
- 19-step acceptance (in-app T-shirt: draw → draft → mirror → measure →
  constrain → validate → save → reload → seam → assemble → simulate → edit →
  reassemble → re-simulate) passes; 31 welds, 1 component, NaN-free both runs.
- `npx tsc --noEmit`: exit 0, no diagnostics, joint tree.

## Defects found and fixed (own scope)

1. `leftNormal` is not unit-length — perpendicular/offset distances were
   scaled by edge length. Now normalized at use sites.
2. `joinPanelsAtSharedEdge` slice kept duplicated junction points
   (zero-length closing edge) — now keeps chainA whole + chainB interior only.
3. Editor select-drag never opened the history gesture (every mousemove its
   own undo entry) — gesture now opens on first drag move and coalesces.
4. Test-premise errors (Float32/exact equality, duplicate-seam semantics,
   editor split = segment split, 1e-6² ring below the area floor, identical
   rings correctly refusing join) — corrected in tests.

## Assumptions / limits

- Sewing projection is correspondence + placement (G8), not solver constraints.
- Constraint solver is sequential projection (max 100 iters, 1e-9 tol):
  contradictions return `unsatisfied`, never forced geometry.
- Arc support: divide/intersection/angle re-aim are arc-safe (rigid);
  loop offset, panel split/join are line-only and say so explicitly.
- G9D is headless by necessity (repo has no frontend); a renderer binds
  pixels to `EditorSession` methods with no geometry changes.
