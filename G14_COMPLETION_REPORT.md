# G14 — Marker Making, Fabric Nesting & Manufacturing Optimization

G13-equivalent export (G11E) and G12 grading pre-existed. This round adds
`src/marker/` + `tests/marker/` (33 tests). No solver, FEM, collision, GPU,
grading, CAD, or view file was modified.

## Interfaces reused

- G12 `deriveSize` (per-size `PatternDocument`s; stable entity IDs) and
  `tests/grading/fixtures` for graded-garment tests.
- G9A `sampleLoopLocal`, `pointInPolygon`, `signedArea`-class predicates;
  `pattern-geometry.distToSegment`; kernel `PatternCadError` taxonomy.
- Export idiom (canonical JSON, R12 LINE/CIRCLE/TEXT writer shape) follows
  G11E; marker DXF/JSON are separate artifacts by design.

## Files added

- `src/marker/model.ts` — G14A: Fabric/Roll, CutPlan/CutItem (quantities
  live here, never in the pattern), MarkerPiece expansion (mirror-required
  tracked in instance ids), NestingConstraint, Marker, structural validation,
  canonical persistence.
- `src/marker/nest.ts` — G14B: deterministic bottom-left-fill (size sort →
  rule orientations → extent candidates → validity → length/width/seeded
  tie-break scoring); eps-robust overlap (strict crossing + strict
  containment, touching allowed) and gap predicates; `placedPolygon` and
  full `auditPlacements` (bounds/overlap/spacing/orientation/grain).
- `src/marker/fabric.ts` — G14C: `FabricRules` (usable width, selvedge,
  grain axis, directional, repeat snapping, defaults), item-over-fabric
  precedence, directional 0°/180° filtering, modulo-180° grain checks,
  centroid rotation, pre-nesting feasibility (too-wide caught before search).
- `src/marker/optimize.ts` — G14D: documented metric definitions,
  fully-placed-first comparison with min-length/max-utilization/min-waste/
  weighted objectives, deterministic multi-seed search with cancellation,
  explicit-price cost model, benchmark records.
- `src/marker/workspace.ts` — G14E: headless `MarkerWorkspace` (optimize,
  gated manual move/rotate, restore-auto, selection/inspect, undo/redo,
  save/load, live preview + audit, settings re-audit); `rebuildMarker` for
  quantity/fabric changes.
- `src/marker/export.ts` — marker JSON package + own R12 DXF writer
  (MARKER_BOUND/PIECES/LABELS, mm); package entrypoint refuses unplaced
  quantities and audit failures.
- `src/marker/index.ts` — barrel.
- Tests: `model` (5), `nest` (14: rules/predicates/engine/failures/graded),
  `workflow` (13: metrics/comparison/cost/multi-seed/workspace/export),
  `g14-workflow` (1 × full commercial acceptance).

## Verification

- Scope gate: marker + grading + cad + pattern + garment + view → **42
  files / 496 tests pass** (463 pre-G14 + 33 new), 0 failures.
- `npx tsc --noEmit` exit 0.
- Acceptance (`g14-workflow`): graded S/M/L two-panel garment → 5 instances
  → fabric/rules/feasibility → optimize (3 seeds × 2 strategies, all placed)
  → hand-verified utilization math → cost → gated manual edits → re-audit →
  marker JSON + DXF → master pattern byte-identical → deterministic rerun.
- Benchmarks: engine runs are millisecond-scale on test garments
  (iterations counted per run and asserted > 0); nesting time is isolated
  from serialization/validation by construction (sync engine, timed runs,
  cancel checked between runs). No GPU transfers anywhere in this layer.

## Defects found and fixed (own scope)

1. Allowance-style sampler lesson re-applied: no shared-vertex duplication
   in piece polygons (grading samples are already distinct).
2. Single-seed runs skipped objective validation (comparison never reached)
   — objectives now validate up front in `optimizeMarker`.
3. Test-premise errors: open-ended length always stacks (structural failure
   needs a length cap → `over-length`); "mismatched" seam reused the same
   edge; coincident-touch is legal contact, not overlap.
4. Cross-talk (not a defect in this scope): a concurrent grading refactor
   broke `deriveSize` mid-session (their own 13 failures); it cleared before
   this report with no action needed here. Graded-input tests were written
   against the stable `deriveSize` contract and pass unmodified.

## Assumptions / limits

- Holes are ignored for packing (conservative: never over-reports utilization).
- Repeat snaps candidate origins only; no print matching.
- Length is unbounded unless `lengthLimitM` is set; width is the hard boundary.
- max-utilization ≡ min-length for identical placed sets; the comparison
  still matters across partial results (full placement always wins).
- No AI nesting, ERP, cutting protocols, grading changes, or physics.
