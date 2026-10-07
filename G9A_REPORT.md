# G9A Professional 2D Pattern CAD — Geometry & Editing Engine

## Pre-implementation interface audit

Branch: `feature/g8a-pattern-cad`. Scope: G9A only (first track of G9). No G8 files were edited; G8A, G8B, and related files are treated as read-only dependencies.

### Interfaces reused

- **G8A pattern CAD kernel** — `src/pattern/cad.ts` is the document model G9A builds on. It provides the immutable-style `PatternDocument` with stable panel/loop/point/segment IDs, panel-local point coordinates, `CircularArcSegment {centerPointId, sweepRad}` authored arcs, `Transform2D`, construction geometry, dimensions and constraints, `validatePatternDocument`, `serializePatternDocument` (canonical JSON), `triangulateCadPanel` (adapter to G7D), and the `PatternCadError` error taxonomy. G9A deliberately rebased onto this document rather than introducing a parallel model: all G9A operations accept and return a `PatternDocument` (or `{document, ...}` result objects) exactly like the G8A kernel operations.
- **G7D pattern geometry** — `src/pattern/pattern-geometry.ts` remains the triangulation layer; nothing in G9A bypasses the G8A adapter to reach it.
- **Error codes** — `PatternCadError` has no dedicated "no intersection found" code, so G9A operations that legitimately find no candidate geometry throw `unsupported-operation` with the specific reason in the message. All other failures reuse the existing codes (`missing-reference`, `zero-length-edge`, `invalid-arc`, `invalid-transform`, `invalid-dimension`, `degenerate-panel`, `self-intersection`, ...).
- **Working-tree preservation** — the user's live G8 edits (`src/pattern/cad.ts`, `src/pattern/pattern-geometry.ts`, `src/garment/**`, `tests/garment/**`) and all `.worktrees/**` checkouts were left untouched. G9A is purely additive: `src/cad/**` and `tests/cad/**`.

### G9A boundary

`src/pattern/cad.ts` provides the document, validation, serialization, and basic panel/point/segment operations, but no editing-session machinery: no undo/redo, no selection, no snapping, no hit-testing, and no advanced editing verbs (trim/extend/offset/mirror/merge/split/delete-point). Those are G9A's additive scope. G9A never mutates the meaning of the kernel: it only calls existing kernel operations or constructs fresh documents through the same public constructors, keeping IDs, transforms, and validation invariants intact.

## G9A implementation result

### Files added for G9A

- `src/cad/geom.ts` — scale-aware geometric predicates (collinearity/left-turn via sine with scale-relative epsilon), segment projection and parameters, line/segment intersection classification (proper, interior touch, endpoint touch, parallel probes), authored-arc geometry (`ArcGeometry`, `pointOnArc`, `arcLength`, sagitta-bounded deterministic `sampleArc`, `arcSegmentIntersections`, `arcArcIntersections` with concentric handling, `nearestOnArc`, `angleOnArc`), polygon utilities (`signedArea`, `perimeter`, `bbox`, `pointInPolygon`, `selfIntersections`), and 2D transforms (`rotateAround`, `reflectAcrossLine`, `leftNormal`, `lerp`).
- `src/cad/queries.ts` — read-side accessors (`getPanel`, `getPoint`, `getSegment`, `getLoop`, `findLoopByRole`), local/global coordinate conversion for points and segments (exact for lines, sampled polyline flagged `isotropic` for arcs), `nearestOnSegment`, `intersectSegments` (exact same-panel; cross-panel through global space, lines only), `measureLoop`/`measurePanel` (exact area including circular-segment correction `r²/2·(θ − sin θ)`), `sampleLoopLocal`, `bboxGlobal`, reference queries (kind-guarded `centerPointId`), `hitTest`/`nearestHit` with point < segment < panel ranking, `pointInPanel` (`pointInPanelGlobal`), box selection (contain/overlap), and unclamped `segmentParam`.
- `src/cad/history.ts` — `History` with record/content-equal no-op suppression, undo/redo stacks, redo invalidation on record, gesture nesting (`beginGesture`/`endGesture`/`cancelGesture`) that blocks stack access while open, an undo limit with constructor validation, and `snapshot`. `CadSession` wraps a document with `run(op)` dispatch for both `PatternDocument` and `{document, ...}` results, plus `undo`, `redo`, `cancelGesture`, `reset`.
- `src/cad/selection.ts` — `Selection` as an ordered unique set of entity IDs, `SelectionView` alias, `selectionOf`/`isSelected`/`setSelection`/`addToSelection`/`removeFromSelection`/`toggleSelection`/`clearSelection`, `pruneSelection` (drop IDs absent from the document), and `selectionStats` (unknown IDs ignored; totals reported only for known entity kinds).
- `src/cad/snap.ts` — `snapPosition(doc, raw, {toleranceM, gridM?, panelIds?, sagittaTolM?})` with fixed priority vertex > intersection > midpoint > on-segment > grid > none, distance-then-source-ID deterministic tie-breaks, arc sampling bounded by sagitta tolerance, panel scoping, `PatternCadError` skipping for degenerate candidate pairs, and raw-position passthrough on "none"; returns `{pos, kind, sourceIds}`.
- `src/cad/ops.ts` — advanced editing operations: `createConstructionPolyline`, `moveSegmentBy` (shifts endpoints and arc centers), `deletePoint` (construction cascade with foreign-reference, arc-center, and dimension guards; degree-2 boundary merge via orientation-aware `chainEndpoints`, line+line direct and arc+arc same-circle-same-direction merging through `buildMergedSegment`; loop-collapse guards for `n <= 2` and line-only `n <= 3`; orphan center cleanup), `mergeSegments` (adjacency + degree + collinear/same-circle checks, preserves segA identity), `trimSegment` (nearest hit from a reference point, moves the hit endpoint; no candidate → `unsupported-operation`), `extendLineBy`, `extendLineTo` (infinite target line vs closed cutter, keeps hits with `t ∈ [−ε, 1+ε]` so extend never trims), `offsetConstructionLine` (left-normal offset with fresh construction points), `mirrorPanel` (local-coordinate reflection, negated arc sweep, flipped loop orientation, IDs and transform preserved), `mirrorPanelCopy` (duplicate + mirror + `"<name> mirror"`), `splitSegment` (lines delegate to the kernel's `splitBoundarySegment`; arcs split into two arcs with `sweep·t` / `sweep·(1−t)` sharing the kernel-allocated center; sub-arc sweep degeneracy guard; ID allocation replicating the kernel's `allocateId` algorithm), `moveEndpointTo` (captures original arc angles before the move; start-move preserves end angle and sign, end-move accumulates `oldSweep + normalizeAngle(newAng − oldEndAng)`; radius mismatch → `invalid-arc`).
- `src/cad/index.ts` — barrel exporting `geom`, `queries`, `history`, `selection`, `snap`, `ops`.
- `tests/cad/fixtures.ts` + 9 test files (below).
- `G9A_REPORT.md` — this report.

### Design assumptions and decisions

1. **Rebase on G8A `PatternDocument`** (decided with the user): no parallel document model. An earlier parallel `src/cad/{types,document,ops}.ts` prototype was removed before implementation.
2. **Pattern-local coordinates are authoritative.** Global-space answers are derived through panel transforms; cross-panel intersection queries fall back to global space and are restricted to line segments (arcs would need sampling policy per query, deferred to G9B).
3. **Arc authoring stays kernel-native.** G9A splits/merges/moves arcs only through center + sweep semantics, and refuses operations that would collapse a sweep to zero or move an arc endpoint onto a radius-incompatible circle.
4. **`unsupported-operation` is the catch-all** for "nothing to do" outcomes (trim with no hit, extend with no crossing, snap pairs that fail), with the specific reason in the message, because the kernel error taxonomy has no `no-intersection` code and must not be edited.
5. **Shared-vertex boundary semantics:** deleting a degree-2 boundary point merges its two adjacent segments only when both are lines (collinear check) or both arcs on the same circle with matching direction; mixed line/arc pairs and D-shape-style 2-segment loops hit explicit guards instead of silently producing garbage geometry.
6. **Offsets:** straight construction-line offset only. Panel-boundary offsetting with arc joins is deferred to G9B (drafting tools) where the offset distance policy belongs.
7. **No G8 file edits, no UI.** Repo has no frontend, so G9D (editor UX) is out of scope here and will need a framework decision later.

### Tests

`tests/cad/` — **9 test files, 153 tests, all passing**:

| File | Tests | Focus |
|---|---:|---|
| `geom.test.ts` | 28 | predicates, intersections, arc sampling/geometry, polygon area/bbox |
| `queries.test.ts` | 27 | accessors, local/global conversion, measurement, hit tests, box select |
| `history.test.ts` | 17 | undo/redo, gestures, no-op suppression, session dispatch |
| `selection.test.ts` | 5 | set ops, pruning, stats |
| `snap.test.ts` | 10 | priority order, tie-breaks, panel scoping, arc sampling |
| `ops.test.ts` | 18 | construction polyline, move, delete-point cascade/guards, merge |
| `trim-extend.test.ts` | 11 | trim by reference, extend-by/extend-to, no-trim-on-extend |
| `mirror-split.test.ts` | 12 | mirror/mirror-copy, line and arc splits, endpoint moves on arcs |
| `adversarial.test.ts` | 25 | degenerate/zero-area, self-crossing, concentric arcs, guard firing, round-trips |

Adversarial coverage includes the bowtie/zero-area distinction, mixed-primitive D-shape merge guards, concentric and equal-radius/distinct-center arc pairs, arc endpoint moves across the wrap boundary, and error-code stability.

### Verification

- `npx tsc --noEmit -p tsconfig.json` — **exit 0**, no diagnostics.
- `npx vitest run tests/cad --reporter=dot --minWorkers=1 --maxWorkers=2` — **9 files / 153 tests passed**, ~5.6 s.
- Full regression (memory-scoped, worktrees excluded, single 2-worker tree). A single all-in-one run exceeded the 600 s command budget, so the suite was executed in two scoped halves with captured exit statuses:
  - Non-WebGPU: `npx vitest run --exclude "**/.worktrees/**" --exclude "tests/webgpu/**" --exclude "**/node_modules/**" --exclude "**/dist/**" --reporter=dot --minWorkers=1 --maxWorkers=2` → **42 files / 344 tests passed**, 382.50 s, `EXIT=0`.
  - WebGPU: `npx vitest run tests/webgpu --exclude "**/.worktrees/**" --exclude "**/node_modules/**" --exclude "**/dist/**" --reporter=dot --minWorkers=1 --maxWorkers=2` → **40 files / 175 tests passed**, 475.36 s, `EXIT=0`.
- **Post-G9A total: 82 files / 519 tests, 0 failures, 0 skipped.** No test or source file outside `src/cad/**`, `tests/cad/**`, and this report was modified; no Node processes were left running after the runs.
- Note on baseline: the G8A report recorded 72 files / 360 tests at G8A close. The current suite is 82 / 519; +9 files / +153 tests are G9A's cad suite, and the remaining delta is G8 test work landed by the user since the G8A report (an independent full-suite baseline was attempted earlier but was killed for memory before completing — the two scoped runs above are the first complete post-G9A suite result).

### Remaining G9 tracks

- **G9B** — drafting tools: parallel/perpendicular/construction helpers, full panel-boundary offset with arc joins, arrays, fillet/chamfer.
- **G9C** — constraints and measurements: dimension driving, constraint solver over the kernel's constraint records, live measurement overlays.
- **G9D** — editor UX: needs a UI framework decision (repo currently has no frontend).
- **G9E** — adversarial QA pass over the complete G9 stack.
- **Integration** — wiring G9A–G9E into an end-to-end editable pattern editor against the G8 garment pipeline.
