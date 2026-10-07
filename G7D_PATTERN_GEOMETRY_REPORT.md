# G7D Pattern Geometry — Report

Branch: `feature/g7d-pattern-geometry`. New code: `src/pattern/pattern-geometry.ts`
(2D only, no solver contact). New tests: `tests/pattern/pattern-geometry.test.ts`
(13), `tests/pattern/pattern-triangulation.test.ts` (9). CPU-only, ~1 s total.

Verdict: foundational 2D pattern geometry complete — polygon/polyline/arc
boundaries, holes, winding normalization, point-in-panel, arclength
parameterization, deterministic constrained triangulation with tagged boundary
edges, quality metrics, rest-space bridge, serialization. 22/22 green.
No solver or seam-graph file modified.

## 1. What was built

`PatternPanel{id, outline, holes, grainAngleRad, materialId}` with boundaries
as `polyline | arc{center, radius, a0, a1}` primitives. Pipeline:
`validatePanel` (sanitize → winding-normalize → reject) →
`bridgeHoles` → `earClip` → `triangulatePatternPanel` →
`{panelId, vertices(xy), triangles, boundaryEdges, grain/material}`.

- **Winding**: outer normalized CCW, holes CW; reversal flags recorded.
  Reversed input triangulates bitwise-identically (tested).
- **Arcs**: deterministic chord subdivision by sagitta tolerance
  (`n = ceil(span / 2acos(1-tol/r))`); full circles via `a0 == a1`.
- **Rejections** (coded `PatternError`): `self-intersecting-boundary`,
  `zero-area-panel`, `empty-loop`, `invalid-hole`, `hole-outside`,
  `hole-touching`, `holes-overlap` (touch/cross/nesting),
  `degenerate-output` (ear stall, sliver under `minTriArea`, bridging
  desync), `unbridgeable`, `invalid-arc`, `invalid-json`.
- **Hole bridging**: deterministic first-valid bridge in (hole-vertex,
  polygon-vertex) index order with endpoint-incident edges skipped; tag
  replay shares the identical probe, so tags cannot diverge from geometry
  (divergence throws instead of mistagging).
- **Boundary edges**: only once-used edges whose endpoints are consecutive on
  one source loop, each tagged `(loop, edgeIndex, t0, t1)` in arclength;
  bridge slits (cross-loop, used-once index pairs) are correctly excluded as
  interior. Sorted deterministically.
- **Metrics** (`measurePanelQuality`): panel/mesh area + rel err, Hausdorff
  true-boundary→mesh-boundary deviation (arcs densely resampled), min
  triangle area, min shape quality `4√3·A/Σe²` (1 = equilateral).
- **Rest bridge**: `panelToRestMesh()` lays the panel flat `(x, 0, y)` with
  `uv = pattern xy` — exactly `preprocess()` inputs, so rest Dm/areas derive
  from pattern space (proven by test: preprocess succeeds, areas match,
  hinges/masses valid).

## 2. Correctness gates (all green)

| check | result |
|---|---|
| rectangle/triangle/L-shape accept, correct vertex counts | pass |
| reversed winding → identical geometry + bitwise-identical triangulation | pass |
| bowtie self-intersection rejected (dedicated code) | pass |
| collinear/duplicate → zero-area/empty-loop rejected | pass |
| hole outside/touching/overlapping/nested rejected (4 distinct codes) | pass |
| needle panel rejected via `minTriArea` | pass |
| arc chordal sagitta ≤ tol; endpoints exact; full circle closes | pass |
| point-in-panel: inside/outside/on-outer/on-hole/hole-interior | pass |
| arclength quarters + wraparound exact | pass |
| rectangle: 2 tris, 4 tagged edges, exact t-quarters | pass |
| Euler `tris = n−2`, indices in range, all areas > 0 (all shapes) | pass |
| hole: 10 verts, 8 tris, 4 outer + 4 hole tags | pass |
| rectangle metrics: area exact, deviation 0, minQ 0.693 | pass |
| quarter disc (r=0.05, tol=1e-4): chordal deficit 2.4e-3, mesh-exact (0.00e+0), dev 9.12e-5, 13 tris | pass |
| rest bridge + preprocess + 1 hinge + masses + metadata/serialization round-trips | pass |
| panel JSON round-trip → bitwise-identical triangulation; input never mutated | pass |

Two test-expectation precisions were calibrated during the run (both test-side,
not module bugs): chordal-vs-true area on arcs, and f32 FEM area summation
(3.5e-10 over exact).

## 3. Full-suite evidence

`npx tsc --noEmit` clean. Full `vitest` run: **71 files / 345 tests green**
(323 pre-existing + 22 G7D). No existing file modified; the only new `src/`
file is `src/pattern/pattern-geometry.ts`.

## 4. Non-goals honored; seams for later tracks

No solver, seam-graph, contact, CCD, Newton, or rendering changes. Open
seam-facing outputs ready for G7C: per-edge `(loop, edgeIndex, t0, t1)` tags,
`boundaryPoint(loop, t)` arclength evaluation, stable panel/vertex identity
through serialization. Grain/material are validated passthrough metadata
(material semantics stay with G7A).
