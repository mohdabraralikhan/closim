# G12D — Multi-Size Presentation — Delivery Report

Track: G12D (multi-size visualization and nesting)
Status: **Complete**
Date: 2026-10-06

## Scope delivered

A designer-facing multi-size presentation core that lets a designer inspect
an entire size set and clearly understand how the master pattern changes
across sizes — headless and renderer-ready, following the repository's
established pattern (G9D/G10): pure TypeScript workspace models with no DOM
and no rendering dependency; a future renderer binds to the produced data
without changing geometry semantics.

## Architecture (`src/grading/presentation.ts`)

### Views

`createSizeView(document, options)` produces the complete view in one call:

- **Layers** — one per active size in size-set order, each carrying:
  panel outlines (closed polylines in **global coordinates**, arcs sampled
  into 24 chords via `panelOutline`), grading-point markers, master→graded
  rule vectors, per-size diagnostics and an `invalid` flag, plus a
  deterministic `nestedOrder` (master = −1).
- **Modes** — `"active"` (single size, defaults to base size), `"overlay"`
  (all visible sizes at their anchored positions) and `"nested"`
  (concentric by construction: grading is anchored to fixed master
  positions, so sizes nest without any marker-nesting algorithm; the spec
  explicitly requires no particular geometric nesting beyond visualization).
- **Master highlighting** — `masterLayer` drawn underneath in nest order,
  labeled from the master pattern name.
- **Per-size visibility** — hidden sizes keep their layer and render order
  (`visible: false`) so visibility switching never reshuffles the scene;
  labels are emitted only for visible layers.
- **Size labels** — placed at the top-center of each layer's combined
  bounding box.
- **Grading points / rule vectors / comparison** — toggled via
  `showGradingPoints`, `showRuleVectors`, `comparisonSizeId`.

Invalid sizes (e.g. a size missing rule deltas) do not fail the view: the
layer is flagged `invalid` with the derivation diagnostic attached, so
malformed derived geometry is visually identifiable. Quality diagnostics
(`seam-mismatch`, `inconsistent-direction`) surface on their size's layer
from the G12B derivation reports.

### Comparison

`compareSizes(document, sizeIdA, sizeIdB)` (also reachable through the view
options) reports per-grading-point displacements A→B with distances, seam
length deltas and per-measurement deltas in canonical metres, plus
`maxPointDistanceM`.

### Selection

Selection always names **both entity and size** (`SizeSelection`).
`pickAtPosition(document, sizeId, position, toleranceM)` resolves the
nearest point, boundary segment (lines sampled at 33 points, arcs at 25) or
defaults against that size's geometry only — a user picking on XL never
accidentally hits M or the master (tested directly, including the case where
an unmoved point of another panel is the true nearest at shared coordinates).

### Editing behavior

`editIntent(selection)` encodes the default contract: derived sizes are not
independently editable — any selection on a derived size redirects editing
to the master (`redirectSizeId: "master"`); the caller then regenerates all
sizes (`regenerateAll`). Master selections are editable directly. Tested
through the view: a master edit + regeneration moves derived outlines to the
new master position.

### Diagnostics

Visible per layer: derivation failures (invalid geometry, missing rules,
arc inconsistencies) as `invalid` + diagnostics; seam mismatch and grade
direction anomalies from the derivation reports.

### Performance

No cloth simulation anywhere in the path — view construction is pure pattern
geometry over `PatternDocument`s (derive + outline sampling). 12-size views
build in milliseconds in tests; nothing touches the solver, GPU backend or
meshing. `documentForSize` serves fresh cached derived documents by
reference and re-derives only stale ones (fingerprint check).

## Tests (tests/grading/presentation.test.ts — 12 tests)

- 2 sizes: layers + master layer, nest order, labels at the combined bbox
  top.
- Rule vectors + grading-point markers on/off; exact master→graded arrows.
- 5, 10 and 12 sizes: deterministic nesting order, all valid, concentric
  hem growth per transition step.
- Active mode; visibility switching (hidden layers keep order, labels
  filtered); invalid size isolated with `missing-rule` diagnostic; empty
  size set rejected.
- Comparison: point displacement 0.005 m, measurement deltas, seam table,
  comparison through view options.
- Picking: master vs size scoping (graded B at (0.41,0) vs unmoved E at
  (0.4,0)), mid-edge segment picking, miss → null, unknown size → error.
- Editing contract: derived selection redirects to master; master selection
  editable.
- Master modification → regeneration reflected in view outlines; fresh
  cache served by reference, stale master re-derived.

## Verification

- `npx tsc --noEmit -p tsconfig.json` — all grading modules clean (remaining
  repo errors are other agents' in-flight files: `src/marker/workspace.ts`,
  `src/render/uvgen.ts`, `src/cad/dxf-export.ts`, `src/cad/dxf-validate.ts`).
- `npx vitest run tests/grading` — **94/94 pass** (model 11, anchors 10,
  derive 21, engine 20, measurements 20, presentation 12).
- Scoped regression (`tests/pattern tests/cad tests/garment tests/view
  tests/serialize`) — **421/421 pass**.

## Definition of done

A designer (or future renderer) can inspect the entire size set as overlay
or nested concentric layers, toggle any size's visibility, label sizes,
show grading points and rule vectors, compare any two sizes numerically,
pick entities scoped to one size, and follow the edit-master → regenerate
workflow — with every diagnostic visually attributable to its size.
