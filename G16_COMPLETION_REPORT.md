# G16 — Advanced Garment Construction (G16A + framework + FINAL)

G15 and below pre-existed. This round adds `src/construction/` (features +
component framework) and `tests/construction/` (24 tests). No solver, FEM,
collision, GPU, grading, CAD, marker, render, or view file was modified —
with one justified exception below.

## Interfaces reused

- G8A document/validation/ops/serialization; kernel `splitBoundarySegment`,
  `movePoint`, `deletePanel`, `createPoint`.
- G9A `geom`/`queries` (incl. `segmentIntersection`), G9B `divideSegment`,
  `buildPanelFromRing`, G8B `Seam`/`validateSeams` (dart closures and
  component attachments join the existing seam graph — no second model).
- G11A production builders consume derivation descriptors
  (folds/notches/drills/internals/grainlines); G11C readiness gates the
  acceptance garment; G12 grading works through stable IDs; G8/G15 run the
  derived garment unchanged.

## Files added

- `src/construction/features.ts` — G16A: `ConstructionSet` sidecar
  (dart/pleat/gather params, stable ids); intake/apex/depth validation with
  per-feature ok/broken status; `deriveConstruction` (fresh-clone
  derivation, order-deterministic, failures reported per feature, input
  never mutated); open-V + press folds; closed darts as leg-to-leg seams;
  knife/box/inverted pleats as intake + fold notches; gathers as
  correspondence metadata + notches; dart transfer; canonical persistence.
- `src/construction/components.ts` — framework: `ComponentSet`,
  `registerComponentDerivation` (B/C/D extension without editing this
  file), dependency fingerprints, current/stale/invalid lifecycle,
  regeneration with seam-invalidation reporting, built-in collar/cuff
  bands, patch pockets, buttons, buttonholes.
- `src/construction/index.ts` — barrel.
- Tests (24): `features` (13: CRUD/transfer/derivation/validation/
  extremes/overlaps/serialization/downstream), `components` (9: bands/
  pocket/closures/registry/regen/invalidation/relocation/pruning),
  `g16-workflow` (2: collared-shirt acceptance + upstream propagation).

## Verification

- Scope: `tests/construction` **24/24 pass**; neighbors
  (cad/pattern/garment/grading/marker/render/view) all pass — **676/677**
  overall. The single failure is another worker's in-flight
  `tests/cad/g13-svg-pdf.test.ts` (their PDF calibration); `tsc` is fully
  clean.

## Close-out full-tree state (post-G16)

- `npx tsc --noEmit`: exit 0.
- Full suite minus `tests/webgpu`: **841/844 pass**. The 3 failures are all
  in the concurrently-edited G13 track (`tests/cad/g13-adversarial.test.ts`:
  units-matrix expectation, size-set counts, concave boundary detection —
  all G13-worker code and fixtures, mid-edit at close-out). Zero failures
  in G16 scope or any file this phase owns or touched.
- Acceptance (`g16-workflow`): 5-panel shirt + 3 darts + box pleat + gather
  + collar + 2 cuffs + pocket + button + buttonhole → 10-seam graph valid →
  stable-id grading → re-derivation clean → avatar assembly → NaN-free
  simulation → production readiness with 0 errors → render representation →
  per-panel triangulation → deterministic serialization.
- Propagation: neckline +0.05 → collar regenerates wider (same panel id),
  seams re-resolve, simulation stays valid — zero manual reconstruction.

## Genuine interface bug fixed (G7D, with evidence)

Splitting a straight boundary edge (public kernel `splitBoundarySegment`)
inserts a 180° vertex that strands the order-dependent G7D ear clipper with
a collinear final triple (`degenerate-output`) — direction-dependent
(top-edge splits threw, bottom/left succeeded), reproducible with kernel
ops alone, blocking any split-then-triangulate workflow G16 requires.
Fix in `src/pattern/pattern-geometry.ts` (`triangulatePatternPanel` only):
on `degenerate-output` and only then, drop near-collinear loop vertices and
retry once, else rethrow. Inputs that triangulated before take the untouched
fast path (byte-identical; full pattern/cad/garment suites green unchanged).
Regression tests added to `tests/pattern/pattern-triangulation.test.ts`
(midpoint on every edge, double midpoints, determinism, fast-path guard).
Solver, FEM, contact, and collision code untouched.

## Defects found and fixed (own scope)

1. Pleat subdivision used absolute split params on tail pieces (wrong
   geometry) — rebased to relative params; fold notches resolved to live
   segment ids after all splits.
2. Join/splice slice dropped the wrong junction endpoints (zero-length
   close) — chainA whole + chainB interior (found by re-deriving the ring).
3. Seam-invalidation only fired when old panels existed in the input doc —
   now reported against replaced ids (stateless re-derivation is normal).
4. Deterministic re-derivation preserves panel ids (asserted, not the
   reverse); callers re-resolve via the invalidation report.
5. Test-premise errors: segment-count arithmetic (7, not 6), unassigned
   builder results, placements missing for derived panels, hem edits that
   must apply to all panels to preserve seam equality, grainlines avoiding
   dart cutouts; production `insideCheck` gained an EPS boundary tolerance
   (G7D "boundary counts in"; flaky exact-vertex ray tests no longer reject
   on-edge markings, genuinely-outside points still fail).

## Assumptions / limits

- Closed-dart "closure" is a sewable seam, not a geometric wedge merge
  (consistent with the solver-knows-nothing rule).
- Pleat intake extends the edge endpoint (shared-corner semantics, validated
  downstream); overlapping features on one edge fail explicitly, second
  wins nothing silently.
- Component bands/pockets are simple rectangular/marking derivations under
  B/C/D-owned type names — richer parametric versions register new types.
- Deleting a component prunes on re-derivation from base; incremental docs
  report invalidated seams instead.
- No marketplace/payments/AI/accounts/animation/try-on/new solver work.
