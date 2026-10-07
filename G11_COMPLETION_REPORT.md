# G11 — Production Pattern Engineering (tracks A–E + integration)

G10 (3D workspace, `src/view/` + `app/`) pre-existed. This round adds the
production layer on the G9 CAD. No solver, FEM, collision, GPU, G8, G9, or
G10 file was modified.

## Interfaces reused

- G8A document/validation/serialization + kernel ops (`src/pattern/cad.ts`).
- G9A `geom` (incl. `sampleArc`), `queries` (`resolveSegment`,
  `sampleLoopLocal`, `measurePanel`, `pointInPanel`, hit-testing),
  `snap`/`history`/`selection` untouched.
- G9B `offsetLoop` miter precedent (G11A generalizes it to per-edge
  distances, arcs, and issue reporting instead of new-panel creation).
- G9C units (`formatLength`) and `measureSeamLength`; G8B
  `resolveStitchPairs`/`validateSeams` for paired-seam truth.
- G11 modules compose forward only: markings → readiness → techsheet →
  export. No cycles.

## Files added

- `src/cad/production.ts` — G11A: `ProductionSet` sidecar (allowances with
  per-edge overrides, notches, grainlines, folds, drills, internal lines, cut
  lines, annotations, label regions, panel metadata incl. cut quantity);
  derived allowance boundaries (exact line miters, sagitta-bounded arc
  sampling, spike/parallel-join/self-intersection reporting); notch frames;
  cut resolution (sewing vs allowance); reference validation; canonical
  persistence. The sewing boundary keeps its ids; derived rings carry none.
- `src/cad/markings.ts` — G11B: notch tick geometry (single/double/custom),
  orphan tracking across boundary edits, centered/cross grainline builders,
  deterministic panel numbering, canonical label fields, marking validation
  (orphans, duplicate positions, vertex-sited notches, escapes, edge-to-edge
  folds allowed).
- `src/cad/readiness.ts` — G11C: `productionReadiness` with
  INVALID/WARNINGS/READY_FOR_EXPORT, severity-coded diagnostics with suggested
  actions, per-panel measurements, total cuttable area, paired-seam comparison
  with configurable tolerance (never auto-fixed), `seamSideLengths` helper.
- `src/cad/techsheet.ts` — G11D: deterministic multi-panel SVG (cut/sewing/
  allowance, notches, grain arrows, folds, drills, internals, annotations,
  label boxes, per-edge dimensions), visibility toggles, selection highlight
  + hit reporting, XML escaping.
- `src/cad/export.ts` — G11E: `ExportIR` (metres, deterministic), canonical
  JSON with lossless import + structural `compareIR`, minimal DXF R12 writer
  (LINE/CIRCLE/TEXT, 10 named layers, mm, `$INSUNITS=4`, tech-sheet layout),
  and `exportProductionPackage`, which throws unless the report is
  READY_FOR_EXPORT. Format note: DXF R12 chosen as the minimal universally
  readable manufacturing path; AAMA/DXF, PDF, and nesting are reported as
  unsupported, not faked.
- `src/cad/index.ts` — barrel extended (production/markings/readiness/
  techsheet/export).
- Tests (43): `production` (14), `markings` (7), `readiness` (7), `techsheet`
  (5), `export` (9), `g11-workflow` (1 × 15-step acceptance).

## Verification

- Scope gate: `tests/cad + tests/pattern + tests/garment` → **28 files /
  348 tests pass** (305 pre-G11 + 43 new), 0 failures.
- Joint tree: `npx tsc --noEmit` exit 0; `tests/view` 7 files / 73 pass
  (concurrent G12 grading track shares the tree; its mid-session breakage
  cleared before this report — no G11 file was involved).
- Acceptance (`g11-workflow`): 4-panel top engineered (allowances, notches,
  grainlines, fold, drill, dart, annotation, labels, cut quantities, cut
  lines) → READY, 0 errors, 5/5 seams within tolerance → SVG preview →
  JSON + DXF export (< 5 s, DXF > 50 entities) → JSON round-trip clean →
  native reload reproduces the IR → allowance edit changes the DXF
  deterministically.
- Performance/memory/GPU: export is pure string building over ≤ hundreds of
  vertices (sub-second in tests); no solver contact, no GPU transfers, no
  new allocations of note. Deliberately not benchmarked beyond the
  in-test timing/size assertions — nothing here is hot-loop code.

## Defects found and fixed (own scope)

1. Allowance sampler emitted the closing vertex twice (phantom 5-gon +
   bogus parallel-join issues) — edges now contribute all-but-last samples.
2. Same dedup bug in the sewing cut-ring path.
3. `leftNormal`-scale lesson re-applied: normals normalized at use sites.
4. Test-premise errors: fold-on-boundary is legal (kept, now covered);
   1e-6² ring is legitimately below the area floor; identical rings correctly
   refuse join (no reversed edge); hem edit must apply to all panels or the
   seam gate — correctly — fires on the 10 mm discrepancy.
5. `notchTicks`/`cutBoundary` orphan paths throw; IR/techsheet skip them
   (validation owns the error).

## Assumptions / limits

- Allowance spikes (>10× local offset) warn; self-intersections error; the
  geometry is still returned for inspection either way.
- Export gate is strict: WARNINGS also refuse. Waivers are a later phase.
- DXF subset documented in every export's warnings (faceting, label
  flattening, R12-only entities).
- Grading/sizes (G12), nesting, marketplace, rendering, and physics are
  untouched.
