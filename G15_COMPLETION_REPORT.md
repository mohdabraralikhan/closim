# G15 — Professional Garment Rendering & Visualization

G10 (3D workspace + three.js browser app) and G14 (marker) pre-existed.
This round adds `src/render/` + `tests/render/` (35 tests) and
`app/render-g15.ts`. No solver, FEM, collision, GPU, grading, CAD, marker,
or view file was modified.

## Interfaces reused

- G8 `AssembledGarment`, avatar specs + `signedDistanceToAvatar`, placements.
- G10 view-core: `OrbitCameraState`, `Bounds`/`computeBounds`,
  `VisibilityFlags` idiom (mirrored, not coupled), `GarmentWorkspace`
  untouched — the adapter reads workspace outputs, never drives them.
- G11E export idiom (canonical JSON, R12 writer shape) for the package;
  three.js (already a dependency) for scene-graph construction only.
- T-shirt builders (`garment/tshirt`) and `CpuSolver` for the integration flow.

## Files added

- `src/render/materials.ts` — G15B: appearance-only `VisualMaterial`
  (color/roughness/metallic/opacity/normal/texture/weave/edge tint,
  `physicalRef` by reference), six documented presets (no physical-accuracy
  claims), library with panel assignment, texture load-state registry,
  explicit fallback, versioned replacement, canonical persistence.
- `src/render/uvgen.ts` — G15C: single-pass deterministic per-panel UVs from
  pattern space (bbox → [0,1]², optional grain-to-+V alignment, exact inverse
  mapping, boundary UV loops). Sim positions cannot move the print (tested).
- `src/render/representation.ts` — G15A: `RenderGarment` (copied positions,
  area-weighted degenerate-safe normals, panel UVs, topology key),
  explicit `syncRenderPositions` (topology change reported, never remapped),
  `RenderAvatar`, seam polylines from weld data, boundary segments,
  collision heat band, render modes, draft/preview/high quality levels,
  visibility flags. Quality never touches physics.
- `src/render/presentation.ts` — G15D: six camera presets (+custom),
  bounds framing at 40° FOV, five data-driven light rigs, backgrounds incl.
  transparent, reference-only `PresentationScene`, deterministic capture plans
  (front/back/three-quarter/garment-only commercial set), serialization.
- `src/render/three-adapter.ts` — scene-graph builder (per-panel material
  groups, wireframe/boundary/seam/avatar overlays, light rig, orbit-state
  camera application), in-place updates with vertex-count guard, texture
  queue (browser loads, never headless), counted disposal. No renderer
  instantiation, no texture fetch — fully headless-testable.
- `src/render/package.ts` — renders/ manifest (scene + materials + captures
  + garment/sim-epoch source), per-scene capture ownership enforced.
- `src/render/index.ts` — barrel.
- `app/render-g15.ts` — browser-only capture execution (camera, clear
  color/alpha, avatar visibility restore, PNG data URL) + texture-loader
  resolution. Typechecked with the app; pixels execute in-browser only.
- Tests (35): `three-headless`, `materials` (8: creation/presets/library/
  fallback/round-trip + UV determinism/grain/invariance), `representation`
  (7), `presentation` (5), `adapter` (5: conversion/groups/update/dispose/
  camera/gating), `lifecycle` (8: load/render/unload/reload, simulate/sync,
  rebuild, material versioning, extremes, missing data, restarts, isolation),
  `g15-workflow` (1 × 12-step commercial acceptance).

## Verification (own scope, all executed)

- `tests/render`: **7 files / 35 tests pass**, 0 failures.
- Adjacent scope re-run: render + cad + pattern + garment + grading + marker
  + view → **612/614 pass**; the 2 failures are another worker's in-flight
  `tests/grading/g12-final.test.ts` (`drillId is not defined`, plain
  unfinished-test ReferenceError), untouched here.
- `npx tsc --noEmit`: clean for every file in this scope. The only repo
  errors are concurrent-worker in-flight files: the G13 `export-ir.ts`
  barrel-name collision in `src/cad/index.ts` and their unfinished
  `tests/cad/g13-fixtures.ts`. Left for their owner — renaming either side
  unilaterally would collide with live edits.
- Acceptance (`g15-workflow`): graded T-shirt → cotton/silk assignment →
  fit + 3 simulation steps → explicit epoch-5 sync → UV stability across
  deformation → framed studio scene → headless preview graph (garment +
  avatar + seams + boundary + lights) → front/back/three-quarter +
  garment-only plans → roughness edit provably changes no physics/pattern
  state → per-scene render packages round-trip.
- Performance: sync is a memcpy + normal pass; adapter updates write
  attributes in place (wireframe rebuild only when visible); capture timing
  is recorded per capture in-browser (`elapsedMs`), never mixed with solve
  time. No GPU readback for rendering (positions flow solver → render copy
  → attributes; reads happen only in CPU reference tests).

## Defects found and fixed (own scope)

1. Dead `mapToUV` stub + duplicated frame computation — unified into one
   exact-inverse pass.
2. Duplicate `class` attributes in SVG-adjacent selection code path
   (same class-suffix helper pattern as G11D, applied correctly here).
3. Capture/package test mixed scenes in one package — the per-scene gate
   correctly refused; test now asserts the refusal plus valid per-scene
   packages.
4. Test-precision errors from Float32 pattern storage (honest 1e-6
   tolerances, documented inline).

## Assumptions / limits

- Pixels require a browser: capture *plans* are fully tested; `dataUrl`
  production runs only in-app. Stated, not hidden.
- Textures never load headlessly by design (`textureQueue` + explicit
  fallback); procedural texture *rendering* is adapter future work.
- Quality levels tune presentation (pixel ratio, AA, shadows, anisotropy,
  overlays) — physics is unreachable from these settings.
- No marketplace/accounts/payments/AI/animation/try-on/fabric scanning.
