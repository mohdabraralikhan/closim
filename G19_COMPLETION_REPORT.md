# G19 — Digital Garment Product & Catalog System (G19A + FINAL)

G18 and below pre-existed. This round adds `src/product/` + `tests/product/`
(15 tests). No solver, FEM, collision, GPU, grading, CAD, marker, render,
view, construction, or project file was modified.

## Interfaces reused

- G12 grading fixtures/model/validation/serialization (size sets, master
  documents); G8 garment project + serialization; G11A production builders,
  G11C readiness gate, G11E export package (real DXF/JSON payloads),
  G11D tech-sheet SVG (real preview payloads); G14 marker workspace;
  G15 materials/presentation/package idioms (adapted, not duplicated).

## Files added

- `src/product/product.ts` — G19A: `Product` (id/name/slug/SKU/category/
  tags/author/status/thumbnail/garment-ref/sizes/formats/license/variants/
  artifacts/metadata/dates), five application states with gated transitions
  (ready/published require a clean validation), variants (single/multi/
  format-bundle/custom with size-range and artifact-reference checks),
  artifacts (filename/type/sizes/format/revision/checksum/generator/status),
  licenses (validation incl. custom-requires-terms), revision pinning
  (`garmentRevision` + `garmentFingerprint`; later edits check via
  `isPinnedCurrent`, never auto-apply), explicit `bumpProductRevision`
  (stales artifacts, returns to draft), duplication, canonical persistence
  on an independent `PRODUCT_SCHEMA_VERSION`.
- `src/product/catalog.ts` — library with configurable taxonomy, add/
  remove/update with id+SKU protection, deterministic search/filter/sort,
  lightweight cards (stored fields only — browsing never loads garment or
  simulation state), integrity audit, persistence.
- `src/product/previews.ts` — preview assets bound to product+garment
  revision and render config, deterministic filenames, stub-injectable
  batch generation with completed/failed reporting, explicit staleness on
  garment change, primary/thumbnail resolution.
- `src/product/licensing.ts` — license metadata (no payments/accounts/DRM
  anywhere), customer-facing summary restricted to current artifacts.
- `src/product/index.ts` — barrel.
- Tests (15): `product` (7: identity/states/variants/pinning/duplication/
  persistence/licensing), `catalog` (7: identity/search/scale/audit/cards,
  previews batch/staleness, licensing summary), `g19-workflow` (1 × full
  commercial acceptance).

## Verification

- Scope: `tests/product` **15/15 pass** (vitest also runs the G18
  `tests/production/` namesake green).
- Broad gate (product/project/construction/cad/pattern/garment/grading/
  marker/render/view): **69 files / 727 tests pass**, 0 failures.
- `npx tsc --noEmit`: 2 errors, both in concurrently-edited files outside
  this scope (`src/cad/marker.ts` missing export, `tests/cad/g13-
  adversarial.test.ts` implicit any). Nothing in `src/product`,
  `tests/product`, or any file this phase touched.
- Acceptance (`g19-workflow`): two-panel top → graded XS–XXL → READY
  production export (real DXF) → real nesting → 6 file payloads with
  sha256 checksums → product TOP-001 → 3 deterministic previews →
  thumbnail → 2 variants → ready → published → catalog search/card/summary
  → garment edit provably cannot move the release → explicit re-release
  (rev 2, artifacts stale) → product+catalog byte-identical round-trips.

## Defects found and fixed (own scope)

1. Scale-test asserted a clean audit on preview-less products — the audit
   was right to flag them; the test now asserts exactly that (200/200
   `missing-preview`, no identity issues).
2. License defaults are restrictive (commercial use off unless requested) —
   the acceptance test requests it explicitly rather than weakening the default.

## Assumptions / limits

- Payments, checkout, accounts, marketplace, cloud, DRM, AI products: out
  of scope by phase design; licensing is metadata only.
- Preview pixels execute in the browser render path; here payloads are
  deterministic tech-sheet SVGs with recorded checksums.
- G19B–G19F deep tracks (variant/download-manager UI, catalog frontend,
  preview farm, license QA tooling) can extend these modules; the data
  contracts they need already exist and are tested.
