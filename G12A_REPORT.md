# G12A — Grading Data Model: Delivery Report

Track: G12A (foundation for G12 grading). Status: **complete**.
Scope per spec: stable-ID grading entities with deterministic serialization; size
definitions; grading anchors referencing stable CAD IDs; grade rules (per-size and
transition); derived geometry generation that never mutates the master; deterministic
invalidation; full test matrix.

## Files

| File | Purpose |
| --- | --- |
| `src/grading/types.ts` | All G12A entities, `GradingError` / `GradingDiagnostic`, resolution + report types |
| `src/grading/anchors.ts` | Anchor resolution against stable CAD IDs (`point`, `corner`, `edge-relative`, `seam-point`) |
| `src/grading/model.ts` | Pure document ops: create/add/insert/remove/move sizes, grading points, rules, cache replace |
| `src/grading/derive.ts` | `deriveSize`, `regenerateAll`, FNV-1a-64 fingerprinting, staleness, delta resolution |
| `src/grading/validate.ts` | Whole-document validation returning diagnostics (never throws) |
| `src/grading/serialize.ts` | `canonicalJson`, `serializeGradingDocument` (validate-then-canonicalize), `parseGradingDocument` |
| `src/grading/index.ts` | Barrel re-export |
| `tests/grading/fixtures.ts` | Two-panel master, arc master, side seam, S/M/L size set, core grading points + rules |
| `tests/grading/model.test.ts` | Size ops, rule errors, conflicting anchors, master integrity |
| `tests/grading/anchors.test.ts` | All four anchor kinds incl. arc mid/quarter points |
| `tests/grading/derive.test.ts` | Full G12A test matrix (derivation, failures, invalidation, persistence) |

## Design decisions

1. **Master stays the single source of truth.** Derivation clones the master document
   and applies displacements only through validated CAD `movePoint` ops. Byte-for-byte
   master invariance is asserted after every derivation (`serializePatternDocument`).
   The master pattern is never distorted to create a size.
2. **Derived documents keep the master's `documentId`.** Entity IDs (points, segments,
   panels, loops) are identical across all sizes, so G12C body-measurement mappings and
   G12D presentation can key on stable IDs. `GradedPattern.document.id === master.document.id`.
3. **Rule modes.**
   - `per-size`: absolute delta Master→Target (`delta[sizeId]` is the offset from master).
   - `transition`: delta Size N→N+1, accumulated from the master through every active
     size up to the target, in size-set order (master = position zero). Inserting a
     middle size therefore re-chains later sizes (they inherit the inserted step unless
     their rules are retuned) — asserted explicitly in the insertion test.
4. **Anchor kinds.** `point` and `corner` (and `edge-relative`/`seam-point` at t=0/1)
   resolve to exactly one pattern point → *vertex* displacement. Intermediate
   `edge-relative` / `seam-point` anchors (0<t<1) are **evaluated-only**: the derived
   document keeps master topology (no segment splitting); the reported
   `gradedPosition` is the **derived-endpoint interpolation plus the anchor's own rule
   delta**, so the evaluated point follows the graded curve. Rule-less evaluated
   anchors still follow the graded curve.
5. **Anchors never use array positions.** Every anchor references entity IDs
   (pointId / segmentId / panelId / seam side segment IDs). Conflicting anchors (two
   grading points resolving to the same pattern point) are rejected at
   `addGradingPoint` and in validation (`conflicting-anchor`).
6. **Deterministic invalidation.** `fingerprintGradingDocument` = FNV-1a-64 over
   `canonicalJson` of {masterId, masterDocument, sizeSet, ruleTable, gradingPoints,
   seams}. `isStale` compares fingerprint + masterId + documentId. Any edit to master
   geometry, sizes, rules, or grading points flips every cached `GradedPattern` stale;
   `regenerateAll` / `upsertGraded` clear it.
7. **`removeSize` prunes** rule delta entries and derived-cache entries keyed to the
   removed size, so remaining documents stay strictly valid. `insertSize` supports
   arbitrary index; base-size replacement on delete is explicit.
8. **No market naming.** Sizes carry `id`, `label`, `displayName`, `order index`,
   `measurements` (open string-keyed record), `active`. Body-measurement IDs are free
   strings for G12C to define.

## Diagnostics / error codes

`GradingErrorCode` = `invalid-argument | invalid-document | duplicate-id |
duplicate-rule | conflicting-anchor | unknown-size | unknown-base-size |
unknown-entity | invalid-grading-point | invalid-rule | missing-rule |
stale-derivation`. Validation (`validateGradingDocument`) never throws; derivation
(`deriveSize`) validates first and throws `GradingError("invalid-document", …)` with
all diagnostics joined. `deriveSize(doc, sizeId, { strict: false })` turns missing
vertex rules into diagnostics and leaves those points at master positions.

## Test matrix coverage (42 grading tests, all green)

One size · multiple sizes with accumulated transitions · unrelated points unmoved ·
evaluated-only anchors · missing rule (strict + lenient) · per-size missing key ·
transition hole · duplicate rule · deleted source entity (panel deletion →
`unknown-entity`) · huge-delta self-intersection rejected by the CAD gate · unknown /
inactive sizes · size insertion · size deletion · master byte-invariance + fingerprint
stability · `regenerateAll` byte-identical output · entity-ID stability · save/load
round trip incl. derived cache · master-edit invalidation → regeneration clears ·
rule-change invalidation → re-derivation clears · canonical JSON key-order
independence · seam context persistence · conflicting anchors · invalid master
document · clone non-aliasing.

## Verification

| Check | Result |
| --- | --- |
| `tsc --noEmit -p tsconfig.json` (root) | clean |
| `tsc --noEmit -p app/tsconfig.json` | clean |
| `vitest run tests/grading` | 42/42 pass (3 files) |
| `vitest run tests/pattern tests/cad tests/garment tests/view` | 421/421 pass (35 files) |

WebGPU files from other agents were not touched or committed.

## Limitations (for G12B/C/D/E/FINAL)

- **Evaluated-only anchors do not split segments.** Intermediate edge/seam anchors are
  reported, not materialized as derived geometry. If G12B needs real split topology
  (e.g. for per-stitch grading of a curved seam midpoint), it should add a
  topology-editing pass in derive; `AnchorEvaluation` already carries `pointIds: []`
  as the extension point.
- **`seam-point` anchors are read-only references today.** They validate against seam
  sides and resolve to segment geometry, but seams themselves are not re-stitched per
  size (stitch counts are size-independent). G12B rule engine can extend `GradeRule`
  if seam-level grading (e.g. ease per size) is required.
- **Transition rules accumulate through *active* sizes only.** Deactivating a middle
  size re-chains the transition across it (documented N→N+1 semantics).
- **FNV-1a-64** is deterministic but not collision-proof; it guards UI staleness, not
  security. Swap for SHA-256 in `fingerprintGradingDocument` alone if FINAL wants it.

## Integration notes

- Import from `src/grading/index.js`. Entry points: `createGradingDocument`,
  `addSize` / `insertSize` / `removeSize`, `addGradingPoint`, `addRule`,
  `setRuleDelta`, `deriveSize`, `regenerateAll`, `isStale`, `serializeGradingDocument`
  / `parseGradingDocument`, `validateGradingDocument`.
- G12B (rule engine): build on `ruleDeltaForSize` + `activeOrderedSizes`; the report
  objects (`RuleApplication`, `AnchorEvaluation`, `GradingDerivationReport`) are the
  audit surface for any solver-grade rule evaluation you add.
- G12C (body measurements): `SizeDefinition.measurements` is an open
  `Record<string, number>`; `assignMeasurement` is the mutation op. Interpolation
  between sizes can key on `GradedPattern.document.id === master.document.id` for
  stable point mapping.
- G12D (nested presentation): `GradedPattern` entries live in `GradingDocument.graded`
  in size-set order after `regenerateAll`; use `findGraded` + `isStale` before render.
- G12E (adversarial): attack surface = `validateGradingDocument` (pure, non-throwing)
  and `deriveSize` (throwing). Conflicting anchors, duplicate rules/IDs, unknown
  entities, stale caches, and self-intersecting derived geometry are all covered —
  see the test matrix for the expected codes.
- G12 FINAL: persistence is `serializeGradingDocument` → `parseGradingDocument`
  (canonical JSON, byte-stable round trip, validated on parse). Master + grading
  document + seams serialize together in one `GradingDocument`.
