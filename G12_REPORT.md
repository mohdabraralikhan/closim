# G12 Report — Grading: sizes, measurements, rules, nested presentation

Solo delivery of all remaining G12 tracks (G12B, G12C, G12D, G12E, G12 FINAL).
Spec: `.g12-spec.md`. Architecture rule honored throughout: **the master
pattern is the source of truth; graded sizes are derived representations** —
no master geometry is ever distorted to produce a size.

## Modules delivered (`src/grading/`)

| File | Track | Contents |
|---|---|---|
| `types.ts` | G12A/B | `GradingDocument`, `SizeDefinition`, `GradeRule`, `GradingPoint`, `MarkingAnchor`, measurement types |
| `model.ts` | G12A/B | Pure clone-on-write ops: sizes, base size, rules (per-size/transition/mirror), grading points, panel adjustments, measurement definitions/assignments, production sidecar link, marking anchors |
| `engine.ts` | G12B | Size ordering, `effectiveDeltaForPoint` (own rule XOR mirror + panel adjustments; transition = cumulative sum in size-set order), `deriveSize` (3 passes: resolve/compose, apply via kernel `movePoint`, report), `regenerateAll`, `isStale` (fingerprint), strict vs lenient missing-rule policy |
| `anchors.ts` | G12B | Corner/midpoint/parameter anchor resolution to exact master positions |
| `derive.ts` | G12B | Quality diagnostics: `seam-mismatch`, `inconsistent-direction` (never blocking); invalid kernel geometry → `GradingError("invalid-document")` |
| `validate.ts` | G12B | Full integrity gate: duplicate IDs, dangling references, anchor/rule/mirror/adjustment validation, measurement assignment checks, production-sidecar revalidation, stale derived-cache detection |
| `measurements.ts`→`engine.ts`/`types.ts` | G12C | Measurement catalogue (unit, ordering, source), per-size assignments, `assignMeasurement` validation |
| `presentation.ts` | G12D | `createSizeView` (active/overlay modes, master layer, labels, visibility), `compareSizes` (point displacements, seam-length deltas, measurement deltas), `documentForSize` (master by reference) |
| `serialize.ts` | G12B | Canonical JSON round-trip (sorted keys, −0→0), versioned parse with rejection of tampered payloads |

## G12E — adversarial QA (29 tests, `tests/grading/adversarial.test.ts`)

No production code was modified. Full findings report: `G12E_FINDINGS.md`.

- 17 attack classes: cache poisoning, duplicate/orphan rules and points, NaN/±∞
  deltas, mirror-bound rules, missing rules (strict + lenient), size
  insert/remove/reorder/re-base, bowtie and zero-length degeneracies, 1e-6
  numeric drift over 10 sizes, direction flips, negative allowances, orphaned
  marking anchors, metadata preservation, 5× regeneration determinism, tampered
  persistence, per-size export gate, master byte-identity.
- Result: **no critical grading, geometry, persistence, or master-pattern
  corruption defects remain** (DoD met). Two low-severity findings documented
  (derived-cache schema check, balanced-bowtie diagnostic label).

## G12 FINAL — commercial acceptance (`tests/grading/g12-final.test.ts`)

The production-ready 4-panel top (G11 layout, 5 seams, full production
sidecar) becomes a graded product: XS–XXL derived from master size M.

### The 16 steps

1–6 open garment / master M / size table / body measurements / grading points
/ grade rules — 6 sizes, 2 measurements (chest, center-back-length), 8 corner
grading points, 8 transition rules.
7 generate — `regenerateAll` produces all six derived documents.
8 validate every size — document integrity 0 diagnostics, per-size positions
equal the size-chart targets.
9 nested inspection — overlay view: 6 valid visible layers + master layer.
10 grading differences — `compareSizes(M, XXL)`: max point distance
√(0.045²+0.06²), chest 0.92→1.01 (+0.09), every seam grows by its grade
(shoulder +0.045 width, sides/sleeves +0.06 length).
11 return to master — `documentForSize(doc, "master")` is the master document
by reference.
12 modify master — widen body 1 cm (front edges +1 cm, back edges −1 cm);
cache fingerprint goes stale.
13–14 regenerate + revalidate — every graded size = widened master + its
grade; base size M tracks the edited master exactly; integrity clean.
15 production validation per size — `deriveProductionSet` per size: 0 issues,
4 allowances / 4 notches / 4 grainlines / drills / 4 panel metas preserved;
anchored drill follows its grading point; every size passes
`productionReadiness` at **READY_FOR_EXPORT, 0 errors, all seams within
tolerance**.
16 export — selected size (XXL) and complete size set to JSON + DXF; IR
round-trips byte-identically; re-export is deterministic.

### Authoring transition rules (canonical pattern)

Transition deltas are **consecutive differences of targets**, not the targets
themselves: `step(i) = target(i) − target(i−1)`, with `step(0) = target(0)`.
This makes the cumulative grade land on each size's target and guarantees the
base size M sums to `[0, 0]`. The test encodes this as a `SIZES` target table
plus a derived `STEP` table. (An earlier draft stored targets directly as
deltas; they cancelled to zero across the set — caught by this test's
master-edit step, which is the first assertion that is not tautological with
respect to the table.)

### Seam-consistent grade scheme

Front-right edge +dx (hem B, top C), back-left edge −dx (hem E, top H), hems
+dy, sleeve tops −dy — all five seam sides stay equal for every cumulative
(dx, dy), so per-size readiness reports zero seam mismatches.

### Production consistency across sizes

T/ID-referenced entities (allowances, notches, cut lines, metadata, cut
quantities) copy verbatim onto each graded boundary; position-based markings
follow their marking anchors (drill → front hem grading point, grainline →
front top grading point) or stay verbatim (fold, dart, annotation, label).
No size is exportable unless its production validation passes — the workflow
asserts READY_FOR_EXPORT for all six.

### Persistence

`serializeGradingDocument` → `parseGradingDocument` round-trip preserves the
master, size set, measurements, grading points, rules, production sidecar,
marking anchors, and derived cache; re-derived output and re-derived production
sets are byte-identical (canonical JSON); reloaded documents pass validation
and re-export at READY_FOR_EXPORT.

## Verification

- Grading suite: **126/126** across 9 files (engine 20, derive 21, anchors 10,
  measurements 20, model 11, presentation 12, adversarial 29, G12 FINAL 2,
  fixtures) — G11 workflow still green.
- Full regression: 939/950 passed; the 11 failures are all in
  `tests/cad/g13-export-ir.test.ts`, another agent's in-flight G13 test
  written against `buildExportIR2`/`compareExportIR2` APIs that do not exist
  yet in `src/cad/export-ir.ts` (not part of G12; untouched).
- `tsc --noEmit` clean on all G12 modules and tests (remaining errors confined
  to foreign files: `g13-export-ir.test.ts`, `src/marker/workspace.ts`,
  `src/render/uvgen.ts`, `src/cad/dxf-export.ts`, `src/cad/dxf-validate.ts`).
- Performance: full 16-step workflow (generate → validate → compare → master
  edit → regenerate → per-size production derivation → 7 exports) completes in
  ~0.37 s; size-set generation well under the 2 s budget.

## Files delivered

- `src/grading/` — types, model, engine, anchors, derive, validate,
  presentation, serialize, units, index
- `tests/grading/` — 8 test files + fixtures (126 tests)
- `G12B_REPORT.md`, `G12C_REPORT.md`, `G12D_REPORT.md`, `G12E_FINDINGS.md`,
  this report
