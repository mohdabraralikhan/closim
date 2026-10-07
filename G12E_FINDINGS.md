# G12E — Adversarial Grading QA Findings

Scope: `src/grading/*` (engine, model, validate, serialize, presentation, units, derive) and its
integration surfaces with `src/pattern/cad.ts`, `src/cad/production.ts`, `src/cad/readiness.ts`.
Method: 29 adversarial tests in `tests/grading/adversarial.test.ts`, run against production code
without modifying it. Attack classes: reference integrity, size-set bookkeeping, geometry
degeneration, production sidecar corruption, persistence tampering, export-gate bypass.

Result: **no critical grading, geometry, persistence or master-pattern corruption defects found.**
Two low-severity findings below. Full regression after the suite: 926/926 tests pass, type-check
clean.

---

## Finding 1 — Corrupted derived-cache entries pass document validation and load

- **Severity:** Low
- **Subsystem:** grading persistence (`src/grading/validate.ts`, `src/grading/serialize.ts`)
- **Reproduction:**
  1. Build and regenerate any grading document; `serializeGradingDocument(regenerated)`.
  2. Parse the string, set `parsed.graded[0].document.schemaVersion = 2`, re-serialize with
     `JSON.stringify` (bypassing the serializer's validation gate) and call `parseGradingDocument`.
  3. Parse succeeds; no diagnostic flags the cache entry.
- **Expected:** A derived-cache entry claiming a foreign schema (or otherwise not a valid pattern
  document) is rejected on load, or the cache is dropped so it is rebuilt from authoritative inputs.
- **Actual:** `validateGradingDocument` only checks `graded.masterId` and `graded.document.id`
  (validate.ts:316-321). A cache entry with matching IDs and a corrupted body loads and is served
  by `documentForSize` whenever the (input-based) fingerprint still matches.
- **Impact:** Bounded. The cache is regenerable, `isStale`/fingerprint invalidation rebuilds on any
  input edit, and master geometry is untouched — this cannot corrupt the master pattern or export
  inputs, only a stale/invalid copy of a derived artifact. Canonical key ordering places `graded`
  before `master` in serialized output, so naive string-level tampering tends to hit the cache
  before the master — which is exactly the harmless target.
- **Recommended fix:** In `validateGradingDocument`, require `graded.document.schemaVersion === 1`;
  alternatively have `parseGradingDocument` drop `document.graded` entirely and let callers
  re-derive from inputs (the cache is a pure function of the fingerprint inputs).

## Finding 2 — Balanced bowtie is diagnosed as zero-area instead of self-intersection

- **Severity:** Low (diagnostic quality only — derivation is still blocked)
- **Subsystem:** pattern CAD geometry validation (`src/pattern/cad.ts` loop-area check + geometry
  kernel `validatePanel`)
- **Reproduction:** Corner grading point on segment pair ab∩bc; per-size rule with delta
  `[-0.2, +0.9]` (moves B=(0.4,0) to (0.2,0.9), folding edge AB across edge CD with lobes of equal
  signed area). `deriveSize` throws with
  `degenerate-panel: loop '...' has area -2.78e-17 m²` / `pattern geometry (zero-area-panel)`.
- **Expected:** A self-intersecting boundary loop is reported as `self-intersection` (the mapping at
  cad.ts:711 exists and does fire for asymmetric bowties — verified with delta `[-0.2, +1.2]`,
  signed area −0.06 m²).
- **Actual:** When the two bowtie lobes cancel, the signed shoelace area is ~0 and the degeneracy
  check classifies the loop as degenerate before any crossing is named.
- **Impact:** None on safety: derivation aborts with `invalid-document`, the master pattern is
  untouched, and the size cannot reach production. Only the diagnostic label is misleading, which
  can send a pattern-maker debugging "zero area" instead of "crossing edges".
- **Recommended fix:** Run the kernel segment-crossing check before (or independently of) the
  signed-area degeneracy check in `validatePanel`/`validatePatternDocument`, so crossing loops are
  always labeled `self-intersection`.

---

## Defenses verified holding (attack → outcome)

| Attack | Defense observed |
| --- | --- |
| Duplicate grading point ID / duplicate anchor target | `duplicate-id` / `conflicting-anchor` at construction; re-checked in validation |
| Duplicated rule, rule on mirror-bound point, orphaned rule, unknown-size delta, non-finite delta, duplicate panel adjustment (all injected via corrupted persistence) | `validateGradingDocument` flags each; `deriveSize` refuses (`invalid-document`) |
| Removing a grading point | Cascades rule, mirror bindings and marking anchors; stale caches flagged via fingerprint |
| Removing/inserting/reordering sizes | Deltas, adjustments and graded entries cleaned or re-accumulated; incomplete rule tables fail loudly (`missing-rule`), never silently skip |
| Inactive size | Excluded from transition accumulation, refused as target |
| Bowtie / zero-length edge from grade deltas | Derivation aborts (`invalid-document`), master byte-identical before/after |
| 1 µm deltas over 10 sizes | No measurable drift (< 1e-12 m) |
| Direction-flipping transition rules | `inconsistent-direction` diagnostics; positions still exact; never blocks |
| Missing rule: strict vs non-strict | Strict aborts; non-strict reports `missing-rule` and leaves the point at master |
| Negative seam allowance, orphaned marking anchor (sidecar corruption) | Validation diagnostic + derivation refusal; direct `deriveProductionSet` reports `unknown-marking-anchor` instead of dropping |
| Notch through grading | ID, segment reference and arclength fraction survive; resolves at t=0.5 of the graded edge (0.215, 0) |
| Production metadata through per-size derivation | Allowances/notches/grainlines/drills/annotations/panel-meta counts and IDs preserved; anchored grainline displaced, never dropped |
| 5× regeneration | Byte-identical canonical derived output |
| Save/reload round-trip | Fingerprint and canonical derived output identical |
| Master tampering (schema version, duplicate grading point) | `parseGradingDocument` rejects |
| Full workflow (regenerate → production derive → views → insert/remove size → save/reload → regenerate ×2) | Master pattern byte-identical (canonical) at every step |
| Clean sizes into G11 gate | `productionReadiness` = `READY_FOR_EXPORT`, zero errors, metadata intact per size |
| Over-graded size with anchored markings outside the panel | `deriveProductionSet` reports `outside-panel` issues; G11 gate returns `INVALID` — corruption cannot slip to export |
