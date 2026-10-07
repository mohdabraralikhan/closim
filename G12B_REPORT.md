# G12B — Grade Rule Engine — Delivery Report

Track: G12B (size grading, rule engine, precedence, propagation, diagnostics)
Status: **Complete**
Date: 2026-10-06

## Scope delivered

Turns grading definitions into deterministic per-size deltas and derived
geometry, on top of the G12A data model. The master pattern remains the source
of truth: derivation reads the master and produces cloned, CAD-op-built
derived documents; the master is never mutated (asserted byte-for-byte in
tests).

## Architecture

`src/grading/engine.ts` (new) is the pure rule core:

```
effective delta (grading point, size) =
    own rule delta            — per-size (master→size) or accumulated transition
  OR mirrored source delta    — axis component negated (mutually exclusive with own rule)
  + Σ panel adjustments       — for (anchor panel, size); at most one per pair
```

- **Precedence is explicit**: own rule XOR mirror binding (enforced at
  `addRule`/`addMirrorBinding` and re-checked in validation as
  `conflicting-rule`), plus at most one panel adjustment per (panel, size)
  (`conflicting-rule` on duplicates). Mirror chains are rejected
  ("mirror chains are ambiguous") in both model ops and validation.
- **Transition accumulation**: `ruleDeltaForSize` sums the transition steps
  from the master through every active size up to the target, in size-set
  order. Inserting a middle size re-chains later sizes by design.
- **Composition provenance**: `composeDelta` returns `mirrorOfId` and
  `panelAdjustmentIds` so every applied-delta row explains where its delta
  came from (`RuleApplication` rows carry both).
- `-0` is normalized away in mirror negation so serialized documents and
  fingerprints never carry `-0`.

### Quality diagnostics (never block derivation)

Only invalid geometry (`validatePatternDocument`, including arc radius
consistency) blocks derivation via `GradingError("invalid-document")`.
Quality signals are reported:

| Signal | Code | Source |
|---|---|---|
| Paired seam sides differ after grading | `seam-mismatch` | `checkSeams`/`seamSideLengths` over `resolveStitchPairs`; tolerance `DEFAULT_SEAM_TOLERANCE_M = 0.003 m` (overridable via `DeriveOptions.seamMismatchM`), rows in `report.seams` |
| Transition sign flip between consecutive active sizes | `inconsistent-direction` | `checkRuleConsistency` (first step master→S is exempt) |
| Marking outside the derived panel | `outside-panel` | production derivation issues |
| Marking anchor dangling or unresolvable | `unknown-marking-anchor` | production derivation issues |

Document-level validation (`validateGradingDocument`) additionally covers
panel adjustments, mirror bindings, the production sidecar
(`validateProductionSet` mapped to grading diagnostics) and marking anchors
(unknown references, panel mismatch, anchors without a production set).

### Production-set derivation

`deriveProductionSet(document, gradedDocument, sizeId)` produces the per-size
production sidecar:

- T/ID-referenced entities (allowances, notches, cut lines, panel meta) copy
  verbatim — they key on stable IDs and arclength fractions.
- Position-based markings (grainlines, folds, drills, annotations, label
  regions) copy verbatim unless a `MarkingAnchor` binds them to a grading
  point's effective delta, in which case every position is shifted by that
  delta.
- Every marked position is checked with `pointInPanel` against the derived
  geometry; violations are explicit `outside-panel` issues — never silent
  drops or moves.
- Callable standalone (defense-in-depth orphan/issue reporting) and after
  `deriveSize` (validation-clean path). Throws `invalid-argument` when no
  production set is assigned.

### Fingerprint / invalidation

`fingerprintGradingDocument` (FNV-1a-64 over canonical JSON) now includes
`production` and `markingAnchors`, so production edits invalidate the derived
cache deterministically.

## API surface (new exports via `src/grading`)

`DEFAULT_SEAM_TOLERANCE_M`, `activeOrderedSizes`, `ruleDeltaForSize`,
`buildDeltaContext`, `gradingPointPanel`, `composeDelta`,
`effectiveDeltaForPoint`, `checkRuleConsistency`, `seamSideLengths`,
`checkSeams`, `deriveProductionSet`; model ops `createPanelAdjustment`/
`addPanelAdjustment`/`removePanelAdjustment`, `createMirrorBinding`/
`addMirrorBinding`/`removeMirrorBinding`, `assignProduction`,
`createMarkingAnchor`/`addMarkingAnchor`/`removeMarkingAnchor`.
`derive.ts` is rewired onto the engine (evaluated anchors now follow the
graded curve of their endpoints plus their own delta) and re-exports
`activeOrderedSizes`/`ruleDeltaForSize`.

## Test matrix (tests/grading/engine.test.ts — 20 tests)

- Delta propagation: 5-size linear transition accumulation (positions,
  cumulative `ruleDeltaForSize`, applied-row provenance); asymmetric per-size
  deltas; inactive sizes excluded from regeneration and refused at derive.
- Mirror bindings: mirrored delta with negated axis component + `mirrorOfId`
  provenance + `effectiveDeltaForPoint` agreement; rule-on-mirror-bound
  rejected; chains rejected; self-mirror rejected; source-without-rule
  flagged by validation.
- Panel adjustments: composed with point deltas for every grading point on
  the anchored panel, `panelAdjustmentIds` provenance, other panels
  unaffected; removal restores base deltas; duplicate (panel, size) rejected.
- Quality diagnostics: matched seams within tolerance; seam desync produces
  `seam-mismatch` + row `withinTolerance: false` with exact side lengths;
  direction flips produce `inconsistent-direction` without blocking.
- Curved boundaries: radial arc grading derives a valid document and the
  evaluated arc midpoint follows the graded curve (0.81, 0.4); asymmetric
  radial grade throws `invalid-document` (arc radius consistency).
- Production derivation: verbatim ID/t entities; anchored marking displaced
  by the effective delta; unanchored inside marking verbatim; outside-panel
  markings produce exactly 2 issues (one per endpoint); orphan anchors
  reported on the direct query path; missing production set throws.
- Master integrity: master byte-identical across derivations; master edit
  marks the cache stale (`isStale`) and `regenerateAll` replaces entries with
  re-derived geometry at the new master position.

## Verification

- `npx tsc --noEmit -p tsconfig.json` — clean for all grading modules (the
  only repo error is a pre-existing, unrelated in-flight file from another
  agent: `src/marker/workspace.ts` "Cannot find name 'nestPieces'").
- `npx vitest run tests/grading` — **62/62 pass** (model 11, anchors 10,
  derive 21, engine 20; all 42 pre-existing tests pass unchanged — the new
  engine is backward compatible).
- Scoped regression `tests/pattern tests/cad tests/garment tests/view
  tests/serialize` — **421/421 pass**.

## Master-pattern invariant

Held throughout: the master document is the source of truth; every graded
size is a derived representation built by CAD ops from master geometry plus
composed deltas; `tests` assert `JSON.stringify(master)` is unchanged across
derivations and that regeneration after a master edit produces fresh derived
geometry with a new source fingerprint.

## Notes / known limits

- Orphan-anchor issues in `deriveProductionSet` are reachable only via the
  direct query path; a full `deriveSize` run is blocked earlier by
  document-level validation (unknown marking-anchor reference). This is
  intentional defense-in-depth.
- `inconsistent-direction` is warning-level by design: inward-then-outward
  grade plans are a legitimate patternmaking choice; the diagnostic exists so
  the choice is visible.
