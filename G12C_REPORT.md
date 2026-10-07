# G12C — Measurement & Size-System Layer — Delivery Report

Track: G12C (size definition and body-measurement layer)
Status: **Complete**
Date: 2026-10-06

## Scope delivered

A designer-definable size system, independent of grading mathematics:
arbitrary size sets (alphabetical, numeric, custom labels), a reusable
measurement catalogue with a typed schema, unit conversion at the boundary,
explicit size-set operations and whole-document validation + persistence.

## Architecture

### Measurement schema (`src/grading/types.ts`)

```
SizeSet
  ├─ measurementDefinitions[]   ← reusable catalogue (shared across sizes)
  └─ sizes[]
       └─ measurements[]        ← per-size values referencing the catalogue
```

- `MeasurementDefinition`: stable ID, name, optional description, display
  unit (`m | cm | mm | in`), source/type tag (`"body"`, `"block"`,
  `"imported"`, ... — free-form, no hard-coded sizing standard), optional
  tolerance (canonical metres) and an optional expected progression
  (`ordering: "none" | "increasing" | "decreasing"`) across the ordered
  active sizes.
- `SizeMeasurement` (per size): `measurementId` (catalogue reference),
  `valueM` (canonical metres), `unit` (entry/display fidelity),
  optional `source` and `toleranceM` (overrides the definition's).
- **No auto-implication**: measurements never generate pattern deltas.
  Derivation only moves geometry through explicit grading rules — tested
  directly (a fully measured size system with a rule-less grading point
  still refuses to derive).

### Units (`src/grading/units.ts`)

`toMetres` / `fromMetres` / `convertLength` over `m, cm, mm, in`
(`INCH_TO_M = 0.0254`); unknown units and non-finite values throw
`GradingError("invalid-unit" | "invalid-argument")`. Storage is always
canonical metres; the recorded unit round-trips for display.

### Size-set operations (`src/grading/model.ts`)

Existing: create size set, insert size (`insertSize`), remove size
(`removeSize`), reorder (`moveSize`), select base (`setBaseSize`), select
active (`setSizeActive`). New:

- `duplicateSize(doc, sizeId, newId, {label?, atIndex?})` — clones a size
  with its measurement values; default label `"<label> copy"`; label
  collisions are rejected with a request for an explicit label; inserts
  after the source by default.
- `renameSize(doc, sizeId, label, displayName?)` — non-empty, unique labels.
- `createMeasurementDefinition` / `addMeasurementDefinition` /
  `removeMeasurementDefinition` — catalogue management; removing a
  definition cascades per-size values so the document stays consistent.
- `assignMeasurement(doc, sizeId, measurementId, value, {unit?, source?,
  toleranceM?})` — converts to canonical metres, replaces existing entries.
- `removeMeasurement(doc, sizeId, measurementId)`.

### Validation (`src/grading/validate.ts`)

New detections on top of the existing duplicate-ID/label checks:

| Detection | Code |
|---|---|
| Unknown unit (definition or entry) | `invalid-unit` |
| Zero value (impossible body measurement) / bad tolerance / negative input | `invalid-measurement` |
| Size missing a catalogue measurement (catalogue is authoritative) | `missing-measurement` |
| Entry referencing an undefined measurement | `unknown-entity` |
| Declared progression violated across ordered active sizes (1e-9 m epsilon) | `inconsistent-ordering` |

`GradingErrorCode` gains `invalid-unit`, `invalid-measurement`,
`missing-measurement`, `inconsistent-ordering`; `GradingDiagnostic` gains
`measurementId`.

### Persistence

`serializeGradingDocument`/`parseGradingDocument` are canonical JSON over
the whole document, so the catalogue and entries persist automatically; the
serialize/parse gate runs full validation, so an incomplete size system
(missing measurements) refuses to serialize. Measurement entries are part of
`sizeSet`, hence part of the G12B derivation fingerprint — measurement edits
invalidate the derived cache deterministically.

## Migration note

`SizeDefinition.measurements` changed from `Record<string, number>` to
structured `SizeMeasurement[]` (spec schema requires unit/source/tolerance
per value). Internal API, pre-release: only one test and `assignMeasurement`
touched; no compat shims kept. Documents serialized by G12A versions remain
parseable (missing `measurementDefinitions` is tolerated as empty).

## Tests (tests/grading/measurements.test.ts — 20 tests)

- Units: all four units through canonical metres, inch conversions, unknown
  unit / non-finite rejections.
- Size systems: alphabetical (XS–XL), numeric (38–44) and custom
  (petite/regular/tall) sets; measurements with unit/source/tolerance in
  canonical metres; re-assignment replaces; single-value removal.
- Size-set operations: duplicate (values, label, position, base-size
  untouched), duplicate defaults + collision/unknown-source rejections,
  rename (+ empty/colliding/unknown rejections), reorder/insert/remove
  identity.
- Validation: missing measurements, unknown measurement references, invalid
  units, zero-value impossibility, inconsistent ordering (increasing marked
  catalogue, both pass and fail paths), cascade removal, and the
  no-auto-grading guarantee.
- Persistence: full size-system round-trip (including a duplicated size with
  a custom-source entry), refusal to serialize an incomplete system,
  deterministic fingerprint invalidation on measurement edits.

## Verification

- `npx tsc --noEmit -p tsconfig.json` — clean for grading modules (only the
  pre-existing foreign error in another agent's `src/marker/workspace.ts`).
- `npx vitest run tests/grading` — **82/82 pass** (model 11, anchors 10,
  derive 21, engine 20, measurements 20; all pre-G12C tests pass unchanged).
- Scoped regression (`tests/pattern tests/cad tests/garment tests/view
  tests/serialize`) — **421/421 pass**.

## Definition of done

The grading document now carries a reusable size system — catalogue plus
sizes plus measured values — fully independent of the rule table, validated
as a whole and persistent through the existing canonical serialization.
